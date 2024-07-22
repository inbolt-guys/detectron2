# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from .double_resnet import build_double_resnet_backbone

__all__ = ["build_resnet_fpn_backbone", "build_retinanet_resnet_fpn_backbone", "FPN"]


class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x, x2=None):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        if x2 is None:
            bottom_up_features = self.bottom_up(x)
        else:
            bottom_up_features = self.bottom_up(x, x2)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

class FusionLayer(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_depth, out_channels, attention = False):
        super().__init__()
        self.attention = attention
        self.out_channels = out_channels
        self.rgb_conv = nn.Conv2d(in_channels_rgb, out_channels, kernel_size=1)
        self.depth_conv = nn.Conv2d(in_channels_depth, out_channels, kernel_size=1)
        if attention:
            self.attention_conv = nn.Conv2d(2 * out_channels, 2, kernel_size=1)
        self.fusion_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
    
    def forward(self, rgb_features, depth_features):
        rgb_transformed = self.rgb_conv(rgb_features)
        depth_transformed = self.depth_conv(depth_features)
        concatenated_features = torch.cat([rgb_transformed, depth_transformed], dim=1)
        if self.attention:
            attention_weights = torch.sigmoid(self.attention_conv(concatenated_features))
            rgb_weight = attention_weights[:, 0:1, :, :]
            depth_weight = attention_weights[:, 1:2, :, :]

            weighted_rgb = rgb_weight * rgb_transformed
            weighted_depth = depth_weight * depth_transformed
            
            fused_features = torch.cat([weighted_rgb, weighted_depth], dim=1)
            fused_features = self.fusion_conv(fused_features)
            return fused_features
        else:
            fused_features = self.fusion_conv(concatenated_features)
            return fused_features

class DFPN(Backbone):
    """
    Double FPN for BGR and Depth / Normals
    """


    def __init__(
        self,
        fpnRGB,
        fpnDepth,
        mode: str = "simple",
        conv_dims: List[int] = (-1,),
        fusion_out_channels: int = -1
    ):
        super(DFPN, self).__init__()
        assert isinstance(fpnRGB, FPN) and isinstance(fpnDepth, FPN)
        assert mode in ["simple", "conv", "cat", "late_fpn"]

        self.fpnRGB = fpnRGB
        self.fpnDepth = fpnDepth
        self.mode = mode
        
        if mode == "conv":
            self.convs = nn.ModuleDict()
            fpnRGB_output_shapes = self.fpnRGB.output_shape()
            fpnDepth_output_shapes = self.fpnDepth.output_shape()
            for name in fpnRGB_output_shapes:
                cur_channels = fpnRGB_output_shapes[name].channels + fpnDepth_output_shapes[name].channels
                if len(conv_dims) == 1:
                    out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
                    # 3x3 conv for the hidden representation
                    self.convs[name] = self._get_conv(cur_channels, out_channels)
                    cur_channels = out_channels
                else:
                    self.convs[name] = nn.Sequential()
                    for k, conv_dim in enumerate(conv_dims):
                        out_channels = cur_channels if conv_dim == -1 else conv_dim
                        if out_channels <= 0:
                            raise ValueError(
                                f"Conv output channels should be greater than 0. Got {out_channels}"
                            )
                        conv = self._get_conv(cur_channels, out_channels)
                        self.convs[name].add_module(f"conv{k}", conv)
                        cur_channels = out_channels
            self.conv_out_channels = out_channels
        elif mode == "late_fpn":
            self.fusion_layers = nn.ModuleDict()
            fpnRGB_output_shapes = self.fpnRGB.output_shape()
            fpnDepth_output_shapes = self.fpnDepth.output_shape()
            for name in fpnRGB_output_shapes:
                rgb_shape = fpnRGB_output_shapes[name]
                depth_shape = fpnDepth_output_shapes[name]
                if fusion_out_channels == -1:
                    fusion_out_channels = rgb_shape.channels + depth_shape.channels
                self.fusion_layers[name] = FusionLayer(rgb_shape.channels, depth_shape.channels, fusion_out_channels, attention=True)

                

    def _get_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @property
    def size_divisibility(self):
        assert self.fpnRGB.size_divisibility == self.fpnDepth.size_divisibility
        return self.fpnRGB.size_divisibility

    @property
    def padding_constraints(self):
        assert self.fpnRGB._square_pad == self.fpnDepth._square_pad
        return {"square_size": self.fpnRGB._square_pad}

    def forward(self, rgb, depth):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        fpnFeaturesRGB = self.fpnRGB(rgb)
        fpnFeaturesDepth = self.fpnDepth(depth)

        out = {}
        if self.mode == "simple":
            for f, v in fpnFeaturesRGB.items():
                out[f+"c"] = v
            for f, v in fpnFeaturesDepth.items():
                out[f+"d"] = v
            assert len(out) == len(self.fpnRGB._out_features) + len(self.fpnDepth._out_features)
        elif self.mode == "cat":
            for f in fpnFeaturesRGB:
                out[f] = torch.cat((fpnFeaturesRGB[f], fpnFeaturesDepth[f]), dim=1)
            assert len(out) == len(self.fpnRGB._out_features) == len(self.fpnDepth._out_features)
        elif self.mode == "conv":
            for f in fpnFeaturesRGB:
                out[f] = self.convs[f](torch.cat((fpnFeaturesRGB[f], fpnFeaturesDepth[f]), dim=1))
        elif self.mode == "late_fpn":
            for f in fpnFeaturesRGB:
                out[f] = self.fusion_layers[f](fpnFeaturesRGB[f], fpnFeaturesDepth[f])
        return out

    def output_shape(self):
        out = {}
        if self.mode == "simple":
            for f, v in self.fpnRGB.output_shape().items():
                out[f+"c"] = v
            for f, v in self.fpnDepth.output_shape().items():
                out[f+"d"] = v
        elif self.mode == "cat":
            rgb_shape = self.fpnRGB.output_shape()
            depth_shape = self.fpnDepth.output_shape()
            for f in rgb_shape:
                assert rgb_shape[f].stride == depth_shape[f].stride 
                out[f] = ShapeSpec(channels=rgb_shape[f].channels + depth_shape[f].channels, stride=rgb_shape[f].stride)
        elif self.mode == "conv":
            assert self.conv_out_channels > 0
            rgb_shape = self.fpnRGB.output_shape()
            for f in rgb_shape:
                out[f] = ShapeSpec(channels=self.conv_out_channels, stride=rgb_shape[f].stride)
        elif self.mode == "late_fpn":
            rgb_shape = self.fpnRGB.output_shape()
            for f in rgb_shape:
                out[f] = ShapeSpec(channels=self.fusion_layers[f].out_channels, stride=rgb_shape[f].stride)
        return out


@BACKBONE_REGISTRY.register()
def build_resnet_double_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.INPUT.FORMAT == "RGBD" or cfg.INPUT.FORMAT == "RGBRD":
        bottom_upRGB = build_resnet_backbone(cfg, ShapeSpec(channels=3))
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        cfg.MODEL.BACKBONE.FREEZE_AT = cfg.MODEL.BACKBONE.FREEZE_AT_DEPTH
        bottom_upDepth = build_resnet_backbone(cfg, ShapeSpec(channels=1))
        cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
    elif cfg.INPUT.FORMAT == "GN":
        bottom_upRGB = build_resnet_backbone(cfg, ShapeSpec(channels=1))
        cfg.MODEL.BACKBONE.FREEZE_AT = cfg.MODEL.BACKBONE.FREEZE_AT_DEPTH
        bottom_upDepth = build_resnet_backbone(cfg, ShapeSpec(channels=3))
        cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at
    else:
        raise ValueError(cfg.INPUT.FORMAT, "cfg.INPUT.FORMAT must be in RGBD, RGBN, GD, GN")
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    fpnRGB = FPN(
        bottom_up=bottom_upRGB,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    fpnDepth = FPN(
        bottom_up=bottom_upDepth,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    backbone = DFPN(
        fpnRGB = fpnRGB,
        fpnDepth = fpnDepth,
        mode = cfg.MODEL.BACKBONE.MODE,
        conv_dims=cfg.MODEL.BACKBONE.CONV_DIMS if cfg.MODEL.BACKBONE.MODE == "conv" else None,
        fusion_out_channels=cfg.MODEL.BACKBONE.FUSION_OUT_CHANNELS if cfg.MODEL.BACKBONE.MODE == "late_fpn" else None
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_double_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.INPUT.FORMAT == "RGBD" or cfg.INPUT.FORMAT == "RGBRD":
        bottom_up = build_double_resnet_backbone(cfg, ShapeSpec(channels=3), ShapeSpec(channels=1))
    elif cfg.INPUT.FORMAT == "GN":
        bottom_up = build_double_resnet_backbone(cfg, ShapeSpec(channels=1), ShapeSpec(channels=3))
    else:
        raise ValueError(cfg.INPUT.FORMAT, "cfg.INPUT.FORMAT must be in RGBD, RGBRD, GN")
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
