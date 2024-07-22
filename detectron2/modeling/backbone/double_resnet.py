# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.modeling.backbone.resnet import (
    BasicBlock, 
    BottleneckBlock, 
    DeformBottleneckBlock, 
    BasicStem, 
    ResNet,
    make_stage, 
    build_resnet_backbone,
    ResNetBlockBase, 
)


from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "DoubleResNet",
    "build_double_resnet_backbone",
]

class FusionLayer(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_depth, out_channels, attention = False, relu = False, bias=False):
        super().__init__()
        self.attention = attention
        self.relu = relu
        self.rgb_conv = nn.Conv2d(in_channels_rgb, out_channels, kernel_size=1, bias=bias)
        self.depth_conv = nn.Conv2d(in_channels_depth, out_channels, kernel_size=1, bias=bias)
        if attention:
            self.attention_conv = nn.Conv2d(2 * out_channels, 2, kernel_size=1, bias=True)
        self.fusion_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, bias=bias)

    
    def forward(self, rgb_features, depth_features):
        rgb_transformed = self.rgb_conv(rgb_features)
        depth_transformed = self.depth_conv(depth_features)
        fused_features = torch.cat([rgb_transformed, depth_transformed], dim=1)
        if self.attention:
            attention_weights = torch.sigmoid(self.attention_conv(fused_features))
            rgb_weight = attention_weights[:, 0:1, :, :]
            depth_weight = attention_weights[:, 1:2, :, :]

            weighted_rgb = rgb_weight * rgb_transformed
            weighted_depth = depth_weight * depth_transformed
            
            fused_features = torch.cat([weighted_rgb, weighted_depth], dim=1)

        fused_features = self.fusion_conv(fused_features)

        if self.relu:
            fused_features = F.relu(fused_features)
        return fused_features


class DoubleResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, resnet1, resnet2, out_features=None, fusion_steps = [], attention=False, relu=False, bias=False, fuse_in = "DEPTH"):
        super().__init__()
        self.resnet1 = resnet1
        self.resnet2 = resnet2
        self.fusion_steps = fusion_steps
        self.fuse_in = fuse_in
        assert fuse_in in ["DEPTH", "COLOR"]

        self.in_shape1 = self.resnet1.stem.in_channels
        self.in_shape2 = self.resnet2.stem.in_channels


        self._out_feature_strides = self.resnet2._out_feature_strides
        self._out_feature_channels = self.resnet2._out_feature_channels

        self.stage_names = self.resnet1.stage_names

        if out_features is None:
            out_features = self.resnet1.out_features
        
        self._out_features = out_features
        assert len(self._out_features)

        #assert all(elem in fusion_steps for elem in out_features), "Cannot output a layer that is not the result of a fusion"


        fusion_features_dict = dict()
        if "stem" in fusion_steps or "stem" in out_features:
            fusion_features_dict["stem"] = [resnet1.stem.out_channels, resnet2.stem.out_channels]
        for i, name in enumerate(self.stage_names):
            fusion_features_dict[name] = [resnet1.stages[i].out_channels, resnet2.stages[i].out_channels]

        self.fusion_layers = nn.ModuleDict()
        for name, out_channels in fusion_features_dict.items():
            if name in out_features:
                self.fusion_layers[name] = FusionLayer(out_channels[0], out_channels[1], out_channels[1],
                                                        attention=attention, relu=relu, bias=bias)

    def forward(self, x1, x2):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x1.dim() == 4 and x2.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x1.shape} and {x2.shape} instead!"
        outputs = {}
        x1 = self.resnet1.stem(x1)
        x2 = self.resnet2.stem(x2)
        if self.fuse_in=="DEPTH":
            if "stem" in self.fusion_steps:
                x2 = self.fusion_layers["stem"](x1, x2)
                if "stem" in self._out_features:
                    outputs["stem"] = x2
            elif "stem" in self._out_features:
                outputs["stem"] = self.fusion_layers["stem"](x1, x2)
            for name, stage1, stage2 in zip(self.stage_names, self.resnet1.stages, self.resnet2.stages):
                x1 = stage1(x1)
                x2 = stage2(x2)
                if name in self.fusion_steps:
                    x2 = self.fusion_layers[name](x1, x2)
                    if name in self._out_features:
                        outputs[name] = x2
                elif name in self._out_features:
                    outputs[name] = self.fusion_layers[name](x1, x2)
        else:
            if "stem" in self.fusion_steps:
                x1 = self.fusion_layers["stem"](x1, x2)
                if "stem" in self._out_features:
                    outputs["stem"] = x1
            elif "stem" in self._out_features:
                outputs["stem"] = self.fusion_layers["stem"](x1, x2)
            for name, stage1, stage2 in zip(self.stage_names, self.resnet1.stages, self.resnet2.stages):
                x1 = stage1(x1)
                x2 = stage2(x2)
                if name in self.fusion_steps:
                    x1 = self.fusion_layers[name](x1, x2)
                    if name in self._out_features:
                        outputs[name] = x1
                elif name in self._out_features:
                    outputs[name] = self.fusion_layers[name](x1, x2)
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_double_resnet_backbone(cfg, input_shape1, input_shape2):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    resnet1 = build_resnet_backbone(cfg, input_shape1)

    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    norm                = cfg.MODEL.RESNETS.NORM

    cfg.MODEL.BACKBONE.FREEZE_AT = cfg.MODEL.BACKBONE.get("FREEZE_AT_DEPTH", cfg.MODEL.BACKBONE.FREEZE_AT)
    cfg.MODEL.RESNETS.NUM_GROUPS = cfg.MODEL.RESNETS.get("NUM_GROUPS_DEPTH", cfg.MODEL.RESNETS.NUM_GROUPS)
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = cfg.MODEL.RESNETS.get("WIDTH_PER_GROUP_DEPTH", cfg.MODEL.RESNETS.WIDTH_PER_GROUP)
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = cfg.MODEL.RESNETS.get("STEM_OUT_CHANNELS_DEPTH", cfg.MODEL.RESNETS.STEM_OUT_CHANNELS)
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = cfg.MODEL.RESNETS.get("RES2_OUT_CHANNELS_DEPTH", cfg.MODEL.RESNETS.RES2_OUT_CHANNELS)
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = cfg.MODEL.RESNETS.get("STRIDE_IN_1X1_DEPTH", cfg.MODEL.RESNETS.STRIDE_IN_1X1)
    cfg.MODEL.RESNETS.RES5_DILATION = cfg.MODEL.RESNETS.get("RES5_DILATION_DEPTH", cfg.MODEL.RESNETS.RES5_DILATION)
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = cfg.MODEL.RESNETS.get("DEFORM_ON_PER_STAGE_DEPTH", cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE)
    cfg.MODEL.RESNETS.DEFORM_MODULATED = cfg.MODEL.RESNETS.get("DEFORM_MODULATED_DEPTH", cfg.MODEL.RESNETS.DEFORM_MODULATED)
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = cfg.MODEL.RESNETS.get("DEFORM_NUM_GROUPS_DEPTH", cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS)
    cfg.MODEL.RESNETS.NORM = cfg.MODEL.RESNETS.get("NORM_DEPTH", cfg.MODEL.RESNETS.NORM)
    
    resnet2 = build_resnet_backbone(cfg, input_shape2)

    cfg.MODEL.BACKBONE.FREEZE_AT          = freeze_at
    cfg.MODEL.RESNETS.NUM_GROUPS          = num_groups
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP     = width_per_group
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS   = in_channels
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS   = out_channels
    cfg.MODEL.RESNETS.STRIDE_IN_1X1       = stride_in_1x1
    cfg.MODEL.RESNETS.RES5_DILATION       = res5_dilation
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = deform_on_per_stage
    cfg.MODEL.RESNETS.DEFORM_MODULATED    = deform_modulated
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS   = deform_num_groups
    cfg.MODEL.RESNETS.NORM                = norm

    fusion_steps = cfg.MODEL.BACKBONE.FUSION.STEPS
    attention = cfg.MODEL.BACKBONE.FUSION.ATTENTION
    relu = cfg.MODEL.BACKBONE.FUSION.RELU
    bias = cfg.MODEL.BACKBONE.FUSION.BIAS
    fuse_in = cfg.MODEL.BACKBONE.FUSION.FUSE_IN

    return DoubleResNet(resnet1, resnet2, out_features=out_features, 
                        fusion_steps=fusion_steps, attention=attention, relu=relu, bias=bias, fuse_in=fuse_in)
