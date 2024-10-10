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

from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from . import build_mpvit_backbone, MPViT_Backbone

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

__all__ = ["DoubleMPViT", "build_double_mpvit_backbone", "build_double_mpvit_fpn_backbone"]


class FusionLayer(nn.Module):
    def __init__(
        self,
        in_channels_rgb,
        in_channels_depth,
        out_channels,
        attention=False,
        relu=False,
        bias=False,
    ):
        super().__init__()
        self.attention = attention
        self.relu = relu
        self.rgb_conv = nn.Conv2d(in_channels_rgb, out_channels // 2, kernel_size=1, bias=bias)
        self.depth_conv = nn.Conv2d(in_channels_depth, out_channels // 2, kernel_size=1, bias=bias)
        self.out_channels = out_channels
        if attention:
            self.attention_conv = nn.Conv2d(out_channels, 2, kernel_size=1, bias=True)
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias)

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


class DoubleMPViT(Backbone):
    def __init__(
        self,
        mpvit1: MPViT_Backbone,
        mpvit2: MPViT_Backbone,
        out_features=None,
        fusion_steps=[],
        attention=False,
        relu=False,
        bias=False,
        fuse_in="DEPTH",
    ):
        super().__init__()
        self.mpvit1 = mpvit1
        self.mpvit2 = mpvit2
        self.fusion_steps = fusion_steps
        self.fuse_in = fuse_in
        assert fuse_in in ["DEPTH", "COLOR"]

        self._out_feature_strides = self.mpvit2._out_feature_strides
        self._out_feature_channels = self.mpvit2._out_feature_channels

        self.stage_names = ["stage2", "stage3", "stage4", "stage5"]

        if out_features is None:
            out_features = self.mpvit1.out_features

        self._out_features = out_features
        assert len(self._out_features)

        # assert all(elem in fusion_steps for elem in out_features), "Cannot output a layer that is not the result of a fusion"

        fusion_features_dict = dict()
        if "stem" in fusion_steps or "stem" in out_features:
            fusion_features_dict["stem"] = [
                mpvit1.backbone.stem.out_channels,
                mpvit2.backbone.stem.out_channels,
            ]
        for i, name in enumerate(self.stage_names):
            fusion_features_dict[name] = [
                mpvit1.backbone.mhca_stages[i].out_channels,
                mpvit2.backbone.mhca_stages[i].out_channels,
            ]

        self.fusion_layers = nn.ModuleDict()
        for name, out_channels in fusion_features_dict.items():
            if name in out_features:
                self.fusion_layers[name] = FusionLayer(
                    out_channels[0],
                    out_channels[1],
                    out_channels[1],
                    attention=attention,
                    relu=relu,
                    bias=bias,
                )

    def forward(self, x1, x2):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x1.dim() == 4 and x2.dim() == 4
        ), f"MPViT takes an input of shape (N, C, H, W). Got {x1.shape} and {x2.shape} instead!"
        outputs = {}
        x1 = self.mpvit1.backbone.stem(x1)
        x2 = self.mpvit2.backbone.stem(x2)
        if self.fuse_in == "DEPTH":
            if "stem" in self.fusion_steps:
                x2 = self.fusion_layers["stem"](x1, x2)
                if "stem" in self._out_features:
                    outputs["stem"] = x2
            elif "stem" in self._out_features:
                outputs["stem"] = self.fusion_layers["stem"](x1, x2)
            for name, patch_stage1, patch_stage2, stage1, stage2 in zip(
                self.stage_names,
                self.mpvit1.backbone.patch_embed_stages,
                self.mpvit2.backbone.patch_embed_stages,
                self.mpvit1.backbone.mhca_stages,
                self.mpvit2.backbone.mhca_stages,
            ):

                x1 = stage1(patch_stage1(x1))
                x2 = stage2(patch_stage2(x2))
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
            for name, patch_stage1, patch_stage2, stage1, stage2 in zip(
                self.stage_names,
                self.mpvit1.backbone.patch_embed_stages,
                self.mpvit2.backbone.patch_embed_stages,
                self.mpvit1.backbone.mhca_stages,
                self.mpvit2.backbone.mhca_stages,
            ):
                x1 = stage1(patch_stage1(x1))
                x2 = stage2(patch_stage2(x2))
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
def build_double_mpvit_backbone(cfg, input_shape1, input_shape2):
    mpvit1 = build_mpvit_backbone(cfg, input_channels=input_shape1.channels)
    mpvit2 = build_mpvit_backbone(cfg, input_channels=input_shape2.channels)

    out_features = cfg.MODEL.MPVIT.OUT_FEATURES

    fusion_steps = cfg.MODEL.BACKBONE.FUSION.STEPS
    attention = cfg.MODEL.BACKBONE.FUSION.ATTENTION
    relu = cfg.MODEL.BACKBONE.FUSION.RELU
    bias = cfg.MODEL.BACKBONE.FUSION.BIAS
    fuse_in = cfg.MODEL.BACKBONE.FUSION.FUSE_IN

    return DoubleMPViT(
        mpvit1,
        mpvit2,
        out_features=out_features,
        fusion_steps=fusion_steps,
        attention=attention,
        relu=relu,
        bias=bias,
        fuse_in=fuse_in,
    )


@BACKBONE_REGISTRY.register()
def build_double_mpvit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.INPUT.FORMAT == "RGBD" or cfg.INPUT.FORMAT == "RGBRD":
        bottom_up = build_double_mpvit_backbone(cfg, ShapeSpec(channels=3), ShapeSpec(channels=1))
    elif cfg.INPUT.FORMAT == "GN":
        bottom_up = build_double_mpvit_backbone(cfg, ShapeSpec(channels=1), ShapeSpec(channels=3))
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
