# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_backbone, BACKBONE_REGISTRY  # noqa F401 isort:skip

from .backbone import Backbone
from .fpn import FPN, DFPN
from .regnet import RegNet
from .resnet import (
    BasicStem,
    ResNet,
    ResNetBlockBase,
    build_resnet_backbone,
    make_stage,
    BottleneckBlock,
)
from .double_resnet import build_double_resnet_backbone
from .MPViT import build_mpvit_double_fpn_backbone, build_mpvit_fpn_backbone, build_double_mpvit_fpn_backbone
from .MPViT.double_mpvit_early_fusion import build_double_mpvit_fpn_backbone
from .vit import ViT, SimpleFeaturePyramid, get_vit_lr_decay_rate
from .mvit import MViT
from .swin import SwinTransformer

__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
