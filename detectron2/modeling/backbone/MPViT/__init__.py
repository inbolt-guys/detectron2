# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------

from .config import add_mpvit_config
from .backbone import build_mpvit_fpn_backbone, build_mpvit_backbone, MPViT_Backbone, build_mpvit_double_fpn_backbone
from .double_mpvit_early_fusion import build_double_mpvit_fpn_backbone
from .dataset_mapper import DetrDatasetMapper