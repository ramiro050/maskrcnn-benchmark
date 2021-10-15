# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from torchvision.ops import nms as vision_nms

# Only valid with fp32 inputs - give AMP the hint
nms = vision_nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
