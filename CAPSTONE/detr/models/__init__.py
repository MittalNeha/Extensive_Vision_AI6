# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .merge_segmentation import build as build_merge_segmentation


def build_model(args):
    return build(args)
