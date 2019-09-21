"""
Wrapper for PatternAttribution.
"""

import torch
import numpy as np
from attribution_bottleneck.attribution.base import AttributionMethod
from attribution_bottleneck.attribution.pattern_util import ExplainableNet
from attribution_bottleneck.utils.misc import to_np


class PatternAttribution(AttributionMethod):

    def __init__(self, vgg_model: torch.nn.Module, pattern_weights=None):
        """
        :param model: has to be a VGG variant
        """
        assert hasattr(vgg_model, "features") and hasattr(vgg_model, "classifier"), \
            "PatternAttribution currently requires VGG-16 network"
        data_mean = np.array([0.485, 0.456, 0.406])
        data_std = np.array([0.229, 0.224, 0.225])
        self.vgg_model = vgg_model
        self.device = next(iter(vgg_model.parameters())).device
        self.explainable_model = ExplainableNet(
            vgg_model, data_mean=data_mean, data_std=data_std).to(self.device)

        # load patterns for pattern attribution
        default_pattern = 'pretrained/pattern_vgg16/model_vgg16_pattern_small.pth'
        self.data_location = (pattern_weights or default_pattern)
        self.explainable_model.load_state_dict(
            torch.load(self.data_location, map_location=self.device), strict=False)

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):

        acc, class_idx = self.explainable_model.classify(input_t)

        # analyse with pattern attribution
        heatmap = self.explainable_model.analyze(R=None, index=class_idx)
        heatmap = torch.sum(torch.abs(heatmap), dim=1).squeeze()
        heatmap = heatmap / torch.max(heatmap)

        return to_np(heatmap)
