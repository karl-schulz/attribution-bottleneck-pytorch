import torch.nn as nn
from attribution.base import *
from information.readout_bottleneck import ReadoutBottleneck
from utils.misc import resize, replace_layer, to_np


class ReadoutBottleneckReader(AttributionMethod):

    def __init__(self, model, target_layer, bn_layer: ReadoutBottleneck):
        super().__init__()

        self.bn_layer = bn_layer
        self.target_layer = target_layer
        self.sequential = nn.Sequential(target_layer, bn_layer)
        self.model = model

    def _inject_bottleneck(self):
        replace_layer(self.model, self.target_layer, self.sequential)
        self.bn_layer.active = True

    def _remove_bottleneck(self):
        replace_layer(self.model, self.sequential, self.target_layer)
        self.bn_layer.active = False

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):

        self._inject_bottleneck()
        with torch.no_grad():
            self.model(input_t)
        self._remove_bottleneck()

        htensor = to_np(self.bn_layer.buffer_capacity)

        hmap = htensor.mean(axis=(0,1))
        hmap = resize(hmap, input_t.shape[2:])

        hmap = hmap - hmap.min()

        return hmap