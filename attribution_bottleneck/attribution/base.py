import torch
import numpy as np


class AttributionMethod:
    """ Something than can make a attribution heatmap """

    def heatmap(self, input_t: torch.Tensor, target_t) -> np.ndarray:
        """
        Generate a attribution map. If the model is modified in the meantime, it is restored after the attribution.
        :param input_t: The input (image) as a BxHxWxC torch tensor with B=1. Already transformed to be passed to the model.
        :param target_t: A target label as 1x1 tensor, indicating the correct label. Alternatively an integer.
        :return: A HxW numpy array representing the resulting heatmap.
        """
        raise NotImplementedError


class Random(AttributionMethod):
    """ random heatmap from -1 to 1, no duplicate values. sum ~ 0 """
    def __init__(self, vmin=-1, vmax=1):
        self.vmin = vmin
        self.vmax = vmax

    def heatmap(self, input_t, target_t):
        shape = input_t[0, 0].shape
        hmap = np.linspace(start=self.vmin, stop=self.vmax, num=shape[0] * shape[1])
        np.random.shuffle(hmap)
        hmap = hmap.reshape((shape[0], shape[1]))
        return hmap


class Zero(AttributionMethod):
    """ random heatmap from -1 to 1, no duplicate values. sum ~ 0 """

    def heatmap(self, input_t, target_t):
        shape = input_t[0, 0].shape
        hmap = np.zeros((shape[0], shape[1]))
        return hmap


