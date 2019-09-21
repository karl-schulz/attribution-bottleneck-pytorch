from __future__ import print_function

import numpy as np
import torch.nn.functional as F
import torch.autograd
from scipy.ndimage import gaussian_filter

# from ..attribution.base import AttributionMethod
from attribution_bottleneck.attribution.backprop import ModifiedBackpropMethod
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image

from ..utils.baselines import Minimum, Mean, Baseline
from ..utils.misc import *


class SumMethod(ModifiedBackpropMethod):
    """
    Something than can make a attribution heatmap from several inputs and a attribution method
    The resulting heatmap is the sum of all the other methods

    It is also just a backprop method: the projection on the input space is the sum of the
    backpropagation over the samples.
    """

    def __init__(self, backprop: ModifiedBackpropMethod, cc_transforms):
        super().__init__(cc_transforms=cc_transforms)
        self.verbose = False
        self.progbar = False
        self.backprop = backprop

    def backpropagate(self, input_t: torch.Tensor, target_t: torch.Tensor):
        # generate sample list (different per method)
        images = self._get_samples(input_t)

        assert isinstance(target_t, torch.Tensor)
        assert len(images[0].shape) == 4, "{} makes dim {} !".format(images[0].shape, len(images[0].shape))  # C x N x N

        grads = self._backpropagate_multiple(images, target_t)

        grad_mean = np.mean(grads, axis=0)

        return grad_mean

    def _backpropagate_multiple(self, inputs: list, target_t: torch.Tensor):
        """
        returns an array with all the computed gradients
        shape: N_Inputs x Batches=1 x Color Channels x Height x Width
        """
        grads = np.zeros((len(inputs), *inputs[0].shape))
        it = tqdm(range(len(inputs)), ncols=100, desc="calc grad") if len(inputs) > 1 and self.progbar else range(len(inputs))

        for i in it:
            grad = self.backprop.backpropagate(input_t=inputs[i], target_t=target_t)

            # add color dimension
            if len(grad.shape) == 3:
                np.expand_dims(grad, axis=0)

            grads[i, :, :, :, :] = grad

        return grads

    def _get_samples(self, img_t: torch.Tensor) -> list:
        """ yield the samples to analyse """
        raise NotImplementedError


class SmoothGrad(SumMethod):
    def __init__(self, backprop: ModifiedBackpropMethod, std, steps=50, cc_transforms=None):
        cc_transforms = cc_transforms if cc_transforms is not None else ["abs", "max"]
        super().__init__(backprop=backprop, cc_transforms=cc_transforms)
        self.std = std
        self.steps = steps

    def _get_samples(self, img_t: torch.Tensor) -> list:
        relative_std = (img_t.max().item() - img_t.min().item()) * self.std
        noises = [torch.randn(*img_t.shape).to(img_t.device) * relative_std for _ in range(0, self.steps)]
        noise_images = [img_t + noises[i] for i in range(0, self.steps)]

        # maybe clamp ?
        # noise_images = [torch.clamp(img, 0, 1) for img in noise_images]

        return noise_images


class IntegratedGradients(SumMethod):

    def __init__(self, backprop: ModifiedBackpropMethod, baseline: Baseline = None, steps=50, cc_transforms=None):
        """
        :param baseline: start point for interpolation (0-1 grey, or "inv", or "avg")
        :param steps: resolution
        :param cc_transforms: how to evaluate the color channel gradients
        """
        cc_transforms = cc_transforms if cc_transforms is not None else ["abs", "max"]
        super().__init__(backprop=backprop, cc_transforms=cc_transforms)
        self.baseline = baseline if baseline is not None else Mean()
        self.steps = steps

    def _get_samples(self, img_t: torch.Tensor) -> np.array:
        bl_img = self.baseline.apply(to_np_img(img_t))
        bl_img_t = to_img_tensor(bl_img, device=img_t.device)
        return [((i / self.steps) * (img_t - bl_img_t) + bl_img_t) for i in range(1, self.steps + 1)]

