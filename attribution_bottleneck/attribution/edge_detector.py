
from attribution.base import *
import torch
from scipy.misc import imresize
import numpy as np
from skimage import feature
from scipy import ndimage
import math
from utils.misc import *


class EdgeDetecor(AttributionMethod):

    def __init__(self):
        self.sigma_relative = 1
        self.canny_sigma = 3
        self.blur_sigma = 5

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):
        img = to_np_img(input_t)
        img = ndimage.gaussian_filter(img, sigma=2)
        edges =   feature.canny(img[:, :, 0], sigma=self.canny_sigma, ).astype(np.double) \
                + feature.canny(img[:, :, 1], sigma=self.canny_sigma).astype(np.double) \
                + feature.canny(img[:, :, 2], sigma=self.canny_sigma).astype(np.double)
        edges_t = torch.tensor(edges, device=input_t.device)
        kernel_size = input_t.shape[-2]
        sigma = kernel_size * self.sigma_relative
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.double, device=input_t.device)  # 1, 2, 3, 4
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)  # 1, 2, 3 \ 1, 2, 3 \ 1, 2, 3
        y_grid = x_grid.t()  # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d /= kernel_2d.max()
        hmap = edges_t * kernel_2d
        hmap = to_np(hmap)
        #hmap = ndimage.gaussian_filter(hmap, sigma=10)
        hmap = hmap / hmap.max()
        return hmap

