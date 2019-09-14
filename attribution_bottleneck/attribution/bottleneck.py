import torch.nn as nn
from tqdm import tqdm

from attribution.base import *
from attribution_bottleneck.evaluate.gui import TensorView
from information.fitted_bottleneck import BetaLoss, BottleneckAdaptive
from information.estimator import Estimator
from utils.misc import resize, replace_layer, to_np


class FittedBottleneckReader(AttributionMethod):

    def __init__(self, model, estim: Estimator, beta=10, steps=10, lr=1, batch_size=10, sigma=0.5, relu=False, mode="capacity"):
        self.model = model
        self.original_layer = estim.get_layer()
        self.shape = estim.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.mode = mode
        self.progbar = False
        self.show_steps = False
        self.device = list(model.parameters())[0].device
        self.lr = lr
        self.train_steps = steps
        self.bottleneck = BottleneckAdaptive(estim.mean(), estim.std(), device=self.device, sigma=sigma, relu=relu)
        self.sequential = nn.Sequential(self.original_layer, self.bottleneck)

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):

        self.run_training(input_t, target_t)

        return self.current_heatmap(shape=input_t.shape[2:])

    def run_training(self, input_t, target_t):
        # Attach layer and train the bottleneck
        replace_layer(self.model, self.original_layer, self.sequential)
        self.train_bottleneck(input_t, target_t)
        replace_layer(self.model, self.sequential, self.original_layer)

    def current_heatmap(self, shape=None):
        # Read bottleneck
        if self.mode == "a":
            heatmap = self.bottleneck.buffer_tensors["a"]
        elif self.mode == "capacity":
            heatmap = self.bottleneck.buffer_capacity
        elif self.mode == "mu":
            heatmap = self.bottleneck.buffer_mu
        else:
            raise ValueError(self.mode)

        heatmap = to_np(heatmap[0])
        heatmap = heatmap.sum(axis=0)  # Sum over channel dim
        heatmap = heatmap - heatmap.min()  # min=0
        heatmap = heatmap / heatmap.max()  # max=0

        if shape is not None:
            heatmap = resize(heatmap, shape)

        return heatmap

    def train_bottleneck(self, input_t: torch.Tensor, target_t: torch.Tensor):

        assert input_t.shape[0] == 1, "We can only fit on one sample"
        assert target_t.shape[0] == 1, "We can only fit on one label"

        batch = input_t.expand(self.batch_size, -1, -1, -1), target_t.expand(self.batch_size)
        optimizer = torch.optim.Adam(lr=self.lr, params=self.bottleneck.parameters())
        loss = BetaLoss(model=self.model, beta=self.beta)

        # Reset from previous run or modifications
        self.bottleneck.reset()

        # Train
        self.model.eval()
        for _ in tqdm(range(self.train_steps), desc="Training Bottleneck", disable=not self.progbar):
            optimizer.zero_grad()
            out = self.model(batch[0])
            loss_t = loss.calc_loss(b_labels=batch[1], b_outputs=out)
            loss_t.backward()
            optimizer.step(closure=None)

            if self.show_steps:
                TensorView(self.bottleneck.buffer_tensors["a"])
