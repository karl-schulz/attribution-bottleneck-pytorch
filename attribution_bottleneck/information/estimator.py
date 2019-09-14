from math import inf

import torch
from tqdm import tqdm
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from attribution_bottleneck.evaluate.tensorview import TensorView
from utils.misc import to_np

class Estimator:

    def get_layer(self):
        raise NotImplementedError

    def shape(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def std(self, stabilize=True):
        raise NotImplementedError

    def p_zero(self):
        raise NotImplementedError

    def normalize(self, z):
        raise NotImplementedError


class ReluEstimator(Estimator):
    """ is fed single points of any-dimensional data and computes the running mean and std per cell """
    def __init__(self, layer):
        self.layer = layer
        self.M = None  # running mean for each entry
        self.S = None  # running std for each entry
        self.N = None  # running num_seen for each entry
        self.num_seen = 0  # total samples seen
        self.relu = True
        self.eps = 1e-5

    def feed(self, z: np.ndarray):

        # Maybe initialize
        if self.N is None:
            self.M = np.zeros_like(z, dtype=float)
            self.S = np.zeros_like(z, dtype=float)
            self.N = np.zeros_like(z, dtype=float)

        self.num_seen += 1

        if self.relu:
            nz_idx = z.nonzero()
            self.N[nz_idx] += 1
            self.S[nz_idx] += z[nz_idx] * z[nz_idx]

        else:
            diff = (z - self.M)
            self.N += 1
            self.M += diff / self.num_seen
            self.S += diff * (z - self.M)

    def feed_batch(self, batch: np.ndarray):
        for point in batch:
            self.feed(point)

    def shape(self):
        return self.M.shape

    def is_complete(self):
        return self.num_seen > 0 and (not self.relu or np.all(self.N))

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M

    def p_zero(self):
        return 1 - self.N / (self.num_seen + 1)  # Adding 1 for stablility, so that p_zero > 0 everywhere

    def std(self, stabilize=True):
        if stabilize:
            # Add small numbers, so that dead neurons are not a problem
            return np.sqrt(np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0))

        else:
            return np.sqrt(self.S / self.N)

    def estimate_density(self, z):
        z_norm = (z - self.mean()) / self.std()
        p = norm.pdf(z_norm, 0, 1)
        return p

    def normalize(self, z):
        return (z - self.mean()) / self.std()

    def estimate_probability(self, z):
        """ return a np.ndarray of probability densities of the same shape as z """
        if self.relu and not np.all(self.N[z.nonzero()]):
            raise RuntimeError(
                "cannot estimate: not every non-zero z has a non-zero non-zero observation count. "
                "{} missing.".format((self.N[z.nonzero()] == 0).sum()))

        z_norm = self.normalize(z)
        p = np.zeros_like(self.M)

        bins = 20
        std_range = 3
        borders = np.linspace(-std_range, std_range, bins-1, endpoint=True)
        cdf = [0, *norm.cdf(borders, 0, 1), 1.0]
        quantities = [cdf[i+1] - cdf[i] for i in range(bins)]

        for i, quantity in enumerate(quantities):
            lower_bound = -inf if i == 0 else borders[i-1]
            upper_bound = inf if i == len(borders) else borders[i]
            p[np.where((z_norm >= lower_bound) & (z_norm < upper_bound))] = quantity

        if self.relu:
            zero_idx = z == 0
            p_zero = self.p_zero()
            # Scale nonzeros up, as the left side of the gaussian has to have integral 1
            p = p * (1 - p_zero)
            # Set pz
            p[zero_idx] = p_zero[zero_idx]

        return p

    def test_mean_std(self):
        """ Show the mean/std tensors """
        TensorView(self.mean(), title="mean")
        TensorView(self.std(True), title="std")

    def test_probability_distribution(self):
        """ Plot the graph of p: x -> p(x), as well as the coresponding information """
        m, s = self.mean(), self.std()
        pz = self.p_zero()
        print(pz)

        if self.relu:
            z = np.linspace(m+5*s, 0, 500, endpoint=False)

            # Plot zero probability
            xz = [np.zeros_like(pz), np.zeros_like(pz)]
            yz = [np.zeros_like(pz), pz]
            plt.plot(xz, yz, marker="o", linestyle=':', markevery=[1])
            plt.ylim(bottom=0)

        else:
            z = np.linspace(m-5*s, m+5*s, 500)

        # Plot z probability
        p = [self.estimate_probability(zi) for zi in z]
        plt.plot(z, p)
        plt.xlabel("z")
        plt.ylabel("Probability (Binarized)")
        plt.ylim(bottom=0)
        plt.show()

        # Plot z information
        plt.plot(z, -np.log(p))
        plt.xlabel("z")
        plt.ylabel("Information (bits)")
        plt.ylim(bottom=0)

        if self.relu:
            # Plot zero information
            xz = [np.zeros_like(pz)]
            yz = [pz]
            plt.plot(xz, -np.log(yz), marker="o", linestyle=':')
            plt.ylim(bottom=0)
        plt.show()

    def test_probability_tensors(self, z, z_norm):
        """ Plot the probability maps of a certain tensor of latent variables """
        TensorView(self.std(), title="stds")
        TensorView(z, title="z")
        TensorView(z_norm, title="z_norm")
        TensorView(norm.pdf(z_norm, 0, 1), title="p")
        TensorView(-norm.logpdf(z_norm, 0, 1), title="i")

    def test_histograms(self, z):

        z_norm = (z - self.mean()) / self.std()
        p = norm.pdf(z_norm, 0, 1)

        plt.hist(z_norm.ravel(), bins=100)
        plt.title("z_norm")
        plt.show()

        plt.hist(p.ravel(), bins=100)
        plt.title("p_norm")
        plt.show()

        plt.hist(z.ravel(), bins=100)
        plt.title("z")
        plt.show()

        plt.hist(self.std().ravel(), bins=100)
        plt.title("std")
        plt.show()

        plt.hist(self.mean().ravel(), bins=100)
        plt.title("mean")
        plt.show()

class GaussianEstimator(ReluEstimator):
    def __init__(self, layer):
        super().__init__(layer)
        self.relu = False

class EstimatorGroup:
    def __init__(self, model, estimators, data_gen=None):
        self.model = model
        self.estimators = estimators
        if data_gen:
            self.feed(data_gen)

    @staticmethod
    def auto(model, layers, data_gen=None):
        estimators = []
        for l in layers:
            if isinstance(l, nn.ReLU) or (isinstance(l, nn.Sequential) and isinstance(l[-1], nn.ReLU)):
                print("ReluEstimator for "+l.__class__.__name__)
                estimators.append(ReluEstimator(l))
            else:
                print("GaussianEstimator for "+l.__class__.__name__)
                estimators.append(GaussianEstimator(l))
        group = EstimatorGroup(model, estimators=estimators, data_gen=data_gen)
        return group

    def _make_feed_hook(self, i):
        def hook(m, x, z):
            self.estimators[i].feed_batch(to_np(z))
        return hook

    def _make_estimate_hook(self, estimates, i):
        def hook(m, x, z):
            estimates[i] = self.estimators[i].estimate_probability(to_np(z).squeeze())
        return hook

    def feed(self, gen):
        print("Feeding estimator from generator...")
        hook_handles = [e.layer.register_forward_hook(self._make_feed_hook(i)) for i, e in enumerate(self.estimators)]

        for batch, labels in tqdm(gen):
            self.model(batch)

        for handle in hook_handles:
            handle.remove()

    def save(self, path):
        torch.save({
            "estimators": [{
                "class": e.__class__.__name__,
                "layer_class": e.layer.__class__.__name__,
                "N": e.N,
                "M": e.M,
                "S": e.S,
                "num_seen": e.num_seen,
            } for e in self.estimators]
        }, path)

    def load(self, path):
        state = torch.load(path)
        assert len(self.estimators) == len(state["estimators"])
        for e, state in zip(self.estimators, state["estimators"]):
            if e.__class__.__name__ != state["class"]:
                raise RuntimeError("Estimator is {}, but not {}".format(e.__class__.__name__, state["class"]))
            if e.layer.__class__.__name__ != state["layer_class"]:
                raise RuntimeError("Layer is {}, but not {}".format(e.layer.__class__.__name__, state["layer_class"]))
            e.N = state["N"]
            e.S = state["S"]
            e.M = state["M"]
            e.num_seen = state["num_seen"]

    def stds(self, stabilize=True):
        return [e.std(stabilize) for e in self.estimators]

    def means(self):
        return [e.mean()for e in self.estimators]

    def shapes(self):
        return [e.shape for e in self.estimators]

    def layers(self):
        return [e.layer for e in self.estimators]

    def estimate(self, x_t):

        estimates = [None for _ in self.estimators]
        hook_handles = [e.layer.register_forward_hook(self._make_estimate_hook(estimates, i)) for i, e in enumerate(self.estimators)]

        self.model(x_t)

        for handle in hook_handles:
            handle.remove()

        return estimates
