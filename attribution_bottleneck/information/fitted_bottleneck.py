import torch
import torch.nn as nn
import numpy as np

from attribution_bottleneck.evaluate.tensorview import TensorView
from information.estimator import EstimatorGroup
from models.layers import SpatialGaussianKernel
from train.loss import Loss, CrossEntropyLoss, Accuracy


class KLStdNormalBottleneck(nn.Module):
    """
    Bottleneck KL[Q(z|x)||P(z)] where P(z) = N(0, 1) and Q(z|x) = N(mu(x), log_noise_var(x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, log_noise_var, force_noise=False):
        """ return mu with additive noise """
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        if self.training or force_noise:
            return mu + noise_std * eps
        else:
            return mu

    def capacity(self, mu, log_var) -> torch.Tensor:
        # KL[Q(z|x)||P(z)]
        # 0.5 * (tr(noise_cov) + mu ^ T mu - k  -  log det(noise)
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

        self.buffer_capacity = None  # filled on forward pass, used for loss
        self.buffer_mu = None  # filled on forward pass
        self.buffer_log_var = None  # filled on forward pass
        self.buffer_tensors = {}

    def _buffer(self, cap, mu, log_var):
        self.buffer_capacity = cap
        self.buffer_mu = mu
        self.buffer_log_var = log_var

    def test_visualize(self):
        if self.buffer_capacity is None:
            print("no buffers set yet, pass something with buffer_flow = True")
            return

        all_plots = {**{
            "Mu": self.buffer_mu,
            "Var": self.buffer_log_var.exp(),
            "Capacity": self.buffer_capacity,
        }, **self.buffer_tensors}

        for name, ele in all_plots.items():
            TensorView(ele, title=name)

class BottleneckAdaptive(Bottleneck):
    def __init__(self, mean: np.ndarray, std: np.ndarray, sigma=None, device=None, relu=False):
        super().__init__()
        self.bottleneck = KLStdNormalBottleneck()
        self.device = device
        self.relu = relu
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.store_buffers = False
        self.normalize_empirically = True
        self.normalize_sample = False

        if sigma is not None:

            # Construct static conv layer with gaussian kernel
            kernel_size = int(round(2 * sigma)) * 2 + 1  # Cover 2.5 stds in both directions
            channels = self.mean.shape[0]

            self.smooth = SpatialGaussianKernel(kernel_size, sigma, channels, device=self.device)
        else:
            self.smooth = None

        self.reset()

    def reset(self):
        """ Used to reset the mask to train on another sample """
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x):

        # Clamp, smoothen and fit a
        a_raw = self.sigmoid(self.alpha)
        a_fit = a_raw.expand(x.shape[0], x.shape[1], -1, -1)
        a = self.smooth(a_fit) if self.smooth is not None else a_fit

        # Normalize x
        x_norm = (x - self.mean) / self.std

        # Get parameters
        mu, log_var = x_norm * a, torch.log(torch.sqrt(1-a**2))
        #mu, log_var = x_norm * a, torch.log(1-a)

        # Calc BN
        z_norm = self.bottleneck(mu, log_var, True)
        cap = self.bottleneck.capacity(mu, log_var)

        # Denormalize x
        z = z_norm * self.std + self.mean

        # Maybe relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        if self.store_buffers:
            self.buffer_tensors["x"] = x.clone()
            self.buffer_tensors["x_norm"] = x_norm.clone()
            self.buffer_tensors["xz_norm"] = (x_norm - z_norm).clone()
            self.buffer_tensors["a_fit"] = a_fit.clone()
            self.buffer_tensors["a"] = a.clone()
            self.buffer_tensors["z_norm"] = z_norm.clone()
            self.buffer_tensors["z"] = z.clone()
        self._buffer(mu=mu, log_var=log_var, cap=cap)

        return z

class BetaLoss(Loss):
    def __init__(self, model, beta, target=None):
        self.cross_entropy = CrossEntropyLoss(model)
        self.target = target
        self.beta = beta
        self.model = model
        self.bn_layers = []
        self.top5 = Accuracy(5)

        def collect_bn_layers(module):
            # TODO we could also walk the module on calc_loss
            if isinstance(module, Bottleneck):
                self.bn_layers.append(module)

        self.model.apply(collect_bn_layers)
        # print("Observing {} bottleneck layers".format(len(self.bn_layers)))

    def calc_kl_loss(self):
        return self.beta * torch.stack(self.calc_kls()).sum()

    def calc_kls(self):
        return [torch.mean(bn_layer.buffer_capacity) for bn_layer in self.bn_layers]

    def calc_metrics(self, b_labels, b_outputs):

        # first term: the normal cross entropy
        cross_entropy = self.cross_entropy.calc_loss(b_labels=b_labels, b_outputs=b_outputs)

        # collect the KL terms of all the bottleneck layers
        # TODO account for mean vs. sum effect
        # TODO separate beta for each layer?
        mus = [torch.mean(bn_layer.buffer_mu) for bn_layer in self.bn_layers]
        vars = [torch.mean(bn_layer.buffer_log_var.exp()) for bn_layer in self.bn_layers]
        kls = self.calc_kls()

        # combine loss together
        kl_loss = self.calc_kl_loss()
        total_loss = cross_entropy + kl_loss

        # return dict with all infos
        metrics = {
            "loss": total_loss,
            "cross_entropy": cross_entropy,
            "KL_loss": kl_loss,
            'accuracy-top5': self.top5(b_labels, b_outputs),
            **{"KL_{}".format(i): v for i, v in enumerate(kls)},
            **{"mu_{}".format(i): v for i, v in enumerate(mus)},
            **{"var_{}".format(i): v for i, v in enumerate(vars)},
        }
        if self.target is not None:
            metrics["logit_score"] = b_outputs[:,self.target].mean()  # average score per sample
        return metrics

    def calc_loss(self, b_labels, b_outputs):
        # More performant than calc_metrics()["loss"]
        return self.cross_entropy.calc_loss(b_labels=b_labels, b_outputs=b_outputs) + self.calc_kl_loss()

    def test(self, sample=None, *args, **kwargs):

        if sample is not None:
            # pass to fill buffers
            self.model(sample)

        for i, bn_layer in enumerate(self.bn_layers):
            print("BN Layer #{}".format(i))
            bn_layer.test_visualize(*args, **kwargs)

    def calc_heatmap(self, img_t, target_t):
        pass