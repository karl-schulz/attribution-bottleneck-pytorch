import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from attribution_bottleneck.bottleneck.gaussian_kernel import SpatialGaussianKernel
from attribution_bottleneck.utils.misc import replace_layer

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot


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
        mu, log_var = x_norm * a, torch.log(1-a)

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


class ReadoutBottleneck(Bottleneck):
    """ Base class for readout bottlenecks - handles the logic for the readout pass """
    def __init__(self, model, readout_layers: list):
        super().__init__()

        self.model = [model]
        self.device = next(model.parameters()).device
        self.readout_layers = readout_layers
        self.bottleneck = KLStdNormalBottleneck()

        self.input = None  # the last input tensor (image)
        self.observed_acts = []  # the last intermediate activations
        self.forward_hooks = []  # registered hooks
        self.input_hook = None  # registered hooks
        self.is_nested_pass = False
        self.active = True
        self.attach_hooks()

    def attach_hooks(self):
        """ attach hooks """
        def forward_hook(m, t_in, t_out):
            if self.active and self.is_nested_pass:
                # print("Recording act with shape {} from {}".format(t_out.shape, m.__class__.__name__))
                self.observed_acts.append(t_out.clone())

        def input_hook(m, t_in):
            if self.active and not self.is_nested_pass:
                # print("Captured input: {}".format(t_in[0].shape))
                self.input = t_in[0].clone()

        # attach hooks to intermediate layers
        for m in self.readout_layers:
            self.forward_hooks.append(m.register_forward_hook(forward_hook))

        # attach hook to model
        self.input_hook = self.model[0].register_forward_pre_hook(input_hook)

    def detach_hooks(self):
        """ detach hooks """
        for h in self.forward_hooks:
            h.remove()

        self.input_hook.remove()

        self.forward_hooks = []
        self.input_hook = None

    def forward(self, x_in):

        if self.is_nested_pass:
            return x_in

        assert self.input is not None, "no input registered - activated?"

        # clear memory
        self.observed_acts = []

        with autograd.no_grad():
            # Pass input again and collect readout
            self.is_nested_pass = True
            self.model[0](self.input)
            self.is_nested_pass = False

        # ok done with refeeding. now use stuff

        return self.forward_augmented(x_in, self.observed_acts)

    def forward_augmented(self, x_in, readouts):
        raise NotImplementedError

class AdaptiveReadoutBottleneck(ReadoutBottleneck):
    """ a bottleneck which noises with emprical dists """
    def __init__(self, model, layers, means, stds, kernel_size, relu=False):
        super().__init__(model, layers)
        self.device = next(model.parameters()).device
        self.feat_out = means[0].shape[0]
        self.relu = relu
        self.feat_in = 0
        self.limit_value = 5.0
        self.store_buffers = False
        self.sigmoid = nn.Sigmoid()

        for i, (mean, std) in enumerate(zip(means, stds)):
            self.register_buffer("std_{}".format(i), torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False))
            self.register_buffer("mean_{}".format(i), torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False))
            self.feat_in += mean.shape[0]

        self.kernel_size = kernel_size
        if kernel_size is not None:
            # Construct static conv layer with gaussian kernel
            sigma = kernel_size * 0.25  # Cover 2 stds in both directions
            channels = self.std_0.shape[0]
            self.smooth = [SpatialGaussianKernel(kernel_size, sigma, channels, self.device)]
        else:
            self.smooth = None

    def attach(self, model, layer):
        replace_layer(model, layer, nn.Sequential(layer, self))
        return self

    def infer_mask(self, readout):
        raise NotImplementedError

    def forward_augmented(self, x, readouts):

        # Preprocess readout
        target_shape = x.shape[-2:]
        buffers = dict(self.named_buffers())
        # TODO substract mean
        readouts = [r / buffers["std_{}".format(i)] for i, r in enumerate(readouts)]
        readouts = [r.unsqueeze(-1).unsqueeze(-1).expand(*r.shape, *target_shape[-2:]) if len(r.shape) == 2 else r for r in readouts]
        readouts = [F.interpolate(input=r, size=target_shape, mode="bilinear", align_corners=True) for r in readouts]

        # Stack readout
        readout = torch.cat(readouts, dim=1)

        # Pass through net to obtain mask
        f = self.infer_mask(readout)
        f = torch.clamp(f, -self.limit_value, self.limit_value)
        a_raw = self.sigmoid(f)

        # Postprocess map
        a_fit = a_raw.expand(-1, x.shape[1], -1, -1)
        a = self.smooth[0](a_fit) if self.smooth is not None else a_fit

        # Normalize x
        x_norm = (x - self.mean_0) / self.std_0
        mu, log_var = x_norm * a, torch.log(torch.sqrt(1-a**2))
        #mu, log_var = x_norm * a, torch.log(1-a)

        # Calc BN
        z_norm = self.bottleneck(mu, log_var, True)
        cap = self.bottleneck.capacity(a, log_var)

        # Denormalize x
        z = z_norm * self.std_0 + self.mean_0

        # Maybe relu
        if self.relu:
            z = torch.clamp(z, 0.0)

        # Write buffers
        if self.store_buffers:
            self.buffer_tensors["readout"] = readout.clone()
            self.buffer_tensors["x"] = x.clone()
            self.buffer_tensors["x_norm"] = x_norm.clone()
            self.buffer_tensors["a_raw"] = a_raw.clone()
            self.buffer_tensors["a"] = a.clone()
            self.buffer_tensors["z_norm"] = z_norm.clone()
            self.buffer_tensors["z"] = z.clone()
        self._buffer(cap, a, log_var)

        return z

    def shapes(self):
        buffers = dict(self.named_buffers())
        return [buffers["std_{}".format(i)].shape for i in range(len(self.readout_layers))]

    def make_save_action(self):
        def tmp_cb(path):
            print("Saving only bottleneck state to {}".format(path))
            return self.save(path)
        return tmp_cb

    def save(self, path):
        """ Save the state of the bottleneck to restore it later without an estimator rerun """
        state = {
            "model_state": self.state_dict(),
            "shapes": self.shapes(),
            "kernel_size": self.kernel_size,
        }
        return torch.save(state, path)

    @classmethod
    def reduce_checkpoint(cls, model, layers, shapes, kernel_size, path):
        state = torch.load(path)
        bn_dict = {
            name: state["model_state"]["layer1.1.layer1.1."+name] for name in
            ["mean_0", "mean_1", "mean_2", "mean_3", "mean_4", "std_0", "std_1",
             "std_2", "std_3", "std_4", "conv.bias", "conv.weight"]
        }
        bottleneck = cls.load(model, layers, bn_dict, shapes, kernel_size)
        bottleneck.save(path)
        return

    @classmethod
    def load(cls, model, layers, layer_state, shapes, kernel_size):
        means = [np.zeros(s) for s in shapes]
        stds = [np.ones(s) for s in shapes]
        # Initialize with dummy mean/stds
        bottleneck = cls(model, layers, means, stds, kernel_size).to(list(model.parameters())[0].device)
        # Override with saved state
        bottleneck.load_state_dict(layer_state)
        return bottleneck

    @classmethod
    def load_path(cls, model, layers, path):
        """ Load a bottleneck from a file """
        device = next(model.parameters()).device
        state = torch.load(path, map_location=device)
        return cls.load(model, layers, state["model_state"], state["shapes"], state["kernel_size"])


class ShallowAdaptiveReadoutBottleneck(AdaptiveReadoutBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=self.feat_in, out_channels=self.feat_out, kernel_size=1)

        with torch.no_grad():
            # Initialize with identity mapping
            nn.init.constant_(self.conv.bias, 5.0)
            self.conv.weight *= 1e-3

    def infer_mask(self, readout):
        f = readout
        f = self.conv(f)
        return f

class MediumDenseAdaptiveReadoutBottleneck(AdaptiveReadoutBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=self.feat_in, out_channels=self.feat_in//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.feat_in//2, out_channels=self.feat_out, kernel_size=1)

        with torch.no_grad():
            # Initialize with identity mapping
            nn.init.constant_(self.conv2.bias, 5.0)
            self.conv2.weight *= 1e-3

    def infer_mask(self, readout):
        f = readout
        f = self.conv1(f)
        f = self.relu(f)
        f = self.conv2(f)
        return f

class DenseAdaptiveReadoutBottleneck(AdaptiveReadoutBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=self.feat_in, out_channels=self.feat_in//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.feat_in//2, out_channels=self.feat_out//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.feat_out//4, out_channels=self.feat_out, kernel_size=1)

        with torch.no_grad():
            # Initialize with identity mapping
            nn.init.constant_(self.conv3.bias, 5.0)
            self.conv3.weight *= 1e-3

    def infer_mask(self, readout):
        f = readout
        f = self.conv1(f)
        f = self.relu(f)
        f = self.conv2(f)
        f = self.relu(f)
        f = self.conv3(f)
        return f

class OldDenseAdaptiveReadoutBottleneck(AdaptiveReadoutBottleneck):
    """ By error some models were trained with this architecture """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels=self.feat_in, out_channels=self.feat_in//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.feat_in//2, out_channels=self.feat_out*2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.feat_out*2, out_channels=self.feat_out, kernel_size=1)

        with torch.no_grad():
            # Initialize with identity mapping
            nn.init.constant_(self.conv3.bias, 5.0)
            self.conv3.weight *= 1e-3

    def infer_mask(self, readout):
        f = readout
        f = self.conv1(f)
        f = self.relu(f)
        f = self.conv2(f)
        f = self.relu(f)
        f = self.conv3(f)
        return f
