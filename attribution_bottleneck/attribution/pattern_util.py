"""
This code was provided by Pan Kessel <pan.kessel@gmail.com> and Maximilian Alber <work.alber.maximilian@gmail.com>.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                 ceil_mode=ceil_mode, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size, stride, padding)
        self.indices = None

    def forward(self, x):
        self.X = x

        output, self.indices = self.pool(x)

        return output

    def analyze(self, R):
        batch_size, channels, height, width = self.X.shape
        height = int(height / 2)
        width = int(width / 2)

        if R.shape != torch.Size([batch_size, channels, height, width]):
            R = R.view(batch_size, channels, height, width)

        return self.unpool(R, self.indices)


class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation_fn=False, data_mean=None, data_std=None):
        super(Convolutional, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.activation_fn = activation_fn

        self.X = 0
        self.pre_activation = 0

        if data_mean is not None and data_std is not None:
            self.lowest = float(np.min((0 - data_mean) / data_std))
            self.highest = float(np.max((1 - data_mean) / data_std))

        self.register_buffer('pattern', torch.zeros(self.conv.weight.shape))

        # initialize parameters
        nn.init.xavier_uniform_(self.conv.weight.data)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        self.X = x

        out = self.conv.forward(x)
        self.pre_activation = out

        if self.activation_fn:
            out = self.activation_fn(out)

        return out

    def analyze(self, R):
        # if previous layer was a dense layer, R needs to be reshaped
        # to the form of self.X after the convolution in the forward pass
        batch_size, _, height, width = self.pre_activation.shape

        if R.shape != torch.Size([batch_size, self.conv.out_channels, height, width]):
            R = R.view(batch_size, self.conv.out_channels, height, width)

        if self.activation_fn is not None:
            if hasattr(self.activation_fn, "beta"):
                R = R * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = R * (self.pre_activation >= 0).float()

        if self.pattern is None:
            raise RuntimeError('Pattern need to be set in order to use pattern attribution.')

        newR = self.deconvolve(R, self.conv.weight * self.pattern)

        return newR

    def deconvolve(self, y, weights):

        # dimensions before convolution in forward pass
        # the deconvolved image has to have the same dimension
        _, _, org_height, org_width = self.X.shape

        # stride and padding from forward convolution
        padding = self.conv.padding
        stride = self.conv.stride

        _, _, filter_height, filter_width = weights.shape

        # the deconvolved image has minimal size
        # to obtain an image with the same size as the image before the convolution in the forward pass
        # we pad the output of the deconvolution
        output_padding = ((org_height + 2 * padding[0] - filter_height) % stride[0],
                          (org_width + 2 * padding[1] - filter_width) % stride[1])  # a=(i+2p−k) mod s

        # perform actual deconvolution
        # this is basically a forward convolution with flipped (and permuted) filters/weights
        deconvolved = torch.nn.functional.conv_transpose2d(input=y, weight=weights, bias=None,
                                                           padding=self.conv.padding, stride=self.conv.stride,
                                                           groups=self.conv.groups, dilation=self.conv.dilation,
                                                           output_padding=output_padding)

        return deconvolved


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, activation_fn=None,
                 data_mean=0.0,
                 data_std=1.0):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation_fn = activation_fn

        self.X = 0
        self.pre_activation = 0

        self.lowest = np.min((0 - data_mean) / data_std)
        self.highest = np.max((1 - data_mean) / data_std)

        self.register_buffer('pattern', torch.zeros(out_dim, in_dim))

        # initialize parameters
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        self.X = x
        out = self.linear(x)
        self.pre_activation = out

        if self.activation_fn:
            out = self.activation_fn(out)

        return out

    def analyze(self, R):
        if self.pattern is None:
            raise RuntimeError('Pattern needs to be set in order to use pattern attribution.')

        weight = self.linear.weight * self.pattern

        if self.activation_fn is not None:
            if hasattr(self.activation_fn, "beta"):
                R = R * torch.sigmoid(self.activation_fn.beta * self.pre_activation)
            else:
                R = R * (self.pre_activation >= 0).float()

        newR = torch.matmul(R, weight)

        return newR


class ExplainableNet(nn.Module):
    def __init__(self, model=None, data_mean=0, data_std=1):
        super(ExplainableNet, self).__init__()

        self.activation_fn = F.relu

        self.layers = nn.ModuleList([])

        self.data_mean = data_mean
        self.data_std = data_std

        if model is not None:
            self.fill_layers(model)

        # remove activation function in last layer
        self.layers[-1].activation_fn = None

        self.R = 0

    def fill_layers(self, model):

        for layer in model.features:
            new_layer = self.create_layer(layer)
            if new_layer == 0:
                continue
            self.layers.append(new_layer)

        for layer in model.classifier:
            new_layer = self.create_layer(layer)
            if new_layer == 0:
                continue
            self.layers.append(new_layer)

    def create_layer(self, layer):
        if type(layer) == torch.nn.Conv2d:
            new_layer = Convolutional(in_channels=layer.in_channels,
                                      out_channels=layer.out_channels,
                                      kernel_size=layer.kernel_size,
                                      stride=layer.stride,
                                      padding=layer.padding,
                                      activation_fn=self.activation_fn,
                                      data_mean=self.data_mean,
                                      data_std=self.data_std)
            new_layer.conv.weight.data = layer.weight.data
            new_layer.conv.bias.data = layer.bias.data

        elif type(layer) == nn.MaxPool2d:
            new_layer = MaxPool(kernel_size=layer.kernel_size,
                                stride=layer.stride,
                                padding=layer.padding)
        elif type(layer) == nn.Linear:
            new_layer = Dense(in_dim=layer.in_features, out_dim=layer.out_features, activation_fn=self.activation_fn)
            new_layer.linear.weight.data = layer.weight.data
            new_layer.linear.bias.data = layer.bias.data

        elif type(layer) == (nn.Dropout or nn.Dropout2d):
            new_layer = layer

        elif type(layer) == nn.ReLU:
            return 0

        else:
            print('ERROR: unknown layer')
            return None

        return new_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        self.R = x

        return x

    def classify(self, x):
        outputs = self.forward(x)

        return F.softmax(outputs, dim=1), torch.max(outputs, 1)[1]

    def analyze(self, R=None, index=None):
        if R is None:
            R = self.R
        if index is not None:
            R = self.R.clone()
            indices = np.ones(1000).astype(bool)
            indices[index] = False
            indices = np.where(indices)[0]
            R[0][indices] = 0

        for layer in reversed(self.layers):
            if type(layer) == nn.Dropout or type(layer) == nn.Dropout2d:  # ignore Dropout layer
                continue
            R = layer.analyze(R)

        return R

