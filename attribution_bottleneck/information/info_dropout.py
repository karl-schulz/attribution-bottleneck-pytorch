import torch.nn as nn
import torch
import numpy as np
from utils.misc import *
"""
contains classes:
 - network
 - ID layer
 - ID loss
"""
class AllCNN96_InfoDropout(nn.Module):
    """
    All-CNN-96 from https://arxiv.org/pdf/1611.01353.pdf
    (without softmax layer)
    """
    def __init__(self, config):
        super().__init__()
        self.beta = config.get("beta")
        # 96x96
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        self.info1 = InformationDropoutLayer(32, 64, kernel_size=3, padding=1, stride=2)
        # 48x48
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(96),
            # nn.ReLU(),
        )
        self.info2 = InformationDropoutLayer(64, 96, kernel_size=3, padding=1, stride=2)
        # 24x24
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            # nn.Conv2d(96, 192, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(192),
            # nn.ReLU(),
        )
        self.info3 = InformationDropoutLayer(96, 192, kernel_size=3, padding=1, stride=2)
        # 12x12
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            # nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(192),
            # nn.ReLU(),
        )
        self.info4 = InformationDropoutLayer(192, 192, kernel_size=3, padding=1, stride=2)
        # 6x6
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        # 1x1
        self.avg_pool = nn.AvgPool2d(kernel_size=6)

    @staticmethod
    def family() -> str:
        return "All-CNN-96 InfoDropout"

    def get_info_layers(self):
        return [self.info1, self.info2, self.info3, self.info4]

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.info1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.info2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.info3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.info4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.avg_pool(x)
        # print(x.shape)
        x = x.view(-1, 10)
        # print("STOP")
        return x


class InformationDropoutLayer(nn.Module):
    def __init__(self, f_in: int, f_out: int, kernel_size: int=3, padding: int=1, stride: int=2):
        super().__init__()
        self.stride = 2
        self.max_alpha = 0.7  # 0.7
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(f_out)
        self.conv = nn.Conv2d(f_in, f_out, padding=padding, kernel_size=kernel_size, stride=stride)
        self.alpha_conv = nn.Conv2d(f_in, f_out, padding=padding, kernel_size=kernel_size, stride=stride)
        self.alpha_act = nn.Sigmoid()
        self.alpha_bn = nn.BatchNorm2d(f_out)
        self.avg_kl = None
        self.last_alpha = None
        self.last_kls = None

    @staticmethod
    def sample_lognormal(means, sigmas=None, sigma0=1.):
        """
        Samples from a log-normal distribution using the reparametrization
        trick so that we can backprogpagate the gradients through the sampling.
        By setting sigma0=0 we make the operation deterministic (useful at testing time)
        """
        e = torch.normal(mean=torch.zeros_like(means), std=torch.ones_like(means))
        return torch.exp(means + sigmas * sigma0 * e)

    def information_dropout(self, input, sigma0=1.):
        # from: https://github.com/ucla-vision/information-dropout

        # Creates a convolutional layer to compute the noiseless output
        x = self.conv(input)
        x = self.bn(x)
        x = self.act(x)

        # Computes the noise parameter alpha for the new layer based on the input
        alpha = self.alpha_conv(input)
        alpha = self.alpha_bn(alpha)
        alpha = self.alpha_act(alpha)

        # Rescale alpha in the allowed range and add a small value for numerical stability
        alpha = 0.001 + self.max_alpha * alpha
        self.last_alpha = alpha.detach().cpu().numpy()

        # Similarly to variational dropout we renormalize so that
        # the KL term is zero for alpha == max_alpha
        # avg_kl = sum of all kl, averaged over batch size
        kls = - torch.log(alpha / (self.max_alpha + 0.001))
        self.last_kls = kls.detach().cpu().numpy()
        self.avg_kl = kls.sum() / float(alpha.shape[0])

        e = self.sample_lognormal(means=torch.zeros_like(x), sigmas=alpha, sigma0=sigma0)
        # Noisy output of Information Dropout
        return x * e

    def forward(self, input: torch.Tensor):
        return self.information_dropout(input)

class InfoDropoutLoss:
    def __init__(self, model: nn.Module, beta: float, beta_fadein: np.array = None):
        self.model = model
        self.cross_entropy = nn.CrossEntropyLoss()
        self.beta = beta
        self.beta_fadein = beta_fadein
        self.epoch = 0

    def update_epoch(self, epoch):
        self.epoch = epoch
        print("new epoch - beta is {} * {}".format(self.get_fadein(), self.beta))

    def get_loss(self, inputs, labels, as_dict=False):
        out = self.model(inputs)  # type: torch.Tensor
        Lx = self._get_loss_lx(out, labels)
        Lz = self.beta * self.get_fadein() * self._get_loss_lz()
        if as_dict:
            return {"Lx": to_np(Lx), "Lz": to_np(Lz)}  # type: dict
        else:
            return Lx + Lz  # type: torch.Tensor

    def _get_loss_lx(self, out, labels) -> torch.Tensor:
        """ assumes a batch was just fed forward """
        return self.cross_entropy(out, labels)

    def _get_loss_lz(self):
        """ assumes a batch was just fed forward """
        return np.sum([l.avg_kl for l in self.model.get_info_layers()])

    def get_fadein(self):
        if not self.beta_fadein:
            return 1.0
        else:
            return self.beta_fadein[self.epoch]
