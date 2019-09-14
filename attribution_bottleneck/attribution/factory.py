from torch.nn import Softmax

from attribution_bottleneck.attribution.base import Zero, Random
from ..attribution.backprop import Gradient, GradientTimesInput
from ..attribution.guided_backprop import *
from ..attribution.integrating import *
from ..attribution.grad_cam import *
from ..attribution.occlusion import Occlusion
from ..attribution.pattern import *
from ..utils.baselines import Mean


class Factory:

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        # Check if the given model's last layer are the logits
        for m in model.modules():
            if isinstance(m, Softmax):
                raise RuntimeError("model has to be pre-softmax!")

    def Random(self):
        return Random()

    def Zero(self):
        return Zero()

    def Gradient(self):
        return Gradient(self.model)

    def GradientTimesInput(self):
        """ https://arxiv.org/abs/1605.01713 """
        return GradientTimesInput(self.model)

    def PatternAttribution(self):
        return PatternAttribution(self.model)

    def Saliency(self):
        """ https://arxiv.org/abs/1312.6034 """
        return Gradient(self.model)

    def GuidedBackprop(self):
        """ https://arxiv.org/abs/1412.6806 """
        return GuidedBackprop(self.model)

    def DeconvNet(self):
        """ https://arxiv.org/abs/1412.6806 """
        return DeconvNet(self.model)

    def IntegratedGradients(self):
        """ https://arxiv.org/abs/1703.01365 """
        return IntegratedGradients(Saliency(self.model))

    def Occlusion(self, patch_size):
        """ TODO CITATION """
        return Occlusion(self.model, size=patch_size)

    def SmoothGrad(self):
        """ https://arxiv.org/abs/1706.03825 """
        return SmoothGrad(Saliency(self.model))

    def GradCAM(self, layer):
        """ https://arxiv.org/abs/1610.02391 """
        return GradCAM(self.model, layer=layer)

    def InfoBottleneck(self):
        raise NotImplementedError

    def InfoReadout(self):
        raise NotImplementedError
