import torch
import numpy as np
from ..utils.misc import resize
from ..attribution.base import AttributionMethod


class GradCAM(AttributionMethod):

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module, interp="nearest"):
        """
        :param model: model containing the softmax-layer
        :param device: dev
        :param layer: evaluation layer - object or name or id
        """
        self.layer = layer
        self.model = model
        self.interp = interp
        self.grads = None
        self.probs = None
        self.eps = 1e-5

    def pass_through(self, img):
        self.model.eval()
        return

    def heatmap(self, input_t: torch.Tensor, target_t: torch.Tensor):
        shape = tuple(input_t[0,0].shape)

        # feed in
        self.model.eval()

        fmaps, grads = None, None

        def hook_forward(module, input, output):
            nonlocal fmaps
            fmaps = output.detach()

        def hook_backward(module, grad_in, grad_out):
            nonlocal grads
            grads = grad_out[0].detach()

        # Pass and collect activations + gradient of feature map
        forward_handle = self.layer.register_forward_hook(hook_forward)
        backward_handle = self.layer.register_backward_hook(hook_backward)
        self.model.zero_grad()
        preds = self.model(input_t)
        forward_handle.remove()
        backward_handle.remove()

        # Calc grads
        grad_eval_point = torch.Tensor(1, preds.size()[-1]).zero_()
        grad_eval_point[0][preds.argmax().item()] = 1.0
        grad_eval_point = grad_eval_point.to(input_t.device)
        preds.backward(gradient=grad_eval_point, retain_graph=True)

        # Weight maps
        maps = fmaps.detach().cpu().numpy()[0,]
        weights = grads.detach().cpu().numpy().mean(axis=(2,3))[0,:]

        # Average maps
        gcam = np.zeros(maps.shape[0:], dtype=np.float32)
        # Sum up weighted fmaps
        for i, w in enumerate(weights):
            gcam += w * maps[i, :, :]

        # Average pool over feature maps
        gcam = np.mean(gcam, axis=0)
        # relu
        gcam = np.maximum(gcam, 0)
        # Fit to input shape
        gcam = resize(gcam, shape, interp=self.interp)
        # Rescale to sum 1 for comparability
        gcam = gcam / (gcam.sum() + self.eps)

        return gcam

