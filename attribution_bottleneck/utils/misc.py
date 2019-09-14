
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        # print("searching ", model.__class__.__name__)
        for name, submodule in model.named_children():
            # print("is it member?", name, submodule == target)
            if submodule == target:
                # we found it!
                if isinstance(model, nn.ModuleList):
                    # replace in module list
                    model[name] = replacement

                elif isinstance(model, nn.Sequential):
                    # replace in sequential layer
                    model[int(name)] = replacement
                else:
                    # replace as member
                    model.__setattr__(name, replacement)

                # print("Replaced " + target.__class__.__name__ + " with "+replacement.__class__.__name__+" in " + model.__class__.__name__)
                return True

            elif len(list(submodule.named_children())) > 0:
                # print("Browsing {} children...".format(len(list(submodule.named_children()))))
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

def resize(arr, shape, interp="bilinear"):
    if interp == "nearest":
        interp = cv2.INTER_NEAREST
    elif interp == "bilinear" or interp == "linear":
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError(interp)
    return cv2.resize(arr, dsize=shape, interpolation=interp)

def mono_to_rgb(img):
    if len(img.shape) == 2:
        return np.stack((img, img, img), axis=2)
    elif img.shape[2] == 1:
        return np.dstack((img, img, img))
    else:
        # nothing to do
        return img

def show_img(img, title="", place=None):
    img = to_np_img(img)
    if place is None:
        place = plt
    try:
        if len(img.shape) == 3 and img.shape[2] == 1:
            # remove single grey channel
            img = img[...,0]

        if len(img.shape) == 2:
            place.imshow(img, cmap="Greys_r")
        else:
            place.imshow(img)
    except TypeError:
        print("type error: shape is {}".format(img.shape))
        raise TypeError

    if not isinstance(place, Axes):
        place.title(title)
        plt.show()
    else:
        place.set_title(title)

def prepare_image(img):
    return Compose([
        Resize(224),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(img).unsqueeze(0)

def normalize_img(img: np.ndarray):
    img = img - np.min(img)
    img = img / max(np.max(img), 0.001)
    return img

def chw_to_hwc(img: np.ndarray):
    return np.swapaxes(np.swapaxes(img, 2, 0), 1, 0)

def hwc_to_chw(img: np.ndarray):
    return np.swapaxes(np.swapaxes(img, 1, 0), 2, 0)

def to_img_tensor(img, device=None):

    # add color channel
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)

    # add batch dimension 1
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    # move color channel to front
    img = np.swapaxes(np.swapaxes(img, 2, 1), 3, 1)

    if device is not None:
        t = torch.tensor(img, device=device)
    else:
        t = torch.from_numpy(img)

    return t

def denormalize(img: np.ndarray):
    img = img - img.min()  # force min 0
    img = img / np.max(img)  # force max 1
    return img

def to_np_img(img: torch.Tensor, denorm=False):

    # force 2-3 dims
    if len(img.shape) == 4:
        img = img[0]

    # tensor to np
    if isinstance(img, torch.Tensor):
        img = img.detach()
        if img.is_cuda:
            img = img.cpu()
        img = img.numpy()

    # if color is not last
    if len(img.shape) > 2 and img.shape[0] < img.shape[2]:
        img = np.swapaxes(np.swapaxes(img, 2, 0), 1, 0)

    if denorm:
        img = denormalize(img)

    return img

def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()
