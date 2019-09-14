import torch.nn as nn
from tqdm import tqdm
from typing import *
from utils.misc import *
import numpy as np

class NetworkObserver:
    """ Holds layer observers, uses them to call info analysers """
    def __init__(self, model: nn.Module, layers, info_mapper_class, progress_bar=False):
        self.observers = []  # type: List[LayerObserver]
        self.mappers = []  # type: List[LayerInfoMapper]
        self.mapper_args = []
        self.mapper_kwargs = []

        self.progress_bar = progress_bar
        self.model = model
        self.info_mapper_class = info_mapper_class

        self.observe(layers)

    def observe(self, layers, *args, **kwargs):
        if not isinstance(layers, list):
            layers = [layers]
        for layer in layers:
            self.observers.append(LayerObserver(layer))
            self.mapper_args.append(args)
            self.mapper_kwargs.append(kwargs)

    def _get_observations(self, batches) -> List[np.ndarray]:
        """ data: list of tuple(4-dim-batches, labels), or just list of tensors, or just tensor """

        batches = make_batch_list(batches)

        # set all observers to observe
        for obs in self.observers:
            obs.start_observing()

        # pass data
        self.model.eval()
        it = tqdm(batches, desc="collecting data") if self.progress_bar else batches
        for batch, labels in it:
            self.model(batch)

        # stop all observers and collect the data
        return [obs.get_forward_observations() for obs in self.observers]

    def prepare(self, batches):
        # collect empirical data
        observations = self._get_observations(batches)
        # build activation to information mappers
        self.mappers = [self.info_mapper_class(obs, *args, **kwargs) for obs, args, kwargs in zip(observations, self.mapper_args, self.mapper_kwargs)]

    def get_info_maps(self, sample, clip_results=0.0):
        """
        sample: 4-D input data
        return a tuple of 2D information maps, one for each observed layer, rescaled to [0,1]
        """

        if len(self.observers) == 0:
            raise RuntimeError("no observers registered yet")

        if self.mappers is None:
            raise RuntimeError("first usage: no info mappers created yet. run create_info_mappers")

        # collect hists
        activations = self._get_observations(sample)
        iterator = zip(activations, self.mappers)
        if self.progress_bar:
            iterator = tqdm(iterator, desc="act --> info")

        maps = []
        for act, mapper in iterator:

            # map activations to information
            info_map = mapper.get_layer_info_map(act)

            # maybe clip and replace with min of unclipped area
            if clip_results > 0.0:
                x, y = get_bolder_px(info_map, clip_results)
                minval = info_map[x or None:-x or None, y or None:-y or None].min()
                fill_border(info_map, clip_ratio=clip_results, val=minval)

            maps.append(info_map)

        def fit(arr, eps=0.0001):
            arr = arr - arr.min()  # min = 0
            arr = arr / max(eps, arr.max())  # max = 1
            return arr

        # fit to [0,1]
        maps = [fit(m) for m in maps]

        return tuple(maps)

class LayerObserver(object):
    """ attach this to a layer to collect the passing activations through this layer """
    def __init__(self, layer: nn.Module):
        self.hook = None  # None: not observing
        self.layer = layer
        self.forward_observations = None

    def layer_name(self):
        return self.layer.__class__.__name__

    def start_observing(self):
        """ start recording the passing data to build distribution estimate """

        def forward_hook(module: torch.nn.Module, data_in: torch.Tensor, data_out: torch.Tensor):
            self.forward_observations.append(to_np(data_out))

        self.forward_observations = []
        self.hook = self.layer.register_forward_hook(hook=forward_hook)

    def get_forward_observations(self):
        """ finish recording, analyse the results and return the results as a tensor"""
        if self.hook is None:
            raise RuntimeError("not observing anyways")
        self.hook.remove()
        self.hook = None
        obs_stack = np.concatenate(self.forward_observations, axis=0)  # concatenate along batch axis
        self.forward_observations = None
        if obs_stack.shape[0] > 1:
            # print("returning observations: ", obs_stack.shape)
            pass
        return obs_stack

class LayerInfoMapper(object):
    """ maps sample activations to information. maybe has to be instantiated with a bunch of data. """

    def get_layer_info_map(self, acts):
        raise NotImplementedError
