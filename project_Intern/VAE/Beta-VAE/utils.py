import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Union, Optional, Callable, Any, Dict


def stack_layers(net_arch: List[int],
                 activation: Optional[Callable] = nn.ReLU,
                 batch_norm: bool = False,
                 reverse: bool = False,
                 layer_type: Optional[str] = "linear",
                 batch_size: int = 64) -> List[nn.Module]:

    if reverse:
        # net_arch = net_arch[::-1]
        extractor = nn.Linear if layer_type == "linear" else nn.ConvTranspose2d
    else:
        extractor = nn.Conv2d if layer_type == "conv" else nn.Linear

    kwargs = {"bias": not batch_norm}
    if layer_type == "conv":
        kwargs["padding"] = 1
        kwargs["stride"] = 2
        kwargs["kernel_size"] = 4

    layers = []

    if layer_type == "linear":
        layers.append(nn.Flatten())

    for i in range(len(net_arch) - 2):
        start_dim, end_dim = net_arch[i:i+2]
        layers.append(extractor(start_dim, end_dim, **kwargs))
        if batch_norm:
            layers.append(nn.BatchNorm2d(end_dim) if layer_type == "conv" else nn.BatchNorm1d(batch_size))
        layers.append(activation())

    layers.append(extractor(net_arch[-2], net_arch[-1], **kwargs))
    if not reverse:
        if batch_norm:
            layers.append(nn.BatchNorm2d(net_arch[-1]) if layer_type == "conv" else nn.BatchNorm1d(batch_size))
        layers.append(activation())

    return layers


def stack_layers_conv(net_arch: List[int],
                      activation: Optional[Callable] = nn.ReLU,
                      batch_norm: bool = False,
                      reverse: bool = False):

    extractor = nn.ConvTranspose2d if reverse else nn.Conv2d
    kwargs = {"bias": not batch_norm, "padding": 1, "stride": 2, "kernel_size": 4}
    layers = []

    for i in range(len(net_arch) - 2):
        start_dim, end_dim = net_arch[i:i+2]
        layers.append(extractor(start_dim, end_dim, **kwargs))
        if batch_norm:
            layers.append(nn.BatchNorm2d(end_dim))
        layers.append(activation())

    layers.append(extractor(net_arch[-2], net_arch[-1], **kwargs))
    if not reverse:
        if batch_norm:
            layers.append(nn.BatchNorm2d(net_arch[-1]))
        layers.append(activation())

    return layers


def stack_layers_linear(net_arch: List[int],
                        activation: Optional[Callable] = nn.ReLU,
                        batch_norm: bool = False,
                        reverse: bool = False):

    layers = []
    input_size = int(np.sqrt(net_arch[-1] // 3))

    if not reverse:
        layers.append(nn.Flatten())

    for i in range(len(net_arch) - 2):
        start_dim, end_dim = net_arch[i:i+2]
        layers.append(nn.Linear(start_dim, end_dim, bias=not batch_norm))
        layers.append(activation())

    layers.append(nn.Linear(net_arch[-2], net_arch[-1], bias=not batch_norm))
    if not reverse:
        layers.append(activation())
    else:
        layers.append(nn.Unflatten(1, (3, input_size, input_size)))
    return layers
