import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import stack_layers_conv, stack_layers_linear
from typing import Tuple, List, Union, Optional, Callable, Any, Dict


class BetaVAE(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 net_arch: Union[List[int], Dict[str, List[int]]],
                 activation: Optional[Callable] = nn.ReLU,
                 batch_norm: bool = False,
                 layer_type: Optional[str] = "linear"):
        super().__init__()
        if isinstance(net_arch, list):
            net_arch = {"encoder": net_arch, "decoder": net_arch[::-1]}

        self.net_arch = net_arch
        self.latent_dim = latent_dim
        layer_const_func = stack_layers_conv if layer_type == "conv" else stack_layers_linear
        self.encoder = layer_const_func(net_arch=net_arch["encoder"],
                                        activation=activation,
                                        batch_norm=batch_norm,
                                        reverse=False)
        self.encoder = nn.Sequential(*self.encoder)

        with th.no_grad():
            dummy = th.zeros(1, 3, 64, 64)
            dummy = self.encoder(dummy)
            dummy_shape = dummy.shape[1:]

        self.flatten = nn.Flatten()
        self.mu = nn.Linear(int(np.prod(dummy_shape)), latent_dim)
        self.log_var = nn.Linear(int(np.prod(dummy_shape)), latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, int(np.prod(dummy_shape))),
            activation(),
            nn.Unflatten(1, dummy_shape),
            *layer_const_func(net_arch=net_arch["decoder"],
                              activation=activation,
                              batch_norm=batch_norm,
                              reverse=True),
            nn.Sigmoid()
        )

    def encode(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x = self.encoder(x)
        x = self.flatten(x)
        return self.mu(x), self.log_var(x)

    def reparameterize(self, mu, log_var):
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var


def log_normal_pdf(sample: th.Tensor, mean: th.Tensor, log_var: th.Tensor):
    log_2pi = np.log(2 * np.pi)
    log_dist = -0.5 * (log_var + log_2pi + (sample - mean) ** 2 / th.exp(log_var))
    return th.sum(log_dist, dim=1)


def compute_loss(model: nn.Module, x: th.Tensor, beta: float = 1.0) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    mu, log_var = model.encode(x)
    z = model.reparameterize(mu, log_var)
    x_hat = model.decoder(z)

    assert x_hat.min() >= 0 and x_hat.max() <= 1, f"min: {x_hat.min()}, max: {x_hat.max()}"
    cross_entropy = F.binary_cross_entropy(x_hat, x, reduction="none")
    log_px_z = th.sum(cross_entropy, dim=[1, 2, 3])
    log_pz = log_normal_pdf(z, th.zeros_like(z), th.zeros_like(z))
    log_qz_x = log_normal_pdf(z, mu, log_var)

    kl = (log_qz_x - log_pz) * beta
    return th.mean(log_px_z + kl), kl, log_px_z
