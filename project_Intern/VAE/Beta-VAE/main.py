import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Callable

from model import BetaVAE, compute_loss


image_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

data = CelebA(root="../dataset/", download=False, transform=image_transform)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def train(model: nn.Module,
          optimizer: th.optim.Optimizer,
          scheduler: th.optim.lr_scheduler,
          train_loader: DataLoader,
          epochs: int = 10,
          beta: float = 1.0,
          device: th.device = device):

    model.train()
    losses = []
    for epoch in range(epochs):
        desc = f"Epoch {epoch + 1}/{epochs}"
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=desc, leave=True)
        epoch_loss = 0

        for i, (img, _) in progress_bar:
            img = img.to(device)
            optimizer.zero_grad()
            loss, kl_div, recon_loss = compute_loss(model, img, beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(),
                                     recon_loss=recon_loss.mean().item(),
                                     kl_div=kl_div.mean().item())

        scheduler.step(loss)
        losses.append(epoch_loss / len(train_loader))
    return losses

batch_size = 512
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

model = BetaVAE(latent_dim=5,
                net_arch=[3, 32, 64, 128, 256],
                activation=nn.Softplus,
                batch_norm=True,
                layer_type="conv").to(device)

optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    patience=3,
                                                    factor=0.25,
                                                    verbose=True)

history = train(model, optimizer, scheduler, data_loader, epochs=30, beta=100.0)