import torch
import torch.nn as nn
import torch.nn.functional as F


class PVSCNet(nn.Module):
    """Probabilistic variational classifier for single-channel spectrograms."""

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int],
        z_dim: int = 16,
        latent_noise_scale: float = 0.1,
    ):
        super().__init__()
        self.latent_noise_scale = float(latent_noise_scale)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
        )
        self.hidden = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training or self.latent_noise_scale <= 0:
            return mu
        std = torch.exp(0.5 * logvar.clamp(min=-6.0, max=2.0))
        return mu + self.latent_noise_scale * torch.randn_like(std) * std

    def forward(self, inputs: torch.Tensor):
        features = self.encoder(inputs)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(features, 1).flatten(1),
                F.adaptive_max_pool2d(features, 1).flatten(1),
            ],
            dim=1,
        )
        hidden = self.hidden(pooled)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(min=-6.0, max=2.0)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)
        return logits, mu, logvar, latent


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=0.05)