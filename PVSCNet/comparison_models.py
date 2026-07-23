import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_fusion import AuxiliaryFeatureEncoder


def _build_encoder() -> nn.Sequential:
    return nn.Sequential(
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


def _build_classifier(z_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(z_dim),
        nn.Linear(z_dim, 128),
        nn.SiLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, num_classes),
    )


def _pool_features(features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            F.adaptive_avg_pool2d(features, 1).flatten(1),
            F.adaptive_max_pool2d(features, 1).flatten(1),
        ],
        dim=1,
    )


class DeterministicFusionNet(nn.Module):
    """PVSC-Net control without the stochastic probabilistic bottleneck."""

    def __init__(self, num_classes: int, z_dim: int = 16, auxiliary_dim: int = 3):
        super().__init__()
        self.encoder = _build_encoder()
        self.auxiliary_encoder = AuxiliaryFeatureEncoder(auxiliary_dim, output_dim=32)
        self.hidden = nn.Sequential(
            nn.Linear(512 + self.auxiliary_encoder.output_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fc_latent = nn.Linear(256, z_dim)
        self.classifier = _build_classifier(z_dim, num_classes)

    def forward(
        self,
        inputs: torch.Tensor,
        auxiliary_features: torch.Tensor = None,
    ) -> torch.Tensor:
        pooled = _pool_features(self.encoder(inputs))
        auxiliary_embedding = self.auxiliary_encoder(
            auxiliary_features,
            batch_size=inputs.size(0),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        latent = self.fc_latent(
            self.hidden(torch.cat([pooled, auxiliary_embedding], dim=1))
        )
        return self.classifier(latent)


class SpectrogramOnlyPVNet(nn.Module):
    """PVSC-Net control retaining stochastic inference without auxiliary features."""

    def __init__(
        self,
        num_classes: int,
        z_dim: int = 16,
        latent_noise_scale: float = 0.1,
    ):
        super().__init__()
        self.latent_noise_scale = float(latent_noise_scale)
        self.encoder = _build_encoder()
        self.hidden = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        self.classifier = _build_classifier(z_dim, num_classes)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training or self.latent_noise_scale <= 0:
            return mu
        std = torch.exp(0.5 * logvar.clamp(min=-6.0, max=2.0))
        return mu + self.latent_noise_scale * torch.randn_like(std) * std

    def forward(
        self,
        inputs: torch.Tensor,
        auxiliary_features: torch.Tensor = None,
    ):
        hidden = self.hidden(_pool_features(self.encoder(inputs)))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(min=-6.0, max=2.0)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)
        return logits, mu, logvar, latent