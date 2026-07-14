import torch
import torch.nn as nn
import torch.nn.functional as F


class PVSCNet(nn.Module):
    """Probabilistic variational classifier for single-channel spectrograms."""

    def __init__(self, num_classes: int, input_shape: tuple[int, int], z_dim: int = 16):
        super().__init__()
        input_height, input_width = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_height, input_width)
            feature_dim = self.encoder(dummy_input).numel()

        self.hidden = nn.Linear(feature_dim, 512)
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, inputs: torch.Tensor):
        features = self.encoder(inputs).flatten(1)
        hidden = F.leaky_relu(self.hidden(features), negative_slope=0.2)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)
        return logits, mu, logvar, latent


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)