import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        features = self.pointwise(self.depthwise(inputs))
        return F.silu(self.norm(features) + residual, inplace=True)


class MixStyle(nn.Module):
    """Mix per-sample feature statistics to simulate unseen recording conditions."""

    def __init__(self, probability: float = 0.5, alpha: float = 0.1, epsilon: float = 1e-6):
        super().__init__()
        self.probability = probability
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or inputs.size(0) < 2 or torch.rand((), device=inputs.device) > self.probability:
            return inputs

        mean = inputs.mean(dim=(2, 3), keepdim=True)
        std = (inputs.var(dim=(2, 3), keepdim=True, unbiased=False) + self.epsilon).sqrt()
        normalized = (inputs - mean) / std
        permutation = torch.randperm(inputs.size(0), device=inputs.device)
        beta = torch.distributions.Beta(
            torch.tensor(self.alpha, device=inputs.device),
            torch.tensor(self.alpha, device=inputs.device),
        )
        mixing = beta.sample((inputs.size(0), 1, 1, 1))
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_std = mixing * std + (1.0 - mixing) * std[permutation]
        return normalized * mixed_std + mixed_mean


class SpectralContextBranch(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv1d(64, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spectrum = torch.cat(
            [
                inputs.mean(dim=3),
                inputs.std(dim=3, unbiased=False),
                inputs.amax(dim=3),
            ],
            dim=1,
        )
        features = self.encoder(spectrum)
        return torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1)


class DVSCNet(nn.Module):
    """Domain-generalized dual-branch variational spectrogram classifier."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        z_dim: int = 32,
        dropout: float = 0.4,
        latent_noise_scale: float = 0.15,
        use_spec_augment: bool = True,
        freq_mask: int = 3,
        time_mask: int = 6,
        augment_probability: float = 0.3,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.latent_noise_scale = float(latent_noise_scale)
        self.use_spec_augment = use_spec_augment
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.augment_probability = float(augment_probability)
        self.normalize_input = normalize_input

        self.stem = ConvNormAct2d(in_channels, 32, stride=2)
        self.domain_mixer = MixStyle(probability=0.5, alpha=0.1)
        self.local_branch = nn.Sequential(
            SeparableResidualBlock(32, 64, stride=2),
            SeparableResidualBlock(64, 96, stride=2),
            SeparableResidualBlock(96, 128, stride=2),
        )
        self.local_projection = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.context_branch = SpectralContextBranch(output_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(32 * 4 * 5 * 2 + 256, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def _spec_augment(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or not self.use_spec_augment
            or torch.rand((), device=inputs.device).item() > self.augment_probability
        ):
            return inputs

        augmented = inputs.clone()
        _, _, frequency_bins, time_steps = augmented.shape

        if frequency_bins > 1 and self.freq_mask > 0:
            max_width = min(self.freq_mask, frequency_bins - 1)
            width = torch.randint(1, max_width + 1, (), device=inputs.device).item()
            start = torch.randint(0, frequency_bins - width + 1, (), device=inputs.device).item()
            augmented[:, :, start:start + width, :] = 0

        if time_steps > 1 and self.time_mask > 0:
            max_width = min(self.time_mask, time_steps - 1)
            width = torch.randint(1, max_width + 1, (), device=inputs.device).item()
            start = torch.randint(0, time_steps - width + 1, (), device=inputs.device).item()
            augmented[:, :, :, start:start + width] = 0

        return augmented

    def _normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return inputs
        mean = inputs.mean(dim=(2, 3), keepdim=True)
        std = inputs.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        return (inputs - mean) / std

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training or self.latent_noise_scale <= 0:
            return mu
        bounded_logvar = logvar.clamp(min=-6.0, max=2.0)
        std = torch.exp(0.5 * bounded_logvar)
        return mu + self.latent_noise_scale * torch.randn_like(std) * std

    def forward(self, inputs: torch.Tensor):
        augmented_inputs = self._spec_augment(inputs)
        normalized_inputs = self._normalize(augmented_inputs)
        shared_features = self.domain_mixer(self.stem(augmented_inputs))
        local_map = self.local_branch(shared_features)
        local_map = self.local_projection(local_map)
        local_features = torch.cat(
            [
                F.adaptive_avg_pool2d(local_map, (4, 5)).flatten(1),
                F.adaptive_max_pool2d(local_map, (4, 5)).flatten(1),
            ],
            dim=1,
        )
        context_features = self.context_branch(normalized_inputs)
        fused_features = self.fusion(torch.cat([local_features, context_features], dim=1))
        mu = self.fc_mu(fused_features)
        logvar = self.fc_logvar(fused_features).clamp(min=-6.0, max=2.0)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)
        return logits, mu, logvar, latent


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=0.05)