import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


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
        mixing = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (inputs.size(0), 1, 1, 1)
        ).to(device=inputs.device, dtype=inputs.dtype)
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_std = mixing * std + (1.0 - mixing) * std[permutation]
        return normalized * mixed_std + mixed_mean


class AxisAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.frequency_gate = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )
        self.time_gate = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        frequency_weights = torch.sigmoid(self.frequency_gate(inputs.mean(dim=3))).unsqueeze(3)
        time_weights = torch.sigmoid(self.time_gate(inputs.mean(dim=2))).unsqueeze(2)
        return inputs * (0.5 + frequency_weights) * (0.5 + time_weights)


class MultiScaleAxialBlock(nn.Module):
    """Depthwise time-frequency block with separate harmonic and temporal receptive fields."""

    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1)):
        super().__init__()
        self.input_projection = ConvNormAct2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.local = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.frequency = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(7, 1),
            stride=stride,
            padding=(3, 0),
            groups=out_channels,
            bias=False,
        )
        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 9),
            stride=stride,
            padding=(0, 4),
            groups=out_channels,
            bias=False,
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            AxisAttention(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
            if stride != (1, 1) or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        projected = self.input_projection(inputs)
        features = (self.local(projected) + self.frequency(projected) + self.temporal(projected)) / 3.0
        return F.silu(self.output_projection(features) + residual, inplace=True)


class SpectralContextEncoder(nn.Module):
    def __init__(self, output_dim: int = 160):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
        )
        self.attention = nn.Conv1d(96, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Linear(96 * 3, output_dim),
            nn.LayerNorm(output_dim),
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
        weights = torch.softmax(self.attention(features), dim=2)
        mean = (features * weights).sum(dim=2)
        variance = (weights * (features - mean.unsqueeze(2)).square()).sum(dim=2)
        statistics = torch.cat(
            [mean, variance.clamp_min(1e-6).sqrt(), features.amax(dim=2)],
            dim=1,
        )
        return self.projection(statistics)


class DVSCNet(nn.Module):
    """Single-stream variational classifier with multi-scale axial convolutions."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        z_dim: int = 32,
        dropout: float = 0.4,
        latent_noise_scale: float = 0.1,
        kl_weight: float = 5e-4,
        use_spec_augment: bool = True,
        freq_mask: int = 4,
        time_mask: int = 8,
        augment_probability: float = 0.3,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.latent_noise_scale = float(latent_noise_scale)
        self.kl_weight = float(kl_weight)
        self.use_spec_augment = use_spec_augment
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.augment_probability = float(augment_probability)
        self.normalize_input = normalize_input

        self.stem = ConvNormAct2d(
            in_channels,
            32,
            kernel_size=(5, 7),
            stride=(2, 2),
            padding=(2, 3),
        )
        self.domain_mixer = MixStyle(probability=0.3, alpha=0.3)
        self.encoder = nn.Sequential(
            MultiScaleAxialBlock(32, 48, stride=(2, 2)),
            MultiScaleAxialBlock(48, 72, stride=(2, 2)),
            MultiScaleAxialBlock(72, 96, stride=(1, 2)),
            MultiScaleAxialBlock(96, 128, stride=(1, 1)),
        )
        self.layout_projection = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.context_encoder = SpectralContextEncoder(output_dim=160)
        self.fusion = nn.Sequential(
            nn.Linear(32 * 4 * 5 * 2 + 128 * 3 + 160, 256),
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
        if not self.training or not self.use_spec_augment:
            return inputs

        augmented = inputs.clone()
        batch_size, _, frequency_bins, time_steps = augmented.shape
        for sample_index in range(batch_size):
            if torch.rand((), device=inputs.device) > self.augment_probability:
                continue
            if frequency_bins > 1 and self.freq_mask > 0:
                max_width = min(self.freq_mask, frequency_bins - 1)
                width = int(torch.randint(1, max_width + 1, (), device=inputs.device))
                start = int(torch.randint(0, frequency_bins - width + 1, (), device=inputs.device))
                augmented[sample_index, :, start:start + width, :] = 0
            if time_steps > 1 and self.time_mask > 0:
                max_width = min(self.time_mask, time_steps - 1)
                width = int(torch.randint(1, max_width + 1, (), device=inputs.device))
                start = int(torch.randint(0, time_steps - width + 1, (), device=inputs.device))
                augmented[sample_index, :, :, start:start + width] = 0
        return augmented

    def _normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return inputs
        mean = inputs.mean(dim=(2, 3), keepdim=True)
        std = inputs.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-5)
        return (inputs - mean) / std

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training or self.latent_noise_scale <= 0:
            return mu
        std = torch.exp(0.5 * logvar.clamp(min=-6.0, max=2.0))
        return mu + self.latent_noise_scale * torch.randn_like(std) * std

    def regularization_loss(self, outputs) -> torch.Tensor:
        _, mu, logvar, _ = outputs
        kl_divergence = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
        return self.kl_weight * kl_divergence.mean()

    def forward(self, inputs: torch.Tensor):
        augmented_inputs = self._spec_augment(inputs)
        normalized_inputs = self._normalize(augmented_inputs)
        feature_map = self.encoder(self.domain_mixer(self.stem(augmented_inputs)))
        layout_map = self.layout_projection(feature_map)
        layout_features = torch.cat(
            [
                F.adaptive_avg_pool2d(layout_map, (4, 5)).flatten(1),
                F.adaptive_max_pool2d(layout_map, (4, 5)).flatten(1),
            ],
            dim=1,
        )
        global_statistics = torch.cat(
            [
                feature_map.mean(dim=(2, 3)),
                feature_map.std(dim=(2, 3), unbiased=False),
                feature_map.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        context_features = self.context_encoder(normalized_inputs)
        fused_features = self.fusion(
            torch.cat([layout_features, global_statistics, context_features], dim=1)
        )
        mu = self.fc_mu(fused_features)
        logvar = self.fc_logvar(fused_features).clamp(min=-6.0, max=2.0)
        latent = self.reparameterize(mu, logvar)
        logits = self.classifier(latent)
        return logits, mu, logvar, latent


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=0.05)