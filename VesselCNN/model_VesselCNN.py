import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_fusion import AuxiliaryFeatureEncoder


class VesselCNN(nn.Module):
    """Compact convolutional baseline for vessel classification."""

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int],
        auxiliary_dim: int = 3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.auxiliary_encoder = AuxiliaryFeatureEncoder(auxiliary_dim, output_dim=32)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 8 + self.auxiliary_encoder.output_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        auxiliary_features: torch.Tensor = None,
    ) -> torch.Tensor:
        deep_features = self.features(inputs).flatten(1)
        auxiliary_embedding = self.auxiliary_encoder(
            auxiliary_features,
            batch_size=inputs.size(0),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return self.classifier(torch.cat([deep_features, auxiliary_embedding], dim=1))


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=0.05)