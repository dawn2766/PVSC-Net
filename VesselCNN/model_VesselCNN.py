import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselCNN(nn.Module):
    """Compact convolutional baseline for vessel classification."""

    def __init__(self, num_classes: int, input_shape: tuple[int, int]):
        super().__init__()
        input_height, input_width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (input_height // 4) * (input_width // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)