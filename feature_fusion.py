import torch
import torch.nn as nn
from typing import Optional


class AuxiliaryFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.output_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        auxiliary_features: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if auxiliary_features is None:
            auxiliary_features = torch.zeros(
                batch_size,
                self.input_dim,
                device=device,
                dtype=dtype,
            )
        if auxiliary_features.ndim != 2 or auxiliary_features.shape != (
            batch_size,
            self.input_dim,
        ):
            raise ValueError(
                "Expected auxiliary features with shape "
                f"({batch_size}, {self.input_dim}), got {tuple(auxiliary_features.shape)}"
            )
        return self.encoder(auxiliary_features.to(device=device, dtype=dtype))
