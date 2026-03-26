import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    """Basic conv block for spectrogram encoding."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureExtractionHead(nn.Module):
    """Extract compact branch-specific embeddings from feature maps."""

    def __init__(self, in_ch: int, hidden_ch: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_ch, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DVSCNet(nn.Module):
    """
    Dual-branch PVSC-style model.

    Design goals:
    - Keep the probabilistic latent-variable pipeline from model.py
    - Add feature extraction heads on two complementary branches
    - Fuse dual-branch embeddings before variational classification
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 1,
        dims=(32, 64, 128, 256),
        z_dim: int = 128,
        branch_dim: int = 128,
        head_dropout: float = 0.3,
        use_spec_augment: bool = True,
        freq_mask: int = 8,
        time_mask: int = 12,
        aug_prob: float = 0.5,
    ):
        super().__init__()
        if len(dims) != 4:
            raise ValueError("dims must have length 4")

        self.use_spec_augment = use_spec_augment
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.aug_prob = float(aug_prob)

        # Shared encoder stem (inherits the spirit of PVSC-Net convolutional encoder).
        self.stem = nn.Sequential(
            ConvBlock(in_chans, dims[0], stride=2),
            ConvBlock(dims[0], dims[1], stride=2),
        )

        # Branch A: local-detail branch with standard convolutions.
        self.branch_local = nn.Sequential(
            ConvBlock(dims[1], dims[2], stride=2),
            ConvBlock(dims[2], dims[3], stride=2),
            ConvBlock(dims[3], dims[3], stride=1),
        )

        # Branch B: context branch with dilated receptive fields.
        self.branch_context = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(dims[3], dims[3], stride=1),
        )

        # Feature extraction heads for each branch.
        self.local_head = FeatureExtractionHead(dims[3], hidden_ch=dims[3], out_dim=branch_dim)
        self.context_head = FeatureExtractionHead(dims[3], hidden_ch=dims[3], out_dim=branch_dim)

        # Fusion head.
        fusion_dim = branch_dim * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )

        # Variational latent projection (same core idea as model.py).
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        # Classifier from latent variable z.
        self.classifier = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Dropout(head_dropout),
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (not self.use_spec_augment) or torch.rand(1).item() > self.aug_prob:
            return x

        x = x.clone()
        b, c, f, t = x.shape

        if f > 1 and self.freq_mask > 0:
            max_f = min(self.freq_mask, f - 1)
            f_width = torch.randint(1, max_f + 1, (1,), device=x.device).item()
            f_start = torch.randint(0, f - f_width + 1, (1,), device=x.device).item()
            x[:, :, f_start:f_start + f_width, :] = 0

        if t > 1 and self.time_mask > 0:
            max_t = min(self.time_mask, t - 1)
            t_width = torch.randint(1, max_t + 1, (1,), device=x.device).item()
            t_start = torch.randint(0, t - t_width + 1, (1,), device=x.device).item()
            x[:, :, :, t_start:t_start + t_width] = 0

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._spec_augment(x)
        shared = self.stem(x)

        local_feat_map = self.branch_local(shared)
        context_feat_map = self.branch_context(shared)

        local_feat = self.local_head(local_feat_map)
        context_feat = self.context_head(context_feat_map)

        fused = torch.cat([local_feat, context_feat], dim=1)
        return self.fusion_head(fused)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        feats = self.forward_features(x)
        mu = self.fc_mu(feats)
        log_var = self.fc_logvar(feats)
        z = self.reparameterize(mu, log_var)
        class_logits = self.classifier(z)
        return class_logits, mu, log_var, z


def compute_loss(class_logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.1):
    """
    Classification loss with optional label smoothing.

    Signature is kept close to the old pipeline for easy replacement.
    """
    loss_class = F.cross_entropy(class_logits, labels, label_smoothing=label_smoothing)
    loss_dict = {
        "total": loss_class.item(),
        "class": loss_class.item(),
    }
    return loss_class, loss_dict


if __name__ == "__main__":
    num_classes = 4
    model = DVSCNet(num_classes=num_classes)
    x = torch.randn(8, 1, 64, 160)
    logits, mu, log_var, z = model(x)
    y = torch.randint(0, num_classes, (8,))
    loss, loss_dict = compute_loss(logits, y)

    print("=" * 50)
    print("ShipConvNeXtFormer test")
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log-var shape: {log_var.shape}")
    print(f"z shape: {z.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    print("=" * 50)
