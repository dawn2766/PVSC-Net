import argparse
import sys
from pathlib import Path

import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PVSCNET_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PVSCNET_DIR / "comparisons"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PVSCNET_DIR))

from comparison_models import DeterministicFusionNet, SpectrogramOnlyPVNet
from model_PVSCNet import PVSCNet, compute_loss
from training_utils import TrainingConfig, train_experiment


MODEL_NAMES = ("PVSCNet", "DeterministicFusionNet", "SpectrogramOnlyPVNet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train controlled PVSC-Net comparison models."
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--windows-per-source", type=int, default=128)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-noise-scale", type=float, default=0.1)
    parser.add_argument(
        "--output-group",
        default=None,
        help="Optional subdirectory under PVSCNet/comparisons for an isolated run group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=0,
        windows_per_source=args.windows_per_source,
        early_stopping_patience=None,
        gradient_clip_norm=args.gradient_clip_norm,
    )

    output_root = OUTPUT_ROOT / args.output_group if args.output_group else OUTPUT_ROOT
    for model_name in args.models:
        if model_name == "PVSCNet":
            build_model = lambda num_classes, input_shape, auxiliary_dim: PVSCNet(
                num_classes=num_classes,
                input_shape=input_shape,
                z_dim=args.z_dim,
                latent_noise_scale=args.latent_noise_scale,
                auxiliary_dim=auxiliary_dim,
            )
            removed_component = None
        elif model_name == "DeterministicFusionNet":
            build_model = lambda num_classes, input_shape, auxiliary_dim: DeterministicFusionNet(
                num_classes=num_classes,
                z_dim=args.z_dim,
                auxiliary_dim=auxiliary_dim,
            )
            removed_component = "stochastic_probabilistic_bottleneck"
        else:
            build_model = lambda num_classes, input_shape, auxiliary_dim: SpectrogramOnlyPVNet(
                num_classes=num_classes,
                z_dim=args.z_dim,
                latent_noise_scale=args.latent_noise_scale,
            )
            removed_component = "auxiliary_feature_fusion"

        train_experiment(
            model_name=model_name,
            project_root=PROJECT_ROOT,
            model_dir=output_root / model_name,
            config=config,
            build_model=build_model,
            build_optimizer=lambda parameters: optim.AdamW(
                parameters,
                lr=config.learning_rate,
                weight_decay=args.weight_decay,
            ),
            criterion=compute_loss,
            extra_config={
                "z_dim": args.z_dim,
                "weight_decay": args.weight_decay,
                "latent_noise_scale": args.latent_noise_scale,
                "comparison_design": "paired_controlled_ablation",
                "removed_component": removed_component,
                "reference_model": "PVSCNet",
                "output_group": args.output_group,
            },
        )


if __name__ == "__main__":
    main()