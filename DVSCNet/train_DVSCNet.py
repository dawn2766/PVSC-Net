import os
import sys
from pathlib import Path

import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_DIR))

from model_DVSCNet import DVSCNet, compute_loss
from training_utils import TrainingConfig, create_argument_parser, train_experiment


def main() -> None:
    parser = create_argument_parser("DVSCNet")
    parser.add_argument("--z-dim", type=int, default=int(os.getenv("DVSC_Z_DIM", "32")))
    parser.add_argument("--dropout", type=float, default=float(os.getenv("DVSC_DROPOUT", "0.4")))
    parser.add_argument("--kl-weight", type=float, default=float(os.getenv("DVSC_KL_WEIGHT", "5e-4")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "1e-3")))
    parser.add_argument(
        "--latent-noise-scale",
        type=float,
        default=float(os.getenv("DVSC_LATENT_NOISE_SCALE", "0.05")),
    )
    parser.add_argument("--disable-spec-augment", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        windows_per_source=args.windows_per_source,
        early_stopping_patience=(
            None if args.disable_early_stopping else args.early_stopping_patience
        ),
        gradient_clip_norm=args.gradient_clip_norm,
    )
    train_experiment(
        model_name="DVSCNet",
        project_root=PROJECT_ROOT,
        model_dir=MODEL_DIR,
        config=config,
        build_model=lambda num_classes, _input_shape, auxiliary_dim: DVSCNet(
            num_classes=num_classes,
            z_dim=args.z_dim,
            dropout=args.dropout,
            latent_noise_scale=args.latent_noise_scale,
            kl_weight=args.kl_weight,
            use_spec_augment=not args.disable_spec_augment,
            auxiliary_dim=auxiliary_dim,
        ),
        build_optimizer=lambda parameters: optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=args.weight_decay,
        ),
        criterion=compute_loss,
        extra_config={
            "architecture_version": "single_stream_axial_context_aux_pca_lda_v4",
            "z_dim": args.z_dim,
            "dropout": args.dropout,
            "kl_weight": args.kl_weight,
            "weight_decay": args.weight_decay,
            "latent_noise_scale": args.latent_noise_scale,
            "use_spec_augment": not args.disable_spec_augment,
        },
    )


if __name__ == "__main__":
    main()