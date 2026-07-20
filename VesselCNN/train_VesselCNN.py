import os
import sys
from pathlib import Path

import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_DIR))

from model_VesselCNN import VesselCNN, compute_loss
from training_utils import TrainingConfig, create_argument_parser, train_experiment


def main() -> None:
    parser = create_argument_parser("VesselCNN")
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "1e-4")))
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
        model_name="VesselCNN",
        project_root=PROJECT_ROOT,
        model_dir=MODEL_DIR,
        config=config,
        build_model=lambda num_classes, input_shape, auxiliary_dim: VesselCNN(
            num_classes,
            input_shape,
            auxiliary_dim=auxiliary_dim,
        ),
        build_optimizer=lambda parameters: optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=args.weight_decay,
        ),
        criterion=compute_loss,
        extra_config={"weight_decay": args.weight_decay},
    )


if __name__ == "__main__":
    main()