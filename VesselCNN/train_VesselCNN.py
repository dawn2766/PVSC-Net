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
    args = create_argument_parser("VesselCNN").parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train_experiment(
        model_name="VesselCNN",
        project_root=PROJECT_ROOT,
        model_dir=MODEL_DIR,
        config=config,
        build_model=lambda num_classes, input_shape: VesselCNN(num_classes, input_shape),
        build_optimizer=lambda parameters: optim.Adam(parameters, lr=config.learning_rate),
        criterion=compute_loss,
    )


if __name__ == "__main__":
    main()