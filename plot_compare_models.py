import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_LOGS = {
    "PVSCNet": PROJECT_ROOT / "PVSCNet" / "log_latest_PVSCNet.json",
    "VesselCNN": PROJECT_ROOT / "VesselCNN" / "log_latest_VesselCNN.json",
    "DVSCNet": PROJECT_ROOT / "DVSCNet" / "log_latest_DVSCNet.json",
}


def load_latest_logs() -> dict:
    missing_logs = [path for path in MODEL_LOGS.values() if not path.exists()]
    if missing_logs:
        missing_text = "\n".join(f"  - {path.relative_to(PROJECT_ROOT)}" for path in missing_logs)
        raise FileNotFoundError(
            "Latest training logs are missing. Train all three models before plotting:\n" + missing_text
        )

    records = {}
    for model_name, log_path in MODEL_LOGS.items():
        with log_path.open("r", encoding="utf-8") as log_file:
            record = json.load(log_file)
        history = record.get("history", {})
        required_metrics = {"train_loss", "train_accuracy", "val_accuracy"}
        missing_metrics = required_metrics.difference(history)
        if missing_metrics:
            raise ValueError(f"{log_path.name} is missing metrics: {sorted(missing_metrics)}")
        records[model_name] = record

    protocol_fields = (
        "preprocessing_signature",
        "split_strategy",
        "train_sampling",
        "feature_normalization",
    )
    reference_name = next(iter(records))
    reference_dataset = records[reference_name].get("dataset", {})
    for field in protocol_fields:
        reference_value = reference_dataset.get(field)
        if reference_value is None:
            raise ValueError(f"{reference_name} log is missing dataset protocol field: {field}")
        mismatches = {
            model_name: record.get("dataset", {}).get(field)
            for model_name, record in records.items()
            if record.get("dataset", {}).get(field) != reference_value
        }
        if mismatches:
            raise ValueError(
                f"Cannot compare logs with different {field}: "
                f"expected {reference_value!r}, got {mismatches}"
            )
    return records


def plot_comparison(records: dict, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    styles = {
        "PVSCNet": {"color": "#2364AA", "linestyle": "-"},
        "VesselCNN": {"color": "#D1495B", "linestyle": "--"},
        "DVSCNet": {"color": "#2A9D8F", "linestyle": "-."},
    }
    metric_panels = (
        ("train_loss", "Training Loss", "Loss", None),
        ("train_accuracy", "Training Accuracy", "Accuracy", (0, 1)),
        ("val_accuracy", "Validation Accuracy", "Accuracy", (0, 1)),
    )

    for axis, (metric, title, ylabel, ylim) in zip(axes, metric_panels):
        for model_name, record in records.items():
            values = record["history"][metric]
            epochs = np.arange(1, len(values) + 1)
            mark_every = max(1, len(epochs) // 10)
            axis.plot(
                epochs,
                values,
                label=model_name,
                linewidth=2,
                marker="o",
                markersize=5,
                markevery=mark_every,
                **styles[model_name],
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_ylim(ylim)
        if all(len(record["history"][metric]) == 1 for record in records.values()):
            axis.set_xlim(0.5, 1.5)
            axis.set_xticks([1])
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle("Three-Model Training Comparison", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot three-model curves from the latest JSON logs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "curve_comparison_three_models.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_latest_logs()
    plot_comparison(records, output_path)
    print(f"Comparison curve saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()