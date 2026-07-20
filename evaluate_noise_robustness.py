import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from evaluate_enhanced_models import MODEL_DIRS, build_model, load_validation_data
from training_utils import _extract_logits


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("PVSCNet", "VesselCNN", "DVSCNet")
MODEL_COLORS = {
    "PVSCNet": "#2364AA",
    "VesselCNN": "#D1495B",
    "DVSCNet": "#2A9D8F",
}


def load_models(class_count: int, raw_auxiliary_features: np.ndarray, device: torch.device):
    models = {}
    auxiliary_features = {}
    preprocessing_signatures = set()

    for model_name in MODEL_NAMES:
        model_dir = MODEL_DIRS[model_name]
        log = json.loads(
            (model_dir / f"log_latest_{model_name}.json").read_text(encoding="utf-8")
        )
        selector = joblib.load(model_dir / f"selector_best_{model_name}.joblib")
        selected_auxiliary = selector.transform(raw_auxiliary_features).astype(
            np.float32,
            copy=False,
        )
        model = build_model(
            model_name,
            log["training_config"],
            class_count,
            selected_auxiliary.shape[1],
        ).to(device)
        checkpoint = torch.load(
            model_dir / f"checkpoint_best_{model_name}.pt",
            map_location=device,
        )
        checkpoint_signature = checkpoint["preprocessing_signature"]
        if checkpoint_signature != log["dataset"]["preprocessing_signature"]:
            raise ValueError(
                f"{model_name} checkpoint and latest log use different preprocessing"
            )
        preprocessing_signatures.add(checkpoint_signature)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[model_name] = model
        auxiliary_features[model_name] = selected_auxiliary

    if len(preprocessing_signatures) != 1:
        raise ValueError(
            f"Model preprocessing signatures differ: {preprocessing_signatures}"
        )
    return models, auxiliary_features, preprocessing_signatures.pop()


def predict_accuracy(
    model: torch.nn.Module,
    features: np.ndarray,
    auxiliary_features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(1),
        torch.from_numpy(np.ascontiguousarray(auxiliary_features)),
        torch.from_numpy(np.ascontiguousarray(targets)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    expected = []
    with torch.no_grad():
        for batch_features, batch_auxiliary, batch_targets in loader:
            logits = _extract_logits(
                model(
                    batch_features.to(device, non_blocking=True),
                    batch_auxiliary.to(device, non_blocking=True),
                )
            )
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            expected.extend(batch_targets.tolist())
    return float(accuracy_score(expected, predictions))


def run_experiment(
    noise_strengths: list[float],
    repeats: int,
    batch_size: int,
    seed: int,
) -> dict:
    reference_log = json.loads(
        (MODEL_DIRS["PVSCNet"] / "log_latest_PVSCNet.json").read_text(
            encoding="utf-8"
        )
    )
    config = reference_log["training_config"]
    (
        features,
        raw_auxiliary_features,
        targets,
        groups,
        class_names,
        train_indices,
        validation_indices,
    ) = load_validation_data(config["seed"], config["val_split"])
    if set(groups[train_indices]) & set(groups[validation_indices]):
        raise RuntimeError("Source-file leakage detected during evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, auxiliary_by_model, preprocessing_signature = load_models(
        len(class_names),
        raw_auxiliary_features,
        device,
    )
    validation_features = features[validation_indices]
    validation_targets = targets[validation_indices]
    trial_accuracies = {
        model_name: [[] for _ in noise_strengths] for model_name in MODEL_NAMES
    }

    for strength_index, noise_strength in enumerate(noise_strengths):
        trial_count = 1 if noise_strength == 0 else repeats
        for repeat_index in range(trial_count):
            random_state = np.random.default_rng(
                seed + strength_index * repeats + repeat_index
            )
            if noise_strength == 0:
                noisy_features = validation_features
            else:
                noise = random_state.normal(
                    loc=0.0,
                    scale=noise_strength,
                    size=validation_features.shape,
                ).astype(np.float32)
                noisy_features = np.clip(validation_features + noise, 0.0, 1.0)

            for model_name in MODEL_NAMES:
                accuracy = predict_accuracy(
                    models[model_name],
                    noisy_features,
                    auxiliary_by_model[model_name][validation_indices],
                    validation_targets,
                    batch_size,
                    device,
                )
                trial_accuracies[model_name][strength_index].append(accuracy)

    model_results = {}
    for model_name in MODEL_NAMES:
        model_results[model_name] = {
            "mean_accuracy": [
                float(np.mean(values)) for values in trial_accuracies[model_name]
            ],
            "std_accuracy": [
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                for values in trial_accuracies[model_name]
            ],
            "trial_accuracies": trial_accuracies[model_name],
        }

    return {
        "device": str(device),
        "noise": {
            "type": "additive_zero_mean_gaussian",
            "strength_definition": "standard deviation on normalized [0, 1] features",
            "clipping_range": [0.0, 1.0],
            "strengths": noise_strengths,
            "repeats_per_nonzero_strength": repeats,
            "seed": seed,
        },
        "validation_samples": int(len(validation_indices)),
        "validation_source_files": int(len(np.unique(groups[validation_indices]))),
        "preprocessing_signature": preprocessing_signature,
        "models": model_results,
    }


def plot_results(results: dict, output_path: Path) -> None:
    noise_strengths = results["noise"]["strengths"]
    positions = np.arange(len(noise_strengths))
    bar_width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.8))

    for model_index, model_name in enumerate(MODEL_NAMES):
        model_result = results["models"][model_name]
        offsets = positions + (model_index - 1) * bar_width
        axis.bar(
            offsets,
            np.asarray(model_result["mean_accuracy"]) * 100.0,
            bar_width,
            yerr=np.asarray(model_result["std_accuracy"]) * 100.0,
            capsize=3,
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.8,
        )

    axis.set_xlabel("Noise Strength (Standard Deviation)")
    axis.set_ylabel("Classification Accuracy (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{strength:.2f}" for strength in noise_strengths])
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_noise_strengths(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("At least one noise strength is required")
    if any(value < 0 for value in values):
        raise ValueError("Noise strengths must be non-negative")
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate three classifiers under additive Gaussian feature noise."
    )
    parser.add_argument(
        "--noise-strengths",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "noise_robustness_three_models.png",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "noise_robustness_results.json",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    noise_strengths = parse_noise_strengths(args.noise_strengths)
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    results_path = args.results if args.results.is_absolute() else PROJECT_ROOT / args.results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_experiment(
        noise_strengths,
        args.repeats,
        args.batch_size,
        args.seed,
    )
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_results(results, output_path)

    for model_name in MODEL_NAMES:
        accuracies = results["models"][model_name]["mean_accuracy"]
        summary = ", ".join(
            f"sigma={strength:.2f}: {accuracy:.4f}"
            for strength, accuracy in zip(noise_strengths, accuracies)
        )
        print(f"{model_name}: {summary}")
    print(f"Results saved to: {results_path.relative_to(PROJECT_ROOT)}")
    print(f"Figure saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()