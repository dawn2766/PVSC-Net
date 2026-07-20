import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from DVSCNet.model_DVSCNet import DVSCNet
from PVSCNet.model_PVSCNet import PVSCNet
from VesselCNN.model_VesselCNN import VesselCNN
from training_utils import _extract_logits, _stratified_group_split


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIRS = {
    "PVSCNet": PROJECT_ROOT / "PVSCNet",
    "VesselCNN": PROJECT_ROOT / "VesselCNN",
    "DVSCNet": PROJECT_ROOT / "DVSCNet",
}


def load_validation_data(seed: int, val_split: float):
    processed_dir = PROJECT_ROOT / "processed"
    features = np.load(processed_dir / "X.npy").astype(np.float32)
    raw_auxiliary_features = np.load(
        processed_dir / "auxiliary_features.npy"
    ).astype(np.float32)
    targets = np.load(processed_dir / "y.npy").astype(np.int64)
    groups = np.load(processed_dir / "groups.npy").astype(str)
    class_names = np.load(processed_dir / "labels.npy", allow_pickle=True).astype(str).tolist()

    train_indices, validation_indices = _stratified_group_split(
        targets,
        groups,
        val_split,
        seed,
    )
    value_min = float(features[train_indices].min())
    value_range = float(features[train_indices].max() - value_min)
    if value_range > 0:
        features = (features - value_min) / value_range
    else:
        features = np.zeros_like(features)
    return (
        features,
        raw_auxiliary_features,
        targets,
        groups,
        class_names,
        train_indices,
        validation_indices,
    )


def build_model(model_name: str, config: dict, class_count: int, auxiliary_dim: int):
    if model_name == "PVSCNet":
        return PVSCNet(
            class_count,
            (64, 157),
            z_dim=config["z_dim"],
            latent_noise_scale=config["latent_noise_scale"],
            auxiliary_dim=auxiliary_dim,
        )
    if model_name == "VesselCNN":
        return VesselCNN(
            class_count,
            (64, 157),
            auxiliary_dim=auxiliary_dim,
        )
    if model_name == "DVSCNet":
        return DVSCNet(
            num_classes=class_count,
            z_dim=config["z_dim"],
            dropout=config["dropout"],
            latent_noise_scale=config["latent_noise_scale"],
            kl_weight=config["kl_weight"],
            use_spec_augment=config["use_spec_augment"],
            auxiliary_dim=auxiliary_dim,
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    model_name: str,
    features: np.ndarray,
    raw_auxiliary_features: np.ndarray,
    targets: np.ndarray,
    validation_indices: np.ndarray,
    class_names: list,
    device: torch.device,
):
    model_dir = MODEL_DIRS[model_name]
    log = json.loads(
        (model_dir / f"log_latest_{model_name}.json").read_text(encoding="utf-8")
    )
    selector = joblib.load(model_dir / f"selector_best_{model_name}.joblib")
    auxiliary_features = selector.transform(raw_auxiliary_features)
    auxiliary_dim = auxiliary_features.shape[1]
    model = build_model(
        model_name,
        log["training_config"],
        len(class_names),
        auxiliary_dim,
    ).to(device)
    checkpoint = torch.load(
        model_dir / f"checkpoint_best_{model_name}.pt",
        map_location=device,
    )
    if checkpoint["preprocessing_signature"] != log["dataset"]["preprocessing_signature"]:
        raise ValueError(f"{model_name} checkpoint and latest log use different preprocessing")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(features[validation_indices]).unsqueeze(1),
        torch.from_numpy(auxiliary_features[validation_indices]),
        torch.from_numpy(targets[validation_indices]),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
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

    recalls = recall_score(
        expected,
        predictions,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
    return {
        "best_epoch": int(checkpoint["epoch"]),
        "accuracy": float(accuracy_score(expected, predictions)),
        "macro_f1": float(f1_score(expected, predictions, average="macro")),
        "recall_by_class": {
            class_name: float(recall)
            for class_name, recall in zip(class_names, recalls)
        },
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "validation_samples": len(expected),
        "preprocessing_signature": checkpoint["preprocessing_signature"],
        "auxiliary_dim": int(auxiliary_dim),
    }


def main() -> None:
    reference_log = json.loads(
        (MODEL_DIRS["PVSCNet"] / "log_latest_PVSCNet.json").read_text(encoding="utf-8")
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
    results = {
        "device": str(device),
        "validation_source_files": int(len(np.unique(groups[validation_indices]))),
        "models": {},
    }
    for model_name in MODEL_DIRS:
        results["models"][model_name] = evaluate_model(
            model_name,
            features,
            raw_auxiliary_features,
            targets,
            validation_indices,
            class_names,
            device,
        )

    signatures = {
        result["preprocessing_signature"]
        for result in results["models"].values()
    }
    if len(signatures) != 1:
        raise ValueError(f"Model preprocessing signatures differ: {signatures}")

    output_path = PROJECT_ROOT / "evaluation_summary_enhanced_features.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for model_name, result in results["models"].items():
        print(
            f"{model_name}: accuracy={result['accuracy']:.6f} "
            f"macro_f1={result['macro_f1']:.6f} epoch={result['best_epoch']}"
        )
    print(f"Evaluation summary saved to: {output_path.name}")


if __name__ == "__main__":
    main()
