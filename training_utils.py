import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    val_split: float
    seed: int
    num_workers: int


class VesselDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index].unsqueeze(0), self.targets[index]


def create_argument_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {model_name} from the project root.")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NUM_EPOCHS", "100")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "256")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LR", "1e-4")))
    parser.add_argument("--val-split", type=float, default=float(os.getenv("VAL_SPLIT", "0.2")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "0")))
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stratified_split(targets: np.ndarray, val_split: float, seed: int):
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    random_state = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        random_state.shuffle(class_indices)
        if len(class_indices) < 2:
            raise ValueError(f"Class {class_id} needs at least two samples for a train/validation split")
        val_count = min(max(1, int(np.ceil(len(class_indices) * val_split))), len(class_indices) - 1)
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    random_state.shuffle(train_indices)
    random_state.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def _stratified_group_split(
    targets: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    seed: int,
):
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")
    if len(targets) != len(groups):
        raise ValueError("targets and groups must contain the same number of samples")

    random_state = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        class_groups, group_counts = np.unique(groups[class_indices], return_counts=True)
        if len(class_groups) < 2:
            raise ValueError(f"Class {class_id} needs at least two source files for a grouped split")

        order = random_state.permutation(len(class_groups))
        class_groups = class_groups[order]
        group_counts = group_counts[order]

        reachable = {0: 0}
        for group_index, count in enumerate(group_counts):
            for subtotal, mask in list(reachable.items()):
                candidate = subtotal + int(count)
                if candidate not in reachable:
                    reachable[candidate] = mask | (1 << group_index)

        target_count = len(class_indices) * val_split
        valid_counts = [count for count in reachable if 0 < count < len(class_indices)]
        selected_count = min(valid_counts, key=lambda count: (abs(count - target_count), count))
        selected_mask = reachable[selected_count]
        selected_groups = {
            class_groups[index]
            for index in range(len(class_groups))
            if selected_mask & (1 << index)
        }
        is_validation = np.isin(groups[class_indices], list(selected_groups))
        val_indices.extend(class_indices[is_validation].tolist())
        train_indices.extend(class_indices[~is_validation].tolist())

    random_state.shuffle(train_indices)
    random_state.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def _load_data(project_root: Path, config: TrainingConfig):
    processed_dir = project_root / "processed"
    features = np.load(processed_dir / "X.npy")
    targets = np.load(processed_dir / "y.npy")
    groups_path = processed_dir / "groups.npy"
    if not groups_path.exists():
        raise FileNotFoundError(
            "processed/groups.npy is required for leakage-free evaluation; rerun data_preprocess.py"
        )
    groups = np.load(groups_path, allow_pickle=False).astype(str)
    class_names = np.load(processed_dir / "labels.npy", allow_pickle=True).astype(str).tolist()

    if features.ndim == 4 and features.shape[-1] == 1:
        features = features[..., 0]
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape (N, H, W), got {features.shape}")

    features = features.astype(np.float32, copy=False)
    train_indices, val_indices = _stratified_group_split(targets, groups, config.val_split, config.seed)
    value_min = float(features[train_indices].min())
    value_range = float(features[train_indices].max() - value_min)
    if value_range > 0:
        features = (features - value_min) / value_range
    else:
        features = np.zeros_like(features)

    dataset = VesselDataset(features, targets)
    generator = torch.Generator().manual_seed(config.seed)

    train_targets = targets[train_indices]
    train_groups = groups[train_indices]
    class_group_counts = {
        class_id: len(np.unique(train_groups[train_targets == class_id]))
        for class_id in np.unique(train_targets)
    }
    group_sample_counts = dict(zip(*np.unique(train_groups, return_counts=True)))
    sample_weights = np.asarray(
        [
            1.0 / (class_group_counts[class_id] * group_sample_counts[group_id])
            for class_id, group_id in zip(train_targets, train_groups)
        ],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names, features.shape[1:], targets, groups, train_indices, val_indices


def _extract_logits(outputs: torch.Tensor) -> torch.Tensor:
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _run_epoch(model, data_loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in data_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = _extract_logits(model(features))
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def _collect_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    targets_all = []
    with torch.no_grad():
        for features, targets in data_loader:
            logits = _extract_logits(model(features.to(device, non_blocking=True)))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.tolist())
    return targets_all, predictions


def _save_training_curve(history: dict, model_name: str, output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    mark_every = max(1, len(epochs) // 10)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=mark_every,
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=mark_every,
    )
    axes[0].set(title=f"{model_name} Loss", xlabel="Epoch", ylabel="Loss")

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=mark_every,
    )
    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation",
        linewidth=2,
        marker="o",
        markersize=4,
        markevery=mark_every,
    )
    axes[1].set(title=f"{model_name} Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))

    for axis in axes:
        if len(epochs) == 1:
            axis.set_xlim(0.5, 1.5)
            axis.set_xticks([1])
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_matrix(targets, predictions, class_names, model_name: str, output_path: Path) -> None:
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(class_names)))
    figure, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=axis, xticks_rotation=45, colorbar=False)
    axis.set_title(f"{model_name} Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _read_best_accuracy(
    path: Path,
    split_strategy: str,
    train_sampling: str,
    feature_normalization: str,
    preprocessing_signature: str,
) -> float:
    if not path.exists():
        return float("-inf")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset", {})
        if (
            dataset.get("split_strategy") != split_strategy
            or dataset.get("train_sampling") != train_sampling
            or dataset.get("feature_normalization") != feature_normalization
            or dataset.get("preprocessing_signature") != preprocessing_signature
        ):
            return float("-inf")
        return float(payload["result_summary"]["best_val_accuracy"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return float("-inf")


def train_experiment(
    model_name: str,
    project_root: Path,
    model_dir: Path,
    config: TrainingConfig,
    build_model: Callable[[int, tuple[int, int]], torch.nn.Module],
    build_optimizer: Callable,
    criterion: Callable,
    extra_config: Optional[dict] = None,
) -> dict:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")

    set_seed(config.seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, class_names, input_shape, targets, groups, train_indices, val_indices = _load_data(
        project_root, config
    )
    preprocess_config_path = project_root / "processed" / "preprocess_config.json"
    if not preprocess_config_path.exists():
        raise FileNotFoundError(
            "processed/preprocess_config.json is required; rerun data_preprocess.py"
        )
    preprocess_config = json.loads(preprocess_config_path.read_text(encoding="utf-8"))
    preprocessing_signature = hashlib.sha256(
        json.dumps(preprocess_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    train_source_file_names = sorted(np.unique(groups[train_indices]).tolist())
    val_source_file_names = sorted(np.unique(groups[val_indices]).tolist())
    source_file_overlap = sorted(set(train_source_file_names) & set(val_source_file_names))
    if source_file_overlap:
        raise RuntimeError(f"Source-file leakage detected: {source_file_overlap}")
    if preprocess_config.get("num_samples") != len(targets):
        raise ValueError("Preprocessing metadata sample count does not match processed arrays")
    if preprocess_config.get("num_source_files") != len(np.unique(groups)):
        raise ValueError("Preprocessing metadata source-file count does not match groups.npy")
    split_strategy = "file_group_stratified"
    train_sampling = "class_and_source_file_balanced"
    feature_normalization = "train_only_global_minmax"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), input_shape).to(device)
    optimizer = build_optimizer(model.parameters())

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    run_best_accuracy = float("-inf")
    run_best_epoch = 0
    run_best_path = model_dir / f".checkpoint_run_best_{model_name}.pt"
    latest_checkpoint_path = model_dir / f"checkpoint_latest_{model_name}.pt"
    best_checkpoint_path = model_dir / f"checkpoint_best_{model_name}.pt"
    latest_log_path = model_dir / f"log_latest_{model_name}.json"
    best_log_path = model_dir / f"log_best_{model_name}.json"

    print(f"Training {model_name} on {device}: {len(train_indices)} train / {len(val_indices)} validation samples")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_accuracy = _run_epoch(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > run_best_accuracy:
            run_best_accuracy = val_accuracy
            run_best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "input_shape": input_shape,
                    "preprocessing_signature": preprocessing_signature,
                    "split_seed": config.seed,
                },
                run_best_path,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )

    torch.save(
        {
            "model_name": model_name,
            "epoch": config.epochs,
            "val_accuracy": history["val_accuracy"][-1],
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "input_shape": input_shape,
            "preprocessing_signature": preprocessing_signature,
            "split_seed": config.seed,
        },
        latest_checkpoint_path,
    )

    run_best_checkpoint = torch.load(run_best_path, map_location=device)
    model.load_state_dict(run_best_checkpoint["model_state_dict"])
    val_targets, val_predictions = _collect_predictions(model, val_loader, device)

    curve_path = model_dir / f"curve_training_{model_name}.png"
    matrix_path = model_dir / f"matrix_confusion_{model_name}.png"
    _save_training_curve(history, model_name, curve_path)
    _save_confusion_matrix(val_targets, val_predictions, class_names, model_name, matrix_path)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    record = {
        "schema_version": 1,
        "model_name": model_name,
        "created_at": timestamp,
        "device": str(device),
        "training_config": {**asdict(config), **(extra_config or {})},
        "dataset": {
            "input_shape": list(input_shape),
            "class_names": class_names,
            "split_strategy": split_strategy,
            "train_sampling": train_sampling,
            "feature_normalization": feature_normalization,
            "preprocessing_signature": preprocessing_signature,
            "preprocessing": preprocess_config,
            "total_samples": int(len(targets)),
            "total_source_files": int(len(np.unique(groups))),
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            "train_source_files": int(len(np.unique(groups[train_indices]))),
            "val_source_files": int(len(np.unique(groups[val_indices]))),
            "source_file_overlap": source_file_overlap,
            "train_source_file_names": train_source_file_names,
            "val_source_file_names": val_source_file_names,
        },
        "history": history,
        "result_summary": {
            "best_epoch": run_best_epoch,
            "best_val_accuracy": run_best_accuracy,
            "latest_val_accuracy": history["val_accuracy"][-1],
            "latest_val_loss": history["val_loss"][-1],
        },
        "artifacts": {
            "latest_checkpoint": latest_checkpoint_path.relative_to(project_root).as_posix(),
            "best_checkpoint": best_checkpoint_path.relative_to(project_root).as_posix(),
            "training_curve": curve_path.relative_to(project_root).as_posix(),
            "confusion_matrix": matrix_path.relative_to(project_root).as_posix(),
        },
    }

    previous_best_accuracy = _read_best_accuracy(
        best_log_path,
        split_strategy,
        train_sampling,
        feature_normalization,
        preprocessing_signature,
    )
    _write_json(latest_log_path, record)
    if run_best_accuracy > previous_best_accuracy:
        run_best_path.replace(best_checkpoint_path)
        _write_json(best_log_path, record)
        print(f"Updated historical best result: {run_best_accuracy:.4f}")
    else:
        run_best_path.unlink(missing_ok=True)
        print(f"Historical best remains {previous_best_accuracy:.4f}")

    print(f"Latest log: {latest_log_path.relative_to(project_root)}")
    print(f"Training curve: {curve_path.relative_to(project_root)}")
    print(f"Confusion matrix: {matrix_path.relative_to(project_root)}")
    return record