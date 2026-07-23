import csv
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PVSCNET_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PVSCNET_DIR / "comparisons" / "results"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PVSCNET_DIR))

from comparison_models import DeterministicFusionNet, SpectrogramOnlyPVNet
from model_PVSCNet import PVSCNet
from training_utils import _extract_logits, _stratified_group_split


MODEL_NAMES = ("PVSCNet", "DeterministicFusionNet", "SpectrogramOnlyPVNet")
DISPLAY_NAMES = {
    "PVSCNet": "PVSC-Net",
    "DeterministicFusionNet": "确定性融合模型",
    "SpectrogramOnlyPVNet": "仅频谱概率模型",
}
COLORS = {
    "PVSCNet": "#0072B2",
    "DeterministicFusionNet": "#D55E00",
    "SpectrogramOnlyPVNet": "#009E73",
}
SEED_LOCATIONS = {
    2024: {
        model_name: PVSCNET_DIR / "comparisons" / "seed_2024" / model_name
        for model_name in MODEL_NAMES
    },
    2025: {
        model_name: PVSCNET_DIR / "comparisons" / "seed_2025" / model_name
        for model_name in MODEL_NAMES
    },
    2026: {
        "PVSCNet": PVSCNET_DIR,
        "DeterministicFusionNet": (
            PVSCNET_DIR / "comparisons" / "DeterministicFusionNet"
        ),
        "SpectrogramOnlyPVNet": (
            PVSCNET_DIR / "comparisons" / "SpectrogramOnlyPVNet"
        ),
    },
}


def _load_arrays(seed: int, val_split: float):
    processed_dir = PROJECT_ROOT / "processed"
    features = np.load(processed_dir / "X.npy").astype(np.float32)
    raw_auxiliary = np.load(processed_dir / "auxiliary_features.npy").astype(np.float32)
    targets = np.load(processed_dir / "y.npy").astype(np.int64)
    groups = np.load(processed_dir / "groups.npy").astype(str)
    class_names = np.load(
        processed_dir / "labels.npy", allow_pickle=True
    ).astype(str).tolist()
    train_indices, validation_indices = _stratified_group_split(
        targets, groups, val_split, seed
    )
    value_min = float(features[train_indices].min())
    value_range = float(features[train_indices].max() - value_min)
    features = (
        (features - value_min) / value_range
        if value_range > 0
        else np.zeros_like(features)
    )
    return (
        features,
        raw_auxiliary,
        targets,
        groups,
        class_names,
        train_indices,
        validation_indices,
    )


def _build_model(
    model_name: str,
    config: dict,
    class_count: int,
    input_shape: tuple[int, int],
    auxiliary_dim: int,
):
    if model_name == "PVSCNet":
        return PVSCNet(
            num_classes=class_count,
            input_shape=input_shape,
            z_dim=config["z_dim"],
            latent_noise_scale=config["latent_noise_scale"],
            auxiliary_dim=auxiliary_dim,
        )
    if model_name == "DeterministicFusionNet":
        return DeterministicFusionNet(
            num_classes=class_count,
            z_dim=config["z_dim"],
            auxiliary_dim=auxiliary_dim,
        )
    return SpectrogramOnlyPVNet(
        num_classes=class_count,
        z_dim=config["z_dim"],
        latent_noise_scale=config["latent_noise_scale"],
    )


def _source_level_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    class_count: int,
):
    source_targets = []
    source_predictions = []
    source_records = []
    for source_name in sorted(np.unique(groups)):
        mask = groups == source_name
        unique_targets = np.unique(targets[mask])
        if len(unique_targets) != 1:
            raise ValueError(f"Source file {source_name} contains multiple target classes")
        target = int(unique_targets[0])
        prediction = int(logits[mask].mean(axis=0).argmax())
        source_targets.append(target)
        source_predictions.append(prediction)
        source_records.append(
            {
                "source_file": source_name,
                "target": target,
                "prediction": prediction,
                "window_count": int(mask.sum()),
            }
        )
    return {
        "accuracy": float(accuracy_score(source_targets, source_predictions)),
        "macro_f1": float(
            f1_score(
                source_targets,
                source_predictions,
                labels=np.arange(class_count),
                average="macro",
                zero_division=0,
            )
        ),
        "source_count": len(source_targets),
        "records": source_records,
    }


def _evaluate_model(
    model_name: str,
    model_dir: Path,
    features: np.ndarray,
    raw_auxiliary: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    validation_indices: np.ndarray,
    class_names: list[str],
    device: torch.device,
):
    log = json.loads(
        (model_dir / f"log_latest_{model_name}.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        model_dir / f"checkpoint_best_{model_name}.pt", map_location=device
    )
    selector = joblib.load(model_dir / f"selector_best_{model_name}.joblib")
    auxiliary = selector.transform(raw_auxiliary).astype(np.float32)
    input_shape = tuple(log["dataset"]["input_shape"])
    model = _build_model(
        model_name,
        log["training_config"],
        len(class_names),
        input_shape,
        auxiliary.shape[1],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(features[validation_indices]).unsqueeze(1),
        torch.from_numpy(auxiliary[validation_indices]),
        torch.from_numpy(targets[validation_indices]),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    logits_batches = []
    with torch.no_grad():
        for batch_features, batch_auxiliary, _ in loader:
            logits_batches.append(
                _extract_logits(
                    model(
                        batch_features.to(device, non_blocking=True),
                        batch_auxiliary.to(device, non_blocking=True),
                    )
                )
                .cpu()
                .numpy()
            )
    logits = np.concatenate(logits_batches)
    expected = targets[validation_indices]
    predictions = logits.argmax(axis=1)
    recalls = recall_score(
        expected,
        predictions,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
    source_metrics = _source_level_metrics(
        logits,
        expected,
        groups[validation_indices],
        len(class_names),
    )
    return {
        "best_epoch": int(checkpoint["epoch"]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "window_level": {
            "accuracy": float(accuracy_score(expected, predictions)),
            "macro_f1": float(f1_score(expected, predictions, average="macro")),
            "sample_count": int(len(expected)),
            "recall_by_class": {
                class_name: float(recall)
                for class_name, recall in zip(class_names, recalls)
            },
            "confusion_matrix": confusion_matrix(
                expected, predictions, labels=np.arange(len(class_names))
            ).tolist(),
        },
        "source_level": source_metrics,
        "preprocessing_signature": checkpoint["preprocessing_signature"],
        "validation_source_files": sorted(np.unique(groups[validation_indices]).tolist()),
    }


def _summarize(results: dict):
    summary = {}
    for model_name in MODEL_NAMES:
        model_results = [
            results["seeds"][str(seed)]["models"][model_name]
            for seed in sorted(SEED_LOCATIONS)
        ]
        summary[model_name] = {}
        for level in ("window_level", "source_level"):
            summary[model_name][level] = {}
            for metric in ("accuracy", "macro_f1"):
                values = np.asarray(
                    [result[level][metric] for result in model_results], dtype=float
                )
                summary[model_name][level][metric] = {
                    "values": values.tolist(),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
        summary[model_name]["parameter_count"] = model_results[0]["parameter_count"]

    reference = np.asarray(
        summary["PVSCNet"]["window_level"]["accuracy"]["values"]
    )
    for model_name in MODEL_NAMES[1:]:
        comparator = np.asarray(
            summary[model_name]["window_level"]["accuracy"]["values"]
        )
        differences = reference - comparator
        summary[model_name]["paired_accuracy_difference_from_pvscnet"] = {
            "values": differences.tolist(),
            "mean": float(differences.mean()),
            "sample_std": float(differences.std(ddof=1)),
        }
    return summary


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def _save_architecture_figure() -> None:
    _apply_style()
    figure, axis = plt.subplots(figsize=(10.4, 4.8))
    axis.set_xlim(0, 12.5)
    axis.set_ylim(0, 6.1)
    axis.axis("off")

    boxes = [
        (0.2, 3.5, 1.35, 1.1, "对数梅尔谱\n64 x 157", "#D9EAF7"),
        (2.0, 3.25, 2.0, 1.6, "四级卷积编码器\n32/64/128/256\n3 x 3，步长2", "#BFD7EA"),
        (4.5, 3.5, 1.65, 1.1, "全局平均与\n最大池化", "#D9EAF7"),
        (0.55, 0.85, 2.0, 1.35, "33维人工\n声学特征", "#F6D7B0"),
        (3.05, 0.85, 1.8, 1.35, "仅训练集拟合\nPCA12 + LDA3", "#F6D7B0"),
        (5.35, 0.85, 1.55, 1.35, "辅助编码器\n3 -> 32 -> 32", "#F6D7B0"),
        (7.25, 2.75, 1.55, 1.25, "特征拼接\n512 + 32", "#E8E8E8"),
        (9.2, 2.75, 1.55, 1.25, "融合层\n544 -> 256", "#E8E8E8"),
        (11.05, 3.65, 1.15, 0.85, "均值mu\n256 -> 16", "#CDECCF"),
        (11.05, 2.45, 1.15, 0.85, "对数方差\n256 -> 16", "#CDECCF"),
        (9.6, 0.75, 1.65, 1.1, "高斯采样\nz=mu+0.1eps sigma", "#CDECCF"),
        (11.55, 0.75, 0.75, 1.1, "分类头\n16 -> 4", "#B7DFD0"),
    ]
    for x_value, y_value, width, height, text, color in boxes:
        axis.add_patch(
            Rectangle(
                (x_value, y_value),
                width,
                height,
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        axis.text(
            x_value + width / 2,
            y_value + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
        )

    arrows = [
        ((1.55, 4.05), (2.0, 4.05)),
        ((4.0, 4.05), (4.5, 4.05)),
        ((6.15, 4.05), (7.25, 3.55)),
        ((2.55, 1.52), (3.05, 1.52)),
        ((4.85, 1.52), (5.35, 1.52)),
        ((6.9, 1.52), (7.25, 3.05)),
        ((8.8, 3.38), (9.2, 3.38)),
        ((10.75, 3.38), (11.05, 4.08)),
        ((10.75, 3.38), (11.05, 2.88)),
        ((11.62, 3.65), (10.75, 1.85)),
        ((11.62, 2.45), (10.75, 1.85)),
        ((11.25, 1.3), (11.55, 1.3)),
    ]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color="#333333",
                connectionstyle="arc3,rad=0.0",
            )
        )
    axis.text(0.2, 5.55, "PVSC-Net网络结构", fontsize=12, weight="bold")
    axis.text(
        0.2,
        5.15,
        "训练阶段启用随机潜变量采样，推理阶段采用潜变量均值。",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "figure_1_pvscnet_architecture.png", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "figure_1_pvscnet_architecture.pdf", bbox_inches="tight")
    plt.close(figure)


def _save_comparison_figure(results: dict) -> None:
    _apply_style()
    seeds = sorted(SEED_LOCATIONS)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    x_positions = np.arange(len(MODEL_NAMES))
    means = [
        results["summary"][model_name]["window_level"]["accuracy"]["mean"]
        for model_name in MODEL_NAMES
    ]
    standard_deviations = [
        results["summary"][model_name]["window_level"]["accuracy"]["sample_std"]
        for model_name in MODEL_NAMES
    ]
    axes[0].bar(
        x_positions,
        means,
        yerr=standard_deviations,
        capsize=4,
        color=[COLORS[model_name] for model_name in MODEL_NAMES],
        edgecolor="#222222",
        linewidth=0.8,
    )
    axes[0].set_xticks(x_positions, [DISPLAY_NAMES[name] for name in MODEL_NAMES], rotation=15, ha="right")
    axes[0].set_ylabel("窗口级准确率")
    axes[0].set_ylim(0.55, 0.95)
    axes[0].set_title("三种子均值与样本标准差")
    axes[0].grid(axis="y", alpha=0.25)
    for x_value, mean in zip(x_positions, means):
        axes[0].text(x_value, mean + 0.012, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    for model_name in MODEL_NAMES:
        values = [
            results["seeds"][str(seed)]["models"][model_name]["window_level"]["accuracy"]
            for seed in seeds
        ]
        axes[1].plot(
            seeds,
            values,
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=COLORS[model_name],
            label=DISPLAY_NAMES[model_name],
        )
    axes[1].set_xticks(seeds)
    axes[1].set_xlabel("划分及训练种子")
    axes[1].set_ylabel("窗口级准确率")
    axes[1].set_ylim(0.55, 0.95)
    axes[1].set_title("各文件分组划分上的配对结果")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontsize=11, weight="bold")
    figure.tight_layout(w_pad=2.5)
    figure.savefig(OUTPUT_DIR / "figure_2_comparison_results.png", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "figure_2_comparison_results.pdf", bbox_inches="tight")
    plt.close(figure)


def _save_confusion_figure(results: dict) -> None:
    _apply_style()
    class_names = ["货船", "客船", "油轮", "拖船"]
    matrix = np.asarray(
        results["seeds"]["2026"]["models"]["PVSCNet"]["window_level"][
            "confusion_matrix"
        ]
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n({value:.1%})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "#222222",
            )
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set_xlabel("预测类别")
    axis.set_ylabel("真实类别")
    axis.set_title("PVSC-Net混淆矩阵（种子2026）")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="按行归一化比例")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "figure_3_pvscnet_confusion.png", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "figure_3_pvscnet_confusion.pdf", bbox_inches="tight")
    plt.close(figure)


def _write_csv(results: dict) -> None:
    rows = []
    for seed in sorted(SEED_LOCATIONS):
        for model_name in MODEL_NAMES:
            metrics = results["seeds"][str(seed)]["models"][model_name]
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "best_epoch": metrics["best_epoch"],
                    "parameters": metrics["parameter_count"],
                    "window_accuracy": metrics["window_level"]["accuracy"],
                    "window_macro_f1": metrics["window_level"]["macro_f1"],
                    "source_accuracy": metrics["source_level"]["accuracy"],
                    "source_macro_f1": metrics["source_level"]["macro_f1"],
                    "validation_windows": metrics["window_level"]["sample_count"],
                    "validation_sources": metrics["source_level"]["source_count"],
                }
            )
    with (OUTPUT_DIR / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "schema_version": 1,
        "device": str(device),
        "experiment_design": {
            "type": "three-seed paired controlled ablation",
            "seeds": sorted(SEED_LOCATIONS),
            "split": "class-stratified source-file grouped holdout",
            "validation_fraction": 0.2,
            "epochs": 35,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "early_stopping": False,
            "selection": "highest validation window accuracy within each 35-epoch run",
            "limitations": (
                "Three repeats support descriptive robustness analysis only; "
                "no inferential significance claim is made."
            ),
        },
        "seeds": {},
    }

    for seed, model_locations in SEED_LOCATIONS.items():
        reference_log = json.loads(
            (
                model_locations["PVSCNet"] / "log_latest_PVSCNet.json"
            ).read_text(encoding="utf-8")
        )
        config = reference_log["training_config"]
        (
            features,
            raw_auxiliary,
            targets,
            groups,
            class_names,
            train_indices,
            validation_indices,
        ) = _load_arrays(seed, config["val_split"])
        overlap = sorted(
            set(groups[train_indices]) & set(groups[validation_indices])
        )
        if overlap:
            raise RuntimeError(f"Source-file leakage for seed {seed}: {overlap}")
        seed_result = {
            "train_windows": int(len(train_indices)),
            "validation_windows": int(len(validation_indices)),
            "train_source_files": int(len(np.unique(groups[train_indices]))),
            "validation_source_files": int(len(np.unique(groups[validation_indices]))),
            "source_file_overlap": overlap,
            "models": {},
        }
        for model_name, model_dir in model_locations.items():
            seed_result["models"][model_name] = _evaluate_model(
                model_name,
                model_dir,
                features,
                raw_auxiliary,
                targets,
                groups,
                validation_indices,
                class_names,
                device,
            )
        results["seeds"][str(seed)] = seed_result

    results["class_names"] = class_names
    results["summary"] = _summarize(results)
    output_path = OUTPUT_DIR / "comparison_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(results)
    _save_architecture_figure()
    _save_comparison_figure(results)
    _save_confusion_figure(results)

    for model_name in MODEL_NAMES:
        metrics = results["summary"][model_name]["window_level"]
        print(
            f"{model_name}: accuracy={metrics['accuracy']['mean']:.6f} "
            f"+/- {metrics['accuracy']['sample_std']:.6f}; "
            f"macro_f1={metrics['macro_f1']['mean']:.6f} "
            f"+/- {metrics['macro_f1']['sample_std']:.6f}"
        )
    print(f"Results written to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()