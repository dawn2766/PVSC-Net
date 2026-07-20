import json
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "PVSCNet"))
sys.path.insert(0, str(PROJECT_ROOT / "VesselCNN"))
sys.path.insert(0, str(PROJECT_ROOT / "DVSCNet"))

from data_preprocess import SAMPLE_RATE, extract_mel
from feature_extraction.feature_selection import PCALDASelector
from feature_extraction.features_aux import AUXILIARY_FEATURE_DIM, extract_aux_features
from model_DVSCNet import DVSCNet
from model_PVSCNet import PVSCNet
from model_VesselCNN import VesselCNN
from training_utils import TrainingConfig, _load_data


def validate_real_audio_features() -> None:
    audio_path = PROJECT_ROOT / "data" / "Cargo" / "103.wav"
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=5.0)
    mel = extract_mel(audio, SAMPLE_RATE)
    auxiliary = extract_aux_features(audio, sr=SAMPLE_RATE)
    assert mel.shape == (64, 157)
    assert auxiliary.shape == (AUXILIARY_FEATURE_DIM,)
    assert np.isfinite(mel).all()
    assert np.isfinite(auxiliary).all()


def validate_selector() -> None:
    random_generator = np.random.default_rng(17)
    features = random_generator.normal(size=(80, AUXILIARY_FEATURE_DIM)).astype(np.float32)
    labels = np.repeat(np.arange(4), 20)
    features += labels[:, None] * 0.15
    selector = PCALDASelector()
    train_selected = selector.fit_transform(features[:64], labels[:64])
    validation_selected = selector.transform(features[64:])
    assert train_selected.shape == (64, 3)
    assert validation_selected.shape == (16, 3)
    assert np.isfinite(train_selected).all()
    assert np.isfinite(validation_selected).all()
    np.testing.assert_allclose(train_selected.mean(axis=0), 0.0, atol=1e-5)


def validate_training_loader() -> None:
    random_generator = np.random.default_rng(23)
    samples_per_group = 3
    groups_per_class = 2
    class_count = 4
    sample_count = samples_per_group * groups_per_class * class_count
    targets = np.repeat(np.arange(class_count), samples_per_group * groups_per_class)
    groups = np.asarray(
        [
            f"class_{class_id}/source_{group_id}.wav"
            for class_id in range(class_count)
            for group_id in range(groups_per_class)
            for _ in range(samples_per_group)
        ]
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        project_root = Path(temporary_directory)
        processed_dir = project_root / "processed"
        processed_dir.mkdir()
        np.save(
            processed_dir / "X.npy",
            random_generator.normal(size=(sample_count, 64, 157)).astype(np.float32),
        )
        np.save(
            processed_dir / "auxiliary_features.npy",
            random_generator.normal(size=(sample_count, AUXILIARY_FEATURE_DIM)).astype(np.float32),
        )
        np.save(processed_dir / "y.npy", targets)
        np.save(processed_dir / "groups.npy", groups)
        np.save(processed_dir / "labels.npy", np.asarray(["a", "b", "c", "d"]))
        (processed_dir / "preprocess_config.json").write_text(
            json.dumps({"num_samples": sample_count, "num_source_files": len(np.unique(groups))}),
            encoding="utf-8",
        )

        config = TrainingConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            val_split=0.5,
            seed=2026,
            num_workers=0,
            windows_per_source=2,
        )
        loaded = _load_data(project_root, config)
        train_loader, val_loader = loaded[:2]
        auxiliary_dim = loaded[4]
        train_indices, val_indices = loaded[-2:]
        batch = next(iter(train_loader))
        validation_batch = next(iter(val_loader))
        assert len(batch) == 3 and len(validation_batch) == 3
        assert batch[0].shape[1:] == (1, 64, 157)
        assert batch[1].shape[1] == auxiliary_dim == 3
        assert not set(groups[train_indices]) & set(groups[val_indices])


def validate_processed_dataset() -> None:
    processed_dir = PROJECT_ROOT / "processed"
    features = np.load(processed_dir / "X.npy", mmap_mode="r")
    auxiliary_features = np.load(
        processed_dir / "auxiliary_features.npy",
        mmap_mode="r",
    )
    targets = np.load(processed_dir / "y.npy")
    groups = np.load(processed_dir / "groups.npy").astype(str)
    metadata = json.loads(
        (processed_dir / "preprocess_config.json").read_text(encoding="utf-8")
    )
    assert features.shape == (4416, 64, 157)
    assert auxiliary_features.shape == (4416, AUXILIARY_FEATURE_DIM)
    assert len(targets) == len(groups) == len(features)
    assert np.isfinite(auxiliary_features).all()
    assert metadata["num_samples"] == len(features)
    assert metadata["num_source_files"] == len(np.unique(groups)) == 63
    assert metadata["feature_extraction"]["auxiliary_dim"] == AUXILIARY_FEATURE_DIM

    config = TrainingConfig(
        epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        val_split=0.2,
        seed=2026,
        num_workers=0,
        windows_per_source=128,
    )
    loaded = _load_data(PROJECT_ROOT, config)
    train_loader, val_loader = loaded[:2]
    auxiliary_dim = loaded[4]
    selector_metadata = loaded[6]
    train_indices, val_indices = loaded[-2:]
    train_batch = next(iter(train_loader))
    validation_batch = next(iter(val_loader))
    assert auxiliary_dim == 3
    assert selector_metadata["raw_dim"] == AUXILIARY_FEATURE_DIM
    assert selector_metadata["selected_dim"] == 3
    assert train_batch[1].shape[1] == validation_batch[1].shape[1] == 3
    assert not set(groups[train_indices]) & set(groups[val_indices])


def validate_models() -> None:
    torch.manual_seed(31)
    inputs = torch.randn(2, 1, 64, 157)
    auxiliary = torch.randn(2, 3)
    labels = torch.tensor([0, 1])
    models = (
        PVSCNet(4, (64, 157), auxiliary_dim=3),
        VesselCNN(4, (64, 157), auxiliary_dim=3),
        DVSCNet(4, auxiliary_dim=3),
    )
    for model in models:
        outputs = model(inputs, auxiliary)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = torch.nn.functional.cross_entropy(logits, labels)
        regularization_loss = getattr(model, "regularization_loss", None)
        if regularization_loss is not None:
            loss = loss + regularization_loss(outputs)
        loss.backward()
        auxiliary_gradients = [
            parameter.grad
            for parameter in model.auxiliary_encoder.parameters()
            if parameter.requires_grad
        ]
        assert logits.shape == (2, 4)
        assert torch.isfinite(logits).all()
        assert all(gradient is not None for gradient in auxiliary_gradients)
        assert sum(gradient.abs().sum().item() for gradient in auxiliary_gradients) > 0


if __name__ == "__main__":
    validate_real_audio_features()
    validate_selector()
    validate_training_loader()
    validate_processed_dataset()
    validate_models()
    print("Feature extraction, selection, loading, and model fusion validation OK")