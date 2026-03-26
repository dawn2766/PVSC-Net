import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from model import PVSCNet, VesselCNN, compute_loss as compute_loss_pvsc
from model2 import DVSCNet, compute_loss as compute_loss_dvsc


# Global hyperparameters
LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 256
VAL_SPLIT = 0.2
SEED = 42
Z_DIM = 16
DVSC_Z_DIM = 128


def set_seed(seed):
    """Fix random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# Load preprocessed features, labels and class names
X = np.load("processed/X.npy")
y = np.load("processed/y.npy")
labels = np.load("processed/labels.npy", allow_pickle=True)

# Normalize features to [0, 1]
X = (X - X.min()) / (X.max() - X.min())
X = X[..., np.newaxis]


class VesselDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(2, 0, 1)
        return x, self.y[idx]


indices = np.arange(len(y))
train_indices, val_indices = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    stratify=y,
    random_state=SEED,
)

print("=" * 60)
print("数据集划分统计 (分层采样):")
print(f"总样本数: {len(y)}")
print(f"训练集样本数: {len(train_indices)}")
print(f"验证集样本数: {len(val_indices)}")
print("-" * 60)

for i, label_name in enumerate(labels):
    total_count = np.sum(y == i)
    train_count = np.sum(y[train_indices] == i)
    val_count = np.sum(y[val_indices] == i)
    train_ratio = train_count / total_count * 100
    val_ratio = val_count / total_count * 100
    print(f"类别 '{label_name}':")
    print(
        f"  总数: {total_count} | 训练集: {train_count} ({train_ratio:.1f}%) | "
        f"验证集: {val_count} ({val_ratio:.1f}%)"
    )

print("=" * 60 + "\n")


dataset = VesselDataset(X, y)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (X.shape[1], X.shape[2])


def forward_and_loss(model_name, model, xb, yb):
    if model_name == "VesselCNN":
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        return logits, loss.item(), loss

    if model_name == "PVSC-Net":
        class_logits, _, _, _ = model(xb)
        loss, loss_dict = compute_loss_pvsc(class_logits, yb)
        return class_logits, loss_dict["total"], loss

    if model_name == "DVSC-Net":
        class_logits, _, _, _ = model(xb)
        loss, loss_dict = compute_loss_dvsc(class_logits, yb)
        return class_logits, loss_dict["total"], loss

    raise ValueError(f"Unknown model name: {model_name}")


def train_one_model(model_name, model, optimizer, num_epochs):
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    print("=" * 60)
    print(f"开始训练: {model_name}")
    print("=" * 60 + "\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits, batch_loss_value, loss = forward_and_loss(model_name, model, xb, yb)
            loss.backward()
            optimizer.step()

            running_loss += batch_loss_value * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss_hist.append(running_loss / total)
        train_acc_hist.append(correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, batch_loss_value, _ = forward_and_loss(model_name, model, xb, yb)

                val_loss += batch_loss_value * xb.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss_hist.append(val_loss / val_total)
        val_acc_hist.append(val_correct / val_total)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss_hist[-1]:.4f}, "
            f"Train Acc: {train_acc_hist[-1]:.4f}, "
            f"Val Acc: {val_acc_hist[-1]:.4f}"
        )

    return {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": train_acc_hist,
        "val_acc": val_acc_hist,
    }


def collect_predictions(model_name, model):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            if model_name == "VesselCNN":
                logits = model(xb)
            else:
                logits, _, _, _ = model(xb)

            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    return all_labels, all_preds


# Initialize three models
model_pvsc = PVSCNet(num_classes=len(labels), input_shape=input_shape, z_dim=Z_DIM).to(device)
optimizer_pvsc = optim.Adam(model_pvsc.parameters(), lr=LR)

model_cnn = VesselCNN(X, num_classes=len(labels)).to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=LR)

model_dvsc = DVSCNet(num_classes=len(labels), in_chans=1, z_dim=DVSC_Z_DIM).to(device)
optimizer_dvsc = optim.AdamW(model_dvsc.parameters(), lr=LR, weight_decay=1e-4)

print("\n" + "=" * 60)
print("训练计划:")
print("  第一阶段: 训练 PVSC-Net")
print("  第二阶段: 训练 VesselCNN")
print("  第三阶段: 训练 DVSC-Net")
print(f"  每个模型训练轮数: {NUM_EPOCHS}")
print("=" * 60 + "\n")

# Train all models
pvsc_hist = train_one_model("PVSC-Net", model_pvsc, optimizer_pvsc, NUM_EPOCHS)
cnn_hist = train_one_model("VesselCNN", model_cnn, optimizer_cnn, NUM_EPOCHS)
dvsc_hist = train_one_model("DVSC-Net", model_dvsc, optimizer_dvsc, NUM_EPOCHS)

# Save model weights
torch.save(model_pvsc.state_dict(), "vessel_pvsc.pt")
torch.save(model_cnn.state_dict(), "vessel_cnn.pt")
torch.save(model_dvsc.state_dict(), "vessel_dvsc.pt")

print("\n模型训练完成，权重已保存:")
print("  vessel_pvsc.pt")
print("  vessel_cnn.pt")
print("  vessel_dvsc.pt\n")

# Plot 3-model comparison
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(pvsc_hist["train_loss"], label="PVSC-Net Train Loss", linewidth=2)
plt.plot(cnn_hist["train_loss"], label="VesselCNN Train Loss", linewidth=2, linestyle="--")
plt.plot(dvsc_hist["train_loss"], label="DVSC-Net Train Loss", linewidth=2, linestyle=":")
plt.legend()
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(pvsc_hist["train_acc"], label="PVSC-Net Train Acc", linewidth=2)
plt.plot(cnn_hist["train_acc"], label="VesselCNN Train Acc", linewidth=2, linestyle="--")
plt.plot(dvsc_hist["train_acc"], label="DVSC-Net Train Acc", linewidth=2, linestyle=":")
plt.legend()
plt.title("Training Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(pvsc_hist["val_acc"], label="PVSC-Net Val Acc", linewidth=2)
plt.plot(cnn_hist["val_acc"], label="VesselCNN Val Acc", linewidth=2, linestyle="--")
plt.plot(dvsc_hist["val_acc"], label="DVSC-Net Val Acc", linewidth=2, linestyle=":")
plt.legend()
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison_3models.png", dpi=300)
print("模型对比图已保存至: model_comparison_3models.png")


# Confusion matrices
for name, model, fig_name in [
    ("PVSC-Net", model_pvsc, "confusion_matrix_pvsc.png"),
    ("VesselCNN", model_cnn, "confusion_matrix_cnn.png"),
    ("DVSC-Net", model_dvsc, "confusion_matrix_dvsc.png"),
]:
    gt, pred = collect_predictions(name, model)
    cm = confusion_matrix(gt, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(10, 8))
    disp.plot(xticks_rotation=45)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_name)
    print(f"{name} 混淆矩阵已保存至: {fig_name}")


# Final comparison summary
pvsc_final = pvsc_hist["val_acc"][-1]
cnn_final = cnn_hist["val_acc"][-1]
dvsc_final = dvsc_hist["val_acc"][-1]

print("\n" + "=" * 60)
print("最终结果对比:")
print(f"  PVSC-Net 验证准确率:          {pvsc_final:.4f}")
print(f"  VesselCNN 验证准确率:        {cnn_final:.4f}")
print(f"  DVSC-Net 验证准确率: {dvsc_final:.4f}")
print("-" * 60)
print(f"  DVSC-Net - PVSC 提升: {dvsc_final - pvsc_final:.4f}")
print(f"  DVSC-Net - CNN  提升: {dvsc_final - cnn_final:.4f}")
print("=" * 60)
