import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from model import AcousticVAE, compute_loss  # 导入模型和损失函数
import random

# 全局超参数配置
LR = 1e-4
NUM_EPOCHS = 80
BATCH_SIZE = 128
VAL_SPLIT = 0.2
SEED = 42
Z_DIM = 16  # 隐变量维度

# ========== 设置所有随机种子，确保可复现 ==========
def set_seed(seed):
    """固定所有随机源，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    # 确保卷积算法的确定性（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# 从processed目录加载预处理后的特征、标签和类别名
X = np.load('processed/X.npy')
y = np.load('processed/y.npy')
labels = np.load('processed/labels.npy', allow_pickle=True)

# 归一化特征到[0,1]区间
X = (X - X.min()) / (X.max() - X.min())
X = X[..., np.newaxis]  # 增加通道维度，形状变为(N, H, W, 1)

class VesselDataset(Dataset):
    def __init__(self, X, y):
        # 将numpy数组转为torch张量
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        # 返回样本数量
        return len(self.X)
    def __getitem__(self, idx):
        # 返回单个样本，调整通道顺序为(C, H, W)
        x = self.X[idx].permute(2, 0, 1)
        return x, self.y[idx]  # (C,H,W), label

# 使用分层划分确保训练集和验证集中各类别比例一致
# 先获取所有索引，然后按类别分层划分
indices = np.arange(len(y))
train_indices, val_indices = train_test_split(
    indices, 
    test_size=VAL_SPLIT, 
    stratify=y,  # 按照y的类别分布进行分层
    random_state=SEED
)

# 打印各类别在训练集和验证集中的分布
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
    print(f"  总数: {total_count} | 训练集: {train_count} ({train_ratio:.1f}%) | "
          f"验证集: {val_count} ({val_ratio:.1f}%)")

print("=" * 60 + "\n")

# 创建完整数据集
dataset = VesselDataset(X, y)

# 使用Subset创建训练集和验证集
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# DataLoader也需要设置worker的随机种子
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    worker_init_fn=seed_worker, generator=g
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    worker_init_fn=seed_worker, generator=g
)

# 自动检测GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用AcousticVAE模型，传入输入形状和类别数
input_shape = (X.shape[1], X.shape[2])  # (height, width)
model = AcousticVAE(num_classes=len(labels), input_shape=input_shape, z_dim=Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练与验证
num_epochs = NUM_EPOCHS
train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        class_logits, mu, log_var, z = model(xb)
        
        # 计算分类损失
        loss, loss_dict = compute_loss(class_logits, yb)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss_dict['total'] * xb.size(0)
        
        # 使用类别logits计算准确率
        _, preds = class_logits.max(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    
    train_loss_hist.append(running_loss / total)
    train_acc_hist.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # 前向传播
            class_logits, mu, log_var, z = model(xb)
            
            # 计算损失
            loss, loss_dict = compute_loss(class_logits, yb)
            
            val_loss += loss_dict['total'] * xb.size(0)
            
            # 使用类别logits计算准确率
            _, preds = class_logits.max(1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    
    val_loss_hist.append(val_loss / val_total)
    val_acc_hist.append(val_correct / val_total)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss_hist[-1]:.4f} - "
          f"Train Acc: {train_acc_hist[-1]:.4f} - Val Acc: {val_acc_hist[-1]:.4f}")

# 保存模型权重
torch.save(model.state_dict(), 'vessel_vae.pt')

# 可视化训练过程：损失和准确率曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_hist, label='train_loss')
plt.plot(val_loss_hist, label='val_loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1,2,2)
plt.plot(train_acc_hist, label='train_acc')
plt.plot(val_acc_hist, label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('train_val_curve.png')

# 计算并可视化混淆矩阵
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        # 使用模型的类别logits输出
        class_logits, _, _, _ = model(xb)
        preds = class_logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print(f"\n最终验证准确率: {val_acc_hist[-1]:.4f}")
print(f"模型已保存至: vessel_vae.pt")
