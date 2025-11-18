import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from model import PVSCNet, VesselCNN, compute_loss  # 导入PVSC-Net模型和损失函数
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

# ========== 初始化两个模型 ==========
# 使用PVSC-Net模型（概率变分船舶分类网络）
input_shape = (X.shape[1], X.shape[2])  # (height, width)
model_pvsc = PVSCNet(num_classes=len(labels), input_shape=input_shape, z_dim=Z_DIM).to(device)
optimizer_pvsc = optim.Adam(model_pvsc.parameters(), lr=LR)

# 使用VesselCNN模型（简单CNN对照）
model_cnn = VesselCNN(X, num_classes=len(labels)).to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=LR)

print("\n" + "=" * 60)
print("训练计划:")
print(f"  第一阶段: 训练 PVSC-Net (概率变分船舶分类网络)")
print(f"  第二阶段: 训练 VesselCNN (简单CNN对照)")
print(f"  每个模型训练轮数: {NUM_EPOCHS}")
print("=" * 60 + "\n")

# 训练与验证
num_epochs = NUM_EPOCHS

# 存储两个模型的训练历史
pvsc_train_loss_hist, pvsc_val_loss_hist = [], []
pvsc_train_acc_hist, pvsc_val_acc_hist = [], []

cnn_train_loss_hist, cnn_val_loss_hist = [], []
cnn_train_acc_hist, cnn_val_acc_hist = [], []

# ========== 第一阶段: 训练 PVSC-Net 模型 ==========
print("=" * 60)
print("第一阶段: 开始训练 PVSC-Net 模型")
print("=" * 60 + "\n")

for epoch in range(num_epochs):
    # 训练 PVSC-Net
    model_pvsc.train()
    pvsc_running_loss, pvsc_correct, pvsc_total = 0, 0, 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_pvsc.zero_grad()
        
        # 前向传播
        class_logits, mu, log_var, z = model_pvsc(xb)
        
        # 计算分类损失
        loss, loss_dict = compute_loss(class_logits, yb)
        
        loss.backward()
        optimizer_pvsc.step()
        
        pvsc_running_loss += loss_dict['total'] * xb.size(0)
        
        # 使用类别logits计算准确率
        _, preds = class_logits.max(1)
        pvsc_correct += (preds == yb).sum().item()
        pvsc_total += xb.size(0)
    
    pvsc_train_loss_hist.append(pvsc_running_loss / pvsc_total)
    pvsc_train_acc_hist.append(pvsc_correct / pvsc_total)

    # 验证 PVSC-Net
    model_pvsc.eval()
    pvsc_val_loss, pvsc_val_correct, pvsc_val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # 前向传播
            class_logits, mu, log_var, z = model_pvsc(xb)
            
            # 计算损失
            loss, loss_dict = compute_loss(class_logits, yb)
            
            pvsc_val_loss += loss_dict['total'] * xb.size(0)
            
            # 使用类别logits计算准确率
            _, preds = class_logits.max(1)
            pvsc_val_correct += (preds == yb).sum().item()
            pvsc_val_total += xb.size(0)
    
    pvsc_val_loss_hist.append(pvsc_val_loss / pvsc_val_total)
    pvsc_val_acc_hist.append(pvsc_val_correct / pvsc_val_total)
    
    # 打印训练进度
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {pvsc_train_loss_hist[-1]:.4f}, "
          f"Train Acc: {pvsc_train_acc_hist[-1]:.4f}, "
          f"Val Acc: {pvsc_val_acc_hist[-1]:.4f}")

# 保存 PVSC-Net 模型
torch.save(model_pvsc.state_dict(), 'vessel_pvsc.pt')
print(f"\nPVSC-Net 模型训练完成，权重已保存至: vessel_pvsc.pt")
print(f"最终验证准确率: {pvsc_val_acc_hist[-1]:.4f}\n")

# ========== 第二阶段: 训练 VesselCNN 模型 ==========
print("=" * 60)
print("第二阶段: 开始训练 VesselCNN 模型（对照）")
print("=" * 60 + "\n")

for epoch in range(num_epochs):
    # 训练 VesselCNN
    model_cnn.train()
    cnn_running_loss, cnn_correct, cnn_total = 0, 0, 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_cnn.zero_grad()
        
        # 前向传播
        logits = model_cnn(xb)
        
        # 计算损失
        loss = torch.nn.functional.cross_entropy(logits, yb)
        
        loss.backward()
        optimizer_cnn.step()
        
        cnn_running_loss += loss.item() * xb.size(0)
        
        # 计算准确率
        _, preds = logits.max(1)
        cnn_correct += (preds == yb).sum().item()
        cnn_total += xb.size(0)
    
    cnn_train_loss_hist.append(cnn_running_loss / cnn_total)
    cnn_train_acc_hist.append(cnn_correct / cnn_total)

    # 验证 VesselCNN
    model_cnn.eval()
    cnn_val_loss, cnn_val_correct, cnn_val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # 前向传播
            logits = model_cnn(xb)
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(logits, yb)
            
            cnn_val_loss += loss.item() * xb.size(0)
            
            # 计算准确率
            _, preds = logits.max(1)
            cnn_val_correct += (preds == yb).sum().item()
            cnn_val_total += xb.size(0)
    
    cnn_val_loss_hist.append(cnn_val_loss / cnn_val_total)
    cnn_val_acc_hist.append(cnn_val_correct / cnn_val_total)
    
    # 打印训练进度
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {cnn_train_loss_hist[-1]:.4f}, "
          f"Train Acc: {cnn_train_acc_hist[-1]:.4f}, "
          f"Val Acc: {cnn_val_acc_hist[-1]:.4f}")

# 保存 VesselCNN 模型
torch.save(model_cnn.state_dict(), 'vessel_cnn.pt')
print(f"\nVesselCNN 模型训练完成，权重已保存至: vessel_cnn.pt")
print(f"最终验证准确率: {cnn_val_acc_hist[-1]:.4f}\n")

# ========== 可视化对比：在同一张图上绘制两种模型的曲线 ==========
plt.figure(figsize=(18, 5))

# 子图1: 训练损失对比
plt.subplot(1, 3, 1)
plt.plot(pvsc_train_loss_hist, label='PVSC-Net Train Loss', linewidth=2)
plt.plot(cnn_train_loss_hist, label='VesselCNN Train Loss', linewidth=2, linestyle='--')
plt.legend()
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 子图2: 训练准确率对比
plt.subplot(1, 3, 2)
plt.plot(pvsc_train_acc_hist, label='PVSC-Net Train Acc', linewidth=2)
plt.plot(cnn_train_acc_hist, label='VesselCNN Train Acc', linewidth=2, linestyle='--')
plt.legend()
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

# 子图3: 验证准确率对比
plt.subplot(1, 3, 3)
plt.plot(pvsc_val_acc_hist, label='PVSC-Net Val Acc', linewidth=2)
plt.plot(cnn_val_acc_hist, label='VesselCNN Val Acc', linewidth=2, linestyle='--')
plt.legend()
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("\n模型对比图已保存至: model_comparison.png")

# ========== 计算并可视化 PVSC-Net 的混淆矩阵 ==========
model_pvsc.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        # 使用模型的类别logits输出
        class_logits, _, _, _ = model_pvsc(xb)
        preds = class_logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(10, 8))
disp.plot(xticks_rotation=45)
plt.title('PVSC-Net Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_pvsc.png')

# ========== 计算并可视化 VesselCNN 的混淆矩阵 ==========
model_cnn.eval()
all_preds_cnn, all_labels_cnn = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model_cnn(xb)
        preds = logits.argmax(1).cpu().numpy()
        all_preds_cnn.extend(preds)
        all_labels_cnn.extend(yb.numpy())

cm_cnn = confusion_matrix(all_labels_cnn, all_preds_cnn)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=labels)
plt.figure(figsize=(10, 8))
disp_cnn.plot(xticks_rotation=45)
plt.title('VesselCNN Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png')

# ========== 打印最终结果对比 ==========
print("\n" + "=" * 60)
print("最终结果对比:")
print(f"  PVSC-Net 验证准确率:  {pvsc_val_acc_hist[-1]:.4f}")
print(f"  VesselCNN 验证准确率: {cnn_val_acc_hist[-1]:.4f}")
print(f"  准确率提升: {(pvsc_val_acc_hist[-1] - cnn_val_acc_hist[-1]):.4f}")
print("=" * 60)
print(f"\n模型权重已保存:")
print(f"  vessel_pvsc.pt (PVSC-Net)")
print(f"  vessel_cnn.pt (VesselCNN)")
