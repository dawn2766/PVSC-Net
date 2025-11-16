import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import AcousticVAE, compute_loss  # 导入VAE模型和损失函数
import random

# 全局超参数配置
LR = 5e-4
NUM_EPOCHS = 120
BATCH_SIZE = 256
VAL_SPLIT = 0.2
SEED = 2
Z_DIM = 16  # 隐变量维度
LAMBDA_KL = 1.0  # KL散度损失权重
LAMBDA_CLASS = 10.0  # 分类损失权重

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

# 划分训练集和验证集，比例8:2，保证可复现
dataset = VesselDataset(X, y)
val_size = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
# 使用固定种子的生成器
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(SEED)
)
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
# 记录各部分损失
train_recon_hist, train_kl_hist, train_class_hist = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    epoch_recon, epoch_kl, epoch_class = 0, 0, 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        
        # VAE前向传播
        x_recon, mu, log_var, class_probs, z = model(xb)
        
        # 计算VAE损失
        loss, loss_dict = compute_loss(
            xb, x_recon, mu, log_var, class_probs, yb,
            lambda_kl=LAMBDA_KL, lambda_class=LAMBDA_CLASS
        )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss_dict['total'] * xb.size(0)
        epoch_recon += loss_dict['recon'] * xb.size(0)
        epoch_kl += loss_dict['kl'] * xb.size(0)
        epoch_class += loss_dict['class'] * xb.size(0)
        
        # 使用类别概率计算准确率
        _, preds = class_probs.max(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    
    train_loss_hist.append(running_loss / total)
    train_acc_hist.append(correct / total)
    train_recon_hist.append(epoch_recon / total)
    train_kl_hist.append(epoch_kl / total)
    train_class_hist.append(epoch_class / total)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # VAE前向传播
            x_recon, mu, log_var, class_probs, z = model(xb)
            
            # 计算损失
            loss, loss_dict = compute_loss(
                xb, x_recon, mu, log_var, class_probs, yb,
                lambda_kl=LAMBDA_KL, lambda_class=LAMBDA_CLASS
            )
            
            val_loss += loss_dict['total'] * xb.size(0)
            
            # 使用类别概率计算准确率
            _, preds = class_probs.max(1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    
    val_loss_hist.append(val_loss / val_total)
    val_acc_hist.append(val_correct / val_total)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss_hist[-1]:.4f} (Recon: {train_recon_hist[-1]:.4f}, "
          f"KL: {train_kl_hist[-1]:.4f}, Class: {train_class_hist[-1]:.4f}) - "
          f"Train Acc: {train_acc_hist[-1]:.4f} - Val Acc: {val_acc_hist[-1]:.4f}")

# 保存模型权重
torch.save(model.state_dict(), 'vessel_vae.pt')

# 可视化训练过程：损失和准确率曲线
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.plot(train_loss_hist, label='train_loss')
plt.plot(val_loss_hist, label='val_loss')
plt.legend()
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1,3,2)
plt.plot(train_recon_hist, label='recon_loss')
plt.plot(train_kl_hist, label='kl_loss')
plt.plot(train_class_hist, label='class_loss')
plt.legend()
plt.title('Training Loss Components')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1,3,3)
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
        # 使用VAE的类别概率输出
        _, _, _, class_probs, _ = model(xb)
        preds = class_probs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print(f"\n最终验证准确率: {val_acc_hist[-1]:.4f}")
print(f"模型已保存至: vessel_vae.pt")
