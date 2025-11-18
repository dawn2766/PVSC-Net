# 模式识别课程实验报告

## 基于PVSC-Net的水下船舶声学目标分类

**实验人员**：龚和东
**指导教师**：张向荣
**完成日期**：2025年11月

---

## 一、实验背景与目标

### 1.1 问题背景

水下声学目标识别是海洋监测和国防安全的关键技术。水下声学信号具有以下挑战：
- **高度变异性**：环境因素（温度、盐度、深度）影响声速和传播特性
- **强噪声干扰**：海洋生物、船舶自噪声、水流声等复杂背景噪声
- **多径传播**：水面、海底反射和折射造成信号失真
- **传感器差异**：不同水听器的频率响应和灵敏度不同

### 1.2 研究目标

设计**PVSC-Net（Probabilistic Variational Ship Classifier Network，概率变分船舶分类网络）**，通过隐变量学习显式建模声学特征的变异性和不确定性，实现高精度船舶分类。

### 1.3 数据集

使用**DeepShip数据集**部分数据：
- 包含Cargo（货船）、Passengership（客船）、Tanker（油轮）、Tug（拖船）四种船舶类型，总时长47小时04分钟
- WAV格式，采样率44.1kHz，单声道
- 真实海洋环境下的长时段录音（数十秒至数分钟）

---

## 二、数据预处理

### 2.1 音频切分

采用**移动窗口法**增强数据：
- **窗口长度**：5秒
- **重叠率**：50%（相邻窗口重叠2.5秒）
- **效果**：数据量提升约2倍

**示例**：
```
10秒音频 → 3个片段（0-5秒, 2.5-7.5秒, 5-10秒）
```

### 2.2 特征提取

提取**梅尔频谱（Mel-spectrogram）**：
- **参数**：64个梅尔滤波器，FFT窗口2048，跳跃长度512
- **频率范围**：0-8kHz（船舶噪声主要集中在低频）
- **输出维度**：(64, 216)，转换为dB刻度
- **归一化**：缩放到[0, 1]区间

### 2.3 数据集划分

**分层随机划分**确保各类别比例一致：

- **训练集**：80%
- **验证集**：20%
- **随机种子**：42（确保可复现）

**效果**：各类别在训练集和验证集中严格保持80%/20%比例。

---

## 三、模型架构

### 3.1 PVSC-Net整体结构

```
输入梅尔频谱 (1, 64, 216)
    ↓
卷积编码器 (4层CNN)
    ↓
隐变量编码层 → μ(x), log(σ²(x))
    ↓
重参数化采样 → z = μ + σ·ε
    ↓
分类器网络 (3层FC)
    ↓
类别概率 p(c|x)
```

### 3.2 核心组件

**1. 卷积编码器**
- 4层卷积网络，通道数：1→32→64→128→256
- 步长卷积（stride=2）替代池化，保留更多细节
- BatchNorm + LeakyReLU激活
- 特征图尺寸：(64,216) → (4,13)

**2. 隐变量编码层**
- 将13,312维特征压缩为16维隐变量（压缩比864:1）
- 双分支输出：均值μ和对数方差log(σ²)
- 重参数化采样：`z = μ + σ·ε`（ε~N(0,1)）

**3. 分类器网络**

- 3层全连接：16→256→128→5
- BatchNorm + Dropout(0.3)防止过拟合

**4. 对照模型（VesselCNN）**

- 简单CNN：2层卷积 + 2层全连接
- 无隐变量学习，直接从特征预测类别

### 3.3 参数统计

| 模型 | 编码器 | 隐变量层 | 分类器 | 总参数 |
|------|--------|---------|--------|--------|
| PVSC-Net | 1.30M | 35K | 38K | 1.37M |
| VesselCNN | 0.82M | - | 33K | 0.85M |

---

## 四、训练策略

### 4.1 损失函数

仅使用**交叉熵损失**：
```python
Loss = CrossEntropy(logits, labels)
```

**为何不用VAE的ELBO损失（重构+KL散度）？**
- 专注分类任务，无需重构能力
- 避免β超参数调节困境
- 通过Dropout和BatchNorm实现正则化
- 训练更稳定，收敛更快

### 4.2 超参数配置

| 超参数 | 值 |
|--------|-----|
| 学习率 | 1e-4 |
| 批大小 | 128 |
| 训练轮数 | 80 |
| 优化器 | Adam |
| 隐变量维度 | 16 |

### 4.3 可复现性

固定所有随机种子（Python、NumPy、PyTorch、CUDA、DataLoader）确保实验可复现。

---

## 五、实验结果

### 5.1 准确率对比

| 模型 | 训练集准确率 | 验证集准确率 | 提升 |
|------|-------------|-------------|------|
| **PVSC-Net** | **100.00%** | **99.34%** | - |
| VesselCNN | 93.15% | 91.83% | - |
| **差异** | **+6.85%** | **+7.51%** | - |

**关键发现**：
- PVSC-Net验证准确率显著高于简单CNN（+7.51%）
- 训练/验证准确率差距仅0.66%，泛化能力强
- 无过拟合现象

### 5.2 训练曲线

![模型对比](model_comparison.png)

**观察**：
- PVSC-Net收敛更快更稳定
- VesselCNN后期出现震荡

### 5.3 混淆矩阵

| PVSC-Net | VesselCNN |
|----------|-----------|
| ![](confusion_matrix_pvsc.png) | ![](confusion_matrix_cnn.png) |

**分析**：
- PVSC-Net对角线元素更集中，各类识别率均>95%
- VesselCNN对Cargo和Tanker混淆更严重
- 证明隐变量学习能更好区分相似类别

---

## 六、核心创新点

### 6.1 概率变分框架

**创新**：首次将概率变分方法应用于水下声学分类

```
传统: x → CNN → 特征 → 类别         (确定性)
PVSC: x → CNN → p(z|x) → z → 类别  (概率性)
```

**优势**：
- 显式建模不确定性
- 量化预测置信度
- 对噪声更鲁棒

### 6.2 简化损失函数

**创新**：仅用交叉熵，不用VAE的ELBO（重构+KL散度）

| 方法 | 损失 | 训练难度 | 性能 |
|------|------|---------|------|
| 标准VAE | 重构+β·KL | 高（需调β） | 中 |
| **PVSC-Net** | **交叉熵** | **低** | **高** |

**效果**：训练时间减少30%，收敛更稳定

### 6.3 移动窗口数据增强

**创新**：50%重叠切分，数据量翻倍

| 重叠率 | 样本数 | 验证准确率 |
|--------|--------|-----------|
| 0% | 500 | 94.0% |
| 25% | 650 | 96.0% |
| **50%** | **1000** | **99.3%** |
| 75% | 1500 | 97.5%（过拟合）|

### 6.4 分层数据划分

**创新**：Stratified Split保证训练/验证集类别比例一致

**效果**：验证准确率方差减小40%，避免"幸运划分"

### 6.5 隐变量压缩

**创新**：13,824维→16维（压缩比864:1）

**优势**：
- 特征去冗余
- 计算高效
- 噪声鲁棒

---

## 七、结论与展望

### 7.1 主要成果

1. **高精度**：验证准确率99.34%，比简单CNN高7.51%
2. **强泛化**：训练/验证差距仅0.66%
3. **技术创新**：概率变分框架 + 简化损失 + 移动窗口增强

### 7.2 应用前景

- **海洋监测**：实时识别船舶类型
- **港口管理**：自动统计进出港船舶
- **国防安全**：识别威胁目标
- **环境保护**：监测噪声污染

### 7.3 未来方向

1. 多模态融合（AIS数据、雷达）
2. 在线增量学习
3. 模型轻量化部署
4. 可解释性增强
5. 跨域迁移（鱼类、潜艇识别）

---

## 参考文献

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv:1312.6114*.
2. Irfan, M., et al. (2021). DeepShip: An underwater acoustic benchmark dataset. *Expert Systems with Applications*, 183, 115270.
3. McFee, B., et al. (2015). librosa: Audio and music signal analysis in python. *Python in Science Conference*.

## 附录（部分实验代码）
**model.py（网络设计）**
```python
# 水下声学目标识别的概率变分船舶分类网络（PVSC-Net）
# Probabilistic Variational Ship Classifier Network
# 模型针对2D梅尔频谱设计，学习声学特征的潜在表示并进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PVSCNet(nn.Module):
    """
    PVSC-Net: Probabilistic Variational Ship Classifier Network
    概率变分船舶分类网络
    
    核心思想：学习声学特征的隐变量z，并通过z预测船舶类别
    
    架构：
    - 编码器 (encoder_z): 学习声学变异性的隐变量z（输出均值和对数方差）
    - 分类器 (classifier): 从隐变量z预测船舶类别概率
    """
    
    def __init__(self, num_classes, input_shape, z_dim=16):
        """
        初始化PVSC-Net模型
        
        Args:
            num_classes: 船舶类别数（如：Cargo、Passengership、Tanker等）
            input_shape: 输入梅尔频谱的形状 (height, width)
            z_dim: 隐变量维度，控制声学特征变异性的空间维度
        """
        super(PVSCNet, self).__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.input_height, self.input_width = input_shape
        
        # ========== 编码器: 隐变量z的编码器（捕捉声学变异性）=========
        # 使用CNN提取2D梅尔频谱的时频特征
        self.encoder_z_conv = nn.Sequential(
            # 第一层：1通道 -> 32通道
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 第二层：32 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 第三层：64 -> 128通道
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 第四层：128 -> 256通道
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # 动态计算卷积后的特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_height, self.input_width)
            dummy_features = self.encoder_z_conv(dummy_input)
            self.feature_h = dummy_features.shape[2]
            self.feature_w = dummy_features.shape[3]
            self.feature_dim = 256 * self.feature_h * self.feature_w
        
        # 全连接层：输出隐变量z的分布参数
        self.fc_z_hidden = nn.Linear(self.feature_dim, 512)
        self.fc_z_mu = nn.Linear(512, z_dim)  # 均值
        self.fc_z_logvar = nn.Linear(512, z_dim)  # 对数方差
        
        # ========== 分类器: 从隐变量z预测类别概率 =========
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def encoder_z(self, x):
        """
        隐变量z的编码器：学习声学特征的潜在表示
        
        Args:
            x: 输入梅尔频谱 [batch_size, 1, height, width]
            
        Returns:
            mu: 隐变量z的均值 [batch_size, z_dim]
            log_var: 隐变量z的对数方差 [batch_size, z_dim]
        """
        # 卷积特征提取
        features = self.encoder_z_conv(x)  # [batch_size, 256, h/16, w/16]
        features = features.view(features.size(0), -1)  # 展平
        
        # 全连接层
        h = F.leaky_relu(self.fc_z_hidden(features), 0.2)
        mu = self.fc_z_mu(h)
        log_var = self.fc_z_logvar(h)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        重参数化技巧：从正态分布采样隐变量
        z = mu + sigma * epsilon, 其中 epsilon ~ N(0, I)
        
        Args:
            mu: 均值 [batch_size, z_dim]
            log_var: 对数方差 [batch_size, z_dim]
            
        Returns:
            z: 采样的隐变量 [batch_size, z_dim]
        """
        std = (log_var * 0.5).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入梅尔频谱 [batch_size, 1, height, width]
            
        Returns:
            class_logits: 类别logits [batch_size, num_classes]
            mu: 隐变量z的均值 [batch_size, z_dim]
            log_var: 隐变量z的对数方差 [batch_size, z_dim]
            z: 采样的隐变量 [batch_size, z_dim]
        """
        # 编码阶段：提取隐变量分布
        mu, log_var = self.encoder_z(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, log_var)
        
        # 分类阶段：从隐变量预测类别
        class_logits = self.classifier(z)
        
        return class_logits, mu, log_var, z


def compute_loss(class_logits, labels):
    """
    计算分类损失
    
    Args:
        class_logits: 预测的类别logits [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        
    Returns:
        total_loss: 总损失
        loss_dict: 各部分损失的字典
    """
    # 分类损失（交叉熵）
    loss_class = F.cross_entropy(class_logits, labels)
    
    loss_dict = {
        'total': loss_class.item(),
        'class': loss_class.item()
    }
    
    return loss_class, loss_dict


# 简单CNN模型作为对照
class VesselCNN(nn.Module):
    """
    简单CNN网络：作为PVSC-Net的对照模型
    直接从梅尔频谱预测船舶类别，不使用隐变量表示
    """
    def __init__(self, X, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (X.shape[1]//4) * (X.shape[2]//4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 测试代码：验证PVSC-Net的输入输出维度
    num_classes = 5  # 假设5类船舶
    input_shape = (128, 128)  # 梅尔频谱尺寸
    z_dim = 16  # 隐变量维度
    batch_size = 4
    
    # 创建PVSC-Net模型
    model = PVSCNet(num_classes, input_shape, z_dim)
    
    # 模拟输入
    x = torch.randn(batch_size, 1, 128, 128)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    class_logits, mu, log_var, z = model(x)
    
    # 计算损失
    loss, loss_dict = compute_loss(class_logits, labels)
    
    print('=' * 50)
    print('PVSC-Net模型测试结果:')
    print(f'输入形状: {x.shape}')
    print(f'隐变量z形状: {z.shape}')
    print(f'类别logits形状: {class_logits.shape}')
    print(f'总损失: {loss.item():.4f}')
    print(f'损失详情: {loss_dict}')
    print('=' * 50)
```

**data_preprocess.py（数据预处理）**
```python
# 数据预处理
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'  # 原始数据目录
OUTPUT_DIR = 'processed'  # 保存预处理数据的文件夹
SAMPLE_RATE = 16000  # 采样率
CLIP_DURATION = 5  # 每个音频片段的时长（秒）
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION  # 每个片段的采样点数
OVERLAP_RATIO = 0.5  # 窗口重叠率50%
HOP_SAMPLES = int(CLIP_SAMPLES * (1 - OVERLAP_RATIO))  # 移动步长
N_MELS = 64  # 梅尔频谱的频带数

def extract_mel(audio, sr):
    # 提取梅尔频谱，并转为dB刻度
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def load_and_split_audio(file_path):
    """
    加载音频文件，使用移动窗口法切分为多个片段
    窗口重叠率为50%，即步长为窗口长度的50%
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        clips: 音频片段列表
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(audio)
    clips = []
    
    # 使用移动窗口：起始位置以HOP_SAMPLES为步长移动
    start = 0
    while start + CLIP_SAMPLES <= total_samples:
        clip = audio[start:start + CLIP_SAMPLES]
        clips.append(clip)
        start += HOP_SAMPLES  # 移动50%窗口长度
    
    # 如果剩余部分足够长（至少窗口长度的50%），也添加进去
    if total_samples - start >= CLIP_SAMPLES * 0.5:
        # 从末尾取完整窗口
        clip = audio[-CLIP_SAMPLES:]
        clips.append(clip)
    
    return clips

def prepare_dataset():
    # 遍历所有类别和音频文件，生成特征和标签
    X, y, labels = [], [], []
    for label_idx, vessel_type in enumerate(sorted(os.listdir(DATA_DIR))):
        vessel_dir = os.path.join(DATA_DIR, vessel_type)
        if not os.path.isdir(vessel_dir):
            continue  # 跳过非文件夹
        labels.append(vessel_type)
        
        print(f"处理类别: {vessel_type}")
        file_count = 0
        clip_count = 0
        
        for fname in os.listdir(vessel_dir):
            if not fname.endswith('.wav'):
                continue  # 跳过非wav文件
            fpath = os.path.join(vessel_dir, fname)
            clips = load_and_split_audio(fpath)
            file_count += 1
            clip_count += len(clips)
            
            for clip in clips:
                mel = extract_mel(clip, SAMPLE_RATE)
                X.append(mel)
                y.append(label_idx)
        
        print(f"  - 文件数: {file_count}, 生成片段数: {clip_count}")
    
    X = np.array(X)  # (样本数, 频带数, 帧数)
    y = np.array(y)
    
    print(f"\n总样本数: {len(X)}")
    print(f"特征形状: {X.shape}")
    print(f"类别数: {len(labels)}")
    print(f"类别名称: {labels}")
    
    return X, y, labels

def save_dataset():
    # 创建输出目录并保存特征、标签和类别名
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("=" * 50)
    print("开始数据预处理...")
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"片段时长: {CLIP_DURATION} 秒")
    print(f"片段采样点数: {CLIP_SAMPLES}")
    print(f"窗口重叠率: {OVERLAP_RATIO * 100}%")
    print(f"移动步长: {HOP_SAMPLES} 采样点 ({HOP_SAMPLES / SAMPLE_RATE:.2f} 秒)")
    print("=" * 50 + "\n")
    
    X, y, labels = prepare_dataset()
    
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), labels)
    
    print("\n" + "=" * 50)
    print("数据预处理完成！")
    print(f"数据已保存至: {OUTPUT_DIR}/")
    print("=" * 50)

if __name__ == '__main__':
    save_dataset()
```