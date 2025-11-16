# 水下声学目标识别的概率生成模型（基于编码器-分类器架构）
# 模型针对2D梅尔频谱设计，学习声学特征的潜在表示并进行分类

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AcousticVAE(nn.Module):
    """
    水下声学目标识别的编码器-分类器模型
    核心思想：学习声学特征的隐变量z，并通过z预测目标类别
    
    架构：
    - 编码器 (encoder_z): 学习声学变异性的隐变量z（输出均值和对数方差）
    - 分类器 (classifier): 从隐变量z预测目标类别概率
    """
    
    def __init__(self, num_classes, input_shape, z_dim=16):
        """
        初始化声学分类模型
        
        Args:
            num_classes: 目标类别数（如：潜艇、鱼雷、水面舰艇等）
            input_shape: 输入梅尔频谱的形状 (height, width)
            z_dim: 隐变量维度，控制声学特征变异性的空间维度
        """
        super(AcousticVAE, self).__init__()
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


# 保留原始简单CNN模型以便兼容
class VesselCNN(nn.Module):
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
    # 测试代码：验证模型的输入输出维度
    num_classes = 5  # 假设5类水下目标
    input_shape = (128, 128)  # 梅尔频谱尺寸
    z_dim = 16  # 隐变量维度
    batch_size = 4
    
    # 创建模型
    model = AcousticVAE(num_classes, input_shape, z_dim)
    
    # 模拟输入
    x = torch.randn(batch_size, 1, 128, 128)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    class_logits, mu, log_var, z = model(x)
    
    # 计算损失
    loss, loss_dict = compute_loss(class_logits, labels)
    
    print('=' * 50)
    print('模型测试结果:')
    print(f'输入形状: {x.shape}')
    print(f'隐变量z形状: {z.shape}')
    print(f'类别logits形状: {class_logits.shape}')
    print(f'总损失: {loss.item():.4f}')
    print(f'损失详情: {loss_dict}')
    print('=' * 50)