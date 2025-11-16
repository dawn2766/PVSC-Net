# 水下声学目标识别的概率生成模型（基于VAE架构）
# 模型针对2D梅尔频谱设计，学习声学特征的潜在表示和目标类别分布

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AcousticVAE(nn.Module):
    """
    水下声学目标识别的变分自编码器
    核心思想：同时学习声学特征的潜在表示（z）和目标类别概率（c）
    
    架构：
    - 编码器1 (encoder_z): 学习声学变异性的隐变量z
    - 编码器2 (encoder_c): 学习目标类别的概率分布c
    - 解码器 (decoder): 从隐变量z重构梅尔频谱
    """
    
    def __init__(self, num_classes, input_shape, z_dim=16):
        """
        初始化声学VAE模型
        
        Args:
            num_classes: 目标类别数（如：潜艇、鱼雷、水面舰艇等）
            input_shape: 输入梅尔频谱的形状 (height, width)
            z_dim: 隐变量维度，控制声学特征变异性的空间维度
        """
        super(AcousticVAE, self).__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.input_height, self.input_width = input_shape
        
        # ========== 编码器1: 隐变量z的编码器（捕捉声学变异性）=========
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
        
        # ========== 编码器2: 类别c的编码器（学习目标类别）=========
        # 共享前期卷积特征提取
        self.encoder_c_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # 全连接层：输出类别概率
        self.fc_c1 = nn.Linear(self.feature_dim, 512)
        self.fc_c2 = nn.Linear(512, 256)
        self.fc_c3 = nn.Linear(256, num_classes)  # 输出类别logits
        
        # ========== 解码器: 从隐变量z重构梅尔频谱 =========
        self.fc_decoder = nn.Linear(z_dim, 512)
        self.fc_decoder2 = nn.Linear(512, self.feature_dim)
        
        # 反卷积网络：从特征图重构梅尔频谱
        self.decoder_conv = nn.Sequential(
            # 256 -> 128通道
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 128 -> 64通道
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 64 -> 32通道
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32 -> 1通道（输出重构的梅尔频谱）
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 归一化到[0,1]范围
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
    
    def encoder_c(self, x):
        """
        类别c的编码器：学习目标类别的概率分布
        
        Args:
            x: 输入梅尔频谱 [batch_size, 1, height, width]
            
        Returns:
            class_probs: 类别概率分布 [batch_size, num_classes]
        """
        # 卷积特征提取
        features = self.encoder_c_conv(x)
        features = features.view(features.size(0), -1)
        
        # 全连接层
        h = F.leaky_relu(self.fc_c1(features), 0.2)
        h = F.leaky_relu(self.fc_c2(h), 0.2)
        logits = self.fc_c3(h)
        
        # Softmax得到类别概率（满足非负性和和为1）
        class_probs = F.softmax(logits, dim=1)
        
        return class_probs
    
    def reparameterize(self, mu, log_var):
        """
        重参数化技巧：使VAE可微分
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
    
    def decoder(self, z):
        """
        解码器：从隐变量z重构梅尔频谱
        
        Args:
            z: 隐变量 [batch_size, z_dim]
            
        Returns:
            x_recon: 重构的梅尔频谱 [batch_size, 1, height, width]
        """
        # 全连接层扩展维度
        h = F.leaky_relu(self.fc_decoder(z), 0.2)
        h = F.leaky_relu(self.fc_decoder2(h), 0.2)
        
        # 重塑为特征图
        h = h.view(-1, 256, self.feature_h, self.feature_w)
        
        # 反卷积重构
        x_recon = self.decoder_conv(h)
        
        # 自适应调整尺寸以匹配原始输入
        if x_recon.shape[2] != self.input_height or x_recon.shape[3] != self.input_width:
            x_recon = F.interpolate(
                x_recon, 
                size=(self.input_height, self.input_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return x_recon
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入梅尔频谱 [batch_size, 1, height, width]
            
        Returns:
            x_recon: 重构的梅尔频谱 [batch_size, 1, height, width]
            mu: 隐变量z的均值 [batch_size, z_dim]
            log_var: 隐变量z的对数方差 [batch_size, z_dim]
            class_probs: 类别概率分布 [batch_size, num_classes]
            z: 采样的隐变量 [batch_size, z_dim]
        """
        # 编码阶段
        mu, log_var = self.encoder_z(x)  # 提取隐变量分布
        class_probs = self.encoder_c(x)  # 提取类别分布
        
        # 重参数化采样
        z = self.reparameterize(mu, log_var)
        
        # 解码阶段
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var, class_probs, z


def compute_loss(x, x_recon, mu, log_var, class_probs, labels, 
                 lambda_kl=1.0, lambda_class=10.0):
    """
    计算VAE的总损失
    
    Args:
        x: 原始梅尔频谱 [batch_size, 1, height, width]
        x_recon: 重构的梅尔频谱 [batch_size, 1, height, width]
        mu: 隐变量均值 [batch_size, z_dim]
        log_var: 隐变量对数方差 [batch_size, z_dim]
        class_probs: 预测的类别概率 [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        lambda_kl: KL散度损失权重
        lambda_class: 分类损失权重
        
    Returns:
        total_loss: 总损失
        loss_dict: 各部分损失的字典
    """
    batch_size = x.size(0)
    
    # 1. 重构损失（MSE）
    loss_recon = F.mse_loss(x_recon, x, reduction='sum') / batch_size
    
    # 2. KL散度损失（正则化隐变量分布）
    # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    # KL平衡：防止后验崩塌
    kl_div = torch.max(kl_div, torch.tensor(0.1).to(x.device))
    
    # 3. 分类损失（交叉熵）
    loss_class = F.cross_entropy(class_probs, labels)
    
    # 总损失
    total_loss = loss_recon + lambda_kl * kl_div + lambda_class * loss_class
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': loss_recon.item(),
        'kl': kl_div.item(),
        'class': loss_class.item()
    }
    
    return total_loss, loss_dict


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
    x_recon, mu, log_var, class_probs, z = model(x)
    
    # 计算损失
    loss, loss_dict = compute_loss(x, x_recon, mu, log_var, class_probs, labels)
    
    print('=' * 50)
    print('模型测试结果:')
    print(f'输入形状: {x.shape}')
    print(f'重构形状: {x_recon.shape}')
    print(f'隐变量z形状: {z.shape}')
    print(f'类别概率形状: {class_probs.shape}')
    print(f'总损失: {loss.item():.4f}')
    print(f'损失详情: {loss_dict}')
    print('=' * 50)