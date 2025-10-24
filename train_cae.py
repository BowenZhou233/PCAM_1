"""
问题4: 卷积自编码器 (CAE)
用于压缩和重建PCam图像
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# ========== 编码器 ==========
class Encoder(nn.Module):
    """编码器: 将图像压缩到潜在空间"""
    def __init__(self):
        super(Encoder, self).__init__()
        
        # 输入: (batch, 3, 28, 28)
        self.encoder = nn.Sequential(
            # 第1层卷积: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 第2层卷积: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 第3层卷积: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        return self.encoder(x)


# ========== 解码器 ==========
class Decoder(nn.Module):
    """解码器: 从潜在空间重建图像"""
    def __init__(self):
        super(Decoder, self).__init__()
        
        # 输入: (batch, 128, 4, 4)
        self.decoder = nn.Sequential(
            # 第1层转置卷积: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 第2层转置卷积: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),  # -> (32, 15, 15)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 第3层转置卷积: 32 -> 3
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (3, 30, 30)
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 额外的卷积层来调整到精确的28x28
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding=0)
        
    def forward(self, x):
        x = self.decoder(x)
        x = self.final_conv(x)  # (3, 30, 30) -> (3, 28, 28)
        return x


# ========== 卷积自编码器 ==========
class ConvAutoencoder(nn.Module):
    """完整的卷积自编码器"""
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """仅编码"""
        return self.encoder(x)
    
    def decode(self, x):
        """仅解码"""
        return self.decoder(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    
    for images, _ in tqdm(dataloader, desc='训练中'):
        images = images.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='验证中'):
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def visualize_reconstruction(model, dataloader, device, epoch, num_images=8):
    """可视化重建结果"""
    model.eval()
    
    # 获取一批图像
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # 转换到CPU并调整范围
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # 创建图像网格
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    
    for i in range(num_images):
        # 原始图像
        img = np.transpose(images[i], (1, 2, 0))
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('原始图像', fontsize=10)
        
        # 重建图像
        rec = np.transpose(reconstructed[i], (1, 2, 0))
        rec = (rec + 1) / 2  # 从[-1,1]转换到[0,1]
        rec = np.clip(rec, 0, 1)  # 确保在有效范围内
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('重建图像', fontsize=10)
    
    plt.tight_layout()
    
    # 创建输出目录
    os.makedirs('reconstructions', exist_ok=True)
    plt.savefig(f'reconstructions/epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已保存重建图像到: reconstructions/epoch_{epoch+1}.png')


def main():
    # 超参数
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.001
    
    print('正在加载PCam数据集...')
    
    # 数据变换 (归一化到[-1, 1])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载数据集
    train_dataset = PathMNIST(split='train', transform=data_transform, download=True)
    val_dataset = PathMNIST(split='val', transform=data_transform, download=True)
    test_dataset = PathMNIST(split='test', transform=data_transform, download=True)
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    model = ConvAutoencoder().to(device)
    print(f'\n模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 打印模型结构
    print("\n========== 编码器结构 ==========")
    print(model.encoder)
    print("\n========== 解码器结构 ==========")
    print(model.decoder)
    
    # 损失函数和优化器
    # 使用MSE损失函数来衡量重建质量
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练
    best_val_loss = float('inf')
    print('\n========== 开始训练 ==========\n')
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'训练损失: {train_loss:.6f}')
        print(f'验证损失: {val_loss:.6f}')
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_autoencoder.pth')
            print(f'✓ 保存最佳模型 (验证损失: {val_loss:.6f})')
        
        # 每5个epoch可视化一次重建结果
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_reconstruction(model, val_loader, device, epoch)
        
        print()
    
    # 测试
    print('\n========== 在测试集上评估 ==========')
    checkpoint = torch.load('best_autoencoder.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, test_loader, criterion, device)
    print(f'测试损失 (MSE): {test_loss:.6f}')
    
    # 生成最终的重建可视化
    print('\n生成最终重建图像...')
    visualize_reconstruction(model, test_loader, device, epoch=999, num_images=10)
    
    print('\n========== 训练完成 ==========')
    print(f'最佳模型已保存为: best_autoencoder.pth')
    print(f'编码器可用于问题5的迁移学习')
    print(f'重建图像保存在: reconstructions/ 目录')
    
    # 打印编码后的特征维度
    sample_input = torch.randn(1, 3, 28, 28).to(device)
    encoded = model.encode(sample_input)
    print(f'\n编码后的特征维度: {encoded.shape}')


if __name__ == '__main__':
    main()

