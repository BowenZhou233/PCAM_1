"""
问题5: 基于迁移学习的CNN分类器 (分类器2)
使用预训练的CAE编码器 + 新的分类层
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
import numpy as np
from tqdm import tqdm
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


# ========== 从CAE导入编码器结构 ==========
class Encoder(nn.Module):
    """编码器: 与train_cae.py中的结构相同"""
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        return self.encoder(x)


# ========== 迁移学习分类器 ==========
class TransferClassifier(nn.Module):
    """
    迁移学习分类器
    
    原理说明:
    1. 重用预训练的编码器: 编码器已经学会了从图像中提取有用的特征表示
    2. 冻结/微调策略: 可以选择冻结编码器权重或允许微调
    3. 新的分类层: 在编码器之后添加全连接层进行分类
    4. 优势: 利用无监督学习的特征表示，加速有监督分类训练
    """
    def __init__(self, num_classes=2, freeze_encoder=False):
        super(TransferClassifier, self).__init__()
        
        # 1. 加载预训练的编码器
        self.encoder = Encoder()
        
        # 2. 是否冻结编码器参数
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ 编码器权重已冻结 (仅训练分类层)")
        else:
            print("✓ 编码器权重可微调 (端到端训练)")
        
        # 3. 新的分类层
        # 编码器输出: (batch, 128, 4, 4)
        # 展平后: 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # 第一个全连接层
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            # 第二个全连接层
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            
            # 输出层
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 通过编码器提取特征
        features = self.encoder(x)
        
        # 通过分类器得到预测
        output = self.classifier(features)
        
        return output
    
    def load_pretrained_encoder(self, checkpoint_path):
        """加载预训练的编码器权重"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"✓ 成功加载预训练编码器: {checkpoint_path}")
            print(f"  预训练epoch: {checkpoint['epoch'] + 1}")
            print(f"  预训练验证损失: {checkpoint['val_loss']:.6f}")
            return True
        else:
            print(f"⚠ 未找到预训练模型: {checkpoint_path}")
            print("  将从随机初始化开始训练")
            return False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc='训练中'):
        inputs, labels = inputs.to(device), labels.to(device).squeeze().long()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='验证中'):
            inputs, labels = inputs.to(device), labels.to(device).squeeze().long()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def compare_with_baseline(baseline_path='best_model.pth'):
    """与问题3的基线模型进行比较"""
    if os.path.exists(baseline_path):
        print(f"\n{'='*50}")
        print("检测到问题3的基线模型，可以进行性能对比")
        print(f"{'='*50}")
        return True
    return False


def main():
    # 超参数
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    freeze_encoder = False  # 设置为True则冻结编码器，False则微调
    
    print('='*60)
    print('问题5: 基于迁移学习的CNN分类器')
    print('='*60)
    
    print('\n【迁移学习原理】')
    print('1. 重用预训练编码器: 利用自编码器学到的特征表示能力')
    print('2. 添加分类层: 在编码器之后添加全连接层用于分类任务')
    print('3. 训练策略: 可以冻结编码器(特征提取)或微调(端到端优化)')
    print('4. 优势: 减少训练时间，提高泛化能力，需要更少的标注数据\n')
    
    print('正在加载PCam数据集...')
    
    # 数据变换
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
    
    # 创建迁移学习模型
    print('\n创建迁移学习分类器...')
    model = TransferClassifier(num_classes=2, freeze_encoder=freeze_encoder).to(device)
    
    # 加载预训练的编码器权重
    pretrained_loaded = model.load_pretrained_encoder('best_autoencoder.pth')
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n总参数量: {total_params/1e6:.2f}M')
    print(f'可训练参数量: {trainable_params/1e6:.2f}M')
    
    # 打印新增的分类层结构
    print("\n========== 新的分类层结构 ==========")
    print(model.classifier)
    print("\n【分类层设计原理】")
    print("- Flatten: 将特征图展平为向量 (128*4*4=2048维)")
    print("- FC1: 2048 -> 512, 提取高级特征")
    print("- FC2: 512 -> 128, 进一步压缩特征")
    print("- Output: 128 -> 2, 二分类输出")
    print("- BatchNorm + Dropout: 防止过拟合，提高泛化能力")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 检查基线模型
    compare_with_baseline()
    
    # 训练
    best_val_acc = 0.0
    print('\n========== 开始训练 ==========\n')
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'freeze_encoder': freeze_encoder,
                'pretrained_loaded': pretrained_loaded,
            }, 'best_transfer_model.pth')
            print(f'✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)')
        
        print()
    
    # 测试
    print('\n========== 在测试集上评估 ==========')
    checkpoint = torch.load('best_transfer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f'\n最终测试准确率: {test_acc:.2f}%')
    print(f'最佳验证准确率: {checkpoint["val_acc"]:.2f}%')
    print(f'是否使用预训练: {"是" if checkpoint["pretrained_loaded"] else "否"}')
    print(f'编码器训练策略: {"冻结" if checkpoint["freeze_encoder"] else "微调"}')
    
    print('\n========== 训练完成 ==========')
    print(f'模型已保存为: best_transfer_model.pth')
    
    # 性能总结
    print('\n========== 性能总结 ==========')
    print(f'迁移学习分类器 (分类器2):')
    print(f'  - 测试准确率: {test_acc:.2f}%')
    print(f'  - 可训练参数: {trainable_params/1e6:.2f}M')
    
    if os.path.exists('best_model.pth'):
        print(f'\n提示: 可以与问题3的基线分类器(best_model.pth)进行性能对比')
    
    print('\n【迁移学习的优势】')
    if pretrained_loaded:
        print('✓ 利用了无监督预训练的特征表示')
        print('✓ 训练收敛更快，需要更少的epoch')
        print('✓ 可能在小数据集上表现更好')
    else:
        print('⚠ 未使用预训练权重，建议先运行train_cae.py')


if __name__ == '__main__':
    main()

