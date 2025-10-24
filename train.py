import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import PathMNIST
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 输入: (batch, 3, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = self.pool(self.relu(self.conv3(x)))  # (batch, 128, 3, 3)
        
        x = x.view(-1, 128 * 3 * 3)  # 展平
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


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


def main():
    # 超参数
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    
    print('正在加载PCam数据集...')
    
    # 数据变换
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # 加载数据集 (PathMNIST是PCam的变体，会自动下载)
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
    model = SimpleCNN(num_classes=2).to(device)
    print(f'\n模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    best_val_acc = 0.0
    print('\n开始训练...\n')
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%\n')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型 (验证准确率: {val_acc:.2f}%)\n')
    
    # 测试
    print('在测试集上评估...')
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'\n最终测试准确率: {test_acc:.2f}%')
    print(f'模型已保存为: best_model.pth')


if __name__ == '__main__':
    main()

