import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 定义常量
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
BATCH_SIZE = 128
NUM_WORKERS = 4
INITIAL_LR = 0.1
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 250
PATIENCE = 10  # 恢复为10个epoch的耐心值
BN_MOMENTUM = 0.1

class Cutout:
    """Cutout数据增强"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        
        for _ in range(self.n_holes):
            y = torch.randint(h, (1,)).item()
            x = torch.randint(w, (1,)).item()
            y1 = max(y - self.length // 2, 0)
            y2 = min(y + self.length // 2, h)
            x1 = max(x - self.length // 2, 0)
            x2 = min(x + self.length // 2, w)
            mask[y1:y2, x1:x2] = 0
        
        return img * mask.expand_as(img)

def prepare_data(data_dir='./data'):
    """准备CIFAR10数据集和数据加载器"""
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 填充后随机裁剪
        transforms.RandomHorizontalFlip(),         # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(n_holes=1, length=16)              # Cutout
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    
    # 加载数据集并创建数据加载器
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, test_loader

def create_model(device):
    """创建改进的ResNet18模型"""
    # 创建模型参数
    params = {
        'conv1.weight': torch.randn(32, 3, 3, 3, requires_grad=True, device=device),  # 32通道
        'bn1.weight': torch.ones(32, requires_grad=True, device=device),
        'bn1.bias': torch.zeros(32, requires_grad=True, device=device),
        'bn1.running_mean': torch.zeros(32, requires_grad=False, device=device),
        'bn1.running_var': torch.ones(32, requires_grad=False, device=device),
        'fc.weight': torch.randn(10, 256, requires_grad=True, device=device),  # 最后一层使用256通道
        'fc.bias': torch.zeros(10, requires_grad=True, device=device)
    }
    
    # 创建每个block的参数
    in_planes = 32  # 起始通道数为32
    for planes, stride in [(32,1), (64,2), (128,2), (256,2)]:  # 恢复原始通道数配置
        for i in range(2):  # 每层2个BasicBlock
            block_name = f'block_{planes}_{i}'
            s = stride if i == 0 else 1
            
            # 主分支参数
            params[f'{block_name}.conv1.weight'] = torch.randn(planes, in_planes, 3, 3, requires_grad=True, device=device)
            params[f'{block_name}.bn1.weight'] = torch.ones(planes, requires_grad=True, device=device)
            params[f'{block_name}.bn1.bias'] = torch.zeros(planes, requires_grad=True, device=device)
            params[f'{block_name}.bn1.running_mean'] = torch.zeros(planes, requires_grad=False, device=device)
            params[f'{block_name}.bn1.running_var'] = torch.ones(planes, requires_grad=False, device=device)
            
            params[f'{block_name}.conv2.weight'] = torch.randn(planes, planes, 3, 3, requires_grad=True, device=device)
            params[f'{block_name}.bn2.weight'] = torch.ones(planes, requires_grad=True, device=device)
            params[f'{block_name}.bn2.bias'] = torch.zeros(planes, requires_grad=True, device=device)
            params[f'{block_name}.bn2.running_mean'] = torch.zeros(planes, requires_grad=False, device=device)
            params[f'{block_name}.bn2.running_var'] = torch.ones(planes, requires_grad=False, device=device)
            
            # shortcut参数
            if s != 1 or in_planes != planes:
                params[f'{block_name}.shortcut.weight'] = torch.randn(planes, in_planes, 1, 1, requires_grad=True, device=device)
                params[f'{block_name}.shortcut_bn.weight'] = torch.ones(planes, requires_grad=True, device=device)
                params[f'{block_name}.shortcut_bn.bias'] = torch.zeros(planes, requires_grad=True, device=device)
                params[f'{block_name}.shortcut_bn.running_mean'] = torch.zeros(planes, requires_grad=False, device=device)
                params[f'{block_name}.shortcut_bn.running_var'] = torch.ones(planes, requires_grad=False, device=device)
            
            in_planes = planes
    
    # 初始化参数
    for k, v in params.items():
        if 'weight' in k and 'bn' not in k and v.requires_grad:
            nn.init.kaiming_normal_(v)
    
    def forward(x, params, training=True):
        # 初始层 - 使用3x3卷积替代7x7卷积和池化层
        out = F.conv2d(x, params['conv1.weight'], padding=1, stride=1)  # stride=1保留更多信息
        if training:
            batch_mean = out.mean([0, 2, 3]).detach()  # 不计算梯度
            batch_var = out.var([0, 2, 3], unbiased=False).detach()  # 不计算梯度
            params['bn1.running_mean'] = params['bn1.running_mean'].detach() * (1 - BN_MOMENTUM) + batch_mean * BN_MOMENTUM
            params['bn1.running_var'] = params['bn1.running_var'].detach() * (1 - BN_MOMENTUM) + batch_var * BN_MOMENTUM
        out = F.batch_norm(out, params['bn1.running_mean'].detach(), params['bn1.running_var'].detach(),
                          params['bn1.weight'], params['bn1.bias'], training)
        out = F.relu(out)
        
        # ResNet层
        in_planes = 32
        for planes, stride in [(32,1), (64,2), (128,2), (256,2)]:
            for i in range(2):
                block_name = f'block_{planes}_{i}'
                identity = out
                s = stride if i == 0 else 1
                
                # 主分支
                out = F.conv2d(out, params[f'{block_name}.conv1.weight'], 
                             stride=s, padding=1)
                if training:
                    batch_mean = out.mean([0, 2, 3]).detach()
                    batch_var = out.var([0, 2, 3], unbiased=False).detach()
                    params[f'{block_name}.bn1.running_mean'] = params[f'{block_name}.bn1.running_mean'].detach() * (1 - BN_MOMENTUM) + batch_mean * BN_MOMENTUM
                    params[f'{block_name}.bn1.running_var'] = params[f'{block_name}.bn1.running_var'].detach() * (1 - BN_MOMENTUM) + batch_var * BN_MOMENTUM
                out = F.batch_norm(out, params[f'{block_name}.bn1.running_mean'].detach(),
                                 params[f'{block_name}.bn1.running_var'].detach(),
                                 params[f'{block_name}.bn1.weight'],
                                 params[f'{block_name}.bn1.bias'], training)
                out = F.relu(out)
                
                out = F.conv2d(out, params[f'{block_name}.conv2.weight'], 
                             padding=1)
                if training:
                    batch_mean = out.mean([0, 2, 3]).detach()
                    batch_var = out.var([0, 2, 3], unbiased=False).detach()
                    params[f'{block_name}.bn2.running_mean'] = params[f'{block_name}.bn2.running_mean'].detach() * (1 - BN_MOMENTUM) + batch_mean * BN_MOMENTUM
                    params[f'{block_name}.bn2.running_var'] = params[f'{block_name}.bn2.running_var'].detach() * (1 - BN_MOMENTUM) + batch_var * BN_MOMENTUM
                out = F.batch_norm(out, params[f'{block_name}.bn2.running_mean'].detach(),
                                 params[f'{block_name}.bn2.running_var'].detach(),
                                 params[f'{block_name}.bn2.weight'],
                                 params[f'{block_name}.bn2.bias'], training)
                
                # shortcut
                if s != 1 or in_planes != planes:
                    identity = F.conv2d(identity, params[f'{block_name}.shortcut.weight'],
                                      stride=s)
                    if training:
                        batch_mean = identity.mean([0, 2, 3]).detach()
                        batch_var = identity.var([0, 2, 3], unbiased=False).detach()
                        params[f'{block_name}.shortcut_bn.running_mean'] = params[f'{block_name}.shortcut_bn.running_mean'].detach() * (1 - BN_MOMENTUM) + batch_mean * BN_MOMENTUM
                        params[f'{block_name}.shortcut_bn.running_var'] = params[f'{block_name}.shortcut_bn.running_var'].detach() * (1 - BN_MOMENTUM) + batch_var * BN_MOMENTUM
                    identity = F.batch_norm(identity, 
                                         params[f'{block_name}.shortcut_bn.running_mean'].detach(),
                                         params[f'{block_name}.shortcut_bn.running_var'].detach(),
                                         params[f'{block_name}.shortcut_bn.weight'],
                                         params[f'{block_name}.shortcut_bn.bias'], training)
                
                out = F.relu(out + identity)
                in_planes = planes
        
        # 最后的层
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = F.linear(out, params['fc.weight'], params['fc.bias'])
        return out
    
    # 创建模型
    model = lambda x, training=True: forward(x, params, training)
    model.parameters = lambda: [p for p in params.values() if p.requires_grad]
    
    # 检查参数量
    num_params = sum(p.numel() for p in params.values() if p.requires_grad)
    print(f"模型总参数量: {num_params:,}")
    assert num_params < 5000000, f"模型参数量 ({num_params:,}) 超过500万限制"
    
    return model

def train(model, train_loader, test_loader, device):
    """训练模型的完整流程"""
    # 初始化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    
    best_acc = 0
    best_params = None
    best_epoch = -1
    no_improve_count = 0  # 记录验证准确率没有提升的轮数
    current_lr = INITIAL_LR  # 当前学习率
    
    print(f"\n开始训练...")
    print(f"初始学习率: {INITIAL_LR}")
    print(f"总训练轮数: {NUM_EPOCHS}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"批次大小: {train_loader.batch_size}")
    print("=" * 50)
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train = True
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(total=len(train_loader), 
                   desc=f'训练 Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 训练步骤
            optimizer.zero_grad()
            outputs = model(inputs, training=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 更新统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            train_acc = 100. * correct / total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_acc:.2f}%',
                'lr': f'{current_lr:.6f}'
            })
            train_pbar.update(1)
        
        train_pbar.close()
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.train = False
        running_loss = 0.0
        correct = 0
        total = 0
        
        test_pbar = tqdm(total=len(test_loader), 
                  desc=f'验证 Epoch {epoch+1}/{NUM_EPOCHS}')
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, training=False)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                val_acc = 100. * correct / total
                test_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_acc:.2f}%'
                })
                test_pbar.update(1)
        
        test_pbar.close()
        val_loss = running_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        # 更新学习率
        if val_acc > best_acc:
            print(f"\n发现更好的模型！")
            print(f"验证准确率: {val_acc:.2f}% > {best_acc:.2f}%")
            
            best_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0  # 重置计数器
            # 保存所有参数
            best_params = []
            for param in model.parameters():
                best_params.append(param.data.clone())
            
            # 保存新的最佳模型
            try:
                torch.save({
                    'epoch': epoch,
                    'params': best_params,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                    'current_lr': current_lr
                }, 'best_model.pth')
                print("最佳模型已保存")
            except Exception as e:
                print(f"保存模型时出错: {str(e)}")
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                current_lr *= 0.5  # 学习率减半
                print(f"\n连续{PATIENCE}个epoch验证准确率未提升，学习率从 {current_lr*2:.6f} 降低到 {current_lr:.6f}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                no_improve_count = 0  # 重置计数器
        
        # 打印统计信息
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')
        print(f'学习率: {current_lr:.6f}')
        print(f'最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch+1})')
        print("=" * 50)
    
    return best_params, best_acc

if __name__ == '__main__':
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')
    
    # 准备数据
    train_loader, test_loader = prepare_data()
    
    # 创建并训练模型
    model = create_model(device)
    
    best_params, best_acc = train(model, train_loader, test_loader, device)
    print(f'训练完成。最佳准确率: {best_acc:.2f}%') 