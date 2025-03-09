import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F

from train import create_model  # 从train.py导入

# 定义常量
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
BATCH_SIZE = 128
NUM_WORKERS = 0  # 使用单进程避免Mac上的问题

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

def load_test_data(data_path='cifar_test_nolabel.pkl'):
    """加载无标签测试数据"""
    print(f"正在从 {data_path} 加载数据...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 打印数据结构信息
    print(f"数据类型: {type(data)}")
    if isinstance(data, dict):
        print(f"字典键: {list(data.keys())}")
    
    # 如果数据是字典格式，获取图像数据
    if isinstance(data, dict):
        if b'data' in data:
            data = data[b'data']
            print("使用 b'data' 键的数据")
        elif 'data' in data:
            data = data['data']
            print("使用 'data' 键的数据")
        elif 'images' in data:
            data = data['images']
            print("使用 'images' 键的数据")
        elif 'x' in data:
            data = data['x']
            print("使用 'x' 键的数据")
        else:
            print(f"警告：未找到标准键名，尝试使用第一个可用的键")
            first_key = list(data.keys())[0]
            data = data[first_key]
            print(f"使用键 '{first_key}' 的数据")
    
    # 确保数据格式正确
    data = np.array(data)
    print(f"原始数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{data.min()}, {data.max()}]")
    
    # 如果数据是平铺的，重新整形
    if len(data.shape) == 2 and data.shape[1] == 3072:  # 32*32*3 = 3072
        print("检测到平铺数据，重新整形...")
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    elif len(data.shape) == 2:
        print("检测到未知的2D数据格式...")
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    print(f"处理后数据形状: {data.shape}")
    print(f"处理后数据范围: [{data.min()}, {data.max()}]")
    
    # 转换为tensor并应用预处理
    processed_data = []
    for i, img in enumerate(tqdm(data, desc="处理图像")):
        if i == 0:  # 打印第一张图片的信息
            print(f"\n第一张图片信息:")
            print(f"形状: {img.shape}")
            print(f"类型: {img.dtype}")
            print(f"范围: [{img.min()}, {img.max()}]")
        
        img = transforms.ToPILImage()(img)
        img = transform(img)
        processed_data.append(img)
        
        if i == 0:  # 打印处理后的第一张图片信息
            print(f"\n处理后第一张图片信息:")
            print(f"形状: {img.shape}")
            print(f"类型: {img.dtype}")
            print(f"范围: [{img.min()}, {img.max()}]")
    
    processed_data = torch.stack(processed_data)
    print(f"\n最终数据形状: {processed_data.shape}")
    print(f"最终数据类型: {processed_data.dtype}")
    print(f"最终数据范围: [{processed_data.min().item()}, {processed_data.max().item()}]")
    return processed_data

def predict():
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')
    
    # 加载最佳模型
    if os.path.exists('best_model.pth'):
        print("找到模型文件，正在加载...")
        checkpoint = torch.load('best_model.pth', map_location=device)
        
        print("\n检查点信息:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], (int, float, str)):
                print(f"{key}: {checkpoint[key]}")
            elif isinstance(checkpoint[key], list):
                print(f"{key}: List[{len(checkpoint[key])} items]")
            else:
                print(f"{key}: {type(checkpoint[key])}")
        
        if 'model_state_dict' in checkpoint:
            print("\n检测到旧版本模型格式，重新训练模型以获得兼容的格式")
            return
        elif 'params' in checkpoint:
            # 新版本模型格式
            saved_params = checkpoint['params']
            
            # 创建模型并获取参数
            model = create_model(device)
            model_params = [p for p in model.parameters()]
            
            print(f"\n参数检查:")
            print(f"存档参数数量: {len(saved_params)}")
            print(f"模型参数数量: {len(model_params)}")
            
            # 加载参数
            try:
                # 更新所有参数
                for param, saved_param in zip(model_params, saved_params):
                    param.data.copy_(saved_param)
                
                print(f"\n加载模型成功!")
                print(f"训练信息:")
                print(f"- 轮次: {checkpoint['epoch']}")
                print(f"- 训练损失: {checkpoint['train_loss']:.4f}")
                print(f"- 验证损失: {checkpoint['val_loss']:.4f}")
                print(f"- 训练准确率: {checkpoint['train_acc']:.2f}%")
                print(f"- 验证准确率: {checkpoint['val_acc']:.2f}%")
                print(f"- 当前学习率: {checkpoint.get('current_lr', 'N/A')}")
            except Exception as e:
                print(f"加载参数时出错: {str(e)}")
                return
        else:
            print("警告：模型格式不正确！")
            return
    else:
        print("警告：未找到最佳模型文件！")
        return
    
    # 加载无标签测试集
    print("\n正在加载测试数据...")
    test_data = load_test_data()
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 预测
    predictions = []
    print("\n开始预测测试集...")
    model.train = False  # 确保模型在评估模式
    
    with torch.no_grad():
        for batch_idx, (inputs,) in enumerate(tqdm(test_loader, desc='Predicting')):
            inputs = inputs.to(device)
            outputs = model(inputs)  # 移除training=False参数
            
            # 打印第一个批次的详细信息
            if batch_idx == 0:
                print(f"\n第一个批次信息:")
                print(f"输入形状: {inputs.shape}")
                print(f"输入范围: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
                print(f"输出形状: {outputs.shape}")
                print(f"输出示例 (第一个样本):")
                print(f"Logits: {outputs[0].cpu().numpy()}")
                probs = F.softmax(outputs[0], dim=0)
                print(f"Softmax: {probs.cpu().numpy()}")
                print(f"预测类别: {probs.argmax().item()}")
                print(f"置信度: {probs.max().item():.4f}")
            
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            # 记录预测结果和置信度
            for pred, conf in zip(predicted.cpu().numpy(), confidence.cpu().numpy()):
                predictions.append((pred, conf))
    
    # 分离预测标签和置信度
    pred_labels, pred_confidences = zip(*predictions)
    pred_labels = np.array(pred_labels)
    
    # 保存结果
    submission = pd.DataFrame({
        'ID': range(len(pred_labels)),
        'Label': pred_labels
    })
    submission.to_csv('submission.csv', index=False)
    
    print(f'\n预测完成！')
    print(f'结果已保存到 submission.csv')

if __name__ == '__main__':
    predict() 