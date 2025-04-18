from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from exp import *

if __name__ == "__main__":
    # 1. 从CSV文件加载数据
    original_data = pd.read_csv('U.csv').values[:, 31].reshape(-1, 1)

    # 2. 在40%-60%区间添加异常值
    np.random.seed(42)
    n_samples = len(original_data)
    
    # 计算40%和60%的位置索引
    start_idx = int(0.4 * n_samples)
    end_idx = int(0.6 * n_samples)
    middle_range = end_idx - start_idx
    
    # 计算原始数据均值，用于判断异常值方向
    data_mean = np.mean(original_data)
    print(f"数据均值: {data_mean:.4f}")
    
    # 在40%-60%区间添加异常值，根据原始值大小决定异常方向
    noisy_data = original_data.copy()
    for i in range(middle_range):
        idx = start_idx + i
        original_val = original_data[idx][0]
        
        # 根据值的大小方向添加不同的偏移
        if original_val > data_mean:  # 大于均值的点
            # 添加正向偏移，使其更大
            max_deviation = 1 * abs(original_val)  # 使用较大的偏移比例
            deviation = np.abs(np.random.normal(loc=0.7*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] += deviation  # 确保是加上正值
        else:  # 小于均值的点
            # 添加负向偏移，使其更小
            max_deviation = 1 * abs(original_val)
            deviation = np.abs(np.random.normal(loc=0.7*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] -= deviation  # 确保是减去正值
    
    # 统计异常值情况
    larger_count = np.sum((noisy_data > original_data)[start_idx:end_idx])
    smaller_count = np.sum((noisy_data < original_data)[start_idx:end_idx])
    print(f"创建了 {larger_count} 个增大的异常点和 {smaller_count} 个减小的异常点")

    # 创建用于训练的干净数据集（移除40%-60%区间）
    clean_indices = np.ones(n_samples, dtype=bool)
    clean_indices[start_idx:end_idx] = False
    clean_data = original_data[clean_indices]
    print(f"用于训练的干净数据点数量: {len(clean_data)}")

    # 3. 使用干净数据训练模型
    X_train, X_val = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    train_data = torch.FloatTensor(X_train)
    val_data = torch.FloatTensor(X_val)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    
    # 4. 训练模型（仅在干净数据上）
    model = VAE(input_dim=X_train.shape[1], latent_dim=2)
    train_vae(model, train_loader, epochs=50)
    
    # 5. 在未知数据（含有异常）上进行检测
    print("\n=== 在未知数据上应用异常检测 ===")

    # 5.1 使用指定区间的检测方法
    print("\n=== 基于预定义区间的检测 ===")
    anomalies_range, scores_range, reconstructed_range = detect_anomalies2(
        model, noisy_data, original_data, anomaly_range=(0.4, 0.6))
    print(f"检测到 {sum(anomalies_range)} 个异常点（基于40%-60%区间）")
    
    # 5.2 添加VAE自动异常检测
    print("\n=== 基于VAE的自动异常检测 ===")
    anomalies_vae, scores_vae, reconstructed_vae = detect_anomalies_vae(
                     model, noisy_data, threshold_percentile=80)
    print(f"检测到 {sum(anomalies_vae)} 个异常点（基于VAE重构误差）")
    
    # 6. 比较两种方法的检测结果
    print("\n=== 两种检测方法比较 ===")
    print(f"区间检测: 检测到 {sum(anomalies_range)} 个异常点")
    print(f"VAE自动检测: 检测到 {sum(anomalies_vae)} 个异常点")
    
    # 计算各检测方法与真实异常区间的重叠度
    true_anomalies = np.zeros(n_samples, dtype=bool)
    true_anomalies[start_idx:end_idx] = True
    
    overlap_range = np.sum(anomalies_range & true_anomalies) / np.sum(true_anomalies)
    overlap_vae = np.sum(anomalies_vae & true_anomalies) / np.sum(true_anomalies)
    
    print(f"区间检测与真实异常的重叠率: {overlap_range:.2f}")
    print(f"VAE自动检测与真实异常的重叠率: {overlap_vae:.2f}")

    # 7. 使用两种不同检测结果进行修复
    print("\n=== 使用不同检测结果进行修复 ===")
    
    # 7.1 使用区间检测结果修复
    repaired_data_range, _, _, _ = optimize_repair(
        model, original_data, noisy_data, anomalies_range)
    
    # 7.2 使用VAE自动检测结果修复
    repaired_data_vae, _, _, _ = optimize_repair(
        model, original_data, noisy_data, anomalies_vae)
    
    # 8. 计算两种方法的修复误差
    repair_mse_range = np.mean((repaired_data_range[anomalies_range] - original_data[anomalies_range])**2)
    repair_mape_range = np.mean(np.abs((repaired_data_range[anomalies_range] - original_data[anomalies_range]) / original_data[anomalies_range])*100)
    
    repair_mse_vae = np.mean((repaired_data_vae[anomalies_vae] - original_data[anomalies_vae])**2)
    repair_mape_vae = np.mean(np.abs((repaired_data_vae[anomalies_vae] - original_data[anomalies_vae]) / original_data[anomalies_vae])*100)
    
    print("\n=== 修复效果比较 ===")
    print(f"区间检测修复 - MSE: {repair_mse_range:.6f}, MAPE: {repair_mape_range:.2f}%")
    print(f"VAE自动检测修复 - MSE: {repair_mse_vae:.6f}, MAPE: {repair_mape_vae:.2f}%")
    
    # 9. 可视化结果比较
    # 9.1 两种检测方法对比图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 区间检测结果
    axes[0].scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c='blue', alpha=0.4)
    axes[0].scatter(np.where(anomalies_range)[0], noisy_data[anomalies_range, 0], 
                   c='red', marker='x', s=80, label='区间检测异常')
    axes[0].scatter(np.where(anomalies_range)[0], repaired_data_range[anomalies_range, 0], 
                   c='green', marker='o', s=50, label='修复值')
    axes[0].set_title("基于区间的异常检测与修复")
    axes[0].set_xlabel("样本索引")
    axes[0].set_ylabel("特征值")
    axes[0].legend()
    axes[0].grid(True)
    
    # VAE检测结果
    axes[1].scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c='blue', alpha=0.4)
    axes[1].scatter(np.where(anomalies_vae)[0], noisy_data[anomalies_vae, 0], 
                   c='red', marker='x', s=80, label='VAE检测异常')
    axes[1].scatter(np.where(anomalies_vae)[0], repaired_data_vae[anomalies_vae, 0], 
                   c='green', marker='o', s=50, label='修复值')
    axes[1].set_title("VAE自动异常检测与修复")
    axes[1].set_xlabel("样本索引")
    axes[1].set_ylabel("特征值")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 9.2 VAE检测分数可视化
    plt.figure(figsize=(10, 6))
    plt.title("VAE自动异常检测结果")
    plt.scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c=scores_vae, cmap='coolwarm', alpha=0.6)
    plt.scatter(np.where(anomalies_vae)[0], noisy_data[anomalies_vae, 0], 
                facecolors='none', edgecolors='black', s=100, linewidths=1.5, label='VAE检测异常')
    plt.colorbar(label='异常分数')
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 9.3 异常修复前后对比
    fig = plt.figure(figsize=(14, 10))
    
    # 原始数据
    plt.subplot(2, 2, 1)
    plt.scatter(np.arange(len(original_data)), original_data[:, 0], c='blue', alpha=0.6)
    plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.2, label='异常区间(40%-60%)')
    plt.title("原始数据")
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)

    # 加入异常值后的数据
    plt.subplot(2, 2, 2)
    plt.scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c='blue', alpha=0.6)
    plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.2, label='异常区间(40%-60%)')
    plt.scatter(np.arange(start_idx, end_idx), noisy_data[start_idx:end_idx, 0],
                c='red', marker='x', s=50, label='人工添加异常区域')
    plt.title("添加异常值后数据")
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)

    # 两种方法修复结果对比
    plt.subplot(2, 2, 3)
    plt.scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c='blue', alpha=0.2, label='含异常数据')
    plt.scatter(np.where(anomalies_range)[0], repaired_data_range[anomalies_range, 0],
                c='green', marker='o', s=30, label='区间方法修复')
    plt.scatter(np.where(anomalies_vae)[0], repaired_data_vae[anomalies_vae, 0],
                c='purple', marker='s', s=30, label='VAE方法修复')
    plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.1, label='异常区间')
    plt.title("两种方法修复结果对比")
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.legend()
    plt.grid(True)
    
    # 异常区域放大
    plt.subplot(2, 2, 4)
    buffer = 20  # 显示异常区域前后的缓冲区
    plt.scatter(np.arange(start_idx-buffer, end_idx+buffer), 
                noisy_data[start_idx-buffer:end_idx+buffer, 0], 
                c='blue', alpha=0.2, label='含异常数据')
    plt.scatter(np.where(anomalies_range)[0], repaired_data_range[anomalies_range, 0],
                c='green', marker='o', s=50, label='区间方法修复')
    plt.scatter(np.where(anomalies_vae)[0], repaired_data_vae[anomalies_vae, 0],
                c='purple', marker='s', s=50, label='VAE方法修复')
    plt.axvspan(start_idx, end_idx, color='yellow', alpha=0.1)
    plt.title("异常区域修复结果(放大)")
    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.xlim(start_idx-buffer, end_idx+buffer)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

