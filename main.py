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
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
from exp import *


if __name__ == "__main__":
    # 1. 从CSV文件加载数据
    original_data = pd.read_csv('U.csv').values[:, 31].reshape(-1, 1)
    
    # 2. 计算原始数据均值
    data_mean = np.mean(original_data)
    
    # 3. 确定异常值范围（均值±0.5%）
    threshold = 0.005 * abs(data_mean)
    lower_bound = data_mean - threshold
    upper_bound = data_mean + threshold
    
    # 4. 标记原始数据中的异常点
    original_anomalies = (original_data < lower_bound) | (original_data > upper_bound)
    print(f"原始数据中已有 {np.sum(original_anomalies)} 个自然异常点")
    
    # 5. 创建干净数据集（移除异常值）用于训练
    clean_indices = ~original_anomalies.flatten()  # 取非异常点索引
    clean_data = original_data[clean_indices]
    print(f"用于训练的干净数据点数量: {len(clean_data)}")
    
    # 6. 创建带异常值的数据集用于后续检测（使大值更大，小值更小）
    np.random.seed(42)
    noisy_data = original_data.copy()
    anomaly_indices = np.where(original_anomalies.flatten())[0]
    
    for idx in anomaly_indices:
        original_val = original_data[idx][0]
        # 根据值的大小方向添加不同的偏移
        if original_val > data_mean:  # 大于均值的点
            # 添加正向偏移，使其更大
            max_deviation = 1 * abs(original_val)  # 使用更大的偏移比例
            deviation = np.abs(np.random.normal(loc=0.4*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] += deviation  # 确保是加上正值
        else:  # 小于均值的点
            # 添加负向偏移，使其更小
            max_deviation = 1 * abs(original_val)  # 使用更大的偏移比例
            deviation = np.abs(np.random.normal(loc=0.4*max_deviation, scale=0.5*max_deviation))
            noisy_data[idx] -= deviation  # 确保是减去正值
    
    # 输出信息，了解异常值修改的程度
    larger_anomalies = np.sum(noisy_data > original_data)
    smaller_anomalies = np.sum(noisy_data < original_data)
    print(f"创建了 {larger_anomalies} 个增大的异常点和 {smaller_anomalies} 个减小的异常点")
    
    # 7. 使用干净数据训练模型
    X_train, X_val = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    train_data = torch.FloatTensor(X_train)
    val_data = torch.FloatTensor(X_val)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    
    # 8. 训练模型（仅在干净数据上）
    model = VAE(input_dim=X_train.shape[1], latent_dim=2)
    train_vae(model, train_loader, epochs=50)
    
    # 9. 在未知数据（含有异常）上进行检测
    print("\n=== 在未知数据上应用异常检测 ===")
    
    # 10. 使用两种方法进行异常检测和修复
    print("\n=== 基于预定义阈值的检测 ===")
    anomalies_threshold, scores_threshold, reconstructed_threshold = detect_anomalies(
                     model, noisy_data, original_data, threshold=0.005)
    print(f"检测到 {sum(anomalies_threshold)} 个异常点（基于均值±5%范围）")
    
    # 添加VAE自动异常检测
    print("\n=== 基于VAE的自动异常检测 ===")
    anomalies_vae, scores_vae, reconstructed_vae = detect_anomalies_vae(
                     model, noisy_data, threshold_percentile=95.95)
    
    data_mean = np.mean(original_data)
    threshold = 0.005 * np.abs(data_mean)
    original_anomalies = (original_data < (data_mean - threshold)) | (original_data > (data_mean + threshold))
    print(f"实际原始异常点数量: {np.sum(original_anomalies)}")
    
    print("\n=== 检测方法比较 ===")
    print(f"基于阈值检测到的异常点: {sum(anomalies_threshold)}")
    print(f"VAE自动检测到的异常点: {sum(anomalies_vae)}")
    
    print("\n=== 使用VAE自动检测结果进行修复 ===")
    repaired_data_vae, anomaly_indices_vae, anomaly_values_vae, original_values_vae = optimize_repair(
        model, original_data, noisy_data, anomalies_vae)

    repair_mse_vae = np.mean((repaired_data_vae[anomalies_vae] - original_data[anomalies_vae])**2)
    repair_mape_vae = np.mean(np.abs((repaired_data_vae[anomalies_vae] - original_data[anomalies_vae]) / original_data[anomalies_vae])*100)
    print(f"VAE自动检测-修复MSE误差: {repair_mse_vae:.6f}")
    print(f"VAE自动检测-修复MAPE误差: {repair_mape_vae:.6f} %")
    
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
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c='blue', alpha=0.4)
    axes[0].scatter(np.where(anomalies_threshold)[0], noisy_data[anomalies_threshold, 0], 
                   c='red', marker='x', s=80, label='阈值检测异常')
    axes[0].scatter(np.where(anomalies_threshold)[0], repaired_data_vae[anomalies_threshold, 0], 
                   c='green', marker='o', s=50, label='修复值')
    axes[0].set_title("基于阈值的异常检测与修复")
    axes[0].set_xlabel("样本索引")
    axes[0].set_ylabel("特征值")
    axes[0].legend()
    axes[0].grid(True)
    
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
    
    print("\n=== 两种检测方法比较 ===")
    print(f"阈值检测: 检测到 {sum(anomalies_threshold)} 个异常点")
    print(f"基本VAE: 检测到 {sum(anomalies_vae)} 个异常点")
    
    print("\n=== 修复效果比较 ===")
    print(f"阈值检测修复 - MSE: {repair_mse_vae:.6f}, MAPE: {repair_mape_vae:.2f}%")  
    print(f"基本VAE修复 - MSE: {repair_mse_vae:.6f}, MAPE: {repair_mape_vae:.2f}%")

# 计算各检测方法与真实异常点的匹配程度
print("\n=== 检测方法与真实异常的匹配度 ===")
# 使用原始数据中标记的异常点作为真实异常
true_anomaly_count = np.sum(original_anomalies)

# 计算每种方法的准确率、召回率和F1分数
tp_threshold = np.sum(anomalies_threshold & original_anomalies.flatten())
precision_threshold = tp_threshold / np.sum(anomalies_threshold) if np.sum(anomalies_threshold) > 0 else 0
recall_threshold = tp_threshold / true_anomaly_count if true_anomaly_count > 0 else 0
f1_threshold = 2 * (precision_threshold * recall_threshold) / (precision_threshold + recall_threshold) if (precision_threshold + recall_threshold) > 0 else 0

tp_vae = np.sum(anomalies_vae & original_anomalies.flatten())
precision_vae = tp_vae / np.sum(anomalies_vae) if np.sum(anomalies_vae) > 0 else 0
recall_vae = tp_vae / true_anomaly_count if true_anomaly_count > 0 else 0
f1_vae = 2 * (precision_vae * recall_vae) / (precision_vae + recall_vae) if (precision_vae + recall_vae) > 0 else 0

print(f"阈值检测 - 准确率: {precision_threshold:.4f}, 召回率: {recall_threshold:.4f}, F1分数: {f1_threshold:.4f}")
print(f"基本VAE - 准确率: {precision_vae:.4f}, 召回率: {recall_vae:.4f}, F1分数: {f1_vae:.4f}")