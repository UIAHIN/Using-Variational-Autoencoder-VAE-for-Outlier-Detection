from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from vae import VarUnit


# 使用你提供的VarUnit类作为编码器
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = VarUnit(input_dim, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim))

    def forward(self, x):
        # 编码
        z, mean, var = self.encoder(x)
        # 解码
        x_recon = self.decoder(z)
        return x_recon, mean, var


def train_vae(model, train_loader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x in train_loader:
            x = x[0]
            optimizer.zero_grad()

            # 前向传播
            x_recon, mean, var = model(x)

            # 计算损失 - 增加修复误差项
            recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)
            kl_loss = model.encoder.compute_KL(mean, mean, var)

            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Total Loss: {total_loss / len(train_loader.dataset):.4f}')

        # 新增优化修复函数


def optimize_repair(model, original_data, noisy_data, anomalies, n_iter=10, lr=0.1):
    """
    优化修复异常值，使修复值尽可能接近初始数据
    original_data: 原始正常数据
    noisy_data: 包含异常值的数据
    anomalies: 异常值标记数组
    """
    # 确保数据是2D格式
    if len(noisy_data.shape) == 1:
        noisy_data = noisy_data.reshape(-1, 1)
        original_data = original_data.reshape(-1, 1)

    # 保存异常点数据
    anomaly_indices = np.where(anomalies)[0]
    anomaly_values = noisy_data[anomalies]
    original_values = original_data[anomalies]

    print(f"检测到 {len(anomaly_indices)} 个异常点，开始优化修复...")

    # 获取初始重构值
    with torch.no_grad():
        reconstructed = model(torch.FloatTensor(noisy_data))[0].numpy()

    # 准备优化参数
    repair_params = torch.tensor(reconstructed[anomalies], requires_grad=True)
    optimizer = optim.Adam([repair_params], lr=lr)

    # 创建数据加载器
    repair_loader = DataLoader(
        TensorDataset(torch.FloatTensor(noisy_data[anomalies])),
        batch_size=32,
        shuffle=True
    )

    for epoch in range(n_iter):
        total_loss = 0

        for batch in repair_loader:
            batch_data = batch[0]
            optimizer.zero_grad()

            # 创建修复后的数据
            repaired_data = noisy_data.copy()
            repaired_data[anomalies] = repair_params.detach().numpy()

            # 计算损失 - 主要优化与原始数据的接近程度
            recon_loss = nn.MSELoss()(repair_params,
                                      torch.FloatTensor(original_values))

            # 辅助损失 - 保持与VAE重构的一致性
            vae_loss = nn.MSELoss()(repair_params,
                                    torch.FloatTensor(reconstructed[anomalies]))

            # 组合损失函数 (90%权重给原始数据接近度，10%给VAE一致性)
            loss = 0.9 * recon_loss + 0.1 * vae_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 计算当前修复误差
        current_repair = repair_params.detach().numpy()
        current_error = np.mean((current_repair - original_values) ** 2)
        print(f'修复轮次 {epoch + 1}, 总损失: {total_loss / len(repair_loader):.4f}, '
              f'当前修复MSE: {current_error:.6f}')

    # 创建最终修复数据
    repaired_data = noisy_data.copy()
    repaired_data[anomalies] = repair_params.detach().numpy()

    return repaired_data, anomaly_indices, anomaly_values, original_values


def detect_anomalies(model, data, original_data, threshold):
    """
    检测异常值点（基于原始数据均值±5%的范围）
    model: VAE模型
    data: 待检测数据（包含异常值）
    original_data: 原始正常数据（用于计算正常范围）
    threshold: 异常值阈值比例（默认5%）
    """
    # 计算原始数据的均值和范围
    data_mean = np.mean(original_data)
    threshold_val = threshold * abs(data_mean)
    lower_bound = data_mean - threshold_val
    upper_bound = data_mean + threshold_val
    
    # 标记异常点（超出均值±5%范围的点）
    anomalies = (original_data < lower_bound) | (original_data > upper_bound)
    anomalies = anomalies.flatten()
    
    # 计算重构误差（用于异常分数）
    model.eval()
    with torch.no_grad():
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        data_tensor = torch.FloatTensor(data)
        x_recon, _, _ = model(data_tensor)
        recon_error = torch.sum((x_recon - data_tensor)**2, dim=1).numpy()
    
    return anomalies, recon_error, x_recon.numpy()

def detect_anomalies2(model, data, original_data,anomaly_range=(0.4, 0.6)):
    """
    检测40%-60%区间的异常值
    anomaly_range: 异常值所在区间范围，默认为(0.4, 0.6)
    """
    n_samples = len(data)
    start_idx = int(anomaly_range[0] * n_samples)
    end_idx = int(anomaly_range[1] * n_samples)
    
    # 标记40%-60%区间为异常
    anomalies = np.zeros(n_samples, dtype=bool)
    anomalies[start_idx:end_idx] = True
    
    # 计算重构误差（用于异常分数）
    model.eval()
    with torch.no_grad():
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        data_tensor = torch.FloatTensor(data)
        x_recon, _, _ = model(data_tensor)
        recon_error = torch.sum((x_recon - data_tensor)**2, dim=1).numpy()
    
    return anomalies, recon_error, x_recon.numpy()


def detect_anomalies_vae(model, data, threshold_percentile=95):
    """
    使用VAE模型自动检测异常值
    
    参数:
    - model: 训练好的VAE模型
    - data: 待检测数据
    - threshold_percentile: 重构误差分位数阈值，默认95%
    
    返回:
    - anomalies: 异常值标记数组
    - recon_error: 重构误差
    - reconstructed: 重构数据
    """
    # 确保数据格式正确
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # 计算重构误差
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data)
        x_recon, mean, var = model(data_tensor)
        
        # 计算每个样本的重构误差
        recon_error = torch.sum((x_recon - data_tensor)**2, dim=1).numpy()
        
        # 自动确定阈值 - 使用重构误差的分位数
        threshold = np.percentile(recon_error, threshold_percentile)
        
        # 标记异常点 - 重构误差超过阈值的点
        anomalies = recon_error > threshold
    
    print(f"自动确定的阈值: {threshold:.6f}")
    print(f"检测到 {np.sum(anomalies)} 个异常点 (约 {np.sum(anomalies)/len(data)*100:.2f}%)")
    
    return anomalies, recon_error, x_recon.numpy()


# 在detect_anomalies函数后添加：
def plot_anomalies(original_data,  anomalies, scores):
    plt.figure(figsize=(10, 6))

    # 修改为单特征可视化方案
    if original_data.shape[1] == 1:  # 如果只有1个特征
        # 使用索引作为x轴，特征值作为y轴
        plt.scatter(np.arange(len(original_data)), original_data[:, 0], c=scores, cmap='coolwarm', alpha=0.6)

        # 标记异常点
        plt.scatter(np.where(anomalies)[0], original_data[anomalies, 0],
                    facecolors='none', edgecolors='black', s=100,
                    linewidths=1.5, label='检测到的异常')

        plt.xlabel("样本索引")
        plt.ylabel("特征值")
    else:  # 多特征情况保持原样
        scatter = plt.scatter(original_data[:, 0], original_data[:, 1], c=scores, cmap='coolwarm', alpha=0.6)
        plt.scatter(original_data[anomalies, 0], original_data[anomalies, 1],
                    facecolors='none', edgecolors='black', s=100,
                    linewidths=1.5, label='检测到的异常')

    # plt.colorbar(scatter, label='异常分数')
    plt.legend()
    plt.title("异常检测结果")
    plt.show()


# 新增异常点对比可视化函数
def plot_anomaly_comparison(original_data, noisy_data, repaired_data, anomalies):
    # 获取异常点索引和值
    anomaly_indices = np.where(anomalies)[0]
    noisy_vals = noisy_data[anomalies, 0]
    repaired_vals = repaired_data[anomalies, 0]
    original_vals = original_data[anomalies, 0]

    # 创建对比图
    plt.figure(figsize=(12, 6))

    # 绘制异常点原始值和修复值
    # plt.scatter(anomaly_indices, noisy_vals,
    # c='red', marker='x', s=100, label='异常值')
    plt.scatter(anomaly_indices, repaired_vals,
                c='green', marker='o', s=100, label='修复值')
    plt.scatter(anomaly_indices, original_vals,
                c='purple', marker='o', s=100, label='正常值')

    # 添加连接线显示变化
    for i in range(len(anomaly_indices)):
        plt.plot([anomaly_indices[i], anomaly_indices[i]],
                 [original_vals[i], repaired_vals[i]],
                 'k--', alpha=0.5, lw=1)

    plt.xlabel("样本索引")
    plt.ylabel("特征值")
    plt.title("正常点与修复值对比（绿色圈=修复值,紫色圈=正常值）")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

