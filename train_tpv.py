import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from types import SimpleNamespace


# 从指定目录加载两个 .pkl 文件
def load_datasets(dataset_dir):
    train_path = os.path.join(dataset_dir, "train_regression_dataset.pkl")
    test_path = os.path.join(dataset_dir, "test_regression_dataset.pkl")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found in {dataset_dir}")
    print(f"从 {train_path} 加载训练集...")
    with open(train_path, "rb") as f:
        train_dataset = pickle.load(f)
    print(f"从 {test_path} 加载训练集...")
    with open(test_path, "rb") as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset

# 将隐藏状态和标签 转换为 PyTorch 张量
def prepare_tensors(dataset, device="cpu"):
    hidden_states = [item[0] for item in dataset] # 数据集的长度，每一个样本的长度为4096
    labels = [item[1] for item in dataset]
    H = torch.tensor(np.array(hidden_states), dtype=torch.float32, device=device)
    y = torch.tensor(np.array(labels), dtype=torch.float32, device=device).reshape(
        -1, 1
    )
    return H, y


def fit_linear_regression(H, y, epsilon=1e-10):
    H_transpose = H.t()
    # 添加小正则化以确保数值稳定性
    reg_term = epsilon * torch.eye(H.shape[1], device=H.device)
    # 求解正规方程组: w = (H^T H + ε*I)^(-1) H^T y
    w = torch.linalg.solve(H_transpose @ H + reg_term, H_transpose @ y).squeeze()
    return w # torch.Size([4096])



def predict(H, w):
    return H @ w

# 计算均方误差
def evaluate(predictions, labels):
    predictions_np = predictions.cpu()
    labels_np = labels.cpu().squeeze()
    criterion = nn.MSELoss()
    mse = criterion(labels_np, predictions_np)
    return mse

# 绘制预测值与真实值的对比图
def plot_results(predictions, labels, output_dir):
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy().squeeze()
    # Sort by true labels (i/N values) for better visualization
    sorted_indices = np.argsort(labels_np)
    plot_labels = labels_np[sorted_indices]
    plot_preds = predictions_np[sorted_indices]
    # Create figure with predictions vs true labels
    plt.figure(figsize=(10, 6))
    # Sample a subset of points if there are too many
    sample_size = min(1000, len(plot_labels))
    sample_indices = np.linspace(0, len(plot_labels) - 1, sample_size, dtype=int)
    plt.scatter(
        np.arange(len(plot_preds[sample_indices])),
        plot_preds[sample_indices],
        alpha=0.5,
        s=10,
        label="Predictions vs True",
    )
    # Exponential smoothing
    alpha = 0.1  # smoothing factor
    smoothed_preds = np.zeros_like(plot_preds[sample_indices])
    smoothed_preds[0] = plot_preds[sample_indices][0]
    for t in range(1, len(plot_preds[sample_indices])):
        smoothed_preds[t] = (
            alpha * plot_preds[sample_indices][t] + (1 - alpha) * smoothed_preds[t - 1]
        )
    # 添加对角线
    plt.plot(
        np.arange(len(plot_labels[sample_indices])),
        plot_labels[sample_indices],
        color="orange",
        label="Perfect Predictions",
        alpha=0.6,
    )
    plt.plot(smoothed_preds, color="blue", linewidth=2, label="Smoothed Predictions")
    plt.xlabel("Decoding Step")
    plt.ylabel("Predicted Relative Position")
    plt.title("TPV Predictions vs True Relative Positions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Save figure
    plot_path = os.path.join(output_dir, "tpv_predictions_vs_true.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    args = SimpleNamespace(
        input_dir="llama_math_tpv_dataset",  # Directory containing the processed train and test dataset .pkl files
        output_dir="tpv_model",  # 保存经过训练的模型权重和绘图的目录
        epsilon=1e-10,  # 数值稳定性的正则化参数
        device="cuda",
    )
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Load datasets
    train_dataset, test_dataset = load_datasets(args.input_dir)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    print("Preparing tensors...")
    H_train, y_train = prepare_tensors(train_dataset, device)
    H_test, y_test = prepare_tensors(test_dataset, device)
    print(f"Hidden state dimension: {H_train.shape[1]}") # 4096
    print("Training linear regressor...")
    weights = fit_linear_regression(H_train, y_train, epsilon=args.epsilon)
    # Make predictions
    print("Making predictions...")
    train_preds = predict(H_train, weights)
    test_preds = predict(H_test, weights)
    # Evaluate performance
    train_mse = evaluate(train_preds, y_train)
    test_mse = evaluate(test_preds, y_test)
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    # 创建可视化绘图
    print("Creating visualization plots...")
    plot_results(test_preds, y_test, args.output_dir)
    print("Retraining on combined train and test data...")
    # 合并训练 = 训练集 + 测试集-->重新训练模型
    H_c = torch.cat([H_train, H_test], dim=0)
    y_c = torch.cat([y_train, y_test], dim=0)
    # 在组合数据集上重新训练
    combined_weights = fit_linear_regression(H_c, y_c, epsilon=args.epsilon)
    # 将组合模型权重另存为numpy数组
    weights_np_path = os.path.join(args.output_dir, "tpv_linear_weights.npy")
    np.save(weights_np_path, combined_weights.cpu().numpy())
    print(f"Combined model weights saved to {weights_np_path} as numpy array")
