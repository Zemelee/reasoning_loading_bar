# This software is for non-commercial use only.
# Commercial use requires a separate license.

import os
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def load_datasets(dataset_dir):
    """
    Load the training and test datasets from the specified directory
    
    Args:
        dataset_dir (str): Directory containing train_regression_dataset.pkl and test_regression_dataset.pkl
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_path = os.path.join(dataset_dir, "train_regression_dataset.pkl")
    test_path = os.path.join(dataset_dir, "test_regression_dataset.pkl")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found in {dataset_dir}")
    
    print(f"Loading training dataset from {train_path}")
    with open(train_path, "rb") as f:
        train_dataset = pickle.load(f)
    
    print(f"Loading test dataset from {test_path}")
    with open(test_path, "rb") as f:
        test_dataset = pickle.load(f)
    
    return train_dataset, test_dataset

def prepare_tensors(dataset, device="cpu"):
    """
    Convert dataset into feature and label tensors
    
    Args:
        dataset (list): List of (hidden_state, relative_position) tuples
        device (str): Device to load tensors onto ('cpu' or 'cuda')
        
    Returns:
        tuple: (H, y) where H contains hidden states and y contains relative positions
    """
    hidden_states = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    
    H = torch.tensor(np.array(hidden_states), dtype=torch.float32, device=device)
    y = torch.tensor(np.array(labels), dtype=torch.float32, device=device).reshape(-1, 1)
    
    return H, y

def fit_linear_regression(H, y, epsilon=1e-10):
    """
    Compute closed-form solution for linear regression weights
    
    Args:
        H (torch.Tensor): Input features (hidden states)
        y (torch.Tensor): Target values (relative positions)
        epsilon (float): Small regularization term to ensure invertibility
        
    Returns:
        torch.Tensor: Computed weights
    """
    H_transpose = H.t()
    # Add small regularization to ensure numerical stability
    reg_term = epsilon * torch.eye(H.shape[1], device=H.device)
    # Solve the normal equations: w = (H^T H + ε*I)^(-1) H^T y
    w = torch.linalg.solve(H_transpose @ H + reg_term, H_transpose @ y).squeeze()
    
    return w

def predict(H, w):
    """
    Make predictions using trained weights
    
    Args:
        H (torch.Tensor): Input features
        w (torch.Tensor): Trained weights
        
    Returns:
        torch.Tensor: Predictions
    """
    return H @ w

def evaluate(predictions, labels):
    """
    Evaluate the model's performance
    
    Args:
        predictions (torch.Tensor): Predicted values
        labels (torch.Tensor): True values
        
    Returns:
        float: Mean squared error
    """
    predictions_np = predictions.cpu()
    labels_np = labels.cpu().squeeze()
    criterion = nn.MSELoss()

    mse = criterion(labels_np, predictions_np)
    
    return mse

def plot_results(predictions, labels, output_dir):
    """
    Create visualization plots for predictions vs true labels
    
    Args:
        predictions (torch.Tensor): Predicted values
        labels (torch.Tensor): True values
        output_dir (str): Directory to save plots
    """
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
    sample_indices = np.linspace(0, len(plot_labels)-1, sample_size, dtype=int)
    
    plt.scatter(np.arange(len(plot_preds[sample_indices])), plot_preds[sample_indices], 
                alpha=0.5, s=10, label="Predictions vs True")
    
    # Exponential smoothing
    alpha = 0.1  # smoothing factor
    smoothed_preds = np.zeros_like(plot_preds[sample_indices])
    smoothed_preds[0] = plot_preds[sample_indices][0]
    for t in range(1, len(plot_preds[sample_indices])):
        smoothed_preds[t] = alpha * plot_preds[sample_indices][t] + (1 - alpha) * smoothed_preds[t - 1]

    # Add diagonal line for perfect predictions
    plt.plot(np.arange(len(plot_labels[sample_indices])), plot_labels[sample_indices], color="orange", label="Perfect Predictions", alpha=0.6)
    plt.plot(smoothed_preds, color="blue", linewidth=2, label='Smoothed Predictions')
    
    plt.xlabel("Decoding Step")
    plt.ylabel("Predicted Relative Position")
    plt.title("TPV Predictions vs True Relative Positions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plot_path = os.path.join(output_dir, "tpv_predictions_vs_true.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset, test_dataset = load_datasets(args.input_dir)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Prepare tensors
    print("Preparing tensors...")
    H_train, y_train = prepare_tensors(train_dataset, device)
    H_test, y_test = prepare_tensors(test_dataset, device)
    
    print(f"Hidden state dimension: {H_train.shape[1]}")
    
    # Train linear regressor
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
    
    # Create visualization plots
    print("Creating visualization plots...")
    plot_results(test_preds, y_test, args.output_dir)
    
    # Retrain on combined train and test data
    print("Retraining on combined train and test data...")
    # Combine datasets
    H_combined = torch.cat([H_train, H_test], dim=0)
    y_combined = torch.cat([y_train, y_test], dim=0)
    
    # Retrain on the combined dataset
    combined_weights = fit_linear_regression(H_combined, y_combined, epsilon=args.epsilon)
    
    # Save the combined model weights as numpy array
    weights_np_path = os.path.join(args.output_dir, "tpv_linear_weights.npy")
    np.save(weights_np_path, combined_weights.cpu().numpy())
    print(f"Combined model weights saved to {weights_np_path} as numpy array")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a token progress value (TPV) regressor using linear regression"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="llama_math_tpv_dataset",
        help="Directory containing the processed train and test dataset .pkl files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tpv_model",
        help="Directory to save the trained model weights and plots"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
        help="Regularization parameter for numerical stability"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for training (cpu or cuda)"
    )
    
    args = parser.parse_args()
    main(args)
