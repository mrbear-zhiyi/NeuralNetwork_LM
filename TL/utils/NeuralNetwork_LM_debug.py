import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings


def comprehensive_data_check(x, y, model):
    print("======================================== comprehensive data check ========================================\n")

    print("Input data analysis:")
    for i, name in enumerate(['Z', 'A', 'Q_alpha']):
        print(f"  {name}: min={x[:, i].min():.3f}, max={x[:, i].max():.3f}, "
              f"mean={x[:, i].mean():.3f}, std={x[:, i].std():.3f}")

    print(f"Target: min={y.min():.3f}, max={y.max():.3f}, "
          f"mean={y.mean():.3f}, std={y.std():.3f}")

    with torch.no_grad():
        initial_output = model(x)
        initial_loss = nn.MSELoss()(initial_output, y)
        print(f"Initial Model - Output range: [{initial_output.min():.3f}, {initial_output.max():.3f}]")
        print(f"Initial Model - Loss: {initial_loss.item():.6f}")

    print("Verify Data normalization:")
    print(f"  Feature mean -> 0: {torch.allclose(x.mean(dim=0), torch.zeros(3), atol=1e-3)}")
    print(f"  Feature standard deviation -> 1: {torch.allclose(x.std(dim=0), torch.ones(3), atol=0.1)}")

    unique_counts = [len(torch.unique(x[:, i])) for i in range(3)]
    print(f"Number of unique feature values: Z={unique_counts[0]}, A={unique_counts[1]}, Q_alpha={unique_counts[2]}")



def run_diagnostic_checks(dataset, model, optimizer, num_samples=3):
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    for x_batch, y_batch in train_loader:
        x_data, y_data = x_batch, y_batch
        break

    comprehensive_data_check(x_data, y_data, model)
    optimizer = optimizer(model, lambda_=0.001, max_iter=200)

    print("======================================== Jacobian matrix check ========================================\n")
    J = optimizer.compute_jacobian(x_data[:num_samples], y_data[:num_samples])
    print(f"Shape of Jacobian matrix: {J.shape}")
    print(f"Range of Jacobian matrix: [{J.min().item():.6f}, {J.max().item():.6f}]")
    print(f"Norm of Jacobian matrix: {J.norm().item():.6f}")

    for i in range(min(3, J.shape[0])):
        grad_norm = J[i].norm().item()
        print(f"Sample{i}Norm of grad: {grad_norm:.6f}")



# gradient descend verification
def verify_with_gradient_descent(model, x, y, epochs=20):
    print("\n======================================== gradient descend verification ======================================== \n")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    initial_loss = criterion(model(x), y).item()

    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    final_loss = criterion(model(x), y).item()
    print(f"Initial loss: {initial_loss:.6f} → Final loss: {final_loss:.6f}")
    print("✓ Model is prepared for train" if final_loss < initial_loss else "✗ Model need to be check")
    return final_loss


def test_simplified_problem(optimizer):
    print("======================================== test simplified problem ========================================\n")
    torch.manual_seed(42)  # 固定随机种子，保证结果可复现
    n_samples = 100
    x_simple = torch.randn(n_samples, 1)  # 输入特征（100个样本，1维）
    y_simple = 2 * x_simple + 1 + 0.1 * torch.randn(n_samples, 1)  # 带噪声的标签

    # Simple Linear Model（y = wx + b）
    simple_model = nn.Sequential(
        nn.Linear(1, 1)  # 1输入1输出的线性层
    )

    optimizer_instance = optimizer(simple_model.parameters(), lambda_=0.01)

    losses = []
    for epoch in range(10):
        loss = optimizer_instance.step(x_simple, y_simple, simple_model)
        losses.append(loss)
        print(f"Simple problem Epoch {epoch + 1}: Loss={loss:.6f}")

    if losses[-1] < losses[0]:
        print("✓ Optimizer correct")
    else:
        print("✗ Optimizer doesn't work")

