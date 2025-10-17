import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# 1. 数据集类（精简注释，保留核心逻辑）
class AlphaDecayDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 2. 神经网络模型（α衰变预测专用，结构不变）
class AlphaDecayNN(nn.Module):
    def __init__(self, input_dim=3, hidden_neurons=10):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.output(x)


# 3. LM优化器（删除旧注释代码，保留稳定核心逻辑）
class LevenbergMarquardtOptimizer:
    def __init__(self, model, lambda_=0.01, max_iter=100, tol=1e-6):
        self.model = model
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.min_lambda = 1e-8  # 避免lambda过小
        self.max_lambda = 1e6  # 避免lambda过大

    def compute_jacobian(self, x, y):
        batch_size = x.shape[0]
        num_params = sum(p.numel() for p in self.model.parameters())
        jacobian = torch.zeros(batch_size, num_params, device=x.device)

        x.requires_grad_(True)
        outputs = self.model(x)

        for i in range(batch_size):
            self.model.zero_grad()
            grads = []
            for param in self.model.parameters():
                if param.requires_grad:
                    grad = torch.autograd.grad(outputs[i], param, retain_graph=True, allow_unused=True)[0]
                    grad = grad if grad is not None else torch.zeros_like(param)
                    grads.append(grad.view(-1))
                else:
                    grads.append(torch.zeros_like(param).view(-1))
            jacobian[i] = torch.cat(grads)

        x.requires_grad_(False)
        return jacobian

    def step(self, x, y):
        # 获取参数并处理目标维度
        params = list(self.model.parameters())
        flat_params = torch.cat([p.data.view(-1) for p in params])
        num_params = len(flat_params)
        y = y.unsqueeze(1) if y.dim() == 1 else y

        # 计算当前误差
        output = self.model(x)
        error = torch.clamp((output - y).squeeze(), -10.0, 10.0)  # 误差裁剪
        current_loss = error.pow(2).mean().item()

        # 计算雅可比与LM矩阵
        J = self.compute_jacobian(x, y)
        if torch.norm(J, dim=1).mean() < 1e-10:
            self.lambda_ = min(self.lambda_ * 10, self.max_lambda)
            return float('inf')

        JTJ = torch.matmul(J.t(), J)
        JTJ_plus_lambdaI = JTJ + min(self.lambda_, self.max_lambda) * torch.eye(num_params, device=x.device)

        # SVD求解稳定逆矩阵
        try:
            U, S, V = torch.svd(JTJ_plus_lambdaI)
            if S.min() < 1e-10:
                raise ValueError("矩阵接近奇异")
            inv_matrix = U @ torch.diag(1.0 / S) @ V.t()
        except Exception:
            self.lambda_ = min(self.lambda_ * 10, self.max_lambda)
            return current_loss

        # 计算并裁剪更新量
        delta = -torch.matmul(inv_matrix, torch.matmul(J.t(), error.unsqueeze(1))).squeeze()
        delta_norm = torch.norm(delta)
        if delta_norm > 1.0:
            delta = delta * (1.0 / delta_norm)

        # 更新参数
        with torch.no_grad():
            new_params = flat_params + delta
            start_idx = 0
            for param in params:
                end_idx = start_idx + param.numel()
                param.data = new_params[start_idx:end_idx].view(param.shape)
                start_idx = end_idx

        # 调整lambda
        new_loss = nn.MSELoss()(self.model(x), y).item()
        improvement = current_loss - new_loss
        if improvement > 0:
            improvement_ratio = improvement / current_loss
            self.lambda_ = max(self.lambda_ / (10 if improvement_ratio > 0.05 else 2), self.min_lambda)
        else:
            self.lambda_ = min(self.lambda_ * 5, self.max_lambda)

        return new_loss


# 4. 数据加载与预处理（删除冗余变量，简化打印）
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # 提取特征（Z,A,Q_alpha）和目标（log10(半衰期)）
    features = np.array([[row['Z'], row['A'], row['Q_alpha']] for _, row in df.iterrows()])
    targets = np.array([np.log10(row['half_life_alpha']) for _, row in df.iterrows()])
    print(f"训练集数量: {len(features)}")

    # 标准化（修正过小标准差）
    feature_mean, feature_std = features.mean(axis=0), features.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-8)
    target_mean, target_std = targets.mean(), targets.std()

    features_norm = (features - feature_mean) / feature_std
    targets_norm = (targets - target_mean) / target_std

    return features_norm, targets_norm, feature_mean, feature_std, target_mean, target_std, features, targets


# 5. 评估指标（保留核心RMSE）
def calculate_rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# 6. 梯度下降验证（精简打印，验证模型可学习性）
def verify_with_gradient_descent(model, x, y, epochs=20):
    print("\n=== 梯度下降验证（确保模型可学习）===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    initial_loss = criterion(model(x), y).item()

    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    final_loss = criterion(model(x), y).item()
    print(f"初始损失: {initial_loss:.6f} → 最终损失: {final_loss:.6f}")
    print("✓ 模型可学习" if final_loss < initial_loss else "✗ 模型需检查")
    return final_loss


# 7. 核心训练函数（整合流程，简化打印）
def NeuralNetwork_LM_train(file_path):
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    print("=" * 50 + " α衰变模型训练 " + "=" * 50)

    # 1. 数据预处理
    features_norm, targets_norm, f_mean, f_std, t_mean, t_std, _, targets = load_and_preprocess_data(file_path)
    dataset = AlphaDecayDataset(features_norm, targets_norm)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    x_data, y_data = next(iter(dataloader))

    # 2. 初始化模型与验证
    model = AlphaDecayNN(input_dim=3, hidden_neurons=10)
    print(f"\n模型结构: 输入层(3) → 隐藏层(10, Tanh) → 输出层(1)")
    verify_with_gradient_descent(model, x_data, y_data)  # 验证模型可学习性

    # 3. LM优化器训练
    print("\n=== 开始LM训练（200轮）===")
    model = AlphaDecayNN()  # 重新初始化模型
    lm_optimizer = LevenbergMarquardtOptimizer(model, lambda_=0.001, max_iter=200)
    train_losses = []

    for epoch in range(200):
        loss = lm_optimizer.step(x_data, y_data)
        train_losses.append(loss)
        # 每40轮打印一次进度
        if (epoch + 1) % 40 == 0:
            print(f"轮次 {epoch + 1:3d}/200 | 损失: {loss:.6f} | Lambda: {lm_optimizer.lambda_:.6f}")

    # 4. 模型评估
    print("\n=== 模型评估 ===")
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(torch.FloatTensor(features_norm))
        y_pred = y_pred_norm.numpy().flatten() * t_std + t_mean  # 反标准化

    rmse = calculate_rms_error(targets, y_pred)
    r2 = 1 - (np.sum((targets - y_pred) ** 2) / np.sum((targets - targets.mean()) ** 2))
    print(f"RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # 5. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': f_mean, 'feature_std': f_std,
        'target_mean': t_mean, 'target_std': t_std,
        'hidden_neurons': 10
    }, 'alpha_decay_model.pth')
    print(f"\n模型已保存为: alpha_decay_model.pth")


# 运行训练（需传入你的CSV文件路径）
if __name__ == "__main__":
    NeuralNetwork_LM_train(file_path="./results/01_preprocess/nuclide_data_modified.csv")  # 替换为实际文件路径