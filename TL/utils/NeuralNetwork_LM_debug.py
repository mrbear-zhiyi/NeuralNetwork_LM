import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


class AlphaDecayDataset(Dataset):
    #Train set class
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class AlphaDecayNN(nn.Module):
    #Model class
    def __init__(self, input_dim, hidden_neurons):
        super(AlphaDecayNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons

        self.hidden = nn.Linear(input_dim, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x


class LevenbergMarquardtOptimizer:
    #optimizer for parameter
    def __init__(self, model, lambda_=0.01, max_iter=100, tol=1e-6):
        self.model = model
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol

    # def compute_jacobian(self, x, y):
    #     batch_size = x.shape[0]
    #     num_params = sum(p.numel() for p in self.model.parameters())
    #
    #     jacobian = torch.zeros(batch_size, num_params)                    # (batch_size, num_params)
    #
    #     for i in range(batch_size):
    #         self.model.zero_grad()
    #         output_i = self.model(x[i:i + 1])
    #         loss_i = nn.MSELoss()(output_i, y[i:i + 1])
    #         # 计算单个样本的梯度
    #         grad = torch.autograd.grad(loss_i, self.model.parameters(),
    #                                    retain_graph=True)
    #         flat_grad = torch.cat([g.view(-1) for g in grad])             # (num_params, )
    #         jacobian[i] = flat_grad                                       # (num_params, )
    #
    #     return jacobian                                                   # (num_params, )

    def compute_jacobian(self, x, y):
        """完全重写的雅可比计算函数"""
        batch_size = x.shape[0]
        num_params = sum(p.numel() for p in self.model.parameters())

        jacobian = torch.zeros(batch_size, num_params, device=x.device)

        # 启用输入的梯度计算
        x.requires_grad_(True)

        # 一次性计算所有输出
        outputs = self.model(x)

        for i in range(batch_size):
            self.model.zero_grad()

            # 计算第i个样本的输出对参数的梯度
            output_i = outputs[i]

            # 手动计算每个参数的梯度
            grads = []
            for param in self.model.parameters():
                if param.requires_grad:
                    # 计算output_i对当前参数的梯度
                    grad = torch.autograd.grad(
                        output_i, param,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    if grad is not None:
                        grads.append(grad.contiguous().view(-1))
                    else:
                        grads.append(torch.zeros_like(param).view(-1))
                else:
                    grads.append(torch.zeros_like(param).view(-1))

            flat_grad = torch.cat(grads)
            jacobian[i] = flat_grad

        x.requires_grad_(False)
        return jacobian

    # def step(self, x, y):
    #     # 添加调试信息
    #     print(f"\nLambda: {self.lambda_}")
    #
    #     #Model initialization
    #     params = list(self.model.parameters())
    #     flat_params = torch.cat([p.data.view(-1) for p in params])
    #     num_params = len(flat_params)
    #
    #     #Error calculate
    #     if y.dim() == 1:
    #         y = y.unsqueeze(1)                                            # From(batch_size, )to(batch_size, 1)
    #     output = self.model(x)
    #     error = (output - y).view(-1)                                     # (batch_size,)
    #
    #
    #     # Delta calculate
    #     J = self.compute_jacobian(x, y)                                   # (batch_size, num_params)
    #     # LM: Δθ = -(J^T J + λI)^(-1) J^T e
    #     JTJ = torch.matmul(J.t(), J)                                      # (num_params, num_params)
    #     I = torch.eye(num_params)                                         # (num_params, num_params)
    #     JTJ_plus_lambdaI = JTJ + self.lambda_ * I                         # (num_params, num_params)
    #
    #     # 检查矩阵条件数
    #     cond_number = torch.linalg.cond(JTJ_plus_lambdaI)
    #     print(f"Condition number: {cond_number.item():.6f}")
    #
    #     if torch.det(JTJ_plus_lambdaI) == 0:
    #         print("Matrix is singular!")
    #         self.lambda_ *= 10
    #         return float('inf')
    #
    #     # 检查逆矩阵
    #     try:
    #         inv_matrix = torch.inverse(JTJ_plus_lambdaI)
    #         print(f"Inverse matrix range: [{inv_matrix.min().item():.6f}, {inv_matrix.max().item():.6f}]")
    #     except:
    #         print("Matrix inversion failed!")
    #         self.lambda_ *= 10
    #         return float('inf')
    #
    #     JT_error = torch.matmul(J.t(), error.unsqueeze(1))
    #     delta = -torch.matmul(inv_matrix, JT_error)
    #
    #     print(f"Delta range: [{delta.min().item():.6f}, {delta.max().item():.6f}]")
    #
    #     # # check inverible matrix
    #     # if torch.det(JTJ_plus_lambdaI) == 0:
    #     #     self.lambda_ *= 10
    #     #     return float('inf')
    #     # JT_error = torch.matmul(J.t(), error.unsqueeze(1))                # (num_params, 1)
    #     # delta = -torch.matmul(torch.inverse(JTJ_plus_lambdaI), JT_error)  # (num_params, 1)
    #
    #
    #     #Parameters, Parameter_Indices, Error Updates
    #     with torch.no_grad():
    #         new_params = flat_params + delta.squeeze()  # (num_params,)
    #         start_idx = 0
    #         for param in params:
    #             end_idx = start_idx + param.numel()
    #             param.data = new_params[start_idx:end_idx].view(param.shape)
    #             start_idx = end_idx
    #     new_output = self.model(x)
    #     new_error = nn.MSELoss()(new_output, y)
    #
    #     # parameter optimization
    #     if new_error < error.pow(2).mean():
    #         self.lambda_ /= 10
    #     else:
    #         self.lambda_ *= 10
    #
    #     return new_error.item()

    def step(self, x, y):
        params = list(self.model.parameters())
        flat_params = torch.cat([p.data.view(-1) for p in params])
        num_params = len(flat_params)

        # Error calculate
        if y.dim() == 1:
            y = y.unsqueeze(1)
        output = self.model(x)
        error = (output - y).squeeze()

        # 误差裁剪
        error = torch.clamp(error, -10.0, 10.0)

        # Delta calculate
        J = self.compute_jacobian(x, y)

        # 检查雅可比矩阵质量
        jacobian_norm = torch.norm(J, dim=1).mean()
        print(f"Jacobian norm: {jacobian_norm.item():.6f}")

        if jacobian_norm < 1e-10:
            print("警告: 雅可比矩阵范数过小")
            return float('inf')

        # 改进的LM计算
        JTJ = torch.matmul(J.t(), J)
        I = torch.eye(num_params, device=JTJ.device)

        # 动态调整lambda，避免过大
        max_lambda = 1e6  # 设置上限
        current_lambda = min(self.lambda_, max_lambda)

        JTJ_plus_lambdaI = JTJ + current_lambda * I

        # 改进的矩阵可逆性检查
        try:
            # 使用SVD提高稳定性
            U, S, V = torch.svd(JTJ_plus_lambdaI)

            # 检查奇异值
            min_singular = S.min()
            max_singular = S.max()
            cond_number = max_singular / min_singular

            print(f"\nSVD - Min singular: {min_singular:.6f}, Max singular: {max_singular:.6f}")
            print(f"Condition number: {cond_number:.6f}")

            if min_singular < 1e-10:
                print("矩阵接近奇异，增大lambda")
                self.lambda_ = min(self.lambda_ * 10, max_lambda)
                return float('inf')

            # 稳定的求逆
            S_inv = 1.0 / S
            inv_matrix = U @ torch.diag(S_inv) @ V.t()

        except Exception as e:
            print(f"SVD失败: {e}")
            self.lambda_ = min(self.lambda_ * 10, max_lambda)
            return float('inf')

        JT_error = torch.matmul(J.t(), error.unsqueeze(1))
        delta = -torch.matmul(inv_matrix, JT_error)

        # 限制更新幅度
        delta_norm = torch.norm(delta)
        max_delta_norm = 1.0  # 最大更新范数
        if delta_norm > max_delta_norm:
            delta = delta * (max_delta_norm / delta_norm)
            print(f"裁剪更新量: {delta_norm.item():.6f} -> {max_delta_norm}")

        # 参数更新
        with torch.no_grad():
            new_params = flat_params + delta.squeeze()
            # 关键：将new_params按参数形状拆分并赋值给模型
            start_idx = 0
            for param in params:  # params是list(self.model.parameters())
                end_idx = start_idx + param.numel()  # 计算当前参数的元素总数
                param.data = new_params[start_idx:end_idx].view(param.shape)  # 按形状赋值
                start_idx = end_idx  # 更新索引，准备下一个参数

        # 改进的lambda调整策略
        new_output = self.model(x)
        new_error = nn.MSELoss()(new_output, y)

        improvement = error.pow(2).mean() - new_error

        # 改进的lambda调整
        if improvement > 0:
            # 显著改进（损失下降超5%）：更大幅度减小lambda
            improvement_ratio = improvement / error.pow(2).mean()
            if improvement_ratio > 0.05:
                self.lambda_ = max(self.lambda_ / 10, 1e-8)  # 最小lambda=1e-8，避免过小
            else:
                self.lambda_ = max(self.lambda_ / 2, 1e-8)
        else:
            # 损失未下降：增大lambda，但限制最大倍数（避免瞬间过大）
            self.lambda_ = min(self.lambda_ * 5, 1e6)  # 增大倍数从10→5，减缓增长

        return new_error.item()


def load_and_preprocess_data(file_path):
    #get_data & standization
    df = pd.read_csv(file_path)

    features = []
    targets = []

    for _, row in df.iterrows():
        # features
        Z = row['Z']
        A = row['A']
        Q_alpha = row['Q_alpha']

        # target
        half_life_alpha = row['half_life_alpha']
        log_T = np.log10(half_life_alpha)
        features.append([Z, A, Q_alpha])
        targets.append(log_T)

    # output
    features = np.array(features)
    targets = np.array(targets)
    print(f"trian set: {len(features)}")

    #standization???!!!
    print("\n======================================== standardization =========================================")
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    target_mean = targets.mean()
    target_std = targets.std()
    # 检查并修正过小的标准差
    feature_stds = np.maximum(feature_std, 1e-8)
    features_normalized = (features - feature_mean) / feature_std
    targets_normalized = (targets - target_mean) / target_std

    return (features_normalized, targets_normalized,
            feature_mean, feature_std, target_mean, target_std,
            features, targets)

def calculate_rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def NeuralNetwork_LM_train_debug(file_path):
    torch.manual_seed(42)
    np.random.seed(42)
    print("======================================== Data preprocessing for NeuralNetwork_LM_train ========================================\n")

    #preprocessing
    (features_normalized, targets_normalized,
     feature_mean, feature_std, target_mean, target_std,
     features_original, targets_original) = load_and_preprocess_data(file_path)
    dataset = AlphaDecayDataset(features_normalized, targets_normalized)

    #set_parameter
    input_dim = 3
    hidden_neurons = 10

    print("======================================== Begin NeuralNetwork_LM_train ========================================\n")
    model = AlphaDecayNN(input_dim, hidden_neurons)
    print(f"model structure: Input layer({input_dim}) -> Hidden layer({hidden_neurons}, tanh) -> Output layer(1)")
    # train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    # ==================== 添加诊断步骤 ====================
    print("\n" + "=" * 50 + " 诊断阶段 " + "=" * 50)

    # 准备数据
    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    for x_batch, y_batch in train_loader:
        x_data, y_data = x_batch, y_batch
        break

    # 1. 全面数据检查
    comprehensive_data_check(x_data, y_data, model)

    # ==================== 添加雅可比验证 ====================
    print("\n" + "=" * 50 + " 雅可比矩阵验证 " + "=" * 50)

    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    for x_batch, y_batch in train_loader:
        x_data, y_data = x_batch, y_batch
        break

    optimizer = LevenbergMarquardtOptimizer(model, lambda_=0.001, max_iter=200)

    # 验证雅可比计算
    J = optimizer.compute_jacobian(x_data[:3], y_data[:3])  # 只计算前3个样本
    print(f"雅可比矩阵形状: {J.shape}")
    print(f"雅可比矩阵范围: [{J.min().item():.6f}, {J.max().item():.6f}]")
    print(f"雅可比矩阵范数: {J.norm().item():.6f}")

    # 检查梯度多样性
    for i in range(min(3, J.shape[0])):
        grad_norm = J[i].norm().item()
        print(f"样本{i}梯度范数: {grad_norm:.6f}")

    print("\n" + "=" * 50 + " 开始LM训练 " + "=" * 50)

    # 2. 梯度下降验证
    gd_losses = verify_with_gradient_descent(model, x_data, y_data, epochs=20)

    # 3. 简化问题测试
    test_simplified_problem()

    print("\n" + "=" * 50 + " 开始LM训练 " + "=" * 50)
    # ==================== 诊断结束 ====================

    # 重新初始化模型（因为梯度下降改变了参数）
    model = AlphaDecayNN(input_dim, hidden_neurons)

    train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # Create LM optimizer
    optimizer = LevenbergMarquardtOptimizer(model, lambda_=0.001, max_iter=200)
    train_losses = []

    #200 training epochs
    for epoch in range(200):
        for x_batch, y_batch in train_loader:
            loss = optimizer.step(x_batch, y_batch)
            train_losses.append(loss)
        if (epoch + 1) % 20 == 0:
            print(f"  epoch: {epoch + 1}/200, loss: {loss:.6f}")

    print("======================================== NeuralNetwork_LM evaluation ========================================\n")
    model.eval()

    with torch.no_grad():
        y_pred_norm = model(torch.FloatTensor(features_normalized))
        y_pred = y_pred_norm.numpy().flatten() * target_std + target_mean
        y_true = targets_original

    # Regression Metrics回归指标
    rms_error = calculate_rms_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"RMSE: {rms_error:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r_squared:.4f}")

    print("======================================== NeuralNetwork_LM saving ========================================\n")
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'hidden_neurons': hidden_neurons
    }, 'alpha_decay_model.pth')

    # print("======================================== NeuralNetwork_LM prediction ========================================\n")
    # print("Origin\t\tPrediction\t\tResidual")
    # for i in range(min(10, len(y_true))):
    #     print(f"{y_true[i]:.4f}\t\t{y_pred[i]:.4f}\t\t{y_true[i] - y_pred[i]:.4f}")

    print("======================================== End NeuralNetwork_LM train ========================================\n")
    print(f"Model save as： 'alpha_decay_model.pth'")


# def predict_new_nuclide(model_path, Z, A, Q_alpha):
#     checkpoint = torch.load(model_path)
#
#     model = AlphaDecayNN(input_dim=3, hidden_neurons=checkpoint['hidden_neurons'])
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     # input data
#     features = np.array([[Z, A, Q_alpha]])
#     features_normalized = (features - checkpoint['feature_mean']) / checkpoint['feature_std']
#
#     # prediction
#     with torch.no_grad():
#         prediction_normalized = model(torch.FloatTensor(features_normalized))
#         prediction = prediction_normalized.numpy().flatten()[0] * checkpoint['target_std'] + checkpoint['target_mean']
#
#     return prediction
#     # print("\n=== prediction ===")
#     # predicted_logT = predict_new_nuclide('alpha_decay_model.pth', Z=92, A=238, Q_alpha=4.27)
#     # print(f"U-238_prediction: log10(T) = {predicted_logT:.4f}")

def comprehensive_data_check(x, y, model):
    """全面数据检查函数"""
    print("=== 全面数据检查 ===")

    # 1. 检查输入数据
    print("输入数据统计:")
    for i, name in enumerate(['Z', 'A', 'Q_alpha']):
        print(f"  {name}: min={x[:, i].min():.3f}, max={x[:, i].max():.3f}, "
              f"mean={x[:, i].mean():.3f}, std={x[:, i].std():.3f}")

    # 2. 检查目标数据
    print(f"目标数据: min={y.min():.3f}, max={y.max():.3f}, "
          f"mean={y.mean():.3f}, std={y.std():.3f}")

    # 3. 检查模型初始输出
    with torch.no_grad():
        initial_output = model(x)
        initial_loss = nn.MSELoss()(initial_output, y)
        print(f"初始模型 - 输出范围: [{initial_output.min():.3f}, {initial_output.max():.3f}]")
        print(f"初始模型 - 损失: {initial_loss.item():.6f}")

    # 4. 检查数据标准化
    print("数据标准化验证:")
    print(f"  特征均值是否接近0: {torch.allclose(x.mean(dim=0), torch.zeros(3), atol=1e-3)}")
    print(f"  特征标准差是否接近1: {torch.allclose(x.std(dim=0), torch.ones(3), atol=0.1)}")

    # 5. 检查数据多样性
    unique_counts = [len(torch.unique(x[:, i])) for i in range(3)]
    print(f"特征唯一值数量: Z={unique_counts[0]}, A={unique_counts[1]}, Q_alpha={unique_counts[2]}")


def verify_with_gradient_descent(model, x, y, epochs=50):
    """用梯度下降验证模型是否能学习"""
    print("=== 梯度下降验证 ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # 检查梯度
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()

        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"GD Epoch {epoch + 1}: Loss={loss.item():.6f}, GradNorm={total_grad_norm:.6f}")

    # 绘制损失曲线
    if len(losses) > 1:
        print(f"梯度下降: 初始损失={losses[0]:.6f}, 最终损失={losses[-1]:.6f}")
        if losses[-1] < losses[0]:
            print("✓ 模型能够学习")
        else:
            print("✗ 模型无法学习")

    return losses


def test_simplified_problem():
    """用简化问题测试LM算法"""
    print("=== 简化问题测试 ===")

    # 创建简单的线性回归问题
    torch.manual_seed(42)
    n_samples = 100
    x_simple = torch.randn(n_samples, 1)
    y_simple = 2 * x_simple + 1 + 0.1 * torch.randn(n_samples, 1)

    # 简单线性模型
    simple_model = nn.Sequential(
        nn.Linear(1, 1)
    )

    # 使用LM优化器
    simple_optimizer = LevenbergMarquardtOptimizer(simple_model, lambda_=0.01)

    losses = []
    for epoch in range(10):
        loss = simple_optimizer.step(x_simple, y_simple)
        losses.append(loss)
        print(f"简单问题 Epoch {epoch + 1}: Loss={loss:.6f}")

    if losses[-1] < losses[0]:
        print("✓ LM算法在简单问题上工作正常")
    else:
        print("✗ LM算法本身有问题")


def debug_compute_jacobian(self, x, y):
    """调试版本的雅可比计算"""
    batch_size = x.shape[0]
    num_params = sum(p.numel() for p in self.model.parameters())

    jacobian = torch.zeros(batch_size, num_params, device=x.device)

    print("=== 雅可比矩阵调试 ===")

    for i in range(min(3, batch_size)):  # 只检查前3个样本
        self.model.zero_grad()
        x_i = x[i:i + 1]
        y_i = y[i:i + 1]

        output_i = self.model(x_i)
        loss_i = nn.MSELoss()(output_i, y_i)

        print(f"样本{i}: 输入{x_i}, 目标{y_i}, 输出{output_i}, 损失{loss_i}")

        grad = torch.autograd.grad(loss_i, self.model.parameters(),
                                   retain_graph=True, create_graph=False)

        # 检查每个参数的梯度
        for j, (param, g) in enumerate(zip(self.model.parameters(), grad)):
            print(f"  参数{j}: 形状{g.shape}, 范围[{g.min():.6f}, {g.max():.6f}]")

        flat_grad = torch.cat([g.view(-1) for g in grad])
        jacobian[i] = flat_grad

    return jacobian