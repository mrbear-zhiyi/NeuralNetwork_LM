import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,Subset
import warnings
from utils.NeuralNetwork_LM_debug import comprehensive_data_check, test_simplified_problem,run_diagnostic_checks,verify_with_gradient_descent

warnings.filterwarnings('ignore')


# Train set class
class AlphaDecayDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


#Model class
class AlphaDecayNN(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=1, hidden_neurons=10):
        # super().__init__()
        # self.hidden = nn.Linear(input_dim, hidden_neurons)
        # self.output = nn.Linear(hidden_neurons, 1)
        # self.activation = nn.Tanh()

        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = nn.Tanh()

        # Actication function automatically concatenate
        self.layers = nn.Sequential()
        self.layers.add_module("hidden_0", nn.Linear(input_dim, hidden_neurons))
        self.layers.add_module("act_0", self.activation)

        for i in range(1, hidden_layers):
            self.layers.add_module(f"hidden_{i}", nn.Linear(hidden_neurons, hidden_neurons))
            self.layers.add_module(f"act_{i}", self.activation)

        self.layers.add_module("output", nn.Linear(hidden_neurons, 1))

    def forward(self, x):
        # x = self.activation(self.hidden(x))
        # x = self.output(x)
        return self.layers(x)


# optimizer for parameter
class LevenbergMarquardtOptimizer:
    def __init__(self, model, lambda_=0.01, max_iter=100, tol=1e-6):
        self.model = model
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        #lambda range specification
        self.min_lambda = 1e-8
        self.max_lambda = 1e6

    def compute_jacobian(self, x, y):
        batch_size = x.shape[0]
        num_params = sum(p.numel() for p in self.model.parameters())
        jacobian = torch.zeros(batch_size, num_params, device=x.device)                                         # (batch_size, num_params)

        x.requires_grad_(True)
        outputs = self.model(x)

        for i in range(batch_size):
            self.model.zero_grad()
            grads = []
            for param in self.model.parameters():
                if param.requires_grad:

                    #core code: Perform gradient descent for each parameter in one batch
                    grad = torch.autograd.grad(outputs[i], param, retain_graph=True, allow_unused=True)[0]
                    grad = grad if grad is not None else torch.zeros_like(param)
                    grads.append(grad.view(-1))                                                                 # (num_params, )
                else:
                    grads.append(torch.zeros_like(param).view(-1))
            jacobian[i] = torch.cat(grads)                                                                      # (num_params, )

        x.requires_grad_(False)
        return jacobian                                                                                         # (batch_size, num_params)

    def step(self, x, y):
        # Parameter initialisation
        params = list(self.model.parameters())
        flat_params = torch.cat([p.data.view(-1) for p in params])
        num_params = len(flat_params)
        y = y.unsqueeze(1) if y.dim() == 1 else y                                                               # From(batch_size, )to(batch_size, 1)

        # Compute current error
        output = self.model(x)
        error = torch.clamp((output - y).squeeze(), -10.0, 10.0)  # 误差裁剪                                # (batch_size,)
        current_loss = error.pow(2).mean().item()


        # Compute LM matrices-----objective:calculate delta
        J = self.compute_jacobian(x, y)                                                                         # (batch_size, num_params)
        if torch.norm(J, dim=1).mean() < 1e-10:
            self.lambda_ = min(self.lambda_ * 10, self.max_lambda)
            return float('inf')

        JTJ = torch.matmul(J.t(), J)                                                                            # (batch_size, num_params)
        JTJ_plus_lambdaI = JTJ + min(self.lambda_, self.max_lambda) * torch.eye(num_params, device=x.device)    # (batch_size, num_params)

        #positive definitization of a singular matrix
        try:
            U, S, V = torch.svd(JTJ_plus_lambdaI)
            if S.min() < 1e-10:
                raise ValueError("matrix is nearly singular")
            inv_matrix = U @ torch.diag(1.0 / S) @ V.t()
        except Exception:
            self.lambda_ = min(self.lambda_ * 10, self.max_lambda)
            return current_loss

        delta = -torch.matmul(inv_matrix, torch.matmul(J.t(), error.unsqueeze(1))).squeeze()                    # (num_params, 1)
        delta_norm = torch.norm(delta)
        if delta_norm > 1.0:
            delta = delta * (1.0 / delta_norm)


        # Update parameters and indices
        with torch.no_grad():
            new_params = flat_params + delta
            start_idx = 0
            for param in params:
                end_idx = start_idx + param.numel()
                param.data = new_params[start_idx:end_idx].view(param.shape)
                start_idx = end_idx

        # Update lambda
        new_loss = nn.MSELoss()(self.model(x), y).item()
        improvement = current_loss - new_loss
        if improvement > 0:
            improvement_ratio = improvement / current_loss
            self.lambda_ = max(self.lambda_ / (10 if improvement_ratio > 0.05 else 2), self.min_lambda)
        else:
            self.lambda_ = min(self.lambda_ * 5, self.max_lambda)

        return new_loss


def hyperparameter_tuning(file_path, k_folds=9, max_epochs=200):

    features_norm, targets_norm, f_mean, f_std, t_mean, t_std,features ,targets = load_and_preprocess_data(file_path)
    dataset = AlphaDecayDataset(features_norm, targets_norm)
    total_samples = len(dataset)
    fold_size = total_samples // k_folds

    # Custom grid for grid search
    search_space = {
        "hidden_layers": [3,5,7,9],
        "hidden_neurons": [10, 21, 32],
        "lambda_init": [0.001]
    }

    tuning_results = []

    # Grid search hyperparameters
    for n_layers in search_space["hidden_layers"]:
        for n_neurons in search_space["hidden_neurons"]:
            for lambda_init in search_space["lambda_init"]:
                print(f"\n======================================== Tuning: Layers={n_layers}, Neurons={n_neurons}, Lambda={lambda_init} =========================================")
                fold_rmse = []

                # 9-KFolds cross varidation
                for fold in range(k_folds):
                    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

                    val_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
                    train_indices = [i for i in range(total_samples) if i not in val_indices]
                    train_subset = Subset(dataset, train_indices)
                    val_subset = Subset(dataset, val_indices)
                    train_loader = DataLoader(train_subset, batch_size=len(train_subset), shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)

                    x_train, y_train = next(iter(train_loader))
                    x_val, y_val = next(iter(val_loader))

                    model = AlphaDecayNN(input_dim=3, hidden_layers=n_layers, hidden_neurons=n_neurons)
                    lm_optimizer = LevenbergMarquardtOptimizer(model, lambda_=lambda_init, max_iter=max_epochs)

                    # Begin Train
                    train_losses = []
                    for epoch in range(max_epochs):
                        loss = lm_optimizer.step(x_train, y_train)
                        train_losses.append(loss)
                        if (epoch + 1) % 40 == 0:
                            print(f"  Epoch {epoch + 1:3d}/{max_epochs} | Train Loss: {loss:.6f} | Lambda: {lm_optimizer.lambda_:.6f}")

                    model.eval()
                    with torch.no_grad():

                        y_val_pred_norm = model(x_val)
                        y_val_pred = y_val_pred_norm.numpy().flatten() * t_std + t_mean
                        y_val_true = y_val.numpy().flatten() * t_std + t_mean

                        rmse = np.sqrt(np.mean((y_val_true - y_val_pred) ** 2))
                        fold_rmse.append(rmse)
                        print(f"  Fold {fold + 1} Val RMSE: {rmse:.4f}")

                #Output for each combination
                avg_rmse = np.mean(fold_rmse)
                std_rmse = np.std(fold_rmse)
                tuning_results.append({
                    "hidden_layers": n_layers,
                    "hidden_neurons": n_neurons,
                    "lambda_init": lambda_init,
                    "avg_val_rmse": avg_rmse,
                    "std_val_rmse": std_rmse,
                })
                print(f"\n======================================== Tuning Result: Layers={n_layers}, Neurons={n_neurons} =========================================")
                print(f"Average Val RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")

    # Output
    results_df = pd.DataFrame(tuning_results)
    best_idx = results_df["avg_val_rmse"].idxmin()
    best_hparams = results_df.iloc[best_idx].to_dict()

    print("\n======================================== Final Tuning Results =========================================")
    print("All hyperparameter combinations:")
    print(results_df.round(4))
    print(f"\nOptimal Hyperparameters:")
    print(f"- Hidden Layers: {best_hparams['hidden_layers']}")
    print(f"- Hidden Neurons per Layer: {best_hparams['hidden_neurons']}")
    print(f"- Initial Lambda: {best_hparams['lambda_init']}")
    print(f"- Average Val RMSE: {best_hparams['avg_val_rmse']:.4f} ± {best_hparams['std_val_rmse']:.4f}")

    results_df.to_csv('./results/02_NeuralNetwork/Gridsearch_hyperparameters.csv', index=False)
    print("\nGrid search hyperparameters RMSE：'./results/02_NeuralNetwork/Gridsearch_hyperparameters.csv'\n")

    return best_hparams, results_df


#get_data & standization
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    features = np.array([[row['Z'], row['A'], row['Q_alpha']] for _, row in df.iterrows()])
    targets = np.array([np.log10(row['half_life_alpha']) for _, row in df.iterrows()])
    print(f"Total number of train set : {len(features)}")

    print("\n======================================== standardization =========================================\n")
    feature_mean, feature_std = features.mean(axis=0), features.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-8)
    target_mean, target_std = targets.mean(), targets.std()
    features_norm = (features - feature_mean) / feature_std
    targets_norm = (targets - target_mean) / target_std

    return features_norm, targets_norm, feature_mean, feature_std, target_mean, target_std, features, targets


def extract_model_parameters(model):
    params = {}
    linear_layer_count = 0
    for layer_name, layer in model.layers.named_modules():
        if isinstance(layer, nn.Linear):
            linear_layer_count += 1
            if "output" in layer_name:
                params["W_out"] = layer.weight.detach().numpy()
                params["b_out"] = layer.bias.detach().numpy()
            else:
                params[f"W{linear_layer_count}"] = layer.weight.detach().numpy()
                params[f"b{linear_layer_count}"] = layer.bias.detach().numpy()
    return params


# Train function
def NeuralNetwork_LM_train(file_path,best_hparams, max_epochs=200):
    torch.manual_seed(42)
    np.random.seed(42)


    print("======================================== Data preprocessing for NeuralNetwork_LM_train ========================================\n")
    features_norm, targets_norm, f_mean, f_std, t_mean, t_std, _, targets = load_and_preprocess_data(file_path)
    dataset = AlphaDecayDataset(features_norm, targets_norm)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    x_data, y_data = next(iter(dataloader))

    #set_parameter
    input_dim = 3
    hidden_neurons = 10
    model = AlphaDecayNN(input_dim, hidden_neurons)
    print(f"model structure: Input layer({input_dim}) -> Hidden layer({hidden_neurons}, tanh) -> Output layer(1)")


    # print("======================================== NeuralNetwork_LM debug ========================================\n")
    # run_diagnostic_checks(dataset,model,LevenbergMarquardtOptimizer)
    # verify_with_gradient_descent(model, x_data, y_data)
    # test_simplified_problem(LevenbergMarquardtOptimizer)



    print("======================================== Begin NeuralNetwork_LM_train for 200 epoch ========================================\n")
    model = AlphaDecayNN(
        input_dim=3,
        hidden_layers=best_hparams["hidden_layers"],
        hidden_neurons=best_hparams["hidden_neurons"]
    )
    lm_optimizer = LevenbergMarquardtOptimizer(model, lambda_=0.001, max_iter=200)
    train_losses = []

    for epoch in range(max_epochs):
        loss = lm_optimizer.step(x_data, y_data)
        train_losses.append(loss)
        if (epoch + 1) % 40 == 0:
            print(f"epoch {epoch + 1:3d}/200 | loss: {loss:.6f} | Lambda: {lm_optimizer.lambda_:.6f}")



    print("======================================== NeuralNetwork_LM evaluation ========================================\n")
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(torch.FloatTensor(features_norm))
        #Denormalize
        y_pred = y_pred_norm.numpy().flatten() * t_std + t_mean

    rmse = np.sqrt(np.mean((targets - y_pred) ** 2))
    r2 = 1 - (np.sum((targets - y_pred) ** 2) / np.sum((targets - targets.mean()) ** 2))
    print(f"RMSE: {rmse:.4f} | R²: {r2:.4f}")




    print("\n======================================== Best model NeuralNetwork paramters（W/b） =========================================")
    best_model_params = extract_model_parameters(model)
    for param_name, param_value in best_model_params.items():
        print(f"\n{param_name} (Shape: {param_value.shape}):")
        print(param_value.round(6))

    target_dir = "./result/02_NeuralNetwork"
    file_path = os.path.join(target_dir, "best_model_parameters.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Best model NeuralNetwork paramters（W/b）\n")
        f.write(f"hyperparamters：hidden_layers={best_hparams['hidden_layers']}, hidden_neurons={best_hparams['hidden_neurons']}\n\n")
        for param_name, param_value in best_model_params.items():
            f.write(f"{param_name} (Shape: {param_value.shape}):\n")
            f.write(f"{param_value.round(6)}\n\n")
    print(f"\nBest model NeuralNetwork paramters（W/b） save as：{file_path} ")




    print("======================================== NeuralNetwork_LM saving ========================================\n")
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_mean': f_mean, 'feature_std': f_std,
        'target_mean': t_mean, 'target_std': t_std,
        'hidden_neurons': 10
    }, 'alpha_decay_model.pth')
    print(f"Model save as： 'alpha_decay_model.pth'")


if __name__ == "__main__":
    NeuralNetwork_LM_train(file_path="./results/01_preprocess/nuclide_data_modified.csv")