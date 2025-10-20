import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def alpha_decay_fit(data: pd.DataFrame):
    df_alpha = data.copy()

    (Z, N, A, Q_alpha, half_life_alpha) = (
        df_alpha[col].to_numpy() for col in ["Z", "N", "A", "Q_alpha", "half_life_alpha"]
    )

    log10_hl_alpha = np.log10(half_life_alpha)
    #系数
    Z_alpha, A_alpha = 2., 4.
    miu = A_alpha * (A - A_alpha) / A
    x1 = Z_alpha * (Z - Z_alpha) * np.sqrt(miu / Q_alpha)
    x2 = np.sqrt(miu * Z_alpha * (Z - Z_alpha) * (A_alpha ** (1 / 3) + (A - A_alpha) ** (1 / 3)))
    X = np.column_stack((x1, x2))

    #fit
    def linear_model(x, a, b, c):
        return a * x[:, 0] + b * x[:, 1] + c

    params, _ = curve_fit(linear_model, X, log10_hl_alpha)
    a, b, c = params
    log10_hl_alpha_cal = linear_model(X, a, b, c)

    residual = log10_hl_alpha - log10_hl_alpha_cal
    rmse_alpha = np.sqrt(np.mean(residual ** 2))

    df_alpha['log10_T_exp'] = log10_hl_alpha
    df_alpha['log10_T_cal'] = log10_hl_alpha_cal
    df_alpha['residual'] = residual

    #Output
    parameters_df = pd.DataFrame({
        'parameter': ['type','a1', 'b1', 'c1', 'RMSE'],
        'fitted_value': ['fitted_value',a, b, c, rmse_alpha],
        'standard_value1': ['RF_value',0.407, -0.382, -23.896, 0.883],
        'standard_value2': ['UDL_value',0.4065, -0.4311, -20.7889, None] # 这里填入你的标准值
    })
    parameters_df.to_csv('./results/01_preprocess/fitting_parameters.csv', index=False)
    print('\n================================ parameters of Eq.(1) are obtained ================================\n')
    print(f'Fitting Results(alpha): a1={a:.6f}\tb1={b:.6f}\tc1={c:.6f}')
    print(f'Fitting residual(alpha): RMSE={rmse_alpha:.6f}')

    return df_alpha