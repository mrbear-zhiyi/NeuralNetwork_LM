from scipy.stats import energy_distance
from utils.get_data import get_data, unit_unified,Data_modified
from utils.alpha_linear_fit import alpha_decay_fit
from utils.NeuralNetwork_LM import  NeuralNetwork_LM_train, hyperparameter_tuning
import pandas as pd


# 主函数
def main(path_json):

    #get_data & preprocessing
    data_NNDC = get_data(path_json)
    data_unit_unified = unit_unified(data_NNDC)
    data_modified = Data_modified(data_unit_unified)
    print('======================================== data is obtained =========================================\n')
    data_NNDC.to_csv('./results/01_preprocess/nuclide_data_origin_NNDC.csv')
    print(f'Original data is stored in: ./results/01_preprocess/nuclide_data_origin_NNDC.csv')
    data_unit_unified.to_csv('./results/01_preprocess/nuclide_data_unit_unified.csv')
    print(f'Unit-unified data is stored in: ./results/01_preprocess/nuclide_data_unit_unified.csv')
    data_modified.to_csv('./results/01_preprocess/nuclide_data_modified.csv')
    print(f'Original data is stored in: ./results/01_preprocess/nuclide_data_modified.csv')

    # print('\n======================================== Linear verification =========================================\n')
    # df_alpha = alpha_decay_fit(data_modified)
    # df_alpha.to_csv('./results/01_preprocess/alpha_linear.csv')
    # print(f'Original data is stored in: ./results/01_preprocess/nuclide_data_modified.csv')


    print('\n======================================== Gird search hyperparameter =========================================\n')
    (best_hparams, results_df)=hyperparameter_tuning('./results/01_preprocess/nuclide_data_modified.csv')


    print('\n======================================== NeuralNetwork =========================================\n')
    NeuralNetwork_LM_train('./results/01_preprocess/nuclide_data_modified.csv',best_hparams)

if __name__ == '__main__':
    # json文件的路径(NNDC)
    path_json = 'Nuclear_decay_data_all.json'
    main(path_json)
