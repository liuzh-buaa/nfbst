"""
    【Plot ROC and Joint Together】
    Note this maybe needs to fix `interpret_method, algorithms, model_names` in the code.

    `python joint_different_curves.py --data XXX --eps XXX --control XXX`
"""
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.utils_file import generate_bayes_factors_filename, generate_auc_curve_filename, \
    generate_local_auc_excel_filename, generate_binary_global_label_filename, generate_binary_local_label_filename
from utils.utils_parser import DefaultArgumentParser, init_config
from utils.utils_plot import plot_curves, calculate_auc

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    plt.rc('font', family='Times New Roman')

    # model settings
    parser.add_argument('--model_name', default='all_models', type=str, help='choose which model to get distri')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--eps', default=0, type=float, help='eps for local binary label')

    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--control', default='all', type=str,
                        choices=['all', 'gradient', 'DeepLIFT', 'DeepSHAP', 'LRP', 'gradientXinput', 'LIME'])

    opt = parser.parse_args()
    opt.exp_name = 'joint_different_curves'
    init_config(opt)

    all_plot_names = {
        # 'SHAP(cat)': {
        #     'interpret_method': 'SHAP(cat)',
        #     'algorithm': 'first',
        #     'model_name': 'cat_1'
        # },
        # 'SHAP(lgb)': {
        #     'interpret_method': 'SHAP(lgb)',
        #     'algorithm': 'first',
        #     'model_name': 'lgb_1'
        # },
        # 'SHAP(xgb)': {
        #     'interpret_method': 'SHAP(xgb)',
        #     'algorithm': 'first',
        #     'model_name': 'xgb_1'
        # },
        'DeepLIFT': {
            'interpret_method': 'DeepLIFT',
            'algorithm': 'mean',
            'model_name': 'nn_1'
        },
        # 'DeepSHAP': {
        #     'interpret_method': 'DeepSHAP',
        #     'algorithm': 'first',
        #     'model_name': 'nn_1'
        # },
        'LRP': {
            'interpret_method': 'LRP',
            'algorithm': 'mean',
            'model_name': 'nn_1'
        },
        'Gradient': {
            'interpret_method': 'gradient',
            'algorithm': 'abs_mean',
            'model_name': 'nn_1'
        },
        # 'GradientXInput': {
        #     'interpret_method': 'gradientXinput',
        #     'algorithm': 'mean',
        #     'model_name': 'nn_1'
        # },
        'LIME': {
            'interpret_method': 'LIME',
            'algorithm': 'mean',
            'model_name': 'nn_1'
        },
        'DeepLIFT-nFBST': {
            'interpret_method': 'DeepLIFT',
            'algorithm': 'p_s',
            'model_name': 'gaussian_e'
        },
        # 'DeepSHAP-nFBST': {
        #     'interpret_method': 'DeepSHAP',
        #     'algorithm': 'p_s',
        #     'model_name': 'gaussian_e'
        # },
        'LRP-nFBST': {
            'interpret_method': 'LRP',
            'algorithm': 'p_s',
            'model_name': 'gaussian_e'
        },
        'Grad-nFBST': {
            'interpret_method': 'gradient',
            'algorithm': 'p_s',
            'model_name': 'gaussian_e'
        },
        # 'GradXInput-nFBST': {
        #     'interpret_method': 'gradientXinput',
        #     'algorithm': 'p_s',
        #     'model_name': 'gaussian_e'
        # },
        'LIME-nFBST': {
            'interpret_method': 'LIME',
            'algorithm': 'p_s',
            'model_name': 'gaussian_e'
        },
    }

    if opt.control == 'DeepSHAP':
        plot_names = {
            k: all_plot_names[k] for k in ['DeepSHAP', 'DeepSHAP-nFBST']
        }
    elif opt.control == 'gradient':
        plot_names = {
            k: all_plot_names[k] for k in ['Gradient', 'Grad-nFBST']
        }
    elif opt.control == 'LRP':
        plot_names = {
            k: all_plot_names[k] for k in ['LRP', 'LRP-nFBST']
        }
    elif opt.control == 'DeepLIFT':
        plot_names = {
            k: all_plot_names[k] for k in ['DeepLIFT', 'DeepLIFT-nFBST']
        }
    elif opt.control == 'gradientXinput':
        plot_names = {
            k: all_plot_names[k] for k in ['GradientXInput', 'GradXInput-nFBST']
        }
    elif opt.control == 'LIME':
        plot_names = {
            k: all_plot_names[k] for k in ['LIME', 'LIME-nFBST']
        }
    else:
        plot_names = all_plot_names

    global_labels = np.loadtxt(generate_binary_global_label_filename(opt, True))
    local_labels = np.loadtxt(generate_binary_local_label_filename(opt, True))

    bayes_factors_models, local_auc_models = [], []
    for k, v in plot_names.items():
        opt.model_name = v['model_name']
        opt.interpret_method = v['interpret_method']
        opt.algorithm = v['algorithm']
        bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))
        bayes_factors = np.abs(bayes_factors)
        local_auc = [calculate_auc(local_labels[:, j], bayes_factors[:, j]) for j in range(opt.n_features)]
        bayes_factors_models.append(bayes_factors)
        local_auc_models.append(local_auc)

    opt.model_name = f'{opt.control}_models'

    features = [f'x{i}' for i in range(opt.n_features)]
    writer = pd.ExcelWriter(generate_local_auc_excel_filename(opt))
    pd_data = pd.DataFrame(np.array(local_auc_models), index=plot_names.keys(), columns=features)
    pd_data.to_excel(writer, float_format='%.3f')
    writer.close()

    # print(f'==> Plotting bayes factors thresholds...')
    # thresholds = np.arange(0, 1.01, 0.01)
    # for j in range(opt.n_features):
    #     method2rates = {}
    #     for model_name, bayes_factors in zip(model_names, bayes_factors_models):
    #         bayes_factors_xj = bayes_factors[:, j]  # (n_samples, )
    #         result_thresholds = []
    #         for threshold in thresholds:
    #             locs = np.argwhere(bayes_factors_xj <= threshold)
    #             result_thresholds.append(len(locs))
    #         result_thresholds_rate = [_ / len(bayes_factors_xj) for _ in result_thresholds]
    #         method2rates[model_name] = result_thresholds_rate
    #     plot_curves(thresholds, method2rates, f'{opt.model_name} x{j}',
    #                 generate_bayes_factors_thresholds_curve_filename(opt, f'x{j}'),
    #                 xlabel='Thresholds', ylabel='Insignificant Data Rate')
    #
    # print(f'==> Plotting local roc curves...')
    # for j in range(opt.n_features):
    #     method2prob = {
    #         model_name: bayes_factors[:, j] for model_name, bayes_factors in
    #         zip(model_names, bayes_factors_models)
    #     }
    #     plot_roc_curves(local_labels[:, j], method2prob, f'x{j} ROC curve',
    #                     generate_local_roc_curve_filename(opt, f'x{j}'))
    #
    # method2prob = {
    #     model_name: np.reshape(bayes_factors, -1) for model_name, bayes_factors in
    #     zip(model_names, bayes_factors_models)
    # }
    # plot_roc_curves(np.reshape(local_labels, -1), method2prob, f'ROC curve',
    #                 generate_local_roc_curve_filename(opt, 'total'))

    opt.logger.info(f'==> Plotting local auc curves...')
    method2local_aucs = {
        plot_model_name: local_auc[0:50] for plot_model_name, local_auc in zip(plot_names.keys(), local_auc_models)
    }
    plot_curves(range(0, 50), method2local_aucs, f'eps={opt.eps}', generate_auc_curve_filename(opt),
                xlabel='features', ylabel='AUC',
                xlim=[0, 49], ylim=[0.0, 1.0], diagonal=False)

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
