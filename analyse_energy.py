"""
    【Plot x8 Distribution】

    `python analyse_energy.py --data energy_16 --interpret_method gradient/DeepLIFT/LRP/LIME`
"""
import shutil
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datasets.regdata import build_reg_dataset
from utils.utils_file import generate_bayes_factors_filename, generate_bayes_factors_thresholds_curve_filename, \
    generate_bayes_factors_thresholds_excel_filename, generate_local_roc_curve_filename, \
    generate_global_roc_curve_filename, generate_local_auc_excel_filename, \
    generate_global_auc_excel_filename, generate_auc_curve_filename, generate_bayes_factors_thresholds_area_filename, \
    generate_binary_global_label_filename, generate_binary_local_label_filename, generate_bayes_factors_distri_filename, \
    generate_bayes_factors_scatter_filename, generate_X_Y_scatter_filename, generate_data_filename, \
    generate_feature_distri_filename
from utils.utils_parser import DefaultArgumentParser, init_config, report_args
from utils.utils_plot import plot_curve, plot_roc_curve, plot_area, plot_feature_distribution

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    plt.rc('font', family='Times New Roman')

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='choose which model to get distri')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--eps', default=0, type=float, help='eps for local binary label')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--algorithm', type=str, default='p_s')

    opt = parser.parse_args()
    opt.exp_name = 'analyse_energy'
    init_config(opt)

    data = np.loadtxt(generate_data_filename(opt, True))
    bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
    bayes_factors = np.abs(bayes_factors)
    bayes_factors = 1 - bayes_factors

    assert data.shape == bayes_factors.shape
    n_data, n_features = data.shape

    # global_labels = np.loadtxt(generate_binary_global_label_filename(opt, True))
    # local_labels = np.loadtxt(generate_binary_local_label_filename(opt, True))
    #
    # for i in range(opt.n_features):
    #     plt.hist(bayes_factors[:, i])
    #     plt.title(f'X{i + 1}')
    #     plt.savefig(generate_bayes_factors_distri_filename(opt, f'x{i}'))
    #     plt.close()
    #
    # for i in range(opt.n_features):
    #     plt.scatter(dataset.data[:, i], bayes_factors[:, i])
    #     plt.title(f'X{i + 1}')
    #     plt.savefig(generate_bayes_factors_scatter_filename(opt, f'x{i}'))
    #     plt.close()
    #
    # for i in range(opt.n_features):
    #     plt.scatter(dataset.data[:, i], dataset.targets)
    #     plt.xlabel(f'X{i + 1}')
    #     plt.ylabel(f'y1')
    #     plt.savefig(generate_X_Y_scatter_filename(opt, f'x{i}', 'y1'))
    #     plt.close()

    # values, counts = np.unique(dataset.data[:, 6], return_counts=True)
    # for value, count in zip(values, counts):
    #     print(f'{value}:{count}')
    #     locs = np.squeeze(np.argwhere(dataset.data[:, 6] == value))
    #     y = bayes_factors[locs, 7]
    #     plt.scatter([value] * len(locs), y, alpha=0.1)
    #     print(sum(y == 1))
    # plt.xlabel(f'X7')
    # plt.ylabel(f'bayes factors (X8)')
    # plt.savefig(f'{opt.log_dir}/test.png')
    # plt.close()

    values = np.unique(data[:, 7])
    for value in values:
        locs = np.squeeze(np.argwhere(data[:, 7] == value))
        plt.hist(bayes_factors[locs, 7])
        plt.xlabel(xlabel='Bayesian Evidence', fontsize=20)
        plt.ylabel(ylabel='Frequency', fontsize=20)
        plt.yticks(fontproperties='Times New Roman', size=15)
        plt.xticks(fontproperties='Times New Roman', size=15)
        plt.savefig(generate_bayes_factors_distri_filename(opt, f'x{7}={value}'), bbox_inches='tight')
        plt.close()

    tot_locs = sorted(range(n_data), key=lambda k: bayes_factors[k, 7])
    threshold = 0.05
    numb = int(threshold * n_data)
    plot_feature_distribution(data, tot_locs[:numb], tot_locs[numb:], 7, 7,
                              f'x7 distribution based on x7 tested',
                              generate_feature_distri_filename(opt, 'lambda', threshold, f'x7'))

    # locs = np.argwhere(bayes_factors[:, 7] < 0.6)
    # selected_features = dataset.data[locs, 6]
    # selected_values, selected_counts = np.unique(selected_features, return_counts=True)
    # other_locs = [i for i in range(dataset.data.size(0)) if i not in locs]
    # other_features = dataset.data[other_locs, 6]
    # other_values, other_counts = np.unique(other_features, return_counts=True)
    #
    # print(selected_values)
    # print(selected_counts)
    #
    # print(other_values)
    # print(other_counts)

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
