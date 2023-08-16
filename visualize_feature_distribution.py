import heapq
import time

import numpy as np
from matplotlib import pyplot as plt

from datasets.regdata import build_reg_dataset
from utils.utils_file import generate_bayes_factors_filename, generate_feature_distri_filename, \
    generate_binary_global_label_filename, generate_binary_local_label_filename
from utils.utils_parser import DefaultArgumentParser, init_config, report_args

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='choose which model to get distri')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--eps', default=0, type=float, help='eps for local binary label')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--algorithm', type=str, default='p_s')

    opt = parser.parse_args()
    opt.exp_name = 'visualize_feature_distribution'
    init_config(opt)

    bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_samples, **n_features)

    dataset = build_reg_dataset(opt)

    global_labels = np.loadtxt(generate_binary_global_label_filename(opt, True))
    local_labels = np.loadtxt(generate_binary_local_label_filename(opt, True))

    sub_plot_thresholds_rate = [0.5, 0.9, 1]
    sub_plot_thresholds_num = np.floor(sub_plot_thresholds_rate * opt.n_samples).astype(int)

    print(f'==> Visualizing feature distribution of strategy1...')
    for j in range(opt.n_features):
        for threshold in sub_plot_thresholds_rate:
            locs = np.argwhere(bayes_factors[:, j] <= threshold)
            selected_features = dataset.data[locs, j]
            y = np.zeros_like(selected_features)
            plt.scatter(selected_features, y, label='insignificant', alpha=0.2)

            other_locs = [i for i in range(dataset.data.size(0)) if i not in locs]
            other_features = dataset.data[other_locs, j]
            y = np.ones_like(other_features)
            plt.scatter(other_features, y, label='significant', alpha=0.2)

            plt.legend()
            plt.xlabel(f'x{j}')
            plt.title(f'{opt.model_name} threshold:{threshold}')
            plt.savefig(generate_feature_distri_filename(opt, 'threshold1', threshold, f'x{j}'))
            plt.close()

    print(f'==> Visualizing feature distribution of strategy2...')
    for j in range(opt.n_features):
        for threshold, num in zip(sub_plot_thresholds_rate, sub_plot_thresholds_num):
            locs = list(map(bayes_factors[:, j].index, heapq.nsmallest(num, bayes_factors[:, j])))
            selected_features = dataset.data[locs, j]
            y = np.zeros_like(selected_features)
            plt.scatter(selected_features, y, label='insignificant', alpha=0.2)

            other_locs = [i for i in range(dataset.data.size(0)) if i not in locs]
            other_features = dataset.data[other_locs, j]
            y = np.ones_like(other_features)
            plt.scatter(other_features, y, label='significant', alpha=0.2)

            plt.legend()
            plt.xlabel(f'x{j}')
            plt.title(f'{opt.model_name} threshold:{threshold}')
            plt.savefig(generate_feature_distri_filename(opt, 'threshold2', threshold, f'x{j}'))
            plt.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
