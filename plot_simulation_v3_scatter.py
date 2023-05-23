"""
    【Analyze】
    Plot Scatters and Histograms of Toy Example

    python plot_simulation_v3_scatter.py --eps 0.001 --interpret_method gradient/DeepLIFT/LRP/LIME
"""
import time

import numpy as np
from matplotlib import pyplot as plt

from utils.utils_file import generate_bayes_factors_filename, generate_data_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='choose which model to get distri')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--eps', default=0.001, type=float, help='eps for local binary label')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--algorithm', type=str, default='p_s')

    opt = parser.parse_args()
    opt.data = 'simulation_v3'
    opt.exp_name = 'plot_simulation_v3_scatter'
    init_config(opt)

    data = np.loadtxt(generate_data_filename(opt, True))
    bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
    bayes_factors = np.abs(bayes_factors)
    bayes_factors = 1 - bayes_factors

    n_data, n_features = bayes_factors.shape

    fontsize = 25

    plt.scatter(data[:, 2], bayes_factors[:, 1], alpha=0.2)
    plt.xlabel('x2', fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel('Bayesian Evidence of x1', fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.savefig(f'{opt.log_dir}/scatter_x1_{opt.interpret_method}.svg', bbox_inches='tight')
    plt.close()

    plt.scatter(data[:, 1], bayes_factors[:, 2], alpha=0.2)
    plt.xlabel('x1', fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel('Bayesian Evidence of x2', fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.savefig(f'{opt.log_dir}/scatter_x2_{opt.interpret_method}.svg', bbox_inches='tight')
    plt.close()

    plt.scatter(data[:, 3], bayes_factors[:, 3], alpha=0.2)
    plt.xlabel('x3', fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel('Bayesian Evidence of x3', fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontproperties='Times New Roman', size=fontsize)
    plt.savefig(f'{opt.log_dir}/scatter_x3_{opt.interpret_method}.svg', bbox_inches='tight')
    plt.close()

    for threshold1, threshold2 in zip([0, 0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 1.001]):
        locs = np.setdiff1d(np.argwhere(threshold1 <= np.abs(data[:, 3])),
                            np.argwhere(threshold2 <= np.abs(data[:, 3])))
        plt.xlabel(xlabel='Bayesian Evidence of x3', fontproperties='Times New Roman', fontsize=fontsize)
        plt.ylabel(ylabel='Frequency', fontproperties='Times New Roman', fontsize=fontsize)
        plt.yticks(fontproperties='Times New Roman', size=fontsize)
        plt.xticks(fontproperties='Times New Roman', size=fontsize)
        plt.hist(bayes_factors[locs, 3])
        plt.savefig(f'{opt.log_dir}/hist_x3_{threshold1}_{threshold2}_{opt.interpret_method}.svg', bbox_inches='tight')
        plt.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
