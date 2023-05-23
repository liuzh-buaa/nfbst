import matplotlib

from utils.utils_file import generate_feature_margin_distri_filename, generate_feature_mutual_distri_filename

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def visualize_feature_margin_distr(opt, data, n_max=None):
    for i in range(opt.n_features):
        plt.hist(data[:, i])
        plt.title(f'x{i}')
        plt.savefig(generate_feature_margin_distri_filename(opt, f'x{i}', False))
        plt.close()

    for i in range(opt.n_features):
        if n_max is None:
            j_list = range(i + 1, opt.n_features)
        else:
            j_list = np.random.permutation(np.arange(i + 1, opt.n_features))[:n_max]
        for j in j_list:
            feature_i = f'x{i}'
            feature_j = f'x{j}'
            plt.title(f'{feature_i} vs. {feature_j}')
            plt.scatter(data[:, i], data[:, j])
            plt.savefig(generate_feature_mutual_distri_filename(opt, feature_i, feature_j, False))
            plt.close()
