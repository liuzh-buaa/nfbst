import time

import captum.attr
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch import nn

from datasets.regdata import build_reg_loader
from models.model_utils import load_model
from utils.utils_file import generate_bayes_factors_filename
from utils.utils_parser import DefaultArgumentParser, init_config


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones((4, 1))

    def forward(self, x):
        return torch.mm(x, self.w)


# if __name__ == '__main__':
#     data = pd.read_excel('data/simulation_v4/joint_different_curves/2022-12-24 17-57-12/local_auc_DeepSHAP(nn_1).xlsx').to_numpy()
#     for model2auc in data:
#         model_name, auc = model2auc[0], model2auc[1:51]
#         plt.boxplot(auc, patch_artist=True)
#         plt.show()
#         exit(0)

def test_gradient_correctness():
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='nn', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='nn_1', type=str, help='choose which model to get distri')
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs to run in parallel for kde')
    parser.add_argument('--start', default=0, type=int, help='which data to start')
    parser.add_argument('--end', default=-1, type=int, help='which data to end;; If -1, then to all data')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--interpret_method', default='gradient', type=str, help='feature importance metric')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')

    opt = parser.parse_args()
    opt.data = 'mnist'
    init_config(opt, model_config=True)

    model = load_model(opt.model_name, opt, resume=True, last=True)
    model.eval()
    dataset, dataloader = build_reg_loader(opt, train=False)
    for inputs, targets in dataloader:
        inputs = inputs.to(opt.device)
        gradient = model.get_interpret(inputs, opt)

        model.convert_to_standard_model()
        saliency = captum.attr.Saliency(model)
        attribution = saliency.attribute(inputs, 0)

        print('ok')

        break

    bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_samples, **n_features)
    bayes_factor = bayes_factors[0]
    print('ok')


def plot_data0_X6_X7():
    plt.rc('font', family='Times New Roman')

    # plt.figure(figsize=(6, 6))
    data = 101
    feature = 6
    # distri_data = np.load(f'data/simulation_v3/results/data_distri/gaussian_e/data_distri_{data}.npy')
    distri_data = np.load(f'data/simulation_v3/results/gradient_distri/gaussian_e/gradient_distri_{data}.npy')
    distri_data_feature = distri_data[:, feature]
    plt.hist(distri_data_feature)
    plt.xlabel('gradient', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xticks(fontproperties='Times New Roman', size=20)
    # plt.title(f'data{data} X{feature}', fontsize=18)
    plt.savefig(f'data{data} X{feature}.svg', bbox_inches='tight')
    plt.close()

    # plt.figure(figsize=(6, 6))
    feature = 7
    # distri_data = np.load(f'data/simulation_v3/results/data_distri/gaussian_e/data_distri_{data}.npy')
    distri_data = np.load(f'data/simulation_v3/results/gradient_distri/gaussian_e/gradient_distri_{data}.npy')
    distri_data_feature = distri_data[:, feature]
    plt.hist(distri_data_feature)
    plt.xlabel('gradient', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xticks(fontproperties='Times New Roman', size=15)
    # plt.title(f'data{data} X{feature}', fontsize=18)
    plt.savefig(f'data{data} X{feature}.svg', bbox_inches='tight')
    plt.close()


def plot_significance_test_example():
    mu1, sigma1 = 0.1, 0.1
    mu2, sigma2 = 0.6, 0.15
    X_start, X_end = -0.4, 1.2
    X = np.linspace(X_start, X_end, 10000)

    def exp(x):
        return 0.5 * stats.norm.pdf(x, mu1, sigma1) + 0.5 * stats.norm.pdf(x, mu2, sigma2)

    y = exp(X)
    plt.plot(X, y, color='black')

    X_base = 0.0
    y_base = exp(X_base)
    plt.plot([X_start, X_end], [y_base, y_base], color='red', linestyle='--')

    plt.plot([0.0, 0.0], [0.0, y_base], color='red', linestyle='--')

    X1 = [x for x in X if exp(x) > y_base]
    X2 = [x for x in X if x not in X1]
    plt.fill_between(X2, 0.0, exp(X2), color=(50 / 255, 76 / 255, 127 / 255), label=r'$Ev(H_0)$')
    # plt.fill_between(X1, y_base, exp(X1), color=(146 / 255, 208 / 255, 80 / 255), label='evidence against H0')
    # plt.fill_between(X1, y_base, exp(X1), color=(0 / 255, 176 / 255, 240 / 255), label='evidence against H0')
    plt.fill_between(X1, y_base, exp(X1), color=(175 / 255, 171 / 255, 171 / 255))

    plt.yticks([], fontproperties='Times New Roman', size=20)
    plt.xticks([], fontproperties='Times New Roman', size=20)
    # plt.xlabel(r'$\eta(\theta)$', fontproperties='Times New Roman', size=20)
    # plt.ylabel(r'$p(\eta(\theta)|\mathcal{D})$', fontproperties='Times New Roman', size=20)
    plt.legend(prop={'family': 'Times New Roman', 'size': 20})

    plt.savefig('test.svg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_data0_X6_X7()
    # plot_significance_test_example()
