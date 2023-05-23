"""
    【Analyze】
    Sort All Bayesian Evidence

    python local_2_global.py --data XXX --eps 0.001/0.01/0.02/0.03/0.04/0.05 --interpret_method gradient/DeepLIFT/LRP/LIME
"""
import time

import numpy as np
import pandas as pd

from utils.utils_file import generate_bayes_factors_filename, generate_local_2_global_QGI_filename, \
    generate_data_filename, generate_feature_distri_filename
from utils.utils_parser import DefaultArgumentParser, init_config
from utils.utils_plot import plot_curve, plot_feature_distribution

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
    opt.exp_name = 'local_2_global'
    init_config(opt)

    data = np.loadtxt(generate_data_filename(opt, True))
    bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
    bayes_factors = np.abs(bayes_factors)

    assert data.shape == bayes_factors.shape
    n_data, n_features = data.shape

    sorted_bayes_factors = np.sort(bayes_factors, axis=0)  # (n_data, **n_features)
    opt.logger.info(f'n_data={n_data}, n_features={n_features}')

    columns = [f'x{i}' for i in range(n_features)]
    writer = pd.ExcelWriter(f'{opt.log_dir}/sorted_{opt.interpret_method}_{opt.algorithm}_{opt.model_name}.xlsx')
    pd_data = pd.DataFrame(sorted_bayes_factors, columns=columns)
    pd_data.to_excel(writer, float_format='%.4f')
    writer.close()

    rates = np.arange(0, n_data + 1, 1) / n_data
    opt.logger.info(f'rates={rates}')

    for j in range(n_features):
        sorted_bayes_factors_xj = sorted_bayes_factors[:, j]
        sorted_bayes_factors_xj = np.insert(sorted_bayes_factors_xj, 0, 0)
        plot_curve(rates, sorted_bayes_factors_xj, f'x{j} Q-GI with lambda from 0 to 1',
                   generate_local_2_global_QGI_filename(opt, f'x{j}'), xlabel='lambda', ylabel='Q-GI')

    if opt.data == 'simulation_v3':
        plot_thresholds_rate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        plot_thresholds_numb = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        target2auxiliary = {0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4, 6: 6, 7: 7}
        for j in range(n_features):
            tot_locs = sorted(range(n_data), key=lambda k: bayes_factors[k, j])
            for threshold, numb in zip(plot_thresholds_rate, plot_thresholds_numb):
                plot_feature_distribution(data, tot_locs[:numb], tot_locs[numb:], j, target2auxiliary[j],
                                          f'x{target2auxiliary[j]} distribution based on x{j} tested',
                                          generate_feature_distri_filename(opt, 'lambda', threshold, f'x{j}'))

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
