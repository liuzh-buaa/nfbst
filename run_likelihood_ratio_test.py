"""
    【Run Baseline t_test】
    Reference1: https://gist.github.com/rnowling/ec9c9038e492d55ffae2ae257aa4acd9
    Reference2: https://www.statology.org/likelihood-ratio-test-in-python/

    python run_t_test.py --data XXX
"""

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from utils.utils_file import generate_targets_filename, generate_data_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()
    opt = parser.parse_args()
    opt.exp_name = 'run_likelihood_ratio_test'
    init_config(opt)

    opt.logger.info('\nStep 1: Load the Data.')
    X = np.loadtxt(generate_data_filename(opt, True))
    y = np.loadtxt(generate_targets_filename(opt, True))
    n_data, n_features = X.shape
    opt.logger.info(f'X.shape={X.shape}, y.shape={y.shape}, n_data={n_data}, n_features={n_features}')

    opt.logger.info('\nStep 2: Fit the Regression Models.')
    full_model = sm.OLS(y, X).fit()
    full_ll = full_model.llf
    opt.logger.info(f'log-likelihood of full model={full_ll}')

    opt.logger.info('\nStep 3: Perform the Log-Likelihood Test.')
    lr_stats, p_values = [], []
    for i in range(n_features):
        reduced_X = X[:, [_ for _ in range(i)] + [_ for _ in range(i + 1, n_features)]]
        opt.logger.info(f'Deleting {i} feature, reduced_X.shape={reduced_X.shape}')
        reduced_model = sm.OLS(y, reduced_X).fit()
        reduced_ll = reduced_model.llf
        lr_stat = -2 * (reduced_ll - full_ll)
        p_value = chi2.sf(lr_stat, 1)
        opt.logger.info(f'log-likelihood of reduced model={reduced_ll}, lr_stat={lr_stat}, p_value={p_value}')
        lr_stats.append(lr_stat)
        p_values.append(p_value)

    indices = [f'x{i}' for i in range(n_features)]
    writer = pd.ExcelWriter(f'{opt.log_dir}/likelihood_ratio_test.xlsx')
    pd_data = pd.DataFrame(np.array(lr_stats)[:, None], index=indices)
    pd_data.to_excel(writer, 'lr_stats', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(p_values)[:, None], index=indices)
    pd_data.to_excel(writer, 'p_values', header=False, float_format='%.5f')
    writer.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
