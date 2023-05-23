"""
    【Run Baseline t_test】

    python run_bootstrap.py --data XXX --n_samples_per_bootstrap 1000
"""
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample

from utils.utils_file import generate_targets_filename, generate_data_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    parser.add_argument('--n_bootstraps', default=1000, type=int)
    parser.add_argument('--n_samples_per_bootstrap', default=1000, type=int)

    opt = parser.parse_args()
    opt.exp_name = 'run_bootstrap'
    init_config(opt)

    opt.logger.info('\nStep 1: Load the Data.')
    X = np.loadtxt(generate_data_filename(opt, True))
    y = np.loadtxt(generate_targets_filename(opt, True))
    n_data, n_features = X.shape
    opt.logger.info(f'X.shape={X.shape}, y.shape={y.shape}, n_data={n_data}, n_features={n_features}')

    opt.logger.info('\nStep 2: Fit the Regression Models.')
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    opt.logger.info(f'Finish training lr model, MSE={mean_squared_error(y, y_pred)}')
    opt.logger.info(f'model.coef_={model.coef_}')
    opt.logger.info(f'model.intercept_={model.intercept_}')

    opt.logger.info('\nStep 3: Running bootstraps.')
    bootstrapped_coefs = []
    for i in range(opt.n_bootstraps):
        # selecting samples randomly
        indices = np.random.choice(n_data, opt.n_samples_per_bootstrap, replace=True)
        # X_samples, y_samples = resample(X, y, n_samples=opt.n_samples_per_bootstrap)
        X_samples, y_samples = X[indices], y[indices]
        # fit the regression model
        t_model = LinearRegression()
        t_model.fit(X_samples, y_samples)
        bootstrapped_coefs.append(t_model.coef_)
        if (i + 1) % 100 == 0:
            y_pred_samples = t_model.predict(X_samples)
            opt.logger.info(f'Finish bootstrap {i}, MSE={mean_squared_error(y_samples, y_pred_samples)}.')

    bootstrapped_coefs = np.array(bootstrapped_coefs)   # (n_bootstraps, n_features)
    mean_coefs = np.mean(bootstrapped_coefs, axis=0)
    std_coefs = np.std(bootstrapped_coefs, axis=0)
    t_statistics = (model.coef_ - mean_coefs) / std_coefs
    opt.logger.info(f'Bootstrap mean coef={mean_coefs}')
    opt.logger.info(f'Bootstrap std coef={std_coefs}')
    opt.logger.info(f'Bootstrap t_statistics={t_statistics}')

    t_statistics2, p_values2 = ttest_1samp(bootstrapped_coefs, 0, axis=0)
    opt.logger.info(f'Bootstrap t_statistics2={t_statistics2}')
    opt.logger.info(f'Bootstrap p_values2={p_values2}')

    z_scores, p_values3 = sm.stats.ztest(bootstrapped_coefs, value=0)
    opt.logger.info(f'Bootstrap z-scores={z_scores}')
    opt.logger.info(f'Bootstrap p_values3={p_values3}')

    indices = [f'x{i}' for i in range(n_features)]
    writer = pd.ExcelWriter(f'{opt.log_dir}/bootstrap.xlsx')
    pd_data = pd.DataFrame(np.array(bootstrapped_coefs), columns=indices)
    pd_data.to_excel(writer, 'bootstrapped_coefs', index=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(mean_coefs)[:, None], index=indices)
    pd_data.to_excel(writer, 'mean_coefs', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(std_coefs)[:, None], index=indices)
    pd_data.to_excel(writer, 'std_coefs', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(t_statistics)[:, None], index=indices)
    pd_data.to_excel(writer, 't_statistics', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(model.coef_), index=indices)
    pd_data.to_excel(writer, 'coef_', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(t_statistics2)[:, None], index=indices)
    pd_data.to_excel(writer, 't_statistics2', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(p_values2)[:, None], index=indices)
    pd_data.to_excel(writer, 'p_values2', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(z_scores)[:, None], index=indices)
    pd_data.to_excel(writer, 'z_scores', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(p_values3)[:, None], index=indices)
    pd_data.to_excel(writer, 'p_values3', header=False, float_format='%.5f')
    writer.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
