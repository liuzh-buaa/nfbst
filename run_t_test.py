"""
    【Run Baseline t_test】

    python run_t_test.py --data XXX
"""
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from datasets.regdata import build_reg_loader, build_reg_dataset
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    parser.add_argument('--indices', default=None, type=int, help='dataset indices')

    opt = parser.parse_args()
    opt.exp_name = 'run_t_test'
    init_config(opt, model_config=False)

    # Note: get the whole dataset, and its order must be fixed (shuffle=False)
    dataset = build_reg_dataset(opt, train=False, indices=opt.indices)
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    X, y = next(iter(dataloader))
    X, y = X.numpy(), y.numpy()

    n_data, n_features = X.shape
    opt.logger.info(f'X.shape={X.shape}, y.shape={y.shape}, n_data={n_data}, n_features={n_features}')

    X_extend = np.ones(n_data)
    X = np.concatenate((X, X_extend[:, None]), axis=1)

    opt.logger.info(f'X_extend.shape={X.shape}')

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    opt.logger.info(f'Finish training lr model, MSE={mean_squared_error(y, y_pred)}')
    opt.logger.info(f'model.coef_={model.coef_}')
    opt.logger.info(f'model.intercept_={model.intercept_}')

    XTX = np.matmul(np.transpose(X), X)
    XTX_inv = np.linalg.inv(XTX)
    theta_hat = np.matmul(np.matmul(XTX_inv, np.transpose(X)), y)

    opt.logger.info(f'theta_hat={theta_hat}')

    # RSS = np.sum(np.square(y_pred - np.matmul(X, theta_hat)))
    # opt.logger.info(f'RSS={RSS}')
    #
    # SE2 = np.array([XTX_inv[i, i] for i in range(n_features + 1)]) * RSS / (n_data - n_features - 1)
    # SE2 = SE2[:, None]
    # opt.logger.info(f'SE^2={SE2}')

    dfe = n_data - n_features - 1

    y_hat = np.matmul(X, theta_hat)
    resid = y - y_pred
    ssr = np.sum((y_hat - np.mean(y)) ** 2)
    sse = np.sum(resid ** 2)
    sst = sse + ssr

    se2 = (sse / dfe) / (np.sum((X - np.mean(X, axis=0)) ** 2, axis=0))
    se2 = se2[:, None]
    opt.logger.info(f'se2={se2}')

    t_statistics = theta_hat / np.sqrt(se2)
    opt.logger.info(f't_statistics={t_statistics}')

    p_values = []
    for t_statistic in t_statistics:
        if t_statistic > 0:
            p_values.append(2 * (1 - stats.t.cdf(t_statistic, df=dfe)))
        else:
            p_values.append(2 * stats.t.cdf(t_statistic, df=dfe))

    indices = [f'x{i}' for i in range(n_features)]
    indices.append('intercept')
    writer = pd.ExcelWriter(f'{opt.log_dir}/t_test.xlsx')
    pd_data = pd.DataFrame(t_statistics, index=indices)
    pd_data.to_excel(writer, 't_statistics', header=False, float_format='%.5f')
    pd_data = pd.DataFrame(np.array(p_values), index=indices)
    pd_data.to_excel(writer, 'p_values', header=False, float_format='%.5f')
    writer.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
