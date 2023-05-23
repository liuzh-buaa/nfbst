"""
    【Ensemble Different BNNs】
    In practice, we adopt diagonal Gaussian distribution as our variational family of model parameters and
    ensemble three Bayesian neural networks to spread the range of model parameters further.

    `python ensemble_statistic_distri.py --log True --data XXX --model_type gaussian --model_name gaussian_e --interpret_method XXX --y_index 0`
"""
import shutil
import time

import numpy as np

from utils.utils_file import generate_statistic_sample_filename, generate_statistic_distri_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='ensemble model name')
    parser.add_argument('--model_indices', default=[1, 2, 3], nargs='+', help='choose which model to get distri')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')

    opt = parser.parse_args()
    opt.exp_name = 'ensemble_statistic_distri'
    init_config(opt)

    init_model_name = opt.model_name

    model_names = [f'{opt.model_type}_{index}' for index in opt.model_indices]
    opt.logger.info(f'==> Ensemble {opt.interpret_method} distri of {model_names}...')
    data = []
    for model_name in model_names:
        opt.model_name = model_name
        data.append(np.load(generate_statistic_sample_filename(opt, 'total', last=True)))

    opt.model_name = init_model_name
    np_data = np.concatenate(data, axis=0)  # (sample_T * len(model_names), n_data, **n_features)
    np.save(generate_statistic_sample_filename(opt, 'total', last=False), np_data)

    for i in range(np_data.shape[1]):
        grad = np_data[:, i, :]
        np.save(generate_statistic_distri_filename(opt, i, last=False), grad)
        if i % 100 == 0:
            opt.logger.info(f'Finish getting data point{i} {opt.interpret_method} distri of {opt.model_name}: {grad.shape}')

    if opt.log:
        opt.logger.info(f'Copying total {opt.interpret_method} samples of {opt.model_name}...')
        shutil.copyfile(generate_statistic_sample_filename(opt, 'total', last=False),
                        generate_statistic_sample_filename(opt, 'total', last=True))

        opt.logger.info(f'Copying data {opt.interpret_method} distri of {opt.model_name}...')
        for k in range(np_data.shape[1]):
            shutil.copyfile(generate_statistic_distri_filename(opt, k, last=False),
                            generate_statistic_distri_filename(opt, k, last=True))

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
