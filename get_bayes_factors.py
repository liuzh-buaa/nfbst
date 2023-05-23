"""
    【Bayesian Evidence Generation Process】

    `python get_bayes_factors.py --log True --data XXX --model_type gaussian --model_name gaussian_e --interpret_method XXX --algorithm p_s`(default)

        To accelerate, we can set start,end from 0 to n_samples with gaps. E.g.
        `python get_bayes_factors.py --log True --data XXX --model_type gaussian --model_name gaussian_e --interpret_method XXX --algorithm p_s --start 0 --end 1000`
        `python get_bayes_factors.py --log True --data XXX --model_type gaussian --model_name gaussian_e --interpret_method XXX --algorithm p_s --start 1000 --end 2000`
        ...
        `python get_bayes_factors.py --log True --data XXX --model_type gaussian --model_name gaussian_e --interpret_method XXX --algorithm p_s --start 9000 --end 10000`

    `python get_bayes_factors.py --log True --data XXX --model_type nn --model_name nn_1 --interpret_method XXX --algorithm mean`
"""
import shutil
import sys
import time

import numpy as np
import pandas as pd

from utils.utils_file import generate_statistic_distri_filename, generate_bayes_factors_filename, \
    generate_bayes_factors_excel_filename, generate_bayes_factors_cache_filename, generate_bandwidths_cache_filename
from utils.utils_parser import DefaultArgumentParser, init_config
from utils.utils_stat import run_hypothesis_test

if __name__ == '__main__':
    init_sys_stdout = sys.stdout

    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='choose which model to get distri')
    parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs to run in parallel for kde')
    parser.add_argument('--start', default=0, type=int, help='which data to start')
    parser.add_argument('--end', default=-1, type=int, help='which data to end; If -1, then to all data')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--algorithm', type=str, default='p_s',
                        choices=['first', 'mean', 'mean_abs', 'abs_mean', 'p_s'])

    opt = parser.parse_args()
    opt.exp_name = 'get_bayes_factors'
    init_config(opt)

    assert (opt.model_type == 'gaussian' and opt.algorithm == 'p_s') or (
            opt.model_type == 'nn' and opt.algorithm != 'p_s')

    if opt.data in ['mnist', 'cifar10']:  # testloader
        opt.n_samples = 10000

    if opt.end == -1:
        opt.end = opt.n_samples

    data, bandwidths = [], []
    for index in range(opt.start, opt.end):
        grads = np.load(generate_statistic_distri_filename(opt, index, last=True))  # (sample_T, **n_features)
        if opt.algorithm == 'p_s':
            p_s_results, _ = run_hypothesis_test(grads, opt)
            data.append(p_s_results[None, :])  # (1, **n_features)
            bandwidths.append(_[None, :])  # (1, **n_features)
        elif opt.algorithm == 'mean':
            data.append(np.mean(grads, axis=0)[None, :])  # (1, **n_features)
        elif opt.algorithm == 'mean_abs':
            data.append(np.abs(np.mean(grads, axis=0))[None, :])  # (1, **n_features)
        elif opt.algorithm == 'abs_mean':
            data.append(np.mean(np.abs(grads), axis=0)[None, :])  # (1, **n_features)
        elif opt.algorithm == 'first':
            data.append(grads[0][None, :])  # (1, **n_features)

        if (index + 1) % 100 == 0 or index == opt.end - 1:
            opt.logger.info(f'==> Running {opt.algorithm} for {opt.interpret_method} of data {index}...')
            np_data = np.concatenate(data, axis=0)
            np.save(generate_bayes_factors_cache_filename(opt, opt.start, index), np_data)
            if opt.algorithm == 'p_s':
                np_bandwidths = np.concatenate(bandwidths, axis=0)
                np_bandwidths = np.reshape(np_bandwidths, (np_bandwidths.shape[0], -1))
                writer = pd.ExcelWriter(generate_bandwidths_cache_filename(opt, opt.start, index))
                pd_bandwidth = pd.DataFrame(np_bandwidths)
                pd_bandwidth.to_excel(writer, index=False, header=False)
                writer.close()

    np_data = np.concatenate(data, axis=0)
    np.save(generate_bayes_factors_filename(opt), np_data)  # (n_data, **n_features)

    if type(opt.n_features) == int:
        features = [f'x{i}' for i in range(opt.n_features)]
        assert np_data.shape[-1] == opt.n_features
        writer = pd.ExcelWriter(generate_bayes_factors_excel_filename(opt))
        pd_data = pd.DataFrame(np_data, columns=features)
        for i in range(opt.n_targets):
            pd_data.to_excel(writer, f'y{i}', float_format='%.3f')
        writer.close()

    try:
        if opt.log and opt.start == 0 and opt.end == opt.n_samples:
            opt.logger.info(f'==> Copying {opt.algorithm} for {opt.interpret_method} from `timestamp` to `results`...')
            shutil.copyfile(generate_bayes_factors_filename(opt, last=False),
                            generate_bayes_factors_filename(opt, last=True))

            shutil.copyfile(generate_bayes_factors_excel_filename(opt, last=False),
                            generate_bayes_factors_excel_filename(opt, last=True))
    except Exception as e:
        opt.logger.warning(repr(e))

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
