"""
    【Analyze】
    Compare with Feature Importance Analysis

    `python analyse_bayes_factors.py --log True --data XXX --model_type gaussian --model_name gaussian_e --eps XXX --interpret_method XXX --algorithm p_s`
    `python analyse_bayes_factors.py --log True --data XXX --model_type nn --model_name nn_1 --eps XXX --interpret_method XXX --algorithm mean`
"""
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datasets.regdata import build_reg_dataset
from utils.utils_file import generate_bayes_factors_filename, generate_bayes_factors_thresholds_curve_filename, \
    generate_bayes_factors_thresholds_excel_filename, generate_local_roc_curve_filename, \
    generate_global_roc_curve_filename, generate_local_auc_excel_filename, \
    generate_global_auc_excel_filename, generate_auc_curve_filename, generate_bayes_factors_thresholds_area_filename, \
    generate_binary_global_label_filename, generate_binary_local_label_filename
from utils.utils_parser import DefaultArgumentParser, init_config
from utils.utils_plot import plot_curve, plot_roc_curve, plot_area

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    plt.rc('font', family='Times New Roman')

    # model settings
    parser.add_argument('--model_type', default='gaussian', type=str, help='Variation inference family')
    parser.add_argument('--model_name', default='gaussian_e', type=str, help='choose which model to get distri')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--eps', default=0, type=float, help='eps for local binary label')

    parser.add_argument('--interpret_method', default='gradient', type=str, help='testing statistic')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')
    parser.add_argument('--algorithm', type=str, default='p_s',
                        choices=['first', 'mean', 'mean_abs', 'abs_mean', 'p_s'])

    opt = parser.parse_args()
    opt.exp_name = 'analyse_bayes_factors'
    init_config(opt)

    bayes_factors = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
    bayes_factors = np.abs(bayes_factors)

    dataset = build_reg_dataset(opt)

    global_labels = np.loadtxt(generate_binary_global_label_filename(opt, True))
    local_labels = np.loadtxt(generate_binary_local_label_filename(opt, True))

    print(f'==> Plotting local roc curves...')
    local_auc = []
    for j in range(opt.n_features):
        local_auc.append(plot_roc_curve(local_labels[:, j], bayes_factors[:, j],
                                        f'{opt.model_name} {opt.interpret_method}_{opt.algorithm} x{j}',
                                        generate_local_roc_curve_filename(opt, f'x{j}')))

    features = [f'x{i}' for i in range(opt.n_features)]
    writer = pd.ExcelWriter(generate_local_auc_excel_filename(opt))
    pd_data2 = pd.DataFrame(np.array(local_auc), index=features).T
    pd_data2.to_excel(writer, opt.model_name, float_format='%.3f')
    writer.close()

    plot_roc_curve(np.reshape(local_labels, -1), np.reshape(bayes_factors, -1),
                   f'{opt.model_name} {opt.interpret_method}_{opt.algorithm}',
                   generate_local_roc_curve_filename(opt, 'total'))

    plot_curve(range(0, opt.n_features), local_auc, f'{opt.model_name} {opt.interpret_method}_{opt.algorithm}',
               generate_auc_curve_filename(opt), xlabel='features', ylabel='auc',
               xlim=[0, opt.n_features], ylim=[0.0, 1.0], diagonal=False)

    print(f'==> Plotting selected insignificant data rate based threshold1...')
    thresholds = np.linspace(0, 1, 101, endpoint=True)
    print(f'==> Threshold1: {thresholds}')

    result_thresholds = []
    for j in range(opt.n_features):
        bayes_factors_xj = bayes_factors[:, j]  # (n_samples, )
        result_thresholds_xj = []
        for threshold in thresholds:
            locs = np.argwhere(bayes_factors_xj <= threshold)
            result_thresholds_xj.append(len(locs))
        result_thresholds_xj = [_ / len(bayes_factors_xj) for _ in result_thresholds_xj]
        plot_curve(thresholds, result_thresholds_xj, f'{opt.model_name} x{j}',
                   generate_bayes_factors_thresholds_curve_filename(opt, f'x{j}'),
                   xlabel='Thresholds', ylabel='Insignificant Data Rate')
        result_thresholds.append(np.array(result_thresholds_xj)[None, :])

    print(f'==> Plotting area for significant and insignificant features...')
    np_result_thresholds = np.concatenate(result_thresholds, axis=0)  # (n_features, |thresholds|)
    plot_area(thresholds, np_result_thresholds, np.where(global_labels == 1)[0], np.where(global_labels == 0)[0],
              f'{opt.model_name}', generate_bayes_factors_thresholds_area_filename(opt),
              xlabel='Thresholds', ylabel='Insignificant Data Rate')

    print(f'==> Analysing global feature importance based on thresholds of two strategies...')
    # avoid round precision. e.g. np.arange(0.991, 1, 0.001) may be not right
    if opt.n_samples == 10000:
        sub_analyse_thresholds_num = np.arange(0, 10000, 1000)
        sub_analyse_thresholds_num = np.append(sub_analyse_thresholds_num, np.arange(9100, 10000, 100))
        sub_analyse_thresholds_num = np.append(sub_analyse_thresholds_num, np.arange(9910, 10000, 10))
        sub_analyse_thresholds_num = np.append(sub_analyse_thresholds_num, np.arange(9991, 10000, 1))
        sub_analyse_thresholds_num = np.append(sub_analyse_thresholds_num, [10000])
        # [0, 1000, ..., 9000, 9100, ..., 9900, 9910, ..., 9990, 9991, ..., 9999, 10000]
        sub_analyse_thresholds_rate = sub_analyse_thresholds_num / 10000  # [0, 0.1, ..., 0.9, 0.91, ..., 0.99, 0.991, ..., 0.999, 0.9991, ..., 0.9999, 1.]
        sub_analyse_thresholds_num -= 1
        sub_analyse_thresholds_num[
            0] = 0  # [0, 999, ..., 8999, 9099, ..., 9899, 9909, ..., 9989, 9990, ..., 9998, 9999]
    else:
        sub_analyse_thresholds_rate = np.linspace(0, 1., 10, endpoint=False)
        sub_analyse_thresholds_rate = np.append(sub_analyse_thresholds_rate, np.linspace(0.91, 1., 9, endpoint=False))
        sub_analyse_thresholds_rate = np.append(sub_analyse_thresholds_rate, np.linspace(0.991, 1., 10, endpoint=True))
        sub_analyse_thresholds_num = np.floor(sub_analyse_thresholds_rate * opt.n_samples).astype(int)
        sub_analyse_thresholds_num[-1] -= 1
    print(f'==> Rate: {sub_analyse_thresholds_rate}\n'
          f'==> Number: {sub_analyse_thresholds_num}')

    print(f'==> Analysing strategy1: fix threshold1, adjust threshold2...')
    result_thresholds = []
    for j in range(opt.n_features):
        bayes_factors_xj = bayes_factors[:, j]
        result_thresholds_xj = []
        for threshold in sub_analyse_thresholds_rate:
            locs = np.argwhere(bayes_factors_xj <= threshold)
            result_thresholds_xj.append(len(locs))
        result_thresholds_xj = [_ / len(bayes_factors_xj) for _ in result_thresholds_xj]
        result_thresholds.append(np.array(result_thresholds_xj)[None, :])

    np_result_thresholds = np.concatenate(result_thresholds, axis=0)  # (n_features, |sub_thresholds|)
    probs = 1 - np_result_thresholds  # (n_features, |sub_thresholds|)
    global_auc_threshold1 = []
    for i, threshold in enumerate(sub_analyse_thresholds_rate):
        print(f'==> Set threshold1={threshold}')
        global_auc_threshold1.append(
            plot_roc_curve(global_labels, probs[:, i], f'{opt.model_name} threshold1={threshold}',
                           generate_global_roc_curve_filename(opt, 'threshold1', threshold)))

    df = pd.DataFrame(columns=sub_analyse_thresholds_rate, index=features, data=probs)
    writer = pd.ExcelWriter(generate_bayes_factors_thresholds_excel_filename(opt, 'thresholds1'))
    df.to_excel(writer, opt.model_name)
    print(df)
    writer.close()

    print(f'==> Analysing strategy2: fix threshold2, adjust threshold1...')
    sorted_bayes_factors = np.sort(bayes_factors, axis=0)
    probs = sorted_bayes_factors[sub_analyse_thresholds_num, :].T  # (n_features, |sub_thresholds|)
    global_auc_threshold2 = []
    for i, threshold in enumerate(sub_analyse_thresholds_rate):
        print(f'==> Set threshold2={threshold}')
        global_auc_threshold2.append(
            plot_roc_curve(global_labels, probs[:, i], f'{opt.model_name} threshold2={threshold}',
                           generate_global_roc_curve_filename(opt, 'threshold2', threshold)))

    df = pd.DataFrame(columns=sub_analyse_thresholds_rate, index=features, data=probs)
    writer = pd.ExcelWriter(generate_bayes_factors_thresholds_excel_filename(opt, 'thresholds2'))
    df.to_excel(writer, opt.model_name)
    print(df)
    writer.close()

    writer = pd.ExcelWriter(generate_global_auc_excel_filename(opt))
    pd_data1 = pd.DataFrame(np.array(global_auc_threshold1), index=sub_analyse_thresholds_rate)
    pd_data1.to_excel(writer, 'threshold1', float_format='%.4f')
    pd_data2 = pd.DataFrame(np.array(global_auc_threshold2), index=sub_analyse_thresholds_rate)
    pd_data2.to_excel(writer, 'threshold2', float_format='%.4f')
    writer.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
