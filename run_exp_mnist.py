"""
    【显著性图对比】
    MNIST数据集上不同方法的显著性图对比。

    `python run_exp_mnist.py --log True --start 0 --end 200`
"""
import os.path
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datasets.regdata import build_reg_dataset
from utils.utils_file import generate_bayes_factors_filename, generate_mnist_feature_importance_filename, \
    generate_bayes_factors_cache_filename
from utils.utils_parser import DefaultArgumentParser, init_config

FBST2timestamps = {
    'Grad-FBST': {
        0: '2023-04-27 15-20-50',
        1: '2023-04-27 15-20-57',
        2: '2023-04-27 15-21-04',
        3: '2023-04-27 11-00-34',
        # 3: '2023-04-30 20-35-32',
        4: '2023-04-27 15-21-12',
        5: '2023-04-27 15-21-20',
        6: '2023-04-27 15-21-27',
        7: '2023-04-27 15-21-33',
        8: '2023-04-27 09-33-19',
        9: '2023-04-27 15-21-39'
    },
    'DeepLIFT-FBST': {
        0: '2023-04-28 01-51-35',
        1: '2023-04-28 01-51-46',
        2: '2023-04-28 01-51-49',
        3: '2023-04-28 01-57-42',
        4: '2023-04-28 02-16-42',
        5: '2023-04-28 02-32-41',
        6: '2023-04-28 02-53-18',
        7: '2023-04-28 03-06-00',
        8: '2023-04-28 03-19-26',
        9: '2023-04-28 03-37-25'
    },
    'LRP-FBST': {
        0: '2023-04-28 08-25-55',
        1: '2023-04-28 08-49-39',
        2: '2023-04-28 09-16-34',
        3: '2023-04-28 10-55-45',
        4: '2023-04-28 11-42-47',
        5: '2023-04-28 12-20-25',
        6: '2023-04-30 00-39-57',
        7: '2023-04-30 01-09-31',
        8: '2023-04-30 01-38-02',
        9: '2023-04-30 02-03-02'
    },
    'gradientXinput-FBST': {
        0: '2023-05-07 08-49-42',
        1: '2023-05-07 08-50-13',
        2: '2023-05-07 17-18-50',
        3: '2023-05-07 10-12-09',
        4: '2023-05-07 17-24-37',
        5: '2023-05-07 17-49-58',
        6: '2023-05-07 17-54-29',
        7: '2023-05-07 17-54-31',
        8: '2023-05-07 10-12-02',
        9: '2023-05-07 17-56-06'
    }
}

target2similar = {
    0: 9,
    9: 0,
    1: 4,
    4: 1,
    2: 7,
    7: 2,
    3: 8,
    8: 3,
    5: 6,
    6: 5
}


# Function to plot scores of an MNIST figure
def viz_scores(_scores, _ax):
    reshaped_scores = _scores.reshape(28, 28)
    the_min = np.min(reshaped_scores)
    the_max = np.max(reshaped_scores)
    center = 0.0
    negative_vals = (reshaped_scores < 0.0) * reshaped_scores / (the_min + 10 ** -7)
    positive_vals = (reshaped_scores > 0.0) * reshaped_scores / float(the_max)
    reshaped_scores = -negative_vals + positive_vals
    _ax.imshow(reshaped_scores, cmap="Greys")
    _ax.set_xticks([])
    _ax.set_yticks([])


# Function that masks out the top n pixels where the score for task_1 is higher than the score for task_2
def get_masked_image(X_test, scores, task_1, task_2, n_to_erase):
    difference = scores[task_1].ravel() - scores[task_2].ravel()
    # highlight the top n
    top_nth_threshold = max(sorted(difference, reverse=True)[n_to_erase], 0.0)
    threshold_points = 1.0 * (difference <= top_nth_threshold)
    masked_inp = threshold_points.reshape(1, 28, 28) * X_test
    return masked_inp


# Function to plot the result of masking on a single example, for converting from task1 -> task2 and task1 -> task3
def plot_two_way_figures(X_test, task1, task2, method_names, n_to_erase,
                         method_to_task_to_scores=None, save_file=None):
    f, axes = plt.subplots(len(method_names), 4, figsize=(15, 10))
    for i, method_name in enumerate(method_names):
        scores = method_to_task_to_scores[method_name]
        # mean_scores_over_all_tasks = np.mean(np.array([scores[i] for i in range(10)]), axis=0)
        mean_scores_over_all_tasks = 0
        viz_scores(X_test, axes[i][0])
        viz_scores(scores[task1] - mean_scores_over_all_tasks, axes[i][1])
        viz_scores(scores[task2] - mean_scores_over_all_tasks, axes[i][2])
        viz_scores(get_masked_image(X_test, scores, task1, task2, n_to_erase), axes[i][3])
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    parser.add_argument('--start', default=0, type=int, help='which data to start')
    parser.add_argument('--end', default=200, type=int, help='which data to end')

    opt = parser.parse_args()
    opt.exp_name = 'run_exp_mnist'
    opt.data = 'mnist'
    init_config(opt)

    dataset = build_reg_dataset(opt, train=False)

    primary_log_dir = opt.log_dir
    for idx in range(opt.start, opt.end):
        data, target = dataset[idx]
        similar_target = target2similar[target]
        opt.logger.info(f'Choosing testset of {idx}, target={target}, similar_target={similar_target}...')

        opt.log_dir = f'{primary_log_dir}/{target}/{idx}'
        if not os.path.isdir(opt.log_dir):
            os.makedirs(opt.log_dir)

        method_names = ['gradient', '|gradient|', 'gradientXinput', '|gradient|Xinput', 'DeepLIFT', 'LRP', 'Grad-FBST', 'gradientXinput-FBST', 'DeepLIFT-FBST', 'LRP-FBST']
        method_to_task_to_scores = {method_name: {} for method_name in method_names}

        for task_idx in range(0, opt.n_targets):
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'nn', 'nn_1'
            opt.interpret_method = 'gradient'
            opt.algorithm = 'mean'
            scores = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
            method_to_task_to_scores['gradient'][task_idx] = scores[idx]

        for task_idx in range(0, opt.n_targets):
            method_to_task_to_scores['|gradient|'][task_idx] = np.abs(method_to_task_to_scores['gradient'][task_idx])

        for task_idx in range(0, opt.n_targets):
            method_to_task_to_scores['gradientXinput'][task_idx] = method_to_task_to_scores['gradient'][task_idx] * data.cpu().numpy()

        for task_idx in range(0, opt.n_targets):
            method_to_task_to_scores['|gradient|Xinput'][task_idx] = method_to_task_to_scores['|gradient|'][task_idx] * data.cpu().numpy()

        for task_idx in range(0, opt.n_targets):
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'nn', 'nn_1'
            opt.interpret_method = 'DeepLIFT'
            opt.algorithm = 'mean'
            scores = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
            method_to_task_to_scores['DeepLIFT'][task_idx] = scores[idx]

        for task_idx in range(0, opt.n_targets):
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'nn', 'nn_1'
            opt.interpret_method = 'LRP'
            opt.algorithm = 'mean'
            scores = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
            method_to_task_to_scores['LRP'][task_idx] = scores[idx]

        init_log_dir = opt.log_dir
        for task_idx in range(0, opt.n_targets):
            timestamp = FBST2timestamps['Grad-FBST'][task_idx]
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
            opt.interpret_method = 'gradient'
            opt.algorithm = 'p_s'
            opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
            scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 999))  # (n_data, **n_features)
            method_to_task_to_scores['Grad-FBST'][task_idx] = scores[idx]
        opt.log_dir = init_log_dir

        # for task_idx in range(0, opt.n_targets):
        #     method_to_task_to_scores['Grad-FBSTXinput'][task_idx] = method_to_task_to_scores['Grad-FBST'][task_idx] * data.cpu().numpy()

        init_log_dir = opt.log_dir
        for task_idx in range(0, opt.n_targets):
            timestamp = FBST2timestamps['gradientXinput-FBST'][task_idx]
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
            opt.interpret_method = 'gradientXinput'
            opt.algorithm = 'p_s'
            opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
            scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 199))  # (n_data, **n_features)
            method_to_task_to_scores['gradientXinput-FBST'][task_idx] = scores[idx]
        opt.log_dir = init_log_dir

        init_log_dir = opt.log_dir
        for task_idx in range(0, opt.n_targets):
            timestamp = FBST2timestamps['DeepLIFT-FBST'][task_idx]
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
            opt.interpret_method = 'DeepLIFT'
            opt.algorithm = 'p_s'
            opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
            scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 999))  # (n_data, **n_features)
            method_to_task_to_scores['DeepLIFT-FBST'][task_idx] = scores[idx]
        opt.log_dir = init_log_dir

        init_log_dir = opt.log_dir
        for task_idx in range(0, opt.n_targets):
            timestamp = FBST2timestamps['LRP-FBST'][task_idx]
            opt.y_index = task_idx
            opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
            opt.interpret_method = 'LRP'
            opt.algorithm = 'p_s'
            opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
            scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 999))  # (n_data, **n_features)
            method_to_task_to_scores['LRP-FBST'][task_idx] = scores[idx]
        opt.log_dir = init_log_dir

        plot_two_way_figures(np.array(dataset.dataset.data)[idx], target, similar_target, method_names, n_to_erase=157,
                             method_to_task_to_scores=method_to_task_to_scores,
                             save_file=generate_mnist_feature_importance_filename(opt, idx))

        for method_name in method_names:
            writer = pd.ExcelWriter(f'{opt.log_dir}/{method_name}_{idx}.xlsx')
            pd_bayes_factors_similar_target = pd.DataFrame(np.squeeze(method_to_task_to_scores[method_name][similar_target]))
            pd_bayes_factors_similar_target.to_excel(writer, sheet_name=f'target{similar_target}', index=False, header=False)
            pd_bayes_factors_target = pd.DataFrame(np.squeeze(method_to_task_to_scores[method_name][target]))
            pd_bayes_factors_target.to_excel(writer, sheet_name=f'target{target}', index=False, header=False)
            writer.close()

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
