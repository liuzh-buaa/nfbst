"""
    【Saliency Map Comparison】
    Note this process may be slow, and we needn't run on the whole dataset.
    Please set the directory in the code.

    `python run_exp_mnist2.py`
"""
import time

import numpy as np
from matplotlib import pyplot as plt

from datasets.regdata import build_reg_dataset
from utils.utils_file import generate_bayes_factors_filename, generate_mnist_feature_importance_filename, \
    generate_bayes_factors_cache_filename
from utils.utils_parser import DefaultArgumentParser, init_config

FBST2timestamps = {
    'Grad-nFBST': {
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
    'DeepLIFT-nFBST': {
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
    'LRP-nFBST': {
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
    'GradXInput-nFBST': {
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
def plot_two_way_figures(X_test_total, indices, method_names, method_to_task_to_scores_indices, save_file):
    f, axes = plt.subplots(len(method_names) + 1, 10, figsize=(15, 10))
    for j, idx in enumerate(indices):
        X_test = X_test_total[idx]
        viz_scores(X_test, axes[0][j])
        method_to_task_to_scores = method_to_task_to_scores_indices[j]
        for i, method_name in enumerate(method_names):
            scores = method_to_task_to_scores[method_name]
            # mean_scores_over_all_tasks = np.mean(np.array([scores[i] for i in range(10)]), axis=0)
            mean_scores_over_all_tasks = 0
            viz_scores(scores - mean_scores_over_all_tasks, axes[i + 1][j])
    method_names.insert(0, "Image")
    for i, method_name in enumerate(method_names):
        axes[i][0].text(-4,
                        15,
                        method_name,
                        fontproperties='Times New Roman',
                        fontsize=25,
                        fontweight='bold',
                        verticalalignment="center",
                        horizontalalignment="right"
                        )
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()
    opt = parser.parse_args()
    opt.exp_name = 'run_exp_mnist2'
    opt.data = 'mnist'
    init_config(opt)

    dataset = build_reg_dataset(opt, train=False)

    indices = [3, 2, 186, 18, 4, 15, 11, 0, 84, 9]

    method_names = ['Gradient', '|Gradient|', 'GradientXInput', '|Gradient|XInput', 'DeepLIFT', 'LRP', 'Grad-nFBST',
                    'Grad-nFBSTXInput',
                    'GradXInput-nFBST', 'DeepLIFT-nFBST', 'LRP-nFBST']

    method_names = ['Gradient', '|Gradient|', 'Grad-nFBST', 'GradientXInput',
                    # 'Grad-nFBSTXInput',
                    'GradXInput-nFBST',
                    # '|Gradient|XInput',
                    'DeepLIFT',
                    'DeepLIFT-nFBST',
                    'LRP',
                    'LRP-nFBST']

    method_to_task_to_scores_indices = []
    for i, idx in enumerate(indices):
        data, target = dataset[idx]
        similar_target = target2similar[target]
        opt.logger.info(f'Choosing testset of {idx}, target={target}, similar_target={similar_target}...')
        if target == 2:
            target = similar_target
        method_to_task_to_scores = {method_name: {} for method_name in method_names}

        opt.y_index = target
        opt.model_type, opt.model_name = 'nn', 'nn_1'
        opt.interpret_method = 'gradient'
        opt.algorithm = 'mean'
        scores = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
        method_to_task_to_scores['Gradient'] = scores[idx]

        method_to_task_to_scores['|Gradient|'] = np.abs(method_to_task_to_scores['Gradient'])

        method_to_task_to_scores['GradientXInput'] = method_to_task_to_scores['Gradient'] * data.cpu().numpy()

        method_to_task_to_scores['|Gradient|XInput'] = method_to_task_to_scores['|Gradient|'] * data.cpu().numpy()

        opt.y_index = target
        opt.model_type, opt.model_name = 'nn', 'nn_1'
        opt.interpret_method = 'DeepLIFT'
        opt.algorithm = 'mean'
        scores = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
        method_to_task_to_scores['DeepLIFT'] = scores[idx]

        opt.y_index = target
        opt.model_type, opt.model_name = 'nn', 'nn_1'
        opt.interpret_method = 'LRP'
        opt.algorithm = 'mean'
        scores = np.load(generate_bayes_factors_filename(opt, last=True))  # (n_data, **n_features)
        method_to_task_to_scores['LRP'] = scores[idx]

        init_log_dir = opt.log_dir
        timestamp = FBST2timestamps['Grad-nFBST'][target]
        opt.y_index = target
        opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
        opt.interpret_method = 'gradient'
        opt.algorithm = 'p_s'
        opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
        scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 999))  # (n_data, **n_features)
        method_to_task_to_scores['Grad-nFBST'] = scores[idx]
        opt.log_dir = init_log_dir

        method_to_task_to_scores['Grad-nFBSTXInput'] = method_to_task_to_scores['Grad-nFBST'] * data.cpu().numpy()

        init_log_dir = opt.log_dir
        timestamp = FBST2timestamps['GradXInput-nFBST'][target]
        opt.y_index = target
        opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
        opt.interpret_method = 'gradientXinput'
        opt.algorithm = 'p_s'
        opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
        scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 199))  # (n_data, **n_features)
        method_to_task_to_scores['GradXInput-nFBST'] = scores[idx]
        opt.log_dir = init_log_dir

        init_log_dir = opt.log_dir
        timestamp = FBST2timestamps['DeepLIFT-nFBST'][target]
        opt.y_index = target
        opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
        opt.interpret_method = 'DeepLIFT'
        opt.algorithm = 'p_s'
        opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
        scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 999))  # (n_data, **n_features)
        method_to_task_to_scores['DeepLIFT-nFBST'] = scores[idx]
        opt.log_dir = init_log_dir

        init_log_dir = opt.log_dir
        timestamp = FBST2timestamps['LRP-nFBST'][target]
        opt.y_index = target
        opt.model_type, opt.model_name = 'gaussian', 'gaussian_e'
        opt.interpret_method = 'LRP'
        opt.algorithm = 'p_s'
        opt.log_dir = f'{opt.data_root}/log/get_bayes_factors/{timestamp}'
        scores = np.load(generate_bayes_factors_cache_filename(opt, 0, 999))  # (n_data, **n_features)
        method_to_task_to_scores['LRP-nFBST'] = scores[idx]
        opt.log_dir = init_log_dir

        method_to_task_to_scores_indices.append(method_to_task_to_scores)

    plot_two_way_figures(np.array(dataset.dataset.data), indices, method_names, method_to_task_to_scores_indices,
                         save_file=generate_mnist_feature_importance_filename(opt, 'compare'))

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
