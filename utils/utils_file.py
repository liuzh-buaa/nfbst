import os.path


def generate_log_filename(opt):
    return f'{opt.log_dir}/{opt.timestamp}.log'


def generate_data_filename(opt, last=False, suffix=""):
    if last:
        return f'{opt.data_dir}/data{suffix}.txt'
    else:
        return f'{opt.log_dir}/data{suffix}.txt'


def generate_targets_filename(opt, last=False, suffix=""):
    if last:
        return f'{opt.data_dir}/targets{suffix}.txt'
    else:
        return f'{opt.log_dir}/targets{suffix}.txt'


def generate_noise_filename(opt, last=False, suffix=""):
    if last:
        return f'{opt.data_dir}/noise{suffix}.txt'
    else:
        return f'{opt.log_dir}/noise{suffix}.txt'


def generate_data_model_filename(opt, last=False):
    if last:
        return f'{opt.data_dir}/gen_model'
    else:
        return f'{opt.log_dir}/gen_model'


def generate_f0_importance_filename(opt):
    return f'{opt.log_dir}/f0_importance.txt'


def generate_binary_global_label_filename(opt, last=False):
    if last:
        return f'{opt.data_dir}/binary_label_global.txt'
    else:
        return f'{opt.log_dir}/binary_label_global.txt'


def generate_binary_local_label_filename(opt, last=False):
    if last:
        return f'{opt.data_dir}/binary_label_local_{opt.eps:.4f}.txt'
    else:
        return f'{opt.log_dir}/binary_label_local_{opt.eps:.4f}.txt'


def generate_binary_local_label_excel_filename(opt, last=False):
    if last:
        return f'{opt.data_dir}/binary_label_local.xlsx'
    else:
        return f'{opt.log_dir}/binary_label_local.xlsx'


def generate_feature_margin_distri_filename(opt, feature, last=False):
    if last:
        new_dir = f'{opt.data_dir}/feature_margin_distri'
    else:
        new_dir = f'{opt.log_dir}/feature_margin_distri'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/hist_{feature}.svg'


def generate_feature_mutual_distri_filename(opt, feature1, feature2, last=False):
    if last:
        new_dir = f'{opt.data_dir}/feature_mutual_distri'
    else:
        new_dir = f'{opt.log_dir}/feature_mutual_distri'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/scatter_{feature1}_{feature2}.svg'


def generate_model_filename(opt, model_name, last=False):
    if last:
        new_dir = f'{opt.results_dir}/models'
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        return f'{new_dir}/{model_name}'
    else:
        return f'{opt.log_dir}/{model_name}'


def generate_history_filename(opt, model_name):
    return f'{opt.log_dir}/history_{model_name}.npy'


def generate_history_figname(opt, model_name):
    return f'{opt.log_dir}/history_{model_name}.svg'


def generate_grad_distri_filename(opt, data, last=False):
    if last:
        new_dir = f'{opt.results_dir}/grad_distri/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/grad_distri/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/grad_distri_{data}.npy'
    else:
        return f'{new_dir}/grad_distri_{data}_y{opt.y_index}.npy'


def generate_statistic_sample_filename(opt, sample, last=False):
    if last:
        new_dir = f'{opt.results_dir}/{opt.interpret_method}_samples/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/{opt.interpret_method}_samples/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/{opt.interpret_method}_sample{sample}.npy'
    else:
        return f'{new_dir}/{opt.interpret_method}_sample{sample}_y{opt.y_index}.npy'


def generate_statistic_distri_filename(opt, data, last=False):
    if last:
        new_dir = f'{opt.results_dir}/{opt.interpret_method}_distri/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/{opt.interpret_method}_distri/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/{opt.interpret_method}_distri_{data}.npy'
    else:
        return f'{new_dir}/{opt.interpret_method}_distri_{data}_y{opt.y_index}.npy'


def generate_kde_model_filename(opt, feature, last=False):
    if last:
        new_dir = f'{opt.results_dir}/kde_models/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/kde_models/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/kde_{feature}.pkl'
    else:
        return f'{new_dir}/kde_{feature}_y{opt.y_index}.pkl'


def generate_bayes_factors_filename(opt, last=False):
    if last:
        new_dir = f'{opt.results_dir}/bayes_factors/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/bayes_factors/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}.npy'
    else:
        return f'{new_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}_y{opt.y_index}.npy'


def generate_bayes_factors_cache_filename(opt, start, end):
    new_dir = f'{opt.log_dir}/bayes_factors/cache_{start}_{end}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}.npy'
    else:
        return f'{new_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}_y{opt.y_index}.npy'


def generate_bandwidths_cache_filename(opt, start, end):
    new_dir = f'{opt.log_dir}/bayes_factors/cache_{start}_{end}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/bandwidths_{opt.model_name}.xlsx'
    else:
        return f'{new_dir}/bandwidths_{opt.model_name}_y{opt.y_index}.xlsx'


def generate_bayes_factors_excel_filename(opt, last=False):
    if last:
        new_dir = f'{opt.results_dir}/bayes_factors/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/bayes_factors/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}.xlsx'


def generate_bayes_factors_thresholds_curve_filename(opt, feature):
    new_dir = f'{opt.log_dir}/bayes_factors_thresholds'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/curve_{opt.model_name}_{feature}.svg'


def generate_bayes_factors_thresholds_area_filename(opt):
    new_dir = f'{opt.log_dir}/bayes_factors_thresholds'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/thresholds_{opt.model_name}.svg'


def generate_bayes_factors_thresholds_excel_filename(opt, thresholds='thresholds1'):
    new_dir = f'{opt.log_dir}/bayes_factors_thresholds'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/{thresholds}_{opt.model_name}.xlsx'


def generate_local_roc_curve_filename(opt, feature):
    new_dir = f'{opt.log_dir}/local_roc_curves'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/roc_curve_{opt.model_name}_{feature}.svg'


def generate_auc_curve_filename(opt):
    return f'{opt.log_dir}/auc_curve_{opt.model_name}_{opt.data}_{opt.eps}.svg'


def generate_local_auc_excel_filename(opt, last=False):
    if last:
        new_dir = f'{opt.results_dir}/local_auc/{opt.model_name}'

        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

    else:
        new_dir = f'{opt.log_dir}'

    return f'{new_dir}/local_auc_{opt.model_name}__{opt.data}_{opt.eps}.xlsx'


def generate_global_roc_curve_filename(opt, threshold_name, threshold):
    new_dir = f'{opt.log_dir}/global_roc_curves'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/roc_curve_{opt.model_name}_{threshold_name}_{threshold:.4f}.svg'


def generate_global_auc_excel_filename(opt):
    return f'{opt.log_dir}/global_auc_{opt.model_name}.xlsx'


def generate_feature_distri_filename(opt, threshold_name, threshold, feature):
    new_dir = f'{opt.log_dir}/feature_distri/{feature}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return f'{new_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}_{threshold_name}_{threshold:.4f}.svg'


def generate_mnist_feature_importance_filename(opt, idx):
    return f'{opt.log_dir}/saliency_{idx}.svg'


def generate_deeplift_filename(opt, last=False):
    if last:
        new_dir = f'{opt.results_dir}/deeplift/{opt.model_name}'
    else:
        new_dir = f'{opt.log_dir}/deeplift/{opt.model_name}'

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    if opt.y_index == 0:
        return f'{new_dir}/deeplift.npy'
    else:
        return f'{new_dir}/deeplift_y{opt.y_index}.npy'


def generate_bayes_factors_distri_filename(opt, feature):
    return f'{opt.log_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}_distri_{feature}.svg'


def generate_bayes_factors_scatter_filename(opt, feature):
    return f'{opt.log_dir}/{opt.interpret_method}_{opt.algorithm}_{opt.model_name}_scatter_{feature}.svg'


def generate_X_Y_scatter_filename(opt, feature, target):
    return f'{opt.log_dir}/{feature}_{target}.svg'


def generate_mnist_or_cifar10_targets_filename(opt, train=False, last=False):
    if last:
        if train:
            return f'{opt.data_dir}/targets_train.xlsx'
        else:
            return f'{opt.data_dir}/targets_test.xlsx'
    else:
        if train:
            return f'{opt.log_dir}/targets_train.xlsx'
        else:
            return f'{opt.log_dir}/targets_test.xlsx'


def generate_local_2_global_QGI_filename(opt, feature):
    return f'{opt.log_dir}/{feature}_{opt.interpret_method}_{opt.algorithm}_{opt.model_name}.svg'
