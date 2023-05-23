import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve


def visualize_loss(history, model_name, save_file):
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.title(f'{model_name} training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_file)
    plt.close()


def plot_curve(x, y, title, savefile, xlabel, ylabel, xlim=None, ylim=None, diagonal=True):
    if xlim is None:
        xlim = [0.0, 1.0]
    if ylim is None:
        ylim = [0.0, 1.0]

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.xlim(xlim)
    plt.ylim(ylim)

    if diagonal:
        plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='navy', lw=lw, linestyle='--')

    plt.plot(x, y, color='darkorange', lw=lw)
    plt.xlabel(xlabel=xlabel, fontsize=15)
    plt.ylabel(ylabel=ylabel, fontsize=15)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.title(title, fontsize=15)

    plt.savefig(savefile, bbox_inches='tight')
    plt.close()


def plot_curves(x, method2y, title, savefile, xlabel, ylabel, xlim=None, ylim=None, diagonal=True):
    if xlim is None:
        xlim = [0.0, 1.0]
    if ylim is None:
        ylim = [0.0, 1.0]

    lw = 2
    plt.figure(figsize=(8, 8))
    plt.xlim(xlim)
    plt.ylim(ylim)
    ticks, labels = plt.xticks([0, 10, 20, 30, 40, 49], [0, 10, 20, 30, 40, 49])
    print(ticks)
    print(labels)

    if diagonal:
        plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], lw=lw, linestyle='--')

    for method, y in method2y.items():
        plt.plot(x, y, lw=lw, label=f'{method}')  # x-FPR,y-TPR

    plt.xlabel(xlabel=xlabel, fontsize=25)
    plt.ylabel(ylabel=ylabel, fontsize=25)
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)
    plt.legend(loc="lower right", labelspacing=0.3, prop={'family': 'Times New Roman', 'size': 20})
    # plt.title(title, fontsize=20)

    plt.savefig(savefile, bbox_inches='tight')
    plt.close()


def calculate_auc(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def plot_roc_curve(y, prob, title, savefile):
    fpr, tpr, _ = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  # x-FPR,y-TPR
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig(savefile)
    plt.close()

    return roc_auc


def plot_roc_curves(y, method2prob, title, savefile):
    lw = 2
    plt.figure(figsize=(10, 10))

    for method, prob in method2prob.items():
        fpr, tpr, _ = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f'{method} (AUC = %0.3f)' % roc_auc)  # x-FPR,y-TPR

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.savefig(savefile)
    plt.close()


def plot_area(x, y, idx1, idx2, title, savefile, xlabel, ylabel):
    plt.figure(figsize=(6, 6))

    y1_max = np.max(y[idx1, :], axis=0)
    y1_min = np.min(y[idx1, :], axis=0)

    y2_max = np.max(y[idx2, :], axis=0)
    y2_min = np.min(y[idx2, :], axis=0)

    plt.fill_between(x, y1_min, y1_max, label='significant')
    plt.fill_between(x, y2_min, y2_max, label='insignificant')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel(xlabel=xlabel, fontsize=20)
    plt.ylabel(ylabel=ylabel, fontsize=20)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.legend(loc="upper left", prop={'family': 'Times New Roman', 'size': 18})
    if title == 'gaussian_e':
        title = 'Grad-FBST'
    plt.title(title, fontsize=18)

    plt.savefig(savefile, bbox_inches='tight')
    plt.close()


def plot_feature_distribution(data, locs_insignificant, locs_significant, target, auxiliary, title, savefile):
    # insignificant_target_features = data[locs_insignificant, target]
    # insignificant_auxiliary_features = data[locs_insignificant, auxiliary]
    # plt.scatter(insignificant_target_features, insignificant_auxiliary_features, label='insignificant', alpha=0.2)
    #
    # significant_target_features = data[locs_significant, target]
    # significant_auxiliary_features = data[locs_significant, auxiliary]
    # plt.scatter(significant_target_features, significant_auxiliary_features, label='significant', alpha=0.2)
    #
    # plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 15})
    # plt.xlabel(xlabel=f'x{target}', fontsize=15)
    # plt.ylabel(ylabel=f'x{auxiliary}', fontsize=15)
    # plt.yticks(fontproperties='Times New Roman', size=15)
    # plt.xticks(fontproperties='Times New Roman', size=15)
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])

    insignificant_auxiliary_features = data[locs_insignificant, auxiliary]
    y = np.zeros_like(insignificant_auxiliary_features)
    plt.scatter(insignificant_auxiliary_features, y, label='insignificant', alpha=0.01)

    significant_auxiliary_features = data[locs_significant, auxiliary]
    y = np.ones_like(significant_auxiliary_features)
    plt.scatter(significant_auxiliary_features, y, label='significant', alpha=0.01)

    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 15})
    plt.xlabel(xlabel=f'x{auxiliary}', fontsize=15)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.xlim([-1, 1])
    plt.ylim([-0.05, 1.05])

    plt.title(title, fontsize=15)
    plt.savefig(savefile)
    plt.close()
