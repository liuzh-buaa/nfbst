"""
    【Plot Average AUC under Different Eps】

    `python plot_avg_auc.py`
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman')
    timestamps = {
        '0.001': '2023-05-10 21-28-33',
        '0.01': '2023-05-16 14-44-21',
        '0.02': '2023-05-16 14-44-52',
        '0.03': '2023-05-16 14-45-16',
        '0.04': '2023-05-16 14-45-32',
        '0.05': '2023-05-16 14-45-54',
    }

    aucs_simulation_v4 = {
        eps: pd.read_excel(f'data/simulation_v4/log/joint_different_curves/{timestamp}/local_auc_all_models.xlsx').to_numpy()
        for eps, timestamp in timestamps.items()
    }

    timestamps = {
        '0.001': '2023-05-10 21-29-28',
        '0.01': '2023-05-16 14-46-20',
        '0.02': '2023-05-16 14-46-47',
        '0.03': '2023-05-16 14-47-01',
        '0.04': '2023-05-16 14-47-16',
        '0.05': '2023-05-16 14-47-32',
    }
    aucs_simulation_v12 = {
        eps: pd.read_excel(
            f'data/simulation_v12/log/joint_different_curves/{timestamp}/local_auc_all_models.xlsx').to_numpy()
        for eps, timestamp in timestamps.items()
    }

    plt.figure(figsize=(15, 6))

    aucs = aucs_simulation_v4
    eps = aucs.keys()

    plt.subplot(1, 2, 1)
    plt.plot(eps, [np.mean(aucs[x][0][1:51]) for x in eps], color='#0072BD', marker='o', linestyle='--', label='DeepLIFT', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][1][1:51]) for x in eps], color='#D95319', marker='v', linestyle='--', label='LRP', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][2][1:51]) for x in eps], color='#EDB120', marker='^', linestyle='--', label='Gradient', linewidth=3)
    # plt.plot(eps, [np.mean(aucs[x][3][1:51]) for x in eps], color='#D95319', marker='v', linestyle='--', label='GradXInput', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][4][1:51]) for x in eps], color='#7E2F8E', marker='s', linestyle='--', label='LIME', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][5][1:51]) for x in eps], color='#0072BD', marker='o', linestyle='-', label='DeepLIFT-nFBST', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][6][1:51]) for x in eps], color='#D95319', marker='v', linestyle='-', label='LRP-nFBST', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][7][1:51]) for x in eps], color='#EDB120', marker='^', linestyle='-', label='Grad-nFBST', linewidth=3)
    # plt.plot(eps, [np.mean(aucs[x][8][1:51]) for x in eps], color='#D95319', marker='v', linestyle='-', label='GradXInput-nFBST', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][9][1:51]) for x in eps], color='#7E2F8E', marker='s', linestyle='-', label='LIME-nFBST', linewidth=3)

    plt.legend(bbox_to_anchor=(0.05, 1.05), loc='lower left', ncol=4, prop={'size': 20})
    # plt.legend(loc='best')
    plt.xlabel('Eps', fontproperties='Times New Roman', fontsize=20)
    plt.ylabel('Average AUC', fontproperties='Times New Roman', fontsize=20)
    plt.xlim(0.001, 5)
    plt.xticks([0.001, 1, 2, 3, 4, 5], [0.001, 0.01, 0.02, 0.03, 0.04, 0.05], fontproperties='Times New Roman', size=20)
    plt.yticks(np.arange(0.55, 0.8, 0.05), fontproperties='Times New Roman', size=20)
    plt.title('Dataset 1', fontproperties='Times New Roman', fontsize=20)

    aucs = aucs_simulation_v12
    eps = aucs.keys()

    plt.subplot(1, 2, 2)
    plt.plot(eps, [np.mean(aucs[x][0][1:51]) for x in eps], color='#0072BD', marker='o', linestyle='--', label='DeepLIFT', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][1][1:51]) for x in eps], color='#D95319', marker='v', linestyle='--', label='LRP', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][2][1:51]) for x in eps], color='#EDB120', marker='^', linestyle='--', label='Gradient', linewidth=3)
    # plt.plot(eps, [np.mean(aucs[x][3][1:51]) for x in eps], color='#D95319', marker='v', linestyle='--', label='GradXInput', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][4][1:51]) for x in eps], color='#7E2F8E', marker='s', linestyle='--', label='LIME', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][5][1:51]) for x in eps], color='#0072BD', marker='o', linestyle='-', label='DeepLIFT-nFBST', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][6][1:51]) for x in eps], color='#D95319', marker='v', linestyle='-', label='LRP-nFBST', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][7][1:51]) for x in eps], color='#EDB120', marker='^', linestyle='-', label='Grad-nFBST', linewidth=3)
    # plt.plot(eps, [np.mean(aucs[x][8][1:51]) for x in eps], color='#D95319', marker='v', linestyle='-', label='GradXInput-nFBST', linewidth=3)
    plt.plot(eps, [np.mean(aucs[x][9][1:51]) for x in eps], color='#7E2F8E', marker='s', linestyle='-', label='LIME-nFBST', linewidth=3)

    plt.xlabel('Eps', fontproperties='Times New Roman', fontsize=20)
    plt.ylabel('Average AUC', fontproperties='Times New Roman', fontsize=20)
    plt.xlim(0.001, 5)
    plt.xticks([0.001, 1, 2, 3, 4, 5], [0.001, 0.01, 0.02, 0.03, 0.04, 0.05], fontproperties='Times New Roman', size=20)
    plt.yticks(np.arange(0.55, 0.75, 0.05), fontproperties='Times New Roman', size=20)
    plt.title('Dataset 2', fontproperties='Times New Roman', fontsize=20)

    plt.savefig(f'Avg AUC of Dataset1 2.svg', bbox_inches='tight')
