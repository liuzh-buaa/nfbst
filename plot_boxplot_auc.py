import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman')
    data = 'simulation_v4'
    timestamps = {
        '0.001': '2023-05-10 21-28-33'
    }

    # data = 'simulation_v12'
    # timestamps = {
    #     '0.001': '2023-05-10 21-29-28',
    # }

    aucs = {
        eps: pd.read_excel(f'data/{data}/log/joint_different_curves/{timestamp}/local_auc_all_models.xlsx').to_numpy()
        for eps, timestamp in timestamps.items()
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for eps in aucs.keys():
        plt.figure(figsize=(8, 8))
        f = plt.boxplot([auc[1:51] for auc in aucs[eps]], patch_artist=True,)
                    # labels=['DeepLIFT(nn)', 'LRP(nn)', 'gradient(NN)', 'gradientXinput(nn)', 'LIME(nn)',
                    #         'DeepLIFT(FBST)', 'LRP(FBST)', 'gradient(FBST)', 'gradientXinput(FBST)', 'LIME(FBST)'])
        for box, color in zip(f['boxes'], colors):
            box.set_facecolor(color)

        # plt.legend(bbox_to_anchor=(1.005, 0), loc=3)
        # plt.legend(loc='best')
        plt.xlabel('methods', fontsize=15)
        plt.ylabel('AUC', fontsize=15)
        plt.yticks(fontproperties='Times New Roman', size=15)
        plt.xticks(fontproperties='Times New Roman', size=15)
        # plt.ylim((0, 1.0))
        plt.title(f'eps={eps}', fontsize=20)
        plt.savefig(f'boxplot auc of {data}({eps}).png', bbox_inches='tight')
        plt.close()
