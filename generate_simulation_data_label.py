"""
    【Binary Label Generation Process】
    Generate binary label for instance-wise significance and global significance.
    Here, `eps` only affects instance-wise significance but not global significance.

    `python generate_simulation_data_label.py --log True --data simulation_v3 --eps 0.001`
    `python generate_simulation_data_label.py --log True --data simulation_v4/simulation_v12 --eps 0.001/0.01/0.02/0.03/0.04/0.05`
    `python generate_simulation_data_label.py --log True --data energy_16_3`
"""
import shutil
import time

import numpy as np
import pandas as pd
import torch

from utils.utils_file import generate_data_filename, generate_binary_local_label_excel_filename, \
    generate_data_model_filename, generate_f0_importance_filename, \
    generate_binary_global_label_filename, generate_binary_local_label_filename
from utils.utils_parser import DefaultArgumentParser, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    parser.add_argument('--eps', default=0, type=float, help='eps for local binary label')
    parser.add_argument('--interpret_method', default='gradient', type=str, help='feature importance metric')
    parser.add_argument('--y_index', default=0, type=int, help='gradient to which output (for multi-outputs)')

    opt = parser.parse_args()
    opt.exp_name = 'generate_simulation_data_label'
    init_config(opt)

    data = np.loadtxt(generate_data_filename(opt, last=True))

    if opt.data in ['simulation_v1']:
        insignificant_features = [_ for _ in range(50, 100)]

        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

        special_features_indices = np.argwhere(data[:, 0] < 0)  # x0
        special_features_indices = [_[0] for _ in special_features_indices]
        local_labels[special_features_indices, 0] = 0

    elif opt.data in ['simulation_v2', 'simulation_v4', 'simulation_v7']:
        from models.nn.nn import NN

        model = NN(hidden=[20, 20, 20], in_features=opt.n_features, out_features=opt.n_targets).to(opt.device)
        model.load_state_dict(torch.load(generate_data_model_filename(opt, True), map_location=opt.device))

        insignificant_features = [_ for _ in range(50, 100)]

        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = model.get_interpret(torch.tensor(data, dtype=torch.float32).to(opt.device),
                                           opt).detach().cpu().numpy()
        np.savetxt(generate_f0_importance_filename(opt), local_labels, fmt='%9f')

        local_labels[np.abs(local_labels) <= opt.eps] = 0
        local_labels[local_labels != 0] = 1

        assert np.sum(local_labels[:, insignificant_features]) == 0

    elif opt.data in ['simulation_v12']:
        from models.nn.nn import NN

        model = NN(hidden=[16, 16, 16], in_features=opt.n_features, out_features=opt.n_targets).to(opt.device)
        model.load_state_dict(torch.load(generate_data_model_filename(opt, True), map_location=opt.device))

        insignificant_features = [_ for _ in range(50, 100)]

        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = model.get_interpret(torch.tensor(data, dtype=torch.float32).to(opt.device),
                                           opt).detach().cpu().numpy()
        np.savetxt(generate_f0_importance_filename(opt), local_labels, fmt='%9f')

        local_labels[np.abs(local_labels) <= opt.eps] = 0
        local_labels[local_labels != 0] = 1

        assert np.sum(local_labels[:, insignificant_features]) == 0

    elif opt.data in ['simulation_v3']:
        insignificant_features = [7]

        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels_x0 = 2 * data[:, 0]
        local_labels_x1 = data[:, 2]
        local_labels_x2 = data[:, 1]
        local_labels_x3 = -np.sin(data[:, 3])
        local_labels_x4 = data[:, 5] * np.exp(data[:, 4] * data[:, 5])
        local_labels_x5 = data[:, 4] * np.exp(data[:, 4] * data[:, 5])
        local_labels_x6 = 0.1 * np.ones_like(data[:, 6])
        local_labels_x7 = np.zeros_like(data[:, 7])

        local_labels = np.stack(
            [local_labels_x0, local_labels_x1, local_labels_x2, local_labels_x3, local_labels_x4, local_labels_x5,
             local_labels_x6, local_labels_x7], axis=1)
        local_labels[np.abs(local_labels) <= opt.eps] = 0
        local_labels[local_labels != 0] = 1

        assert np.sum(local_labels[:, insignificant_features]) == 0

    elif opt.data in ['simulation_v8', 'simulation_v9']:

        global_labels = np.ones(opt.n_features)

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:1000, :50] = 0
        local_labels[5000:, 50:] = 0

    elif opt.data in ['simulation_v5']:
        from models.nn.nn import NN

        model = NN(hidden=[20, 20], in_features=opt.n_features, out_features=opt.n_targets).to(opt.device)
        model.load_state_dict(torch.load(generate_data_model_filename(opt, True), map_location=opt.device))

        insignificant_features = [_ for _ in range(50, 100)]

        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = model.get_interpret(torch.tensor(data, dtype=torch.float32).to(opt.device),
                                           opt).detach().cpu().numpy()
        np.savetxt(generate_f0_importance_filename(opt), local_labels, fmt='%9f')

        local_labels[np.abs(local_labels) <= opt.eps] = 0
        local_labels[local_labels != 0] = 1

        assert np.sum(local_labels[:, insignificant_features]) == 0

    elif opt.data in ['simulation_v6']:
        from models.nn.nn import NN

        model = NN(hidden=[50, 50], in_features=opt.n_features, out_features=opt.n_targets).to(opt.device)
        model.load_state_dict(torch.load(generate_data_model_filename(opt, True), map_location=opt.device))

        insignificant_features = [_ for _ in range(50, 100)]

        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = model.get_interpret(torch.tensor(data, dtype=torch.float32).to(opt.device),
                                           opt).detach().cpu().numpy()
        np.savetxt(generate_f0_importance_filename(opt), local_labels, fmt='%9f')

        local_labels[np.abs(local_labels) <= opt.eps] = 0
        local_labels[local_labels != 0] = 1

        assert np.sum(local_labels[:, insignificant_features]) == 0

    elif opt.data.startswith('boston'):

        insignificant_features = [_ for _ in range(13, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('concrete') or opt.data.startswith('kin8nm'):

        insignificant_features = [_ for _ in range(8, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('energy') or opt.data.startswith('efficient'):

        insignificant_features = [5]
        insignificant_features.extend([_ for _ in range(8, opt.n_features)])
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('naval_y1') or opt.data.startswith('naval_y2'):

        insignificant_features = [_ for _ in range(16, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('power'):

        insignificant_features = [_ for _ in range(4, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('wine'):

        insignificant_features = [_ for _ in range(11, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('protein'):

        insignificant_features = [_ for _ in range(9, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    elif opt.data.startswith('yacht'):

        insignificant_features = [_ for _ in range(6, opt.n_features)]
        global_labels = np.ones(opt.n_features)
        global_labels[insignificant_features] = 0

        local_labels = np.ones((opt.n_samples, opt.n_features))
        local_labels[:, insignificant_features] = 0

    else:
        raise NotImplementedError(f'No such data type of {opt.data} for generating label.')

    np.savetxt(generate_binary_global_label_filename(opt, False), global_labels, fmt="%d")
    np.savetxt(generate_binary_local_label_filename(opt, False), local_labels, fmt="%d")

    print(f'==> Analysing binary labels for each feature...')
    p_f_sum = [[np.sum(local_labels[:, i] == 1), np.sum(local_labels[:, i] == 0)] for i in range(opt.n_features)]
    writer = pd.ExcelWriter(generate_binary_local_label_excel_filename(opt))
    df = pd.DataFrame(data=p_f_sum, index=range(opt.n_features), columns=['P', 'F'])
    df.to_excel(writer, )
    writer.close()

    print(df)

    if opt.log:
        print('Copying binary_label.txt from `timestamp` to `data`')
        shutil.copyfile(generate_binary_local_label_filename(opt, False),
                        generate_binary_local_label_filename(opt, True))
        shutil.copyfile(generate_binary_global_label_filename(opt, False),
                        generate_binary_global_label_filename(opt, True))
        shutil.copyfile(generate_binary_local_label_excel_filename(opt, False),
                        generate_binary_local_label_excel_filename(opt, True))

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
