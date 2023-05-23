"""
    【Data Generation Process】
    Generate different types of simulation datasets.

    `python generate_simulation_data.py --log True --data simulation_v4`
"""
import os.path
import shutil
import time

from datasets.generate_simulation_data import generate_simulation_data_v1, generate_simulation_data_v2, \
    generate_simulation_data_v3, generate_simulation_data_v4, generate_simulation_data_v5, generate_simulation_data_v6, \
    generate_simulation_data_v7, generate_simulation_data_v8, generate_simulation_data_v9, generate_simulation_data_v10, \
    generate_simulation_data_v11, generate_simulation_data_v12, generate_uci_boston_data, generate_uci_concrete_data, \
    generate_uci_energy_data, generate_uci_kin8nm_data, generate_uci_naval_data, generate_uci_power_data, \
    generate_uci_wine_data, generate_uci_protein_data, generate_uci_yacht_data, generate_uci_efficient_data
from datasets.visualize_feature_margin_distr import visualize_feature_margin_distr
from utils.utils_file import generate_data_filename, generate_targets_filename, generate_noise_filename, \
    generate_data_model_filename
from utils.utils_parser import DefaultArgumentParser, report_args, init_config

if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    opt = parser.parse_args()
    opt.exp_name = 'generate_simulation_data'
    init_config(opt)

    n_max = 5
    print(f'==> Generating regression data...')
    if opt.data == 'simulation_v1':
        X, y, noise = generate_simulation_data_v1(opt)
    elif opt.data == 'simulation_v2':
        X, y, noise = generate_simulation_data_v2(opt)
    elif opt.data == 'simulation_v3':
        X, y, noise = generate_simulation_data_v3(opt)
    elif opt.data == 'simulation_v4':
        X, y, noise = generate_simulation_data_v4(opt)
    elif opt.data == 'simulation_v5':
        X, y, noise = generate_simulation_data_v5(opt)
    elif opt.data == 'simulation_v6':
        X, y, noise = generate_simulation_data_v6(opt)
    elif opt.data == 'simulation_v7':
        X, y, noise = generate_simulation_data_v7(opt)
    elif opt.data == 'simulation_v8':
        X, y, noise = generate_simulation_data_v8(opt)
    elif opt.data == 'simulation_v9':
        X, y, noise = generate_simulation_data_v9(opt)
    elif opt.data == 'simulation_v10':
        X, y, noise = generate_simulation_data_v10(opt)
    elif opt.data == 'simulation_v11':
        X, y, noise = generate_simulation_data_v11(opt)
    elif opt.data == 'simulation_v12':
        X, y, noise = generate_simulation_data_v12(opt)
    elif opt.data.startswith('boston'):
        X, y, noise = generate_uci_boston_data(opt)
    elif opt.data.startswith('concrete'):
        X, y, noise = generate_uci_concrete_data(opt)
    elif opt.data.startswith('energy'):
        X, y, noise = generate_uci_energy_data(opt)
    elif opt.data.startswith('kin8nm'):
        X, y, noise = generate_uci_kin8nm_data(opt)
    elif opt.data.startswith('naval_y1'):
        X, y, noise = generate_uci_naval_data(opt, y1=True)
    elif opt.data.startswith('naval_y2'):
        X, y, noise = generate_uci_naval_data(opt, y2=True)
    elif opt.data.startswith('power'):
        X, y, noise = generate_uci_power_data(opt)
    elif opt.data.startswith('wine'):
        X, y, noise = generate_uci_wine_data(opt)
    elif opt.data.startswith('protein'):
        X, y, noise = generate_uci_protein_data(opt)
    elif opt.data.startswith('yacht'):
        X, y, noise = generate_uci_yacht_data(opt)
    elif opt.data.startswith('efficient'):
        X, y, noise = generate_uci_efficient_data(opt)
    else:
        raise NotImplementedError(f'No such generating data of {opt.data}')

    print(f'==> Visualizing feature margin distribution...')
    visualize_feature_margin_distr(opt, X, n_max)

    if opt.log:
        print('Copying data.txt and targets.txt from `timestamp` to `data`...')
        shutil.copyfile(generate_data_filename(opt, False), generate_data_filename(opt, True))
        shutil.copyfile(generate_targets_filename(opt, False), generate_targets_filename(opt, True))
        shutil.copyfile(generate_noise_filename(opt, False), generate_noise_filename(opt, True))

        print('Copying log from `timestamp` to `data`...')
        with open(f'{opt.data_dir}/last.log', 'w') as fp:
            fp.write(opt.timestamp)

        print('Copying data generating model from `timestamp` to `data`...')
        if os.path.exists(generate_data_model_filename(opt, False)):
            shutil.copyfile(generate_data_model_filename(opt, False),
                            generate_data_model_filename(opt, True))

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
