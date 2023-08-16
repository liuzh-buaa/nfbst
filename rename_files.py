import os

directories = [
    'data/simulation_v12/results/gradient_distri/gaussian_1',
    # 'data/simulation_v12/results/gradient_distri/nn_1',
    # 'data/simulation_v12/results/gradient_distri/nn_2',
    # 'data/simulation_v12/results/gradient_distri/nn_3',
    # 'data/simulation_v12/results/gradient_distri/nn_4',
    # 'data/simulation_v12/results/gradient_distri/nn_e',
    # 'data/simulation_v4/results/gradient_distri/nn_1',
    # 'data/simulation_v4/results/gradient_distri/nn_2',
    # 'data/simulation_v4/results/gradient_distri/nn_3',
    # 'data/simulation_v4/results/gradient_distri/nn_4',
    # 'data/simulation_v4/results/gradient_distri/nn_e',
]
for directory in directories:
    filenames = os.listdir(directory)

    for filename in filenames:
        old_filename = os.path.join(directory, filename)
        new_filename = os.path.join(directory, "gradient" + filename[4:])
        print(f'{old_filename} --> {new_filename}')
        os.rename(old_filename, new_filename)
