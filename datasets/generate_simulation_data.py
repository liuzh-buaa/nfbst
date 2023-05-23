"""
    根据数据集生成不同的数据
    generate_reg_data_v{X} 生成公式和数据分布不同
    generate_boston_data 生成包含随机噪声的波士顿房价数据集
"""
import math
import numpy as np
import torch
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

from datasets.data_utils import run_baseline_lr, standardization
from utils.utils_file import generate_data_filename, generate_targets_filename, generate_noise_filename, \
    generate_data_model_filename


def expr_v1(x):
    y = 0 if x[0] < 0 else x[0]
    for i in range(1, 10):
        y += (x[i] ** (i + 1)) / (i + 1)
    for i in range(10, 20):
        y += math.sin(i * x[i]) / i
    for i in range(20, 30):
        y += math.cos(i * x[i]) / i
    for i in range(30, 40):
        y += math.exp(i * x[i]) / i / math.exp(i)
    for i in range(40, 50):
        y += math.log(x[i] + i)
    return y


def generate_simulation_data_v1(opt):
    X = np.random.uniform(-1, 1, size=(opt.n_samples, opt.n_features))
    y = np.array([expr_v1(sample) for sample in X])
    noise = np.random.normal(0, 0.1, size=(opt.n_samples,))
    y_noise = y + noise
    np.savetxt(generate_data_filename(opt, False), X, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise, fmt='%9f')
    opt.logger.info(f'MSE: {mean_squared_error(y, y_noise)}, noise: {np.sum(np.square(noise))}, '
                    f'lr: {mean_squared_error(y, run_baseline_lr(X, y_noise))}')

    return X, y_noise, noise


def generate_simulation_data_v2(opt):
    from models.nn.nn import NN
    model = NN(hidden=[20, 20, 20], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)

    w = model.get_layer1_w_data()
    w[:, 50:] = 0
    model.set_layer1_w_data(w)  # last 50 variables are set to 0

    var_e = 0.1
    noise = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)

    X = torch.randn(opt.n_samples, opt.n_features, device=opt.device)
    y = model.predict(X)
    y_noise = y + noise

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_numpy


def expr_v3(x):
    return 8 + x[0] ** 2 + x[1] * x[2] + math.cos(x[3]) + math.exp(x[4] * x[5]) + 0.1 * x[6]


def generate_simulation_data_v3(opt):
    X = np.random.uniform(-1, 1, size=(opt.n_samples, opt.n_features))
    y = np.array([expr_v3(sample) for sample in X])

    var_e = 1
    noise = np.random.normal(0, var_e, size=(opt.n_samples,))
    y_noise = y + noise

    np.savetxt(generate_data_filename(opt, False), X, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y, y_noise)}, noise={np.sum(np.square(noise))}, '
                    f'targets: {mean_squared_error(y_noise, np.zeros_like(y_noise))}, '
                    f'lr: {mean_squared_error(y_noise, run_baseline_lr(X, y_noise))}')

    return X, y_noise, noise


def generate_simulation_data_v4(opt):
    from models.nn.nn import NN
    model = NN(hidden=[20, 20, 20], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)

    w = model.get_layer1_w_data()
    w[:, 50:] = 0
    model.set_layer1_w_data(w)  # last 50 variables are set to 0

    var_e = 0.1
    noise = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)

    X = 2.0 * torch.rand(opt.n_samples, opt.n_features, device=opt.device) - 1.
    y = model.predict(X)
    y_noise = y + noise

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_numpy


def generate_simulation_data_v5(opt):
    from models.nn.nn import NN
    model = NN(hidden=[20, 20], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)

    w = model.get_layer1_w_data()
    w[:, 50:] = 0
    model.set_layer1_w_data(w)  # last 50 variables are set to 0

    var_e = 0.1
    noise = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)

    X = 2.0 * torch.rand(opt.n_samples, opt.n_features, device=opt.device) - 1.
    y = model.predict(X)
    y_noise = y + noise

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_numpy


def generate_simulation_data_v6(opt):
    from models.nn.nn import NN
    model = NN(hidden=[50, 50], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)

    w = model.get_layer1_w_data()
    w[:, 50:] = 0
    model.set_layer1_w_data(w)  # last 50 variables are set to 0

    var_e = 0.1
    noise = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)

    X = 2.0 * torch.rand(opt.n_samples, opt.n_features, device=opt.device) - 1.
    y = model.predict(X)
    y_noise = y + noise

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_numpy


def generate_simulation_data_v7(opt):
    from models.nn.nn import NN
    model = NN(hidden=[20, 20, 20], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)

    w = model.get_layer1_w_data()
    w[:, 50:] = 0
    model.set_layer1_w_data(w)  # last 50 variables are set to 0

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    assert opt.n_samples == 10000 and opt.n_features == 100
    X = 2.0 * torch.rand(opt.n_samples, opt.n_features, device=opt.device) - 1.

    var_e = 0.1
    noise_y = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 0.1 ** 2)
    y = model.predict(X)
    y_noise = y + noise_y

    var_e = 0.01
    noise_X = var_e * torch.randn(opt.n_samples, opt.n_features, device=opt.device)  # N (0, 0.01 ** 2)
    X_noise = X + noise_X

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_y_numpy = noise_y.detach().cpu().numpy()
    X_noise_numpy = X_noise.detach().cpu().numpy()
    noise_X_numpy = noise_X.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False, suffix="_plain"), X_numpy, fmt='%9f')
    np.savetxt(generate_data_filename(opt, False), X_noise_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False, suffix="_plain"), y_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False, suffix="_X"), noise_X_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_y_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_y_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_y_numpy


def generate_simulation_data_v8(opt):
    from models.nn.nn import NN
    model = NN(hidden=[20, 20, 20], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)
    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())
    w = model.get_layer1_w_data()

    assert opt.n_samples == 10000 and opt.n_features == 100
    X = torch.randn(opt.n_samples, opt.n_features, device=opt.device)

    w1 = torch.zeros_like(w)  # first 50 variables are set to insignificant
    w1[:, 50:] = w[:, 50:]
    model.set_layer1_w_data(w1)
    y1 = model.predict(X[:1000, :])

    model.set_layer1_w_data(w)
    y2 = model.predict(X[1000:5000, :])

    w3 = torch.zeros_like(w)  # last 50 variables are set to insignificant
    w3[:, :50] = w[:, :50]
    model.set_layer1_w_data(w3)
    y3 = model.predict(X[5000:, :])

    var_e = 0.1
    noise = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)
    y = torch.cat((y1, y2, y3), dim=0)
    y_noise = y + noise

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_numpy


def generate_simulation_data_v9(opt):
    from models.nn.nn import NN
    model = NN(hidden=[20, 20, 20], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)
    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())
    w = model.get_layer1_w_data()

    assert opt.n_samples == 10000 and opt.n_features == 100
    X = torch.randn(opt.n_samples, opt.n_features, device=opt.device)

    w1 = torch.zeros_like(w)  # first 50 variables are set to insignificant
    w1[:, 50:] = w[:, 50:]
    model.set_layer1_w_data(w1)
    y1 = model.predict(X[:1000, :])

    model.set_layer1_w_data(w)
    y2 = model.predict(X[1000:5000, :])

    w3 = torch.zeros_like(w)  # last 50 variables are set to insignificant
    w3[:, :50] = w[:, :50]
    model.set_layer1_w_data(w3)
    y3 = model.predict(X[5000:, :])

    var_e = 0.1
    noise_y = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)
    y = torch.cat((y1, y2, y3), dim=0)
    y_noise = y + noise_y

    var_e = 0.01
    noise_X = var_e * torch.randn(opt.n_samples, opt.n_features, device=opt.device)  # N (0, 0.01 ** 2)
    X_noise = X + noise_X

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_y_numpy = noise_y.detach().cpu().numpy()
    X_noise_numpy = X_noise.detach().cpu().numpy()
    noise_X_numpy = noise_X.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False, suffix="_plain"), X_numpy, fmt='%9f')
    np.savetxt(generate_data_filename(opt, False), X_noise_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False, suffix="_plain"), y_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False, suffix="_X"), noise_X_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_y_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_y_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_y_numpy


def generate_simulation_data_v10(opt):
    return


def generate_simulation_data_v11(opt):
    return


def generate_simulation_data_v12(opt):
    from models.nn.nn import NN
    model = NN(hidden=[16, 16, 16], in_features=opt.n_features, out_features=opt.n_targets,
               init_func=torch.nn.init.kaiming_normal_).to(opt.device)

    w = model.get_layer1_w_data()
    w[:, 50:] = 0
    model.set_layer1_w_data(w)  # last 50 variables are set to 0

    var_e = 0.1
    noise = var_e * torch.randn(opt.n_samples, opt.n_targets, device=opt.device)  # N(0, 1)

    X = 2.0 * torch.rand(opt.n_samples, opt.n_features, device=opt.device) - 1.
    y = model.predict(X)
    y_noise = y + noise

    torch.save(model.state_dict(), generate_data_model_filename(opt, last=False))
    for k, v in model.state_dict().items():
        np.savetxt(f'{opt.log_dir}/{k}.txt', v.detach().cpu().numpy())

    X_numpy = X.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()
    y_noise_numpy = y_noise.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_noise_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')

    opt.logger.info(f'MSE: {mean_squared_error(y_numpy, y_noise_numpy)}, noise: {np.sum(np.square(noise_numpy))}, '
                    f'targets: {mean_squared_error(y_noise_numpy, np.zeros_like(y_noise_numpy))}, '
                    f'lr: {mean_squared_error(y_noise_numpy, run_baseline_lr(X_numpy, y_noise_numpy))}')

    return X_numpy, y_noise_numpy, noise_numpy


def generate_uci_boston_data(opt):
    X_numpy, y_numpy = load_boston(return_X_y=True)

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 13 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_concrete_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/concrete/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 8 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_kin8nm_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/kin8nm/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 8 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_naval_data(opt, y1=False, y2=False):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/naval-propulsion-plant/data/data.txt')
    assert y1 ^ y2
    if y1:
        X_numpy, y_numpy = data[:, :-2], data[:, -2]
    else:
        X_numpy, y_numpy = data[:, :-2], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 16 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_power_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/power-plant/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 4 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_wine_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/wine-quality-red/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 11 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_protein_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/protein-tertiary-structure/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 9 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_yacht_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/yacht/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 6 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_energy_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/energy/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 8 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy


def generate_uci_efficient_data(opt):
    data = np.loadtxt(f'{opt.data_root}/../UCI_Datasets/energy/data/data.txt')
    X_numpy, y_numpy = data[:, :-1], data[:, -1]

    locs = np.squeeze(np.argwhere(X_numpy[:, 6] == 0))
    X_numpy_add_1 = X_numpy[locs, :]
    X_numpy_add_1[:, -1] = 1
    y_numpy_add_1 = y_numpy[locs]
    X_numpy_add_2 = X_numpy[locs, :]
    X_numpy_add_2[:, -1] = 2
    y_numpy_add_2 = y_numpy[locs]
    X_numpy_add_3 = X_numpy[locs, :]
    X_numpy_add_3[:, -1] = 3
    y_numpy_add_3 = y_numpy[locs]
    X_numpy_add_4 = X_numpy[locs, :]
    X_numpy_add_4[:, -1] = 4
    y_numpy_add_4 = y_numpy[locs]
    X_numpy_add_5 = X_numpy[locs, :]
    X_numpy_add_5[:, -1] = 5
    y_numpy_add_5 = y_numpy[locs]

    other_locs = [i for i in range(X_numpy.shape[0]) if i not in locs]
    X_numpy = X_numpy[other_locs, :]
    y_numpy = y_numpy[other_locs]

    X_numpy = np.concatenate((X_numpy, X_numpy_add_1, X_numpy_add_2, X_numpy_add_3, X_numpy_add_4, X_numpy_add_5),
                             axis=0)
    y_numpy = np.concatenate((y_numpy, y_numpy_add_1, y_numpy_add_2, y_numpy_add_3, y_numpy_add_4, y_numpy_add_5),
                             axis=0)

    opt.logger.info(X_numpy)
    opt.logger.info(y_numpy)

    X_numpy = standardization(X_numpy)

    assert X_numpy.shape[0] == opt.n_samples and X_numpy.shape[1] == 8 and opt.n_features >= X_numpy.shape[1]
    if opt.n_features > X_numpy.shape[1]:
        X_rand = np.random.randn(X_numpy.shape[0], opt.n_features - X_numpy.shape[1])
        X_numpy = np.concatenate((X_numpy, X_rand), axis=1)

    noise_numpy = np.zeros(y_numpy.shape)

    np.savetxt(generate_data_filename(opt, False), X_numpy, fmt='%9f')
    np.savetxt(generate_targets_filename(opt, False), y_numpy, fmt='%9f')
    np.savetxt(generate_noise_filename(opt, False), noise_numpy, fmt='%9f')
    opt.logger.info(f'lr: {mean_squared_error(y_numpy, run_baseline_lr(X_numpy, y_numpy))}')
    return X_numpy, y_numpy, noise_numpy
