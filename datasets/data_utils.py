import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def normalization(data):
    """ 归一化 """
    if type(data) == np.ndarray:
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)
        _range = _max - _min
    elif type(data) == torch.Tensor:
        _min = torch.min(data, dim=0)[0]
        _max = torch.max(data, dim=0)[0]
        _range = _max - _min
    else:
        raise NotImplementedError('Unsupported normalization type.')

    return (data - _min) / _range


def standardization(data):
    """ 标准化 """
    if type(data) == np.ndarray:
        _mean = np.mean(data, axis=0)
        _std = np.std(data, axis=0)
        if type(_std) == np.ndarray:
            _std[_std == 0] = 1
    elif type(data) == torch.Tensor:
        _mean = torch.mean(data, dim=0)
        _std = torch.std(data, dim=0)  # Bessel's Correction
        if type(_std) == torch.Tensor:
            _std[_std == 0] = 1
    else:
        raise NotImplementedError('Unsupported standardization type.')

    return (data - _mean) / _std


def init_config_data(opt):
    if opt.data in ['simulation_v1', 'simulation_v2', 'simulation_v4', 'simulation_v5', 'simulation_v6',
                    'simulation_v7', 'simulation_v8', 'simulation_v9', 'simulation_v11', 'simulation_v12']:
        opt.n_features = 100
        opt.n_targets = 1
        opt.n_samples = 10000
    elif opt.data in ['simulation_v3', 'simulation_v10']:
        opt.n_features = 8
        opt.n_targets = 1
        opt.n_samples = 10000
    elif opt.data in ['mnist']:
        opt.n_features = (28, 28)
        opt.n_targets = 10
        opt.n_samples = (60000, 10000)
    elif opt.data in ['cifar10']:
        opt.n_features = (3, 32, 32)
        opt.n_targets = 10  # ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        opt.n_samples = (50000, 10000)
    elif opt.data.startswith('boston'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'boston'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 506
    elif opt.data.startswith('concrete'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'concrete'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 1030
    elif opt.data.startswith('energy'):
        strs = opt.data.split('_')
        assert strs[0] == 'energy'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 768
    elif opt.data.startswith('efficient'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'efficient'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 960
    elif opt.data.startswith('kin8nm'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'kin8nm'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 8192
    elif opt.data.startswith('naval_y1') or opt.data.startswith('naval_y2'):
        strs = opt.data.split('_')
        assert len(strs) == 3
        opt.n_features = int(strs[2])
        opt.n_targets = 1
        opt.n_samples = 11934
    elif opt.data.startswith('power'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'power'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 9568
    elif opt.data.startswith('wine'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'wine'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 1599
    elif opt.data.startswith('protein'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'protein'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 45730
    elif opt.data.startswith('yacht'):
        strs = opt.data.split('_')
        assert len(strs) == 2 and strs[0] == 'yacht'
        opt.n_features = int(strs[1])
        opt.n_targets = 1
        opt.n_samples = 308
    else:
        raise NotImplementedError(f'No such data type of {opt.data}.')


def run_baseline_lr(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model.predict(x)


if __name__ == '__main__':
    print(mean_squared_error(np.loadtxt('../data/simulation_v3/data/targets.txt'),
                             run_baseline_lr(np.loadtxt('../data/simulation_v3/data/data.txt'),
                                             np.loadtxt('../data/simulation_v3/data/targets.txt'))))
