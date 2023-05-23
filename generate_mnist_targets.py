"""
    【Data Generation Process】
    Select the corresponding serial number according to the classification label for the following comparison.

    `python generate_mnist_targets.py --log True`
"""
import shutil
import time

import numpy as np
import pandas as pd
import torchvision

from utils.utils_file import generate_mnist_or_cifar10_targets_filename
from utils.utils_parser import DefaultArgumentParser, init_config


def generate_mnist_targets():
    _dataset = torchvision.datasets.MNIST(root=opt.data_dir,
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
    _targets = _dataset.targets.cpu().numpy()
    _writer = pd.ExcelWriter(generate_mnist_or_cifar10_targets_filename(opt, train=True))
    for target in range(10):
        _indices = np.argwhere(_targets == target)
        _pd_data = pd.DataFrame(_indices)
        _pd_data.to_excel(_writer, sheet_name=f'target{target}', index=False, header=False)
    _writer.close()

    _dataset = torchvision.datasets.MNIST(root=opt.data_dir,
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
    _targets = _dataset.targets.cpu().numpy()
    _writer = pd.ExcelWriter(generate_mnist_or_cifar10_targets_filename(opt, train=False))
    for target in range(10):
        _indices = np.argwhere(_targets == target)
        _pd_data = pd.DataFrame(_indices)
        _pd_data.to_excel(_writer, sheet_name=f'target{target}', index=False, header=False)
    _writer.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = DefaultArgumentParser().get_parser()

    opt = parser.parse_args()
    opt.exp_name = 'generate_mnist_targets'
    opt.data = 'mnist'
    init_config(opt)

    generate_mnist_targets()

    if opt.log:
        opt.logger.info('Copying targets indices.xlsx from `timestamp` to `data`')
        shutil.copyfile(generate_mnist_or_cifar10_targets_filename(opt, train=True, last=False),
                        generate_mnist_or_cifar10_targets_filename(opt, train=True, last=True))
        shutil.copyfile(generate_mnist_or_cifar10_targets_filename(opt, train=False, last=False),
                        generate_mnist_or_cifar10_targets_filename(opt, train=False, last=True))

    end_time = time.time()
    elapse_time = end_time - start_time
    opt.logger.info(f'All end  in {elapse_time // 60:.0f}m {elapse_time % 60:.0f}s.')
