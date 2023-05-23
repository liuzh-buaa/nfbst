import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms

from datasets.data_utils import normalization, standardization
from utils.utils_file import generate_data_filename, generate_targets_filename


class SimulationDataset(Dataset):
    def __init__(self, opt, normalize=False, standard=False, indices=None):
        super(SimulationDataset, self).__init__()
        self.data = np.loadtxt(generate_data_filename(opt, True))
        self.targets = np.loadtxt(generate_targets_filename(opt, True))

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.targets = torch.unsqueeze(self.targets, dim=1)

        assert not (normalize and standard), 'cannot both normalization and standardization'
        if normalize:
            self.data = normalization(self.data)
            self.targets = normalization(self.targets)

        if standard:
            self.data = standardization(self.data)
            self.targets = standardization(self.targets)

        if indices is None:
            self.indices = list(range(self.data.size()[0]))
        elif isinstance(indices, int):
            self.indices = list(range(indices))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> T_co:
        return self.data[self.indices[index]], self.targets[self.indices[index]]


class MnistDataset(Dataset):
    def __init__(self, opt, train, indices=None):
        super(MnistDataset, self).__init__()
        self.dataset = torchvision.datasets.MNIST(root=opt.data_dir,
                                                  train=train,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)

        if indices is None:
            self.indices = list(range(len(self.dataset)))
        elif isinstance(indices, int):
            self.indices = list(range(indices))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> T_co:
        return self.dataset[self.indices[index]]


class Cifar10Dataset(Dataset):
    def __init__(self, opt, train, indices=None):
        super(Cifar10Dataset, self).__init__()
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = torchvision.datasets.cifar.CIFAR10(root=opt.data_dir,
                                                          train=train,
                                                          transform=transform,
                                                          download=True)

        if indices is None:
            self.indices = list(range(len(self.dataset)))
        elif isinstance(indices, int):
            self.indices = list(range(indices))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> T_co:
        return self.dataset[self.indices[index]]


def build_reg_dataset(opt, train=True, indices=None):
    if opt.data in ['mnist']:
        dataset = MnistDataset(opt, train, indices=indices)
    elif opt.data in ['cifar10']:
        dataset = Cifar10Dataset(opt, train, indices=indices)
    else:
        dataset = SimulationDataset(opt, indices=indices)

    return dataset


def build_reg_loader(opt, shuffle=False, train=True, indices=None):
    dataset = build_reg_dataset(opt, train, indices=indices)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle)
    return dataset, dataloader
