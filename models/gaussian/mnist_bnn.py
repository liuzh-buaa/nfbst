from collections import OrderedDict

import torch
from torch import nn

from models.base import BaseModel
from models.gaussian.layers.conv2d import RandConv2d
from models.gaussian.layers.linear import RandLinear


class MnistBNN(BaseModel):
    def __init__(self, sigma_pi, sigma_start):
        super(MnistBNN, self).__init__()

        self.conv = nn.Sequential(
            RandConv2d(sigma_pi, sigma_start, 1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            RandConv2d(sigma_pi, sigma_start, 32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            RandLinear(sigma_pi, sigma_start, 1600, 64),
            nn.ReLU(),
            RandLinear(sigma_pi, sigma_start, 64, 10)
        )

    def forward_(self, x):
        out = x
        for layer in self.conv:
            if type(layer).__name__.startswith("Rand"):
                out = layer.forward_(out)
            else:
                out = layer.forward(out)
        out = torch.flatten(out, start_dim=1)
        for layer in self.fc:
            if type(layer).__name__.startswith("Rand"):
                out = layer.forward_(out)
            else:
                out = layer.forward(out)
        return out

    def forward(self, x):
        if self.standard:
            return self.forward_(x)

        kl_sum, out = 0, x
        for layer in self.conv:
            if type(layer).__name__.startswith("Rand"):
                out, kl = layer.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = layer.forward(out)
        out = torch.flatten(out, start_dim=1)
        for layer in self.fc:
            if type(layer).__name__.startswith("Rand"):
                out, kl = layer.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = layer.forward(out)

        return out, kl_sum

    def sample_4_generate_standard_model(self):
        params = OrderedDict()
        for i, layer in enumerate(self.conv):
            if type(layer).__name__.startswith("Rand"):
                params[f'conv.{i}.weight'], params[f'conv.{i}.bias'] = layer.sample_mu_weight_bias()
        for i, layer in enumerate(self.fc):
            if type(layer).__name__.startswith("Rand"):
                params[f'fc.{i}.weight'], params[f'fc.{i}.bias'] = layer.sample_mu_weight_bias()
        from models.nn.mnist_cnn import MnistCNN
        model = MnistCNN().to(params['conv.0.weight'].device)
        model.load_state_dict(params)
        model.convert_to_standard_model()
        return model

    def predict(self, x, rep=1):
        outputs_agg = 0.0
        for _ in range(rep):
            outputs = self.forward_(x)
            outputs_agg += outputs
        return outputs_agg / rep
