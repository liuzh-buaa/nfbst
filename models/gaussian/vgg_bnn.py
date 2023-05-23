# reference to torchvision/models/vgg.py
from collections import OrderedDict
from typing import Any, Union, List, cast

import torch
from torch import nn

from models.base import BaseModel
from models.gaussian.layers.batchnorm2d import RandBatchNorm2d
from models.gaussian.layers.conv2d import RandConv2d
from models.gaussian.layers.linear import RandLinear

# A-VGG11, B-VGG13, D-VGG16, E-VGG19
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGBNN(BaseModel):
    def __init__(
            self,
            sigma_pi,
            sigma_start,
            features: nn.Module,
            batch_norm: bool = False,
            num_classes: int = 10,
            init_weights: bool = True
    ) -> None:
        super(VGGBNN, self).__init__()

        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            RandLinear(sigma_pi, sigma_start, 512 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
            RandLinear(sigma_pi, sigma_start, 256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            RandLinear(sigma_pi, sigma_start, 256, num_classes),
        )

        self.batch_norm = batch_norm

    def forward_(self, x):
        out = x
        for layer in self.features:
            if type(layer).__name__.startswith("Rand"):
                out = layer.forward_(out)
            else:
                out = layer.forward(out)
        # out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        for layer in self.classifier:
            if type(layer).__name__.startswith("Rand"):
                out = layer.forward_(out)
            else:
                out = layer.forward(out)
        return out

    def forward(self, x):
        if self.standard:
            return self.forward_(x)

        kl_sum, out = 0, x
        for layer in self.features:
            if type(layer).__name__.startswith("Rand"):
                out, kl = layer.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = layer.forward(out)
        # out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        for layer in self.classifier:
            if type(layer).__name__.startswith("Rand"):
                out, kl = layer.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = layer.forward(out)
        return out, kl_sum

    def sample_4_generate_standard_model(self):
        params = OrderedDict()
        for i, layer in enumerate(self.features):
            if type(layer).__name__.startswith("Rand"):
                params[f'features.{i}.weight'], params[f'features.{i}.bias'] = layer.sample_mu_weight_bias()
                if type(layer).__name__ == "RandBatchNorm2d":
                    params[f'features.{i}.running_mean'] = layer.running_mean
                    params[f'features.{i}.running_var'] = layer.running_var
                    params[f'features.{i}.num_batches_tracked'] = layer.num_batches_tracked
        for i, layer in enumerate(self.classifier):
            if type(layer).__name__.startswith("Rand"):
                params[f'classifier.{i}.weight'], params[f'classifier.{i}.bias'] = layer.sample_mu_weight_bias()
                if type(layer).__name__ == "RandBatchNorm2d":
                    params[f'features.{i}.running_mean'] = layer.running_mean
                    params[f'features.{i}.running_var'] = layer.running_var
                    params[f'features.{i}.num_batches_tracked'] = layer.num_batches_tracked
        if self.batch_norm:
            from models.nn.vgg import vgg16_bn
            model = vgg16_bn().to(params['features.0.weight'].device)
        else:
            from models.nn.vgg import vgg16
            model = vgg16().to(params['features.0.weight'].device)
        model.load_state_dict(params)
        model.convert_to_standard_model()
        return model

    def predict(self, x, rep=1):
        outputs_agg = 0.0
        for _ in range(rep):
            outputs, _ = self.forward(x)
            outputs_agg += outputs
        return outputs_agg / rep


def make_layers(sigma_pi, sigma_start, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = RandConv2d(sigma_pi, sigma_start, in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, RandBatchNorm2d(sigma_pi, sigma_start, v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vggbnn(sigma_pi, sigma_start, arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool,
            **kwargs: Any) -> VGGBNN:
    model = VGGBNN(sigma_pi, sigma_start, make_layers(sigma_pi, sigma_start, cfgs[cfg], batch_norm=batch_norm),
                   batch_norm=batch_norm, **kwargs)
    return model


def vgg16bnn(sigma_pi, sigma_start, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGBNN:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    return _vggbnn(sigma_pi, sigma_start, 'vgg16', 'D', False, False, progress, **kwargs)


def vgg16bnn_bn(sigma_pi, sigma_start, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGBNN:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    return _vggbnn(sigma_pi, sigma_start, 'vgg16_bn', 'D', True, False, progress, **kwargs)
