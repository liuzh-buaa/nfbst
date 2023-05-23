import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class RandConv2d(nn.Module):
    def __init__(self, sigma_pi, sigma_start, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(RandConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self._sigma_pi = sigma_pi

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups
        # self.padding_mode = padding_mode
        # self.bias = bias

        # self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.log_sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer('eps_weight', torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            # self.bias = Parameter(torch.Tensor(out_channels))
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.log_sigma_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer('eps_bias', torch.Tensor(out_channels))
        else:
            # self.register_parameter('bias', None)
            self.register_parameter('mu_bias', None)
            self.register_parameter('log_sigma_bias', None)
            self.register_buffer('eps_bias', None)

        # pytorch/pytorch/blob/08891b0a4e08e2c642deac2042a02238a4d34c67/torch/nn/modules/conv.py#L40-L47
        # def reset_parameters(self):
        #     n = self.in_channels
        #     for k in self.kernel_size:
        #         n *= k
        #     stdv = 1. / math.sqrt(n)
        #     self.weight.data.uniform_(-stdv, stdv)
        #     if self.bias is not None:
        #         self.bias.data.uniform_(-stdv, stdv)

        torch.nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.mu_bias, -bound, bound)

        self.log_sigma_weight.data.fill_(math.log(sigma_start))
        self.eps_weight.data.zero_()
        if bias:
            self.log_sigma_bias.data.fill_(math.log(sigma_start))
            self.eps_bias.data.zero_()

    def forward_(self, x):
        sig_weight = torch.exp(self.log_sigma_weight)
        # self.eps_weight = torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size,
        #                               self.kernel_size).to(self.mu_weight.device)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()  # .normal_()
        bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.log_sigma_bias)
            # self.eps_bias = torch.randn(self.out_channels).to(self.mu_weight.device)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()  # .normal_()
        out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def forward(self, x):
        sig_weight = torch.exp(self.log_sigma_weight)
        # self.eps_weight = torch.randn(self.out_channels, self.in_channels // self.groups, self.kernel_size,
        #                               self.kernel_size).to(self.mu_weight.device)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()  # .normal_()
        kl_weight = math.log(self._sigma_pi) - self.log_sigma_weight + (sig_weight ** 2 + self.mu_weight ** 2) / (
                2 * self._sigma_pi ** 2) - 0.5
        bias = None
        kl_bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.log_sigma_bias)
            # self.eps_bias = torch.randn(self.out_channels).to(self.mu_weight.device)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()  # .normal_()
            kl_bias = math.log(self._sigma_pi) - self.log_sigma_bias + (sig_bias ** 2 + self.mu_bias ** 2) / (
                    2 * self._sigma_pi ** 2) - 0.5
        out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        kl = kl_weight.sum() + kl_bias.sum() if kl_bias is not None else kl_weight.sum()
        return out, kl

    def sample_mu_weight_bias(self):
        sig_weight = torch.exp(self.log_sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()  # .normal_()
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.log_sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()  # .normal_()
            return weight, bias
        else:
            return weight
