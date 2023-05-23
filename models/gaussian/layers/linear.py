import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class RandLinear(nn.Module):
    def __init__(self, sigma_pi, sigma_start, in_features, out_features, bias=True):
        super(RandLinear, self).__init__()

        self._sigma_pi = sigma_pi

        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.log_sigma_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer('eps_bias', torch.Tensor(out_features))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('log_sigma_bias', None)
            self.register_buffer('eps_bias', None)

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
        # self.eps_weight = torch.randn(self.out_features, self.in_features).to(self.mu_weight.device)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()  # .normal_()
        bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.log_sigma_bias)
            # self.eps_bias = torch.randn(self.out_features).to(self.mu_weight.device)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()  # .normal_()
        return F.linear(x, weight, bias)

    def forward(self, x):
        sig_weight = torch.exp(self.log_sigma_weight)
        # self.eps_weight = torch.randn(self.out_features, self.in_features).to(self.mu_weight.device)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()  # .normal_()
        kl_weight = math.log(self._sigma_pi) - self.log_sigma_weight + (sig_weight ** 2 + self.mu_weight ** 2) / (
                2 * self._sigma_pi ** 2) - 0.5
        bias = None
        kl_bias = None
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.log_sigma_bias)
            # self.eps_bias = torch.randn(self.out_features).to(self.mu_weight.device)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()  # .normal_()
            kl_bias = math.log(self._sigma_pi) - self.log_sigma_bias + (sig_bias ** 2 + self.mu_bias ** 2) / (
                    2 * self._sigma_pi ** 2) - 0.5
        out = F.linear(x, weight, bias)
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
