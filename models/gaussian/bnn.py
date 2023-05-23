from collections import OrderedDict

from torch import nn

from models.base import BaseModel
from models.gaussian.layers.linear import RandLinear


class BNN(BaseModel):
    def __init__(self, sigma_pi, sigma_start, hidden, in_features, out_features=1, activation='relu'):
        super(BNN, self).__init__()

        self._hidden = hidden
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation

        self._n_hidden = len(hidden)
        assert self._n_hidden > 0

        self.fc = nn.Sequential()
        units = [in_features] + hidden
        for i in range(0, self._n_hidden):
            self.fc.add_module(f'linear{i}', RandLinear(sigma_pi, sigma_start, units[i], units[i + 1]))
            if activation == 'relu':
                self.fc.add_module(f'relu{i}', nn.ReLU())
            elif activation == 'tanh':
                self.fc.add_module(f'tanh{i}', nn.Tanh())
            elif activation == 'sigmoid':
                self.fc.add_module(f'sigmoid{i}', nn.Sigmoid())
            else:
                raise NotImplementedError(f'No such activation type of {activation}')
        self.fc.add_module(f'linear{self._n_hidden}', RandLinear(sigma_pi, sigma_start, units[-1], out_features))

    def forward_(self, x):
        out = x
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
        for i in range(self._n_hidden + 1):
            params[f'fc.linear{i}.weight'], params[f'fc.linear{i}.bias'] = self.fc[2 * i].sample_mu_weight_bias()
        from models.nn.nn import NN
        model = NN(self._hidden, self._in_features, self._out_features, self._activation).to(
            params['fc.linear0.weight'].device)
        model.load_state_dict(params)
        model.convert_to_standard_model()
        return model

    def predict(self, x, rep=1):
        outputs_agg = 0.0
        for _ in range(rep):
            outputs = self.forward_(x)
            outputs_agg += outputs
        return outputs_agg / rep
