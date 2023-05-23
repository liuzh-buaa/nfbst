from torch import nn

from models.base import BaseModel


class NN(BaseModel):
    def __init__(self, hidden, in_features, out_features=1, activation='relu', init_func=None):
        super(NN, self).__init__()

        self._hidden = hidden
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation

        n_hidden = len(hidden)
        assert n_hidden > 0

        self.fc = nn.Sequential()
        units = [in_features] + hidden
        for i in range(0, n_hidden):
            self.fc.add_module(f'linear{i}', nn.Linear(units[i], units[i + 1]))
            if init_func is not None:
                init_func(self.fc[-1].weight)
            if activation == 'relu':
                self.fc.add_module(f'relu{i}', nn.ReLU())
            elif activation == 'tanh':
                self.fc.add_module(f'tanh{i}', nn.Tanh())
            elif activation == 'sigmoid':
                self.fc.add_module(f'sigmoid{i}', nn.Sigmoid())
            else:
                raise NotImplementedError(f'No such activation type of {activation}')
        self.fc.add_module(f'linear{n_hidden}', nn.Linear(units[-1], out_features))
        if init_func is not None:
            init_func(self.fc[-1].weight)

    def forward_(self, x):
        return self.fc(x)

    def forward(self, x):
        if self.standard:
            return self.forward_(x)

        return self.fc(x), 0

    def sample_4_generate_standard_model(self):
        self.convert_to_standard_model()
        return self

    def predict(self, x, rep=1):
        return self.forward_(x)

    # def get_layer1_w_data(self):
    #     """ for aglnet """
    #     return self.fc[0].weight.data
    #
    # def set_layer1_w_data(self, data):
    #     """ for aglnet """
    #     self.fc[0].weight.data = data
    #
    # def proximal(self, lam, eta):
    #     """ for aglnet """
    #     tmp = torch.norm(self.get_layer1_w_data().t(), dim=1) - lam * eta
    #     alpha = torch.clamp(tmp, min=0)
    #     v = torch.nn.functional.normalize(self.get_layer1_w_data().t(), dim=1) * alpha[:, None]
    #     self.set_layer1_w_data(v.t())
