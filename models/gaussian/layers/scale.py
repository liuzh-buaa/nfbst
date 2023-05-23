from torch import nn
import torch.nn.functional as F


class Scale(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Scale, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, False, self.inplace)

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)
