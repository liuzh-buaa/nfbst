import torch
from torch import nn

from models.base import BaseModel


class MnistCNN(BaseModel):
    def __init__(self):
        super(MnistCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),  # (batch_size, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (batch_size, 32, 13, 13)
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # (batch_size, 64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (batch_size, 64, 5, 5)
        )

        self.fc = nn.Sequential(
            nn.Linear(1600, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward_(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def forward(self, x):
        if self.standard:
            return self.forward_(x)

        return self.forward_(x), 0

    def sample_4_generate_standard_model(self):
        self.convert_to_standard_model()
        return self

    def predict(self, x, rep=1):
        return self.forward_(x)
