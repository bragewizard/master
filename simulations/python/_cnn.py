import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5, stride=1, padding=0)
        self.map_to_grid = nn.Conv2d(2, 2, kernel_size=5, stride=3, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return torch.sigmoid(self.map_to_grid(x))
