import torch.nn as nn


class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        # 784 pixels -> 128 hidden spiking neurons
        # CRITICAL: bias=False is required for zero-shot SNN translation
        self.fc1 = nn.Linear(28 * 28, 128, bias=False)
        self.relu = nn.ReLU()

        # 128 hidden -> 10 output class neurons
        # CRITICAL: bias=False
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.log_softmax(x)
