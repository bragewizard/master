import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1 channel, 64x64
        # Kernel 3, Stride 2 -> Output: floor((64-3)/2 + 1) = 31x31
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2)
        self.relu = nn.ReLU()

        # Output Layer (Readout)
        # 8 filters * 31 * 31 = 7,688 inputs
        # Output 4: [x_sq, y_sq, x_tri, y_tri]
        self.fc = nn.Linear(8 * 31 * 31, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    def count_ops(self):
        # Calculation for 64x64 input
        h_out, w_out = 31, 31
        conv_ops = 2 * (1 * 8 * 3 * 3) * h_out * w_out
        fc_input = 8 * 31 * 31
        fc_ops = 2 * fc_input * 4

        total = conv_ops + fc_ops
        print("--- CNN EFFICIENCY METRIC (64x64) ---")
        print(f"Conv Layer Ops:  {conv_ops:,}")
        print(f"FC Layer Ops:    {fc_ops:,}")
        print(f"Total Ops/Frame: {total:,}")
        return total
