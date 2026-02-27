import torch.nn as nn

# --- CONFIG ---
FRAME_WIDTH, FRAME_HEIGHT = 28, 28
BATCH_SIZE = 32
TRAIN_STEPS = 1000
LEARNING_RATE = 0.001


# --- THE NETWORK ---
class SimpleLineCNN(nn.Module):
    def __init__(self):
        super(SimpleLineCNN, self).__init__()

        # Layer 1: Convolution
        # We use a large stride (2) to reduce dimensionality quickly,
        # simulating a "strided" receptive field in an SNN.
        # Input: 1x28x28 -> Output: 1x13x13
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.relu = nn.ReLU()

        # Layer 2: Fully Connected (Readout)
        # 4 filters * 13 * 13 = 676 inputs
        self.fc = nn.Linear(1 * 13 * 13, 2)  # Output: X, Y

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    def count_ops(self):
        # 1. Conv Ops: 2 * (Kin * Kout * Kh * Kw) * H_out * W_out
        # Input 28x28, Kernel 3x3, Stride 2 -> Output 13x13
        h_out, w_out = 13, 13
        conv_ops = 2 * (1 * 1 * 3 * 3) * h_out * w_out

        # 2. Linear Ops: 2 * Input * Output
        fc_input = 1 * 13 * 13
        fc_ops = 2 * fc_input * 2

        total = conv_ops + fc_ops
        print("--- CNN EFFICIENCY METRIC ---")
        print(f"Conv Layer Ops: {conv_ops:,}")
        print(f"FC Layer Ops:   {fc_ops:,}")
        print(f"Total Ops/Frame: {total:,} (Constant)")
        return total
