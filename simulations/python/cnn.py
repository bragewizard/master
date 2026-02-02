import torch
import torch.nn as nn


class SimpleLineCNN(nn.Module):
    def __init__(self):
        super(SimpleLineCNN, self).__init__()
        # Layer 1: Convolution (simulating SNN receptive fields)
        # 4 filters, 3x3 kernel, stride 2 to reduce size
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()

        # Layer 2: Fully Connected (simulating final neuron readout)
        # Input image 28x28 -> /2 -> 14x14.
        # Flattened size = 4 * 14 * 14 = 784 neurons
        self.fc = nn.Linear(4 * 14 * 14, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def count_cnn_ops(model, input_dim):
    # Theoretical FLOPs (Floating Point Operations)
    # Conv2d Ops: 2 * (Kin * Kout * Kh * Kw) * H_out * W_out
    # FC Ops: 2 * Cin * Cout

    # Let's assume input 1x28x28
    # Conv1: 4 filters, 3x3 kernels. Output is 14x14
    conv_ops = (2 * 1 * 4 * 3 * 3) * 14 * 14

    # FC: Input 784, Output 1
    fc_ops = 2 * 784 * 1

    total_ops = conv_ops + fc_ops
    print(f"CNN Operations per Frame: {total_ops}")
    return total_ops


# Initialize
cnn = SimpleLineCNN()
count_cnn_ops(cnn, (1, 28, 28))

# --- HANDCRAFTING WEIGHTS ---
# Instead of training, let's manually set weights to detect vertical lines
# Filter 0: Vertical Edge Detector
vertical_filter = torch.tensor(
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32
)
with torch.no_grad():
    cnn.conv1.weight[0] = vertical_filter.unsqueeze(0)
    # Set other filters to 0 or random
