import torch
import torch.nn as nn
import torch.optim as optim
from _cnn import (
    SimpleLineCNN,
    BATCH_SIZE,
    TRAIN_STEPS,
    LEARNING_RATE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
import numpy as np
from _data import LineVideoGenerator

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


# --- TRAINING HELPERS ---
def generate_batch(gen, batch_size):
    images = []
    targets = []
    for _ in range(batch_size):
        img, (x, y) = gen.get_next_frame()
        # Normalize Image 0-1
        images.append(img[np.newaxis, :, :] / 255.0)
        # Normalize Targets to range -1 to 1 (helps training)
        t_x = (x / gen.width) * 2 - 1
        t_y = (y / gen.height) * 2 - 1
        targets.append([t_x, t_y])

    return torch.tensor(np.array(images), dtype=torch.float32), torch.tensor(
        np.array(targets), dtype=torch.float32
    )


# --- MAIN ---
if __name__ == "__main__":
    # 1. Setup
    gen = LineVideoGenerator(FRAME_WIDTH, FRAME_HEIGHT)
    model = SimpleLineCNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 2. Benchmark Ops
    model.count_ops()

    # 3. Train
    print("\nStarting Training...")
    for step in range(TRAIN_STEPS):
        inputs, targets = generate_batch(gen, BATCH_SIZE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.5f}")

    print("Training Complete.")

    # 4. Save Weights (for the SNN later)
    torch.save(model.state_dict(), "cnn_weights.pth")
    print("Weights saved to 'cnn_weights.pth'")
