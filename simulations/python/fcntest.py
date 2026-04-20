import pyglet
from pyglet.window import key
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Added for weight viz
from matplotlib.colors import TwoSlopeNorm  # Add this to your imports
from _data import MNISTProvider
from _fcn import SimpleFCN

# --- CONFIG ---
WINDOW_SIZE = 800
BATCH_SIZE = 32
paused = False
step_once = False

window = pyglet.window.Window(
    WINDOW_SIZE,
    WINDOW_SIZE,
    caption="MNIST Baseline: [Space] Pause | [S] Step | [V] Viz Weights | [E] Export to SNN",
)

provider = MNISTProvider()
batch_group = pyglet.graphics.Batch()

model = SimpleFCN()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.NLLLoss()

# Sprite and Bar setup (Same as before)
sprite = pyglet.sprite.Sprite(
    pyglet.image.ImageData(
        28, 28, "RGB", np.zeros((28, 28, 3), dtype=np.uint8).tobytes()
    ),
    batch=batch_group,
)
sprite.scale = (WINDOW_SIZE // 2) / 28
sprite.y = WINDOW_SIZE // 4

bar_container_x = WINDOW_SIZE // 2
bar_width = (WINDOW_SIZE // 2) // 10
bars = [
    pyglet.shapes.Rectangle(
        bar_container_x + (i * bar_width),
        100,
        bar_width - 10,
        10,
        color=(100, 100, 255),
        batch=batch_group,
    )
    for i in range(10)
]


def visualize_weights():
    """Reshapes the 784 weights of hidden neurons back into 28x28 images."""
    weights = model.fc1.weight.detach().cpu().numpy()  # Shape: [128, 784]

    plt.figure(figsize=(10, 10))
    plt.suptitle("Hidden Layer Feature Detectors (SNN Synapse Maps)")

    # Calculate min/max for the whole set to keep color scale consistent
    w_min, w_max = weights.min(), weights.max()

    # Ensure 0 is in the middle. If all weights are positive,
    # we just use a standard norm to avoid errors.
    if w_min < 0 < w_max:
        norm = TwoSlopeNorm(vmin=w_min, vcenter=0, vmax=w_max)
    else:
        norm = None

    for i in range(16):  # Show the first 16 hidden neurons
        plt.subplot(4, 4, i + 1)
        w_img = weights[i].reshape(28, 28)
        plt.imshow(w_img, cmap="RdBu", norm=norm)
        plt.axis("off")
        plt.title(f"Neuron {i}")

    plt.tight_layout()
    plt.show()


@window.event
def on_key_press(symbol, modifiers):
    global paused, step_once
    if symbol == key.SPACE:
        paused = not paused
    elif symbol == key.S:
        step_once = True
    elif symbol == key.V:
        visualize_weights()

    elif symbol == key.E:
        print("\n--- The Parameter Bridge: Exporting Baseline ---")
        torch.save(model.state_dict(), "phase2_baseline_fcn.pth")

        # 1. Evaluate Baseline Accuracy
        print("Evaluating Baseline ANN Accuracy...")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            # Grabbing a large test batch (adapt to your _data.py provider if needed)
            test_images, test_labels = provider.get_batch(1000)
            outputs = model(test_images)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == test_labels).sum().item()
            total = test_labels.size(0)

        ann_acc = (correct / total) * 100
        print(f"[+] Baseline ANN Accuracy (FP32): {ann_acc:.2f}%")

        # 2. Extract and Quantize Weights
        with torch.no_grad():
            w1_fp = model.fc1.weight.data.cpu().numpy().flatten()
            w1_int8 = (
                (model.fc1.weight.data * 64)
                .round()
                .clamp(-128, 127)
                .cpu()
                .numpy()
                .flatten()
            )

            print(f"  -> FP32 Range: [{w1_fp.min():.4f}, {w1_fp.max():.4f}]")
            print(f"  -> INT8 Range: [{w1_int8.min():.0f}, {w1_int8.max():.0f}]")

        # 3. Export to CSV for Typst/CeTZ
        import csv

        # We sample 5000 weights to keep the CSV lightweight for Typst compilation
        sample_indices = np.random.choice(len(w1_fp), 5000, replace=False)
        with open("phase2_weights.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "FP32_Value", "INT8_Value"])
            for idx in sample_indices:
                writer.writerow([idx, w1_fp[idx], w1_int8[idx]])
        print("[+] Exported weight sample to 'phase2_weights.csv'")

        # 4. Generate the overlaid Histogram Plot
        plt.figure(figsize=(10, 5))
        plt.hist(
            w1_fp,
            bins=100,
            alpha=0.6,
            color="blue",
            label="FP32 (Original)",
            density=True,
        )
        # Scale INT8 down by 64 just for the visual overlay comparison
        plt.hist(
            w1_int8 / 64.0,
            bins=100,
            alpha=0.6,
            color="orange",
            label="INT8 (Quantized)",
            density=True,
        )
        plt.title("Phase II: Synaptic Weight Distribution (Hidden Layer N1)")
        plt.xlabel("Synaptic Weight Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("weight_distribution.png", dpi=300)
        print("[+] Saved 'weight_distribution.png'")
        print("Ready for Phase II Zero-Shot Transfer!")


def update(dt):
    global step_once
    if paused and not step_once:
        return

    # 1. Train
    images, target_labels = provider.get_batch(BATCH_SIZE)
    model.train()
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, target_labels)
    loss.backward()
    optimizer.step()

    raw_img = (images[0, 0].detach().numpy() * 255).astype(np.uint8)
    raw_img = np.flipud(raw_img)  # Corrects for the Pyglet/OpenGL bottom-up origin

    sprite.image = pyglet.image.ImageData(
        28, 28, "RGB", np.dstack([raw_img] * 3).tobytes()
    )

    # 3. Update Bars
    probs = torch.exp(output[0]).detach().numpy()
    winner = np.argmax(probs)
    for i in range(10):
        bars[i].height = max(5, int(probs[i] * 400))
        bars[i].color = (
            (50, 255, 50)
            if i == winner and winner == target_labels[0].item()
            else (100, 100, 255)
        )
        if i == winner and winner != target_labels[0].item():
            bars[i].color = (255, 50, 50)

    step_once = False


@window.event
def on_draw():
    window.clear()
    batch_group.draw()
    if paused:
        pyglet.text.Label(
            "PAUSED - Press 'S' to step",
            x=WINDOW_SIZE // 2,
            y=20,
            anchor_x="center",
            color=(255, 255, 0, 255),
        ).draw()


pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
