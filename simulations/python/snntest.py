import pyglet
from pyglet.window import key
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from _data import MNISTProvider
from _fcn import SimpleFCN
from _snn import SimpleSNN  # Your GPU SNN logic
import matplotlib.pyplot as plt  # Added for weight viz
from matplotlib.colors import TwoSlopeNorm  # Add this to your imports

# --- CONFIG ---
WINDOW_SIZE = 800
BATCH_SIZE = 32
paused = False
step_once = False

window = pyglet.window.Window(
    WINDOW_SIZE, WINDOW_SIZE, caption="SNN Saccade Test: [Space] Pause | [S] Step"
)
provider = MNISTProvider()
batch_group = pyglet.graphics.Batch()

# The CNN (Teacher)
model = SimpleFCN()
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.NLLLoss()

# The SNN (Student/Inference)
# We will update the SNN's weights from the CNN every frame
snn = SimpleSNN(model, device="cuda")

# UI Setup
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


# ... (existing imports)


def visualize_snn_weights():
    """Visualize the actual i8 weights currently sitting on the GPU."""
    # Pull from GPU back to CPU for plotting
    weights = snn.w1.detach().cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.suptitle("SNN Quantized Synapse Maps (i8 Precision)")

    w_min, w_max = weights.min(), weights.max()
    norm = (
        TwoSlopeNorm(vmin=w_min, vcenter=0, vmax=w_max) if w_min < 0 < w_max else None
    )

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        w_img = weights[i].reshape(28, 28)
        # Use 'nearest' interpolation to see the individual i8 pixel weights
        plt.imshow(w_img, cmap="RdBu", norm=norm, interpolation="nearest")
        plt.axis("off")
        plt.title(f"SNN Neur {i}")
    plt.show()


@window.event
def on_key_press(symbol, modifiers):
    global paused, step_once
    if symbol == key.SPACE:
        paused = not paused
    elif symbol == key.S:
        step_once = True
    elif symbol == key.W:  # New key for SNN specific weights
        visualize_snn_weights()
    elif symbol == key.UP:  # Hotkey to increase sensitivity
        snn.threshold_h -= 10
        print(f"Hidden Threshold: {snn.threshold_h}")
    elif symbol == key.DOWN:  # Hotkey to decrease sensitivity
        snn.threshold_h += 10
        print(f"Hidden Threshold: {snn.threshold_h}")


def update(dt):
    global step_once
    if paused and not step_once:
        return

    # 1. Train the CNN (Teacher)
    images, target_labels = provider.get_batch(BATCH_SIZE)
    model.train()
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, target_labels)
    loss.backward()
    optimizer.step()

    # 2. Run the SNN (Inference)
    # We pass the weights of the current model to the SNN
    snn.update_weights(model)
    # Process the first image in the batch through the spiking logic
    snn_spikes = snn.run_saccade(images[0])  # Returns [10] array of tick times

    # 3. Update Visuals
    raw_img = (images[0, 0].detach().numpy() * 255).astype(np.uint8)
    raw_img = np.flipud(raw_img)
    sprite.image = pyglet.image.ImageData(
        28, 28, "RGB", np.dstack([raw_img] * 3).tobytes()
    )

    # 4. Update Bars based on SPIKE TIME
    # We invert the time: Firing at T=0 is a tall bar, T=64 is a short bar
    for i in range(10):
        tick = snn_spikes[i].item()
        if tick == -1:  # Never fired
            bars[i].height = 5
            bars[i].color = (50, 50, 50)
        else:
            # Map tick 0-64 to height 400-5
            bars[i].height = max(5, int((1.0 - (tick / 64)) * 400))

            # Winner logic
            winner = torch.argmin(
                torch.where(
                    snn_spikes == -1, torch.tensor(999, device="cuda"), snn_spikes
                )
            ).item()
            if i == winner:
                bars[i].color = (
                    (50, 255, 50)
                    if winner == target_labels[0].item()
                    else (255, 50, 50)
                )
            else:
                bars[i].color = (100, 100, 255)

    step_once = False


@window.event
def on_draw():
    window.clear()
    batch_group.draw()
    pyglet.text.Label("CNN Training...", x=10, y=WINDOW_SIZE - 30).draw()
    pyglet.text.Label(
        "SNN Spike Latency (Higher = Earlier Spike)", x=420, y=WINDOW_SIZE - 30
    ).draw()


pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
