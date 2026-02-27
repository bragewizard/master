import numpy as np
from _data import (
    scale_image,
    create_random_image_one_channel,
    inject_pattern,
    parse_MNIST,
    MNIST_train_images_path,
    MNIST_test_images_path,
)
import random
import pyglet
from pyglet.window import key
from pyglet.window import mouse
from pyglet.shapes import Circle, Rectangle, Line

from _snn import SpikingNet

NODE_C = 64
EDGE_C = 128

NETWORK_SHAPE = (64, 32)
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
LAYER_SPACING = 400
X_MARGIN = 32

# --- Import the simulation code ---


def np_to_pyglet_image(np_array):
    np_array = scale_image(np_array, 16)
    if np_array.dtype != np.uint8:
        np_array = np_array.astype(np.uint8)
    height, width = np_array.shape
    rgb_array = np.dstack((np_array, np_array, np_array))
    raw_data = rgb_array.tobytes()
    pitch = width * 3
    return pyglet.image.ImageData(width, height, "RGB", raw_data, pitch=pitch)


def update_image(np_array, image):
    np_array = scale_image(np_array, 16)
    if np_array.dtype != np.uint8:
        np_array = np_array.astype(np.uint8)
    height, width = np_array.shape
    rgb_array = np.dstack((np_array, np_array, np_array))
    raw_data = rgb_array.tobytes()
    pitch = width * 3
    image.set_data("RGB", pitch, raw_data)


def center_image(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


def get_neuron_coords(neuron_index):
    if 0 <= neuron_index < NETWORK_SHAPE[0]:
        i = neuron_index
        height = NETWORK_SHAPE[0] * 16
        y_offset = (WINDOW_HEIGHT - height) // 2
        neuron_x = (i // 100) * 16 + X_MARGIN
        neuron_y = (i % 100) * 16 + y_offset
        return neuron_x, neuron_y
    # Output Layer
    elif NETWORK_SHAPE[0] <= neuron_index < NETWORK_SHAPE[0] + NETWORK_SHAPE[1]:
        j = neuron_index - NETWORK_SHAPE[0]
        height = NETWORK_SHAPE[1] * 16
        y_offset = (WINDOW_HEIGHT - height) // 2
        neuron_x = (j // 100) * 16 + LAYER_SPACING + X_MARGIN
        neuron_y = (j % 100) * 16 + y_offset
        return neuron_x, neuron_y
    # Unknown
    return -1, -1


def draw_neurons(batch, group):
    circles = []
    # Use our new helper function!
    for i in range(NETWORK_SHAPE[0] + NETWORK_SHAPE[1]):
        neuron_x, neuron_y = get_neuron_coords(i)
        if neuron_x != -1:
            new_circle = Circle(
                x=neuron_x,
                y=neuron_y,
                radius=3,
                color=(NODE_C, NODE_C, NODE_C),
                batch=batch,
                group=group,
            )
            circles.append(new_circle)
    return circles


playback_speed = 100
# image_path = 'data/doomguy.jpg'
# image_path = 'data/lenna.png'
# original_image = Image.open(image_path)
# grayscale_image = original_image.convert('L')
# MNIST = parse_MNIST(MNIST_train_images_path)
# max_dim = 28
# low_res_image_pil = grayscale_image.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
random_image1 = create_random_image_one_channel(
    int(np.sqrt(NETWORK_SHAPE[0])), int(np.sqrt(NETWORK_SHAPE[0]))
)
random_image2 = create_random_image_one_channel(
    int(np.sqrt(NETWORK_SHAPE[0])), int(np.sqrt(NETWORK_SHAPE[0]))
)
random_image3 = create_random_image_one_channel(
    int(np.sqrt(NETWORK_SHAPE[0])), int(np.sqrt(NETWORK_SHAPE[0]))
)
rows1 = [4, 4, 4]
cols1 = [3, 4, 5]
rows2 = [7, 7, 7]
cols2 = [4, 5, 6]
rows3 = [2, 2, 2]
cols3 = [2, 3, 4]
inject_pattern(random_image1, rows1, cols1, 255)
inject_pattern(random_image2, rows2, cols2, 255)
inject_pattern(random_image3, rows3, cols3, 255)
# input_image = np.array(low_res_image_pil)
# input_image = generate_checkerboard(size=max_dim, block_size=int(max_dim/7))
# input_image = MNIST[24]
# visualize_image(input_image)
data = np.array([random_image1, random_image2, random_image3])
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

snn = SpikingNet()
# pyglet.resource.path = ['../data']
# pyglet.resource.reindex()
image = np_to_pyglet_image(data[0])
center_image(image)
batch = pyglet.graphics.Batch()
background_group = pyglet.graphics.Group(order=0)
foreground_group = pyglet.graphics.Group(order=1)
neurons = draw_neurons(batch, foreground_group)
connections = {}
label = pyglet.text.Label(
    "SPIKING NEURAL NETWORK snnULATOR V1",
    font_name="GeistMono NF Medium",
    font_size=11,
    x=window.width // 2,
    y=window.height - 18,
    anchor_x="center",
    anchor_y="center",
)
time_label = pyglet.text.Label(
    "",
    font_name="GeistMono NF Medium",
    font_size=11,
    x=window.width - 20,
    y=20,
    anchor_x="right",
    anchor_y="center",
)
input_label = pyglet.text.Label(
    "input image:",
    font_name="GeistMono NF Medium",
    font_size=11,
    x=window.width - 140,
    y=window.height - 20,
    anchor_x="right",
    anchor_y="center",
)


def update(dt):
    global playback_speed
    time_slice = dt * playback_speed
    if time_slice <= 0:
        return

    spiked_indices, current_ops = snn.advance(time_slice)
    if is_empty:
        snn.next_data(data)
        update_image(data[snn.iteration], image)

    for neuron in neurons:
        if neuron.color != (NODE_C, NODE_C, NODE_C):
            neuron.color = (
                max(NODE_C, neuron.color[0] - 5),
                max(NODE_C, neuron.color[1] - 5),
                max(NODE_C, neuron.color[2] - 5),
            )

    for idx in spiked_indices:
        if 0 <= idx < len(neurons):
            neurons[idx].color = (255, 255, 255)  # White spike

    for pre_idx, post_idx in grown_synapses:
        if (pre_idx, post_idx) not in connections:
            pre_x, pre_y = get_neuron_coords(pre_idx)
            post_x, post_y = get_neuron_coords(post_idx)
            new_line = Line(
                pre_x,
                pre_y,
                post_x,
                post_y,
                thickness=2,
                color=(EDGE_C, EDGE_C, EDGE_C, 64),  # Blue-ish synapse
                batch=batch,
                group=background_group,
            )
            connections[(pre_idx, post_idx)] = new_line
    for pre_idx, post_idx in pruned_synapses:
        # Check if we are tracking this connection
        if (pre_idx, post_idx) in connections:
            # Get the actual Line object
            line_to_delete = connections[(pre_idx, post_idx)]

            # Check if it's not already None (just in case)
            if line_to_delete:
                # Tell Pyglet to delete the visual object
                line_to_delete.delete()

            # Remove the key from our tracking dictionary
            del connections[(pre_idx, post_idx)]
    time_label.text = f"{snn.now_time:.2f}ms"


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('The "A" key was pressed.')
    elif symbol == key.LEFT:
        print("The left arrow key was pressed.")
    elif symbol == key.ENTER:
        print("The enter key was pressed.")


@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print(f"The left mouse button was pressed. [{x}, {y}]")


@window.event
def on_draw():
    window.clear()
    image.blit(window.width - 64, window.height - 64)
    label.draw()
    time_label.draw()
    input_label.draw()
    batch.draw()


pyglet.clock.schedule_interval(update, 1 / 120)
pyglet.app.run()
