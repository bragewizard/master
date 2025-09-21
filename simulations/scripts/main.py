from PIL import Image
from data import generate_checkerboard, generate_spatiotemporal_stimuli
from visualize import visualize_image
from experiments import pipeline
# from gui import run_gui
from gui2 import run_gui
from MNIST import parse_MNIST, MNIST_train_images_path, MNIST_test_images_path

if __name__ == "__main__":
    # image_path = 'data/doomguy.jpg'
    image_path = 'data/lenna.png'
    original_image = Image.open(image_path)
    grayscale_image = original_image.convert('L')
    MNIST = parse_MNIST(MNIST_train_images_path)
    max_dim = 16
    low_res_image_pil = grayscale_image.resize((max_dim, max_dim), Image.Resampling.LANCZOS)
    # input_image = np.array(low_res_image_pil)
    input_image = generate_checkerboard(size=max_dim, block_size=int(max_dim/4))
    # input_image = MNIST[0]
    # visualize_image(input_image)
    run_gui(input_image)
