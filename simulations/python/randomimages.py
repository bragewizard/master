from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from visualize import visualize_image
from data import create_random_image_one_channel, to_grayscale, three_to_one_channel, one_to_three_channel

def luminance_to_vector_2d(luminance_image: np.ndarray) -> np.ndarray:
    luminance_image = luminance_image.flatten()
    angles = (luminance_image / 255.0) * 2 * np.pi
    vectors = np.zeros(luminance_image.shape + (2,), dtype=np.float32)
    vectors[..., 0] = np.cos(angles)  # x component
    vectors[..., 1] = np.sin(angles)  # y component
    return vectors

def average_images(image_list):
    stacked_images = np.stack(image_list, axis=0).astype(np.float32)
    averaged_image = np.mean(stacked_images, axis=0)
    return averaged_image.astype(np.uint8)

if __name__ == "__main__":
    pattern = [8,14,10],[7,12,17]
    # pattern_color = (48, 158, 100) # arbitrary
    pattern_color = (255, 255, 255) # arbitrary

    vectors = []
    images = []
    for i in range(30):
        image1 = create_random_image_one_channel(28,28)
        image1[pattern] = pattern_color
        vector1 = luminance_to_vector_2d(image1)
        origin = np.zeros((2,vector1.shape[0]))
        images.append(image1)
        vectors.append(vector1)

    vectors = np.array(vectors)
    visualize_image(images[0])
    # for i in range(30):
        # visualize_image(images[i])
    fig,ax = plt.subplots()
    ax.quiver(*origin,vectors[0,:,0],vectors[0,:,1],scale=2)
    ax.set_aspect('equal')
    plt.show()
    summed_vectors = np.sum(vectors, axis=0)
    fig,ax = plt.subplots()
    ax.quiver(*origin,summed_vectors[:,0],summed_vectors[:,1],scale=60)
    ax.set_aspect('equal')
    plt.show()
    magnitudes = np.linalg.norm(summed_vectors, axis=-1)
    num_images = len(vectors)
    processed_image = (magnitudes / num_images) * 255.0
    visualize_image(processed_image.reshape(28,28))
