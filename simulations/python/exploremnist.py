import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from _data import MNISTProvider


def create_thesis_visuals():
    provider = MNISTProvider()
    sns.set_theme(style="whitegrid")  # Clean academic look

    # --- 1. 3x3 Grid of Examples ---
    plt.figure(figsize=(8, 8))
    for i in range(9):
        img, lbl = provider.get_single_visualization_frame()
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Label: {lbl}", fontsize=10)
        plt.axis("off")
    plt.suptitle("Sample MNIST Digits (28x28 pixels)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("mnist_grid.png", dpi=300)
    plt.show()

    # --- 2. Class Distribution (Bar Chart) ---
    dist = provider.get_class_distribution()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(dist.keys()), y=list(dist.values()), palette="viridis")
    plt.title("Distribution of Digits in Training Set", fontsize=14)
    plt.xlabel("Digit")
    plt.ylabel("Number of Samples")
    plt.savefig("mnist_distribution.png", dpi=300)
    plt.show()

    # --- 3. Pixel Intensity Histogram ---
    # Sampling 1000 images for the histogram to represent the general spread
    sample_imgs = provider.images[
        np.random.choice(provider.num_samples, 1000)
    ].flatten()
    plt.figure(figsize=(10, 5))
    plt.hist(sample_imgs, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.yscale("log")  # Log scale helps see the middle-gray values
    plt.title("Pixel Intensity Distribution (Log Scale)", fontsize=14)
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency (Log)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("mnist_histogram.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    create_thesis_visuals()
