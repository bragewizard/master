import numpy as np
import torch
from datasets import load_dataset


class MNISTProvider:
    def __init__(self):
        print("Loading MNIST from Hugging Face...")
        dataset = load_dataset("mnist", split="train")

        self.images = np.array([np.array(x) for x in dataset["image"]])
        self.labels = np.array(dataset["label"])
        self.num_samples = len(self.images)
        print(f"Loaded {self.num_samples} samples.")

    def get_batch(self, batch_size=32):
        indices = np.random.choice(self.num_samples, batch_size)
        imgs = self.images[indices].astype(np.float32) / 255.0
        imgs_tensor = torch.from_numpy(imgs).unsqueeze(1)
        lbls_tensor = torch.from_numpy(self.labels[indices]).long()
        return imgs_tensor, lbls_tensor

    def get_class_distribution(self):
        """Returns counts for each digit (0-9)."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_single_visualization_frame(self):
        idx = np.random.randint(0, self.num_samples)
        return self.images[idx], self.labels[idx]
