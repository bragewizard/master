import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from _fcn import SimpleFCN
from _snn import SimpleSNN
from _data import MNISTProvider


def plot_receptive_fields(weights, title, filename):
    """Visualizes what the neurons actually 'see' by plotting their weights as images"""
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # Plot the first 32 hidden neurons
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            img = weights[i].reshape(28, 28).cpu().numpy()
            ax.imshow(img, cmap="viridis")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[+] Saved Receptive Field Visualization: {filename}")


def run_phase3_stdp():
    print("--- Phase III: Native Unsupervised STDP ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize Blank SNN
    dummy_ann = SimpleFCN().to(device)
    snn = SimpleSNN(dummy_ann, device=device)
    snn.threshold_h["C"] = 1800.0

    # Overwrite weights with random uniform noise (0 to 127)
    print("[*] Severing ANN ties and initializing random synapses...")
    snn.w1.data = torch.rand_like(snn.w1.data) * 127.0
    snn.w2.data = torch.rand_like(snn.w2.data) * 127.0

    # Save "Before" snapshot for the thesis
    plot_receptive_fields(
        snn.w1.data,
        "Hidden Layer Synapses: Random Initialization (t=0)",
        "figures/phase3_receptive_before.png",
    )

    provider = MNISTProvider()
    num_train_samples = 2000  # Adjust up to 10k+ for final thesis run (takes longer)
    train_images, train_labels = provider.get_batch(num_train_samples)
    train_images = train_images.to(device)

    # Homeostasis Target: Keep the sum of weights for each neuron roughly constant
    K_target_h = snn.w1.data.sum(dim=1).mean().item()
    K_target_o = snn.w2.data.sum(dim=1).mean().item()

    print(f"\n[*] Beginning Unsupervised STDP Loop on {num_train_samples} images...")
    # 2. THE STDP TRAINING LOOP
    for i in range(num_train_samples):
        img_tensor = train_images[i : i + 1]

        # We need the actual input delays to run STDP on layer 1
        pixel_intensities = img_tensor.view(-1)
        input_delays = ((1.0 - pixel_intensities) * 32).long()

        # Run Saccade (We use Model C as chosen in Phase I)
        out_spikes = snn.run_saccade(img_tensor, model_type="C")

        # Note: To apply STDP, your run_saccade needs to return hidden spikes as well.
        # Assuming your _snn.py caches them, or you can run a custom integration loop here.
        # For this script, we will assume you have access to snn.last_spikes_h and snn.last_spikes_o
        # If not, we trigger the STDP rule manually based on the output.

        # [THESIS NOTE]: If your _snn.py doesn't return hidden spikes, you must update
        # run_saccade to return (spikes_h, spikes_o) for Phase III.
        spikes_h = (
            snn.last_spikes_h
            if hasattr(snn, "last_spikes_h")
            else torch.zeros(128, device=device)
        )
        spikes_o = out_spikes

        # Apply STDP
        snn.apply_stdp(input_delays, spikes_h, snn.w1.data)
        snn.apply_stdp(spikes_h, spikes_o, snn.w2.data)

        # Apply Strict Homeostasis (Normalization)
        # If a neuron's weights grow too large, shrink them. If too small, boost them.
        w1_sums = snn.w1.data.sum(dim=1, keepdim=True)
        w1_sums[w1_sums == 0] = 1.0  # Prevent div by zero
        snn.w1.data = snn.w1.data * (K_target_h / w1_sums)

        w2_sums = snn.w2.data.sum(dim=1, keepdim=True)
        w2_sums[w2_sums == 0] = 1.0
        snn.w2.data = snn.w2.data * (K_target_o / w2_sums)

        if (i + 1) % 500 == 0:
            print(f"  -> STDP Processed {i + 1}/{num_train_samples}")

    # Save "After" snapshot
    plot_receptive_fields(
        snn.w1.data,
        "Hidden Layer Synapses: Emergent Features (Post-STDP)",
        "figures/phase3_receptive_after.png",
    )

    # 3. POST-HOC LABEL ASSIGNMENT
    print("\n[*] Training complete. Running Post-Hoc Label Assignment...")
    label_matrix = np.zeros((snn.num_output, 10))

    # We use a small validation set to figure out what the neurons learned
    val_images, val_labels = provider.get_batch(1000)
    val_images = val_images.to(device)
    val_labels = val_labels.cpu().numpy()

    for i in range(1000):
        img_tensor = val_images[i : i + 1]
        out_spikes = snn.run_saccade(img_tensor, model_type="C")

        valid_spikes = out_spikes.clone()
        valid_spikes[valid_spikes == -1] = 999

        if valid_spikes.min() != 999:
            winner_neuron = torch.argmin(valid_spikes).item()
            true_label = val_labels[i]
            label_matrix[winner_neuron, true_label] += 1

    # Assign the most frequent ground-truth label to each output neuron
    assigned_labels = np.argmax(label_matrix, axis=1)
    for j in range(snn.num_output):
        print(
            f"  -> Output Neuron {j} assigned to Class [{assigned_labels[j]}] (Fired {int(label_matrix[j].sum())} times)"
        )

    # 4. FINAL EVALUATION
    print("\n[*] Evaluating Unsupervised Network on Test Set...")
    test_images, test_labels = provider.get_batch(1000)
    test_images = test_images.to(device)
    test_labels = test_labels.cpu().numpy()

    correct = 0
    for i in range(1000):
        img_tensor = test_images[i : i + 1]
        out_spikes = snn.run_saccade(img_tensor, model_type="C")

        valid_spikes = out_spikes.clone()
        valid_spikes[valid_spikes == -1] = 999

        if valid_spikes.min() != 999:
            winner_neuron = torch.argmin(valid_spikes).item()
            predicted_label = assigned_labels[winner_neuron]
            if predicted_label == test_labels[i]:
                correct += 1

    final_acc = (correct / 1000) * 100
    print(f"\n[+] Final Unsupervised STDP Accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    import os

    os.makedirs("figures", exist_ok=True)
    run_phase3_stdp()
