import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from _fcn import SimpleFCN
from _snn import SimpleSNN
from _data import MNISTProvider

# Set global font family and size
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Geist"],
        "font.weight": "medium",
        "font.size": 12,
    }
)


def plot_receptive_fields(weights, title, filename):
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    # Dynamic scale: Center at 0, fit to the max absolute value of the current weights
    max_w = weights.abs().max().item()
    vmin = -max_w
    vmax = max_w

    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            img = weights[i].reshape(28, 28).cpu().numpy()
            im = ax.imshow(img, cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.axis("off")

    # Add a colorbar to the side so the scale is explicit
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {filename}")


def plot_confusion_matrix(matrix, filename):
    plt.figure(figsize=(10, 8))
    # Normalize by row (actual class) to show percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0
    )

    plt.imshow(norm_matrix, cmap="Blues")
    plt.colorbar(label="Proportion of Predictions")

    plt.xlabel("Predicted Digit", fontsize=12)
    plt.ylabel("Actual Digit", fontsize=12)

    ticks = np.arange(10)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    # Add text annotations
    for i in range(10):
        for j in range(10):
            color = "white" if norm_matrix[i, j] > 0.5 else "black"
            plt.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  -> Saved {filename}")


def run_phase3_full_pipeline():
    print("--- Phase III: Unsupervised STDP Training & Routing ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_ann = SimpleFCN().to(device)
    snn = SimpleSNN(dummy_ann, device=device)

    snn.w1.data = torch.rand_like(snn.w1.data) * 1.0
    snn.w2.data.zero_()

    plot_receptive_fields(
        snn.w1.data,
        "Hidden Layer (t=0): Random Noise",
        "phase3_01_receptive_before.png",
    )

    provider = MNISTProvider()
    num_train_samples = 5000
    train_images, train_labels = provider.get_batch(num_train_samples)
    train_images = train_images.to(device)

    K_target_h = snn.w1.data.sum(dim=1).mean().item() * 0.5

    win_counts = torch.zeros(snn.num_hidden, device=device)
    dnf_count_train = 0

    print(f"\n[*] 1. Running Training ({num_train_samples} images)...")
    for i in range(num_train_samples):
        # 1. Decay all thresholds (Homeostasis)
        snn.adaptive_thresholds *= snn.theta_decay

        img_tensor = train_images[i : i + 1]
        input_delays = snn.encode_to_delays(img_tensor)

        # 2. Run Forward Pass (Turn Lateral Inhibition ON)
        _ = snn.run_saccade(img_tensor, model_type="C", use_lateral=False)

        winning_neurons = snn.apply_stdp(
            input_delays,
            snn.last_spikes_h,
            snn.w1.data,
            k_target=K_target_h,
            use_wta=True,
        )

        if len(winning_neurons) > 0:
            # We convert winners to a tensor to update thresholds all at once
            winner_tensor = torch.tensor(winning_neurons, device=device)
            snn.adaptive_thresholds[winner_tensor] += snn.theta_plus

            # Increment win counts for each neuron in the list
            for w_idx in winning_neurons:
                win_counts[w_idx] += 1
        else:
            dnf_count_train += 1

        # Simple terminal tracker
        if (i + 1) % 100 == 0:
            active_neurons = (win_counts > 0).sum().item()
            print(
                f"  -> {i + 1}/{num_train_samples} | Active Neurons: {active_neurons}/{snn.num_hidden} | DNFs: {dnf_count_train}"
            )

    plot_receptive_fields(
        snn.w1.data,
        f"Hidden Layer (t={num_train_samples}): STDP Emergence",
        "phase3_02_receptive_after.png",
    )

    print("\n[*] 2. Running Post-Hoc Label Assignment...")
    val_images, val_labels = provider.get_batch(1000)
    val_images, val_labels = val_images.to(device), val_labels.cpu().numpy()

    label_matrix = np.zeros((snn.num_hidden, 10))
    snn.adaptive_thresholds.zero_()  # Disable threshold penalties for deterministic inference!

    snn.thresholds["C"] = 180  # set inference threshold

    for i in range(1000):
        img_tensor = val_images[i : i + 1]
        _ = snn.run_saccade(img_tensor, model_type="C")

        fired_neurons = torch.where(snn.last_spikes_h != -1)[0]
        for winner in fired_neurons:
            label_matrix[winner.item(), val_labels[i]] += 1

    assigned_labels = np.argmax(label_matrix, axis=1)

    for j in range(snn.num_hidden):
        assigned_class = assigned_labels[j]
        snn.w2.data[assigned_class, j] = snn.thresholds["C"] * 2

    print("\n[*] 4. Final Evaluation (Full Network Inference)...")
    test_images, test_labels = provider.get_batch(1000)
    test_images, test_labels = test_images.to(device), test_labels.cpu().numpy()

    results = snn.evaluate_dataset(test_images, test_labels, model_type="C")
    plot_confusion_matrix(results["Confusion_Matrix"], "phase3_04_confusion_matrix.png")

    print(f"\n[+] Phase 3 Final STDP Results:")
    print(f"    - Accuracy:        {results['Accuracy']:.2f}%")
    print(f"    - Mean Latency:    {results['Mean_Latency']:.1f} Ticks")
    print(f"    - Mean Operations: {results['Mean_Ops']:,.0f} per image")
    print(f"    - Compute Saved:   {results['Efficiency_Gain']:.1f}%")
    print(f"    --- Network Activity ---")
    print(
        f"    - Hidden Sparsity: {results['Mean_Active_Hidden']:.1f} / 128 neurons fired per image"
    )
    print(
        f"    - Hidden DNFs:     {results['Hidden_DNFs']} (Images where NO hidden neurons fired)"
    )
    print(
        f"    - Output DNFs:     {results['Output_DNFs']} (Images where NO decision was made)"
    )

    # [*] 5. Visualizing Phase 2 FCN Baseline Weights for Comparison
    print("\n[*] 5. Visualizing Phase 2 FCN Baseline Weights for Comparison...")
    try:
        baseline_ann = SimpleFCN().to(device)
        baseline_ann.load_state_dict(
            torch.load("phase2_baseline_fcn.pth", map_location=device)
        )

        # 1. Scale the FCN weights exactly like we do for the SNN import
        fcn_weights_scaled = baseline_ann.fc1.weight.data * 64.0

        # 2. Plot ANN Weights (dynamically scaled to its own max)
        plot_receptive_fields(
            fcn_weights_scaled,
            "Phase 2 Baseline FCN Weights (ANN - Scaled x64)",
            "phase3_03_baseline_fcn_weights.png",
        )

        # Note: Phase 3 STDP weights were already plotted earlier as
        # "figures/phase3_02_receptive_after.png", so we don't need to plot them again.

    except FileNotFoundError:
        print("  -> [!] Could not find 'phase2_baseline_fcn.pth'.")


if __name__ == "__main__":
    run_phase3_full_pipeline()
