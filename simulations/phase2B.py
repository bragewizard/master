import torch
from _fcn import SimpleFCN
from _snn import SimpleSNN
from _data import MNISTProvider
import matplotlib.pyplot as plt
import numpy as np

# Set global font family and size
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Geist"],
        "font.weight": "medium",
        "font.size": 14,
    }
)


def plot_confusion_matrix(matrix, filename):
    plt.figure(figsize=(10, 8))
    # Normalize by row (actual class) to show percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0
    )

    plt.imshow(norm_matrix, cmap="Blues")
    plt.colorbar()

    plt.xlabel("Predicted Digit", fontsize=16)
    plt.ylabel("Actual Digit", fontsize=16)

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


def plot_cumulative_accuracy(
    latencies, correctness, max_time=64, filename="phase2_cumulative_accuracy.png"
):
    """
    Plots accuracy over simulation time.
    Shows a rising S-curve from 0 to max accuracy as evidence integrates over time.
    """
    latencies = np.array(latencies)
    correctness = np.array(correctness)
    total_images = len(latencies)

    times = np.arange(0, max_time + 1)
    accuracies = []

    for t in times:
        # Count how many images were correctly classified at OR before tick t
        # (Ignore DNF images where latency might be recorded as 999 or -1)
        correct_by_t = np.sum(correctness[latencies <= t])
        acc = (correct_by_t / total_images) * 100.0
        accuracies.append(acc)

    plt.figure(figsize=(8, 6))
    plt.plot(times, accuracies, color="#1f77b4", linewidth=3, label="FP32 SNN")

    plt.xlabel("Simulation Tick", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.xlim(0, max_time)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Highlight the absolute maximum accuracy achieved
    max_acc = max(accuracies)
    max_acc_time = times[np.argmax(accuracies)]
    plt.scatter([max_acc_time], [max_acc], color="green", zorder=5, s=60)
    plt.annotate(
        f"{max_acc:.1f}%",
        (max_acc_time, max_acc),
        textcoords="offset points",
        xytext=(-25, -15),
        ha="center",
        fontsize=14,
        weight="bold",
        color="green",
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {filename}")


def run_phase2_evaluation():
    print("--- Phase II: Zero-Shot SNN Evaluation (FP32 Baseline) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = SimpleFCN().to(device)
    ann.load_state_dict(torch.load("phase2_baseline_fcn.pth"))

    snn = SimpleSNN(ann, device=device)
    provider = MNISTProvider()

    test_images, test_labels = provider.get_batch(1000)
    test_images, test_labels = test_images.to(device), test_labels.cpu().numpy()

    # Pass 1: FP32 Zero-Shot Transfer
    print("\n[*] Evaluating FP32 Continuous...")
    # Scale continuous weights to match TTFS threshold scaling
    snn.w1.data = ann.fc1.weight.data * 64.0
    snn.w2.data = ann.fc2.weight.data * 64.0

    res_fp32 = snn.evaluate_dataset(test_images, test_labels)

    # Generate the academic plots for the thesis
    plot_confusion_matrix(res_fp32["Confusion_Matrix"], "phase2_confusion_matrix.png")

    # Pass the raw arrays of latencies and correct booleans into the plotting function
    plot_cumulative_accuracy(
        res_fp32["All_Latencies"],
        res_fp32["All_Correct"],
        max_time=64,
        filename="phase2_cumulative_accuracy.png",
    )

    print(f"\n[+] Results (FP32 SNN):")
    print(f"    - Accuracy:        {res_fp32['Accuracy']:.2f}%")
    print(f"    - Mean Latency:    {res_fp32['Mean_Latency']:.1f} Ticks")
    print(f"    - Mean Operations: {res_fp32['Mean_Ops']:,.0f} per image")
    print(f"    - Compute Saved:   {res_fp32['Efficiency_Gain']:.1f}%")
    print(f"    --- Network Activity ---")
    print(
        f"    - Hidden Sparsity: {res_fp32['Mean_Active_Hidden']:.1f} / 128 neurons fired per image"
    )
    print(
        f"    - Hidden DNFs:     {res_fp32['Hidden_DNFs']} (Images where NO hidden neurons fired)"
    )
    print(
        f"    - Output DNFs:     {res_fp32['Output_DNFs']} (Images where NO decision was made)"
    )


if __name__ == "__main__":
    run_phase2_evaluation()
