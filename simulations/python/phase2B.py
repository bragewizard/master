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
        "font.size": 12,
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
    plt.colorbar(label="Proportion of Predictions")

    plt.xlabel("Predicted Digit", fontsize=10)
    plt.ylabel("Actual Digit", fontsize=10)

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


def plot_cumulative_accuracy():
    """
    Plots accuracy over simulation time,
    should see a rising s curve from 0 to max accuracy
    """


def run_phase2_evaluation():
    print("--- Phase II: Zero-Shot SNN Evaluation (Strict Ops Counting) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = SimpleFCN().to(device)
    ann.load_state_dict(torch.load("phase2_baseline_fcn.pth"))

    snn = SimpleSNN(ann, device=device)
    provider = MNISTProvider()

    test_images, test_labels = provider.get_batch(1000)
    test_images, test_labels = test_images.to(device), test_labels.cpu().numpy()

    # Pass 1: FP32
    print("\n[*] Evaluating FP32 Continuous...")
    snn.w1.data = ann.fc1.weight.data * 64.0
    snn.w2.data = ann.fc2.weight.data * 64.0
    res_fp32 = snn.evaluate_dataset(test_images, test_labels)

    # Pass 2: INT8
    print("[*] Evaluating INT8 Quantized...")
    snn.w1.data = snn.w1.data.round().clamp(-128, 127)
    snn.w2.data = snn.w2.data.round().clamp(-128, 127)
    res_int8 = snn.evaluate_dataset(test_images, test_labels)

    plot_confusion_matrix(res_int8["Confusion_Matrix"], "phase2_confusion_matrix.png")
    plot_cumulative_accuracy()

    for name, res in [("FP32", res_fp32), ("INT8", res_int8)]:
        print(f"\n[+] Results:")
        print(f"    - Accuracy:        {res['Accuracy']:.2f}%")
        print(f"    - Mean Latency:    {res['Mean_Latency']:.1f} Ticks")
        print(f"    - Mean Operations: {res['Mean_Ops']:,.0f} per image")
        print(f"    - Compute Saved:   {res['Efficiency_Gain']:.1f}%")
        print(f"    --- Network Activity ---")
        print(
            f"    - Hidden Sparsity: {res['Mean_Active_Hidden']:.1f} / 128 neurons fired per image"
        )
        print(
            f"    - Hidden DNFs:     {res['Hidden_DNFs']} (Images where NO hidden neurons fired)"
        )
        print(
            f"    - Output DNFs:     {res['Output_DNFs']} (Images where NO decision was made)"
        )


if __name__ == "__main__":
    run_phase2_evaluation()
