import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import csv
from _fcn import SimpleFCN
from _snn import SimpleSNN
from _data import MNISTProvider


def evaluate_snn_mode(snn, images, labels, mode_name, num_samples):
    print(f"\n[*] Evaluating SNN Mode: {mode_name}...")
    predictions = []
    spike_times = []
    total_synops = []

    # The dense baseline for 784->128->10
    # 784*128 + 128*10 = 101,632 operations
    ANN_MACs = 101632

    for i in range(num_samples):
        img_tensor = images[i : i + 1]  # [1, 1, 28, 28]
        out_spikes = snn.run_saccade(img_tensor, model_type="C")

        valid_spikes = out_spikes.clone()
        valid_spikes[valid_spikes == -1] = 999

        if valid_spikes.min() == 999:
            pred, t_fire = -1, 64
        else:
            pred = torch.argmin(valid_spikes).item()
            t_fire = valid_spikes.min().item()

        predictions.append(pred)
        spike_times.append(t_fire)

        # --- COMPUTE SYNOPS ---
        # Assuming your input delay mapping is (1.0 - pixel) * 32
        # Any pixel with a delay <= t_fire generated a spike before the network decided
        pixel_intensities = img_tensor.view(-1)
        delays = ((1.0 - pixel_intensities) * 32).long()

        input_spikes_processed = (delays <= t_fire).sum().item()

        # SynOps = (Input Spikes * 128 hidden neurons) + (Hidden Spikes * 10)
        # We estimate hidden spikes as ~10% active for this calculation
        synops = (input_spikes_processed * 128) + (12 * 10)
        total_synops.append(synops)

        if (i + 1) % 200 == 0:
            print(f"  -> {i + 1}/{num_samples} images processed")

    preds = np.array(predictions)
    times = np.array(spike_times)
    synops_arr = np.array(total_synops)

    correct_mask = preds == labels
    final_acc = (correct_mask.sum() / num_samples) * 100

    # Calculate Latency and Efficiency
    mean_t_fire = times[times != 64].mean()
    mean_synops = synops_arr.mean()
    compute_reduction = ((ANN_MACs - mean_synops) / ANN_MACs) * 100

    # Calculate Temporal Curve
    temporal_acc = np.zeros(64)
    for t in range(64):
        temporal_acc[t] = ((correct_mask) & (times <= t)).sum() / num_samples * 100

    print(f"\n[+] {mode_name} Results:")
    print(f"    - Final Accuracy:     {final_acc:.2f}%")
    print(f"    - Mean Decision Time: {mean_t_fire:.1f} Ticks")
    print(f"    - Mean SynOps/Image:  {mean_synops:,.0f} (vs {ANN_MACs:,} MACs)")
    print(f"    - Compute Reduction:  {compute_reduction:.1f}%")

    return final_acc, temporal_acc, preds, mean_t_fire


def run_phase2_evaluation():
    print("--- Phase II: Zero-Shot SNN Transfer Evaluation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann = SimpleFCN().to(device)
    ann.load_state_dict(torch.load("phase2_baseline_fcn.pth"))

    snn = SimpleSNN(ann, device=device)
    snn.threshold_h["C"] = 1800.0

    provider = MNISTProvider()
    num_samples = 1000
    test_images, test_labels = provider.get_batch(num_samples)
    test_images = test_images.to(device)
    test_labels = test_labels.cpu().numpy()

    # --- ISOLATE WEIGHTS ---
    w1_fp32 = ann.fc1.weight.data * 64.0
    w2_fp32 = ann.fc2.weight.data * 64.0

    w1_int8 = w1_fp32.round().clamp(-128, 127)
    w2_int8 = w2_fp32.round().clamp(-128, 127)

    # --- PASS 1: SNN (Continuous FP32) ---
    snn.w1.data = w1_fp32
    snn.w2.data = w2_fp32
    acc_fp32, curve_fp32, _, t_fp32 = evaluate_snn_mode(
        snn, test_images, test_labels, "FP32 Continuous", num_samples
    )

    # --- PASS 2: SNN (Discrete INT8) ---
    snn.w1.data = w1_int8
    snn.w2.data = w2_int8
    acc_int8, curve_int8, preds_int8, t_int8 = evaluate_snn_mode(
        snn, test_images, test_labels, "INT8 Quantized", num_samples
    )

    # --- VISUALIZATION: DUAL TEMPORAL CURVE ---
    plt.figure(figsize=(9, 5))
    plt.plot(
        range(64),
        curve_fp32,
        color="blue",
        linewidth=2.5,
        label=f"SNN FP32 (Peak: {acc_fp32:.1f}%)",
    )
    plt.plot(
        range(64),
        curve_int8,
        color="orange",
        linewidth=2.5,
        linestyle="--",
        label=f"SNN INT8 (Peak: {acc_int8:.1f}%)",
    )

    plt.axvline(
        x=t_fp32,
        color="blue",
        linestyle=":",
        alpha=0.6,
        label=f"FP32 Mean Decision (t={t_fp32:.1f})",
    )
    plt.axvline(
        x=t_int8,
        color="orange",
        linestyle=":",
        alpha=0.6,
        label=f"INT8 Mean Decision (t={t_int8:.1f})",
    )

    plt.title("Phase II: Temporal Accuracy & Latency (Model C)")
    plt.xlabel("Simulation Time (Ticks)")
    plt.ylabel("Cumulative Accuracy (%)")
    plt.grid(axis="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("snn_temporal_comparison.png", dpi=300)
    print("\n[+] Saved 'snn_temporal_comparison.png'")


if __name__ == "__main__":
    run_phase2_evaluation()
