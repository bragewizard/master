import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import csv
from _snn import SimpleSNN

# Set global font family and size
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Geist"],
        "font.weight": "medium",
        "font.size": 18,
    }
)


# --- 1. DUMMY CNN FOR INITIALIZATION ---
class DummyFCN(nn.Module):
    def __init__(self):
        super(DummyFCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)


def run_phase1_experiment():
    print("--- Phase I: Temporal Dynamics & Threshold Sweeps ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_model = DummyFCN()
    snn = SimpleSNN(dummy_model, device=device)

    # --- 2. ISOLATED UNIT SETUP ---
    snn.w1.zero_()
    weights = [100.0, 80.0, 60.0, 40.0, 20.0]
    for i, w in enumerate(weights):
        snn.w1[0, i] = w

    snn.gamma = 1.0
    snn.tau_m = 20.0
    weight_sum = sum(weights)

    # --- 3. SYNTHETIC SPIKE TRAINS ---
    c_delays = [2, 6, 10, 14, 18]
    d_delays = [18, 14, 10, 6, 2]

    delays_concordant = torch.full((784,), 63, device=device, dtype=torch.long)
    delays_concordant[0:5] = torch.tensor(c_delays, device=device)

    delays_discordant = torch.full((784,), 63, device=device, dtype=torch.long)
    delays_discordant[0:5] = torch.tensor(d_delays, device=device)

    # --- 4. THRESHOLD REGIMES ---
    regimes = {
        "Low (Saturation)": {"A": 150.0, "B": 140.0, "C": 500.0, "D": 100.0},
        "Critical (Balanced)": {"A": 290.0, "B": 220.0, "C": 1300.0, "D": 200.0},
        "High (Deficit)": {"A": 310.0, "B": 310.0, "C": 2000.0, "D": 310.0},
    }

    models = {
        "A": ("Simple IF", snn.run_model_a_integrator),
        "B": ("Standard LIF", snn.run_model_b_lif),
        "C": ("Linear Ramp", snn.run_model_c_linear_ramp),
        "D": ("State Discount", snn.run_model_d_threshold_sensitive),
    }

    results_data = {r: {} for r in regimes.keys()}
    csv_data = []  # For CeTZ export

    # --- 5. EXECUTE SWEEPS ---
    for regime_name, thresholds in regimes.items():
        print(f"\nEvaluating Regime: {regime_name}")
        for mod_key, (mod_name, run_func) in models.items():
            thresh = thresholds[mod_key]
            snn.thresholds[mod_key] = thresh

            # Concordant
            spikes_h_c = torch.full(
                (snn.num_hidden,), -1, device=device, dtype=torch.long
            )
            spikes_o_c = torch.full(
                (snn.num_output,), -1, device=device, dtype=torch.long
            )
            run_func(delays_concordant, spikes_h_c, spikes_o_c)
            t_c = spikes_h_c[0].item()

            # Discordant
            spikes_h_d = torch.full(
                (snn.num_hidden,), -1, device=device, dtype=torch.long
            )
            spikes_o_d = torch.full(
                (snn.num_output,), -1, device=device, dtype=torch.long
            )
            run_func(delays_discordant, spikes_h_d, spikes_o_d)
            t_d = spikes_h_d[0].item()

            results_data[regime_name][mod_name] = (t_c, t_d)

            # Store for CSV
            str_tc = t_c if t_c != -1 else "DNF"
            str_td = t_d if t_d != -1 else "DNF"
            csv_data.append(
                [regime_name, mod_name, thresh, weight_sum, "Concordant", str_tc]
            )
            csv_data.append(
                [regime_name, mod_name, thresh, weight_sum, "Discordant", str_td]
            )

            print(
                f"  {mod_name:<15} [Thresh: {thresh:<6}] | Conc: {str_tc:<4} | Disc: {str_td:<4}"
            )

    # --- 6. EXPORT CSV FOR Typst/CeTZ ---
    with open("phase1_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Regime", "Model", "Threshold", "WeightSum", "Pattern", "SpikeTime"]
        )
        writer.writerows(csv_data)
    print("\n[+] Exported raw data to 'phase1_results.csv' for CeTZ visualization.")

    # --- 7. COMPOSITE VISUALIZATION ---
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3)
    # Bottom Row: Regime Results
    axes_bars = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2]),
    ]

    model_names = [m[0] for m in models.values()]
    x = np.arange(len(model_names))
    width = 0.35

    for ax, (regime_name, data) in zip(axes_bars, results_data.items()):
        c_times = [data[m][0] if data[m][0] != -1 else 64 for m in model_names]
        d_times = [data[m][1] if data[m][1] != -1 else 64 for m in model_names]

        ax.bar(x - width / 2, c_times, width, color="#93c572")
        ax.bar(x + width / 2, d_times, width, color="#a2bffe")

        for i, (ct, dt) in enumerate(zip(c_times, d_times)):
            if ct == 64:
                ax.text(
                    x[i] - width / 2,
                    60,
                    "DNF",
                    ha="center",
                    color="black",
                    weight="bold",
                    fontsize=10,
                )
            if dt == 64:
                ax.text(
                    x[i] + width / 2,
                    60,
                    "DNF",
                    ha="center",
                    color="black",
                    weight="bold",
                    fontsize=10,
                )

        ax.set_title(regime_name, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15)
        ax.axhline(y=64, color="gray", linestyle="--", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0, 68)

    axes_bars[0].set_ylabel("Output Spike Latency (Ticks)")
    plt.tight_layout()  # Automatically adjusts margins
    plt.savefig("phase1_composite_sweep.png", dpi=300)
    print("[+] Saved composite plot to 'phase1_composite_sweep.png'.")
    plt.show()


if __name__ == "__main__":
    run_phase1_experiment()
