import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

df = pd.read_csv("actionpotentials.csv")
threshold = 0.1
offset_step = 2.0

plt.figure(figsize=(14, 10))
time = df.iloc[:, 0].values

for i in range(1, 2):
    col_name = df.columns[i]
    signal = df[col_name].values

    # 1. Create a mask of everything above threshold
    mask = signal > threshold

    # 2. Label each "island" of True values with a unique ID
    # 'labels' will be an array like [0, 0, 1, 1, 1, 0, 2, 2, 0...]
    labels, num_features = label(mask)

    spike_indices = []

    # 3. Iterate through each unique cluster (skipping 0, which is background)
    for cluster_id in range(1, num_features + 1):
        # Find all indices belonging to this specific cluster
        cluster_indices = np.where(labels == cluster_id)[0]

        # Find the index of the maximum value WITHIN this cluster
        peak_idx = cluster_indices[np.argmax(signal[cluster_indices])]
        spike_indices.append(peak_idx)

    # 4. Save to DataFrame
    spike_array = np.zeros(len(df))
    spike_array[spike_indices] = 1
    df[f"{col_name}_Spike"] = spike_array.astype(int)

    # --- Plotting to Verify ---
    y_offset = i * offset_step
    plt.plot(time, signal + y_offset, color="gray", alpha=0.4, linewidth=0.8)

    # Plot the detected peaks
    plt.scatter(
        time[spike_indices],
        signal[spike_indices] + y_offset,
        color="red",
        s=25,
        edgecolor="black",
        zorder=3,
        label=f"Ch{i} Peaks",
    )

plt.title("Cluster-Based Peak Detection (Labeling Algorithm)")
plt.xlabel("Time (s)")
plt.ylabel("Channels (Offset)")
plt.tight_layout()
plt.show()

df.to_csv("actionpotentialfiltered.csv", index=False)
