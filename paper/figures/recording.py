import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

data = scipy.io.loadmat("1554.mat")

rawdata = data["rawdata"]
df = pd.DataFrame(rawdata, columns=["time", "ch1", "ch2", "ch3", "ch4", "ch5"])


# 1. Define your window of interest (in seconds)
t_start = 0.114
t_end = 0.132

# 2. Filter the data for that time range
# Assuming column 0 is 'Time'
mask = (df.iloc[:, 0] >= t_start) & (df.iloc[:, 0] <= t_end)
zoom_df = df.loc[mask]

# 3. Plotting with offsets
plt.figure(figsize=(14, 6))

time_zoom = zoom_df.iloc[:, 0]
offset_step = 0.5  # Adjust this based on your signal scale

for i in range(1, 6):
    # Center each channel at 0 first so the offset is predictable
    signal = zoom_df.iloc[:, i]
    centered_signal = signal - signal.mean()

    plt.plot(time_zoom, centered_signal + (i * offset_step), label=f"Ch {i}")

# 4. Clean up the view
plt.xlim(t_start, t_end)
plt.title(f"Zoomed Neural Activity: {t_start}s to {t_end}s")
plt.xlabel("Time (s)")
plt.ylabel("Channels (Offset)")
plt.legend(loc="upper right")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.show()

zoom_df.to_csv("actionpotentials.csv", index=False)
