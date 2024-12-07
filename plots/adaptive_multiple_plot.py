import matplotlib.pyplot as plt

# Data
cache_block_sizes = [8, 16, 32, 64, 128, 256]
thresholds = {
    "Threshold-1": [888696, 830923, 762851, 690955, 621653, 504151],
    "Threshold-2": [888716, 829874, 762079, 689374, 622071, 501680],
    "Threshold-4": [893828, 831343, 763110, 689148, 622213, 501791],
    "Threshold-8": [889928, 833828, 764591, 690349, 622055, 504493],
    "Threshold-16": [890184, 832967, 767478, 691316, 622336, 506645],
}

# Determine layout
num_plots = len(thresholds)
num_cols = 2  # Number of columns for subplots
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate required rows

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten in case of multiple rows/columns
fig.suptitle("Cache Block Size vs Thresholds (Subplots)", fontsize=18)

# Plot each threshold
for i, (threshold, values) in enumerate(thresholds.items()):
    axes[i].plot(
        cache_block_sizes, values, marker="o", linewidth=2, markersize=8, label=threshold
    )
    axes[i].set_title(threshold, fontsize=14)
    axes[i].grid(True, linestyle="--", alpha=0.6)
    axes[i].set_xlabel("Cache Block Size", fontsize=12)
    axes[i].set_ylabel("No. of misses", fontsize=12)
    axes[i].legend(fontsize=10)

# Remove extra empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for the main title
plt.show()
