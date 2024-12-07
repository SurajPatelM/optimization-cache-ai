import matplotlib.pyplot as plt
import numpy as np

# Data
cache_block_sizes = [8, 16, 32, 64, 128, 256]
thresholds = {
    "Threshold-1": [888696, 830923, 762851, 690955, 621653, 504151],
    "Threshold-2": [888716, 829874, 762079, 689374, 622071, 501680],
    "Threshold-4": [893828, 831343, 763110, 689148, 622213, 501791],
    "Threshold-8": [889928, 833828, 764591, 690349, 622055, 504493],
    "Threshold-16": [890184, 832967, 767478, 691316, 622336, 506645],
}

# Normalize the data (optional)
max_value = max(max(values) for values in thresholds.values())
normalized_thresholds = {key: [v / max_value * 100 for v in values] for key, values in thresholds.items()}

# Function to plot
def plot_trends():
    plt.figure(figsize=(14, 8))

    # Line plot for normalized values
    for threshold, values in normalized_thresholds.items():
        plt.plot(
            cache_block_sizes,
            values,
            marker="o",
            label=threshold,
            linewidth=2,
            markersize=8,
        )

    # Customization
    plt.title("Cache Block Size vs Normalized Thresholds", fontsize=16)
    plt.xlabel("Cache Block Size", fontsize=14)
    plt.ylabel("Normalized Value (%)", fontsize=14)
    plt.xticks(cache_block_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.show()

def plot_bar_comparison():
    plt.figure(figsize=(14, 8))
    bar_width = 0.15
    x = np.arange(len(cache_block_sizes))

    for i, (threshold, values) in enumerate(normalized_thresholds.items()):
        plt.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=threshold,
            edgecolor="black",
            alpha=0.8,
        )

    # Customization
    plt.title("Cache Block Size vs Normalized Thresholds (Bar Plot)", fontsize=16)
    plt.xlabel("Cache Block Size", fontsize=14)
    plt.ylabel("Normalized Value (%)", fontsize=14)
    plt.xticks(x + bar_width * 2, cache_block_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.show()

def plot_stacked_bars():
    plt.figure(figsize=(14, 8))
    cumulative_values = np.zeros(len(cache_block_sizes), dtype=np.float64)

    for threshold, values in normalized_thresholds.items():
        plt.bar(
            cache_block_sizes,
            values,
            bottom=cumulative_values,
            label=threshold,
            edgecolor="black",
            alpha=0.8,
        )
        cumulative_values += np.array(values)

    # Customization
    plt.title("Cache Block Size vs Normalized Thresholds (Stacked Bar Plot)", fontsize=16)
    plt.xlabel("Cache Block Size", fontsize=14)
    plt.ylabel("Cumulative Normalized Value (%)", fontsize=14)
    plt.xticks(cache_block_sizes, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.show()

def plot_subplots():
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle("Cache Block Size vs Thresholds (Subplots)", fontsize=18)

    axes = axes.flatten()
    for i, (threshold, values) in enumerate(thresholds.items()):
        axes[i].plot(
            cache_block_sizes, values, marker="o", linewidth=2, markersize=8, label=threshold
        )
        axes[i].set_title(threshold, fontsize=14)
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].set_xlabel("Cache Block Size", fontsize=12)
        axes[i].set_ylabel("Value", fontsize=12)
        axes[i].legend(fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for title
    plt.show()

# Call the plotting functions
plot_trends()
plot_bar_comparison()
plot_stacked_bars()
plot_subplots()
