import matplotlib.pyplot as plt
import numpy as np

# Data
cache_block_sizes = [8, 16, 32, 64, 128, 256]
models = {
    "FIFO": [898900, 842107, 773857, 700620, 633247, 519654],
    "LRU": [883320, 825447, 756443, 685522, 616437, 501024],
    "Perceptron": [883320, 652615, 652615, 551735, 458196, 373448],
    "ML Model": [907979, 860932, 793717, 727604, 665577, 548945],
}

# Colors for better visualization
colors = ["blue", "green", "red", "purple"]

# 1. Line Plot for All Policies
plt.figure(figsize=(12, 6))
for i, (model, values) in enumerate(models.items()):
    plt.plot(cache_block_sizes, values, marker="o", linewidth=2, label=model, color=colors[i])
plt.title("Cache Block Size vs Replacement Policies (Line Plot)", fontsize=16)
plt.xlabel("Cache Block Size", fontsize=14)
plt.ylabel("No. of misses", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# 2. Bar Plot for All Policies
x = np.arange(len(cache_block_sizes))  # Bar positions
bar_width = 0.2  # Width of each bar
plt.figure(figsize=(12, 6))
for i, (model, values) in enumerate(models.items()):
    plt.bar(x + i * bar_width, values, bar_width, label=model, color=colors[i])
plt.xticks(x + bar_width * 1.5, cache_block_sizes, fontsize=12)  # Adjust x-tick positions
plt.title("Cache Block Size vs Replacement Policies (Bar Plot)", fontsize=16)
plt.xlabel("Cache Block Size", fontsize=14)
plt.ylabel("No. of misses", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", axis="y", alpha=0.6)
plt.tight_layout()
plt.show()

# 3. Scatter Plot for All Policies
plt.figure(figsize=(12, 6))
for i, (model, values) in enumerate(models.items()):
    plt.scatter(cache_block_sizes, values, label=model, s=100, color=colors[i], alpha=0.8)
plt.title("Cache Block Size vs Replacement Policies (Scatter Plot)", fontsize=16)
plt.xlabel("Cache Block Size", fontsize=14)
plt.ylabel("No. of misses", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 4. Grouped Bar Plot for Better Comparison
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.2  # Width of bars
x = np.arange(len(cache_block_sizes))  # Positions for the bars
for i, (model, values) in enumerate(models.items()):
    ax.bar(x + i * bar_width, values, bar_width, label=model, color=colors[i])

ax.set_title("Cache Block Size vs Replacement Policies (Grouped Bar Plot)", fontsize=16)
ax.set_xlabel("Cache Block Size", fontsize=14)
ax.set_ylabel("No. of misses", fontsize=14)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(cache_block_sizes, fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle="--", axis="y", alpha=0.6)
plt.tight_layout()
plt.show()

# 5. Subplots for Each Policy (For Trend Analysis)
num_models = len(models)
num_cols = 2  # Columns in the subplot grid
num_rows = (num_models + num_cols - 1) // num_cols  # Calculate rows dynamically
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()
fig.suptitle("Cache Block Size vs Replacement Policies (Subplots)", fontsize=18)

for i, (model, values) in enumerate(models.items()):
    axes[i].plot(cache_block_sizes, values, marker="o", linewidth=2, label=model, color=colors[i])
    axes[i].set_title(model, fontsize=14)
    axes[i].grid(True, linestyle="--", alpha=0.6)
    axes[i].set_xlabel("Cache Block Size", fontsize=12)
    axes[i].set_ylabel("No. of misses", fontsize=12)
    axes[i].legend(fontsize=10)

# Remove extra blank subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
