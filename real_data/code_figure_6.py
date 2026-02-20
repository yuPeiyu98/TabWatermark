import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# ========== Configuration ==========
DATA_PATH = Path(r'path/to/synthetic/data')  # for the first sub-plot in figure 6, we visualize california dataset's column 1, so the input file should be california/0/X_num_train_raw_syn.npy
DATASET = 'California'  # for printing
COLUMN_INDEX = 1  # Column index to visualize; 1 for california, 2 for house
BINS = 100        # Number of histogram bins
# ====================================

# Load data
data = np.load(DATA_PATH)
print(f"Data shape: {data.shape}")
print(f"Number of columns: {data.shape[1]}")
print(f"Selected column: {COLUMN_INDEX}")

# Extract the specified column
col_data = data[:, COLUMN_INDEX]
print(f"Column stats: min={col_data.min():.4f}, max={col_data.max():.4f}, mean={col_data.mean():.4f}, std={col_data.std():.4f}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(col_data, bins=BINS, color='steelblue', edgecolor='black', alpha=0.85)
plt.title(f"Histogram of the Spike Column in {DATASET} Dataset", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()

# Save as EPS
save_path = f"histogram_{DATASET.lower()}_column{COLUMN_INDEX}.eps"
plt.savefig(save_path, format='eps')
plt.show()
print(f"Figure saved to {save_path}")