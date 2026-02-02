"""
Compute Wasserstein distance:
1. Original data vs Synthetic data
2. Synthetic data vs Watermarked data
"""

import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================
# DATA_PATH: Original data from TabDDPM (https://github.com/rotot0/tab-ddpm)
# BASE_PATH: Directory containing synthetic data files
DATA_PATH = Path(r'path/to/tabddpm/data')      
BASE_PATH = Path(r'path/to/synthetic/data')   
TASKS = ['california', 'gesture', 'house', 'wilt', 'higgs-small', 'miniboone']
NUM_FILES = 50


def normalize_array(x):
    """In-place normalization."""
    for col_id in range(x.shape[1]):
        x[:, col_id] = (x[:, col_id] - np.mean(x[:, col_id])) / np.std(x[:, col_id])


def main():
    for task in TASKS:
        print('-----------------------')
        print('Task:', task)
        
        original = np.load(DATA_PATH / task / 'X_num_train.npy')
        distance_raw_ori = []
        distance_marked_raw = []

        for i in tqdm(range(NUM_FILES)):
            raw_syn = np.load(BASE_PATH / task / str(i) / 'X_num_train_raw_syn.npy')
            marked_syn = np.load(BASE_PATH / task / str(i) / 'X_num_train_marked_syn.npy')

            normalize_array(original)
            normalize_array(raw_syn)
            normalize_array(marked_syn)

            unit_distance_raw_ori = []
            unit_distance_marked_raw = []

            for col_id in range(original.shape[1]):
                unit_distance_raw_ori.append(wasserstein_distance(original[:, col_id], raw_syn[:, col_id]))
                unit_distance_marked_raw.append(wasserstein_distance(marked_syn[:, col_id], raw_syn[:, col_id]))

            distance_raw_ori.append(sum(unit_distance_raw_ori) / len(unit_distance_raw_ori))
            distance_marked_raw.append(sum(unit_distance_marked_raw) / len(unit_distance_marked_raw))

        print('Wasserstein distance (original vs synthetic):', 
              sum(distance_raw_ori) / len(distance_raw_ori))
        print('Wasserstein distance (synthetic vs watermarked):', 
              sum(distance_marked_raw) / len(distance_marked_raw))


if __name__ == '__main__':
    main()