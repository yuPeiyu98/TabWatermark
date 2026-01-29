"""
Experiment: Additive Noise Attack

The attacker adds Gaussian noise to the watermarked data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import os
from pathlib import Path

from watermark import (
    set_seed,
    singleColumnWatermark,
    singleColumnNoise,
    get_optimal_col_and_m,
    get_p_value
)
from run_experiment import load_raw_data, build_greenlist_info, compute_auc

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path(r'path/to/synthetic/data')  
TASKS = ['california', 'gesture', 'house', 'wilt', 'higgs-small', 'miniboone']
NUM_FILES = 50
NUM_SAMPLE_DATA = 5

# Experiment parameters
GAMMA = 0.5
NOISE_PROP = 0.95   # Proportion of rows to add noise
STD_PROP = 0.01     # Noise std as proportion of column std


# =============================================================================
# Run Experiments
# =============================================================================

def run_noise_experiment(base_path, task, gamma, noise_prop, std_prop, num_files, num_sample_data):
    """Run additive noise attack experiment."""
    
    print('-------------------------')
    print(f'Start processing task: {task}')
    
    base_path = Path(base_path)
    
    # Load data
    X, sub_dir_path = load_raw_data(base_path, task, num_files)
    
    # Build empirical distribution
    X_sample = np.concatenate([X[i] for i in range(num_sample_data)], axis=0)
    
    # Filter columns and find optimal m
    filter_continuous, m = get_optimal_col_and_m(X_sample, gamma=gamma)
    
    if len(filter_continuous) == 0 or m is None:
        print(f'No suitable columns found for task: {task}')
        return {'task': task, 'status': 'no_columns'}
    
    print(f'optimal_m: {m}')
    print(f'original_features: {X_sample.shape[1]}')
    print(f'good_features: {len(filter_continuous)}')
    
    # Build GreenListsInfo
    GreenListsInfo = build_greenlist_info(X_sample, gamma, m, filter_continuous)
    
    with open(os.path.join(sub_dir_path, f'GreenListsInfo.pkl'), 'wb') as f:
        pickle.dump(GreenListsInfo, f)
    print('GreenList is successfully generated and saved.')
    
    # Apply watermarks and noise attack
    for i in range(num_files):
        if i % 10 == 0:
            print(i)
        
        X_num = X[i].copy()
        
        # Apply watermark
        for col_id in range(X_num.shape[1]):
            col_arr_wm, _, _ = singleColumnWatermark(X_num[:, col_id], GreenListsInfo, col_id, m=m)
            X_num[:, col_id] = col_arr_wm
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_marked_syn.npy'), X_num)
        
        # Apply noise attack
        for col_id in range(X_num.shape[1]):
            col_arr = X_num[:, col_id]
            col_arr_noise = singleColumnNoise(col_arr, noise_prop, std_prop)
            X_num[:, col_id] = col_arr_noise
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_attacked_syn.npy'), X_num)
    
    # Compute AUC
    auc_original, auc_attacked = compute_auc(sub_dir_path, num_files, gamma)
    
    print(f'original auc: {auc_original}')
    print(f'noised auc: {auc_attacked}')
    print(f'Finished processing task: {task}')
    print('-------------------------')
    
    return {
        'task': task,
        'gamma': gamma,
        'm': m,
        'noise_prop': noise_prop,
        'std_prop': std_prop,
        'auc_original': auc_original,
        'auc_attacked': auc_attacked
    }


def main():
    set_seed(42) 
    results = []
    
    print(f'\n{"="*50}')
    print(f'Noise Attack: gamma={GAMMA}, noise_prop={NOISE_PROP}, std_prop={STD_PROP}')
    print(f'{"="*50}\n')
    
    for task in TASKS:
        result = run_noise_experiment(
            base_path=BASE_PATH,
            task=task,
            gamma=GAMMA,
            noise_prop=NOISE_PROP,
            std_prop=STD_PROP,
            num_files=NUM_FILES,
            num_sample_data=NUM_SAMPLE_DATA
        )
        results.append(result)
    
    # Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    for r in results:
        if r.get('status') == 'no_columns':
            print(f"{r['task']}: No suitable columns")
        else:
            print(f"{r['task']}: AUC_orig={r['auc_original']:.4f}, AUC_noised={r['auc_attacked']:.4f}")


if __name__ == '__main__':
    main()
