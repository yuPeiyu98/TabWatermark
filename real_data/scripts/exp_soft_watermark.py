"""
Experiment: Soft Watermarking

Apply watermark with probability p < 1 to each element.
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
    soft_singleColumnWatermark,
    singleColumnNoise,  # 加这个
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
P_LIST = [0.8]  # Watermarking probability
NOISE_PROP = 0.95
STD_PROP = 0.01

# =============================================================================
# Run Experiments
# =============================================================================

def run_soft_watermark_experiment(base_path, task, gamma, p, noise_prop, std_prop, num_files, num_sample_data, delta):
    """Run soft watermarking experiment with noise attack."""
    
    print('-------------------------')
    print(f'Start processing task: {task}, p={p}')
    
    base_path = Path(base_path)
    
    # Load data
    X, sub_dir_path = load_raw_data(base_path, task, num_files)
    
    # Build empirical distribution
    X_sample = np.concatenate([X[i] for i in range(num_sample_data)], axis=0)
    
    # Filter columns and find optimal m
    filter_continuous, m = get_optimal_col_and_m(X_sample, gamma=gamma, delta=delta)
    
    if len(filter_continuous) == 0 or m is None:
        print(f'No suitable columns found for task: {task}')
        return {'task': task, 'p': p, 'status': 'no_columns'}
    
    print(f'optimal_m: {m}')
    print(f'original_features: {X_sample.shape[1]}')
    print(f'good_features: {len(filter_continuous)}')
    
    # Build GreenListsInfo
    GreenListsInfo = build_greenlist_info(X_sample, gamma, m, filter_continuous)
    
    with open(os.path.join(sub_dir_path, f'GreenListsInfo.pkl'), 'wb') as f:
        pickle.dump(GreenListsInfo, f)
    print('GreenList is successfully generated and saved.')
    
    # Apply soft watermarks and noise attack
    for i in range(num_files):
        if i % 10 == 0:
            print(i)
        
        X_num = X[i].copy()
        
        # Apply soft watermark
        for col_id in range(X_num.shape[1]):
            col_arr_wm, _, _ = soft_singleColumnWatermark(
                X_num[:, col_id], GreenListsInfo, col_id, m=m, p=p
            )
            X_num[:, col_id] = col_arr_wm
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_marked_syn.npy'), X_num)
        
        # Apply noise attack
        for col_id in range(X_num.shape[1]):
            col_arr = X_num[:, col_id]
            col_arr_noise = singleColumnNoise(col_arr, noise_prop, std_prop)
            X_num[:, col_id] = col_arr_noise
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_attacked_syn.npy'), X_num)
    
    # Compute AUC
    auc_original, auc_noised = compute_auc(sub_dir_path, num_files, gamma)
    
    print(f'original auc: {auc_original}')
    print(f'noised auc: {auc_noised}')
    print(f'Finished processing task: {task}')
    print('-------------------------')
    
    return {
        'task': task,
        'gamma': gamma,
        'p': p,
        'm': m,
        'auc_original': auc_original,
        'auc_noised': auc_noised
    }

def main():
    set_seed(42) 
    results = []
    
    for p in P_LIST:
        print(f'\n{"="*50}')
        print(f'Soft Watermarking: gamma={GAMMA}, p={p}')
        print(f'{"="*50}\n')
        
        for task in TASKS:
            if task in ['california', 'gesture', 'house', 'wilt', 'higgs-small']:
                delta = 0.01
            else:
                delta = 0.005
            
            result = run_soft_watermark_experiment(
                base_path=BASE_PATH,
                task=task,
                gamma=GAMMA,
                p=p,
                noise_prop=NOISE_PROP,
                std_prop=STD_PROP,
                num_files=NUM_FILES,
                num_sample_data=NUM_SAMPLE_DATA,
                delta=delta
            )
            results.append(result)
    
    # Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    for r in results:
        if r.get('status') == 'no_columns':
            print(f"{r['task']} (p={r['p']}): No suitable columns")
        else:
            print(f"{r['task']} (p={r['p']}): AUC_orig={r['auc_original']:.4f}, AUC_noised={r['auc_noised']:.4f}")


if __name__ == '__main__':
    main()
