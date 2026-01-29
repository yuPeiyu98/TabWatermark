"""
Experiment: Re-watermarking Attack

The attacker attempts to overwrite the watermark with their own.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

from watermark import (
    getGreenList,
    set_seed,
    singleColumnWatermark,
    get_optimal_col_with_fix_m,
    get_p_value,
    Know_rewm_singleColumnWatermark
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
M = 2000      # watermarker's m
REM = [100, 1000, 10000]    # attacker's m for re-watermarking


# =============================================================================
# Run Experiments
# =============================================================================

def run_rewm_experiment(base_path, task, gamma, m, rem, num_files, num_sample_data, delta):
    """Run re-watermarking attack experiment."""
    
    print('-------------------------')
    print(f'Start processing task: {task}')
    
    base_path = Path(base_path)
    
    # Load data
    X, sub_dir_path = load_raw_data(base_path, task, num_files)
    
    # Build empirical distribution
    X_sample = np.concatenate([X[i] for i in range(num_sample_data)], axis=0)
    
    # Filter columns
    filter_continuous, _ = get_optimal_col_with_fix_m(X_sample, gamma=gamma, delta=delta)
    
    if len(filter_continuous) == 0:
        print(f'No suitable columns found for task: {task}')
        return {'task': task, 'status': 'no_columns'}
    
    print(f'original_features: {X_sample.shape[1]}')
    print(f'good_features: {len(filter_continuous)}')
    
    # Build GreenListsInfo
    GreenListsInfo = build_greenlist_info(X_sample, gamma, m, filter_continuous)
    
    with open(os.path.join(sub_dir_path, f'GreenListsInfo.pkl'), 'wb') as f:
        pickle.dump(GreenListsInfo, f)
    print('GreenList is successfully generated and saved.')
    
    # Apply watermarks and re-watermarking attack
    for i in range(num_files):
        if i % 10 == 0:
            print(i)
        
        X_num = X[i].copy()
        
        # Apply original watermark
        for col_id in range(X_num.shape[1]):
            col_arr_wm, _, _ = singleColumnWatermark(X_num[:, col_id], GreenListsInfo, col_id, m=m)
            X_num[:, col_id] = col_arr_wm
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_marked_syn.npy'), X_num)
        
        # Apply re-watermarking attack
        for col_id in range(X_num.shape[1]):
            col_arr = X_num[:, col_id]
            # Attacker generates their own green list
            re_col_GreenList = getGreenList(m=rem, gamma=gamma)
            col_arr_attacked = Know_rewm_singleColumnWatermark(
                arr=col_arr, 
                GreenListsInfo=GreenListsInfo, 
                col_id=col_id,
                green_list=re_col_GreenList, 
                rem=rem
            )
            X_num[:, col_id] = col_arr_attacked
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_attacked_syn.npy'), X_num)
    
    # Compute AUC
    auc_original, auc_attacked = compute_auc(sub_dir_path, num_files, gamma)
    
    print(f'original auc: {auc_original}')
    print(f'attacked auc: {auc_attacked}')
    print(f'Finished processing task: {task}')
    print('-------------------------')
    
    return {
        'task': task,
        'gamma': gamma,
        'm': m,
        'rem': rem,
        'auc_original': auc_original,
        'auc_attacked': auc_attacked
    }

def main():
    set_seed(42) 
    results = []
    
    for rem in REM:
        print(f'\n{"="*50}')
        print(f'Re-watermarking Attack: gamma={GAMMA}, m={M}, rem={rem}')
        print(f'{"="*50}\n')
        
        for task in TASKS:
            if task in ['california', 'gesture', 'house', 'wilt', 'higgs-small']:
                delta = 0.005
            else:
                delta = 0.0025
            
            result = run_rewm_experiment(
                base_path=BASE_PATH,
                task=task,
                gamma=GAMMA,
                m=M,
                rem=rem,
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
            print(f"{r['task']}: No suitable columns")
        else:
            print(f"{r['task']} (rem={r['rem']}): AUC_orig={r['auc_original']:.4f}, AUC_atk={r['auc_attacked']:.4f}")


if __name__ == '__main__':
    main()
