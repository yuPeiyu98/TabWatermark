"""
Common experiment runner for watermarking experiments.
"""

import pickle
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

from watermark import (
    getGreenList, 
    singleColumnWatermark,
    get_optimal_col_and_m,
    get_optimal_col_with_fix_m,
    get_p_value
)


def load_raw_data(base_path, task, num_files):
    """Load all raw synthetic data files for a task."""
    sub_dir_path = base_path / task
    X = []
    for i in range(num_files):
        X.append(np.load(os.path.join(sub_dir_path / str(i), f'X_num_train_raw_syn.npy')))
    return X, sub_dir_path


def build_greenlist_info(X_sample, gamma, m, filter_continuous):
    """Build GreenListsInfo dictionary from sample data."""
    GreenLists = []
    GreenListsInfo = {
        'GreenLists': [], 
        'col_mean': [], 
        'col_std': [], 
        'filter_continuous': [], 
        'm': [], 
        'gamma': []
    }
    
    for col_id in range(X_sample.shape[1]):
        col_GreenList = getGreenList(m=m, gamma=gamma)
        GreenLists.append(col_GreenList)
        col_mean, col_std = np.mean(X_sample[:, col_id]), np.std(X_sample[:, col_id])
        GreenListsInfo['col_mean'].append(col_mean)
        GreenListsInfo['col_std'].append(col_std)

    GreenListsInfo['GreenLists'] = GreenLists
    GreenListsInfo['filter_continuous'] = filter_continuous
    GreenListsInfo['m'] = m
    GreenListsInfo['gamma'] = gamma
    
    return GreenListsInfo


def apply_watermark_to_all(X, GreenListsInfo, m):
    """Apply watermark to all data files, return watermarked copies."""
    X_marked = []
    for X_num in X:
        X_num_wm = X_num.copy()
        for col_id in range(X_num_wm.shape[1]):
            col_arr_wm, _, _ = singleColumnWatermark(X_num_wm[:, col_id], GreenListsInfo, col_id, m=m)
            X_num_wm[:, col_id] = col_arr_wm
        X_marked.append(X_num_wm)
    return X_marked


def compute_auc(sub_dir_path, num_files, gamma):
    """Compute AUC scores for detection."""
    p_raw_list = []
    p_wm_list = []
    p_attacked_list = []

    y = np.concatenate([np.ones(num_files), np.zeros(num_files)])
    
    for i in range(num_files):
        p_raw, p_watermarked, p_attacked = get_p_value(
            file_path=sub_dir_path, 
            file_index=i, 
            gamma=gamma
        )
        p_raw_list.append(1 - p_raw)
        p_wm_list.append(1 - p_watermarked)
        p_attacked_list.append(1 - p_attacked)

    auc_original = roc_auc_score(y, np.array(p_wm_list + p_raw_list))
    auc_attacked = roc_auc_score(y, np.array(p_attacked_list + p_raw_list))
    
    return auc_original, auc_attacked


def run_experiment(
    base_path,
    task,
    gamma,
    m,
    attack_fn,
    attack_kwargs=None,
    num_files=50,
    num_sample_data=5,
    delta=0.005,
    use_fixed_m=True,
    verbose=True
):
    """
    Run a single watermarking experiment.
    
    Args:
        base_path: Base path containing task directories
        task: Task name (e.g., 'california', 'gesture')
        gamma: Green list proportion
        m: Number of intervals
        attack_fn: Attack function to apply
        attack_kwargs: Additional kwargs for attack function
        num_files: Number of synthetic data files
        num_sample_data: Number of files to use for statistics estimation
        delta: Tolerance for column filtering
        use_fixed_m: Whether to use fixed m or search for optimal
        verbose: Print progress
    
    Returns:
        Dictionary with experiment results
    """
    if attack_kwargs is None:
        attack_kwargs = {}
    
    base_path = Path(base_path)
    
    if verbose:
        print('-------------------------')
        print(f'Start processing task: {task}')
    
    # Step 1: Load data
    X, sub_dir_path = load_raw_data(base_path, task, num_files)
    
    # Step 2: Build empirical distribution from sample
    X_sample = np.concatenate([X[i] for i in range(num_sample_data)], axis=0)
    
    # Step 3: Filter columns
    if use_fixed_m:
        filter_continuous, _ = get_optimal_col_with_fix_m(X_sample, gamma=gamma, delta=delta)
    else:
        filter_continuous, m = get_optimal_col_and_m(X_sample, gamma=gamma, delta=delta)
    
    if len(filter_continuous) == 0:
        if verbose:
            print(f'No suitable columns found for task: {task}')
        return {'task': task, 'status': 'no_columns'}
    
    if verbose:
        print(f'm= {m}')
        print(f'original_features: {X_sample.shape[1]}')
        print(f'good_features: {len(filter_continuous)}')
    
    # Step 4: Build and save GreenListsInfo
    GreenListsInfo = build_greenlist_info(X_sample, gamma, m, filter_continuous)
    
    with open(os.path.join(sub_dir_path, f'GreenListsInfo.pkl'), 'wb') as f:
        pickle.dump(GreenListsInfo, f)
    
    if verbose:
        print('GreenList is successfully generated and saved.')
    
    # Step 5: Apply watermarks and attacks
    for i in range(num_files):
        if verbose and i % 10 == 0:
            print(i)
        
        X_num = X[i].copy()
        
        # Apply watermark
        for col_id in range(X_num.shape[1]):
            col_arr_wm, _, _ = singleColumnWatermark(X_num[:, col_id], GreenListsInfo, col_id, m=m)
            X_num[:, col_id] = col_arr_wm
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_marked_syn.npy'), X_num)
        
        # Apply attack
        for col_id in range(X_num.shape[1]):
            col_arr = X_num[:, col_id]
            col_arr_attacked = attack_fn(
                arr=col_arr, 
                GreenListsInfo=GreenListsInfo, 
                col_id=col_id, 
                **attack_kwargs
            )
            X_num[:, col_id] = col_arr_attacked
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_attacked_syn.npy'), X_num)
    
    # Step 6: Compute AUC
    auc_original, auc_attacked = compute_auc(sub_dir_path, num_files, gamma)
    
    if verbose:
        print(f'original auc: {auc_original}')
        print(f'attacked auc: {auc_attacked}')
        print(f'Finished processing task: {task}')
        print('-------------------------')
    
    return {
        'task': task,
        'gamma': gamma,
        'm': m,
        'auc_original': auc_original,
        'auc_attacked': auc_attacked,
        'num_features': X_sample.shape[1],
        'good_features': len(filter_continuous)
    }


def run_experiment_simple_attack(
    base_path,
    task,
    gamma,
    m,
    attack_fn,
    attack_kwargs=None,
    num_files=50,
    num_sample_data=5,
    delta=0.005,
    verbose=True
):
    """
    Run experiment with simple attack function (no GreenListsInfo needed).
    
    For attacks like Guesswatermark that only need the array and mtwo.
    """
    if attack_kwargs is None:
        attack_kwargs = {}
    
    base_path = Path(base_path)
    
    if verbose:
        print('-------------------------')
        print(f'Start processing task: {task}')
    
    # Step 1: Load data
    X, sub_dir_path = load_raw_data(base_path, task, num_files)
    
    # Step 2: Build empirical distribution from sample
    X_sample = np.concatenate([X[i] for i in range(num_sample_data)], axis=0)
    
    # Step 3: Filter columns
    filter_continuous, _ = get_optimal_col_with_fix_m(X_sample, gamma=gamma, delta=delta)
    
    if len(filter_continuous) == 0:
        if verbose:
            print(f'No suitable columns found for task: {task}')
        return {'task': task, 'status': 'no_columns'}
    
    if verbose:
        print(f'm= {m}')
        print(f'original_features: {X_sample.shape[1]}')
        print(f'good_features: {len(filter_continuous)}')
    
    # Step 4: Build and save GreenListsInfo
    GreenListsInfo = build_greenlist_info(X_sample, gamma, m, filter_continuous)
    
    with open(os.path.join(sub_dir_path, f'GreenListsInfo.pkl'), 'wb') as f:
        pickle.dump(GreenListsInfo, f)
    
    if verbose:
        print('GreenList is successfully generated and saved.')
    
    # Step 5: Apply watermarks and attacks
    for i in range(num_files):
        if verbose and i % 10 == 0:
            print(i)
        
        X_num = X[i].copy()
        
        # Apply watermark
        for col_id in range(X_num.shape[1]):
            col_arr_wm, _, _ = singleColumnWatermark(X_num[:, col_id], GreenListsInfo, col_id, m=m)
            X_num[:, col_id] = col_arr_wm
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_marked_syn.npy'), X_num)
        
        # Apply simple attack (no GreenListsInfo)
        for col_id in range(X_num.shape[1]):
            col_arr = X_num[:, col_id]
            col_arr_attacked = attack_fn(arr=col_arr, **attack_kwargs)
            X_num[:, col_id] = col_arr_attacked
        
        np.save(os.path.join(sub_dir_path / str(i), f'X_num_train_attacked_syn.npy'), X_num)
    
    # Step 6: Compute AUC
    auc_original, auc_attacked = compute_auc(sub_dir_path, num_files, gamma)
    
    if verbose:
        print(f'original auc: {auc_original}')
        print(f'attacked auc: {auc_attacked}')
        print(f'Finished processing task: {task}')
        print('-------------------------')
    
    return {
        'task': task,
        'gamma': gamma,
        'm': m,
        'auc_original': auc_original,
        'auc_attacked': auc_attacked,
        'num_features': X_sample.shape[1],
        'good_features': len(filter_continuous)
    }
