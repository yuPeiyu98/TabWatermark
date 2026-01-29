"""
Core watermarking functions for tabular data.
"""

import pickle
import numpy as np
import os
from pathlib import Path
from scipy.stats import binom, norm, chi2
from sklearn.metrics import roc_auc_score
import numpy as np
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

# =============================================================================
# Green List Generation
# =============================================================================

def getGreenList(gamma, lo=0, hi=1, m=1000):
    """Generate a green list of intervals for watermarking.
    
    Args:
        gamma: Proportion of green list coverage
        lo: Lower bound of the range
        hi: Upper bound of the range
        m: Number of intervals (must be even)
    
    Returns:
        List of [left, right] interval pairs
    """
    coarse_waymarks = np.linspace(lo, hi, m // 2 + 1)

    green_list = []
    for i in range(0, m // 2):
        if np.random.uniform() > .5:
            l_ed_pt = coarse_waymarks[i]
            r_ed_pt = coarse_waymarks[i] + (coarse_waymarks[i + 1] - coarse_waymarks[i]) * gamma
        else:
            l_ed_pt = coarse_waymarks[i] + (coarse_waymarks[i + 1] - coarse_waymarks[i]) * (1 - gamma)
            r_ed_pt = coarse_waymarks[i + 1]

        green_list.append([l_ed_pt, r_ed_pt])

    return green_list


def findNearestInterval(e, green_list):
    """Return the nearest interval to the given element e (brute force)."""
    min_dist, min_indx = np.inf, -1
    e = e - np.floor(e)

    for i, intv in enumerate(green_list):
        cur_dist = np.abs(e - (intv[0] + intv[1]) / 2)
        if cur_dist < min_dist:
            min_dist = cur_dist
            min_indx = i

    return green_list[min_indx]


def findNearestIntervalHash(e, green_list, m):
    """Return the nearest interval using hash-based lookup (faster)."""
    min_dist, min_indx = np.inf, -1
    e = e - np.floor(e)

    # Hashing for faster lookup
    idx_c = int(e // (2 / m))
    idx_l0, idx_r0 = max(0, idx_c - 1), min(idx_c + 1, len(green_list) - 1)
    idx_l1, idx_r1 = max(0, idx_c - 2), min(idx_c + 2, len(green_list) - 1)

    local_g_list = [green_list[idx_l1], green_list[idx_l0], green_list[idx_c], 
                    green_list[idx_r0], green_list[idx_r1]]
    
    for i, intv in enumerate(local_g_list):
        cur_dist = np.abs(e - (intv[0] + intv[1]) / 2)
        if cur_dist < min_dist:
            min_dist = cur_dist
            min_indx = i

    return local_g_list[min_indx]

def singleColumnWatermark(arr, GreenListsInfo, col_id, m):
    """Add watermark to a single column based on green list.
    
    Args:
        arr: Input array (single column)
        GreenListsInfo: Dictionary containing green lists and statistics
        col_id: Column index
        m: Number of intervals
    
    Returns:
        Tuple of (watermarked array, mean, std)
    """
    arr_wm = arr.copy()
    mean = GreenListsInfo['col_mean'][col_id]
    std = GreenListsInfo['col_std'][col_id]
    green_list = GreenListsInfo['GreenLists'][col_id]
    arr_wm = (arr_wm - mean) / std
    
    for i in range(len(arr_wm)):
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestIntervalHash(e, green_list, m)

        if e > g[1] or e < g[0]:
            arr_wm[i] = np.random.uniform(g[0], g[1]) + e_flr
    
    arr_wm = arr_wm * std + mean
    return arr_wm, mean, std


def soft_singleColumnWatermark(arr, GreenListsInfo, col_id, m, p=1):
    """Add soft watermark with probability p.
    
    Args:
        arr: Input array
        GreenListsInfo: Dictionary containing green lists and statistics
        col_id: Column index
        m: Number of intervals
        p: Probability of applying watermark to each element
    
    Returns:
        Tuple of (watermarked array, mean, std)
    """
    arr_wm = arr.copy()
    mean = GreenListsInfo['col_mean'][col_id]
    std = GreenListsInfo['col_std'][col_id]
    green_list = GreenListsInfo['GreenLists'][col_id]
    arr_wm = (arr_wm - mean) / std
    
    for i in range(len(arr_wm)):
        if np.random.random() > p:
            continue  
        
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestIntervalHash(e, green_list, m)

        if e > g[1] or e < g[0]:
            arr_wm[i] = np.random.uniform(g[0], g[1]) + e_flr

    arr_wm = arr_wm * std + mean
    return arr_wm, mean, std


def countElemInGreenList(arr, green_list, m):
    """Count elements falling within green list intervals."""
    T = 0
    for e in arr:
        g = findNearestIntervalHash(e, green_list, m)
        e_ofst = e - np.floor(e)
        T += (e_ofst <= g[1] and e_ofst >= g[0])
    return T


def binomTesting(arr, green_list, m, gamma):
    """Perform binomial test for watermark detection."""
    T = countElemInGreenList(arr, green_list, m)
    p = binom.cdf(T, len(arr), gamma)
    return 1 - p


def gaussTesting(arr, green_list, m, gamma):
    """Perform Gaussian approximation test for watermark detection."""
    T = countElemInGreenList(arr, green_list, m)
    mv = len(arr) * gamma
    std = np.sqrt(len(arr) * gamma * (1 - gamma))
    p = norm.cdf((T - mv) / std)
    return 1 - p        


def gaussMultTesting(arrs, green_lists, m, gamma):
    """Perform chi-squared test across multiple columns.
    
    This is the proposed method for multi-column watermark detection.
    """
    X = 0
    mv = len(arrs[0]) * gamma
    std = np.sqrt(len(arrs[0]) * gamma * (1 - gamma))

    for arr, green_list in zip(arrs, green_lists):
        T = countElemInGreenList(arr, green_list, m)
        T_c = (T - mv) / std
        X += T_c ** 2

    p = chi2.cdf(X, len(arrs))
    return 1 - p


def exactMaxGaussCDF(x, n):
    """CDF for maximum of n standard Gaussian variables."""
    return norm.cdf(x)**n


def maxGaussTesting(arrs, green_lists, m):
    """Alternative test using extreme statistics."""
    max_Tc = float('-inf')
    mv = len(arrs[0]) / 2
    std = np.sqrt(len(arrs[0])) / 2

    for arr, green_list in zip(arrs, green_lists):
        T = countElemInGreenList(arr, green_list, m)
        T_c = (T - mv) / std
        if T_c > max_Tc:
            max_Tc = T_c

    p = exactMaxGaussCDF(max_Tc, len(arrs))
    return 1 - p


def singleColumnNoise(arr, noise_prop, std_prop):
    """Add Gaussian noise attack to a column."""
    arr_noise = arr.copy()
    num_rows_to_modify = int(len(arr) * noise_prop)
    noise_std = np.sqrt(np.var(arr)) * std_prop
    noise = np.random.normal(0, noise_std, num_rows_to_modify)
    arr_noise[:num_rows_to_modify] += noise
    return arr_noise


def Guesswatermark(arr, mtwo):
    """Guess attack with unknown mean/std (uses empirical statistics)."""
    arr_rwm = arr.copy()
    mean, std = np.mean(arr), np.std(arr)
    arr_rwm = (arr_rwm - mean) / std
    
    m = mtwo // 2

    for i in range(len(arr_rwm)):
        e_flr = np.floor(arr_rwm[i])
        e = arr_rwm[i] - e_flr
        idx_c = int(e // (1 / m))
        if np.random.uniform() > .5:
            if e <= (idx_c + 0.5) / m and e >= (idx_c) / m:
                continue
            arr_rwm[i] = np.random.uniform((idx_c) / m, (idx_c + 0.5) / m) + e_flr
        else:
            if e <= (idx_c + 1) / m and e >= (idx_c + 0.5) / m:
                continue
            arr_rwm[i] = np.random.uniform((idx_c + 0.5) / m, (idx_c + 1) / m) + e_flr
    
    arr_rwm = arr_rwm * std + mean
    return arr_rwm


def KnowGuesswatermark(arr, GreenListsInfo, col_id, mtwo):
    """Guess attack with known mean/std from GreenListsInfo."""
    arr_rwm = arr.copy()
    mean = GreenListsInfo['col_mean'][col_id]
    std = GreenListsInfo['col_std'][col_id]
    arr_rwm = (arr_rwm - mean) / std
    
    m = mtwo // 2

    for i in range(len(arr_rwm)):
        e_flr = np.floor(arr_rwm[i])
        e = arr_rwm[i] - e_flr
        idx_c = int(e // (1 / m))
        if np.random.uniform() > .5:
            if e <= (idx_c + 0.5) / m and e >= (idx_c) / m:
                continue
            arr_rwm[i] = np.random.uniform((idx_c) / m, (idx_c + 0.5) / m) + e_flr
        else:
            if e <= (idx_c + 1) / m and e >= (idx_c + 0.5) / m:
                continue
            arr_rwm[i] = np.random.uniform((idx_c + 0.5) / m, (idx_c + 1) / m) + e_flr
    
    arr_rwm = arr_rwm * std + mean
    return arr_rwm


def Know_rewm_singleColumnWatermark(arr, GreenListsInfo, col_id, green_list, rem):
    """Re-watermarking attack with known statistics."""
    arr_wm = arr.copy()
    mean = GreenListsInfo['col_mean'][col_id]
    std = GreenListsInfo['col_std'][col_id]
    arr_wm = (arr_wm - mean) / std
    
    m = rem
    for i in range(len(arr_wm)):
        e_flr = np.floor(arr_wm[i])
        e = arr_wm[i] - e_flr

        g = findNearestIntervalHash(e, green_list, m)

        if e > g[1] or e < g[0]:
            arr_wm[i] = np.random.uniform(g[0], g[1]) + e_flr

    arr_wm = arr_wm * std + mean
    return arr_wm


def get_optimal_col_and_m(X_num, gamma, delta=0.1):
    """Filter poorly distributed columns and find optimal m.
    
    Args:
        X_num: Input data matrix
        gamma: Target green list proportion
        delta: Tolerance for filtering
    
    Returns:
        Tuple of (filtered column indices, optimal m)
    """
    dataset = X_num.copy()
    
    for col_id in range(dataset.shape[1]):
        dataset[:, col_id] = (dataset[:, col_id] - np.mean(dataset[:, col_id])) / np.std(dataset[:, col_id])
    
    prop_min = gamma - delta
    prop_max = gamma + delta
    
    m_list = list(range(1000, 5001, 500))
    filter_arr = np.zeros(dataset.shape[1])
    n = dataset.shape[0]
    
    test_times = 5
    
    for time in range(test_times):
        print('test:', time)
        for m in m_list:
            print('m:', m)
            unit_filter_arr = np.zeros(dataset.shape[1])
            
            for col_id in range(dataset.shape[1]):
                arr = dataset[:, col_id]
                col_GreenList = getGreenList(m=m, gamma=gamma)
                prop_col = countElemInGreenList(arr, col_GreenList, m=m) / n
                
                if prop_col < prop_min or prop_col > prop_max:
                    unit_filter_arr[col_id] += 1
            
            filter_arr = np.vstack((filter_arr, unit_filter_arr))
    
    filter_arr = filter_arr[1:, :]
    filter_continuous = np.where(np.sum(filter_arr, axis=0)/(test_times*len(m_list)) < 0.1)[0]
        
    if not len(filter_continuous):
        return np.array([]), None
    
    else:
        filter_arr = filter_arr[:, filter_continuous]
        print('filter_continuous:', filter_continuous, len(filter_continuous))
        
        agg_by_m = np.array([])

        if len(filter_continuous):
            for step in range(len(m_list)):
                agg_by_m = np.append(agg_by_m, filter_arr[step::len(m_list)].sum())
            
            print('pass_times', agg_by_m)
            optimal_m = m_list[np.argmin(agg_by_m)]
        else:
            return filter_continuous, None
        
        print('optimal_m:', optimal_m)
        
    return filter_continuous, optimal_m


def get_optimal_col_with_fix_m(X_num, gamma, delta=0.1, m_list=[1000]):
    """Filter columns with fixed m value(s) for faster computation."""
    dataset = X_num.copy()
    
    for col_id in range(dataset.shape[1]):
        dataset[:, col_id] = (dataset[:, col_id] - np.mean(dataset[:, col_id])) / np.std(dataset[:, col_id])
    
    prop_min = gamma - delta
    prop_max = gamma + delta
    
    filter_arr = np.zeros(dataset.shape[1])
    n = dataset.shape[0]
    
    test_times = 5
    
    for time in range(test_times):
        print('test:', time)
        for m in m_list:
            unit_filter_arr = np.zeros(dataset.shape[1])
            
            for col_id in range(dataset.shape[1]):
                arr = dataset[:, col_id]
                col_GreenList = getGreenList(m=m, gamma=gamma)
                prop_col = countElemInGreenList(arr, col_GreenList, m=m) / n
                
                if prop_col < prop_min or prop_col > prop_max:
                    unit_filter_arr[col_id] += 1
            
            filter_arr = np.vstack((filter_arr, unit_filter_arr))
    
    filter_arr = filter_arr[1:, :]
    filter_continuous = np.where(np.sum(filter_arr, axis=0)/(test_times*len(m_list)) < 0.1)[0]
        
    if not len(filter_continuous):
        return np.array([]), None
    
    else:
        filter_arr = filter_arr[:, filter_continuous]
        print('filter_continuous:', filter_continuous, len(filter_continuous))
        
        agg_by_m = np.array([])

        if len(filter_continuous):
            for step in range(len(m_list)):
                agg_by_m = np.append(agg_by_m, filter_arr[step::len(m_list)].sum())
            
            print('pass_times', agg_by_m)
            optimal_m = m_list[np.argmin(agg_by_m)]
        else:
            return filter_continuous, None
        
    return filter_continuous, optimal_m


def get_p_value(file_path, file_index, gamma):
    """Get p-values for raw, watermarked, and attacked data.
    
    Args:
        file_path: Path to experiment directory
        file_index: Index of the data file
        gamma: Green list proportion
    
    Returns:
        List of [p_raw, p_watermarked, p_attacked]
    """
    sub_dir_path = Path(file_path)
    with open(os.path.join(sub_dir_path, f'GreenListsInfo.pkl'), 'rb') as f:
        GreenListsInfo = pickle.load(f)

    continuous_indexes = GreenListsInfo['filter_continuous']
    GreenLists = [GreenListsInfo['GreenLists'][i] for i in continuous_indexes]
    col_mean = np.array(GreenListsInfo['col_mean'])
    col_std = np.array(GreenListsInfo['col_std'])
    
    # Raw data
    raw_data = np.load(sub_dir_path / str(file_index) / f'X_num_train_raw_syn.npy')
    raw_data = (raw_data - col_mean) / col_std
    raw_data = raw_data[:, continuous_indexes]
    p_raw = gaussMultTesting(raw_data.T, GreenLists, m=len(GreenLists[0])*2, gamma=gamma)

    # Watermarked data
    watermarked_data = np.load(sub_dir_path / str(file_index) / f'X_num_train_marked_syn.npy')
    watermarked_data = (watermarked_data - col_mean) / col_std
    watermarked_data = watermarked_data[:, continuous_indexes]
    p_watermarked = gaussMultTesting(watermarked_data.T, GreenLists, m=len(GreenLists[0])*2, gamma=gamma)

    # Attacked data
    attacked_data = np.load(sub_dir_path / str(file_index) / f'X_num_train_attacked_syn.npy')
    attacked_data = (attacked_data - col_mean) / col_std
    attacked_data = attacked_data[:, continuous_indexes]
    p_attacked = gaussMultTesting(attacked_data.T, GreenLists, m=len(GreenLists[0])*2, gamma=gamma)
    
    return [p_raw, p_watermarked, p_attacked]
