"""
Experiment: Utility Evaluation Before and After Watermarking (used in Table 4)
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    classification_report,
    roc_auc_score,
)
from catboost import CatBoostRegressor, CatBoostClassifier

# =============================================================================
# Configuration
# =============================================================================
# DATA_PATH: Original data from TabDDPM;
#            each DATA_PATH/<task>/ provides X_num_val/test.npy, y_val/test.npy, info.json
# BASE_PATH: Directory containing the synthetic data files
DATA_PATH = Path(r'path/to/tabddpm/data')
BASE_PATH = Path(r'path/to/synthetic/data')
TASKS = ['california', 'gesture', 'house', 'wilt', 'higgs-small', 'miniboone']
NUM_FILES = 50           # number of synthetic samples to average over

# Synthetic training files compared in Table 4: before vs after watermarking.
# (Add 'noised': 'X_num_train_noised_syn.npy' here to also score the noise-attacked data.)
VARIANTS = {
    'raw':    'X_num_train_raw_syn.npy',     # before watermarking
    'marked': 'X_num_train_marked_syn.npy',  # after watermarking
}

# Tuned CatBoost hyper-parameters, copied verbatim from the TabDDPM repository
# (tuned_models/catboost/<task>_cv.json). Kept here so the evaluation is fully
# reproducible without cloning the external repo.
CATBOOST_CONFIGS = {
    'california':  {'learning_rate': 0.04790555955853721, 'depth': 9, 'l2_leaf_reg': 3.5239162123685057, 'bagging_temperature': 0.25057429631750866, 'leaf_estimation_iterations': 5, 'iterations': 2000, 'early_stopping_rounds': 50, 'od_pval': 0.001, 'task_type': 'CPU', 'thread_count': 4, 'cat_features': []},
    'house':       {'learning_rate': 0.04196246501291052, 'depth': 8, 'l2_leaf_reg': 7.667680541186549, 'bagging_temperature': 0.7200947860987692, 'leaf_estimation_iterations': 4, 'iterations': 2000, 'early_stopping_rounds': 50, 'od_pval': 0.001, 'task_type': 'CPU', 'thread_count': 4, 'cat_features': []},
    'gesture':     {'learning_rate': 0.06605210083761866, 'depth': 7, 'l2_leaf_reg': 1.014366822804501, 'bagging_temperature': 0.3224895199122277, 'leaf_estimation_iterations': 9, 'iterations': 2000, 'early_stopping_rounds': 50, 'od_pval': 0.001, 'task_type': 'CPU', 'thread_count': 4, 'cat_features': []},
    'higgs-small': {'learning_rate': 0.059421510534574236, 'depth': 8, 'l2_leaf_reg': 4.164640566364768, 'bagging_temperature': 0.7435082899382577, 'leaf_estimation_iterations': 4, 'iterations': 2000, 'early_stopping_rounds': 50, 'od_pval': 0.001, 'task_type': 'CPU', 'thread_count': 4, 'cat_features': []},
    'miniboone':   {'learning_rate': 0.09465329870285419, 'depth': 9, 'l2_leaf_reg': 3.987514085877886, 'bagging_temperature': 0.45834726300696005, 'leaf_estimation_iterations': 5, 'iterations': 2000, 'early_stopping_rounds': 50, 'od_pval': 0.001, 'task_type': 'CPU', 'thread_count': 4, 'cat_features': []},
    'wilt':        {'learning_rate': 0.13877980376409904, 'depth': 6, 'l2_leaf_reg': 1.1040918394803323, 'bagging_temperature': 0.9966446926502672, 'leaf_estimation_iterations': 4, 'iterations': 2000, 'early_stopping_rounds': 50, 'od_pval': 0.001, 'task_type': 'CPU', 'thread_count': 4, 'cat_features': []},
}

PRIMARY_METRIC = {'regression': 'r2', 'binclass': 'f1', 'multiclass': 'f1'}


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_single(real_dir, syn_file, config, seed=EVAL_SEED):
    """Train CatBoost on one synthetic file, score on the real test set.

    Mirrors TabDDPM scripts/eval_catboost.py for eval_type='synthetic':
      - X_num is fed at its original scale (eval normalization is disabled).
      - For regression, y is standardized by the train mean/std and the RMSE is
        rescaled back; for classification y is left untouched.
    Returns (task_type, {split: {metric: value}}).
    """
    random.seed(seed)
    np.random.seed(seed)

    info = json.load(open(os.path.join(real_dir, 'info.json')))
    task_type = info['task_type']
    n_classes = info.get('n_classes', 2)

    X = {
        'train': np.load(syn_file, allow_pickle=True).astype(float),
        'val':   np.load(os.path.join(real_dir, 'X_num_val.npy'),  allow_pickle=True),
        'test':  np.load(os.path.join(real_dir, 'X_num_test.npy'), allow_pickle=True),
    }
    y = {
        'train': np.load(os.path.join(os.path.dirname(syn_file), 'y_train.npy'), allow_pickle=True).ravel(),
        'val':   np.load(os.path.join(real_dir, 'y_val.npy'),  allow_pickle=True).ravel(),
        'test':  np.load(os.path.join(real_dir, 'y_test.npy'), allow_pickle=True).ravel(),
    }

    n_features = X['train'].shape[1]
    X = {k: pd.DataFrame(v, columns=range(n_features)).astype(float) for k, v in X.items()}

    out = {}
    if task_type == 'regression':
        mean, std = float(y['train'].mean()), float(y['train'].std())
        y = {k: (v - mean) / std for k, v in y.items()}
        model = CatBoostRegressor(**config, eval_metric='RMSE', random_seed=seed)
        model.fit(X['train'], y['train'], eval_set=(X['val'], y['val']), verbose=False)
        for split in ['val', 'test']:
            pred = model.predict(X[split])
            out[split] = {
                'r2': r2_score(y[split], pred),
                'rmse': mean_squared_error(y[split], pred) ** 0.5 * std,
            }
    else:
        y = {k: v.astype(int) for k, v in y.items()}
        is_multiclass = task_type == 'multiclass'
        model = CatBoostClassifier(
            loss_function='MultiClass' if is_multiclass else 'Logloss',
            **config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(n_classes)] if is_multiclass else ['0', '1'],
        )
        model.fit(X['train'], y['train'], eval_set=(X['val'], y['val']), verbose=False)
        for split in ['val', 'test']:
            if is_multiclass:
                probs = model.predict_proba(X[split])
                labels = probs.argmax(axis=1).astype('int64')
            else:
                probs = model.predict_proba(X[split])[:, 1]
                labels = np.round(probs).astype('int64')
            report = classification_report(y[split], labels, output_dict=True)
            out[split] = {'acc': report['accuracy'], 'f1': report['macro avg']['f1-score']}
            if not is_multiclass:
                out[split]['roc_auc'] = roc_auc_score(y[split], probs)
    return task_type, out


def run_utility_evaluation(data_path, base_path, task, num_files):
    """Average the test-set utility over all synthetic samples for one task."""
    print('-------------------------')
    print(f'Start processing task: {task}')

    real_dir = data_path / task
    task_type = json.load(open(real_dir / 'info.json'))['task_type']
    key = PRIMARY_METRIC[task_type]

    scores = {v: [] for v in VARIANTS}
    for i in range(num_files):
        for variant, fname in VARIANTS.items():
            syn_file = base_path / task / str(i) / fname
            if not syn_file.exists():
                continue
            _, out = evaluate_single(real_dir, str(syn_file), CATBOOST_CONFIGS[task])
            scores[variant].append(out['test'][key])

    result = {'task': task, 'task_type': task_type, 'metric': key}
    for variant in VARIANTS:
        vals = scores[variant]
        if vals:
            result[variant] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
            print(f'  {variant:7s} {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  (n={len(vals)})')
        else:
            result[variant] = None
            print(f'  {variant:7s} {key}: no files found')

    print(f'Finished processing task: {task}')
    print('-------------------------')
    return result


def main():
    results = []

    print(f'\n{"="*50}')
    print(f'Utility Evaluation (Table 4): CatBoost on synthetic, scored on real')
    print(f'{"="*50}\n')

    for task in TASKS:
        results.append(run_utility_evaluation(DATA_PATH, BASE_PATH, task, NUM_FILES))

    # Print summary
    print('\n' + '=' * 70)
    print('SUMMARY (test set, mean +/- std)')
    print('=' * 70)
    header = f'{"task":12s} {"metric":7s} ' + ' '.join(f'{v:>18s}' for v in VARIANTS)
    print(header)
    for r in results:
        cells = []
        for v in VARIANTS:
            cells.append('--' if r[v] is None else f'{r[v]["mean"]:.4f}+/-{r[v]["std"]:.4f}')
        print(f'{r["task"]:12s} {r["metric"]:7s} ' + ' '.join(f'{c:>18s}' for c in cells))


if __name__ == '__main__':
    main()