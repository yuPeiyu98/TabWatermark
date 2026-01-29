"""
Experiment: Guess Attack with Unknown Mean/Std

This attack uses empirical mean/std from the data itself.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from run_experiment import run_experiment_simple_attack
from watermark import Guesswatermark, set_seed

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path(r'path/to/synthetic/data')  
TASKS = ['california', 'gesture', 'house', 'wilt', 'higgs-small', 'miniboone']
NUM_FILES = 50
NUM_SAMPLE_DATA = 5

# Experiment parameters
GAMMA_LIST = [0.5]
M = 1000
REM = 1000  # attacker's guess for m


# =============================================================================
# Run Experiments
# =============================================================================

def main():
    set_seed(42) 
    results = []
    
    for gamma in GAMMA_LIST:
        print(f'\n{"="*50}')
        print(f'Running experiments with gamma={gamma}')
        print(f'{"="*50}\n')
        
        for task in TASKS:
            # Set delta based on task
            if task in ['california', 'gesture', 'house', 'wilt', 'higgs-small']:
                delta = 0.005
            else:
                delta = 0.0025
            
            result = run_experiment_simple_attack(
                base_path=BASE_PATH,
                task=task,
                gamma=gamma,
                m=M,
                attack_fn=Guesswatermark,
                attack_kwargs={'mtwo': REM},
                num_files=NUM_FILES,
                num_sample_data=NUM_SAMPLE_DATA,
                delta=delta,
                verbose=True
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
            print(f"{r['task']} (gamma={r['gamma']}): "
                  f"AUC_orig={r['auc_original']:.4f}, "
                  f"AUC_atk={r['auc_attacked']:.4f}")


if __name__ == '__main__':
    main()
