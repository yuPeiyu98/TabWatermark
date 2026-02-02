"""
Experiment: Guess Attack with Known Mean/Std

This attack assumes the attacker knows the mean and std used for normalization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from run_experiment import run_experiment
from watermark import KnowGuesswatermark, set_seed

# =============================================================================
# Configuration
# =============================================================================
BASE_PATH = Path(r'path/to/synthetic/data')  
TASKS = ['california', 'gesture', 'house', 'wilt', 'higgs-small', 'miniboone']
NUM_FILES = 50
NUM_SAMPLE_DATA = 5

# Experiment parameters
GAMMA_LIST = [0.3, 0.5, 0.7]
M = 1000
REM = [100, 1000, 10000]  # attacker's guess for m


# =============================================================================
# Run Experiments
# =============================================================================

def main():
    set_seed(42) 
    results = []
    
    for rem in REM:
        for gamma in GAMMA_LIST:
            print(f'\n{"="*50}')
            print(f'Running experiments with gamma={gamma}, rem={rem}')
            print(f'{"="*50}\n')
            
            for task in TASKS:
                # Set delta based on task
                if task in ['california', 'gesture', 'house', 'wilt', 'higgs-small']:
                    delta = 0.005
                else:
                    delta = 0.0025
                
                result = run_experiment(
                    base_path=BASE_PATH,
                    task=task,
                    gamma=gamma,
                    m=M,
                    attack_fn=KnowGuesswatermark,
                    attack_kwargs={'mtwo': rem},
                    num_files=NUM_FILES,
                    num_sample_data=NUM_SAMPLE_DATA,
                    delta=delta,
                    use_fixed_m=True,
                    verbose=True
                )
                result['rem'] = rem
                results.append(result)
    
    # Print summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    for r in results:
        if r.get('status') == 'no_columns':
            print(f"{r['task']}: No suitable columns")
        else:
            print(f"{r['task']} (gamma={r['gamma']}, rem={r['rem']}): "
                  f"AUC_orig={r['auc_original']:.4f}, "
                  f"AUC_atk={r['auc_attacked']:.4f}")


if __name__ == '__main__':
    main()
