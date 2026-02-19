# Tabular Data Watermarking

This repository contains the implementation of watermarking methods for synthetic tabular data, along with various attack experiments.

## Project Structure

```
tabular_watermark/
├── watermark.py           # Core watermarking functions
├── run_experiment.py      # Common experiment utilities (used in Table 2)
├── scripts/
│   ├── exp_know_guess_attack.py    # Guess attack (known mean/std)
│   ├── exp_guess_attack.py         # Guess attack (unknown mean/std)
│   ├── exp_rewm_attack.py          # Re-watermarking attack
│   ├── exp_noise_attack.py         # Additive noise attack (used in Table 3)
│   ├── exp_wasserstein.py         # Evaluate wasserstein distance (used in Tables 1, 13, and 14)
│   └── exp_soft_watermark.py       # Soft watermarking experiment
└── README.md
```

## Dependencies

This experiment builds upon [TabDDPM](https://github.com/rotot0/tab-ddpm) for synthetic data generation. We fully use their implementation to train and generate synthetic tabular data with TabDDPM / CTGAN / TVAE models. Our most experiment scenarios use TabDDPM due to its good generation quality.

## Data Preparation

```
base_path/
├── california/
│   ├── 0/
│   │   └── X_num_train_raw_syn.npy
│   ├── 1/
│   │   └── X_num_train_raw_syn.npy
│   └── ...
├── gesture/
│   └── ...
└── ...
```

## Usage

### Running Experiments

Each experiment script can be run independently. First, modify the `BASE_PATH` in each script to point to your data directory.

```bash
cd scripts

# Run guess attack with known mean/std
python exp_know_guess_attack.py

# Run guess attack with unknown mean/std
python exp_guess_attack.py

# Run re-watermarking attack
python exp_rewm_attack.py

# Run noise attack
python exp_noise_attack.py

# Run soft watermarking experiment
python exp_soft_watermark.py

# Run wasserstein distance experiment
python exp_wasserstein.py
```
### Utility Evaluation Before and After Watermarking

We evaluate the utility of synthetic tabular data using the official utility
evaluation pipeline provided by the TabDDPM repository.

The evaluation procedure is as follows:

1. Train tabular generative models (TabDDPM / CTGAN / TVAE) and generate synthetic
   datasets following the original TabDDPM implementation.
2. Measure the utility of the generated data using TabDDPM's provided evaluation
   code.
3. Apply our watermarking method to the generated synthetic data.
4. Replace the original synthetic data with the watermarked version and
   re-run the same TabDDPM utility evaluation code.

The only difference between the two evaluations is the presence of watermarking
in the synthetic data. All evaluation metrics, models, and procedures remain
unchanged.

### Using Core Functions

```python
from watermark import (
    getGreenList,
    singleColumnWatermark,
    gaussMultTesting,
    get_p_value
)

# Generate green list
green_list = getGreenList(gamma=0.5, m=1000)

# Apply watermark to a column
arr_wm, mean, std = singleColumnWatermark(arr, GreenListsInfo, col_id, m=1000)

# Perform statistical test
p_value = gaussMultTesting(arrs, green_lists, m=1000, gamma=0.5)
```

## Method Overview

### Watermark Embedding

The watermarking method works by:
1. Normalizing data using empirical mean/std
2. Mapping values to [0, 1] intervals
3. Shifting values into "green list" regions
4. Denormalizing back to original scale

### Watermark Detection

Detection uses chi-squared hypothesis testing:
- Count elements falling in green list regions
- Aggregate test statistics across columns
- Compare against chi-squared distribution

### Attacks

- **Guess Attack**: Attacker tries to guess and overwrite the watermark intervals
- **Re-watermarking Attack**: Attacker applies their own watermark
- **Noise Attack**: Attacker adds Gaussian noise to watermarked data

## Parameters

- `gamma`: Proportion of green list coverage (default: 0.5)
- `m`: Number of intervals for quantization (default: 1000)
- `delta`: Tolerance for column filtering (default: 0.005)

## Citation

If you use this code, please cite:

```bibtex
@article{he2024watermarking, 
title={Watermarking Generative Tabular Data}, 
author={He, H. and Yu, P. and Ren, J. and Wu, Y. N. and Cheng, G.}, 
journal={arXiv preprint arXiv:2405.14018}, 
year={2024} }
```

