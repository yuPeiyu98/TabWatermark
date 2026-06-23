# TabWatermark — Repro Guide

This README documents **only the root-level scripts in this repository**, except for the “Mapping from results to scripts” section, which also provides a high-level mapping between the reported results and the corresponding scripts. The `real_data/` folder is intentionally excluded from the root-level script documentation, as it has its own README.


---

## 0) Folder layout (root only)

Scripts covered here:

- `add_watermark.py`
- `add_watermark_attack.py`
- `add_watermark_curves_fig_5_ab.py`
- `add_watermark_attack_curves_fig_5_cd.py`
- `kde_plot_fig_3.py`
- `heatmap_plot_fig_4.py`
- `rewatermark_with_plot.py`
- `gamma_watermark_guessing_attack_tab_5.py`
- `Tab_8-13.py`
- `p_watermark.py`
- `add_watermark_tab_15.py`
- `add_watermark_tab_16.py`
- `add_watermark_attack_tab_17.py`

Most table scripts **print results to stdout**.  
Most figure scripts **save `.png` files** to the current working directory.

## Mapping from results to scripts

| Result | Location | Script |
|---|---|---|
| Fig. 3 | Main paper | `kde_plot_fig_3.py` |
| Fig. 4 | Main paper | `heatmap_plot_fig_4.py` |
| Fig. 5(a,b) | Main paper | `add_watermark_curves_fig_5_ab.py` |
| Fig. 5(c,d) | Main paper | `add_watermark_attack_curves_fig_5_cd.py` |
| Fig. 6 | Appendix | `real_data/code_figure_6.py` |
| Tab. 1 | Main paper | `real_data/scripts/exp_wasserstein.py` |
| Tab. 2 | Main paper | `real_data/scripts/exp_noise_attack.py` |
| Tab. 3 | Main paper | `real_data/scripts/exp_noise_attack.py` |
| Tab. 4 | Main paper | `real_data/scripts/exp_utility_eval.py` |
| Tab. 5 | Appendix | `gamma_watermark_guessing_attack_tab_5.py` |
| Tab. 6 | Appendix | `real_data/scripts/exp_know_guess_attack.py` |
| Tab. 7 | Appendix | `real_data/scripts/exp_guess_attack.py` |
| Tab. 8--13 | Appendix | `Tab_8-13.py` |
| Tab. 14 | Appendix | Manuscript dataset-information table; no script |
| Tab. 15 | Appendix | `add_watermark_tab_15.py` |
| Tab. 16 | Appendix | `add_watermark_tab_16.py` |
| Tab. 17 | Appendix | `add_watermark_attack_tab_17.py` |
| Tab. 18 | Appendix | `real_data/scripts/exp_wasserstein.py` |
| Tab. 19 | Appendix | `real_data/scripts/exp_wasserstein.py` |
| Tab. 20 | Appendix | `rewatermark_with_plot.py` |
| Tab. 21 | Appendix | `rewatermark_with_plot.py` |
| Tab. 22 | Appendix | `real_data/scripts/exp_rewm_attack.py` |
| Tab. 23 | Appendix | `p_watermark.py` |
| Tab. 24 | Appendix | `real_data/scripts/exp_soft_watermark.py` |

---

## 1) Environment setup

### Python
Use **Python 3.10+** (3.10/3.11 recommended).

### Dependencies
From the root folder:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install numpy scipy matplotlib seaborn scikit-learn tqdm pandas
```

We provide `requirements.txt` with the exact package versions used to produce the reported results. To install the same package versions as used in our experiments, run:

```bash
pip install -r requirements.txt
```
---

## 2) Mapping from scripts to tables / figures (Note: Scripts without special labels correspond to tables and figures in the manuscript (including both the main text and appendix). Scripts associated with the response letter are explicitly labeled.)

### Figures

#### Fig. 3 — KDE distributions before vs after watermark
**Script:** `kde_plot_fig_3.py`

**Run:**
```bash
python kde_plot_fig_3.py --seed 1
```

**Outputs:**
- `0_before.png`, `0_after.png`, `1_before.png`, `1_after.png`, …

Each pair corresponds to one synthetic column. In our manuscript, we display the figures for the first column.

---

#### Fig. 4 — Correlation / heatmap visualization
**Script:** `heatmap_plot_fig_4.py`

**Run:**
```bash
python heatmap_plot_fig_4.py --seed 1
```

**Outputs:**
- `corr_no_wm_n{n}_c{c}_m{m}.png`

(Other heatmaps exist in the code but are commented out.)

---

#### Fig. 5 (a,b) — Detection accuracy vs n
**Script:** `add_watermark_curves_fig_5_ab.py`

**Run:**
```bash
python add_watermark_curves_fig_5_ab.py --seed 1 --m 10000
```

**Outputs:**
- Figure 5 (a,b)


---

#### Fig. 5 (c,d) — Robustness under attack
**Script:** `add_watermark_attack_curves_fig_5_cd.py`

**Run:**
```bash
python add_watermark_attack_curves_fig_5_cd.py --seed 1 --m 1000
```

**Outputs:**
- Figure 5 (c,d)



---

### Tables

---

#### Tab. 5 — Gamma watermark guessing attack
**Script:** `gamma_watermark_guessing_attack_tab_5.py`

**Run:**
```bash
python gamma_watermark_guessing_attack_tab_5.py --seed 1 --m 1000
```

**Outputs:**
- Printed TPR, TNR, and AUC for all sweep settings

Recommended:
```bash
python gamma_watermark_guessing_attack_tab_5.py --seed 1 --m 1000 | tee tab5.log
```

---

#### Tab. 8--13 - Soft watermark utility--robustness trade-off
**Script:** `Tab_8-13.py`

**Run:**
```bash
python Tab_8-13.py --seed 20260608
```

**Outputs:**
- Printed fixed-m and adaptive-m(q) summary tables for m_hard = 500, 1000, 1500
- CSV summary saved to `q1_soft_watermark_multi_m_summary/all_summary.csv`

---

#### Tab. 15 
**Script:** `add_watermark_tab_15.py`

**Run:**
```bash
python add_watermark_tab_15.py --m 1000
```
---

#### Tab. 16 
**Script:** add_watermark_tab_16.py

**Run:**
```bash
python add_watermark_tab_16.py
```
---

#### Tab. 17 
**Script:** add_watermark_attack_tab_17.py

**Run:**
```bash
python add_watermark_attack_tab_17.py
```
---

#### Tab. 20 
**Script:** rewatermark_with_plot.py

**Run:**
```bash
python rewatermark_with_plot.py --m 1000
```

---

#### Tab. 21 
**Script:** rewatermark_with_plot.py

**Run:**
```bash
python rewatermark_with_plot.py --m 2000
```

#### Tab. 23 
**Script:** `p_watermark.py`

**Run:**
```bash
python p_watermark.py --seed 1 --m 1000
```

**Outputs:**
- Printed TPR, TNR, and AUC for different q settings(Table 23 in the Appendix)

---

## 3) Utility / sanity scripts

### Basic watermark sanity check
**Script:** `add_watermark.py`

```bash
python add_watermark.py --seed 1 --m 1000
```

---

### Attack sanity check
**Script:** `add_watermark_attack.py`

```bash
python add_watermark_attack.py --seed 1 --m 1000 | tee attack.log
```

---

### Rewatermark plotting helper
**Script:** `rewatermark_with_plot.py`

```bash
python rewatermark_with_plot.py --seed 1 --m 2000
```

Saved plots depend on which `savefig` calls are enabled in the script.

---

## 4) Repro tips

- Run figure scripts from a dedicated directory to keep PNGs organized.
- Always capture stdout (`tee`) for table scripts.
- All randomness is controlled via `--seed`.
- Numerical resolution is controlled by `--m` where available.

---

This README is intentionally limited to **root-folder synthetic experiments only**.
