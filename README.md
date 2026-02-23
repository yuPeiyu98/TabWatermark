# TabWatermark — Repro Guide

This README documents **only the root-level scripts** in this repository and explains how to reproduce the **tables (Tab. *) and figures (Fig. *)** reported by the code.  
The `real_data/` folder is intentionally **excluded**, as it already has its own README.

---

## 0) Folder layout (root only)

Scripts covered here:

- `add_watermark.py`
- `add_watermark_attack.py`
- `add_watermark_curves_fig_5_ab.py`
- `add_watermark_attack_curves_fig_5_cd.py`
- `kde_plot_fig_3.py`
- `heatmap_plot_fig_4.py`
- `random_rescale_response_tab1_tab3.py`
- `rewatermark_response_tab4.py`
- `rewatermark_with_plot.py`
- `gamma_watermark_guessing_attack_tab_5.py`
- `p_watermark.py`
- `soft_robustness_tab_8.py`
- `add_watermark_tab_10.py`
- `add_watermark_tab_11.py`
- `add_watermark_attack_tab_12.py`

Most table scripts **print results to stdout**.  
Most figure scripts **save `.png` files** to the current working directory.

---

## 1) Environment setup

### Python
Use **Python 3.9+** (3.10/3.11 recommended).

### Dependencies
From the root folder:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install numpy scipy matplotlib seaborn scikit-learn tqdm
```

---

## 2) Mapping from scripts to tables / figures

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

#### Tab. 1 & Tab. 3 in the response letter — Random rescale response
**Script:** `random_rescale_response_tab1_tab3.py`

**Run (auto-executes):**
```bash
python random_rescale_response_tab1_tab3.py
```

**Outputs:**
- Printed TPR, TNR, and AUC for each (n, p)

Recommended:
```bash
python random_rescale_response_tab1_tab3.py | tee tab1_tab3.log
```

---

#### Tab. 4 in the response letter — Rewatermark response
**Script:** `rewatermark_response_tab4.py`

**Run (auto-executes):**
```bash
python rewatermark_response_tab4.py
```

**Outputs:**
- Printed AUC values for each (n, p)

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

#### Tab. 18 in the Appendix - q-Watermark sensitivity table
**Script:** `p_watermark.py`

**Run:**
```bash
python p_watermark.py --seed 1 --m 1000
```

**Outputs:**
- Printed TPR, TNR, and AUC for different q settings(Table 18 in the Appendix)

---

#### Tab. 8 — Soft watermark robustness
**Script:** `soft_robustness_tab_8.py`

**Run:**
```bash
python soft_robustness_tab_8.py
```

**Outputs:**
- Printed summary table with columns:
  - q
  - m
  - sigma_break
  - W1_wm
  - Delta_W1

Recommended:
```bash
python soft_robustness_tab_8.py | tee tab8.log
```

---

#### Tab. 10 in Appendix
**Script:** `add_watermark_tab_10.py`

**Run:**
```bash
python add_watermark_tab_10.py --m 1000
```
---

#### Tab. 11 in Appendix
**Script:** add_watermark_tab_11.py

**Run:**
```bash
python add_watermark_tab_11.py
```
---

#### Tab. 12 in Appendix
**Script:** add_watermark_attack_tab_12.py

**Run:**
```bash
python add_watermark_attack_tab_12.py
```
---

#### Tab. 15 in Appendix
**Script:** rewatermark_with_plot.py

**Run:**
```bash
python rewatermark_with_plot.py --m 1000
```

---

#### Tab. 16 in Appendix
**Script:** rewatermark_with_plot.py

**Run:**
```bash
python rewatermark_with_plot.py --m 2000
```

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
