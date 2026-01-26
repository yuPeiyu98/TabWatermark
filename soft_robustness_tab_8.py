
"""
Experiment (1D):
- Data: Gaussian Mixture (GMM)
- Watermark: nearest green bin (within {pair-1, pair, pair+1})
              chosen by BIN-CENTER distance + random position in that green bin
- Soft watermark: apply with prob q, else unchanged
- Granularity: m(q) = round_even(q * m_hard), with m_hard = 500
- Break attack: additive Gaussian noise with std sigma
- Break search: GRID SCAN over sigma in (1/m)*[0,5] (no bisection)
- Metrics per q:
    W1_wm      = mean over 5 reps of W1(X, X_wm)
    sigma_break = minimal sigma on grid s.t. mean AUC <= auc_thresh
    Delta_W1   = mean over 5 reps of W1(X_wm, X_wm + eps_break)
"""

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score


# =============================
# GMM sampler
# =============================

GMM_WEIGHTS = np.array([0.5, 0.5], dtype=float)
GMM_MEANS   = np.array([-2.0,  2.0], dtype=float)
GMM_STDS    = np.array([ 1.0,  1.0], dtype=float)

def sample_gmm(rng: np.random.Generator, size: int) -> np.ndarray:
    z = rng.choice(2, size=size, p=GMM_WEIGHTS)
    return rng.normal(loc=GMM_MEANS[z], scale=GMM_STDS[z], size=size)


# =============================
# Helpers: bins / green
# =============================

def frac_part(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 1.0)

def next_even(v: int) -> int:
    return v if v % 2 == 0 else v + 1

def make_green_is_left(m: int, rng: np.random.Generator) -> np.ndarray:
    if m % 2 != 0:
        raise ValueError("m must be even.")
    return rng.integers(0, 2, size=m // 2, dtype=np.int8)

def is_green_bin(bin_idx: np.ndarray, green_is_left: np.ndarray) -> np.ndarray:
    pair = bin_idx // 2
    side = bin_idx % 2
    return ((side == 0) & (green_is_left[pair] == 1)) | \
           ((side == 1) & (green_is_left[pair] == 0))

def green_bin_of_pair(pair_idx: np.ndarray, green_is_left: np.ndarray) -> np.ndarray:
    return 2 * pair_idx + (green_is_left[pair_idx] == 0).astype(np.int64)


# =============================
# Watermarking
# =============================

def watermark_hard_nearest_green_randompos_1d(
    x: np.ndarray,
    m: int,
    green_is_left: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    u = frac_part(x)
    b = np.floor(u * m).astype(np.int64)
    pair = b // 2
    P = m // 2

    green_mask = is_green_bin(b, green_is_left)
    red_mask = ~green_mask
    if not np.any(red_mask):
        return x.copy()

    pair0 = pair[red_mask]
    cand_pairs = np.stack(
        [(pair0 - 1) % P, pair0, (pair0 + 1) % P], axis=1
    )
    cand_gbins = green_bin_of_pair(cand_pairs, green_is_left)

    # 🔑 distance = |center - original fractional part|
    u0 = u[red_mask][:, None]
    centers = (cand_gbins + 0.5) / m
    dist = np.abs(centers - u0)

    min_dist = dist.min(axis=1, keepdims=True)
    is_min = np.isclose(dist, min_dist, rtol=0.0, atol=1e-12)

    choice = np.empty(is_min.shape[0], dtype=np.int64)
    for i in range(is_min.shape[0]):
        opts = np.flatnonzero(is_min[i])
        choice[i] = opts[rng.integers(0, opts.size)]

    chosen_gbin = cand_gbins[np.arange(cand_gbins.shape[0]), choice]
    tprime = rng.random(size=chosen_gbin.shape[0])

    u_new = u.copy()
    u_new[red_mask] = (chosen_gbin + tprime) / m

    x_new = np.floor(x) + u_new
    x_new[green_mask] = x[green_mask]
    return x_new

def watermark_soft_nearest_green_randompos_1d(
    x: np.ndarray,
    m: int,
    q: float,
    green_is_left: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x_h = watermark_hard_nearest_green_randompos_1d(x, m, green_is_left, rng)
    mask = rng.random(size=x.shape) < q
    out = x.copy()
    out[mask] = x_h[mask]
    return out


# =============================
# Metrics
# =============================

def w1(x: np.ndarray, y: np.ndarray) -> float:
    return float(wasserstein_distance(x, y))

def green_zscore_dataset(y: np.ndarray, m: int, green_is_left: np.ndarray) -> float:
    u = frac_part(y)
    b = np.floor(u * m).astype(np.int64)
    g = is_green_bin(b, green_is_left)
    p_hat = np.mean(g)
    N = y.size
    return (p_hat - 0.5) / np.sqrt(0.25 / N)


def w1_metrics_repeat_1d(
    rng: np.random.Generator,
    n_fid: int,
    m: int,
    q: float,
    green_is_left: np.ndarray,
    sigma_break: float | None,
    reps: int = 5,
):
    w1_wm_vals = []
    delta_vals = []

    for _ in range(reps):
        x = sample_gmm(rng, n_fid)
        xw = watermark_soft_nearest_green_randompos_1d(x, m, q, green_is_left, rng)

        w1_wm_vals.append(w1(x, xw))

        if sigma_break is not None:
            eps = rng.normal(0.0, sigma_break, size=n_fid)
            y = xw + eps
            delta_vals.append(w1(xw, y))

    return float(np.mean(w1_wm_vals)), float(np.mean(delta_vals))


# =============================
# AUC at a given sigma
# =============================

def auc_at_sigma(
    n: int,
    K: int,
    q: float,
    m: int,
    green_is_left: np.ndarray,
    sigma: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    scores = np.empty(2 * K)
    labels = np.zeros(2 * K)

    for k in range(K):
        x = sample_gmm(rng, n)
        y = x + rng.normal(0.0, sigma, size=n)
        scores[k] = green_zscore_dataset(y, m, green_is_left)

        x = sample_gmm(rng, n)
        xw = watermark_soft_nearest_green_randompos_1d(x, m, q, green_is_left, rng)
        y = xw + rng.normal(0.0, sigma, size=n)
        scores[K + k] = green_zscore_dataset(y, m, green_is_left)
        labels[K + k] = 1

    return float(roc_auc_score(labels, scores))

def auc_mean(
    n: int,
    K: int,
    q: float,
    m: int,
    green_is_left: np.ndarray,
    sigma: float,
    seed: int,
    reps: int = 3,
) -> float:
    return float(np.mean([
        auc_at_sigma(n, K, q, m, green_is_left, sigma, seed + 10_000 * r)
        for r in range(reps)
    ]))


# =============================
# Break sigma by grid scan
# =============================

def find_break_sigma_grid(
    n: int,
    K: int,
    q: float,
    m: int,
    green_is_left: np.ndarray,
    auc_thresh: float,
    grid_points: int = 81,
    sigma_max_multiplier: float = 5.0,
    reps_per_eval: int = 3,
    seed0: int = 12345,
):
    sigma_grid = (1.0 / m) * np.linspace(0.0, sigma_max_multiplier, grid_points)
    for i, sigma in enumerate(sigma_grid):
        a = auc_mean(n, K, q, m, green_is_left, sigma,
                     seed=seed0 + i, reps=reps_per_eval)
        print(f"  sigma={sigma:.6g} (sigma*m={sigma*m:.3f}), AUC={a:.3f}")
        if a <= auc_thresh:
            return float(sigma)
    return None


def run_experiment():
    q_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    m_hard = next_even(500)
    n = 1_000
    n_fid = 1_000
    K = 250
    auc_thresh = 0.55

    results = []

    for q in q_list:
        m = next_even(max(2, int(round(q * m_hard))))
        rng_key = np.random.default_rng(2026 + m)
        green = make_green_is_left(m, rng_key)

        print(f"\n===== q={q}, m={m} =====")
        sigma_break = find_break_sigma_grid(
            n, K, q, m, green, auc_thresh
        )
        print(f"sigma_break = {sigma_break}")

        rng = np.random.default_rng(123)
        W1_wm, Delta_W1 = w1_metrics_repeat_1d(
            rng, n_fid, m, q, green, sigma_break, reps=5
        )

        print(f"W1_wm    = {W1_wm:.6g}  (avg over 5 reps)")
        print(f"Delta_W1 = {Delta_W1:.6g}  (avg over 5 reps)")

        results.append({
            "q": q,
            "m": m,
            "sigma_break": sigma_break,
            "W1_wm": W1_wm,
            "Delta_W1": Delta_W1,
        })

    print("\n========== SUMMARY TABLE ==========")
    print(" q     m     sigma_break     W1_wm        Delta_W1")
    print("----------------------------------------------------")
    for r in results:
        print(
            f"{r['q']:<4.1f}  {r['m']:<4d}  "
            f"{str(r['sigma_break']):<14}  "
            f"{r['W1_wm']:<10.4g}  {r['Delta_W1']:<10.4g}"
        )

    return results

if __name__ == "__main__":
  run_experiment()