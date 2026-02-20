import numpy as np
import pandas as pd
from scipy import stats
import warnings
from math import comb

warnings.filterwarnings("ignore")

def max_t_permutation_test(
    csv_file_path,
    n_permutations=5000,
    tail=0,                 # 0 two-tailed, 1 positive (group1>group0), -1 negative (group1<group0)
    random_state=42,
    alpha=0.05,
    nan_policy="omit",      # default: per-feature omission
    n_bootstrap=2000,       # bootstrap resamples for effect-size CI
    ci_level=0.95           # e.g., 0.95 -> 95% CI
):
    """
    Mass univariate permutation t-test with tmax (FWER) correction (Welch t),
    Hedges' g based on d_av denominator, and bootstrap CIs (within-group).

    CSV format:
      - Column 0: labels (two groups; any values -> mapped to {0,1})
      - Columns 1..N: features or variables (numeric; NaNs handled via nan_policy)

    Notes:
      - Uses Welch's t-test ALWAYS (equal_var=False).
      - Effect size: d_av = (mean1-mean0)/sqrt((s0^2+s1^2)/2),
        then Hedges correction g_av = J * d_av with J ≈ 1 - 3/(4*df - 1), df = n0+n1-2 (per feature).
      - Bootstrap CI: resample within each group for each feature (per-feature valid rows only).
      - nan_policy='omit' (default): per-feature omission (pairwise deletion).

    Returns
    -------
    results : dict
    df_out : pd.DataFrame
        Summary table for all features.
    sig_mask : np.ndarray of shape (C,)
        Boolean mask where p_corr_tmax < alpha.
    sig_features : list[str]
        Names of significant features (corrected p < alpha).
    raw_data : pd.DataFrame
        The original data as read (no imputation performed for nan_policy='omit').
    """
    if nan_policy not in ("omit",):
        raise ValueError("This version supports only nan_policy='omit'.")

    rng = np.random.default_rng(random_state)

    # 1) load dataset (CSV)
    raw_data = pd.read_csv(csv_file_path)
    if raw_data.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns: labels + ≥1 feature.")

    # labels -> exactly two groups mapped to {0,1}
    label_raw = raw_data.iloc[:, 0]
    cats = pd.Categorical(label_raw)
    if len(cats.categories) != 2:
        raise ValueError(f"Expected exactly 2 groups, found {len(cats.categories)}.")
    labels = cats.codes  # 0/1
    group_names = list(cats.categories)

    # features as float array (keep NaNs; we will omit per feature)
    X = raw_data.iloc[:, 1:].to_numpy(dtype=float)
    ch_names = raw_data.columns[1:].tolist()
    n = X.shape[0]
    C = X.shape[1]

    # group sizes (based on labels only; per-feature sizes may differ under nan_policy='omit')
    n0_all = int(np.sum(labels == 0))
    n1_all = int(np.sum(labels == 1))
    if n0_all == 0 or n1_all == 0:
        raise ValueError("One of the groups is empty after label parsing.")

    # print total unique label permutations (choose n1 from n)
    total_unique = comb(n, n1_all)
    print(f"Total unique label permutations (n={n}, n1={n1_all}): {total_unique}")

    # 2) observed Welch t and uncorrected p + effect size g_av and bootstrap CIs
    t_obs = np.full(C, np.nan, dtype=float)
    p_unc = np.full(C, np.nan, dtype=float)

    g_av = np.full(C, np.nan, dtype=float)
    g_ci_low = np.full(C, np.nan, dtype=float)
    g_ci_high = np.full(C, np.nan, dtype=float)

    # store Welch–Satterthwaite degrees of freedom per feature
    df_welch = np.full(C, np.nan, dtype=float)


    # for one-sided p from two-sided p
    def one_sided_from_two_sided(p2, tval, direction):  # direction: 1 or -1
        if not np.isfinite(p2) or not np.isfinite(tval):
            return np.nan
        if direction == 1:
            return (p2 / 2.0) if (tval >= 0) else (1.0 - p2 / 2.0)
        else:
            return (p2 / 2.0) if (tval <= 0) else (1.0 - p2 / 2.0)

    # Hedges correction factor J (per feature df)
    def hedges_J(df):
        # common approximation; guard for tiny df
        if df is None or df <= 1:
            return np.nan
        return 1.0 - (3.0 / (4.0 * df - 1.0))

    # compute g_av for given arrays
    def compute_g_av(x0, x1):
        n0 = x0.size
        n1 = x1.size
        if n0 < 2 or n1 < 2:
            return np.nan
        m0 = np.mean(x0)
        m1 = np.mean(x1)
        v0 = np.var(x0, ddof=1)
        v1 = np.var(x1, ddof=1)
        denom = np.sqrt((v0 + v1) / 2.0)
        if not np.isfinite(denom) or denom == 0:
            return 0.0
        d_av = (m1 - m0) / denom
        df = n0 + n1 - 2
        J = hedges_J(df)
        if not np.isfinite(J):
            return np.nan
        return J * d_av

    # >>> Welch–Satterthwaite degrees of freedom for each feature
    def welch_satterthwaite_df(x0, x1):
        """
        Compute Welch–Satterthwaite degrees of freedom for Welch's two-sample t-test.
        """
        n0 = x0.size
        n1 = x1.size
        if n0 < 2 or n1 < 2:
            return np.nan
        v0 = np.var(x0, ddof=1)
        v1 = np.var(x1, ddof=1)
        a = v0 / n0
        b = v1 / n1
        denom = (a * a) / (n0 - 1) + (b * b) / (n1 - 1)
        if denom == 0 or not np.isfinite(denom):
            return np.nan
        return (a + b) * (a + b) / denom


    # bootstrap CI for g_av (within groups)
    def bootstrap_ci_g(x0, x1, B, level, rng_local):
        n0 = x0.size
        n1 = x1.size
        if n0 < 2 or n1 < 2:
            return (np.nan, np.nan)
        if B is None or B <= 0:
            return (np.nan, np.nan)

        g_samples = np.empty(B, dtype=float)
        for b in range(B):
            s0 = x0[rng_local.integers(0, n0, size=n0)]
            s1 = x1[rng_local.integers(0, n1, size=n1)]
            g_samples[b] = compute_g_av(s0, s1)

        g_samples = g_samples[np.isfinite(g_samples)]
        if g_samples.size < max(20, int(0.1 * B)):
            return (np.nan, np.nan)

        alpha_tail = (1.0 - level) / 2.0
        low = np.quantile(g_samples, alpha_tail)
        high = np.quantile(g_samples, 1.0 - alpha_tail)
        return (low, high)

    # observed stats per feature with nan_policy='omit'
    for j in range(C):
        col = X[:, j]
        valid = np.isfinite(col)
        if not np.any(valid):
            continue

        xj = col[valid]
        lj = labels[valid]

        x0 = xj[lj == 0]
        x1 = xj[lj == 1]
        if x0.size < 2 or x1.size < 2:
            continue

        # Welch t-test
        t, p2 = stats.ttest_ind(x1, x0, equal_var=False)
        t_obs[j] = t

        # compute and store Welch df for this feature
        df_welch[j] = welch_satterthwaite_df(x0, x1)

        # p-values
        if tail == 0:
            p_unc[j] = p2
        elif tail == 1:
            p_unc[j] = one_sided_from_two_sided(p2, t, direction=1)
        else:  # tail == -1
            p_unc[j] = one_sided_from_two_sided(p2, t, direction=-1)

        # effect size: Hedges g_av + bootstrap CI
        g_av[j] = compute_g_av(x0, x1)
        lo, hi = bootstrap_ci_g(x0, x1, n_bootstrap, ci_level, rng)
        g_ci_low[j] = lo
        g_ci_high[j] = hi

    # 3) permutation null of tmax (Welch t; nan_policy='omit')
    def reduce_max_t(t_vec):
        # handle all-nan safely
        tv = t_vec[np.isfinite(t_vec)]
        if tv.size == 0:
            return 0.0
        if tail == 0:
            return float(np.max(np.abs(tv)))
        elif tail == 1:
            return float(np.max(tv))
        else:
            return float(np.max(-tv))  # largest magnitude in negative direction

    T_max = np.zeros(n_permutations, dtype=float)

    for b in range(n_permutations):
        perm = rng.permutation(labels)
        t_perm = np.full(C, np.nan, dtype=float)

        for j in range(C):
            col = X[:, j]
            valid = np.isfinite(col)
            if not np.any(valid):
                continue

            xj = col[valid]
            pj = perm[valid]

            g0 = xj[pj == 0]
            g1 = xj[pj == 1]
            if g0.size < 2 or g1.size < 2:
                continue

            t, _ = stats.ttest_ind(g1, g0, equal_var=False)  # Welch
            t_perm[j] = t

        T_max[b] = reduce_max_t(t_perm)

    # 4) corrected p-values 
    if tail == 0:
        stat_obs = np.abs(t_obs)
    elif tail == 1:
        stat_obs = t_obs
    else:
        stat_obs = -t_obs

    p_corr = np.full(C, np.nan, dtype=float)
    finite_obs = np.isfinite(stat_obs)
    if np.any(finite_obs):
        counts = (T_max[:, None] >= stat_obs[None, finite_obs]).sum(axis=0)
        p_corr[finite_obs] = (1.0 + counts) / (n_permutations + 1.0)

    # 5) assemble outputs
    df_out = pd.DataFrame({
        "#": ch_names,
        "t_obs_welch": t_obs,
        "df_welch": df_welch,
        "p_uncorrected": p_unc,
        "p_corr_tmax": p_corr,
        "hedges_g_av": g_av,
        "hedges_g_ci_low": g_ci_low,
        "hedges_g_ci_high": g_ci_high,
        "Sign": np.where(t_obs > 0, "pos", np.where(t_obs < 0, "neg", "nan_or_zero")),
    }).sort_values("p_corr_tmax", na_position="last").reset_index(drop=True)

    # 6) significant features
    sig_mask = (p_corr < alpha) & np.isfinite(p_corr)
    sig_features = [ch for ch, keep in zip(ch_names, sig_mask) if keep]

    results = {
        "#": ch_names,
        "t_obs_welch": t_obs,
        "p_uncorrected": p_unc,
        "p_corr_tmax": p_corr,
        "hedges_g_av": g_av,
        "hedges_g_ci_low": g_ci_low,
        "hedges_g_ci_high": g_ci_high,
        "H0_Tmax": T_max,
        "labels_mapped": {"group0": group_names[0], "group1": group_names[1]},
        "group_sizes_labels_only": {"n0": n0_all, "n1": n1_all},
        "permutation_space": {"total_unique_label_permutations": total_unique},
        "test_parameters": {
            "n_permutations": n_permutations,
            "tail": tail,
            "random_state": random_state,
            "alpha": alpha,
            "nan_policy": nan_policy,
            "welch_equal_var": False,
            "bootstrap": {"n_bootstrap": n_bootstrap, "ci_level": ci_level},
        },
    }

    return results, df_out, sig_mask, sig_features, raw_data


def save_max_t_results(df_out, output_file="max_t_results.csv"):
    df_out.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def save_significant_feature_values(
    raw_data: pd.DataFrame,
    sig_features: list,
    output_file: str = "significant_feature_values.csv",
    include_label: bool = True,
):
    """
    Save a CSV with only the significant feature columns (and optionally the label column).
    Note: raw_data is saved as-is (may include NaNs) since inference used nan_policy='omit'.
    """
    if include_label:
        cols = [raw_data.columns[0]] + sig_features
    else:
        cols = sig_features

    if len(sig_features) == 0:
        print(
            "No features met the corrected p-value threshold; writing file with"
            f" {'label column only' if include_label else 'no columns'}."
        )

    raw_data.loc[:, cols].to_csv(output_file, index=False)
    print(f"Significant feature values saved to {output_file}")