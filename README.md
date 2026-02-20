# Mass Univariate Permutation T-Test with Max-T FWER Correction

## Overview

This repository implements a **mass univariate two-sample permutation t-test** with:
- Welch’s t-statistic (heteroscedasticity-robust)
- Max-T family-wise error rate (FWER) correction
- Phipson–Smyth adjusted permutation $$p$$-values
- Hedges’ $$g$$ effect size (based on heteroscedastic $$d_{av}$$)
- Bootstrap confidence intervals (within-group resampling)

The implementation is written from scratch using NumPy, SciPy, and Pandas. It accepts **any tabular feature matrix** as input, without relying on any pipeline-specific data structures.


## Motivation

Established mass univariate and permutation-based frameworks (MNE-Python, FieldTrip, Nilearn, dmgroppe toolbox, etc.) are typically coupled to proprietary data structures or domain-specific pipelines. Therefore, these frameworks are not suitable for **custom scalar features** extracted from biosignals.

This repository aims to provide a two-group comparison framework that allows for arbitrary tabular input. The sole requirement is that features are organized as a standard CSV file (see **Input Format**). No pipeline integration, toolbox installation, or domain-specific preprocessing is required.


## Statistical Framework

### 1. Hypothesis Testing

For each feature (variable), the observed statistic is:

Welch’s two-sample t-statistic:

$$
t = \frac{\bar{x}_1 - \bar{x}_0}
{\sqrt{\frac{s_0^2}{n_0} + \frac{s_1^2}{n_1}}}
$$

This test does not assume equal variances and is robust under heteroscedasticity.

For each feature, the Welch–Satterthwaite approximation is used to compute the effective degrees of freedom:

$$
df =
\frac{
\left(\frac{s_0^2}{n_0} + \frac{s_1^2}{n_1}\right)^2
}{
\frac{\left(\frac{s_0^2}{n_0}\right)^2}{n_0 - 1}
+
\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1 - 1}
}
$$

### 2. Permutation-Based Inference

A nonparametric permutation test is performed by:
- Randomly permuting group labels
- Computing Welch’s t-statistic for each feature
- Recording the maximum statistic across all features (Max-T)

This constructs a null distribution of the maximum statistic:

$$
T_{\max}^{(b)}
$$

Family-wise error rate (FWER) control is achieved by comparing observed statistics to this distribution.

Corrected $$p$$-values use the Phipson–Smyth adjustment:

$$
p = \frac{b + 1}{m + 1}
$$

where:
- $b$ = number of permuted statistics ≥ observed statistic
- $m$ = number of permutations

This avoids zero $$p$$-values in Monte Carlo testing.

### 3. Effect Size

Effect size is computed as Hedges’ $$g$$, based on the heteroscedastic standardized mean difference:

$$
d_{av} = \frac{\bar{x}_1 - \bar{x}_0}
{\sqrt{\frac{s_0^2 + s_1^2}{2}}}
$$

Hedges’ correction:

$$
g = J \cdot d_{av}$$ Where $$J = 1 - \frac{3}{4(n_0 + n_1 - 2) - 1}
$$

This provides a small-sample bias-corrected standardized mean difference consistent with Welch’s test.

### 4. Confidence Intervals

Confidence intervals for Hedges’ $$g$$ are computed via bootstrap:
- Resampling is performed independently within each group.
- Percentile-based confidence intervals are reported.
- Default: 2000 bootstrap resamples, 95% CI.

This avoids reliance on normal approximation assumptions.

### 5. Missing Data Handling

Default: `nan_policy="omit"`

Per-feature omission (pairwise deletion):
- For each feature, only subjects with valid values are used.
- No mean imputation is performed.
- Effective group sizes may vary across features.


## Input Format

The input CSV must contain:
- Column 0 → group labels (exactly two groups)
- Columns 1..N → numeric feature values


## Output

The output CSV contains:
| # | t_obs_welch | df_welch | p_uncorrected | p_corr_tmax | hedges_g_av | hedges_g_ci_low | hedges_g_ci_high | Sign |

Where:
- `#`: feature name
- `t_obs_welch`: observed Welch t-statistic
- `df_welch`: Welch–Satterthwaite degrees of freedom
- `p_corr_tmax`: FWER-corrected permutation $$p$$-value
- `hedges_g_av`: heteroscedastic Hedges’ $$g$$
- `hedges_g_ci_low/high`: bootstrap confidence interval
- `Sign`: direction of effect

Additionally, the total number of unique label permutations is printed at runtime.

> **Note on $p$-values:** `p_uncorrected` is derived from the parametric Welch t-distribution
> (per-feature, independently). `p_corr_tmax` is fully nonparametric, derived from the
> permutation-based Max-T null distribution. These are intentionally different: the former
> provides a standard parametric reference, while the latter provides FWER-controlled
> nonparametric inference. The corrected value is the primary inferential output.



## Intended Use Cases

### 1. Primary use case: custom feature matrices

This implementation is specifically designed for researchers who extract scalar features
from signals and require simultaneous two-group testing across all features with FWER control.

Supported feature types include, but are not limited to:

- **Dynamical time-series features**: Higuchi fractal dimension, Lyapunov exponents,
  approximate entropy, sample entropy, permutation entropy
- **Spectral features**: absolute or relative band power, spectral edge frequency,
  peak frequency, spectral entropy
- **Connectivity measures**: coherence, phase-locking value, or any pairwise scalar summary
- **Model-derived features**: latent representations, classification scores, regression coefficients
- **Morphological features**: peak amplitude, latency, area under curve per condition

### 2. General applicability

This implementation is also suitable for:

- EEG / MEG mass univariate analysis across channels or time–frequency bins
- Neuroimaging ROI-wise testing
- Multichannel biological signals
- Any two-group, high-dimensional comparison where FWER control is required


## Dependencies

- Python ≥ 3.8
- NumPy
- Pandas
- SciPy


## References

- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples
  and Methods for p-Value Adjustment*. Wiley Series in Probability and Statistics.
- Nichols, T. E., & Holmes, A. P. (2001). Nonparametric permutation tests for functional
  neuroimaging: A primer with examples. *Human Brain Mapping*, 15(1), 1–25. https://doi.org/10.1002/hbm.1058 
- Phipson, B., & Smyth, G. K. (2010). Permutation P-values Should Never Be Zero: Calculating
  Exact P-values When Permutations Are Randomly Drawn. *Statistical Applications in Genetics
  and Molecular Biology*, 9(1). https://doi.org/10.2202/1544-6115.1585 
- Delacre, M., Lakens, D., & Leys, C. (2017). Why Psychologists Should by Default Use
  Welch’s t-test Instead of Student’s t-test. *International Review of Social Psychology*,
  30(1), 92–101. https://doi.org/10.5334/irsp.82 
- Hedges, L. V. (1981). Distribution Theory for Glass’s Estimator of Effect size and Related
  Estimators. *Journal of Educational Statistics*, 6(2), 107–128. https://doi.org/10.3102/10769986006002107 


## Authors

- **Yousef Dehghan**
