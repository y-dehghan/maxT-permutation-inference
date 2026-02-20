
from max_t_permutation_test import maxT

results, table, sig_mask, sig_feats, raw = maxT.max_t_permutation_test(
"sample_data.csv",  # "your_data.csv"
n_permutations=5000,
tail=0,
alpha=0.05,
nan_policy="omit",
n_bootstrap=2000,
ci_level=0.95,
)

maxT.save_max_t_results(table, "max_t_results.csv")

maxT.save_significant_feature_values(raw, sig_feats, "significant_feature_values.csv", include_label=True)