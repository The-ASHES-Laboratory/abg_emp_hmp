# 23_power_analysis.py
# By Carter Clinton, Ph.D.
"""
Expanded Power Analysis

Minimum detectable effect sizes at n=6 controls, alpha=0.05, 80% power.
Also reports power for the observed effects.

Outputs:
  results/power_analysis/power_analysis_expanded.tsv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results/power_analysis/")
    return p.parse_args()


def power_mann_whitney(n1, n2, effect_d, alpha=0.05, n_sim=10000, seed=42):
    """Simulate power for Mann-Whitney U test via simulation."""
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        x = rng.standard_normal(n1)
        y = rng.standard_normal(n2) + effect_d
        _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        if p < alpha:
            rejections += 1
    return rejections / n_sim


def power_fisher_exact(n1, n2, p1, p2, alpha=0.05, n_sim=10000, seed=42):
    """Simulate power for Fisher exact test."""
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        x1 = rng.binomial(n1, p1)
        x2 = rng.binomial(n2, p2)
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        _, p = stats.fisher_exact(table)
        if p < alpha:
            rejections += 1
    return rejections / n_sim


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n_burial = 69
    n_control = 6
    alpha = 0.05
    target_power = 0.80

    results = []

    # 1. Mann-Whitney: minimum detectable Cohen's d at n=6 vs n=69
    print("Simulating power for Mann-Whitney U...")
    for d in np.arange(0.2, 3.0, 0.1):
        power = power_mann_whitney(n_burial, n_control, d, alpha=alpha, n_sim=5000)
        results.append({
            "test": "Mann-Whitney U",
            "parameter": "Cohen_d",
            "value": round(d, 1),
            "power": power,
            "n1": n_burial,
            "n2": n_control,
            "alpha": alpha,
        })
        if power >= target_power:
            print(f"  Min detectable d={d:.1f} at {target_power*100:.0f}% power "
                  f"(n1={n_burial}, n2={n_control})")
            break

    # 2. Fisher exact: minimum detectable difference in proportions
    print("Simulating power for Fisher exact...")
    # Burial detection = 45%, what control rate can we distinguish?
    p_burial = 0.45
    for p_diff in np.arange(0.1, 0.9, 0.05):
        p_ctrl = max(0, p_burial - p_diff)
        power = power_fisher_exact(n_burial, n_control, p_burial, p_ctrl,
                                    alpha=alpha, n_sim=5000)
        results.append({
            "test": "Fisher exact",
            "parameter": "proportion_difference",
            "value": round(p_diff, 2),
            "power": power,
            "n1": n_burial,
            "n2": n_control,
            "alpha": alpha,
        })
        if power >= target_power:
            print(f"  Min detectable diff={p_diff:.2f} at {target_power*100:.0f}% power")
            break

    # 3. Observed effect: cranium (n=15) vs non-cranium (n=54)
    print("Power for cranium analysis...")
    for d in np.arange(0.2, 2.0, 0.1):
        power = power_mann_whitney(15, 54, d, alpha=alpha, n_sim=5000)
        results.append({
            "test": "Mann-Whitney U (cranium)",
            "parameter": "Cohen_d",
            "value": round(d, 1),
            "power": power,
            "n1": 15,
            "n2": 54,
            "alpha": alpha,
        })

    # 4. Burial-level (n=47 burials vs n=6 controls)
    print("Power at burial level (pseudoreplication corrected)...")
    for d in np.arange(0.2, 3.0, 0.1):
        power = power_mann_whitney(47, n_control, d, alpha=alpha, n_sim=5000)
        results.append({
            "test": "Mann-Whitney U (burial-level)",
            "parameter": "Cohen_d",
            "value": round(d, 1),
            "power": power,
            "n1": 47,
            "n2": n_control,
            "alpha": alpha,
        })
        if power >= target_power:
            print(f"  Burial-level min detectable d={d:.1f}")
            break

    # 5. Subadult age effect (n~12 subadult vs n~30 adult, estimated)
    print("Power for subadult age effect...")
    for d in np.arange(0.2, 3.0, 0.1):
        power = power_mann_whitney(12, 30, d, alpha=alpha, n_sim=5000)
        results.append({
            "test": "Mann-Whitney U (age effect)",
            "parameter": "Cohen_d",
            "value": round(d, 1),
            "power": power,
            "n1": 12,
            "n2": 30,
            "alpha": alpha,
        })

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.out_dir, "power_analysis_expanded.tsv")
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path} ({len(results_df)} rows)")

    # Report minimum detectable effects at 80% power
    print("\n=== Minimum Detectable Effects (80% power) ===")
    for test_name in results_df["test"].unique():
        test_data = results_df[results_df["test"] == test_name]
        above_80 = test_data[test_data["power"] >= target_power]
        if len(above_80) > 0:
            min_val = above_80["value"].min()
            print(f"  {test_name}: {above_80.iloc[0]['parameter']} >= {min_val}")
        else:
            max_power = test_data["power"].max()
            print(f"  {test_name}: Max power={max_power:.2f} (underpowered)")


if __name__ == "__main__":
    main()
