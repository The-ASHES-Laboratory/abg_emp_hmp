# 26_compare_networks.py
# By Carter Clinton, Ph.D.
"""
Network Comparison

For each taxon present in both burial and control networks:
- Track assignment shift (emp→hmp, hmp→emp, stable_emp, stable_hmp, stable_ambiguous)
- Compute IndVal difference (burial IndVal_HMP - control IndVal_HMP)
- Identify burial-specific HMP indicator taxa
- Fisher's exact test on 2x2 table of assignments
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--burial-taxa", default="results/networks/burial/overall/source_network_taxa.tsv")
    ap.add_argument("--control-taxa", default="results/networks/control/overall/source_network_taxa.tsv")
    ap.add_argument("--burial-summary", default="results/networks/burial/overall/assignment_summary.tsv")
    ap.add_argument("--control-summary", default="results/networks/control/overall/assignment_summary.tsv")
    ap.add_argument("--out-dir", default="results/comparison/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load taxa tables
    print("Loading network taxa tables...")
    b_taxa = pd.read_csv(args.burial_taxa, sep="\t")
    c_taxa = pd.read_csv(args.control_taxa, sep="\t")

    key_col = "taxon_key" if "taxon_key" in b_taxa.columns else b_taxa.columns[0]

    # Identify assignment and IndVal columns
    assign_col = "assignment" if "assignment" in b_taxa.columns else None
    indval_hmp_cols = [c for c in b_taxa.columns if "indval" in c.lower() and "hmp" in c.lower()]
    indval_emp_cols = [c for c in b_taxa.columns if "indval" in c.lower() and "emp" in c.lower()]

    if not assign_col:
        print("Warning: no 'assignment' column found in taxa table")
        return

    # Merge on taxon key
    merged = b_taxa[[key_col, assign_col] + indval_hmp_cols + indval_emp_cols].merge(
        c_taxa[[key_col, assign_col] + indval_hmp_cols + indval_emp_cols],
        on=key_col, suffixes=("_burial", "_control"),
        how="outer"
    )
    print(f"  {len(merged)} total taxa ({len(b_taxa)} burial, {len(c_taxa)} control)")

    # Classify shifts
    rows = []
    shift_counts = {}
    burial_specific_hmp = []

    for _, row in merged.iterrows():
        taxon = row[key_col]
        b_assign = str(row.get(f"{assign_col}_burial", "")).strip().lower()
        c_assign = str(row.get(f"{assign_col}_control", "")).strip().lower()

        # Determine shift category
        if b_assign == "nan" or b_assign == "":
            shift = "burial_only"
        elif c_assign == "nan" or c_assign == "":
            shift = "control_only"
        elif b_assign == c_assign:
            shift = f"stable_{b_assign}"
        else:
            shift = f"{c_assign}_to_{b_assign}"

        shift_counts[shift] = shift_counts.get(shift, 0) + 1

        # IndVal difference (burial - control)
        indval_diff = np.nan
        b_indval = np.nan
        c_indval = np.nan
        if indval_hmp_cols:
            col = indval_hmp_cols[0]
            b_indval = row.get(f"{col}_burial", np.nan)
            c_indval = row.get(f"{col}_control", np.nan)
            try:
                indval_diff = float(b_indval) - float(c_indval)
            except (TypeError, ValueError):
                pass

        # Burial-specific HMP indicator
        is_burial_hmp = (b_assign == "hmp" and c_assign in ("emp", "ambiguous", "nan", ""))

        if is_burial_hmp:
            burial_specific_hmp.append(taxon)

        rows.append({
            "taxon_key": taxon,
            "burial_assignment": b_assign if b_assign != "nan" else "",
            "control_assignment": c_assign if c_assign != "nan" else "",
            "shift": shift,
            "indval_hmp_burial": b_indval if str(b_indval) != "nan" else "",
            "indval_hmp_control": c_indval if str(c_indval) != "nan" else "",
            "indval_hmp_diff": f"{indval_diff:.6g}" if not np.isnan(indval_diff) else "",
            "burial_specific_hmp": is_burial_hmp,
        })

    # Write comparison table
    out_path = os.path.join(args.out_dir, "network_comparison_taxa.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {out_path} ({len(rows)} taxa)")

    # Write shift summary
    shift_path = os.path.join(args.out_dir, "network_shift_summary.tsv")
    with open(shift_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["shift_category", "n_taxa"])
        for shift, count in sorted(shift_counts.items(), key=lambda x: -x[1]):
            writer.writerow([shift, count])
    print(f"Wrote {shift_path}")

    # Shift counts summary
    for shift, count in sorted(shift_counts.items(), key=lambda x: -x[1]):
        print(f"  {shift}: {count}")

    # Burial-specific HMP indicator taxa
    hmp_path = os.path.join(args.out_dir, "burial_specific_hmp_taxa.tsv")
    with open(hmp_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["taxon_key"])
        for t in sorted(burial_specific_hmp):
            writer.writerow([t])
    print(f"\nBurial-specific HMP indicator taxa: {len(burial_specific_hmp)}")

    # Fisher's exact test on 2x2 assignment table
    # Only for taxa present in BOTH networks
    both = merged.dropna(subset=[f"{assign_col}_burial", f"{assign_col}_control"])
    b_emp = (both[f"{assign_col}_burial"] == "emp").sum()
    b_hmp = (both[f"{assign_col}_burial"] == "hmp").sum()
    c_emp = (both[f"{assign_col}_control"] == "emp").sum()
    c_hmp = (both[f"{assign_col}_control"] == "hmp").sum()

    if b_emp + b_hmp > 0 and c_emp + c_hmp > 0:
        table = np.array([[b_emp, b_hmp], [c_emp, c_hmp]])
        odds, p = stats.fisher_exact(table, alternative="two-sided")
        print(f"\nFisher's exact: burial[emp={b_emp}, hmp={b_hmp}] vs control[emp={c_emp}, hmp={c_hmp}]")
        print(f"  OR={odds:.4f}, p={p:.4g}")

        fisher_path = os.path.join(args.out_dir, "network_fisher_test.tsv")
        with open(fisher_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["burial_emp", "burial_hmp", "control_emp", "control_hmp",
                             "odds_ratio", "p_value"])
            writer.writerow([b_emp, b_hmp, c_emp, c_hmp, f"{odds:.6g}", f"{p:.6g}"])


if __name__ == "__main__":
    main()
