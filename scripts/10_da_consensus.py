# 10_da_consensus.py
# By Carter Clinton, Ph.D.
"""
Cross-method differential abundance consensus.

Identifies taxa flagged as significant by ANCOMBC2 AND ALDEx2 for each
pairwise comparison, creating a high-confidence DA consensus set.

Outputs:
  results/diff_abun/da_consensus.tsv         — merged table
  results/diff_abun/da_consensus_summary.tsv  — summary counts
"""

import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ancombc2-dir", default="results/diff_abun/ancombc2_split/")
    p.add_argument("--aldex2-dir", default="results/diff_abun/aldex2/")
    p.add_argument("--out-dir", default="results/diff_abun/")
    return p.parse_args()


def load_ancombc2(path):
    """Load ANCOMBC2 result; return significant taxa."""
    df = pd.read_csv(path, sep="\t")
    # Column naming: q_cohort<group1> or diff_cohort<group1>
    # Find the q-value column for the comparison
    q_cols = [c for c in df.columns if c.startswith("q_") and c != "q_(Intercept)"]
    diff_cols = [c for c in df.columns if c.startswith("diff_") and c != "diff_(Intercept)"]
    lfc_cols = [c for c in df.columns if c.startswith("lfc_") and c != "lfc_(Intercept)"]

    if not q_cols:
        return pd.DataFrame()

    q_col = q_cols[0]
    diff_col = diff_cols[0] if diff_cols else None
    lfc_col = lfc_cols[0] if lfc_cols else None

    sig = df[df[q_col] < 0.05].copy()
    sig = sig.rename(columns={"taxon": "taxon"})
    if lfc_col:
        sig["ancombc2_lfc"] = sig[lfc_col]
    sig["ancombc2_q"] = sig[q_col]
    return sig[["taxon", "ancombc2_lfc", "ancombc2_q"]].copy()


def load_aldex2(path):
    """Load ALDEx2 result; return significant taxa."""
    df = pd.read_csv(path, sep="\t")
    # ALDEx2 output: row names as first col (unnamed) + may have trailing 'taxon' col
    # Drop any duplicate 'taxon' columns to avoid merge errors
    if df.columns.tolist().count("taxon") > 1:
        # Keep only the first occurrence
        cols = df.columns.tolist()
        seen = set()
        keep = []
        for i, c in enumerate(cols):
            if c == "taxon" and c in seen:
                keep.append(False)
            else:
                keep.append(True)
                seen.add(c)
        df = df.loc[:, keep]
    # If first column has the taxa (unnamed index), rename it
    taxon_col = df.columns[0]
    if taxon_col != "taxon":
        # Check if there's a 'taxon' col at the end that's the same
        if "taxon" in df.columns and taxon_col != "taxon":
            # Use the named 'taxon' column if it has actual taxonomy strings
            if df["taxon"].str.contains(";", na=False).any():
                df = df.drop(columns=[taxon_col])
            else:
                df = df.drop(columns=["taxon"])
                df = df.rename(columns={taxon_col: "taxon"})
        else:
            df = df.rename(columns={taxon_col: "taxon"})

    # Check for BH-corrected columns
    q_col = None
    for c in ["wi.eBH", "we.eBH", "wi.ep.adj"]:
        if c in df.columns:
            q_col = c
            break
    if q_col is None:
        # Fall back to raw p
        for c in ["wi.ep", "we.ep"]:
            if c in df.columns:
                q_col = c
                break
    if q_col is None:
        return pd.DataFrame()

    effect_col = "effect" if "effect" in df.columns else None

    sig = df[df[q_col] < 0.05].copy()
    sig["aldex2_q"] = sig[q_col]
    if effect_col:
        sig["aldex2_effect"] = sig[effect_col]
    else:
        sig["aldex2_effect"] = float("nan")
    return sig[["taxon", "aldex2_effect", "aldex2_q"]].copy()


def extract_genus(taxon_str):
    """Extract genus name from full taxonomy string."""
    parts = str(taxon_str).split(";")
    for p in reversed(parts):
        p = p.strip()
        if p.startswith("g__"):
            return p.replace("g__", "")
        if p and not p.startswith("__"):
            return p
    return taxon_str


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    comparisons = [
        "burial_vs_emp",
        "burial_vs_hmp",
        "burial_vs_control",
        "control_vs_emp",
        "control_vs_hmp",
    ]
    levels = ["L5", "L6"]

    all_consensus = []
    summary_rows = []

    for comp in comparisons:
        for level in levels:
            anc_file = os.path.join(args.ancombc2_dir, f"ancombc2_{comp}_{level}.tsv")
            ald_file = os.path.join(args.aldex2_dir, f"aldex2_{comp}_{level}.tsv")

            tag = f"{comp}_{level}"

            if not os.path.exists(anc_file):
                print(f"  {tag}: ANCOMBC2 file missing, skipping")
                summary_rows.append({
                    "comparison": comp, "level": level,
                    "n_ancombc2_sig": 0, "n_aldex2_sig": 0,
                    "n_consensus": 0, "note": "ancombc2 missing",
                })
                continue
            if not os.path.exists(ald_file):
                print(f"  {tag}: ALDEx2 file missing, skipping")
                summary_rows.append({
                    "comparison": comp, "level": level,
                    "n_ancombc2_sig": 0, "n_aldex2_sig": 0,
                    "n_consensus": 0, "note": "aldex2 missing",
                })
                continue

            anc = load_ancombc2(anc_file)
            ald = load_aldex2(ald_file)

            print(f"  {tag}: {len(anc)} ANCOMBC2 sig, {len(ald)} ALDEx2 sig")

            if anc.empty or ald.empty:
                summary_rows.append({
                    "comparison": comp, "level": level,
                    "n_ancombc2_sig": len(anc), "n_aldex2_sig": len(ald),
                    "n_consensus": 0, "note": "one method has 0 sig",
                })
                continue

            # Merge on taxon
            merged = anc.merge(ald, on="taxon", how="inner")
            print(f"    → {len(merged)} consensus taxa")

            if len(merged) > 0:
                merged["comparison"] = comp
                merged["level"] = level
                merged["genus"] = merged["taxon"].apply(extract_genus)
                all_consensus.append(merged)

            summary_rows.append({
                "comparison": comp, "level": level,
                "n_ancombc2_sig": len(anc), "n_aldex2_sig": len(ald),
                "n_consensus": len(merged), "note": "",
            })

    # Write consensus
    if all_consensus:
        consensus_df = pd.concat(all_consensus, ignore_index=True)
        out = os.path.join(args.out_dir, "da_consensus.tsv")
        consensus_df.to_csv(out, sep="\t", index=False)
        print(f"\nWrote {out} ({len(consensus_df)} rows)")
    else:
        print("\nNo consensus taxa found across any comparison.")

    # Write summary
    summary_df = pd.DataFrame(summary_rows)
    out_s = os.path.join(args.out_dir, "da_consensus_summary.tsv")
    summary_df.to_csv(out_s, sep="\t", index=False)
    print(f"Wrote {out_s}")


if __name__ == "__main__":
    main()
