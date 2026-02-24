# 00_harmonize_taxonomy.py
# By Carter Clinton, Ph.D.
"""
Harmonize taxonomy across Greengenes (Root;p__...) and SILVA/QIIME2 (d__;p__...) formats
in the merged genus table.

The merged_L6genus.tsv contains taxa from two naming schemes:
  - QIIME2/SILVA: d__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus
  - Greengenes:   Root;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Veillonellaceae;g__Acidaminococcus

Even the same genus can have different intermediate taxonomy (class, order, family)
between SILVA and Greengenes due to reclassification. Therefore we merge by GENUS NAME
rather than by full taxonomy path.

Strategy:
  1. Extract genus name from each taxon (the g__ level, or last level)
  2. For taxa with the same genus name across both databases, SUM their counts
  3. Keep the SILVA-format taxonomy string as the canonical ID (since burial/EMP use SILVA)
  4. For taxa unique to one database, keep them as-is

Output: results/compositional/merged_L6genus_harmonized.tsv
"""

import argparse
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def extract_genus(taxon_str):
    """Extract the genus name from a taxonomy string.

    Returns (genus_name, has_genus_prefix) where genus_name is the bare name
    (e.g., "Bacillus") and has_genus_prefix indicates if it had g__ prefix.
    """
    parts = [p.strip() for p in taxon_str.split(";")]
    last = parts[-1]

    if last.startswith("g__"):
        name = last[3:]  # strip g__ prefix
        if name and name not in ("", "__"):
            return name
    elif last.startswith("f__") or last.startswith("o__") or last.startswith("c__") or last.startswith("p__"):
        # Higher-level classification with empty genus
        return None
    elif last in ("", "__", "Root", "Unassigned"):
        return None
    else:
        # No prefix — might be a bare genus or family-level
        if last and last not in ("__",):
            return last

    return None


def get_database_prefix(taxon_str):
    """Determine if a taxon is from SILVA (d__) or Greengenes (Root;)."""
    if taxon_str.startswith("d__"):
        return "silva"
    elif taxon_str.startswith("Root"):
        return "greengenes"
    elif taxon_str == "Unassigned":
        return "unassigned"
    else:
        return "unknown"


def choose_canonical_taxonomy(taxon_list):
    """Choose the SILVA taxonomy as canonical (since burial/EMP use SILVA).

    If no SILVA entry exists, use the Greengenes one.
    """
    silva = [t for t in taxon_list if t.startswith("d__")]
    gg = [t for t in taxon_list if t.startswith("Root")]
    other = [t for t in taxon_list if not t.startswith("d__") and not t.startswith("Root")]

    # Prefer SILVA
    if silva:
        return silva[0]
    elif gg:
        return gg[0]
    elif other:
        return other[0]
    return taxon_list[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/compositional/merged_L6genus_original.tsv",
                    help="Input merged genus table (original, with SILVA + GG taxa)")
    ap.add_argument("--output", default="results/compositional/merged_L6genus_harmonized.tsv",
                    help="Output harmonized genus table")
    ap.add_argument("--mapping", default="results/compositional/taxonomy_mapping.tsv",
                    help="Output: mapping from original to canonical taxonomy")
    args = ap.parse_args()

    print("Loading genus table...")
    # Read with comment skipping
    with open(args.input) as f:
        skip_lines = 0
        for line in f:
            if line.startswith("#") and not line.startswith("#OTU") and not line.startswith("#Feature"):
                skip_lines += 1
                continue
            break

    df = pd.read_csv(args.input, sep="\t", skiprows=skip_lines, dtype={0: str})
    taxon_col = df.columns[0]
    print(f"  {len(df)} taxa x {len(df.columns) - 1} samples")

    # Check for taxonomy column at end
    if df.columns[-1].lower() == "taxonomy":
        df = df.drop(columns=[df.columns[-1]])

    # Extract genus from each taxon
    print("Extracting genus names...")
    df["_genus"] = df[taxon_col].apply(extract_genus)
    df["_db"] = df[taxon_col].apply(get_database_prefix)

    n_with_genus = df["_genus"].notna().sum()
    n_without = df["_genus"].isna().sum()
    print(f"  Taxa with named genus: {n_with_genus}")
    print(f"  Taxa without named genus (higher-level): {n_without}")

    # Find genera that appear in BOTH databases
    genus_to_taxa = defaultdict(list)
    for _, row in df.iterrows():
        genus = row["_genus"]
        if genus and isinstance(genus, str) and genus.strip():
            genus_to_taxa[genus].append(row[taxon_col])

    cross_db = {}
    for genus, taxa_list in genus_to_taxa.items():
        dbs = set(get_database_prefix(t) for t in taxa_list)
        if "silva" in dbs and "greengenes" in dbs:
            cross_db[genus] = taxa_list

    print(f"\n  Genera present in BOTH SILVA and Greengenes: {len(cross_db)}")
    for g in sorted(cross_db.keys())[:5]:
        ts = cross_db[g]
        print(f"    {g}: {len(ts)} entries → will merge")

    # Also handle higher-level taxa (no genus) by normalizing their path
    # Strip domain/Root prefix and normalize phylum names, then merge if paths match
    phylum_map = {
        "Acidobacteriota": "Acidobacteria",
        "Actinobacteriota": "Actinobacteria",
        "Bacteroidota": "Bacteroidetes",
        "Chloroflexota": "Chloroflexi",
        "Cyanobacteriota": "Cyanobacteria",
        "Firmicutes": "Firmicutes",
        "Fusobacteriota": "Fusobacteria",
        "Gemmatimonadota": "Gemmatimonadetes",
        "Planctomycetota": "Planctomycetes",
        "Proteobacteria": "Proteobacteria",
        "Spirochaetota": "Spirochaetes",
        "Verrucomicrobiota": "Verrucomicrobia",
    }

    def normalize_path(taxon_str):
        """Normalize a full taxonomy path by stripping domain and standardizing phylum."""
        parts = [p.strip() for p in taxon_str.split(";")]
        if parts[0].startswith("d__") or parts[0].startswith("k__") or parts[0] == "Root":
            parts = parts[1:]
        # Normalize phylum
        for i, p in enumerate(parts):
            if p.startswith("p__"):
                pname = p[3:]
                if pname in phylum_map:
                    parts[i] = f"p__{phylum_map[pname]}"
        return ";".join(parts) if parts else ""

    # Group higher-level taxa by normalized path
    higher_taxa = df[df["_genus"].isna()]
    path_to_taxa = defaultdict(list)
    for _, row in higher_taxa.iterrows():
        norm = normalize_path(row[taxon_col])
        if norm:
            path_to_taxa[norm].append(row[taxon_col])

    higher_cross_db = {}
    for path, taxa_list in path_to_taxa.items():
        dbs = set(get_database_prefix(t) for t in taxa_list)
        if len(taxa_list) > 1:
            higher_cross_db[path] = taxa_list

    print(f"  Higher-level taxa merged by path: {len(higher_cross_db)}")

    # Build merge mapping: assign each taxon a canonical ID
    # For cross-database genera, merge under the SILVA taxonomy
    # For same-database genera with multiple entries, also merge
    taxon_to_canonical = {}
    for _, row in df.iterrows():
        taxon = row[taxon_col]
        genus = row["_genus"]

        if genus and isinstance(genus, str) and genus in cross_db:
            # Merge under canonical (SILVA) taxonomy
            taxon_to_canonical[taxon] = choose_canonical_taxonomy(cross_db[genus])
        elif genus and isinstance(genus, str) and len(genus_to_taxa.get(genus, [])) > 1:
            # Same genus name, same database — still merge
            taxon_to_canonical[taxon] = choose_canonical_taxonomy(genus_to_taxa[genus])
        else:
            # Check higher-level merge
            norm = normalize_path(taxon)
            if norm in higher_cross_db:
                taxon_to_canonical[taxon] = choose_canonical_taxonomy(higher_cross_db[norm])
            else:
                # Unique taxon — keep as-is
                taxon_to_canonical[taxon] = taxon

    df["_canonical"] = df[taxon_col].map(taxon_to_canonical)

    # Count merges
    n_unique_canonical = df["_canonical"].nunique()
    n_merged_genera = len(df) - n_unique_canonical
    print(f"\n  {len(df)} original taxa → {n_unique_canonical} canonical taxa")
    print(f"  {n_merged_genera} taxa will be merged into existing rows")

    # Save mapping
    mapping = df[[taxon_col, "_canonical", "_genus", "_db"]].copy()
    mapping.columns = ["original", "canonical", "genus", "database"]
    mapping.to_csv(args.mapping, sep="\t", index=False)
    print(f"  Wrote taxonomy mapping: {args.mapping}")

    # Group by canonical taxonomy and sum counts
    print("\nMerging rows...")
    sample_cols = [c for c in df.columns if c != taxon_col and not c.startswith("_")]
    for col in sample_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = df.groupby("_canonical")[sample_cols].sum()
    grouped.index.name = taxon_col
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={grouped.columns[0]: taxon_col})

    print(f"  Result: {len(grouped)} taxa x {len(sample_cols)} samples")

    # Verify: check overlap between burial and HMP samples
    try:
        meta = pd.read_csv("results/metadata_cohort_split.tsv", sep="\t", dtype={0: str})
        meta.columns = ["sample_id"] + list(meta.columns[1:])
        hmp_ids = set(meta.loc[meta["cohort"] == "hmp", "sample_id"])
        burial_ids = set(meta.loc[meta["cohort"] == "burial", "sample_id"])
        emp_ids = set(meta.loc[meta["cohort"] == "emp", "sample_id"])

        hmp_cols = [c for c in sample_cols if c in hmp_ids]
        burial_cols = [c for c in sample_cols if c in burial_ids]
        emp_cols = [c for c in sample_cols if c in emp_ids]

        if hmp_cols and burial_cols:
            hmp_present = set(grouped.loc[grouped[hmp_cols].sum(axis=1) > 0, taxon_col])
            burial_present = set(grouped.loc[grouped[burial_cols].sum(axis=1) > 0, taxon_col])
            emp_present = set(grouped.loc[grouped[emp_cols].sum(axis=1) > 0, taxon_col])
            print(f"\n  Sanity check after harmonization:")
            print(f"    HMP taxa (>0):    {len(hmp_present)}")
            print(f"    EMP taxa (>0):    {len(emp_present)}")
            print(f"    Burial taxa (>0): {len(burial_present)}")
            print(f"    HMP ∩ Burial:     {len(hmp_present & burial_present)}")
            print(f"    EMP ∩ Burial:     {len(emp_present & burial_present)}")
            print(f"    HMP ∩ EMP:        {len(hmp_present & emp_present)}")
    except Exception as e:
        print(f"  Could not verify: {e}")

    # Write output
    grouped.to_csv(args.output, sep="\t", index=False)
    print(f"\nWrote harmonized table: {args.output}")


if __name__ == "__main__":
    main()
