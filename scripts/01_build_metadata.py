# 01_build_metadata.py
# By Carter Clinton, Ph.D.
"""
Build Split Metadata

Reads NYABG metadata and the current master metadata, produces split cohort files
that separate burial and control samples. Also adds HMP body-site and EMP environment-
type subsite labels for downstream stratified analyses.

Outputs:
  - <out-dir>/metadata_cohort_split.tsv (full split metadata)
  - <out-dir>/metadata_burial_only.tsv (burial + emp + hmp)
  - <out-dir>/metadata_control_only.tsv (control + emp + hmp)
  - <out-dir>/metadata_subsite_map.tsv (lookup: subsite labels and sample counts)
"""

import argparse
import os
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv",
                   help="NYABG sample metadata (QIIME2 format)")
    p.add_argument("--master-meta", default="results/metadata_cohort.prev.tsv",
                   help="Master metadata with sample-id and cohort columns")
    p.add_argument("--hmp-meta", default="data/hmp/hmp_metadata.tsv",
                   help="HMP metadata with body-site labels")
    p.add_argument("--emp-meta", default="data/emp/emp_metadata.tsv",
                   help="EMP metadata with EMPO labels")
    p.add_argument("--out-dir", default="results/",
                   help="Output directory")
    return p.parse_args()


# Lab/QC controls to exclude entirely
LAB_QC_IDS = {"S93", "S91", "Water-neg-plateD", "Water-neg-TD-plateD",
              "Zymo-mock-pos-plateD", "Zymo-mock-pos-TD-plateD", "PC"}
# Edge cases to exclude
EDGE_CASE_IDS = {"NA", "MISC"}


def classify_nyabg(row):
    sid = str(row["sample-id"])
    bc = str(row.get("BurialControl", ""))
    br = str(row.get("Body_Region", ""))

    if sid in LAB_QC_IDS or sid in EDGE_CASE_IDS:
        return "exclude"
    if bc == "Burial":
        return "burial"
    if bc == "Control" and br == "Control":
        return "control"
    return "exclude"


# Map HMP body subsites to 5 groups
HMP_SITE_MAP = {
    "Attached Keratinized Gingiva": "hmp_oral",
    "Buccal Mucosa": "hmp_oral",
    "Hard Palate": "hmp_oral",
    "Palatine Tonsils": "hmp_oral",
    "Saliva": "hmp_oral",
    "Subgingival Plaque": "hmp_oral",
    "Supragingival Plaque": "hmp_oral",
    "Throat": "hmp_oral",
    "Tongue Dorsum": "hmp_oral",
    "Left Antecubital Fossa": "hmp_skin",
    "Right Antecubital Fossa": "hmp_skin",
    "Left Retroauricular Crease": "hmp_skin",
    "Right Retroauricular Crease": "hmp_skin",
    "Stool": "hmp_stool",
    "Mid Vagina": "hmp_vaginal",
    "Posterior Fornix": "hmp_vaginal",
    "Vaginal Introitus": "hmp_vaginal",
    "Anterior Nares": "hmp_nasal",
}

# Map EMPO_3 to simplified environment types
EMPO_MAP = {
    "Soil (non-saline)": "emp_soil",
    "Water (non-saline)": "emp_water",
    "Water (saline)": "emp_water",
    "Sediment (non-saline)": "emp_sediment",
    "Sediment (saline)": "emp_sediment",
    "Animal distal gut": "emp_animal",
    "Animal proximal gut": "emp_animal",
    "Animal corpus": "emp_animal",
    "Animal surface": "emp_animal",
    "Animal secretion": "emp_animal",
    "Plant surface": "emp_plant",
    "Plant rhizosphere": "emp_plant",
    "Plant corpus": "emp_plant",
    "Surface (non-saline)": "emp_other",
    "Surface (saline)": "emp_other",
    "Aerosol (non-saline)": "emp_other",
    "Hypersaline (saline)": "emp_other",
    "Mock community": "emp_exclude",
    "Sterile water blank": "emp_exclude",
}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Read NYABG metadata ─────────────────────────────────────────────
    # QIIME2 format: first line is #SampleID header, second line is #q2:types
    with open(args.nyabg_meta) as f:
        header_line = f.readline().strip().rstrip("\r")
    header_cols = [c.strip() for c in header_line.lstrip("#").split("\t") if c.strip()]
    n_cols = len(header_cols)

    nyabg_meta = pd.read_csv(
        args.nyabg_meta, sep="\t", skiprows=2, header=None,
        usecols=range(n_cols), names=header_cols, dtype=str, keep_default_na=False
    )
    nyabg_meta.rename(columns={nyabg_meta.columns[0]: "sample-id"}, inplace=True)

    # ── 2. Classify each NYABG sample ─────────────────────────────────────
    nyabg_meta["split_cohort"] = nyabg_meta.apply(classify_nyabg, axis=1)

    nyabg_lookup = nyabg_meta[nyabg_meta["split_cohort"] != "exclude"].set_index("sample-id")[
        ["split_cohort", "Body_Region"]
    ].to_dict("index")

    print(f"NYABG classification: "
          f"{sum(1 for v in nyabg_lookup.values() if v['split_cohort']=='burial')} burial, "
          f"{sum(1 for v in nyabg_lookup.values() if v['split_cohort']=='control')} control, "
          f"{sum(1 for r in nyabg_meta['split_cohort'] if r=='exclude')} excluded")

    # ── 3. Read HMP metadata, assign body-site subgroups ──────────────────
    hmp_meta = pd.read_csv(
        args.hmp_meta, sep="\t", dtype=str, keep_default_na=False
    )
    hmp_meta.rename(columns={hmp_meta.columns[0]: "sample-id"}, inplace=True)

    hmp_subsite = {}
    for _, row in hmp_meta.iterrows():
        sid = row["sample-id"]
        subsite = HMP_SITE_MAP.get(row.get("hmp_body_subsite", ""), "hmp_other")
        hmp_subsite[sid] = subsite

    # ── 4. Read EMP metadata, assign environment-type subgroups ───────────
    emp_meta = pd.read_csv(
        args.emp_meta, sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    emp_meta.rename(columns={emp_meta.columns[0]: "sample-id"}, inplace=True)

    emp_subsite = {}
    for _, row in emp_meta.iterrows():
        sid = row["sample-id"]
        empo3 = row.get("empo_3", "")
        subsite = EMPO_MAP.get(empo3, "emp_other")
        if subsite != "emp_exclude":
            emp_subsite[sid] = subsite

    # ── 5. Build the split metadata ───────────────────────────────────────
    master = pd.read_csv(args.master_meta, sep="\t", dtype=str)

    rows = []
    for _, row in master.iterrows():
        sid = row["sample-id"]
        old_cohort = row["cohort"]

        if old_cohort == "nyabg":
            if sid in nyabg_lookup:
                info = nyabg_lookup[sid]
                cohort = info["split_cohort"]
                body_region = info["Body_Region"]
                cohort_subsite = cohort  # burial/control (no further subsite)
            else:
                continue  # excluded NYABG sample
        elif old_cohort == "hmp":
            cohort = "hmp"
            body_region = ""
            cohort_subsite = hmp_subsite.get(sid, "hmp_other")
        elif old_cohort == "emp":
            cohort = "emp"
            body_region = ""
            cohort_subsite = emp_subsite.get(sid, "emp_other")
        else:
            continue

        rows.append({
            "sample-id": sid,
            "cohort": cohort,
            "cohort_subsite": cohort_subsite,
            "body_region": body_region,
        })

    split_df = pd.DataFrame(rows)

    # ── 6. Write outputs ──────────────────────────────────────────────────
    split_df.to_csv(os.path.join(args.out_dir, "metadata_cohort_split.tsv"), sep="\t", index=False)
    print(f"Wrote metadata_cohort_split.tsv: {len(split_df)} rows")

    burial_only = split_df[split_df["cohort"].isin(["burial", "emp", "hmp"])]
    burial_only.to_csv(os.path.join(args.out_dir, "metadata_burial_only.tsv"), sep="\t", index=False)
    print(f"Wrote metadata_burial_only.tsv: {len(burial_only)} rows")

    control_only = split_df[split_df["cohort"].isin(["control", "emp", "hmp"])]
    control_only.to_csv(os.path.join(args.out_dir, "metadata_control_only.tsv"), sep="\t", index=False)
    print(f"Wrote metadata_control_only.tsv: {len(control_only)} rows")

    subsite_counts = split_df.groupby(["cohort", "cohort_subsite"]).size().reset_index(name="n_samples")
    subsite_counts.to_csv(os.path.join(args.out_dir, "metadata_subsite_map.tsv"), sep="\t", index=False)
    print(f"\nSubsite summary:")
    print(subsite_counts.to_string(index=False))

    cohort_counts = split_df["cohort"].value_counts()
    print(f"\nCohort summary:")
    for c, n in cohort_counts.items():
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
