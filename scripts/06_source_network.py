# 06_source_network.py
# By Carter Clinton, Ph.D.
"""
Build a weighted source-taxon network for dynamic cohort combinations.

Generalizes the original EMP/HMP/NYABG-only network to handle arbitrary cohort
labels (e.g., emp, hmp, burial, control, hmp_oral, emp_soil) via CLI flags.

Reads `cohort` and `cohort_subsite` columns from split metadata to support
both overall and subsite-stratified runs.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default="results/compositional/merged_L6genus.tsv",
        help="Merged L6 genus table (TSV, taxon + samples)",
    )
    parser.add_argument(
        "--metadata",
        default="results/metadata_cohort_split.tsv",
        help="Metadata TSV with sample-id, cohort, cohort_subsite columns",
    )
    parser.add_argument(
        "--cohorts",
        required=True,
        help="Comma-separated cohort labels to include (e.g., emp,hmp,burial)",
    )
    parser.add_argument(
        "--source-cohorts",
        required=True,
        help="Comma-separated reference cohorts (e.g., emp,hmp)",
    )
    parser.add_argument(
        "--target-cohort",
        required=True,
        help="Target cohort being attributed (e.g., burial or control)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/networks",
        help="Output directory",
    )
    parser.add_argument(
        "--key",
        choices=["taxon", "genus"],
        default="genus",
        help="Aggregate by full taxon or genus name (default: genus)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N indicator taxa per source for co-occurrence scoring",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help="Minimum IndVal delta to prefer a source",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.2,
        help="Minimum IndVal ratio to prefer a source",
    )
    parser.add_argument(
        "--cooccur-cohort",
        default=None,
        help="Cohort scope for co-occurrence preference (default: target cohort)",
    )
    return parser.parse_args()


def read_metadata(meta_path, valid_cohorts):
    """Read metadata, mapping sample-id to cohort label.

    Matches against both the `cohort` and `cohort_subsite` columns so that
    labels like 'hmp_oral' or 'emp_soil' are correctly resolved.
    """
    cohort_map = {}
    with open(meta_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sample = row.get("sample-id") or row.get("sample_id") or row.get("sample")
            cohort = (row.get("cohort") or "").strip().lower()
            subsite = (row.get("cohort_subsite") or "").strip().lower()
            if not sample:
                continue
            # Check subsite first (more specific), then cohort
            if subsite in valid_cohorts:
                cohort_map[str(sample)] = subsite
            elif cohort in valid_cohorts:
                cohort_map[str(sample)] = cohort
    return cohort_map


def parse_taxonomy(taxon):
    out = {"k": "", "p": "", "c": "", "o": "", "f": "", "g": ""}
    for part in taxon.split(";"):
        part = part.strip()
        if "__" not in part:
            continue
        prefix, value = part.split("__", 1)
        prefix = prefix.strip().lower()[:1]
        if prefix in out:
            out[prefix] = value.strip()
    return out


def taxon_key(taxon, key_level):
    if key_level == "taxon":
        return taxon, parse_taxonomy(taxon)
    meta = parse_taxonomy(taxon)
    genus = meta.get("g", "").strip()
    genus_lower = genus.lower()
    if not genus or genus_lower in {"uncultured", "unclassified", "unknown"} or genus_lower.startswith(
        "uncultured"
    ):
        return taxon, meta
    return genus, meta


def write_tsv(path, rows, header):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse cohort lists
    all_cohorts = [c.strip() for c in args.cohorts.split(",")]
    source_cohorts = [c.strip() for c in args.source_cohorts.split(",")]
    target_cohort = args.target_cohort.strip()
    cooccur_cohort = (args.cooccur_cohort or target_cohort).strip()

    valid_cohorts = set(all_cohorts)

    cohort_map = read_metadata(args.metadata, valid_cohorts)
    if not cohort_map:
        raise SystemExit(f"No samples found for cohorts {all_cohorts} in metadata.")

    # Pass 1: totals per sample
    with open(args.table, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]

        col_info = []
        sample_index = {}
        cohort_counts = {c: 0 for c in all_cohorts}
        for idx, sid in enumerate(sample_ids):
            cohort = cohort_map.get(sid)
            if cohort in cohort_counts:
                sample_index[sid] = len(sample_index)
                col_info.append((idx + 1, sample_index[sid], cohort))
                cohort_counts[cohort] += 1

        if not col_info:
            raise SystemExit("No sample columns in table overlap with metadata.")

        print(f"Cohort counts: {cohort_counts}")

        totals = {sid_idx: 0.0 for _, sid_idx, _ in col_info}
        for row in reader:
            for col_idx, sid_idx, _ in col_info:
                try:
                    val = float(row[col_idx])
                except (ValueError, IndexError):
                    val = 0.0
                totals[sid_idx] += val

    # Pass 2: per-taxon metrics
    taxa_sums = defaultdict(lambda: {c: 0.0 for c in all_cohorts})
    present_sets = defaultdict(lambda: {c: set() for c in all_cohorts})
    taxa_meta = {}
    taxa_example = {}
    with open(args.table, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            taxon = row[0]
            key, meta = taxon_key(taxon, args.key)
            if key not in taxa_meta:
                taxa_meta[key] = meta
                taxa_example[key] = taxon
            for col_idx, sid_idx, cohort in col_info:
                try:
                    val = float(row[col_idx])
                except (ValueError, IndexError):
                    val = 0.0
                if val > 0:
                    present_sets[key][cohort].add(sid_idx)
                denom = totals.get(sid_idx, 0.0)
                if denom > 0:
                    taxa_sums[key][cohort] += val / denom

    # Compute per-taxon metrics
    taxa_metrics = {}
    shared_counts = defaultdict(int)
    for key, sums in taxa_sums.items():
        means = {}
        prev = {}
        for cohort in all_cohorts:
            n = cohort_counts[cohort]
            means[cohort] = sums[cohort] / n if n else 0.0
            prev[cohort] = len(present_sets[key][cohort]) / n if n else 0.0

        mean_sum = sum(means.values())
        spec = {c: (means[c] / mean_sum if mean_sum > 0 else 0.0) for c in all_cohorts}
        indval = {c: spec[c] * prev[c] for c in all_cohorts}
        present_flags = {c: prev[c] > 0 for c in all_cohorts}
        shared_key = "+".join(sorted([c for c, v in present_flags.items() if v]))
        shared_counts[shared_key or "none"] += 1

        taxa_metrics[key] = {
            "taxon": taxa_example.get(key, ""),
            "meta": taxa_meta.get(key, {}),
            "means": means,
            "prev": prev,
            "spec": spec,
            "indval": indval,
            "present": present_flags,
        }

    # Identify top indicator taxa for each source cohort
    top_sets = {}
    signature_counts = {}
    for src in source_cohorts:
        top = sorted(
            taxa_metrics.items(),
            key=lambda kv: kv[1]["indval"][src],
            reverse=True,
        )[: args.top_n]
        top_sets[src] = {key for key, _ in top}

        sig_counts = defaultdict(int)
        for key in top_sets[src]:
            for cohort in all_cohorts:
                for sid_idx in present_sets[key][cohort]:
                    sig_counts[sid_idx] += 1
        signature_counts[src] = sig_counts

    # Ambiguous taxa: present in both source cohorts
    src0, src1 = source_cohorts[0], source_cohorts[1]
    ambiguous_taxa = {
        key
        for key, info in taxa_metrics.items()
        if info["present"].get(src0) and info["present"].get(src1)
    }
    ambiguous_stats = {}
    for key in ambiguous_taxa:
        if cooccur_cohort == "all":
            scope = list(all_cohorts)
        else:
            scope = [cooccur_cohort]
        src0_sum = 0.0
        src1_sum = 0.0
        present = 0
        for cohort in scope:
            for sid_idx in present_sets[key][cohort]:
                src0_sum += signature_counts[src0].get(sid_idx, 0)
                src1_sum += signature_counts[src1].get(sid_idx, 0)
                present += 1
        ambiguous_stats[key] = {"src0_sum": src0_sum, "src1_sum": src1_sum, "present": present}

    # Write outputs
    # Build dynamic column headers
    taxa_rows = []
    edge_rows = []
    node_strength = defaultdict(float)
    assignment_rows = []

    for key, info in taxa_metrics.items():
        taxon = info["taxon"]
        tax = info["meta"]
        ind_src0 = info["indval"][src0]
        ind_src1 = info["indval"][src1]

        assign = "ambiguous"
        delta = abs(ind_src0 - ind_src1)
        ratio = max(ind_src0, ind_src1) / (min(ind_src0, ind_src1) + 1e-12)
        if delta >= args.delta_threshold and ratio >= args.ratio_threshold:
            assign = src0 if ind_src0 > ind_src1 else src1

        cooccur_pref = ""
        if key in ambiguous_stats:
            stats = ambiguous_stats[key]
            if stats["present"] > 0:
                s0_mean = stats["src0_sum"] / stats["present"]
                s1_mean = stats["src1_sum"] / stats["present"]
                if s0_mean > s1_mean:
                    cooccur_pref = src0
                elif s1_mean > s0_mean:
                    cooccur_pref = src1

        row_dict = {
            "taxon_key": key,
            "taxon": taxon,
            "phylum": tax.get("p", ""),
            "family": tax.get("f", ""),
            "genus": tax.get("g", ""),
        }
        for c in all_cohorts:
            row_dict[f"{c}_mean_rel_abund"] = f"{info['means'][c]:.6g}"
            row_dict[f"{c}_prevalence"] = f"{info['prev'][c]:.6g}"
            row_dict[f"{c}_spec"] = f"{info['spec'][c]:.6g}"
            row_dict[f"{c}_indval"] = f"{info['indval'][c]:.6g}"
            row_dict[f"present_{c}"] = "yes" if info["present"][c] else ""
        row_dict["assignment"] = assign
        row_dict["cooccurrence_pref"] = cooccur_pref
        taxa_rows.append(row_dict)

        for source in all_cohorts:
            weight = info["indval"][source]
            if weight <= 0:
                continue
            edge_rows.append(
                {
                    "source": source,
                    "taxon_key": key,
                    "taxon": taxon,
                    "weight_indval": f"{weight:.6g}",
                    "mean_rel_abund": f"{info['means'][source]:.6g}",
                    "prevalence": f"{info['prev'][source]:.6g}",
                }
            )
            node_strength[source] += weight
            node_strength[key] += weight

        if info["present"].get(src0) and info["present"].get(src1):
            assignment_rows.append(
                {
                    "taxon_key": key,
                    "taxon": taxon,
                    f"{src0}_indval": f"{ind_src0:.6g}",
                    f"{src1}_indval": f"{ind_src1:.6g}",
                    "indval_delta": f"{delta:.6g}",
                    "indval_ratio": f"{ratio:.6g}",
                    "assignment": assign,
                    "cooccurrence_pref": cooccur_pref,
                }
            )

    # Build header dynamically
    taxa_header = ["taxon_key", "taxon", "phylum", "family", "genus"]
    for c in all_cohorts:
        taxa_header.extend([
            f"{c}_mean_rel_abund", f"{c}_prevalence", f"{c}_spec", f"{c}_indval", f"present_{c}"
        ])
    taxa_header.extend(["assignment", "cooccurrence_pref"])

    write_tsv(out_dir / "source_network_taxa.tsv", taxa_rows, taxa_header)
    write_tsv(
        out_dir / "source_network_edges.tsv",
        edge_rows,
        ["source", "taxon_key", "taxon", "weight_indval", "mean_rel_abund", "prevalence"],
    )

    node_rows = []
    for node, strength in node_strength.items():
        node_rows.append(
            {
                "node": node,
                "type": "source" if node in valid_cohorts else "taxon",
                "strength": f"{strength:.6g}",
            }
        )
    write_tsv(out_dir / "source_network_nodes.tsv", node_rows, ["node", "type", "strength"])

    assignment_header = [
        "taxon_key", "taxon",
        f"{src0}_indval", f"{src1}_indval",
        "indval_delta", "indval_ratio",
        "assignment", "cooccurrence_pref",
    ]
    write_tsv(out_dir / "ambiguous_taxa_assignments.tsv", assignment_rows, assignment_header)

    summary_rows = []
    for key, count in sorted(shared_counts.items()):
        summary_rows.append({"presence_pattern": key, "taxa_count": count})
    write_tsv(out_dir / "source_network_summary.tsv", summary_rows, ["presence_pattern", "taxa_count"])
    write_tsv(
        out_dir / "assignment_summary.tsv",
        [{"metric": f"presence_{k}", "value": v} for k, v in sorted(shared_counts.items())],
        ["metric", "value"],
    )

    print(f"Network outputs written to {out_dir}/")
    print(f"  source_network_taxa.tsv: {len(taxa_rows)} taxa")
    print(f"  source_network_edges.tsv: {len(edge_rows)} edges")
    print(f"  ambiguous_taxa_assignments.tsv: {len(assignment_rows)} ambiguous taxa")


if __name__ == "__main__":
    main()
