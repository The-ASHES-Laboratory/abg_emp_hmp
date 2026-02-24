# 27_integrate_feast_network.py
# By Carter Clinton, Ph.D.
"""Integrate FEAST proportions with ambiguous taxa from the source network."""

import argparse
import csv
import random
import statistics
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
        default="results/metadata_cohort.prev.tsv",
        help="Metadata TSV with sample-id and cohort columns",
    )
    parser.add_argument(
        "--feast",
        default="results/feast/nyabg_source_props.tsv",
        help="FEAST source proportions for NYABG",
    )
    parser.add_argument(
        "--ambiguous",
        default="results/networks/ambiguous_taxa_assignments.tsv",
        help="Ambiguous taxa assignments from source network",
    )
    parser.add_argument(
        "--out-dir",
        default="results/networks/feast_integration",
        help="Output directory",
    )
    parser.add_argument(
        "--cyto-out",
        default="results/networks/cytoscape/source_network_nodes_feast.tsv",
        help="Cytoscape node attributes output",
    )
    parser.add_argument(
        "--target-cohort",
        default=None,
        help="Target cohort to filter (burial/control). If not set, uses 'nyabg' matching.",
    )
    parser.add_argument(
        "--key",
        choices=["taxon", "genus"],
        default="genus",
        help="Aggregate by full taxon or genus name (default: genus)",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Permutation count for group comparison",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for permutation test",
    )
    return parser.parse_args()


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
        return taxon
    meta = parse_taxonomy(taxon)
    genus = meta.get("g", "").strip()
    genus_lower = genus.lower()
    if not genus or genus_lower in {"uncultured", "unclassified", "unknown"} or genus_lower.startswith(
        "uncultured"
    ):
        return taxon
    return genus


def read_metadata(meta_path, target_cohort=None):
    """Read metadata and return set of target sample IDs.

    If target_cohort is specified (e.g. 'burial', 'control'), filters by that cohort.
    Otherwise falls back to matching 'nyabg' for backward compatibility.
    """
    target_ids = set()
    match_val = (target_cohort or "nyabg").strip().lower()
    with open(meta_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sample = row.get("sample-id") or row.get("sample_id") or row.get("sample")
            cohort = (row.get("cohort") or row.get("dataset") or row.get("group") or "").strip().lower()
            if not sample:
                continue
            if cohort == match_val:
                target_ids.add(str(sample))
    return target_ids


def read_feast(feast_path):
    with open(feast_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit("FEAST file has no header")
        fields = {name.lower(): name for name in reader.fieldnames}
        sample_col = (
            fields.get("sample_id")
            or fields.get("sample-id")
            or fields.get("sample")
        )
        emp_col = fields.get("emp")
        hmp_col = fields.get("hmp")
        if not sample_col or not emp_col:
            raise SystemExit("FEAST file missing sample_id or emp columns")
        data = {}
        for row in reader:
            sid = row.get(sample_col)
            if not sid:
                continue
            try:
                emp_val = float(row.get(emp_col, "0") or 0)
            except ValueError:
                emp_val = 0.0
            try:
                hmp_val = float(row.get(hmp_col, "0") or 0) if hmp_col else 0.0
            except ValueError:
                hmp_val = 0.0
            data[str(sid)] = {"emp": emp_val, "hmp": hmp_val}
    return data


def read_ambiguous(path):
    taxa = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = row.get("taxon_key") or row.get("taxon")
            if not key:
                continue
            taxa[key] = {
                "assignment": (row.get("assignment") or "").strip(),
                "cooccurrence_pref": (row.get("cooccurrence_pref") or "").strip(),
            }
    return taxa


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def stdev(values):
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def permutation_pvalue(values_a, values_b, permutations, seed):
    if not values_a or not values_b:
        return None
    observed = statistics.mean(values_a) - statistics.mean(values_b)
    combined = list(values_a) + list(values_b)
    n_a = len(values_a)
    random.seed(seed)
    count = 0
    for _ in range(permutations):
        random.shuffle(combined)
        diff = statistics.mean(combined[:n_a]) - statistics.mean(combined[n_a:])
        if abs(diff) >= abs(observed):
            count += 1
    return (count + 1) / (permutations + 1)


def weighted_mean(values, weights):
    total = sum(weights)
    if total == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total


def rank_weights(values):
    n = len(values)
    if n == 0:
        return []
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            idx = sorted_pairs[k][0]
            ranks[idx] = avg_rank
        i = j + 1
    return [r / n for r in ranks]


def weighted_permutation_values(values_a, weights_a, values_b, weights_b, permutations, seed):
    if not values_a or not values_b:
        return None, None
    mean_a = weighted_mean(values_a, weights_a)
    mean_b = weighted_mean(values_b, weights_b)
    if mean_a is None or mean_b is None:
        return None, None
    observed = mean_a - mean_b

    combined = list(zip(values_a, weights_a)) + list(zip(values_b, weights_b))
    n_a = len(values_a)
    random.seed(seed)
    count = 0
    for _ in range(permutations):
        random.shuffle(combined)
        grp_a = combined[:n_a]
        grp_b = combined[n_a:]
        vals_a = [item[0] for item in grp_a]
        wts_a = [item[1] for item in grp_a]
        vals_b = [item[0] for item in grp_b]
        wts_b = [item[1] for item in grp_b]
        perm_mean_a = weighted_mean(vals_a, wts_a)
        perm_mean_b = weighted_mean(vals_b, wts_b)
        if perm_mean_a is None or perm_mean_b is None:
            continue
        diff = perm_mean_a - perm_mean_b
        if abs(diff) >= abs(observed):
            count += 1
    return observed, (count + 1) / (permutations + 1)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_label = args.target_cohort or "nyabg"
    nyabg_ids = read_metadata(args.metadata, target_cohort=args.target_cohort)
    if not nyabg_ids:
        raise SystemExit(f"No {target_label} samples found in metadata.")

    feast = read_feast(args.feast)
    feast_emp = {sid: v["emp"] for sid, v in feast.items() if sid in nyabg_ids}
    if not feast_emp:
        raise SystemExit("No FEAST proportions matched NYABG samples.")

    ambiguous = read_ambiguous(args.ambiguous)
    ambiguous_keys = set(ambiguous)

    nyabg_present = {key: set() for key in ambiguous_keys}
    nyabg_rel_sums = {key: 0.0 for key in ambiguous_keys}

    with open(args.table, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        nyabg_indices = [(idx + 1, sid) for idx, sid in enumerate(sample_ids) if sid in nyabg_ids]
        if not nyabg_indices:
            raise SystemExit("No NYABG sample columns found in table.")

        nyabg_totals = {sid: 0.0 for _, sid in nyabg_indices}
        for row in reader:
            for col_idx, sid in nyabg_indices:
                try:
                    val = float(row[col_idx])
                except (ValueError, IndexError):
                    val = 0.0
                nyabg_totals[sid] += val

    with open(args.table, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            key = taxon_key(row[0], args.key)
            if key not in nyabg_present:
                continue
            for col_idx, sid in nyabg_indices:
                try:
                    val = float(row[col_idx])
                except (ValueError, IndexError):
                    val = 0.0
                if val > 0:
                    nyabg_present[key].add(sid)
                denom = nyabg_totals.get(sid, 0.0)
                if denom > 0:
                    nyabg_rel_sums[key] += val / denom

    rows = []
    group_values = defaultdict(list)
    detailed_values = defaultdict(list)
    weighted_items = defaultdict(list)

    for key, info in sorted(ambiguous.items()):
        assignment = info["assignment"]
        cooccur = info["cooccurrence_pref"]
        if assignment == "emp":
            simple_group = "emp_leaning"
            detailed_group = "assigned_emp"
        elif assignment == "hmp":
            simple_group = "hmp_leaning"
            detailed_group = "assigned_hmp"
        elif assignment == "ambiguous":
            if cooccur == "emp":
                simple_group = "emp_leaning"
                detailed_group = "ambiguous_emp"
            elif cooccur == "hmp":
                simple_group = "hmp_leaning"
                detailed_group = "ambiguous_hmp"
            else:
                simple_group = "ambiguous"
                detailed_group = "ambiguous_none"
        else:
            simple_group = "ambiguous"
            detailed_group = "ambiguous_none"

        present_ids = nyabg_present.get(key, set())
        emp_vals = [feast_emp[sid] for sid in present_ids if sid in feast_emp]
        nyabg_prev = len(present_ids) / len(nyabg_ids) if nyabg_ids else 0.0
        nyabg_mean_rel = nyabg_rel_sums.get(key, 0.0) / len(nyabg_ids) if nyabg_ids else 0.0

        row = {
            "taxon_key": key,
            "assignment": assignment,
            "cooccurrence_pref": cooccur,
            "leaning_group": simple_group,
            "detailed_group": detailed_group,
            "nyabg_present_n": len(emp_vals),
            "nyabg_prevalence": f"{nyabg_prev:.6g}",
            "nyabg_mean_rel_abund": f"{nyabg_mean_rel:.6g}",
            "nyabg_emp_fraction_mean": "",
            "nyabg_emp_fraction_median": "",
            "nyabg_emp_fraction_sd": "",
            "nyabg_emp_fraction_min": "",
            "nyabg_emp_fraction_max": "",
        }
        if emp_vals:
            emp_mean = mean(emp_vals)
            row.update(
                {
                    "nyabg_emp_fraction_mean": f"{emp_mean:.6g}",
                    "nyabg_emp_fraction_median": f"{median(emp_vals):.6g}",
                    "nyabg_emp_fraction_sd": f"{stdev(emp_vals):.6g}" if stdev(emp_vals) is not None else "",
                    "nyabg_emp_fraction_min": f"{min(emp_vals):.6g}",
                    "nyabg_emp_fraction_max": f"{max(emp_vals):.6g}",
                }
            )
            group_values[simple_group].append(emp_mean)
            detailed_values[detailed_group].append(emp_mean)
            weighted_items[simple_group].append(
                {
                    "emp_mean": emp_mean,
                    "prev": nyabg_prev,
                    "abund": nyabg_mean_rel,
                }
            )
        rows.append(row)

    out_path = out_dir / "ambiguous_taxa_feast.tsv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_rows = []
    for group, values in sorted(group_values.items()):
        summary_rows.append(
            {
                "group": group,
                "n_taxa": len(values),
                "mean_of_means": f"{mean(values):.6g}" if values else "",
                "median_of_means": f"{median(values):.6g}" if values else "",
            }
        )
    for group, values in sorted(detailed_values.items()):
        summary_rows.append(
            {
                "group": group,
                "n_taxa": len(values),
                "mean_of_means": f"{mean(values):.6g}" if values else "",
                "median_of_means": f"{median(values):.6g}" if values else "",
            }
        )

    with open(out_dir / "feast_group_summary.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "n_taxa", "mean_of_means", "median_of_means"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    emp_vals = group_values.get("emp_leaning", [])
    hmp_vals = group_values.get("hmp_leaning", [])
    p_value = permutation_pvalue(emp_vals, hmp_vals, args.permutations, args.seed)
    emp_items = weighted_items.get("emp_leaning", [])
    hmp_items = weighted_items.get("hmp_leaning", [])
    all_items = emp_items + hmp_items
    if all_items:
        prev_ranks = rank_weights([item["prev"] for item in all_items])
        abund_ranks = rank_weights([item["abund"] for item in all_items])
        for item, prev_rank, abund_rank in zip(all_items, prev_ranks, abund_ranks):
            item["rank_prev"] = prev_rank
            item["rank_abund"] = abund_rank
    else:
        for item in all_items:
            item["rank_prev"] = 0.0
            item["rank_abund"] = 0.0

    schemes = [
        ("prev", lambda item: item["prev"]),
        ("abund", lambda item: item["abund"]),
        ("prev_x_abund", lambda item: item["prev"] * item["abund"]),
        ("prev_plus_abund", lambda item: item["prev"] + item["abund"]),
        ("prev_plus_abund_norm", lambda item: (item["prev"] + item["abund"]) / 2),
        ("max_prev_abund", lambda item: max(item["prev"], item["abund"])),
        ("min_prev_abund", lambda item: min(item["prev"], item["abund"])),
        ("rank_prev", lambda item: item.get("rank_prev", 0.0)),
        ("rank_abund", lambda item: item.get("rank_abund", 0.0)),
        ("rank_prev_abund_mean", lambda item: (item.get("rank_prev", 0.0) + item.get("rank_abund", 0.0)) / 2),
        ("rank_prev_abund_prod", lambda item: item.get("rank_prev", 0.0) * item.get("rank_abund", 0.0)),
    ]

    scheme_results = {}
    for idx, (name, func) in enumerate(schemes):
        values_a = [item["emp_mean"] for item in emp_items]
        values_b = [item["emp_mean"] for item in hmp_items]
        weights_a = [func(item) for item in emp_items]
        weights_b = [func(item) for item in hmp_items]
        diff, pval = weighted_permutation_values(
            values_a, weights_a, values_b, weights_b, args.permutations, args.seed + idx
        )
        scheme_results[name] = (diff, pval)

    prev_diff, prev_p = scheme_results.get("prev", (None, None))
    abund_diff, abund_p = scheme_results.get("abund", (None, None))
    combo_diff, combo_p = scheme_results.get("prev_x_abund", (None, None))
    combo_add_diff, combo_add_p = scheme_results.get("prev_plus_abund", (None, None))

    with open(out_dir / "feast_emp_fraction_test.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "emp_leaning_n",
                "hmp_leaning_n",
                "mean_diff",
                "permutation_p",
                "weighted_prev_mean_diff",
                "weighted_prev_p",
                "weighted_abund_mean_diff",
                "weighted_abund_p",
                "weighted_prev_abund_mean_diff",
                "weighted_prev_abund_p",
                "weighted_prev_abund_add_mean_diff",
                "weighted_prev_abund_add_p",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        mean_diff = ""
        if emp_vals and hmp_vals:
            mean_diff = f"{(statistics.mean(emp_vals) - statistics.mean(hmp_vals)):.6g}"
        writer.writerow(
            {
                "emp_leaning_n": len(emp_vals),
                "hmp_leaning_n": len(hmp_vals),
                "mean_diff": mean_diff,
                "permutation_p": f"{p_value:.6g}" if p_value is not None else "",
                "weighted_prev_mean_diff": f"{prev_diff:.6g}" if prev_diff is not None else "",
                "weighted_prev_p": f"{prev_p:.6g}" if prev_p is not None else "",
                "weighted_abund_mean_diff": f"{abund_diff:.6g}" if abund_diff is not None else "",
                "weighted_abund_p": f"{abund_p:.6g}" if abund_p is not None else "",
                "weighted_prev_abund_mean_diff": f"{combo_diff:.6g}" if combo_diff is not None else "",
                "weighted_prev_abund_p": f"{combo_p:.6g}" if combo_p is not None else "",
                "weighted_prev_abund_add_mean_diff": f"{combo_add_diff:.6g}"
                if combo_add_diff is not None
                else "",
                "weighted_prev_abund_add_p": f"{combo_add_p:.6g}"
                if combo_add_p is not None
                else "",
            }
        )

    with open(out_dir / "feast_emp_fraction_test_allweights.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scheme", "mean_diff", "permutation_p"],
            delimiter="\t",
        )
        writer.writeheader()
        for name, _ in schemes:
            diff_val, p_val = scheme_results.get(name, (None, None))
            writer.writerow(
                {
                    "scheme": name,
                    "mean_diff": f"{diff_val:.6g}" if diff_val is not None else "",
                    "permutation_p": f"{p_val:.6g}" if p_val is not None else "",
                }
            )

    cyto_path = Path(args.cyto_out)
    cyto_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cyto_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "nyabg_emp_fraction_mean", "nyabg_present_n", "leaning_group"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["taxon_key"],
                    "nyabg_emp_fraction_mean": row["nyabg_emp_fraction_mean"],
                    "nyabg_present_n": row["nyabg_present_n"],
                    "leaning_group": row["leaning_group"],
                }
            )

    # Plot distribution by group (SVG)
    plot_groups = ["emp_leaning", "hmp_leaning", "ambiguous"]
    data = {group: group_values.get(group, []) for group in plot_groups}

    def quantile(values, q):
        if not values:
            return None
        vals = sorted(values)
        pos = (len(vals) - 1) * q
        low = int(pos)
        high = min(low + 1, len(vals) - 1)
        if low == high:
            return vals[low]
        frac = pos - low
        return vals[low] * (1 - frac) + vals[high] * frac

    width = 600
    height = 400
    margin_left = 80
    margin_right = 20
    margin_top = 40
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_vals = [v for values in data.values() for v in values]
    y_max = max(all_vals) if all_vals else 1.0
    y_min = min(all_vals) if all_vals else 0.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    def y_scale(val):
        return margin_top + (1 - (val - y_min) / y_range) * plot_h

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="{margin_top - 10}" font-size="14" font-family="Arial">EMP fraction by ambiguous taxon leaning</text>',
    ]

    svg_lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>'
    )
    svg_lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>'
    )

    step = plot_w / max(len(plot_groups), 1)
    box_w = step * 0.5

    for idx, group in enumerate(plot_groups):
        values = data.get(group, [])
        if not values:
            continue
        q1 = quantile(values, 0.25)
        q2 = quantile(values, 0.5)
        q3 = quantile(values, 0.75)
        vmin = min(values)
        vmax = max(values)

        cx = margin_left + idx * step + step / 2
        box_x = cx - box_w / 2
        box_y = y_scale(q3)
        box_h = y_scale(q1) - box_y

        svg_lines.append(
            f'<rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_w:.2f}" height="{box_h:.2f}" fill="#c6dbef" stroke="#2171b5"/>'
        )
        svg_lines.append(
            f'<line x1="{box_x:.2f}" y1="{y_scale(q2):.2f}" x2="{box_x + box_w:.2f}" y2="{y_scale(q2):.2f}" stroke="#08306b"/>'
        )
        svg_lines.append(
            f'<line x1="{cx:.2f}" y1="{y_scale(vmin):.2f}" x2="{cx:.2f}" y2="{y_scale(vmax):.2f}" stroke="#08306b"/>'
        )
        svg_lines.append(
            f'<line x1="{cx - box_w * 0.3:.2f}" y1="{y_scale(vmin):.2f}" x2="{cx + box_w * 0.3:.2f}" y2="{y_scale(vmin):.2f}" stroke="#08306b"/>'
        )
        svg_lines.append(
            f'<line x1="{cx - box_w * 0.3:.2f}" y1="{y_scale(vmax):.2f}" x2="{cx + box_w * 0.3:.2f}" y2="{y_scale(vmax):.2f}" stroke="#08306b"/>'
        )
        svg_lines.append(
            f'<text x="{cx:.2f}" y="{margin_top + plot_h + 20}" font-size="11" font-family="Arial" text-anchor="middle">{group}</text>'
        )

    svg_lines.append(
        f'<text x="{margin_left - 50}" y="{margin_top + plot_h / 2}" font-size="12" font-family="Arial" transform="rotate(-90 {margin_left - 50},{margin_top + plot_h / 2})">NYABG mean FEAST EMP fraction</text>'
    )
    svg_lines.append("</svg>")

    with open(out_dir / "feast_emp_fraction_by_group.svg", "w") as handle:
        handle.write("\n".join(svg_lines))


if __name__ == "__main__":
    main()
