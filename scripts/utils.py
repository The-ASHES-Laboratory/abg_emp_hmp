# utils.py
# By Carter Clinton, Ph.D.
"""Shared utility functions for QIIME2 table I/O and compositional transforms."""

import pandas as pd, numpy as np

def read_qiime_tsv(table_tsv):
    """
    Load a QIIME2-exported feature table (TSV) and return a sample x feature matrix.
    Handles the common "# Constructed..." preamble that biom adds so that the header
    row beginning with "#OTU ID" is preserved rather than discarded as a comment.
    """
    header_tokens = (
        "#OTU ID", "#OTUID", "#OTU_ID", "#Feature ID", "#FEATURE ID",
        "feature-id", "feature id", "OTU ID", "Feature ID"
    )
    skiprows = 0
    with open(table_tsv, "r") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                skiprows += 1
                continue
            if stripped.startswith("#") and not stripped.startswith(header_tokens):
                skiprows += 1
                continue
            # We found the header row (either #OTU ID or the first non-comment line)
            break
    df = pd.read_csv(table_tsv, sep="\t", skiprows=skiprows, index_col=0, low_memory=False)
    if df.index.name:
        df.index.name = df.index.name.lstrip("#")
    df.columns = [str(c).lstrip("#") for c in df.columns]
    if len(df.columns) and df.columns[-1].lower().startswith("taxonomy"):
        df = df.iloc[:, :-1]
    return df

def clr_transform(counts, pseudo=0.5):
    X = counts + pseudo
    gm = np.exp(np.log(X).mean(axis=0))  # geometric mean per sample
    return np.log(X / gm)
