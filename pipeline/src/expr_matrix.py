#!/usr/bin/env python3
“””merge featureCounts outputs into raw, tpm, and mor matrices”””

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd

ANNOT_COLS = ["Geneid", "Chr", "Start", "End", "Strand", "Length"]


def parse_args():
    p = argparse.ArgumentParser(
        description="merge featurecounts files and write expr_counts.tsv, expr_tpm.tsv, expr_mor.tsv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="paths to featureCounts files (single- or multi-sample TSV)",
    )
    p.add_argument(
        "--strip-version",
        action="store_true",
        help="strip Ensembl version suffix from gene IDs (e.g., ENSG000001234.5 → ENSG000001234)",
    )
    p.add_argument("--counts-out", default="expr_counts.tsv", help="output path for raw counts matrix")
    p.add_argument("--tpm-out", default="expr_tpm.tsv", help="output path for TPM matrix")
    p.add_argument("--mor-out", default="expr_mor.tsv", help="output path for median-of-ratios normalized counts")
    return p.parse_args()


def sample_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.counts\.txt$", "", base, flags=re.IGNORECASE)
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}__{base}"


def read_one(path: str):
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    if "Geneid" not in df.columns or "Length" not in df.columns:
        raise SystemExit(f"{path}: missing required columns Geneid/Length")

    annot = [c for c in ANNOT_COLS if c in df.columns]
    count_cols = [c for c in df.columns if c not in annot]
    if not count_cols:
        raise SystemExit(f"{path}: no count columns detected")

    # Pick the last count column (typical for single-sample outputs; still okay for multi-sample)
    counts = df[count_cols[-1]].astype("Int64").fillna(0).astype(int)

    out = pd.DataFrame(
        {
            "gene": df["Geneid"].astype(str),
            "length": df["Length"].astype("Int64"),
            "count": counts,
        }
    )
    sample = sample_name_from_path(path)
    return sample, out


def maybe_strip_version(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gene"] = df["gene"].str.replace(r"\.\d+$", "", regex=True)
    return df


def main():
    args = parse_args()

    # Read all inputs
    tables = []
    samples = []
    for p in args.inputs:
        s, t = read_one(p)
        if args.strip_version:
            t = maybe_strip_version(t)
        tables.append(t)
        samples.append(s)

    # Fail fast if names still collide for any reason
    vc = pd.Series(samples).value_counts()
    dups = vc[vc > 1]
    if not dups.empty:
        dup_list = ", ".join(list(dups.index))
        raise SystemExit(f"duplicate sample names after derivation: {dup_list}")

    # Union of genes across samples
    all_genes = pd.Index(sorted(set().union(*[t["gene"] for t in tables])))

    # Build counts matrix
    counts = pd.DataFrame(index=all_genes)
    for s, t in zip(samples, tables):
        counts[s] = t.set_index("gene")["count"].reindex(all_genes).fillna(0).astype(int)

    # Write raw counts
    counts_out = counts.copy()
    counts_out.insert(0, "gene_id", counts_out.index)
    counts_out.to_csv(args.counts_out, sep="\t", index=False)

    # Lengths: take from first file (assumes same GTF across samples)
    first_len = tables[0].set_index("gene")["length"]
    lengths = first_len.reindex(all_genes).astype("float64")
    lengths[lengths <= 0] = np.nan

    # TPM
    rpk = counts.div(lengths.values / 1000.0, axis=0)             # counts / kb
    tpm = rpk.div(rpk.sum(axis=0) / 1e6, axis=1).fillna(0.0)      # columns sum to 1e6
    tpm_out = tpm.copy()
    tpm_out.insert(0, "gene_id", tpm_out.index)
    tpm_out.to_csv(args.tpm_out, sep="\t", index=False, float_format="%.6f")

    # DESeq-style median-of-ratios size-factor normalization
    X = counts.astype(float).values
    with np.errstate(divide="ignore"):
        logX = np.where(X > 0, np.log(X), np.nan)
    gm = np.exp(np.nanmean(logX, axis=1))
    valid = gm > 0
    ratios = X[valid] / gm[valid, None]
    sf = np.nanmedian(np.where(ratios > 0, ratios, np.nan), axis=0)
    sf[sf <= 0] = np.nan
    mor = counts.astype(float).div(sf, axis=1).fillna(0.0)
    mor_out = mor.copy()
    mor_out.insert(0, "gene_id", mor_out.index)
    mor_out.to_csv(args.mor_out, sep="\t", index=False, float_format="%.6f")

    # Warn if gene lengths disagree across inputs
    lens0 = first_len
    for p, t in zip(args.inputs[1:], tables[1:]):
        lens = t.set_index("gene")["length"].reindex(lens0.index)
        if not np.all((lens0.fillna(-1).values == lens.fillna(-1).values)):
            print(f"warning: gene lengths differ vs first file: {os.path.basename(p)}", file=sys.stderr)


if __name__ == "__main__":
    main()
