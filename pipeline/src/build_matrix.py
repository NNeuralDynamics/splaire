#!/usr/bin/env python3
"""build SSU matrix from SpliSER-annotated TSVs"""
import os
import glob
import argparse
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from tqdm import tqdm


def process_file(file_path):
    """extract fraction column from one sample"""
    # infer sample name from parent folder
    sample = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    df = pd.read_csv(file_path, sep="\t")

    # fill any missing counts with zero
    for col in ["alpha_count", "beta1_count", "beta2Simple_count", "beta2Cryptic_count"]:
        if col in df:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # row label
    df["row_label"] = (
        df.Region.astype(str) + "_" +
        df.Strand.astype(str) + "_" +
        df.Site.astype(str) + "_" +
        df.site_type.astype(str)
    )

    def compute_fraction(row):
        a = row["alpha_count"]
        b1 = row["beta1_count"]
        b2s = row["beta2Simple_count"]
        b2c = row["beta2Cryptic_count"]
        denom = a + b1 + b2s + b2c
        if denom == 0:
            return "0/0"
        return f"{int(a)}/{int(denom)}"

    df["fraction"] = df.apply(compute_fraction, axis=1)
    return sample, df.set_index("row_label")["fraction"]


def build_raw_matrix(directory, suffix, output_file):
    """collect all samples into raw fraction matrix"""
    pattern = os.path.join(directory, "*", "spliser", f"*{suffix}")
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] no files found: {pattern}")
        return None

    print(f"building raw matrix from {len(files)} samples")

    with Pool(min(cpu_count(), len(files))) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="  collecting"))

    df_dict = {sample: series for sample, series in results if series is not None}
    matrix = pd.DataFrame(df_dict).sort_index()
    matrix.to_csv(output_file, sep="\t")
    print(f"  raw matrix: {output_file} ({len(matrix)} sites x {len(df_dict)} samples)")
    return matrix


def _convert_chunk(args):
    """fraction strings to decimals"""
    chunk, denom_thresh = args
    parts = np.char.partition(chunk, '/')
    num = parts[..., 0].astype(float)
    den = parts[..., 2].astype(float)

    mask_low = den < denom_thresh
    safe_den = den.copy()
    safe_den[mask_low] = 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        dec = num / safe_den

    dec[mask_low] = 777.0
    return dec


def process_matrix(raw_matrix_file, processed_matrix_file, denom_thresh):
    """fractions to decimals, mask low coverage, add metadata"""
    print(f"processing matrix (denom_thresh={denom_thresh})")

    raw = pd.read_csv(raw_matrix_file, sep="\t", index_col=0, dtype=str)

    meta_cols = {'Gene_ID'}
    sample_cols = [c for c in raw.columns if c not in meta_cols]
    print(f"  {len(sample_cols)} sample columns")

    # fill missing with 0/0
    raw[sample_cols] = raw[sample_cols].fillna("0/0").replace(r'^\s*$', "0/0", regex=True)

    arr = raw[sample_cols].values.astype(str)

    # count malformed entries
    has_slash = np.char.count(arr, '/') == 1
    parts = np.char.partition(arr, '/')
    left_is_num = np.char.isdigit(parts[..., 0])
    right_is_num = np.char.isdigit(parts[..., 2])
    valid = has_slash & left_is_num & right_is_num
    bad = arr.size - np.count_nonzero(valid)
    if bad > 0:
        print(f"  {bad} malformed entries treated as 0/0")

    # convert in parallel
    nproc = max(1, cpu_count() - 1)
    chunks = np.array_split(arr, nproc, axis=0)
    args = [(chunk, denom_thresh) for chunk in chunks]

    with Pool(nproc) as pool:
        dec_parts = pool.map(_convert_chunk, args)
    dec = np.vstack(dec_parts)

    # assemble dataframe
    meta_df = raw.drop(columns=sample_cols)
    ssu_df = pd.DataFrame(dec, index=raw.index, columns=sample_cols)
    df = pd.concat([meta_df, ssu_df], axis=1)

    # filter fully masked rows
    keep = ~(df[sample_cols] == 777.0).all(axis=1)
    n_dropped = (~keep).sum()
    df = df.loc[keep]
    print(f"  dropped {n_dropped} fully-masked rows, {len(df)} remain")

    # compute pop_mean
    pop_mean = df[sample_cols].replace(777.0, np.nan).mean(axis=1)
    df = pd.concat([df, pop_mean.rename('pop_mean')], axis=1)

    # parse index into metadata columns
    parts_idx = df.index.to_series().str.split('_', n=3, expand=True)
    parts_idx.columns = ['region', 'strand', 'site', 'site_type']
    parts_idx['site'] = parts_idx['site'].astype(int)
    idx_df = pd.DataFrame({
        'event_id': df.index,
        'region': parts_idx['region'],
        'strand': parts_idx['strand'],
        'site': parts_idx['site'],
        'site_type': parts_idx['site_type']
    }, index=df.index)
    df = pd.concat([idx_df, df], axis=1)

    df.to_csv(processed_matrix_file, sep='\t', index=False)
    print(f"  processed matrix: {processed_matrix_file}")


def main():
    parser = argparse.ArgumentParser(description="build SSU matrix from SpliSER-annotated TSVs")
    parser.add_argument("-d", "--directory", required=True, help="project base dir with sample subdirs")
    parser.add_argument("-s", "--suffix", default=".complete_annotated.tsv", help="file suffix")
    parser.add_argument("-o", "--output", default="splicing_matrix.tsv", help="raw matrix output")
    parser.add_argument("-t", "--denom-thresh", type=float, default=5.0, help="coverage threshold for masking")
    args = parser.parse_args()

    raw_out = args.output
    processed_out = f"processed_{args.output}"

    build_raw_matrix(args.directory, args.suffix, raw_out)
    process_matrix(raw_out, processed_out, args.denom_thresh)

    print("[done]")


if __name__ == "__main__":
    main()
