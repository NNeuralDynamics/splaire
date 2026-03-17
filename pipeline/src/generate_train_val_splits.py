#!/usr/bin/env python3
"""split master tsv into train/validation sets for replicate model training.

creates 5-way transcript-level splits:
  - main_train_paralogs.tsv (shared across all splits)
  - split{N}_nonparalog_train.tsv (90% non-paralogs, different per split)
  - split{N}_validation.tsv (10% non-paralogs, different per split)
"""

import argparse
import os

import numpy as np
import pandas as pd


def is_autosomal(chrom):
    """check if chromosome is autosomal (chr1-22)"""
    s = str(chrom).replace('chr', '')
    try:
        return 1 <= int(s) <= 22
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="generate train/validation splits from master tsv"
    )
    parser.add_argument("--input", required=True,
                        help="master tsv (output of fill_gencode or filter_empty_txs)")
    parser.add_argument("--output-dir", required=True,
                        help="output directory for split files")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="number of train/validation splits (default: 5)")
    parser.add_argument("--val-frac", type=float, default=0.10,
                        help="fraction of non-paralogs for validation (default: 0.10)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="base random seed; split N uses seed_base + N (default: 42)")
    parser.add_argument("--exclude-chroms", type=str, default="1,3,5,7",
                        help="comma-separated chrom numbers to exclude non-paralogs from (default: 1,3,5,7)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exclude = set(args.exclude_chroms.split(','))

    # load master tsv
    print(f"loading {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    print(f"loaded {len(df):,} rows")

    # filter to autosomes
    df = df[df['Chromosome'].apply(is_autosomal)].copy().reset_index(drop=True)
    print(f"autosomes only: {len(df):,} rows")

    # strip 'chr' prefix for matching
    chrom_core = df['Chromosome'].astype(str).str.replace('chr', '', regex=False)

    # remove non-paralogs on excluded chroms (test-only)
    is_nonparalog = df['paralog_status'] == 0
    is_excluded = chrom_core.isin(exclude)
    keep = ~(is_nonparalog & is_excluded)
    df = df[keep].reset_index(drop=True)
    print(f"after removing non-paralogs on chr {','.join(sorted(exclude))}: {len(df):,} rows")

    # separate paralogs from non-paralogs
    paralog_df = df.query("paralog_status == 1").reset_index(drop=True)
    nonparalog_df = df.query("paralog_status == 0").reset_index(drop=True)
    print(f"paralogs: {len(paralog_df):,}")
    print(f"non-paralogs: {len(nonparalog_df):,}")

    # save paralogs (shared across all splits)
    paralog_path = os.path.join(args.output_dir, 'main_train_paralogs.tsv')
    paralog_df.to_csv(paralog_path, sep='\t', index=False)
    print(f"saved paralogs -> {paralog_path}")

    # generate splits
    n_nonpara = len(nonparalog_df)
    target_val = int(n_nonpara * args.val_frac)
    print(f"each split: {target_val:,} validation, {n_nonpara - target_val:,} train")
    print("=" * 50)

    for split in range(1, args.n_splits + 1):
        rng = np.random.default_rng(args.seed_base + split)

        # sample validation indices
        val_idx = rng.choice(n_nonpara, size=target_val, replace=False)
        train_idx = np.setdiff1d(np.arange(n_nonpara), val_idx)

        val_df = nonparalog_df.iloc[val_idx].reset_index(drop=True)
        train_df = nonparalog_df.iloc[train_idx].reset_index(drop=True)

        # save
        val_path = os.path.join(args.output_dir, f'split{split}_validation.tsv')
        train_path = os.path.join(args.output_dir, f'split{split}_nonparalog_train.tsv')
        val_df.to_csv(val_path, sep='\t', index=False)
        train_df.to_csv(train_path, sep='\t', index=False)

        print(f"split {split}: validation={len(val_df):,} -> {val_path}")
        print(f"         train={len(train_df):,} -> {train_path}")

    print("=" * 50)
    print(f"done. 1 paralog file + {args.n_splits * 2} split files in {args.output_dir}")


if __name__ == "__main__":
    main()
