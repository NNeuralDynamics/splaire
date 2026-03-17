#!/usr/bin/env python3
"""generate train/valid sample splits with reproducible seeds.

usage:
    python generate_sample_splits.py \
        --train-samples train_samples.txt \
        --output-dir ./splits \
        --n-splits 5 --valid-frac 0.10 --seed-base 42
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="generate train/valid sample splits")
    parser.add_argument("--train-samples", required=True, help="input file with all train samples")
    parser.add_argument("--output-dir", required=True, help="output directory for split files")
    parser.add_argument("--n-splits", type=int, default=5, help="number of splits to generate")
    parser.add_argument("--valid-frac", type=float, default=0.10, help="fraction of samples for validation")
    parser.add_argument("--seed-base", type=int, default=42, help="base seed (split i uses seed_base + i)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load samples (filter out "holder" placeholder)
    samples = pd.read_csv(args.train_samples, header=None)[0].tolist()
    samples = [s for s in samples if s and s.strip() and s.strip() != 'holder']
    n_valid = int(len(samples) * args.valid_frac)

    print(f"loaded {len(samples)} samples from {args.train_samples}")
    print(f"valid fraction: {args.valid_frac} ({n_valid} samples)")
    print(f"generating {args.n_splits} splits with seeds {args.seed_base + 1} to {args.seed_base + args.n_splits}")
    print()

    for split in range(1, args.n_splits + 1):
        seed = args.seed_base + split  # seeds: 43, 44, 45, 46, 47
        rng = np.random.default_rng(seed)
        valid_idx = set(rng.choice(len(samples), size=n_valid, replace=False))

        train = [s for i, s in enumerate(samples) if i not in valid_idx]
        valid = [samples[i] for i in sorted(valid_idx)]

        train_file = out_dir / f"train_split{split}.txt"
        valid_file = out_dir / f"valid_split{split}.txt"

        pd.Series(train).to_csv(train_file, index=False, header=False)
        pd.Series(valid).to_csv(valid_file, index=False, header=False)
        print(f"split {split} (seed={seed}): {len(train)} train, {len(valid)} valid")
        print(f"  -> {train_file}")
        print(f"  -> {valid_file}")


if __name__ == "__main__":
    main()
