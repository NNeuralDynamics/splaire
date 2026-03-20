#!/usr/bin/env python3
"""
add observed and L1-normalized observed attributions to attribution h5 files

computes:
  - obs_<head>: (X * attr).sum(axis=2) - observed attribution at actual base
  - obs_norm_<head>: obs / sum(|obs|) - L1 normalized per sequence

usage:
    python add_normalized_attributions.py --attr data/attr_sphaec_ref_reg.h5 --seq data/sequences.h5
    python add_normalized_attributions.py --attr data/attr_*.h5 --seq data/sequences.h5
"""

import argparse

import numpy as np
import h5py
from glob import glob


def add_normalized_attributions(attr_path, seq_path):
    """add observed and normalized observed attributions to h5 file"""

    print(f"loading {seq_path}")
    with h5py.File(seq_path, 'r') as f:
        X = f['X'][:].astype(np.float32)
    print(f"{X.shape[0]:,} sequences, {X.shape[1]} bp")

    print(f"processing {attr_path}")

    with h5py.File(attr_path, 'r+') as f:
        attr_keys = [k for k in f.keys() if k.startswith('attr_')]
        assert attr_keys, "no attr_ datasets found"

        for attr_key in attr_keys:
            head = attr_key.replace('attr_', '')
            obs_key = f'obs_{head}'
            obs_norm_key = f'obs_norm_{head}'

            print(f"{head}: computing...")

            attr = f[attr_key][:].astype(np.float32)
            assert attr.shape == X.shape, f"shape mismatch attr={attr.shape} vs X={X.shape}"

            # observed attribution at actual base
            obs = (X * attr).sum(axis=2).astype(np.float32)

            # L1 norm per sequence (no safeguard - let it fail if it fails)
            l1_raw = np.abs(obs).sum(axis=1, keepdims=True)

            # track minimum denominator
            min_l1 = l1_raw.min()
            min_idx = l1_raw.argmin()
            print(f"  min denominator: {min_l1:.6e} (seq {min_idx})")

            # normalize without safeguard
            obs_norm = (obs / l1_raw).astype(np.float32)

            # verify L1 norm == 1 for all sequences
            l1_check = np.abs(obs_norm).sum(axis=1)
            tol = 1e-5
            bad_mask = np.abs(l1_check - 1.0) > tol
            n_bad = bad_mask.sum()

            if n_bad > 0:
                bad_indices = np.where(bad_mask)[0]
                print(f"  ERROR: {n_bad} sequences have sum(|obs_norm|) != 1")
                for idx in bad_indices[:10]:  # show first 10
                    print(f"    seq {idx}: L1={l1_check[idx]:.6f}, denom={l1_raw[idx, 0]:.6e}")
                if n_bad > 10:
                    print(f"    ... and {n_bad - 10} more")

            # check for numerical issues
            n_nan = np.isnan(obs_norm).sum()
            n_inf = np.isinf(obs_norm).sum()

            # save
            if obs_key in f:
                del f[obs_key]
            if obs_norm_key in f:
                del f[obs_norm_key]

            f.create_dataset(obs_key, data=obs, compression='gzip', compression_opts=4)
            f.create_dataset(obs_norm_key, data=obs_norm, compression='gzip', compression_opts=4)

            print(f"  {obs_key}: mean={obs.mean():.4f}")
            print(f"  {obs_norm_key}: mean={obs_norm.mean():.6f}")
            print(f"  L1 check: mean={l1_check.mean():.6f}, min={l1_check.min():.6f}, max={l1_check.max():.6f}")
            print(f"  nan: {n_nan}, inf: {n_inf}")

            if n_nan > 0 or n_inf > 0:
                print(f"  ERROR: numerical issues detected (nan={n_nan}, inf={n_inf})")

    print("done")


def main():
    p = argparse.ArgumentParser(description="add normalized attributions to h5 files")
    p.add_argument('--attr', nargs='+', required=True)
    p.add_argument('--seq', required=True)
    args = p.parse_args()

    # expand globs
    attr_paths = []
    for pattern in args.attr:
        matches = glob(pattern)
        if matches:
            attr_paths.extend(matches)
        else:
            attr_paths.append(pattern)

    print(f"{len(attr_paths)} attribution file(s)")

    for attr_path in attr_paths:
        add_normalized_attributions(attr_path, args.seq)
        print()


if __name__ == '__main__':
    main()
