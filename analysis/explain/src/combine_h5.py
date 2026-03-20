#!/usr/bin/env python3
"""
combine split h5 files back into one

combines ALL datasets (including attr_*, pred_*) by concatenating
along first axis.

usage:
    python combine_h5.py sequences.h5 4
    # reads: sequences_0.h5, sequences_1.h5, sequences_2.h5, sequences_3.h5
    # writes combined data to: sequences.h5
    # deletes split files (unless --keep)
"""

import sys
import os
import h5py
import numpy as np


def combine_h5(h5_path, n_splits, delete_splits=True):
    """combine n split files into main h5"""

    split_paths = [h5_path.replace(".h5", f"_{i}.h5") for i in range(n_splits)]

    # verify all splits exist
    for sp in split_paths:
        if not os.path.exists(sp):
            print(f"error: split file not found: {sp}")
            sys.exit(1)

    # collect structure from first split
    with h5py.File(split_paths[0], "r") as f:
        keys = list(f.keys())
        groups = [k for k in keys if isinstance(f[k], h5py.Group)]
        datasets = [k for k in keys if not isinstance(f[k], h5py.Group)]

    print(f"combining {n_splits} splits into {h5_path}")
    print(f"datasets: {datasets}")
    print(f"groups: {groups}")

    # collect data from all splits
    data = {k: [] for k in datasets}
    group_data = {g: {} for g in groups}

    for i, sp in enumerate(split_paths):
        print(f"reading {sp}...")
        with h5py.File(sp, "r") as f:
            # datasets
            for key in datasets:
                data[key].append(f[key][:])

            # groups
            for grp_name in groups:
                grp = f[grp_name]
                for ds_name in grp.keys():
                    full_key = f"{grp_name}/{ds_name}"
                    if full_key not in group_data[grp_name]:
                        group_data[grp_name][ds_name] = []
                    group_data[grp_name][ds_name].append(grp[ds_name][:])

    # write combined file
    print(f"writing {h5_path}...")
    with h5py.File(h5_path, "w") as f:
        # write datasets
        for key in datasets:
            combined = np.concatenate(data[key], axis=0)
            f.create_dataset(key, data=combined, compression="gzip")
            print(f"  {key}: {combined.shape}")

        # write groups
        for grp_name in groups:
            grp = f.create_group(grp_name)
            for ds_name, arrays in group_data[grp_name].items():
                combined = np.concatenate(arrays, axis=0)
                grp.create_dataset(ds_name, data=combined, compression="gzip")
                print(f"  {grp_name}/{ds_name}: {combined.shape}")

        # copy attrs from first split (excluding split-specific ones)
        with h5py.File(split_paths[0], "r") as src:
            for k, v in src.attrs.items():
                if not k.startswith("split_"):
                    f.attrs[k] = v

    n_total = len(data[datasets[0]])
    print(f"combined {n_total:,} samples")

    # cleanup
    if delete_splits:
        for sp in split_paths:
            os.remove(sp)
            print(f"deleted {sp}")

    print("done")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python combine_h5.py <h5_file> <n_splits> [--keep]")
        sys.exit(1)

    delete = "--keep" not in sys.argv
    combine_h5(sys.argv[1], int(sys.argv[2]), delete_splits=delete)
