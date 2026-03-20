#!/usr/bin/env python3
"""
split h5 file into n parts for parallel processing

copies ALL datasets and attributes to each split.
handles vlen arrays with NULL entries by rebuilding them.

usage:
    python split_h5.py sequences.h5 4
    # creates: sequences_0.h5, sequences_1.h5, sequences_2.h5, sequences_3.h5
"""

import sys
import h5py
import numpy as np


def split_h5(h5_path, n_splits):
    """split h5 into n equal parts, preserving all metadata"""

    with h5py.File(h5_path, "r") as f:
        # get total samples from X
        n_total = f["X"].shape[0]
        chunk_size = n_total // n_splits

        print(f"splitting {h5_path} into {n_splits} parts")
        print(f"total samples: {n_total}, ~{chunk_size} per split")

        for i in range(n_splits):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_splits - 1 else n_total

            out_path = h5_path.replace(".h5", f"_{i}.h5")
            print(f"\n{out_path}: indices {start}:{end} ({end-start} samples)")

            with h5py.File(out_path, "w") as out:
                _copy_recursive(f, out, start, end, prefix="")

                # add split info
                out.attrs["split_idx"] = i
                out.attrs["split_total"] = n_splits
                out.attrs["split_start"] = start
                out.attrs["split_end"] = end

    print(f"\ndone. created {n_splits} files")


def _copy_recursive(src, dst, start, end, prefix):
    """recursively copy h5 structure with slicing"""

    # copy attributes
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    # copy items
    for key in src.keys():
        item = src[key]
        full_key = f"{prefix}/{key}" if prefix else key

        if isinstance(item, h5py.Group):
            grp = dst.create_group(key)
            _copy_recursive(item, grp, start, end, full_key)
        else:
            _copy_dataset(item, dst, key, start, end, full_key)


def _copy_dataset(ds, dst, key, start, end, full_key):
    """copy dataset with proper handling of vlen and regular arrays"""

    vlen_type = h5py.check_vlen_dtype(ds.dtype)

    if vlen_type is not None:
        # vlen arrays with potential NULL entries
        # read one by one, replacing errors with empty arrays
        data = []
        for idx in range(start, end):
            try:
                val = ds[idx]
                if val is None:
                    val = np.array([], dtype=vlen_type)
                data.append(val)
            except (ValueError, OSError):
                # NULL pointer - use empty array
                data.append(np.array([], dtype=vlen_type))

        dst.create_dataset(key, data=data, dtype=ds.dtype)
        print(f"  {full_key}: vlen ({end-start} items)")
    else:
        # regular array - slice directly
        data = ds[start:end]

        kwargs = {}
        if ds.compression:
            kwargs["compression"] = ds.compression
            if ds.compression_opts:
                kwargs["compression_opts"] = ds.compression_opts

        dst.create_dataset(key, data=data, **kwargs)
        print(f"  {full_key}: {ds.shape} -> {data.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python split_h5.py <h5_file> <n_splits>")
        sys.exit(1)

    split_h5(sys.argv[1], int(sys.argv[2]))
