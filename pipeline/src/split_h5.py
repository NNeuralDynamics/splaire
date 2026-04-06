#!/usr/bin/env python3
"""split a donor's full h5 into per-fold train/valid h5s using gc transcript ids"""

import argparse
import re
import logging
from pathlib import Path

import numpy as np
import h5py

CHUNK_SIZE = 2000


def count_chunks(h5f):
    """number of X* datasets in an h5 file"""
    return sum(1 for k in h5f.keys() if k.startswith("X"))


def load_fold_uids(fold_tsvs):
    """load unique_ids from each fold tsv, returns {fold_name: set of uids}"""
    import pandas as pd
    folds = {}
    for path in fold_tsvs:
        # parse fold name from filename like split1_train.tsv, split2_validation.tsv
        m = re.match(r'(split\d+)_(train|validation)\.tsv', Path(path).name)
        if not m:
            logging.warning(f"skipping unrecognized fold file: {path}")
            continue
        split_num, split_type = m.group(1), m.group(2)
        label = 'valid' if split_type == 'validation' else 'train'
        fold_name = f"{label}_{split_num}"

        df = pd.read_csv(path, sep='\t', usecols=['Unique_ID'])
        folds[fold_name] = set(df['Unique_ID'].astype(str))
        logging.info(f"{fold_name}: {len(folds[fold_name]):,} transcript ids")
    return folds


def flush_chunk(h5f, buf_x, buf_y, buf_gc, chunk_idx, keep_gc):
    """write buffered arrays as a chunk"""
    x = np.concatenate(buf_x, axis=0)
    y = np.concatenate(buf_y, axis=0)
    h5f.create_dataset(f"X{chunk_idx}", data=x, compression="gzip", compression_opts=4)
    h5f.create_dataset(f"Y{chunk_idx}", data=y, compression="gzip", compression_opts=4)
    if keep_gc and buf_gc:
        gc = np.concatenate(buf_gc, axis=0)
        h5f.create_dataset(f"GC{chunk_idx}", data=gc, compression="gzip", compression_opts=4)
    return chunk_idx + 1


def split_h5(input_path, fold_tsvs, output_dir, donor, keep_gc):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = load_fold_uids(fold_tsvs)
    if not folds:
        logging.error("no valid fold tsvs found")
        return

    with h5py.File(input_path, 'r') as tmp:
        n_chunks_in = count_chunks(tmp)
    logging.info(f"input h5: {n_chunks_in} chunks")

    # open all output files
    writers = {}
    buffers = {}
    chunk_idxs = {}
    window_counts = {}
    for name in folds:
        out_path = output_dir / f"{name}_{donor}.h5"
        writers[name] = h5py.File(out_path, 'w')
        buffers[name] = ([], [], [])  # x, y, gc
        chunk_idxs[name] = 0
        window_counts[name] = 0

    with h5py.File(input_path, 'r') as src:
        for ci in range(n_chunks_in):
            gc = src[f"GC{ci}"][:]
            x = src[f"X{ci}"][:]
            y = src[f"Y{ci}"][:]

            # transcript id from first position in each window
            window_uids = gc["name"][:, 0].astype(str)

            for fold_name, uid_set in folds.items():
                mask = np.array([u in uid_set for u in window_uids])
                if not mask.any():
                    continue

                bx, by, bgc = buffers[fold_name]
                bx.append(x[mask])
                by.append(y[mask])
                bgc.append(gc[mask])
                window_counts[fold_name] += mask.sum()

                # flush at chunk size
                total = sum(len(b) for b in bx)
                if total >= CHUNK_SIZE:
                    chunk_idxs[fold_name] = flush_chunk(
                        writers[fold_name], bx, by, bgc,
                        chunk_idxs[fold_name], keep_gc
                    )
                    buffers[fold_name] = ([], [], [])

    # final flush + close
    for name in folds:
        bx, by, bgc = buffers[name]
        if bx:
            flush_chunk(writers[name], bx, by, bgc, chunk_idxs[name], keep_gc)
        writers[name].close()
        logging.info(f"{name}: {window_counts[name]:,} windows, {chunk_idxs[name] + (1 if bx else 0)} chunks")

    logging.info(f"done, wrote {len(folds)} fold h5s for {donor}")


def main():
    parser = argparse.ArgumentParser(description="split full donor h5 into fold-specific h5s")
    parser.add_argument("--input", required=True, help="full donor h5")
    parser.add_argument("--fold-tsvs", required=True, nargs='+', help="fold tsv files from generate_train_val_splits")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--donor", required=True, help="donor id for output filenames")
    parser.add_argument("--keep-gc", action="store_true", help="keep gc datasets in output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    split_h5(args.input, args.fold_tsvs, args.output_dir, args.donor, args.keep_gc)


if __name__ == "__main__":
    main()
