#!/usr/bin/env python3
"""score sqtl vcf with spliceai models

usage: python score_sa.py input.vcf.gz reference.fa output.h5
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import h5py
import pysam
import tensorflow as tf
from tensorflow.keras.models import load_model
from importlib.resources import files as _pkg_files
def resource_filename(pkg, path): return str(_pkg_files(pkg).joinpath(path))
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_vcf, onehot, extract_sequences

seq_len = 20001
batch_size = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vcf")
    ap.add_argument("fasta")
    ap.add_argument("output_h5")
    ap.add_argument("--batch-size", type=int, default=batch_size)
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"gpu: {gpus[0].name if gpus else 'none'}", flush=True)

    # load vcf
    print(f"loading {args.input_vcf}", flush=True)
    vcf_df = load_vcf(args.input_vcf)
    n = len(vcf_df)
    print(f"variants: {n:,}")

    # extract sequences (reverse complemented for - strand)
    ref_seqs, alt_seqs, strands = extract_sequences(vcf_df, args.fasta)

    # one-hot encode
    ref_oh = np.stack([onehot(s) for s in ref_seqs], axis=0)
    alt_oh = np.stack([onehot(s) for s in alt_seqs], axis=0)

    # load 5 spliceai models
    print("loading spliceai models...", flush=True)
    models = []
    for i in range(1, 6):
        path = resource_filename("spliceai", f"models/spliceai{i}.h5")
        models.append(load_model(path, compile=False))

    # score ref
    print("scoring ref...", flush=True)
    ref_preds = []
    for m in tqdm(models, desc="ref"):
        p = m.predict(ref_oh, batch_size=args.batch_size, verbose=0)
        ref_preds.append(p)
    ref_out = np.mean(np.stack(ref_preds, axis=0), axis=0).astype(np.float32)

    # score alt
    print("scoring alt...", flush=True)
    alt_preds = []
    for m in tqdm(models, desc="alt"):
        p = m.predict(alt_oh, batch_size=args.batch_size, verbose=0)
        alt_preds.append(p)
    alt_out = np.mean(np.stack(alt_preds, axis=0), axis=0).astype(np.float32)

    # reverse outputs for - strand variants to align with genomic coordinates
    for i, strand in enumerate(strands):
        if strand == "-":
            ref_out[i] = ref_out[i, ::-1, :]
            alt_out[i] = alt_out[i, ::-1, :]

    # write output
    print(f"writing {args.output_h5}", flush=True)
    with h5py.File(args.output_h5, "w") as f:
        f.create_dataset("ref", data=ref_out, compression=None)
        f.create_dataset("alt", data=alt_out, compression=None)

        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("var_key", data=vcf_df["var_key"].values.astype(object), dtype=str_dt)

    print(f"done: {n:,} variants")


if __name__ == "__main__":
    main()
