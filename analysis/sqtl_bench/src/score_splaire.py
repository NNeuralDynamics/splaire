#!/usr/bin/env python3
"""score sqtl vcf with splaire models

outputs separate h5 for ref and var models, each with cls and reg predictions

usage: python score_splaire.py input.vcf.gz reference.fa output_prefix --models-dir /path/to/models
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import h5py
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_vcf, onehot, extract_sequences

batch_size = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vcf", help="input vcf(.gz)")
    ap.add_argument("fasta", help="reference genome fasta")
    ap.add_argument("output_prefix", help="output prefix (will create {prefix}.ref.h5, {prefix}.var.h5)")
    ap.add_argument("--models-dir", required=True, help="path to SpHAEC models directory")
    ap.add_argument("--batch-size", type=int, default=batch_size)
    args = ap.parse_args()

    # import model paths
    sys.path.insert(0, args.models_dir)
    from paths import ref_cls, ref_reg, var_cls, var_reg

    gpus = tf.config.list_physical_devices("GPU")
    print(f"gpu: {gpus[0].name if gpus else 'none'}", flush=True)

    # load vcf
    print(f"loading {args.input_vcf}", flush=True)
    vcf_df = load_vcf(args.input_vcf)
    n = len(vcf_df)
    print(f"variants: {n:,}")

    # extract sequences (reverse complemented for - strand)
    ref_seqs, alt_seqs, strands = extract_sequences(vcf_df, args.fasta)
    strands = np.array(strands)
    minus_mask = strands == "-"

    # one-hot encode
    print("one-hot encoding...", flush=True)
    ref_oh = np.stack([onehot(s) for s in ref_seqs], axis=0)
    alt_oh = np.stack([onehot(s) for s in alt_seqs], axis=0)

    # score with ref models
    print("loading ref models...", flush=True)
    ref_cls_models = [tf.keras.models.load_model(str(p), compile=False) for p in ref_cls()]
    ref_reg_models = [tf.keras.models.load_model(str(p), compile=False) for p in ref_reg()]

    print("scoring with ref models...", flush=True)
    # cls output (n, seq_len, 3)
    cls_ref = np.mean([m.predict(ref_oh, batch_size=args.batch_size, verbose=0) for m in ref_cls_models], axis=0).astype(np.float32)
    cls_alt = np.mean([m.predict(alt_oh, batch_size=args.batch_size, verbose=0) for m in ref_cls_models], axis=0).astype(np.float32)

    # reg output (n, seq_len, 1)
    reg_preds_ref = [m.predict(ref_oh, batch_size=args.batch_size, verbose=0) for m in ref_reg_models]
    reg_preds_alt = [m.predict(alt_oh, batch_size=args.batch_size, verbose=0) for m in ref_reg_models]
    reg_ref = tf.nn.sigmoid(np.mean(reg_preds_ref, axis=0)).numpy().squeeze(-1).astype(np.float32)  # (n, seq_len)
    reg_alt = tf.nn.sigmoid(np.mean(reg_preds_alt, axis=0)).numpy().squeeze(-1).astype(np.float32)  # (n, seq_len)

    # reverse outputs for - strand variants to align with genomic coordinates
    if minus_mask.any():
        cls_ref[minus_mask] = cls_ref[minus_mask, ::-1, :]
        cls_alt[minus_mask] = cls_alt[minus_mask, ::-1, :]
        reg_ref[minus_mask] = reg_ref[minus_mask, ::-1]
        reg_alt[minus_mask] = reg_alt[minus_mask, ::-1]

    # write ref model output
    ref_out = f"{args.output_prefix}.ref.h5"
    print(f"writing {ref_out}", flush=True)
    with h5py.File(ref_out, "w") as f:
        f.create_dataset("cls_ref", data=cls_ref, compression="gzip")
        f.create_dataset("cls_alt", data=cls_alt, compression="gzip")
        f.create_dataset("reg_ref", data=reg_ref, compression="gzip")
        f.create_dataset("reg_alt", data=reg_alt, compression="gzip")
        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("var_key", data=vcf_df["var_key"].values.astype(object), dtype=str_dt)

    # clear ref models
    del ref_cls_models, ref_reg_models

    # score with var models
    print("loading var models...", flush=True)
    var_cls_models = [tf.keras.models.load_model(str(p), compile=False) for p in var_cls()]
    var_reg_models = [tf.keras.models.load_model(str(p), compile=False) for p in var_reg()]

    print("scoring with var models...", flush=True)
    cls_ref = np.mean([m.predict(ref_oh, batch_size=args.batch_size, verbose=0) for m in var_cls_models], axis=0).astype(np.float32)
    cls_alt = np.mean([m.predict(alt_oh, batch_size=args.batch_size, verbose=0) for m in var_cls_models], axis=0).astype(np.float32)

    reg_preds_ref = [m.predict(ref_oh, batch_size=args.batch_size, verbose=0) for m in var_reg_models]
    reg_preds_alt = [m.predict(alt_oh, batch_size=args.batch_size, verbose=0) for m in var_reg_models]
    reg_ref = tf.nn.sigmoid(np.mean(reg_preds_ref, axis=0)).numpy().squeeze(-1).astype(np.float32)  # (n, seq_len)
    reg_alt = tf.nn.sigmoid(np.mean(reg_preds_alt, axis=0)).numpy().squeeze(-1).astype(np.float32)  # (n, seq_len)

    # reverse outputs for - strand variants to align with genomic coordinates
    if minus_mask.any():
        cls_ref[minus_mask] = cls_ref[minus_mask, ::-1, :]
        cls_alt[minus_mask] = cls_alt[minus_mask, ::-1, :]
        reg_ref[minus_mask] = reg_ref[minus_mask, ::-1]
        reg_alt[minus_mask] = reg_alt[minus_mask, ::-1]

    # write var model output
    var_out = f"{args.output_prefix}.var.h5"
    print(f"writing {var_out}", flush=True)
    with h5py.File(var_out, "w") as f:
        f.create_dataset("cls_ref", data=cls_ref, compression="gzip")
        f.create_dataset("cls_alt", data=cls_alt, compression="gzip")
        f.create_dataset("reg_ref", data=reg_ref, compression="gzip")
        f.create_dataset("reg_alt", data=reg_alt, compression="gzip")
        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("var_key", data=vcf_df["var_key"].values.astype(object), dtype=str_dt)

    print(f"done: {n:,} variants")


if __name__ == "__main__":
    main()
