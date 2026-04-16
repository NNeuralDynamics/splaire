#!/usr/bin/env python3
"""score sqtl vcf with splaire models

outputs separate h5 for ref and var models, each with cls and reg predictions
accepts multiple vcf/output pairs to avoid gpu idle between files

usage:
    python score_splaire.py input.vcf.gz reference.fa output_prefix --models-dir /path/to/models
    python score_splaire.py vcf1.gz fasta out1 --models-dir /path --extra vcf2.gz out2
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


def _predict_ensemble(models, oh, bs):
    """run predict on each model in list and return mean"""
    return np.mean([m.predict(oh, batch_size=bs, verbose=0) for m in models], axis=0).astype(np.float32)


def score_vcf(vcf_path, fasta, output_prefix, models, mode="both", minus_flip=True, bs=128, chunk_size=20000):
    """score a single vcf with pre-loaded models, write h5 files

    mode: 'ref', 'var', or 'both'
    processes in chunks of chunk_size to stay within memory
    """
    print(f"loading {vcf_path}", flush=True)
    vcf_df = load_vcf(vcf_path)
    n = len(vcf_df)
    print(f"variants: {n:,}")

    ref_seqs, alt_seqs, strands = extract_sequences(vcf_df, fasta)
    strands = np.array(strands)
    var_keys = vcf_df["var_key"].values.astype(object)
    str_dt = h5py.special_dtype(vlen=str)

    # figure out output shapes from a tiny forward pass
    dummy = np.stack([onehot(ref_seqs[0])], axis=0)
    if mode in ("ref", "both"):
        ref_cls_models, ref_reg_models = models[:2]
        cls_shape_tail = ref_cls_models[0].predict(dummy, batch_size=1, verbose=0).shape[1:]
        reg_shape_tail = ref_reg_models[0].predict(dummy, batch_size=1, verbose=0).squeeze(-1).shape[1:]
    if mode in ("var", "both"):
        var_cls_models, var_reg_models = models[-2:]
        vcls_shape_tail = var_cls_models[0].predict(dummy, batch_size=1, verbose=0).shape[1:]
        vreg_shape_tail = var_reg_models[0].predict(dummy, batch_size=1, verbose=0).squeeze(-1).shape[1:]

    # pre-allocate output arrays
    if mode in ("ref", "both"):
        cls_ref = np.zeros((n, *cls_shape_tail), dtype=np.float32)
        cls_alt = np.zeros((n, *cls_shape_tail), dtype=np.float32)
        reg_ref = np.zeros((n, *reg_shape_tail), dtype=np.float32)
        reg_alt = np.zeros((n, *reg_shape_tail), dtype=np.float32)
    if mode in ("var", "both"):
        vcls_ref = np.zeros((n, *vcls_shape_tail), dtype=np.float32)
        vcls_alt = np.zeros((n, *vcls_shape_tail), dtype=np.float32)
        vreg_ref = np.zeros((n, *vreg_shape_tail), dtype=np.float32)
        vreg_alt = np.zeros((n, *vreg_shape_tail), dtype=np.float32)

    # process in chunks
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        i0 = ci * chunk_size
        i1 = min(i0 + chunk_size, n)
        print(f"chunk {ci+1}/{n_chunks}: variants {i0:,}-{i1:,}", flush=True)

        ref_oh = np.stack([onehot(s) for s in ref_seqs[i0:i1]], axis=0)
        alt_oh = np.stack([onehot(s) for s in alt_seqs[i0:i1]], axis=0)
        minus_mask = strands[i0:i1] == "-"

        if mode in ("ref", "both"):
            cr = _predict_ensemble(ref_cls_models, ref_oh, bs)
            ca = _predict_ensemble(ref_cls_models, alt_oh, bs)
            rr_preds = [m.predict(ref_oh, batch_size=bs, verbose=0) for m in ref_reg_models]
            ra_preds = [m.predict(alt_oh, batch_size=bs, verbose=0) for m in ref_reg_models]
            rr = tf.nn.sigmoid(np.mean(rr_preds, axis=0)).numpy().squeeze(-1).astype(np.float32)
            ra = tf.nn.sigmoid(np.mean(ra_preds, axis=0)).numpy().squeeze(-1).astype(np.float32)
            if minus_mask.any():
                cr[minus_mask] = cr[minus_mask, ::-1, :]
                ca[minus_mask] = ca[minus_mask, ::-1, :]
                rr[minus_mask] = rr[minus_mask, ::-1]
                ra[minus_mask] = ra[minus_mask, ::-1]
            cls_ref[i0:i1] = cr
            cls_alt[i0:i1] = ca
            reg_ref[i0:i1] = rr
            reg_alt[i0:i1] = ra
            del cr, ca, rr, ra, rr_preds, ra_preds

        if mode in ("var", "both"):
            vcr = _predict_ensemble(var_cls_models, ref_oh, bs)
            vca = _predict_ensemble(var_cls_models, alt_oh, bs)
            vrr_preds = [m.predict(ref_oh, batch_size=bs, verbose=0) for m in var_reg_models]
            vra_preds = [m.predict(alt_oh, batch_size=bs, verbose=0) for m in var_reg_models]
            vrr = tf.nn.sigmoid(np.mean(vrr_preds, axis=0)).numpy().squeeze(-1).astype(np.float32)
            vra = tf.nn.sigmoid(np.mean(vra_preds, axis=0)).numpy().squeeze(-1).astype(np.float32)
            if minus_mask.any():
                vcr[minus_mask] = vcr[minus_mask, ::-1, :]
                vca[minus_mask] = vca[minus_mask, ::-1, :]
                vrr[minus_mask] = vrr[minus_mask, ::-1]
                vra[minus_mask] = vra[minus_mask, ::-1]
            vcls_ref[i0:i1] = vcr
            vcls_alt[i0:i1] = vca
            vreg_ref[i0:i1] = vrr
            vreg_alt[i0:i1] = vra
            del vcr, vca, vrr, vra, vrr_preds, vra_preds

        del ref_oh, alt_oh

    # write h5 files
    if mode in ("ref", "both"):
        ref_out = f"{output_prefix}.ref.h5"
        print(f"writing {ref_out}", flush=True)
        with h5py.File(ref_out, "w") as f:
            f.create_dataset("cls_ref", data=cls_ref, compression=None)
            f.create_dataset("cls_alt", data=cls_alt, compression=None)
            f.create_dataset("reg_ref", data=reg_ref, compression=None)
            f.create_dataset("reg_alt", data=reg_alt, compression=None)
            f.create_dataset("var_key", data=var_keys, dtype=str_dt)

    if mode in ("var", "both"):
        var_out = f"{output_prefix}.var.h5"
        print(f"writing {var_out}", flush=True)
        with h5py.File(var_out, "w") as f:
            f.create_dataset("cls_ref", data=vcls_ref, compression=None)
            f.create_dataset("cls_alt", data=vcls_alt, compression=None)
            f.create_dataset("reg_ref", data=vreg_ref, compression=None)
            f.create_dataset("reg_alt", data=vreg_alt, compression=None)
            f.create_dataset("var_key", data=var_keys, dtype=str_dt)

    print(f"done: {n:,} variants", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vcf", help="input vcf(.gz)")
    ap.add_argument("fasta", help="reference genome fasta")
    ap.add_argument("output_prefix", help="output prefix (will create {prefix}.ref.h5, {prefix}.var.h5)")
    ap.add_argument("--models-dir", required=True, help="path to SpHAEC models directory")
    ap.add_argument("--batch-size", type=int, default=batch_size)
    ap.add_argument("--mode", choices=["ref", "var", "both"], default="both",
                    help="which model group to run (default: both)")
    ap.add_argument("--extra", nargs=2, action="append", default=[],
                    metavar=("VCF", "PREFIX"), help="additional vcf + output prefix pairs")
    args = ap.parse_args()

    sys.path.insert(0, args.models_dir)
    from paths import ref_cls, ref_reg, var_cls, var_reg

    gpus = tf.config.list_physical_devices("GPU")
    print(f"gpu: {gpus[0].name if gpus else 'none'}", flush=True)
    print(f"mode: {args.mode}", flush=True)

    # only load models needed for this mode
    print("loading models...", flush=True)
    ref_cls_models = [tf.keras.models.load_model(str(p), compile=False) for p in ref_cls()] if args.mode in ("ref", "both") else []
    ref_reg_models = [tf.keras.models.load_model(str(p), compile=False) for p in ref_reg()] if args.mode in ("ref", "both") else []
    var_cls_models = [tf.keras.models.load_model(str(p), compile=False) for p in var_cls()] if args.mode in ("var", "both") else []
    var_reg_models = [tf.keras.models.load_model(str(p), compile=False) for p in var_reg()] if args.mode in ("var", "both") else []
    models = (ref_cls_models, ref_reg_models, var_cls_models, var_reg_models)
    print("models loaded", flush=True)

    # score primary vcf
    score_vcf(args.input_vcf, args.fasta, args.output_prefix, models, mode=args.mode, bs=args.batch_size)

    # score additional vcfs
    for vcf, prefix in args.extra:
        score_vcf(vcf, args.fasta, prefix, models, mode=args.mode, bs=args.batch_size)


if __name__ == "__main__":
    main()
