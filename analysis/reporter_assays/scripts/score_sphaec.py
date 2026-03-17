#!/usr/bin/env python3
"""score reporter assay variants with splaire"""
import argparse
import math
import os
import sys

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# add models to path for paths.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
from paths import ref_cls, ref_reg, var_cls, var_reg

batch_size = int(os.environ.get("SPHAEC_BS", "128"))
use_mixed = bool(int(os.environ.get("SPHAEC_MIXED", "1")))
use_xla = bool(int(os.environ.get("SPHAEC_XLA", "1")))

# tf perf setup
if use_xla:
    tf.config.optimizer.set_jit(True)
if use_mixed and tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input h5 file with pre-encoded sequences")
    p.add_argument("--output", required=True, help="output h5 file for scores")
    p.add_argument("--variant-type", choices=["ref", "var"], default="ref",
                   help="model variant type (ref or var)")
    p.add_argument("--batch-size", type=int, default=batch_size)
    args = p.parse_args()

    assert os.path.exists(args.input), f"input not found: {args.input}"

    # load pre-encoded sequences
    print(f"loading {args.input}")
    with h5py.File(args.input, "r") as f:
        seqs = {
            "exon_start_ref": f["seqs/exon_start_ref"][:],
            "exon_start_alt": f["seqs/exon_start_alt"][:],
            "exon_end_ref": f["seqs/exon_end_ref"][:],
            "exon_end_alt": f["seqs/exon_end_alt"][:],
        }
        # load metadata for copying to output
        meta = {}
        for key in f["meta"].keys():
            meta[key] = f["meta"][key][:]
        input_attrs = dict(f.attrs)

    n = len(seqs["exon_start_ref"])
    print(f"  {n:,} variants")

    # load models
    if args.variant_type == "ref":
        cls_paths = ref_cls("keras")
        reg_paths = ref_reg("keras")
    else:
        cls_paths = var_cls("keras")
        reg_paths = var_reg("keras")

    print(f"loading {args.variant_type} models")
    cls_models = [load_model(str(p), compile=False) for p in tqdm(cls_paths, desc="load cls")]
    reg_models = [load_model(str(p), compile=False) for p in tqdm(reg_paths, desc="load reg")]

    use_fp16 = use_mixed and tf.config.list_physical_devices("GPU")

    @tf.function(jit_compile=use_xla)
    def run_ensembles_center(x):
        # cls ensemble: average center predictions across models -> (B, 3)
        cls_sum = tf.zeros((tf.shape(x)[0], 3), dtype=tf.float32)
        for m in cls_models:
            pred = m(x, training=False)
            pred = tf.cast(pred, tf.float32)
            kc = tf.shape(pred)[1] // 2
            cls_sum += pred[:, kc, :]
        cls_mean = cls_sum / tf.cast(len(cls_models), tf.float32)

        # reg ensemble: average center predictions with sigmoid -> (B,)
        reg_sum = tf.zeros((tf.shape(x)[0],), dtype=tf.float32)
        for m in reg_models:
            q = m(x, training=False)
            q = tf.nn.sigmoid(q)
            q = tf.cast(q, tf.float32)
            kr = tf.shape(q)[1] // 2
            reg_sum += tf.squeeze(q[:, kr, :], axis=-1)
        reg_mean = reg_sum / tf.cast(len(reg_models), tf.float32)
        return cls_mean, reg_mean

    # score each site/allele combination
    combos = [
        ("exon_start", "ref", seqs["exon_start_ref"]),
        ("exon_start", "alt", seqs["exon_start_alt"]),
        ("exon_end", "ref", seqs["exon_end_ref"]),
        ("exon_end", "alt", seqs["exon_end_alt"]),
    ]

    heads = ["cls_neither", "cls_acceptor", "cls_donor", "reg_ssu"]
    all_scores = {h: {} for h in heads}

    for site, allele, X_all in combos:
        key = f"{site}_{allele}"
        print(f"scoring {key}")

        scores = {h: [] for h in heads}
        steps = math.ceil(n / args.batch_size)

        for step in tqdm(range(steps), desc=key):
            i = step * args.batch_size
            j = min((step + 1) * args.batch_size, n)
            X = X_all[i:j]
            if use_fp16:
                X = X.astype(np.float16)
            X = tf.convert_to_tensor(X)

            cls_center, reg_center = run_ensembles_center(X)
            cls_center = cls_center.numpy()
            reg_center = reg_center.numpy()

            scores["cls_neither"].append(cls_center[:, 0])
            scores["cls_acceptor"].append(cls_center[:, 1])
            scores["cls_donor"].append(cls_center[:, 2])
            scores["reg_ssu"].append(reg_center)

        for h in heads:
            all_scores[h][key] = np.concatenate(scores[h]).astype(np.float32)

    # save to h5
    print(f"saving to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as f:
        # scores group - raw ref/alt scores for each head and site
        scores_grp = f.create_group("scores")
        for h in heads:
            for site in ("exon_start", "exon_end"):
                for allele in ("ref", "alt"):
                    key = f"{site}_{allele}"
                    scores_grp.create_dataset(f"{h}_{key}", data=all_scores[h][key])

        # copy metadata from input
        meta_grp = f.create_group("meta")
        for key, arr in meta.items():
            meta_grp.create_dataset(key, data=arr)

        # attributes
        f.attrs["model"] = "sphaec"
        f.attrs["variant_type"] = args.variant_type
        f.attrs["n_variants"] = n
        if "seq_len" in input_attrs:
            f.attrs["seq_len"] = input_attrs["seq_len"]

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
