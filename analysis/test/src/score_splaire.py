#!/usr/bin/env python3
"""score h5 with splaire models (ref and/or var), output to parquet"""

import sys
import os
import h5py
import argparse
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# add models dir to path for paths.py
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))
from paths import ref_cls, ref_reg, var_cls, var_reg

import tensorflow as tf
from tqdm import tqdm


def load_models(paths, name):
    """load model ensemble"""
    print(f"loading {name}: {len(paths)} models", flush=True)
    return [tf.keras.models.load_model(str(p), compile=False) for p in paths]


def warmup(models):
    """pre-compile predict functions"""
    dummy = np.zeros((1, 15000, 4), dtype=np.float32)
    for m in models:
        m.predict(dummy, verbose=0)


def score_ensemble(dataset, cls_models, reg_models):
    """score with ensemble, return combined predictions"""
    cls_preds = [m.predict(dataset, verbose=0) for m in cls_models]
    cls_combined = np.mean(np.stack(cls_preds, axis=0), axis=0)

    reg_preds = [m.predict(dataset, verbose=0) for m in reg_models]
    reg_combined = np.mean(1 / (1 + np.exp(-np.stack(reg_preds, axis=0))), axis=0)

    if reg_combined.ndim == 2:
        reg_combined = reg_combined[..., None]

    return np.concatenate([cls_combined, reg_combined], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", help="input h5 file")
    parser.add_argument("output_dir", help="output directory for predictions")
    parser.add_argument("--ref", action="store_true", help="score with ref models")
    parser.add_argument("--var", action="store_true", help="score with var models")
    parser.add_argument("--prefix", default="splaire", help="output file prefix (default: splaire)")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    args = parser.parse_args()

    # default to ref if neither specified
    if not args.ref and not args.var:
        args.ref = True

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # setup gpu
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"gpu: {gpus[0].name if gpus else 'none'}", flush=True)

    # load models
    ref_cls_m, ref_reg_m = None, None
    var_cls_m, var_reg_m = None, None

    if args.ref:
        ref_cls_m = load_models(ref_cls(), "ref_cls")
        ref_reg_m = load_models(ref_reg(), "ref_reg")
    if args.var:
        var_cls_m = load_models(var_cls(), "var_cls")
        var_reg_m = load_models(var_reg(), "var_reg")

    # enable xla
    tf.config.optimizer.set_jit(True)

    # warmup
    print("warming up...", flush=True)
    all_models = []
    if args.ref:
        all_models.extend(ref_cls_m + ref_reg_m)
    if args.var:
        all_models.extend(var_cls_m + var_reg_m)
    warmup(all_models)

    print(f"scoring {args.input_h5}", flush=True)

    # filter valid positions per chunk to avoid accumulating full predictions
    ref_rows, var_rows = [], []

    with h5py.File(args.input_h5, 'r') as fin:
        chunks = sorted([k for k in fin.keys() if k.startswith('X')])
        print(f"found {len(chunks)} chunks", flush=True)

        for chunk in tqdm(chunks, desc="chunks", file=sys.stdout, mininterval=10):
            x = fin[chunk][:].astype(np.float32)
            dataset = tf.data.Dataset.from_tensor_slices(x).batch(args.batch_size).prefetch(2)

            ref_pred = score_ensemble(dataset, ref_cls_m, ref_reg_m) if args.ref else None
            var_pred = score_ensemble(dataset, var_cls_m, var_reg_m) if args.var else None

            # filter to valid positions immediately
            gc = fin['GC' + chunk[1:]][:].reshape(-1)
            y = fin['Y' + chunk[1:]][:].reshape(-1, 4).astype(np.float32)
            valid = gc["position"] != -1

            common = {
                "chrom": gc["chrom"][valid].astype(np.int8),
                "pos": gc["position"][valid].astype(np.int32),
                "strand": gc["strand"][valid].astype(np.int8),
                "y_acceptor": y[valid, 1],
                "y_donor": y[valid, 2],
                "y_ssu": y[valid, 3],
            }

            # model outputs: [neither, acceptor, donor, ssu]
            if ref_pred is not None:
                p = ref_pred.reshape(-1, 4)
                ref_rows.append(pd.DataFrame({
                    **common,
                    "acceptor": p[valid, 1],
                    "donor": p[valid, 2],
                    "ssu": p[valid, 3],
                }))
                del ref_pred

            if var_pred is not None:
                p = var_pred.reshape(-1, 4)
                var_rows.append(pd.DataFrame({
                    **common,
                    "acceptor": p[valid, 1],
                    "donor": p[valid, 2],
                    "ssu": p[valid, 3],
                }))
                del var_pred

    def write_parquet(rows, name):
        df = pd.concat(rows, ignore_index=True)
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"wrote {path} ({len(df):,} rows)", flush=True)

    if args.ref:
        write_parquet(ref_rows, f"{args.prefix}_ref")
    if args.var:
        write_parquet(var_rows, f"{args.prefix}_var")

    print("done", flush=True)


if __name__ == '__main__':
    main()
