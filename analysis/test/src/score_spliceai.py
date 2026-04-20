#!/usr/bin/env python3
"""score h5 with spliceai ensemble (5 models), output to parquet"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from importlib.resources import files as _pkg_files
def resource_filename(pkg, path): return str(_pkg_files(pkg).joinpath(path))
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", help="input h5 file")
    parser.add_argument("output_parquet", help="output parquet file")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    args = parser.parse_args()

    # setup gpu
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"using gpu: {gpus[0].name}" if gpus else "warning: no gpu", flush=True)

    # load models
    print("loading spliceai models", flush=True)
    models = []
    for i in range(1, 6):
        path = resource_filename("spliceai", f"models/spliceai{i}.h5")
        models.append(load_model(path, compile=False))

    # enable xla
    tf.config.optimizer.set_jit(True)

    # warmup
    print("warming up models...", flush=True)
    dummy = np.zeros((1, 15000, 4), dtype=np.float32)
    for m in models:
        m.predict(dummy, verbose=0)

    print(f"scoring {args.input_h5}", flush=True)

    pred_list, gc_list, y_list = [], [], []

    with h5py.File(args.input_h5, 'r') as fin:
        chunks = sorted([k for k in fin.keys() if k.startswith('X')])
        print(f"found {len(chunks)} chunks", flush=True)

        for chunk in tqdm(chunks, desc="chunks", file=sys.stdout, mininterval=10):
            x = fin[chunk][:].astype(np.float32)
            dataset = tf.data.Dataset.from_tensor_slices(x).batch(args.batch_size).prefetch(2)

            all_preds = [m.predict(dataset, verbose=0) for m in models]
            combined = np.mean(np.stack(all_preds, axis=0), axis=0).astype(np.float32)
            pred_list.append(combined)

            gc_key = 'GC' + chunk[1:]
            y_key = 'Y' + chunk[1:]
            if gc_key in fin:
                gc_list.append(fin[gc_key][:].reshape(-1))
            if y_key in fin:
                y_list.append(fin[y_key][:].reshape(-1, 4))

    # filter valid positions and write parquet
    gc = np.concatenate(gc_list)
    y = np.concatenate(y_list).astype(np.float32)
    preds = np.concatenate(pred_list).reshape(-1, 3)
    valid = gc["position"] != -1

    # y columns: [neither, acceptor, donor, ssu]
    df = pd.DataFrame({
        "chrom": gc["chrom"][valid].astype(np.int8),
        "pos": gc["position"][valid].astype(np.int32),
        "strand": gc["strand"][valid].astype(np.int8),
        "y_acceptor": y[valid, 1],
        "y_donor": y[valid, 2],
        "y_ssu": y[valid, 3],
        "acceptor": preds[valid, 1].astype(np.float32),
        "donor": preds[valid, 2].astype(np.float32),
    })

    df.to_parquet(args.output_parquet, index=False)
    print(f"wrote {args.output_parquet} ({valid.sum():,} rows)", flush=True)


if __name__ == '__main__':
    main()
