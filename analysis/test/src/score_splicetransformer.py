#!/usr/bin/env python3
"""score h5 with splicetransformer model, output to parquet"""

import os
import sys
from pathlib import Path

import h5py
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# add splicetransformer repo to path
script_dir = Path(__file__).parent
spt_path = script_dir.parent.parent / "other_models" / "SpliceTransformer"
sys.path.insert(0, str(spt_path))

from sptransformer import Annotator

tissues = ["adipose", "blood", "blood_vessel", "brain", "colon", "heart", "kidney",
           "liver", "lung", "muscle", "nerve", "small_intestine", "skin", "spleen", "stomach"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", help="input h5 file")
    parser.add_argument("output_parquet", help="output parquet file")
    parser.add_argument("--batch-size", type=int, default=2, help="batch size (small due to memory)")
    args = parser.parse_args()

    # setup gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"using gpu: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("warning: no gpu", flush=True)

    # load model, chdir to repo so annotator finds weights
    print("loading splicetransformer", flush=True)
    prev_dir = os.getcwd()
    os.chdir(spt_path)
    try:
        annotator = Annotator()
        model = annotator.model
        if hasattr(model, "eval"):
            model.eval()
    finally:
        os.chdir(prev_dir)

    # warmup
    print("warming up model...", flush=True)
    with torch.no_grad():
        dummy = torch.zeros((1, 4, 13000), dtype=torch.float32, device=device)
        _ = model.step(dummy)

    print(f"scoring {args.input_h5}", flush=True)

    pred_list, gc_list, y_list = [], [], []

    with h5py.File(args.input_h5, 'r') as fin:
        chunks = sorted([k for k in fin.keys() if k.startswith('X')])
        print(f"found {len(chunks)} chunks", flush=True)

        for chunk in tqdm(chunks, desc="chunks", file=sys.stdout, mininterval=10):
            x = fin[chunk][:]

            # trim to 13000bp and transpose to channels first: (n, 4, 13000)
            x = x[:, 1000:-1000, :].transpose(0, 2, 1).astype(np.float32)
            n = len(x)

            # score in batches
            preds = []
            with torch.no_grad():
                for start in range(0, n, args.batch_size):
                    xb = torch.from_numpy(x[start:start + args.batch_size]).to(device)
                    out = model.step(xb)
                    preds.append(out.cpu().numpy())

            combined = np.concatenate(preds, axis=0).astype(np.float32)
            pred_list.append(combined)

            gc_key = 'GC' + chunk[1:]
            y_key = 'Y' + chunk[1:]
            if gc_key in fin:
                gc_list.append(fin[gc_key][:].reshape(-1))
            if y_key in fin:
                y_list.append(fin[y_key][:].reshape(-1, 4))

    # transpose (n, 18, 5000) -> (n*5000, 18)
    gc = np.concatenate(gc_list)
    y = np.concatenate(y_list).astype(np.float32)
    all_preds = np.concatenate(pred_list, axis=0)
    n_seqs, n_out, seq_len = all_preds.shape

    p = np.empty((n_seqs * seq_len, n_out), dtype=np.float32)
    for i in range(n_seqs):
        p[i * seq_len:(i + 1) * seq_len] = all_preds[i].T
    del all_preds

    valid = gc["position"] != -1

    # y columns: [neither, acceptor, donor, ssu]
    data = {
        "chrom": gc["chrom"][valid].astype(np.int8),
        "pos": gc["position"][valid].astype(np.int32),
        "strand": gc["strand"][valid].astype(np.int8),
        "y_acceptor": y[valid, 1],
        "y_donor": y[valid, 2],
        "y_ssu": y[valid, 3],
        "acceptor": p[valid, 1],
        "donor": p[valid, 2],
    }
    for i, t in enumerate(tissues):
        data[t] = p[valid, 3 + i]
    data["avg_tissue"] = p[valid, 3:18].mean(axis=1).astype(np.float32)

    df = pd.DataFrame(data)
    df.to_parquet(args.output_parquet, index=False)
    print(f"wrote {args.output_parquet} ({valid.sum():,} rows)", flush=True)


if __name__ == '__main__':
    main()
