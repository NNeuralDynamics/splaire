#!/usr/bin/env python3
"""score h5 with pangolin tissue-specific models (5 reps per task), output to parquet"""

import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import torch
from pkg_resources import resource_filename
from tqdm import tqdm

from pangolin.model import Pangolin, L, W, AR

# task_id to (name, output_channel)
tasks = {
    0: ("heart_p_splice", 1),
    1: ("heart_usage", 2),
    2: ("liver_p_splice", 4),
    3: ("liver_usage", 5),
    4: ("brain_p_splice", 7),
    5: ("brain_usage", 8),
    6: ("testis_p_splice", 10),
    7: ("testis_usage", 11),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", help="input h5 file")
    parser.add_argument("output_parquet", help="output parquet file")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    args = parser.parse_args()

    # setup gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"using gpu: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("warning: no gpu", flush=True)

    # load models
    print("loading pangolin models", flush=True)
    models = {}
    for task_id, (task_name, _) in tasks.items():
        reps = []
        for rep in range(1, 6):
            model = Pangolin(L, W, AR).to(device).eval()
            path = resource_filename("pangolin", f"models/final.{rep}.{task_id}.3")
            model.load_state_dict(torch.load(path, map_location=device))
            reps.append(model)
        models[task_id] = reps

    # warmup
    print("warming up models...", flush=True)
    with torch.no_grad():
        dummy = torch.zeros((1, 4, 15000), dtype=torch.float32, device=device)
        for reps in models.values():
            for m in reps:
                _ = m(dummy)

    print(f"scoring {args.input_h5}", flush=True)

    # accumulate per-task predictions
    task_preds = {tasks[tid][0]: [] for tid in tasks}
    gc_list, y_list = [], []

    with h5py.File(args.input_h5, 'r') as fin:
        chunks = sorted([k for k in fin.keys() if k.startswith('X')])
        print(f"found {len(chunks)} chunks", flush=True)

        for chunk in tqdm(chunks, desc="chunks", file=sys.stdout, mininterval=10):
            x = fin[chunk][:]
            # transpose to channels first: (n, 4, length)
            x_t = x.transpose(0, 2, 1).astype(np.float32)
            n = len(x_t)

            for task_id, reps in models.items():
                task_name, channel_idx = tasks[task_id]
                rep_preds = []

                with torch.no_grad():
                    for model in reps:
                        batches = []
                        for start in range(0, n, args.batch_size):
                            xb = torch.from_numpy(x_t[start:start + args.batch_size]).to(device)
                            out = model(xb)[:, channel_idx, :].cpu().numpy()
                            batches.append(out)
                        rep_preds.append(np.concatenate(batches, axis=0))

                # average over 5 reps: (n, length)
                avg = np.mean(np.stack(rep_preds, axis=0), axis=0).astype(np.float32)
                task_preds[task_name].append(avg)

            gc_key = 'GC' + chunk[1:]
            y_key = 'Y' + chunk[1:]
            if gc_key in fin:
                gc_list.append(fin[gc_key][:].reshape(-1))
            if y_key in fin:
                y_list.append(fin[y_key][:].reshape(-1, 4))

    # filter valid positions and write parquet
    gc = np.concatenate(gc_list)
    y = np.concatenate(y_list).astype(np.float32)
    valid = gc["position"] != -1

    # flatten task predictions and filter valid
    flat = {}
    for task_name, arrs in task_preds.items():
        flat[task_name] = np.concatenate(arrs).flatten()[valid].astype(np.float32)

    # compute averages across tissues
    p_splice_keys = [k for k in flat if "_p_splice" in k]
    usage_keys = [k for k in flat if "_usage" in k]
    flat["avg_p_splice"] = np.mean([flat[k] for k in p_splice_keys], axis=0).astype(np.float32)
    flat["avg_usage"] = np.mean([flat[k] for k in usage_keys], axis=0).astype(np.float32)

    # y columns: [neither, acceptor, donor, ssu]
    data = {
        "chrom": gc["chrom"][valid].astype(np.int8),
        "pos": gc["position"][valid].astype(np.int32),
        "strand": gc["strand"][valid].astype(np.int8),
        "y_acceptor": y[valid, 1],
        "y_donor": y[valid, 2],
        "y_ssu": y[valid, 3],
        **flat,
    }

    df = pd.DataFrame(data)
    df.to_parquet(args.output_parquet, index=False)
    print(f"wrote {args.output_parquet} ({valid.sum():,} rows)", flush=True)


if __name__ == '__main__':
    main()
