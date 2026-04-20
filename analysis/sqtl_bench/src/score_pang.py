#!/usr/bin/env python3
"""score sqtl vcf with pangolin models

usage: python score_pang.py input.vcf.gz reference.fa output.h5
       python score_pang.py input.vcf.gz reference.fa output.h5 --v2
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import h5py
import pysam
import torch
from importlib.resources import files as _pkg_files
def resource_filename(pkg, path): return str(_pkg_files(pkg).joinpath(path))
from pangolin.model import Pangolin, L, W, AR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_vcf, onehot, extract_sequences

seq_len = 20001
batch_size = 256

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
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vcf")
    ap.add_argument("fasta")
    ap.add_argument("output_h5")
    ap.add_argument("--batch-size", type=int, default=batch_size)
    ap.add_argument("--v2", action="store_true", help="use v2 weights (human-finetuned, 3 reps)")
    args = ap.parse_args()

    n_reps = 3 if args.v2 else 5
    weight_suffix = ".v2" if args.v2 else ""
    if args.v2:
        print("using v2 weights (human-finetuned, 3 reps)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # load vcf
    print(f"loading {args.input_vcf}", flush=True)
    vcf_df = load_vcf(args.input_vcf)
    n = len(vcf_df)
    print(f"variants: {n:,}")

    # extract sequences (reverse complemented for - strand)
    ref_seqs, alt_seqs, strands = extract_sequences(vcf_df, args.fasta)

    # one-hot encode and transpose for pangolin (n, 4, seq_len)
    print("one-hot encoding...", flush=True)
    ref_oh = np.stack([onehot(s) for s in ref_seqs], axis=0).transpose(0, 2, 1)
    alt_oh = np.stack([onehot(s) for s in alt_seqs], axis=0).transpose(0, 2, 1)
    out_len = seq_len - 10000
    strands = np.array(strands)  # for vectorized reversal later

    # load models
    print("loading pangolin models...", flush=True)
    models = {}
    for task_id in tasks.keys():
        reps = []
        for rep in range(1, n_reps + 1):
            m = Pangolin(L, W, AR).to(device).eval()
            path = resource_filename("pangolin", f"models/final.{rep}.{task_id}.3{weight_suffix}")
            m.load_state_dict(torch.load(path, map_location=device))
            reps.append(m)
        models[task_id] = reps

    # score each task
    results = {}
    for task_id, (task_name, channel) in tqdm(tasks.items(), desc="tasks"):
        reps = models[task_id]

        for prefix, x in [("ref", ref_oh), ("alt", alt_oh)]:
            preds = np.zeros((n, out_len), dtype=np.float32)

            for model in reps:
                for i in range(0, n, args.batch_size):
                    j = min(i + args.batch_size, n)
                    xb = torch.from_numpy(x[i:j]).to(device)
                    with torch.no_grad():
                        yb = model(xb)[:, channel, :].cpu().numpy()
                    preds[i:j] += yb

            preds /= n_reps
            results[f"{prefix}_{task_name}"] = preds

    # reverse outputs for - strand variants to align with genomic coordinates
    minus_mask = strands == "-"
    if minus_mask.any():
        for name, data in results.items():
            data[minus_mask] = data[minus_mask, ::-1]

    # write output
    print(f"writing {args.output_h5}", flush=True)
    with h5py.File(args.output_h5, "w") as f:
        for name, data in results.items():
            f.create_dataset(name, data=data, compression=None)

        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("var_key", data=vcf_df["var_key"].values.astype(object), dtype=str_dt)

    print(f"done: {n:,} variants")


if __name__ == "__main__":
    main()
