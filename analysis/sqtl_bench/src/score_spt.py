#!/usr/bin/env python3
"""score sqtl vcf with splicetransformer

trims 1000bp from each end of 20001bp input for 18001bp sequences

usage: python score_spt.py input.vcf.gz reference.fa output.h5 --spt-dir /path/to/SpliceTransformer
"""
import os
import sys
from pathlib import Path

import argparse
import numpy as np
import h5py
import pysam
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_vcf, onehot, extract_sequences

seq_len = 20001
batch_size = 64
trim = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_vcf")
    ap.add_argument("fasta")
    ap.add_argument("output_h5")
    ap.add_argument("--spt-dir", required=True, help="path to SpliceTransformer directory")
    ap.add_argument("--batch-size", type=int, default=batch_size)
    args = ap.parse_args()

    spt_path = Path(args.spt_dir)
    sys.path.insert(0, str(spt_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # load vcf
    print(f"loading {args.input_vcf}", flush=True)
    vcf_df = load_vcf(args.input_vcf)
    n = len(vcf_df)
    print(f"variants: {n:,}")

    # extract sequences (reverse complemented for - strand)
    ref_seqs, alt_seqs, strands = extract_sequences(vcf_df, args.fasta)
    strands = np.array(strands)

    # one-hot encode
    print("one-hot encoding...", flush=True)
    ref_oh = np.stack([onehot(s) for s in ref_seqs], axis=0)
    alt_oh = np.stack([onehot(s) for s in alt_seqs], axis=0)

    # trim 1000bp from each end: 20001 -> 18001
    ref_trim = ref_oh[:, trim:-trim, :]
    alt_trim = alt_oh[:, trim:-trim, :]
    out_len = seq_len - 2 * trim

    # transpose for spt: (n, 4, seq_len)
    ref_t = ref_trim.transpose(0, 2, 1)
    alt_t = alt_trim.transpose(0, 2, 1)

    # load model
    print("loading splicetransformer...", flush=True)
    prev_dir = os.getcwd()
    os.chdir(spt_path)
    try:
        from sptransformer import Annotator
        annotator = Annotator()
        model = annotator.model
        if hasattr(model, "eval"):
            model.eval()
    finally:
        os.chdir(prev_dir)

    # score ref
    print("scoring ref...", flush=True)
    ref_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, n, args.batch_size)):
            j = min(i + args.batch_size, n)
            xb = torch.from_numpy(ref_t[i:j]).to(device)
            yb = model.step(xb).cpu().numpy()
            ref_preds.append(yb)
    ref_out = np.concatenate(ref_preds, axis=0).astype(np.float32)

    # score alt
    print("scoring alt...", flush=True)
    alt_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, n, args.batch_size)):
            j = min(i + args.batch_size, n)
            xb = torch.from_numpy(alt_t[i:j]).to(device)
            yb = model.step(xb).cpu().numpy()
            alt_preds.append(yb)
    alt_out = np.concatenate(alt_preds, axis=0).astype(np.float32)

    # reverse outputs for - strand variants to align with genomic coordinates
    # spt output shape is (n, channels, seq_len), reverse on seq dimension
    minus_mask = strands == "-"
    if minus_mask.any():
        ref_out[minus_mask] = ref_out[minus_mask, :, ::-1]
        alt_out[minus_mask] = alt_out[minus_mask, :, ::-1]

    # write output
    print(f"writing {args.output_h5}", flush=True)
    with h5py.File(args.output_h5, "w") as f:
        f.create_dataset("ref", data=ref_out, compression="gzip")
        f.create_dataset("alt", data=alt_out, compression="gzip")

        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("var_key", data=vcf_df["var_key"].values.astype(object), dtype=str_dt)

    print(f"done: {n:,} variants")


if __name__ == "__main__":
    main()
