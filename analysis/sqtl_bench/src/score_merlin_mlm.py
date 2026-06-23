#!/usr/bin/env python3
"""zero-shot score sqtl vcf variants with MLM-pretrained MERLIN

masks the variant position in the ref one-hot and reads
  llr = log P(alt | masked context) - log P(ref | masked context)
from the decoder softmax. SNVs only; indels get NaN.

usage:
    python score_merlin_mlm.py input.vcf.gz reference.fa output.h5 \\
        --checkpoint-mlm /path/to/model_MLM_final_data_sp.pth
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_vcf, onehot, extract_sequences  # noqa: E402

# MLM encoder context = 10000 nt → input = CL+1 = 10001, variant at center index 5000
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "other_models" / "MERLIN"))
from merlin_mlm import load_mlm_from_ckpt, score_llr  # noqa: E402

seq_len = 10001
center_idx = seq_len // 2  # 5000
batch_size = 8

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_vcf")
    ap.add_argument("fasta")
    ap.add_argument("output_h5")
    ap.add_argument("--checkpoint-mlm", required=True)
    ap.add_argument("--batch-size", type=int, default=batch_size)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    print(f"loading {args.input_vcf}", flush=True)
    vcf_df = load_vcf(args.input_vcf)
    n = len(vcf_df)
    print(f"variants: {n:,}")

    # extract 10001 nt windows centered on each variant. extract_sequences
    # reverse-complements - strand so the encoder sees transcript direction;
    # for MLM the LLR is symmetric to revcomp anyway (we score from masked context).
    ref_seqs, alt_seqs, strands = extract_sequences(vcf_df, args.fasta, seq_len=seq_len)
    strands = np.array(strands)

    print("one-hot encoding ref...", flush=True)
    ref_oh = np.stack([onehot(s) for s in ref_seqs], axis=0)

    # SNV-only LLR. take ref/alt allele from the ref sequence (post-revcomp).
    ref_lens = vcf_df["ref"].str.len().values
    alt_lens = vcf_df["alt"].str.len().values
    snv_mask = (ref_lens == 1) & (alt_lens == 1)
    n_indel = int((~snv_mask).sum())
    if n_indel:
        print(f"  {n_indel:,} non-SNV variants → NaN", flush=True)

    # the post-revcomp ref base at center is the ref the encoder sees (transcript direction).
    # we read it directly from the one-hot to stay consistent.
    center_oh_ref = ref_oh[:, center_idx, :]                # (n, 4)
    ref_idx = np.full(n, -1, dtype=np.int64)
    alt_idx = np.full(n, -1, dtype=np.int64)
    ok = center_oh_ref.sum(axis=1) == 1                     # exclude N
    ref_idx[ok & snv_mask] = center_oh_ref[ok & snv_mask].argmax(axis=1)

    # for alt index in transcript direction, look at the alt sequence one-hot center.
    # cheaper than one-hotting the full alt window: just pick the base.
    alt_bases = np.array([s[center_idx] if 0 <= center_idx < len(s) else "N"
                          for s in alt_seqs])
    alt_idx_lookup = np.array([BASE_TO_IDX.get(b, -1) for b in alt_bases], dtype=np.int64)
    alt_idx[snv_mask & ok] = alt_idx_lookup[snv_mask & ok]

    valid = snv_mask & ok & (alt_idx >= 0)
    print(f"  {int(valid.sum()):,} variants will be scored "
          f"({int((~valid & snv_mask).sum()):,} dropped: N at center or alt unmapped)",
          flush=True)

    model = load_mlm_from_ckpt(args.checkpoint_mlm, device)

    llr = np.full(n, np.nan, dtype=np.float32)
    if valid.any():
        idxs = np.where(valid)[0]
        offs = np.full(len(idxs), center_idx, dtype=np.int64)
        out = score_llr(model, ref_oh[idxs], offs, ref_idx[idxs], alt_idx[idxs],
                        device, bs=args.batch_size)
        llr[idxs] = out

    print(f"writing {args.output_h5}", flush=True)
    with h5py.File(args.output_h5, "w") as f:
        f.create_dataset("llr", data=llr.astype(np.float32), compression=None)

        str_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("var_key", data=vcf_df["var_key"].values.astype(object), dtype=str_dt)
        f.attrs["model"] = "merlin_mlm"
        f.attrs["seq_len"] = seq_len
        f.attrs["note"] = ("zero-shot single-position LLR at variant center; "
                           "SNVs only, indels NaN")

    print(f"done: {n:,} variants ({int(valid.sum()):,} scored)", flush=True)


if __name__ == "__main__":
    main()
