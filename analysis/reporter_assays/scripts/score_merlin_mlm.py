#!/usr/bin/env python3
"""zero-shot score reporter assay variants with MLM-pretrained MERLIN

masks the variant position in the ref one-hot and reads
  llr = log P(alt | masked context) - log P(ref | masked context)
from the decoder softmax. one forward pass per variant.

uses the exon_start window by default; falls back to exon_end if the variant
sits outside the start window. SNVs only — indels written as NaN.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# shared MERLIN architecture + mlm head
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "other_models" / "MERLIN"))
from merlin_mlm import load_mlm_from_ckpt  # noqa: E402

BASE_TO_IDX = {b"A": 0, b"C": 1, b"G": 2, b"T": 3}
WINDOW_HALF = 5000      # matches analysis.py
SEQ_LEN_FULL = 10001    # exon_*_{ref,alt} h5 dataset width
CONTEXT = 10000         # merlin CL
TARGET_LEN = CONTEXT + 1  # 10001 — single center prediction, but we read off-center positions too

batch_size = int(os.environ.get("MERLIN_BS", "8"))


def base_idx(allele):
    # allele is a bytes scalar (h5 S256). single base → idx, otherwise -1
    if len(allele) != 1:
        return -1
    return BASE_TO_IDX.get(allele.upper(), -1)


def compute_offsets(pos, anchor):
    # variant offset within a window centered on `anchor` (the splice site).
    # window covers [anchor-WINDOW_HALF, anchor+WINDOW_HALF] (1-based, inclusive),
    # stored as SEQ_LEN_FULL = 10001 positions in the h5.
    return (pos - anchor + WINDOW_HALF).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input h5 (same schema as score_merlin.py)")
    p.add_argument("--output", required=True, help="output h5 for llr scores")
    p.add_argument("--checkpoint-mlm", required=True, help="MLM-pretrained MERLIN .pth")
    p.add_argument("--batch-size", type=int, default=batch_size)
    args = p.parse_args()

    assert os.path.exists(args.input), f"input not found: {args.input}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    print(f"loading {args.input}")
    with h5py.File(args.input, "r") as f:
        seq_start_ref = f["seqs/exon_start_ref"][:]
        seq_end_ref   = f["seqs/exon_end_ref"][:]
        meta = {k: f["meta"][k][:] for k in f["meta"].keys()}
        input_attrs = dict(f.attrs)

    n = len(seq_start_ref)
    stored_len = seq_start_ref.shape[1]
    print(f"  {n:,} variants, seq_len={stored_len}")

    if stored_len < TARGET_LEN:
        raise ValueError(f"stored seq_len {stored_len} too small (need >= {TARGET_LEN})")
    trim = (stored_len - TARGET_LEN) // 2

    # variant indices + per-window offsets
    pos = meta["pos"].astype(np.int64)
    ref = meta["ref"]
    alt = meta["alt"]
    exon_start = meta["exon_start"].astype(np.int64)
    exon_end = meta["exon_end"].astype(np.int64)

    ref_idx = np.array([base_idx(b) for b in ref], dtype=np.int64)
    alt_idx = np.array([base_idx(b) for b in alt], dtype=np.int64)
    snv_mask = (ref_idx >= 0) & (alt_idx >= 0)
    n_indel = int((~snv_mask).sum())
    if n_indel:
        print(f"  {n_indel:,} non-SNV variants → NaN", flush=True)

    # offset within the trimmed (10001) window. stored is 10001; trim is 0 for matched length.
    off_start = compute_offsets(pos, exon_start) - trim
    off_end   = compute_offsets(pos, exon_end) - trim

    in_start = (off_start >= 0) & (off_start < TARGET_LEN)
    in_end   = (off_end >= 0) & (off_end < TARGET_LEN)

    # pick the start window if variant lives in it; else fall back to end window.
    use_start = snv_mask & in_start
    use_end   = snv_mask & ~in_start & in_end
    valid = use_start | use_end

    print(f"  use exon_start window: {int(use_start.sum()):,}", flush=True)
    print(f"  fall back to exon_end: {int(use_end.sum()):,}", flush=True)
    print(f"  variant outside both:  {int((snv_mask & ~valid).sum()):,} → NaN", flush=True)

    model = load_mlm_from_ckpt(args.checkpoint_mlm, device)

    llr = np.full(n, np.nan, dtype=np.float32)

    def trim_seqs(arr):
        return arr[:, trim: trim + TARGET_LEN, :] if trim else arr

    for src_seq, mask, offs_full, label in [
        (seq_start_ref, use_start, off_start, "exon_start"),
        (seq_end_ref,   use_end,   off_end,   "exon_end"),
    ]:
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        offs = offs_full[idxs]
        seqs = trim_seqs(src_seq[idxs])
        r = ref_idx[idxs]
        a = alt_idx[idxs]
        print(f"scoring {label} ({len(idxs):,} variants)", flush=True)

        # batched forward + LLR readout
        steps = math.ceil(len(idxs) / args.batch_size)
        out = np.empty(len(idxs), dtype=np.float32)
        from merlin_mlm import score_llr
        # delegate to module helper; it batches internally with its own bs.
        out = score_llr(model, seqs, offs, r, a, device, bs=args.batch_size)
        llr[idxs] = out

    print(f"saving to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        scores_grp = f.create_group("scores")
        scores_grp.create_dataset("llr", data=llr.astype(np.float32))

        meta_grp = f.create_group("meta")
        for key, arr in meta.items():
            meta_grp.create_dataset(key, data=arr)

        f.attrs["model"] = "merlin_mlm"
        f.attrs["n_variants"] = n
        f.attrs["seq_len_in"] = TARGET_LEN
        if "seq_len" in input_attrs:
            f.attrs["seq_len"] = input_attrs["seq_len"]
        f.attrs["note"] = ("single-position LLR at variant; exon_start window preferred, "
                           "falls back to exon_end if variant outside the start window. SNVs only.")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
