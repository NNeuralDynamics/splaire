#!/usr/bin/env python3
"""score reporter assay variants with pangolin"""
import argparse
import os

import h5py
import numpy as np
import torch
from pkg_resources import resource_filename
from pangolin.model import Pangolin, L, W, AR
from tqdm import tqdm

MODEL_NAMES = {
    0: "heart_p_splice", 1: "heart_usage", 2: "liver_p_splice", 3: "liver_usage",
    4: "brain_p_splice", 5: "brain_usage", 6: "testis_p_splice", 7: "testis_usage",
}
INDEX_MAP = {0: 1, 1: 2, 2: 4, 3: 5, 4: 7, 5: 8, 6: 10, 7: 11}

target_bs = int(os.environ.get("PANG_BS", "128"))
use_compile = bool(int(os.environ.get("PANG_COMPILE", "0")))
use_amp = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"


def load_groups():
    groups = {}
    for mn in tqdm(list(range(8)), desc="load pangolin"):
        reps = []
        for rep in range(1, 6):
            m = Pangolin(L, W, AR).to(device).eval()
            if use_compile and hasattr(torch, "compile"):
                m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
            w = torch.load(resource_filename("pangolin", f"models/final.{rep}.{mn}.3"), map_location=device)
            m.load_state_dict(w)
            reps.append(m)
        groups[mn] = reps
    return groups


def score_batch(groups, X):
    with torch.inference_mode():
        with torch.amp.autocast(device_str, enabled=use_amp):
            probe = next(iter(groups.values()))[0](X)
    kc = int(probe.shape[-1] // 2)

    results = {}
    for mn, reps in groups.items():
        vals = []
        with torch.inference_mode(), torch.amp.autocast(device_str, enabled=use_amp):
            for m in reps:
                y = m(X)[:, INDEX_MAP[mn], kc]
                vals.append(y.detach())
        v = torch.stack(vals, dim=0).mean(dim=0).float().cpu().numpy()
        results[MODEL_NAMES[mn]] = v
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input h5 file with pre-encoded sequences")
    p.add_argument("--output", required=True, help="output h5 file for scores")
    p.add_argument("--batch-size", type=int, default=target_bs)
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
        meta = {}
        for key in f["meta"].keys():
            meta[key] = f["meta"][key][:]
        input_attrs = dict(f.attrs)

    n = len(seqs["exon_start_ref"])
    print(f"  {n:,} variants")

    # load models
    groups = load_groups()

    # score each site/allele combination
    combos = [
        ("exon_start", "ref", seqs["exon_start_ref"]),
        ("exon_start", "alt", seqs["exon_start_alt"]),
        ("exon_end", "ref", seqs["exon_end_ref"]),
        ("exon_end", "alt", seqs["exon_end_alt"]),
    ]

    all_scores = {name: {} for name in MODEL_NAMES.values()}

    for site, allele, X_all in combos:
        key = f"{site}_{allele}"
        print(f"scoring {key}")

        scores = {name: [] for name in MODEL_NAMES.values()}
        bs = args.batch_size

        i = 0
        pbar = tqdm(total=n, desc=key)
        while i < n:
            j = min(i + bs, n)
            # transpose from (B, L, 4) to (B, 4, L) for pangolin
            X = torch.from_numpy(X_all[i:j].transpose(0, 2, 1)).to(device)

            try:
                batch_scores = score_batch(groups, X)
                for name, vals in batch_scores.items():
                    scores[name].append(vals)
                pbar.update(j - i)
                i = j
                if bs < args.batch_size:
                    bs = min(args.batch_size, bs * 2)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                else:
                    raise
        pbar.close()

        for name in MODEL_NAMES.values():
            all_scores[name][key] = np.concatenate(scores[name]).astype(np.float32)

    # save to h5
    print(f"saving to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as f:
        # scores group
        scores_grp = f.create_group("scores")
        for name in MODEL_NAMES.values():
            for site in ("exon_start", "exon_end"):
                for allele in ("ref", "alt"):
                    key = f"{site}_{allele}"
                    scores_grp.create_dataset(f"{name}_{key}", data=all_scores[name][key])

        # copy metadata from input
        meta_grp = f.create_group("meta")
        for key, arr in meta.items():
            meta_grp.create_dataset(key, data=arr)

        # attributes
        f.attrs["model"] = "pangolin"
        f.attrs["n_variants"] = n
        if "seq_len" in input_attrs:
            f.attrs["seq_len"] = input_attrs["seq_len"]

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
