#!/usr/bin/env python3
"""score reporter assay variants with splicetransformer"""
import argparse
import math
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

# splicetransformer setup
here = os.path.abspath(os.path.dirname(__file__))
repo_dir = os.environ.get("SPT_REPO") or os.path.abspath(os.path.join(here, "../../other_models/SpliceTransformer"))
weights_path = os.environ.get("SPT_WEIGHTS") or os.path.join(repo_dir, "model", "weights", "SpTransformer_pytorch.ckpt")

for p in (repo_dir, os.path.join(repo_dir, "src"), os.path.join(repo_dir, "sptransformer")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

SPT_TISSUES = [
    "Adipose_Tissue", "Blood", "Blood_Vessel", "Brain", "Colon", "Heart",
    "Kidney", "Liver", "Lung", "Muscle", "Nerve", "Small_Intestine",
    "Skin", "Spleen", "Stomach"
]
SPT_CHANNELS = ["neither", "acceptor", "donor"] + [f"usage_{t}" for t in SPT_TISSUES]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def center_map(l_in, l_out):
    if l_in <= 1:
        return 0
    return int(round((l_out - 1) * (l_in // 2) / (l_in - 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input h5 file with pre-encoded sequences")
    p.add_argument("--output", required=True, help="output h5 file for scores")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    assert os.path.exists(args.input), f"input not found: {args.input}"
    assert os.path.exists(weights_path), f"splicetransformer weights not found: {weights_path}"

    # load model (must be done from repo dir)
    print("loading splicetransformer")
    print(f"device: {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    from sptransformer import Annotator
    annot = Annotator()
    model = annot.model
    if hasattr(model, "eval"):
        model.eval()
    os.chdir(prev_cwd)

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

    # score each site/allele combination
    combos = [
        ("exon_start", "ref", seqs["exon_start_ref"]),
        ("exon_start", "alt", seqs["exon_start_alt"]),
        ("exon_end", "ref", seqs["exon_end_ref"]),
        ("exon_end", "alt", seqs["exon_end_alt"]),
    ]

    all_scores = {ch: {} for ch in SPT_CHANNELS}

    for site, allele, X_all in combos:
        key = f"{site}_{allele}"
        print(f"scoring {key}")

        scores = {ch: [] for ch in SPT_CHANNELS}
        steps = math.ceil(n / args.batch_size)

        for k in tqdm(range(steps), desc=key):
            i = k * args.batch_size
            j = min((k + 1) * args.batch_size, n)

            # trim to 8001bp (4000 each side) - SPT uses 4000bp context not 5000bp
            # input is 10001bp centered, trim 1000bp from each end
            batch = X_all[i:j, 1000:-1000, :]

            # transpose from (B, L, 4) to (B, 4, L) for splicetransformer
            X = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)

            with torch.no_grad():
                Y = model.step(X).cpu().numpy()
            kc = center_map(X.shape[-1], Y.shape[-1])

            for chi, ch in enumerate(SPT_CHANNELS):
                scores[ch].append(Y[:, chi, kc])

        for ch in SPT_CHANNELS:
            all_scores[ch][key] = np.concatenate(scores[ch]).astype(np.float32)

    # save to h5
    print(f"saving to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as f:
        # scores group
        scores_grp = f.create_group("scores")
        for ch in SPT_CHANNELS:
            for site in ("exon_start", "exon_end"):
                for allele in ("ref", "alt"):
                    key = f"{site}_{allele}"
                    scores_grp.create_dataset(f"{ch}_{key}", data=all_scores[ch][key])

        # copy metadata from input
        meta_grp = f.create_group("meta")
        for key, arr in meta.items():
            meta_grp.create_dataset(key, data=arr)

        # attributes
        f.attrs["model"] = "splicetransformer"
        f.attrs["n_variants"] = n
        if "seq_len" in input_attrs:
            f.attrs["seq_len"] = input_attrs["seq_len"]

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
