#!/usr/bin/env python3
"""score reporter assay variants with spliceai"""
import argparse
import math
import os

import h5py
import numpy as np
from pkg_resources import resource_filename
from tensorflow.keras.models import load_model
from tqdm import tqdm


def center_map(l_in, l_out):
    if l_in <= 1:
        return 0
    return int(round((l_out - 1) * ((l_in) // 2) / (l_in - 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input h5 file with pre-encoded sequences")
    p.add_argument("--output", required=True, help="output h5 file for scores")
    p.add_argument("--batch-size", type=int, default=64)
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
    print("loading spliceai models")
    models = [load_model(resource_filename("spliceai", f"models/spliceai{i}.h5"), compile=False)
              for i in tqdm(range(1, 6), desc="load spliceai")]

    # score each site/allele combination
    combos = [
        ("exon_start", "ref", seqs["exon_start_ref"]),
        ("exon_start", "alt", seqs["exon_start_alt"]),
        ("exon_end", "ref", seqs["exon_end_ref"]),
        ("exon_end", "alt", seqs["exon_end_alt"]),
    ]

    heads = ["neither", "acceptor", "donor"]
    all_scores = {h: {} for h in heads}

    for site, allele, X_all in combos:
        key = f"{site}_{allele}"
        print(f"scoring {key}")

        scores = {h: [] for h in heads}
        bs = args.batch_size
        steps = math.ceil(n / bs)

        for k in tqdm(range(steps), desc=key):
            i, j = k * bs, min((k + 1) * bs, n)
            X = X_all[i:j]

            preds = [m.predict(X, batch_size=bs, verbose=0) for m in models]
            y = np.mean(np.stack(preds, axis=0), axis=0)
            kc = center_map(X.shape[1], y.shape[1])

            for hi, h in enumerate(heads):
                scores[h].append(y[:, kc, hi])

        for h in heads:
            all_scores[h][key] = np.concatenate(scores[h]).astype(np.float32)

    # save to h5
    print(f"saving to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as f:
        # scores group
        scores_grp = f.create_group("scores")
        for h in heads:
            for site in ("exon_start", "exon_end"):
                for allele in ("ref", "alt"):
                    key = f"{site}_{allele}"
                    scores_grp.create_dataset(f"{h}_{key}", data=all_scores[h][key])

        # copy metadata from input
        meta_grp = f.create_group("meta")
        for key, arr in meta.items():
            meta_grp.create_dataset(key, data=arr)

        # attributes
        f.attrs["model"] = "spliceai"
        f.attrs["n_variants"] = n
        if "seq_len" in input_attrs:
            f.attrs["seq_len"] = input_attrs["seq_len"]

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
