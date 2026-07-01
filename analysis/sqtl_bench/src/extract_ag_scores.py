#!/usr/bin/env python3
"""strand-aware AG score extraction. outputs SPLAIRE-shape columns
(alphagenome_{don,acc,ssu}_max_{inc,dec}[_off]) so downstream code that
consumes SPLAIRE-shape columns plugs in without changes.

AG SPLICE_SITES output has 4 channels: donor+, acceptor+, donor-, acceptor-.
Minus-strand genes have near-zero signal on channels 0/1 (the +strand tracks);
the real signal is on channels 2/3. Aggregators that use channels 0/1 unconditionally
get pearson ~0.1 on minus-strand rows (vs ~0.3 on plus-strand). This picks the
correct pair per row.

positions are already in genomic frame (score_alphagenome_variant.py flips before
write), so `off` is genomic offset from variant (which is at window center).

usage:
    python src/extract_ag_scores.py \\
        /scratch/runyan.m/sqtl_bench/haec_hungarian/scores/pos.ag \\
        /scratch/runyan.m/sqtl_bench/haec_hungarian/scores/pos.ag_extracted.parquet
"""
import argparse

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# AG SPLICE_SITES channel layout (see analysis/test/src/_ag_common.py)
DON_PLUS, ACC_PLUS, DON_MINUS, ACC_MINUS = 0, 1, 2, 3

WINDOW = 16384
CENTER = WINDOW // 2


def compute_ss_maxmin(h5_path, model_prefix, chunk_size=256):
    """max_inc/max_dec of donor and acceptor delta, strand-picked channels"""
    f = h5py.File(h5_path, "r")
    n = f["ref"].shape[0]
    strand = np.array([s.decode() if isinstance(s, bytes) else s for s in f["gene_strand"][:]])
    var_key = np.array([s.decode() if isinstance(s, bytes) else s for s in f["var_key"][:]])
    is_plus = strand == "+"

    cols = [f"{model_prefix}_{h}_{d}{s}"
            for h in ("don", "acc")
            for d in ("max_inc", "max_dec")
            for s in ("", "_off")]
    results = {c: np.full(n, np.nan, dtype=np.float64) for c in cols}

    for i0 in tqdm(range(0, n, chunk_size), desc=f"maxmin {model_prefix}"):
        i1 = min(i0 + chunk_size, n)
        delta = f["alt"][i0:i1].astype(np.float32) - f["ref"][i0:i1].astype(np.float32)
        plus_mask = is_plus[i0:i1]

        # pick strand-correct channels per row
        don = np.where(plus_mask[:, None], delta[:, :, DON_PLUS], delta[:, :, DON_MINUS])
        acc = np.where(plus_mask[:, None], delta[:, :, ACC_PLUS], delta[:, :, ACC_MINUS])

        for h, sig in (("don", don), ("acc", acc)):
            idx_inc = np.argmax(sig, axis=1)
            val_inc = np.take_along_axis(sig, idx_inc[:, None], axis=1).squeeze(1)
            idx_dec = np.argmin(sig, axis=1)
            val_dec = np.take_along_axis(sig, idx_dec[:, None], axis=1).squeeze(1)

            results[f"{model_prefix}_{h}_max_inc"][i0:i1] = val_inc
            results[f"{model_prefix}_{h}_max_inc_off"][i0:i1] = idx_inc - CENTER
            results[f"{model_prefix}_{h}_max_dec"][i0:i1] = val_dec
            results[f"{model_prefix}_{h}_max_dec_off"][i0:i1] = idx_dec - CENTER

    f.close()
    return pd.DataFrame(results), var_key, strand


def compute_ssu_maxmin(ssu_parquet_path, var_key, model_prefix):
    """max/min ssu delta across (pos, track) per variant. sparse long format,
    positions absent from BOTH ref and alt contribute 0 delta."""
    ssu = pd.read_parquet(ssu_parquet_path)
    ref = ssu.query("allele == 'ref'").drop(columns="allele").rename(columns={"value": "ref"})
    alt = ssu.query("allele == 'alt'").drop(columns="allele").rename(columns={"value": "alt"})
    merged = ref.merge(alt, on=["variant_id", "pos_in_window", "track_idx"], how="outer").fillna(0.0)
    merged["delta"] = merged["alt"] - merged["ref"]

    grp = (merged.groupby("variant_id")["delta"]
                 .agg(max_inc="max", max_dec="min")
                 .reset_index()
                 .rename(columns={"max_inc": f"{model_prefix}_ssu_max_inc",
                                  "max_dec": f"{model_prefix}_ssu_max_dec"}))

    out = pd.DataFrame({"variant_id": var_key}).merge(grp, on="variant_id", how="left")
    # variants with no non-zero ssu → delta is 0 everywhere
    for c in (f"{model_prefix}_ssu_max_inc", f"{model_prefix}_ssu_max_dec"):
        out[c] = out[c].fillna(0.0)
    return out.drop(columns=["variant_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ag_prefix", help="AG h5+parquet prefix (reads {prefix}.h5, {prefix}.ssu.parquet)")
    ap.add_argument("output_parquet")
    ap.add_argument("--model-prefix", default="alphagenome", help="column name prefix")
    args = ap.parse_args()

    h5_path = f"{args.ag_prefix}.h5"
    ssu_path = f"{args.ag_prefix}.ssu.parquet"

    print(f"loading {h5_path}")
    ss_df, var_key, strand = compute_ss_maxmin(h5_path, args.model_prefix)
    print(f"n_variants: {len(ss_df):,}  strand + {(strand=='+').sum()}  - {(strand=='-').sum()}")

    print(f"loading {ssu_path}")
    ssu_df = compute_ssu_maxmin(ssu_path, var_key, args.model_prefix)

    out = pd.concat([pd.DataFrame({"variant_id": var_key, "gene_strand": strand}),
                     ss_df, ssu_df], axis=1)
    print(f"output shape: {out.shape}")
    print(out.head())
    out.to_parquet(args.output_parquet, index=False)
    print(f"saved {args.output_parquet}")


if __name__ == "__main__":
    main()
