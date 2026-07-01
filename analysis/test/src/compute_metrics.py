#!/usr/bin/env python3
"""compute metrics from parquet prediction files"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from metrics import classification, classification_posfirst, regression

sentinel = 777.0
n_workers = 8

chrom_to_int = {f'chr{i}': i for i in range(1, 23)}
chrom_to_int['chrX'] = 23
chrom_to_int['chrY'] = 24

coord_cols = {"chrom", "pos", "strand", "y_acceptor", "y_donor", "y_ssu"}

# each model: filename suffix -> (model_name, list of prediction columns)
# to add a new model, add one entry here
models = {
    "_splaire_ref": ("splaire_ref", ["acceptor", "donor", "ssu"]),
    "_splaire_var": ("splaire_var", ["acceptor", "donor", "ssu"]),
    "_sa": ("spliceai", ["acceptor", "donor"]),
    "_pang": ("pangolin", [
        "heart_p_splice", "heart_usage", "liver_p_splice", "liver_usage",
        "brain_p_splice", "brain_usage", "testis_p_splice", "testis_usage",
        "avg_p_splice", "avg_usage",
    ]),
    "_spt": ("splicetransformer", [
        "acceptor", "donor",
        "adipose", "blood", "blood_vessel", "brain", "colon", "heart", "kidney",
        "liver", "lung", "muscle", "nerve", "small_intestine", "skin", "spleen", "stomach",
        "avg_tissue",
    ]),
    "_merlin": ("merlin", ["acceptor", "donor", "ssu"]),
    "_rinalmo": ("rinalmo", ["acceptor", "donor"]),
    "_ag": ("alphagenome", ["acceptor", "donor", "ssu"]),
    "_dsm": ("deltasplice", ["acceptor", "donor"]),
    "_dsh": ("deltasplice_human", ["acceptor", "donor"]),
}


def find_prediction_files(pred_dir):
    """find parquet files matching models dict"""
    files = {}
    for pq in pred_dir.glob("*.parquet"):
        # AG writes a raw per-position junctions parquet alongside the main
        # prediction parquet; it has its own schema (no chrom col) and is not
        # a model prediction file
        if "_junctions" in pq.stem:
            continue
        for suffix, (name, _cols) in models.items():
            if pq.stem.endswith(suffix):
                files[name] = pq
                break
    return files


def load_predictions(pred_dir, drop_chr9=True):
    """load ground truth and model predictions from parquets

    drop_chr9: filter out chr9 positions (in our validation, not test).
    loads all non-coord columns from each parquet so per-replicate columns
    (e.g. acceptor_rep1) are picked up alongside the averaged columns.
    """
    files = find_prediction_files(pred_dir)
    gt_cols = ["chrom", "pos", "strand", "y_acceptor", "y_donor", "y_ssu"]
    any_df = pd.read_parquet(next(iter(files.values())), columns=gt_cols)
    if drop_chr9:
        keep = any_df["chrom"].values != 9
        any_df = any_df.loc[keep].reset_index(drop=True)
        print(f"dropped chr9: {len(keep) - keep.sum():,} positions removed", flush=True)
    else:
        keep = None
    truth = {
        "acceptor": any_df["y_acceptor"].values.astype(np.float32),
        "donor": any_df["y_donor"].values.astype(np.float32),
        "ssu": any_df["y_ssu"].values.astype(np.float32),
    }
    coords = any_df[["chrom", "pos", "strand"]].copy()
    del any_df

    preds = {}
    for name, path in files.items():
        df = pd.read_parquet(path)
        if keep is not None:
            df = df.loc[keep].reset_index(drop=True)
        # all non-coord, non-truth columns are predictions (incl. per-rep)
        pred_cols = [c for c in df.columns if c not in coord_cols]
        preds[name] = {c: df[c].values.astype(np.float32) for c in pred_cols}
        del df
    return truth, preds, coords


def load_tau_groups_df(csv_path):
    """parse tau-group csv into chrom/pos/strand/group_int"""
    grp_to_int = {"constitutive": 0, "intermediate": 1, "tissue-specific": 2}
    df_grp = pd.read_csv(csv_path)
    if df_grp["chrom"].dtype == object:
        df_grp["chrom"] = df_grp["chrom"].astype(str).str.replace("chr", "", regex=False)
        df_grp["chrom"] = df_grp["chrom"].map(lambda x: int(x) if x.isdigit() else
                                              (23 if x == "X" else 24 if x == "Y" else -1))
    df_grp = df_grp[df_grp["chrom"] >= 0]
    df_grp["group_int"] = df_grp["group"].map(grp_to_int).fillna(-1).astype(np.int8)
    df_grp = df_grp[df_grp["group_int"] >= 0]
    return df_grp[["chrom", "pos", "strand", "group_int"]].copy()


def apply_tau_groups(df_grp, coords_df, verbose=True):
    merged = coords_df[["chrom", "pos", "strand"]].merge(
        df_grp, on=["chrom", "pos", "strand"], how="left")
    out = merged["group_int"].fillna(-1).astype(np.int8).values
    if verbose:
        n_class = int((out >= 0).sum())
        print(f"tau-groups: {n_class:,}/{len(out):,} positions classified", flush=True)
    return out


def load_tau_groups(csv_path, coords_df):
    # per-position int8 group array from csv
    return apply_tau_groups(load_tau_groups_df(csv_path), coords_df)


def build_keep_mask(truth, tau_groups_arr, rng, neg_ratio=20):
    # keep all positives + tau-classified, subsample negatives at neg_ratio:1
    is_pos = (truth["acceptor"] == 1) | (truth["donor"] == 1)
    must_keep = is_pos
    if tau_groups_arr is not None:
        must_keep = must_keep | (tau_groups_arr >= 0)
    n_pos = int(is_pos.sum())
    n_neg_target = n_pos * neg_ratio
    neg_pool = np.where(~must_keep)[0]
    if len(neg_pool) > n_neg_target and n_neg_target > 0:
        sampled = rng.choice(neg_pool, size=n_neg_target, replace=False)
        neg_mask = np.zeros(len(is_pos), dtype=bool)
        neg_mask[sampled] = True
        keep = must_keep | neg_mask
    else:
        keep = np.ones_like(is_pos, dtype=bool)
    return keep


def load_shared_sites(matrix_path):
    """splice sites valid across all samples, filtered to test chroms"""
    test_chroms = {f"chr{c}" for c in [1, 3, 5, 7, 9]}

    with open(matrix_path) as f:
        header = f.readline().strip().split('\t')
        meta = {'event_id', 'region', 'strand', 'site', 'site_type', 'pop_mean'}
        sample_indices = [i for i, h in enumerate(header) if h not in meta]

    shared = set()
    with open(matrix_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            region = parts[1]
            chrom = region if region.startswith('chr') else f'chr{region}'
            if chrom not in test_chroms:
                continue
            if chrom not in chrom_to_int:
                continue

            all_valid = all(float(parts[i]) != 777.0 for i in sample_indices)
            if all_valid:
                shared.add((chrom_to_int[chrom], int(parts[3])))

    return shared


def build_shared_mask(coords_df, shared_sites):
    """boolean mask for positions matching shared-valid splice sites"""
    chroms = coords_df["chrom"].values.astype(np.int64)
    positions = coords_df["pos"].values.astype(np.int64)
    keys = chroms * 1_000_000_000 + positions
    shared_keys = np.array([c * 1_000_000_000 + p for c, p in shared_sites], dtype=np.int64)
    return np.isin(keys, shared_keys)


def load_truth_coords_files(input_path, drop_chr9=True, paralog_exclude=None, only_models=None):
    """load truth + coords once; return file map + combined keep_mask (chr9 + paralog).

    only_models: optional set of model names to restrict the file map to. truth is
    loaded from the first remaining file, so when one model's parquet has a different
    row set (e.g. merlin already excludes chr9), the truth + keep_mask align with it.
    """
    files = find_prediction_files(input_path)
    if not files:
        raise ValueError(f"no prediction parquets found in {input_path}")
    if only_models:
        files = {k: v for k, v in files.items() if k in only_models}
        if not files:
            raise ValueError(f"no parquets match only_models {only_models} in {input_path}")
        print(f"--only-models filter: scoring {list(files.keys())}", flush=True)
    gt_cols = ["chrom", "pos", "strand", "y_acceptor", "y_donor", "y_ssu"]
    any_path = next(iter(files.values()))
    df = pd.read_parquet(any_path, columns=gt_cols)

    keep = np.ones(len(df), dtype=bool)
    if drop_chr9:
        chr9_keep = df["chrom"].values != 9
        keep &= chr9_keep
        print(f"dropped chr9: {(~chr9_keep).sum():,} positions removed", flush=True)
    if paralog_exclude is not None:
        exclude_keys = (paralog_exclude["chrom"].astype(np.int64) * 1_000_000_000
                        + paralog_exclude["pos"].astype(np.int64))
        df_keys = (df["chrom"].values.astype(np.int64) * 1_000_000_000
                    + df["pos"].values.astype(np.int64))
        paralog_keep = ~np.isin(df_keys, exclude_keys)
        keep &= paralog_keep
        print(f"paralog-filter: excluded {(~paralog_keep).sum():,} positions", flush=True)

    if keep.all():
        keep = None
        df_filt = df
    else:
        df_filt = df.loc[keep].reset_index(drop=True)
    print(f"final kept positions: {len(df_filt):,}", flush=True)

    truth = {
        "acceptor": df_filt["y_acceptor"].values.astype(np.float32),
        "donor": df_filt["y_donor"].values.astype(np.float32),
        "ssu": df_filt["y_ssu"].values.astype(np.float32),
    }
    coords = df_filt[["chrom", "pos", "strand"]].copy()
    return truth, coords, files, keep


def group_pred_columns(pred_cols):
    """group prediction columns by rep or fold suffix; returns [(batch_name, [cols]), ...]
    recognizes:
      *_rep<N>                — replicate ensembles (splaire, spliceai, pangolin, deltasplice)
      *_fold_<N>              — alphagenome fold (acceptor/donor)
      *_fold_<N>_track<T>     — alphagenome per-fold per-tissue-track ssu
      everything else         — goes in 'avg' batch
    each batch keeps acceptor/donor/ssu paired so combined-cls metrics still work.
    """
    by_batch = {}
    for c in pred_cols:
        m_rep = re.search(r"_rep(\d+)$", c)
        m_fold = re.search(r"_fold_(\d+)(?:_track_\d+)?$", c)
        if m_rep:
            batch = f"rep{m_rep.group(1)}"
        elif m_fold:
            batch = f"fold{m_fold.group(1)}"
        else:
            batch = "avg"
        by_batch.setdefault(batch, []).append(c)
    keys = []
    if "avg" in by_batch:
        keys.append("avg")
    keys += sorted([k for k in by_batch if k.startswith("rep")], key=lambda x: int(x[3:]))
    keys += sorted([k for k in by_batch if k.startswith("fold")], key=lambda x: int(x[4:]))
    return [(k, by_batch[k]) for k in keys]


def merge_results(target, partial, overwrite=False):
    """recursively merge partial results into target; new keys added, existing
    leaves preserved (default) or replaced (overwrite=True). use overwrite=True
    when refreshing one model's metrics in a JSON that already contains them."""
    for k, v in partial.items():
        if k not in target:
            target[k] = v
        elif isinstance(v, dict) and isinstance(target[k], dict):
            merge_results(target[k], v, overwrite=overwrite)
        elif overwrite:
            target[k] = v
    return target


def precompute_context(truth, valid_all_mask=None, regression_only=False, splice_combined=False):
    """precompute all truth-derived indices and masks; reused across streaming batches.
    expensive flatnonzero/concatenate ops on 100M+ position arrays are done once here."""
    is_acc = truth["acceptor"] == 1
    is_don = truth["donor"] == 1
    is_neither = ~is_acc & ~is_don
    ssu = truth["ssu"]

    if regression_only:
        masks = {
            "ssu_valid": ssu != sentinel,
            "ssu_valid_nonzero": (ssu != sentinel) & (ssu > 0),
        }
    else:
        masks = {
            "all": np.ones(ssu.size, dtype=bool),
            "ssu_valid": ssu != sentinel,
            "ssu_valid_nonzero": (ssu != sentinel) & (ssu > 0),
        }
    if valid_all_mask is not None:
        masks["ssu_shared"] = valid_all_mask
        masks["ssu_shared_nonzero"] = valid_all_mask & (ssu > 0)

    neither_idx = np.flatnonzero(is_neither)
    reg_subsets = [k for k in masks if k != "all"]
    cls_subsets = [k for k in reg_subsets if not k.startswith("ssu_shared")]

    counts = {
        "n_positions": int(ssu.size),
        "n_acceptors": int(is_acc.sum()),
        "n_donors": int(is_don.sum()),
        "n_neither": int(neither_idx.size),
    }

    # cls indices per (subset, target)
    cls_indices = {}
    if not regression_only:
        is_splice = is_acc | is_don
        for subset, mask in masks.items():
            cls_indices[subset] = {}
            if splice_combined:
                pos_mask = is_splice & mask
                pos_idx = np.flatnonzero(pos_mask)
                include_idx = np.flatnonzero(pos_mask | is_neither)
                y_true = pos_mask[include_idx].astype(np.float32)
                cls_indices[subset]["splice"] = {"pos_idx": pos_idx, "include_idx": include_idx, "y_true": y_true}
            else:
                for target, is_target in [("acceptor", is_acc), ("donor", is_don)]:
                    pos_mask = is_target & mask
                    pos_idx = np.flatnonzero(pos_mask)
                    include_idx = np.flatnonzero(pos_mask | is_neither)
                    y_true = pos_mask[include_idx].astype(np.float32)
                    cls_indices[subset][target] = {"pos_idx": pos_idx, "include_idx": include_idx, "y_true": y_true}

    # regression site indices per subset
    reg_indices = {}
    for subset in reg_subsets:
        mask = masks[subset]
        site_idx = np.flatnonzero((is_acc | is_don) & mask)
        y_true_reg = ssu[site_idx].astype(np.float32)
        acc_idx = np.flatnonzero(is_acc & mask)
        don_idx = np.flatnonzero(is_don & mask)
        y_true_cls = np.concatenate([ssu[acc_idx], ssu[don_idx]]).astype(np.float32)
        reg_indices[subset] = {"site_idx": site_idx, "y_true_reg": y_true_reg,
                               "acc_idx": acc_idx, "don_idx": don_idx, "y_true_cls": y_true_cls}

    # ssu bins
    bin_edges = np.arange(0, 1.1, 0.1)
    site_bins = np.clip(np.digitize(ssu, bin_edges) - 1, 0, 9)

    # binned regression indices per (subset, bin)
    binned_reg_indices = {}
    for subset in reg_subsets:
        mask = masks[subset]
        binned_reg_indices[subset] = {}
        for b in range(10):
            in_bin = site_bins == b
            site_idx = np.flatnonzero((is_acc | is_don) & mask & in_bin)
            if site_idx.size == 0:
                continue
            y_true_reg = ssu[site_idx].astype(np.float32)
            acc_idx = np.flatnonzero(is_acc & mask & in_bin)
            don_idx = np.flatnonzero(is_don & mask & in_bin)
            y_true_cls = np.concatenate([ssu[acc_idx], ssu[don_idx]]).astype(np.float32) if (acc_idx.size > 0 or don_idx.size > 0) else None
            binned_reg_indices[subset][b] = {"site_idx": site_idx, "y_true_reg": y_true_reg,
                                              "acc_idx": acc_idx, "don_idx": don_idx, "y_true_cls": y_true_cls}

    # pos_per_bin for ratio-preserving binned classification
    pos_per_bin_cache = {}
    if not regression_only:
        for subset in cls_subsets:
            mask = masks[subset]
            pos_per_bin_cache[subset] = {}
            for target, is_target in [("acceptor", is_acc), ("donor", is_don)]:
                pos_mask = is_target & mask
                pos_per_bin = {b: np.flatnonzero(pos_mask & (site_bins == b)) for b in range(10)}
                pos_per_bin = {b: idx for b, idx in pos_per_bin.items() if idx.size > 0}
                pos_per_bin_cache[subset][target] = pos_per_bin

    return {
        "is_acc": is_acc, "is_don": is_don, "is_neither": is_neither, "ssu": ssu,
        "masks": masks, "neither_idx": neither_idx,
        "reg_subsets": reg_subsets, "cls_subsets": cls_subsets,
        "counts": counts,
        "cls_indices": cls_indices, "reg_indices": reg_indices,
        "site_bins": site_bins, "binned_reg_indices": binned_reg_indices,
        "pos_per_bin_cache": pos_per_bin_cache,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_json")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-binned", action="store_true")
    parser.add_argument("--no-binned-cls", action="store_true", help="skip ratio-preserving binned classification (slowest stage); keep binned regression")
    parser.add_argument("--paralog-mask", help="npy file with structured array (chrom int8, pos int32) of (chrom, pos) positions to EXCLUDE (paralog filter). truth/preds at these positions are set to non-splice / dropped before metrics.")
    parser.add_argument("--no-cls", action="store_true", help="skip all classification metrics")
    parser.add_argument("--regression-only", action="store_true", help="only compute regression (overall + binned) on valid-SSU splice sites")
    parser.add_argument("--splicing-matrix", help="processed_splicing_matrix.tsv for shared-valid site filtering")
    parser.add_argument("--splice-combined", action="store_true", help="use max(acc,don) for classification (for data where all sites are both)")
    parser.add_argument("--tau-groups", help="CSV (chrom, pos, strand, group) for per-site tau group classification; adds classification_tau_group block to output")
    parser.add_argument("--phase", choices=["both", "reg", "cls"], default="both",
                        help="which phase to run. 'cls' loads existing JSON and only runs Phase 2 (classification); useful for resuming after a timeout.")
    parser.add_argument("--ensemble-only", action="store_true",
                        help="skip per-replicate batches; only compute metrics on the ensemble-averaged batch.")
    parser.add_argument("--only-models", help="comma-separated list of model names; only score these (e.g. 'merlin'). when output JSON already exists, results are merged into it — non-listed models in the existing JSON are preserved, listed models are refreshed.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.all:
        pred_dirs = sorted(input_path.glob("*/"))
        print(f"pooling {len(pred_dirs)} individuals (pre-filter per indiv)", flush=True)

        # parse tau-group CSV once so we can classify per individual before concat
        tau_df = load_tau_groups_df(args.tau_groups) if args.tau_groups else None
        rng = np.random.default_rng(0)

        all_truth = {"acceptor": [], "donor": [], "ssu": []}
        all_preds = {}
        all_coords = []
        all_tau = [] if tau_df is not None else None
        total_kept = 0
        total_raw = 0

        for pred_dir in pred_dirs:
            truth_i, preds_i, coords_i = load_predictions(pred_dir)
            tau_i = apply_tau_groups(tau_df, coords_i, verbose=False) if tau_df is not None else None
            keep = build_keep_mask(truth_i, tau_i, rng, neg_ratio=20)
            total_raw += len(keep)
            total_kept += int(keep.sum())

            for k in all_truth:
                all_truth[k].append(truth_i[k][keep])
            for model, p in preds_i.items():
                if model not in all_preds:
                    all_preds[model] = {k: [] for k in p}
                for k, v in p.items():
                    all_preds[model][k].append(v[keep])
            all_coords.append(coords_i.iloc[keep].reset_index(drop=True))
            if tau_i is not None:
                all_tau.append(tau_i[keep])

            del truth_i, preds_i, coords_i, tau_i, keep

        print(f"kept {total_kept:,}/{total_raw:,} positions "
              f"({100 * total_kept / total_raw:.2f}%) after pre-filter", flush=True)

        truth = {k: np.concatenate(v) for k, v in all_truth.items()}
        preds = {m: {k: np.concatenate(v) for k, v in p.items()} for m, p in all_preds.items()}
        coords = pd.concat(all_coords, ignore_index=True) if args.tau_groups else None
        tau_groups = np.concatenate(all_tau) if all_tau is not None else None
        del all_truth, all_preds, all_coords, all_tau

        skip_cls = args.no_cls or args.regression_only
        skip_binned = args.no_binned and not args.regression_only
        results = compute_metrics(truth, preds, skip_binned=skip_binned, skip_cls=skip_cls, skip_binned_cls=args.no_binned_cls, splice_combined=args.splice_combined, regression_only=args.regression_only, tau_groups=tau_groups)
        with open(output, 'w') as f:
            json.dump({"overall": results}, f, indent=2, default=float)

    else:
        print(f"processing {input_path.name}", flush=True)
        # optional paralog exclusion mask
        paralog_exclude = None
        if args.paralog_mask:
            paralog_exclude = np.load(args.paralog_mask)
            print(f"loaded paralog exclude mask: {len(paralog_exclude):,} (chrom, pos) entries", flush=True)
        # stream per-model, per-rep batch to keep memory bounded.
        # filter files BEFORE truth is loaded so keep_mask aligns with the model's row count
        # (e.g. merlin parquet already excludes chr9 — wrong-sized mask if truth is loaded from another model)
        only_models = set(args.only_models.split(",")) if args.only_models else None
        truth, coords, files, keep_mask = load_truth_coords_files(
            input_path, drop_chr9=True, paralog_exclude=paralog_exclude, only_models=only_models)
        print(f"{truth['acceptor'].size:,} positions, {(truth['acceptor'] == 1).sum():,} acceptors", flush=True)

        shared_mask = None
        if args.splicing_matrix:
            shared_sites = load_shared_sites(args.splicing_matrix)
            shared_mask = build_shared_mask(coords, shared_sites)
            is_splice = (truth["acceptor"] == 1) | (truth["donor"] == 1)
            print(f"shared-valid: {len(shared_sites):,} sites in matrix, {(shared_mask & is_splice).sum():,} splice sites matched", flush=True)

        tau_groups = load_tau_groups(args.tau_groups, coords) if args.tau_groups else None
        del coords

        skip_cls = args.no_cls or args.regression_only
        skip_binned = args.no_binned and not args.regression_only
        skip_binned_cls = args.no_binned_cls

        # precompute truth-derived context once; reused across all batches
        print("precomputing truth-derived context...", flush=True)
        ctx = precompute_context(truth, valid_all_mask=shared_mask,
                                  regression_only=args.regression_only,
                                  splice_combined=args.splice_combined)

        # collect (model, batch_name, batch_cols, path) tuples once
        all_batches = []
        for model_name, path in files.items():
            schema_cols = pq.ParquetFile(path).schema_arrow.names
            pred_cols = [c for c in schema_cols if c not in coord_cols]
            for batch_name, batch_cols in group_pred_columns(pred_cols):
                if args.ensemble_only and batch_name != "avg":
                    continue
                all_batches.append((model_name, batch_name, batch_cols, path))
        if args.ensemble_only:
            print(f"ensemble-only: {len(all_batches)} batches (per-rep skipped)", flush=True)

        def load_batch(path, batch_cols):
            df = pd.read_parquet(path, columns=batch_cols)
            if keep_mask is not None:
                arr_dict = {c: df[c].values[keep_mask].astype(np.float32) for c in batch_cols}
            else:
                arr_dict = {c: df[c].values.astype(np.float32) for c in batch_cols}
            del df
            return arr_dict

        results = None

        # load existing JSON if:
        #   - --phase=cls (resume Phase 2 without losing Phase 1 results), or
        #   - --only-models (refresh some models, preserve the rest already in the JSON)
        if (args.phase == "cls" or only_models) and output.exists():
            with open(output) as f:
                results = json.load(f).get("overall", {})
            print(f"loaded existing results from {output} ({len(results)} top-level keys)", flush=True)

        # Phase 1: regression across all batches (fast) — skip if --phase=cls
        if args.phase == "cls":
            print(f"\n=== Phase 1 skipped (--phase=cls) ===", flush=True)
        else:
            print(f"\n=== Phase 1: regression across {len(all_batches)} batches ===", flush=True)
        for model_name, batch_name, batch_cols, path in (all_batches if args.phase != "cls" else []):
            print(f"  [reg] {model_name} batch {batch_name}: {len(batch_cols)} cols", flush=True)
            arr_dict = load_batch(path, batch_cols)
            preds_batch = {model_name: arr_dict}
            partial = compute_metrics(truth, preds_batch,
                                      skip_binned=skip_binned, skip_cls=True,
                                      skip_binned_cls=True,
                                      splice_combined=args.splice_combined,
                                      regression_only=args.regression_only,
                                      valid_all_mask=shared_mask, tau_groups=tau_groups,
                                      ctx=ctx)
            if results is None:
                results = partial
            else:
                merge_results(results, partial, overwrite=bool(only_models))
            with open(output, 'w') as f:
                json.dump({"overall": results}, f, indent=2, default=float)
            del preds_batch, arr_dict, partial
        print(f"=== Phase 1 done; regression saved to {output} ===", flush=True)

        # Phase 2: classification across all batches (slow) — skip if --phase=reg
        if not skip_cls and args.phase != "reg":
            print(f"\n=== Phase 2: classification across {len(all_batches)} batches ===", flush=True)
            for model_name, batch_name, batch_cols, path in all_batches:
                print(f"  [cls] {model_name} batch {batch_name}: {len(batch_cols)} cols", flush=True)
                arr_dict = load_batch(path, batch_cols)
                preds_batch = {model_name: arr_dict}
                partial = compute_metrics(truth, preds_batch,
                                          skip_binned=skip_binned, skip_cls=False,
                                          skip_reg=True, skip_binned_reg=True,
                                          skip_binned_cls=skip_binned_cls,
                                          splice_combined=args.splice_combined,
                                          regression_only=args.regression_only,
                                          valid_all_mask=shared_mask, tau_groups=tau_groups,
                                          ctx=ctx)
                merge_results(results, partial, overwrite=bool(only_models))
                with open(output, 'w') as f:
                    json.dump({"overall": results}, f, indent=2, default=float)
                del preds_batch, arr_dict, partial
            print(f"=== Phase 2 done; full results in {output} ===", flush=True)

    print(f"wrote {args.output_json}", flush=True)
    return 0


def compute_metrics(truth, preds, skip_binned=False, skip_cls=False, skip_reg=False, skip_binned_cls=False, skip_binned_reg=False, splice_combined=False, regression_only=False, valid_all_mask=None, tau_groups=None, ctx=None):
    """classification + regression metrics for all models. ctx is the optional precomputed
    truth-derived context dict; if None, computed inline (used by --all path)."""
    if ctx is None:
        ctx = precompute_context(truth, valid_all_mask=valid_all_mask,
                                  regression_only=regression_only, splice_combined=splice_combined)

    is_acc = ctx["is_acc"]
    is_don = ctx["is_don"]
    is_neither = ctx["is_neither"]
    ssu = ctx["ssu"]
    masks = ctx["masks"]
    neither_idx = ctx["neither_idx"]
    reg_subsets = ctx["reg_subsets"]
    cls_indices = ctx["cls_indices"]
    reg_indices = ctx["reg_indices"]

    if splice_combined:
        is_splice = is_acc | is_don

    results = {
        "counts": ctx["counts"],
        "classification": {},
        "regression": {},
    }

    # add combined predictions (max of acc/don) for models with both outputs
    if splice_combined:
        for model, p in preds.items():
            if "acceptor" in p and "donor" in p:
                p["splice"] = np.maximum(p["acceptor"], p["donor"])

    # add mean-across-tracks ssu aggregate for ag per-fold (matches splaire cross-tissue-mean target)
    for model, p in preds.items():
        fold_tracks = {}
        for key in list(p.keys()):
            m = re.match(r"ssu_fold_(\d+)_track_\d+$", key)
            if m:
                fold_tracks.setdefault(m.group(1), []).append(key)
        for fold, keys in fold_tracks.items():
            mean_key = f"ssu_fold_{fold}_mean"
            if mean_key in p:
                continue
            acc = np.zeros_like(p[keys[0]], dtype=np.float64)
            for k in keys:
                acc += p[k]
            p[mean_key] = (acc / len(keys)).astype(np.float32)

    # regression — use cached indices from ctx
    if skip_reg:
        print("skipping regression (--phase=cls)", flush=True)
        reg_subsets_iter = []
    else:
        print("computing regression", flush=True)
        reg_subsets_iter = reg_subsets
    for subset in reg_subsets_iter:
        ri = reg_indices[subset]
        site_idx = ri["site_idx"]
        y_true = ri["y_true_reg"]
        # acc_idx/don_idx/y_true_cls used by the per-fold classification-regression
        # branch (line ~665), which fires for AG's acceptor_fold_N / donor_fold_N
        # entries even in splice_combined mode
        acc_idx = ri["acc_idx"]
        don_idx = ri["don_idx"]
        y_true_cls = ri["y_true_cls"]

        results["regression"][subset] = {}
        for model, p in preds.items():
            # acceptor/donor outputs combined (averaged + per-rep + per-fold)
            acc_keys = sorted(k for k in p if k == "acceptor" or k.startswith("acceptor_rep") or k.startswith("acceptor_fold_"))
            don_keys = sorted(k for k in p if k == "donor" or k.startswith("donor_rep") or k.startswith("donor_fold_"))
            paired = []
            if "acceptor" in acc_keys and "donor" in don_keys:
                paired.append(("acceptor", "donor", "_cls"))
            for ak in acc_keys:
                if ak == "acceptor":
                    continue
                suffix = ak[len("acceptor_"):]  # rep1, fold_0, ...
                dk = f"donor_{suffix}"
                if dk in don_keys:
                    paired.append((ak, dk, f"_cls_{suffix}"))

            for ak, dk, col_suffix in paired:
                col = f"{model}{col_suffix}"
                if splice_combined and col_suffix == "_cls":
                    y_pred_cls = p["splice"][site_idx].astype(np.float32)
                    results["regression"][subset][col] = regression(y_true, y_pred_cls)
                else:
                    y_pred_cls = np.concatenate([p[ak][acc_idx], p[dk][don_idx]]).astype(np.float32)
                    results["regression"][subset][col] = regression(y_true_cls, y_pred_cls)

            # single-score outputs like ssu, tissue predictions
            paired_keys = {k for ak, dk, _ in paired for k in (ak, dk)}
            for key, arr in p.items():
                if key in paired_keys or key in ("neither", "splice"):
                    continue
                col = f"{model}_{key}"
                y_pred = arr[site_idx].astype(np.float32)
                results["regression"][subset][col] = regression(y_true, y_pred)

    # classification
    if skip_cls:
        print(f"skipping classification ({'--regression-only' if regression_only else '--no-cls'})", flush=True)
    else:
        print("computing classification", flush=True)
        targets = ["splice"] if splice_combined else ["acceptor", "donor"]
        for subset, mask in masks.items():
            results["classification"][subset] = {}
            for target in targets:
                idx = cls_indices[subset][target]
                include_idx, y_true = idx["include_idx"], idx["y_true"]

                tasks = []
                for model, p in preds.items():
                    for key, arr in p.items():
                        if key == "neither":
                            continue
                        # skip regression-only outputs (ssu tracks/mean, per-tissue usage).
                        # these are NaN-masked at non-splice rows for ag → cls would drop all negatives.
                        if key == "ssu" or key.startswith("ssu_") or key.endswith("_usage"):
                            continue
                        if splice_combined:
                            if key in ["acceptor", "donor"] or key.startswith("acceptor_rep") or key.startswith("donor_rep"):
                                continue
                            col = model if key == "splice" else f"{model}_{key}"
                        else:
                            # skip wrong-target acceptor/donor (incl. per-rep + per-fold)
                            other = "donor" if target == "acceptor" else "acceptor"
                            if key == other or key.startswith(f"{other}_rep") or key.startswith(f"{other}_fold_"):
                                continue
                            col = model if key in ["acceptor", "donor"] else f"{model}_{key}"
                        tasks.append((col, arr[include_idx].astype(np.float32)))

                target_results = {}
                if tasks:
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futures = {pool.submit(classification, y_true, y_pred): name for name, y_pred in tasks}
                        for fut in as_completed(futures):
                            target_results[futures[fut]] = fut.result()

                results["classification"][subset][target] = target_results

    # binned metrics
    if not skip_binned:
        results["binned"] = compute_binned(preds, ctx, regression_only=regression_only, skip_binned_cls=skip_binned_cls, skip_binned_reg=skip_binned_reg, splice_combined=splice_combined, tau_groups=tau_groups)

    return results


def compute_binned(preds, ctx, regression_only=False, skip_binned_cls=False, skip_binned_reg=False, splice_combined=False, tau_groups=None):
    """binned metrics - ordered fast to slow. uses precomputed ctx for indices."""
    is_acc = ctx["is_acc"]
    is_don = ctx["is_don"]
    ssu = ctx["ssu"]
    masks = ctx["masks"]
    neither_idx = ctx["neither_idx"]
    site_bins = ctx["site_bins"]
    binned_reg_indices = ctx["binned_reg_indices"]
    pos_per_bin_cache = ctx["pos_per_bin_cache"]
    reg_subsets = ctx["reg_subsets"]
    cls_subsets = ctx["cls_subsets"]

    results = {
        "counts": {},
        "regression": {},
    }

    if not regression_only and not skip_binned_cls:
        n_neg = neither_idx.size
        rng = np.random.default_rng(seed=42)
        shuffled_neg = rng.permutation(n_neg)

        results["classification_ratio"] = {}

    # counts
    for b in range(10):
        in_bin = site_bins == b
        results["counts"][f"bin_{b}"] = {
            "n_acc": int((is_acc & masks["ssu_valid"] & in_bin).sum()),
            "n_don": int((is_don & masks["ssu_valid"] & in_bin).sum()),
        }

    # regression — use cached binned_reg_indices
    if skip_binned_reg:
        print("skipping binned regression (--phase=cls)", flush=True)
        binned_reg_iter = []
    else:
        print("computing binned regression", flush=True)
        binned_reg_iter = reg_subsets
    for subset in binned_reg_iter:
        results["regression"][subset] = {}

        for b in range(10):
            if b not in binned_reg_indices.get(subset, {}):
                continue
            bri = binned_reg_indices[subset][b]
            site_idx = bri["site_idx"]
            y_true = bri["y_true_reg"]
            acc_idx = bri["acc_idx"]
            don_idx = bri["don_idx"]

            for model, p in preds.items():
                # combine acc+don pairs (averaged + per-rep + per-fold)
                acc_keys = sorted(k for k in p if k == "acceptor" or k.startswith("acceptor_rep") or k.startswith("acceptor_fold_"))
                don_keys = sorted(k for k in p if k == "donor" or k.startswith("donor_rep") or k.startswith("donor_fold_"))
                paired = []
                if "acceptor" in acc_keys and "donor" in don_keys:
                    paired.append(("acceptor", "donor", "_cls"))
                for ak in acc_keys:
                    if ak == "acceptor":
                        continue
                    suffix = ak[len("acceptor_"):]
                    dk = f"donor_{suffix}"
                    if dk in don_keys:
                        paired.append((ak, dk, f"_cls_{suffix}"))

                for ak, dk, col_suffix in paired:
                    col = f"{model}{col_suffix}"
                    if bri["y_true_cls"] is not None:
                        y_true_cls = bri["y_true_cls"]
                        y_pred_cls = np.concatenate([p[ak][acc_idx], p[dk][don_idx]]).astype(np.float32)
                        if col not in results["regression"][subset]:
                            results["regression"][subset][col] = {}
                        results["regression"][subset][col][f"bin_{b}"] = regression(y_true_cls, y_pred_cls)

                paired_keys = {k for ak, dk, _ in paired for k in (ak, dk)}
                for key, arr in p.items():
                    if key in paired_keys or key == "neither":
                        continue
                    col = f"{model}_{key}"
                    if col not in results["regression"][subset]:
                        results["regression"][subset][col] = {}
                    results["regression"][subset][col][f"bin_{b}"] = regression(y_true, arr[site_idx].astype(np.float32))

    # ratio-preserving binned classification
    if not regression_only and not skip_binned_cls:
        print("computing binned classification (ratio-preserving)", flush=True)
        for subset in cls_subsets:
            results["classification_ratio"][subset] = {}
            for target in ["acceptor", "donor"]:
                pos_per_bin = pos_per_bin_cache[subset][target]
                if not pos_per_bin:
                    continue
                neg_per_bin = balance_ratio_preserving(pos_per_bin, shuffled_neg, n_neg)
                target_results = run_binned_classification(pos_per_bin, neg_per_bin, neither_idx, preds, target)
                results["classification_ratio"][subset][target] = target_results
    elif skip_binned_cls:
        print("skipping binned classification (--no-binned-cls)", flush=True)

    # ratio-preserving classification per tau group (constitutive / intermediate / tissue-specific)
    # other-group positives are excluded entirely (positive only counts in its own group)
    if (not regression_only) and (tau_groups is not None):
        print("computing tau-group classification (ratio-preserving)", flush=True)
        results["classification_tau_group"] = {}
        group_names = {0: "constitutive", 1: "intermediate", 2: "tissue-specific"}
        for subset in cls_subsets:
            mask = masks[subset]
            results["classification_tau_group"][subset] = {}
            for target, is_target in [("acceptor", is_acc), ("donor", is_don)]:
                pos_per_grp = {}
                for g_int, g_name in group_names.items():
                    pos_mask = is_target & mask & (tau_groups == g_int)
                    idx = np.flatnonzero(pos_mask)
                    if idx.size > 0:
                        pos_per_grp[g_name] = idx
                if not pos_per_grp:
                    continue
                neg_per_grp = balance_ratio_preserving(pos_per_grp, shuffled_neg, n_neg)
                target_results = run_grouped_classification(pos_per_grp, neg_per_grp, neither_idx, preds, target)
                results["classification_tau_group"][subset][target] = target_results

    return results


def run_grouped_classification(pos_per_grp, neg_per_grp, neither_idx, preds, target):
    """run classification for all tau groups, parallelized; same shape as run_binned_classification"""
    other = "donor" if target == "acceptor" else "acceptor"
    tasks = []
    for model, p in preds.items():
        for key, arr in p.items():
            if key == "neither":
                continue
            if key == "ssu" or key.startswith("ssu_") or key.endswith("_usage"):
                continue
            if key == other or key.startswith(f"{other}_rep") or key.startswith(f"{other}_fold_"):
                continue
            col = model if key in ["acceptor", "donor"] else f"{model}_{key}"
            for g in pos_per_grp:
                pos = arr[pos_per_grp[g]].astype(np.float32)
                neg = arr[neither_idx[neg_per_grp[g]]].astype(np.float32)
                tasks.append((g, col, pos, neg))

    def worker(t):
        g, name, pos, neg = t
        return g, name, classification_posfirst(pos, neg)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        out = list(pool.map(worker, tasks))

    target_results = {}
    for g, name, m in out:
        if name not in target_results:
            target_results[name] = {}
        target_results[name][g] = m

    return target_results


def run_binned_classification(pos_per_bin, neg_per_bin, neither_idx, preds, target):
    """run classification for all bins, parallelized"""
    other = "donor" if target == "acceptor" else "acceptor"
    tasks = []
    for model, p in preds.items():
        for key, arr in p.items():
            if key == "neither":
                continue
            if key == "ssu" or key.startswith("ssu_") or key.endswith("_usage"):
                continue
            # skip wrong-target acc/don keys (incl. per-rep + per-fold)
            if key == other or key.startswith(f"{other}_rep") or key.startswith(f"{other}_fold_"):
                continue
            col = model if key in ["acceptor", "donor"] else f"{model}_{key}"
            for b in pos_per_bin:
                pos = arr[pos_per_bin[b]].astype(np.float32)
                neg = arr[neither_idx[neg_per_bin[b]]].astype(np.float32)
                tasks.append((b, col, pos, neg))

    def worker(t):
        b, name, pos, neg = t
        return b, name, classification_posfirst(pos, neg)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        out = list(pool.map(worker, tasks))

    target_results = {}
    for b, name, m in out:
        if name not in target_results:
            target_results[name] = {}
        target_results[name][f"bin_{b}"] = m

    return target_results


def balance_ratio_preserving(pos_per_bin, shuffled_neg, n_neg):
    """ratio-preserving negative subsampling"""
    bins_by_size = sorted(pos_per_bin, key=lambda b: pos_per_bin[b].size, reverse=True)
    max_bin = bins_by_size[0]
    ratio = n_neg / pos_per_bin[max_bin].size

    out = {max_bin: shuffled_neg}
    for b in bins_by_size[1:]:
        out[b] = shuffled_neg[:int(pos_per_bin[b].size * ratio)]
    return out


if __name__ == "__main__":
    sys.exit(main())
