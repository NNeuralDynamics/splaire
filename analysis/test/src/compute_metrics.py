#!/usr/bin/env python3
"""compute metrics from parquet prediction files"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

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
}


def find_prediction_files(pred_dir):
    """find parquet files matching models dict"""
    files = {}
    for pq in pred_dir.glob("*.parquet"):
        for suffix, (name, _cols) in models.items():
            if suffix in pq.stem:
                files[name] = pq
                break
    return files


def load_predictions(pred_dir):
    """load ground truth and model predictions from parquets"""
    files = find_prediction_files(pred_dir)
    gt_cols = ["chrom", "pos", "strand", "y_acceptor", "y_donor", "y_ssu"]
    any_df = pd.read_parquet(next(iter(files.values())), columns=gt_cols)
    truth = {
        "acceptor": any_df["y_acceptor"].values.astype(np.float32),
        "donor": any_df["y_donor"].values.astype(np.float32),
        "ssu": any_df["y_ssu"].values.astype(np.float32),
    }
    coords = any_df[["chrom", "pos", "strand"]].copy()
    del any_df
    # reverse lookup: model name -> expected columns
    name_to_cols = {name: cols for _suffix, (name, cols) in models.items()}
    preds = {}
    for name, path in files.items():
        cols = name_to_cols[name]
        df = pd.read_parquet(path, columns=cols)
        preds[name] = {c: df[c].values.astype(np.float32) for c in cols}
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
    test_chroms = {f"chr{c}" for c in [1, 3, 5, 7]}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_json")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-binned", action="store_true")
    parser.add_argument("--no-cls", action="store_true", help="skip all classification metrics")
    parser.add_argument("--regression-only", action="store_true", help="only compute regression (overall + binned) on valid-SSU splice sites")
    parser.add_argument("--splicing-matrix", help="processed_splicing_matrix.tsv for shared-valid site filtering")
    parser.add_argument("--splice-combined", action="store_true", help="use max(acc,don) for classification (for data where all sites are both)")
    parser.add_argument("--tau-groups", help="CSV (chrom, pos, strand, group) for per-site tau group classification; adds classification_tau_group block to output")
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
        results = compute_metrics(truth, preds, skip_binned=skip_binned, skip_cls=skip_cls, splice_combined=args.splice_combined, regression_only=args.regression_only, tau_groups=tau_groups)
        with open(output, 'w') as f:
            json.dump({"overall": results}, f, indent=2, default=float)

    else:
        print(f"processing {input_path.name}", flush=True)
        truth, preds, coords = load_predictions(input_path)

        print(f"{truth['acceptor'].size:,} positions, {(truth['acceptor'] == 1).sum():,} acceptors", flush=True)

        shared_mask = None
        if args.splicing_matrix:
            shared_sites = load_shared_sites(args.splicing_matrix)
            shared_mask = build_shared_mask(coords, shared_sites)
            is_splice = (truth["acceptor"] == 1) | (truth["donor"] == 1)
            print(f"shared-valid: {len(shared_sites):,} sites in matrix, {(shared_mask & is_splice).sum():,} splice sites matched", flush=True)

        tau_groups = load_tau_groups(args.tau_groups, coords) if args.tau_groups else None

        skip_cls = args.no_cls or args.regression_only
        skip_binned = args.no_binned and not args.regression_only
        results = compute_metrics(truth, preds, skip_binned=skip_binned, skip_cls=skip_cls, splice_combined=args.splice_combined, regression_only=args.regression_only, valid_all_mask=shared_mask, tau_groups=tau_groups)
        with open(output, 'w') as f:
            json.dump({"overall": results}, f, indent=2, default=float)

    print(f"wrote {args.output_json}", flush=True)
    return 0


def compute_metrics(truth, preds, skip_binned=False, skip_cls=False, splice_combined=False, regression_only=False, valid_all_mask=None, tau_groups=None):
    """classification + regression metrics for all models"""
    is_acc = truth["acceptor"] == 1
    is_don = truth["donor"] == 1
    is_neither = ~is_acc & ~is_don
    ssu = truth["ssu"]

    # all splice sites marked as both acceptor and donor
    if splice_combined:
        is_splice = is_acc | is_don
        print(f"splice_combined mode: {is_splice.sum():,} splice sites", flush=True)

    if regression_only:
        # only positions with valid ssu
        n_valid = (ssu != sentinel).sum()
        n_splice_valid = ((is_acc | is_don) & (ssu != sentinel)).sum()
        print(f"regression-only mode: {n_splice_valid:,} splice sites with valid SSU (of {n_valid:,} valid)", flush=True)
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

    # sites valid in all samples from --splicing-matrix
    if valid_all_mask is not None:
        masks["ssu_shared"] = valid_all_mask
        masks["ssu_shared_nonzero"] = valid_all_mask & (ssu > 0)

    neither_idx = np.flatnonzero(is_neither)

    # regression subsets, everything except "all" which has no ssu ground truth
    reg_subsets = [k for k in masks if k != "all"]

    results = {
        "counts": {
            "n_positions": int(ssu.size),
            "n_acceptors": int(is_acc.sum()),
            "n_donors": int(is_don.sum()),
            "n_neither": int(neither_idx.size),
        },
        "classification": {},
        "regression": {},
    }

    if not regression_only:
        # pre-compute indices for classification
        cls_indices = {}
        for subset, mask in masks.items():
            cls_indices[subset] = {}
            if splice_combined:
                pos_mask = is_splice & mask
                pos_idx = np.flatnonzero(pos_mask)
                include_idx = np.flatnonzero(pos_mask | is_neither)
                y_true = pos_mask[include_idx].astype(np.float16)
                cls_indices[subset]["splice"] = {"pos_idx": pos_idx, "include_idx": include_idx, "y_true": y_true}
            else:
                for target, is_target in [("acceptor", is_acc), ("donor", is_don)]:
                    pos_mask = is_target & mask
                    pos_idx = np.flatnonzero(pos_mask)
                    include_idx = np.flatnonzero(pos_mask | is_neither)
                    y_true = pos_mask[include_idx].astype(np.float16)
                    cls_indices[subset][target] = {"pos_idx": pos_idx, "include_idx": include_idx, "y_true": y_true}

    # add combined predictions (max of acc/don) for models with both outputs
    if splice_combined:
        for model, p in preds.items():
            if "acceptor" in p and "donor" in p:
                p["splice"] = np.maximum(p["acceptor"], p["donor"])

    # regression
    print("computing regression", flush=True)
    for subset in reg_subsets:
        mask = masks[subset]
        site_idx = np.flatnonzero((is_acc | is_don) & mask)
        y_true = ssu[site_idx].astype(np.float32)

        if not splice_combined:
            acc_idx = np.flatnonzero(is_acc & mask)
            don_idx = np.flatnonzero(is_don & mask)
            y_true_cls = np.concatenate([ssu[acc_idx], ssu[don_idx]]).astype(np.float32)

        results["regression"][subset] = {}
        for model, p in preds.items():
            # acceptor/donor outputs combined
            if "acceptor" in p and "donor" in p:
                col = f"{model}_cls"
                if splice_combined:
                    y_pred_cls = p["splice"][site_idx].astype(np.float32)
                    results["regression"][subset][col] = regression(y_true, y_pred_cls)
                else:
                    y_pred_cls = np.concatenate([p["acceptor"][acc_idx], p["donor"][don_idx]]).astype(np.float32)
                    results["regression"][subset][col] = regression(y_true_cls, y_pred_cls)

            # single-score outputs like ssu, tissue predictions
            for key, arr in p.items():
                if key in ["acceptor", "donor", "neither", "splice"]:
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
                        if splice_combined:
                            if key in ["acceptor", "donor"]:
                                continue
                            col = model if key == "splice" else f"{model}_{key}"
                        else:
                            if key != target and key in ["acceptor", "donor"]:
                                continue
                            col = model if key in ["acceptor", "donor"] else f"{model}_{key}"
                        tasks.append((col, arr[include_idx].astype(np.float16)))

                target_results = {}
                if tasks:
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futures = {pool.submit(classification, y_true, y_pred): name for name, y_pred in tasks}
                        for fut in as_completed(futures):
                            target_results[futures[fut]] = fut.result()

                results["classification"][subset][target] = target_results

    # binned metrics
    if not skip_binned:
        results["binned"] = compute_binned(is_acc, is_don, ssu, masks, neither_idx, preds, regression_only=regression_only, tau_groups=tau_groups)

    return results


def compute_binned(is_acc, is_don, ssu, masks, neither_idx, preds, regression_only=False, tau_groups=None):
    """binned metrics - ordered fast to slow"""
    bin_edges = np.arange(0, 1.1, 0.1)
    site_bins = np.clip(np.digitize(ssu, bin_edges) - 1, 0, 9)

    reg_subsets = [k for k in masks if k != "all"]
    # classification only uses ssu_valid/ssu_valid_nonzero, not ssu_shared
    cls_subsets = [k for k in reg_subsets if not k.startswith("ssu_shared")]

    results = {
        "counts": {},
        "regression": {},
    }

    if not regression_only:
        n_neg = neither_idx.size
        rng = np.random.default_rng(seed=42)
        shuffled_neg = rng.permutation(n_neg)

        results["classification_ratio"] = {}

        # pre-compute pos_per_bin for all subsets/targets
        pos_per_bin_cache = {}
        for subset in cls_subsets:
            mask = masks[subset]
            pos_per_bin_cache[subset] = {}
            for target, is_target in [("acceptor", is_acc), ("donor", is_don)]:
                pos_mask = is_target & mask
                pos_per_bin = {b: np.flatnonzero(pos_mask & (site_bins == b)) for b in range(10)}
                pos_per_bin = {b: idx for b, idx in pos_per_bin.items() if idx.size > 0}
                pos_per_bin_cache[subset][target] = pos_per_bin

    # counts
    for b in range(10):
        in_bin = site_bins == b
        results["counts"][f"bin_{b}"] = {
            "n_acc": int((is_acc & masks["ssu_valid"] & in_bin).sum()),
            "n_don": int((is_don & masks["ssu_valid"] & in_bin).sum()),
        }

    # regression
    print("computing binned regression", flush=True)
    for subset in reg_subsets:
        mask = masks[subset]
        results["regression"][subset] = {}

        for b in range(10):
            in_bin = site_bins == b
            site_idx = np.flatnonzero((is_acc | is_don) & mask & in_bin)
            if site_idx.size == 0:
                continue

            y_true = ssu[site_idx].astype(np.float32)
            acc_idx = np.flatnonzero(is_acc & mask & in_bin)
            don_idx = np.flatnonzero(is_don & mask & in_bin)

            for model, p in preds.items():
                if "acceptor" in p and "donor" in p:
                    col = f"{model}_cls"
                    if acc_idx.size > 0 or don_idx.size > 0:
                        y_true_cls = np.concatenate([ssu[acc_idx], ssu[don_idx]]).astype(np.float32)
                        y_pred_cls = np.concatenate([p["acceptor"][acc_idx], p["donor"][don_idx]]).astype(np.float32)
                        if col not in results["regression"][subset]:
                            results["regression"][subset][col] = {}
                        results["regression"][subset][col][f"bin_{b}"] = regression(y_true_cls, y_pred_cls)

                for key, arr in p.items():
                    if key in ["acceptor", "donor", "neither"]:
                        continue
                    col = f"{model}_{key}"
                    if col not in results["regression"][subset]:
                        results["regression"][subset][col] = {}
                    results["regression"][subset][col][f"bin_{b}"] = regression(y_true, arr[site_idx].astype(np.float32))

    # ratio-preserving binned classification
    if not regression_only:
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
    tasks = []
    for model, p in preds.items():
        for key, arr in p.items():
            if key == "neither":
                continue
            if key != target and key in ["acceptor", "donor"]:
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
    tasks = []
    for model, p in preds.items():
        for key, arr in p.items():
            if key == "neither":
                continue
            # skip acc/don keys that dont match current target
            if key != target and key in ["acceptor", "donor"]:
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
