#!/usr/bin/env python
# coding: utf-8

import os
import re
import shutil
import subprocess
import sys
import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.image import imread as _imread_panel
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.transforms import blended_transform_factory
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import precision_recall_curve, average_precision_score, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
from tqdm import tqdm

# cli flags
# usage: python analysis.py [--skip-bootstrap]
SKIP_BOOTSTRAP = "--skip-bootstrap" in sys.argv

# tracks figures skipped because their inputs (GTF, gnomAD VCF) are not in the
# Zenodo bundle. read by analysis.qmd's show() helper to render a callout.
_ZENODO_SKIPPED = set()

# 5kb on each side of splice site = 10,001 bp total window
WINDOW_HALF = 5000
SEQ_LEN = 2 * WINDOW_HALF + 1

# reference genome path — canonical location: pipeline/reference/GRCh37/hg19.fa
# falls back to local vex_seq/ or mfass/ caches. not on zenodo.
_FASTA_CANDIDATES = [
    Path(__file__).resolve().parent / ".." / ".." / "pipeline" / "reference" / "GRCh37" / "hg19.fa",
    Path("../../pipeline/reference/GRCh37/hg19.fa"),
    Path("vex_seq/hg19.fa"),
    Path("mfass/hg19.fa"),
]
FASTA_PATH = next((p for p in _FASTA_CANDIDATES if p.exists()), _FASTA_CANDIDATES[0])

# reverse complement
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

# data roots resolved via report_utils (zenodo bundle)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import report_utils as _ru
_paths = _ru.get_paths()
_DATA = _paths.reporter

vex_data_dir = _DATA / "vex_seq" / "data"
mfass_data_dir = _DATA / "mfass" / "data"

# output directories
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)
fig3_main = Path("figures/main")
fig3_sup = Path("figures/sup")
fig3_main.mkdir(parents=True, exist_ok=True)
fig3_sup.mkdir(parents=True, exist_ok=True)

ANNOT_SIZE = 12

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.titleweight": "normal",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def _save(out_dir, stem, *, tight=True, dpi=300, pdf=True):
    """save current figure as png (+pdf) to {out_dir}/{stem}, close all figs"""
    if tight:
        plt.tight_layout()
    out = Path(out_dir) / stem
    plt.savefig(str(out) + ".png", dpi=dpi, bbox_inches="tight")
    if pdf:
        plt.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# sequence helpers
# ---------------------------------------------------------------------------

def revcomp(seq):
    return seq.translate(COMP)[::-1]


def one_hot(seq):
    seq = seq.upper()
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in BASE_TO_IDX:
            arr[i, BASE_TO_IDX[base]] = 1.0
    return arr


def get_window(chrom, center, fasta, half=WINDOW_HALF):
    """fetch a window of sequence centered on a position, pad with Ns at edges"""
    chrom_len = fasta.get_reference_length(str(chrom))
    start1 = int(center) - half
    end1 = int(center) + half

    left_pad = max(0, 1 - start1)
    right_pad = max(0, end1 - chrom_len)
    fetch_start = max(0, start1 - 1)
    fetch_end = min(chrom_len, end1)
    seq = fasta.fetch(str(chrom), fetch_start, fetch_end).upper()

    if left_pad:
        seq = "N" * left_pad + seq
    if right_pad:
        seq = seq + "N" * right_pad
    return seq, start1, end1


def build_ref_alt_pair(chrom, center, var_pos, ref_allele, alt_allele, fasta):
    """build reference and alternate sequences for a variant"""
    ref_seq, start1, end1 = get_window(chrom, center, fasta)
    ref_allele = ref_allele.upper()
    alt_allele = alt_allele.upper()

    idx = int(var_pos) - start1
    variant_is_upstream = idx < WINDOW_HALF

    if not (0 <= idx < len(ref_seq) - len(ref_allele) + 1):
        return None, None, "variant outside window"

    genome_ref = ref_seq[idx:idx + len(ref_allele)]
    if genome_ref != ref_allele:
        return None, None, f"ref mismatch: expected {ref_allele}, got {genome_ref}"

    alt_seq = ref_seq[:idx] + alt_allele + ref_seq[idx + len(ref_allele):]

    expected_len = 2 * WINDOW_HALF + 1
    len_diff = len(alt_seq) - expected_len

    if len_diff != 0:
        chrom_len = fasta.get_reference_length(str(chrom))
        if variant_is_upstream:
            if len_diff > 0:
                alt_seq = alt_seq[len_diff:]
            else:
                need = -len_diff
                new_start = start1 - 1 - need
                if new_start < 0:
                    extra = fasta.fetch(str(chrom), 0, start1 - 1).upper()
                    extra = "N" * (need - len(extra)) + extra
                else:
                    extra = fasta.fetch(str(chrom), new_start, start1 - 1).upper()
                alt_seq = extra + alt_seq
        else:
            if len_diff > 0:
                alt_seq = alt_seq[:expected_len]
            else:
                need = -len_diff
                extra_end = min(chrom_len, end1 + need)
                extra = fasta.fetch(str(chrom), end1, extra_end).upper()
                if len(extra) < need:
                    extra = extra + "N" * (need - len(extra))
                alt_seq = alt_seq + extra

    assert len(ref_seq) == expected_len, f"ref_seq wrong length: {len(ref_seq)}"
    assert len(alt_seq) == expected_len, f"alt_seq wrong length: {len(alt_seq)}"

    return ref_seq, alt_seq, None


# ---------------------------------------------------------------------------
# model config + colors
# ---------------------------------------------------------------------------

model_colors = {
    "pangolin":          "#009E73",
    "pangolin_v2":       "#2CA02C",
    "spliceai":          "#D55E00",
    "splicetransformer": "#E69F00",
    "alphagenome":       "#9467BD",
    "splaire_ref":        "#56B4E9",
    "splaire_var":        "#0072B2",
    "splaire_avg":        "#CC79A7",
    "gencode":           "#666666",
}


def _get_base_model(key):
    """extract base model name from output key"""
    key = key.lower().replace("-", "_").replace(" ", "_")
    if key in model_colors:
        return key
    for base in sorted(model_colors.keys(), key=len, reverse=True):
        if key.startswith(base):
            return base
    return None


def get_color(key):
    """get color for model output"""
    base = _get_base_model(key)
    if base:
        return model_colors[base]
    return "#999999"


models = ["pangolin", "pangolin_v2", "spliceai", "splicetransformer", "alphagenome", "splaire_ref", "splaire_var"]

model_names = {
    "pangolin": "Pangolin",
    "pangolin_usage": "Pangolin (max usage)",
    "pangolin_v2": "Pangolin v2",
    "pangolin_v2_usage": "Pangolin v2 (max usage)",
    "spliceai": "SpliceAI",
    "splicetransformer": "SpliceTransformer",
    "splicetransformer_usage": "SpTransformer (max usage)",
    "alphagenome": "AlphaGenome",
    "alphagenome_usage": "AlphaGenome (max usage)",
    "splaire_ref": "SPLAIRE",
    "splaire_var": "SPLAIRE-var",
}

datasets = {
    "vexseq": {"path": str(vex_data_dir), "prefix": "vex_seq", "thr": 0.10, "psi_scale": 100},
    "mfass": {"path": str(mfass_data_dir), "prefix": "mfass", "thr": 0.50, "psi_scale": 1},
}

dataset_names = {
    "vexseq": "Vex-seq", "mfass": "MFASS",
    "compass_minigene_refs":    "COMPASS (minigene, refs)",
    "compass_minigene_snvs":    "COMPASS (minigene, SNVs)",
    "compass_minigene_doubles": "COMPASS (minigene, doubles)",
    "compass_genome_refs":      "COMPASS (genome, refs)",
    "compass_genome_snvs":      "COMPASS (genome, SNVs)",
    "compass_genome_doubles":   "COMPASS (genome, doubles)",
    "opensplice_minigene_snvs": "OpenSplice (minigene, SNVs)",
    "opensplice_minigene_dels": "OpenSplice (minigene, dels)",
    "opensplice_genome_snvs":   "OpenSplice (genome, SNVs)",
    "opensplice_genome_dels":   "OpenSplice (genome, dels)",
    # merged buckets — see _merge_compass_opensplice in notebook
    "compass_minigene":   "COMPASS (minigene)",
    "compass_genome":     "COMPASS (genome)",
    "opensplice_minigene":"OpenSplice (minigene)",
    "opensplice_genome":  "OpenSplice (genome)",
}


_MERGE_BUCKETS = {
    # compass_minigene / compass_genome = refs + snvs only. doubles stay as standalone
    # compass_*_doubles datasets — all 6 sequence-only models we benchmark get Pearson ~0.05
    # on 2-substitution variants (none of them can resolve epistasis), so bundling them in
    # would drag combined-bucket Pearson from ~0.46 to ~0.20. See notebook cell 29.
    "compass_minigene":   ["compass_minigene_refs", "compass_minigene_snvs"],
    "compass_genome":     ["compass_genome_refs", "compass_genome_snvs"],
    "opensplice_minigene":["opensplice_minigene_snvs", "opensplice_minigene_dels"],
    "opensplice_genome":  ["opensplice_genome_snvs", "opensplice_genome_dels"],
}


def merge_compass_opensplice(d):
    """bundle compass/opensplice subsets into {genome, minigene} buckets in-place

    each row gets `sub_dataset` recording its origin; raw_dfs, dfs_filt,
    dfs_neutral, sd_info, and the global `datasets` thr_sd entry are all
    rebuilt for the merged names. sub-dataset entries are removed.
    """
    for new_name, subs in _MERGE_BUCKETS.items():
        present = [s for s in subs if s in d.dfs]
        if not present:
            continue
        parts = []
        for s in present:
            sub = d.dfs[s].copy()
            sub["sub_dataset"] = s
            parts.append(sub)
        merged = pd.concat(parts, ignore_index=True)
        d.dfs[new_name] = merged
        # recompute sd_info, dfs_filt, dfs_neutral, labels
        y = merged["y"].values
        finite = np.isfinite(y)
        if finite.sum() > 1:
            mu = float(y[finite].mean())
            sd = float(y[finite].std())
            d.sd_info[new_name] = (mu, sd, mu - sd, mu + sd)
            merged.loc[:, "label"] = ((np.abs(merged["y"]) > sd) & finite).astype(int)
            merged.loc[:, "label_fixed"] = ((np.abs(merged["y"]) > 0.50) & finite).astype(int)
            mask = (np.abs(merged["y"]) > sd) & finite
            d.dfs_filt[new_name] = merged[mask].reset_index(drop=True)
            d.dfs_neutral[new_name] = merged[~mask & finite].reset_index(drop=True)
        else:
            d.sd_info[new_name] = (np.nan, np.nan, np.nan, np.nan)
        # backfill datasets dict (used by figs for thr_sd lookups)
        datasets[new_name] = {"path": "", "prefix": new_name, "thr": 0.50,
                              "psi_scale": 1.0, "thr_sd": d.sd_info[new_name][1]}
        # merge raw_dfs per model
        all_models = set()
        for s in present:
            all_models.update(d.raw_dfs.get(s, {}).keys())
        if all_models:
            new_raw = {}
            for m in all_models:
                parts_m = [d.raw_dfs[s][m] for s in present if s in d.raw_dfs and m in d.raw_dfs[s]]
                if parts_m:
                    new_raw[m] = pd.concat(parts_m, ignore_index=True)
            d.raw_dfs[new_name] = new_raw
        # drop sub-datasets from all per-dataset stores
        for s in present:
            d.dfs.pop(s, None)
            d.dfs_filt.pop(s, None)
            d.dfs_neutral.pop(s, None)
            d.sd_info.pop(s, None)
            d.raw_dfs.pop(s, None)
            datasets.pop(s, None)
        n_lab = int(merged["y"].notna().sum())
        print(f"{new_name}: merged {len(present)} subs -> {len(merged):,} rows ({n_lab:,} labeled)")

# max-aggregated output column per model
MODEL_COLS = {
    "pangolin": "pangolin_max_p_splice",
    "pangolin_v2": "pangolin_v2_max_p_splice",
    "spliceai": "spliceai_cls_delta",
    "splicetransformer": "splicetransformer_cls_delta",
    "alphagenome": "alphagenome_cls_delta",
    "splaire_ref": "splaire_ref_cls_delta",
    "splaire_var": "splaire_var_cls_delta",
}
MODEL_LIST = list(MODEL_COLS.keys())
split_cols = MODEL_COLS
split_models = MODEL_LIST


def _fig_setup(d, name):
    """common per-dataset prelude: returns (df, y_true, mods)"""
    df = d.dfs[name]
    return df, df["y"].values, [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]

pr_cols = {
    "vexseq": {
        "pangolin": "pangolin_testis_p_splice_delta",
        "pangolin_v2": "pangolin_v2_testis_p_splice_delta",
        "spliceai": "spliceai_cls_delta",
        "splicetransformer": "splicetransformer_usage_Blood_Vessel_delta",
        "alphagenome": "alphagenome_cls_delta",
        "splaire_ref": "splaire_ref_cls_delta",
        "splaire_var": "splaire_var_cls_delta",
    },
    "mfass": {
        "pangolin": "pangolin_testis_p_splice_delta",
        "pangolin_v2": "pangolin_v2_testis_p_splice_delta",
        "spliceai": "spliceai_cls_delta",
        "splicetransformer": "splicetransformer_usage_Lung_delta",
        "alphagenome": "alphagenome_cls_delta",
        "splaire_ref": "splaire_ref_cls_delta",
        "splaire_var": "splaire_var_cls_delta",
    },
}

thresh_cols = pr_cols  # same column set

# ref/alt column mapping (matching MODEL_COLS pattern)
REF_COLS = {
    "pangolin": "pangolin_max_p_splice_ref",
    "pangolin_v2": "pangolin_v2_max_p_splice_ref",
    "spliceai": "spliceai_cls_ref",
    "splicetransformer": "splicetransformer_cls_ref",
    "alphagenome": "alphagenome_cls_ref",
    "splaire_ref": "splaire_ref_cls_ref",
    "splaire_var": "splaire_var_cls_ref",
}
ALT_COLS = {
    "pangolin": "pangolin_max_p_splice_alt",
    "pangolin_v2": "pangolin_v2_max_p_splice_alt",
    "spliceai": "spliceai_cls_alt",
    "splicetransformer": "splicetransformer_cls_alt",
    "alphagenome": "alphagenome_cls_alt",
    "splaire_ref": "splaire_ref_cls_alt",
    "splaire_var": "splaire_var_cls_alt",
}

combined_cols = [
    "pangolin_brain_p_splice_delta", "pangolin_brain_usage_delta",
    "pangolin_heart_p_splice_delta", "pangolin_heart_usage_delta",
    "pangolin_liver_p_splice_delta", "pangolin_liver_usage_delta",
    "pangolin_testis_p_splice_delta", "pangolin_testis_usage_delta",
    "pangolin_max_p_splice", "pangolin_max_usage",
    "pangolin_v2_brain_p_splice_delta", "pangolin_v2_brain_usage_delta",
    "pangolin_v2_heart_p_splice_delta", "pangolin_v2_heart_usage_delta",
    "pangolin_v2_liver_p_splice_delta", "pangolin_v2_liver_usage_delta",
    "pangolin_v2_testis_p_splice_delta", "pangolin_v2_testis_usage_delta",
    "pangolin_v2_max_p_splice", "pangolin_v2_max_usage",
    "splaire_ref_cls_delta", "splaire_ref_reg_ssu_delta",
    "splaire_var_cls_delta", "splaire_var_reg_ssu_delta",
    "spliceai_cls_delta", "splicetransformer_cls_delta",
    "splicetransformer_max_usage",
    "splicetransformer_usage_Adipose_Tissue_delta", "splicetransformer_usage_Blood_Vessel_delta",
    "splicetransformer_usage_Blood_delta", "splicetransformer_usage_Brain_delta",
    "splicetransformer_usage_Colon_delta", "splicetransformer_usage_Heart_delta",
    "splicetransformer_usage_Kidney_delta", "splicetransformer_usage_Liver_delta",
    "splicetransformer_usage_Lung_delta", "splicetransformer_usage_Muscle_delta",
    "splicetransformer_usage_Nerve_delta", "splicetransformer_usage_Skin_delta",
    "splicetransformer_usage_Small_Intestine_delta", "splicetransformer_usage_Spleen_delta",
    "splicetransformer_usage_Stomach_delta",
]


def col_display(c):
    """pretty name for delta column"""
    has_delta = c.endswith('_delta')
    base = c.replace('_delta', '') if has_delta else c
    suffix = ' Delta' if has_delta else ''
    if base == 'splaire_ref_cls': return f'SPLAIRE CLS{suffix}'
    if base == 'splaire_ref_reg_ssu': return f'SPLAIRE SSU{suffix}'
    if base == 'splaire_var_cls': return f'SPLAIRE-var CLS{suffix}'
    if base == 'splaire_var_reg_ssu': return f'SPLAIRE-var SSU{suffix}'
    if base == 'spliceai_cls': return f'SpliceAI CLS{suffix}'
    if base == 'splicetransformer_cls': return f'SpliceTransformer CLS{suffix}'
    if base == 'splicetransformer_max_usage': return f'SpliceTransformer Max Usage{suffix}'
    if base.startswith('splicetransformer_usage_'):
        tissue = base.replace('splicetransformer_usage_', '').replace('_', ' ')
        return f'SpliceTransformer {tissue}{suffix}'
    if base.startswith('pangolin_v2_'):
        rest = base.replace('pangolin_v2_', '')
        if rest.endswith('_p_splice'):
            tissue = rest.replace('_p_splice', '').replace('_', ' ').title()
            return f'Pangolin v2 {tissue} CLS{suffix}'
        if rest.endswith('_usage'):
            tissue = rest.replace('_usage', '').replace('_', ' ').title()
            return f'Pangolin v2 {tissue} Usage{suffix}'
        return f'Pangolin v2 {rest.replace("_", " ").title()}{suffix}'
    if base.startswith('pangolin_'):
        rest = base.replace('pangolin_', '')
        if rest.endswith('_p_splice'):
            tissue = rest.replace('_p_splice', '').replace('_', ' ').title()
            return f'Pangolin {tissue} CLS{suffix}'
        if rest.endswith('_usage'):
            tissue = rest.replace('_usage', '').replace('_', ' ').title()
            return f'Pangolin {tissue} Usage{suffix}'
        return f'Pangolin {rest.replace("_", " ").title()}{suffix}'
    return c


# ---------------------------------------------------------------------------
# location masks (Chong 2019 / Mount et al. 2019 categories)
# ---------------------------------------------------------------------------

def get_location_masks(pos, exon_start, exon_end):
    """boolean masks for variant location categories — combined + acc/don split"""
    in_exon = (pos >= exon_start) & (pos <= exon_end)
    ss_3p = (pos >= exon_start - 8) & (pos <= exon_start + 2)
    ss_5p = (pos >= exon_end - 2) & (pos <= exon_end + 8)
    ss_broad = ss_3p | ss_5p
    ss_can_3p = (pos >= exon_start - 2) & (pos <= exon_start - 1)
    ss_can_5p = (pos >= exon_end + 1) & (pos <= exon_end + 2)
    splice_site = ss_can_3p | ss_can_5p
    splice_region = ss_broad & ~splice_site
    splice_region_3p = ss_3p & ~ss_can_3p
    splice_region_5p = ss_5p & ~ss_can_5p
    exon = in_exon & ~ss_broad
    intron = ~in_exon & ~ss_broad
    closer_to_acc = np.abs(pos - exon_start) <= np.abs(pos - exon_end)
    return {
        "all": np.ones(len(pos), dtype=bool),
        "exon": exon,
        "intron": intron,
        "splice_site": splice_site,
        "splice_region": splice_region,
        "splice_site_acc": ss_can_3p,
        "splice_site_don": ss_can_5p,
        "splice_region_acc": splice_region_3p,
        "splice_region_don": splice_region_5p,
        "exon_acc": exon & closer_to_acc,
        "exon_don": exon & ~closer_to_acc,
        "intron_acc": intron & (pos < exon_start),
        "intron_don": intron & (pos > exon_end),
    }


LOCATION_SUBSETS = ["all", "exon", "intron", "splice_region", "splice_site"]
LOCATION_LABELS = {
    "all": "All", "exon": "Exon", "intron": "Intron",
    "splice_site": "Splice site", "splice_region": "Splice region",
}
LOCATION_SUBSETS_AD = [
    "splice_site_acc", "splice_site_don",
    "splice_region_acc", "splice_region_don",
    "exon_acc", "exon_don",
    "intron_acc", "intron_don",
]
LOCATION_LABELS_AD = {
    "splice_site_acc": "SS (acc)", "splice_site_don": "SS (don)",
    "splice_region_acc": "SR (acc)", "splice_region_don": "SR (don)",
    "exon_acc": "Exon (acc)", "exon_don": "Exon (don)",
    "intron_acc": "Intron (acc)", "intron_don": "Intron (don)",
}
ALL_LOCATION_LABELS = {**LOCATION_LABELS, **LOCATION_LABELS_AD}


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------

def get_metrics(y, pred, label):
    mask = np.isfinite(y) & np.isfinite(pred)
    yt, yp, lab = y[mask], pred[mask], label[mask]
    if len(yt) < 3:
        return np.nan, np.nan, np.nan
    r = stats.pearsonr(yt, yp)[0]
    rho = stats.spearmanr(yt, yp)[0]
    auprc = average_precision_score(lab, np.abs(yp)) if lab.sum() > 0 else np.nan
    return r, rho, auprc


def compute_corrs(y_true, y_pred):
    """pearson, spearman, r2, n for finite pairs"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, 0
    yt, yp = y_true[mask], y_pred[mask]
    r = stats.pearsonr(yt, yp)[0]
    rho = stats.spearmanr(yt, yp)[0]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r, rho, r2, int(mask.sum())


# ---------------------------------------------------------------------------
# score loading helpers
# ---------------------------------------------------------------------------

def load_scores_as_deltas(path):
    """load h5 and compute deltas (alt - ref) for each head.

    non-AG models (spliceai, spt, splaire, pangolin, merlin) store 1D per-head
    columns (acceptor, donor, usage_<tissue>). AG stores 4-channel SS + 734-track
    SSU + 367-track SJ, so we split SS into strand-aware acceptor/donor columns
    (channels 0=don+ 1=acc+ 2=don- 3=acc-) and collapse SSU across tracks to a
    signed max-abs scalar per boundary. SJ is skipped (100% dead in vex/mfass).
    """
    with h5py.File(path, "r") as f:
        meta = {}
        # take n_variants from attrs when present; otherwise infer from the longest meta array
        n_variants = int(f.attrs.get("n_variants", 0)) or max(f["meta"][k].shape[0] for k in f["meta"].keys())
        for k in f["meta"].keys():
            arr = f["meta"][k][:]
            # skip auxiliary keys with a different row count (e.g. dropped_indices_hg38_liftover in vex_seq)
            if arr.ndim >= 1 and arr.shape[0] != n_variants:
                continue
            # decode bytes regardless of dtype (h5py vlen strings come back as object-dtype bytes)
            if arr.dtype.kind == "S":
                meta[k] = np.array([x.decode() for x in arr])
            elif arr.dtype == object and len(arr) and isinstance(arr[0], bytes):
                meta[k] = np.array([x.decode() for x in arr])
            else:
                meta[k] = arr

        is_ag = ("ss_exon_start_ref" in f["scores"]
                 and f["scores"]["ss_exon_start_ref"].ndim == 2
                 and f["scores"]["ss_exon_start_ref"].shape[-1] == 4)
        deltas = {}
        if is_ag:
            strand = meta["strand"]
            is_plus = strand == "+"
            n = len(strand); ridx = np.arange(n)
            # +strand rows use ch1/ch0 (acc+/don+); -strand rows use ch3/ch2 (acc-/don-)
            acc_ch = np.where(is_plus, 1, 3)
            don_ch = np.where(is_plus, 0, 2)
            for bnd in ("exon_start", "exon_end"):
                d = f["scores"][f"ss_{bnd}_alt"][:] - f["scores"][f"ss_{bnd}_ref"][:]
                deltas[f"acceptor_{bnd}_delta"] = d[ridx, acc_ch]
                deltas[f"donor_{bnd}_delta"] = d[ridx, don_ch]
                sd = f["scores"][f"ssu_{bnd}_alt"][:] - f["scores"][f"ssu_{bnd}_ref"][:]
                idx = np.abs(sd).argmax(axis=1)
                deltas[f"usage_max_{bnd}_delta"] = sd[ridx, idx]
        else:
            heads = set()
            for k in list(f["scores"].keys()):
                for suffix in ["_exon_start_ref", "_exon_start_alt", "_exon_end_ref", "_exon_end_alt"]:
                    if k.endswith(suffix):
                        heads.add(k.replace(suffix, ""))
            for head in sorted(heads):
                deltas[f"{head}_exon_start_delta"] = f["scores"][f"{head}_exon_start_alt"][:] - f["scores"][f"{head}_exon_start_ref"][:]
                deltas[f"{head}_exon_end_delta"] = f["scores"][f"{head}_exon_end_alt"][:] - f["scores"][f"{head}_exon_end_ref"][:]
    return pd.DataFrame(meta), pd.DataFrame(deltas)


def load_raw_scores(path):
    """load h5 and return per-boundary ref/alt scores for each head"""
    with h5py.File(path, "r") as f:
        keys = list(f["scores"].keys())
        heads = set()
        for k in keys:
            for suffix in ["_exon_start_ref", "_exon_start_alt", "_exon_end_ref", "_exon_end_alt"]:
                if k.endswith(suffix):
                    heads.add(k.replace(suffix, ""))
        scores = {}
        for head in sorted(heads):
            for bnd in ["exon_start", "exon_end"]:
                for allele in ["ref", "alt"]:
                    k = f"{head}_{bnd}_{allele}"
                    scores[k] = f["scores"][k][:]
    return pd.DataFrame(scores)


def _h5_name(prefix, m):
    if m == "pangolin": return f"{prefix}_pang.h5"
    if m == "pangolin_v2": return f"{prefix}_pang_v2.h5"
    if m == "spliceai": return f"{prefix}_sa.h5"
    if m == "splicetransformer": return f"{prefix}_spt.h5"
    if m == "alphagenome": return f"{prefix}_ag.h5"
    return f"{prefix}_{m}.h5"


# ---------------------------------------------------------------------------
# unified dataset config + registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    path: Path                            # data dir holding the input h5 + score h5s
    prefix: str                           # used by _h5_name to find {prefix}_<model>.h5
    input_h5_name: str = None             # filename for the meta-bearing h5 (defaults to {prefix}.h5)
    psi_scale: float = 1.0
    thr: float = 0.50
    force_plus_strand: bool = False       # compass/opensplice: sequences are pre-oriented to sense
    filter_unlabeled: bool = False        # mfass: drop rows with NaN delta_psi after load
    label_source: str = "h5_meta"         # h5_meta | vex_csv | compass_psi | opensplice_master
    compass_primary_cell: str = None      # used when label_source == "compass_psi"

    def input_h5_path(self):
        fn = self.input_h5_name or f"{self.prefix}.h5"
        return self.path / fn

    def score_h5_path(self, model):
        return self.path / _h5_name(self.prefix, model)


def _build_dataset_registry():
    """one canonical registry covering vex/mfass + all 10 compass/opensplice configs"""
    reg = [
        DatasetConfig(
            name="vexseq", path=vex_data_dir, prefix="vex_seq",
            psi_scale=100, thr=0.10, label_source="h5_meta",
        ),
        DatasetConfig(
            name="mfass", path=mfass_data_dir, prefix="mfass",
            psi_scale=1, thr=0.50, label_source="h5_meta",
            filter_unlabeled=True,
        ),
    ]
    # compass + opensplice live under _DATA / reporter root; not vex/mfass dirs
    compass_dir = _DATA / "compass" / "data"
    opensplice_dir = _DATA / "opensplice" / "data"
    for stem in ["compass_minigene_refs", "compass_minigene_snvs", "compass_minigene_doubles",
                 "compass_genome_refs", "compass_genome_snvs", "compass_genome_doubles"]:
        reg.append(DatasetConfig(
            name=stem, path=compass_dir, prefix=stem,
            psi_scale=1.0, thr=0.50,
            force_plus_strand=True,
            label_source="compass_psi", compass_primary_cell="HeLa",
        ))
    for stem in ["opensplice_minigene_snvs", "opensplice_minigene_dels",
                 "opensplice_genome_snvs", "opensplice_genome_dels"]:
        reg.append(DatasetConfig(
            name=stem, path=opensplice_dir, prefix=stem,
            psi_scale=1.0, thr=0.50,
            force_plus_strand=True,
            label_source="opensplice_master",
        ))
    return reg


DATASET_CONFIGS = _build_dataset_registry()
DATASET_CONFIGS_BY_NAME = {c.name: c for c in DATASET_CONFIGS}


# label sourcing helpers (one per cfg.label_source)
# each returns a tuple: (labels_array, extra_cols_dict) — extras get assigned onto df

def _labels_from_h5_meta(cfg, meta_df):
    """vexseq + mfass: delta_psi already lives in meta"""
    labels = meta_df["delta_psi"].values.astype(float)
    return labels, {}


_VEX_CSV_CACHE = {"df": None}
def _vex_ref_psi_table():
    """one-shot cache: train + test CSV merged with HepG2_ref_psi keyed by (chrom,pos,ref,alt)"""
    if _VEX_CSV_CACHE["df"] is not None:
        return _VEX_CSV_CACHE["df"]
    _train = pd.read_csv(vex_data_dir / "train.csv")
    _test = pd.read_csv(vex_data_dir / "test.csv")
    for d in (_train, _test):
        d["chrom"] = d["seqnames"].astype(str)
        d["pos"] = d["hg19_variant_position"].astype(int)
        d["ref"] = d["reference"].astype(str)
        d["alt"] = d["variant"].astype(str)
    out = pd.concat([
        _train[["chrom", "pos", "ref", "alt", "HepG2_ref_psi"]],
        _test[["chrom", "pos", "ref", "alt", "HepG2_ref_psi"]],
    ], ignore_index=True).drop_duplicates(subset=["chrom", "pos", "ref", "alt"])
    _VEX_CSV_CACHE["df"] = out
    return out


_COMPASS_PSI_CACHE = {"df": None}
def _compass_psi_table(compass_dir):
    if _COMPASS_PSI_CACHE["df"] is not None:
        return _COMPASS_PSI_CACHE["df"]
    psi_csv = compass_dir / "compass_psi.csv.gz"
    if not psi_csv.exists():
        _COMPASS_PSI_CACHE["df"] = pd.DataFrame()
        return _COMPASS_PSI_CACHE["df"]
    _COMPASS_PSI_CACHE["df"] = pd.read_csv(psi_csv)
    return _COMPASS_PSI_CACHE["df"]


def _labels_from_compass_psi(cfg, meta_df):
    psi = _compass_psi_table(cfg.path)
    if psi.empty or "Reference" not in meta_df.columns:
        return np.full(len(meta_df), np.nan), {}
    cell = cfg.compass_primary_cell
    keep = ["Reference", f"{cell}_dpsi_pooled", f"{cell}_pooled_psi_clipped", f"{cell}_wt_pooled_psi_raw"]
    keep = [k for k in keep if k in psi.columns]
    sub = psi[keep]
    merged = meta_df[["Reference"]].merge(sub, on="Reference", how="left")
    labels = merged[f"{cell}_dpsi_pooled"].values.astype(float) if f"{cell}_dpsi_pooled" in merged else np.full(len(meta_df), np.nan)
    extras = {}
    if f"{cell}_pooled_psi_clipped" in merged:
        extras["measured_alt_inclusion"] = merged[f"{cell}_pooled_psi_clipped"].values
    if f"{cell}_wt_pooled_psi_raw" in merged:
        extras["measured_ref_inclusion"] = merged[f"{cell}_wt_pooled_psi_raw"].values
    return labels, extras


_OS_MASTER_CACHE = {"df": None}
def _opensplice_master_table(opensplice_dir):
    if _OS_MASTER_CACHE["df"] is not None:
        return _OS_MASTER_CACHE["df"]
    master_path = opensplice_dir / "open_splice_master.tsv.gz"
    if not master_path.exists():
        _OS_MASTER_CACHE["df"] = pd.DataFrame()
        return _OS_MASTER_CACHE["df"]
    df = pd.read_csv(master_path, sep="\t", usecols=["variant_id", "delta_psi", "psi", "wt_psi"], low_memory=False)
    df = df.drop_duplicates(subset=["variant_id"])
    _OS_MASTER_CACHE["df"] = df
    return df


def _labels_from_opensplice_master(cfg, meta_df):
    master = _opensplice_master_table(cfg.path)
    if master.empty or "variant_id" not in meta_df.columns:
        return np.full(len(meta_df), np.nan), {}
    drop_cols = [c for c in ("delta_psi", "psi", "wt_psi") if c in meta_df.columns]
    merged = meta_df.drop(columns=drop_cols).merge(master, on="variant_id", how="left")
    labels = merged["delta_psi"].values.astype(float)
    return labels, {
        "measured_alt_inclusion": merged["psi"].values,
        "measured_ref_inclusion": merged["wt_psi"].values,
    }


LABEL_SOURCES = {
    "h5_meta": _labels_from_h5_meta,
    "compass_psi": _labels_from_compass_psi,
    "opensplice_master": _labels_from_opensplice_master,
}


def _load_meta_from_h5(h5_path):
    """returns DataFrame of /meta with bytes decoded"""
    with h5py.File(h5_path, "r") as f:
        meta = {}
        for k in f["meta"].keys():
            arr = f["meta"][k][:]
            if arr.dtype.kind in ("S", "O"):
                meta[k] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
            else:
                meta[k] = arr
    return pd.DataFrame(meta)


# ---------------------------------------------------------------------------
# Data dataclass + load_data
# ---------------------------------------------------------------------------

@dataclass
class Data:
    dfs: dict = field(default_factory=dict)              # name -> df with merged predictions
    raw_dfs: dict = field(default_factory=dict)          # name -> dict[model -> raw scores df]
    dfs_filt: dict = field(default_factory=dict)         # |y| > 1SD subset
    dfs_neutral: dict = field(default_factory=dict)      # |y| <= 1SD subset
    sd_info: dict = field(default_factory=dict)          # name -> (mu, sd, lo, hi)
    vex_combined: pd.DataFrame = field(default_factory=pd.DataFrame)
    mfass_df_mut: pd.DataFrame = field(default_factory=pd.DataFrame)
    boot_results: dict = field(default_factory=dict)
    cboot_results: dict = field(default_factory=dict)


def _download_vexseq():
    """download vex-seq csv/tsv if missing"""
    vex_data_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "train": "https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/vexseq/HepG2_delta_PSI_CAGI_training.csv",
        "test": "https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/vexseq/HepG2_delta_PSI_CAGI_testing.csv",
        "truth": "https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/vexseq/Vexseq_HepG2_delta_PSI_CAGI_test_true.tsv",
    }
    for nm, url in urls.items():
        ext = "csv" if url.endswith(".csv") else "tsv"
        path = vex_data_dir / f"{nm}.{ext}"
        if not path.exists():
            print(f"downloading {nm}...")
            os.system(f'curl -sL "{url}" -o "{path}"')


def _build_vex_combined():
    """load + merge vex-seq train/test/truth → combined dataframe"""
    train = pd.read_csv(vex_data_dir / "train.csv")
    test = pd.read_csv(vex_data_dir / "test.csv")
    truth = pd.read_csv(vex_data_dir / "truth.tsv", sep="\t")
    print(f"train: {len(train):,} variants")
    print(f"test:  {len(test):,} variants")
    print(f"truth: {len(truth):,} labels")

    computed_width = train["end"] - train["start"] + 1
    assert (computed_width == train["width"]).all()
    print(f"delta-PSI range: {train['HepG2_delta_psi'].min():.1f} to {train['HepG2_delta_psi'].max():.1f}")
    print(f"delta-PSI mean: {train['HepG2_delta_psi'].mean():.2f}")

    for d in (train, test):
        d["chrom"] = d["seqnames"].astype(str)
        d["pos"] = d["hg19_variant_position"].astype(int)
        d["ref"] = d["reference"].astype(str)
        d["alt"] = d["variant"].astype(str)
    truth["chrom"] = truth["chromosome"].astype(str)
    truth["pos"] = truth["hg19_variant_position"].astype(int)
    truth["ref"] = truth["reference"].astype(str)
    truth["alt"] = truth["variant"].astype(str)

    key_cols = ["chrom", "pos", "ref", "alt"]
    truth_labels = truth[key_cols + ["HepG2_delta_psi"]].drop_duplicates(subset=key_cols)
    test_merged = test.merge(truth_labels, on=key_cols, how="left")
    missing = test_merged["HepG2_delta_psi"].isna().sum()
    print(f"test variants missing labels: {missing}")
    assert missing == 0

    train["split"] = "train"
    test_merged["split"] = "test"
    keep_cols = ["seqnames", "start", "end", "strand", "hg19_variant_position",
                 "reference", "variant", "HepG2_delta_psi", "split"]
    vex_combined = pd.concat([train[keep_cols], test_merged[keep_cols]], ignore_index=True)
    print(f"combined dataset: {len(vex_combined):,} variants")
    n_before = len(vex_combined)
    vex_combined = vex_combined.drop_duplicates()
    print(f"removed {n_before - len(vex_combined)} duplicate rows")
    print(f"final: {len(vex_combined):,} unique variants")

    ref_lens = vex_combined["reference"].str.len()
    alt_lens = vex_combined["variant"].str.len()
    is_snv = (ref_lens == 1) & (alt_lens == 1)
    print(f"  SNVs:       {is_snv.sum():,} ({100 * is_snv.mean():.1f}%)")
    print(f"  insertions: {(ref_lens < alt_lens).sum():,}")
    print(f"  deletions:  {(ref_lens > alt_lens).sum():,}")
    return vex_combined


def _build_vex_h5(vex_combined):
    """build vex_seq.h5 with one-hot ref/alt sequences (skip if exists)"""
    vex_out = vex_data_dir / "vex_seq.h5"
    if vex_out.exists():
        print(f"skipping vexseq sequence build — {vex_out} exists")
        return
    assert FASTA_PATH.exists(), f"need hg19 reference genome at {FASTA_PATH}"
    fasta = pysam.FastaFile(str(FASTA_PATH))
    print(f"loaded {FASTA_PATH}")

    vex_seqs = {"exon_start_ref": [], "exon_start_alt": [],
                "exon_end_ref": [], "exon_end_alt": []}
    vex_meta = {"chrom": [], "pos": [], "ref": [], "alt": [], "strand": [],
                "exon_start": [], "exon_end": [], "delta_psi": [], "split": []}
    skipped = []
    for r in tqdm(vex_combined.itertuples(index=False), total=len(vex_combined), desc="vexseq sequences"):
        chrom, strand = str(r.seqnames), str(r.strand)
        pos, ref, alt = int(r.hg19_variant_position), str(r.reference), str(r.variant)
        ex_start, ex_end = int(r.start), int(r.end)
        if strand not in {"+", "-"}:
            skipped.append((pos, f"invalid strand: {strand}")); continue
        ref_start, alt_start, err = build_ref_alt_pair(chrom, ex_start, pos, ref, alt, fasta)
        if err:
            skipped.append((pos, f"exon_start: {err}")); continue
        ref_end, alt_end, err = build_ref_alt_pair(chrom, ex_end, pos, ref, alt, fasta)
        if err:
            skipped.append((pos, f"exon_end: {err}")); continue
        if strand == "-":
            ref_start, alt_start = revcomp(ref_start), revcomp(alt_start)
            ref_end, alt_end = revcomp(ref_end), revcomp(alt_end)
        vex_seqs["exon_start_ref"].append(ref_start)
        vex_seqs["exon_start_alt"].append(alt_start)
        vex_seqs["exon_end_ref"].append(ref_end)
        vex_seqs["exon_end_alt"].append(alt_end)
        for k, v in zip(vex_meta.keys(),
                        [chrom, pos, ref, alt, strand, ex_start, ex_end,
                         float(r.HepG2_delta_psi), str(r.split)]):
            vex_meta[k].append(v)
    fasta.close()
    n_vex = len(vex_meta["chrom"])
    print(f"built sequences for {n_vex:,} variants ({len(skipped):,} skipped)")
    vex_encoded = {}
    for key, seq_list in vex_seqs.items():
        arr = np.zeros((n_vex, SEQ_LEN, 4), dtype=np.float32)
        for i, seq in enumerate(seq_list):
            arr[i] = one_hot(seq)
        vex_encoded[key] = arr
        print(f"  {key}: {arr.shape}")
    with h5py.File(vex_out, "w") as f:
        seq_grp = f.create_group("seqs")
        for key, arr in vex_encoded.items():
            seq_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("chrom", data=np.array(vex_meta["chrom"], dtype="S24"))
        meta_grp.create_dataset("pos", data=np.array(vex_meta["pos"], dtype=np.int64))
        meta_grp.create_dataset("ref", data=np.array(vex_meta["ref"], dtype="S256"))
        meta_grp.create_dataset("alt", data=np.array(vex_meta["alt"], dtype="S256"))
        meta_grp.create_dataset("strand", data=np.array(vex_meta["strand"], dtype="S1"))
        meta_grp.create_dataset("exon_start", data=np.array(vex_meta["exon_start"], dtype=np.int64))
        meta_grp.create_dataset("exon_end", data=np.array(vex_meta["exon_end"], dtype=np.int64))
        meta_grp.create_dataset("delta_psi", data=np.array(vex_meta["delta_psi"], dtype=np.float32))
        meta_grp.create_dataset("split", data=np.array(vex_meta["split"], dtype="S8"))
    print(f"saved {vex_out}")


def _build_mfass_df_mut():
    """download mfass and build the mutant-only dataframe"""
    mfass_data_dir.mkdir(parents=True, exist_ok=True)
    snv_url = "https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt"
    snv_path = mfass_data_dir / "snv_data_clean.txt"
    if not snv_path.exists():
        print("downloading mfass data...")
        os.system(f'curl -sL "{snv_url}" -o "{snv_path}"')
    df = pd.read_csv(snv_path, sep="\t")
    print(f"loaded {len(df):,} rows")
    print(df["category"].value_counts())
    natural = df[df["category"] == "natural"]
    assert natural["ref_allele"].isna().all()

    df_mut = df[df["category"] == "mutant"].copy()
    df_mut = df_mut.rename(columns={"chr": "chrom", "ref_allele": "ref",
                                    "alt_allele": "alt", "snp_position": "pos"})
    df_mut["pos"] = df_mut["pos"].astype(int)
    print(f"mutant rows: {len(df_mut):,}")

    df_mut["exon_start"] = df_mut["start"] + df_mut["intron1_len"]
    df_mut["exon_end"] = df_mut["exon_start"] + df_mut["exon_len"] - 1

    fasta = pysam.FastaFile(str(FASTA_PATH))
    print(f"reference genome: {FASTA_PATH.name}")
    orientations = []
    for r in tqdm(df_mut.itertuples(index=False), total=len(df_mut), desc="checking"):
        genome_seq = fasta.fetch(str(r.chrom), int(r.start) - 1, int(r.end)).upper()
        natural_seq = str(r.natural_seq).upper()
        if genome_seq == natural_seq:
            orientations.append("forward")
        elif genome_seq == revcomp(natural_seq):
            orientations.append("revcomp")
        else:
            orientations.append("mismatch")
    df_mut["orientation"] = orientations
    n_fwd = (df_mut["orientation"] == "forward").sum()
    n_rev = (df_mut["orientation"] == "revcomp").sum()
    n_mis = (df_mut["orientation"] == "mismatch").sum()
    print(f"  forward: {n_fwd:,}, revcomp: {n_rev:,}, mismatch: {n_mis:,}")

    n_mismatch = (df_mut["orientation"] == "mismatch").sum()
    if n_mismatch > 0:
        df_mut = df_mut[df_mut["orientation"] != "mismatch"].reset_index(drop=True)
    df_mut["strand"] = df_mut["orientation"].map({"forward": "+", "revcomp": "-"})

    unlabeled = df_mut["v2_dpsi"].isna()
    if unlabeled.sum() > 0:
        df_mut = df_mut[~unlabeled].reset_index(drop=True)
    n_sdv = (df_mut["v2_dpsi"].abs() > 0.5).sum()
    print(f"labeled: {len(df_mut):,} ({n_sdv:,} splice-disrupting)")

    mfass_out = mfass_data_dir / "mfass.h5"
    if not mfass_out.exists():
        _build_mfass_h5(df_mut, fasta, mfass_out)
    fasta.close()
    return df_mut


def _build_mfass_h5(df_mut, fasta, mfass_out):
    """write mfass.h5 with one-hot sequences"""
    mfass_seqs = {"exon_start_ref": [], "exon_start_alt": [],
                  "exon_end_ref": [], "exon_end_alt": []}
    mfass_meta = {"chrom": [], "pos": [], "ref": [], "alt": [], "strand": [],
                  "exon_start": [], "exon_end": [], "delta_psi": [], "exon_id": []}
    skipped = []
    for r in tqdm(df_mut.itertuples(index=False), total=len(df_mut), desc="mfass sequences"):
        chrom, strand = str(r.chrom), str(r.strand)
        pos, ref, alt = int(r.pos), str(r.ref), str(r.alt)
        ex_start, ex_end = int(r.exon_start), int(r.exon_end)
        if strand not in {"+", "-"}:
            skipped.append((pos, f"invalid strand: {strand}")); continue
        ref_start, alt_start, err = build_ref_alt_pair(chrom, ex_start, pos, ref, alt, fasta)
        if err:
            skipped.append((pos, f"exon_start: {err}")); continue
        ref_end, alt_end, err = build_ref_alt_pair(chrom, ex_end, pos, ref, alt, fasta)
        if err:
            skipped.append((pos, f"exon_end: {err}")); continue
        if strand == "-":
            ref_start, alt_start = revcomp(ref_start), revcomp(alt_start)
            ref_end, alt_end = revcomp(ref_end), revcomp(alt_end)
        mfass_seqs["exon_start_ref"].append(ref_start)
        mfass_seqs["exon_start_alt"].append(alt_start)
        mfass_seqs["exon_end_ref"].append(ref_end)
        mfass_seqs["exon_end_alt"].append(alt_end)
        dpsi = r.v2_dpsi if hasattr(r, "v2_dpsi") and pd.notna(r.v2_dpsi) else np.nan
        for k, v in zip(mfass_meta.keys(),
                        [chrom, pos, ref, alt, strand, ex_start, ex_end,
                         float(dpsi), str(r.ensembl_id) if hasattr(r, "ensembl_id") else ""]):
            mfass_meta[k].append(v)
    n_mfass = len(mfass_meta["chrom"])
    print(f"built sequences for {n_mfass:,} variants ({len(skipped):,} skipped)")
    mfass_encoded = {}
    for key, seq_list in mfass_seqs.items():
        arr = np.zeros((n_mfass, SEQ_LEN, 4), dtype=np.float32)
        for i, seq in enumerate(seq_list):
            arr[i] = one_hot(seq)
        mfass_encoded[key] = arr
    with h5py.File(mfass_out, "w") as f:
        seq_grp = f.create_group("seqs")
        for key, arr in mfass_encoded.items():
            seq_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("chrom", data=np.array(mfass_meta["chrom"], dtype="S24"))
        meta_grp.create_dataset("pos", data=np.array(mfass_meta["pos"], dtype=np.int64))
        meta_grp.create_dataset("ref", data=np.array(mfass_meta["ref"], dtype="S256"))
        meta_grp.create_dataset("alt", data=np.array(mfass_meta["alt"], dtype="S256"))
        meta_grp.create_dataset("strand", data=np.array(mfass_meta["strand"], dtype="S1"))
        meta_grp.create_dataset("exon_start", data=np.array(mfass_meta["exon_start"], dtype=np.int64))
        meta_grp.create_dataset("exon_end", data=np.array(mfass_meta["exon_end"], dtype=np.int64))
        meta_grp.create_dataset("delta_psi", data=np.array(mfass_meta["delta_psi"], dtype=np.float32))
        meta_grp.create_dataset("exon_id", data=np.array(mfass_meta["exon_id"], dtype="S64"))
    print(f"saved {mfass_out}")


_OS_SNV_RE = re.compile(r"_([ACGT])(\d+)([ACGT])$")
_OS_DEL_RE = re.compile(r"_del(\d+)to(\d+)$")


def _normalize_canonical_cols(df):
    """fill canonical fig schema (chrom, pos, ref, alt, exon_start, exon_end) in-place
    vex/mfass already have them; compass parses variant_hg38 + renames exon_*_hg38;
    opensplice maps acc/donor_in_variant -> exon_start/exon_end + parses variant_id"""
    n = len(df)
    if "exon_start" not in df.columns:
        if "exon_start_hg38" in df.columns:
            df["exon_start"] = df["exon_start_hg38"].values
            df["exon_end"] = df["exon_end_hg38"].values
        elif "acc_in_variant" in df.columns and "donor_in_variant" in df.columns:
            df["exon_start"] = df["acc_in_variant"].values
            df["exon_end"] = df["donor_in_variant"].values
    if "pos" not in df.columns:
        pos = np.full(n, -1, dtype=np.int64)
        ref = np.array([""] * n, dtype=object)
        alt = np.array([""] * n, dtype=object)
        if "variant_hg38" in df.columns:
            # compass: "chr10:101775270:A>T" (empty for refs/genome refs)
            for i, s in enumerate(df["variant_hg38"].astype(str).values):
                if not s or s == "nan":
                    continue
                parts = s.split(":")
                if len(parts) == 3 and ">" in parts[2]:
                    try:
                        pos[i] = int(parts[1])
                    except ValueError:
                        continue
                    ra = parts[2].split(">")
                    if len(ra) == 2:
                        ref[i] = ra[0]; alt[i] = ra[1]
        elif "variant_id" in df.columns:
            # opensplice: "GENE_eN_REFposALT" SNV or "GENE_eN_delSTARTtoEND"
            del_start = df["del_start"].values if "del_start" in df.columns else None
            for i, s in enumerate(df["variant_id"].astype(str).values):
                m = _OS_SNV_RE.search(s)
                if m:
                    ref[i] = m.group(1); pos[i] = int(m.group(2)); alt[i] = m.group(3)
                    continue
                m = _OS_DEL_RE.search(s)
                if m:
                    pos[i] = int(m.group(1)); ref[i] = "del"; alt[i] = ""
                    continue
                if del_start is not None and del_start[i] >= 0:
                    pos[i] = int(del_start[i])
        df["pos"] = pos
        df["ref"] = ref
        df["alt"] = alt


def _load_scores_into_dfs(configs=None):
    """load h5 scores for each cfg into per-dataset dataframes (config-driven)"""
    if configs is None:
        configs = DATASET_CONFIGS
    dfs = {}
    for cfg in configs:
        # source meta from each score h5 — choose the largest row count as canonical
        # (some h5s were scored pre-filter at n_raw, others post-filter at n_filt)
        per_model_meta = {}
        delta_dfs = {}
        for m in models:
            path = cfg.score_h5_path(m)
            if not path.exists():
                continue
            try:
                meta, deltas = load_scores_as_deltas(path)
            except Exception as e:
                print(f"  {cfg.name}/{m}: skip ({type(e).__name__}: {e})")
                continue
            deltas.columns = [f"{m}_{c}" for c in deltas.columns]
            delta_dfs[m] = deltas
            per_model_meta[m] = meta

        if not delta_dfs:
            print(f"{cfg.name}: no model scores loaded, skipping")
            continue

        # canonical meta = the one with the largest row count (pre-filter if any model is pre-filter)
        canonical_m = max(per_model_meta, key=lambda k: len(per_model_meta[k]))
        meta_df = per_model_meta[canonical_m]
        n_raw = len(meta_df)

        # source labels from canonical meta
        labels, extras = LABEL_SOURCES[cfg.label_source](cfg, meta_df)
        meta_df = meta_df.copy()
        meta_df["delta_psi"] = labels

        # apply per-cfg row mask (mfass drops NaN-label rows; some delta_dfs are
        # already post-filter and pass through unchanged)
        if cfg.filter_unlabeled:
            mask = np.isfinite(meta_df["delta_psi"].values)
            meta_df = meta_df[mask].reset_index(drop=True)
            n_filt = len(meta_df)
            for k, v in list(extras.items()):
                extras[k] = np.asarray(v)[mask]
            filtered = {}
            for m, mdf in delta_dfs.items():
                if len(mdf) == n_raw:
                    filtered[m] = mdf[mask].reset_index(drop=True)
                elif len(mdf) == n_filt:
                    filtered[m] = mdf.reset_index(drop=True)
                else:
                    print(f"  WARNING: {cfg.name}/{m} has {len(mdf)} rows, expected {n_raw}/{n_filt}, skipping")
                    continue
            delta_dfs = filtered
            print(f"{cfg.name}: filtered {n_raw - n_filt} unlabeled rows")

        df = meta_df.copy()
        if cfg.force_plus_strand:
            df["strand"] = "+"
        elif "strand" not in df.columns:
            df["strand"] = "+"
        _normalize_canonical_cols(df)
        df["y"] = df["delta_psi"].values / cfg.psi_scale
        y_clean = df["y"].values[np.isfinite(df["y"].values)]
        sd_val = y_clean.std() if len(y_clean) > 1 else np.nan
        df["label"] = (np.abs(df["y"]) > sd_val).astype(int) if np.isfinite(sd_val) else 0
        df["label_fixed"] = (np.abs(df["y"]) > cfg.thr).astype(int)
        # backfill global datasets dict so downstream code that reads thr_sd works
        if cfg.name not in datasets:
            datasets[cfg.name] = {"path": str(cfg.path), "prefix": cfg.prefix,
                                  "thr": cfg.thr, "psi_scale": cfg.psi_scale}
        datasets[cfg.name]["thr_sd"] = float(sd_val) if np.isfinite(sd_val) else np.nan
        for k, v in extras.items():
            df[k] = v
        for m, delta_df in delta_dfs.items():
            df = pd.concat([df, delta_df], axis=1)
        dfs[cfg.name] = df
        n_pos = int(df["label"].sum())
        n_pos_fixed = int(df["label_fixed"].sum())
        n_lab = int(np.isfinite(y_clean).sum())
        print(f"{cfg.name}: loaded {len(delta_dfs)} models, {len(df):,} variants, {n_lab:,} labeled")
        print(f"  labels (1-SD): {n_pos:,} positives, threshold={sd_val:.3f}")
        print(f"  labels (fixed): {n_pos_fixed:,} positives, threshold={cfg.thr:.2f}")
    return dfs


def _compute_combined_deltas(dfs):
    """add strand-aware + max-aggregated columns to each df"""
    for name, df in dfs.items():
        is_plus = (df["strand"].values == "+")
        def strand_select(start_col, end_col):
            return np.where(is_plus, df[start_col], df[end_col])
        def avg_boundaries(prefix):
            return (df[f"{prefix}_exon_start_delta"] + df[f"{prefix}_exon_end_delta"]) / 2

        # spliceai
        if "spliceai_acceptor_exon_start_delta" in df.columns:
            acc = strand_select("spliceai_acceptor_exon_start_delta", "spliceai_acceptor_exon_end_delta")
            don = strand_select("spliceai_donor_exon_end_delta", "spliceai_donor_exon_start_delta")
            df["spliceai_cls_delta"] = (acc + don) / 2

        # splicetransformer cls + per-tissue usage
        if "splicetransformer_acceptor_exon_start_delta" in df.columns:
            acc = strand_select("splicetransformer_acceptor_exon_start_delta", "splicetransformer_acceptor_exon_end_delta")
            don = strand_select("splicetransformer_donor_exon_end_delta", "splicetransformer_donor_exon_start_delta")
            df["splicetransformer_cls_delta"] = (acc + don) / 2
            usage_cols = []
            for c in df.columns:
                if c.startswith("splicetransformer_usage_") and c.endswith("_exon_start_delta"):
                    tissue = c.replace("splicetransformer_usage_", "").replace("_exon_start_delta", "")
                    col = f"splicetransformer_usage_{tissue}_delta"
                    df[col] = avg_boundaries(f"splicetransformer_usage_{tissue}")
                    usage_cols.append(col)
            if usage_cols:
                vals = df[usage_cols].values
                df["splicetransformer_max_usage"] = vals[np.arange(len(df)), np.abs(vals).argmax(axis=1)]

        # alphagenome cls + per-tissue usage (mirrors splicetransformer structure)
        if "alphagenome_acceptor_exon_start_delta" in df.columns:
            acc = strand_select("alphagenome_acceptor_exon_start_delta", "alphagenome_acceptor_exon_end_delta")
            don = strand_select("alphagenome_donor_exon_end_delta", "alphagenome_donor_exon_start_delta")
            df["alphagenome_cls_delta"] = (acc + don) / 2
            usage_cols = []
            for c in df.columns:
                if c.startswith("alphagenome_usage_") and c.endswith("_exon_start_delta"):
                    tissue = c.replace("alphagenome_usage_", "").replace("_exon_start_delta", "")
                    col = f"alphagenome_usage_{tissue}_delta"
                    df[col] = avg_boundaries(f"alphagenome_usage_{tissue}")
                    usage_cols.append(col)
            if usage_cols:
                vals = df[usage_cols].values
                df["alphagenome_max_usage"] = vals[np.arange(len(df)), np.abs(vals).argmax(axis=1)]

        # pangolin variants
        for _pang in ["pangolin", "pangolin_v2"]:
            _pang_cols = [c for c in df.columns if c.startswith(f"{_pang}_") and c.endswith("_p_splice_exon_start_delta")]
            if not _pang_cols:
                continue
            tissues = [c.replace(f"{_pang}_", "").replace("_p_splice_exon_start_delta", "") for c in _pang_cols]
            p_cols, u_cols = [], []
            for t in tissues:
                df[f"{_pang}_{t}_p_splice_delta"] = avg_boundaries(f"{_pang}_{t}_p_splice")
                df[f"{_pang}_{t}_usage_delta"] = avg_boundaries(f"{_pang}_{t}_usage")
                p_cols.append(f"{_pang}_{t}_p_splice_delta")
                u_cols.append(f"{_pang}_{t}_usage_delta")
            if p_cols:
                vals = df[p_cols].values
                df[f"{_pang}_max_p_splice"] = vals[np.arange(len(df)), np.abs(vals).argmax(axis=1)]
            if u_cols:
                vals = df[u_cols].values
                df[f"{_pang}_max_usage"] = vals[np.arange(len(df)), np.abs(vals).argmax(axis=1)]

        for v in ["ref", "var"]:
            p = f"splaire_{v}"
            if f"{p}_cls_acceptor_exon_start_delta" not in df.columns:
                continue
            acc = strand_select(f"{p}_cls_acceptor_exon_start_delta", f"{p}_cls_acceptor_exon_end_delta")
            don = strand_select(f"{p}_cls_donor_exon_end_delta", f"{p}_cls_donor_exon_start_delta")
            df[f"{p}_cls_delta"] = (acc + don) / 2
            df[f"{p}_reg_ssu_delta"] = avg_boundaries(f"{p}_reg_ssu")

        dfs[name] = df


def _load_raw_scores_into_dfs(dfs, configs=None):
    """load raw (ref/alt) scores and merge ref/alt combined columns onto each df"""
    if configs is None:
        configs = DATASET_CONFIGS
    cfgs_by_name = {c.name: c for c in configs}
    raw_dfs = {}
    for name in dfs:
        cfg = cfgs_by_name.get(name)
        if cfg is None:
            print(f"{name}: no config for raw-score load, skipping")
            continue
        raw_scores = {}
        for m in models:
            path = cfg.score_h5_path(m)
            if not path.exists():
                continue
            try:
                scores = load_raw_scores(path)
            except Exception as e:
                print(f"  {cfg.name}/{m}: raw skip ({type(e).__name__}: {e})")
                continue
            scores.columns = [f"{m}_{c}" for c in scores.columns]
            raw_scores[m] = scores
        n_main = len(dfs[name])
        # cfgs that filter unlabeled rows produce dfs with fewer rows than score h5s;
        # derive a row mask from the score h5 meta to realign raw scores
        _row_mask = None
        if cfg.filter_unlabeled:
            for m in raw_scores:
                first_h5 = raw_scores[m]
                if len(first_h5) > n_main:
                    _meta_tmp, _ = load_scores_as_deltas(cfg.score_h5_path(m))
                    _row_mask = _meta_tmp["delta_psi"].notna().values
                    break
        for m, sdf in raw_scores.items():
            if len(sdf) != n_main and _row_mask is not None and len(sdf) == len(_row_mask):
                raw_scores[m] = sdf[_row_mask].reset_index(drop=True)
        raw_dfs[name] = raw_scores
        print(f"{name}: loaded raw scores for {len(raw_scores)} models")

    # combine into strand-aware ref/alt columns
    for name in dfs:
        df = dfs[name]
        is_plus = (df["strand"].values == "+")
        raw = raw_dfs[name]
        def _strand_sel(col_start, col_end):
            return np.where(is_plus, col_start, col_end)

        if "spliceai" in raw:
            rdf = raw["spliceai"]
            for tag in ["ref", "alt"]:
                acc = _strand_sel(rdf[f"spliceai_acceptor_exon_start_{tag}"].values,
                                  rdf[f"spliceai_acceptor_exon_end_{tag}"].values)
                don = _strand_sel(rdf[f"spliceai_donor_exon_end_{tag}"].values,
                                  rdf[f"spliceai_donor_exon_start_{tag}"].values)
                df[f"spliceai_cls_{tag}"] = (acc + don) / 2

        if "splicetransformer" in raw:
            rdf = raw["splicetransformer"]
            for tag in ["ref", "alt"]:
                acc = _strand_sel(rdf[f"splicetransformer_acceptor_exon_start_{tag}"].values,
                                  rdf[f"splicetransformer_acceptor_exon_end_{tag}"].values)
                don = _strand_sel(rdf[f"splicetransformer_donor_exon_end_{tag}"].values,
                                  rdf[f"splicetransformer_donor_exon_start_{tag}"].values)
                df[f"splicetransformer_cls_{tag}"] = (acc + don) / 2

        if "alphagenome" in raw:
            rdf = raw["alphagenome"]
            for tag in ["ref", "alt"]:
                if f"alphagenome_acceptor_exon_start_{tag}" not in rdf.columns:
                    continue
                acc = _strand_sel(rdf[f"alphagenome_acceptor_exon_start_{tag}"].values,
                                  rdf[f"alphagenome_acceptor_exon_end_{tag}"].values)
                don = _strand_sel(rdf[f"alphagenome_donor_exon_end_{tag}"].values,
                                  rdf[f"alphagenome_donor_exon_start_{tag}"].values)
                df[f"alphagenome_cls_{tag}"] = (acc + don) / 2

        for _pang in ["pangolin", "pangolin_v2"]:
            if _pang not in raw:
                continue
            rdf = raw[_pang]
            start_cols = [c for c in rdf.columns if c.endswith("_p_splice_exon_start_ref")]
            if not start_cols:
                continue
            tissues = [c.replace(f"{_pang}_", "").replace("_p_splice_exon_start_ref", "") for c in start_cols]
            for tag in ["ref", "alt"]:
                vals = np.column_stack([
                    (rdf[f"{_pang}_{t}_p_splice_exon_start_{tag}"].values +
                     rdf[f"{_pang}_{t}_p_splice_exon_end_{tag}"].values) / 2
                    for t in tissues
                ])
                df[f"{_pang}_max_p_splice_{tag}"] = vals[np.arange(len(df)), np.abs(vals).argmax(axis=1)]

        for v in ["ref", "var"]:
            p = f"splaire_{v}"
            if p not in raw:
                continue
            rdf = raw[p]
            for tag in ["ref", "alt"]:
                if f"{p}_cls_acceptor_exon_start_{tag}" in rdf.columns:
                    acc = _strand_sel(rdf[f"{p}_cls_acceptor_exon_start_{tag}"].values,
                                      rdf[f"{p}_cls_acceptor_exon_end_{tag}"].values)
                    don = _strand_sel(rdf[f"{p}_cls_donor_exon_end_{tag}"].values,
                                      rdf[f"{p}_cls_donor_exon_start_{tag}"].values)
                    df[f"{p}_cls_{tag}"] = (acc + don) / 2
        dfs[name] = df
    return raw_dfs


def _merge_measured_inclusion(dfs):
    """attach measured ref/alt inclusion onto each df (vex/mfass-specific external merges;
    compass/opensplice already populated by the label-source helpers)"""
    if "vexseq" in dfs:
        vex_df = dfs["vexseq"]
        _vex_ref_psi = _vex_ref_psi_table()
        _before = len(vex_df)
        vex_df = vex_df.merge(_vex_ref_psi, on=["chrom", "pos", "ref", "alt"], how="left")
        assert len(vex_df) == _before
        vex_df["measured_ref_psi"] = vex_df["HepG2_ref_psi"].values / 100.0
        vex_df.drop(columns=["HepG2_ref_psi"], inplace=True)
        dfs["vexseq"] = vex_df
        n_ref = np.isfinite(vex_df["measured_ref_psi"].values).sum()
        print(f"vexseq: merged ref PSI for {n_ref:,} / {len(vex_df):,} variants")

    if "mfass" in dfs:
        mfass_df = dfs["mfass"]
        _mfass_raw = pd.read_csv(mfass_data_dir / "snv_data_clean.txt", sep="\t")
        _mfass_mut = _mfass_raw[_mfass_raw["category"] == "mutant"].copy()
        _mfass_mut = _mfass_mut.rename(columns={"chr": "chrom", "ref_allele": "ref",
                                                "alt_allele": "alt", "snp_position": "pos"})
        _mfass_mut["pos"] = _mfass_mut["pos"].astype(int)
        extra_cols = ["nat_v2_index", "v2_index", "v1_dpsi", "v2_dpsi_R1", "v2_dpsi_R2"]
        _mfass_merge = _mfass_mut[["chrom", "pos", "ref", "alt"] + extra_cols].drop_duplicates(
            subset=["chrom", "pos", "ref", "alt"])
        _before = len(mfass_df)
        mfass_df = mfass_df.merge(_mfass_merge, on=["chrom", "pos", "ref", "alt"], how="left")
        assert len(mfass_df) == _before
        mfass_df["measured_ref_inclusion"] = mfass_df["nat_v2_index"].values
        mfass_df["measured_alt_inclusion"] = mfass_df["v2_index"].values
        dfs["mfass"] = mfass_df
        for col in extra_cols:
            n_valid = mfass_df[col].notna().sum()
            print(f"mfass: {col} — {n_valid:,} valid")


def _compute_filtered_subsets(dfs):
    """split each dataset into |y|>1SD (filt) and |y|<=1SD (neutral)"""
    sd_info = {}
    dfs_filt = {}
    dfs_neutral = {}
    for name, df in dfs.items():
        y = df["y"].values
        y_clean = y[np.isfinite(y)]
        mu, sd = y_clean.mean(), y_clean.std()
        lo, hi = mu - sd, mu + sd
        sd_info[name] = (mu, sd, lo, hi)
        n_outside = ((y_clean < lo) | (y_clean > hi)).sum()
        print(f"{dataset_names.get(name, name)}: mean={mu:.4f}, SD={sd:.4f}, kept {n_outside:,}/{len(y_clean):,}")
        mask = (df["y"] < lo) | (df["y"] > hi)
        dfs_filt[name] = df[mask].reset_index(drop=True)
        dfs_neutral[name] = df[~mask].reset_index(drop=True)
    return sd_info, dfs_filt, dfs_neutral


def load_data(only=None) -> Data:
    """load every dataset, run merges, build h5s if missing, returns Data.

    only: optional list of dataset names to include (defaults to all in DATASET_CONFIGS).
    """
    configs = DATASET_CONFIGS
    if only is not None:
        keep = set(only)
        configs = [c for c in configs if c.name in keep]

    # dataset-specific prerequisites (only run if their dataset is included)
    vex_combined = pd.DataFrame()
    mfass_df_mut = pd.DataFrame()
    cfg_names = {c.name for c in configs}
    if "vexseq" in cfg_names:
        _download_vexseq()
        vex_combined = _build_vex_combined()
        _build_vex_h5(vex_combined)
    if "mfass" in cfg_names:
        mfass_df_mut = _build_mfass_df_mut()

    dfs = _load_scores_into_dfs(configs)
    _compute_combined_deltas(dfs)
    raw_dfs = _load_raw_scores_into_dfs(dfs, configs)
    _merge_measured_inclusion(dfs)
    sd_info, dfs_filt, dfs_neutral = _compute_filtered_subsets(dfs)

    # save filtered data for figures.ipynb (vex/mfass-only — used by main paper figs)
    if {"vexseq", "mfass"} <= set(dfs):
        _reporter_pkl = fig3_main / "_reporter_data.pkl"
        with open(_reporter_pkl, "wb") as f:
            pickle.dump({"dfs_filt": {k: dfs_filt[k] for k in ("vexseq", "mfass")},
                         "dfs_neutral": {k: dfs_neutral[k] for k in ("vexseq", "mfass")}}, f)
        print(f"saved reporter data to {_reporter_pkl}")

    return Data(
        dfs=dfs, raw_dfs=raw_dfs,
        dfs_filt=dfs_filt, dfs_neutral=dfs_neutral,
        sd_info=sd_info,
        vex_combined=vex_combined, mfass_df_mut=mfass_df_mut,
    )


# ---------------------------------------------------------------------------
# bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_compare(y, preds_dict, label, n_boot=10_000, seed=42):
    """paired bootstrap significance for all model pairs, returns results dict"""
    models_b = list(preds_dict.keys())
    n = len(y)
    for m, pred in preds_dict.items():
        n_nonfinite = (~np.isfinite(pred)).sum()
        if n_nonfinite > 0:
            raise ValueError(f"model {m} has {n_nonfinite} non-finite predictions")
    # auprc-only mode when y has NaN (e.g. neutrals concatenated for label balance)
    y_finite = np.isfinite(y).all()
    if not y_finite:
        print(f"  y has {(~np.isfinite(y)).sum()} non-finite values — auprc-only (skip Pearson/Spearman/R2)")
    if n < 10:
        return {}
    rng = np.random.default_rng(seed)
    boot_metrics = {m: np.empty((n_boot, 4)) for m in models_b}
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yb, lb = y[idx], label[idx]
        for m in models_b:
            pb = preds_dict[m][idx]
            if y_finite:
                r = stats.pearsonr(yb, pb)[0]
                rho = stats.spearmanr(yb, pb)[0]
                r2 = r2_score(yb, pb)
            else:
                r = rho = r2 = np.nan
            auprc = average_precision_score(lb, np.abs(pb)) if lb.sum() > 0 else np.nan
            boot_metrics[m][b] = [r, rho, r2, auprc]
    metric_names = ["Pearson", "Spearman", "R2", "AUPRC"]
    results = {}
    for a, b in combinations(models_b, 2):
        results[(a, b)] = {}
        for mi, mname in enumerate(metric_names):
            diff = boot_metrics[a][:, mi] - boot_metrics[b][:, mi]
            p_left = np.mean(diff <= 0)
            p_val = 2 * min(p_left, 1 - p_left)
            p_val = min(p_val, 1.0)
            results[(a, b)][mname] = {
                "diff_mean": np.mean(diff),
                "ci_lo": np.percentile(diff, 2.5),
                "ci_hi": np.percentile(diff, 97.5),
                "p": p_val,
            }
    return results


def cluster_bootstrap_compare(y, preds_dict, label, clusters, n_boot=10_000, seed=42):
    """cluster bootstrap: resample exons (clusters) instead of variants"""
    models_b = list(preds_dict.keys())
    for m, pred in preds_dict.items():
        n_nonfinite = (~np.isfinite(pred)).sum()
        if n_nonfinite > 0:
            raise ValueError(f"model {m} has {n_nonfinite} non-finite predictions")
    # auprc-only mode when y has NaN (e.g. neutrals concatenated for label balance)
    y_finite = np.isfinite(y).all()
    if not y_finite:
        print(f"  y has {(~np.isfinite(y)).sum()} non-finite values — auprc-only (skip Pearson/Spearman/R2)")
    n = len(y)
    if n < 10:
        return {}
    unique_clusters = np.unique(clusters)
    cluster_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
    n_clusters = len(unique_clusters)
    print(f"  cluster bootstrap: {n} variants in {n_clusters} clusters")
    rng = np.random.default_rng(seed)
    boot_metrics = {m: np.empty((n_boot, 4)) for m in models_b}
    for b in range(n_boot):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_idx[c] for c in sampled])
        yb, lb = y[idx], label[idx]
        for m in models_b:
            pb = preds_dict[m][idx]
            if y_finite:
                r = stats.pearsonr(yb, pb)[0]
                rho = stats.spearmanr(yb, pb)[0]
                r2 = r2_score(yb, pb)
            else:
                r = rho = r2 = np.nan
            auprc = average_precision_score(lb, np.abs(pb)) if lb.sum() > 0 else np.nan
            boot_metrics[m][b] = [r, rho, r2, auprc]
    metric_names = ["Pearson", "Spearman", "R2", "AUPRC"]
    results = {}
    for a, b_ in combinations(models_b, 2):
        results[(a, b_)] = {}
        for mi, mname in enumerate(metric_names):
            diff = boot_metrics[a][:, mi] - boot_metrics[b_][:, mi]
            p_left = np.mean(diff <= 0)
            p_val = 2 * min(p_left, 1 - p_left)
            p_val = min(p_val, 1.0)
            results[(a, b_)][mname] = {
                "diff_mean": np.mean(diff),
                "ci_lo": np.percentile(diff, 2.5),
                "ci_hi": np.percentile(diff, 97.5),
                "p": p_val,
            }
    return results


def significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def holm_bonferroni(results):
    """holm-bonferroni per metric (k=10 for all-pairs)"""
    by_metric = {}
    for pair, metrics in results.items():
        for mname, d in metrics.items():
            by_metric.setdefault(mname, []).append((pair, d["p"]))
    for mname, pvals in by_metric.items():
        pvals.sort(key=lambda x: x[1])
        k = len(pvals)
        prev = 0.0
        for i, (pair, p) in enumerate(pvals):
            p_adj = min(max(prev, p * (k - i)), 1.0)
            prev = p_adj
            results[pair][mname]["p_adj_holm_k10"] = p_adj
            results[pair][mname]["sig_holm_k10"] = significance_stars(p_adj)


def get_boot_sig(results, a, b, metric):
    """look up pairwise significance, handles reversed pair order"""
    if (a, b) in results:
        return results[(a, b)].get(metric)
    elif (b, a) in results:
        d = results[(b, a)].get(metric)
        if d is None:
            return None
        return {"diff_mean": -d["diff_mean"], "ci_lo": -d["ci_hi"], "ci_hi": -d["ci_lo"],
                "p": d["p"],
                "p_adj_holm_k10": d.get("p_adj_holm_k10", d["p"]),
                "sig_holm_k10": d.get("sig_holm_k10", "")}
    return None


def print_bootstrap_table(results, model_names_map, title):
    """formatted pairwise comparison table"""
    if not results:
        print(f"{title}: insufficient data")
        return
    metric_names = ["Pearson", "Spearman", "R2", "AUPRC"]
    w = 120
    print(f"\n{'=' * w}")
    print(title)
    print(f"{'=' * w}")
    header = f"{'model A':<20} {'model B':<20}"
    for mn in metric_names:
        header += f" {mn:>8} {'p':>7} {'':>3}"
    print(header)
    print("-" * w)
    for (a, b), metrics in results.items():
        na = model_names_map.get(a, a)
        nb = model_names_map.get(b, b)
        row = f"{na:<20} {nb:<20}"
        for mn in metric_names:
            d = metrics[mn]
            diff_str = f"{d['diff_mean']:+.3f}"
            p_str = f"{d['p_adj_holm_k10']:.4f}" if d['p_adj_holm_k10'] >= 0.0001 else "<.0001"
            sig_str = d['sig_holm_k10']
            row += f" {diff_str:>8} {p_str:>7} {sig_str:>3}"
        print(row)
    print()


def draw_all_sig_brackets(ax, mi, bar_w, n_models, models_avail, model_vals, metric,
                           boot_key, boot_dict, y_offset=0.05, dh=0.04):
    """draw significance brackets for all significant model pairs within one metric"""
    if boot_key not in boot_dict:
        return 0
    pairs = []
    for i in range(len(models_avail)):
        for j in range(i + 1, len(models_avail)):
            a, b = models_avail[i], models_avail[j]
            sig_info = get_boot_sig(boot_dict[boot_key], a, b, metric)
            if sig_info and sig_info.get("sig_holm_k10", ""):
                pairs.append((i, j, sig_info["sig_holm_k10"]))
    if not pairs:
        return 0
    def bar_x(bi):
        return mi + (bi - n_models / 2 + 0.5) * bar_w
    y_base = max(model_vals[m][metric] for m in models_avail) + y_offset
    pairs.sort(key=lambda p: p[1] - p[0])
    for k, (i, j, sig) in enumerate(pairs):
        y = y_base + k * dh
        x1, x2 = bar_x(i), bar_x(j)
        ax.plot([x1, x1, x2, x2], [y - 0.008, y, y, y - 0.008], lw=0.7, color="black", clip_on=False)
        ax.text((x1 + x2) / 2, y + 0.003, sig, ha="center", va="bottom", fontsize=5.5, clip_on=False)
    return len(pairs)


def holm_correct(pvals_list):
    """apply holm-bonferroni to [(label, p), ...], returns {label: (p_adj, sig_str)}"""
    pvals_list.sort(key=lambda x: x[1])
    k = len(pvals_list)
    result = {}
    prev = 0.0
    for i, (label, p) in enumerate(pvals_list):
        p_adj = min(max(prev, p * (k - i)), 1.0)
        prev = p_adj
        result[label] = (p_adj, significance_stars(p_adj))
    return result


def bonferroni_correct_k(results, k):
    for pair, metrics in results.items():
        for mname, d in metrics.items():
            p_adj = min(d["p"] * k, 1.0)
            results[pair][mname][f"p_adj_bonf_k{k}"] = p_adj
            results[pair][mname][f"sig_bonf_k{k}"] = significance_stars(p_adj)


def holm_correct_k(results, k):
    by_metric = {}
    for pair, metrics in results.items():
        for mname, d in metrics.items():
            by_metric.setdefault(mname, []).append((pair, d["p"]))
    for mname, pvals in by_metric.items():
        pvals.sort(key=lambda x: x[1])
        prev = 0.0
        for i, (pair, p) in enumerate(pvals):
            p_adj = min(max(prev, p * max(k - i, 1)), 1.0)
            prev = p_adj
            results[pair][mname][f"p_adj_holm_k{k}"] = p_adj
            results[pair][mname][f"sig_holm_k{k}"] = significance_stars(p_adj)


def run_bootstrap(d: Data):
    """run paired + cluster bootstrap, cache to fig3_main/_boot_results.pkl"""
    _boot_pkl = fig3_main / "_boot_results.pkl"
    _cboot_pkl = fig3_main / "_cboot_results.pkl"

    if _boot_pkl.exists():
        with open(_boot_pkl, "rb") as f:
            _boot_results = pickle.load(f)
        print(f"loaded {len(_boot_results)} cached results from {_boot_pkl}")
    else:
        _boot_results = {}
    if _cboot_pkl.exists():
        with open(_cboot_pkl, "rb") as f:
            _cboot_results = pickle.load(f)
        print(f"loaded {len(_cboot_results)} cached cluster bootstrap results")
    else:
        _cboot_results = {}

    if SKIP_BOOTSTRAP:
        print("--skip-bootstrap: skipping bootstrap computation")

    # standard bootstrap
    _n_new = 0
    for tag, data_src in [("full", d.dfs), ("filtered", d.dfs_filt)]:
        for name, df in data_src.items():
            y = df["y"].values
            label = df["label"].values
            pos = df["pos"].values
            loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)
            preds_dict = {m: df[c].values for m, c in MODEL_COLS.items() if c in df.columns}
            if tag == "filtered":
                df_neut = d.dfs_neutral[name]
                df_comb = pd.concat([df.assign(_nonneutral=1), df_neut.assign(_nonneutral=0)], ignore_index=True)
                y_comb = df_comb["y"].values
                label_comb = df_comb["_nonneutral"].values
                pos_comb = df_comb["pos"].values
                loc_comb = get_location_masks(pos_comb, df_comb["exon_start"].values, df_comb["exon_end"].values)
                preds_dict_comb = {m: df_comb[c].values for m, c in MODEL_COLS.items() if c in df_comb.columns}
            for subset_name in LOCATION_SUBSETS:
                mask = loc[subset_name]
                key = (name, tag, subset_name)
                if key in _boot_results:
                    print(f"cached: {dataset_names[name]} — {tag} — {subset_name}")
                    continue
                if SKIP_BOOTSTRAP:
                    print(f"skipped: {dataset_names[name]} — {tag} — {subset_name}")
                    continue
                n_sub = mask.sum()
                title = f"{dataset_names[name]} — {tag} — {subset_name} (n={n_sub:,})"
                if tag == "filtered":
                    results_corr = bootstrap_compare(
                        y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask])
                    mask_comb = loc_comb[subset_name]
                    results_auprc = bootstrap_compare(
                        y_comb[mask_comb], {m: p[mask_comb] for m, p in preds_dict_comb.items()},
                        label_comb[mask_comb])
                    results = {}
                    for pair in results_corr:
                        results[pair] = {}
                        for metric in ["Pearson", "Spearman", "R2"]:
                            results[pair][metric] = results_corr[pair][metric]
                        if pair in results_auprc:
                            results[pair]["AUPRC"] = results_auprc[pair]["AUPRC"]
                else:
                    results = bootstrap_compare(
                        y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask])
                if results:
                    holm_bonferroni(results)
                    _boot_results[key] = results
                    _n_new += 1
                    print_bootstrap_table(results, model_names, title)
    if _n_new > 0:
        with open(_boot_pkl, "wb") as f:
            pickle.dump(_boot_results, f)
        print(f"\nsaved {len(_boot_results)} results to {_boot_pkl} ({_n_new} new)")

    # cluster bootstrap (resample by exon)
    _cn_new = 0
    for tag, data_src in [("full", d.dfs), ("filtered", d.dfs_filt)]:
        for name, df in data_src.items():
            y = df["y"].values
            label = df["label"].values
            pos = df["pos"].values
            loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)
            clusters = (df["exon_start"].astype(str) + "_" + df["exon_end"].astype(str)).values
            preds_dict = {m: df[c].values for m, c in MODEL_COLS.items() if c in df.columns}
            if tag == "filtered":
                df_neut = d.dfs_neutral[name]
                df_comb = pd.concat([df.assign(_nonneutral=1), df_neut.assign(_nonneutral=0)], ignore_index=True)
                y_comb = df_comb["y"].values
                label_comb = df_comb["_nonneutral"].values
                pos_comb = df_comb["pos"].values
                loc_comb = get_location_masks(pos_comb, df_comb["exon_start"].values, df_comb["exon_end"].values)
                clusters_comb = (df_comb["exon_start"].astype(str) + "_" + df_comb["exon_end"].astype(str)).values
                preds_dict_comb = {m: df_comb[c].values for m, c in MODEL_COLS.items() if c in df_comb.columns}
            for subset_name in LOCATION_SUBSETS:
                mask = loc[subset_name]
                key = (name, tag, subset_name)
                if key in _cboot_results:
                    print(f"cluster cached: {dataset_names[name]} — {tag} — {subset_name}")
                    continue
                if SKIP_BOOTSTRAP:
                    print(f"cluster skipped: {dataset_names[name]} — {tag} — {subset_name}")
                    continue
                title = f"CLUSTER: {dataset_names[name]} — {tag} — {subset_name}"
                if tag == "filtered":
                    results_corr = cluster_bootstrap_compare(
                        y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask], clusters[mask])
                    mask_comb = loc_comb[subset_name]
                    results_auprc = cluster_bootstrap_compare(
                        y_comb[mask_comb], {m: p[mask_comb] for m, p in preds_dict_comb.items()},
                        label_comb[mask_comb], clusters_comb[mask_comb])
                    results = {}
                    for pair in results_corr:
                        results[pair] = {}
                        for metric in ["Pearson", "Spearman", "R2"]:
                            results[pair][metric] = results_corr[pair][metric]
                        if pair in results_auprc:
                            results[pair]["AUPRC"] = results_auprc[pair]["AUPRC"]
                else:
                    results = cluster_bootstrap_compare(
                        y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask], clusters[mask])
                if results:
                    holm_bonferroni(results)
                    _cboot_results[key] = results
                    _cn_new += 1
                    print_bootstrap_table(results, model_names, title)
    if _cn_new > 0:
        with open(_cboot_pkl, "wb") as f:
            pickle.dump(_cboot_results, f)
        print(f"\nsaved {len(_cboot_results)} cluster results ({_cn_new} new)")

    # re-apply corrections
    for key, results in _cboot_results.items():
        holm_bonferroni(results)
        holm_correct_k(results, 4)
        bonferroni_correct_k(results, 4)
        bonferroni_correct_k(results, 10)
    for key, results in _boot_results.items():
        holm_bonferroni(results)
        holm_correct_k(results, 4)
        bonferroni_correct_k(results, 4)
        bonferroni_correct_k(results, 10)
    print("re-applied all corrections")

    d.boot_results = _boot_results
    d.cboot_results = _cboot_results


# ---------------------------------------------------------------------------
# small plot helpers
# ---------------------------------------------------------------------------

def _add_exon_schematic(fig, ax_acc, ax_don, y_fig, rect_h=0.012, line_lw=1.0):
    """split exon schematic with // break between acceptor and donor panels"""
    fig.canvas.draw()
    def _d2f(ax, xd):
        return fig.transFigure.inverted().transform(
            ax.transData.transform((xd, 0)))[0]
    xL = _d2f(ax_acc, ax_acc.get_xlim()[0])
    xEL = _d2f(ax_acc, 0)
    xELR = _d2f(ax_acc, ax_acc.get_xlim()[1])
    xERL = _d2f(ax_don, ax_don.get_xlim()[0])
    xER = _d2f(ax_don, 0)
    xR = _d2f(ax_don, ax_don.get_xlim()[1])
    fig.add_artist(Line2D([xL, xEL], [y_fig, y_fig],
        transform=fig.transFigure, color="black", lw=line_lw, clip_on=False, zorder=10))
    fig.add_artist(Rectangle((xEL, y_fig - rect_h/2), xELR - xEL, rect_h,
        transform=fig.transFigure, facecolor="black", edgecolor="black",
        clip_on=False, zorder=10))
    _gcx = (xELR + xERL) / 2
    _sw = 0.005; _sh = rect_h * 2.5
    for _dx in [-0.004, 0.004]:
        fig.add_artist(Line2D(
            [_gcx + _dx - _sw/2, _gcx + _dx + _sw/2],
            [y_fig - _sh/2, y_fig + _sh/2],
            transform=fig.transFigure, color="black", lw=line_lw,
            clip_on=False, zorder=10))
    fig.add_artist(Rectangle((xERL, y_fig - rect_h/2), xER - xERL, rect_h,
        transform=fig.transFigure, facecolor="black", edgecolor="black",
        clip_on=False, zorder=10))
    fig.add_artist(Line2D([xER, xR], [y_fig, y_fig],
        transform=fig.transFigure, color="black", lw=line_lw, clip_on=False, zorder=10))


def _bootstrap_lowess(x, y, frac=0.15, n_boot=200, seed=42, ci=0.95):
    """lowess with bootstrap confidence interval"""
    rng = np.random.default_rng(seed)
    n = len(x)
    base = _lowess(y, x, frac=frac, return_sorted=True)
    x_grid = base[:, 0]
    boot_curves = np.full((n_boot, len(x_grid)), np.nan)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            curve = _lowess(y[idx], x[idx], frac=frac, return_sorted=True)
            boot_curves[b] = np.interp(x_grid, curve[:, 0], curve[:, 1])
        except Exception:
            pass
    alpha = (1 - ci) / 2
    lo = np.nanpercentile(boot_curves, 100 * alpha, axis=0)
    hi = np.nanpercentile(boot_curves, 100 * (1 - alpha), axis=0)
    return x_grid, base[:, 1], lo, hi


# kde caches keyed by mode
_kde_caches = {"filt": {}, "full_raw": {}, "full_log": {}}


def _kde(mode, name, m, y, pred):
    """gaussian KDE density with mode-specific transform + cache. mode in {filt, full_raw, full_log}"""
    cache = _kde_caches[mode]
    key = (name, m)
    if key not in cache:
        fin = np.isfinite(y) & np.isfinite(pred)
        xm, ym = y[fin], pred[fin]
        z = gaussian_kde(np.vstack([xm, ym]))(np.vstack([xm, ym]))
        if mode == "filt":
            z = np.log1p(z * 1000)
        elif mode == "full_log":
            z = np.log1p(z)
        order = z.argsort()
        cache[key] = (xm, ym, z, order)
    return cache[key]


def _get_kde_filt(name, m, y, pred):
    return _kde("filt", name, m, y, pred)


def _get_kde_full_raw(name, m, y, pred):
    return _kde("full_raw", name, m, y, pred)


def _get_kde(name, m, y, pred):
    return _kde("full_log", name, m, y, pred)


def _compute_bar_metrics(df, models_, loc_subsets):
    """compute metrics per model per location subset"""
    y = df["y"].values
    lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    out = {}
    for sn in loc_subsets:
        mask = loc[sn]
        out[sn] = {}
        for m in models_:
            col = MODEL_COLS[m]
            if col not in df.columns:
                continue
            pred = df[col].values
            r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
            out[sn][m] = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}
    return out


def _cs_get_sig(source, ctx_key, pair, metric, sig_key):
    """extract significance level as integer"""
    _star_to_int = {"": 0, "*": 1, "**": 2, "***": 3}
    if ctx_key not in source:
        return 0
    res = source[ctx_key]
    for key in [pair, pair[::-1]]:
        if key in res and metric in res[key]:
            return _star_to_int.get(res[key][metric].get(sig_key, ""), 0)
    return 0


# ---------------------------------------------------------------------------
# figure functions
# ---------------------------------------------------------------------------

def fig_data_overview(d: Data, name: str,
                      # ---- figure sizing (width is paper-fixed by journal_size clamp) ----
                      height=8,                          # fig height before journal-width clamp; bigger = taller aspect ratio
                      # ---- fonts (None = inherit from rcParams) ----
                      title_fontsize=None,               # panel title size, eg "(a) ΔPSI distribution"
                      label_fontsize=None,               # x/y axis label size
                      tick_fontsize=None,                # tick label size
                      annot_fontsize=9,                  # stats text in upper-right of each panel
                      # ---- histogram styling (used for panels a, c, d) ----
                      hist_bins=60,                      # bin count for the ΔPSI histogram (panel a)
                      dist_bins=50,                      # bin count for distance + exon-width histograms (c, d)
                      hist_color="#4a90d9",              # bar fill color for all 3 histograms
                      hist_edge="white",                 # bar edge color
                      hist_edge_width=0.5,               # bar edge thickness
                      # ---- reference lines on panel (a): zero + ±1 SD ----
                      zero_line_color="red",             # the |dpsi|=0 vertical line
                      zero_line_style="--",              # dashed
                      zero_line_width=1,
                      zero_line_alpha=0.7,
                      sd_line_color="black",             # ±1 SD vertical lines
                      sd_line_style=":",                 # dotted
                      sd_line_width=1,
                      sd_line_alpha=0.6,
                      # ---- location bar chart on panel (b) ----
                      loc_colors=("#4a90d9", "#7fc97f", "#e7298a"),  # one color per category (exon, intron, splice site)
                      loc_edge="black",                  # bar outline
                      loc_edge_width=0.5,
                      # ---- annotation box around stats text ----
                      annot_bbox_alpha=0.8,              # 0=transparent, 1=opaque white background behind stats
                      ) -> None:
    """data overview 2x2: dpsi, location, distance, exon width"""
    df = d.dfs[name]
    y = df["y"].values
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    loc = get_location_masks(pos, exon_start, exon_end)

    # 2x2 grid: dpsi hist | location bars | distance hist | unique-exon-width hist
    fig, axes = plt.subplots(2, 2, figsize=_ru.journal_size(12, height), constrained_layout=True)

    # panel (a): measured dpsi histogram with vertical guides at 0 and ±1 SD
    ax = axes[0, 0]
    ax.hist(y[np.isfinite(y)], bins=hist_bins, edgecolor=hist_edge, linewidth=hist_edge_width, color=hist_color)
    ax.axvline(0, color=zero_line_color, ls=zero_line_style, lw=zero_line_width, alpha=zero_line_alpha)
    sd = np.nanstd(y)
    ax.axvline(sd, color=sd_line_color, ls=sd_line_style, lw=sd_line_width, alpha=sd_line_alpha)
    ax.axvline(-sd, color=sd_line_color, ls=sd_line_style, lw=sd_line_width, alpha=sd_line_alpha)
    ax.set_xlabel("Measured ΔPSI", fontsize=label_fontsize)
    ax.set_ylabel("Count", fontsize=label_fontsize)
    ax.set_title("(a) ΔPSI distribution", fontsize=title_fontsize)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    stats_txt = f"mean={np.nanmean(y):.3f}\nSD={sd:.3f}\nn={np.isfinite(y).sum():,}"
    ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes, fontsize=annot_fontsize,
            va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=annot_bbox_alpha))

    # panel (b): variant counts by Mount-style location (exon / intron / splice site)
    ax = axes[0, 1]
    loc_counts = {LOCATION_LABELS[k]: loc[k].sum() for k in LOCATION_SUBSETS if k != "all"}
    bars = ax.bar(loc_counts.keys(), loc_counts.values(), color=list(loc_colors),
                  edgecolor=loc_edge, linewidth=loc_edge_width)
    for bar, v in zip(bars, loc_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + len(df) * 0.01,
                f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", fontsize=annot_fontsize)
    ax.set_ylabel("Count", fontsize=label_fontsize)
    ax.set_title("(b) Variants by location", fontsize=title_fontsize)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)

    # panel (c): distance to the closer of the two splice sites for each variant
    ax = axes[1, 0]
    dist_nearest = np.minimum(np.abs(pos - exon_start), np.abs(pos - exon_end))
    ax.hist(dist_nearest, bins=dist_bins, edgecolor=hist_edge, linewidth=hist_edge_width, color=hist_color)
    ax.set_xlabel("Distance to nearest splice site (bp)", fontsize=label_fontsize)
    ax.set_ylabel("Count", fontsize=label_fontsize)
    ax.set_title("(c) Distance to splice site", fontsize=title_fontsize)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    ax.text(0.97, 0.95, f"median={np.median(dist_nearest):.0f} bp\nmax={dist_nearest.max():,} bp",
            transform=ax.transAxes, fontsize=annot_fontsize, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=annot_bbox_alpha))

    # panel (d): exon width distribution — one entry per unique exon, not per variant
    ax = axes[1, 1]
    exon_widths = exon_end - exon_start + 1
    unique_widths = pd.Series(exon_widths).groupby(
        [df["chrom"].values, exon_start, exon_end]).first().values
    ax.hist(unique_widths, bins=dist_bins, edgecolor=hist_edge, linewidth=hist_edge_width, color=hist_color)
    ax.set_xlabel("Exon width (bp)", fontsize=label_fontsize)
    ax.set_ylabel("Count (unique exons)", fontsize=label_fontsize)
    ax.set_title("(d) Exon width distribution", fontsize=title_fontsize)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    ax.text(0.97, 0.95, f"median={np.median(unique_widths):.0f} bp\nn={len(unique_widths):,} exons",
            transform=ax.transAxes, fontsize=annot_fontsize, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=annot_bbox_alpha))

    _save(out_dir, f"{name}_data_overview", tight=False)


def fig_pr(d: Data, name: str,
           # ---- figure sizing ----
           height=9,                              # fig height; width fixed by journal_size clamp
           # ---- fonts ----
           title_fontsize=None,                   # panel title (eg "Exon (n=1,002, pos=42)")
           label_fontsize=None,                   # x/y axis labels
           tick_fontsize=None,
           legend_fontsize=6,                     # in-panel legend text (small — many models fit)
           # ---- PR curve styling (per-model lines) ----
           model_line_width=2,                    # line thickness for each model's PR curve
           palette=None,                          # dict {model: color}, None = default model_colors
           # ---- baseline (random-classifier) line ----
           baseline_color="#888888",              # gray
           baseline_style="--",                   # dashed
           baseline_width=1,
           # ---- legend + grid ----
           legend_loc="upper right",              # where each panel's legend sits
           grid_alpha=0.3,                        # 0 = no grid, 1 = solid
           ) -> None:
    """2x2 PR curves stratified by location"""
    df = d.dfs[name]
    cols = pr_cols[name]
    pos = df["pos"].values
    loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)

    # 2x2: one PR curve panel per location category (all / exon / intron / splice region+site)
    fig, axes = plt.subplots(2, 2, figsize=_ru.journal_size(10, height), constrained_layout=True)
    for ax, loc_name in zip(axes.flatten(), LOCATION_SUBSETS):
        mask = loc[loc_name]
        label = df["label"].values[mask]
        n_pos = label.sum()
        for m in models:
            if m not in cols or cols[m] not in df.columns:
                continue
            delta = df[cols[m]].values[mask]
            if label.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(label, np.abs(delta))
            auprc = average_precision_score(label, np.abs(delta))
            mcolor = (palette or {}).get(m, get_color(m))
            ax.plot(rec, prec, color=mcolor, lw=model_line_width, label=f"{model_names[m]} ({auprc:.3f})")
        # baseline = prevalence of positive class in this subset (random classifier perf)
        baseline = label.mean() if len(label) > 0 else 0
        ax.axhline(baseline, color=baseline_color, ls=baseline_style, lw=baseline_width, label=f"baseline ({baseline:.3f})")
        ax.set_xlabel("recall", fontsize=label_fontsize)
        ax.set_ylabel("precision", fontsize=label_fontsize)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{LOCATION_LABELS.get(loc_name, loc_name)} (n={mask.sum():,}, pos={n_pos:,})", fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)
        ax.grid(alpha=grid_alpha)
    _save(out_dir, f"{name}_pr", tight=False)


def fig_threshold(d: Data, name: str,
                  # ---- figure sizing ----
                  height=4.5,                         # fig height
                  # ---- fonts ----
                  title_fontsize=None,
                  label_fontsize=None,
                  tick_fontsize=None,
                  legend_fontsize=9,                  # legend inside each panel
                  sd_marker_fontsize=8,               # the rotated "1 SD = X" text near sd line
                  # ---- model curves (left panel) ----
                  model_line_width=2,                 # line thickness per model
                  palette=None,                       # {model: color} override
                  # ---- 1-SD vertical marker line ----
                  sd_line_color="black",
                  sd_line_style=":",                  # dotted
                  sd_line_width=1.5,
                  sd_line_alpha=0.7,
                  # ---- class distribution colors (right panel) ----
                  pos_color="#D55E00",                # positive class line color (default orange)
                  neg_color="#0072B2",                # negative class line color (default blue)
                  class_line_width=2,
                  # ---- legend + grid ----
                  legend_loc="best",
                  grid_alpha=0.3,
                  ) -> None:
    """AUPRC vs threshold + class distribution"""
    df = d.dfs[name]
    cols = thresh_cols[name]
    thresholds = np.arange(0.01, 1.01, 0.01)
    y = df["y"].values
    sd_thr = np.std(y)

    # 1x2: auprc-vs-threshold | positive/negative counts at each threshold
    fig, axes = plt.subplots(1, 2, figsize=_ru.journal_size(12, height), constrained_layout=True)

    # left panel: sweep |dpsi| threshold from 0.01..1.0, recompute AUPRC at each
    ax = axes[0]
    for m in models:
        if cols[m] not in df.columns: continue
        delta = df[cols[m]].values
        auprcs = []
        for thr in thresholds:
            lab = (np.abs(y) > thr).astype(int)
            auprcs.append(average_precision_score(lab, np.abs(delta)) if lab.sum() > 0 else np.nan)
        mcolor = (palette or {}).get(m, get_color(m))
        ax.plot(thresholds, auprcs, lw=model_line_width, color=mcolor, label=model_names[m])
    # vertical marker at 1 SD of measured dpsi (the threshold we use in main text)
    ax.axvline(sd_thr, color=sd_line_color, ls=sd_line_style, lw=sd_line_width, alpha=sd_line_alpha)
    ax.text(sd_thr + 0.003, 0.95, f"1 SD = {sd_thr:.3f}", fontsize=sd_marker_fontsize, va="top", rotation=90, alpha=sd_line_alpha)
    ax.set_xlabel("|∆PSI| threshold", fontsize=label_fontsize)
    ax.set_ylabel("auprc", fontsize=label_fontsize)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, 1)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    ax.grid(alpha=grid_alpha)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    ax.set_title("auprc vs threshold", fontsize=title_fontsize)

    # right panel: how class balance shifts with the threshold
    ax2 = axes[1]
    n_pos = [(np.abs(y) > thr).sum() for thr in thresholds]
    n_neg = [(np.abs(y) <= thr).sum() for thr in thresholds]
    ax2.plot(thresholds, n_pos, lw=class_line_width, color=pos_color, label="positives")
    ax2.plot(thresholds, n_neg, lw=class_line_width, color=neg_color, label="negatives")
    ax2.axvline(sd_thr, color=sd_line_color, ls=sd_line_style, lw=sd_line_width, alpha=sd_line_alpha)
    ax2.text(sd_thr + 0.003, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 1,
             f"1 SD", fontsize=sd_marker_fontsize, va="top", rotation=90, alpha=sd_line_alpha)
    ax2.set_xlabel("|∆PSI| threshold", fontsize=label_fontsize)
    ax2.set_ylabel("count", fontsize=label_fontsize)
    ax2.set_xlim(thresholds[0], thresholds[-1])
    ax2.legend(loc=legend_loc, fontsize=legend_fontsize)
    ax2.grid(alpha=grid_alpha)
    if tick_fontsize is not None: ax2.tick_params(labelsize=tick_fontsize)
    ax2.set_title("class distribution", fontsize=title_fontsize)

    _save(out_dir, f"{name}_threshold", tight=False)


def fig_all_outputs(d: Data, name: str,
                    # ---- figure sizing (auto-scales with n_bars; override below to force) ----
                    height_per_bar=0.35,            # inches of height per output bar
                    min_height=8,                   # minimum fig height regardless of n_bars
                    # ---- fonts ----
                    title_fontsize=13,              # metric panel titles ("Pearson", "Spearman", "Auprc")
                    label_fontsize=12,              # x-axis label per panel
                    tick_fontsize=None,             # x-tick labels
                    ytick_fontsize=9,               # left-panel y-tick labels (the model output names)
                    value_fontsize=8,               # numeric value text printed next to each bar
                    # ---- bar styling ----
                    bar_edge="white",
                    bar_edge_width=0.5,
                    palette=None,                   # {model: color}; None = get_color(model)
                    # ---- axis + grid ----
                    xlim_max=1.0,                   # x-axis upper bound for all 3 metric panels
                    grid_alpha=0.3,
                    ) -> None:
    """compute metrics tables + 1x3 horizontal bars per model output"""
    df = d.dfs[name]
    y, label = df["y"].values, df["label"].values
    n_pos, n_neg = label.sum(), (1 - label).sum()
    results = []
    for col in combined_cols:
        if col not in df.columns:
            continue
        r, rho, auprc = get_metrics(y, df[col].values, label)
        model = col.split("_")[0]
        if "splaire" in col:
            model = "splaire_ref" if "ref" in col else "splaire_var"
        results.append({"col": col, "model": model, "pearson": r, "spearman": rho, "auprc": auprc})
    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"{dataset_names[name]} (n={len(df)}, pos={n_pos}, neg={n_neg})")
    print(f"{'='*80}")
    print(f"{'column':<45} {'pearson':>10} {'spearman':>10} {'auprc':>10} ")
    print("-" * 80)
    for _, row in results_df.sort_values("auprc", ascending=False).iterrows():
        print(f"{col_display(row['col']):<45} {row['pearson']:>10.3f} {row['spearman']:>10.3f} {row['auprc']:>10.3f}")

    results_df["display_name"] = results_df["col"].apply(col_display)
    results_df["threshold"] = "1-SD"
    results_df.sort_values("auprc", ascending=False).to_csv(f"{out_dir}/{name}_metrics.csv", index=False)
    print(f"  saved {out_dir}/{name}_metrics.csv")

    label_fixed = df["label_fixed"].values
    results_fixed = []
    for col in combined_cols:
        if col not in df.columns:
            continue
        r_f, rho_f, auprc_f = get_metrics(y, df[col].values, label_fixed)
        model = col.split("_")[0]
        if "splaire" in col:
            model = "_".join(col.split("_")[:2])
        results_fixed.append({"col": col, "model": model, "pearson": r_f, "spearman": rho_f, "auprc": auprc_f})
    results_fixed_df = pd.DataFrame(results_fixed)
    results_fixed_df["display_name"] = results_fixed_df["col"].apply(col_display)
    results_fixed_df["threshold"] = "fixed"
    results_fixed_df.sort_values("auprc", ascending=False).to_csv(
        f"{out_dir}/{name}_metrics_fixed_thr.csv", index=False)

    thr_sd = datasets[name]["thr_sd"]
    thr_fixed = datasets[name]["thr"]
    n_sd = label.sum(); n_fixed = label_fixed.sum()
    print(f"\n  threshold comparison ({dataset_names[name]}):")
    print(f"    1-SD (|ΔPSI| > {thr_sd:.3f}): {n_sd:,} positives")
    print(f"    fixed (|ΔPSI| > {thr_fixed:.2f}): {n_fixed:,} positives")
    print(f"    {'model':<25} {'AUPRC (1-SD)':>14} {'AUPRC (fixed)':>14} {'diff':>8}")
    for m, col in MODEL_COLS.items():
        if col not in df.columns:
            continue
        _, _, auprc_sd = get_metrics(y, df[col].values, label)
        _, _, auprc_fx = get_metrics(y, df[col].values, label_fixed)
        diff = auprc_sd - auprc_fx if not (np.isnan(auprc_sd) or np.isnan(auprc_fx)) else np.nan
        print(f"    {model_names.get(m, m):<25} {auprc_sd:>14.4f} {auprc_fx:>14.4f} {diff:>+8.4f}")
    print(f"  saved {out_dir}/{name}_metrics_fixed_thr.csv")

    # metrics by region table
    _all_regions = LOCATION_SUBSETS + LOCATION_SUBSETS_AD
    _pos_r = df["pos"].values.astype(int)
    _loc_here = get_location_masks(_pos_r, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    _region_rows = []
    for rk in _all_regions:
        rmask = _loc_here[rk]
        n_r = rmask.sum()
        if n_r < 5:
            continue
        for m, col in MODEL_COLS.items():
            if col not in df.columns:
                continue
            pred = df[col].values
            r_val, rho_val, auprc_val = get_metrics(y[rmask], pred[rmask], label[rmask])
            _region_rows.append({
                "dataset": name, "region": rk,
                "region_label": ALL_LOCATION_LABELS.get(rk, rk),
                "model": m, "model_label": model_names.get(m, m),
                "n": n_r, "n_sdv": int(label[rmask].sum()),
                "pearson": r_val, "spearman": rho_val, "auprc": auprc_val,
            })
    region_df = pd.DataFrame(_region_rows)
    region_df.to_csv(f"{out_dir}/{name}_metrics_by_region.csv", index=False, float_format="%.4f")
    print(f"\n  metrics by region ({dataset_names[name]}):")
    print(f"  {'region':<18} {'n':>7} {'SDV':>6} {'model':<20} {'Pearson':>8} {'Spearman':>9} {'AUPRC':>8}")
    print(f"  {'-'*80}")
    for rk in _all_regions:
        sub = region_df[region_df["region"] == rk]
        if sub.empty:
            continue
        best = sub.loc[sub["auprc"].idxmax()] if sub["auprc"].notna().any() else sub.iloc[0]
        print(f"  {ALL_LOCATION_LABELS.get(rk, rk):<18} {int(best['n']):>7,} {int(best['n_sdv']):>6} "
              f"{best['model_label']:<20} {best['pearson']:>8.3f} {best['spearman']:>9.3f} {best['auprc']:>8.3f}")
    print(f"  saved {out_dir}/{name}_metrics_by_region.csv")

    # 1x3 horizontal bars — one panel per metric (pearson, spearman, auprc), sorted by auprc
    _metrics_list = ["pearson", "spearman", "auprc"]
    sorted_df = results_df.sort_values("auprc", ascending=True).reset_index(drop=True)
    colors_bar = [(palette or {}).get(row["model"], get_color(row["model"])) for _, row in sorted_df.iterrows()]
    y_pos = np.arange(len(sorted_df))
    n_bars = len(sorted_df)
    fig, axes = plt.subplots(1, 3, figsize=_ru.journal_size(18, max(min_height, n_bars * height_per_bar)),
                             sharey=True, constrained_layout=True)
    for ai, metric in enumerate(_metrics_list):
        ax = axes[ai]
        ax.barh(y_pos, sorted_df[metric], color=colors_bar, edgecolor=bar_edge, linewidth=bar_edge_width)
        ax.set_xlabel(metric.capitalize(), fontsize=label_fontsize)
        ax.set_xlim(0, xlim_max)
        ax.grid(axis="x", alpha=grid_alpha)
        # print numeric value next to each bar (skip non-finite)
        for i, v in enumerate(sorted_df[metric]):
            if np.isfinite(v):
                ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=value_fontsize)
        if ai == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([col_display(c) for c in sorted_df["col"]], fontsize=ytick_fontsize)
        ax.set_title(metric.capitalize(), fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(axis="x", labelsize=tick_fontsize)
    _save(fig3_sup, f"{name}_all_outputs", tight=False)


def fig_dpsi_by_position(d: Data, name: str,
                         # ---- figure sizing ----
                         height=4,                          # fig height
                         wspace=0.15,                       # horizontal gap between acceptor / donor panels
                         bottom_margin=0.22,                # bottom padding (leaves room for exon schematic)
                         # ---- fonts ----
                         title_fontsize=None,
                         label_fontsize=None,
                         tick_fontsize=None,
                         annot_fontsize=9,                  # "n=X" text in upper-left of each panel
                         # ---- scatter styling ----
                         scatter_size=8,                    # marker size (s= arg)
                         scatter_alpha=0.4,
                         scatter_color="#555555",           # dark gray
                         # ---- zero / vertical splice-site reference lines ----
                         ref_line_color="gray",
                         ref_line_style="--",
                         ref_line_width=0.5,
                         ref_line_alpha=0.5,
                         # ---- y-axis range ----
                         ylim_percentile=(1, 99),           # use these percentiles to set symmetric ylim
                         ylim_pad=1.1,                      # multiplier on the percentile-based limit
                         ) -> None:
    """measured dpsi scatter vs position (acceptor + donor panels)"""
    df = d.dfs[name]
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values

    # split variants into acceptor-side vs donor-side based on which splice site is closer
    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)
    x_acc = dist_to_acc[is_acc]
    y_acc = y_true[is_acc]
    x_don = -(exon_end[~is_acc] - pos[~is_acc])
    y_don = y_true[~is_acc]
    fin_acc = np.isfinite(y_acc)
    fin_don = np.isfinite(y_don)

    fig, (ax_acc, ax_don) = plt.subplots(1, 2, figsize=_ru.journal_size(10, height), gridspec_kw={"wspace": wspace})

    # acceptor side
    ax_acc.scatter(x_acc[fin_acc], y_acc[fin_acc], s=scatter_size, alpha=scatter_alpha, color=scatter_color, edgecolors="none")
    ax_acc.axhline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
    ax_acc.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
    ax_acc.set_ylabel("Measured ΔPSI", fontsize=label_fontsize)
    ax_acc.set_xlabel("Position relative to acceptor", labelpad=14, fontsize=label_fontsize)
    ax_acc.set_title("Acceptor", fontsize=title_fontsize)
    if tick_fontsize is not None: ax_acc.tick_params(labelsize=tick_fontsize)
    ax_acc.text(0.02, 0.98, f"n={fin_acc.sum():,}", transform=ax_acc.transAxes, va="top", fontsize=annot_fontsize)

    # donor side
    ax_don.scatter(x_don[fin_don], y_don[fin_don], s=scatter_size, alpha=scatter_alpha, color=scatter_color, edgecolors="none")
    ax_don.axhline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
    ax_don.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
    ax_don.set_xlabel("Position relative to donor", labelpad=14, fontsize=label_fontsize)
    ax_don.set_title("Donor", fontsize=title_fontsize)
    if tick_fontsize is not None: ax_don.tick_params(labelsize=tick_fontsize)
    ax_don.text(0.02, 0.98, f"n={fin_don.sum():,}", transform=ax_don.transAxes, va="top", fontsize=annot_fontsize)

    # symmetric y-limits driven by data percentiles so the zero line stays centered
    all_y = np.concatenate([y_acc[fin_acc], y_don[fin_don]])
    ylim = max(np.abs(np.percentile(all_y, list(ylim_percentile)))) * ylim_pad
    ax_acc.set_ylim(-ylim, ylim)
    ax_don.set_ylim(-ylim, ylim)

    plt.subplots_adjust(bottom=bottom_margin, wspace=wspace)
    # exon schematic strip pinned below the bottom margin
    bot = min(ax_acc.get_position().y0, ax_don.get_position().y0)
    _add_exon_schematic(fig, ax_acc, ax_don, bot - 8/72/fig.get_figheight())
    _save(fig3_sup, f"{name}_dpsi_by_position", tight=False)


def fig_error_by_position(d: Data, name: str,
                          # ---- figure sizing (one fig per model) ----
                          height=4,
                          wspace=0.15,
                          bottom_margin=0.22,
                          # ---- fonts ----
                          title_fontsize=None,
                          label_fontsize=None,
                          tick_fontsize=None,
                          annot_fontsize=9,
                          # ---- scatter styling ----
                          scatter_size=8,
                          scatter_alpha=0.4,
                          palette=None,                       # {model: color}; None uses get_color() so each model gets its branded color
                          # ---- zero / vertical reference lines ----
                          ref_line_color="gray",
                          ref_line_style="--",
                          ref_line_width=0.5,
                          ref_line_alpha=0.5,
                          # ---- y-limits ----
                          ylim_percentile=(1, 99),
                          ylim_pad=1.1,
                          # ---- model selection ----
                          models_to_plot=("spliceai", "splaire_ref", "pangolin", "splicetransformer"),
                          ) -> None:
    """prediction error vs position, per error-model (spliceai, splaire_ref)"""
    df = d.dfs[name]
    _err_cols = MODEL_COLS
    for m in models_to_plot:
        col = _err_cols[m]
        if col not in df.columns:
            continue
        pos = df["pos"].values.astype(int)
        exon_start = df["exon_start"].values.astype(int)
        exon_end = df["exon_end"].values.astype(int)
        y_true = df["y"].values
        y_pred = df[col].values
        error = y_pred - y_true
        # acceptor side vs donor side
        dist_to_acc = pos - exon_start
        dist_to_don = exon_end - pos
        is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)
        x_acc = dist_to_acc[is_acc]
        err_acc = error[is_acc]
        x_don = -(exon_end[~is_acc] - pos[~is_acc])
        err_don = error[~is_acc]
        fin_acc = np.isfinite(err_acc)
        fin_don = np.isfinite(err_don)
        mcolor = (palette or {}).get(m, get_color(m))

        fig, (ax_acc, ax_don) = plt.subplots(1, 2, figsize=_ru.journal_size(10, height), gridspec_kw={"wspace": wspace})

        # acceptor side
        ax_acc.scatter(x_acc[fin_acc], err_acc[fin_acc], s=scatter_size, alpha=scatter_alpha, color=mcolor, edgecolors="none")
        ax_acc.axhline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        ax_acc.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        ax_acc.set_ylabel("Prediction error (pred - measured)", fontsize=label_fontsize)
        ax_acc.set_xlabel("Position relative to acceptor", labelpad=14, fontsize=label_fontsize)
        ax_acc.set_title("Acceptor", fontsize=title_fontsize)
        if tick_fontsize is not None: ax_acc.tick_params(labelsize=tick_fontsize)
        n_acc = fin_acc.sum()
        ax_acc.text(0.02, 0.98, f"n={n_acc:,}", transform=ax_acc.transAxes, va="top", fontsize=annot_fontsize)

        # donor side
        ax_don.scatter(x_don[fin_don], err_don[fin_don], s=scatter_size, alpha=scatter_alpha, color=mcolor, edgecolors="none")
        ax_don.axhline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        ax_don.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        ax_don.set_xlabel("Position relative to donor", labelpad=14, fontsize=label_fontsize)
        ax_don.set_title("Donor", fontsize=title_fontsize)
        if tick_fontsize is not None: ax_don.tick_params(labelsize=tick_fontsize)
        n_don = fin_don.sum()
        ax_don.text(0.02, 0.98, f"n={n_don:,}", transform=ax_don.transAxes, va="top", fontsize=annot_fontsize)

        # symmetric y-limits from data percentiles
        all_err = np.concatenate([err_acc[fin_acc], err_don[fin_don]])
        ylim = max(np.abs(np.percentile(all_err, list(ylim_percentile)))) * ylim_pad
        ax_acc.set_ylim(-ylim, ylim)
        ax_don.set_ylim(-ylim, ylim)
        plt.subplots_adjust(bottom=bottom_margin, wspace=wspace)
        bot = min(ax_acc.get_position().y0, ax_don.get_position().y0)
        _add_exon_schematic(fig, ax_acc, ax_don, bot - 8/72/fig.get_figheight())
        _save(fig3_sup, f"{name}_{m}_error_by_position", tight=False)
        print(f"{dataset_names[name]} — {model_names[m]}: {n_acc:,} acceptor, {n_don:,} donor")


def fig_measured_psi_by_position(d: Data, name: str,
                                 # ---- figure sizing ----
                                 height=4.5,
                                 # ---- fonts ----
                                 title_fontsize=None,
                                 label_fontsize=None,
                                 tick_fontsize=None,
                                 legend_fontsize=8,
                                 # ---- binning ----
                                 bin_width=1,                     # bp per position bin
                                 min_bin_count=5,                 # skip a bin if it has fewer variants
                                 # ---- aggregation curves ----
                                 mean_color="#d62728",            # red — running mean of |dpsi|
                                 mean_style="-",
                                 mean_width=1.5,
                                 mean_alpha=0.8,
                                 median_color="#1f77b4",          # blue — running median of |dpsi|
                                 median_style="--",
                                 median_width=1.5,
                                 median_alpha=0.8,
                                 # ---- zero / splice-site vertical reference line ----
                                 ref_line_color="gray",
                                 ref_line_style="--",
                                 ref_line_width=0.5,
                                 ref_line_alpha=0.5,
                                 # ---- grid + legend ----
                                 grid_alpha=0.2,
                                 legend_loc="best",
                                 ) -> None:
    """mean/median |dpsi| per base position, acceptor + donor"""
    df = d.dfs[name]
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values
    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    fig, axes = plt.subplots(1, 2, figsize=_ru.journal_size(12, height), sharey=True, constrained_layout=True)
    # bin variants by their position offset, then plot per-bin mean + median |dpsi|
    for ax, (side, smask, dvals) in zip(axes, [
        ("Acceptor", is_acc, dist_to_acc[is_acc]),
        ("Donor", ~is_acc, -(exon_end[~is_acc] - pos[~is_acc])),
    ]):
        ys = np.abs(y_true[smask])
        fin = np.isfinite(ys)
        dd, yy = dvals[fin], ys[fin]
        bins = np.arange(dd.min() - bin_width/2, dd.max() + bin_width, bin_width)
        bi = np.digitize(dd, bins)
        cx, cy_mean, cy_med = [], [], []
        for b in range(1, len(bins)):
            bm = bi == b
            if bm.sum() >= min_bin_count:
                cx.append((bins[b-1] + bins[b]) / 2)
                cy_mean.append(np.mean(yy[bm]))
                cy_med.append(np.median(yy[bm]))
        ax.plot(cx, cy_mean, color=mean_color, lw=mean_width, alpha=mean_alpha, ls=mean_style, label="Mean |ΔPSI|")
        ax.plot(cx, cy_med, color=median_color, lw=median_width, alpha=median_alpha, ls=median_style, label="Median |ΔPSI|")
        ax.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        ax.set_xlabel(f"Position relative to {side.lower()}", fontsize=label_fontsize)
        ax.set_title(side, fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.grid(alpha=grid_alpha)
    axes[0].set_ylabel("Measured |ΔPSI|", fontsize=label_fontsize)
    axes[0].legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)
    _save(fig3_sup, f"{name}_measured_psi_by_position", tight=False)


def fig_effect_by_position_rigorous(d: Data, name: str,
                                    # ---- figure sizing ----
                                    height=7,
                                    # ---- fonts ----
                                    title_fontsize=11,                # panel titles "Acceptor"/"Donor"
                                    label_fontsize=None,              # axis labels
                                    tick_fontsize=None,
                                    annot_fontsize=8,                 # "n=X" text
                                    legend_fontsize=8,
                                    # ---- scatter (the raw cloud) ----
                                    scatter_size=4,
                                    scatter_alpha=0.15,
                                    scatter_color="#888888",
                                    scatter_subsample=2000,           # cap shown points for rendering speed
                                    scatter_seed=0,                   # rng seed for the subsample
                                    # ---- LOWESS curve + CI band ----
                                    lowess_frac=0.15,                 # smoothing fraction (smaller = wigglier)
                                    lowess_color="#d62728",           # red
                                    lowess_width=1.8,
                                    ci_alpha=0.15,                    # transparency of the 95% CI band
                                    # ---- zero reference line ----
                                    ref_line_color="gray",
                                    ref_line_style="--",
                                    ref_line_width=0.5,
                                    ref_line_alpha=0.5,
                                    # ---- grid ----
                                    grid_alpha=0.15,
                                    legend_loc="upper right",
                                    ) -> None:
    """effect size by position with lowess + bootstrap CI + per-exon z-score"""
    df = d.dfs[name]
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values

    # build per-exon z-score so each exon's variants are normalized to its own mean/SD
    exon_key = [f"{es}_{ee}" for es, ee in zip(exon_start, exon_end)]
    df["_exon_key_tmp"] = exon_key
    exon_std = df.groupby("_exon_key_tmp")["y"].transform("std")
    exon_mean = df.groupby("_exon_key_tmp")["y"].transform("mean")
    z_score = np.where(exon_std > 0, (y_true - exon_mean) / exon_std, 0.0)
    df.drop(columns=["_exon_key_tmp"], inplace=True)

    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    # 2x2: rows = (raw |dpsi|, exon-normalized |z|), cols = (acceptor, donor)
    fig, axes = plt.subplots(2, 2, figsize=_ru.journal_size(12, height), constrained_layout=True)
    for row, (y_plot, y_label, title_tag) in enumerate([
        (np.abs(y_true), r"$|\Delta\mathrm{PSI}|$", "Raw"),
        (np.abs(z_score), r"$|z|$ (exon-normalized)", "Exon-normalized"),
    ]):
        for col, (side, smask, dvals) in enumerate([
            ("Acceptor", is_acc, dist_to_acc[is_acc]),
            ("Donor", ~is_acc, -(exon_end[~is_acc] - pos[~is_acc])),
        ]):
            ax = axes[row, col]
            ys = y_plot[smask]
            fin = np.isfinite(ys)
            dd, yy = dvals[fin].astype(float), ys[fin]
            n_pts = len(dd)
            # subsample the raw cloud so the plot renders fast even with 27k MFASS variants
            if n_pts > scatter_subsample:
                sub = np.random.default_rng(scatter_seed).choice(n_pts, scatter_subsample, replace=False)
            else:
                sub = np.arange(n_pts)
            ax.scatter(dd[sub], yy[sub], s=scatter_size, alpha=scatter_alpha, color=scatter_color,
                       edgecolors="none", rasterized=True, zorder=1)
            # lowess + bootstrap CI on the FULL data (not the subsample)
            x_grid, y_smooth, lo, hi = _bootstrap_lowess(dd, yy, frac=lowess_frac)
            ax.plot(x_grid, y_smooth, color=lowess_color, lw=lowess_width, zorder=3, label="LOWESS")
            ax.fill_between(x_grid, lo, hi, color=lowess_color, alpha=ci_alpha, zorder=2, label="95% CI")
            ax.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
            if row == 0:
                ax.set_title(side, fontsize=title_fontsize)
            if col == 0:
                ax.set_ylabel(f"{title_tag}\n{y_label}", fontsize=label_fontsize)
            if row == 1:
                ax.set_xlabel(f"Position relative to {side.lower()}", fontsize=label_fontsize)
            if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
            ax.grid(alpha=grid_alpha)
            ax.text(0.02, 0.95, f"n={n_pts:,}", transform=ax.transAxes, va="top", fontsize=annot_fontsize)
    axes[0, 1].legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)
    _save(fig3_sup, f"{name}_effect_by_position_rigorous", tight=False)


def fig_pct_sdv_by_position(d: Data, name: str,
                            # ---- overall figure sizing ----
                            height_pad=1,                       # added to sum of h_ratios for total height
                            hspace=0.15,                        # vertical gap between row panels
                            # row heights (relative units) — change these to resize individual rows
                            h_ratios=None,                      # None = default sdv 2.5, mae 1.8, disagree 1.5, phylop 1.2, density 1.0, bars 2.0
                            # ---- fonts ----
                            title_fontsize=12,                  # main title at top of sdv panel
                            ylabel_fontsize=9,                  # per-row y-axis labels (%SDV, MAE, etc.)
                            sdv_ylabel_fontsize=10,             # sdv-row y-label (slightly larger)
                            density_xlabel_fontsize=10,         # main x-axis label at very bottom
                            xtick_fontsize=8,                   # bottom-row x-tick labels (intron coords + Acc/Exon/Don)
                            ad_xtick_fontsize=6.5,              # ad-region bar panels x-tick labels
                            ad_xtick_rotation=30,               # rotation of ad-region x-tick labels (deg)
                            legend_fontsize=6,                  # in-panel legend
                            annot_fontsize=5,                   # small "n=X,SDVs=Y" labels under ad bars
                            # ---- model curves + bars ----
                            model_line_width=1,                 # mae line thickness
                            palette=None,                       # {model: color}
                            bar_edge="black",                   # ad-region bar outline
                            bar_edge_width=0.3,                 # ad-region bar outline width
                            # ---- region background shading ----
                            region_shade_alpha=0.06,            # 0=no shade, 1=opaque
                            region_shade_colors=None,           # {"intron": .., "splice_region": .., "splice_site": .., "exon": ..}
                            # ---- exon/splice-site dashed vertical guides ----
                            guide_line_color="#333333",
                            guide_line_style="--",
                            guide_line_width=0.5,
                            guide_line_alpha=0.3,
                            # ---- grid ----
                            grid_alpha=0.15,
                            ) -> None:
    """%SDV by normalized position along exon + flanking intron (Chong 2019 Fig 4C)"""
    # internal styling default tables (kept here so kwargs above stay short)
    if h_ratios is None:
        h_ratios = {"sdv": 2.5, "mae": 1.8, "disagreement": 1.5, "phylop": 1.2, "density": 1.0,
                    "pearson_ad": 2.0, "spearman_ad": 2.0, "auprc_ad": 2.0}
    if region_shade_colors is None:
        region_shade_colors = {"intron": "#009E73", "splice_region": "#E69F00",
                               "splice_site": "#D55E00", "exon": "#0072B2"}
    df = d.dfs[name]
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y = df["y"].values
    label = df["label"].values
    thr = datasets[name]["thr"]

    exon_width = (exon_end - exon_start).astype(float)
    in_exon = (pos >= exon_start) & (pos <= exon_end)
    in_upstream = pos < exon_start
    in_downstream = pos > exon_end
    upstream_dist = pos - exon_start
    downstream_dist = pos - exon_end
    exon_frac = np.where(exon_width > 0, (pos - exon_start) / exon_width, 0.5)

    n_exon_bins = 50
    exon_edges = np.linspace(0, 1, n_exon_bins + 1)
    up_dists = upstream_dist[in_upstream]
    up_labels = label[in_upstream]
    up_min, up_max = int(up_dists.min()), int(up_dists.max())
    up_bins = np.arange(up_min - 0.5, up_max + 1.5, 1)
    dn_dists = downstream_dist[in_downstream]
    dn_labels = label[in_downstream]
    dn_min, dn_max = int(dn_dists.min()), int(dn_dists.max())
    dn_bins = np.arange(dn_min - 0.5, dn_max + 1.5, 1)
    ex_fracs = exon_frac[in_exon]
    ex_labels = label[in_exon]

    def pct_sdv(values, labels, bin_edges):
        bi = np.digitize(values, bin_edges)
        cx, cy, cn = [], [], []
        for b in range(1, len(bin_edges)):
            mask = bi == b
            n = mask.sum()
            if n >= 3:
                cx.append((bin_edges[b-1] + bin_edges[b]) / 2)
                cy.append(100 * labels[mask].sum() / n)
                cn.append(n)
        return np.array(cx), np.array(cy), np.array(cn)

    def bin_mae(values, pred, truth, bin_edges):
        bi = np.digitize(values, bin_edges)
        cx, cy = [], []
        for b in range(1, len(bin_edges)):
            mask = bi == b
            fin = mask & np.isfinite(pred) & np.isfinite(truth)
            if fin.sum() >= 3:
                cx.append((bin_edges[b-1] + bin_edges[b]) / 2)
                cy.append(np.mean(np.abs(pred[fin] - truth[fin])))
        return np.array(cx), np.array(cy)

    def bin_stat(values, scores, bin_edges, func=np.nanmean):
        bi = np.digitize(values, bin_edges)
        cx, cy = [], []
        for b in range(1, len(bin_edges)):
            mask = bi == b
            fin = mask & np.isfinite(scores)
            if fin.sum() >= 3:
                cx.append((bin_edges[b-1] + bin_edges[b]) / 2)
                cy.append(func(scores[fin]))
        return np.array(cx), np.array(cy)

    up_cx, up_cy, up_cn = pct_sdv(up_dists, up_labels, up_bins)
    dn_cx, dn_cy, dn_cn = pct_sdv(dn_dists, dn_labels, dn_bins)
    ex_cx, ex_cy, ex_cn = pct_sdv(ex_fracs, ex_labels, exon_edges)
    exon_plot_width = 40
    ex_x_mapped = ex_cx * exon_plot_width
    dn_x_mapped = dn_cx + exon_plot_width

    _sdv_models = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]

    # phyloP — try loading
    _phylop_path = Path("vex_seq/hg19.phyloP46way.bw")
    _has_phylop = False
    _bw = None
    try:
        import pyBigWig
        if _phylop_path.exists():
            _bw = pyBigWig.open(str(_phylop_path))
            _has_phylop = True
            print(f"loaded phyloP: {_phylop_path}")
    except (ImportError, Exception):
        pass

    _row_names = ["sdv", "mae", "disagreement"]
    if _has_phylop:
        _row_names.append("phylop")
    _row_names.extend(["density", "pearson_ad", "spearman_ad", "auprc_ad"])
    n_rows = len(_row_names)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=_ru.journal_size(12, sum(h_ratios[r] for r in _row_names) + height_pad),
        height_ratios=[h_ratios[r] for r in _row_names],
        gridspec_kw={"hspace": hspace})
    _ax = {r: axes[i] for i, r in enumerate(_row_names)}

    _bar_rows = {"pearson_ad", "spearman_ad", "auprc_ad"}
    _pos_rows = [r for r in _row_names if r not in _bar_rows]
    _bar_row_list = [r for r in _row_names if r in _bar_rows]
    for r in _pos_rows[1:]:
        _ax[r].sharex(_ax[_pos_rows[0]])
    for r in _bar_row_list[1:]:
        _ax[r].sharex(_ax[_bar_row_list[0]])
    for r in _bar_row_list[:-1]:
        _ax[r].tick_params(labelbottom=False)

    _region_shade = region_shade_colors
    _shade_alpha = region_shade_alpha
    _x_left = (up_cx.min() if len(up_cx) else -30) - 2
    _x_right = (dn_x_mapped.max() if len(dn_x_mapped) else exon_plot_width + 30) + 2
    _region_spans = [
        ("intron", _x_left, -8),
        ("splice_region", -8, -2),
        ("splice_site", -2, 0),
        ("exon", 0, exon_plot_width),
        ("splice_site", exon_plot_width, exon_plot_width + 2),
        ("splice_region", exon_plot_width + 2, exon_plot_width + 8),
        ("intron", exon_plot_width + 8, _x_right),
    ]
    def _shade_regions(ax):
        for rname, x0, x1 in _region_spans:
            ax.axvspan(x0, x1, color=_region_shade[rname], alpha=_shade_alpha, zorder=0)
        for xb in [0, exon_plot_width]:
            ax.axvline(xb, color=guide_line_color, ls=guide_line_style, lw=guide_line_width, alpha=guide_line_alpha)
    _shade_exon = _shade_regions

    def _plot_3region(ax, up_x, up_y, ex_x, ex_y, dn_x, dn_y, color, lw=1, alpha=0.8, label=None):
        ax.plot(up_x, up_y, color=color, lw=lw, alpha=alpha)
        ax.plot(ex_x * exon_plot_width, ex_y, color=color, lw=lw, alpha=alpha, label=label)
        ax.plot(dn_x + exon_plot_width, dn_y, color=color, lw=lw, alpha=alpha)

    ax = _ax["sdv"]
    ax.fill_between(up_cx, 0, up_cy, color="#888888", alpha=0.3, step="mid")
    ax.plot(up_cx, up_cy, color="#333333", lw=1.2, label="All")
    ax.fill_between(ex_x_mapped, 0, ex_cy, color="#4a86c8", alpha=0.3, step="mid")
    ax.plot(ex_x_mapped, ex_cy, color="#2b5d8e", lw=1.2)
    ax.fill_between(dn_x_mapped, 0, dn_cy, color="#888888", alpha=0.3, step="mid")
    ax.plot(dn_x_mapped, dn_cy, color="#333333", lw=1.2)
    ax.set_ylabel("%SDV", fontsize=sdv_ylabel_fontsize)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=grid_alpha, axis="y")
    ax.set_title(
        f"{dataset_names[name]} — positional analysis "
        f"({len(df):,} variants, {label.sum():,} SDVs, |ΔPSI| > {thr})",
        fontsize=title_fontsize)
    _shade_exon(ax)

    loc = get_location_masks(pos, exon_start, exon_end)
    _ad_regions = [
        "intron_acc", "splice_region_acc", "splice_site_acc",
        "exon_acc", "exon_don",
        "splice_site_don", "splice_region_don", "intron_don",
    ]
    n_ad = len(_ad_regions)
    n_mods = len(_sdv_models)
    bar_w = 0.7 / n_mods

    for metric_key, metric_name, metric_fn in [
        ("pearson_ad", "Pearson", lambda ym, pm, lm: stats.pearsonr(ym, pm)[0] if len(ym) >= 3 else np.nan),
        ("spearman_ad", "Spearman", lambda ym, pm, lm: stats.spearmanr(ym, pm)[0] if len(ym) >= 3 else np.nan),
        ("auprc_ad", "AUPRC", lambda ym, pm, lm: average_precision_score(lm, np.abs(pm)) if lm.sum() > 0 and len(lm) >= 5 else np.nan),
    ]:
        ax = _ax[metric_key]
        x_pos = np.arange(n_ad)
        for mi, m in enumerate(_sdv_models):
            col = MODEL_COLS[m]
            pred = df[col].values
            vals = []
            for rk in _ad_regions:
                rmask = loc[rk]
                fin = rmask & np.isfinite(pred) & np.isfinite(y)
                if fin.sum() >= 3:
                    vals.append(metric_fn(y[fin], pred[fin], label[fin]))
                else:
                    vals.append(np.nan)
            c = (palette or {}).get(m, get_color(m))
            x = x_pos + (mi - n_mods / 2 + 0.5) * bar_w
            ax.bar(x, vals, bar_w, color=c, edgecolor=bar_edge, linewidth=bar_edge_width,
                   label=model_names.get(m, m) if metric_key == "pearson_ad" else "")
        for ri, rk in enumerate(_ad_regions):
            rmask = loc[rk]
            n_r = rmask.sum()
            n_sdv_r = label[rmask].sum()
            ax.text(ri, -0.12, f"{n_r:,}\n({n_sdv_r})", ha="center", fontsize=annot_fontsize,
                    color="#666666", transform=ax.get_xaxis_transform())
        _ad_parent = {
            "splice_site_acc": "splice_site", "splice_site_don": "splice_site",
            "splice_region_acc": "splice_region", "splice_region_don": "splice_region",
            "exon_acc": "exon", "exon_don": "exon",
            "intron_acc": "intron", "intron_don": "intron",
        }
        for ri, rk in enumerate(_ad_regions):
            parent = _ad_parent[rk]
            ax.axvspan(ri - 0.5, ri + 0.5, color=_region_shade[parent],
                       alpha=_shade_alpha, zorder=0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([LOCATION_LABELS_AD[rk] for rk in _ad_regions], fontsize=ad_xtick_fontsize, rotation=ad_xtick_rotation, ha="right")
        ax.set_ylabel(metric_name, fontsize=ylabel_fontsize)
        ax.grid(alpha=grid_alpha, axis="y")
        if metric_key == "pearson_ad":
            ax.legend(fontsize=legend_fontsize, ncol=3, loc="upper right")

    ax = _ax["mae"]
    for m in _sdv_models:
        col = MODEL_COLS[m]
        pred = df[col].values
        up_mx, up_my = bin_mae(up_dists, pred[in_upstream], y[in_upstream], up_bins)
        ex_mx, ex_my = bin_mae(ex_fracs, pred[in_exon], y[in_exon], exon_edges)
        dn_mx, dn_my = bin_mae(dn_dists, pred[in_downstream], y[in_downstream], dn_bins)
        c = (palette or {}).get(m, get_color(m))
        _plot_3region(ax, up_mx, up_my, ex_mx, ex_my, dn_mx, dn_my,
                      color=c, lw=model_line_width, label=model_names.get(m, m))
    ax.set_ylabel("MAE", fontsize=ylabel_fontsize)
    ax.grid(alpha=grid_alpha, axis="y")
    ax.legend(fontsize=legend_fontsize, ncol=3, loc="upper right")
    _shade_exon(ax)

    ax = _ax["disagreement"]
    model_preds = []
    for m in _sdv_models:
        col = MODEL_COLS[m]
        model_preds.append(np.abs(df[col].values))
    model_stack = np.column_stack(model_preds)
    model_var = np.nanstd(model_stack, axis=1)
    up_vx, up_vy = bin_stat(up_dists, model_var[in_upstream], up_bins)
    ex_vx, ex_vy = bin_stat(ex_fracs, model_var[in_exon], exon_edges)
    dn_vx, dn_vy = bin_stat(dn_dists, model_var[in_downstream], dn_bins)
    ax.fill_between(up_vx, 0, up_vy, color="#d95f02", alpha=0.3)
    ax.plot(up_vx, up_vy, color="#d95f02", lw=1)
    ax.fill_between(ex_vx * exon_plot_width, 0, ex_vy, color="#d95f02", alpha=0.3)
    ax.plot(ex_vx * exon_plot_width, ex_vy, color="#d95f02", lw=1)
    ax.fill_between(dn_vx + exon_plot_width, 0, dn_vy, color="#d95f02", alpha=0.3)
    ax.plot(dn_vx + exon_plot_width, dn_vy, color="#d95f02", lw=1)
    ax.set_ylabel("Model\ndisagree.", fontsize=9)
    ax.grid(alpha=grid_alpha, axis="y")
    _shade_exon(ax)

    if _has_phylop:
        ax = _ax["phylop"]
        chroms_arr = df["chrom"].values if "chrom" in df.columns else None
        if chroms_arr is not None:
            phylop_vals = np.full(len(df), np.nan)
            for i in range(len(df)):
                try:
                    phylop_vals[i] = _bw.values(str(chroms_arr[i]), int(pos[i]) - 1, int(pos[i]))[0]
                except Exception:
                    pass
            up_px, up_py = bin_stat(up_dists, phylop_vals[in_upstream], up_bins)
            ex_px, ex_py = bin_stat(ex_fracs, phylop_vals[in_exon], exon_edges)
            dn_px, dn_py = bin_stat(dn_dists, phylop_vals[in_downstream], dn_bins)
            _plot_3region(ax, up_px, up_py, ex_px, ex_py, dn_px, dn_py,
                          color="#7570b3", lw=1.2)
            ax.set_ylabel("phyloP", fontsize=9)
            ax.grid(alpha=grid_alpha, axis="y")
        _shade_exon(ax)

    ax = _ax["density"]
    ax.bar(up_cx, up_cn, width=1, color="#888888", alpha=0.5, edgecolor="none")
    ax.bar(ex_x_mapped, ex_cn, width=exon_plot_width / n_exon_bins,
           color="#4a86c8", alpha=0.5, edgecolor="none")
    ax.bar(dn_x_mapped, dn_cn, width=1, color="#888888", alpha=0.5, edgecolor="none")
    ax.set_ylabel("SNV\ncount", fontsize=9)
    ax.set_xlabel("Position (intron bp / normalized exon / intron bp)", fontsize=10)
    _shade_exon(ax)

    x_left = (up_cx.min() if len(up_cx) else -30)
    x_right = (dn_x_mapped.max() if len(dn_x_mapped) else exon_plot_width + 30)
    _ax["density"].set_xlim(x_left - 2, x_right + 2)
    intron_ticks_left = [t for t in [-30, -20, -10, -5, -2] if t >= x_left]
    intron_ticks_right_raw = [2, 5, 10, 20, 30]
    intron_ticks_right = [t + exon_plot_width for t in intron_ticks_right_raw
                          if t + exon_plot_width <= x_right]
    exon_ticks = [0, exon_plot_width / 2, exon_plot_width]
    all_ticks = intron_ticks_left + exon_ticks + intron_ticks_right
    all_tick_labels = ([str(t) for t in intron_ticks_left] +
                       ["Acc", "Exon", "Don"] +
                       [f"+{t}" for t in intron_ticks_right_raw
                        if t + exon_plot_width <= x_right])
    _ax["density"].set_xticks(all_ticks)
    _ax["density"].set_xticklabels(all_tick_labels, fontsize=8)
    for r in _pos_rows[:-1]:
        _ax[r].tick_params(labelbottom=False)

    _save(fig3_sup, f"{name}_pct_sdv_by_position", tight=False)

    _loc_summary = get_location_masks(pos, exon_start, exon_end)
    print(f"{dataset_names[name]}:")
    for lk in ["splice_site", "splice_region", "exon", "intron"]:
        lm = _loc_summary[lk]
        print(f"  SDVs in {LOCATION_LABELS[lk]}: {label[lm].sum()} / {lm.sum()} "
              f"({100 * label[lm].mean():.1f}% SDV rate)")
    ss_mount = _loc_summary["splice_site"] | _loc_summary["splice_region"]
    print(f"  SDVs in SS+region (Mount): {label[ss_mount].sum()} / {ss_mount.sum()} "
          f"({100 * label[ss_mount].mean():.1f}%)")
    if _has_phylop and _bw is not None:
        _bw.close()


def fig_pct_sdv_by_position_combined(d: Data,
                                     # ---- panel-letter label (top-left "a" / "b") ----
                                     letter_fontsize=24,
                                     letter_weight="bold",
                                     # ---- dataset-name caption (top-center) ----
                                     name_fontsize=16,
                                     name_weight="bold",
                                     # ---- figure auto-height (driven by stitched panel aspect ratios) ----
                                     fig_w=24,                            # composed width before journal_size clamp
                                     dpi=300,
                                     ) -> None:
    """combined 1x2 panel: vex-seq (a) + mfass (b)"""
    # reads pre-rendered per-dataset PNGs from disk and tiles them side by side
    _combined_paths = [
        ("a", f"{fig3_sup}/vexseq_pct_sdv_by_position.png", "Vex-seq"),
        ("b", f"{fig3_sup}/mfass_pct_sdv_by_position.png", "MFASS"),
    ]
    _panel_imgs = []
    for letter, path, dname in _combined_paths:
        if Path(path).exists():
            _panel_imgs.append((letter, _imread_panel(path), dname))
    if len(_panel_imgs) != 2:
        print(f"skipping combined panel — only {len(_panel_imgs)}/2 per-dataset panels available")
        return
    # match composed height to the taller of the two panels' aspect ratios
    h_a, w_a = _panel_imgs[0][1].shape[:2]
    h_b, w_b = _panel_imgs[1][1].shape[:2]
    fig_h = fig_w / 2 * max(h_a / w_a, h_b / w_b)
    fig_cmb, axes_cmb = plt.subplots(1, 2, figsize=_ru.journal_size(fig_w, fig_h), constrained_layout=True)
    for ax, (letter, img, dname) in zip(axes_cmb, _panel_imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.text(-0.01, 1.01, letter, transform=ax.transAxes,
                fontsize=letter_fontsize, fontweight=letter_weight, va="bottom", ha="left")
        ax.text(0.5, 1.01, dname, transform=ax.transAxes,
                fontsize=name_fontsize, fontweight=name_weight, va="bottom", ha="center")
    _save(fig3_sup, "pct_sdv_by_position_combined", dpi=dpi, tight=False)
    print(f"saved combined panel: {fig3_sup}/pct_sdv_by_position_combined.png")


# transition/transversion lookup
_TI = {("A","G"), ("G","A"), ("C","T"), ("T","C")}


def fig_mae_by_position(d: Data, name: str,
                        # ---- figure sizing ----
                        height=4.5,
                        # ---- fonts ----
                        title_fontsize=None,
                        label_fontsize=None,
                        tick_fontsize=None,
                        legend_fontsize=8,
                        # ---- binning ----
                        bin_width=1,                       # bp per position bin
                        min_bin_count=5,                   # skip bins with fewer variants than this
                        # ---- per-model lines ----
                        model_line_width=1.5,
                        model_line_alpha=0.8,
                        palette=None,                      # {model: color} override
                        # ---- zero / splice-site reference line ----
                        ref_line_color="gray",
                        ref_line_style="--",
                        ref_line_width=0.5,
                        ref_line_alpha=0.5,
                        # ---- grid + legend ----
                        grid_alpha=0.2,
                        legend_loc="best",
                        ) -> None:
    """binned MAE by position, acceptor + donor panels"""
    df, y_true, mods = _fig_setup(d, name)
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)

    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    fig, axes = plt.subplots(1, 2, figsize=_ru.journal_size(12, height), sharey=True, constrained_layout=True)
    # one line per model showing mean |err| in each 1-bp position bin
    for ax, (side, smask, dvals) in zip(axes, [
        ("Acceptor", is_acc, dist_to_acc[is_acc]),
        ("Donor", ~is_acc, -(exon_end[~is_acc] - pos[~is_acc])),
    ]):
        for m in mods:
            err_side = np.abs(df[MODEL_COLS[m]].values - y_true)[smask]
            fin = np.isfinite(err_side)
            dd, ee = dvals[fin], err_side[fin]
            bins = np.arange(dd.min() - bin_width/2, dd.max() + bin_width, bin_width)
            bi = np.digitize(dd, bins)
            cx, cy = [], []
            for b in range(1, len(bins)):
                bm = bi == b
                if bm.sum() >= min_bin_count:
                    cx.append((bins[b-1] + bins[b]) / 2)
                    cy.append(np.mean(ee[bm]))
            mcolor = (palette or {}).get(m, get_color(m))
            ax.plot(cx, cy, color=mcolor, lw=model_line_width, alpha=model_line_alpha, label=model_names[m])
        ax.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        ax.set_xlabel(f"Position relative to {side.lower()}", fontsize=label_fontsize)
        ax.set_title(side, fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.grid(alpha=grid_alpha)
    axes[0].set_ylabel("Mean absolute error", fontsize=label_fontsize)
    axes[0].legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)
    _save(fig3_sup, f"{name}_mae_by_position", tight=False)


def fig_error_vs_effect(d: Data, name: str,
                        # ---- figure sizing (one panel per model, n_panels = len(mods)) ----
                        height=4.5,
                        width_per_panel=3,              # inches per model panel (input to journal_size; clamped by paper width)
                        min_width=12,                   # floor on total width
                        # ---- fonts ----
                        title_fontsize=None,            # model name above each panel
                        label_fontsize=None,
                        tick_fontsize=None,
                        # ---- scatter (the raw error cloud) ----
                        scatter_size=4,
                        scatter_alpha=0.15,
                        palette=None,                   # {model: color} override; defaults to get_color()
                        # ---- binned-mean overlay (black line through the cloud) ----
                        n_percentile_bins=21,           # number of percentile-cut bins along x
                        min_bin_count=3,                # skip bins with fewer points
                        binned_line_color="black",
                        binned_line_width=2,
                        binned_line_z=5,                # zorder so it draws on top of scatter
                        # ---- grid ----
                        grid_alpha=0.2,
                        ) -> None:
    """|prediction error| vs |measured ΔPSI| per model"""
    df, y_true, mods = _fig_setup(d, name)

    fig, axes = plt.subplots(1, len(mods),
                             figsize=_ru.journal_size(max(min_width, width_per_panel * len(mods)), height),
                             sharey=True, squeeze=False, constrained_layout=True)
    axes = axes[0]
    for ax, m in zip(axes, mods):
        abs_err = np.abs(df[MODEL_COLS[m]].values - y_true)
        abs_y = np.abs(y_true)
        fin = np.isfinite(abs_err) & np.isfinite(abs_y)
        mcolor = (palette or {}).get(m, get_color(m))
        ax.scatter(abs_y[fin], abs_err[fin], s=scatter_size, alpha=scatter_alpha, color=mcolor,
                   edgecolors="none", rasterized=True)
        # percentile-binned mean error overlay so the trend through the cloud is visible
        pct = np.percentile(abs_y[fin], np.linspace(0, 100, n_percentile_bins))
        bi = np.digitize(abs_y[fin], pct)
        cx, cy = [], []
        for b in range(1, len(pct)):
            bm = bi == b
            if bm.sum() >= min_bin_count:
                cx.append((pct[b-1] + pct[b]) / 2)
                cy.append(np.mean(abs_err[fin][bm]))
        ax.plot(cx, cy, color=binned_line_color, lw=binned_line_width, zorder=binned_line_z)
        ax.set_xlabel("|Measured ΔPSI|", fontsize=label_fontsize)
        ax.set_title(model_names[m], fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.grid(alpha=grid_alpha)
    axes[0].set_ylabel("|Prediction error|", fontsize=label_fontsize)
    _save(fig3_sup, f"{name}_error_vs_effect", tight=False)


# gtf cache shared across error_vs_intron_length calls
_gtf_exons_by_strand_cache = None


def _load_gtf_exons():
    """load gencode v19 exon intervals indexed by (chrom, strand)"""
    global _gtf_exons_by_strand_cache
    if _gtf_exons_by_strand_cache is not None:
        return _gtf_exons_by_strand_cache
    _gtf_candidates = []
    if os.environ.get("SPLAIRE_MFASS_GTF"):
        _gtf_candidates.append(Path(os.environ["SPLAIRE_MFASS_GTF"]))
    _gtf_candidates.extend([
        # canonical: pipeline/reference/GRCh37/ (mirrors hg19.fa location)
        Path(__file__).resolve().parent / ".." / ".." / "pipeline" / "reference" / "GRCh37" / "gencode.v19.annotation.gtf",
        Path("../../pipeline/reference/GRCh37/gencode.v19.annotation.gtf"),
        # legacy locations from pre-refactor scripts
        Path(__file__).resolve().parent / ".." / ".." / "pipeline" / "python" / "mfass" / "gencode.v19.annotation.gtf",
        Path("../../pipeline/python/mfass/gencode.v19.annotation.gtf"),
        Path("gencode.v19.annotation.gtf"),
    ])
    gtf_path = next((p for p in _gtf_candidates if p.exists()), None)
    if gtf_path is None:
        print(f"GTF not found (tried {[str(p) for p in _gtf_candidates]}); "
              f"set $SPLAIRE_MFASS_GTF to a gencode.v19.annotation.gtf to enable intron-length figures")
        return None
    print(f"loading GTF: {gtf_path}")
    out = {}
    with open(gtf_path) as fgtf:
        for line in fgtf:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom = parts[0]
            strand = parts[6]
            s, e = int(parts[3]), int(parts[4])
            key = (chrom, strand)
            if key not in out:
                out[key] = []
            out[key].append((s, e))
    for key in out:
        out[key] = np.array(sorted(set(out[key])))
    n_total = sum(len(v) for v in out.values())
    print(f"loaded {n_total:,} exon intervals across {len(out)} chrom/strand pairs")
    _gtf_exons_by_strand_cache = out
    return out


def fig_error_vs_intron_length(d: Data, name: str,
                               # ---- figure sizing (one fig per aggregation: min/max/mean) ----
                               height=4.5,
                               width_per_panel=3,             # inches per model panel before journal_size clamp
                               min_width=12,
                               # ---- fonts ----
                               title_fontsize=None,           # model name above each panel
                               label_fontsize=None,
                               tick_fontsize=None,
                               # ---- scatter (raw cloud, points = variants) ----
                               scatter_size=4,
                               scatter_alpha=0.15,
                               palette=None,                  # {model: color} override
                               # ---- binned-mean trend line ----
                               n_percentile_bins=16,          # number of percentile cuts along x
                               min_bin_count=5,               # skip bins with fewer points
                               binned_line_color="black",
                               binned_line_width=2,
                               binned_line_z=5,
                               # ---- grid ----
                               grid_alpha=0.2,
                               ) -> None:
    """error vs min/max/mean flanking intron length (3 figures per dataset)"""
    gtf = _load_gtf_exons()
    if gtf is None:
        _ZENODO_SKIPPED.add("fig_error_vs_intron_length")
        print(f"skipping fig_error_vs_intron_length: no GTF available")
        return
    df, y_true, mods = _fig_setup(d, name)
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)

    # look up nearest upstream/downstream exon in gencode for each variant's exon
    chroms = df["chrom"].values if "chrom" in df.columns else df["seqnames"].values
    strands = df["strand"].values
    up_intron = np.full(len(df), np.nan)
    dn_intron = np.full(len(df), np.nan)
    for i in range(len(df)):
        ch = chroms[i]
        st = strands[i]
        es, ee = exon_start[i], exon_end[i]
        key = (ch, st)
        if key not in gtf or len(gtf[key]) == 0:
            continue
        gtf_starts = gtf[key][:, 0]
        gtf_ends = gtf[key][:, 1]
        upstream_mask = gtf_ends < es
        if upstream_mask.any():
            up_intron[i] = es - gtf_ends[upstream_mask].max()
        downstream_mask = gtf_starts > ee
        if downstream_mask.any():
            dn_intron[i] = gtf_starts[downstream_mask].min() - ee
    # 3 aggregations across the two flanking introns
    min_intron = np.fmin(up_intron, dn_intron)
    max_intron = np.fmax(up_intron, dn_intron)
    mean_intron = (up_intron + dn_intron) / 2
    print(f"\n{dataset_names[name]} flanking intron lengths:")
    for lbl, arr in [("upstream", up_intron), ("downstream", dn_intron),
                     ("min", min_intron), ("max", max_intron), ("mean", mean_intron)]:
        print(f"  {lbl}: median={np.nanmedian(arr):.0f}, mean={np.nanmean(arr):.0f}, "
              f"range=[{np.nanmin(arr):.0f}, {np.nanmax(arr):.0f}]")

    # one figure per aggregation, with one panel per model
    for agg_label, agg_arr, agg_tag in [
        ("Min flanking intron", min_intron, "min"),
        ("Max flanking intron", max_intron, "max"),
        ("Mean flanking intron", mean_intron, "mean"),
    ]:
        fin_agg = np.isfinite(agg_arr)
        fig, axes = plt.subplots(1, len(mods),
                                 figsize=_ru.journal_size(max(min_width, width_per_panel * len(mods)), height),
                                 sharey=True, squeeze=False, constrained_layout=True)
        axes = axes[0]
        for ax, m in zip(axes, mods):
            abs_err = np.abs(df[MODEL_COLS[m]].values - y_true)
            fin = np.isfinite(abs_err) & fin_agg
            x_val = agg_arr[fin]
            y_val = abs_err[fin]
            mcolor = (palette or {}).get(m, get_color(m))
            ax.scatter(x_val, y_val, s=scatter_size, alpha=scatter_alpha, color=mcolor, edgecolors="none", rasterized=True)
            # percentile-binned mean of |err| at each x bin → trend line through the cloud
            pct = np.percentile(x_val, np.linspace(0, 100, n_percentile_bins))
            pct = np.unique(pct)
            bi = np.digitize(x_val, pct)
            cx, cy = [], []
            for b in range(1, len(pct)):
                bm = bi == b
                if bm.sum() >= min_bin_count:
                    cx.append((pct[b-1] + pct[b]) / 2)
                    cy.append(np.mean(y_val[bm]))
            ax.plot(cx, cy, color=binned_line_color, lw=binned_line_width, zorder=binned_line_z)
            ax.set_xlabel(f"{agg_label} length (bp)", fontsize=label_fontsize)
            ax.set_title(model_names[m], fontsize=title_fontsize)
            if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
            ax.grid(alpha=grid_alpha)
        axes[0].set_ylabel("|Prediction error|", fontsize=label_fontsize)
        _save(fig3_sup, f"{name}_error_vs_{agg_tag}_intron_length", tight=False)


_GNOMAD_DATASETS = ("vexseq", "mfass")  # hg19-coord with chrom/pos/ref/alt; compass=hg38, opensplice=variant_id
_PAPER_DATASETS = ("vexseq", "mfass")   # main paper figs; compass/opensplice are table-only


def _paper_pairs(d):
    """ordered (name, df) for the paper-figure datasets (vexseq, mfass)"""
    return [(n, d.dfs[n]) for n in _PAPER_DATASETS if n in d.dfs]


def _annotate_gnomad(d: Data) -> None:
    """add gnomAD AF column to vex/mfass dfs (cache to fig3_sup)"""
    gnomad_path = os.environ.get("SPLAIRE_GNOMAD_VCF", "/scratch/runyan.m/gnomad.exomes.r2.1.1.sites.vcf.bgz")
    _gnomad_cache = fig3_sup / "_gnomad_af_cache.pkl"
    target_dfs = {name: d.dfs[name] for name in _GNOMAD_DATASETS if name in d.dfs}
    if _gnomad_cache.exists():
        with open(_gnomad_cache, "rb") as f:
            _cached = pickle.load(f)
        for name, df in target_dfs.items():
            if name in _cached and len(_cached[name]) == len(df):
                df["gnomad_af"] = _cached[name]
        print(f"loaded gnomAD AF from {_gnomad_cache}")

    needs_lookup = [name for name, df in target_dfs.items() if "gnomad_af" not in df.columns]
    if needs_lookup:
        vcf = pysam.VariantFile(gnomad_path)
        for name in needs_lookup:
            df = d.dfs[name]
            chrom = df["chrom"].values.astype(str)
            pos = df["pos"].values.astype(int)
            ref_al = df["ref"].values.astype(str)
            alt_al = df["alt"].values.astype(str)
            mafs = np.full(len(df), np.nan)
            for i in tqdm(range(len(df)), desc=f"{dataset_names.get(name, name)} gnomAD lookup"):
                c = chrom[i].replace("chr", "")
                p = int(pos[i])
                r, a = ref_al[i], alt_al[i]
                try:
                    for rec in vcf.fetch(c, p - 1, p):
                        if rec.pos == p and rec.ref == r and a in (rec.alts or []):
                            af = rec.info.get("AF")
                            if af is not None:
                                ai = list(rec.alts).index(a)
                                mafs[i] = af[ai] if isinstance(af, tuple) else af
                            break
                except (ValueError, KeyError):
                    pass
            df["gnomad_af"] = mafs
            n_found = np.isfinite(mafs).sum()
            print(f"{dataset_names.get(name, name)}: {n_found:,} / {len(df):,} in gnomAD")
        vcf.close()
        _to_cache = {name: df["gnomad_af"].values for name, df in target_dfs.items()
                     if "gnomad_af" in df.columns}
        with open(_gnomad_cache, "wb") as f:
            pickle.dump(_to_cache, f)
        print(f"saved gnomAD AF cache to {_gnomad_cache}")


def fig_error_vs_maf(d: Data, name: str,
                     # ---- figure sizing (one panel per model) ----
                     height=4.5,
                     width_per_panel=3,
                     min_width=12,
                     # ---- fonts ----
                     title_fontsize=None,
                     label_fontsize=None,
                     tick_fontsize=None,
                     # ---- scatter ----
                     scatter_size=6,
                     scatter_alpha=0.2,
                     palette=None,                     # {model: color}
                     # ---- binned-mean trend line ----
                     n_percentile_bins=11,
                     min_bin_count=3,
                     binned_line_color="black",
                     binned_line_width=2,
                     binned_line_z=5,
                     # ---- grid ----
                     grid_alpha=0.2,
                     ) -> None:
    """error vs log10(MAF), scatter per model + binned median trend"""
    df, y_true, mods = _fig_setup(d, name)
    if "gnomad_af" not in df.columns:
        _ZENODO_SKIPPED.add("fig_error_vs_maf")
        print(f"skipping fig_error_vs_maf ({name}): no gnomAD AF column "
              f"(call _annotate_gnomad(d) first; needs gnomAD VCF or cached pkl)")
        return
    af = df["gnomad_af"].values
    has_af = np.isfinite(af)
    if has_af.sum() <= 50:
        return
    fig, axes = plt.subplots(1, len(mods),
                             figsize=_ru.journal_size(max(min_width, width_per_panel * len(mods)), height),
                             sharey=True, squeeze=False, constrained_layout=True)
    axes = axes[0]
    for ax, m in zip(axes, mods):
        abs_err = np.abs(df[MODEL_COLS[m]].values - y_true)
        fin = has_af & np.isfinite(abs_err)
        log_af = np.log10(af[fin].clip(1e-7))                  # clip tiny AFs so log doesn't blow up
        err_fin = abs_err[fin]
        mcolor = (palette or {}).get(m, get_color(m))
        ax.scatter(log_af, err_fin, s=scatter_size, alpha=scatter_alpha, color=mcolor, edgecolors="none", rasterized=True)
        pct = np.percentile(log_af, np.linspace(0, 100, n_percentile_bins))
        bi = np.digitize(log_af, pct)
        cx, cy = [], []
        for b in range(1, len(pct)):
            bm = bi == b
            if bm.sum() >= min_bin_count:
                cx.append((pct[b-1] + pct[b]) / 2)
                cy.append(np.mean(err_fin[bm]))
        ax.plot(cx, cy, color=binned_line_color, lw=binned_line_width, zorder=binned_line_z)
        ax.set_xlabel("log₁₀(gnomAD AF)", fontsize=label_fontsize)
        ax.set_title(model_names[m], fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.grid(alpha=grid_alpha)
    axes[0].set_ylabel("|Prediction error|", fontsize=label_fontsize)
    _save(fig3_sup, f"{name}_error_vs_maf", tight=False)


def fig_effect_vs_maf(d: Data, name: str,
                      # ---- figure sizing ----
                      height=5,
                      # ---- fonts ----
                      title_fontsize=None,
                      label_fontsize=None,
                      tick_fontsize=None,
                      legend_fontsize=None,
                      # ---- scatter (raw cloud) ----
                      scatter_size=6,
                      scatter_alpha=0.2,
                      scatter_color="#666666",            # neutral gray
                      # ---- binned-median trend line ----
                      n_percentile_bins=16,
                      min_bin_count=3,
                      median_line_color="black",
                      median_line_width=2.5,
                      median_line_z=5,
                      # ---- legend + grid ----
                      legend_loc="best",
                      grid_alpha=0.2,
                      ) -> None:
    """|measured ΔPSI| vs log10(MAF)"""
    df = d.dfs[name]
    if "gnomad_af" not in df.columns:
        _ZENODO_SKIPPED.add("fig_effect_vs_maf")
        print(f"skipping fig_effect_vs_maf ({name}): no gnomAD AF column")
        return
    af = df["gnomad_af"].values
    y_true = df["y"].values
    has_af = np.isfinite(af)
    if has_af.sum() <= 50:
        return
    fig, ax = plt.subplots(figsize=_ru.journal_size(6, height), constrained_layout=True)
    fin = has_af & np.isfinite(y_true)
    log_af = np.log10(af[fin].clip(1e-7))
    abs_y = np.abs(y_true[fin])
    ax.scatter(log_af, abs_y, s=scatter_size, alpha=scatter_alpha, color=scatter_color, edgecolors="none", rasterized=True)
    # binned median overlay across the log AF range
    pct = np.percentile(log_af, np.linspace(0, 100, n_percentile_bins))
    pct = np.unique(pct)
    bi = np.digitize(log_af, pct)
    cx, cy = [], []
    for b in range(1, len(pct)):
        bm = bi == b
        if bm.sum() >= min_bin_count:
            cx.append((pct[b-1] + pct[b]) / 2)
            cy.append(np.median(abs_y[bm]))
    ax.plot(cx, cy, color=median_line_color, lw=median_line_width, zorder=median_line_z, label="median")
    ax.set_xlabel("log₁₀(gnomAD AF)", fontsize=label_fontsize)
    ax.set_ylabel("|Measured ΔPSI|", fontsize=label_fontsize)
    ax.set_title(f"{dataset_names[name]} — Effect size vs allele frequency", fontsize=title_fontsize)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    ax.legend(loc=legend_loc, frameon=False, fontsize=legend_fontsize)
    ax.grid(alpha=grid_alpha)
    _save(fig3_sup, f"{name}_effect_vs_maf", tight=False)
    rho = stats.spearmanr(log_af, abs_y)[0]
    print(f"  {dataset_names[name]} Spearman(log10 AF, |ΔPSI|) = {rho:.3f}")


def fig_reporter_full_grouped_bars(d: Data,
                                   # ---- figure sizing ----
                                   panel_width_per_model=3.5,        # scatter panel width per model
                                   panel_height_per_dataset=7,       # total vertical inches per dataset
                                   scatter_to_bar_ratio=(1, 0.7),    # height ratio: scatter row vs bar row
                                   hspace=0.70,                      # vertical gap between dataset rows
                                   sc_wspace=0.3,                    # gap between scatter panels
                                   bar_wspace=0.3,                   # gap between bar panels
                                   # ---- fonts ----
                                   title_fontsize=None,              # model name above each scatter
                                   label_fontsize=None,
                                   tick_fontsize=None,
                                   dataset_label_scale=1.4,          # multiplier on rcParams font.size for "Vex-seq" / "MFASS" header
                                   legend_scale=1.1,                 # multiplier on rcParams font.size for shared legend
                                   r2_fontsize=None,                 # R² text in scatter; None = ANNOT_SIZE
                                   sig_star_fontsize=7,              # significance star text above bars
                                   colorbar_label_fontsize=9,
                                   # ---- scatter ----
                                   scatter_size=12,
                                   scatter_alpha=0.8,
                                   scatter_cmap="viridis",
                                   # ---- zero/baseline lines on scatter panels ----
                                   ref_line_color="gray",
                                   ref_line_style="--",
                                   ref_line_width=0.5,
                                   ref_line_alpha=0.5,
                                   # ---- bar styling ----
                                   bar_edge="black",
                                   bar_edge_width=0.5,
                                   palette=None,
                                   # ---- grid + dataset header offset ----
                                   grid_alpha=0.3,
                                   ds_header_offset=0.035,           # y offset above scatter row for "Vex-seq"/"MFASS"
                                   ) -> None:
    """main figure: scatter (log density) + grouped bars per location"""
    _cols_af = MODEL_COLS
    _models_af = MODEL_LIST
    _metrics_af = ["Pearson", "Spearman", "AUPRC"]
    paper_pairs = _paper_pairs(d)
    all_mods_af = set()
    for name, df in paper_pairs:
        all_mods_af.update(m for m in _models_af if _cols_af[m] in df.columns)
    mods_af = [m for m in _models_af if m in all_mods_af]
    nm_af = len(mods_af)
    _scatter_ticks = [-1, -0.5, 0, 0.5, 1]
    n_datasets = len(paper_pairs)

    _r2_fs = r2_fontsize if r2_fontsize is not None else ANNOT_SIZE
    # don't clamp to journal full_width — 6 scatter + 5 bar panels per row needs real width
    fig = plt.figure(figsize=(panel_width_per_model * nm_af, panel_height_per_dataset * n_datasets))
    gs = fig.add_gridspec(n_datasets * 2, 1, height_ratios=list(scatter_to_bar_ratio) * n_datasets, hspace=hspace)
    _sc_axes_af, _bar_axes_af, _sc_refs_af = {}, {}, {}

    for d_idx, (name, df) in enumerate(paper_pairs):
        y, label = df["y"].values, df["label"].values
        pos = df["pos"].values
        loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)
        gs_sc = gs[d_idx * 2].subgridspec(1, nm_af, wspace=sc_wspace)
        scatter_data = {}
        z_min, z_max = float('inf'), float('-inf')
        for m in mods_af:
            if _cols_af[m] not in df.columns: continue
            xm, ym, z_log, order = _get_kde(name, m, y, df[_cols_af[m]].values)
            scatter_data[m] = (xm, ym, z_log, order)
            z_min = min(z_min, z_log.min())
            z_max = max(z_max, z_log.max())
        sc_ref = None
        sc_axes = []
        for j, m in enumerate(mods_af):
            ax = fig.add_subplot(gs_sc[0, j])
            sc_axes.append(ax)
            if m not in scatter_data:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
                continue
            xm, ym, z, order = scatter_data[m]
            sc = ax.scatter(xm[order], ym[order], c=z[order], cmap=scatter_cmap,
                            s=scatter_size, alpha=scatter_alpha, edgecolors="none", vmin=z_min, vmax=z_max)
            sc_ref = sc
            r2 = 1 - np.sum((xm - ym) ** 2) / np.sum((xm - xm.mean()) ** 2)
            ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, fontsize=_r2_fs, va="top", ha="left")
            ax.axhline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
            ax.axvline(0, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
            ax.set_xticks(_scatter_ticks); ax.set_yticks(_scatter_ticks)
            if j == 0: ax.set_ylabel("Predicted ΔPSI", fontsize=label_fontsize)
            else: ax.set_ylabel("")
            ax.set_xlabel("Measured ΔPSI", fontsize=label_fontsize)
            ax.set_title(model_names[m], fontsize=title_fontsize)
            if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        _sc_axes_af[d_idx] = sc_axes
        _sc_refs_af[d_idx] = sc_ref

        gs_bar = gs[d_idx * 2 + 1].subgridspec(1, len(LOCATION_SUBSETS), wspace=bar_wspace)
        b_axes = []
        for si, subset_name in enumerate(LOCATION_SUBSETS):
            mask = loc[subset_name]
            ax = fig.add_subplot(gs_bar[0, si])
            b_axes.append(ax)
            n_sub = mask.sum()
            models_avail = [m for m in mods_af if _cols_af[m] in df.columns]
            model_vals = {}
            for m in models_avail:
                pred = df[_cols_af[m]].values
                r, rho, auprc = get_metrics(y[mask], pred[mask], label[mask])
                model_vals[m] = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}
            n_models_h = len(models_avail)
            bar_w = 0.8 / n_models_h
            for mi, metric in enumerate(_metrics_af):
                best_m = max(models_avail, key=lambda m: model_vals[m][metric])
                # significance vs best model per metric (holm-corrected)
                boot_key = (name, "full", subset_name)
                vb_pvals = []
                if boot_key in d.boot_results:
                    for m in models_avail:
                        if m == best_m: continue
                        si_ = get_boot_sig(d.boot_results[boot_key], best_m, m, metric)
                        if si_: vb_pvals.append((m, si_["p"]))
                vb_corr = holm_correct(vb_pvals)
                for bi, m in enumerate(models_avail):
                    x = mi + (bi - n_models_h / 2 + 0.5) * bar_w
                    val = model_vals[m][metric]
                    mcolor = (palette or {}).get(m, get_color(m))
                    ax.bar(x, val, bar_w, color=mcolor, edgecolor=bar_edge, linewidth=bar_edge_width)
                    if m != best_m and m in vb_corr:
                        p_adj, sig = vb_corr[m]
                        if sig:
                            ax.text(x, val + 0.01, sig, ha="center", va="bottom",
                                    fontsize=sig_star_fontsize, fontweight="bold")
            ax.set_xticks(range(len(_metrics_af)))
            ax.set_xticklabels(_metrics_af)
            ax.set_ylim(0, 1.0); ax.set_ylabel("Value", fontsize=label_fontsize)
            ax.set_title(f"{LOCATION_LABELS[subset_name]} ({n_sub:,})", fontsize=title_fontsize)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=grid_alpha)
            if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        _bar_axes_af[d_idx] = b_axes
    fig.canvas.draw()

    for d_idx, (name, df) in enumerate(paper_pairs):
        sc_axes = _sc_axes_af[d_idx]
        b_axes = _bar_axes_af[d_idx]
        bbox_l, bbox_r = sc_axes[0].get_position(), sc_axes[-1].get_position()
        fig.text((bbox_l.x0 + bbox_r.x1) / 2, bbox_l.y1 + ds_header_offset,
                 dataset_names[name], fontsize=plt.rcParams["font.size"] * dataset_label_scale, va="bottom", ha="center")
        # panel labels: vex-seq -> a (scatter), b (bars); mfass -> c (scatter), d (bars)
        sc_lab, bar_lab = ("a", "b") if d_idx == 0 else ("c", "d")
        fig.text(bbox_l.x0 - 0.015, bbox_l.y1 + 0.005, sc_lab, **_ru.panel_kw)
        bbox_b = b_axes[0].get_position()
        fig.text(bbox_b.x0 - 0.015, bbox_b.y1 + 0.005, bar_lab, **_ru.panel_kw)
        if _sc_refs_af[d_idx] is not None:
            cax = fig.add_axes([bbox_r.x1 + 0.01, bbox_r.y0, 0.01, bbox_r.height])
            cb = fig.colorbar(_sc_refs_af[d_idx], cax=cax)
            cb.set_ticks([])
            cax.set_ylabel("log density", fontsize=colorbar_label_fontsize, rotation=270, labelpad=12)
    _lh = [Patch(facecolor=(palette or {}).get(m, get_color(m)), edgecolor=bar_edge, label=model_names[m]) for m in mods_af]
    fig.legend(handles=_lh, loc="upper center", ncol=len(mods_af), frameon=False,
               prop={"size": plt.rcParams["font.size"] * legend_scale}, bbox_to_anchor=(0.5, 0.999))
    _save(fig3_main, "figure4", tight=False)
    print(f"saved {fig3_main}/figure4.png")


def fig_correction_sensitivity(d: Data,
                               # ---- figure sizing ----
                               height_per_pair=0.6,                # rows scale with n_pairs (15 for 6 models)
                               min_height=9,                        # floor on total height
                               height_pad=3,                        # extra height beyond pair-scaled rows
                               # ---- fonts ----
                               col_title_fontsize=9,                # top-row column titles
                               row_label_fontsize=10,               # right-side metric label
                               pair_label_fontsize=7,               # left y-axis pair names
                               context_tick_fontsize=6,             # bottom-row context tick labels
                               context_group_fontsize=7,            # "V.full" / "M.filt" group labels under tick row
                               legend_fontsize=9,
                               # ---- heatmap colors (ns / * / ** / ***) ----
                               cell_colors=("#e8e8e8", "#b2dfdb", "#4db6ac", "#00695c"),
                               # ---- diff highlight (box around cells that differ from holm k=4) ----
                               diff_edge_color="black",
                               diff_edge_width=1.5,
                               # ---- inter-group vertical separator lines ----
                               group_sep_color="white",
                               group_sep_width=2,
                               ) -> None:
    """correction comparison heatmap across all bootstrap contexts"""
    _cs_models = MODEL_LIST
    _cs_pairs = list(combinations(_cs_models, 2))
    _cs_short = {"pangolin": "Pang", "pangolin_v2": "Pang-v2", "spliceai": "SA",
                 "splicetransformer": "ST", "splaire_ref": "SPL", "splaire_var": "SPL-v"}
    _cs_pair_labels = [f"{_cs_short[a]} vs {_cs_short[b]}" for a, b in _cs_pairs]
    _cs_contexts = [
        (ds, filt, loc)
        for ds in ["vexseq", "mfass"]
        for filt in ["full", "filtered"]
        for loc in LOCATION_SUBSETS
    ]
    _cs_metrics = ["Pearson", "Spearman", "AUPRC"]
    _cs_corrections = [
        ("Holm k=4\n(standard)", d.boot_results, "sig_holm_k4"),
        ("Holm k=10\n(standard)", d.boot_results, "sig_holm_k10"),
        ("Bonf k=4\n(standard)", d.boot_results, "sig_bonf_k4"),
        ("Holm k=10\n(cluster)", d.cboot_results, "sig_holm_k10"),
    ]
    _cs_mats = {}
    for mi, metric in enumerate(_cs_metrics):
        for ci, (_, source, sig_key) in enumerate(_cs_corrections):
            mat = np.zeros((len(_cs_pairs), len(_cs_contexts)), dtype=int)
            for pi, pair in enumerate(_cs_pairs):
                for xi, ctx in enumerate(_cs_contexts):
                    mat[pi, xi] = _cs_get_sig(source, ctx, pair, metric, sig_key)
            _cs_mats[(mi, ci)] = mat
    _cs_col_diffs = {}
    for ci in range(1, 4):
        _cs_col_diffs[ci] = sum(
            np.sum(_cs_mats[(mi, ci)] != _cs_mats[(mi, 0)]) for mi in range(3)
        )
    _cs_cmap = ListedColormap(list(cell_colors))
    _cs_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _cs_cmap.N)
    fig, axes = plt.subplots(3, 4, figsize=_ru.journal_size(16, max(min_height, len(_cs_pairs) * height_per_pair + height_pad)), constrained_layout=True)
    for mi, metric in enumerate(_cs_metrics):
        ref = _cs_mats[(mi, 0)]
        for ci, (col_title, _, _) in enumerate(_cs_corrections):
            ax = axes[mi, ci]
            mat = _cs_mats[(mi, ci)]
            ax.imshow(mat, cmap=_cs_cmap, norm=_cs_norm, aspect="auto", interpolation="nearest")
            for pi in range(len(_cs_pairs)):
                for xi in range(len(_cs_contexts)):
                    if ci > 0 and mat[pi, xi] != ref[pi, xi]:
                        ax.add_patch(plt.Rectangle(
                            (xi - 0.5, pi - 0.5), 1, 1,
                            fill=False, edgecolor=diff_edge_color, linewidth=diff_edge_width
                        ))
            if mi == 0:
                title = col_title
                if ci > 0:
                    title += f"\n({_cs_col_diffs[ci]} differ)"
                ax.set_title(title, fontsize=col_title_fontsize)
            ax.set_yticks(range(len(_cs_pairs)))
            if ci == 0:
                ax.set_yticklabels(_cs_pair_labels, fontsize=pair_label_fontsize)
            else:
                ax.set_yticklabels([])
            _n_loc = len(LOCATION_SUBSETS)
            _n_ctx = len(_cs_contexts)
            ax.set_xticks(range(_n_ctx))
            if mi == 2:
                _loc_short = {"all": "a", "exon": "e", "intron": "i",
                              "splice_region": "r", "splice_site": "s"}
                _tick_labels = [_loc_short.get(loc, loc[0]) for _, _, loc in _cs_contexts]
                ax.set_xticklabels(_tick_labels, fontsize=context_tick_fontsize)
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                _groups = ["V.full", "V.filt", "M.full", "M.filt"]
                for gi, gl in enumerate(_groups):
                    gx = gi * _n_loc + (_n_loc - 1) / 2
                    ax.text(gx, -0.18, gl, ha="center", va="top", fontsize=context_group_fontsize,
                            transform=trans, clip_on=False)
            else:
                ax.set_xticklabels([])
            if ci == 3:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(metric, fontsize=row_label_fontsize, rotation=270, labelpad=15)
            # vertical separators between context groups (V.full | V.filt | M.full | M.filt)
            for gx in [3.5, 7.5, 11.5]:
                ax.axvline(gx, color=group_sep_color, linewidth=group_sep_width)
    _cs_legend = [
        mpatches.Patch(facecolor=cell_colors[0], edgecolor="gray", label="ns"),
        mpatches.Patch(facecolor=cell_colors[1], edgecolor="gray", label="*"),
        mpatches.Patch(facecolor=cell_colors[2], edgecolor="gray", label="**"),
        mpatches.Patch(facecolor=cell_colors[3], edgecolor="gray", label="***"),
        mpatches.Patch(facecolor="none", edgecolor=diff_edge_color, linewidth=diff_edge_width,
                       label="differs from Holm k=4"),
    ]
    fig.legend(handles=_cs_legend, loc="outside lower center", ncol=5, fontsize=legend_fontsize,
               frameon=False)
    _save(fig3_sup, "correction_sensitivity", tight=False)
    for ci, (title, _, _) in enumerate(_cs_corrections):
        if ci == 0:
            continue
        print(f"{title.replace(chr(10), ' ')}: {_cs_col_diffs[ci]} cells differ from Holm k=4")


def _inclusion_scatter_panels(df, y_target, cols_map, x_label, y_label,
                              # ---- panel sizing ----
                              panel_width=5,                     # inches per panel column (input to journal_size)
                              panel_height=4.5,                  # inches per panel row
                              max_cols=3,                        # wrap to a new row after this many panels
                              # ---- fonts ----
                              title_fontsize=None,
                              label_fontsize=None,
                              tick_fontsize=None,
                              annot_fontsize=9,                  # r/rho/R2/n stats box
                              # ---- scatter (kde-colored) ----
                              scatter_size=3,
                              scatter_alpha=0.6,
                              scatter_alpha_fallback=0.3,        # used when kde fails
                              cmap="viridis",
                              # ---- stats annotation box ----
                              annot_bbox_alpha=0.8,
                              ):
    """grid of per-model scatter (predicted vs measured inclusion), kde-colored"""
    avail = [m for m in MODEL_LIST if cols_map[m] in df.columns]
    n_mod = len(avail)
    if n_mod == 0:
        return False
    ncols = min(n_mod, max_cols)
    nrows = (n_mod + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=_ru.journal_size(panel_width * ncols, panel_height * nrows),
                             squeeze=False, constrained_layout=True)
    for i, m in enumerate(avail):
        ax = axes[i // ncols, i % ncols]
        pred = df[cols_map[m]].values
        mask = np.isfinite(y_target) & np.isfinite(pred)
        x, y_plot = pred[mask], y_target[mask]
        # kde-color the density; fall back to flat scatter if kde fails (eg degenerate points)
        try:
            xy = np.vstack([x, y_plot])
            z = gaussian_kde(xy)(xy)
            order = z.argsort()
            ax.scatter(x[order], y_plot[order], c=np.log10(z[order]), s=scatter_size, alpha=scatter_alpha, cmap=cmap, rasterized=True)
        except Exception:
            ax.scatter(x, y_plot, s=scatter_size, alpha=scatter_alpha_fallback, color=get_color(m), rasterized=True)
        r, rho, r2, n = compute_corrs(y_target, pred)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(model_names.get(m, m), fontsize=title_fontsize)
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.text(0.03, 0.97, f"$r$={r:.3f}\n$\\rho$={rho:.3f}\n$R^2$={r2:.3f}\n$n$={n:,}",
                transform=ax.transAxes, fontsize=annot_fontsize, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=annot_bbox_alpha))
    # hide leftover empty subplots in the last row
    for i in range(n_mod, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    return True


def fig_ref_inclusion_scatter(d: Data, name: str,
                              # all styling forwarded to _inclusion_scatter_panels — see that helper for defaults
                              panel_width=5,
                              panel_height=4.5,
                              max_cols=3,
                              title_fontsize=None,
                              label_fontsize=None,
                              tick_fontsize=None,
                              annot_fontsize=9,
                              scatter_size=3,
                              scatter_alpha=0.6,
                              scatter_alpha_fallback=0.3,
                              cmap="viridis",
                              annot_bbox_alpha=0.8,
                              ) -> None:
    """ref prediction vs measured ref inclusion scatter per model"""
    df = d.dfs[name]
    if name == "vexseq":
        y_ref, y_label = df["measured_ref_psi"].values, "Measured ref PSI"
    elif name == "mfass":
        y_ref, y_label = df["measured_ref_inclusion"].values, "Measured ref inclusion"
    else:
        return
    if not _inclusion_scatter_panels(df, y_ref, REF_COLS, "Predicted ref score", y_label,
                                     panel_width=panel_width, panel_height=panel_height, max_cols=max_cols,
                                     title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
                                     annot_fontsize=annot_fontsize, scatter_size=scatter_size,
                                     scatter_alpha=scatter_alpha, scatter_alpha_fallback=scatter_alpha_fallback,
                                     cmap=cmap, annot_bbox_alpha=annot_bbox_alpha):
        return
    _save(out_dir, f"{name}_ref_inclusion_scatter", tight=False)


def fig_exon_ref_psi(d: Data,
                     # ---- figure sizing ----
                     height=4.5,
                     # ---- fonts ----
                     title_fontsize=None,                  # "Vex-seq (n=...)" / "MFASS (n=...)"
                     label_fontsize=None,
                     tick_fontsize=None,
                     annot_fontsize=9,                     # stats text (median/mean/SD)
                     # ---- histogram styling ----
                     hist_bins=30,
                     hist_color="#4a90d9",                 # bar fill (default blue)
                     hist_edge="white",
                     hist_edge_width=0.5,
                     # ---- stats text box ----
                     annot_bbox_alpha=0.85,                # opacity of white bg behind stats
                     ) -> None:
    """per-exon ref PSI histogram (vex + mfass)"""
    _ref_specs = [
        ("vexseq", "measured_ref_psi", "Vex-seq", "Ref PSI (HepG2)"),
        ("mfass", "measured_ref_inclusion", "MFASS", "Ref inclusion (nat_v2_index)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=_ru.journal_size(10, height), constrained_layout=True)
    # one panel per dataset; each is a histogram of per-exon ref PSI
    for ax, (name, col, ds_label, x_label) in zip(axes, _ref_specs):
        df_ = d.dfs[name]
        if col not in df_.columns:
            ax.text(0.5, 0.5, f"{col} missing", ha="center", va="center", transform=ax.transAxes)
            continue
        # one row per unique exon — collapse the per-variant frame
        exon_ref = (df_[["chrom", "exon_start", "exon_end", col]]
                    .dropna(subset=[col])
                    .groupby(["chrom", "exon_start", "exon_end"], as_index=False)
                    .first())
        vals = exon_ref[col].values
        ax.hist(vals, bins=hist_bins, range=(0, 1), edgecolor=hist_edge, linewidth=hist_edge_width, color=hist_color)
        ax.set_xlim(0, 1)
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel("Exons", fontsize=label_fontsize)
        ax.set_title(f"{ds_label} ($n$ = {len(vals):,} exons)", fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        stats_txt = (f"median={np.median(vals):.3f}\n"
                     f"mean={np.mean(vals):.3f}\n"
                     f"SD={np.std(vals):.3f}")
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=annot_fontsize,
                va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=annot_bbox_alpha))
    _save(out_dir, "exon_ref_psi", tight=False)


def fig_mfass_alt_inclusion_scatter(d: Data,
                                    # forwarded to _inclusion_scatter_panels
                                    panel_width=5,
                                    panel_height=4.5,
                                    max_cols=3,
                                    title_fontsize=None,
                                    label_fontsize=None,
                                    tick_fontsize=None,
                                    annot_fontsize=9,
                                    scatter_size=3,
                                    scatter_alpha=0.6,
                                    scatter_alpha_fallback=0.3,
                                    cmap="viridis",
                                    annot_bbox_alpha=0.8,
                                    ) -> None:
    """alt prediction vs measured alt inclusion (mfass only)"""
    df = d.dfs["mfass"]
    y_alt = df["measured_alt_inclusion"].values
    if not _inclusion_scatter_panels(df, y_alt, ALT_COLS, "Predicted alt score", "Measured alt inclusion",
                                     panel_width=panel_width, panel_height=panel_height, max_cols=max_cols,
                                     title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize,
                                     annot_fontsize=annot_fontsize, scatter_size=scatter_size,
                                     scatter_alpha=scatter_alpha, scatter_alpha_fallback=scatter_alpha_fallback,
                                     cmap=cmap, annot_bbox_alpha=annot_bbox_alpha):
        return
    _save(out_dir, "mfass_alt_inclusion_scatter", tight=False)


def fig_mfass_replicate_consistency(d: Data,
                                    # ---- figure sizing ----
                                    height=5,
                                    # ---- fonts ----
                                    title_fontsize=None,            # "Pearson" / "Spearman" panel titles
                                    label_fontsize=None,            # y-axis label per panel
                                    xtick_fontsize=11,              # group labels under bars (v2 main / R1 / R2 / v1)
                                    tick_fontsize=None,
                                    legend_fontsize=9,              # color legend at top
                                    # ---- bar styling ----
                                    bar_edge="black",
                                    bar_edge_width=0.5,
                                    palette=None,                   # {model: color}
                                    # ---- legend ----
                                    legend_loc="outside upper center",
                                    ) -> None:
    """replicate consistency bar chart (mfass)"""
    df = d.dfs["mfass"]
    # collect every available replicate-target column; main is v2 mean of R1/R2
    _rep_targets_bar = [
        ("v2 (main)", df["y"].values),
        ("v2 R1", df["v2_dpsi_R1"].values if "v2_dpsi_R1" in df.columns else None),
        ("v2 R2", df["v2_dpsi_R2"].values if "v2_dpsi_R2" in df.columns else None),
        ("v1", df["v1_dpsi"].values if "v1_dpsi" in df.columns else None),
    ]
    _rep_targets_bar = [(n, y) for n, y in _rep_targets_bar if y is not None]
    fig, axes = plt.subplots(1, 2, figsize=_ru.journal_size(14, height), constrained_layout=True)
    # one panel per metric; bars grouped by target with one bar per model
    for mi, metric in enumerate(["Pearson", "Spearman"]):
        ax = axes[mi]
        for ti, (tname, y_target) in enumerate(_rep_targets_bar):
            for bi, m in enumerate(MODEL_LIST):
                col = MODEL_COLS.get(m)
                if col not in df.columns:
                    continue
                pred = df[col].values
                mask = np.isfinite(y_target) & np.isfinite(pred)
                yt, yp = y_target[mask], pred[mask]
                if len(yt) < 3:
                    continue
                val = stats.pearsonr(yt, yp)[0] if metric == "Pearson" else stats.spearmanr(yt, yp)[0]
                x_pos = ti * (len(MODEL_LIST) + 1) + bi
                mcolor = (palette or {}).get(m, get_color(m))
                ax.bar(x_pos, val, color=mcolor, edgecolor=bar_edge, linewidth=bar_edge_width)
        n_m = len(MODEL_LIST)
        group_centers = [ti * (n_m + 1) + (n_m - 1) / 2 for ti in range(len(_rep_targets_bar))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([t[0] for t in _rep_targets_bar], fontsize=xtick_fontsize)
        ax.set_ylabel(metric, fontsize=label_fontsize)
        ax.set_title(metric, fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(axis="y", labelsize=tick_fontsize)
    _leg = [Patch(facecolor=(palette or {}).get(m, get_color(m)), edgecolor=bar_edge,
                  label=model_names.get(m, m)) for m in MODEL_LIST]
    fig.legend(handles=_leg, loc=legend_loc, ncol=len(MODEL_LIST), fontsize=legend_fontsize, frameon=False)
    _save(out_dir, "mfass_replicate_consistency", tight=False)


def fig_error_vs_baseline(d: Data, name: str,
                          # ---- per-panel sizing ----
                          panel_width=5,
                          panel_height=4.5,
                          max_cols=3,
                          # ---- fonts ----
                          title_fontsize=None,
                          label_fontsize=None,
                          tick_fontsize=None,
                          annot_fontsize=9,
                          # ---- scatter ----
                          scatter_size=3,
                          scatter_alpha=0.2,
                          palette=None,                       # {model: color}
                          # ---- binned-mean overlay (black line) ----
                          n_bins=20,                          # linearly-spaced bins along x
                          binned_line_color="black",
                          binned_line_style="-",
                          binned_line_width=2,
                          # ---- stats text box ----
                          annot_bbox_alpha=0.8,
                          ) -> None:
    """|prediction error| vs ref inclusion baseline, per model"""
    df = d.dfs[name]
    # both datasets carry a ref inclusion column but under different names
    if name == "vexseq":
        baseline = df["measured_ref_psi"].values
        bl_label = "Ref PSI"
    elif name == "mfass":
        baseline = df["measured_ref_inclusion"].values
        bl_label = "Ref inclusion (nat_v2_index)"
    else:
        return
    avail = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]
    if not avail:
        return
    ncols = min(len(avail), max_cols)
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=_ru.journal_size(panel_width * ncols, panel_height * nrows),
                             squeeze=False, constrained_layout=True)
    y = df["y"].values
    for i, m in enumerate(avail):
        ax = axes[i // ncols, i % ncols]
        pred = df[MODEL_COLS[m]].values
        err = np.abs(pred - y)
        mask = np.isfinite(baseline) & np.isfinite(err)
        bm, em = baseline[mask], err[mask]
        mcolor = (palette or {}).get(m, get_color(m))
        ax.scatter(bm, em, s=scatter_size, alpha=scatter_alpha, color=mcolor, rasterized=True)
        # linearly-spaced bins → per-bin mean of |err| as a trend line
        bin_edges = np.linspace(np.nanmin(bm), np.nanmax(bm), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = np.array([
            np.nanmean(em[(bm >= bin_edges[j]) & (bm < bin_edges[j + 1])])
            for j in range(n_bins)
        ])
        valid_bins = np.isfinite(bin_means)
        ax.plot(bin_centers[valid_bins], bin_means[valid_bins],
                color=binned_line_color, ls=binned_line_style, lw=binned_line_width)
        rho = stats.spearmanr(bm, em)[0] if len(bm) > 2 else np.nan
        ax.set_title(model_names.get(m, m), fontsize=title_fontsize)
        ax.set_xlabel(bl_label, fontsize=label_fontsize)
        ax.set_ylabel("|Prediction error|", fontsize=label_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.text(0.03, 0.97, f"$\\rho$={rho:.3f}\n$n$={mask.sum():,}",
                transform=ax.transAxes, fontsize=annot_fontsize, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=annot_bbox_alpha))
    # hide unused subplots in trailing row
    for i in range(len(avail), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    _save(out_dir, f"{name}_error_vs_baseline", tight=False)


def fig_dpsi_perf_by_ref_psi(d: Data, name: str,
                             # ---- figure sizing ----
                             height=6.5,
                             height_ratios=(3, 1),               # main metric panels : n-per-bin panels
                             hspace=0.05,                        # vertical gap between top and bottom rows
                             # ---- fonts ----
                             title_fontsize=None,                # "Pearson r" / "Spearman ρ" / "MAE"
                             label_fontsize=None,
                             tick_fontsize=None,
                             legend_fontsize=7,                  # in-panel legend
                             # ---- binning ----
                             n_bins=10,                          # number of ref-PSI bins (0..1)
                             min_bin_count=5,                    # bins with fewer variants get NaN'd
                             # ---- per-model lines ----
                             model_line_width=1.5,
                             marker_style="o",
                             marker_size=5,
                             palette=None,                       # {model: color}
                             # ---- zero reference line on metric panels ----
                             ref_line_color="gray",
                             ref_line_width=0.5,
                             ref_line_alpha=0.5,
                             # ---- n-per-bin bar chart (bottom row) ----
                             n_bar_width=0.08,                   # bar width in x units
                             n_bar_color="#888",
                             n_bar_edge="black",
                             n_bar_edge_width=0.4,
                             # ---- grid + legend ----
                             grid_alpha=0.2,
                             legend_loc="best",
                             ) -> None:
    """binned ΔPSI prediction performance vs ref PSI"""
    df = d.dfs[name]
    if name == "vexseq":
        baseline = df["measured_ref_psi"].values
        bl_label = "Ref PSI"
    elif name == "mfass":
        baseline = df["measured_ref_inclusion"].values
        bl_label = "Ref inclusion"
    else:
        return
    avail = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]
    if not avail:
        return
    # uniform ref-PSI bins from 0..1
    _bin_edges = np.linspace(0, 1, n_bins + 1)
    _bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
    _bin_labels = [f"[{_bin_edges[i]:.1f},{_bin_edges[i+1]:.1f})" for i in range(n_bins)]
    y = df["y"].values
    rows = []
    for m in avail:
        pred = df[MODEL_COLS[m]].values
        for bi in range(n_bins):
            mask = (baseline >= _bin_edges[bi]) & (baseline < _bin_edges[bi + 1] if bi < n_bins - 1 else baseline <= _bin_edges[bi + 1])
            mask &= np.isfinite(baseline) & np.isfinite(y) & np.isfinite(pred)
            n = int(mask.sum())
            if n < min_bin_count:
                rows.append({"model": model_names.get(m, m), "bin": _bin_labels[bi],
                             "bin_center": _bin_centers[bi], "n": n,
                             "pearson": np.nan, "spearman": np.nan, "r2": np.nan, "mae": np.nan})
                continue
            yt, yp = y[mask], pred[mask]
            if np.std(yt) == 0 or np.std(yp) == 0:
                r, rho, r2 = np.nan, np.nan, np.nan
            else:
                r = stats.pearsonr(yt, yp)[0]
                rho = stats.spearmanr(yt, yp)[0]
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            mae = float(np.mean(np.abs(yt - yp)))
            rows.append({"model": model_names.get(m, m), "bin": _bin_labels[bi],
                         "bin_center": _bin_centers[bi], "n": n,
                         "pearson": r, "spearman": rho, "r2": r2, "mae": mae})
    bin_df = pd.DataFrame(rows)
    bin_df.to_csv(f"{out_dir}/{name}_dpsi_perf_by_ref_psi.csv", index=False, float_format="%.4f")
    print(f"saved {out_dir}/{name}_dpsi_perf_by_ref_psi.csv")

    # 2 rows × 3 cols — top row is metric lines, bottom row is per-bin n
    fig, axes = plt.subplots(2, 3, figsize=_ru.journal_size(15, height),
                              gridspec_kw={"height_ratios": list(height_ratios), "hspace": hspace})
    for ci, metric in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[0, ci]
        for m in avail:
            sub = bin_df[bin_df.model == model_names.get(m, m)].sort_values("bin_center")
            mcolor = (palette or {}).get(m, get_color(m))
            ax.plot(sub["bin_center"].values, sub[metric].values, marker=marker_style,
                    color=mcolor, lw=model_line_width, markersize=marker_size,
                    label=model_names.get(m, m))
        ax.set_title({"pearson": "Pearson $r$", "spearman": "Spearman $\\rho$", "mae": "MAE"}[metric],
                     fontsize=title_fontsize)
        ax.set_xlim(0, 1)
        ax.grid(alpha=grid_alpha)
        ax.tick_params(labelbottom=False)              # hide x labels on top row; bottom row carries them
        if tick_fontsize is not None: ax.tick_params(axis="y", labelsize=tick_fontsize)
        if ci == 0:
            ax.set_ylabel("metric", fontsize=label_fontsize)
        ax.axhline(0, color=ref_line_color, lw=ref_line_width, alpha=ref_line_alpha)
    axes[0, 2].legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)

    # bottom row: log-scale n-per-bin bars (same data shape for every model so just take one)
    n_per_bin = bin_df[bin_df.model == model_names.get(avail[0], avail[0])].sort_values("bin_center")["n"].values
    for ci in range(3):
        ax_n = axes[1, ci]
        ax_n.bar(_bin_centers, n_per_bin, width=n_bar_width, color=n_bar_color, edgecolor=n_bar_edge, linewidth=n_bar_edge_width)
        ax_n.set_xlim(0, 1)
        ax_n.set_xlabel(bl_label, fontsize=label_fontsize)
        if ci == 0:
            ax_n.set_ylabel("$n$", fontsize=label_fontsize)
        ax_n.set_yscale("log")
        if tick_fontsize is not None: ax_n.tick_params(labelsize=tick_fontsize)
    _save(out_dir, f"{name}_dpsi_perf_by_ref_psi", tight=False)


def fig_ref_alt_tables(d: Data) -> None:
    """save ref/alt correlation tables"""
    print("\n" + "=" * 60)
    print("CORRELATION TABLES")
    print("=" * 60)
    ref_corr_rows = []
    for name, df in d.dfs.items():
        if name == "vexseq":
            y_ref = df["measured_ref_psi"].values
        elif name == "mfass":
            y_ref = df["measured_ref_inclusion"].values
        else:
            continue
        for m in MODEL_LIST:
            col = REF_COLS.get(m)
            if col is None or col not in df.columns:
                continue
            pred = df[col].values
            r, rho, r2, n = compute_corrs(y_ref, pred)
            ref_corr_rows.append({"dataset": name, "model": model_names.get(m, m),
                                  "pearson": r, "spearman": rho, "r2": r2, "n": n})
    ref_corr_df = pd.DataFrame(ref_corr_rows)
    print("\nRef prediction vs measured ref inclusion:")
    print(ref_corr_df.to_string(index=False, float_format="%.3f"))

    alt_corr_rows = []
    df = d.dfs["mfass"]
    y_alt = df["measured_alt_inclusion"].values
    for m in MODEL_LIST:
        col = ALT_COLS.get(m)
        if col is None or col not in df.columns:
            continue
        pred = df[col].values
        r, rho, r2, n = compute_corrs(y_alt, pred)
        alt_corr_rows.append({"model": model_names.get(m, m),
                              "pearson": r, "spearman": rho, "r2": r2, "n": n})
    alt_corr_df = pd.DataFrame(alt_corr_rows)
    print("\nAlt prediction vs measured alt inclusion (MFASS):")
    print(alt_corr_df.to_string(index=False, float_format="%.3f"))

    rep_corr_rows = []
    df = d.dfs["mfass"]
    rep_targets = {
        "v2_dpsi (main)": df["y"].values,
        "v2_dpsi_R1": df["v2_dpsi_R1"].values if "v2_dpsi_R1" in df.columns else None,
        "v2_dpsi_R2": df["v2_dpsi_R2"].values if "v2_dpsi_R2" in df.columns else None,
        "v1_dpsi": df["v1_dpsi"].values if "v1_dpsi" in df.columns else None,
    }
    for target_name, y_target in rep_targets.items():
        if y_target is None:
            continue
        for m in MODEL_LIST:
            col = MODEL_COLS.get(m)
            if col is None or col not in df.columns:
                continue
            pred = df[col].values
            r, rho, r2, n = compute_corrs(y_target, pred)
            rep_corr_rows.append({"target": target_name, "model": model_names.get(m, m),
                                  "pearson": r, "spearman": rho, "n": n})
    rep_corr_df = pd.DataFrame(rep_corr_rows)
    print("\nDelta score vs replicate ΔPSI (MFASS):")
    print(rep_corr_df.to_string(index=False, float_format="%.3f"))

    ref_corr_df.to_csv(f"{out_dir}/ref_inclusion_corr.csv", index=False, float_format="%.4f")
    alt_corr_df.to_csv(f"{out_dir}/alt_inclusion_corr.csv", index=False, float_format="%.4f")
    rep_corr_df.to_csv(f"{out_dir}/replicate_corr.csv", index=False, float_format="%.4f")
    print(f"\nsaved correlation tables to {out_dir}/")


def fig_error_refpsi_spearman_table(d: Data) -> None:
    """Spearman ρ between |prediction error| and ref PSI per model"""
    err_corr_rows = []
    for name, df in d.dfs.items():
        if name == "vexseq":
            baseline = df["measured_ref_psi"].values
        elif name == "mfass":
            baseline = df["measured_ref_inclusion"].values
        else:
            continue
        y = df["y"].values
        for m in [m_ for m_ in MODEL_LIST if MODEL_COLS[m_] in df.columns]:
            pred = df[MODEL_COLS[m]].values
            err = np.abs(pred - y)
            mask = np.isfinite(baseline) & np.isfinite(err)
            if mask.sum() < 5:
                continue
            rho = stats.spearmanr(baseline[mask], err[mask])[0]
            err_corr_rows.append({"dataset": name, "model": model_names.get(m, m),
                                  "spearman_err_vs_refpsi": rho, "n": int(mask.sum())})
    err_corr_df = pd.DataFrame(err_corr_rows)
    err_corr_df.to_csv(f"{out_dir}/error_vs_refpsi_spearman.csv", index=False, float_format="%.4f")
    print(f"\nSpearman ρ (|prediction error| vs reference PSI):")
    print(err_corr_df.to_string(index=False, float_format="%.3f"))
    print(f"saved {out_dir}/error_vs_refpsi_spearman.csv")


# ---------------------------------------------------------------------------
# per-exon analysis
# ---------------------------------------------------------------------------

MIN_VARIANTS_PER_EXON = 5


# cache: id(df) -> (exon_df, mods); persists for the lifetime of the Data object
_per_exon_cache = {}


def _build_per_exon_df(df):
    """compute per-exon metrics, returns dataframe with ranked models"""
    _key = id(df)
    if _key in _per_exon_cache:
        return _per_exon_cache[_key]
    y = df["y"].values
    label = df["label"].values
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    exon_key = [f"{es}_{ee}" for es, ee in zip(exon_start, exon_end)]
    df["_exon_key"] = exon_key
    mods = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]
    exon_rows = []
    for ek, grp in df.groupby("_exon_key"):
        n_var = len(grp)
        if n_var < MIN_VARIANTS_PER_EXON:
            continue
        yt = grp["y"].values
        lab = grp["label"].values
        es, ee = int(ek.split("_")[0]), int(ek.split("_")[1])
        width = ee - es + 1
        row = {"exon_key": ek, "exon_start": es, "exon_end": ee,
               "width": width, "n_variants": n_var,
               "mean_abs_dpsi": np.nanmean(np.abs(yt)),
               "n_sdv": int(lab.sum())}
        for m in mods:
            pred = grp[MODEL_COLS[m]].values
            fin = np.isfinite(yt) & np.isfinite(pred)
            if fin.sum() < 3:
                row[f"{m}_pearson"] = np.nan
                row[f"{m}_spearman"] = np.nan
                row[f"{m}_mae"] = np.nan
                continue
            row[f"{m}_pearson"] = stats.pearsonr(yt[fin], pred[fin])[0]
            row[f"{m}_spearman"] = stats.spearmanr(yt[fin], pred[fin])[0]
            row[f"{m}_mae"] = np.nanmean(np.abs(pred[fin] - yt[fin]))
        exon_rows.append(row)
    exon_df = pd.DataFrame(exon_rows)
    for metric_suffix, ascending in [("pearson", False), ("spearman", False), ("mae", True)]:
        metric_cols = [f"{m}_{metric_suffix}" for m in mods]
        vals = exon_df[metric_cols].values
        if ascending:
            ranks = np.argsort(np.argsort(vals, axis=1), axis=1) + 1
        else:
            ranks = np.argsort(np.argsort(-vals, axis=1), axis=1) + 1
        for i, m in enumerate(mods):
            exon_df[f"{m}_{metric_suffix}_rank"] = ranks[:, i]
    df.drop(columns=["_exon_key"], inplace=True)
    _per_exon_cache[_key] = (exon_df, mods)
    return exon_df, mods


def fig_per_exon_rank_heatmap(d: Data, name: str,
                              # ---- figure auto-sizing (scales with n_models) ----
                              height_per_model=0.5,
                              min_height=4,
                              height_pad=1.5,
                              # ---- fonts ----
                              title_fontsize=None,            # metric panel titles (Pearson/Spearman/Mae)
                              label_fontsize=None,
                              xtick_fontsize=9,               # rank labels "#1", "#2", ...
                              ytick_fontsize=9,               # model names on y
                              cell_fontsize=7,                # per-cell percentage text
                              cmap="YlGnBu",                  # heatmap colormap
                              ) -> None:
    """rank frequency heatmap"""
    df = d.dfs[name]
    exon_df, mods = _build_per_exon_df(df)
    exon_df.to_csv(f"{out_dir}/{name}_per_exon_metrics.csv", index=False, float_format="%.4f")
    print(f"{dataset_names[name]}: {len(exon_df):,} exons with >= {MIN_VARIANTS_PER_EXON} variants")
    fig, axes = plt.subplots(1, 3, figsize=_ru.journal_size(15, max(min_height, len(mods) * height_per_model + height_pad)), constrained_layout=True)
    for ai, metric_suffix in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[ai]
        # for each model, count how often it ranks #1, #2, ..., #N across exons
        rank_matrix = np.zeros((len(mods), len(mods)), dtype=int)
        for mi, m in enumerate(mods):
            col = f"{m}_{metric_suffix}_rank"
            for rank_val in range(1, len(mods) + 1):
                rank_matrix[mi, rank_val - 1] = (exon_df[col] == rank_val).sum()
        rank_frac = rank_matrix / rank_matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(rank_frac, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([f"#{i+1}" for i in range(len(mods))], fontsize=xtick_fontsize)
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels([model_names[m] for m in mods], fontsize=ytick_fontsize)
        ax.set_xlabel("Rank", fontsize=label_fontsize)
        ax.set_title(metric_suffix.capitalize(), fontsize=title_fontsize)
        for mi in range(len(mods)):
            for ri in range(len(mods)):
                pct = rank_frac[mi, ri]
                ax.text(ri, mi, f"{100*pct:.0f}%", ha="center", va="center",
                        fontsize=cell_fontsize, color="white" if pct > 0.5 else "black")
    _save(fig3_sup, f"{name}_per_exon_rank_heatmap", tight=False)


def fig_per_exon_mean_rank(d: Data, name: str,
                           # ---- figure sizing ----
                           height=4.5,
                           # ---- fonts ----
                           title_fontsize=None,
                           label_fontsize=None,
                           xtick_fontsize=8,
                           xtick_rotation=30,
                           value_fontsize=7,                  # numeric value above each bar
                           # ---- bar styling ----
                           bar_edge="black",
                           bar_edge_width=0.5,
                           palette=None,
                           # ---- median reference line ----
                           ref_line_color="gray",
                           ref_line_style="--",
                           ref_line_width=0.5,
                           ref_line_alpha=0.5,
                           ) -> None:
    """mean rank bar chart"""
    df = d.dfs[name]
    exon_df, mods = _build_per_exon_df(df)
    fig, axes = plt.subplots(1, 3, figsize=_ru.journal_size(12, height), constrained_layout=True)
    for ai, metric_suffix in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[ai]
        mean_ranks = [exon_df[f"{m}_{metric_suffix}_rank"].mean() for m in mods]
        bars = ax.bar(range(len(mods)), mean_ranks,
                      color=[(palette or {}).get(m, get_color(m)) for m in mods],
                      edgecolor=bar_edge, linewidth=bar_edge_width)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([model_names[m] for m in mods], fontsize=xtick_fontsize, rotation=xtick_rotation, ha="right")
        ax.set_ylabel("Mean rank", fontsize=label_fontsize)
        ax.set_title(metric_suffix.capitalize(), fontsize=title_fontsize)
        ax.set_ylim(0.5, len(mods) + 0.5)
        # horizontal line at the "expected" rank under random ordering
        ax.axhline(len(mods) / 2 + 0.5, color=ref_line_color, ls=ref_line_style, lw=ref_line_width, alpha=ref_line_alpha)
        for bi, v in enumerate(mean_ranks):
            ax.text(bi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=value_fontsize)
    _save(fig3_sup, f"{name}_per_exon_mean_rank", tight=False)


def fig_per_exon_win_matrix(d: Data, name: str,
                            # ---- figure auto-sizing ----
                            height_per_model=0.7,
                            min_height=4.5,
                            height_pad=1.5,
                            # ---- fonts ----
                            title_fontsize=None,
                            label_fontsize=None,
                            xtick_fontsize=8,
                            xtick_rotation=45,
                            ytick_fontsize=8,
                            cell_fontsize=7,                  # per-cell win count
                            # ---- heatmap colormap ----
                            cmap="RdYlGn",
                            ) -> None:
    """win/loss matrix"""
    df = d.dfs[name]
    exon_df, mods = _build_per_exon_df(df)
    n_exons = len(exon_df)
    fig, axes = plt.subplots(1, 3, figsize=_ru.journal_size(15, max(min_height, len(mods) * height_per_model + height_pad)), constrained_layout=True)
    for ai, metric_suffix in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[ai]
        # row model wins over column model if its metric is better (smaller for MAE, larger else)
        win_matrix = np.zeros((len(mods), len(mods)), dtype=int)
        for mi, ma in enumerate(mods):
            for mj, mb in enumerate(mods):
                if mi == mj:
                    continue
                col_a = f"{ma}_{metric_suffix}"
                col_b = f"{mb}_{metric_suffix}"
                if metric_suffix == "mae":
                    wins = (exon_df[col_a] < exon_df[col_b]).sum()
                else:
                    wins = (exon_df[col_a] > exon_df[col_b]).sum()
                win_matrix[mi, mj] = wins
        win_frac = win_matrix / n_exons
        im = ax.imshow(win_frac, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([model_names[m] for m in mods], fontsize=xtick_fontsize, rotation=xtick_rotation, ha="right")
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels([model_names[m] for m in mods], fontsize=ytick_fontsize)
        ax.set_xlabel("Column model", fontsize=label_fontsize)
        ax.set_ylabel("Row model wins →", fontsize=label_fontsize)
        ax.set_title(metric_suffix.capitalize(), fontsize=title_fontsize)
        for mi in range(len(mods)):
            for mj in range(len(mods)):
                if mi == mj:
                    ax.text(mj, mi, "—", ha="center", va="center", fontsize=cell_fontsize)
                else:
                    ax.text(mj, mi, f"{win_matrix[mi, mj]}", ha="center", va="center",
                            fontsize=cell_fontsize, color="white" if win_frac[mi, mj] > 0.6 else "black")
    _save(fig3_sup, f"{name}_per_exon_win_matrix", tight=False)


def fig_exon_difficulty(d: Data, name: str,
                        # ---- figure sizing (one panel per property) ----
                        height=4.5,
                        width_per_panel=4,
                        # ---- fonts ----
                        title_fontsize=None,
                        label_fontsize=None,
                        tick_fontsize=None,
                        annot_fontsize=9,                  # ρ text top-left
                        # ---- scatter ----
                        scatter_size=20,
                        scatter_alpha=0.5,
                        scatter_color="#4a90d9",
                        # ---- grid ----
                        grid_alpha=0.2,
                        ) -> None:
    """correlate avg per-exon MAE with exon properties"""
    df = d.dfs[name]
    exon_df, mods = _build_per_exon_df(df)
    avg_mae = exon_df[[f"{m}_mae" for m in mods]].mean(axis=1)
    properties = [
        ("Exon width (bp)", exon_df["width"].values),
        ("Variants tested", exon_df["n_variants"].values),
        ("Mean |ΔPSI|", exon_df["mean_abs_dpsi"].values),
        ("SDV fraction", exon_df["n_sdv"].values / exon_df["n_variants"].values),
    ]
    fig, axes = plt.subplots(1, len(properties), figsize=_ru.journal_size(width_per_panel * len(properties), height), constrained_layout=True)
    for ax, (prop_name, prop_vals) in zip(axes, properties):
        fin = np.isfinite(prop_vals) & np.isfinite(avg_mae.values)
        ax.scatter(prop_vals[fin], avg_mae.values[fin], s=scatter_size, alpha=scatter_alpha,
                   color=scatter_color, edgecolors="none")
        if fin.sum() > 3:
            rho = stats.spearmanr(prop_vals[fin], avg_mae.values[fin])[0]
            ax.text(0.03, 0.97, f"$\\rho$={rho:.2f}", transform=ax.transAxes,
                    fontsize=annot_fontsize, va="top")
        ax.set_xlabel(prop_name, fontsize=label_fontsize)
        ax.set_ylabel("Mean MAE (all models)", fontsize=label_fontsize)
        ax.grid(alpha=grid_alpha)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    _save(fig3_sup, f"{name}_exon_difficulty", tight=False)


def fig_per_exon_concordance(d: Data, name: str,
                             # ---- figure auto-sizing (one panel per model pair) ----
                             panel_width=3.5,
                             panel_height=3.5,
                             max_cols=5,
                             # ---- fonts ----
                             label_fontsize=8,
                             annot_fontsize=8,                # rho text upper-left
                             tick_fontsize=None,
                             # ---- scatter ----
                             scatter_size=15,
                             scatter_alpha=0.5,
                             scatter_color="#555555",
                             # ---- y=x diagonal reference ----
                             diag_color="black",
                             diag_style="--",
                             diag_width=0.5,
                             diag_alpha=0.4,
                             # ---- axis padding around data ----
                             lim_pad=0.05,
                             ) -> None:
    """scatter of per-exon Pearson for each model pair"""
    df = d.dfs[name]
    exon_df, mods = _build_per_exon_df(df)
    pairs = list(combinations(mods, 2))
    n_pairs = len(pairs)
    ncols = min(max_cols, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=_ru.journal_size(panel_width * ncols, panel_height * nrows), squeeze=False, constrained_layout=True)
    for pi, (ma, mb) in enumerate(pairs):
        ax = axes[pi // ncols, pi % ncols]
        xa = exon_df[f"{ma}_pearson"].values
        xb = exon_df[f"{mb}_pearson"].values
        fin = np.isfinite(xa) & np.isfinite(xb)
        ax.scatter(xa[fin], xb[fin], s=scatter_size, alpha=scatter_alpha, color=scatter_color, edgecolors="none")
        lims = [min(np.nanmin(xa[fin]), np.nanmin(xb[fin])) - lim_pad,
                max(np.nanmax(xa[fin]), np.nanmax(xb[fin])) + lim_pad]
        ax.plot(lims, lims, color=diag_color, ls=diag_style, lw=diag_width, alpha=diag_alpha)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(model_names[ma], fontsize=label_fontsize)
        ax.set_ylabel(model_names[mb], fontsize=label_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        rho = stats.spearmanr(xa[fin], xb[fin])[0] if fin.sum() > 3 else np.nan
        ax.text(0.03, 0.97, f"$\\rho$={rho:.2f}", transform=ax.transAxes, fontsize=annot_fontsize, va="top")
        ax.set_aspect("equal")
    # hide empty trailing panels
    for pi in range(n_pairs, nrows * ncols):
        axes[pi // ncols, pi % ncols].set_visible(False)
    _save(fig3_sup, f"{name}_per_exon_concordance", tight=False)


# ---------------------------------------------------------------------------
# CLS vs SSU/usage head comparison
# ---------------------------------------------------------------------------

_HEAD_PAIRS = [
    ("splaire_ref", "splaire_ref_cls_delta", "splaire_ref_reg_ssu_delta", "SPLAIRE", "CLS", "SSU"),
    ("splaire_var", "splaire_var_cls_delta", "splaire_var_reg_ssu_delta", "SPLAIRE-var", "CLS", "SSU"),
    ("pangolin", "pangolin_max_p_splice", "pangolin_max_usage", "Pangolin", "CLS", "Usage"),
    ("pangolin_v2", "pangolin_v2_max_p_splice", "pangolin_v2_max_usage", "Pangolin v2", "CLS", "Usage"),
]

_LOC_COLORS_HEAD = {"splice_site": "#D55E00", "splice_region": "#E69F00", "exon": "#0072B2", "intron": "#009E73"}
_LOC_LABELS_HEAD = {"splice_site": "Splice site", "splice_region": "Splice region", "exon": "Exon", "intron": "Intron"}


def fig_cls_vs_ssu(d: Data, name: str,
                   # ---- figure sizing (one fig per model pair) ----
                   height=7,
                   # ---- fonts ----
                   title_fontsize=None,
                   label_fontsize=None,
                   tick_fontsize=None,
                   annot_fontsize=9,                       # AUPRC text upper-left
                   legend_fontsize=8,
                   # ---- scatter styling (per location) ----
                   scatter_size=10,
                   scatter_alpha=0.3,
                   loc_colors=None,                        # None = default 4-region palette
                   loc_labels=None,                        # None = default labels
                   # ---- y=x diagonal reference ----
                   diag_color="black",
                   diag_style="--",
                   diag_width=1,
                   diag_alpha=0.3,
                   # ---- axis padding ----
                   lim_pad=1.05,                           # multiplier on max(|a|, |b|) for axis limits
                   # ---- annotation box + grid ----
                   annot_bbox_alpha=0.8,
                   grid_alpha=0.2,
                   legend_loc="lower right",
                   ) -> None:
    """|delta| CLS vs SSU/usage scatter colored by location, per model"""
    df = d.dfs[name]
    y = df["y"].values
    lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    _loc_cols = loc_colors if loc_colors is not None else _LOC_COLORS_HEAD
    _loc_lbls = loc_labels if loc_labels is not None else _LOC_LABELS_HEAD
    for model_base, col_a, col_b, model_label, a_name, b_name in _HEAD_PAIRS:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        pred_a = df[col_a].values
        pred_b = df[col_b].values
        valid = np.isfinite(pred_a) & np.isfinite(pred_b) & np.isfinite(y)
        abs_a = np.abs(pred_a[valid])
        abs_b = np.abs(pred_b[valid])
        fig, ax = plt.subplots(figsize=_ru.journal_size(7, height), constrained_layout=True)
        for loc_name in ["intron", "exon", "splice_region", "splice_site"]:
            mask = loc[loc_name][valid]
            n = mask.sum()
            ax.scatter(abs_a[mask], abs_b[mask], s=scatter_size, alpha=scatter_alpha,
                      color=_loc_cols[loc_name],
                      label=f"{_loc_lbls[loc_name]} (n={n:,})",
                      rasterized=True)
        lim = max(abs_a.max(), abs_b.max()) * lim_pad
        ax.plot([0, lim], [0, lim], color=diag_color, ls=diag_style, alpha=diag_alpha, lw=diag_width)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"|delta| {a_name}", fontsize=label_fontsize)
        ax.set_ylabel(f"|delta| {b_name}", fontsize=label_fontsize)
        ax.set_title(f"{model_label} — {dataset_names[name]}", fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        auprc_a = average_precision_score(lab[valid], abs_a) if lab[valid].sum() > 0 else np.nan
        auprc_b = average_precision_score(lab[valid], abs_b) if lab[valid].sum() > 0 else np.nan
        ax.text(0.03, 0.97, f"{a_name} AUPRC: {auprc_a:.3f}\n{b_name} AUPRC: {auprc_b:.3f}",
                transform=ax.transAxes, fontsize=annot_fontsize, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=annot_bbox_alpha))
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, frameon=False)
        ax.set_aspect("equal")
        ax.grid(alpha=grid_alpha)
        _save(fig3_sup, f"{name}_{model_base}_cls_vs_ssu", tight=False)
        print(f"  {model_label} — {dataset_names[name]}: {a_name} AUPRC={auprc_a:.3f}, {b_name} AUPRC={auprc_b:.3f}")


_CELL_TISSUE = {"vexseq": "liver", "mfass": None}
_TISSUE_HEAD_CFG = {
    "Pangolin": {
        "aggregate": "pangolin_max_p_splice",
        "agg_label": "max p(splice)",
        "tissues": {t: f"pangolin_{t}_p_splice_delta" for t in ["brain", "heart", "liver", "testis"]},
    },
    "Pangolin v2": {
        "aggregate": "pangolin_v2_max_p_splice",
        "agg_label": "max p(splice)",
        "tissues": {t: f"pangolin_v2_{t}_p_splice_delta" for t in ["brain", "heart", "liver", "testis"]},
    },
}


def fig_tissue_head_eval(d: Data, name: str,
                         # ---- figure auto-sizing (one panel per family) ----
                         width_per_panel=5,
                         height_per_tissue=0.3,
                         min_height=4,
                         height_pad=2,
                         # ---- fonts ----
                         title_fontsize=None,                # family name above each panel
                         label_fontsize=None,
                         tick_fontsize=None,
                         ytick_fontsize=9,                   # tissue names on y
                         value_fontsize=8,                   # numeric AUPRC text next to each bar
                         # ---- bar styling ----
                         bar_alpha=0.8,
                         bar_height=0.7,
                         agg_color="#333333",                # color for the aggregate ("max" row)
                         expected_tissue_color="#D55E00",    # color for the matching cell-line tissue
                         palette=None,                       # {family-base: color} for non-agg/non-expected bars
                         # ---- x-axis padding ----
                         xlim_pad=1.15,                      # multiplier on max AUPRC for x-axis room
                         # ---- grid ----
                         grid_alpha=0.2,
                         ) -> None:
    """per-head AUPRC for each model family with tissue-specific heads"""
    df = d.dfs[name]
    lab = df["label"].values
    expected_tissue = _CELL_TISSUE.get(name)
    # auto-build splicetransformer tissue list from columns
    spt_tissues = {}
    for c in df.columns:
        if c.startswith("splicetransformer_usage_") and c.endswith("_delta") and "exon_" not in c:
            tissue = c.replace("splicetransformer_usage_", "").replace("_delta", "")
            spt_tissues[tissue] = c
    spt_cfg = {
        "aggregate": "splicetransformer_max_usage",
        "agg_label": "max usage",
        "tissues": spt_tissues,
    }
    families = {**_TISSUE_HEAD_CFG, "SpliceTransformer": spt_cfg}
    n_families = len(families)
    fig, axes = plt.subplots(1, n_families,
        figsize=_ru.journal_size(width_per_panel * n_families,
                                 max(min_height, height_per_tissue * max(len(c["tissues"]) for c in families.values()) + height_pad)),
        constrained_layout=True)
    if n_families == 1:
        axes = [axes]
    for ax, (family, cfg_th) in zip(axes, families.items()):
        agg_col = cfg_th["aggregate"]
        if agg_col not in df.columns:
            ax.text(0.5, 0.5, f"{family}: no data", ha="center", va="center", transform=ax.transAxes)
            continue
        tissue_auprcs = {}
        for tissue, col in cfg_th["tissues"].items():
            if col in df.columns:
                vals = np.abs(df[col].values)
                tissue_auprcs[tissue] = average_precision_score(lab, vals) if lab.sum() > 0 else np.nan
        agg_auprc = average_precision_score(lab, np.abs(df[agg_col].values)) if lab.sum() > 0 else np.nan
        sorted_tissues = sorted(tissue_auprcs.items(), key=lambda x: x[1])
        bar_labels = [t for t, _ in sorted_tissues] + [cfg_th["agg_label"]]
        bar_vals = [v for _, v in sorted_tissues] + [agg_auprc]
        # color bars: aggregate, cell-line-matching tissue, all other tissues (model family color)
        colors = []
        family_base = family.lower().split()[0]
        family_color = (palette or {}).get(family_base, get_color(family_base))
        for t in bar_labels:
            if t == cfg_th["agg_label"]:
                colors.append(agg_color)
            elif expected_tissue and t.lower().replace("_", " ") == expected_tissue.lower():
                colors.append(expected_tissue_color)
            else:
                colors.append(family_color)
        y_pos = np.arange(len(bar_labels))
        ax.barh(y_pos, bar_vals, color=colors, alpha=bar_alpha, height=bar_height)
        # numeric value next to each bar
        for i, v in enumerate(bar_vals):
            if np.isfinite(v):
                ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=value_fontsize)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=ytick_fontsize)
        ax.set_xlabel("AUPRC", fontsize=label_fontsize)
        ax.set_title(family, fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.set_xlim(0, max(v for v in bar_vals if np.isfinite(v)) * xlim_pad if bar_vals else 1)
        ax.grid(alpha=grid_alpha, axis="x")
    _save(fig3_sup, f"{name}_tissue_head_eval", tight=False)


def fig_threshold_by_location(d: Data, name: str,
                              # ---- figure sizing ----
                              height=10,
                              # ---- fonts ----
                              title_fontsize=None,             # per-location title "Exon (n=...)"
                              label_fontsize=None,
                              tick_fontsize=None,
                              legend_fontsize=7,
                              sd_marker_fontsize=7,            # rotated "1 SD=X" annotation
                              # ---- model curves ----
                              model_line_width=2,
                              palette=None,                    # {model: color}
                              # ---- 1-SD vertical marker ----
                              sd_line_color="black",
                              sd_line_style=":",
                              sd_line_width=1.5,
                              sd_line_alpha=0.7,
                              # ---- legend + grid ----
                              legend_loc="best",
                              grid_alpha=0.3,
                              ) -> None:
    """threshold sensitivity per location category"""
    df = d.dfs[name]
    cols = thresh_cols[name]
    y = df["y"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    thresholds = np.arange(0.01, 1.01, 0.01)
    fig, axes = plt.subplots(2, 2, figsize=_ru.journal_size(12, height), constrained_layout=True)
    # one panel per location category — auprc as threshold sweeps 0..1
    for ax, loc_name in zip(axes.flatten(), LOCATION_SUBSETS):
        mask = loc[loc_name]
        y_loc = y[mask]
        sd_thr = np.std(y_loc)
        for m in models:
            if cols[m] not in df.columns:
                continue
            delta = df[cols[m]].values[mask]
            auprcs = []
            for thr in thresholds:
                lab_t = (np.abs(y_loc) > thr).astype(int)
                if lab_t.sum() > 0 and (1 - lab_t).sum() > 0:
                    auprcs.append(average_precision_score(lab_t, np.abs(delta)))
                else:
                    auprcs.append(np.nan)
            mcolor = (palette or {}).get(m, get_color(m))
            ax.plot(thresholds, auprcs, lw=model_line_width, color=mcolor, label=model_names[m])
        # per-location 1-SD marker line
        ax.axvline(sd_thr, color=sd_line_color, ls=sd_line_style, lw=sd_line_width, alpha=sd_line_alpha)
        ax.text(sd_thr + 0.01, 0.95, f"1 SD={sd_thr:.2f}", fontsize=sd_marker_fontsize, va="top", rotation=90, alpha=sd_line_alpha)
        ax.set_xlabel("|ΔPSI| threshold", fontsize=label_fontsize)
        ax.set_ylabel("AUPRC", fontsize=label_fontsize)
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_ylim(0, 1)
        ax.set_title(f"{LOCATION_LABELS[loc_name]} (n={mask.sum():,})", fontsize=title_fontsize)
        if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)
        ax.grid(alpha=grid_alpha)
    _save(fig3_sup, f"{name}_threshold_by_location", tight=False)


def fig_sfig_neutral_mae(d: Data, name: str,
                         # ---- figure sizing ----
                         height=4,
                         # ---- fonts ----
                         title_fontsize=None,                # the long ds-specific title
                         label_fontsize=None,
                         tick_fontsize=None,
                         value_fontsize=9,                   # numeric value text next to each bar
                         # ---- bar styling ----
                         bar_edge="none",
                         palette=None,                       # {model: color}
                         # ---- x-axis padding ----
                         xlim_pad=1.25,                      # multiplier on max MAE for x-axis room
                         ) -> None:
    """per-model MAE on neutral variants (|y| within 1 SD) near splice sites.
    counterpart to table_neutral_mae.tex (figures_pub.py). near splice site =
    splice_site + splice_region (3 exonic + 8 intronic nt). saves as
    figures/sup/sfig_neutral_mae_{name}.png."""
    df = d.dfs[name]
    y = df["y"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)
    # "neutral" = within 1 SD of the mean |dpsi| (Mount-style cutoff)
    neutral_mask = np.abs(y - y_mean) < y_std
    near_ss_mask = loc["splice_site"] | loc["splice_region"]
    final_mask = neutral_mask & near_ss_mask
    delta_cols = thresh_cols[name]
    rows = []
    for m in models:
        if delta_cols[m] not in df.columns:
            continue
        delta = df[delta_cols[m]].values[final_mask]
        y_neu = y[final_mask]
        valid = np.isfinite(delta) & np.isfinite(y_neu)
        if valid.sum() == 0:
            continue
        mae = float(np.mean(np.abs(delta[valid] - y_neu[valid])))
        rows.append({"model": m, "mae": mae, "n": int(valid.sum())})
    if not rows:
        print(f"  {name}: no valid models for sfig_neutral_mae")
        return
    mae_df = pd.DataFrame(rows).sort_values("mae")
    fig, ax = plt.subplots(figsize=_ru.journal_size(8, height), constrained_layout=True)
    ax.barh(range(len(mae_df)), mae_df["mae"].values,
            color=[(palette or {}).get(m, get_color(m)) for m in mae_df["model"]],
            edgecolor=bar_edge)
    xmax = mae_df["mae"].max() * xlim_pad
    # print numeric MAE + n next to each bar
    for i, (_, row) in enumerate(mae_df.iterrows()):
        ax.text(row["mae"] + 0.005 * xmax, i, f"{row['mae']:.3f} (n={row['n']:,})",
                va="center", fontsize=value_fontsize)
    ax.set_yticks(range(len(mae_df)))
    ax.set_yticklabels([model_names[m] for m in mae_df["model"]])
    ax.set_xlabel("MAE on neutral variants", fontsize=label_fontsize)
    ax.set_xlim(0, xmax)
    ax.set_title(f"{dataset_names[name]} — neutral MAE near splice site (|y|<1 SD, SS + SR)", fontsize=title_fontsize)
    if tick_fontsize is not None: ax.tick_params(labelsize=tick_fontsize)
    ax.invert_yaxis()
    _save(fig3_sup, f"sfig_neutral_mae_{name}", tight=False)
    print(f"  saved sfig_neutral_mae_{name}: {len(mae_df)} models, n_variants={mae_df['n'].iloc[0]:,}")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def main():
    d = load_data()

    # data overview + PR + threshold + all_outputs (per dataset)
    for name in d.dfs:
        fig_data_overview(d, name)
        fig_pr(d, name)
        fig_threshold(d, name)
        fig_all_outputs(d, name)

    # per-position figures
    for name in d.dfs:
        fig_dpsi_by_position(d, name)
        fig_error_by_position(d, name)
        fig_measured_psi_by_position(d, name)
        fig_effect_by_position_rigorous(d, name)
        fig_pct_sdv_by_position(d, name)
    fig_pct_sdv_by_position_combined(d)

    # variant error analysis
    for name in d.dfs:
        fig_mae_by_position(d, name)
        fig_error_vs_effect(d, name)
        fig_error_vs_intron_length(d, name)

    # gnomAD annotation + MAF figures
    _annotate_gnomad(d)
    for name in d.dfs:
        fig_error_vs_maf(d, name)
        fig_effect_vs_maf(d, name)

    # bootstrap (cached, skippable)
    run_bootstrap(d)

    # main manuscript figure (single composite, embedded panel labels)
    fig_reporter_full_grouped_bars(d)

    # correction sensitivity
    fig_correction_sensitivity(d)

    # ref/alt inclusion analysis
    fig_ref_alt_tables(d)
    for name in d.dfs:
        fig_ref_inclusion_scatter(d, name)
    fig_exon_ref_psi(d)
    fig_mfass_alt_inclusion_scatter(d)
    fig_mfass_replicate_consistency(d)
    for name in d.dfs:
        fig_error_vs_baseline(d, name)
        fig_dpsi_perf_by_ref_psi(d, name)
    fig_error_refpsi_spearman_table(d)

    # per-exon analysis
    for name in d.dfs:
        fig_per_exon_rank_heatmap(d, name)
        fig_per_exon_mean_rank(d, name)
        fig_per_exon_win_matrix(d, name)
        fig_exon_difficulty(d, name)
        fig_per_exon_concordance(d, name)

    # head comparison + tissue head + threshold by location + neutral MAE
    for name in d.dfs:
        fig_cls_vs_ssu(d, name)
        fig_tissue_head_eval(d, name)
        fig_threshold_by_location(d, name)
        fig_sfig_neutral_mae(d, name)

    # dump tables as markdown for the static qmd to include
    _dump_tables()


def _tex_tabular_to_md(tex: str) -> str:
    # extract rows between \toprule / \midrule / \bottomrule, strip latex artifacts
    # multi-row header is collapsed into single header with <br>
    import re
    body = tex
    # cut to tabular block
    m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", body, re.DOTALL)
    if not m:
        return "```latex\n" + tex + "\n```\n"
    body = m.group(1)
    # remove rule commands
    body = re.sub(r"\\(top|mid|bottom)rule\b", "", body)
    rows = [r.strip().rstrip(r"\\").strip() for r in body.split(r"\\") if r.strip()]
    # parse rows: split on & (not preceded by \), strip cells
    def clean(cell):
        cell = cell.strip()
        cell = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", cell)
        cell = re.sub(r"\\text(it|tt|rm|sf)\{([^}]*)\}", r"\2", cell)
        cell = cell.replace(r"\%", "%").replace(r"\$", "$")
        return cell
    parsed = [[clean(c) for c in row.split("&")] for row in rows]
    # find midrule split: in source, header rows come before midrule
    # since we already stripped rules, infer header by counting rows: typically 1-2 header rows
    # heuristic: rows whose first cell is non-numeric AND row appears in first 3 rows are headers
    n_cols = len(parsed[0])
    header_count = 1
    for i in range(1, min(3, len(parsed))):
        first = parsed[i][0]
        if first == "" or not any(c.replace(".", "").replace(",", "").replace("-", "").isdigit() for c in parsed[i]):
            header_count = i + 1
        else:
            break
    headers = parsed[:header_count]
    data = parsed[header_count:]
    # collapse multi-row headers with <br>
    final_header = []
    for j in range(n_cols):
        parts = [h[j] if j < len(h) else "" for h in headers]
        parts = [p for p in parts if p]
        final_header.append(" ".join(parts))
    out = "| " + " | ".join(final_header) + " |\n"
    out += "|" + "|".join(["---"] * n_cols) + "|\n"
    for row in data:
        out += "| " + " | ".join(row) + " |\n"
    return out


def _dump_tables():
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)

    # neutral mae table — parse latex tabular into markdown so html renders properly
    # canonical source: ../figures/fig3/sup/ (paper-wide), with local fallbacks
    tex_candidates = [
        Path("../figures/fig3/sup/table_neutral_mae.tex"),
        Path("figures/fig3/sup/table_neutral_mae.tex"),
        Path("figures/sup/table_neutral_mae.tex"),
    ]
    tex_path = next((p for p in tex_candidates if p.exists()), tex_candidates[0])
    md_path = out / "_neutral_mae_table.md"
    if tex_path.exists():
        md_path.write_text(_tex_tabular_to_md(tex_path.read_text()))
        print(f"  table -> {md_path}  (src: {tex_path})")
    else:
        md_path.write_text(f"_missing: `{tex_path}`_\n")

    # reporter assay table — split into vexseq and mfass markdown
    csv_candidates = [
        Path("../figures/fig3/results/reporter_assay_table.csv"),
        Path("figures/fig3/results/reporter_assay_table.csv"),
        Path("figures/results/reporter_assay_table.csv"),
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), csv_candidates[0])
    if csv_path.exists():
        t = pd.read_csv(csv_path)
        for ds in ("vexseq", "mfass"):
            sub = t.query("dataset == @ds").drop(columns=["dataset", "boot_id"], errors="ignore").round(3)
            sub_md = out / f"_reporter_assay_table_{ds}.md"
            sub.to_markdown(sub_md, index=False)
            print(f"  table -> {sub_md}  (src: {csv_path})")
    else:
        for ds in ("vexseq", "mfass"):
            (out / f"_reporter_assay_table_{ds}.md").write_text(f"_missing: `{csv_path}`_\n")


if __name__ == "__main__":
    main()
