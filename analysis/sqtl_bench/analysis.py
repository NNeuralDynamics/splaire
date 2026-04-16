#!/usr/bin/env python
# coding: utf-8

# # sQTL Benchmark Analysis
# 
# ## Data Sources
# 
# **GTEx (txrevise & leafcutter)**
# - Fine-mapped sQTLs and summary statistics from [eQTL Catalogue](https://www.ebi.ac.uk/eqtl/)
# - FTP: `ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats`
# - 49 tissues, GRCh38
# 
# **HAEC185**
# - Credible sets: `/scratch/runyan.m/sqtl_bench/haec/raw/HAEC185_sQTL_credible_sets_120825.tsv`
# - Sumstats: `/projects/talisman/shared-data/HAEC-185/sQTL/sQTL_redo/*.parquet`
# - TPM: `/projects/talisman/shared-data/HAEC-185/bams/salmon.merged.gene_tpm.tsv`
# - Single tissue, GRCh38
# 
# **Reference**
# - GTEx v8 median TPM: `/scratch/runyan.m/sqtl_bench/reference/GTEx_v8_median_tpm.gct.gz`
# - GENCODE v39 GTF (txrevise/leafcutter): `/scratch/runyan.m/sqtl_bench/reference/gencode.v39.basic.annotation.gtf.gz`
# - GENCODE v45 GTF (HAEC): `/projects/talisman/mrunyan/paper/SpHAEC/pipeline/reference/GRCh38/gencode.v45.primary_assembly.annotation.gtf`
# 
# ---
# 
# Evaluation of splice variant effect prediction models on three sQTL datasets (txrevise, leafcutter, HAEC). For each dataset, positive variants (fine-mapped sQTLs with high PIP) are matched 1:1 to negative variants (tested but non-significant variants from summary statistics). 
# 
# A fourth analysis tests within-credible-set ranking on cases where fine-mapping is uncertain.

# In[ ]:


import argparse
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # avoid NFS stale lock errors

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script mode
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import h5py
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--nohbar', action='store_true', help='skip hbar plots and per-tissue-head scoring')
args = parser.parse_args()

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False

DATA_BASE = Path(os.environ.get("SPLAIRE_SQTL_DIR", "/scratch/runyan.m/sqtl_bench"))
FIG_BASE = DATA_BASE / 'figures'
FIG_BASE.mkdir(exist_ok=True)

# logging — tee all output to file + stdout
LOG_PATH = DATA_BASE / 'sqtl_benchmark_results.log'
_log_file = open(LOG_PATH, 'w')

class _Tee:
    """write to both stdout and log file"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)

# colorblind-friendly model palette
# base models + output variants share the same base color
COLORS = {
    'pangolin': '#66C2A5',
    'pangolin_usage': '#66C2A5',
    'pangolin_all': '#66C2A5',
    'pangolin_v2': '#009E73',
    'pangolin_v2_usage': '#009E73',
    'pangolin_v2_all': '#009E73',
    'spliceai': '#D55E00',
    'splicetransformer': '#E69F00',
    'splicetransformer_usage': '#E69F00',
    'splicetransformer_all': '#E69F00',
    'sphaec_ref': '#56B4E9',
    'sphaec_ref_reg': '#56B4E9',
    'sphaec_ref_all': '#56B4E9',
    'sphaec_var': '#0072B2',
    'sphaec_var_reg': '#0072B2',
    'sphaec_var_all': '#0072B2',
}
_PANG_TISSUES = ['brain', 'heart', 'liver', 'testis']
_SPT_TISSUES = [
    'adipose', 'blood', 'blood_vessel', 'brain', 'colon',
    'heart', 'kidney', 'liver', 'lung', 'muscle',
    'nerve', 'small_intestine', 'skin', 'spleen', 'stomach',
]
MODEL_NAMES = {
    'pangolin': 'Pangolin v1 CLS',
    'pangolin_usage': 'Pangolin v1 Usage',
    'pangolin_all': 'Pangolin v1 All',
    'pangolin_v2': 'Pangolin CLS',
    'pangolin_v2_usage': 'Pangolin Usage',
    'pangolin_v2_all': 'Pangolin All',
    'spliceai': 'SpliceAI',
    'splicetransformer': 'SpliceTransformer CLS',
    'splicetransformer_usage': 'SpliceTransformer Usage',
    'splicetransformer_all': 'SpliceTransformer All',
    'sphaec_ref': 'SPLAIRE CLS',
    'sphaec_ref_reg': 'SPLAIRE SSU',
    'sphaec_ref_all': 'SPLAIRE All',
    'sphaec_var': 'SPLAIRE-var CLS',
    'sphaec_var_reg': 'SPLAIRE-var SSU',
    'sphaec_var_all': 'SPLAIRE-var All',
}
# pangolin per-tissue heads
for _t in _PANG_TISSUES:
    MODEL_NAMES[f'pangolin_{_t}_cls'] = f'Pangolin v1 {_t} CLS'
    MODEL_NAMES[f'pangolin_{_t}_usage'] = f'Pangolin v1 {_t} Usage'
    MODEL_NAMES[f'pangolin_v2_{_t}_cls'] = f'Pangolin {_t} CLS'
    MODEL_NAMES[f'pangolin_v2_{_t}_usage'] = f'Pangolin {_t} Usage'
    COLORS[f'pangolin_{_t}_cls'] = '#66C2A5'
    COLORS[f'pangolin_{_t}_usage'] = '#66C2A5'
    COLORS[f'pangolin_v2_{_t}_cls'] = '#009E73'
    COLORS[f'pangolin_v2_{_t}_usage'] = '#009E73'
# splicetransformer per-tissue usage
for _t in _SPT_TISSUES:
    MODEL_NAMES[f'spt_{_t}'] = f'SPT {_t}'
    COLORS[f'spt_{_t}'] = '#E69F00'
# primary models used for swarm/PR/ROC/distance plots
MODELS = ['pangolin_v2', 'spliceai', 'splicetransformer', 'sphaec_ref', 'sphaec_var']
# all output variants for the bar chart comparison
MODELS_ALL = [
    'pangolin', 'pangolin_usage', 'pangolin_all',
    *[f'pangolin_{t}_cls' for t in _PANG_TISSUES],
    *[f'pangolin_{t}_usage' for t in _PANG_TISSUES],
    'pangolin_v2', 'pangolin_v2_usage', 'pangolin_v2_all',
    *[f'pangolin_v2_{t}_cls' for t in _PANG_TISSUES],
    *[f'pangolin_v2_{t}_usage' for t in _PANG_TISSUES],
    'spliceai',
    'splicetransformer', 'splicetransformer_usage', 'splicetransformer_all',
    *[f'spt_{t}' for t in _SPT_TISSUES],
    'sphaec_ref', 'sphaec_ref_reg', 'sphaec_ref_all',
    'sphaec_var', 'sphaec_var_reg', 'sphaec_var_all',
]

DATA_COLORS = {
    'pos': '#66c2a5', 'neg': '#fc8d62',
    'ge': '#8da0cb', 'txrevise': '#e78ac3',
    'set1': '#a6d854', 'set2': '#ffd92f',
    'tier1': '#66c2a5', 'tier2': '#8da0cb', 'tier3': '#fc8d62',
    'tier4': '#e78ac3', 'tier5': '#a6d854', 'tier6': '#ffd92f',
}

RAW = {'tpm_gtex': DATA_BASE / 'reference' / 'GTEx_v8_median_tpm.gct.gz'}


def format_axis_plain(ax, axis="both"):
    """disable scientific notation, use plain numbers with commas"""
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f"{int(x):,}"
        elif x == int(x):
            return f"{int(x)}"
        else:
            return f"{x:.2f}"
    formatter = FuncFormatter(thousands_formatter)
    if axis in ("both", "x"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("both", "y"):
        ax.yaxis.set_major_formatter(formatter)


def savefig(path, fig=None, dpi=300):
    """save figure as png and pdf"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white", edgecolor="none")


print(f'data: {DATA_BASE}')
print(f'figures: {FIG_BASE}')
print(f'log: {LOG_PATH}')


# In[ ]:


# helper functions

WINDOW = 2000  # ±2000bp around variant center for max delta score


def load_scores(score_dir, vcf_base, primary_only=False):
    """load scores from h5 files, returns dict of model -> {var_key: max |delta|} within ±WINDOW bp
    primary_only: skip per-tissue heads (faster, only compute aggregate models)"""
    # check parquet cache
    cache_sfx = '_deltas_primary.parquet' if primary_only else '_deltas_all.parquet'
    cache = score_dir / f'{vcf_base}.{cache_sfx}'
    if cache.exists():
        df = pd.read_parquet(cache)
        scores = {}
        for col in df.columns:
            if col == 'var_key': continue
            scores[col] = dict(zip(df['var_key'], df[col]))
        print(f'  cached: {cache.name} ({len(scores)} models, {len(df):,} variants)')
        return scores

    scores = {}

    def _wslice(seq_len):
        """position slice for ±WINDOW bp around center"""
        c = seq_len // 2
        return slice(max(0, c - WINDOW), min(seq_len, c + WINDOW + 1))

    # spliceai: ref/alt shape (n, seq_len, 3)
    sa_path = score_dir / f'{vcf_base}.sa.h5'
    if sa_path.exists():
        with h5py.File(sa_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            ref, alt = f['ref'][:], f['alt'][:]
            sl = _wslice(ref.shape[1])
            delta = np.max(np.abs(alt[:, sl, :] - ref[:, sl, :]), axis=(1, 2))
            scores['spliceai'] = dict(zip(keys, delta))

    # pangolin v1 — skip entirely in primary_only (v2 is primary)
    pang_path = score_dir / f'{vcf_base}.pang.h5'
    if pang_path.exists() and not primary_only:
        with h5py.File(pang_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            all_tasks = [k[4:] for k in f.keys() if k.startswith('ref_')]
            ps_tasks = [t for t in all_tasks if t.endswith('_p_splice')]
            us_tasks = [t for t in all_tasks if t.endswith('_usage')]
            sl = _wslice(f[f'ref_{ps_tasks[0]}'].shape[1])
            max_ps = None
            for task in ps_tasks:
                d = np.max(np.abs(f[f'alt_{task}'][:, sl] - f[f'ref_{task}'][:, sl]), axis=1)
                max_ps = d if max_ps is None else np.maximum(max_ps, d)
                tissue = task.replace('_p_splice', '')
                scores[f'pangolin_{tissue}_cls'] = dict(zip(keys, d))
            scores['pangolin'] = dict(zip(keys, max_ps))
            max_us = None
            for task in us_tasks:
                d = np.max(np.abs(f[f'alt_{task}'][:, sl] - f[f'ref_{task}'][:, sl]), axis=1)
                max_us = d if max_us is None else np.maximum(max_us, d)
                tissue = task.replace('_usage', '')
                scores[f'pangolin_{tissue}_usage'] = dict(zip(keys, d))
            scores['pangolin_usage'] = dict(zip(keys, max_us))
            scores['pangolin_all'] = dict(zip(keys, np.maximum(max_ps, max_us)))

    # pangolin v2 (variant-finetuned) — primary model, always load CLS aggregate
    pang_v2_path = score_dir / f'{vcf_base}.pang_v2.h5'
    if pang_v2_path.exists():
        with h5py.File(pang_v2_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            all_tasks = [k[4:] for k in f.keys() if k.startswith('ref_')]
            ps_tasks = [t for t in all_tasks if t.endswith('_p_splice')]
            us_tasks = [t for t in all_tasks if t.endswith('_usage')]
            sl = _wslice(f[f'ref_{ps_tasks[0]}'].shape[1])
            max_ps = None
            for task in ps_tasks:
                d = np.max(np.abs(f[f'alt_{task}'][:, sl] - f[f'ref_{task}'][:, sl]), axis=1)
                max_ps = d if max_ps is None else np.maximum(max_ps, d)
                if not primary_only:
                    tissue = task.replace('_p_splice', '')
                    scores[f'pangolin_v2_{tissue}_cls'] = dict(zip(keys, d))
            scores['pangolin_v2'] = dict(zip(keys, max_ps))
            if not primary_only:
                max_us = None
                for task in us_tasks:
                    d = np.max(np.abs(f[f'alt_{task}'][:, sl] - f[f'ref_{task}'][:, sl]), axis=1)
                    max_us = d if max_us is None else np.maximum(max_us, d)
                    tissue = task.replace('_usage', '')
                    scores[f'pangolin_v2_{tissue}_usage'] = dict(zip(keys, d))
                scores['pangolin_v2_usage'] = dict(zip(keys, max_us))
                scores['pangolin_v2_all'] = dict(zip(keys, np.maximum(max_ps, max_us)))

    # splaire-var: cls + reg from variant-aware model
    sphaec_var_path = score_dir / f'{vcf_base}.splaire.var.h5'
    if not sphaec_var_path.exists():
        sphaec_var_path = score_dir / f'{vcf_base}.sphaec.var.h5'
    if sphaec_var_path.exists():
        with h5py.File(sphaec_var_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            cls_ref, cls_alt = f['cls_ref'][:], f['cls_alt'][:]
            sl = _wslice(cls_ref.shape[1])
            delta_cls = np.max(np.abs(cls_alt[:, sl, :] - cls_ref[:, sl, :]), axis=(1, 2))
            scores['sphaec_var'] = dict(zip(keys, delta_cls))
            if 'reg_ref' in f and not primary_only:
                reg_ref, reg_alt = f['reg_ref'][:], f['reg_alt'][:]
                delta_reg = np.max(np.abs(reg_alt[:, sl] - reg_ref[:, sl]), axis=1)
                scores['sphaec_var_reg'] = dict(zip(keys, delta_reg))
                scores['sphaec_var_all'] = dict(zip(keys, np.maximum(delta_cls, delta_reg)))

    # splaire-ref: cls + reg from reference model
    sphaec_ref_path = score_dir / f'{vcf_base}.splaire.ref.h5'
    if not sphaec_ref_path.exists():
        sphaec_ref_path = score_dir / f'{vcf_base}.sphaec.ref.h5'
    if sphaec_ref_path.exists():
        with h5py.File(sphaec_ref_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            cls_ref, cls_alt = f['cls_ref'][:], f['cls_alt'][:]
            sl = _wslice(cls_ref.shape[1])
            delta_cls = np.max(np.abs(cls_alt[:, sl, :] - cls_ref[:, sl, :]), axis=(1, 2))
            scores['sphaec_ref'] = dict(zip(keys, delta_cls))
            if 'reg_ref' in f and not primary_only:
                reg_ref, reg_alt = f['reg_ref'][:], f['reg_alt'][:]
                delta_reg = np.max(np.abs(reg_alt[:, sl] - reg_ref[:, sl]), axis=1)
                scores['sphaec_ref_reg'] = dict(zip(keys, delta_reg))
                scores['sphaec_ref_all'] = dict(zip(keys, np.maximum(delta_cls, delta_reg)))

    # splicetransformer: ref/alt shape (n, 18, seq_len) — channels 0-2 cls, 3-17 usage
    spt_path = score_dir / f'{vcf_base}.spt.h5'
    SPT_USAGE_TISSUES = [
        'adipose', 'blood', 'blood_vessel', 'brain', 'colon',
        'heart', 'kidney', 'liver', 'lung', 'muscle',
        'nerve', 'small_intestine', 'skin', 'spleen', 'stomach',
    ]
    if spt_path.exists():
        with h5py.File(spt_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            ref = f['ref'][:]
            alt = f['alt'][:]
            sl = _wslice(ref.shape[2])  # position is axis 2 for spt
            # cls only (channels 0-2)
            delta_cls = np.max(np.abs(alt[:, :3, sl] - ref[:, :3, sl]), axis=(1, 2))
            scores['splicetransformer'] = dict(zip(keys, delta_cls))
            if not primary_only:
                # usage only (channels 3-17)
                delta_usage = np.max(np.abs(alt[:, 3:, sl] - ref[:, 3:, sl]), axis=(1, 2))
                scores['splicetransformer_usage'] = dict(zip(keys, delta_usage))
                # all channels
                delta_all = np.max(np.abs(alt[:, :, sl] - ref[:, :, sl]), axis=(1, 2))
                scores['splicetransformer_all'] = dict(zip(keys, delta_all))
                # per-tissue usage channels
                for ch_idx, tissue in enumerate(SPT_USAGE_TISSUES):
                    ch = ch_idx + 3
                    d = np.max(np.abs(alt[:, ch, sl] - ref[:, ch, sl]), axis=1)
                    scores[f'spt_{tissue}'] = dict(zip(keys, d))

    # write cache
    if scores:
        var_keys = list(next(iter(scores.values())).keys())
        df = pd.DataFrame({'var_key': var_keys})
        for model, d in scores.items():
            df[model] = df['var_key'].map(d).astype(np.float32)
        df.to_parquet(cache)
        print(f'  saved cache: {cache.name} ({len(scores)} models, {len(df):,} variants)')

    return scores


def compute_auprc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def compute_auroc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def get_ys(pairs, pos_scores, neg_scores, model, neg_col='neg_var_key'):
    """get y_true, y_score arrays for a model"""
    pv = np.array([pos_scores[model].get(k, np.nan) for k in pairs['pos_var_key']])
    nv = np.array([neg_scores[model].get(k, np.nan) for k in pairs[neg_col]])
    valid = ~(np.isnan(pv) | np.isnan(nv))
    y_true = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
    y_score = np.concatenate([pv[valid], nv[valid]])
    return y_true, y_score, valid.sum()


def compute_tissue_metrics(pairs, pos_scores, neg_scores):
    """AUPRC/AUROC per tissue"""
    results = []
    for tissue in pairs['tissue'].unique():
        t = pairs[pairs['tissue'] == tissue]
        for model in pos_scores:
            if model not in neg_scores:
                continue
            yt, ys, n = get_ys(t, pos_scores, neg_scores, model)
            if n < 10:
                continue
            results.append({
                'tissue': tissue, 'model': model, 'n': n,
                'auprc': compute_auprc(yt, ys),
                'auroc': compute_auroc(yt, ys),
            })
    return pd.DataFrame(results)


def compute_tissue_counts(pairs):
    """count unique pos/neg variants per tissue and shared across tissues"""
    pos_tcounts = pairs.groupby('pos_var_key')['tissue'].nunique()
    neg_tcounts = pairs.groupby('neg_var_key')['tissue'].nunique()
    results = []
    for tissue in pairs['tissue'].unique():
        tp = pairs[pairs['tissue'] == tissue]
        pk = tp['pos_var_key']
        nk = tp['neg_var_key']
        results.append({
            'tissue': tissue,
            'n_pos': pk.nunique(),
            'n_neg': nk.nunique(),
            'n_pos_shared': (pos_tcounts[pk.unique()] > 1).sum(),
            'n_neg_shared': (neg_tcounts[nk.unique()] > 1).sum(),
            'n_pos_unique': (pos_tcounts[pk.unique()] == 1).sum(),
            'n_neg_unique': (neg_tcounts[nk.unique()] == 1).sum(),
        })
    return pd.DataFrame(results)


def plot_summary(metrics, title, save_path=None):
    """bar or swarm plot of AUPRC per model"""
    single_tissue = metrics['tissue'].nunique() == 1
    fig, ax = plt.subplots(figsize=(8, 5 if not single_tissue else 6))
    mp = [m for m in MODELS if m in metrics['model'].values]

    if single_tissue:
        for i, model in enumerate(mp):
            data = metrics[metrics['model'] == model]['auprc']
            if len(data):
                val = data.iloc[0]
                ax.bar(i, val, color=COLORS[model], alpha=0.8)
                ax.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)
        ax.set_xticks(range(len(mp)))
        ax.set_xticklabels([MODEL_NAMES[m] for m in mp], rotation=45, ha='right')
        ax.set_ylabel('AUPRC')
    else:
        for i, model in enumerate(mp):
            data = metrics[metrics['model'] == model]['auprc']
            ax.scatter(np.random.normal(i, 0.08, len(data)), data, alpha=0.5,
                       color=COLORS[model], s=25)
            med = data.median()
            ax.hlines(med, i - 0.3, i + 0.3, colors=COLORS[model], lw=2.5)
            ax.text(i + 0.35, med, f'{med:.3f}', va='center', fontsize=9)
        ax.set_xticks(range(len(mp)))
        ax.set_xticklabels([MODEL_NAMES[m] for m in mp], rotation=45, ha='right')
        ax.set_ylabel('AUPRC per tissue')

    ax.set_title(title)
    all_vals = metrics[metrics['model'].isin(mp)]['auprc']
    ylo = np.floor(all_vals.min() * 10) / 10
    yhi = np.ceil(all_vals.max() * 10) / 10
    ax.set_ylim(ylo, yhi)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_median_auprc_hbar(metrics, title, save_path=None):
    """horizontal bar chart of median AUPRC for all model output variants"""
    single_tissue = metrics['tissue'].nunique() == 1
    mp = [m for m in MODELS_ALL if m in metrics['model'].values]
    if not mp:
        return

    medians = []
    for model in mp:
        data = metrics[metrics['model'] == model]['auprc']
        medians.append(data.median() if not single_tissue else data.iloc[0])

    # sort by median AUPRC descending (top = best)
    order = np.argsort(medians)  # ascending, plotted bottom-to-top
    mp_sorted = [mp[i] for i in order]
    med_sorted = [medians[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(mp_sorted))))
    y = np.arange(len(mp_sorted))
    colors = [COLORS.get(m, 'gray') for m in mp_sorted]

    bars = ax.barh(y, med_sorted, color=colors, alpha=0.8, height=0.7)
    for yi, val in zip(y, med_sorted):
        ax.text(val + 0.005, yi, f'{val:.3f}', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in mp_sorted], fontsize=9)
    ax.set_xlabel('Median AUPRC' if not single_tissue else 'AUPRC')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='gray', ls='--', alpha=0.3)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_pr(pairs, pos_scores, neg_scores, title, save_path=None):
    """precision-recall curves per model"""
    single_tissue = 'tissue' not in pairs.columns
    mp = [m for m in MODELS if m in pos_scores]
    fig, ax = plt.subplots(figsize=(7, 6))

    for model in mp:
        if single_tissue:
            yt, ys, n = get_ys(pairs, pos_scores, neg_scores, model)
            if n < 10: continue
            prec, rec, _ = precision_recall_curve(yt, ys)
            ap = compute_auprc(yt, ys)
            ax.plot(rec, prec, color=COLORS[model], label=f'{MODEL_NAMES[model]} ({ap:.3f})', lw=2)
        else:
            # tissue-median PR: compute per tissue, plot median
            tissue_precs = {}
            for tissue in pairs['tissue'].unique():
                tp = pairs[pairs['tissue'] == tissue]
                yt, ys, n = get_ys(tp, pos_scores, neg_scores, model)
                if n < 10: continue
                prec, rec, _ = precision_recall_curve(yt, ys)
                # interpolate to common recall grid
                rec_grid = np.linspace(0, 1, 200)
                prec_interp = np.interp(rec_grid, rec[::-1], prec[::-1])
                tissue_precs[tissue] = prec_interp
            if not tissue_precs: continue
            med_prec = np.median(list(tissue_precs.values()), axis=0)
            med_auprc = np.trapz(med_prec, rec_grid)
            ax.plot(rec_grid, med_prec, color=COLORS[model],
                    label=f'{MODEL_NAMES[model]} ({med_auprc:.3f})', lw=2)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_delta_dist(pairs, pos_scores, neg_scores, title, save_path=None):
    mp = [m for m in MODELS if m in pos_scores]
    fig, axes = plt.subplots(2, len(mp), figsize=(3 * len(mp), 6))
    if len(mp) == 1: axes = axes.reshape(-1, 1)
    for j, model in enumerate(mp):
        pv = np.array([pos_scores[model].get(k, np.nan) for k in pairs['pos_var_key']])
        nv = np.array([neg_scores[model].get(k, np.nan) for k in pairs['neg_var_key']])
        pv = pv[~np.isnan(pv)]
        nv = nv[~np.isnan(nv)]

        bins = np.linspace(0, max(pv.max(), nv.max()) * 1.05, 50)

        ax = axes[0, j]
        ax.hist(pv, bins=bins, color='#e74c3c', alpha=0.7, edgecolor='#c0392b')
        if len(pv): ax.axvline(np.median(pv), color='black', ls='--', lw=1.5)
        ax.set_title(f'{MODEL_NAMES[model]} pos')
        ax.set_xlabel('|delta|')
        ax.set_title(f'{MODEL_NAMES[model]} pos (n={len(pv):,})')
        ax.set_yscale('log')

        ax = axes[1, j]
        ax.hist(nv, bins=bins, color='#27ae60', alpha=0.7, edgecolor='#1e8449')
        if len(nv): ax.axvline(np.median(nv), color='black', ls='--', lw=1.5)
        ax.set_xlabel('|delta|')
        ax.set_title(f'(n={len(nv):,})')
        ax.set_yscale('log')

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_roc(pairs, pos_scores, neg_scores, title, save_path=None, annotate_youden=False):
    """ROC curves per model, optionally annotate Youden's J point"""
    single_tissue = 'tissue' not in pairs.columns
    mp = [m for m in MODELS if m in pos_scores]
    fig, ax = plt.subplots(figsize=(7, 6))

    for model in mp:
        if single_tissue:
            yt, ys, n = get_ys(pairs, pos_scores, neg_scores, model)
            if n < 10: continue
            fpr, tpr, thresholds = roc_curve(yt, ys)
            auroc_val = compute_auroc(yt, ys)
            ax.plot(fpr, tpr, color=COLORS[model], label=f'{MODEL_NAMES[model]} ({auroc_val:.3f})', lw=2)
            if annotate_youden:
                j_idx = np.argmax(tpr - fpr)
                ax.scatter(fpr[j_idx], tpr[j_idx], color=COLORS[model], s=80, zorder=15, edgecolors='black', linewidths=1.5)
                ax.annotate(f't={thresholds[j_idx]:.3f}',
                    (fpr[j_idx], tpr[j_idx]), textcoords='offset points',
                    xytext=(10, -15), fontsize=8, color=COLORS[model], fontweight='bold')
        else:
            tissue_thresholds = []
            tissue_tprs = {}
            for tissue in pairs['tissue'].unique():
                tp = pairs[pairs['tissue'] == tissue]
                yt, ys, n = get_ys(tp, pos_scores, neg_scores, model)
                if n < 10: continue
                fpr, tpr, thresholds = roc_curve(yt, ys)
                fpr_grid = np.linspace(0, 1, 200)
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tissue_tprs[tissue] = tpr_interp
                if annotate_youden:
                    j_idx = np.argmax(tpr - fpr)
                    tissue_thresholds.append(thresholds[j_idx])
            if not tissue_tprs: continue
            med_tpr = np.median(list(tissue_tprs.values()), axis=0)
            med_auroc = np.trapz(med_tpr, fpr_grid)
            ax.plot(fpr_grid, med_tpr, color=COLORS[model],
                    label=f'{MODEL_NAMES[model]} ({med_auroc:.3f})', lw=2)
            if annotate_youden and tissue_thresholds:
                med_thresh = np.median(tissue_thresholds)
                j_vals = med_tpr - fpr_grid
                j_idx = np.argmax(j_vals)
                ax.scatter(fpr_grid[j_idx], med_tpr[j_idx], color=COLORS[model],
                    s=80, zorder=15, edgecolors='black', linewidths=1.5)
                ax.annotate(f't={med_thresh:.3f}',
                    (fpr_grid[j_idx], med_tpr[j_idx]), textcoords='offset points',
                    xytext=(10, -15), fontsize=8, color=COLORS[model], fontweight='bold')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_distance_auprc(pairs, pos_scores, neg_scores, distances, title, save_path=None):
    """line plot of AUPRC at cumulative distance thresholds with count bars"""
    thresholds = [50, 100, 200, 500, 2000, 10000]
    labels = ['<=50', '<=100', '<=200', '<=500', '<=2k', '<=10k']

    pos_dist = distances[distances['type'] == 'pos'].set_index('var_key')['splice_dist']
    p_dists = pairs['pos_var_key'].map(pos_dist)

    mp = [m for m in MODELS if m in pos_scores]
    single_tissue = 'tissue' not in pairs.columns
    x = np.arange(len(thresholds))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax2 = ax.twinx()

    pos_counts, neg_counts = [], []
    bw = 0.25
    for d in thresholds:
        mask = (p_dists <= d)
        sub = pairs[mask]
        pos_counts.append(sub['pos_var_key'].nunique())
        neg_counts.append(sub['neg_var_key'].nunique())
    ax2.bar(x - bw/2, pos_counts, bw, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.bar(x + bw/2, neg_counts, bw, color='#606060', alpha=0.4, label='neg variants')
    ax2.set_ylabel('unique variants', color='gray', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='gray')

    for model in mp:
        auprcs = []
        for d in thresholds:
            mask = (p_dists <= d)
            tp = pairs[mask]
            if len(tp) < 20:
                auprcs.append(np.nan)
                continue
            if single_tissue:
                yt, ys, n = get_ys(tp, pos_scores, neg_scores, model)
                auprcs.append(compute_auprc(yt, ys) if n >= 10 else np.nan)
            else:
                tissue_auprcs = []
                for tissue in tp['tissue'].unique():
                    tt = tp[tp['tissue'] == tissue]
                    yt, ys, n = get_ys(tt, pos_scores, neg_scores, model)
                    if n >= 10:
                        tissue_auprcs.append(compute_auprc(yt, ys))
                auprcs.append(np.median(tissue_auprcs) if tissue_auprcs else np.nan)

        ax.plot(x, auprcs, marker='o', color=COLORS[model], label=MODEL_NAMES[model],
                lw=2, markersize=6, zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Splice Distance (bp)')
    ax.set_ylabel('AUPRC (tissue median)' if not single_tissue else 'AUPRC')
    all_y = [v for line in ax.get_lines() for v in line.get_ydata() if np.isfinite(v)]
    if all_y:
        ax.set_ylim(np.floor(min(all_y) * 10) / 10, np.ceil(max(all_y) * 10) / 10)
    ax.grid(alpha=0.3, zorder=0)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels1) + len(labels2), fontsize=9, frameon=False)
    fig.suptitle(title, y=1.08)
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_distance_auprc_unique(pairs, pos_scores, neg_scores, distances, title, save_path=None):
    """AUPRC by distance with unique positives — each pos used once, one neg per pos"""
    thresholds = [50, 100, 200, 500, 2000, 10000]
    labels = ['<=50', '<=100', '<=200', '<=500', '<=2k', '<=10k']

    pos_dist = distances[distances['type'] == 'pos'].set_index('var_key')['splice_dist']
    p_dists = pairs['pos_var_key'].map(pos_dist)

    mp = [m for m in MODELS if m in pos_scores]
    x = np.arange(len(thresholds))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax2 = ax.twinx()

    pos_counts, neg_counts = [], []
    bw = 0.25
    for d in thresholds:
        mask = (p_dists <= d)
        sub = pairs[mask]
        pos_counts.append(sub['pos_var_key'].nunique())
        neg_counts.append(sub['neg_var_key'].nunique())
    ax2.bar(x - bw/2, pos_counts, bw, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.bar(x + bw/2, neg_counts, bw, color='#606060', alpha=0.4, label='neg variants')
    ax2.set_ylabel('unique variants', color='gray', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='gray')

    for model in mp:
        auprcs = []
        for d in thresholds:
            mask = (p_dists <= d)
            unique_pairs = pairs[mask].drop_duplicates(subset='pos_var_key', keep='first')
            if len(unique_pairs) < 20:
                auprcs.append(np.nan)
                continue
            yt, ys, n = get_ys(unique_pairs, pos_scores, neg_scores, model)
            auprcs.append(compute_auprc(yt, ys) if n >= 10 else np.nan)

        ax.plot(x, auprcs, marker='o', color=COLORS[model], label=MODEL_NAMES[model],
                lw=2, markersize=6, zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Splice Distance (bp)')
    ax.set_ylabel('AUPRC (unique positives)')
    all_y = [v for line in ax.get_lines() for v in line.get_ydata() if np.isfinite(v)]
    if all_y:
        ax.set_ylim(np.floor(min(all_y) * 10) / 10, np.ceil(max(all_y) * 10) / 10)
    ax.grid(alpha=0.3, zorder=0)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels1) + len(labels2), fontsize=9, frameon=False)
    fig.suptitle(title, y=1.08)
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_wins(metrics, title, save_path=None, win_margin=0.02):
    """2-row layout: top=bar chart of wins, bottom=deficit scatter"""
    mp = [m for m in MODELS if m in metrics['model'].values]
    tissues = metrics['tissue'].unique()
    n_tissues = len(tissues)

    wins = {m: 0 for m in mp}
    deficits = {m: [] for m in mp}
    for tissue in tissues:
        t_df = metrics[metrics['tissue'] == tissue]
        best = t_df['auprc'].max()
        for _, row in t_df.iterrows():
            deficit = best - row['auprc']
            deficits[row['model']].append(deficit)
            if deficit <= win_margin:
                wins[row['model']] += 1

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1.5]})

    ax = axes[0]
    for i, model in enumerate(mp):
        ax.bar(i, wins[model], color=COLORS[model], alpha=0.8)
        ax.text(i, wins[model] + 0.5, str(wins[model]), ha='center', fontsize=10)
    ax.set_xticks(range(len(mp)))
    ax.set_xticklabels([MODEL_NAMES[m] for m in mp])
    ax.set_ylabel(f'Tissues within {win_margin} of best')
    ax.set_title(f'{title} — Wins (of {n_tissues} tissues)')

    ax = axes[1]
    for i, model in enumerate(mp):
        d = np.array(deficits[model])
        ax.scatter(np.random.normal(i, 0.08, len(d)), d, alpha=0.5,
                   color=COLORS[model], s=20)
        med = np.median(d)
        ax.hlines(med, i - 0.3, i + 0.3, colors=COLORS[model], lw=2.5)
        ax.text(i + 0.35, med, f'{med:.3f}', va='center', fontsize=9)
    ax.set_xticks(range(len(mp)))
    ax.set_xticklabels([MODEL_NAMES[m] for m in mp])
    ax.set_ylabel('Deficit from best AUPRC')
    ax.set_title('Per-Tissue Deficit Distribution')
    ax.axhline(win_margin, color='gray', ls='--', alpha=0.5)

    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_per_tissue_stacked(pairs, title, save_path=None):
    """stacked bar chart: tissue-specific vs shared variants per tissue"""
    tc = compute_tissue_counts(pairs)
    tc = tc.sort_values('n_pos', ascending=True)
    tissues = tc['tissue'].values
    x = np.arange(len(tissues))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.bar(x, tc['n_pos_unique'], color='#66c2a5', alpha=0.8, label='tissue-specific')
    ax.bar(x, tc['n_pos_shared'], bottom=tc['n_pos_unique'], color='#8da0cb', alpha=0.8, label='shared')
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Positive variants')
    ax.legend(fontsize=8)
    format_axis_plain(ax, axis='y')

    ax = axes[1]
    ax.bar(x, tc['n_neg_unique'], color='#fc8d62', alpha=0.8, label='tissue-specific')
    ax.bar(x, tc['n_neg_shared'], bottom=tc['n_neg_unique'], color='#e78ac3', alpha=0.8, label='shared')
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Negative variants')
    ax.legend(fontsize=8)
    format_axis_plain(ax, axis='y')

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


# In[3]:


# additional helper functions

def plot_tier_dist(tiers, title, save_path=None, n_pos_total=None):
    """bar chart of matching tier distribution"""
    tiers = tiers[tiers['tier'] != 'unmatched'].copy()
    tiers['tier'] = tiers['tier'].astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [DATA_COLORS.get(f'tier{t}', 'gray') for t in tiers['tier']]
    bars = ax.bar(tiers['tier'], tiers['count'], color=colors, alpha=0.8)
    for bar, v in zip(bars, tiers['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tiers['count'])*0.02,
                f'{int(v):,}', ha='center', fontsize=9)
    ax.set_xlabel('Tier')
    ax.set_ylabel('Matched pairs')
    format_axis_plain(ax, axis='y')

    subtitle = title
    if n_pos_total is not None:
        n_matched = int(tiers['count'].sum())
        subtitle += f'\n({n_matched:,}/{n_pos_total:,} matched = {n_matched/n_pos_total:.1%})'
    ax.set_title(subtitle)

    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def plot_swarm(metrics, title, save_path=None):
    """swarm plot of per-tissue AUPRC by model"""
    mp = [m for m in MODELS if m in metrics['model'].values]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(mp):
        data = metrics[metrics['model'] == model]['auprc']
        ax.scatter(np.random.normal(i, 0.08, len(data)), data, alpha=0.5,
                   color=COLORS[model], s=25)
        med = data.median()
        ax.hlines(med, i - 0.3, i + 0.3, colors=COLORS[model], lw=2.5)
        ax.text(i + 0.35, med, f'{med:.3f}', va='center', fontsize=9)

    ax.set_xticks(range(len(mp)))
    ax.set_xticklabels([MODEL_NAMES[m] for m in mp], rotation=45, ha='right')
    ax.set_ylabel('AUPRC per tissue')
    ax.set_title(title)
    all_vals = metrics[metrics['model'].isin(mp)]['auprc']
    buf = (all_vals.max() - all_vals.min()) * 0.1
    ax.set_ylim(max(0, all_vals.min() - buf), min(1, all_vals.max() + buf))
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path: savefig(save_path)
    plt.show()
    plt.close()


def compute_pooled_metrics(pairs, pos_scores, neg_scores):
    """compute pooled AUPRC/AUROC across all pairs (for single-tissue datasets)"""
    results = {}
    for model in pos_scores:
        if model not in neg_scores:
            continue
        yt, ys, n = get_ys(pairs, pos_scores, neg_scores, model)
        if n < 10:
            continue
        results[model] = {
            'auprc': compute_auprc(yt, ys),
            'auroc': compute_auroc(yt, ys),
            'n': n,
        }
    return results


# 
# # Txrevise Analysis
# 
# **Positives:** For each GTEx txrevise sQTL credible set (eQTL Catalogue), the variant with the highest posterior inclusion probability (PIP) is selected. Credible sets are filtered to "contained" events only (within gene body), max PIP >= 0.9, SNVs only, and splice distance <= 10 kb. Splice distance = distance to nearest exon boundary within the same gene.
# 
# **Negatives:** Variants from GTEx gene expression (GE) summary statistics with PIP < 0.01, SNVs only, splice distance <= 10 kb. Splice distance = distance to nearest exon boundary within the same gene (per-gene, not chromosome-wide). Each positive is matched 1:1 to a negative within tissue using a 4-tier cascade: same gene + matching alleles, same gene, same TPM bin + matching alleles, same TPM bin. Within each tier, the negative with the closest splice distance is selected.

# ## Txrevise QC
# 
# These figures show the raw input data before matching.
# 
# **GE Sumstats Per-Tissue Counts:** Per-tissue counts from gene expression summary statistics: total rows, unique variants, unique genes.
# 
# 
# **Credible Set Per-Tissue Counts:** Per-tissue counts from txrevise credible sets: total rows, unique variants, unique genes, and high-PIP variants.
# 
# **PIP Distribution:** Distribution of posterior inclusion probabilities across all variants in credible sets. Linear and log scale histograms with threshold lines at PIP = 0.9 (positive cutoff) and PIP = 0.01 (negative cutoff).
# 
# **GTEx TPM Distribution:** Distribution of gene expression levels from GTEx v8 median TPM.
# 
# **PIP Sum Per Credible Set:** Sum of PIP values per credible set. Full distribution on log scale with median annotated, and a zoomed view of the 0.8-1.1 range.
# 
# **CS Size vs Max PIP:** Relationship between credible set size and maximum PIP.

# In[4]:


# credible set stats
txrevise_dir = DATA_BASE / 'txrevise'
credset_stats = pd.read_csv(txrevise_dir / 'credset_stats.csv')
print(f'credset stats: {len(credset_stats)} tissues')
print(credset_stats.describe().round(0))

metrics_cfg = [
    ('n_rows', 'Total Rows'),
    ('n_variants', 'Unique Variants'),
    ('n_genes', 'Unique Genes'),
]
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
for ax, (col, label) in zip(axes, metrics_cfg):
    ss = credset_stats.sort_values(col, ascending=True)
    ax.bar(range(len(ss)), ss[col], color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(ss)))
    ax.set_xticklabels(ss['tissue'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(label)
    format_axis_plain(ax, axis='y')
plt.suptitle('Txrevise Credible Sets Per Tissue', y=1.01)
plt.tight_layout()
savefig(FIG_BASE / 'txrevise' / 'qc' / '07_credsets_counts')
plt.show()
plt.close()


# In[5]:


# PIP distribution
pip_values = pd.read_csv(txrevise_dir / 'pip_values.csv')
print(f'pip values: {len(pip_values):,} rows')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(pip_values['pip'], bins=100, color='steelblue', alpha=0.8, edgecolor='white')
axes[0].axvline(0.9, color='#e74c3c', ls='--', lw=2, label='PIP=0.9')
axes[0].axvline(0.01, color='#3498db', ls='--', lw=2, label='PIP=0.01')
axes[0].set_xlabel('PIP'); axes[0].set_ylabel('Count')
axes[0].set_title('PIP Distribution'); axes[0].legend()

axes[1].hist(pip_values['pip'], bins=100, color='steelblue', alpha=0.8, edgecolor='white')
axes[1].set_yscale('log')
axes[1].axvline(0.9, color='#e74c3c', ls='--', lw=2)
axes[1].axvline(0.01, color='#3498db', ls='--', lw=2)
axes[1].set_xlabel('PIP'); axes[1].set_ylabel('Count (log)')
axes[1].set_title('PIP Distribution (log scale)')
plt.suptitle('txrevise: PIP Distribution', y=1.02)
plt.tight_layout()
savefig(FIG_BASE / 'txrevise' / 'qc' / '08_credsets_pip_dist')
plt.show()
plt.close()


# In[6]:


# GTEx TPM distribution
tpm_path = RAW['tpm_gtex']
if tpm_path.exists():
    tpm = pd.read_csv(tpm_path, sep='\t', skiprows=2)
    tpm_vals = tpm.iloc[:, 2:].values.flatten()
    tpm_vals = tpm_vals[tpm_vals > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(np.log10(tpm_vals), bins=100, color='steelblue', alpha=0.8, edgecolor='white')
    axes[0].axvline(0, color='#e74c3c', ls='--', lw=2, label='TPM=1')
    axes[0].set_xlabel('log10(TPM)'); axes[0].set_ylabel('Count')
    axes[0].set_title('GTEx TPM Distribution'); axes[0].legend()

    genes_per_tissue = (tpm.iloc[:, 2:] > 1).sum().sort_values(ascending=False)
    med_genes = genes_per_tissue.median()
    axes[1].bar(range(len(genes_per_tissue)), genes_per_tissue.values,
                color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].axhline(med_genes, color='#e74c3c', ls='--', lw=2,
                     label=f'median={med_genes:,.0f}')
    axes[1].set_xticks(range(len(genes_per_tissue)))
    axes[1].set_xticklabels(genes_per_tissue.index, rotation=45, ha='right', fontsize=5)
    axes[1].set_ylabel('Genes with TPM > 1')
    axes[1].set_title('Expressed Genes Per Tissue')
    axes[1].legend(fontsize=8)
    format_axis_plain(axes[1], axis='y')
    plt.suptitle('GTEx v8 TPM reference', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'txrevise' / 'qc' / '05_gtex_tpm')
    plt.show()
    plt.close()
    del tpm, tpm_vals
else:
    print(f'TPM file not found: {tpm_path}')


# In[7]:


# GE sumstats per-tissue counts
ge_counts_path = txrevise_dir / 'sumstats_ge_counts.csv'
if ge_counts_path.exists():
    ge_counts = pd.read_csv(ge_counts_path)
    n_tissues = len(ge_counts)
    print(f'GE sumstats: {n_tissues} tissues')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    metrics = [
        ('n_variants', 'Variants'),
        ('n_phenos', 'Phenotypes'),
        ('n_var_pheno', 'Variant-Phenotype Pairs'),
    ]
    for ax, (col, title) in zip(axes, metrics):
        if col not in ge_counts.columns:
            ax.text(0.5, 0.5, f'{col} not available', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        sorted_df = ge_counts.sort_values(col, ascending=False)
        x = np.arange(len(sorted_df))
        ax.bar(x, sorted_df[col].values, color='steelblue', alpha=0.8, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_df['tissue'].values, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('count')
        ax.set_title(title)
        ax.set_xlim(-0.5, len(sorted_df) - 0.5)
        format_axis_plain(ax, axis='y')
    plt.suptitle(f'GE Sumstats per Tissue ({n_tissues} tissues)', fontsize=14, y=1.01)
    plt.tight_layout()
    savefig(FIG_BASE / 'txrevise' / 'qc' / '03_sumstats_ge')
    plt.show()
    plt.close()
else:
    print(f'not found: {ge_counts_path} — run gen_qc_data.py first')


# In[8]:


# PIP sum per credible set
cs_path = txrevise_dir / 'cs_summary.csv'
if cs_path.exists():
    cs = pd.read_csv(cs_path)
    n_cs = len(cs)
    pip_sums = cs['pip_sum'].values
    med = np.median(pip_sums)
    mx = pip_sums.max()
    print(f'txrevise CS summary: {n_cs:,} credible sets, median pip_sum={med:.4f}, max={mx:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # full distribution
    ax = axes[0]
    ax.hist(pip_sums, bins=100, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('sum(PIP) per credible set')
    ax.set_ylabel('count (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'PIP sum per CS (n={n_cs:,})')
    # zoomed 0.8-1.1
    ax = axes[1]
    in_range = pip_sums[(pip_sums > 0.8) & (pip_sums < 1.1)]
    ax.hist(in_range, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('sum(PIP) per credible set')
    ax.set_ylabel('count (log scale)')
    ax.set_yscale('log')
    ax.set_xlim(0.8, 1.1)
    ax.set_title(f'PIP sum (zoomed 0.8-1.1, n={len(in_range):,})')
    plt.suptitle('txrevise: PIP Sum per Credible Set', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'txrevise' / 'qc' / '09_credsets_pip_sum')
    plt.show()
    plt.close()
    # diagnostic
    n_over_1 = (pip_sums > 1.0).sum()
    if n_over_1 > 0:
        print(f'  [WARNING] {n_over_1:,} CS have PIP sum > 1.0, max={mx:.4f}')
    else:
        print(f'  [OK] all CS have PIP sum <= 1.0 (max: {mx:.4f})')
else:
    print(f'not found: {cs_path} — run gen_qc_data.py first')


# In[9]:


# CS size vs max PIP
if cs_path.exists():
    cs_sizes = cs['cs_size'].values
    max_pips = cs['max_pip'].values
    n_cs = len(cs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # left: hexbin density
    ax = axes[0]
    hb = ax.hexbin(cs_sizes, max_pips, gridsize=50, cmap='viridis', mincnt=1, bins='log')
    ax.set_xlabel('variants per credible set')
    ax.set_ylabel('max PIP in credible set')
    ax.set_title(f'CS size vs max PIP (n={n_cs:,})')
    ax.axhline(0.9, color='#e74c3c', ls='--', lw=1.5, alpha=0.8, label='PIP=0.9')
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(hb, ax=ax, label='count (log)')
    # right: box plot by size bins
    ax = axes[1]
    size_bins = [1, 2, 3, 5, 10, 20, 50, 100, np.inf]
    bin_labels = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
    size_bin_idx = np.digitize(cs_sizes, size_bins[1:])
    boxplot_data = []
    valid_labels = []
    for i, label in enumerate(bin_labels):
        mask = size_bin_idx == i
        if mask.sum() > 0:
            boxplot_data.append(max_pips[mask])
            valid_labels.append(f'{label}\n(n={mask.sum():,})')
    bp = ax.boxplot(boxplot_data, labels=valid_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax.axhline(0.9, color='#e74c3c', ls='--', lw=1.5, alpha=0.8, label='PIP=0.9')
    ax.set_xlabel('credible set size')
    ax.set_ylabel('max PIP in credible set')
    ax.set_title('Max PIP by CS size')
    ax.legend(loc='lower right', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle('txrevise: CS Size vs Maximum PIP', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'txrevise' / 'qc' / '10_credsets_size_vs_pip')
    plt.show()
    plt.close()


# In[10]:


# load matching data
pairs = pd.read_csv(txrevise_dir / 'pairs.csv')
tiers = pd.read_csv(txrevise_dir / 'tiers.csv')
tissue_stats = pd.read_csv(txrevise_dir / 'tissue_stats.csv')
distances = pd.read_csv(txrevise_dir / 'distances.csv')

n_pos_unique = pairs['pos_var_key'].nunique()
n_pos_pairs = len(pairs)
n_tissues = pairs['tissue'].nunique()
neg_unique = pairs['neg_var_key'].nunique()
print(f'positives: {n_pos_unique:,} unique variants x {n_tissues} tissues = {n_pos_pairs:,} variant-tissue pairs')
print(f'negatives (GE source): {neg_unique:,} unique variants, {len(pairs):,} variant-tissue pairs')


# ## Txrevise Matching
# 
# 4-tier matching cascade (GE negatives). Splice distance = distance to nearest exon boundary within the same gene. Each positive attempts tier 1 first, falls through to tier 2 if no match, etc.
# 
# | Tier | Gene | Allele | Description |
# |------|------|--------|-------------|
# | 1 | same gene | same ref/alt | Best: same genomic context + same mutation type |
# | 2 | same gene | any | Same gene, different alleles |
# | 3 | same TPM bin | same ref/alt | Similar expression level + same mutation type |
# | 4 | same TPM bin | any | Loosest: just similar expression level |

# In[11]:


# txrevise matching - tier distribution
n_pos = int(tissue_stats['n_pos'].sum()) if tissue_stats is not None else None
plot_tier_dist(tiers, 'Txrevise (GE) - Tier Distribution',
    FIG_BASE / 'txrevise' / 'matching' / 'ge_set1' / '01_tier_distribution',
    n_pos_total=n_pos)


# In[12]:


# txrevise matching - splice distance + per-tissue stacked
fig, ax = plt.subplots(figsize=(8, 4))
pos_d = distances[distances['type'] == 'pos']['splice_dist']
neg_d = distances[distances['type'] == 'neg']['splice_dist']
bins = np.logspace(0, 4, 50)
ax.hist(pos_d, bins=bins, alpha=0.7, label=f'pos (n={len(pos_d):,})', color=DATA_COLORS['pos'])
ax.hist(neg_d, bins=bins, alpha=0.7, label=f'neg (n={len(neg_d):,})', color=DATA_COLORS['neg'])
ax.set_xscale('log'); ax.set_xlabel('Splice distance (bp)')
ax.set_ylabel('Count'); ax.set_title('Txrevise (GE) - Splice Distance'); ax.legend()
plt.tight_layout()
savefig(FIG_BASE / 'txrevise' / 'matching' / 'ge_set1' / '05_distance_dist')
plt.show()
plt.close()

plot_per_tissue_stacked(pairs,
    'Txrevise (GE) - Per Tissue (tissue-specific vs shared)',
    FIG_BASE / 'txrevise' / 'matching' / 'ge_set1' / '06_per_tissue')


# ## Txrevise Results
# 
# Model performance using GE negatives. We compute delta scores (max |alt - ref|) for each variant and evaluate separation between positives and negatives.
# 

# In[13]:


# load scores
score_dir = txrevise_dir / 'scores'
pos_scores = load_scores(score_dir, 'pos', primary_only=args.nohbar)
neg_scores = load_scores(score_dir, 'neg', primary_only=args.nohbar)
print('GE models:', list(pos_scores.keys()))


# In[14]:


plot_delta_dist(pairs, pos_scores, neg_scores,
    'Txrevise (GE neg) - Delta Distributions',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '00a_delta_distributions')


# In[15]:


metrics = compute_tissue_metrics(pairs, pos_scores, neg_scores)
n_tissues = metrics['tissue'].nunique()
n_models = metrics['model'].nunique()
print(f'GE: {n_models} models x {n_tissues} tissues = {len(metrics)} tissue-model combos')

plot_swarm(metrics, 'Txrevise (GE neg) - Per-tissue AUPRC',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '01_swarm')

if not args.nohbar:
    plot_median_auprc_hbar(metrics, 'Txrevise (GE neg) - All Output Heads',
        FIG_BASE / 'txrevise' / 'results' / 'set1' / '01b_hbar')


# In[16]:


plot_pr(pairs, pos_scores, neg_scores,
    'Txrevise (GE neg) - PR Curves',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '01b_pr_curves')


# In[17]:


plot_roc(pairs, pos_scores, neg_scores,
    'Txrevise (GE neg) - ROC Curves',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '01d_roc_curves')


# In[18]:


plot_distance_auprc(pairs, pos_scores, neg_scores, distances,
    'Txrevise (GE neg) - AUPRC by Splice Distance',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '02_distance')


# In[19]:


plot_distance_auprc_unique(pairs, pos_scores, neg_scores, distances,
    'Txrevise (GE neg) - AUPRC by Distance (unique positives)',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '02b_distance_unique')


# ## Txrevise pip90 vs pip50 Verification
#
# Data integrity check: load txrevise_pip50, filter to PIP >= 0.9, and compare counts + AUPRCs
# against the pip90 dataset. Both datasets were generated independently — if the pipeline is
# correct, the pip90 subset of pip50 should produce identical results.

# In[ ]:


txrevise_pip50_dir = DATA_BASE / 'txrevise_pip50'
if (txrevise_pip50_dir / 'pairs.csv').exists():
    pairs_tx50 = pd.read_csv(txrevise_pip50_dir / 'pairs.csv')
    score_dir_tx50 = txrevise_pip50_dir / 'scores'
    pos_scores_tx50 = load_scores(score_dir_tx50, 'pos')
    neg_scores_tx50 = load_scores(score_dir_tx50, 'neg')

    print('=' * 70)
    print('TXREVISE PIP90 vs PIP50 VERIFICATION')
    print('=' * 70)

    # pip50 full counts
    print(f'\npip50 (all):  {pairs_tx50["pos_var_key"].nunique():,} pos, {pairs_tx50["neg_var_key"].nunique():,} neg unique, {len(pairs_tx50):,} pairs')

    # filter to pip >= 0.9
    pairs_tx50_90 = pairs_tx50.query('pos_pip >= 0.9')
    print(f'pip50 @0.9:   {pairs_tx50_90["pos_var_key"].nunique():,} pos, {pairs_tx50_90["neg_var_key"].nunique():,} neg unique, {len(pairs_tx50_90):,} pairs')
    print(f'pip90 (orig): {pairs["pos_var_key"].nunique():,} pos, {pairs["neg_var_key"].nunique():,} neg unique, {len(pairs):,} pairs')

    # compare pos variant sets
    pip90_pos = set(pairs['pos_var_key'])
    pip50_90_pos = set(pairs_tx50_90['pos_var_key'])
    shared = pip90_pos & pip50_90_pos
    only_90 = pip90_pos - pip50_90_pos
    only_50 = pip50_90_pos - pip90_pos
    print(f'\npos variant overlap: {len(shared):,} shared, {len(only_90):,} only-in-pip90, {len(only_50):,} only-in-pip50')

    # compare neg variant sets
    pip90_neg = set(pairs['neg_var_key'])
    pip50_90_neg = set(pairs_tx50_90['neg_var_key'])
    shared_neg = pip90_neg & pip50_90_neg
    only_90_neg = pip90_neg - pip50_90_neg
    only_50_neg = pip50_90_neg - pip90_neg
    print(f'neg variant overlap: {len(shared_neg):,} shared, {len(only_90_neg):,} only-in-pip90, {len(only_50_neg):,} only-in-pip50')

    # compare AUPRCs per model
    mp = [m for m in MODELS if m in pos_scores and m in neg_scores and m in pos_scores_tx50 and m in neg_scores_tx50]
    print(f'\n{"Model":<25s} {"pip90 AUPRC":>12s} {"pip50@0.9 AUPRC":>15s} {"diff":>8s}')
    print(f'{"-"*25} {"-"*12} {"-"*15} {"-"*8}')
    for m in mp:
        # pip90 tissue-median
        auprcs_90 = []
        for tissue in pairs['tissue'].unique():
            tp = pairs[pairs['tissue'] == tissue]
            yt, ys, n = get_ys(tp, pos_scores, neg_scores, m)
            if n >= 10:
                auprcs_90.append(compute_auprc(yt, ys))
        med_90 = np.median(auprcs_90) if auprcs_90 else np.nan

        # pip50@0.9 tissue-median
        auprcs_50 = []
        for tissue in pairs_tx50_90['tissue'].unique():
            tp = pairs_tx50_90[pairs_tx50_90['tissue'] == tissue]
            yt, ys, n = get_ys(tp, pos_scores_tx50, neg_scores_tx50, m)
            if n >= 10:
                auprcs_50.append(compute_auprc(yt, ys))
        med_50 = np.median(auprcs_50) if auprcs_50 else np.nan

        diff = med_50 - med_90
        print(f'{MODEL_NAMES.get(m, m):<25s} {med_90:12.4f} {med_50:15.4f} {diff:+8.4f}')

    # negative match degradation
    if 'neg_var_key_ideal' in pairs_tx50.columns:
        mismatch = (pairs_tx50['neg_var_key'] != pairs_tx50['neg_var_key_ideal']).sum()
        print(f'\npip50 negative degradation: {mismatch:,}/{len(pairs_tx50):,} ({100*mismatch/len(pairs_tx50):.1f}%) got non-ideal negative')

    del pairs_tx50, pairs_tx50_90, pos_scores_tx50, neg_scores_tx50
else:
    print('txrevise_pip50: pairs.csv not found, skipping verification')


#
# # Txrevise Comprehensive GTF Analysis
#
# Rerun of the txrevise benchmark using GENCODE v39 **comprehensive** annotation
# (`gencode.v39.annotation.gtf.gz`) instead of the basic subset
# (`gencode.v39.basic.annotation.gtf.gz`). The eQTL Catalogue itself uses the
# comprehensive GTF for txrevise event quantification, so this is the matched
# annotation. The comprehensive set has ~250k transcripts vs ~114k basic,
# giving more exon boundaries per gene and therefore potentially different
# splice distances, which affects which variants pass the 10 kb filter and
# which negatives get picked by the 4-tier matching cascade.
#
# SPLAIRE scores may not be ready yet (long running job). Any missing model
# scores are skipped automatically — rerun this section when scores are ready.

# In[ ]:


txrevise_comp_dir = DATA_BASE / 'txrevise_comp'
if (txrevise_comp_dir / 'pairs.csv').exists():
    print('\n' + '=' * 70)
    print('TXREVISE COMPREHENSIVE GTF')
    print('=' * 70)

    # --- QC: credible set stats per tissue ---
    credset_stats_txc = pd.read_csv(txrevise_comp_dir / 'credset_stats.csv')
    print(f'credset stats: {len(credset_stats_txc)} tissues')

    metrics_cfg = [
        ('n_rows', 'Total Rows'),
        ('n_variants', 'Unique Variants'),
        ('n_genes', 'Unique Genes'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, (col, label) in zip(axes, metrics_cfg):
        if col not in credset_stats_txc.columns:
            ax.set_visible(False)
            continue
        ss = credset_stats_txc.sort_values(col, ascending=True)
        ax.bar(range(len(ss)), ss[col], color='steelblue', alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(ss)))
        ax.set_xticklabels(ss['tissue'], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(label)
        format_axis_plain(ax, axis='y')
    plt.suptitle('Txrevise (comprehensive GTF) Credible Sets Per Tissue', y=1.01)
    plt.tight_layout()
    savefig(FIG_BASE / 'txrevise_comp' / 'qc' / '07_credsets_counts')
    plt.show()
    plt.close()

    # --- load matching data ---
    pairs_txc = pd.read_csv(txrevise_comp_dir / 'pairs.csv')
    tiers_txc = pd.read_csv(txrevise_comp_dir / 'tiers.csv')
    tissue_stats_txc = pd.read_csv(txrevise_comp_dir / 'tissue_stats.csv')
    distances_txc = pd.read_csv(txrevise_comp_dir / 'distances.csv')

    n_pos_uniq = pairs_txc['pos_var_key'].nunique()
    n_neg_uniq = pairs_txc['neg_var_key'].nunique()
    n_tissues_txc = pairs_txc['tissue'].nunique()
    print(f'positives: {n_pos_uniq:,} unique x {n_tissues_txc} tissues = {len(pairs_txc):,} pairs')
    print(f'negatives: {n_neg_uniq:,} unique')

    # direct count comparison with basic-GTF run
    if 'pairs' in dir():
        n_pos_basic = pairs['pos_var_key'].nunique()
        n_neg_basic = pairs['neg_var_key'].nunique()
        print(f'vs basic GTF: pos {n_pos_basic:,} -> {n_pos_uniq:,} ({n_pos_uniq - n_pos_basic:+,})')
        print(f'              neg {n_neg_basic:,} -> {n_neg_uniq:,} ({n_neg_uniq - n_neg_basic:+,})')
        print(f'              pairs {len(pairs):,} -> {len(pairs_txc):,} ({len(pairs_txc) - len(pairs):+,})')

    # --- matching: tier distribution ---
    n_pos_total = int(tissue_stats_txc['n_pos'].sum())
    plot_tier_dist(tiers_txc, 'Txrevise comp GTF - Tier Distribution',
        FIG_BASE / 'txrevise_comp' / 'matching' / 'ge_set1' / '01_tier_distribution',
        n_pos_total=n_pos_total)

    # --- matching: splice distance hist ---
    fig, ax = plt.subplots(figsize=(8, 4))
    pos_d = distances_txc[distances_txc['type'] == 'pos']['splice_dist']
    neg_d = distances_txc[distances_txc['type'] == 'neg']['splice_dist']
    bins = np.logspace(0, 4, 50)
    ax.hist(pos_d, bins=bins, alpha=0.7, label=f'pos (n={len(pos_d):,})', color=DATA_COLORS['pos'])
    ax.hist(neg_d, bins=bins, alpha=0.7, label=f'neg (n={len(neg_d):,})', color=DATA_COLORS['neg'])
    ax.set_xscale('log'); ax.set_xlabel('Splice distance (bp)')
    ax.set_ylabel('Count'); ax.set_title('Txrevise comp GTF - Splice Distance'); ax.legend()
    plt.tight_layout()
    savefig(FIG_BASE / 'txrevise_comp' / 'matching' / 'ge_set1' / '05_distance_dist')
    plt.show()
    plt.close()

    # --- results: load scores ---
    score_dir_txc = txrevise_comp_dir / 'scores'
    pos_scores_txc = load_scores(score_dir_txc, 'pos', primary_only=args.nohbar)
    neg_scores_txc = load_scores(score_dir_txc, 'neg', primary_only=args.nohbar)
    # filter to intersection — splaire may only be scored for one side while
    # the other is still running; plot funcs assume symmetric dicts
    common_models = set(pos_scores_txc.keys()) & set(neg_scores_txc.keys())
    pos_scores_txc = {m: d for m, d in pos_scores_txc.items() if m in common_models}
    neg_scores_txc = {m: d for m, d in neg_scores_txc.items() if m in common_models}
    txc_models_loaded = sorted(common_models)
    print(f'models loaded: {txc_models_loaded}')
    missing = [m for m in MODELS if m not in txc_models_loaded]
    if missing:
        print(f'  missing (will be skipped): {missing}')

    # --- results: delta distributions ---
    plot_delta_dist(pairs_txc, pos_scores_txc, neg_scores_txc,
        'Txrevise comp GTF - Delta Distributions',
        FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '00a_delta_distributions')

    # --- results: per-tissue AUPRC swarm ---
    metrics_txc = compute_tissue_metrics(pairs_txc, pos_scores_txc, neg_scores_txc)
    print(f'tissue-model combos: {len(metrics_txc)}')
    plot_swarm(metrics_txc, 'Txrevise comp GTF - Per-tissue AUPRC',
        FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '01_swarm')

    if not args.nohbar:
        plot_median_auprc_hbar(metrics_txc, 'Txrevise comp GTF - All Output Heads',
            FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '01b_hbar')

    # --- results: PR / ROC curves ---
    plot_pr(pairs_txc, pos_scores_txc, neg_scores_txc,
        'Txrevise comp GTF - PR Curves',
        FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '01b_pr_curves')

    plot_roc(pairs_txc, pos_scores_txc, neg_scores_txc,
        'Txrevise comp GTF - ROC Curves',
        FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '01d_roc_curves')

    # --- results: AUPRC by splice distance ---
    plot_distance_auprc(pairs_txc, pos_scores_txc, neg_scores_txc, distances_txc,
        'Txrevise comp GTF - AUPRC by Splice Distance',
        FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '02_distance')

    plot_distance_auprc_unique(pairs_txc, pos_scores_txc, neg_scores_txc, distances_txc,
        'Txrevise comp GTF - AUPRC by Distance (unique positives)',
        FIG_BASE / 'txrevise_comp' / 'results' / 'set1' / '02b_distance_unique')
else:
    print('txrevise_comp: pairs.csv not found, skipping comprehensive GTF analysis')


#
# # Leafcutter Analysis
#
# **Data source:** leafcutter_pip50 (PIP >= 0.5 matching). The pip90 dataset has a known bug
# (positives matched in wrong tissues) and is not used. Results filter to PIP >= 0.9 post-hoc.
#
# **Positives:** Variants from GTEx leafcutter sQTL fine-mapping credible sets (eQTL Catalogue). For each credible set, the variant with highest PIP is selected as the positive. Matching includes all variants with PIP >= 0.5. Results filter to PIP >= 0.9. Introns must fall within a gene body (intergenic introns dropped). Splice distance = distance to the specific intron the variant was tested against.
#
# **Negatives:** Variants from GTEx gene expression (GE) summary statistics with PIP < 0.01, SNVs only, splice distance <= 10 kb. Splice distance = distance to nearest leafcutter intron boundary within the same gene. Intron boundaries are from Zenodo phenotype metadata (all leafcutter-detected introns per tissue, not just QTL-significant). Each positive is matched 1:1 to a negative within tissue using a 4-tier cascade: same gene + matching alleles, same gene, same TPM bin + matching alleles, same TPM bin.

# ## Leafcutter QC
#
# **Credible Set Per-Tissue Counts:** Per-tissue counts from leafcutter credible sets: total rows, unique variants, unique genes, and high-PIP variants.
#
# **PIP Distribution:** Distribution of posterior inclusion probabilities across all credible set variants.

# In[ ]:


# use pip50 dataset (pip90 has known matching bug)
lc_dir = DATA_BASE / 'leafcutter_pip50'


# In[ ]:


# credset stats
credset_stats_lc = pd.read_csv(lc_dir / 'credset_stats.csv')

metrics_cfg = [
    ('n_variants', 'Unique Variants'),
    ('n_genes', 'Unique Genes'),
    ('n_high_pip', 'High-PIP Variants'),
]
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
for ax, (col, label) in zip(axes, metrics_cfg):
    ss = credset_stats_lc.sort_values(col, ascending=True)
    ax.bar(range(len(ss)), ss[col], color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(ss)))
    ax.set_xticklabels(ss['tissue'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(label)
    format_axis_plain(ax, axis='y')
plt.suptitle('Leafcutter Credible Sets Per Tissue', y=1.01)
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter' / 'qc' / '05_credsets_counts')
plt.show()
plt.close()


# In[22]:


# PIP distribution
pip_lc = pd.read_csv(lc_dir / 'pip_values.csv')
print(f'pip values: {len(pip_lc):,} rows')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(pip_lc['pip'], bins=100, color='steelblue', alpha=0.8, edgecolor='white')
axes[0].axvline(0.9, color='#e74c3c', ls='--', lw=2, label='PIP=0.9')
axes[0].axvline(0.01, color='#3498db', ls='--', lw=2, label='PIP=0.01')
axes[0].set_xlabel('PIP'); axes[0].set_ylabel('Count')
axes[0].set_title('Leafcutter PIP Distribution'); axes[0].legend()

axes[1].hist(pip_lc['pip'], bins=100, color='steelblue', alpha=0.8, edgecolor='white')
axes[1].set_yscale('log')
axes[1].axvline(0.9, color='#e74c3c', ls='--', lw=2)
axes[1].axvline(0.01, color='#3498db', ls='--', lw=2)
axes[1].set_xlabel('PIP'); axes[1].set_ylabel('Count (log)')
axes[1].set_title('PIP Distribution (log scale)')
plt.suptitle('leafcutter: PIP Distribution', y=1.02)
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter' / 'qc' / '06_credsets_pip_dist')
plt.show()
plt.close()


# In[23]:


# PIP sum per credible set
lc_cs_path = lc_dir / 'cs_summary.csv'
if lc_cs_path.exists():
    lc_cs = pd.read_csv(lc_cs_path)
    n_cs = len(lc_cs)
    pip_sums = lc_cs['pip_sum'].values
    med = np.median(pip_sums)
    mx = pip_sums.max()
    print(f'leafcutter CS summary: {n_cs:,} credible sets, median pip_sum={med:.4f}, max={mx:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist(pip_sums, bins=100, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('sum(PIP) per credible set')
    ax.set_ylabel('count (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'PIP sum per CS (n={n_cs:,})')
    ax = axes[1]
    in_range = pip_sums[(pip_sums > 0.8) & (pip_sums < 1.1)]
    ax.hist(in_range, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('sum(PIP) per credible set')
    ax.set_ylabel('count (log scale)')
    ax.set_yscale('log')
    ax.set_xlim(0.8, 1.1)
    ax.set_title(f'PIP sum (zoomed 0.8-1.1, n={len(in_range):,})')
    plt.suptitle('leafcutter: PIP Sum per Credible Set', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'leafcutter' / 'qc' / '07_credsets_pip_sum')
    plt.show()
    plt.close()
    n_over_1 = (pip_sums > 1.0).sum()
    if n_over_1 > 0:
        print(f'  [WARNING] {n_over_1:,} CS have PIP sum > 1.0, max={mx:.4f}')
    else:
        print(f'  [OK] all CS have PIP sum <= 1.0 (max: {mx:.4f})')
else:
    print(f'not found: {lc_cs_path} — run gen_qc_data.py first')


# In[24]:


# CS size vs max PIP
if lc_cs_path.exists():
    cs_sizes = lc_cs['cs_size'].values
    max_pips = lc_cs['max_pip'].values
    n_cs = len(lc_cs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    hb = ax.hexbin(cs_sizes, max_pips, gridsize=50, cmap='viridis', mincnt=1, bins='log')
    ax.set_xlabel('variants per credible set')
    ax.set_ylabel('max PIP in credible set')
    ax.set_title(f'CS size vs max PIP (n={n_cs:,})')
    ax.axhline(0.9, color='#e74c3c', ls='--', lw=1.5, alpha=0.8, label='PIP=0.9')
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(hb, ax=ax, label='count (log)')
    ax = axes[1]
    size_bins = [1, 2, 3, 5, 10, 20, 50, 100, np.inf]
    bin_labels = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
    size_bin_idx = np.digitize(cs_sizes, size_bins[1:])
    boxplot_data = []
    valid_labels = []
    for i, label in enumerate(bin_labels):
        mask = size_bin_idx == i
        if mask.sum() > 0:
            boxplot_data.append(max_pips[mask])
            valid_labels.append(f'{label}\n(n={mask.sum():,})')
    bp = ax.boxplot(boxplot_data, labels=valid_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax.axhline(0.9, color='#e74c3c', ls='--', lw=1.5, alpha=0.8, label='PIP=0.9')
    ax.set_xlabel('credible set size')
    ax.set_ylabel('max PIP in credible set')
    ax.set_title('Max PIP by CS size')
    ax.legend(loc='lower right', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle('leafcutter: CS Size vs Maximum PIP', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'leafcutter' / 'qc' / '08_credsets_size_vs_pip')
    plt.show()
    plt.close()


# In[25]:


# CS structure: per-tissue counts + size distribution
if lc_cs_path.exists():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # left: CS per tissue
    ax = axes[0]
    tissue_counts = lc_cs.groupby('tissue').size().sort_index()
    x = np.arange(len(tissue_counts))
    ax.bar(x, tissue_counts.values, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tissue_counts.index, rotation=90, fontsize=6)
    ax.set_ylabel('count')
    ax.set_title('Number of credible sets per tissue')
    ax.set_xlim(-0.5, len(tissue_counts) - 0.5)
    # right: CS size distribution
    ax = axes[1]
    cs_sizes = lc_cs['cs_size'].values
    n_cs = len(lc_cs)
    ax.hist(cs_sizes, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('variants per credible set')
    ax.set_ylabel('count')
    ax.set_title(f'CS size distribution (n={n_cs:,})')
    ax.axvline(np.median(cs_sizes), color='#e74c3c', ls='--', lw=2,
               label=f'median={np.median(cs_sizes):.0f}')
    ax.legend()
    plt.suptitle('leafcutter: Credible Set Structure', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'leafcutter' / 'qc' / '09_credsets_structure')
    plt.show()
    plt.close()


# ## Leafcutter Matching
# 
# 4-tier matching cascade (GE negatives). Splice distance for negatives = distance to nearest leafcutter intron boundary within the same gene (intron boundaries from Zenodo phenotype metadata — all detected introns, not just QTL-tested).
# 
# | Tier | Gene | Allele | Description |
# |------|------|--------|-------------|
# | 1 | same gene | same ref/alt | Best: same genomic context + same mutation type |
# | 2 | same gene | any | Same gene, different alleles |
# | 3 | same TPM bin | same ref/alt | Similar expression level + same mutation type |
# | 4 | same TPM bin | any | Loosest: just similar expression level |

# In[26]:


# load leafcutter matching data from pip50, filter to pip >= 0.9 for results
if (lc_dir / 'pairs.csv').exists():
    pairs_lc_all = pd.read_csv(lc_dir / 'pairs.csv')
    tiers_lc = pd.read_csv(lc_dir / 'tiers.csv')
    distances_lc = pd.read_csv(lc_dir / 'distances.csv') if (lc_dir / 'distances.csv').exists() else None
    tissue_stats_lc = pd.read_csv(lc_dir / 'tissue_stats.csv') if (lc_dir / 'tissue_stats.csv').exists() else None

    # counts before filtering
    n_pos_all = pairs_lc_all['pos_var_key'].nunique()
    n_neg_all = pairs_lc_all['neg_var_key'].nunique()
    n_t = pairs_lc_all['tissue'].nunique()
    print(f'leafcutter pip50 (all): {n_pos_all:,} pos + {n_neg_all:,} neg unique, {n_t} tissues, {len(pairs_lc_all):,} pairs')

    # filter to pip >= 0.9 for results
    pairs_lc = pairs_lc_all.query('pos_pip >= 0.9').copy()
    n_pos = pairs_lc['pos_var_key'].nunique()
    n_neg = pairs_lc['neg_var_key'].nunique()
    print(f'leafcutter pip90 (filtered): {n_pos:,} pos + {n_neg:,} neg unique, {n_t} tissues, {len(pairs_lc):,} pairs')
else:
    pairs_lc_all = pairs_lc = tiers_lc = distances_lc = tissue_stats_lc = None
    print('not ready (pairs.csv not found)')


# In[27]:


# leafcutter matching figs (show full pip50 matching data)
if tiers_lc is not None:
    n_pos = int(tissue_stats_lc['n_pos'].sum()) if tissue_stats_lc is not None else None
    plot_tier_dist(tiers_lc, 'Leafcutter (pip50) - Tier Distribution',
        FIG_BASE / 'leafcutter' / 'matching' / 'set1' / '01_tier_distribution',
        n_pos_total=n_pos)

if distances_lc is not None:
    fig, ax = plt.subplots(figsize=(8, 4))
    pos_d = distances_lc[distances_lc['type'] == 'pos']['splice_dist']
    neg_d = distances_lc[distances_lc['type'] == 'neg']['splice_dist']
    bins = np.logspace(0, 4, 50)
    ax.hist(pos_d, bins=bins, alpha=0.7, label=f'pos (n={len(pos_d):,})', color=DATA_COLORS['pos'])
    ax.hist(neg_d, bins=bins, alpha=0.7, label=f'neg (n={len(neg_d):,})', color=DATA_COLORS['neg'])
    ax.set_xscale('log'); ax.set_xlabel('Splice distance (bp)')
    ax.set_ylabel('Count'); ax.set_title('Leafcutter (pip50) - Splice Distance'); ax.legend()
    plt.tight_layout()
    savefig(FIG_BASE / 'leafcutter' / 'matching' / 'set1' / '05_distance_dist')
    plt.show()
    plt.close()

if pairs_lc_all is not None and 'tissue' in pairs_lc_all.columns:
    plot_per_tissue_stacked(pairs_lc_all,
        'Leafcutter (pip50) - Per Tissue (tissue-specific vs shared)',
        FIG_BASE / 'leafcutter' / 'matching' / 'set1' / '06_per_tissue')


# In[ ]:


# diagnose unmatched positives (against full pip50 matching)
if pairs_lc_all is not None and tissue_stats_lc is not None:
    n_total = int(tissue_stats_lc['n_pos'].sum())
    n_matched = int(tissue_stats_lc['n_matched'].sum())
    n_unmatched = n_total - n_matched
    print(f'total pos: {n_total:,}, matched: {n_matched:,}, unmatched: {n_unmatched:,}')

    if n_unmatched > 0:
        # which tissues have unmatched?
        ts = tissue_stats_lc.copy()
        ts['unmatched'] = ts['n_pos'] - ts['n_matched']
        ts_um = ts[ts['unmatched'] > 0].sort_values('unmatched', ascending=False)
        print(f'\ntissues with unmatched: {len(ts_um)}/{len(ts)}')
        print(ts_um[['tissue', 'n_pos', 'n_matched', 'unmatched']].to_string(index=False))

        # load full positives to identify unmatched var_keys per tissue
        pip_lc = pd.read_csv(lc_dir / 'pip_values.csv')
        high_pip = pip_lc[pip_lc['pip'] >= 0.9].copy()
        high_pip = high_pip.sort_values('pip', ascending=False).drop_duplicates('var_key', keep='first')
        # rebuild pos_tissues: which tissues each pos appears in
        pos_keys = set(high_pip['var_key'])
        pos_tissues = pip_lc[pip_lc['var_key'].isin(pos_keys)].groupby('var_key')['tissue'].apply(set).to_dict()

        matched_pairs = set(zip(pairs_lc_all['pos_var_key'], pairs_lc_all['tissue']))

        unmatched = []
        for vk, tissues in pos_tissues.items():
            for t in tissues:
                if (vk, t) not in matched_pairs:
                    gene = high_pip.loc[high_pip['var_key'] == vk, 'gene_id'].iloc[0] if 'gene_id' in high_pip.columns else 'unknown'
                    unmatched.append({'var_key': vk, 'tissue': t, 'gene_id': gene})

        if unmatched:
            um_df = pd.DataFrame(unmatched)
            print(f'\nunmatched variant-tissue pairs: {len(um_df):,}')
            if 'gene_id' in um_df.columns and um_df['gene_id'].iloc[0] != 'unknown':
                gene_counts = um_df['gene_id'].value_counts()
                print(f'unique genes in unmatched: {len(gene_counts):,}')
                print(f'\ntop 10 genes with most unmatched:')
                print(gene_counts.head(10).to_string())

            # check neg availability: load one tissue parquet to see if genes have negs
            tmp_dir = lc_dir / '.tmp_ge'
            if tmp_dir.exists() and 'gene_id' in um_df.columns:
                um_genes = set(um_df['gene_id'].unique())
                # sample: check a few tissues
                for t in um_df['tissue'].value_counts().head(3).index:
                    pq = tmp_dir / f'{t}.parquet'
                    if pq.exists():
                        neg = pd.read_parquet(pq, columns=['gene_id'])
                        neg_genes = set(neg['gene_id'].str.split('.').str[0])
                        t_um_genes = set(um_df[um_df['tissue'] == t]['gene_id'])
                        in_neg = t_um_genes & neg_genes
                        not_in_neg = t_um_genes - neg_genes
                        print(f'\n{t}: {len(t_um_genes)} unmatched genes, {len(in_neg)} have negs, {len(not_in_neg)} have NO negs')
                        if not_in_neg:
                            print(f'  genes missing from GE negs: {sorted(not_in_neg)[:10]}')
        else:
            print('no unmatched pairs found (mismatch in counting?)')


# ## Leafcutter Results
# 
# Model performance on leafcutter sQTL benchmark.
# 

# In[28]:


# load leafcutter scores (from pip50 dataset, results use pip >= 0.9 filter)
score_dir_lc = lc_dir / 'scores'
pos_scores_lc = load_scores(score_dir_lc, 'pos', primary_only=args.nohbar)
neg_scores_lc = load_scores(score_dir_lc, 'neg', primary_only=args.nohbar)
print('leafcutter models:', list(pos_scores_lc.keys()))
print(f'leafcutter results (pip >= 0.9): {pairs_lc["pos_var_key"].nunique():,} pos, {pairs_lc["neg_var_key"].nunique():,} neg unique, {len(pairs_lc):,} pairs')


# In[ ]:


plot_delta_dist(pairs_lc, pos_scores_lc, neg_scores_lc,
    'Leafcutter (pip50, filtered >=0.9) - Delta Distributions',
    FIG_BASE / 'leafcutter' / 'results' / 'set1' / '00a_delta_distributions')


# In[ ]:


metrics_lc = compute_tissue_metrics(pairs_lc, pos_scores_lc, neg_scores_lc)
n_tissues = metrics_lc['tissue'].nunique()
n_models = metrics_lc['model'].nunique()
print(f'{n_models} models x {n_tissues} tissues = {len(metrics_lc)} tissue-model combos')

plot_swarm(metrics_lc, 'Leafcutter (pip50, filtered >=0.9) - Per-tissue AUPRC',
    FIG_BASE / 'leafcutter' / 'results' / 'set1' / '01_swarm')

if not args.nohbar:
    plot_median_auprc_hbar(metrics_lc, 'Leafcutter (pip50, filtered >=0.9) - All Output Heads',
        FIG_BASE / 'leafcutter' / 'results' / 'set1' / '01b_hbar')


# In[ ]:


plot_pr(pairs_lc, pos_scores_lc, neg_scores_lc,
    'Leafcutter (pip50, filtered >=0.9) - PR Curves',
    FIG_BASE / 'leafcutter' / 'results' / 'set1' / '01b_pr_curves')


# In[ ]:


plot_roc(pairs_lc, pos_scores_lc, neg_scores_lc,
    'Leafcutter (pip50, filtered >=0.9) - ROC Curves',
    FIG_BASE / 'leafcutter' / 'results' / 'set1' / '01d_roc_curves')


# In[ ]:


if distances_lc is not None:
    plot_distance_auprc(pairs_lc, pos_scores_lc, neg_scores_lc, distances_lc,
        'Leafcutter - AUPRC by Splice Distance',
        FIG_BASE / 'leafcutter' / 'results' / 'set1' / '02_distance')


# In[ ]:


if distances_lc is not None:
    plot_distance_auprc_unique(pairs_lc, pos_scores_lc, neg_scores_lc, distances_lc,
        'Leafcutter - AUPRC by Distance (unique positives)',
        FIG_BASE / 'leafcutter' / 'results' / 'set1' / '02b_distance_unique')


# 
# # HAEC Analysis
# 
# **Positives:** Variants from HAEC185 sQTL fine-mapping credible sets. For each credible set, the variant with highest PIP is selected as the positive. Only credible sets with max PIP >= 0.9 are included. Single tissue dataset.
# 
# **Negatives:** Variants from HAEC sQTL summary statistics with PIP < 0.01. Splice distance for negatives is computed as distance to the nearest leafcutter-detected intron boundary in the same gene (using all introns from phenotype bed files). Each positive is matched 1:1 to a negative using the same 4-tier cascade as leafcutter and txrevise.

# ## HAEC QC
# 
# **Sumstats QC:** Summary statistics quality metrics for the HAEC dataset.
# 
# **Credible Set QC:** Credible set statistics: variant counts, high-PIP variants.
# 
# **PIP Distribution:** Distribution of posterior inclusion probabilities.
# 
# **CS Summary:** PIP sum per credible set, CS size distribution, CS size vs max PIP.

# In[29]:


haec_dir = DATA_BASE / 'haec'

# --- sumstats QC: 2x2 grid ---
if (haec_dir / 'sumstats_counts.csv').exists():
    haec_ss = pd.read_csv(haec_dir / 'sumstats_counts.csv')
    print('HAEC sumstats:')
    print(haec_ss)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ('n_variants', 'Total rows'),
        ('n_variants', 'Unique variants'),
        ('n_introns', 'Unique phenotypes'),
        ('n_genes', 'Unique genes'),
    ]
    for ax, (col, label) in zip(axes.flatten(), metrics):
        if col not in haec_ss.columns:
            ax.set_visible(False)
            continue
        val = haec_ss[col].iloc[0]
        ax.bar(['HAEC'], [val], color='steelblue', alpha=0.8)
        ax.set_ylabel('count')
        ax.set_title(f'{label}: {val:,}')
    plt.suptitle('HAEC: Sumstats QC (single tissue)', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'haec' / 'qc' / '02_sumstats_qc')
    plt.show()
    plt.close()

# --- credset QC: 2x2 grid (counts, PIP dist, PIP sum, CS size dist) ---
has_pip = (haec_dir / 'pip_values.csv').exists()
has_cs_summary = (haec_dir / 'cs_summary.csv').exists()

if has_cs_summary or has_pip:
    # load cs_summary once (used by count, pip calibration, and cs size panels)
    haec_cs_sum = pd.read_csv(haec_dir / 'cs_summary.csv') if has_cs_summary else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # top-left: total credible sets count
    if has_cs_summary:
        n_cs = len(haec_cs_sum)
        axes[0, 0].bar(['Credible Sets'], [n_cs], color='steelblue', alpha=0.8)
        axes[0, 0].text(0, n_cs + n_cs * 0.02, f'{n_cs:,}', ha='center', fontsize=10)
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'HAEC Credible Sets: {n_cs:,}')
        axes[0, 0].set_ylim(0, n_cs * 1.15)
    else:
        axes[0, 0].text(0.5, 0.5, 'run gen_qc_data.py', ha='center', va='center',
                         transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Credible Set Count')

    # top-right: PIP distribution
    if has_pip:
        pip_haec = pd.read_csv(haec_dir / 'pip_values.csv')
        axes[0, 1].hist(pip_haec['pip'], bins=50, color='steelblue', alpha=0.8, edgecolor='white')
        axes[0, 1].axvline(0.9, color='#2ecc71', ls='--', lw=2, label='pos threshold (0.9)')
        axes[0, 1].axvline(0.01, color='#e74c3c', ls='--', lw=2, label='neg threshold (0.01)')
        axes[0, 1].set_xlabel('PIP')
        axes[0, 1].set_ylabel('Variant count')
        axes[0, 1].set_title('PIP Distribution')
        axes[0, 1].legend(fontsize=8)

    # bottom-left: PIP calibration check
    if has_cs_summary:
        pip_sums = haec_cs_sum['pip_sum'].values
        med = np.median(pip_sums)
        mx = pip_sums.max()
        axes[1, 0].hist(pip_sums, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
        axes[1, 0].axvline(0.95, color='#e74c3c', ls='--', lw=2, label='expected (0.95)')
        axes[1, 0].axvline(med, color='#2ecc71', ls=':', lw=2, label=f'median={med:.4f}')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('PIP Sum per Credible Set')
        axes[1, 0].set_ylabel('Count (log scale)')
        axes[1, 0].set_title('PIP Calibration Check')
        axes[1, 0].legend(fontsize=8, title=f'max={mx:.4f}')
    else:
        axes[1, 0].text(0.5, 0.5, 'run gen_qc_data.py', ha='center', va='center',
                         transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('PIP Calibration Check')

    # bottom-right: CS size distribution
    if has_cs_summary:
        cs_sizes = haec_cs_sum['cs_size'].values
        med_size = np.median(cs_sizes)
        axes[1, 1].hist(cs_sizes, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
        axes[1, 1].axvline(med_size, color='#e74c3c', ls='--', lw=2,
                            label=f'median={med_size:.0f}')
        axes[1, 1].set_xlabel('Variants per Credible Set')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Credible Set Size Distribution')
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, 'run gen_qc_data.py', ha='center', va='center',
                         transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('CS Size Distribution')

    plt.suptitle('haec: Credible Sets QC', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / 'haec' / 'qc' / '04_credsets_qc')
    plt.show()
    plt.close()


# ## HAEC Matching
# 
# 4-tier matching with sQTL-sourced negatives (same cascade as leafcutter/txrevise). Negative splice distance computed from all leafcutter-detected intron boundaries per gene.
# 
# - **Tier 1:** same gene + same alleles (closest splice distance)
# - **Tier 2:** same gene only
# - **Tier 3:** same expression bin + same alleles
# - **Tier 4:** same expression bin only

# In[30]:


# load HAEC matching data
haec_dir = DATA_BASE / 'haec'

if (haec_dir / 'pairs.csv').exists():
    pairs_haec = pd.read_csv(haec_dir / 'pairs.csv')
    tiers_haec = pd.read_csv(haec_dir / 'tiers.csv')
    distances_haec = pd.read_csv(haec_dir / 'distances.csv') if (haec_dir / 'distances.csv').exists() else None
    n_pos = pairs_haec['pos_var_key'].nunique()
    n_neg = pairs_haec['neg_var_key'].nunique()
    print(f'{n_pos:,} pos + {n_neg:,} neg unique variants, {len(pairs_haec):,} pairs (single tissue)')
else:
    pairs_haec = tiers_haec = distances_haec = None
    print('not ready (pairs.csv not found)')


# In[31]:


# HAEC matching figs
if tiers_haec is not None:
    plot_tier_dist(tiers_haec, 'HAEC - Tier Distribution',
        FIG_BASE / 'haec' / 'matching' / 'set1' / '01_tier_distribution')

if distances_haec is not None:
    fig, ax = plt.subplots(figsize=(8, 4))
    pos_d = distances_haec[distances_haec['type'] == 'pos']['splice_dist']
    neg_d = distances_haec[distances_haec['type'] == 'neg']['splice_dist']
    bins = np.logspace(0, 4, 50)
    ax.hist(pos_d, bins=bins, alpha=0.7, label=f'pos (n={len(pos_d):,})', color=DATA_COLORS['pos'])
    ax.hist(neg_d, bins=bins, alpha=0.7, label=f'neg (n={len(neg_d):,})', color=DATA_COLORS['neg'])
    ax.set_xscale('log'); ax.set_xlabel('Splice distance (bp)')
    ax.set_ylabel('Count'); ax.set_title('HAEC - Splice Distance'); ax.legend()
    plt.tight_layout()
    savefig(FIG_BASE / 'haec' / 'matching' / 'set1' / '05_distance_dist')
    plt.show()
    plt.close()


# ## HAEC Results
# 
# Model performance on HAEC sQTL benchmark.
# 

# In[32]:


# load HAEC scores
score_dir_haec = haec_dir / 'scores'
pos_scores_haec = load_scores(score_dir_haec, 'pos', primary_only=args.nohbar)
neg_scores_haec = load_scores(score_dir_haec, 'neg', primary_only=args.nohbar)
print('HAEC models:', list(pos_scores_haec.keys()))


# In[33]:


plot_delta_dist(pairs_haec, pos_scores_haec, neg_scores_haec,
    'HAEC - Delta Distributions',
    FIG_BASE / 'haec' / 'results' / 'set1' / '00a_delta_distributions')

# overall AUPRC (single tissue)
pooled_haec = compute_pooled_metrics(pairs_haec, pos_scores_haec, neg_scores_haec)
n_pos = pairs_haec['pos_var_key'].nunique()
n_neg = pairs_haec['neg_var_key'].nunique()
print(f'HAEC: {n_pos:,} pos + {n_neg:,} neg unique variants, {len(pairs_haec):,} pairs (single tissue)')
for model, m in pooled_haec.items():
    print(f'  {MODEL_NAMES[model]}: AUPRC={m["auprc"]:.3f}, AUROC={m["auroc"]:.3f} (n_pairs={m["n"]:,})')

fig, ax = plt.subplots(figsize=(8, 5))
mp = [m for m in MODELS if m in pooled_haec]
for i, model in enumerate(mp):
    val = pooled_haec[model]['auprc']
    ax.bar(i, val, color=COLORS[model], alpha=0.8)
    ax.text(i, val + 0.01, f'{val:.3f}', ha='center', fontsize=9)
ax.set_xticks(range(len(mp)))
ax.set_xticklabels([MODEL_NAMES[m] for m in mp], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('AUPRC'); ax.set_title('HAEC - Overall AUPRC')
ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig(FIG_BASE / 'haec' / 'results' / 'set1' / '01_overall_auprc')
plt.show()
plt.close()

# all output heads comparison
if not args.nohbar:
    metrics_haec = compute_tissue_metrics(
        pairs_haec.assign(tissue='HAEC'), pos_scores_haec, neg_scores_haec)
    plot_median_auprc_hbar(metrics_haec, 'HAEC - All Output Heads',
        FIG_BASE / 'haec' / 'results' / 'set1' / '01b_hbar')


# In[34]:


plot_pr(pairs_haec, pos_scores_haec, neg_scores_haec,
    'HAEC - PR Curves',
    FIG_BASE / 'haec' / 'results' / 'set1' / '01b_pr_curves')


# In[35]:


plot_roc(pairs_haec, pos_scores_haec, neg_scores_haec,
    'HAEC - ROC Curves',
    FIG_BASE / 'haec' / 'results' / 'set1' / '01b_roc_curves')


# In[36]:


if distances_haec is not None:
    plot_distance_auprc(pairs_haec, pos_scores_haec, neg_scores_haec, distances_haec,
        'HAEC - AUPRC by Splice Distance',
        FIG_BASE / 'haec' / 'results' / 'set1' / '02_distance')


# ## Cross-Dataset Summary (Main Classification)
# 
# Per-tissue median AUPRC and AUROC across all three sQTL datasets for the primary pos-vs-neg classification benchmark. Each cell is the median across tissues (49 for GTEx, 1 for HAEC).

# In[ ]:


# cross-dataset summary table
datasets = {
    'GTEx Txrevise': (pairs, pos_scores, neg_scores),
    'GTEx Leafcutter': (pairs_lc, pos_scores_lc, neg_scores_lc),
    'HAEC': (pairs_haec, pos_scores_haec, neg_scores_haec),
}

summary_rows = []
for ds_name, (p, ps, ns) in datasets.items():
    if p is None or not ps:
        continue
    has_tissue = 'tissue' in p.columns
    n_tissues = p['tissue'].nunique() if has_tissue else 1
    n_pairs = len(p)
    n_pos = p['pos_var_key'].nunique()
    for m in MODELS:
        if m not in ps or m not in ns:
            continue
        if has_tissue:
            tissue_auprc, tissue_auroc = [], []
            for tissue in p['tissue'].unique():
                tp = p[p['tissue'] == tissue]
                yt, ys, n = get_ys(tp, ps, ns, m)
                if n >= 10:
                    tissue_auprc.append(compute_auprc(yt, ys))
                    tissue_auroc.append(compute_auroc(yt, ys))
            med_auprc = np.median(tissue_auprc) if tissue_auprc else np.nan
            med_auroc = np.median(tissue_auroc) if tissue_auroc else np.nan
        else:
            yt, ys, n = get_ys(p, ps, ns, m)
            med_auprc = compute_auprc(yt, ys) if n >= 10 else np.nan
            med_auroc = compute_auroc(yt, ys) if n >= 10 else np.nan
        summary_rows.append({
            'dataset': ds_name, 'model': MODEL_NAMES.get(m, m),
            'median_auprc': med_auprc, 'median_auroc': med_auroc,
            'n_tissues': n_tissues, 'n_pairs': n_pairs, 'n_pos_unique': n_pos,
        })

summary_df = pd.DataFrame(summary_rows)

print('=' * 90)
print('Cross-Dataset Classification Summary (per-tissue median)')
print('=' * 90)
for ds in summary_df['dataset'].unique():
    sub = summary_df[summary_df['dataset'] == ds]
    nt = sub.iloc[0]['n_tissues']
    np_ = sub.iloc[0]['n_pairs']
    nu = sub.iloc[0]['n_pos_unique']
    print(f'\n{ds} ({nt} tissues, {np_:,} pairs, {nu:,} unique positives)')
    print(f'  {"Model":<25s} {"AUPRC":>8s} {"AUROC":>8s}')
    print(f'  {"-"*25} {"-"*8} {"-"*8}')
    for _, row in sub.iterrows():
        print(f'  {row["model"]:<25s} {row["median_auprc"]:8.3f} {row["median_auroc"]:8.3f}')
print()

savefig_path = FIG_BASE / 'cross_dataset_classification_summary'
summary_df.to_csv(str(savefig_path) + '.csv', index=False)
print(f'saved {savefig_path}.csv')


# ## Cross-Dataset Positive Variant Overlap
#
# Compare positive variant sets across txrevise, leafcutter, and HAEC.
# Per-tissue overlap for the two GTEx datasets (49 shared tissues),
# overall deduped overlap across all three datasets, and swarm plots
# for dataset-exclusive variants.

# In[ ]:


# overall deduped positive var_keys per dataset
tx_pos = set(pairs['pos_var_key'].unique())
lc_pos = set(pairs_lc['pos_var_key'].unique())
haec_pos = set(pairs_haec['pos_var_key'].unique()) if pairs_haec is not None else set()

print('=' * 70)
print('CROSS-DATASET POSITIVE VARIANT OVERLAP')
print('=' * 70)
print(f'  txrevise:   {len(tx_pos):,} unique pos variants')
print(f'  leafcutter: {len(lc_pos):,} unique pos variants')
print(f'  HAEC:       {len(haec_pos):,} unique pos variants')

tx_lc = tx_pos & lc_pos
tx_haec = tx_pos & haec_pos
lc_haec = lc_pos & haec_pos
all_three = tx_pos & lc_pos & haec_pos

tx_only = tx_pos - lc_pos - haec_pos
lc_only = lc_pos - tx_pos - haec_pos
haec_only = haec_pos - tx_pos - lc_pos

print(f'\n  txrevise & leafcutter: {len(tx_lc):,}')
print(f'  txrevise & HAEC:      {len(tx_haec):,}')
print(f'  leafcutter & HAEC:    {len(lc_haec):,}')
print(f'  all three:            {len(all_three):,}')
print(f'\n  txrevise only:   {len(tx_only):,}')
print(f'  leafcutter only: {len(lc_only):,}')
print(f'  HAEC only:       {len(haec_only):,}')

# per-tissue overlap (txrevise vs leafcutter, both GTEx 49 tissues)
tx_tissues = set(pairs['tissue'].unique())
lc_tissues = set(pairs_lc['tissue'].unique())
shared_tissues = sorted(tx_tissues & lc_tissues)

tissue_overlap = []
for tissue in shared_tissues:
    tx_t = set(pairs.query('tissue == @tissue')['pos_var_key'])
    lc_t = set(pairs_lc.query('tissue == @tissue')['pos_var_key'])
    shared = tx_t & lc_t
    tissue_overlap.append({
        'tissue': tissue,
        'txrevise': len(tx_t), 'leafcutter': len(lc_t),
        'shared': len(shared),
        'tx_only': len(tx_t - lc_t), 'lc_only': len(lc_t - tx_t),
    })
tissue_overlap_df = pd.DataFrame(tissue_overlap)

print(f'\nper-tissue overlap (txrevise vs leafcutter, {len(shared_tissues)} tissues):')
print(f'  median shared per tissue:     {tissue_overlap_df["shared"].median():.0f}')
print(f'  median txrevise-only:         {tissue_overlap_df["tx_only"].median():.0f}')
print(f'  median leafcutter-only:       {tissue_overlap_df["lc_only"].median():.0f}')
total_shared = tissue_overlap_df['shared'].sum()
total_tx = tissue_overlap_df['txrevise'].sum()
total_lc = tissue_overlap_df['leafcutter'].sum()
print(f'  total variant-tissue shared:  {total_shared:,} / {total_tx:,} tx / {total_lc:,} lc')

# plot: per-tissue stacked bar (shared / tx-only / lc-only)
fig, ax = plt.subplots(figsize=(14, 5))
tdf = tissue_overlap_df.sort_values('shared', ascending=True)
x = np.arange(len(tdf))
ax.barh(x, tdf['shared'], color='#8da0cb', label='shared')
ax.barh(x, tdf['tx_only'], left=tdf['shared'], color='#e78ac3', label='txrevise only')
ax.barh(x, tdf['lc_only'], left=tdf['shared'].values + tdf['tx_only'].values, color='#66c2a5', label='leafcutter only')
ax.set_yticks(x)
ax.set_yticklabels(tdf['tissue'], fontsize=6)
ax.set_xlabel('Positive variants')
ax.set_title('Per-tissue positive variant overlap (txrevise vs leafcutter)')
ax.legend(fontsize=9)
plt.tight_layout()
savefig(FIG_BASE / 'overlap' / 'per_tissue_overlap')
plt.show()
plt.close()

# overall overlap bar chart (3 datasets)
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['txrevise\nonly', 'leafcutter\nonly', 'HAEC\nonly',
              'tx & lc', 'tx & HAEC', 'lc & HAEC', 'all three']
counts = [len(tx_only), len(lc_only), len(haec_only),
          len(tx_lc - haec_pos), len(tx_haec - lc_pos), len(lc_haec - tx_pos), len(all_three)]
colors = ['#e78ac3', '#66c2a5', '#fc8d62', '#8da0cb', '#a6d854', '#ffd92f', '#b3b3b3']
bars = ax.bar(categories, counts, color=colors, edgecolor='white')
for bar, c in zip(bars, counts):
    if c > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{c:,}', ha='center', fontsize=9)
ax.set_ylabel('Unique positive variants')
ax.set_title('Positive variant overlap across datasets')
format_axis_plain(ax, axis='y')
plt.tight_layout()
savefig(FIG_BASE / 'overlap' / 'overall_overlap')
plt.show()
plt.close()

# --- same for negatives ---
tx_neg = set(pairs['neg_var_key'].unique())
lc_neg = set(pairs_lc['neg_var_key'].unique())
haec_neg = set(pairs_haec['neg_var_key'].unique()) if pairs_haec is not None else set()

tx_lc_neg = tx_neg & lc_neg
tx_haec_neg = tx_neg & haec_neg
lc_haec_neg = lc_neg & haec_neg
all_three_neg = tx_neg & lc_neg & haec_neg
tx_only_neg = tx_neg - lc_neg - haec_neg
lc_only_neg = lc_neg - tx_neg - haec_neg
haec_only_neg = haec_neg - tx_neg - lc_neg

print(f'\nnegative variant overlap:')
print(f'  txrevise:   {len(tx_neg):,}')
print(f'  leafcutter: {len(lc_neg):,}')
print(f'  HAEC:       {len(haec_neg):,}')
print(f'  tx & lc: {len(tx_lc_neg):,}, tx & HAEC: {len(tx_haec_neg):,}, lc & HAEC: {len(lc_haec_neg):,}, all: {len(all_three_neg):,}')
print(f'  tx only: {len(tx_only_neg):,}, lc only: {len(lc_only_neg):,}, HAEC only: {len(haec_only_neg):,}')

# per-tissue negative overlap
tissue_overlap_neg = []
for tissue in shared_tissues:
    tx_t = set(pairs.query('tissue == @tissue')['neg_var_key'])
    lc_t = set(pairs_lc.query('tissue == @tissue')['neg_var_key'])
    shared_n = tx_t & lc_t
    tissue_overlap_neg.append({
        'tissue': tissue,
        'txrevise': len(tx_t), 'leafcutter': len(lc_t),
        'shared': len(shared_n),
        'tx_only': len(tx_t - lc_t), 'lc_only': len(lc_t - tx_t),
    })
tissue_overlap_neg_df = pd.DataFrame(tissue_overlap_neg)

fig, ax = plt.subplots(figsize=(14, 5))
tdf = tissue_overlap_neg_df.sort_values('shared', ascending=True)
x = np.arange(len(tdf))
ax.barh(x, tdf['shared'], color='#8da0cb', label='shared')
ax.barh(x, tdf['tx_only'], left=tdf['shared'], color='#e78ac3', label='txrevise only')
ax.barh(x, tdf['lc_only'], left=tdf['shared'].values + tdf['tx_only'].values, color='#66c2a5', label='leafcutter only')
ax.set_yticks(x)
ax.set_yticklabels(tdf['tissue'], fontsize=6)
ax.set_xlabel('Negative variants')
ax.set_title('Per-tissue negative variant overlap (txrevise vs leafcutter)')
ax.legend(fontsize=9)
plt.tight_layout()
savefig(FIG_BASE / 'overlap' / 'per_tissue_overlap_neg')
plt.show()
plt.close()

# overall negative overlap bar chart
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['txrevise\nonly', 'leafcutter\nonly', 'HAEC\nonly',
              'tx & lc', 'tx & HAEC', 'lc & HAEC', 'all three']
counts_neg = [len(tx_only_neg), len(lc_only_neg), len(haec_only_neg),
              len(tx_lc_neg - haec_neg), len(tx_haec_neg - lc_neg), len(lc_haec_neg - tx_neg), len(all_three_neg)]
colors_neg = ['#e78ac3', '#66c2a5', '#fc8d62', '#8da0cb', '#a6d854', '#ffd92f', '#b3b3b3']
bars = ax.bar(categories, counts_neg, color=colors_neg, edgecolor='white')
for bar, c in zip(bars, counts_neg):
    if c > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_neg)*0.01,
                f'{c:,}', ha='center', fontsize=9)
ax.set_ylabel('Unique negative variants')
ax.set_title('Negative variant overlap across datasets')
format_axis_plain(ax, axis='y')
plt.tight_layout()
savefig(FIG_BASE / 'overlap' / 'overall_overlap_neg')
plt.show()
plt.close()


# In[ ]:


# swarm plots for dataset-exclusive variants
# txrevise-only: positives not in leafcutter or HAEC (in any tissue)
# leafcutter-only: positives not in txrevise or HAEC (in any tissue)

pairs_tx_only = pairs[pairs['pos_var_key'].isin(tx_only)]
pairs_lc_only = pairs_lc[pairs_lc['pos_var_key'].isin(lc_only)]

print(f'\ntxrevise-only pairs: {len(pairs_tx_only):,} ({pairs_tx_only["pos_var_key"].nunique():,} unique pos)')
print(f'leafcutter-only pairs: {len(pairs_lc_only):,} ({pairs_lc_only["pos_var_key"].nunique():,} unique pos)')

if len(pairs_tx_only) > 100:
    metrics_tx_only = compute_tissue_metrics(pairs_tx_only, pos_scores, neg_scores)
    plot_swarm(metrics_tx_only, f'Txrevise-only positives (n={pairs_tx_only["pos_var_key"].nunique():,}) - AUPRC',
        FIG_BASE / 'overlap' / 'swarm_txrevise_only')

if len(pairs_lc_only) > 100:
    metrics_lc_only = compute_tissue_metrics(pairs_lc_only, pos_scores_lc, neg_scores_lc)
    plot_swarm(metrics_lc_only, f'Leafcutter-only positives (n={pairs_lc_only["pos_var_key"].nunique():,}) - AUPRC',
        FIG_BASE / 'overlap' / 'swarm_leafcutter_only')

# shared variants — swarm from txrevise pairs subset
pairs_tx_shared = pairs[pairs['pos_var_key'].isin(tx_lc)]
pairs_lc_shared = pairs_lc[pairs_lc['pos_var_key'].isin(tx_lc)]

print(f'shared (tx pairs):  {len(pairs_tx_shared):,} ({pairs_tx_shared["pos_var_key"].nunique():,} unique pos)')
print(f'shared (lc pairs):  {len(pairs_lc_shared):,} ({pairs_lc_shared["pos_var_key"].nunique():,} unique pos)')

if len(pairs_tx_shared) > 100:
    metrics_tx_shared = compute_tissue_metrics(pairs_tx_shared, pos_scores, neg_scores)
    plot_swarm(metrics_tx_shared, f'Shared positives via txrevise pairs (n={pairs_tx_shared["pos_var_key"].nunique():,}) - AUPRC',
        FIG_BASE / 'overlap' / 'swarm_shared_txrevise')

if len(pairs_lc_shared) > 100:
    metrics_lc_shared = compute_tissue_metrics(pairs_lc_shared, pos_scores_lc, neg_scores_lc)
    plot_swarm(metrics_lc_shared, f'Shared positives via leafcutter pairs (n={pairs_lc_shared["pos_var_key"].nunique():,}) - AUPRC',
        FIG_BASE / 'overlap' / 'swarm_shared_leafcutter')


# ## Tissue-Specific Output Head Evaluation
#
# For GTEx tissues that map to a model tissue head (SPT 15 tissues, Pangolin 4 tissues),
# compare the tissue-matched head AUPRC (x) against the best AUPRC from any other
# available output head (y). Points above the diagonal mean the tissue head is NOT the
# best choice for that tissue.

# In[ ]:


# gtex tissue → model tissue head mapping
_GTEX_TO_SPT = {
    'adipose_subcutaneous': 'adipose', 'adipose_visceral': 'adipose',
    'artery_aorta': 'blood_vessel', 'artery_coronary': 'blood_vessel', 'artery_tibial': 'blood_vessel',
    'brain_amygdala': 'brain', 'brain_anterior_cingulate_cortex': 'brain',
    'brain_caudate': 'brain', 'brain_cerebellar_hemisphere': 'brain',
    'brain_cerebellum': 'brain', 'brain_cortex': 'brain',
    'brain_frontal_cortex': 'brain', 'brain_hippocampus': 'brain',
    'brain_hypothalamus': 'brain', 'brain_nucleus_accumbens': 'brain',
    'brain_putamen': 'brain', 'brain_spinal_cord': 'brain',
    'brain_substantia_nigra': 'brain',
    'blood': 'blood',
    'colon_sigmoid': 'colon', 'colon_transverse': 'colon',
    'heart_atrial_appendage': 'heart', 'heart_left_ventricle': 'heart',
    'kidney_cortex': 'kidney',
    'liver': 'liver', 'lung': 'lung', 'muscle': 'muscle',
    'nerve_tibial': 'nerve',
    'skin_not_sun_exposed': 'skin', 'skin_sun_exposed': 'skin',
    'small_intestine': 'small_intestine',
    'spleen': 'spleen', 'stomach': 'stomach',
}
_GTEX_TO_PANG = {
    'brain_amygdala': 'brain', 'brain_anterior_cingulate_cortex': 'brain',
    'brain_caudate': 'brain', 'brain_cerebellar_hemisphere': 'brain',
    'brain_cerebellum': 'brain', 'brain_cortex': 'brain',
    'brain_frontal_cortex': 'brain', 'brain_hippocampus': 'brain',
    'brain_hypothalamus': 'brain', 'brain_nucleus_accumbens': 'brain',
    'brain_putamen': 'brain', 'brain_spinal_cord': 'brain',
    'brain_substantia_nigra': 'brain',
    'heart_atrial_appendage': 'heart', 'heart_left_ventricle': 'heart',
    'liver': 'liver', 'testis': 'testis',
}

# heads that are max-across-tissues/channels aggregates
_AGGREGATE_HEADS = {
    'splicetransformer', 'splicetransformer_usage', 'splicetransformer_all',
    'pangolin', 'pangolin_usage', 'pangolin_all',
    'pangolin_v2', 'pangolin_v2_usage', 'pangolin_v2_all',
    'sphaec_ref_all', 'sphaec_var_all',
}

# load all output heads for this analysis (bypasses --nohbar)
print('\n' + '=' * 70)
print('TISSUE-SPECIFIC OUTPUT HEAD EVALUATION')
print('=' * 70)

# precompute per-tissue AUPRC for all heads (shared across both versions)
_th_tissue_auprc = {}  # {ds_name: {(tissue, head): auprc}}
_th_all_heads = {}

for ds_name, ds_pairs, score_dir_th in [
    ('txrevise', pairs, DATA_BASE / 'txrevise' / 'scores'),
    ('leafcutter', pairs_lc, DATA_BASE / 'leafcutter_pip50' / 'scores'),
]:
    if ds_pairs is None:
        continue
    print(f'\nloading all heads for {ds_name}...')
    all_pos = load_scores(score_dir_th, 'pos', primary_only=False)
    all_neg = load_scores(score_dir_th, 'neg', primary_only=False)
    heads = sorted(set(all_pos.keys()) & set(all_neg.keys()))
    print(f'  {len(heads)} output heads loaded')

    tissue_auprc = {}
    tissues = sorted(ds_pairs['tissue'].unique())
    for tissue in tissues:
        tp = ds_pairs[ds_pairs['tissue'] == tissue]
        for head in heads:
            yt, ys, n = get_ys(tp, all_pos, all_neg, head)
            if n >= 10:
                tissue_auprc[(tissue, head)] = compute_auprc(yt, ys)

    _th_tissue_auprc[ds_name] = tissue_auprc
    _th_all_heads[ds_name] = heads
    del all_pos, all_neg

families = ['SPT', 'Pangolin v1', 'Pangolin v2']

def _run_tissue_head_eval(exclude_heads, label, save_suffix):
    """run tissue-head evaluation, excluding given heads from best-other candidates"""
    results = []
    for ds_name in _th_tissue_auprc:
        tissue_auprc = _th_tissue_auprc[ds_name]
        candidate_heads = [h for h in _th_all_heads[ds_name] if h not in exclude_heads]
        tissues = sorted(set(t for t, _ in tissue_auprc.keys()))

        for family_name, gtex_map, head_fmt in [
            ('SPT', _GTEX_TO_SPT, 'spt_{tissue}'),
            ('Pangolin v1', _GTEX_TO_PANG, 'pangolin_{tissue}_cls'),
            ('Pangolin v2', _GTEX_TO_PANG, 'pangolin_v2_{tissue}_cls'),
        ]:
            for gtex_tissue in tissues:
                if gtex_tissue not in gtex_map:
                    continue
                model_tissue = gtex_map[gtex_tissue]
                matched_head = head_fmt.format(tissue=model_tissue)
                if (gtex_tissue, matched_head) not in tissue_auprc:
                    continue
                matched_val = tissue_auprc[(gtex_tissue, matched_head)]

                best_other_val = -1
                best_other_head = None
                for head in candidate_heads:
                    if head == matched_head:
                        continue
                    val = tissue_auprc.get((gtex_tissue, head), -1)
                    if val > best_other_val:
                        best_other_val = val
                        best_other_head = head
                if best_other_val < 0:
                    continue

                results.append({
                    'dataset': ds_name, 'family': family_name,
                    'gtex_tissue': gtex_tissue, 'model_tissue': model_tissue,
                    'matched_head': matched_head, 'matched_auprc': matched_val,
                    'best_other_head': best_other_head, 'best_other_auprc': best_other_val,
                })

    thr_df = pd.DataFrame(results)
    if len(thr_df) == 0:
        print(f'{label}: no data')
        return thr_df
    print(f'\n{label}: {len(thr_df)} data points')

    ds_names = [d for d in ['txrevise', 'leafcutter'] if d in thr_df['dataset'].values]
    n_ds = len(ds_names)
    thr_df['delta'] = thr_df['matched_auprc'] - thr_df['best_other_auprc']

    tissue_groups = sorted(thr_df['model_tissue'].unique())
    tg_colors_arr = plt.cm.tab20(np.linspace(0, 1, max(len(tissue_groups), 1)))
    tg_cmap = dict(zip(tissue_groups, tg_colors_arr))
    family_markers = {'SPT': 'o', 'Pangolin v1': '^', 'Pangolin v2': 's'}
    ds_colors = {'txrevise': '#0072B2', 'leafcutter': '#D55E00'}

    # --- option A: 3x2 grid scatter (current) ---
    fig, axes = plt.subplots(len(families), n_ds, figsize=(6 * n_ds, 5 * len(families)),
                              squeeze=False)
    for i, family in enumerate(families):
        for j, ds in enumerate(ds_names):
            ax = axes[i, j]
            sub = thr_df[(thr_df['family'] == family) & (thr_df['dataset'] == ds)]
            if len(sub) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{family} — {ds}')
                continue
            for _, row in sub.iterrows():
                ax.scatter(row['matched_auprc'], row['best_other_auprc'],
                          color=tg_cmap[row['model_tissue']], s=40, alpha=0.7,
                          edgecolors='white', linewidths=0.5)
            lo = min(sub['matched_auprc'].min(), sub['best_other_auprc'].min())
            hi = max(sub['matched_auprc'].max(), sub['best_other_auprc'].max())
            buf = (hi - lo) * 0.05
            lims = [lo - buf, hi + buf]
            ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel('Tissue-matched head AUPRC')
            ax.set_ylabel('Best other head AUPRC')
            ax.set_title(f'{family} — {ds}')
            ax.set_aspect('equal'); ax.grid(alpha=0.2)
            n_above = (sub['best_other_auprc'] > sub['matched_auprc']).sum()
            ax.text(0.05, 0.95, f'{n_above}/{len(sub)} above diag',
                    transform=ax.transAxes, fontsize=9, va='top')
            for tg in sorted(sub['model_tissue'].unique()):
                ax.scatter([], [], color=tg_cmap[tg], label=tg, s=30)
            ax.legend(fontsize=7, loc='lower right', ncol=1)
    plt.suptitle(f'{label} (grid scatter)', y=1.02, fontsize=14)
    plt.tight_layout()
    savefig(FIG_BASE / 'tissue_heads' / f'A_grid_scatter{save_suffix}')
    plt.show()
    plt.close()

    # --- option B: single panel, all overlaid ---
    fig, ax = plt.subplots(figsize=(7, 7))
    for _, row in thr_df.iterrows():
        ax.scatter(row['matched_auprc'], row['best_other_auprc'],
                  marker=family_markers[row['family']],
                  color=ds_colors[row['dataset']],
                  s=35, alpha=0.6, edgecolors='white', linewidths=0.3)
    all_vals = pd.concat([thr_df['matched_auprc'], thr_df['best_other_auprc']])
    lo, hi = all_vals.min(), all_vals.max()
    buf = (hi - lo) * 0.05
    lims = [lo - buf, hi + buf]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Tissue-matched head AUPRC')
    ax.set_ylabel('Best other head AUPRC')
    ax.set_aspect('equal'); ax.grid(alpha=0.2)
    # legend: shapes for family, colors for dataset
    for fam, marker in family_markers.items():
        ax.scatter([], [], marker=marker, color='gray', s=40, label=fam)
    for ds, color in ds_colors.items():
        ax.scatter([], [], marker='o', color=color, s=40, label=ds)
    ax.legend(fontsize=8, loc='lower right')
    n_above = (thr_df['best_other_auprc'] > thr_df['matched_auprc']).sum()
    ax.set_title(f'{n_above}/{len(thr_df)} above diagonal')
    plt.tight_layout()
    savefig(FIG_BASE / 'tissue_heads' / f'B_single_scatter{save_suffix}')
    plt.show()
    plt.close()

    # --- option C: delta swarm ---
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), squeeze=False)
    rng = np.random.default_rng(42)
    for j, ds in enumerate(ds_names):
        ax = axes[0, j]
        sub_ds = thr_df[thr_df['dataset'] == ds]
        for fi, family in enumerate(families):
            sub = sub_ds[sub_ds['family'] == family]
            if len(sub) == 0: continue
            jitter = rng.normal(fi, 0.08, len(sub))
            for _, row in sub.iterrows():
                ax.scatter(jitter[sub.index.get_loc(row.name) if row.name in sub.index else 0],
                          row['delta'],
                          color=tg_cmap[row['model_tissue']], s=30, alpha=0.7,
                          edgecolors='white', linewidths=0.3)
            med = sub['delta'].median()
            ax.hlines(med, fi - 0.3, fi + 0.3, colors='black', lw=2)
            ax.text(fi + 0.35, med, f'{med:.3f}', va='center', fontsize=8)
        ax.axhline(0, color='red', ls='--', lw=1, alpha=0.5)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=30, ha='right')
        ax.set_ylabel('AUPRC delta (tissue head - best other)')
        ax.set_title(ds)
        ax.grid(alpha=0.2, axis='y')
    plt.suptitle(f'{label} (delta swarm)', y=1.02, fontsize=14)
    plt.tight_layout()
    savefig(FIG_BASE / 'tissue_heads' / f'C_delta_swarm{save_suffix}')
    plt.show()
    plt.close()

    # --- option D: paired bar (median tissue head vs median best other) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = 0
    xticks, xlabels = [], []
    bar_w = 0.35
    for family in families:
        for ds in ds_names:
            sub = thr_df[(thr_df['family'] == family) & (thr_df['dataset'] == ds)]
            if len(sub) == 0: continue
            med_matched = sub['matched_auprc'].median()
            med_best = sub['best_other_auprc'].median()
            ax.bar(x_pos - bar_w/2, med_matched, bar_w, color='#8da0cb', edgecolor='white')
            ax.bar(x_pos + bar_w/2, med_best, bar_w, color='#fc8d62', edgecolor='white')
            ax.text(x_pos - bar_w/2, med_matched + 0.003, f'{med_matched:.3f}', ha='center', fontsize=7)
            ax.text(x_pos + bar_w/2, med_best + 0.003, f'{med_best:.3f}', ha='center', fontsize=7)
            xticks.append(x_pos)
            xlabels.append(f'{family}\n{ds}')
            x_pos += 1
        x_pos += 0.5  # gap between families
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('Median AUPRC')
    # legend
    ax.bar([], [], color='#8da0cb', label='tissue-matched head')
    ax.bar([], [], color='#fc8d62', label='best other head')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis='y')
    plt.suptitle(f'{label} (paired bar)', y=1.02, fontsize=14)
    plt.tight_layout()
    savefig(FIG_BASE / 'tissue_heads' / f'D_paired_bar{save_suffix}')
    plt.show()
    plt.close()

    # --- option E: heatmap (tissues × model tissue heads) ---
    # for each dataset, make a heatmap of AUPRC per GTEx tissue × model head
    for ds in ds_names:
        sub = thr_df[thr_df['dataset'] == ds]
        if len(sub) == 0: continue
        # pivot: rows = gtex_tissue, cols = matched_head
        all_matched = sorted(sub['matched_head'].unique())
        all_gtex = sorted(sub['gtex_tissue'].unique())
        # build matrix: tissue AUPRC for each head + the best-other AUPRC
        heads_to_show = all_matched + ['best_other']
        mat = np.full((len(all_gtex), len(heads_to_show)), np.nan)
        for ri, gt in enumerate(all_gtex):
            row_data = sub[sub['gtex_tissue'] == gt]
            for ci, head in enumerate(all_matched):
                key = (gt, head)
                if key in _th_tissue_auprc.get(ds, {}):
                    mat[ri, ci] = _th_tissue_auprc[ds][key]
            # best-other column
            if len(row_data) > 0:
                mat[ri, -1] = row_data['best_other_auprc'].max()

        fig, ax = plt.subplots(figsize=(max(6, len(heads_to_show) * 0.8), max(6, len(all_gtex) * 0.25)))
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xticks(range(len(heads_to_show)))
        head_labels = [MODEL_NAMES.get(h, h) for h in all_matched] + ['best other']
        ax.set_xticklabels(head_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(all_gtex)))
        ax.set_yticklabels(all_gtex, fontsize=6)
        plt.colorbar(im, ax=ax, label='AUPRC', shrink=0.8)
        ax.set_title(f'{label} — {ds}')
        plt.tight_layout()
        savefig(FIG_BASE / 'tissue_heads' / f'E_heatmap_{ds}{save_suffix}')
        plt.show()
        plt.close()

    # summary table
    print(f'\n{"Family":<15s} {"Dataset":<12s} {"N":<5s} {"Tissue wins":<14s} {"Median delta":>13s}')
    print('-' * 60)
    for family in families:
        for ds in ds_names:
            sub = thr_df[(thr_df['family'] == family) & (thr_df['dataset'] == ds)]
            if len(sub) == 0: continue
            n = len(sub)
            wins = (sub['matched_auprc'] >= sub['best_other_auprc']).sum()
            delta = (sub['matched_auprc'] - sub['best_other_auprc']).median()
            print(f'{family:<15s} {ds:<12s} {n:<5d} {wins}/{n:<12} {delta:+13.4f}')

    print(f'\nmost common best-other head:')
    for family in families:
        sub = thr_df[thr_df['family'] == family]
        if len(sub) == 0: continue
        top = sub['best_other_head'].value_counts().head(5)
        print(f'  {family}:')
        for head, count in top.items():
            print(f'    {MODEL_NAMES.get(head, head)}: {count}')

    return thr_df

# version 1: all heads (including aggregates)
_run_tissue_head_eval(set(), 'Tissue-specific vs best alternative (all heads)', '')

# version 2: no aggregate heads
_run_tissue_head_eval(_AGGREGATE_HEADS, 'Tissue-specific vs best alternative (no aggregates)', '_no_agg')


# ## CLS vs SSU/Usage Margin Scatter
#
# For each pos-neg pair, compute margin = pos_delta - neg_delta for both heads.
# Scatter shows which pairs each head gets right. Quadrant I = both win,
# II = head B rescues, IV = head A rescues, III = both fail.

# In[ ]:


print('\n' + '=' * 70)
print('CLS vs SSU/USAGE MARGIN SCATTER')
print('=' * 70)

_margin_models = [
    ('sphaec_ref', 'sphaec_ref', 'sphaec_ref_reg', 'SPLAIRE', 'CLS', 'SSU'),
    ('sphaec_var', 'sphaec_var', 'sphaec_var_reg', 'SPLAIRE-var', 'CLS', 'SSU'),
    ('pangolin', 'pangolin', 'pangolin_usage', 'Pangolin v1', 'CLS', 'Usage'),
    ('pangolin_v2', 'pangolin_v2', 'pangolin_v2_usage', 'Pangolin v2', 'CLS', 'Usage'),
]
# add per-tissue pangolin heads
for _t in _PANG_TISSUES:
    _margin_models.append((f'pangolin_{_t}', f'pangolin_{_t}_cls', f'pangolin_{_t}_usage', f'Pangolin v1 {_t}', 'CLS', 'Usage'))
    _margin_models.append((f'pangolin_v2_{_t}', f'pangolin_v2_{_t}_cls', f'pangolin_v2_{_t}_usage', f'Pangolin v2 {_t}', 'CLS', 'Usage'))

for ds_name, ds_pairs, score_dir_cs in [
    ('txrevise', pairs, DATA_BASE / 'txrevise' / 'scores'),
    ('leafcutter', pairs_lc, DATA_BASE / 'leafcutter_pip50' / 'scores'),
]:
    if ds_pairs is None:
        continue

    cs_pos = load_scores(score_dir_cs, 'pos', primary_only=False)
    cs_neg = load_scores(score_dir_cs, 'neg', primary_only=False)

    for model_base, head_a_key, head_b_key, label, a_name, b_name in _margin_models:
        if head_a_key not in cs_pos or head_b_key not in cs_pos:
            continue

        a_pos_d = np.array([cs_pos[head_a_key].get(k, np.nan) for k in ds_pairs['pos_var_key']])
        a_neg_d = np.array([cs_neg[head_a_key].get(k, np.nan) for k in ds_pairs['neg_var_key']])
        b_pos_d = np.array([cs_pos[head_b_key].get(k, np.nan) for k in ds_pairs['pos_var_key']])
        b_neg_d = np.array([cs_neg[head_b_key].get(k, np.nan) for k in ds_pairs['neg_var_key']])

        valid = ~(np.isnan(a_pos_d) | np.isnan(a_neg_d) | np.isnan(b_pos_d) | np.isnan(b_neg_d))
        a_margin = (a_pos_d - a_neg_d)[valid]
        b_margin = (b_pos_d - b_neg_d)[valid]

        a_win = a_margin > 0
        b_win = b_margin > 0
        both_win = a_win & b_win
        a_rescue = a_win & ~b_win
        b_rescue = b_win & ~a_win
        both_fail = ~a_win & ~b_win

        n = valid.sum()
        print(f'\n{label} — {ds_name} ({n:,} valid pairs):')
        print(f'  both win:      {both_win.sum():,} ({100*both_win.mean():.1f}%)')
        print(f'  {a_name} rescue:  {a_rescue.sum():,} ({100*a_rescue.mean():.1f}%)')
        print(f'  {b_name} rescue:  {b_rescue.sum():,} ({100*b_rescue.mean():.1f}%)')
        print(f'  both fail:     {both_fail.sum():,} ({100*both_fail.mean():.1f}%)')

        fig, ax = plt.subplots(figsize=(7, 7))
        if n > 5000:
            idx = np.random.default_rng(42).choice(n, 5000, replace=False)
        else:
            idx = np.arange(n)
        cats = np.where(both_win, 0, np.where(a_rescue, 1, np.where(b_rescue, 2, 3)))
        cat_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        cat_labels = ['both win', f'{a_name} rescue', f'{b_name} rescue', 'both fail']
        for ci in [3, 2, 1, 0]:
            mask = cats[idx] == ci
            if mask.sum() == 0: continue
            ax.scatter(a_margin[idx][mask], b_margin[idx][mask],
                      s=8, alpha=0.3, color=cat_colors[ci], label=f'{cat_labels[ci]} ({(cats == ci).sum():,})')
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.axvline(0, color='black', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'{a_name} margin (pos - neg delta)')
        ax.set_ylabel(f'{b_name} margin (pos - neg delta)')
        ax.set_title(f'{label} — {ds_name}')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_aspect('equal')
        lim = max(abs(a_margin[idx]).max(), abs(b_margin[idx]).max()) * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        savefig(FIG_BASE / 'cls_vs_ssu' / f'{model_base}_{ds_name}_margin_scatter')
        plt.show()
        plt.close()

    del cs_pos, cs_neg


# # PIP Threshold Analysis (HAEC)
# 
# Classification performance as a function of the minimum PIP threshold used to select positive variants. The pip50 dataset includes all variants with PIP >= 0.5, matched to negatives using the same 4-tier cascade. By filtering pairs at different PIP thresholds after scoring, we evaluate how label confidence affects benchmark results without re-scoring.
# 
# The `pos_pip` column in pairs.csv records each positive's actual PIP value. The `neg_var_key_ideal` column records what negative each positive would have received if negatives were shared (ignoring the exclusive matching constraint).

# In[ ]:


# load haec pip50 data
haec_pip50_dir = DATA_BASE / 'haec_pip50'
if (haec_pip50_dir / 'pairs.csv').exists():
    pairs_haec_pip50 = pd.read_csv(haec_pip50_dir / 'pairs.csv')
    score_dir_pip50 = haec_pip50_dir / 'scores'
    pos_scores_haec_pip50 = load_scores(score_dir_pip50, 'pos')
    neg_scores_haec_pip50 = load_scores(score_dir_pip50, 'neg')
    distances_haec_pip50 = pd.read_csv(haec_pip50_dir / 'distances.csv') if (haec_pip50_dir / 'distances.csv').exists() else None

    print(f'HAEC pip50: {len(pairs_haec_pip50):,} pairs, {pairs_haec_pip50["pos_var_key"].nunique():,} unique positives')
    print(f'  models: {list(pos_scores_haec_pip50.keys())}')
    print(f'  PIP range: {pairs_haec_pip50["pos_pip"].min():.2f} - {pairs_haec_pip50["pos_pip"].max():.2f}')
    print()

    # pip distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pairs_haec_pip50['pos_pip'], bins=50, color='#66c2a5', edgecolor='white')
    ax.axvline(0.9, color='red', ls='--', label='PIP >= 0.9')
    ax.set_xlabel('Positive PIP')
    ax.set_ylabel('Count')
    ax.set_title('HAEC pip50: PIP distribution of matched positives')
    ax.legend()
    plt.tight_layout()
    savefig(FIG_BASE / 'haec_pip50' / 'pip_distribution')
    plt.show()
    plt.close()

    # match quality: how often does ideal != actual?
    if 'neg_var_key_ideal' in pairs_haec_pip50.columns:
        mismatch = (pairs_haec_pip50['neg_var_key'] != pairs_haec_pip50['neg_var_key_ideal']).sum()
        print(f'negative match degradation: {mismatch:,}/{len(pairs_haec_pip50):,} pairs ({100*mismatch/len(pairs_haec_pip50):.1f}%) got a different negative than ideal')
else:
    pairs_haec_pip50 = None
    print('HAEC pip50: not found')


# In[ ]:


# AUPRC by PIP threshold — HAEC pip50
if pairs_haec_pip50 is not None and pos_scores_haec_pip50:
    pip_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    mp = [m for m in MODELS if m in pos_scores_haec_pip50 and m in neg_scores_haec_pip50]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) AUPRC by PIP threshold
    ax = axes[0]
    for m in mp:
        auprcs = []
        for pip_thresh in pip_thresholds:
            sub = pairs_haec_pip50[pairs_haec_pip50['pos_pip'] >= pip_thresh]
            if len(sub) < 20:
                auprcs.append(np.nan)
                continue
            yt, ys, n = get_ys(sub, pos_scores_haec_pip50, neg_scores_haec_pip50, m)
            auprcs.append(compute_auprc(yt, ys) if n >= 10 else np.nan)
        ax.plot(pip_thresholds, auprcs, marker='o', color=COLORS.get(m, 'gray'),
                label=MODEL_NAMES.get(m, m), lw=2, markersize=6)
    # variant counts on twin axis
    ax2 = ax.twinx()
    bw = 0.025
    pos_counts = [pairs_haec_pip50[pairs_haec_pip50['pos_pip'] >= t]['pos_var_key'].nunique() for t in pip_thresholds]
    neg_counts = [pairs_haec_pip50[pairs_haec_pip50['pos_pip'] >= t]['neg_var_key'].nunique() for t in pip_thresholds]
    ax2.bar(np.array(pip_thresholds) - bw/2, pos_counts, bw, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.bar(np.array(pip_thresholds) + bw/2, neg_counts, bw, color='#606060', alpha=0.4, label='neg variants')
    ax2.set_ylabel('unique variants', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax.set_xlabel('Minimum PIP Threshold')
    ax.set_ylabel('AUPRC')
    ax.set_title('HAEC: AUPRC by PIP Threshold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # (b) AUROC by PIP threshold
    ax = axes[1]
    for m in mp:
        aurocs = []
        for pip_thresh in pip_thresholds:
            sub = pairs_haec_pip50[pairs_haec_pip50['pos_pip'] >= pip_thresh]
            if len(sub) < 20:
                aurocs.append(np.nan)
                continue
            yt, ys, n = get_ys(sub, pos_scores_haec_pip50, neg_scores_haec_pip50, m)
            aurocs.append(compute_auroc(yt, ys) if n >= 10 else np.nan)
        ax.plot(pip_thresholds, aurocs, marker='o', color=COLORS.get(m, 'gray'),
                label=MODEL_NAMES.get(m, m), lw=2, markersize=6)
    ax2 = ax.twinx()
    ax2.bar(np.array(pip_thresholds) - bw/2, pos_counts, bw, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.bar(np.array(pip_thresholds) + bw/2, neg_counts, bw, color='#606060', alpha=0.4, label='neg variants')
    ax2.set_ylabel('unique variants', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax.set_xlabel('Minimum PIP Threshold')
    ax.set_ylabel('AUROC')
    ax.set_title('HAEC: AUROC by PIP Threshold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig(FIG_BASE / 'haec_pip50' / 'auprc_by_pip_threshold')
    plt.show()
    plt.close()

    # print table
    print('HAEC pip50: AUPRC / AUROC by PIP threshold')
    print(f'{"Model":<25s}', end='')
    for t in pip_thresholds:
        print(f'  PIP>={t:.1f}', end='')
    print()
    for m in mp:
        print(f'{MODEL_NAMES.get(m, m):<25s}', end='')
        for pip_thresh in pip_thresholds:
            sub = pairs_haec_pip50[pairs_haec_pip50['pos_pip'] >= pip_thresh]
            yt, ys, n = get_ys(sub, pos_scores_haec_pip50, neg_scores_haec_pip50, m)
            auprc = compute_auprc(yt, ys) if n >= 10 else np.nan
            print(f'  {auprc:7.3f}', end='')
        print()
    print(f'{"n pairs":<25s}', end='')
    for t in pip_thresholds:
        n = (pairs_haec_pip50['pos_pip'] >= t).sum()
        print(f'  {n:7d}', end='')
    print()


# In[ ]:


# === PIP Threshold Analysis — all datasets ===
# load pip50 data for txrevise and leafcutter alongside haec
# for each: (1) verify pip90 subset matches original, (2) show AUPRC by PIP threshold,
# (3) check negative match degradation

pip50_datasets = {}
for ds_name, ds_dir_name, orig_pairs, orig_pos, orig_neg in [
    ('HAEC', 'haec_pip50', pairs_haec if 'pairs_haec' in dir() else None,
     pos_scores_haec if 'pos_scores_haec' in dir() else {},
     neg_scores_haec if 'neg_scores_haec' in dir() else {}),
    ('GTEx Txrevise', 'txrevise_pip50', pairs if 'pairs' in dir() else None,
     pos_scores if 'pos_scores' in dir() else {},
     neg_scores if 'neg_scores' in dir() else {}),
    # leafcutter: no pip90 original to compare (buggy), skip verification
    ('GTEx Leafcutter', 'leafcutter_pip50', None, {}, {}),
]:
    pip50_dir = DATA_BASE / ds_dir_name
    if not (pip50_dir / 'pairs.csv').exists():
        print(f'{ds_name} pip50: pairs.csv not found, skipping')
        continue

    p50 = pd.read_csv(pip50_dir / 'pairs.csv')
    score_dir_p50 = pip50_dir / 'scores'
    ps50 = load_scores(score_dir_p50, 'pos') if score_dir_p50.exists() else {}
    ns50 = load_scores(score_dir_p50, 'neg') if score_dir_p50.exists() else {}

    if not ps50:
        print(f'{ds_name} pip50: no scores yet, skipping')
        continue

    mp = [m for m in MODELS if m in ps50 and m in ns50]
    has_tissue = 'tissue' in p50.columns
    n_tissues = p50['tissue'].nunique() if has_tissue else 1

    print(f'\n{"=" * 70}')
    print(f'{ds_name} PIP Threshold Analysis')
    print(f'{"=" * 70}')
    print(f'  total pairs: {len(p50):,}, unique positives: {p50["pos_var_key"].nunique():,}')
    print(f'  PIP range: {p50["pos_pip"].min():.2f} - {p50["pos_pip"].max():.2f}')
    print(f'  models: {[MODEL_NAMES.get(m, m) for m in mp]}')

    # negative match degradation
    if 'neg_var_key_ideal' in p50.columns:
        mismatch = (p50['neg_var_key'] != p50['neg_var_key_ideal']).sum()
        print(f'  negative degradation: {mismatch:,}/{len(p50):,} ({100*mismatch/len(p50):.1f}%) got non-ideal negative')
        # breakdown by PIP bin
        for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7)]:
            sub = p50[(p50['pos_pip'] >= lo) & (p50['pos_pip'] < hi)]
            if len(sub) > 0:
                mm = (sub['neg_var_key'] != sub['neg_var_key_ideal']).sum()
                print(f'    PIP [{lo:.1f}, {hi:.1f}): {mm:,}/{len(sub):,} ({100*mm/len(sub):.1f}%) degraded')

    # --- verify pip90 subset matches original ---
    if orig_pairs is not None:
        p90_from_pip50 = p50[p50['pos_pip'] >= 0.9]
        print(f'\n  pip90 verification:')
        print(f'    original pip90 pairs: {len(orig_pairs):,}')
        print(f'    pip50 subset (pip>=0.9): {len(p90_from_pip50):,}')
        # check if same pos_var_key sets
        orig_pos_keys = set(orig_pairs['pos_var_key'])
        p50_pos_keys = set(p90_from_pip50['pos_var_key'])
        overlap = orig_pos_keys & p50_pos_keys
        only_orig = orig_pos_keys - p50_pos_keys
        only_p50 = p50_pos_keys - orig_pos_keys
        print(f'    shared positives: {len(overlap):,}')
        print(f'    only in original: {len(only_orig):,}')
        print(f'    only in pip50: {len(only_p50):,}')
        # compare AUPRC for shared positives
        if mp and len(overlap) > 10:
            m0 = mp[0]
            # original
            if has_tissue:
                orig_auprcs = []
                for tissue in orig_pairs['tissue'].unique():
                    tp = orig_pairs[orig_pairs['tissue'] == tissue]
                    yt, ys, n = get_ys(tp, orig_pos, orig_neg, m0)
                    if n >= 10:
                        orig_auprcs.append(compute_auprc(yt, ys))
                orig_med = np.median(orig_auprcs) if orig_auprcs else np.nan
            else:
                yt, ys, n = get_ys(orig_pairs, orig_pos, orig_neg, m0)
                orig_med = compute_auprc(yt, ys) if n >= 10 else np.nan
            # pip50 subset at pip>=0.9
            if has_tissue:
                p50_auprcs = []
                for tissue in p90_from_pip50['tissue'].unique():
                    tp = p90_from_pip50[p90_from_pip50['tissue'] == tissue]
                    yt, ys, n = get_ys(tp, ps50, ns50, m0)
                    if n >= 10:
                        p50_auprcs.append(compute_auprc(yt, ys))
                p50_med = np.median(p50_auprcs) if p50_auprcs else np.nan
            else:
                yt, ys, n = get_ys(p90_from_pip50, ps50, ns50, m0)
                p50_med = compute_auprc(yt, ys) if n >= 10 else np.nan
            print(f'    {MODEL_NAMES.get(m0, m0)} AUPRC: original={orig_med:.3f}, pip50@0.9={p50_med:.3f}, diff={p50_med-orig_med:+.3f}')

    # --- PIP distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(p50['pos_pip'], bins=50, color='#66c2a5', edgecolor='white')
    ax.axvline(0.9, color='red', ls='--', label='PIP >= 0.9')
    ax.set_xlabel('Positive PIP')
    ax.set_ylabel('Count')
    ax.set_title(f'{ds_name} pip50: PIP distribution of matched positives')
    ax.legend()
    plt.tight_layout()
    savefig(FIG_BASE / ds_dir_name / 'pip_distribution')
    plt.show()
    plt.close()
    plt.close()

    # --- AUPRC/AUROC by PIP threshold (actual vs ideal negatives) ---
    pip_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    has_ideal = 'neg_var_key_ideal' in p50.columns

    def _pip_metric(sub, m, metric_fn, neg_col='neg_var_key'):
        """compute metric for a pip-filtered subset"""
        if has_tissue:
            tissue_vals = []
            for tissue in sub['tissue'].unique():
                tp = sub[sub['tissue'] == tissue]
                yt, ys, n = get_ys(tp, ps50, ns50, m, neg_col=neg_col)
                if n >= 10:
                    tissue_vals.append(metric_fn(yt, ys))
            return np.median(tissue_vals) if tissue_vals else np.nan
        else:
            yt, ys, n = get_ys(sub, ps50, ns50, m, neg_col=neg_col)
            return metric_fn(yt, ys) if n >= 10 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bw_pip = 0.025
    pip_pos_counts = [p50[p50['pos_pip'] >= t]['pos_var_key'].nunique() for t in pip_thresholds]
    pip_neg_counts = [p50[p50['pos_pip'] >= t]['neg_var_key'].nunique() for t in pip_thresholds]

    for ax_idx, (metric_name, metric_fn) in enumerate([('AUPRC', compute_auprc), ('AUROC', compute_auroc)]):
        ax = axes[ax_idx]
        for m in mp:
            # actual negatives (solid)
            vals = []
            for pip_thresh in pip_thresholds:
                sub = p50[p50['pos_pip'] >= pip_thresh]
                if len(sub) < 20: vals.append(np.nan); continue
                vals.append(_pip_metric(sub, m, metric_fn, 'neg_var_key'))
            ax.plot(pip_thresholds, vals, marker='o', color=COLORS.get(m, 'gray'),
                    label=MODEL_NAMES.get(m, m), lw=2, markersize=6)
            # ideal negatives (dashed)
            if has_ideal:
                vals_ideal = []
                for pip_thresh in pip_thresholds:
                    sub = p50[p50['pos_pip'] >= pip_thresh]
                    if len(sub) < 20: vals_ideal.append(np.nan); continue
                    vals_ideal.append(_pip_metric(sub, m, metric_fn, 'neg_var_key_ideal'))
                ax.plot(pip_thresholds, vals_ideal, marker='s', color=COLORS.get(m, 'gray'),
                        ls='--', lw=1.5, markersize=4, alpha=0.6)
        ax2 = ax.twinx()
        ax2.bar(np.array(pip_thresholds) - bw_pip/2, pip_pos_counts, bw_pip, color='#b0b0b0', alpha=0.4, label='pos variants')
        ax2.bar(np.array(pip_thresholds) + bw_pip/2, pip_neg_counts, bw_pip, color='#606060', alpha=0.4, label='neg variants')
        ax2.set_ylabel('unique variants', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax.set_xlabel('Minimum PIP Threshold')
        ax.set_ylabel(metric_name)
        title_sfx = ' (solid=actual, dashed=ideal neg)' if has_ideal else ''
        ax.set_title(f'{ds_name}: {metric_name} by PIP Threshold{title_sfx}')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    savefig(FIG_BASE / ds_dir_name / 'auprc_by_pip_threshold')
    plt.show()
    plt.close()

    # print table: actual vs ideal
    agg_label = 'tissue median' if has_tissue else 'pooled'
    print(f'\n{ds_name} pip50: AUPRC by PIP threshold ({agg_label})')
    print(f'{"Model":<25s}', end='')
    for t in pip_thresholds:
        print(f'  PIP>={t:.1f}', end='')
    print()
    for m in mp:
        # actual
        print(f'{MODEL_NAMES.get(m, m):<25s}', end='')
        for pip_thresh in pip_thresholds:
            sub = p50[p50['pos_pip'] >= pip_thresh]
            val = _pip_metric(sub, m, compute_auprc, 'neg_var_key')
            print(f'  {val:7.3f}', end='')
        print()
        # ideal
        if has_ideal:
            print(f'{"  (ideal neg)":<25s}', end='')
            for pip_thresh in pip_thresholds:
                sub = p50[p50['pos_pip'] >= pip_thresh]
                val = _pip_metric(sub, m, compute_auprc, 'neg_var_key_ideal')
                print(f'  {val:7.3f}', end='')
            print()
    print(f'{"n pairs":<25s}', end='')
    for t in pip_thresholds:
        print(f'  {(p50["pos_pip"] >= t).sum():7d}', end='')
    print()
    # degradation summary per threshold
    if has_ideal:
        print(f'{"degraded neg %":<25s}', end='')
        for t in pip_thresholds:
            sub = p50[p50['pos_pip'] >= t]
            mm = (sub['neg_var_key'] != sub['neg_var_key_ideal']).sum()
            print(f'  {100*mm/len(sub):6.1f}%', end='')
        print()

    pip50_datasets[ds_name] = {'pairs': p50, 'pos_scores': ps50, 'neg_scores': ns50, 'dir': ds_dir_name}


#
# # Leafcutter Ambiguous Analysis
# 
# **Task:** Within-credible-set ranking. Given multiple variants in a credible set, does the model rank them by their fine-mapping probability (PIP)?
# 
# **Variants:** From GTEx leafcutter credible sets where no single variant dominates (max PIP < 0.9). All variants within each credible set are included. Filtered to credible sets with 2+ variants where all variants are near the same intron.
# 
# **Unit of analysis:** Each (credible set, tissue) pair.

# ## Leafcutter Ambig QC
# 
# **CS Filtering Funnel:** How many credible sets pass each filter (size >= 2, same intron, etc.).
# 
# **Variants Per Credible Set:** Distribution of credible set sizes after filtering.

# In[37]:


ambig_dir = DATA_BASE / 'leafcutter_ambig'
cs_data = pd.read_csv(ambig_dir / 'cs_data.csv')
cs_stats = pd.read_csv(ambig_dir / 'cs_stats.csv')
filter_funnel = pd.read_csv(ambig_dir / 'filter_funnel.csv')
size_dist = pd.read_csv(ambig_dir / 'cs_size_distribution.csv')

# merge delta scores from h5 files into cs_data
score_dir = ambig_dir / 'scores'
for sfx, model in [('sa', 'spliceai'), ('spt', 'splicetransformer'),
                    ('sphaec.ref', 'sphaec_ref'), ('sphaec.var', 'sphaec_var')]:
    p = score_dir / f'ambig.{sfx}.h5'
    if not p.exists(): continue
    with h5py.File(p, 'r') as f:
        keys = [k.decode() for k in f['var_key'][:]]
        if 'cls_alt' in f:
            delta = np.max(np.abs(f['cls_alt'][:] - f['cls_ref'][:]), axis=(1, 2))
        else:
            delta = np.max(np.abs(f['alt'][:] - f['ref'][:]), axis=(1, 2))
        mapping = dict(zip(keys, delta))
    cs_data[f'delta_{model}'] = cs_data['var_key'].map(mapping)
    n_mapped = cs_data[f'delta_{model}'].notna().sum()
    print(f'  {model}: {n_mapped:,}/{len(cs_data):,} mapped')

# pangolin v1: max across tissue tasks
pp = score_dir / 'ambig.pang.h5'
if pp.exists():
    with h5py.File(pp, 'r') as f:
        keys = [k.decode() for k in f['var_key'][:]]
        tasks = [k[4:] for k in f.keys() if k.startswith('ref_')]
        md = None
        for t in tasks:
            d = np.max(np.abs(f[f'alt_{t}'][:] - f[f'ref_{t}'][:]), axis=1)
            md = d if md is None else np.maximum(md, d)
        mapping = dict(zip(keys, md))
    cs_data['delta_pangolin'] = cs_data['var_key'].map(mapping)
    n_mapped = cs_data['delta_pangolin'].notna().sum()
    print(f'  pangolin v1: {n_mapped:,}/{len(cs_data):,} mapped')

# pangolin v2 (variant-finetuned): max across tissue tasks
pp2 = score_dir / 'ambig.pang_v2.h5'
if pp2.exists():
    with h5py.File(pp2, 'r') as f:
        keys = [k.decode() for k in f['var_key'][:]]
        tasks = [k[4:] for k in f.keys() if k.startswith('ref_')]
        md = None
        for t in tasks:
            d = np.max(np.abs(f[f'alt_{t}'][:] - f[f'ref_{t}'][:]), axis=1)
            md = d if md is None else np.maximum(md, d)
        mapping = dict(zip(keys, md))
    cs_data['delta_pangolin_v2'] = cs_data['var_key'].map(mapping)
    n_mapped = cs_data['delta_pangolin_v2'].notna().sum()
    print(f'  pangolin v2: {n_mapped:,}/{len(cs_data):,} mapped')

# which models have delta scores
models_present = [m for m in MODELS if f'delta_{m}' in cs_data.columns]

print(f'\ncredible sets: {len(cs_stats):,}')
print(f'variant rows: {len(cs_data):,}')
print(f'unique variants: {cs_data["var_key"].nunique():,}')
print(f'models: {[MODEL_NAMES[m] for m in models_present]}')


# In[38]:


# filter funnel
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(filter_funnel['stage'], filter_funnel['count'], color='steelblue')
ax.set_xlabel('Number of Credible Sets')
ax.set_title('Leafcutter Ambig - Filter Funnel')
ax.invert_yaxis()
for bar, count in zip(bars, filter_funnel['count']):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, f'{count:,}', va='center')
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter_ambig' / 'qc' / '02_cs_filtering_funnel')
plt.show()
plt.close()


# In[39]:


# do credible sets across tissues contain the same variants?
# for each cs_id, get the variant set per tissue and check if they're identical
var_sets = cs_data.groupby(['cs_id', 'tissue'])['var_key'].apply(frozenset).reset_index()
var_sets.columns = ['cs_id', 'tissue', 'var_set']

cs_sharing = var_sets.groupby('cs_id').agg(
    n_tissues=('tissue', 'count'),
    n_unique_sets=('var_set', 'nunique'),
).reset_index()

multi = cs_sharing.query('n_tissues > 1')
same = multi.query('n_unique_sets == 1')
diff = multi.query('n_unique_sets > 1')

print(f'total credible sets: {len(cs_sharing):,}')
print(f'  single tissue only: {(cs_sharing["n_tissues"] == 1).sum():,}')
print(f'  multi-tissue: {len(multi):,}')
print(f'    same variants across tissues: {len(same):,} ({len(same)/len(multi)*100:.1f}%)')
print(f'    different variants across tissues: {len(diff):,} ({len(diff)/len(multi)*100:.1f}%)')

# tissue count distribution for multi-tissue CS
print(f'\ntissue count distribution (multi-tissue CS):')
for n, count in multi['n_tissues'].value_counts().sort_index().items():
    print(f'  {n} tissues: {count:,} CS')


# In[40]:


# CS size distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(size_dist['n_variants'], size_dist['count'], color='steelblue')
ax.set_xlabel('Variants per Credible Set'); ax.set_ylabel('Count')
ax.set_title(f'CS Size Distribution (n={size_dist["count"].sum():,})')
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter_ambig' / 'qc' / '03_variants_per_cs')
plt.show()
plt.close()


# ## Leafcutter Ambig Results
# 
# **Delta Distributions by CS Size:** Delta score distributions stratified by credible set size.
# 
# **PIP vs Delta Correlation:** Correlation between PIP and delta score across all variants.
# 
# **Top-1 Accuracy by CS Size:** For each (cs, tissue) pair, does the model's highest-scoring variant match the highest-PIP variant?
# 
# ### Binary Threshold Analysis
# 
# **Youden's J Per Model:** Optimal threshold using Youden's J statistic.
# 
# **Exactly-1 Positive Call Rate:** Using each model's Youden's J threshold, what percentage of credible sets have exactly one variant called positive? Stratified by CS size.
# 
# **Threshold Sweep:** Precision/recall at various delta score thresholds.

# In[41]:


# delta distributions by CS size
sizes = sorted(cs_data['n_variants_in_cs'].unique())[:5]
fig, axes = plt.subplots(len(models_present), len(sizes), figsize=(3*len(sizes), 3*len(models_present)),
                         sharex=True, sharey='row')
if len(models_present) == 1: axes = axes[np.newaxis, :]
for i, model in enumerate(models_present):
    for j, sz in enumerate(sizes):
        ax = axes[i, j]
        sub = cs_data[cs_data['n_variants_in_cs'] == sz][f'delta_{model}'].dropna()
        ax.hist(sub, bins=30, color=COLORS[model], alpha=0.7)
        if i == 0: ax.set_title(f'CS size={sz}')
        if j == 0: ax.set_ylabel(MODEL_NAMES[model])
fig.suptitle('Delta Distributions by CS Size', y=1.02)
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter_ambig' / 'results' / '02_delta_distributions')
plt.show()
plt.close()


# In[42]:


# PIP vs delta correlation
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
axes = axes.flatten()
for ax, model in zip(axes, models_present):
    valid = cs_data[['pip', f'delta_{model}']].dropna()
    rho, pval = spearmanr(valid['pip'], valid[f'delta_{model}'])
    ax.scatter(valid['pip'], valid[f'delta_{model}'], alpha=0.3, s=10, color=COLORS[model])
    ax.set_xlabel('PIP'); ax.set_ylabel('Delta Score')
    ax.set_title(f'{MODEL_NAMES[model]}\nSpearman rho={rho:.3f}')
for ax in axes[len(models_present):]: ax.set_visible(False)
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter_ambig' / 'results' / '03_pip_vs_delta')
plt.show()
plt.close()


# In[43]:


# top-1 accuracy
def compute_top1(cs_data, model):
    delta_col = f'delta_{model}'
    results = []
    for (tissue, cs_id), group in cs_data.groupby(['tissue', 'cs_id']):
        group = group.dropna(subset=['pip', delta_col])
        if len(group) < 2: continue
        results.append({
            'tissue': tissue, 'cs_id': cs_id, 'n_variants': len(group),
            'match': group.loc[group['pip'].idxmax(), 'var_key'] == group.loc[group[delta_col].idxmax(), 'var_key']
        })
    return pd.DataFrame(results)

top1_results = {}
for model in models_present:
    res = compute_top1(cs_data, model)
    if res is not None and len(res) > 0:
        top1_results[model] = res
        print(f'{MODEL_NAMES[model]}: {res["match"].mean():.1%} top-1 accuracy ({len(res)} CS)')


# Below plot shows the percentage of CS in each CS size bin in which the highest PIP variant is also the highest predicted delta score
# regardless of the magnitude of predicted delta.

# In[44]:


# top-1 by CS size (grey bars for counts, lines for accuracy)
sizes = sorted(cs_data["n_variants_in_cs"].unique())
bar_x = np.arange(len(sizes))

# cs counts per size
cs_counts = [cs_data[cs_data["n_variants_in_cs"] == s][["tissue", "cs_id"]].drop_duplicates().shape[0] for s in sizes]

fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()

# grey bars for counts (secondary axis)
ax2.bar(bar_x, cs_counts, color="lightgray", alpha=0.5, zorder=1, width=0.6)
ax2.set_ylabel("number of (CS, tissue) pairs", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
for i, cnt in enumerate(cs_counts):
    if cnt > 0:
        ax2.text(i, cnt + max(cs_counts) * 0.02, f"{cnt:,}", ha="center", fontsize=8, color="gray")

# line plots for each model (primary axis)
for model in models_present:
    if model not in top1_results: continue
    res = top1_results[model]
    accs = [res[res["n_variants"] == s]["match"].mean() * 100 if len(res[res["n_variants"] == s]) > 0 else np.nan for s in sizes]
    ax.plot(bar_x, accs, "o-", label=MODEL_NAMES[model], color=COLORS[model], markersize=8, lw=2, zorder=3)

# random baseline
ax.plot(bar_x, [100/s for s in sizes], "k--", alpha=0.5, label="random (1/n)", lw=1.5)

ax.set_xticks(bar_x)
ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel("credible set size")
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title("Top-1 Accuracy by CS Size (highest delta = highest PIP)")
ax.legend(fontsize=8, loc="upper right")
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)

plt.tight_layout()
savefig(FIG_BASE / "leafcutter_ambig" / "results" / "09b_top1_by_cs_size")
plt.show()
plt.close()


# In[45]:


# Youden's J thresholds from leafcutter pos/neg classification
lc_binned_thresholds = {}
for model in [m for m in MODELS if m in pos_scores_lc]:
    yt, ys, n = get_ys(pairs_lc, pos_scores_lc, neg_scores_lc, model)
    if n < 10:
        continue
    fpr, tpr, thresholds = roc_curve(yt, ys)
    j_stats = tpr - fpr
    best_idx = np.argmax(j_stats)
    lc_binned_thresholds[model] = thresholds[best_idx]
    print(f"{MODEL_NAMES[model]}: Youden's J threshold = {thresholds[best_idx]:.4f} (J = {j_stats[best_idx]:.3f})")


# In[ ]:


# exactly-1 positive rate by CS size (using Youden's J thresholds from leafcutter binned)
sizes = sorted(cs_data["n_variants_in_cs"].unique())
bar_x = np.arange(len(sizes))

# cs counts per size
cs_counts = [cs_data[cs_data["n_variants_in_cs"] == s][["tissue", "cs_id"]].drop_duplicates().shape[0] for s in sizes]

fig, ax = plt.subplots(figsize=(10, 5))
ax2 = ax.twinx()

# grey bars for counts (secondary axis)
ax2.bar(bar_x, cs_counts, color="lightgray", alpha=0.5, zorder=1, width=0.6)
ax2.set_ylabel("number of (CS, tissue) pairs", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
for i, cnt in enumerate(cs_counts):
    if cnt > 0:
        ax2.text(i, cnt + max(cs_counts) * 0.02, f"{cnt:,}", ha="center", fontsize=8, color="gray")

# line plots: % of CS where exactly 1 variant exceeds threshold
for model in models_present:
    if model not in lc_binned_thresholds:
        continue
    thresh = lc_binned_thresholds[model]
    delta_col = f'delta_{model}'

    pcts = []
    for s in sizes:
        sub = cs_data[cs_data["n_variants_in_cs"] == s].dropna(subset=[delta_col])
        n_exactly_1 = 0
        n_total = 0
        for (tissue, cs_id), group in sub.groupby(["tissue", "cs_id"]):
            n_called = (group[delta_col] >= thresh).sum()
            n_total += 1
            if n_called == 1:
                n_exactly_1 += 1
        pcts.append(100 * n_exactly_1 / n_total if n_total > 0 else np.nan)

    ax.plot(bar_x, pcts, "o-", label=f"{MODEL_NAMES[model]} (t={thresh:.3f})",
            color=COLORS[model], markersize=8, lw=2, zorder=3)

ax.set_xticks(bar_x)
ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel("credible set size")
ax.set_ylabel("% with exactly 1 positive call")
ax.set_title("Exactly-1 Positive Call Rate by CS Size\n(Youden's J threshold from leafcutter binned pos/neg)")
ax.legend(fontsize=8, loc="upper right")
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)

plt.tight_layout()
savefig(FIG_BASE / "leafcutter_ambig" / "results" / "09c_exactly1_by_cs_size")
plt.show()
plt.close()


# In[ ]:


# distribution of positive calls per CS by size (using Youden's J from leafcutter binned)
sizes = sorted(cs_data["n_variants_in_cs"].unique())
mp = [m for m in models_present if m in lc_binned_thresholds]
n_models = len(mp)
max_size = max(sizes)

# categories 0 through max_size
cat_labels = [str(i) for i in range(max_size + 1)]
# distinct colors for each count
cat_colors = [
    "#d62728",  # 0 - red
    "#2ca02c",  # 1 - green
    "#1f77b4",  # 2 - blue
    "#ff7f0e",  # 3 - orange
    "#9467bd",  # 4 - purple
    "#8c564b",  # 5 - brown
    "#e377c2",  # 6 - pink
    "#7f7f7f",  # 7 - gray
    "#bcbd22",  # 8 - olive
    "#17becf",  # 9 - cyan
    "#aec7e8",  # 10 - light blue
]

def _make_ncalled_figure(thresholds_dict, fixed_thresh, title_suffix, save_name):
    """stacked bar: % of CS with 0,1,...,N positive calls per size bin"""
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, mp):
        thresh = fixed_thresh if fixed_thresh is not None else thresholds_dict[model]
        delta_col = f"delta_{model}"

        dist_pcts = {cat: [] for cat in cat_labels}
        for s in sizes:
            sub = cs_data[cs_data["n_variants_in_cs"] == s].dropna(subset=[delta_col])
            counts = {cat: 0 for cat in cat_labels}
            n_total = 0
            for (tissue, cs_id), group in sub.groupby(["tissue", "cs_id"]):
                n_called = int((group[delta_col] >= thresh).sum())
                n_total += 1
                key = str(min(n_called, max_size))
                counts[key] += 1
            for cat in cat_labels:
                dist_pcts[cat].append(100 * counts[cat] / n_total if n_total > 0 else 0)

        bar_x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for cat, color in zip(cat_labels, cat_colors):
            vals = np.array(dist_pcts[cat])
            if vals.sum() == 0:
                continue
            ax.bar(bar_x, vals, bottom=bottom, color=color, label=cat, width=0.7,
                   edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if v >= 5:
                    ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(bar_x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("credible set size")
        ax.set_title(f"{MODEL_NAMES[model]}\n(t={thresh:.4f})", fontsize=10)

    axes[0].set_ylabel("% of credible sets")
    # build legend from all categories 0..max_size
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="white", label=l)
                      for l, c in zip(cat_labels, cat_colors)]
    fig.legend(handles=legend_handles, title="# called positive", loc="upper center",
               ncol=min(len(legend_handles), 6), bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle(f"Distribution of Positive Calls per CS by Size\n({title_suffix})",
                 fontsize=12, y=1.12)
    plt.tight_layout()
    savefig(FIG_BASE / "leafcutter_ambig" / "results" / save_name)
    plt.show()
    plt.close()

# figure 1: Youden's J thresholds
_make_ncalled_figure(lc_binned_thresholds, None,
                     "Youden's J threshold from leafcutter binned pos/neg",
                     "09d_ncalled_dist_by_cs_size")

# figure 2: fixed threshold = 0.2
_make_ncalled_figure(lc_binned_thresholds, 0.2,
                     "fixed threshold = 0.2 for all models",
                     "09e_ncalled_dist_by_cs_size_t02")


# In[ ]:


# same distribution but bucketed: 0, 1, 2, 3+
_cat_labels_b = ["0", "1", "2", "3+"]
_cat_colors_b = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]

def _make_ncalled_figure_bucketed(thresholds_dict, fixed_thresh, title_suffix, save_name):
    """stacked bar with 0/1/2/3+ buckets"""
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, mp):
        thresh = fixed_thresh if fixed_thresh is not None else thresholds_dict[model]
        delta_col = f"delta_{model}"

        dist_pcts = {cat: [] for cat in _cat_labels_b}
        for s in sizes:
            sub = cs_data[cs_data["n_variants_in_cs"] == s].dropna(subset=[delta_col])
            counts = {cat: 0 for cat in _cat_labels_b}
            n_total = 0
            for (tissue, cs_id), group in sub.groupby(["tissue", "cs_id"]):
                n_called = int((group[delta_col] >= thresh).sum())
                n_total += 1
                if n_called == 0:
                    counts["0"] += 1
                elif n_called == 1:
                    counts["1"] += 1
                elif n_called == 2:
                    counts["2"] += 1
                else:
                    counts["3+"] += 1
            for cat in _cat_labels_b:
                dist_pcts[cat].append(100 * counts[cat] / n_total if n_total > 0 else 0)

        bar_x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for cat, color in zip(_cat_labels_b, _cat_colors_b):
            vals = np.array(dist_pcts[cat])
            ax.bar(bar_x, vals, bottom=bottom, color=color, label=cat, width=0.7,
                   edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if v >= 5:
                    ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(bar_x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("credible set size")
        ax.set_title(f"{MODEL_NAMES[model]}\n(t={thresh:.4f})", fontsize=10)

    axes[0].set_ylabel("% of credible sets")
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="white", label=l)
                      for l, c in zip(_cat_labels_b, _cat_colors_b)]
    fig.legend(handles=legend_handles, title="# called positive", loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle(f"Distribution of Positive Calls per CS by Size\n({title_suffix})",
                 fontsize=12, y=1.12)
    plt.tight_layout()
    savefig(FIG_BASE / "leafcutter_ambig" / "results" / save_name)
    plt.show()
    plt.close()

# Youden's J thresholds, bucketed
_make_ncalled_figure_bucketed(lc_binned_thresholds, None,
                              "Youden's J threshold, bucketed 0/1/2/3+",
                              "09f_ncalled_bucketed_by_cs_size")

# fixed threshold = 0.2, bucketed
_make_ncalled_figure_bucketed(lc_binned_thresholds, 0.2,
                              "fixed threshold = 0.2, bucketed 0/1/2/3+",
                              "09g_ncalled_bucketed_by_cs_size_t02")


# In[ ]:


# same bucketed plot but vertical layout (one column)
def _make_ncalled_figure_vertical(thresholds_dict, fixed_thresh, title_suffix, save_name):
    """stacked bar with 0/1/2/3+ buckets, vertical subplot layout"""
    fig, axes = plt.subplots(n_models, 1, figsize=(8, 3.5 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, mp):
        thresh = fixed_thresh if fixed_thresh is not None else thresholds_dict[model]
        delta_col = f"delta_{model}"

        dist_pcts = {cat: [] for cat in _cat_labels_b}
        for s in sizes:
            sub = cs_data[cs_data["n_variants_in_cs"] == s].dropna(subset=[delta_col])
            counts = {cat: 0 for cat in _cat_labels_b}
            n_total = 0
            for (tissue, cs_id), group in sub.groupby(["tissue", "cs_id"]):
                n_called = int((group[delta_col] >= thresh).sum())
                n_total += 1
                if n_called == 0:
                    counts["0"] += 1
                elif n_called == 1:
                    counts["1"] += 1
                elif n_called == 2:
                    counts["2"] += 1
                else:
                    counts["3+"] += 1
            for cat in _cat_labels_b:
                dist_pcts[cat].append(100 * counts[cat] / n_total if n_total > 0 else 0)

        bar_x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for cat, color in zip(_cat_labels_b, _cat_colors_b):
            vals = np.array(dist_pcts[cat])
            ax.bar(bar_x, vals, bottom=bottom, color=color, label=cat, width=0.7,
                   edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if v >= 5:
                    ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals

        ax.set_ylabel("% of credible sets")
        ax.set_title(f"{MODEL_NAMES[model]} (t={thresh:.4f})", fontsize=10)

    axes[-1].set_xticks(np.arange(len(sizes)))
    axes[-1].set_xticklabels([str(s) for s in sizes])
    axes[-1].set_xlabel("credible set size")

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="white", label=l)
                      for l, c in zip(_cat_labels_b, _cat_colors_b)]
    fig.legend(handles=legend_handles, title="# called positive", loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"Distribution of Positive Calls per CS by Size\n({title_suffix})",
                 fontsize=12, y=1.06)
    plt.tight_layout()
    savefig(FIG_BASE / "leafcutter_ambig" / "results" / save_name)
    plt.show()
    plt.close()

# Youden's J thresholds, vertical
_make_ncalled_figure_vertical(lc_binned_thresholds, None,
                              "Youden's J threshold, bucketed 0/1/2/3+",
                              "09h_ncalled_bucketed_vertical")

# fixed threshold = 0.2, vertical
_make_ncalled_figure_vertical(lc_binned_thresholds, 0.2,
                              "fixed threshold = 0.2, bucketed 0/1/2/3+",
                              "09i_ncalled_bucketed_vertical_t02")


# In[ ]:


# SpHAEC-ref only — bucketed 0/1/2/3+
_sphaec_ref = 'sphaec_ref'
if _sphaec_ref in lc_binned_thresholds:
    for fixed_thresh, suffix, save_suffix in [
        (None, "Youden's J threshold", "09j_ncalled_sphaec_ref"),
        (0.2, "fixed threshold = 0.2", "09k_ncalled_sphaec_ref_t02"),
    ]:
        thresh = fixed_thresh if fixed_thresh is not None else lc_binned_thresholds[_sphaec_ref]
        delta_col = f"delta_{_sphaec_ref}"

        fig, ax = plt.subplots(figsize=(8, 5))
        dist_pcts = {cat: [] for cat in _cat_labels_b}
        for s in sizes:
            sub = cs_data[cs_data["n_variants_in_cs"] == s].dropna(subset=[delta_col])
            counts = {cat: 0 for cat in _cat_labels_b}
            n_total = 0
            for (tissue, cs_id), group in sub.groupby(["tissue", "cs_id"]):
                n_called = int((group[delta_col] >= thresh).sum())
                n_total += 1
                if n_called == 0:
                    counts["0"] += 1
                elif n_called == 1:
                    counts["1"] += 1
                elif n_called == 2:
                    counts["2"] += 1
                else:
                    counts["3+"] += 1
            for cat in _cat_labels_b:
                dist_pcts[cat].append(100 * counts[cat] / n_total if n_total > 0 else 0)

        bar_x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for cat, color in zip(_cat_labels_b, _cat_colors_b):
            vals = np.array(dist_pcts[cat])
            ax.bar(bar_x, vals, bottom=bottom, color=color, label=cat, width=0.7,
                   edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if v >= 5:
                    ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(bar_x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("credible set size")
        ax.set_ylabel("% of credible sets")

        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=c, edgecolor="white", label=l)
                          for l, c in zip(_cat_labels_b, _cat_colors_b)]
        fig.legend(handles=legend_handles, title="# called positive",
                   loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=9)
        fig.suptitle(f"SpHAEC-ref — Distribution of Positive Calls per CS by Size\n({suffix}, t={thresh:.4f})",
                     fontsize=12, y=1.10)

        plt.tight_layout()
        savefig(FIG_BASE / "leafcutter_ambig" / "results" / save_suffix)
        plt.show()
    plt.close()


# In[ ]:


# threshold sweep - precision/recall curves
def build_binary_labels(cs_data, model):
    """label top-pip variant in each CS as 1, rest as 0, returns labels + delta scores"""
    delta_col = f'delta_{model}'
    valid = cs_data.dropna(subset=['pip', delta_col])
    # mark top-pip variant per (tissue, cs_id)
    top_idx = valid.groupby(['tissue', 'cs_id'])['pip'].idxmax()
    labels = np.zeros(len(valid), dtype=int)
    labels[valid.index.get_indexer(top_idx)] = 1
    return labels, valid[delta_col].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for model in models_present:
    labels, scores = build_binary_labels(cs_data, model)
    thresholds = np.linspace(np.percentile(scores, 5), np.percentile(scores, 95), 50)
    precs, recs = [], []
    for t in thresholds:
        pred = scores >= t
        tp = (pred & (labels == 1)).sum()
        fp = (pred & (labels == 0)).sum()
        fn = (~pred & (labels == 1)).sum()
        precs.append(tp / max(tp + fp, 1))
        recs.append(tp / max(tp + fn, 1))
    axes[0].plot(thresholds, precs, color=COLORS[model], label=MODEL_NAMES[model], lw=2)
    axes[1].plot(thresholds, recs, color=COLORS[model], label=MODEL_NAMES[model], lw=2)
axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Precision')
axes[0].set_title('Precision vs Threshold'); axes[0].legend()
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Recall')
axes[1].set_title('Recall vs Threshold'); axes[1].legend()
plt.tight_layout()
savefig(FIG_BASE / 'leafcutter_ambig' / 'results' / '17_threshold_sweep')
plt.show()
plt.close()


# ## Supplemental: ROC with Youden's J Threshold
# 
# ROC curves for leafcutter sQTLs with Youden's J annotation showing optimal threshold per model.

# In[ ]:


# ROC with Youden's J annotation — leafcutter
if pairs_lc is not None and pos_scores_lc:
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in [m for m in MODELS if m in pos_scores_lc and m in neg_scores_lc]:
        tissue_thresholds = []
        tissue_tprs = {}
        for tissue in pairs_lc['tissue'].unique():
            tp = pairs_lc[pairs_lc['tissue'] == tissue]
            yt, ys, n = get_ys(tp, pos_scores_lc, neg_scores_lc, m)
            if n < 10: continue
            fpr, tpr, th = roc_curve(yt, ys)
            fpr_grid = np.linspace(0, 1, 200)
            tissue_tprs[tissue] = np.interp(fpr_grid, fpr, tpr)
            j_idx = np.argmax(tpr - fpr)
            tissue_thresholds.append(th[j_idx])
        if not tissue_tprs: continue
        med_tpr = np.median(list(tissue_tprs.values()), axis=0)
        med_auroc = np.trapz(med_tpr, fpr_grid)
        med_thresh = np.median(tissue_thresholds)
        ax.plot(fpr_grid, med_tpr, color=COLORS.get(m, 'gray'),
                label=f'{MODEL_NAMES[m]} (AUROC={med_auroc:.3f}, t={med_thresh:.3f})', lw=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title("GTEx leafcutter: ROC with Youden's J")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(FIG_BASE / 'leafcutter' / 'roc_youden')
    plt.show()
    plt.close()


# ## Tissue Sharing Analysis
# 
# Variants can appear as positives in one tissue or all 49. Constitutive sQTLs (present in all tissues) tend to be near constitutive splice sites and may be easier to classify. Tissue-specific sQTLs reflect alternative splicing and may be harder. We stratify per-tissue AUPRC by the number of tissues each positive appears in.

# In[ ]:


# tissue specificity analysis
# for each variant, compute tissue_fraction = n_tissues_positive / n_tissues_tested
# n_tissues_tested = tissues where variant appears in any credible set (any PIP)
# n_tissues_positive = tissues where PIP >= threshold (variant is in matched pairs)

for ds_name, ds_pairs, ds_pos, ds_neg, pip_path in [
    ('leafcutter', pairs_lc, pos_scores_lc, neg_scores_lc, DATA_BASE / 'leafcutter_pip50' / 'pip_values.csv'),
    ('txrevise', pairs, pos_scores, neg_scores, DATA_BASE / 'txrevise' / 'pip_values.csv'),
]:
    if ds_pairs is None or not ds_pos or 'tissue' not in ds_pairs.columns:
        continue
    if not pip_path.exists():
        print(f'{ds_name}: pip_values.csv not found, skipping tissue specificity')
        continue

    pip_all = pd.read_csv(pip_path)
    n_tissues_total = ds_pairs['tissue'].nunique()
    mp = [m for m in MODELS if m in ds_pos and m in ds_neg]

    # n_tissues_tested: how many tissues each variant appears in (any PIP)
    tissues_tested = pip_all.groupby('var_key')['tissue'].nunique()
    # n_tissues_positive: how many tissues where variant has PIP >= 0.9
    pip_high = pip_all[pip_all['pip'] >= 0.9]
    tissues_positive = pip_high.groupby('var_key')['tissue'].nunique()

    # tissue fraction for matched positives
    pos_keys = set(ds_pairs['pos_var_key'].unique())
    tf_data = pd.DataFrame({
        'var_key': list(pos_keys),
        'n_tested': [tissues_tested.get(k, 0) for k in pos_keys],
        'n_positive': [tissues_positive.get(k, 0) for k in pos_keys],
    })
    tf_data['tissue_fraction'] = tf_data['n_positive'] / tf_data['n_tested'].clip(lower=1)

    # log stats
    n_filtered = len(set(pip_all['var_key'].unique()) - pos_keys)
    print(f'\n{ds_name} tissue specificity:')
    print(f'  variants in credible sets (any PIP): {pip_all["var_key"].nunique():,}')
    print(f'  variants in matched pairs (positive): {len(pos_keys):,}')
    print(f'  variants filtered (not in pairs): {n_filtered:,}')
    print(f'  median tissues tested: {tf_data["n_tested"].median():.0f}')
    print(f'  median tissues positive: {tf_data["n_positive"].median():.0f}')
    print(f'  median tissue fraction: {tf_data["tissue_fraction"].median():.3f}')
    print(f'  tissue fraction distribution:')
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        n = (tf_data['tissue_fraction'] <= q).sum()
        print(f'    fraction <= {q:.1f}: {n:,} variants ({100*n/len(tf_data):.1f}%)')

    # map tissue fraction onto pairs
    tf_map = dict(zip(tf_data['var_key'], tf_data['tissue_fraction']))
    ds_pairs = ds_pairs.copy()
    ds_pairs['_tf'] = ds_pairs['pos_var_key'].map(tf_map)

    # --- plot 1: cumulative AUPRC by tissue fraction ---
    # include positives with tissue_fraction <= threshold
    tf_thresholds = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    tf_labels = ['<=0.05', '<=0.1', '<=0.2', '<=0.4', '<=0.6', '<=0.8', '<=1.0']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    x = np.arange(len(tf_thresholds))

    # variant counts
    bw = 0.25
    tf_pos_counts, tf_neg_counts = [], []
    for t in tf_thresholds:
        sub = ds_pairs[ds_pairs['_tf'] <= t]
        tf_pos_counts.append(sub['pos_var_key'].nunique())
        tf_neg_counts.append(sub['neg_var_key'].nunique())
    ax2.bar(x - bw/2, tf_pos_counts, bw, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.bar(x + bw/2, tf_neg_counts, bw, color='#606060', alpha=0.4, label='neg variants')
    ax2.set_ylabel('unique variants', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    for m in mp:
        auprcs = []
        for t in tf_thresholds:
            sub = ds_pairs[ds_pairs['_tf'] <= t]
            if len(sub) < 20:
                auprcs.append(np.nan); continue
            tissue_vals = []
            for tissue in sub['tissue'].unique():
                tp = sub[sub['tissue'] == tissue]
                yt, ys, n = get_ys(tp, ds_pos, ds_neg, m)
                if n >= 10:
                    tissue_vals.append(compute_auprc(yt, ys))
            auprcs.append(np.median(tissue_vals) if tissue_vals else np.nan)
        ax.plot(x, auprcs, marker='o', color=COLORS.get(m, 'gray'),
                label=MODEL_NAMES.get(m, m), lw=2, markersize=6, zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(tf_labels)
    ax.set_xlabel('Max tissue fraction (n_positive / n_tested)')
    ax.set_ylabel('AUPRC (tissue median)')
    ax.set_title(f'{ds_name}: AUPRC by tissue specificity (cumulative)')
    all_y = [v for line in ax.get_lines() for v in line.get_ydata() if np.isfinite(v)]
    if all_y:
        ax.set_ylim(np.floor(min(all_y) * 10) / 10, np.ceil(max(all_y) * 10) / 10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    savefig(FIG_BASE / ds_name / 'results' / 'set1' / '03a_tissue_fraction_cumulative')
    plt.show()
    plt.close()

    # --- plot 2: per-tissue AUPRC vs fraction of shared variants ---
    # for each tissue: what fraction of its positives are shared (in all tested tissues)?
    fig, ax = plt.subplots(figsize=(7, 6))

    for m in mp:
        tissue_auprcs = []
        tissue_shared_fracs = []
        for tissue in ds_pairs['tissue'].unique():
            tp = ds_pairs[ds_pairs['tissue'] == tissue]
            yt, ys, n = get_ys(tp, ds_pos, ds_neg, m)
            if n < 10:
                continue
            auprc_val = compute_auprc(yt, ys)
            # fraction of this tissue's positives that have tissue_fraction == 1.0
            n_shared = (tp['_tf'] == 1.0).sum()
            frac_shared = n_shared / len(tp)
            tissue_auprcs.append(auprc_val)
            tissue_shared_fracs.append(frac_shared)

        if tissue_auprcs:
            ax.scatter(tissue_shared_fracs, tissue_auprcs,
                      color=COLORS.get(m, 'gray'), alpha=0.5, s=20,
                      label=MODEL_NAMES.get(m, m))
            # spearman correlation
            rho, pval = spearmanr(tissue_shared_fracs, tissue_auprcs)
            print(f'  {ds_name} {MODEL_NAMES.get(m, m)}: tissue AUPRC vs frac_shared: rho={rho:.3f}, p={pval:.2e}')

    ax.set_xlabel('Fraction of positives shared across all tissues')
    ax.set_ylabel('AUPRC')
    ax.set_title(f'{ds_name}: tissue AUPRC vs variant composition')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(FIG_BASE / ds_name / 'results' / 'set1' / '03b_tissue_composition_scatter')
    plt.show()
    plt.close()

    # --- plot 3: tissue fraction histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tf_data['tissue_fraction'], bins=50, color='#66c2a5', edgecolor='white')
    ax.set_xlabel('Tissue fraction (n_positive / n_tested)')
    ax.set_ylabel('Unique variants')
    ax.set_title(f'{ds_name}: distribution of tissue specificity')
    ax.axvline(1.0, color='red', ls='--', alpha=0.5, label='constitutive (fraction=1.0)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(FIG_BASE / ds_name / 'results' / 'set1' / '03c_tissue_fraction_hist')
    plt.show()
    plt.close()


# ## Independent Filtering Distance AUPRC
# 
# Positives and negatives filtered independently by their own splice distance at each threshold. This matches the evaluation used in Linder et al. 2025 (Borzoi) Figure 7d-e. In the paired method, only the positive's splice distance is used to filter pairs. Here, both must independently satisfy the threshold.

# In[ ]:


# independent filtering distance AUPRC helper
def plot_distance_auprc_independent(pairs, pos_scores, neg_scores, distances,
                                    title, save_path=None):
    """filter pos and neg independently by their own splice distance"""
    thresholds = [50, 100, 200, 500, 2000, 10000]
    labels = ['<=50', '<=100', '<=200', '<=500', '<=2k', '<=10k']

    pos_dist = distances[distances['type'] == 'pos'].set_index('var_key')['splice_dist']
    neg_dist = distances[distances['type'] == 'neg'].set_index('var_key')['splice_dist']

    mp = [m for m in MODELS if m in pos_scores]
    single_tissue = 'tissue' not in pairs.columns
    x = np.arange(len(thresholds))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax2 = ax.twinx()

    bw = 0.25
    pos_snv_counts, neg_snv_counts = [], []
    for d in thresholds:
        pos_snv_counts.append(sum(1 for k in pairs['pos_var_key'].unique()
                                  if k in pos_dist.index and pos_dist[k] <= d))
        neg_snv_counts.append(sum(1 for k in pairs['neg_var_key'].unique()
                                  if k in neg_dist.index and neg_dist[k] <= d))
    ax2.bar(x - bw/2, pos_snv_counts, bw, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.bar(x + bw/2, neg_snv_counts, bw, color='#606060', alpha=0.4, label='neg variants')
    ax2.set_ylabel('unique variants', color='gray', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='gray')

    for model in mp:
        auprcs = []
        for d in thresholds:
            if single_tissue:
                pos_pass = {k for k in pos_scores[model]
                            if k in pos_dist.index and pos_dist[k] <= d}
                neg_pass = {k for k in neg_scores[model]
                            if k in neg_dist.index and neg_dist[k] <= d}
                if len(pos_pass) < 10 or len(neg_pass) < 10:
                    auprcs.append(np.nan); continue
                y_true = np.concatenate([np.ones(len(pos_pass)), np.zeros(len(neg_pass))])
                y_score = np.concatenate([[pos_scores[model][k] for k in pos_pass],
                                          [neg_scores[model][k] for k in neg_pass]])
                auprcs.append(compute_auprc(y_true, y_score))
            else:
                tissue_auprcs = []
                for tissue in pairs['tissue'].unique():
                    tp = pairs[pairs['tissue'] == tissue]
                    t_pos_keys = set(tp['pos_var_key'])
                    t_neg_keys = set(tp['neg_var_key'])
                    pos_pass = [k for k in t_pos_keys
                                if k in pos_scores.get(model, {})
                                and k in pos_dist.index and pos_dist[k] <= d]
                    neg_pass = [k for k in t_neg_keys
                                if k in neg_scores.get(model, {})
                                and k in neg_dist.index and neg_dist[k] <= d]
                    if len(pos_pass) < 10 or len(neg_pass) < 10:
                        continue
                    yt = np.concatenate([np.ones(len(pos_pass)), np.zeros(len(neg_pass))])
                    ys = np.concatenate([[pos_scores[model][k] for k in pos_pass],
                                          [neg_scores[model][k] for k in neg_pass]])
                    tissue_auprcs.append(compute_auprc(yt, ys))
                auprcs.append(np.median(tissue_auprcs) if tissue_auprcs else np.nan)

        ax.plot(x, auprcs, marker='o', color=COLORS[model], label=MODEL_NAMES[model],
                lw=2, markersize=6, zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Splice Distance (bp)')
    ax.set_ylabel('AUPRC (tissue median)')
    all_y = [v for line in ax.get_lines() for v in line.get_ydata() if np.isfinite(v)]
    if all_y:
        ax.set_ylim(np.floor(min(all_y) * 10) / 10, np.ceil(max(all_y) * 10) / 10)
    ax.grid(alpha=0.3, zorder=0)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels1) + len(labels2), fontsize=9, frameon=False)
    fig.suptitle(title, y=1.08)
    if save_path: savefig(save_path)
    plt.show()
    plt.close()

# leafcutter
if distances_lc is not None:
    plot_distance_auprc_independent(pairs_lc, pos_scores_lc, neg_scores_lc, distances_lc,
        'Leafcutter - AUPRC by Distance (independent filtering)',
        FIG_BASE / 'leafcutter' / 'results' / 'set1' / '02c_distance_independent')

# txrevise
plot_distance_auprc_independent(pairs, pos_scores, neg_scores, distances,
    'Txrevise - AUPRC by Distance (independent filtering)',
    FIG_BASE / 'txrevise' / 'results' / 'set1' / '02c_distance_independent')


#
# # Hungarian Matching Analysis
#
# Rerun of the three sQTL benchmarks with Hungarian (global one-to-one optimal
# assignment) matching on expression bin + log-distance, dropping the gene
# stratum. This directly addresses the distance-imbalance finding (SMD = -0.38
# on distance under tiered matching): Hungarian produces near-perfect distance
# balance (median dist_diff ~ 0 bp, >99% of pairs within 10 bp) while keeping
# the same number of pairs.
#
# PIP thresholds match the tiered runs:
# - txrevise_hungarian: pos_pip >= 0.9, comprehensive GTF (matches txrevise_comp baseline)
# - leafcutter_hungarian: pos_pip >= 0.5 matching, filter to >= 0.9 in results
# - haec_hungarian: pos_pip >= 0.9
#
# SPLAIRE scoring is missing for leafcutter_hungarian spt neg — any missing
# model scores are skipped automatically.

print('\n' + '=' * 70)
print('HUNGARIAN MATCHING ANALYSIS')
print('=' * 70)


def _hungarian_block(ds_name, out_subdir, ref_pairs=None, ref_pos_s=None,
                    ref_neg_s=None, filter_pip=None, multi_tissue=True):
    """run QC + matching + results pipeline for one hungarian dataset.
    ref_* are the tiered-matching equivalents for side-by-side comparison"""
    d = DATA_BASE / ds_name
    if not (d / 'pairs.csv').exists():
        print(f'\n{ds_name}: pairs.csv not found, skipping')
        return None, None, None, None, None

    print(f'\n--- {ds_name} ---')
    pairs_h = pd.read_csv(d / 'pairs.csv')
    if filter_pip is not None and 'pos_pip' in pairs_h.columns:
        n_before = len(pairs_h)
        pairs_h = pairs_h.query('pos_pip >= @filter_pip').copy()
        print(f'filtered {n_before:,} -> {len(pairs_h):,} pairs (pos_pip >= {filter_pip})')

    tiers_h = pd.read_csv(d / 'tiers.csv') if (d / 'tiers.csv').exists() else None
    distances_h = pd.read_csv(d / 'distances.csv') if (d / 'distances.csv').exists() else None

    n_pos = pairs_h['pos_var_key'].nunique()
    n_neg = pairs_h['neg_var_key'].nunique()
    n_pairs = len(pairs_h)
    print(f'{ds_name}: {n_pos:,} pos, {n_neg:,} neg unique, {n_pairs:,} pairs')

    # compare to tiered baseline if provided
    if ref_pairs is not None:
        ref_np = ref_pairs['pos_var_key'].nunique()
        ref_nn = ref_pairs['neg_var_key'].nunique()
        ref_npairs = len(ref_pairs)
        print(f'vs tiered baseline: pos {ref_np:,} -> {n_pos:,} ({n_pos - ref_np:+,}), '
              f'neg {ref_nn:,} -> {n_neg:,} ({n_neg - ref_nn:+,}), '
              f'pairs {ref_npairs:,} -> {n_pairs:,} ({n_pairs - ref_npairs:+,})')

    # dist_diff distribution
    if distances_h is not None:
        pos_d = distances_h[distances_h['type'] == 'pos'].set_index('var_key')['splice_dist']
        neg_d = distances_h[distances_h['type'] == 'neg'].set_index('var_key')['splice_dist']
        dd = (pairs_h['pos_var_key'].map(pos_d) - pairs_h['neg_var_key'].map(neg_d)).abs().dropna()
        print(f'  dist_diff: median={dd.median():.1f}, mean={dd.mean():.1f}, '
              f'<=10bp={100*(dd<=10).mean():.1f}%, <=500bp={100*(dd<=500).mean():.1f}%')

    # QC — tier distribution (all tier 99 for hungarian)
    if tiers_h is not None:
        plot_tier_dist(tiers_h, f'{ds_name} - Tier Distribution (hungarian)',
            FIG_BASE / out_subdir / 'matching' / '01_tier_distribution')

    # splice distance histogram
    if distances_h is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        pos_d_vals = distances_h[distances_h['type'] == 'pos']['splice_dist']
        neg_d_vals = distances_h[distances_h['type'] == 'neg']['splice_dist']
        bins = np.logspace(0, 4, 50)
        ax.hist(pos_d_vals, bins=bins, alpha=0.7, label=f'pos (n={len(pos_d_vals):,})',
                color=DATA_COLORS['pos'])
        ax.hist(neg_d_vals, bins=bins, alpha=0.7, label=f'neg (n={len(neg_d_vals):,})',
                color=DATA_COLORS['neg'])
        ax.set_xscale('log'); ax.set_xlabel('Splice distance (bp)')
        ax.set_ylabel('Count'); ax.set_title(f'{ds_name} - Splice Distance')
        ax.legend()
        plt.tight_layout()
        savefig(FIG_BASE / out_subdir / 'matching' / '05_distance_dist')
        plt.show()
        plt.close()

    # load scores (graceful if missing models)
    score_dir_h = d / 'scores'
    pos_s_h = load_scores(score_dir_h, 'pos', primary_only=args.nohbar)
    neg_s_h = load_scores(score_dir_h, 'neg', primary_only=args.nohbar)
    common = set(pos_s_h) & set(neg_s_h)
    pos_s_h = {m: d for m, d in pos_s_h.items() if m in common}
    neg_s_h = {m: d for m, d in neg_s_h.items() if m in common}
    loaded = sorted(common)
    missing = [m for m in MODELS if m not in loaded]
    if missing:
        print(f'  models loaded: {len(loaded)} (missing: {missing})')

    if not pos_s_h or not neg_s_h:
        return pairs_h, pos_s_h, neg_s_h, distances_h, None

    # delta distribution
    plot_delta_dist(pairs_h, pos_s_h, neg_s_h,
        f'{ds_name} - Delta Distributions',
        FIG_BASE / out_subdir / 'results' / '00a_delta_distributions')

    # swarm / pooled AUPRC
    if multi_tissue and 'tissue' in pairs_h.columns:
        met_h = compute_tissue_metrics(pairs_h, pos_s_h, neg_s_h)
        plot_swarm(met_h, f'{ds_name} - Per-tissue AUPRC',
            FIG_BASE / out_subdir / 'results' / '01_swarm')
        if not args.nohbar:
            plot_median_auprc_hbar(met_h, f'{ds_name} - All Output Heads',
                FIG_BASE / out_subdir / 'results' / '01b_hbar')
    else:
        pooled = compute_pooled_metrics(pairs_h, pos_s_h, neg_s_h)
        fig, ax = plt.subplots(figsize=(8, 5))
        mp = [m for m in MODELS if m in pooled]
        for i, m in enumerate(mp):
            v = pooled[m]['auprc']
            ax.bar(i, v, color=COLORS[m], alpha=0.8)
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
        ax.set_xticks(range(len(mp)))
        ax.set_xticklabels([MODEL_NAMES[m] for m in mp], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('AUPRC'); ax.set_title(f'{ds_name} - AUPRC')
        ax.axhline(0.5, color='gray', ls='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        savefig(FIG_BASE / out_subdir / 'results' / '01_overall_auprc')
        plt.show()
        plt.close()
        met_h = None

    # PR / ROC
    plot_pr(pairs_h, pos_s_h, neg_s_h,
        f'{ds_name} - PR Curves',
        FIG_BASE / out_subdir / 'results' / '01b_pr_curves')
    plot_roc(pairs_h, pos_s_h, neg_s_h,
        f'{ds_name} - ROC Curves',
        FIG_BASE / out_subdir / 'results' / '01d_roc_curves')

    # distance
    if distances_h is not None:
        plot_distance_auprc(pairs_h, pos_s_h, neg_s_h, distances_h,
            f'{ds_name} - AUPRC by Splice Distance',
            FIG_BASE / out_subdir / 'results' / '02_distance')
        plot_distance_auprc_unique(pairs_h, pos_s_h, neg_s_h, distances_h,
            f'{ds_name} - AUPRC by Distance (unique positives)',
            FIG_BASE / out_subdir / 'results' / '02b_distance_unique')

    return pairs_h, pos_s_h, neg_s_h, distances_h, met_h


# --- run all three hungarian datasets ---
pairs_txh, pos_s_txh, neg_s_txh, dist_txh, met_txh = _hungarian_block(
    'txrevise_hungarian', 'txrevise_hungarian',
    ref_pairs=pairs, ref_pos_s=pos_scores, ref_neg_s=neg_scores,
    multi_tissue=True)

pairs_lch, pos_s_lch, neg_s_lch, dist_lch, met_lch = _hungarian_block(
    'leafcutter_hungarian', 'leafcutter_hungarian',
    ref_pairs=pairs_lc, ref_pos_s=pos_scores_lc, ref_neg_s=neg_scores_lc,
    filter_pip=0.9, multi_tissue=True)

pairs_hch, pos_s_hch, neg_s_hch, dist_hch, met_hch = _hungarian_block(
    'haec_hungarian', 'haec_hungarian',
    ref_pairs=pairs_haec, ref_pos_s=pos_scores_haec, ref_neg_s=neg_scores_haec,
    multi_tissue=False)


# --- hungarian vs tiered: per-model median AUPRC comparison ---
print('\n' + '=' * 70)
print('HUNGARIAN vs TIERED — per-model AUPRC comparison')
print('=' * 70)

_cmp_rows = []
for ds_label, tier_pairs, tier_pos, tier_neg, hun_pairs, hun_pos, hun_neg, multi in [
    ('txrevise',   pairs,        pos_scores,    neg_scores,
     pairs_txh,    pos_s_txh,    neg_s_txh,     True),
    ('leafcutter', pairs_lc,     pos_scores_lc, neg_scores_lc,
     pairs_lch,    pos_s_lch,    neg_s_lch,     True),
    ('haec',       pairs_haec,   pos_scores_haec, neg_scores_haec,
     pairs_hch,    pos_s_hch,    neg_s_hch,     False),
]:
    if hun_pairs is None or not hun_pos:
        continue
    for m in MODELS:
        # tiered
        if tier_pos and m in tier_pos and m in tier_neg:
            if multi and 'tissue' in tier_pairs.columns:
                aucs = []
                for t in tier_pairs['tissue'].unique():
                    tp = tier_pairs[tier_pairs['tissue'] == t]
                    yt, ys, n = get_ys(tp, tier_pos, tier_neg, m)
                    if n >= 10: aucs.append(compute_auprc(yt, ys))
                tier_auprc = np.median(aucs) if aucs else np.nan
            else:
                yt, ys, n = get_ys(tier_pairs, tier_pos, tier_neg, m)
                tier_auprc = compute_auprc(yt, ys) if n >= 10 else np.nan
        else:
            tier_auprc = np.nan
        # hungarian
        if m in hun_pos and m in hun_neg:
            if multi and 'tissue' in hun_pairs.columns:
                aucs = []
                for t in hun_pairs['tissue'].unique():
                    tp = hun_pairs[hun_pairs['tissue'] == t]
                    yt, ys, n = get_ys(tp, hun_pos, hun_neg, m)
                    if n >= 10: aucs.append(compute_auprc(yt, ys))
                hun_auprc = np.median(aucs) if aucs else np.nan
            else:
                yt, ys, n = get_ys(hun_pairs, hun_pos, hun_neg, m)
                hun_auprc = compute_auprc(yt, ys) if n >= 10 else np.nan
        else:
            hun_auprc = np.nan
        _cmp_rows.append({
            'dataset': ds_label, 'model': m, 'tiered_auprc': tier_auprc,
            'hungarian_auprc': hun_auprc, 'delta': hun_auprc - tier_auprc,
        })

_cmp_df = pd.DataFrame(_cmp_rows)
_cmp_df.to_csv(FIG_BASE / 'hungarian_vs_tiered_auprc.csv', index=False)

for ds in _cmp_df['dataset'].unique():
    sub = _cmp_df[_cmp_df['dataset'] == ds]
    print(f'\n{ds}:')
    print(f'  {"model":<25} {"tiered":>8}  {"hungarian":>10}  {"delta":>8}')
    print(f'  {"-"*25} {"-"*8}  {"-"*10}  {"-"*8}')
    for _, r in sub.iterrows():
        t = f'{r["tiered_auprc"]:.4f}' if pd.notna(r['tiered_auprc']) else '  —  '
        h = f'{r["hungarian_auprc"]:.4f}' if pd.notna(r['hungarian_auprc']) else '  —  '
        d = f'{r["delta"]:+.4f}' if pd.notna(r['delta']) else '  —  '
        print(f'  {MODEL_NAMES.get(r["model"], r["model"]):<25} {t:>8}  {h:>10}  {d:>8}')

print(f'\nsaved {FIG_BASE / "hungarian_vs_tiered_auprc.csv"}')


# ==========================================================================
# Delta Position Analysis
# ==========================================================================
# for each variant the delta score is max |alt - ref| within ±2kb
# this section analyzes WHERE that maximum occurs — at the variant itself
# (offset = 0) or at some nearby splice site. offset = argmax_pos - center

print('\n' + '=' * 70)
print('DELTA POSITION ANALYSIS')
print('=' * 70)


def load_delta_positions(score_dir, vcf_base):
    """load argmax position offset for each variant/head
    returns dict {model: {var_key: offset_bp}}"""
    cache = score_dir / f'{vcf_base}_positions.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
        out = {}
        for col in df.columns:
            if col == 'var_key': continue
            out[col] = dict(zip(df['var_key'], df[col]))
        print(f'  cached: {cache.name} ({len(out)} heads, {len(df):,} variants)')
        return out

    out = {}

    def _ci_sl(seq_len):
        c = seq_len // 2
        sl = slice(max(0, c - WINDOW), min(seq_len, c + WINDOW + 1))
        return c - sl.start, sl

    def _off(d2d, ci):
        """d2d shape (n, pos) → (n,) int16 offsets from center"""
        return (np.argmax(d2d, axis=1) - ci).astype(np.int16)

    # spliceai: (n, seq_len, 3)
    h = score_dir / f'{vcf_base}.sa.h5'
    if h.exists():
        with h5py.File(h, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            ref, alt = f['ref'][:], f['alt'][:]
            ci, sl = _ci_sl(ref.shape[1])
            d = np.max(np.abs(alt[:, sl, :] - ref[:, sl, :]), axis=2)
            out['spliceai'] = dict(zip(keys, _off(d, ci)))
            del ref, alt, d

    # pangolin helper
    def _pang(h5_path, prefix):
        if not h5_path.exists():
            return
        with h5py.File(h5_path, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            tasks = [k[4:] for k in f.keys() if k.startswith('ref_')]
            ps = [t for t in tasks if t.endswith('_p_splice')]
            us = [t for t in tasks if t.endswith('_usage')]
            ci, sl = _ci_sl(f[f'ref_{ps[0]}'].shape[1])
            ps_stack, us_stack = [], []
            for t in ps:
                d = np.abs(f[f'alt_{t}'][:, sl] - f[f'ref_{t}'][:, sl])
                tissue = t.replace('_p_splice', '')
                out[f'{prefix}_{tissue}_cls'] = dict(zip(keys, _off(d, ci)))
                ps_stack.append(d)
            if ps_stack:
                agg = np.max(np.stack(ps_stack), axis=0)
                out[prefix] = dict(zip(keys, _off(agg, ci)))
            for t in us:
                d = np.abs(f[f'alt_{t}'][:, sl] - f[f'ref_{t}'][:, sl])
                tissue = t.replace('_usage', '')
                out[f'{prefix}_{tissue}_usage'] = dict(zip(keys, _off(d, ci)))
                us_stack.append(d)
            if us_stack:
                agg = np.max(np.stack(us_stack), axis=0)
                out[f'{prefix}_usage'] = dict(zip(keys, _off(agg, ci)))
            if ps_stack and us_stack:
                agg = np.max(np.stack(ps_stack + us_stack), axis=0)
                out[f'{prefix}_all'] = dict(zip(keys, _off(agg, ci)))

    _pang(score_dir / f'{vcf_base}.pang.h5', 'pangolin')
    _pang(score_dir / f'{vcf_base}.pang_v2.h5', 'pangolin_v2')

    # splaire
    for label, sfx_list in [('sphaec_var', ['splaire.var', 'sphaec.var']),
                             ('sphaec_ref', ['splaire.ref', 'sphaec.ref'])]:
        found = None
        for sfx in sfx_list:
            pp = score_dir / f'{vcf_base}.{sfx}.h5'
            if pp.exists():
                found = pp
                break
        if not found:
            continue
        with h5py.File(found, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            cr, ca = f['cls_ref'][:], f['cls_alt'][:]
            ci, sl = _ci_sl(cr.shape[1])
            d = np.max(np.abs(ca[:, sl, :] - cr[:, sl, :]), axis=2)
            out[label] = dict(zip(keys, _off(d, ci)))
            if 'reg_ref' in f:
                rr, ra = f['reg_ref'][:], f['reg_alt'][:]
                dr = np.abs(ra[:, sl] - rr[:, sl])
                out[f'{label}_reg'] = dict(zip(keys, _off(dr, ci)))
                out[f'{label}_all'] = dict(zip(keys, _off(np.maximum(d, dr), ci)))
            del cr, ca, d

    # splicetransformer: (n, 18, seq_len)
    h = score_dir / f'{vcf_base}.spt.h5'
    _SPT_T = ['adipose', 'blood', 'blood_vessel', 'brain', 'colon',
              'heart', 'kidney', 'liver', 'lung', 'muscle',
              'nerve', 'small_intestine', 'skin', 'spleen', 'stomach']
    if h.exists():
        with h5py.File(h, 'r') as f:
            keys = [k.decode() for k in f['var_key'][:]]
            ref, alt = f['ref'][:], f['alt'][:]
            ci, sl = _ci_sl(ref.shape[2])
            d_cls = np.max(np.abs(alt[:, :3, sl] - ref[:, :3, sl]), axis=1)
            out['splicetransformer'] = dict(zip(keys, _off(d_cls, ci)))
            d_us = np.max(np.abs(alt[:, 3:, sl] - ref[:, 3:, sl]), axis=1)
            out['splicetransformer_usage'] = dict(zip(keys, _off(d_us, ci)))
            d_all = np.max(np.abs(alt[:, :, sl] - ref[:, :, sl]), axis=1)
            out['splicetransformer_all'] = dict(zip(keys, _off(d_all, ci)))
            for ch_i, tissue in enumerate(_SPT_T):
                ch = ch_i + 3
                d = np.abs(alt[:, ch, sl] - ref[:, ch, sl])
                out[f'spt_{tissue}'] = dict(zip(keys, _off(d, ci)))
            del ref, alt

    # write cache
    if out:
        vk = list(next(iter(out.values())).keys())
        df = pd.DataFrame({'var_key': vk})
        for m, d in out.items():
            df[m] = df['var_key'].map(d).astype(np.int16)
        df.to_parquet(cache)
        print(f'  saved: {cache.name} ({len(out)} heads, {len(df):,} variants)')

    return out


# --- run delta position analysis per dataset ---
_dp_datasets = [
    ('txrevise', pairs, distances, DATA_BASE / 'txrevise' / 'scores',
     pos_scores, neg_scores, 'txrevise'),
    ('leafcutter', pairs_lc, distances_lc, DATA_BASE / 'leafcutter_pip50' / 'scores',
     pos_scores_lc, neg_scores_lc, 'leafcutter'),
    ('haec', pairs_haec, distances_haec, DATA_BASE / 'haec' / 'scores',
     pos_scores_haec, neg_scores_haec, 'haec'),
]

for ds_label, ds_pairs, ds_distances, score_dir_dp, ds_ps, ds_ns, fig_dir in _dp_datasets:
    if ds_pairs is None:
        continue

    print(f'\n--- {ds_label} ---')
    pos_positions = load_delta_positions(score_dir_dp, 'pos')
    neg_positions = load_delta_positions(score_dir_dp, 'neg')
    all_heads = sorted(set(pos_positions.keys()) & set(neg_positions.keys()))
    mp = [m for m in MODELS if m in pos_positions and m in neg_positions]
    print(f'  {len(all_heads)} heads, {len(mp)} primary models')

    pos_keys = ds_pairs['pos_var_key'].unique()
    neg_keys = ds_pairs['neg_var_key'].unique()
    single_tissue = 'tissue' not in ds_pairs.columns

    # ---- plot 1: offset distribution (primary models, pos vs neg) ----
    fig, axes = plt.subplots(1, len(mp), figsize=(3.5 * len(mp), 4))
    if len(mp) == 1: axes = [axes]
    bins_h = np.arange(-WINDOW, WINDOW + 1, 50)
    for ax, m in zip(axes, mp):
        p_off = np.array([pos_positions[m].get(k, np.nan) for k in pos_keys])
        n_off = np.array([neg_positions[m].get(k, np.nan) for k in neg_keys])
        p_off = p_off[~np.isnan(p_off)]
        n_off = n_off[~np.isnan(n_off)]
        ax.hist(p_off, bins=bins_h, alpha=0.6, color='#e74c3c', label=f'pos (n={len(p_off):,})')
        ax.hist(n_off, bins=bins_h, alpha=0.6, color='#27ae60', label=f'neg (n={len(n_off):,})')
        ax.set_xlabel('offset (bp)')
        ax.set_title(MODEL_NAMES[m], fontsize=9)
        ax.legend(fontsize=7)
        ax.set_yscale('log')
    fig.suptitle(f'{ds_label} — Delta Position Offset', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / fig_dir / 'delta_position' / '01_offset_distribution')
    plt.show()
    plt.close()

    # ---- plot 2: all-heads median |offset| hbar ----
    head_stats = []
    for h in all_heads:
        p_off = np.array([pos_positions[h].get(k, np.nan) for k in pos_keys])
        n_off = np.array([neg_positions[h].get(k, np.nan) for k in neg_keys])
        p_off = np.abs(p_off[~np.isnan(p_off)])
        n_off = np.abs(n_off[~np.isnan(n_off)])
        head_stats.append({
            'head': h,
            'pos_median': np.median(p_off) if len(p_off) else np.nan,
            'neg_median': np.median(n_off) if len(n_off) else np.nan,
            'pos_at_var': (p_off <= 10).mean() * 100 if len(p_off) else np.nan,
            'neg_at_var': (n_off <= 10).mean() * 100 if len(n_off) else np.nan,
        })
    hs = pd.DataFrame(head_stats).sort_values('pos_median')

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(hs))))
    y = np.arange(len(hs))
    colors = [COLORS.get(h, 'gray') for h in hs['head']]
    ax.barh(y, hs['pos_median'], color=colors, alpha=0.8, height=0.7)
    for i, row in enumerate(hs.itertuples()):
        if np.isfinite(row.pos_median):
            ax.text(row.pos_median + 5, i, f'{row.pos_median:.0f}', va='center', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_NAMES.get(h, h) for h in hs['head']], fontsize=7)
    ax.set_xlabel('Median |offset| from variant (bp)')
    ax.set_title(f'{ds_label} — Median |offset| by Head (positives)')
    ax.grid(alpha=0.2, axis='x')
    plt.tight_layout()
    savefig(FIG_BASE / fig_dir / 'delta_position' / '02_median_offset_all_heads')
    plt.show()
    plt.close()

    # ---- plot 3: fraction at/near variant (|offset| <= 10bp) all heads ----
    hs2 = hs.sort_values('pos_at_var', ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(hs2))))
    y = np.arange(len(hs2))
    bw = 0.35
    ax.barh(y - bw/2, hs2['pos_at_var'], bw, color='#e74c3c', alpha=0.8, label='pos')
    ax.barh(y + bw/2, hs2['neg_at_var'], bw, color='#27ae60', alpha=0.8, label='neg')
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_NAMES.get(h, h) for h in hs2['head']], fontsize=7)
    ax.set_xlabel('% with |offset| ≤ 10 bp')
    ax.set_title(f'{ds_label} — Fraction at/near variant')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis='x')
    plt.tight_layout()
    savefig(FIG_BASE / fig_dir / 'delta_position' / '03_frac_at_variant')
    plt.show()
    plt.close()

    # ---- plot 4: |offset| vs splice distance scatter (pos, primary models) ----
    if ds_distances is not None:
        pos_dist = ds_distances[ds_distances['type'] == 'pos'].set_index('var_key')['splice_dist']

        fig, axes = plt.subplots(1, len(mp), figsize=(3.5 * len(mp), 4))
        if len(mp) == 1: axes = [axes]
        for ax, m in zip(axes, mp):
            off_arr = np.array([pos_positions[m].get(k, np.nan) for k in pos_keys])
            sd_arr = np.array([pos_dist.get(k, np.nan) for k in pos_keys])
            valid = ~(np.isnan(off_arr) | np.isnan(sd_arr))
            ax.scatter(sd_arr[valid], np.abs(off_arr[valid]), s=5, alpha=0.15,
                      color=COLORS.get(m, 'gray'), rasterized=True)
            lim = max(sd_arr[valid].max(), np.abs(off_arr[valid]).max()) * 1.05
            ax.plot([1, lim], [1, lim], 'k--', alpha=0.3, lw=1)
            ax.set_xlabel('splice distance (bp)')
            ax.set_ylabel('|offset| (bp)')
            ax.set_title(MODEL_NAMES[m], fontsize=9)
            rho = spearmanr(sd_arr[valid], np.abs(off_arr[valid]))[0]
            ax.text(0.03, 0.97, f'ρ={rho:.3f}', transform=ax.transAxes, fontsize=8, va='top')
            ax.set_xscale('log'); ax.set_yscale('log')
        fig.suptitle(f'{ds_label} — |Offset| vs Splice Distance (positives)', y=1.02)
        plt.tight_layout()
        savefig(FIG_BASE / fig_dir / 'delta_position' / '04_offset_vs_splice_dist')
        plt.show()
        plt.close()

    # ---- plot 5: AUPRC by |offset| bin (primary models) ----
    offset_bins = [0, 10, 50, 200, 500, 2001]
    offset_labels = ['0-10', '10-50', '50-200', '200-500', '500-2k']
    x = np.arange(len(offset_labels))

    # count bars using first model
    m0 = mp[0]
    p_off_ref = {k: np.abs(pos_positions[m0].get(k, np.nan)) for k in ds_pairs['pos_var_key'].unique()}
    bin_counts = []
    for i in range(len(offset_labels)):
        lo, hi = offset_bins[i], offset_bins[i + 1]
        bin_counts.append(sum(1 for v in p_off_ref.values() if np.isfinite(v) and lo <= v < hi))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()
    ax2.bar(x, bin_counts, 0.3, color='#b0b0b0', alpha=0.4, label='pos variants')
    ax2.set_ylabel('unique variants', color='gray', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='gray')

    for m in mp:
        auprcs = []
        for i in range(len(offset_labels)):
            lo, hi = offset_bins[i], offset_bins[i + 1]
            p_off_m = np.array([np.abs(pos_positions[m].get(k, np.nan)) for k in ds_pairs['pos_var_key']])
            mask = (p_off_m >= lo) & (p_off_m < hi) & ~np.isnan(p_off_m)
            sub = ds_pairs[mask]
            if len(sub) < 20:
                auprcs.append(np.nan); continue
            if single_tissue:
                yt, ys, n = get_ys(sub, ds_ps, ds_ns, m)
                auprcs.append(compute_auprc(yt, ys) if n >= 10 else np.nan)
            else:
                t_auprcs = []
                for tissue in sub['tissue'].unique():
                    tt = sub[sub['tissue'] == tissue]
                    yt, ys, n = get_ys(tt, ds_ps, ds_ns, m)
                    if n >= 10:
                        t_auprcs.append(compute_auprc(yt, ys))
                auprcs.append(np.median(t_auprcs) if t_auprcs else np.nan)
        ax.plot(x, auprcs, marker='o', color=COLORS[m], label=MODEL_NAMES[m],
                lw=2, markersize=6, zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(offset_labels)
    ax.set_xlabel('|offset| from variant (bp)')
    ax.set_ylabel('AUPRC' + ('' if single_tissue else ' (tissue median)'))
    all_y = [v for line in ax.get_lines() for v in line.get_ydata() if np.isfinite(v)]
    if all_y:
        ax.set_ylim(np.floor(min(all_y) * 10) / 10, np.ceil(max(all_y) * 10) / 10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='best')
    ax.grid(alpha=0.3, zorder=0)
    ax.set_title(f'{ds_label} — AUPRC by Delta Position Offset')
    plt.tight_layout()
    savefig(FIG_BASE / fig_dir / 'delta_position' / '05_auprc_by_offset')
    plt.show()
    plt.close()

    # ---- plot 6: all-heads AUPRC by offset hbar (offset <=10 vs >10) ----
    # for each head: compute AUPRC when |offset| <= 10 ("at variant") vs > 10 ("elsewhere")
    near_far = []
    for h in all_heads:
        if h not in ds_ps or h not in ds_ns:
            continue
        p_off_h = np.array([np.abs(pos_positions[h].get(k, np.nan)) for k in ds_pairs['pos_var_key']])
        for label_nf, mask_nf in [('near', (p_off_h <= 10) & ~np.isnan(p_off_h)),
                                   ('far', (p_off_h > 10) & ~np.isnan(p_off_h))]:
            sub = ds_pairs[mask_nf]
            if len(sub) < 20:
                continue
            if single_tissue:
                yt, ys, n = get_ys(sub, ds_ps, ds_ns, h)
                val = compute_auprc(yt, ys) if n >= 10 else np.nan
            else:
                t_auprcs = []
                for tissue in sub['tissue'].unique():
                    tt = sub[sub['tissue'] == tissue]
                    yt, ys, n = get_ys(tt, ds_ps, ds_ns, h)
                    if n >= 10:
                        t_auprcs.append(compute_auprc(yt, ys))
                val = np.median(t_auprcs) if t_auprcs else np.nan
            near_far.append({'head': h, 'where': label_nf, 'auprc': val,
                            'n': mask_nf.sum()})

    if near_far:
        nf_df = pd.DataFrame(near_far)
        nf_near = nf_df[nf_df['where'] == 'near'].set_index('head')
        nf_far = nf_df[nf_df['where'] == 'far'].set_index('head')
        common = sorted(set(nf_near.index) & set(nf_far.index),
                       key=lambda h: nf_near.loc[h, 'auprc'] if h in nf_near.index else 0)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(common))))
        y = np.arange(len(common))
        bw = 0.35
        ax.barh(y - bw/2, [nf_near.loc[h, 'auprc'] for h in common], bw,
               color='#0072B2', alpha=0.8, label='|offset| ≤ 10 bp')
        ax.barh(y + bw/2, [nf_far.loc[h, 'auprc'] for h in common], bw,
               color='#D55E00', alpha=0.8, label='|offset| > 10 bp')
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_NAMES.get(h, h) for h in common], fontsize=7)
        ax.set_xlabel('AUPRC')
        ax.set_title(f'{ds_label} — AUPRC: at-variant vs elsewhere (all heads)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis='x')
        plt.tight_layout()
        savefig(FIG_BASE / fig_dir / 'delta_position' / '06_auprc_near_vs_far')
        plt.show()
        plt.close()

    # ---- plot 7: cross-model offset concordance (primary models, pos) ----
    from itertools import combinations as _comb_dp
    model_pairs_dp = list(_comb_dp(mp, 2))
    n_p = len(model_pairs_dp)
    ncols = min(5, n_p)
    nrows = (n_p + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)
    for pi, (ma, mb) in enumerate(model_pairs_dp):
        ax = axes[pi // ncols, pi % ncols]
        off_a = np.array([pos_positions[ma].get(k, np.nan) for k in pos_keys])
        off_b = np.array([pos_positions[mb].get(k, np.nan) for k in pos_keys])
        valid = ~(np.isnan(off_a) | np.isnan(off_b))
        ax.scatter(off_a[valid], off_b[valid], s=3, alpha=0.15, color='#555555', rasterized=True)
        lim = max(np.abs(off_a[valid]).max(), np.abs(off_b[valid]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, lw=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(MODEL_NAMES[ma], fontsize=7)
        ax.set_ylabel(MODEL_NAMES[mb], fontsize=7)
        rho = spearmanr(off_a[valid], off_b[valid])[0]
        exact = (off_a[valid] == off_b[valid]).mean()
        ax.text(0.03, 0.97, f'ρ={rho:.2f}\nexact={exact:.0%}', transform=ax.transAxes, fontsize=7, va='top')
        ax.set_aspect('equal')
    for pi in range(n_p, nrows * ncols):
        axes[pi // ncols, pi % ncols].set_visible(False)
    fig.suptitle(f'{ds_label} — Delta Position Concordance (positives)', y=1.02)
    plt.tight_layout()
    savefig(FIG_BASE / fig_dir / 'delta_position' / '07_cross_model_concordance')
    plt.show()
    plt.close()

    # ---- summary stats ----
    print(f'\n  {ds_label} offset summary (positives):')
    print(f'  {"Head":<30s} {"median |off|":>12s} {"at var (%)":>10s}')
    print(f'  {"-"*30} {"-"*12} {"-"*10}')
    for _, row in hs.iterrows():
        print(f'  {MODEL_NAMES.get(row["head"], row["head"]):<30s} {row["pos_median"]:12.0f} {row["pos_at_var"]:10.1f}')

    del pos_positions, neg_positions


# render analysis.md to html with quarto if available
import shutil
import subprocess
if shutil.which('quarto'):
    print("quarto found, rendering analysis.md → html")
    subprocess.run([
        'quarto', 'render', 'analysis.md', '--to', 'html',
        '-M', 'toc:false',
        '-M', 'embed-resources:true',
        '-M', 'theme:default',
        '-M', 'minimal:true',
    ], check=True)
    print("quarto render complete")
else:
    print("quarto not found, skipping html render")

