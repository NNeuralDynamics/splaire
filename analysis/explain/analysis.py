#!/usr/bin/env python
"""attribution analysis for splaire models"""

import os
import sys
import json
import shutil
import subprocess
import datetime
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from glob import glob
from collections import Counter

data_dir = "data"
models_dir = "../../models"

# figure output directory
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)
for subdir in ['data_overview', 'attribution_magnitude', 'ref_vs_var',
               'attribution_profiles', 'sequence_logos', 'position_analysis',
               'seqlets', 'model_comparison', 'fig5/main', 'fig5/supp']:
    os.makedirs(f"{fig_dir}/{subdir}", exist_ok=True)

# --- logging infrastructure ---
class TeeLogger:
    """write to both stdout and log file"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
        self.log.write(f"# analysis_script.py — {datetime.datetime.now().isoformat()}\n\n")
    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

_log_path = f'{fig_dir}/analysis.log'
sys.stdout = TeeLogger(_log_path)

# results collector for key numerical outputs
_results = {}

def log_result(key, value, desc=None):
    """save a key result for the results file"""
    _results[key] = {'value': value, 'desc': desc}
    tag = f" ({desc})" if desc else ""
    print(f"  >> RESULT: {key} = {value}{tag}")

def save_fig(path_stem):
    """save current figure as png (300 dpi) + pdf, then close"""
    plt.savefig(f'{path_stem}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{path_stem}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  saved: {path_stem}.{{png,pdf}}")

plt.rcParams['figure.dpi'] = 150


# sequence for attributions
seq_path = f"{data_dir}/sequences.h5"

# load all sequences and metadata
with h5py.File(seq_path, 'r', locking=False) as f:
    n = f.attrs['n_sequences']
    seq_len = f.attrs['sequence_length']
    center = f.attrs['center_index']
    
    X = f['X'][:]
    sequence = np.array([s.decode() for s in f['sequence'][:]])
    splice_label = np.array([s.decode() for s in f['splice_label'][:]])
    chromosome = np.array([c.decode() for c in f['chromosome'][:]])
    position = f['position'][:]
    
    tissues = ['brain_cortex', 'haec', 'lung', 'testis', 'whole_blood']
    ssu_data = {t: f[f'ssu_mean_{t}'][:] for t in tissues if f'ssu_mean_{t}' in f}
    ssu_per_tissue = ssu_data

    # mean SSU across tissues
    ssu_stack = np.stack([ssu_data[t] for t in tissues], axis=1)
    true_ssu = np.nanmean(ssu_stack, axis=1)

acc_mask = splice_label == 'acceptor'
don_mask = splice_label == 'donor'

print(f"loaded {n:,} sequences, {seq_len:,} bp, center={center}")
print(f"  acceptor: {acc_mask.sum():,}, donor: {don_mask.sum():,}")
print(f"  tissues: {tissues}")
log_result('n_sequences', n, 'total splice sites analyzed')
log_result('n_acceptor', int(acc_mask.sum()), 'acceptor splice sites')
log_result('n_donor', int(don_mask.sum()), 'donor splice sites')
log_result('sequence_length', seq_len, 'bp window per site')

# splice sites used in analysis
# test chromosomes (chr1, 3, 5, 7)



# splice type counts
counts = Counter(splice_label)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(counts.keys(), counts.values(), color=['steelblue', 'darkorange'])
ax.set_ylabel('count')
for i, (k, v) in enumerate(counts.items()):
    ax.text(i, v + 500, f'{v:,}', ha='center')
plt.tight_layout()
save_fig(f'{fig_dir}/data_overview/splice_type_distribution')


# **SSU per tissue**: Each sequence has a SSU measurement from one of the 5 tissues (brain_cortex, haec, lung, testis, whole_blood). NaN indicates the site was not observed in that tissue. This is the mean SSU across samples from each tissue.
# 

# sample ssu values
df = pd.DataFrame({t: ssu_data[t][:10] for t in tissues})
df


# ssu distributions per tissue
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes = axes.flatten()

for i, t in enumerate(tissues):
    ax = axes[i]
    valid = ~np.isnan(ssu_data[t])
    vals = ssu_data[t][valid]
    ax.hist(vals, bins=50, color='steelblue', edgecolor='white')
    ax.set_yscale('log')
    ax.set_xlabel('ssu')
    ax.set_ylabel('count')
    ax.set_title(f'{t} (n={valid.sum():,})')
    ax.axvline(vals.mean(), color='red', ls='--', alpha=0.7)

# mean across tissues
ax = axes[5]
ssu_stack = np.stack([ssu_data[t] for t in tissues], axis=1)
true_ssu = np.nanmean(ssu_stack, axis=1)
valid_ssu = ~np.isnan(true_ssu)
ax.hist(true_ssu[valid_ssu], bins=50, color='coral', edgecolor='white')
ax.set_yscale('log')
ax.set_xlabel('ssu')
ax.set_ylabel('count')
ax.set_title(f'mean across tissues (n={valid_ssu.sum():,})')

plt.tight_layout()
save_fig(f'{fig_dir}/data_overview/ssu_distributions')


# annotation command
with h5py.File(seq_path, 'r', locking=False) as f:
  n_nearby = f['nearby/n_sites'][:]

fig, ax = plt.subplots(figsize=(8, 4))

# create integer-aligned bins (edges at 0.5 intervals so bars center on integers)
max_val = int(n_nearby.max())
bins = np.arange(-0.5, max_val + 1.5, 1)

ax.hist(n_nearby, bins=bins, color='steelblue', edgecolor='white')
ax.set_xlabel('nearby sites per sequence')
ax.set_ylabel('count')
ax.set_title('nearby splice sites within +/-5kb')

# set x ticks at integers
ax.set_xticks(np.arange(0, max_val + 1, max(1, max_val // 10)))
ax.set_xlim(-0.5, min(max_val + 0.5, 50))  # cap display at 50 if needed

plt.tight_layout()
save_fig(f'{fig_dir}/data_overview/nearby_sites_histogram')


# model conversion
attr_path = f"{data_dir}/sphaec/attr_sphaec_ref_reg.h5"
with h5py.File(attr_path, 'r', locking=False) as f:
  obs_raw = f['obs_reg'][:]
  obs_norm = f['obs_norm_reg'][:]
  pred = f['pred_reg'][:]

from scipy.ndimage import gaussian_filter
from matplotlib.ticker import ScalarFormatter

def density_scatter(ax, x, y, bins=100, cmap='viridis'):
  """density-colored scatter, returns pearson r"""
  h, xe, ye = np.histogram2d(x, y, bins=bins)
  h = gaussian_filter(h, sigma=1)
  xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
  yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
  c = h[xi, yi]
  idx = c.argsort()
  ax.scatter(x[idx], y[idx], c=c[idx], s=1, alpha=0.5, cmap=cmap, rasterized=True)
  return np.corrcoef(x, y)[0, 1]

# completeness: signed raw sum should equal prediction
fig, ax = plt.subplots(figsize=(6, 5))
r = density_scatter(ax, pred, obs_raw.sum(axis=1))
ax.set_xlabel('prediction')
ax.set_ylabel('sum raw attribution (signed)')
ax.set_title('completeness: signed raw sum vs prediction')
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, va='top')
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.tight_layout()
save_fig(f'{fig_dir}/attribution_magnitude/completeness_check')
print(f"completeness r = {r:.4f}")
log_result('deeplift_completeness_r', round(float(r), 4), 'signed attr sum vs prediction correlation')


# why normalize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: raw magnitude scales with prediction
ax = axes[0]
r = density_scatter(ax, pred, np.abs(obs_raw).sum(axis=1))
ax.set_xlabel('prediction')
ax.set_ylabel('sum |raw attribution|')
ax.set_title('raw magnitude vs prediction')
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, va='top')
ax.ticklabel_format(style='plain', axis='y', useOffset=False)

# right: normalized magnitude is flat at 1.0
ax = axes[1]
r = density_scatter(ax, pred, np.abs(obs_norm).sum(axis=1))
ax.set_xlabel('prediction')
ax.set_ylabel('sum |normalized attribution|')
ax.set_title('normalized magnitude vs prediction')
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, va='top')
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
ax.set_ylim(0, 1.2)

plt.tight_layout()
save_fig(f'{fig_dir}/attribution_magnitude/raw_vs_norm_magnitude')


# what normalization loses
fig, ax = plt.subplots(figsize=(6, 5))
r = density_scatter(ax, pred, obs_norm.sum(axis=1))
ax.set_xlabel('prediction')
ax.set_ylabel('sum normalized attribution (signed)')
ax.set_title('signed normalized sum vs prediction')
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, va='top')
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.tight_layout()
save_fig(f'{fig_dir}/attribution_magnitude/signed_norm_vs_prediction')
print(f"signed normalized sum correlation with prediction: r = {r:.3f}")


# summary
window = 100
x = np.arange(-window, window + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

# top row: signed attribution
for col, (stype, mask, color) in enumerate([
    ('acceptor', acc_mask, 'steelblue'), ('donor', don_mask, 'darkorange')
]):
    ax = axes[0, col]
    vals = obs_norm[mask, center - window:center + window + 1]
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    ax.plot(x, mean, color=color, lw=1.5)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.axvline(0, color='red', ls='--', alpha=0.5)
    ax.axhline(0, color='gray', alpha=0.3)
    ax.set_title(f'{stype} (n={mask.sum():,})', fontweight='bold')
    if col == 0:
        ax.set_ylabel('signed attribution')

# bottom row: absolute attribution
for col, (stype, mask, color) in enumerate([
    ('acceptor', acc_mask, 'steelblue'), ('donor', don_mask, 'darkorange')
]):
    ax = axes[1, col]
    vals = np.abs(obs_norm[mask, center - window:center + window + 1])
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    ax.plot(x, mean, color=color, lw=1.5)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.axvline(0, color='red', ls='--', alpha=0.5)
    ax.axhline(0, color='gray', alpha=0.3)
    ax.set_xlabel('position (bp)')
    if col == 0:
        ax.set_ylabel('|attribution|')

plt.suptitle('mean attribution profile (+/- 1 std)', y=1.02)
plt.tight_layout()
save_fig(f'{fig_dir}/attribution_magnitude/mean_profiles_by_splice_type')


# ---
# 
# detailed analysis
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error
from tqdm.std import tqdm
import logomaker

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style('whitegrid')


# paths
seq_path = "data/sequences.h5"

# splaire attribution files
sphaec_dir = "data/sphaec"
attr_ref_reg_path = f"{sphaec_dir}/attr_sphaec_ref_reg.h5"
attr_ref_cls_path = f"{sphaec_dir}/attr_sphaec_ref_cls.h5"

# seqlet files (ref only, normalized) - multiple p-value thresholds
seqlet_dir = f"{sphaec_dir}/seqlets"

# threshold 0.001
seqlet_paths_001 = {
    'ref_reg': f"{seqlet_dir}/seqlets_ref_reg_norm.h5",
    'ref_cls_acc': f"{seqlet_dir}/seqlets_ref_cls_acceptor_acceptor_norm.h5",
    'ref_cls_don': f"{seqlet_dir}/seqlets_ref_cls_donor_donor_norm.h5",
}

# threshold 0.005
seqlet_paths_005 = {
    'ref_reg': f"{seqlet_dir}/seqlets_005_ref_reg_norm.h5",
    'ref_cls_acc': f"{seqlet_dir}/seqlets_005_ref_cls_acceptor_acceptor_norm.h5",
    'ref_cls_don': f"{seqlet_dir}/seqlets_005_ref_cls_donor_donor_norm.h5",
}

# threshold 0.01
seqlet_paths_01 = {
    'ref_reg': f"{seqlet_dir}/seqlets_01_ref_reg_norm.h5",
    'ref_cls_acc': f"{seqlet_dir}/seqlets_01_ref_cls_acceptor_acceptor_norm.h5",
    'ref_cls_don': f"{seqlet_dir}/seqlets_01_ref_cls_donor_donor_norm.h5",
}

# all thresholds dict
seqlet_paths_by_threshold = {
    0.001: seqlet_paths_001,
    0.005: seqlet_paths_005,
    0.01: seqlet_paths_01,
}


# sequences already loaded above (X, splice_label, true_ssu, etc)
# masks
acc_mask = splice_label == 'acceptor'
don_mask = splice_label == 'donor'

n_acc = acc_mask.sum()
n_don = don_mask.sum()
print(f"sequences: {n:,} ({n_acc:,} acceptor, {n_don:,} donor)")


# load attributions and predictions from all models

# === SPLAIRE ===
print("loading SPLAIRE attributions...", flush=True)
with h5py.File(attr_ref_reg_path, 'r', locking=False) as f:
    obs_ref_reg = f['obs_norm_reg'][:]
    pred_ref_reg = f['pred_reg'][:]
print("  ref_reg loaded", flush=True)

with h5py.File(attr_ref_cls_path, 'r', locking=False) as f:
    obs_ref_acceptor = f['obs_norm_acceptor'][:]
    obs_ref_donor = f['obs_norm_donor'][:]
    pred_ref_acceptor = f['pred_acceptor'][:]
    pred_ref_donor = f['pred_donor'][:]
print("  ref_cls loaded", flush=True)

# === SpliceAI ===
print("loading SpliceAI...", flush=True)
spliceai_path = f"{data_dir}/spliceai/attr_spliceai.h5"
with h5py.File(spliceai_path, 'r', locking=False) as f:
    pred_spliceai_acceptor = f['pred_acceptor'][:]
    pred_spliceai_donor = f['pred_donor'][:]

# === SpliceTransformer ===
print("loading SpliceTransformer...", flush=True)
spt_path = f"{data_dir}/splicetransformer/pred_splicetransformer.h5"
with h5py.File(spt_path, 'r', locking=False) as f:
    pred_spt_acceptor = f['pred_acceptor'][:]
    pred_spt_donor = f['pred_donor'][:]
    spt_tissues = ['adipose', 'blood', 'blood_vessel', 'brain', 'colon', 'heart',
                   'kidney', 'liver', 'lung', 'muscle', 'nerve', 'skin',
                   'small_intestine', 'spleen', 'stomach']
    pred_spt_tissue = {t: f[f'pred_{t}'][:] for t in spt_tissues}

# === Pangolin ===
print("loading Pangolin...", flush=True)
pang_tissues = ['brain', 'heart', 'liver', 'testis']
pred_pang_p_splice = {}
pred_pang_usage = {}
for tissue in pang_tissues:
    with h5py.File(f"{data_dir}/pang/attr_pangolin_{tissue}_p_splice.h5", 'r', locking=False) as f:
        pred_pang_p_splice[tissue] = f[f'pred_{tissue}_p_splice'][:]
    with h5py.File(f"{data_dir}/pang/attr_pangolin_{tissue}_usage.h5", 'r', locking=False) as f:
        pred_pang_usage[tissue] = f[f'pred_{tissue}_usage'][:]

print(f"loaded predictions for {n:,} sequences")
print(f"  SPLAIRE: ref_reg, ref_cls")
print(f"  SpliceAI: acceptor, donor")
print(f"  SpliceTransformer: cls + {len(spt_tissues)} tissues")
print(f"  Pangolin: {len(pang_tissues)} tissues × 2 heads")


# ## Prediction Summary

# build matched cls predictions (acceptor pred for acceptor sites, donor for donor)
acc_mask = splice_label == 'acceptor'
don_mask = splice_label == 'donor'

# === SPLAIRE matched ===
ref_cls_matched = np.zeros(n)
ref_cls_matched[acc_mask] = pred_ref_acceptor[acc_mask]
ref_cls_matched[don_mask] = pred_ref_donor[don_mask]

# === SpliceAI matched ===
spliceai_cls_matched = np.zeros(n)
spliceai_cls_matched[acc_mask] = pred_spliceai_acceptor[acc_mask]
spliceai_cls_matched[don_mask] = pred_spliceai_donor[don_mask]

# === SpliceTransformer matched (cls) ===
spt_cls_matched = np.zeros(n)
spt_cls_matched[acc_mask] = pred_spt_acceptor[acc_mask]
spt_cls_matched[don_mask] = pred_spt_donor[don_mask]

# === SpliceTransformer tissue mean ===
spt_tissue_mean = np.mean([pred_spt_tissue[t] for t in spt_tissues], axis=0)

# === Pangolin tissue means ===
pang_p_splice_mean = np.mean([pred_pang_p_splice[t] for t in pang_tissues], axis=0)
pang_usage_mean = np.mean([pred_pang_usage[t] for t in pang_tissues], axis=0)

print("built matched predictions for classification models")
print(f"  SPLAIRE ref_cls_matched: {ref_cls_matched.mean():.3f} mean")
print(f"  SpliceAI: {spliceai_cls_matched.mean():.3f} mean")
print(f"  SpliceTransformer cls: {spt_cls_matched.mean():.3f} mean")
print(f"  SpliceTransformer tissue mean: {spt_tissue_mean.mean():.3f} mean")
print(f"  Pangolin p_splice mean: {pang_p_splice_mean.mean():.3f} mean")
print(f"  Pangolin usage mean: {pang_usage_mean.mean():.3f} mean")


# Attribution Profiles
obs_ref_cls_matched = np.zeros_like(obs_ref_reg)
obs_ref_cls_matched[acc_mask] = obs_ref_acceptor[acc_mask]
obs_ref_cls_matched[don_mask] = obs_ref_donor[don_mask]

low_thr = 0.1
high_thr = 0.9
colors = {'low': 'tab:blue', 'mid': 'tab:orange', 'high': 'tab:red'}
x = np.arange(-window, window + 1)

valid_ssu_mask = ~np.isnan(true_ssu)






# Standalone Attribution Profiles by SSU
obs_ref_cls_matched = np.zeros_like(obs_ref_reg)
obs_ref_cls_matched[acc_mask] = obs_ref_acceptor[acc_mask]
obs_ref_cls_matched[don_mask] = obs_ref_donor[don_mask]

low_thr = 0.1
high_thr = 0.9

colors = {'high': '#e41a1c', 'mid': '#377eb8', 'low': '#4daf4a'}  # red, blue, green  
plot_window = 30  # show 50bp on each side of splice site
x = np.arange(-plot_window, plot_window + 1)

valid_ssu_mask = ~np.isnan(true_ssu)
print(f"valid SSU: {valid_ssu_mask.sum():,} / {len(true_ssu):,}")
print(f"SSU range: {np.nanmin(true_ssu):.3f} - {np.nanmax(true_ssu):.3f}")

# use true_ssu for all heads (same ground truth)
attr_sources_true = [
  ('Classification', obs_ref_cls_matched, true_ssu),
  ('Regression', obs_ref_reg, true_ssu),
]

# bin definitions: order high → mid → low for legend
bins_ordered = [
    ('high', 'SSU 0.9-1.0'),
    ('mid', 'SSU 0.1-0.9'),
    ('low', 'SSU 0-0.1'),
]

# compute global y limits (only valid SSU sequences)
all_vals = []
for attr_name, arr, ssu in attr_sources_true:
    for stype in ['Acceptor', 'Donor']:
        type_mask = (splice_label == stype.lower()) & valid_ssu_mask
        for key, label in bins_ordered:
            if key == 'low':
                bin_mask = type_mask & (ssu <= low_thr)
            elif key == 'high':
                bin_mask = type_mask & (ssu >= high_thr)
            else:
                bin_mask = type_mask & (ssu > low_thr) & (ssu < high_thr)
            if bin_mask.sum() > 0:
                vals = arr[bin_mask, center-plot_window:center+plot_window+1]
                all_vals.append(vals.mean(axis=0))

if len(all_vals) == 0:
    print("WARNING: no valid data for any SSU bin - check true_ssu values")
    print(f"  acceptor & valid: {(acc_mask & valid_ssu_mask).sum()}")
    print(f"  donor & valid: {(don_mask & valid_ssu_mask).sum()}")
else:
    ymax_true = max(np.abs(v).max() for v in all_vals) * 1.1
    ylim_true = (0, ymax_true)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for row, (attr_name, arr, ssu) in enumerate(attr_sources_true):
        for col, stype in enumerate(['Acceptor', 'Donor']):
            ax = axes[row, col]
            type_mask = (splice_label == stype.lower()) & valid_ssu_mask

            # plot in order: high → mid → low (legend order top to bottom)
            for key, label in bins_ordered:
                if key == 'low':
                    bin_mask = type_mask & (ssu <= low_thr)
                elif key == 'high':
                    bin_mask = type_mask & (ssu >= high_thr)
                else:
                    bin_mask = type_mask & (ssu > low_thr) & (ssu < high_thr)

                n_bin = bin_mask.sum()
                if n_bin > 0:
                    vals = arr[bin_mask, center-plot_window:center+plot_window+1]
                    avg = vals.mean(axis=0)
                    ax.plot(x, avg, label=f'{label} (n={n_bin:,})', color=colors[key], lw=1.4, alpha=0.95)

            vline_pos = 0 if stype == 'Acceptor' else 1
            ax.axvline(vline_pos, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            ax.set_ylim(ylim_true)
            ax.set_xlim(-plot_window, plot_window)
            ax.set_xticks(np.arange(-plot_window, plot_window+1, 10))

            if col == 0:
                ax.set_ylabel(f'{attr_name} Attribution')
            if row == 0:
                ax.set_title(stype, fontweight='bold')
            ax.set_xlabel('Position')
            if row == 0:
                ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle('SPLAIRE Mean Attribution', y=1.02)
    plt.tight_layout()
    save_fig(f'{fig_dir}/attribution_by_true_ssu')


# dinucleotide logo plots
import logomaker
from scipy.ndimage import gaussian_filter
from scipy import stats
from matplotlib.colors import Normalize

ANNOT_SIZE = 7
contour_bins = 100
contour_sigma = 1.1
contour_levels = 12
contour_cmap = 'turbo'
cbar_frac = 0.7
contour_pct = 90
freq_as_logo = False

acc_left, acc_right = 10, 9
don_left, don_right = 9, 10


def get_dinuc_mask(X, idx_mask, pos1, pos2, dinuc):
    """mask for sequences with specific dinucleotide at given positions"""
    bases = np.array(['A', 'C', 'G', 'T'])
    nuc1 = bases[X[idx_mask, pos1, :].argmax(axis=1)]
    nuc2 = bases[X[idx_mask, pos2, :].argmax(axis=1)]
    return np.char.add(nuc1, nuc2) == dinuc


def mean_attr_matrix_1d(X_win, obs_win, mask, norm_factors=None):
    """mean attribution per position per base from 1D obs"""
    n = mask.sum()
    if n == 0:
        return None, 0
    Xm = X_win[mask]
    Om = obs_win[mask]
    W = Xm.shape[1]
    M = np.zeros((W, 4), dtype=np.float64)
    for p in range(W):
        present = Xm[:, p, :].astype(bool)
        vals = Om[:, p]
        for b in range(4):
            sel = present[:, b]
            if sel.any():
                M[p, b] = vals[sel].mean()
    return M.astype(np.float32), n


def density_contour_panel(ax, x, y, shared_vmax=None):
    """contour density plot for pred vs true SSU"""
    H, xedges, yedges = np.histogram2d(x, y, bins=contour_bins, range=[[0, 1], [0, 1]])
    Z = gaussian_filter(H.T, sigma=contour_sigma)
    vmax = shared_vmax if shared_vmax is not None else (
        np.percentile(Z[Z > 0], contour_pct) if np.any(Z > 0) else 1.0)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    Xg, Yg = np.meshgrid(xc, yc)
    lev = np.linspace(0, vmax, contour_levels + 1)
    cf = ax.contourf(Xg, Yg, Z, levels=lev, cmap=contour_cmap, vmin=0, vmax=vmax, extend='max')
    ax.plot([0, 1], [0, 1], '-', color='white', lw=0.8, alpha=0.5)
    r = stats.pearsonr(x, y)[0]
    ax.text(0.5, 0.5, f'r={r:.2f}', transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color='white')
    return cf


def compute_density_vmax(x, y):
    """compute density vmax without plotting"""
    H, _, _ = np.histogram2d(x, y, bins=contour_bins, range=[[0, 1], [0, 1]])
    Z = gaussian_filter(H.T, sigma=contour_sigma)
    return np.percentile(Z[Z > 0], contour_pct) if np.any(Z > 0) else 1.0


def plot_dinuc_triplet(splice_type, dinucs, obs_arr, pred_arr, title, save_path):
    """plot logo + frequency heatmap + density contour for each dinucleotide group"""
    is_acc = splice_type == 'acceptor'
    type_mask = acc_mask if is_acc else don_mask
    left = acc_left if is_acc else don_left
    right = acc_right if is_acc else don_right
    pos1 = center - 2 if is_acc else center + 1
    pos2 = center - 1 if is_acc else center + 2

    s, e = center - left, center + right + 1
    X_win = X[:, s:e, :]
    obs_win = obs_arr[:, s:e]
    boundary = left - 0.5 if is_acc else left + 0.5

    # position labels
    pos_labels = []
    for p in range(-left, right + 1):
        if is_acc:
            pos_labels.append(str(p) if p < 0 else f'+{p+1}')
        else:
            pos_labels.append(str(p - 1) if p <= 0 else f'+{p}')

    # compute shared y-limits and density vmax
    all_matrices = []
    density_vmax = 0
    for dinuc in dinucs:
        dm = get_dinuc_mask(X, type_mask, pos1, pos2, dinuc)
        fm = np.zeros(len(X), dtype=bool); fm[type_mask] = dm
        M, _ = mean_attr_matrix_1d(X_win, obs_win, fm, None)
        if M is not None:
            all_matrices.append(M)
        v = ~np.isnan(true_ssu[fm])
        if v.sum() > 0:
            density_vmax = max(density_vmax, compute_density_vmax(pred_arr[fm][v], true_ssu[fm][v]))

    all_vals = np.concatenate([m.flatten() for m in all_matrices])
    y_pad = (all_vals.max() - all_vals.min()) * 0.15
    y_lim = (all_vals.min() - y_pad, all_vals.max() + y_pad)

    fig, axes = plt.subplots(len(dinucs), 3, figsize=(18, 3.5 * len(dinucs)),
                             gridspec_kw={'width_ratios': [1.5, 1.5, 1]})
    if len(dinucs) == 1:
        axes = axes[np.newaxis, :]

    for i, dinuc in enumerate(dinucs):
        dm = get_dinuc_mask(X, type_mask, pos1, pos2, dinuc)
        fm = np.zeros(len(X), dtype=bool); fm[type_mask] = dm
        n_seqs = fm.sum()

        # logo
        ax = axes[i, 0]
        M, _ = mean_attr_matrix_1d(X_win, obs_win, fm, None)
        df_logo = pd.DataFrame(M, columns=['A', 'C', 'G', 'U'])
        logomaker.Logo(df_logo, ax=ax, color_scheme='classic')
        ax.set_ylim(y_lim); ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(boundary, color='red', ls='--', lw=1)
        ax.set_ylabel(f'{dinuc} ({n_seqs:,})')
        ax.set_xticks(range(len(pos_labels))); ax.set_xticklabels(pos_labels)
        if i == len(dinucs) - 1: ax.set_xlabel('Position')

        # frequency heatmap
        ax = axes[i, 1]
        freq = X_win[fm].mean(axis=0).T
        im = ax.imshow(freq, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['A','C','G','U'])
        ax.set_xticks(range(freq.shape[1])); ax.set_xticklabels(pos_labels)
        ax.axvline(boundary, color='red', ls='--', lw=1)
        for yi in range(4):
            for xi in range(freq.shape[1]):
                f = freq[yi, xi]; clr = 'white' if f > 0.5 else 'black'
                ax.text(xi, yi, f'{f*100:.0f}', ha='center', va='center',
                        fontsize=ANNOT_SIZE, color=clr)
        if i == len(dinucs) - 1: ax.set_xlabel('Position')

        # density contour
        ax = axes[i, 2]
        ps = pred_arr[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
        last_cf = density_contour_panel(ax, ps[v], ts[v], shared_vmax=density_vmax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('Predicted SSU'); ax.set_ylabel('True SSU')
        ax.set_aspect('equal')

    fig.suptitle(title, y=1.02)
    plt.tight_layout(w_pad=6)
    # colorbars
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=Normalize(0, 1)); sm.set_array([])
    bt, bb = axes[0, 1].get_position(), axes[-1, 1].get_position()
    h = (bt.y1 - bb.y0) * cbar_frac
    cax = fig.add_axes([bt.x1 + 0.015, (bt.y1 + bb.y0 - h) / 2, 0.012, h])
    fig.colorbar(sm, cax=cax, label='Frequency')
    bt, bb = axes[0, 2].get_position(), axes[-1, 2].get_position()
    h = (bt.y1 - bb.y0) * cbar_frac
    cax = fig.add_axes([bt.x1 + 0.015, (bt.y1 + bb.y0 - h) / 2, 0.012, h])
    fig.colorbar(last_cf, cax=cax, label='Density')
    save_fig(save_path)


# splaire regression logos
acc_dinucs = ['AG', 'AC']
don_dinucs = ['GT', 'GC', 'AT']
obs_arr = obs_ref_reg
pred_arr = pred_ref_reg

plot_dinuc_triplet('acceptor', acc_dinucs, obs_arr, pred_arr,
                   'Acceptor — SPLAIRE SSU Regression',
                   f'{fig_dir}/logo_acceptor_by_dinucleotide')
plot_dinuc_triplet('donor', don_dinucs, obs_arr, pred_arr,
                   'Donor — SPLAIRE SSU Regression',
                   f'{fig_dir}/logo_donor_by_dinucleotide')

# pangolin usage logos (per tissue)
pang_dir = "data/pang"
for tissue in ["brain", "heart", "liver", "testis"]:
    path = f"{pang_dir}/attr_pangolin_{tissue}_usage.h5"
    with h5py.File(path, 'r', locking=False) as f:
        obs_pang = f[f'obs_norm_{tissue}_usage'][:]
        pred_pang = f[f'pred_{tissue}_usage'][:]
    print(f"loaded pangolin {tissue}", flush=True)
    plot_dinuc_triplet('acceptor', acc_dinucs, obs_pang, pred_pang,
                       f'Acceptor — Pangolin {tissue.title()} Usage',
                       f'{fig_dir}/logo_acceptor_pangolin_{tissue}_usage')
    plot_dinuc_triplet('donor', don_dinucs, obs_pang, pred_pang,
                       f'Donor — Pangolin {tissue.title()} Usage',
                       f'{fig_dir}/logo_donor_pangolin_{tissue}_usage')
    del obs_pang, pred_pang  # free memory

# splaire classification logos
obs_ref_cls_matched = np.zeros_like(obs_ref_reg)
obs_ref_cls_matched[acc_mask] = obs_ref_acceptor[acc_mask]
obs_ref_cls_matched[don_mask] = obs_ref_donor[don_mask]

plot_dinuc_triplet('acceptor', acc_dinucs, obs_ref_cls_matched, ref_cls_matched,
                   'Acceptor — SPLAIRE Classification',
                   f'{fig_dir}/logo_acceptor_cls_by_dinucleotide')
plot_dinuc_triplet('donor', don_dinucs, obs_ref_cls_matched, ref_cls_matched,
                   'Donor — SPLAIRE Classification',
                   f'{fig_dir}/logo_donor_cls_by_dinucleotide')

print(f"all dinucleotide logos saved")


# === pangolin usage attribution profiles ===
# per-tissue DeepLIFT-SHAP for pangolin usage head
# caches mean attr matrices to skip reloading 2.7GB obs_norm per tissue

import pickle

pang_fig_dir = f"{fig_dir}/pangolin"
pang_cache_dir = f"{pang_fig_dir}/cache"
os.makedirs(pang_fig_dir, exist_ok=True)
os.makedirs(pang_cache_dir, exist_ok=True)

pang_attr_tissues = ['brain', 'heart', 'liver', 'testis']

# donor logo window (same as fig5)
_pd_pos1, _pd_pos2 = center + 1, center + 2
_pd_left, _pd_right = 5, 6
_pd_s, _pd_e = center - _pd_left, center + _pd_right + 1
_pd_boundary = _pd_left + 0.5
X_win_pd = X[:, _pd_s:_pd_e, :]

_pd_labels = []
for p in range(-_pd_left, _pd_right + 1):
    _pd_labels.append(str(p - 1) if p <= 0 else f'+{p}')

# acceptor logo window (mirrors donor)
_pa_pos1, _pa_pos2 = center - 2, center - 1
_pa_left, _pa_right = 6, 5
_pa_s, _pa_e = center - _pa_left, center + _pa_right + 1
_pa_boundary = _pa_left - 0.5
X_win_pa = X[:, _pa_s:_pa_e, :]

_pa_labels = []
for p in range(-_pa_left, _pa_right + 1):
    _pa_labels.append(str(p) if p < 0 else f'+{p+1}')

# dinuc masks (model-independent, reused across tissues)
_pd_masks = {}
for dinuc in don_dinucs:
    dm = get_dinuc_mask(X, don_mask, _pd_pos1, _pd_pos2, dinuc)
    fm = np.zeros(n, dtype=bool); fm[don_mask] = dm
    _pd_masks[dinuc] = (fm, dinuc.replace('T', 'U'), fm.sum())

_pa_masks = {}
for dinuc in acc_dinucs:
    dm = get_dinuc_mask(X, acc_mask, _pa_pos1, _pa_pos2, dinuc)
    fm = np.zeros(n, dtype=bool); fm[acc_mask] = dm
    _pa_masks[dinuc] = (fm, dinuc.replace('T', 'U'), fm.sum())

# splice type configs for looping
_pang_stypes = [
    ('donor', don_dinucs, _pd_masks, X_win_pd, _pd_s, _pd_e, _pd_labels, _pd_boundary),
    ('acceptor', acc_dinucs, _pa_masks, X_win_pa, _pa_s, _pa_e, _pa_labels, _pa_boundary),
]

# frequency heatmaps (model-independent, save once)
for stype, dinucs, masks, X_win, _, _, labels, bnd in _pang_stypes:
    nd = len(dinucs)
    fig_h = 5.0 if nd == 3 else 3.3
    fig, axes = plt.subplots(nd, 1, figsize=(5, fig_h))
    if nd == 1: axes = [axes]
    for i, dinuc in enumerate(dinucs):
        ax = axes[i]
        fm, rna, ns = masks[dinuc]
        freq = X_win[fm].mean(axis=0).T
        ax.imshow(freq, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['A', 'C', 'G', 'U'])
        ax.set_xticks(range(freq.shape[1])); ax.set_xticklabels(labels)
        ax.axvline(bnd, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_ylabel(f'{rna} (n={ns:,})')
        for yi in range(4):
            for xi in range(freq.shape[1]):
                f = freq[yi, xi]; clr = 'white' if f > 0.5 else 'black'
                ax.text(xi, yi, f'{f*100:.0f}', ha='center', va='center',
                        fontsize=ANNOT_SIZE, color=clr)
        if i == nd - 1: ax.set_xlabel('Position')
    plt.suptitle(f'{stype.title()} Nucleotide Frequency', y=1.02)
    plt.tight_layout()
    save_fig(f'{pang_fig_dir}/{stype}_freq')

# per-tissue attribution analysis
_pang_summary = []

for tissue in pang_attr_tissues:
    cache_path = f"{pang_cache_dir}/{tissue}_usage.pkl"

    if os.path.exists(cache_path):
        print(f"loading cached pangolin {tissue}...")
        with open(cache_path, 'rb') as cf:
            tc = pickle.load(cf)
    else:
        print(f"loading pangolin {tissue} usage attributions...")
        h5_path = f"{pang_dir}/attr_pangolin_{tissue}_usage.h5"
        with h5py.File(h5_path, 'r', locking=False) as f:
            obs_norm = f[f'obs_norm_{tissue}_usage'][:]
            pred = f[f'pred_{tissue}_usage'][:]
        print(f"  loaded: obs_norm {obs_norm.shape}, pred {pred.shape}")

        tc = {'pred': pred, 'donor': {}, 'acceptor': {}}
        for stype, dinucs, masks, X_win, s, e, _, _ in _pang_stypes:
            obs_win = obs_norm[:, s:e]
            for dinuc in dinucs:
                fm, _, _ = masks[dinuc]
                M, ns = mean_attr_matrix_1d(X_win, obs_win, fm, None)
                tc[stype][dinuc] = {'matrix': M, 'n': ns}

        with open(cache_path, 'wb') as cf:
            pickle.dump(tc, cf)
        print(f"  cached to {cache_path}")
        del obs_norm

    pred = tc['pred']

    for stype, dinucs, masks, _, _, _, labels, bnd in _pang_stypes:
        nd = len(dinucs)
        fig_h = 5.0 if nd == 3 else 3.3

        # shared y-limits for logos
        mats = [tc[stype][d]['matrix'] for d in dinucs]
        _ymax = max(m.clip(min=0).sum(axis=1).max() for m in mats)
        _ymin = min(m.clip(max=0).sum(axis=1).min() for m in mats)
        _ypad = (_ymax - _ymin) * 0.1
        ylim = (_ymin - _ypad, _ymax + _ypad)

        # logos
        fig, axes = plt.subplots(nd, 1, figsize=(5, fig_h), sharex=True)
        if nd == 1: axes = [axes]
        for i, dinuc in enumerate(dinucs):
            ax = axes[i]
            M = tc[stype][dinuc]['matrix']
            ns = tc[stype][dinuc]['n']
            rna = dinuc.replace('T', 'U')
            logomaker.Logo(pd.DataFrame(M, columns=['A', 'C', 'G', 'U']),
                           ax=ax, color_scheme='classic')
            ax.set_ylim(ylim); ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(bnd, color='gray', ls='--', lw=1, alpha=0.5)
            ax.set_ylabel('Attribution')
            ax.text(0.02, 0.95, f'{rna} ({ns:,})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=10)
            ax.set_xticks(range(len(labels)))
            if i == nd - 1:
                ax.set_xticklabels(labels); ax.set_xlabel('Position')
            else:
                ax.set_xticklabels([])
        plt.suptitle(f'Pangolin {tissue.title()} Usage — {stype.title()}', y=1.02)
        plt.tight_layout()
        save_fig(f'{pang_fig_dir}/{tissue}_{stype}_logos')

        # kde density
        fig, axes = plt.subplots(nd, 1, figsize=(4, fig_h), sharex=True)
        if nd == 1: axes = [axes]
        for i, dinuc in enumerate(dinucs):
            ax = axes[i]
            fm, rna, ns = masks[dinuc]
            ps = pred[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
            sns.kdeplot(x=ps[v], y=ts[v], ax=ax, fill=True, cmap='turbo', thresh=0)
            r = stats.pearsonr(ps[v], ts[v])[0]
            ax.text(0.5, 0.5, f'r={r:.2f}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='white')
            ax.plot([0, 1], [0, 1], '-', color='white', lw=0.8, alpha=0.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_ylabel(rna)
            if i == nd - 1:
                ax.set_xlabel('Predicted SSU')
            else:
                ax.set_xticklabels([])
        fig.supylabel('True SSU', x=0.02)
        plt.suptitle(f'Pangolin {tissue.title()} Usage — {stype.title()}', y=1.02)
        plt.tight_layout()
        save_fig(f'{pang_fig_dir}/{tissue}_{stype}_kde')

        # summary stats
        for dinuc in dinucs:
            fm, rna, ns = masks[dinuc]
            ps = pred[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
            r = stats.pearsonr(ps[v], ts[v])[0]
            _pang_summary.append({
                'tissue': tissue, 'type': stype, 'dinuc': rna,
                'n': int(v.sum()), 'r': round(float(r), 3),
            })

    print(f"pangolin {tissue}: done")
    del tc

# summary table
print("\npangolin usage prediction summary by dinucleotide:")
print(f"  {'tissue':<10s} {'type':<10s} {'dinuc':<6s} {'n':>8s} {'r':>6s}")
for row in _pang_summary:
    print(f"  {row['tissue']:<10s} {row['type']:<10s} {row['dinuc']:<6s} "
          f"{row['n']:>8,d} {row['r']:>6.3f}")
log_result('pangolin_n_tissues', len(pang_attr_tissues), 'tissues with attribution profiles')
print("pangolin attribution analysis complete")


# === splaire classification attribution profiles ===
# donor + acceptor logos and kde using cls head
# obs_ref_cls_matched and ref_cls_matched already in memory

cls_fig_dir = f"{fig_dir}/splaire_cls"
os.makedirs(cls_fig_dir, exist_ok=True)

_cls_summary = []

for stype, dinucs, masks, X_win, s, e, labels, bnd in _pang_stypes:
    nd = len(dinucs)
    fig_h = 5.0 if nd == 3 else 3.3

    # mean attr matrices
    obs_win = obs_ref_cls_matched[:, s:e]
    cls_mats = {}
    for dinuc in dinucs:
        fm, _, _ = masks[dinuc]
        M, ns = mean_attr_matrix_1d(X_win, obs_win, fm, None)
        cls_mats[dinuc] = {'matrix': M, 'n': ns}

    # shared y-limits
    mats = [cls_mats[d]['matrix'] for d in dinucs]
    _ymax = max(m.clip(min=0).sum(axis=1).max() for m in mats)
    _ymin = min(m.clip(max=0).sum(axis=1).min() for m in mats)
    _ypad = (_ymax - _ymin) * 0.1
    ylim = (_ymin - _ypad, _ymax + _ypad)

    # logos
    fig, axes = plt.subplots(nd, 1, figsize=(5, fig_h), sharex=True)
    if nd == 1: axes = [axes]
    for i, dinuc in enumerate(dinucs):
        ax = axes[i]
        M = cls_mats[dinuc]['matrix']
        ns = cls_mats[dinuc]['n']
        rna = dinuc.replace('T', 'U')
        logomaker.Logo(pd.DataFrame(M, columns=['A', 'C', 'G', 'U']),
                       ax=ax, color_scheme='classic')
        ax.set_ylim(ylim); ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(bnd, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_ylabel('Attribution')
        ax.text(0.02, 0.95, f'{rna} ({ns:,})', transform=ax.transAxes,
                ha='left', va='top', fontsize=10)
        ax.set_xticks(range(len(labels)))
        if i == nd - 1:
            ax.set_xticklabels(labels); ax.set_xlabel('Position')
        else:
            ax.set_xticklabels([])
    plt.suptitle(f'SPLAIRE Classification — {stype.title()}', y=1.02)
    plt.tight_layout()
    save_fig(f'{cls_fig_dir}/{stype}_logos')

    # kde
    fig, axes = plt.subplots(nd, 1, figsize=(4, fig_h), sharex=True)
    if nd == 1: axes = [axes]
    for i, dinuc in enumerate(dinucs):
        ax = axes[i]
        fm, rna, ns = masks[dinuc]
        ps = ref_cls_matched[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
        sns.kdeplot(x=ps[v], y=ts[v], ax=ax, fill=True, cmap='turbo', thresh=0)
        r = stats.pearsonr(ps[v], ts[v])[0]
        ax.text(0.5, 0.5, f'r={r:.2f}', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='white')
        ax.plot([0, 1], [0, 1], '-', color='white', lw=0.8, alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_ylabel(rna)
        if i == nd - 1:
            ax.set_xlabel('Predicted')
        else:
            ax.set_xticklabels([])
    fig.supylabel('True SSU', x=0.02)
    plt.suptitle(f'SPLAIRE Classification — {stype.title()}', y=1.02)
    plt.tight_layout()
    save_fig(f'{cls_fig_dir}/{stype}_kde')

    for dinuc in dinucs:
        fm, rna, ns = masks[dinuc]
        ps = ref_cls_matched[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
        r = stats.pearsonr(ps[v], ts[v])[0]
        _cls_summary.append({
            'type': stype, 'dinuc': rna, 'n': int(v.sum()), 'r': round(float(r), 3),
        })

print("\nsplaire cls prediction summary by dinucleotide:")
print(f"  {'type':<10s} {'dinuc':<6s} {'n':>8s} {'r':>6s}")
for row in _cls_summary:
    print(f"  {row['type']:<10s} {row['dinuc']:<6s} {row['n']:>8,d} {row['r']:>6.3f}")
print("splaire cls attribution profiles complete")


# === pangolin p_splice attribution profiles ===
# same structure as usage but for the splice probability head

_psplice_summary = []

for tissue in pang_attr_tissues:
    cache_path = f"{pang_cache_dir}/{tissue}_p_splice.pkl"

    if os.path.exists(cache_path):
        print(f"loading cached pangolin {tissue} p_splice...")
        with open(cache_path, 'rb') as cf:
            tc = pickle.load(cf)
    else:
        print(f"loading pangolin {tissue} p_splice attributions...")
        h5_path = f"{pang_dir}/attr_pangolin_{tissue}_p_splice.h5"
        with h5py.File(h5_path, 'r', locking=False) as f:
            obs_norm = f[f'obs_norm_{tissue}_p_splice'][:]
            pred = f[f'pred_{tissue}_p_splice'][:]
        print(f"  loaded: obs_norm {obs_norm.shape}, pred {pred.shape}")

        tc = {'pred': pred, 'donor': {}, 'acceptor': {}}
        for stype, dinucs, masks, X_win, s, e, _, _ in _pang_stypes:
            obs_win = obs_norm[:, s:e]
            for dinuc in dinucs:
                fm, _, _ = masks[dinuc]
                M, ns = mean_attr_matrix_1d(X_win, obs_win, fm, None)
                tc[stype][dinuc] = {'matrix': M, 'n': ns}

        with open(cache_path, 'wb') as cf:
            pickle.dump(tc, cf)
        print(f"  cached to {cache_path}")
        del obs_norm

    pred = tc['pred']

    for stype, dinucs, masks, _, _, _, labels, bnd in _pang_stypes:
        nd = len(dinucs)
        fig_h = 5.0 if nd == 3 else 3.3

        # shared y-limits
        mats = [tc[stype][d]['matrix'] for d in dinucs]
        _ymax = max(m.clip(min=0).sum(axis=1).max() for m in mats)
        _ymin = min(m.clip(max=0).sum(axis=1).min() for m in mats)
        _ypad = (_ymax - _ymin) * 0.1
        ylim = (_ymin - _ypad, _ymax + _ypad)

        # logos
        fig, axes = plt.subplots(nd, 1, figsize=(5, fig_h), sharex=True)
        if nd == 1: axes = [axes]
        for i, dinuc in enumerate(dinucs):
            ax = axes[i]
            M = tc[stype][dinuc]['matrix']
            ns = tc[stype][dinuc]['n']
            rna = dinuc.replace('T', 'U')
            logomaker.Logo(pd.DataFrame(M, columns=['A', 'C', 'G', 'U']),
                           ax=ax, color_scheme='classic')
            ax.set_ylim(ylim); ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(bnd, color='gray', ls='--', lw=1, alpha=0.5)
            ax.set_ylabel('Attribution')
            ax.text(0.02, 0.95, f'{rna} ({ns:,})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=10)
            ax.set_xticks(range(len(labels)))
            if i == nd - 1:
                ax.set_xticklabels(labels); ax.set_xlabel('Position')
            else:
                ax.set_xticklabels([])
        plt.suptitle(f'Pangolin {tissue.title()} P(splice) — {stype.title()}', y=1.02)
        plt.tight_layout()
        save_fig(f'{pang_fig_dir}/{tissue}_psplice_{stype}_logos')

        # kde
        fig, axes = plt.subplots(nd, 1, figsize=(4, fig_h), sharex=True)
        if nd == 1: axes = [axes]
        for i, dinuc in enumerate(dinucs):
            ax = axes[i]
            fm, rna, ns = masks[dinuc]
            ps = pred[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
            sns.kdeplot(x=ps[v], y=ts[v], ax=ax, fill=True, cmap='turbo', thresh=0)
            r = stats.pearsonr(ps[v], ts[v])[0]
            ax.text(0.5, 0.5, f'r={r:.2f}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='white')
            ax.plot([0, 1], [0, 1], '-', color='white', lw=0.8, alpha=0.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_ylabel(rna)
            if i == nd - 1:
                ax.set_xlabel('P(splice)')
            else:
                ax.set_xticklabels([])
        fig.supylabel('True SSU', x=0.02)
        plt.suptitle(f'Pangolin {tissue.title()} P(splice) — {stype.title()}', y=1.02)
        plt.tight_layout()
        save_fig(f'{pang_fig_dir}/{tissue}_psplice_{stype}_kde')

        for dinuc in dinucs:
            fm, rna, ns = masks[dinuc]
            ps = pred[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
            r = stats.pearsonr(ps[v], ts[v])[0]
            _psplice_summary.append({
                'tissue': tissue, 'type': stype, 'dinuc': rna,
                'n': int(v.sum()), 'r': round(float(r), 3),
            })

    print(f"pangolin {tissue} p_splice: done")
    del tc

print("\npangolin p_splice prediction summary by dinucleotide:")
print(f"  {'tissue':<10s} {'type':<10s} {'dinuc':<6s} {'n':>8s} {'r':>6s}")
for row in _psplice_summary:
    print(f"  {row['tissue']:<10s} {row['type']:<10s} {row['dinuc']:<6s} "
          f"{row['n']:>8,d} {row['r']:>6.3f}")
log_result('pangolin_psplice_n_tissues', len(pang_attr_tissues), 'tissues with p_splice profiles')
print("pangolin p_splice attribution analysis complete")


# === figure 5 publication panels ===
# individual panels for LaTeX composition: attribution profiles, logos, freq, density
# plus supplemental spread metrics

import seaborn as _sns
from scipy.stats import kruskal, mannwhitneyu
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

fig5_dir = f'{fig_dir}/fig5/main'
fig5_supp_dir = f'{fig_dir}/fig5/supp'

# attribution window sizes (asymmetric around splice site)
_acc_left, _acc_right = 20, 10
_don_left_attr, _don_right_attr = 10, 20
colors_ssu = {'high': '#e41a1c', 'mid': '#377eb8', 'low': '#4daf4a'}

attr_sources_true = [
    ('Classification', obs_ref_cls_matched, true_ssu),
    ('Regression', obs_ref_reg, true_ssu),
]
bins_ordered = [
    ('high', 'SSU 0.9-1.0'),
    ('mid', 'SSU 0.1-0.9'),
    ('low', 'SSU 0-0.1'),
]

# precompute y-limits for attribution profiles
all_vals_top = []
for attr_name, arr, ssu in attr_sources_true:
    for stype in ['Acceptor', 'Donor']:
        type_mask = (splice_label == stype.lower()) & valid_ssu_mask
        for key, label in bins_ordered:
            if key == 'low': bin_mask = type_mask & (ssu <= low_thr)
            elif key == 'high': bin_mask = type_mask & (ssu >= high_thr)
            else: bin_mask = type_mask & (ssu > low_thr) & (ssu < high_thr)
            if bin_mask.sum() > 0:
                _l = _acc_left if stype == 'Acceptor' else _don_left_attr
                _r = _acc_right if stype == 'Acceptor' else _don_right_attr
                vals = arr[bin_mask, center-_l:center+_r+1]
                all_vals_top.append(vals.mean(axis=0))
ymax_fig5 = max(np.abs(v).max() for v in all_vals_top) * 1.1
ylim_fig5 = (0, ymax_fig5)


def _add_exon_schematic(fig, ax_acc, ax_don, y_fig, rect_h=0.012, line_lw=1.0):
    """draw intron-[exon]-//-[exon]-intron schematic below panels"""
    fig.canvas.draw()
    def _d2f(ax, xd):
        return fig.transFigure.inverted().transform(
            ax.transData.transform((xd, 0)))[0]
    xL   = _d2f(ax_acc, ax_acc.get_xlim()[0])
    xEL  = _d2f(ax_acc, 0)
    xELR = _d2f(ax_acc, ax_acc.get_xlim()[1])
    xERL = _d2f(ax_don, ax_don.get_xlim()[0])
    xER  = _d2f(ax_don, 0)
    xR   = _d2f(ax_don, ax_don.get_xlim()[1])
    # left intron
    fig.add_artist(Line2D([xL, xEL], [y_fig, y_fig],
        transform=fig.transFigure, color='black', lw=line_lw, clip_on=False, zorder=10))
    # left exon half
    fig.add_artist(Rectangle((xEL, y_fig - rect_h/2), xELR - xEL, rect_h,
        transform=fig.transFigure, facecolor='black', edgecolor='black',
        clip_on=False, zorder=10))
    # break: double slash //
    _gcx = (xELR + xERL) / 2
    _sw = 0.005; _sh = rect_h * 2.5
    for _dx in [-0.004, 0.004]:
        fig.add_artist(Line2D(
            [_gcx + _dx - _sw/2, _gcx + _dx + _sw/2],
            [y_fig - _sh/2, y_fig + _sh/2],
            transform=fig.transFigure, color='black', lw=line_lw,
            clip_on=False, zorder=10))
    # right exon half
    fig.add_artist(Rectangle((xERL, y_fig - rect_h/2), xER - xERL, rect_h,
        transform=fig.transFigure, facecolor='black', edgecolor='black',
        clip_on=False, zorder=10))
    # right intron
    fig.add_artist(Line2D([xER, xR], [y_fig, y_fig],
        transform=fig.transFigure, color='black', lw=line_lw, clip_on=False, zorder=10))


# --- panel a: attribution profiles 1x4 ---
from matplotlib.gridspec import GridSpec

fig_a4 = plt.figure(figsize=(14, 2.5))
gs_a4 = fig_a4.add_gridspec(1, 5, width_ratios=[1, 1, 0.15, 1, 1],
                             left=0.08, right=0.98, top=0.82, bottom=0.28,
                             wspace=0.20)
axes_a4 = [fig_a4.add_subplot(gs_a4[0, i]) for i in [0, 1, 3, 4]]
_a4_combos = [(0, 'Acceptor'), (0, 'Donor'), (1, 'Acceptor'), (1, 'Donor')]

for ci, (ri, stype) in enumerate(_a4_combos):
    attr_name, arr, ssu = attr_sources_true[ri]
    ax = axes_a4[ci]
    type_mask = (splice_label == stype.lower()) & valid_ssu_mask
    _pw_l = _acc_left if stype == 'Acceptor' else _don_left_attr
    _pw_r = _acc_right if stype == 'Acceptor' else _don_right_attr
    _x_attr = np.arange(-_pw_l, _pw_r + 1)
    for key, label in bins_ordered:
        if key == 'low': bm = type_mask & (ssu <= low_thr)
        elif key == 'high': bm = type_mask & (ssu >= high_thr)
        else: bm = type_mask & (ssu > low_thr) & (ssu < high_thr)
        if bm.sum() > 0:
            vals = arr[bm, center-_pw_l:center+_pw_r+1]
            ax.plot(_x_attr, vals.mean(axis=0), label=label,
                    color=colors_ssu[key], lw=1.5, alpha=0.95)
    vl = 0 if stype == 'Acceptor' else 1
    ax.axvline(vl, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylim(ylim_fig5); ax.set_xlim(-_pw_l, _pw_r)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.set_xticks(np.arange(-_pw_l, _pw_r+1, 10))
    ax.set_title(['Acceptor', 'Donor', 'Acceptor', 'Donor'][ci])
    ax.tick_params(axis='x', pad=16)
    if ci == 0:
        ax.set_ylabel('Attribution')
        ax.legend(loc='upper left', framealpha=0.9)
    if ci in (1, 3):
        ax.set_ylabel(''); ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False, length=0)
        ax.spines['left'].set_visible(False)
    if ci == 2:
        ax.set_ylabel(''); ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False, length=0)

# group labels
fig_a4.canvas.draw()
for grp_label, ax_l, ax_r in [
    ('Classification', axes_a4[0], axes_a4[1]),
    ('Regression', axes_a4[2], axes_a4[3]),
]:
    x_l = ax_l.get_position().x0
    x_r = ax_r.get_position().x1
    y_t = ax_l.get_position().y1
    fig_a4.text((x_l + x_r) / 2, y_t + 0.06, grp_label,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# exon schematics
fig_a4.canvas.draw()
_row_bot4 = axes_a4[0].get_position().y0
_schem_y4 = _row_bot4 - 10 / 72 / fig_a4.get_figheight()
_add_exon_schematic(fig_a4, axes_a4[0], axes_a4[1], _schem_y4, rect_h=0.018)
_add_exon_schematic(fig_a4, axes_a4[2], axes_a4[3], _schem_y4, rect_h=0.018)
plt.savefig(f'{fig5_dir}/panel_a_attribution_1x4.png', dpi=300, bbox_inches=None)
plt.savefig(f'{fig5_dir}/panel_a_attribution_1x4.pdf', bbox_inches=None)
plt.close()
print("fig5 panel a: attribution profiles 1x4")


# --- precompute donor dinucleotide data for panels b, c, d ---
_don_pos1, _don_pos2 = center + 1, center + 2
_don_left, _don_right = 5, 6
_don_s, _don_e = center - _don_left, center + _don_right + 1
_don_boundary = _don_left + 0.5

_pos_don = []
for p in range(-_don_left, _don_right + 1):
    _pos_don.append(str(p - 1) if p <= 0 else f'+{p}')

X_win_don = X[:, _don_s:_don_e, :]
obs_win_don = obs_ref_reg[:, _don_s:_don_e]

_dinuc_data = []
for dinuc in don_dinucs:
    dm = get_dinuc_mask(X, don_mask, _don_pos1, _don_pos2, dinuc)
    fm = np.zeros(len(X), dtype=bool); fm[don_mask] = dm
    rna = dinuc.replace('T', 'U')
    _dinuc_data.append((dinuc, rna, fm, fm.sum()))

# shared y-limits for logos
all_matrices_don = []
for dinuc, rna, fm, ns in _dinuc_data:
    M, _ = mean_attr_matrix_1d(X_win_don, obs_win_don, fm, None)
    if M is not None:
        all_matrices_don.append(M)
_max_stack = max(m.clip(min=0).sum(axis=1).max() for m in all_matrices_don)
_min_stack = min(m.clip(max=0).sum(axis=1).min() for m in all_matrices_don)
_y_pad = (_max_stack - _min_stack) * 0.1
y_lim_don = (_min_stack - _y_pad, _max_stack + _y_pad)

# shared density vmax
density_vmax_don = 0
for dinuc, rna, fm, ns in _dinuc_data:
    v = ~np.isnan(true_ssu[fm])
    if v.sum() > 0:
        density_vmax_don = max(density_vmax_don,
                               compute_density_vmax(pred_ref_reg[fm][v], true_ssu[fm][v]))

n_don = len(don_dinucs)


# --- panel b: attribution logos ---
fig_b, axes_b = plt.subplots(n_don, 1, figsize=(4, 5), sharex=True)
for i, (dinuc, rna, fm, ns) in enumerate(_dinuc_data):
    ax = axes_b[i]
    M, _ = mean_attr_matrix_1d(X_win_don, obs_win_don, fm, None)
    df_logo = pd.DataFrame(M, columns=['A', 'C', 'G', 'U'])
    logomaker.Logo(df_logo, ax=ax, color_scheme='classic')
    ax.set_ylim(y_lim_don); ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(_don_boundary, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel('Attribution')
    ax.text(0.02, 0.95, f'{rna} ({ns:,})', transform=ax.transAxes,
            ha='left', va='top', fontsize=10)
    ax.set_xticks(range(len(_pos_don)))
    if i == n_don - 1:
        ax.set_xticklabels(_pos_don); ax.set_xlabel('Position')
    else:
        ax.set_xticklabels([])
plt.tight_layout()
save_fig(f'{fig5_dir}/panel_b_logos')
print("fig5 panel b: attribution logos")


# --- panel c: nucleotide frequency heatmaps ---
fig_c, axes_c = plt.subplots(n_don, 1, figsize=(4, 5))
for i, (dinuc, rna, fm, ns) in enumerate(_dinuc_data):
    ax = axes_c[i]
    freq = X_win_don[fm].mean(axis=0).T
    ax.imshow(freq, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['A', 'C', 'G', 'U'])
    ax.set_xticks(range(freq.shape[1])); ax.set_xticklabels(_pos_don)
    ax.axvline(_don_boundary, color='gray', ls='--', lw=1, alpha=0.5)
    for yi in range(4):
        for xi in range(freq.shape[1]):
            f = freq[yi, xi]; clr = 'white' if f > 0.5 else 'black'
            ax.text(xi, yi, f'{f*100:.0f}', ha='center', va='center',
                    fontsize=ANNOT_SIZE, color=clr)
    if i == n_don - 1:
        ax.set_xlabel('Position')
plt.tight_layout()
# add colorbar below
fig_c.canvas.draw()
_fp = axes_c[-1].get_position()
cax_f = fig_c.add_axes([_fp.x0 + _fp.width * 0.1, 0.02, _fp.width * 0.8, 0.015])
sm_f = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=100))
sm_f.set_array([])
cbar_f = fig_c.colorbar(sm_f, cax=cax_f, orientation='horizontal')
cbar_f.set_label('Frequency (%)')
plt.savefig(f'{fig5_dir}/panel_c_freq.png', dpi=300)
plt.savefig(f'{fig5_dir}/panel_c_freq.pdf')
plt.close()
print("  saved: {fig5_dir}/panel_c_freq.{{png,pdf}}")
print("fig5 panel c: nucleotide frequency")


# --- panel d: density (kde) ---
fig_dk, axes_dk = plt.subplots(n_don, 1, figsize=(3, 5), sharex=True)
for i, (dinuc, rna, fm, ns) in enumerate(_dinuc_data):
    ax = axes_dk[i]
    ps = pred_ref_reg[fm]; ts = true_ssu[fm]; v = ~np.isnan(ts)
    _sns.kdeplot(x=ps[v], y=ts[v], ax=ax, fill=True, cmap='turbo', thresh=0)
    r = stats.pearsonr(ps[v], ts[v])[0]
    ax.text(0.5, 0.5, f'r={r:.2f}', transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color='white')
    ax.plot([0, 1], [0, 1], '-', color='white', lw=0.8, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel('')
    if i == n_don - 1:
        ax.set_xlabel('Predicted SSU')
    else:
        ax.set_xticklabels([])
fig_dk.supylabel('True SSU', x=0.02)
plt.tight_layout()
save_fig(f'{fig5_dir}/panel_d_kde')
print("fig5 panel d: kde density")


# --- supplemental: attribution spread metrics ---
core_half = 50
core_start = center - core_half
core_end = center + core_half + 1

ssu_bin_keys = ['low', 'mid', 'high']
ssu_bin_fns = {
    'low': lambda s: s <= low_thr,
    'mid': lambda s: (s > low_thr) & (s < high_thr),
    'high': lambda s: s >= high_thr,
}
ssu_legend_labels = {
    'low': 'SSU 0\u20130.1', 'mid': 'SSU 0.1\u20130.9', 'high': 'SSU 0.9\u20131.0',
}
splice_types = [('Acceptor', acc_mask), ('Donor', don_mask)]

def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def _holm_correct(pvals):
    """holm-bonferroni correction, returns adjusted p-values"""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(pvals[idx] * (n - rank), 1.0)
    for rank in range(1, n):
        idx = order[rank]
        prev = order[rank - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev])
    return adjusted

for head_label, obs_spread in [('Classification', obs_ref_cls_matched),
                                ('Regression', obs_ref_reg)]:
    tag = head_label.lower()[:3]

    # compute metrics per (splice_type, ssu_bin)
    metrics_by_group = {}
    group_sizes = {}
    for stype_name, stype_mask in splice_types:
        metrics_by_group[stype_name] = {}
        group_sizes[stype_name] = {}
        for bin_key in ssu_bin_keys:
            mask = stype_mask & valid_ssu_mask & ssu_bin_fns[bin_key](true_ssu)
            n_sites = mask.sum()
            if n_sites == 0:
                continue
            group_sizes[stype_name][bin_key] = n_sites
            arr_s = obs_spread[mask]
            abs_arr = np.abs(arr_s)
            total = abs_arr.sum(axis=1, keepdims=True)
            p = abs_arr / np.maximum(total, 1e-10)

            core_sum = abs_arr[:, core_start:core_end].sum(axis=1)
            distal = 1.0 - (core_sum / np.maximum(total.squeeze(), 1e-10))

            log_p = np.log2(p + 1e-30)
            entropy = -np.sum(p * log_p, axis=1)

            sorted_desc = np.sort(abs_arr, axis=1)[:, ::-1]
            cs = np.cumsum(sorted_desc, axis=1)
            eff_w = (cs < (0.9 * total)).sum(axis=1) + 1.0

            pos_sum = np.where(arr_s > 0, arr_s, 0.0).sum(axis=1)
            pos_frac = pos_sum / np.maximum(total.squeeze(), 1e-10)

            metrics_by_group[stype_name][bin_key] = {
                'distal': distal, 'entropy': entropy,
                'eff_width': eff_w, 'pos_frac': pos_frac,
            }

    # statistical tests
    pairs = [('low', 'mid'), ('mid', 'high'), ('low', 'high')]
    stat_results = {}
    for stype_name in ['Acceptor', 'Donor']:
        for mkey in ['distal', 'entropy', 'eff_width', 'pos_frac']:
            bins_present = [k for k in ssu_bin_keys if k in metrics_by_group[stype_name]]
            if len(bins_present) < 2:
                continue
            data_per_bin = [metrics_by_group[stype_name][k][mkey] for k in bins_present]
            _, kw_p = kruskal(*data_per_bin)
            raw_pvals = []
            pair_keys = []
            for a, b in pairs:
                if a in metrics_by_group[stype_name] and b in metrics_by_group[stype_name]:
                    _, pw_p = mannwhitneyu(
                        metrics_by_group[stype_name][a][mkey],
                        metrics_by_group[stype_name][b][mkey],
                        alternative='two-sided')
                    raw_pvals.append(pw_p)
                    pair_keys.append((a, b))
            adj_pvals = _holm_correct(np.array(raw_pvals))
            stat_results[(stype_name, mkey)] = {
                'kw_p': kw_p,
                'pairs': {pk: ap for pk, ap in zip(pair_keys, adj_pvals)},
            }

    # plot
    metric_defs = [
        ('distal', f'Distal fraction\n(outside \u00b1{core_half} bp)'),
        ('entropy', 'Shannon entropy\n(bits)'),
        ('eff_width', 'Effective width\n(bp, 90% |attr|)'),
        ('pos_frac', 'Positive attribution\nfraction'),
    ]

    pos_map = {}
    pos_counter = 0
    stype_center = {}
    for stype_name, _ in splice_types:
        stype_pos = []
        for bin_key in ssu_bin_keys:
            if bin_key in metrics_by_group[stype_name]:
                pos_counter += 1
                pos_map[(stype_name, bin_key)] = pos_counter
                stype_pos.append(pos_counter)
        stype_center[stype_name] = np.mean(stype_pos)
        pos_counter += 1

    n_acc_bins = sum(1 for k in ssu_bin_keys if k in metrics_by_group['Acceptor'])
    sep_x = n_acc_bins + 0.5 + 0.5

    fig_s, axes_s = plt.subplots(1, 4, figsize=(16, 4.0))
    for ax_s, (mkey, mlabel) in zip(axes_s, metric_defs):
        box_data = []
        box_colors = []
        positions = []
        for stype_name, _ in splice_types:
            for bin_key in ssu_bin_keys:
                if bin_key in metrics_by_group[stype_name]:
                    box_data.append(metrics_by_group[stype_name][bin_key][mkey])
                    box_colors.append(colors_ssu[bin_key])
                    positions.append(pos_map[(stype_name, bin_key)])

        bp = ax_s.boxplot(box_data, positions=positions, patch_artist=True,
                          showfliers=False, widths=0.65)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color('black'); line.set_linewidth(0.8)

        ax_s.set_xticks([stype_center[s] for s, _ in splice_types])
        ax_s.set_xticklabels([s for s, _ in splice_types])
        ax_s.axvline(sep_x, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax_s.set_ylabel(mlabel)

        if mkey == 'pos_frac':
            ax_s.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)

        # significance brackets
        for stype_name, _ in splice_types:
            key = (stype_name, mkey)
            if key not in stat_results:
                continue
            sr = stat_results[key]
            y_lo, y_hi = ax_s.get_ylim()
            y_range = y_hi - y_lo
            bracket_step = y_range * 0.07
            bracket_y = y_hi - y_range * 0.02
            for (a, b), adj_p in sr['pairs'].items():
                if (stype_name, a) not in pos_map or (stype_name, b) not in pos_map:
                    continue
                stars = _sig_stars(adj_p)
                if stars == 'ns':
                    continue
                x1 = pos_map[(stype_name, a)]
                x2 = pos_map[(stype_name, b)]
                tip = bracket_y + bracket_step * 0.3
                ax_s.plot([x1, x1, x2, x2], [bracket_y, tip, tip, bracket_y],
                          lw=0.8, color='black', clip_on=False)
                ax_s.text((x1 + x2) / 2, tip, stars, ha='center', va='bottom',
                          fontsize=8, color='black')
                bracket_y += bracket_step

    legend_patches = [Patch(facecolor=colors_ssu[k], alpha=0.7, edgecolor='black',
                            linewidth=0.5, label=ssu_legend_labels[k])
                      for k in ssu_bin_keys]
    axes_s[0].legend(handles=legend_patches, loc='best', framealpha=0.9, handlelength=1.2)

    fig_s.suptitle(f'{head_label} Head \u2014 Attribution Spread', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(f'{fig5_supp_dir}/spread_{tag}')
    print(f"fig5 supplemental: spread ({head_label})")

    # summary table
    print(f"\n  {'group':<25s} {'n':>6s} {'distal':>8s} {'entropy':>8s} {'eff_w':>8s} {'pos_frac':>8s}")
    for stype_name in ['Acceptor', 'Donor']:
        for bin_key in ssu_bin_keys:
            if bin_key not in metrics_by_group[stype_name]:
                continue
            m = metrics_by_group[stype_name][bin_key]
            ns = group_sizes[stype_name][bin_key]
            print(f"  {stype_name + ' ' + bin_key:<25s} {ns:>6,d} "
                  f"{m['distal'].mean():>8.4f} {m['entropy'].mean():>8.2f} "
                  f"{m['eff_width'].mean():>8.0f} {m['pos_frac'].mean():>8.4f}")

    # statistical tests
    print(f"\n  kruskal-wallis + pairwise mann-whitney U (holm-corrected):")
    for stype_name in ['Acceptor', 'Donor']:
        for mkey, _ in metric_defs:
            key = (stype_name, mkey)
            if key not in stat_results:
                continue
            sr = stat_results[key]
            pw_str = ', '.join(f'{a}-{b}: {_sig_stars(p)} (p={p:.1e})'
                               for (a, b), p in sr['pairs'].items())
            print(f"  {stype_name:>8s} {mkey:<10s}  KW p={sr['kw_p']:.1e}  {pw_str}")
    print()

print("fig5 all panels complete")


# === write results file ===
results_path = f'{fig_dir}/results.txt'
with open(results_path, 'w') as _rf:
    _rf.write(f"# attribution analysis results — {datetime.datetime.now().isoformat()}\n")
    _rf.write(f"# auto-generated by analysis_script.py\n\n")
    max_key = max(len(k) for k in _results) if _results else 10
    for k, v in _results.items():
        desc = f"  # {v['desc']}" if v['desc'] else ""
        _rf.write(f"{k:<{max_key+2}} = {v['value']}{desc}\n")
print(f"\nresults written to {results_path} ({len(_results)} entries)")

# also save as json for programmatic access
with open(f'{fig_dir}/results.json', 'w') as _jf:
    json.dump({k: v['value'] for k, v in _results.items()}, _jf, indent=2, default=str)
print(f"results json written to {fig_dir}/results.json")

# close logger
print(f"\nlog written to {_log_path}")
print("done.")
sys.stdout.close()
sys.stdout = sys.stdout.terminal

# render analysis_report.qmd (pdf + html) if quarto available
if shutil.which('quarto'):
    print("quarto found, rendering analysis_report.qmd")
    subprocess.run([
        'quarto', 'render', 'analysis_report.qmd', '--to', 'pdf',
    ], check=True)
    print("pdf render complete")
    subprocess.run([
        'quarto', 'render', 'analysis_report.qmd', '--to', 'html',
    ], check=True)
    print("html render complete")
else:
    print("quarto not found, skipping render")

