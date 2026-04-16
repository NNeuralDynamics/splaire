#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

# cli flags
# usage: python analysis.py [--skip-bootstrap]
SKIP_BOOTSTRAP = "--skip-bootstrap" in sys.argv

# 5kb on each side of splice site = 10,001 bp total window
WINDOW_HALF = 5000
SEQ_LEN = 2 * WINDOW_HALF + 1

# reference genome path used by both datasets
FASTA_PATH = Path("vex_seq/hg19.fa")


# reverse complement
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def revcomp(seq):
    return seq.translate(COMP)[::-1]

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot(seq):
    seq = seq.upper()
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in BASE_TO_IDX:
            arr[i, BASE_TO_IDX[base]] = 1.0
    return arr


def get_window(chrom, center, half=WINDOW_HALF):
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


def build_ref_alt_pair(chrom, center, var_pos, ref_allele, alt_allele):
    """build reference and alternate sequences for a variant"""
    ref_seq, start1, end1 = get_window(chrom, center)
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


vex_data_dir = Path("vex_seq/data")
vex_data_dir.mkdir(exist_ok=True)

URLS = {
    "train": "https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/vexseq/HepG2_delta_PSI_CAGI_training.csv",
    "test": "https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/vexseq/HepG2_delta_PSI_CAGI_testing.csv",
    "truth": "https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/master/data/vexseq/Vexseq_HepG2_delta_PSI_CAGI_test_true.tsv",
}

for name, url in URLS.items():
    ext = "csv" if url.endswith(".csv") else "tsv"
    path = vex_data_dir / f"{name}.{ext}"
    if not path.exists():
        print(f"downloading {name}...")
        os.system(f'curl -sL "{url}" -o "{path}"')


train = pd.read_csv(vex_data_dir / "train.csv")
test = pd.read_csv(vex_data_dir / "test.csv")
truth = pd.read_csv(vex_data_dir / "truth.tsv", sep="\t")

print(f"train: {len(train):,} variants")
print(f"test:  {len(test):,} variants")
print(f"truth: {len(truth):,} labels")


train.head()


test.head()


truth.head()


# width should equal end - start + 1
computed_width = train["end"] - train["start"] + 1
assert (computed_width == train["width"]).all()


print(f"delta-PSI range: {train['HepG2_delta_psi'].min():.1f} to {train['HepG2_delta_psi'].max():.1f}")
print(f"delta-PSI mean: {train['HepG2_delta_psi'].mean():.2f}")
print(f"delta-PSI std: {train['HepG2_delta_psi'].std():.2f}")


# The test CSV doesn't include delta-PSI values merge from truth file.

train["chrom"] = train["seqnames"].astype(str)
train["pos"] = train["hg19_variant_position"].astype(int)
train["ref"] = train["reference"].astype(str)
train["alt"] = train["variant"].astype(str)

test["chrom"] = test["seqnames"].astype(str)
test["pos"] = test["hg19_variant_position"].astype(int)
test["ref"] = test["reference"].astype(str)
test["alt"] = test["variant"].astype(str)

truth["chrom"] = truth["chromosome"].astype(str)
truth["pos"] = truth["hg19_variant_position"].astype(int)
truth["ref"] = truth["reference"].astype(str)
truth["alt"] = truth["variant"].astype(str)


key_cols = ["chrom", "pos", "ref", "alt"]
truth_labels = truth[key_cols + ["HepG2_delta_psi"]].drop_duplicates(subset=key_cols)

test_merged = test.merge(truth_labels, on=key_cols, how="left")

missing = test_merged["HepG2_delta_psi"].isna().sum()
print(f"test variants missing labels: {missing}")
assert missing == 0, "some test variants couldn't be matched to truth"


train["split"] = "train"
test_merged["split"] = "test"

keep_cols = ["seqnames", "start", "end", "strand", "hg19_variant_position",
             "reference", "variant", "HepG2_delta_psi", "split"]

vex_combined = pd.concat([train[keep_cols], test_merged[keep_cols]], ignore_index=True)
print(f"combined dataset: {len(vex_combined):,} variants")
print(f"  train: {(vex_combined['split'] == 'train').sum():,}")
print(f"  test:  {(vex_combined['split'] == 'test').sum():,}")


n_before = len(vex_combined)
vex_combined = vex_combined.drop_duplicates()
n_removed = n_before - len(vex_combined)
print(f"removed {n_removed} duplicate rows")
print(f"final: {len(vex_combined):,} unique variants")


# Majority SNVs with some indels.

ref_lens = vex_combined["reference"].str.len()
alt_lens = vex_combined["variant"].str.len()

is_snv = (ref_lens == 1) & (alt_lens == 1)
is_insertion = ref_lens < alt_lens
is_deletion = ref_lens > alt_lens

print(f"variant types:")
print(f"  SNVs:       {is_snv.sum():,} ({100 * is_snv.mean():.1f}%)")
print(f"  insertions: {is_insertion.sum():,} ({100 * is_insertion.mean():.1f}%)")
print(f"  deletions:  {is_deletion.sum():,} ({100 * is_deletion.mean():.1f}%)")


fig, ax = plt.subplots(figsize=(8, 4))

delta_psi = vex_combined["HepG2_delta_psi"].values
ax.hist(delta_psi, bins=50, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)

ax.set_xlabel("delta-PSI (%)")
ax.set_ylabel("count")
ax.set_title(f"VexSeq: distribution of measured delta-PSI (n={len(delta_psi):,})")

textstr = f"mean: {delta_psi.mean():.1f}%\nstd: {delta_psi.std():.1f}%\nrange: [{delta_psi.min():.0f}, {delta_psi.max():.0f}]"
ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()


pos = vex_combined["hg19_variant_position"].values
exon_start = vex_combined["start"].values
exon_end = vex_combined["end"].values

dist_to_start = np.abs(pos - exon_start)
dist_to_end = np.abs(pos - exon_end)
dist_to_nearest = np.minimum(dist_to_start, dist_to_end)

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(dist_to_nearest, bins=50, edgecolor="white", linewidth=0.5)
ax.set_xlabel("distance to nearest splice site (bp)")
ax.set_ylabel("count")
ax.set_title("distance to nearest splice site")
ax.text(0.98, 0.95, f"median: {np.median(dist_to_nearest):.0f} bp\nmax: {dist_to_nearest.max():,} bp",
        transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"distance summary:")
print(f"  variants at splice site (0bp): {(dist_to_nearest == 0).sum():,}")
print(f"  variants within 10bp: {(dist_to_nearest <= 10).sum():,}")
print(f"  variants within 50bp: {(dist_to_nearest <= 50).sum():,}")
print(f"  variants > 100bp away: {(dist_to_nearest > 100).sum():,}")


vex_out = vex_data_dir / "vex_seq.h5"

if vex_out.exists():
    print(f"skipping vexseq sequence build — {vex_out} exists")
else:
    assert FASTA_PATH.exists(), f"need hg19 reference genome at {FASTA_PATH}"
    fasta = pysam.FastaFile(str(FASTA_PATH))
    print(f"loaded {FASTA_PATH}")

    vex_seqs = {
        "exon_start_ref": [], "exon_start_alt": [],
        "exon_end_ref": [], "exon_end_alt": [],
    }
    vex_meta = {
        "chrom": [], "pos": [], "ref": [], "alt": [], "strand": [],
        "exon_start": [], "exon_end": [], "delta_psi": [], "split": [],
    }
    skipped = []

    for r in tqdm(vex_combined.itertuples(index=False), total=len(vex_combined), desc="vexseq sequences"):
        chrom, strand = str(r.seqnames), str(r.strand)
        pos, ref, alt = int(r.hg19_variant_position), str(r.reference), str(r.variant)
        ex_start, ex_end = int(r.start), int(r.end)

        if strand not in {"+", "-"}:
            skipped.append((pos, f"invalid strand: {strand}")); continue
        ref_start, alt_start, err = build_ref_alt_pair(chrom, ex_start, pos, ref, alt)
        if err:
            skipped.append((pos, f"exon_start: {err}")); continue
        ref_end, alt_end, err = build_ref_alt_pair(chrom, ex_end, pos, ref, alt)
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

    print(f"one-hot encoding {n_vex:,} variants x 4 sequences...")
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


mfass_data_dir = Path("mfass/data")
mfass_data_dir.mkdir(exist_ok=True)

snv_url = "https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt"
snv_path = mfass_data_dir / "snv_data_clean.txt"
if not snv_path.exists():
    print(f"downloading mfass data...")
    os.system(f'curl -sL "{snv_url}" -o "{snv_path}"')


# The MFASS data has three categories for each oligo (row). We only use the **mutant** rows.

df = pd.read_csv(snv_path, sep="\t")
print(f"loaded {len(df):,} rows")
print(f"\ncategory breakdown:")
print(df["category"].value_counts())


df.head()


# natural rows have no variant — confirm they're empty
natural = df[df["category"] == "natural"]
assert natural["ref_allele"].isna().all()
print(f"natural rows: {len(natural):,} (no variants, skipping these)")


# Keep only mutant rows and rename columns to match the VexSeq.

df_mut = df[df["category"] == "mutant"].copy()

df_mut = df_mut.rename(columns={
    "chr": "chrom",
    "ref_allele": "ref",
    "alt_allele": "alt",
    "snp_position": "pos",
})
df_mut["pos"] = df_mut["pos"].astype(int)

print(f"mutant rows: {len(df_mut):,}")


# MFASS only contains SNVs

ref_lens = df_mut["ref"].str.len()
alt_lens = df_mut["alt"].str.len()

is_snv = (ref_lens == 1) & (alt_lens == 1)
is_insertion = ref_lens < alt_lens
is_deletion = ref_lens > alt_lens

print(f"snvs: {is_snv.sum():,} ({100 * is_snv.mean():.1f}%)")
print(f"insertions: {is_insertion.sum():,} ({100 * is_insertion.mean():.1f}%)")
print(f"deletions: {is_deletion.sum():,} ({100 * is_deletion.mean():.1f}%)")


computed_span = df_mut["end"] - df_mut["start"]
expected_span = df_mut["intron1_len"] + df_mut["exon_len"] + df_mut["intron2_len"] - 1

mismatch = (computed_span != expected_span).sum()
if mismatch > 0:
    print(f"rows with inconsistent coordinates: {mismatch}")


df_mut["exon_start"] = df_mut["start"] + df_mut["intron1_len"]
df_mut["exon_end"] = df_mut["exon_start"] + df_mut["exon_len"] - 1

exon_end_check = df_mut["end"] - df_mut["intron2_len"]
end_mismatch = (df_mut["exon_end"] != exon_end_check).sum()
if end_mismatch > 0:
    print(f"exon_end verification: {end_mismatch} mismatches")
print(f"exon sizes: min={df_mut['exon_len'].min()}, median={df_mut['exon_len'].median():.0f}, max={df_mut['exon_len'].max()}")


fasta = pysam.FastaFile(str(FASTA_PATH))
print(f"reference genome: {FASTA_PATH.name}")


# check if natural_seq matches genome forward or revcomp
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
print(f"\nstrand validation against hg19 reference:")
print(f"  forward (+ strand): {n_fwd:,}")
print(f"  revcomp (- strand): {n_rev:,}")
print(f"  mismatch (removed): {n_mis:,}")

# cross-check: does annotated strand agree with empirical orientation?
strand_disagree = (
    ((df_mut["strand"] == "+") & (df_mut["orientation"] == "revcomp")) |
    ((df_mut["strand"] == "-") & (df_mut["orientation"] == "forward"))
)
n_disagree = strand_disagree.sum()
if n_disagree > 0:
    print(f"\n  strand/orientation disagreements: {n_disagree:,}")
    print(f"  these will be corrected using empirical orientation")
else:
    print(f"\n  all annotated strands match derived orientation")


# remove sequence mismatches
n_mismatch = (df_mut["orientation"] == "mismatch").sum()
if n_mismatch > 0:
    print(f"removing {n_mismatch} rows with sequence mismatches")
    df_mut = df_mut[df_mut["orientation"] != "mismatch"].reset_index(drop=True)
else:
    print(f"all {len(df_mut):,} sequences match the genome")

# correct strand from empirical orientation
df_mut["strand"] = df_mut["orientation"].map({"forward": "+", "revcomp": "-"})

# remove variants without measured delta-psi
unlabeled = df_mut["v2_dpsi"].isna()
n_unlabeled = unlabeled.sum()
if n_unlabeled > 0:
    print(f"\n{n_unlabeled:,} variants have no measured v2_dpsi:")
    print(df_mut.loc[unlabeled, ["ensembl_id", "chrom", "pos", "v1_dpsi", "v2_dpsi"]].head(10))
    df_mut = df_mut[~unlabeled].reset_index(drop=True)
    print(f"removed, {len(df_mut):,} variants remaining")

# count splice-disrupting variants (SDV = |delta-psi| > 0.5)
n_sdv = (df_mut["v2_dpsi"].abs() > 0.5).sum()
n_neg = len(df_mut) - n_sdv
print(f"\nlabeled: {len(df_mut):,} ({n_sdv:,} splice-disrupting, {n_neg:,} neutral)")


delta_psi = df_mut["v2_dpsi"].values
n_labeled = np.sum(~np.isnan(delta_psi))
print(f"{n_labeled:,} of {len(delta_psi):,} variants have measured delta-psi")


fig, ax = plt.subplots(figsize=(8, 4))

labeled_dpsi = delta_psi[~np.isnan(delta_psi)]
ax.hist(labeled_dpsi, bins=50, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)

ax.set_xlabel("delta-psi")
ax.set_ylabel("count")
ax.set_title(f"MFASS: measured delta-psi (n={len(labeled_dpsi):,})")

stats_text = f"mean: {labeled_dpsi.mean():.2f}\nstd: {labeled_dpsi.std():.2f}\nrange: [{labeled_dpsi.min():.2f}, {labeled_dpsi.max():.2f}]"
ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()


pos = df_mut["pos"].values
exon_start = df_mut["exon_start"].values
exon_end = df_mut["exon_end"].values

dist_to_start = np.abs(pos - exon_start)
dist_to_end = np.abs(pos - exon_end)
dist_to_nearest = np.minimum(dist_to_start, dist_to_end)

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(dist_to_nearest, bins=50, edgecolor="white", linewidth=0.5)
ax.set_xlabel("distance to nearest splice site (bp)")
ax.set_ylabel("count")
ax.set_title("distance to nearest splice site")
ax.text(0.98, 0.95, f"median: {np.median(dist_to_nearest):.0f} bp\nmax: {dist_to_nearest.max():,} bp",
        transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

print(f"distance summary:")
print(f"  at splice site (0bp): {(dist_to_nearest == 0).sum():,}")
print(f"  within 10bp: {(dist_to_nearest <= 10).sum():,}")
print(f"  within 50bp: {(dist_to_nearest <= 50).sum():,}")
print(f"  > 100bp away: {(dist_to_nearest > 100).sum():,}")


mfass_out = mfass_data_dir / "mfass.h5"

if mfass_out.exists():
    print(f"skipping mfass sequence build — {mfass_out} exists")
else:
    mfass_seqs = {
        "exon_start_ref": [], "exon_start_alt": [],
        "exon_end_ref": [], "exon_end_alt": [],
    }
    mfass_meta = {
        "chrom": [], "pos": [], "ref": [], "alt": [], "strand": [],
        "exon_start": [], "exon_end": [], "delta_psi": [], "exon_id": [],
    }
    skipped = []

    for r in tqdm(df_mut.itertuples(index=False), total=len(df_mut), desc="mfass sequences"):
        chrom, strand = str(r.chrom), str(r.strand)
        pos, ref, alt = int(r.pos), str(r.ref), str(r.alt)
        ex_start, ex_end = int(r.exon_start), int(r.exon_end)

        if strand not in {"+", "-"}:
            skipped.append((pos, f"invalid strand: {strand}")); continue
        ref_start, alt_start, err = build_ref_alt_pair(chrom, ex_start, pos, ref, alt)
        if err:
            skipped.append((pos, f"exon_start: {err}")); continue
        ref_end, alt_end, err = build_ref_alt_pair(chrom, ex_end, pos, ref, alt)
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

    fasta.close()
    n_mfass = len(mfass_meta["chrom"])
    print(f"built sequences for {n_mfass:,} variants ({len(skipped):,} skipped)")

    print(f"one-hot encoding {n_mfass:,} variants...")
    mfass_encoded = {}
    for key, seq_list in mfass_seqs.items():
        arr = np.zeros((n_mfass, SEQ_LEN, 4), dtype=np.float32)
        for i, seq in enumerate(seq_list):
            arr[i] = one_hot(seq)
        mfass_encoded[key] = arr
        print(f"  {key}: {arr.shape}")

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

fasta.close()


model_colors = {
    "pangolin":          "#009E73",  # green
    "pangolin_v2":       "#2CA02C",  # darker green (v2, human-finetuned)
    "spliceai":          "#D55E00",  # orange-red
    "splicetransformer": "#E69F00",  # yellow-orange
    "sphaec_ref":        "#56B4E9",  # light blue
    "sphaec_var":        "#0072B2",  # dark blue
    "sphaec_avg":        "#CC79A7",  # pink
    "gencode":           "#666666",  # gray (baseline)
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

ANNOT_SIZE = 12

fig3_main = Path("../../figures/fig3/main")
fig3_sup = Path("../../figures/fig3/sup")
fig3_main.mkdir(parents=True, exist_ok=True)
fig3_sup.mkdir(parents=True, exist_ok=True)

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

out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

models = ["pangolin", "pangolin_v2", "spliceai", "splicetransformer", "sphaec_ref", "sphaec_var"]

model_names = {
    "pangolin": "Pangolin",
    "pangolin_usage": "Pangolin (max usage)",
    "pangolin_v2": "Pangolin v2",
    "pangolin_v2_usage": "Pangolin v2 (max usage)",
    "spliceai": "SpliceAI",
    "splicetransformer": "SpliceTransformer",
    "splicetransformer_usage": "SpTransformer (max usage)",
    "sphaec_ref": "SPLAIRE",
    "sphaec_var": "SPLAIRE-var",
}

datasets = {
    "vexseq": {"path": "vex_seq/data", "prefix": "vex_seq", "thr": 0.10, "psi_scale": 100},
    "mfass": {"path": "mfass/data", "prefix": "mfass", "thr": 0.50, "psi_scale": 1}
}

dataset_names = {"vexseq": "Vex-seq", "mfass": "MFASS"}


def load_scores_as_deltas(path):
    """load h5 and compute deltas (alt - ref) for each head"""
    with h5py.File(path, "r") as f:
        # get meta
        meta = {}
        for k in f["meta"].keys():
            arr = f["meta"][k][:]
            meta[k] = np.array([x.decode() for x in arr]) if arr.dtype.kind == "S" else arr

        # find unique heads and compute deltas
        keys = list(f["scores"].keys())
        heads = set()
        for k in keys:
            for suffix in ["_exon_start_ref", "_exon_start_alt", "_exon_end_ref", "_exon_end_alt"]:
                if k.endswith(suffix):
                    heads.add(k.replace(suffix, ""))

        deltas = {}
        for head in sorted(heads):
            start_ref = f["scores"][f"{head}_exon_start_ref"][:]
            start_alt = f["scores"][f"{head}_exon_start_alt"][:]
            deltas[f"{head}_exon_start_delta"] = start_alt - start_ref

            end_ref = f["scores"][f"{head}_exon_end_ref"][:]
            end_alt = f["scores"][f"{head}_exon_end_alt"][:]
            deltas[f"{head}_exon_end_delta"] = end_alt - end_ref

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


# load all data
dfs = {}
for name, cfg in datasets.items():
    meta_df = None
    delta_dfs = {}

    for m in models:
        if m == "pangolin": fname = f"{cfg['prefix']}_pang.h5"
        elif m == "pangolin_v2": fname = f"{cfg['prefix']}_pang_v2.h5"
        elif m == "spliceai": fname = f"{cfg['prefix']}_sa.h5"
        elif m == "splicetransformer": fname = f"{cfg['prefix']}_spt.h5"
        else: fname = f"{cfg['prefix']}_{m}.h5"

        path = Path(cfg["path"]) / fname
        if not path.exists():
            continue

        meta, deltas = load_scores_as_deltas(path)
        deltas.columns = [f"{m}_{c}" for c in deltas.columns]
        delta_dfs[m] = deltas

        if meta_df is None:
            meta_df = meta

    # filter mfass to variants with labels
    # old h5s have 28972 rows (pre-filter), new ones have 27733 (already filtered)
    if name == "mfass":
        mask = meta_df["delta_psi"].notna().values
        n_raw = len(meta_df)
        meta_df = meta_df[mask].reset_index(drop=True)
        n_filt = len(meta_df)
        filtered = {}
        for m, mdf in delta_dfs.items():
            if len(mdf) == n_raw:
                filtered[m] = mdf[mask].reset_index(drop=True)
            elif len(mdf) == n_filt:
                filtered[m] = mdf.reset_index(drop=True)
            else:
                print(f"  WARNING: {m} has {len(mdf)} rows, expected {n_raw} or {n_filt}, skipping")
                continue
        delta_dfs = filtered

    df = meta_df.copy()
    df["y"] = df["delta_psi"].values / cfg["psi_scale"]

    # 1-SD threshold (primary) — defines classification label and correlation filter
    y_clean = df["y"].values[np.isfinite(df["y"].values)]
    sd_val = y_clean.std()
    df["label"] = (np.abs(df["y"]) > sd_val).astype(int)

    # fixed threshold (secondary) — Chong 0.50 for MFASS, 0.10 for Vex-seq
    df["label_fixed"] = (np.abs(df["y"]) > cfg["thr"]).astype(int)

    # store both thresholds for downstream use
    cfg["thr_sd"] = float(sd_val)

    for m, delta_df in delta_dfs.items():
        df = pd.concat([df, delta_df], axis=1)

    dfs[name] = df

    n_models = len(delta_dfs)
    delta_cols = [c for c in df.columns if "_delta" in c]
    n_pos = df["label"].sum()
    n_pos_fixed = df["label_fixed"].sum()
    print(f"{name}: loaded {n_models} models, {len(df):,} variants")
    print(f"  scores: {len(delta_cols)} delta columns, shape ({len(df):,}, {len(delta_cols)})")
    print(f"  labels (1-SD): {n_pos:,} positives ({100*df['label'].mean():.1f}%), threshold={sd_val:.3f}")
    print(f"  labels (fixed): {n_pos_fixed:,} positives ({100*df['label_fixed'].mean():.1f}%), threshold={cfg['thr']:.2f}")


# compute combined delta columns
for name, df in dfs.items():
    is_plus = (df["strand"].values == "+")

    def strand_select(start_col, end_col):
        return np.where(is_plus, df[start_col], df[end_col])

    def avg_boundaries(prefix):
        return (df[f"{prefix}_exon_start_delta"] + df[f"{prefix}_exon_end_delta"]) / 2

    # spliceai: avg of strand-aware acceptor + donor
    acc = strand_select("spliceai_acceptor_exon_start_delta", "spliceai_acceptor_exon_end_delta")
    don = strand_select("spliceai_donor_exon_end_delta", "spliceai_donor_exon_start_delta")
    df["spliceai_cls_delta"] = (acc + don) / 2

    # splicetransformer: cls + tissue usage
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

    # pangolin + pangolin_v2: tissue p_splice + usage
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

    # sphaec: cls + reg_ssu
    for v in ["ref", "var"]:
        p = f"sphaec_{v}"
        acc = strand_select(f"{p}_cls_acceptor_exon_start_delta", f"{p}_cls_acceptor_exon_end_delta")
        don = strand_select(f"{p}_cls_donor_exon_end_delta", f"{p}_cls_donor_exon_start_delta")
        df[f"{p}_cls_delta"] = (acc + don) / 2
        df[f"{p}_reg_ssu_delta"] = avg_boundaries(f"{p}_reg_ssu")

    dfs[name] = df

    n_raw = len([c for c in df.columns if "_delta" in c and "exon_start" in c])
    n_combined = len([c for c in df.columns if c.endswith("_delta") and "exon_" not in c])
    print(f"{name}: {n_combined} combined columns ({n_raw} raw heads → strand-corrected + cross-tissue max)")


# chong 2019 / mount et al 2019 variant location categories
# splice_site = canonical 2 bp intronic at each junction (chong "splice site")
# splice_region = 3 exonic + 8 intronic nt, excluding splice_site (chong "splice region")
# exon = remaining exonic positions
# intron = remaining intronic positions

def get_location_masks(pos, exon_start, exon_end):
    """returns dict of boolean masks — combined and acc/don split for each category

    combined keys: all, exon, intron, splice_site, splice_region
    acc/don keys:  exon_acc, exon_don, intron_acc, intron_don,
                   splice_site_acc, splice_site_don, splice_region_acc, splice_region_don
    """
    in_exon = (pos >= exon_start) & (pos <= exon_end)

    # mount broad SS: 3 exonic + 8 intronic at each boundary
    ss_3p = (pos >= exon_start - 8) & (pos <= exon_start + 2)  # acceptor side
    ss_5p = (pos >= exon_end - 2) & (pos <= exon_end + 8)      # donor side
    ss_broad = ss_3p | ss_5p

    # chong canonical SS: 2bp intronic at each boundary
    ss_can_3p = (pos >= exon_start - 2) & (pos <= exon_start - 1)  # acceptor
    ss_can_5p = (pos >= exon_end + 1) & (pos <= exon_end + 2)      # donor
    splice_site = ss_can_3p | ss_can_5p

    # splice region = broad minus canonical
    splice_region = ss_broad & ~splice_site
    splice_region_3p = ss_3p & ~ss_can_3p   # acceptor-side region
    splice_region_5p = ss_5p & ~ss_can_5p   # donor-side region

    # exon and intron (excluding broad SS)
    exon = in_exon & ~ss_broad
    intron = ~in_exon & ~ss_broad

    # acc/don split for exon and intron: assign by nearest boundary
    closer_to_acc = np.abs(pos - exon_start) <= np.abs(pos - exon_end)

    return {
        # combined
        "all": np.ones(len(pos), dtype=bool),
        "exon": exon,
        "intron": intron,
        "splice_site": splice_site,
        "splice_region": splice_region,
        # acc/don split
        "splice_site_acc": ss_can_3p,
        "splice_site_don": ss_can_5p,
        "splice_region_acc": splice_region_3p,
        "splice_region_don": splice_region_5p,
        "exon_acc": exon & closer_to_acc,
        "exon_don": exon & ~closer_to_acc,
        "intron_acc": intron & (pos < exon_start),
        "intron_don": intron & (pos > exon_end),
    }

# combined subsets (used by PR curves, bar charts, bootstrap)
LOCATION_SUBSETS = ["all", "exon", "intron", "splice_region", "splice_site"]
LOCATION_LABELS = {
    "all": "All", "exon": "Exon", "intron": "Intron",
    "splice_site": "Splice site", "splice_region": "Splice region",
}

# acc/don subsets (available for finer analysis)
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

# everything merged for convenience
ALL_LOCATION_LABELS = {**LOCATION_LABELS, **LOCATION_LABELS_AD}

# data overview plots — 2x2: top = ΔPSI + location, bottom = distance + exon width
for name, df in dfs.items():
    y = df["y"].values
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    loc = get_location_masks(pos, exon_start, exon_end)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # panel a: ΔPSI histogram
    ax = axes[0, 0]
    ax.hist(y[np.isfinite(y)], bins=60, edgecolor="white", linewidth=0.5, color="#4a90d9")
    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7)
    sd = np.nanstd(y)
    ax.axvline(sd, color="black", ls=":", lw=1, alpha=0.6)
    ax.axvline(-sd, color="black", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Measured \u0394PSI")
    ax.set_ylabel("Count")
    ax.set_title("(a) ΔPSI distribution")
    stats_txt = f"mean={np.nanmean(y):.3f}\nSD={sd:.3f}\nn={np.isfinite(y).sum():,}"
    ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # panel b: variants by location category
    ax = axes[0, 1]
    loc_counts = {LOCATION_LABELS[k]: loc[k].sum() for k in LOCATION_SUBSETS if k != "all"}
    bars = ax.bar(loc_counts.keys(), loc_counts.values(), color=["#4a90d9", "#7fc97f", "#e7298a"],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, loc_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + len(df) * 0.01,
                f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("(b) Variants by location")

    # panel c: distance to nearest splice site
    ax = axes[1, 0]
    dist_start = np.abs(pos - exon_start)
    dist_end = np.abs(pos - exon_end)
    dist_nearest = np.minimum(dist_start, dist_end)
    ax.hist(dist_nearest, bins=50, edgecolor="white", linewidth=0.5, color="#4a90d9")
    ax.set_xlabel("Distance to nearest splice site (bp)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Distance to splice site")
    ax.text(0.97, 0.95, f"median={np.median(dist_nearest):.0f} bp\nmax={dist_nearest.max():,} bp",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # panel d: exon width distribution (unique exons only)
    ax = axes[1, 1]
    exon_widths = exon_end - exon_start + 1
    unique_widths = pd.Series(exon_widths).groupby(
        [df["chrom"].values, exon_start, exon_end]).first().values
    ax.hist(unique_widths, bins=50, edgecolor="white", linewidth=0.5, color="#4a90d9")
    ax.set_xlabel("Exon width (bp)")
    ax.set_ylabel("Count (unique exons)")
    ax.set_title("(d) Exon width distribution")
    ax.text(0.97, 0.95, f"median={np.median(unique_widths):.0f} bp\nn={len(unique_widths):,} exons",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle(f"{dataset_names[name]} — Data Overview ($n$ = {len(df):,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_data_overview.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{name}_data_overview.pdf", bbox_inches="tight")
    plt.show()


def get_metrics(y, pred, label):
    mask = np.isfinite(y) & np.isfinite(pred)
    yt, yp, lab = y[mask], pred[mask], label[mask]
    if len(yt) < 3: return np.nan, np.nan, np.nan
    r = stats.pearsonr(yt, yp)[0]
    rho = stats.spearmanr(yt, yp)[0]
    auprc = average_precision_score(lab, np.abs(yp)) if lab.sum() > 0 else np.nan
    return r, rho, auprc


from scipy.stats import spearmanr

# shared model config — single source of truth
# max-aggregated output column per model (used across all analyses)
MODEL_COLS = {
    "pangolin": "pangolin_max_p_splice",
    "pangolin_v2": "pangolin_v2_max_p_splice",
    "spliceai": "spliceai_cls_delta",
    "splicetransformer": "splicetransformer_cls_delta",
    "sphaec_ref": "sphaec_ref_cls_delta",
    "sphaec_var": "sphaec_var_cls_delta",
}
MODEL_LIST = list(MODEL_COLS.keys())

split_cols = MODEL_COLS
split_models = MODEL_LIST


pr_cols = {
    "vexseq": {
        "pangolin": "pangolin_testis_p_splice_delta",
        "pangolin_v2": "pangolin_v2_testis_p_splice_delta",
        "spliceai": "spliceai_cls_delta",
        "splicetransformer": "splicetransformer_usage_Blood_Vessel_delta",
        "sphaec_ref": "sphaec_ref_cls_delta",
        "sphaec_var": "sphaec_var_cls_delta",
    },
    "mfass": {
        "pangolin": "pangolin_testis_p_splice_delta",
        "pangolin_v2": "pangolin_v2_testis_p_splice_delta",
        "spliceai": "spliceai_cls_delta",
        "splicetransformer": "splicetransformer_usage_Lung_delta",
        "sphaec_ref": "sphaec_ref_cls_delta",
        "sphaec_var": "sphaec_var_cls_delta",
    },
}

for name, df in dfs.items():
    cols = pr_cols[name]
    pos = df["pos"].values
    loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes_flat = axes.flatten()

    for ax, loc_name in zip(axes_flat, LOCATION_SUBSETS):
        mask = loc[loc_name]
        label = df["label"].values[mask]
        n_pos, n_neg = label.sum(), (1 - label).sum()

        for m in models:
            if m not in cols or cols[m] not in df.columns:
                continue
            delta = df[cols[m]].values[mask]
            if label.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(label, np.abs(delta))
            auprc = average_precision_score(label, np.abs(delta))
            ax.plot(rec, prec, color=get_color(m), lw=2, label=f"{model_names[m]} ({auprc:.3f})")

        baseline = label.mean() if len(label) > 0 else 0
        ax.axhline(baseline, color="#888888", ls="--", lw=1, label=f"baseline ({baseline:.3f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{LOCATION_LABELS.get(loc_name, loc_name)} (n={mask.sum():,}, pos={n_pos:,})")
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(alpha=0.3)

    plt.suptitle(f"{dataset_names[name]} — Precision-Recall", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_pr.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{name}_pr.pdf", bbox_inches="tight")
    plt.show()


thresh_cols = {
    "vexseq": {
        "pangolin": "pangolin_testis_p_splice_delta",
        "pangolin_v2": "pangolin_v2_testis_p_splice_delta",
        "spliceai": "spliceai_cls_delta",
        "splicetransformer": "splicetransformer_usage_Blood_Vessel_delta",
        "sphaec_ref": "sphaec_ref_cls_delta",
        "sphaec_var": "sphaec_var_cls_delta",
    },
    "mfass": {
        "pangolin": "pangolin_testis_p_splice_delta",
        "pangolin_v2": "pangolin_v2_testis_p_splice_delta",
        "spliceai": "spliceai_cls_delta",
        "splicetransformer": "splicetransformer_usage_Lung_delta",
        "sphaec_ref": "sphaec_ref_cls_delta",
        "sphaec_var": "sphaec_var_cls_delta",
    },
}

for name, df in dfs.items():
    cols = thresh_cols[name]
    thresholds = np.arange(0.01, 1.01, 0.01)
    y = df["y"].values
    sd_thr = np.std(y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for m in models:
        if cols[m] not in df.columns: continue
        delta = df[cols[m]].values
        auprcs = []
        for thr in thresholds:
            lab = (np.abs(y) > thr).astype(int)
            auprcs.append(average_precision_score(lab, np.abs(delta)) if lab.sum() > 0 else np.nan)
        ax.plot(thresholds, auprcs, lw=2, color=get_color(m), label=model_names[m])
    ax.axvline(sd_thr, color="black", ls=":", lw=1.5, alpha=0.7)
    ax.text(sd_thr + 0.003, 0.95, f"1 SD = {sd_thr:.3f}", fontsize=8, va="top", rotation=90, alpha=0.7)
    ax.set_xlabel("|\u2206PSI| threshold")
    ax.set_ylabel("auprc")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("auprc vs threshold")

    ax2 = axes[1]
    n_pos = [(np.abs(y) > thr).sum() for thr in thresholds]
    n_neg = [(np.abs(y) <= thr).sum() for thr in thresholds]
    ax2.plot(thresholds, n_pos, lw=2, color="#D55E00", label="positives")
    ax2.plot(thresholds, n_neg, lw=2, color="#0072B2", label="negatives")
    ax2.axvline(sd_thr, color="black", ls=":", lw=1.5, alpha=0.7)
    ax2.text(sd_thr + 0.003, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 1, f"1 SD", fontsize=8, va="top", rotation=90, alpha=0.7)
    ax2.set_xlabel("|\u2206PSI| threshold")
    ax2.set_ylabel("count")
    ax2.set_xlim(thresholds[0], thresholds[-1])
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_title("class distribution")

    plt.suptitle(f"{dataset_names[name]} threshold sensitivity (n={len(df)})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_threshold.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{name}_threshold.pdf", bbox_inches="tight")
    plt.show()


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
    "sphaec_ref_cls_delta", "sphaec_ref_reg_ssu_delta",
    "sphaec_var_cls_delta", "sphaec_var_reg_ssu_delta",
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

# map raw column names to display names
def col_display(c):
    """pretty name for delta column"""
    # strip _delta suffix, re-add as title case at end
    has_delta = c.endswith('_delta')
    base = c.replace('_delta', '') if has_delta else c
    suffix = ' Delta' if has_delta else ''
    # SPLAIRE
    if base == 'sphaec_ref_cls': return f'SPLAIRE CLS{suffix}'
    if base == 'sphaec_ref_reg_ssu': return f'SPLAIRE SSU{suffix}'
    if base == 'sphaec_var_cls': return f'SPLAIRE-var CLS{suffix}'
    if base == 'sphaec_var_reg_ssu': return f'SPLAIRE-var SSU{suffix}'
    # SpliceAI
    if base == 'spliceai_cls': return f'SpliceAI CLS{suffix}'
    # SpliceTransformer
    if base == 'splicetransformer_cls': return f'SpliceTransformer CLS{suffix}'
    if base == 'splicetransformer_max_usage': return f'SpliceTransformer Max Usage{suffix}'
    if base.startswith('splicetransformer_usage_'):
        tissue = base.replace('splicetransformer_usage_', '').replace('_', ' ')
        return f'SpliceTransformer {tissue}{suffix}'
    # Pangolin / Pangolin v2
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


for name, df in dfs.items():
    y, label = df["y"].values, df["label"].values
    n_pos, n_neg = label.sum(), (1 - label).sum()
    
    # compute metrics
    results = []
    for col in combined_cols:
        if col not in df.columns:
            continue
        r, rho, auprc = get_metrics(y, df[col].values, label)
        model = col.split("_")[0]
        if "sphaec" in col:
            model = "sphaec_ref" if "ref" in col else "sphaec_var"
        results.append({"col": col, "model": model, "pearson": r, "spearman": rho, "auprc": auprc})
    
    results_df = pd.DataFrame(results)
    
    # print results table
    print(f"\n{'='*80}")
    print(f"{dataset_names[name]} (n={len(df)}, pos={n_pos}, neg={n_neg})")
    print(f"{'='*80}")
    print(f"{'column':<45} {'pearson':>10} {'spearman':>10} {'auprc':>10} ")
    print("-" * 80)
    for _, row in results_df.sort_values("auprc", ascending=False).iterrows():
        print(f"{col_display(row['col']):<45} {row['pearson']:>10.3f} {row['spearman']:>10.3f} {row['auprc']:>10.3f}")

    # save metrics to csv (1-SD threshold)
    results_df["display_name"] = results_df["col"].apply(col_display)
    results_df["threshold"] = "1-SD"
    results_df.sort_values("auprc", ascending=False).to_csv(f"{out_dir}/{name}_metrics.csv", index=False)
    print(f"  saved {out_dir}/{name}_metrics.csv")

    # secondary metrics with fixed threshold (0.10 for vex-seq, 0.50 for mfass)
    label_fixed = df["label_fixed"].values
    results_fixed = []
    for col in combined_cols:
        if col not in df.columns:
            continue
        r_f, rho_f, auprc_f = get_metrics(y, df[col].values, label_fixed)
        model = col.split("_")[0]
        if "sphaec" in col:
            model = "_".join(col.split("_")[:2])
        results_fixed.append({"col": col, "model": model, "pearson": r_f, "spearman": rho_f, "auprc": auprc_f})
    results_fixed_df = pd.DataFrame(results_fixed)
    results_fixed_df["display_name"] = results_fixed_df["col"].apply(col_display)
    results_fixed_df["threshold"] = "fixed"
    results_fixed_df.sort_values("auprc", ascending=False).to_csv(
        f"{out_dir}/{name}_metrics_fixed_thr.csv", index=False)

    # print comparison for MODEL_COLS outputs
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

    # metrics by region (combined + acc/don split) for MODEL_COLS outputs
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

    # print summary table
    print(f"\n  metrics by region ({dataset_names[name]}):")
    print(f"  {'region':<18} {'n':>7} {'SDV':>6} {'model':<20} {'Pearson':>8} {'Spearman':>9} {'AUPRC':>8}")
    print(f"  {'-'*80}")
    for rk in _all_regions:
        sub = region_df[region_df["region"] == rk]
        if sub.empty:
            continue
        # show best model by AUPRC
        best = sub.loc[sub["auprc"].idxmax()] if sub["auprc"].notna().any() else sub.iloc[0]
        print(f"  {ALL_LOCATION_LABELS.get(rk, rk):<18} {int(best['n']):>7,} {int(best['n_sdv']):>6} "
              f"{best['model_label']:<20} {best['pearson']:>8.3f} {best['spearman']:>9.3f} {best['auprc']:>8.3f}")
    print(f"  saved {out_dir}/{name}_metrics_by_region.csv")

    # 1x3 combined figure (shared y-axis) — all model outputs ranked by AUPRC
    _metrics_list = ["pearson", "spearman", "auprc"]
    # sort by auprc for consistent row order
    sorted_df = results_df.sort_values("auprc", ascending=True).reset_index(drop=True)
    colors_bar = [get_color(row["model"]) for _, row in sorted_df.iterrows()]
    y_pos = np.arange(len(sorted_df))
    n_bars = len(sorted_df)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, n_bars * 0.35)), sharey=True)

    for ai, metric in enumerate(_metrics_list):
        ax = axes[ai]
        ax.barh(y_pos, sorted_df[metric], color=colors_bar, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(metric.capitalize(), fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(sorted_df[metric]):
            if np.isfinite(v):
                ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
        if ai == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([col_display(c) for c in sorted_df["col"]], fontsize=9)
        ax.set_title(metric.capitalize(), fontsize=13)

    plt.suptitle(f"{dataset_names[name]} — All Model Outputs", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_all_outputs.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_all_outputs.pdf", bbox_inches="tight")
    plt.show()


# (supplemental PR curves removed — covered by main 2x2 PR grid above)

# placeholder to keep variable scope for downstream code
pr_cols_sup = {
    "pangolin": "pangolin_max_p_splice",
    "pangolin_v2": "pangolin_v2_max_p_splice",
    "pangolin_usage": "pangolin_max_usage",
    "pangolin_v2_usage": "pangolin_v2_max_usage",
    "spliceai": "spliceai_cls_delta",
    "splicetransformer": "splicetransformer_cls_delta",
    "splicetransformer_usage": "splicetransformer_max_usage",
    "sphaec_ref": "sphaec_ref_cls_delta",
    "sphaec_var": "sphaec_var_cls_delta",
}

# (supplemental PR curves removed — covered by main 2x2 PR grid)


# prediction error vs position relative to nearest splice site
# x-axis: acceptor (left panel) and donor (right panel) with exon schematic
# each variant assigned to whichever splice site it is closest to
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

_err_cols = MODEL_COLS
_err_models = ["spliceai", "sphaec_ref"]

def _add_exon_schematic(fig, ax_acc, ax_don, y_fig, rect_h=0.012, line_lw=1.0):
    """split exon schematic with // break between acceptor and donor panels"""
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
    # left intron line
    fig.add_artist(Line2D([xL, xEL], [y_fig, y_fig],
        transform=fig.transFigure, color="black", lw=line_lw, clip_on=False, zorder=10))
    # left exon (acceptor side: pos 0 to x-max)
    fig.add_artist(Rectangle((xEL, y_fig - rect_h/2), xELR - xEL, rect_h,
        transform=fig.transFigure, facecolor="black", edgecolor="black",
        clip_on=False, zorder=10))
    # break: // in the gap
    _gcx = (xELR + xERL) / 2
    _sw = 0.005; _sh = rect_h * 2.5
    for _dx in [-0.004, 0.004]:
        fig.add_artist(Line2D(
            [_gcx + _dx - _sw/2, _gcx + _dx + _sw/2],
            [y_fig - _sh/2, y_fig + _sh/2],
            transform=fig.transFigure, color="black", lw=line_lw,
            clip_on=False, zorder=10))
    # right exon (donor side: x-min to pos 0)
    fig.add_artist(Rectangle((xERL, y_fig - rect_h/2), xER - xERL, rect_h,
        transform=fig.transFigure, facecolor="black", edgecolor="black",
        clip_on=False, zorder=10))
    # right intron line
    fig.add_artist(Line2D([xER, xR], [y_fig, y_fig],
        transform=fig.transFigure, color="black", lw=line_lw, clip_on=False, zorder=10))

# measured dpsi vs position (scatter, same layout as error-by-position)
for name, df in dfs.items():
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values

    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    x_acc = dist_to_acc[is_acc]
    y_acc = y_true[is_acc]
    x_don = -(exon_end[~is_acc] - pos[~is_acc])
    y_don = y_true[~is_acc]

    fin_acc = np.isfinite(y_acc)
    fin_don = np.isfinite(y_don)

    fig, (ax_acc, ax_don) = plt.subplots(1, 2, figsize=(10, 4),
        gridspec_kw={"wspace": 0.15})

    ax_acc.scatter(x_acc[fin_acc], y_acc[fin_acc], s=8, alpha=0.4,
                   color="#555555", edgecolors="none")
    ax_acc.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_acc.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_acc.set_ylabel("Measured \u0394PSI")
    ax_acc.set_xlabel("Position relative to acceptor", labelpad=14)
    ax_acc.set_title("Acceptor")
    ax_acc.text(0.02, 0.98, f"n={fin_acc.sum():,}", transform=ax_acc.transAxes, va="top", fontsize=9)

    ax_don.scatter(x_don[fin_don], y_don[fin_don], s=8, alpha=0.4,
                   color="#555555", edgecolors="none")
    ax_don.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_don.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_don.set_xlabel("Position relative to donor", labelpad=14)
    ax_don.set_title("Donor")
    ax_don.text(0.02, 0.98, f"n={fin_don.sum():,}", transform=ax_don.transAxes, va="top", fontsize=9)

    all_y = np.concatenate([y_acc[fin_acc], y_don[fin_don]])
    ylim = max(np.abs(np.percentile(all_y, [1, 99]))) * 1.1
    ax_acc.set_ylim(-ylim, ylim)
    ax_don.set_ylim(-ylim, ylim)

    plt.suptitle(f"{dataset_names[name]} — Measured \u0394PSI by position", fontsize=13)
    plt.subplots_adjust(bottom=0.22, wspace=0.15)

    bot = min(ax_acc.get_position().y0, ax_don.get_position().y0)
    _add_exon_schematic(fig, ax_acc, ax_don, bot - 8/72/fig.get_figheight())

    plt.savefig(f"{fig3_sup}/{name}_dpsi_by_position.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_dpsi_by_position.pdf", bbox_inches="tight")
    plt.show()

for m in _err_models:
    col = _err_cols[m]
    for name, df in dfs.items():
        if col not in df.columns:
            continue
        pos = df["pos"].values.astype(int)
        exon_start = df["exon_start"].values.astype(int)
        exon_end = df["exon_end"].values.astype(int)
        y_true = df["y"].values
        y_pred = df[col].values
        error = y_pred - y_true

        # signed distance: negative = intronic, positive = exonic
        # relative to exon_start (acceptor) or exon_end (donor)
        dist_to_acc = pos - exon_start  # neg = upstream intron, pos = into exon
        dist_to_don = exon_end - pos    # neg = downstream intron (we flip sign below)

        # assign each variant to nearest splice site
        abs_acc = np.abs(dist_to_acc)
        abs_don = np.abs(dist_to_don)
        is_acc = abs_acc <= abs_don

        # acceptor: x = dist_to_acc (neg = intron, pos = exon)
        x_acc = dist_to_acc[is_acc]
        err_acc = error[is_acc]
        # donor: x = -dist_to_don (neg = exon interior, pos = intron)
        # flip so: negative = exon side, positive = intron side
        x_don = -(exon_end[~is_acc] - pos[~is_acc])
        err_don = error[~is_acc]

        fin_acc = np.isfinite(err_acc)
        fin_don = np.isfinite(err_don)

        fig, (ax_acc, ax_don) = plt.subplots(1, 2, figsize=(10, 4),
            gridspec_kw={"wspace": 0.15})

        ax_acc.scatter(x_acc[fin_acc], err_acc[fin_acc], s=8, alpha=0.4,
                       color=get_color(m), edgecolors="none")
        ax_acc.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax_acc.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax_acc.set_ylabel("Prediction error (pred - measured)")
        ax_acc.set_xlabel("Position relative to acceptor", labelpad=14)
        ax_acc.set_title("Acceptor")
        n_acc = fin_acc.sum()
        ax_acc.text(0.02, 0.98, f"n={n_acc:,}", transform=ax_acc.transAxes,
                    va="top", fontsize=9)

        ax_don.scatter(x_don[fin_don], err_don[fin_don], s=8, alpha=0.4,
                       color=get_color(m), edgecolors="none")
        ax_don.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax_don.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax_don.set_xlabel("Position relative to donor", labelpad=14)
        ax_don.set_title("Donor")
        n_don = fin_don.sum()
        ax_don.text(0.02, 0.98, f"n={n_don:,}", transform=ax_don.transAxes,
                    va="top", fontsize=9)

        # match y-limits
        all_err = np.concatenate([err_acc[fin_acc], err_don[fin_don]])
        ylim = max(np.abs(np.percentile(all_err, [1, 99]))) * 1.1
        ax_acc.set_ylim(-ylim, ylim)
        ax_don.set_ylim(-ylim, ylim)

        plt.suptitle(f"{dataset_names[name]} — {model_names[m]}", fontsize=13)
        plt.subplots_adjust(bottom=0.22, wspace=0.15)

        # exon schematic
        bot = min(ax_acc.get_position().y0, ax_don.get_position().y0)
        _add_exon_schematic(fig, ax_acc, ax_don, bot - 8/72/fig.get_figheight())

        plt.savefig(f"{fig3_sup}/{name}_{m}_error_by_position.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig3_sup}/{name}_{m}_error_by_position.pdf", bbox_inches="tight")
        plt.show()
        print(f"{dataset_names[name]} — {model_names[m]}: "
              f"{n_acc:,} acceptor, {n_don:,} donor variants")


# measured effect size by position relative to nearest splice site
# mean and median |ΔPSI| at each base, acceptor/donor panels

for name, df in dfs.items():
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values

    # assign to acceptor or donor
    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bin_w = 1

    for ax, (side, smask, dvals) in zip(axes, [
        ("Acceptor", is_acc, dist_to_acc[is_acc]),
        ("Donor", ~is_acc, -(exon_end[~is_acc] - pos[~is_acc])),
    ]):
        ys = np.abs(y_true[smask])
        fin = np.isfinite(ys)
        d, y = dvals[fin], ys[fin]
        bins = np.arange(d.min() - bin_w/2, d.max() + bin_w, bin_w)
        bi = np.digitize(d, bins)
        cx, cy_mean, cy_med = [], [], []
        for b in range(1, len(bins)):
            bm = bi == b
            if bm.sum() >= 5:
                cx.append((bins[b-1] + bins[b]) / 2)
                cy_mean.append(np.mean(y[bm]))
                cy_med.append(np.median(y[bm]))

        ax.plot(cx, cy_mean, color="#d62728", lw=1.5, alpha=0.8, label="Mean |ΔPSI|")
        ax.plot(cx, cy_med, color="#1f77b4", lw=1.5, alpha=0.8, ls="--", label="Median |ΔPSI|")
        ax.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"Position relative to {side.lower()}")
        ax.set_title(side)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Measured |ΔPSI|")
    axes[0].legend(fontsize=8)
    plt.suptitle(f"{dataset_names[name]} — Effect size by position", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_measured_psi_by_position.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_measured_psi_by_position.pdf", bbox_inches="tight")
    plt.show()
    n_acc, n_don = is_acc.sum(), (~is_acc).sum()
    print(f"{dataset_names[name]}: {n_acc:,} acceptor, {n_don:,} donor variants")


# rigorous effect size by position — lowess smoothing, exon-normalized z-scores, 95% CI
# addresses three confounds in the binned version above:
# 1) exon identity (z-score removes per-exon baseline)
# 2) unequal n per position (lowess adapts bandwidth)
# 3) missing uncertainty (bootstrap CI bands)

from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess

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


for name, df in dfs.items():
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values

    # per-exon z-score — removes exon-level baseline inclusion
    exon_key = [f"{es}_{ee}" for es, ee in zip(exon_start, exon_end)]
    df["_exon_key_tmp"] = exon_key
    exon_std = df.groupby("_exon_key_tmp")["y"].transform("std")
    exon_mean = df.groupby("_exon_key_tmp")["y"].transform("mean")
    z_score = np.where(exon_std > 0, (y_true - exon_mean) / exon_std, 0.0)
    df.drop(columns=["_exon_key_tmp"], inplace=True)

    # assign to nearest splice site
    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"hspace": 0.45, "wspace": 0.2})

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
            d, y = dvals[fin].astype(float), ys[fin]

            # scatter (subsample for vex-seq readability)
            n_pts = len(d)
            if n_pts > 2000:
                sub = np.random.default_rng(0).choice(n_pts, 2000, replace=False)
            else:
                sub = np.arange(n_pts)
            ax.scatter(d[sub], y[sub], s=4, alpha=0.15, color="#888888", edgecolors="none",
                       rasterized=True, zorder=1)

            # lowess + bootstrap CI
            x_grid, y_smooth, lo, hi = _bootstrap_lowess(d, y, frac=0.15)
            ax.plot(x_grid, y_smooth, color="#d62728", lw=1.8, zorder=3, label="LOWESS")
            ax.fill_between(x_grid, lo, hi, color="#d62728", alpha=0.15, zorder=2, label="95% CI")

            ax.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
            if row == 0:
                ax.set_title(side, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{title_tag}\n{y_label}")
            if row == 1:
                ax.set_xlabel(f"Position relative to {side.lower()}")
            ax.grid(alpha=0.15)
            ax.text(0.02, 0.95, f"n={n_pts:,}", transform=ax.transAxes, va="top", fontsize=8)

    axes[0, 1].legend(fontsize=8, loc="upper right")
    plt.suptitle(f"{dataset_names[name]} — Effect size by position (LOWESS + exon normalization)", fontsize=12)
    plt.savefig(f"{fig3_sup}/{name}_effect_by_position_rigorous.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_effect_by_position_rigorous.pdf", bbox_inches="tight")
    plt.show()
    print(f"{dataset_names[name]}: rigorous position plot saved")


# %SDV by normalized position along exon + flanking intron (cf. Chong 2019 Fig 4C)
# positions are normalized so all exons map to the same coordinate space:
# intron positions stay as raw bp offset, exon positions are rescaled to [0, 1]

for name, df in dfs.items():
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y = df["y"].values
    label = df["label"].values
    thr = datasets[name]["thr"]

    # normalized position: intron = raw offset (negative upstream, positive downstream)
    # exon = fractional position [0, 1] where 0 = acceptor edge, 1 = donor edge
    exon_width = (exon_end - exon_start).astype(float)
    in_exon = (pos >= exon_start) & (pos <= exon_end)
    in_upstream = pos < exon_start
    in_downstream = pos > exon_end

    # raw distances for intron bins
    upstream_dist = pos - exon_start       # negative values
    downstream_dist = pos - exon_end       # positive values
    exon_frac = np.where(exon_width > 0,
                         (pos - exon_start) / exon_width, 0.5)

    # bin intron positions at 1-nt resolution, exon at ~2% bins (50 bins)
    n_exon_bins = 50
    exon_edges = np.linspace(0, 1, n_exon_bins + 1)

    # upstream intron bins (furthest intron on left → acceptor at 0)
    up_dists = upstream_dist[in_upstream]
    up_labels = label[in_upstream]
    up_min, up_max = int(up_dists.min()), int(up_dists.max())
    up_bins = np.arange(up_min - 0.5, up_max + 1.5, 1)

    # downstream intron bins
    dn_dists = downstream_dist[in_downstream]
    dn_labels = label[in_downstream]
    dn_min, dn_max = int(dn_dists.min()), int(dn_dists.max())
    dn_bins = np.arange(dn_min - 0.5, dn_max + 1.5, 1)

    # exon bins
    ex_fracs = exon_frac[in_exon]
    ex_labels = label[in_exon]

    # --- binning helpers ---

    def pct_sdv(values, labels, bin_edges):
        """returns bin centers, %sdv, n_total per bin"""
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
        """returns bin centers and MAE per bin"""
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
        """generic binning: returns bin centers and func(scores) per bin"""
        bi = np.digitize(values, bin_edges)
        cx, cy = [], []
        for b in range(1, len(bin_edges)):
            mask = bi == b
            fin = mask & np.isfinite(scores)
            if fin.sum() >= 3:
                cx.append((bin_edges[b-1] + bin_edges[b]) / 2)
                cy.append(func(scores[fin]))
        return np.array(cx), np.array(cy)

    # overall %SDV bins
    up_cx, up_cy, up_cn = pct_sdv(up_dists, up_labels, up_bins)
    dn_cx, dn_cy, dn_cn = pct_sdv(dn_dists, dn_labels, dn_bins)
    ex_cx, ex_cy, ex_cn = pct_sdv(ex_fracs, ex_labels, exon_edges)

    exon_plot_width = 40
    ex_x_mapped = ex_cx * exon_plot_width
    dn_x_mapped = dn_cx + exon_plot_width

    # exon-width stratification (medium 1)
    median_exon_w = np.median(exon_width)
    short_mask = exon_width < median_exon_w
    long_mask = exon_width >= median_exon_w

    # which models are available
    _sdv_models = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]

    # model prediction thresholds for precision/recall — use median |score| of SDVs per model
    _pred_thrs = {}
    for m in _sdv_models:
        col = MODEL_COLS[m]
        sdv_scores = np.abs(df[col].values[label == 1])
        _pred_thrs[m] = np.nanmedian(sdv_scores) if np.any(np.isfinite(sdv_scores)) else 0.1

    # nucleotide substitution data
    has_alleles = "ref" in df.columns and "alt" in df.columns
    if has_alleles:
        ref_al = df["ref"].values.astype(str)
        alt_al = df["alt"].values.astype(str)
        # only single-nt substitutions
        snv_mask = np.array([len(r) == 1 and len(a) == 1 for r, a in zip(ref_al, alt_al)])
    else:
        snv_mask = np.zeros(len(df), dtype=bool)

    # phyloP — try loading, skip if not available
    _phylop_path = Path("vex_seq/hg19.phyloP46way.bw")
    _has_phylop = False
    try:
        import pyBigWig
        if _phylop_path.exists():
            _bw = pyBigWig.open(str(_phylop_path))
            _has_phylop = True
            print(f"loaded phyloP: {_phylop_path}")
    except (ImportError, Exception):
        pass

    # --- layout ---
    # rows: %SDV, MAE, disagreement, [phyloP], density, pearson_ad, spearman_ad, auprc_ad
    _row_names = [
        "sdv", "mae", "disagreement",
    ]
    if _has_phylop:
        _row_names.append("phylop")
    _row_names.extend(["density", "pearson_ad", "spearman_ad", "auprc_ad"])
    n_rows = len(_row_names)
    h_ratios = {
        "sdv": 2.5, "mae": 1.8,
        "disagreement": 1.5, "phylop": 1.2, "density": 1.0,
        "pearson_ad": 2.0, "spearman_ad": 2.0, "auprc_ad": 2.0,
    }
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, sum(h_ratios[r] for r in _row_names) + 1),
        height_ratios=[h_ratios[r] for r in _row_names],
        gridspec_kw={"hspace": 0.15})

    _ax = {r: axes[i] for i, r in enumerate(_row_names)}

    # share x-axis for positional rows (all except bar chart rows which have categorical x)
    _bar_rows = {"pearson_ad", "spearman_ad", "auprc_ad"}
    _pos_rows = [r for r in _row_names if r not in _bar_rows]
    _bar_row_list = [r for r in _row_names if r in _bar_rows]
    # share x among positional rows
    for r in _pos_rows[1:]:
        _ax[r].sharex(_ax[_pos_rows[0]])
    # share x among bar rows, hide tick labels on all but last
    for r in _bar_row_list[1:]:
        _ax[r].sharex(_ax[_bar_row_list[0]])
    for r in _bar_row_list[:-1]:
        _ax[r].tick_params(labelbottom=False)

    # region colors for shading (matching LOCATION_LABELS_AD colors)
    _region_shade = {
        "intron": "#009E73",       # green
        "splice_region": "#E69F00", # yellow-orange
        "splice_site": "#D55E00",  # red-orange
        "exon": "#0072B2",         # blue
    }
    _shade_alpha = 0.06

    # x-ranges for each region on the unified positional axis
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
        # boundary lines at exon edges
        for xb in [0, exon_plot_width]:
            ax.axvline(xb, color="#333333", ls="--", lw=0.5, alpha=0.3)

    # backward compat
    _shade_exon = _shade_regions

    def _plot_3region(ax, up_x, up_y, ex_x, ex_y, dn_x, dn_y, color, lw=1, alpha=0.8, label=None):
        """plot a curve across upstream intron, exon, downstream intron"""
        ax.plot(up_x, up_y, color=color, lw=lw, alpha=alpha)
        ax.plot(ex_x * exon_plot_width, ex_y, color=color, lw=lw, alpha=alpha, label=label)
        ax.plot(dn_x + exon_plot_width, dn_y, color=color, lw=lw, alpha=alpha)

    # === row: %SDV (overall) ===
    ax = _ax["sdv"]
    ax.fill_between(up_cx, 0, up_cy, color="#888888", alpha=0.3, step="mid")
    ax.plot(up_cx, up_cy, color="#333333", lw=1.2, label="All")
    ax.fill_between(ex_x_mapped, 0, ex_cy, color="#4a86c8", alpha=0.3, step="mid")
    ax.plot(ex_x_mapped, ex_cy, color="#2b5d8e", lw=1.2)
    ax.fill_between(dn_x_mapped, 0, dn_cy, color="#888888", alpha=0.3, step="mid")
    ax.plot(dn_x_mapped, dn_cy, color="#333333", lw=1.2)
    ax.set_ylabel("%SDV", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15, axis="y")
    ax.set_title(
        f"{dataset_names[name]} — positional analysis "
        f"({len(df):,} variants, {label.sum():,} SDVs, |ΔPSI| > {thr})",
        fontsize=12)
    _shade_exon(ax)

    # === rows: Pearson / Spearman / AUPRC by acc/don region ===
    loc = get_location_masks(pos, exon_start, exon_end)
    # genomic order: upstream intron → acceptor SS/SR → exon → donor SS/SR → downstream intron
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
            c = get_color(m)
            x = x_pos + (mi - n_mods / 2 + 0.5) * bar_w
            ax.bar(x, vals, bar_w, color=c, edgecolor="black", linewidth=0.3,
                   label=model_names.get(m, m) if metric_key == "pearson_ad" else "")

        # annotate n and SDV count below bars
        for ri, rk in enumerate(_ad_regions):
            rmask = loc[rk]
            n_r = rmask.sum()
            n_sdv_r = label[rmask].sum()
            ax.text(ri, -0.12, f"{n_r:,}\n({n_sdv_r})", ha="center", fontsize=5,
                    color="#666666", transform=ax.get_xaxis_transform())

        # shade bar background by parent region
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
        ax.set_xticklabels([LOCATION_LABELS_AD[rk] for rk in _ad_regions], fontsize=6.5, rotation=30, ha="right")
        ax.set_ylabel(metric_name, fontsize=9)
        ax.grid(alpha=0.15, axis="y")
        if metric_key == "pearson_ad":
            ax.legend(fontsize=6, ncol=3, loc="upper right")

    # === row: MAE (all models overlaid) ===
    ax = _ax["mae"]
    for m in _sdv_models:
        col = MODEL_COLS[m]
        pred = df[col].values
        up_mx, up_my = bin_mae(up_dists, pred[in_upstream], y[in_upstream], up_bins)
        ex_mx, ex_my = bin_mae(ex_fracs, pred[in_exon], y[in_exon], exon_edges)
        dn_mx, dn_my = bin_mae(dn_dists, pred[in_downstream], y[in_downstream], dn_bins)
        c = get_color(m)
        _plot_3region(ax, up_mx, up_my, ex_mx, ex_my, dn_mx, dn_my,
                      color=c, lw=1, label=model_names.get(m, m))
    ax.set_ylabel("MAE", fontsize=9)
    ax.grid(alpha=0.15, axis="y")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    _shade_exon(ax)

    # === row: model disagreement ===
    ax = _ax["disagreement"]
    # variance of |Δscore| across models at each position
    model_preds = []
    for m in _sdv_models:
        col = MODEL_COLS[m]
        model_preds.append(np.abs(df[col].values))
    model_stack = np.column_stack(model_preds)  # (n_variants, n_models)
    model_var = np.nanstd(model_stack, axis=1)  # std across models per variant

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
    ax.grid(alpha=0.15, axis="y")
    _shade_exon(ax)

    # === row: phyloP conservation (conditional) ===
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
            ax.grid(alpha=0.15, axis="y")
        _shade_exon(ax)

    # === row: SNV density ===
    ax = _ax["density"]
    ax.bar(up_cx, up_cn, width=1, color="#888888", alpha=0.5, edgecolor="none")
    ax.bar(ex_x_mapped, ex_cn, width=exon_plot_width / n_exon_bins,
           color="#4a86c8", alpha=0.5, edgecolor="none")
    ax.bar(dn_x_mapped, dn_cn, width=1, color="#888888", alpha=0.5, edgecolor="none")
    ax.set_ylabel("SNV\ncount", fontsize=9)
    ax.set_xlabel("Position (intron bp / normalized exon / intron bp)", fontsize=10)
    _shade_exon(ax)

    # x-axis limits and ticks
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

    # hide x-tick labels on positional rows except the bottom one (density)
    for r in _pos_rows[:-1]:
        _ax[r].tick_params(labelbottom=False)

    plt.savefig(f"{fig3_sup}/{name}_pct_sdv_by_position.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_pct_sdv_by_position.pdf", bbox_inches="tight")
    plt.show()

    # summary stats using chong categories
    _loc_summary = get_location_masks(pos, exon_start, exon_end)
    print(f"{dataset_names[name]}:")
    for lk in ["splice_site", "splice_region", "exon", "intron"]:
        lm = _loc_summary[lk]
        print(f"  SDVs in {LOCATION_LABELS[lk]}: {label[lm].sum()} / {lm.sum()} "
              f"({100 * label[lm].mean():.1f}% SDV rate)")
    # combined splice site + region = mount definition
    ss_mount = _loc_summary["splice_site"] | _loc_summary["splice_region"]
    print(f"  SDVs in SS+region (Mount): {label[ss_mount].sum()} / {ss_mount.sum()} "
          f"({100 * label[ss_mount].mean():.1f}%)")
    if _has_phylop:
        _bw.close()
    print(f"  saved {fig3_sup}/{name}_pct_sdv_by_position.png")


# --- combined 1x2 panel: vex-seq (a) + mfass (b) ---
from matplotlib.image import imread as _imread_panel
_combined_paths = [
    ("a", f"{fig3_sup}/vexseq_pct_sdv_by_position.png", "Vex-seq"),
    ("b", f"{fig3_sup}/mfass_pct_sdv_by_position.png", "MFASS"),
]
_panel_imgs = []
for letter, path, dname in _combined_paths:
    if Path(path).exists():
        _panel_imgs.append((letter, _imread_panel(path), dname))
if len(_panel_imgs) == 2:
    # get aspect ratios
    h_a, w_a = _panel_imgs[0][1].shape[:2]
    h_b, w_b = _panel_imgs[1][1].shape[:2]
    fig_w = 24
    fig_h = fig_w / 2 * max(h_a / w_a, h_b / w_b)
    fig_cmb, axes_cmb = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, (letter, img, dname) in zip(axes_cmb, _panel_imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.text(-0.01, 1.01, letter, transform=ax.transAxes,
                fontsize=24, fontweight="bold", va="bottom", ha="left")
        ax.text(0.5, 1.01, dname, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="center")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/pct_sdv_by_position_combined.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/pct_sdv_by_position_combined.pdf", bbox_inches="tight")
    plt.show()
    print(f"saved combined panel: {fig3_sup}/pct_sdv_by_position_combined.png")
else:
    print(f"skipping combined panel — only {len(_panel_imgs)}/2 per-dataset panels available")


# variant error analysis — which types of variants do models fail on?
# 1) binned MAE by position  2) Ti/Tv  3) error vs effect size
# 4) location × mutation type heatmap  5) top-k worst predictions

_a_cols = MODEL_COLS
_a_models = MODEL_LIST

# transition/transversion
_TI = {("A","G"), ("G","A"), ("C","T"), ("T","C")}

for name, df in dfs.items():
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)
    y_true = df["y"].values
    loc = get_location_masks(pos, exon_start, exon_end)

    has_alleles = "ref" in df.columns and "alt" in df.columns
    if has_alleles:
        ref_al = df["ref"].values.astype(str)
        alt_al = df["alt"].values.astype(str)
        mut_type = np.array(["Ti" if (r.upper(), a.upper()) in _TI else "Tv"
                             for r, a in zip(ref_al, alt_al)])
        is_ti = mut_type == "Ti"
        n_ti, n_tv = is_ti.sum(), (~is_ti).sum()
        print(f"{dataset_names[name]}: {n_ti:,} transitions, {n_tv:,} transversions")
    else:
        print(f"{dataset_names[name]}: no ref/alt columns, skipping Ti/Tv analyses")

    # assign to acceptor or donor
    dist_to_acc = pos - exon_start
    dist_to_don = exon_end - pos
    is_acc = np.abs(dist_to_acc) <= np.abs(dist_to_don)

    mods = [m for m in _a_models if _a_cols[m] in df.columns]

    # ============================================================
    # 1. Binned MAE by position (acceptor / donor)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bin_w = 1
    for ax, (side, smask, dvals) in zip(axes, [
        ("Acceptor", is_acc, dist_to_acc[is_acc]),
        ("Donor", ~is_acc, -(exon_end[~is_acc] - pos[~is_acc])),
    ]):
        for m in mods:
            err_side = np.abs(df[_a_cols[m]].values - y_true)[smask]
            fin = np.isfinite(err_side)
            d, e = dvals[fin], err_side[fin]
            bins = np.arange(d.min() - bin_w/2, d.max() + bin_w, bin_w)
            bi = np.digitize(d, bins)
            cx, cy = [], []
            for b in range(1, len(bins)):
                bm = bi == b
                if bm.sum() >= 5:
                    cx.append((bins[b-1] + bins[b]) / 2)
                    cy.append(np.mean(e[bm]))
            ax.plot(cx, cy, color=get_color(m), lw=1.5, alpha=0.8, label=model_names[m])
        ax.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"Position relative to {side.lower()}")
        ax.set_title(side)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean absolute error")
    axes[0].legend(fontsize=8)
    plt.suptitle(f"{dataset_names[name]} — MAE by position", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_mae_by_position.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_mae_by_position.pdf", bbox_inches="tight")
    plt.show()

    # ============================================================
    # 3. Error vs effect size (|error| vs |measured ΔPSI|)
    # ============================================================
    fig, axes = plt.subplots(1, len(mods), figsize=(3 * len(mods), 4), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, mods):
        abs_err = np.abs(df[_a_cols[m]].values - y_true)
        abs_y = np.abs(y_true)
        fin = np.isfinite(abs_err) & np.isfinite(abs_y)
        ax.scatter(abs_y[fin], abs_err[fin], s=4, alpha=0.15, color=get_color(m),
                   edgecolors="none", rasterized=True)
        # binned trend
        pct = np.percentile(abs_y[fin], np.linspace(0, 100, 21))
        bi = np.digitize(abs_y[fin], pct)
        cx, cy = [], []
        for b in range(1, len(pct)):
            bm = bi == b
            if bm.sum() >= 3:
                cx.append((pct[b-1] + pct[b]) / 2)
                cy.append(np.mean(abs_err[fin][bm]))
        ax.plot(cx, cy, color="black", lw=2, zorder=5)
        ax.set_xlabel("|Measured ΔPSI|")
        ax.set_title(model_names[m])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("|Prediction error|")
    plt.suptitle(f"{dataset_names[name]} — Error vs effect size", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_error_vs_effect.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_error_vs_effect.pdf", bbox_inches="tight")
    plt.show()

    # ============================================================
    # 5. Error vs flanking intron length
    # ============================================================
    # for each tested exon, find the nearest neighboring exon upstream and downstream
    # from the GTF — this gives the real intron length the model "sees" beyond the construct
    try:
        _gtf_exons_by_strand
    except NameError:
        # GTF search order: env override, script-relative, cwd-relative, legacy cluster path
        _gtf_candidates = []
        if os.environ.get("SPLAIRE_MFASS_GTF"):
            _gtf_candidates.append(Path(os.environ["SPLAIRE_MFASS_GTF"]))
        _gtf_candidates.extend([
            Path(__file__).resolve().parent / ".." / ".." / "pipeline" / "python" / "mfass" / "gencode.v19.annotation.gtf",
            Path("../../pipeline/python/mfass/gencode.v19.annotation.gtf"),
            Path("gencode.v19.annotation.gtf"),
            Path("/projects/talisman/mrunyan/SpHAEC/pipeline/python/mfass/gencode.v19.annotation.gtf"),
        ])
        gtf_path = next((p for p in _gtf_candidates if p.exists()), _gtf_candidates[0])
        print(f"loading GTF: {gtf_path}")
        _gtf_exons_by_strand = {}  # (chrom, strand) -> sorted array of (start, end)
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
                if key not in _gtf_exons_by_strand:
                    _gtf_exons_by_strand[key] = []
                _gtf_exons_by_strand[key].append((s, e))
        for key in _gtf_exons_by_strand:
            _gtf_exons_by_strand[key] = np.array(sorted(set(_gtf_exons_by_strand[key])))
        n_total = sum(len(v) for v in _gtf_exons_by_strand.values())
        print(f"loaded {n_total:,} exon intervals (strand-aware) across {len(_gtf_exons_by_strand)} chrom/strand pairs")

    # compute min flanking intron for each tested exon (strand-matched)
    chroms = df["chrom"].values if "chrom" in df.columns else df["seqnames"].values
    strands = df["strand"].values
    up_intron = np.full(len(df), np.nan)
    dn_intron = np.full(len(df), np.nan)
    for i in range(len(df)):
        ch = chroms[i]
        st = strands[i]
        es, ee = exon_start[i], exon_end[i]
        key = (ch, st)
        if key not in _gtf_exons_by_strand or len(_gtf_exons_by_strand[key]) == 0:
            continue
        gtf_starts = _gtf_exons_by_strand[key][:, 0]
        gtf_ends = _gtf_exons_by_strand[key][:, 1]
        # upstream: nearest exon that ends before our exon starts
        upstream_mask = gtf_ends < es
        if upstream_mask.any():
            up_intron[i] = es - gtf_ends[upstream_mask].max()
        # downstream: nearest exon that starts after our exon ends
        downstream_mask = gtf_starts > ee
        if downstream_mask.any():
            dn_intron[i] = gtf_starts[downstream_mask].min() - ee

    min_intron = np.fmin(up_intron, dn_intron)
    max_intron = np.fmax(up_intron, dn_intron)
    mean_intron = (up_intron + dn_intron) / 2
    fin_intron = np.isfinite(min_intron) & np.isfinite(max_intron)
    print(f"\n{dataset_names[name]} flanking intron lengths:")
    for lbl, arr in [("upstream", up_intron), ("downstream", dn_intron),
                     ("min", min_intron), ("max", max_intron), ("mean", mean_intron)]:
        print(f"  {lbl}: median={np.nanmedian(arr):.0f}, mean={np.nanmean(arr):.0f}, "
              f"range=[{np.nanmin(arr):.0f}, {np.nanmax(arr):.0f}]")

    # three plots: min, max, mean flanking intron length
    for agg_label, agg_arr, agg_tag in [
        ("Min flanking intron", min_intron, "min"),
        ("Max flanking intron", max_intron, "max"),
        ("Mean flanking intron", mean_intron, "mean"),
    ]:
        fin_agg = np.isfinite(agg_arr)
        fig, axes = plt.subplots(1, len(mods), figsize=(3 * len(mods), 4), sharey=True,
                                 squeeze=False)
        axes = axes[0]
        for ax, m in zip(axes, mods):
            abs_err = np.abs(df[_a_cols[m]].values - y_true)
            fin = np.isfinite(abs_err) & fin_agg
            x_val = agg_arr[fin]
            y_val = abs_err[fin]
            ax.scatter(x_val, y_val, s=4, alpha=0.15, color=get_color(m),
                       edgecolors="none", rasterized=True)
            # binned trend
            pct = np.percentile(x_val, np.linspace(0, 100, 16))
            pct = np.unique(pct)
            bi = np.digitize(x_val, pct)
            cx, cy = [], []
            for b in range(1, len(pct)):
                bm = bi == b
                if bm.sum() >= 5:
                    cx.append((pct[b-1] + pct[b]) / 2)
                    cy.append(np.mean(y_val[bm]))
            ax.plot(cx, cy, color="black", lw=2, zorder=5)
            ax.set_xlabel(f"{agg_label} length (bp)")
            ax.set_title(model_names[m])
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("|Prediction error|")
        plt.suptitle(f"{dataset_names[name]} — Error vs {agg_label.lower()} length", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{fig3_sup}/{name}_error_vs_{agg_tag}_intron_length.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig3_sup}/{name}_error_vs_{agg_tag}_intron_length.pdf", bbox_inches="tight")
        plt.show()

    # ============================================================
    # 6. Top 10 worst predictions per model
    # ============================================================
    print(f"\n{'='*80}")
    print(f"{dataset_names[name]} — Top 10 worst predictions per model")
    print(f"{'='*80}")
    for m in mods:
        pred = df[_a_cols[m]].values
        err = pred - y_true
        abs_err = np.abs(err)
        idx = np.argsort(abs_err)[::-1]
        idx = idx[np.isfinite(abs_err[idx])][:10]
        print(f"\n{model_names[m]}:")
        hdr = f"  {'pos':>10}"
        if has_alleles: hdr += f" {'ref':>4} {'alt':>4} {'Ti/Tv':>5}"
        hdr += f" {'loc':>12} {'measured':>10} {'predicted':>10} {'error':>10}"
        print(hdr)
        for i in idx:
            loc_lbl = "?"
            for lk in ["splice_site", "splice_region", "exon", "intron"]:
                if loc[lk][i]:
                    loc_lbl = LOCATION_LABELS[lk]; break
            row = f"  {pos[i]:>10}"
            if has_alleles: row += f" {ref_al[i]:>4} {alt_al[i]:>4} {mut_type[i]:>5}"
            row += f" {loc_lbl:>12} {y_true[i]:>10.4f} {pred[i]:>10.4f} {err[i]:>10.4f}"
            print(row)


# annotate variants with gnomAD MAF
# gnomAD v2.1.1 exomes (hg19) — uses bare chrom names (1, 2, ..., X)
# results cached to disk so tabix lookup only runs once

import pysam, pickle

gnomad_path = os.environ.get("SPLAIRE_GNOMAD_VCF", "/scratch/runyan.m/gnomad.exomes.r2.1.1.sites.vcf.bgz")
_gnomad_cache = fig3_sup / "_gnomad_af_cache.pkl"

# try loading from disk cache
if _gnomad_cache.exists():
    with open(_gnomad_cache, "rb") as f:
        _cached = pickle.load(f)
    for name, df in dfs.items():
        if name in _cached and len(_cached[name]) == len(df):
            df["gnomad_af"] = _cached[name]
    print(f"loaded gnomAD AF from {_gnomad_cache}")

# annotate any datasets still missing
needs_lookup = [name for name, df in dfs.items() if "gnomad_af" not in df.columns]

if needs_lookup:
    vcf = pysam.VariantFile(gnomad_path)

    for name in needs_lookup:
        df = dfs[name]
        chrom = df["chrom"].values.astype(str)
        pos = df["pos"].values.astype(int)
        ref_al = df["ref"].values.astype(str)
        alt_al = df["alt"].values.astype(str)

        mafs = np.full(len(df), np.nan)
        found = 0

        for i in tqdm(range(len(df)), desc=f"{dataset_names[name]} gnomAD lookup"):
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
                            found += 1
                        break
            except (ValueError, KeyError):
                pass

        df["gnomad_af"] = mafs
        n_found = np.isfinite(mafs).sum()
        print(f"{dataset_names[name]}: {n_found:,} / {len(df):,} found in gnomAD "
              f"({100*n_found/len(df):.1f}%)")

    vcf.close()

    # save cache
    _to_cache = {name: df["gnomad_af"].values for name, df in dfs.items()}
    with open(_gnomad_cache, "wb") as f:
        pickle.dump(_to_cache, f)
    print(f"saved gnomAD AF cache to {_gnomad_cache}")

else:
    for name, df in dfs.items():
        n_found = np.isfinite(df["gnomad_af"]).sum()
        print(f"{dataset_names[name]}: {n_found:,} / {len(df):,} in gnomAD (cached)")


# error vs MAF analysis

_maf_cols = MODEL_COLS
_maf_models = MODEL_LIST

# MAF bins
_maf_bins = [
    ("not in gnomAD", lambda af: ~np.isfinite(af)),
    ("singleton\n(AF<1e-5)", lambda af: np.isfinite(af) & (af < 1e-5)),
    ("rare\n(1e-5–1e-3)", lambda af: (af >= 1e-5) & (af < 1e-3)),
    ("low freq\n(1e-3–0.01)", lambda af: (af >= 1e-3) & (af < 0.01)),
    ("common\n(≥0.01)", lambda af: af >= 0.01),
]

for name, df in dfs.items():
    if "gnomad_af" not in df.columns:
        print(f"{dataset_names[name]}: no gnomAD AF, skipping")
        continue

    af = df["gnomad_af"].values
    y_true = df["y"].values
    has_af = np.isfinite(af)
    mods = [m for m in _maf_models if _maf_cols[m] in df.columns]

    print(f"\n{dataset_names[name]}:")
    print(f"  in gnomAD: {has_af.sum():,} / {len(df):,} ({100*has_af.mean():.1f}%)")
    for label, fn in _maf_bins:
        n = fn(af).sum()
        print(f"  {label.replace(chr(10), ' ')}: {n:,}")

    # ============================================================
    # 2. Error vs log10(MAF) — for variants found in gnomAD
    # ============================================================
    if has_af.sum() > 50:
        fig, axes = plt.subplots(1, len(mods), figsize=(3 * len(mods), 4),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, m in zip(axes, mods):
            abs_err = np.abs(df[_maf_cols[m]].values - y_true)
            fin = has_af & np.isfinite(abs_err)
            log_af = np.log10(af[fin].clip(1e-7))
            err_fin = abs_err[fin]

            ax.scatter(log_af, err_fin, s=6, alpha=0.2, color=get_color(m),
                       edgecolors="none", rasterized=True)
            # binned trend
            pct = np.percentile(log_af, np.linspace(0, 100, 11))
            bi = np.digitize(log_af, pct)
            cx, cy = [], []
            for b in range(1, len(pct)):
                bm = bi == b
                if bm.sum() >= 3:
                    cx.append((pct[b-1] + pct[b]) / 2)
                    cy.append(np.mean(err_fin[bm]))
            ax.plot(cx, cy, color="black", lw=2, zorder=5)
            ax.set_xlabel("log₁₀(gnomAD AF)")
            ax.set_title(model_names[m])
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("|Prediction error|")
        plt.suptitle(f"{dataset_names[name]} — Error vs allele frequency (gnomAD variants only)",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{fig3_sup}/{name}_error_vs_maf.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig3_sup}/{name}_error_vs_maf.pdf", bbox_inches="tight")
        plt.show()

    # ============================================================
    # 2b. True effect size vs log10(MAF) — for variants found in gnomAD
    # ============================================================
    if has_af.sum() > 50:
        fig, ax = plt.subplots(figsize=(6, 5))
        fin = has_af & np.isfinite(y_true)
        log_af = np.log10(af[fin].clip(1e-7))
        abs_y = np.abs(y_true[fin])

        ax.scatter(log_af, abs_y, s=6, alpha=0.2, color="#666666",
                   edgecolors="none", rasterized=True)
        # binned trend
        pct = np.percentile(log_af, np.linspace(0, 100, 16))
        pct = np.unique(pct)
        bi = np.digitize(log_af, pct)
        cx, cy = [], []
        for b in range(1, len(pct)):
            bm = bi == b
            if bm.sum() >= 3:
                cx.append((pct[b-1] + pct[b]) / 2)
                cy.append(np.median(abs_y[bm]))
        ax.plot(cx, cy, color="black", lw=2.5, zorder=5, label="median")
        ax.set_xlabel("log\u2081\u2080(gnomAD AF)")
        ax.set_ylabel("|Measured \u0394PSI|")
        ax.set_title(f"{dataset_names[name]} — Effect size vs allele frequency")
        ax.legend()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{fig3_sup}/{name}_effect_vs_maf.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig3_sup}/{name}_effect_vs_maf.pdf", bbox_inches="tight")
        plt.show()
        # print correlation
        rho = stats.spearmanr(log_af, abs_y)[0]
        print(f"  Spearman(log10 AF, |ΔPSI|) = {rho:.3f}")


# (supplemental threshold sweep removed — merged into main threshold figure above)


sd_info = {}
for name, df in dfs.items():
    y = df["y"].values
    y_clean = y[np.isfinite(y)]
    mu, sd = y_clean.mean(), y_clean.std()
    lo, hi = mu - sd, mu + sd
    outside = (y_clean < lo) | (y_clean > hi)
    n_pos = (y_clean > hi).sum()
    n_neg = (y_clean < lo).sum()
    n_neutral = (~outside).sum()
    sd_info[name] = (mu, sd, lo, hi)
    print(f"{dataset_names[name]}:")
    print(f"  mean = {mu:.4f}, SD = {sd:.4f}")
    print(f"  thresholds: < {lo:.4f} (decreasing) or > {hi:.4f} (increasing)")
    print(f"  kept: {n_pos + n_neg:,} / {len(y_clean):,} ({100*(n_pos+n_neg)/len(y_clean):.1f}%)")

# filter dataframes
dfs_filt = {}
for name, df in dfs.items():
    mu, sd, lo, hi = sd_info[name]
    mask = (df["y"] < lo) | (df["y"] > hi)
    dfs_filt[name] = df[mask].reset_index(drop=True)
    print(f"{dataset_names[name]} filtered: {len(dfs_filt[name]):,} variants")

# neutral dataframes (within ±1 SD) for gray background scatter
dfs_neutral = {}
for name, df in dfs.items():
    mu, sd, lo, hi = sd_info[name]
    mask = (df["y"] >= lo) & (df["y"] <= hi)
    dfs_neutral[name] = df[mask].reset_index(drop=True)


# save filtered data for figures.ipynb
# (avoids re-running full H5 loading + delta computation in the figures notebook)
import pickle

_reporter_pkl = fig3_main / "_reporter_data.pkl"
_reporter_data = {
    "dfs_filt": dfs_filt,
    "dfs_neutral": dfs_neutral,
}
with open(_reporter_pkl, "wb") as f:
    pickle.dump(_reporter_data, f)
print(f"saved reporter data to {_reporter_pkl}")
for name, df in dfs_filt.items():
    print(f"  {dataset_names[name]}: {len(df):,} filtered, {len(dfs_neutral[name]):,} neutral")


def bootstrap_compare(y, preds_dict, label, n_boot=10_000, seed=42):
    """paired bootstrap significance for all model pairs, returns results dict"""
    from itertools import combinations
    from sklearn.metrics import r2_score

    models = list(preds_dict.keys())
    n = len(y)

    # all predictions must be finite
    for m, pred in preds_dict.items():
        n_nonfinite = (~np.isfinite(pred)).sum()
        if n_nonfinite > 0:
            raise ValueError(f"model {m} has {n_nonfinite} non-finite predictions")
    if (~np.isfinite(y)).any():
        raise ValueError(f"y has {(~np.isfinite(y)).sum()} non-finite values")
    n_m = len(y)
    y_m, lab_m = y, label
    preds_m = preds_dict
    if n_m < 10:
        return {}

    rng = np.random.default_rng(seed)

    # bootstrap: compute metrics for each model on each resample
    # store as {model: (n_boot, 4)} array — columns: pearson, spearman, r2, auprc
    boot_metrics = {m: np.empty((n_boot, 4)) for m in models}

    for b in range(n_boot):
        idx = rng.choice(n_m, size=n_m, replace=True)
        yb, lb = y_m[idx], lab_m[idx]
        for m in models:
            pb = preds_m[m][idx]
            r = stats.pearsonr(yb, pb)[0]
            rho = stats.spearmanr(yb, pb)[0]
            r2 = r2_score(yb, pb)
            auprc = average_precision_score(lb, np.abs(pb)) if lb.sum() > 0 else np.nan
            boot_metrics[m][b] = [r, rho, r2, auprc]

    metric_names = ["Pearson", "Spearman", "R2", "AUPRC"]
    results = {}
    for a, b in combinations(models, 2):
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
    """cluster bootstrap: resample exons (clusters) instead of individual variants"""
    from itertools import combinations
    from sklearn.metrics import r2_score

    models = list(preds_dict.keys())

    # all predictions must be finite
    for m, pred in preds_dict.items():
        n_nonfinite = (~np.isfinite(pred)).sum()
        if n_nonfinite > 0:
            raise ValueError(f"model {m} has {n_nonfinite} non-finite predictions")
    if (~np.isfinite(y)).any():
        raise ValueError(f"y has {(~np.isfinite(y)).sum()} non-finite values")
    n = len(y)
    if n < 10:
        return {}

    # group indices by cluster
    unique_clusters = np.unique(clusters)
    cluster_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
    n_clusters = len(unique_clusters)
    print(f"  cluster bootstrap: {n} variants in {n_clusters} clusters")

    rng = np.random.default_rng(seed)
    boot_metrics = {m: np.empty((n_boot, 4)) for m in models}

    for b in range(n_boot):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_idx[c] for c in sampled])
        yb, lb = y[idx], label[idx]
        for m in models:
            pb = preds_dict[m][idx]
            r = stats.pearsonr(yb, pb)[0]
            rho = stats.spearmanr(yb, pb)[0]
            r2 = r2_score(yb, pb)
            auprc = average_precision_score(lb, np.abs(pb)) if lb.sum() > 0 else np.nan
            boot_metrics[m][b] = [r, rho, r2, auprc]

    metric_names = ["Pearson", "Spearman", "R2", "AUPRC"]
    results = {}
    for a, b_ in combinations(models, 2):
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


def holm_bonferroni(results):
    """apply holm-bonferroni correction per metric (separate family per metric)"""
    # group p-values by metric
    by_metric = {}
    for pair, metrics in results.items():
        for mname, d in metrics.items():
            by_metric.setdefault(mname, []).append((pair, d["p"]))
    # correct within each metric family
    for mname, pvals in by_metric.items():
        pvals.sort(key=lambda x: x[1])
        k = len(pvals)
        prev = 0.0
        for i, (pair, p) in enumerate(pvals):
            p_adj = min(max(prev, p * (k - i)), 1.0)
            prev = p_adj
            results[pair][mname]["p_adj_holm_k10"] = p_adj
            results[pair][mname]["sig_holm_k10"] = significance_stars(p_adj)


def significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


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
    """print formatted comparison table"""
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
                           boot_key, y_offset=0.05, dh=0.04):
    """draw significance brackets for all significant model pairs within one metric group"""
    try:
        boot = _boot_results
    except NameError:
        return 0
    if boot_key not in boot:
        return 0
    pairs = []
    for i in range(len(models_avail)):
        for j in range(i + 1, len(models_avail)):
            a, b = models_avail[i], models_avail[j]
            sig_info = get_boot_sig(boot[boot_key], a, b, metric)
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
    """apply plain bonferroni with explicit k (e.g. k=4 for vs-best, k=10 for all-pairs)"""
    for pair, metrics in results.items():
        for mname, d in metrics.items():
            p_adj = min(d["p"] * k, 1.0)
            results[pair][mname][f"p_adj_bonf_k{k}"] = p_adj
            results[pair][mname][f"sig_bonf_k{k}"] = significance_stars(p_adj)


def holm_correct_k(results, k):
    """apply holm step-down with explicit k (e.g. k=4 for vs-best)"""
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


import pickle

_boot_pkl = fig3_main / "_boot_results.pkl"

# load from disk if available, else start fresh
try:
    _boot_results
except NameError:
    if _boot_pkl.exists():
        with open(_boot_pkl, "rb") as f:
            _boot_results = pickle.load(f)
        print(f"loaded {len(_boot_results)} cached results from {_boot_pkl}")
    else:
        _boot_results = {}

if SKIP_BOOTSTRAP:
    print("--skip-bootstrap: skipping standard bootstrap (using cached results only)")

_split_cols_boot = MODEL_COLS

_n_new = 0
for tag, data_src in [("full", dfs), ("filtered", dfs_filt)]:
    for name, df in data_src.items():
        y = df["y"].values
        label = df["label"].values
        pos = df["pos"].values
        loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)

        preds_dict = {}
        for m, col in _split_cols_boot.items():
            if col in df.columns:
                preds_dict[m] = df[col].values

        # for filtered: combined data for AUPRC (non-neutral=1, neutral=0)
        if tag == "filtered":
            df_neut = dfs_neutral[name]
            df_comb = pd.concat([df.assign(_nonneutral=1), df_neut.assign(_nonneutral=0)], ignore_index=True)
            y_comb = df_comb["y"].values
            label_comb = df_comb["_nonneutral"].values
            pos_comb = df_comb["pos"].values
            loc_comb = get_location_masks(pos_comb, df_comb["exon_start"].values, df_comb["exon_end"].values)
            preds_dict_comb = {}
            for m, col in _split_cols_boot.items():
                if col in df_comb.columns:
                    preds_dict_comb[m] = df_comb[col].values

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
                # pearson/spearman/r2 from filtered data only
                results_corr = bootstrap_compare(
                    y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask]
                )
                # auprc from combined data (non-neutral vs neutral)
                mask_comb = loc_comb[subset_name]
                results_auprc = bootstrap_compare(
                    y_comb[mask_comb], {m: p[mask_comb] for m, p in preds_dict_comb.items()}, label_comb[mask_comb]
                )
                # merge: Pearson/Spearman/R2 from filtered, AUPRC from combined
                results = {}
                for pair in results_corr:
                    results[pair] = {}
                    for metric in ["Pearson", "Spearman", "R2"]:
                        results[pair][metric] = results_corr[pair][metric]
                    if pair in results_auprc:
                        results[pair]["AUPRC"] = results_auprc[pair]["AUPRC"]
            else:
                results = bootstrap_compare(
                    y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask]
                )

            if results:
                holm_bonferroni(results)
                _boot_results[key] = results
                _n_new += 1
                print_bootstrap_table(results, model_names, title)

# save to disk if new results were computed
if _n_new > 0:
    with open(_boot_pkl, "wb") as f:
        pickle.dump(_boot_results, f)
    print(f"\nsaved {len(_boot_results)} results to {_boot_pkl} ({_n_new} new)")
else:
    print(f"\nall {len(_boot_results)} results cached, nothing to compute")

# === cluster bootstrap (resample by exon) ===
_cboot_pkl = fig3_main / "_cboot_results.pkl"
try:
    _cboot_results
except NameError:
    if _cboot_pkl.exists():
        with open(_cboot_pkl, "rb") as f:
            _cboot_results = pickle.load(f)
        print(f"loaded {len(_cboot_results)} cached cluster bootstrap results")
    else:
        _cboot_results = {}

if SKIP_BOOTSTRAP:
    print("--skip-bootstrap: skipping cluster bootstrap (using cached results only)")

_cn_new = 0
for tag, data_src in [("full", dfs), ("filtered", dfs_filt)]:
    for name, df in data_src.items():
        y = df["y"].values
        label = df["label"].values
        pos = df["pos"].values
        loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)
        # cluster ID = exon boundaries
        clusters = (df["exon_start"].astype(str) + "_" + df["exon_end"].astype(str)).values

        preds_dict = {}
        for m, col in _split_cols_boot.items():
            if col in df.columns:
                preds_dict[m] = df[col].values

        if tag == "filtered":
            df_neut = dfs_neutral[name]
            df_comb = pd.concat([df.assign(_nonneutral=1), df_neut.assign(_nonneutral=0)], ignore_index=True)
            y_comb = df_comb["y"].values
            label_comb = df_comb["_nonneutral"].values
            pos_comb = df_comb["pos"].values
            loc_comb = get_location_masks(pos_comb, df_comb["exon_start"].values, df_comb["exon_end"].values)
            clusters_comb = (df_comb["exon_start"].astype(str) + "_" + df_comb["exon_end"].astype(str)).values
            preds_dict_comb = {}
            for m, col in _split_cols_boot.items():
                if col in df_comb.columns:
                    preds_dict_comb[m] = df_comb[col].values

        for subset_name in LOCATION_SUBSETS:
            mask = loc[subset_name]
            key = (name, tag, subset_name)
            if key in _cboot_results:
                print(f"cluster cached: {dataset_names[name]} — {tag} — {subset_name}")
                continue
            if SKIP_BOOTSTRAP:
                print(f"cluster skipped: {dataset_names[name]} — {tag} — {subset_name}")
                continue

            n_sub = mask.sum()
            title = f"CLUSTER: {dataset_names[name]} — {tag} — {subset_name} (n={n_sub:,})"

            if tag == "filtered":
                results_corr = cluster_bootstrap_compare(
                    y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask], clusters[mask]
                )
                mask_comb = loc_comb[subset_name]
                results_auprc = cluster_bootstrap_compare(
                    y_comb[mask_comb], {m: p[mask_comb] for m, p in preds_dict_comb.items()},
                    label_comb[mask_comb], clusters_comb[mask_comb]
                )
                results = {}
                for pair in results_corr:
                    results[pair] = {}
                    for metric in ["Pearson", "Spearman", "R2"]:
                        results[pair][metric] = results_corr[pair][metric]
                    if pair in results_auprc:
                        results[pair]["AUPRC"] = results_auprc[pair]["AUPRC"]
            else:
                results = cluster_bootstrap_compare(
                    y[mask], {m: p[mask] for m, p in preds_dict.items()}, label[mask], clusters[mask]
                )

            if results:
                holm_bonferroni(results)
                _cboot_results[key] = results
                _cn_new += 1
                print_bootstrap_table(results, model_names, title)

if _cn_new > 0:
    with open(_cboot_pkl, "wb") as f:
        pickle.dump(_cboot_results, f)
    print(f"\nsaved {len(_cboot_results)} cluster results to {_cboot_pkl} ({_cn_new} new)")
else:
    print(f"\nall {len(_cboot_results)} cluster results cached")

# re-apply corrections on cluster results
for key, results in _cboot_results.items():
    holm_bonferroni(results)
    holm_correct_k(results, 4)
    bonferroni_correct_k(results, 4)
    bonferroni_correct_k(results, 10)

# re-apply corrections on all results (handles cached results with stale corrections)
for key, results in _boot_results.items():
    holm_bonferroni(results)        # holm per-metric (k=10 for all-pairs)
    holm_correct_k(results, 4)         # holm per-metric (k=4 for vs-best)
    bonferroni_correct_k(results, 4)   # plain bonferroni vs-best (k=4)
    bonferroni_correct_k(results, 10)  # plain bonferroni all-pairs (k=10)
print("re-applied all corrections (holm k=10, bonferroni k=4, bonferroni k=10)")

# comparison table: holm vs bonferroni
print("\n" + "=" * 90)
print("CORRECTION COMPARISON: Holm k=10 vs Holm k=4 vs Bonferroni k=4 vs Bonferroni k=10")
print("=" * 90)
for key, results in _boot_results.items():
    name, tag, subset = key
    print(f"\n{dataset_names[name]} — {tag} — {subset}")
    print(f"{'pair':<42} {'metric':<10} {'raw p':>8} {'Holm10':>8} {'Holm4':>8} {'Bonf4':>8} {'Bonf10':>8}")
    print("-" * 90)
    for (a, b), metrics in results.items():
        na, nb_ = model_names.get(a, a), model_names.get(b, b)
        for mname, d in metrics.items():
            p_raw = d["p"]
            p_holm = d.get("p_adj_holm_k10", p_raw)
            p_holm4 = d.get("p_adj_holm_k4", p_raw)
            p_b4 = d.get("p_adj_bonf_k4", p_raw)
            p_b10 = d.get("p_adj_bonf_k10", p_raw)
            # only print where any correction differs
            stars = [d.get("sig_holm_k10",""), d.get("sig_holm_k4",""), d.get("sig_bonf_k4",""), d.get("sig_bonf_k10","")]
            if any(s != stars[0] for s in stars) or p_raw < 0.1:
                fmt_p = lambda p: f"{p:.4f}" if p >= 0.0001 else "<.0001"
                print(f"{na:<20} vs {nb_:<17} {mname:<10} {fmt_p(p_raw):>8} {fmt_p(p_holm):>8} {fmt_p(p_holm4):>8} {fmt_p(p_b4):>8} {fmt_p(p_b10):>8}")

# standard vs cluster bootstrap comparison
print("\n" + "=" * 100)
print("STANDARD vs CLUSTER BOOTSTRAP (both Holm k=10)")
print("=" * 100)
for key in _boot_results:
    if key not in _cboot_results:
        continue
    name, tag, subset = key
    std = _boot_results[key]
    clust = _cboot_results[key]
    print(f"\n{dataset_names[name]} — {tag} — {subset}")
    print(f"{'pair':<42} {'metric':<10} {'std p':>8} {'clust p':>8} {'std sig':>8} {'clust sig':>8}")
    print("-" * 85)
    for (a, b_) in std:
        if (a, b_) not in clust:
            continue
        na, nb_ = model_names.get(a, a), model_names.get(b_, b_)
        for mname in std[(a, b_)]:
            if mname not in clust[(a, b_)]:
                continue
            ds = std[(a, b_)][mname]
            dc = clust[(a, b_)][mname]
            fmt_p = lambda p: f"{p:.4f}" if p >= 0.0001 else "<.0001"
            print(f"{na:<20} vs {nb_:<17} {mname:<10} {fmt_p(ds['p']):>8} {fmt_p(dc['p']):>8} {ds.get('sig_holm_k10',''):>8} {dc.get('sig_holm_k10',''):>8}")


from scipy.stats import spearmanr
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score

# KDE cache for filtered data (log-transformed density)
_kde_cache_filt = {}
def _get_kde_filt(name, m, y, pred):
    """kde density with log transform for filtered scatter"""
    key = (name, m)
    if key not in _kde_cache_filt:
        fin = np.isfinite(y) & np.isfinite(pred)
        xm, ym = y[fin], pred[fin]
        z = gaussian_kde(np.vstack([xm, ym]))(np.vstack([xm, ym]))
        z_log = np.log1p(z * 1000)
        order = z_log.argsort()
        _kde_cache_filt[key] = (xm, ym, z_log, order)
    return _kde_cache_filt[key]


_cols_a = MODEL_COLS
_models_a = MODEL_LIST
_metrics_a = ["Pearson", "Spearman", "AUPRC"]

all_mods_a = set()
for name, df in dfs_filt.items():
    all_mods_a.update(m for m in _models_a if _cols_a[m] in df.columns)
mods_a = [m for m in _models_a if m in all_mods_a]
nm_a = len(mods_a)

_scatter_ticks = [-1, -0.5, 0, 0.5, 1]
n_datasets = len(dfs_filt)


# --- third figure: scatter + exonic/intronic only (no "all") — filtered ---

# Option A: scatter + grouped bars by metric — full (unfiltered) reporter assays

from scipy.stats import gaussian_kde, spearmanr
from matplotlib.patches import Patch

# raw KDE cache (no log transform) for full data
_kde_cache_full_raw = {}
def _get_kde_full_raw(name, m, y, pred):
    """kde density without log transform"""
    key = (name, m)
    if key not in _kde_cache_full_raw:
        fin = np.isfinite(y) & np.isfinite(pred)
        xm, ym = y[fin], pred[fin]
        z = gaussian_kde(np.vstack([xm, ym]))(np.vstack([xm, ym]))
        order = z.argsort()
        _kde_cache_full_raw[key] = (xm, ym, z, order)
    return _kde_cache_full_raw[key]

# log-transformed KDE cache for full data
_kde_cache_full_log = {}
def _get_kde(name, m, y, pred):
    """kde density with log transform for full data scatter"""
    key = (name, m)
    if key not in _kde_cache_full_log:
        fin = np.isfinite(y) & np.isfinite(pred)
        xm, ym = y[fin], pred[fin]
        z = gaussian_kde(np.vstack([xm, ym]))(np.vstack([xm, ym]))
        z_log = np.log1p(z)
        order = z_log.argsort()
        _kde_cache_full_log[key] = (xm, ym, z_log, order)
    return _kde_cache_full_log[key]

_cols_af = MODEL_COLS
_models_af = MODEL_LIST
_metrics_af = ["Pearson", "Spearman", "AUPRC"]

all_mods_af = set()
for name, df in dfs.items():
    all_mods_af.update(m for m in _models_af if _cols_af[m] in df.columns)
mods_af = [m for m in _models_af if m in all_mods_af]
nm_af = len(mods_af)

_scatter_ticks = [-1, -0.5, 0, 0.5, 1]
n_datasets = len(dfs)

# --- figure: scatter + grouped bars (log density) ---
fig = plt.figure(figsize=(3.5 * nm_af, 7 * n_datasets))
gs = fig.add_gridspec(n_datasets * 2, 1, height_ratios=[1, 0.7] * n_datasets, hspace=0.70)

_sc_axes_af = {}
_bar_axes_af = {}
_sc_refs_af = {}

for d_idx, (name, df) in enumerate(dfs.items()):
    y, label = df["y"].values, df["label"].values
    pos = df["pos"].values
    loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)

    # === scatter row (log density) ===
    gs_sc = gs[d_idx * 2].subgridspec(1, nm_af, wspace=0.3)
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
        sc = ax.scatter(xm[order], ym[order], c=z[order], cmap="viridis",
                        s=12, alpha=0.8, edgecolors="none", vmin=z_min, vmax=z_max)
        sc_ref = sc
        r2 = 1 - np.sum((xm - ym) ** 2) / np.sum((xm - xm.mean()) ** 2)
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", transform=ax.transAxes, fontsize=ANNOT_SIZE, va="top", ha="left")
        ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xticks(_scatter_ticks); ax.set_yticks(_scatter_ticks)
        if j == 0: ax.set_ylabel("Predicted \u0394PSI")
        else: ax.set_ylabel("")
        ax.set_xlabel("Measured \u0394PSI")
        ax.set_title(model_names[m])
    _sc_axes_af[d_idx] = sc_axes
    _sc_refs_af[d_idx] = sc_ref

    # === grouped bars row ===
    gs_bar = gs[d_idx * 2 + 1].subgridspec(1, len(LOCATION_SUBSETS), wspace=0.3)

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

        n_models = len(models_avail)
        bar_w = 0.8 / n_models
        for mi, metric in enumerate(_metrics_af):
            best_m = max(models_avail, key=lambda m: model_vals[m][metric])
            boot_key = (name, "full", subset_name)
            vb_pvals = []
            if boot_key in _boot_results:
                for m in models_avail:
                    if m == best_m: continue
                    si = get_boot_sig(_boot_results[boot_key], best_m, m, metric)
                    if si: vb_pvals.append((m, si["p"]))
            vb_corr = holm_correct(vb_pvals)
            for bi, m in enumerate(models_avail):
                x = mi + (bi - n_models / 2 + 0.5) * bar_w
                val = model_vals[m][metric]
                ax.bar(x, val, bar_w, color=get_color(m), edgecolor="black", linewidth=0.5)
                if m != best_m and m in vb_corr:
                    p_adj, sig = vb_corr[m]
                    if sig:
                        ax.text(x, val + 0.01, sig, ha="center", va="bottom",
                                fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(_metrics_af)))
        ax.set_xticklabels(_metrics_af)
        ax.set_ylim(0, 1.0); ax.set_ylabel("Value")
        ax.set_title(f"{LOCATION_LABELS[subset_name]} ({n_sub:,})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    _bar_axes_af[d_idx] = b_axes

fig.canvas.draw()

# dataset titles + colorbars
for d_idx, (name, df) in enumerate(dfs.items()):
    sc_axes = _sc_axes_af[d_idx]
    bbox_l, bbox_r = sc_axes[0].get_position(), sc_axes[-1].get_position()
    fig.text((bbox_l.x0 + bbox_r.x1) / 2, bbox_l.y1 + 0.035,
             dataset_names[name], fontsize=plt.rcParams["font.size"] * 1.4, va="bottom", ha="center")
    if _sc_refs_af[d_idx] is not None:
        cax = fig.add_axes([bbox_r.x1 + 0.01, bbox_r.y0, 0.01, bbox_r.height])
        cb = fig.colorbar(_sc_refs_af[d_idx], cax=cax)
        cb.set_ticks([])
        cax.set_ylabel("log density", fontsize=9, rotation=270, labelpad=12)

# model legend
_lh = [Patch(facecolor=get_color(m), edgecolor="black", label=model_names[m]) for m in mods_af]
fig.legend(handles=_lh, loc="upper center", ncol=len(mods_af), frameon=False,
           prop={"size": plt.rcParams["font.size"] * 1.1}, bbox_to_anchor=(0.5, 0.999))

plt.savefig(f"{fig3_main}/reporter_full_grouped_bars.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{fig3_main}/reporter_full_grouped_bars.pdf", bbox_inches="tight")
plt.show()
print(f"saved {fig3_main}/reporter_full_grouped_bars.png")

# --- second figure: all pairwise significance brackets (full) ---
fig2 = plt.figure(figsize=(5 * len(LOCATION_SUBSETS), 4.5 * n_datasets))
gs2 = fig2.add_gridspec(n_datasets, len(LOCATION_SUBSETS), wspace=0.35, hspace=0.5)

for d_idx, (name, df) in enumerate(dfs.items()):
    y, label = df["y"].values, df["label"].values
    pos = df["pos"].values
    loc = get_location_masks(pos, df["exon_start"].values, df["exon_end"].values)

    for si, subset_name in enumerate(LOCATION_SUBSETS):
        mask = loc[subset_name]
        ax = fig2.add_subplot(gs2[d_idx, si])
        n_sub = mask.sum()
        models_avail = [m for m in mods_af if _cols_af[m] in df.columns]
        model_vals = {}
        for m in models_avail:
            pred = df[_cols_af[m]].values
            r, rho, auprc = get_metrics(y[mask], pred[mask], label[mask])
            model_vals[m] = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}
        n_models = len(models_avail)
        bar_w = 0.8 / n_models
        for mi, metric in enumerate(_metrics_af):
            for bi, m in enumerate(models_avail):
                x = mi + (bi - n_models / 2 + 0.5) * bar_w
                ax.bar(x, model_vals[m][metric], bar_w, color=get_color(m), edgecolor="black", linewidth=0.5)
            boot_key = (name, "full", subset_name)
            draw_all_sig_brackets(ax, mi, bar_w, n_models, models_avail, model_vals, metric, boot_key)
        ax.set_xticks(range(len(_metrics_af))); ax.set_xticklabels(_metrics_af)
        ax.set_ylabel("Value"); ax.set_title(f"{dataset_names[name]} — {LOCATION_LABELS[subset_name]} ({n_sub:,})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.5)

_lh2 = [Patch(facecolor=get_color(m), edgecolor="black", label=model_names[m]) for m in mods_af]
fig2.legend(handles=_lh2, loc="upper center", ncol=len(mods_af), frameon=False, prop={"size": plt.rcParams["font.size"] * 1.1}, bbox_to_anchor=(0.5, 0.999))
plt.savefig(f"{fig3_main}/reporter_full_allpairs.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{fig3_main}/reporter_full_allpairs.pdf", bbox_inches="tight")
plt.show()
print(f"saved {fig3_main}/reporter_full_allpairs.png")


# ============================================================
# figure 3 layout alternatives — real plots for visual comparison
# each option renders one figure using MFASS data (more variants)
# ============================================================

_opt_models = [m for m in MODEL_LIST if MODEL_COLS[m] in dfs["mfass"].columns]
_opt_metrics = ["Pearson", "Spearman", "AUPRC"]

def _compute_bar_metrics(df, models, loc_subsets):
    """compute metrics per model per location subset, returns nested dict"""
    y = df["y"].values
    lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    out = {}
    for sn in loc_subsets:
        mask = loc[sn]
        out[sn] = {}
        for m in models:
            col = MODEL_COLS[m]
            if col not in df.columns:
                continue
            pred = df[col].values
            r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
            out[sn][m] = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}
    return out

# option A: 3 panels (exon, intron, splice_site+splice_region combined as "SS")
def _opt_a():
    df = dfs["mfass"]
    y = df["y"].values; lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    # combined SS
    combined = {"exon": loc["exon"], "intron": loc["intron"],
                "ss_combined": loc["splice_site"] | loc["splice_region"]}
    labels = {"exon": "Exon", "intron": "Intron", "ss_combined": "Splice site"}
    subsets = ["exon", "intron", "ss_combined"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for si, sn in enumerate(subsets):
        ax = axes[si]
        mask = combined[sn]
        n_sub = mask.sum()
        nm = len(_opt_models)
        bw = 0.8 / nm
        for mi, metric in enumerate(_opt_metrics):
            for mj, m in enumerate(_opt_models):
                col = MODEL_COLS[m]
                pred = df[col].values
                r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
                val = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}[metric]
                x = mi + (mj - nm/2 + 0.5) * bw
                ax.bar(x, val, bw, color=get_color(m), edgecolor="black", linewidth=0.3,
                       label=model_names.get(m, m) if mi == 0 else "")
        ax.set_xticks(range(len(_opt_metrics)))
        ax.set_xticklabels(_opt_metrics, fontsize=8)
        ax.set_title(f"{labels[sn]} (n={n_sub:,})", fontsize=10)
        ax.grid(alpha=0.15, axis="y")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0))
    plt.suptitle("Option A — Drop All, 3 location panels", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig3_main}/layout_option_A.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved {fig3_main}/layout_option_A.png")

# option B: 4 Chong panels
def _opt_b():
    df = dfs["mfass"]
    y = df["y"].values; lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    subsets = ["exon", "intron", "splice_region", "splice_site"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for si, sn in enumerate(subsets):
        ax = axes[si]
        mask = loc[sn]
        n_sub = mask.sum()
        nm = len(_opt_models)
        bw = 0.8 / nm
        for mi, metric in enumerate(_opt_metrics):
            for mj, m in enumerate(_opt_models):
                col = MODEL_COLS[m]
                pred = df[col].values
                r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
                val = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}[metric]
                x = mi + (mj - nm/2 + 0.5) * bw
                ax.bar(x, val, bw, color=get_color(m), edgecolor="black", linewidth=0.3,
                       label=model_names.get(m, m) if mi == 0 else "")
        ax.set_xticks(range(len(_opt_metrics)))
        ax.set_xticklabels(_opt_metrics, fontsize=8)
        ax.set_title(f"{LOCATION_LABELS.get(sn, sn)} (n={n_sub:,})", fontsize=10)
        ax.grid(alpha=0.15, axis="y")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0))
    plt.suptitle("Option B — 4 Chong categories (no All)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig3_main}/layout_option_B.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved {fig3_main}/layout_option_B.png")

# option C: metrics as panels, locations as grouped bars
def _opt_c():
    df = dfs["mfass"]
    y = df["y"].values; lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    subsets = ["exon", "intron", "splice_region", "splice_site"]
    _loc_cols = {"exon": "#0072B2", "intron": "#009E73", "splice_region": "#E69F00", "splice_site": "#D55E00"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for mi, metric in enumerate(_opt_metrics):
        ax = axes[mi]
        n_loc = len(subsets)
        n_mod = len(_opt_models)
        total_groups = n_mod
        bw = 0.7 / n_loc
        for li, sn in enumerate(subsets):
            mask = loc[sn]
            vals = []
            for m in _opt_models:
                col = MODEL_COLS[m]
                pred = df[col].values
                r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
                vals.append({"Pearson": r, "Spearman": rho, "AUPRC": auprc}[metric])
            x = np.arange(n_mod) + (li - n_loc/2 + 0.5) * bw
            ax.bar(x, vals, bw, color=_loc_cols[sn], edgecolor="black", linewidth=0.3,
                   label=LOCATION_LABELS.get(sn, sn) if mi == 0 else "")
        ax.set_xticks(range(n_mod))
        ax.set_xticklabels([model_names.get(m, m) for m in _opt_models], fontsize=7, rotation=30, ha="right")
        ax.set_title(metric, fontsize=11)
        ax.grid(alpha=0.15, axis="y")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    plt.suptitle("Option C — Metrics as panels, locations as bar groups", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig3_main}/layout_option_C.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved {fig3_main}/layout_option_C.png")

# option D: dot plot / forest plot
def _opt_d():
    df = dfs["mfass"]
    y = df["y"].values; lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    subsets = ["exon", "intron", "splice_region", "splice_site"]
    _loc_markers = {"exon": "o", "intron": "s", "splice_region": "D", "splice_site": "^"}
    _loc_cols = {"exon": "#0072B2", "intron": "#009E73", "splice_region": "#E69F00", "splice_site": "#D55E00"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for mi, metric in enumerate(_opt_metrics):
        ax = axes[mi]
        for mj, m in enumerate(_opt_models):
            col = MODEL_COLS[m]
            pred = df[col].values
            for li, sn in enumerate(subsets):
                mask = loc[sn]
                r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
                val = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}[metric]
                # jitter locations slightly within each model row
                y_pos = mj + (li - len(subsets)/2 + 0.5) * 0.15
                ax.scatter(val, y_pos, marker=_loc_markers[sn], color=_loc_cols[sn],
                           s=50, edgecolors="black", linewidths=0.5, zorder=5,
                           label=LOCATION_LABELS.get(sn, sn) if mj == 0 else "")
        ax.set_yticks(range(len(_opt_models)))
        ax.set_yticklabels([model_names.get(m, m) for m in _opt_models], fontsize=8)
        ax.set_xlabel(metric, fontsize=10)
        ax.grid(alpha=0.15, axis="x")
        ax.invert_yaxis()
        ax.set_title(metric, fontsize=11)
    axes[0].legend(fontsize=7, loc="lower left", ncol=1)
    plt.suptitle("Option D — Dot plot (forest plot style)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig3_main}/layout_option_D.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved {fig3_main}/layout_option_D.png")

# option E: 4 Chong panels, no significance stars (clean bars)
def _opt_e():
    df = dfs["mfass"]
    y = df["y"].values; lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))
    subsets = ["exon", "intron", "splice_region", "splice_site"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for si, sn in enumerate(subsets):
        ax = axes[si]
        mask = loc[sn]
        n_sub = mask.sum()
        n_sdv = lab[mask].sum()
        nm = len(_opt_models)
        bw = 0.8 / nm
        for mi, metric in enumerate(_opt_metrics):
            for mj, m in enumerate(_opt_models):
                col = MODEL_COLS[m]
                pred = df[col].values
                r, rho, auprc = get_metrics(y[mask], pred[mask], lab[mask])
                val = {"Pearson": r, "Spearman": rho, "AUPRC": auprc}[metric]
                x = mi + (mj - nm/2 + 0.5) * bw
                ax.bar(x, val, bw, color=get_color(m), edgecolor="black", linewidth=0.3,
                       label=model_names.get(m, m) if mi == 0 else "")
        ax.set_xticks(range(len(_opt_metrics)))
        ax.set_xticklabels(_opt_metrics, fontsize=8)
        ax.set_title(f"{LOCATION_LABELS.get(sn, sn)}\n(n={n_sub:,}, {n_sdv} SDV)", fontsize=9)
        ax.grid(alpha=0.15, axis="y")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.text(0.5, -0.02, "Statistical comparisons in supplemental Figure S11", ha="center",
             fontsize=9, fontstyle="italic", color="#666666")
    plt.suptitle("Option E — 4 Chong categories, no stars (significance in supplement)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig3_main}/layout_option_E.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"saved {fig3_main}/layout_option_E.png")

_opt_a()
_opt_b()
_opt_c()
_opt_d()
_opt_e()
print("all layout options saved")


from itertools import combinations
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.transforms import blended_transform_factory
import matplotlib.patches as mpatches

_cs_models = MODEL_LIST
_cs_pairs = list(combinations(_cs_models, 2))
_cs_short = {"pangolin": "Pang", "pangolin_v2": "Pang-v2", "spliceai": "SA",
             "splicetransformer": "ST", "sphaec_ref": "SPL", "sphaec_var": "SPL-v"}
_cs_pair_labels = [f"{_cs_short[a]} vs {_cs_short[b]}" for a, b in _cs_pairs]

# 20 contexts: 2 datasets × 2 filter states × 5 subsets
_cs_contexts = [
    (ds, filt, loc)
    for ds in ["vexseq", "mfass"]
    for filt in ["full", "filtered"]
    for loc in LOCATION_SUBSETS
]
_cs_metrics = ["Pearson", "Spearman", "AUPRC"]

# correction methods: (column title, source dict, sig key)
_cs_corrections = [
    ("Holm k=4\n(standard)", _boot_results, "sig_holm_k4"),
    ("Holm k=10\n(standard)", _boot_results, "sig_holm_k10"),
    ("Bonf k=4\n(standard)", _boot_results, "sig_bonf_k4"),
    ("Holm k=10\n(cluster)", _cboot_results, "sig_holm_k10"),
]

_star_to_int = {"": 0, "*": 1, "**": 2, "***": 3}

def _cs_get_sig(source, ctx_key, pair, metric, sig_key):
    """extract significance level as integer"""
    if ctx_key not in source:
        return 0
    res = source[ctx_key]
    for key in [pair, pair[::-1]]:
        if key in res and metric in res[key]:
            return _star_to_int.get(res[key][metric].get(sig_key, ""), 0)
    return 0

# build (10, 12) matrices per (metric, correction)
_cs_mats = {}
for mi, metric in enumerate(_cs_metrics):
    for ci, (_, source, sig_key) in enumerate(_cs_corrections):
        mat = np.zeros((len(_cs_pairs), len(_cs_contexts)), dtype=int)
        for pi, pair in enumerate(_cs_pairs):
            for xi, ctx in enumerate(_cs_contexts):
                mat[pi, xi] = _cs_get_sig(source, ctx, pair, metric, sig_key)
        _cs_mats[(mi, ci)] = mat

# total differences per column
_cs_col_diffs = {}
for ci in range(1, 4):
    _cs_col_diffs[ci] = sum(
        np.sum(_cs_mats[(mi, ci)] != _cs_mats[(mi, 0)]) for mi in range(3)
    )

# colormap: ns=light gray, *=light teal, **=medium teal, ***=dark teal
_cs_cmap = ListedColormap(["#e8e8e8", "#b2dfdb", "#4db6ac", "#00695c"])
_cs_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _cs_cmap.N)

fig, axes = plt.subplots(3, 4, figsize=(16, max(9, len(_cs_pairs) * 0.6 + 3)))

for mi, metric in enumerate(_cs_metrics):
    ref = _cs_mats[(mi, 0)]
    for ci, (col_title, _, _) in enumerate(_cs_corrections):
        ax = axes[mi, ci]
        mat = _cs_mats[(mi, ci)]
        ax.imshow(mat, cmap=_cs_cmap, norm=_cs_norm, aspect="auto", interpolation="nearest")

        # disagreement borders (black outline where sig differs from Holm k=4)
        for pi in range(len(_cs_pairs)):
            for xi in range(len(_cs_contexts)):
                if ci > 0 and mat[pi, xi] != ref[pi, xi]:
                    ax.add_patch(plt.Rectangle(
                        (xi - 0.5, pi - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.5
                    ))

        # column titles (top row only)
        if mi == 0:
            title = col_title
            if ci > 0:
                title += f"\n({_cs_col_diffs[ci]} differ)"
            ax.set_title(title, fontsize=9)

        # y-axis: pair labels (left column only)
        ax.set_yticks(range(len(_cs_pairs)))
        if ci == 0:
            ax.set_yticklabels(_cs_pair_labels, fontsize=7)
        else:
            ax.set_yticklabels([])

        # x-axis: context labels (bottom row only)
        _n_loc = len(LOCATION_SUBSETS)
        _n_ctx = len(_cs_contexts)
        ax.set_xticks(range(_n_ctx))
        if mi == 2:
            # short labels: a=all, e=exon, i=intron, r=splice_region, s=splice_site
            _loc_short = {"all": "a", "exon": "e", "intron": "i",
                          "splice_region": "r", "splice_site": "s"}
            _tick_labels = [_loc_short.get(loc, loc[0]) for _, _, loc in _cs_contexts]
            ax.set_xticklabels(_tick_labels, fontsize=6)
            # group labels below ticks
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            _groups = ["V.full", "V.filt", "M.full", "M.filt"]
            for gi, gl in enumerate(_groups):
                gx = gi * _n_loc + (_n_loc - 1) / 2
                ax.text(gx, -0.18, gl, ha="center", va="top", fontsize=7,
                        transform=trans, clip_on=False)
        else:
            ax.set_xticklabels([])

        # metric label on right margin
        if ci == 3:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(metric, fontsize=10, rotation=270, labelpad=15)

        # group separators (white lines between dataset/filter groups)
        for gx in [3.5, 7.5, 11.5]:
            ax.axvline(gx, color="white", linewidth=2)

# legend
_cs_legend = [
    mpatches.Patch(facecolor="#e8e8e8", edgecolor="gray", label="ns"),
    mpatches.Patch(facecolor="#b2dfdb", edgecolor="gray", label="*"),
    mpatches.Patch(facecolor="#4db6ac", edgecolor="gray", label="**"),
    mpatches.Patch(facecolor="#00695c", edgecolor="gray", label="***"),
    mpatches.Patch(facecolor="none", edgecolor="black", linewidth=1.5,
                   label="differs from Holm k=4"),
]
fig.legend(handles=_cs_legend, loc="lower center", ncol=5, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Correction Sensitivity: Reporter Assay Bootstrap Significance",
             fontsize=12, y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig(f"{fig3_sup}/correction_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{fig3_sup}/correction_sensitivity.pdf", bbox_inches="tight")
plt.show()
print(f"saved {fig3_sup}/correction_sensitivity.png")

# summary
for ci, (title, _, _) in enumerate(_cs_corrections):
    if ci == 0:
        continue
    print(f"{title.replace(chr(10), ' ')}: {_cs_col_diffs[ci]} cells differ from Holm k=4")


# ==========================================================================
# REF/ALT INCLUSION ANALYSIS
# ==========================================================================
# phase 1: load raw ref/alt scores for each model
# phase 2: merge measured ref inclusion from original data
# phase 3: correlation tables
# phase 4: figures

print("=" * 60)
print("REF/ALT INCLUSION ANALYSIS")
print("=" * 60)

# --- phase 1: load raw ref/alt scores ---
raw_dfs = {}
for name, cfg in datasets.items():
    raw_scores = {}
    for m in models:
        if m == "pangolin": fname = f"{cfg['prefix']}_pang.h5"
        elif m == "pangolin_v2": fname = f"{cfg['prefix']}_pang_v2.h5"
        elif m == "spliceai": fname = f"{cfg['prefix']}_sa.h5"
        elif m == "splicetransformer": fname = f"{cfg['prefix']}_spt.h5"
        else: fname = f"{cfg['prefix']}_{m}.h5"

        path = Path(cfg["path"]) / fname
        if not path.exists():
            continue

        scores = load_raw_scores(path)
        scores.columns = [f"{m}_{c}" for c in scores.columns]
        raw_scores[m] = scores

    # apply same mfass filter as main loader
    n_main = len(dfs[name])

    # get the delta_psi mask from the first scored h5 (same for all models)
    _dpsi_mask = None
    if name == "mfass":
        for m in raw_scores:
            _first_h5 = raw_scores[m]
            if len(_first_h5) > n_main:
                # read meta from this model's h5 to get delta_psi
                if m == "pangolin": _fn = f"{cfg['prefix']}_pang.h5"
                elif m == "pangolin_v2": _fn = f"{cfg['prefix']}_pang_v2.h5"
                elif m == "spliceai": _fn = f"{cfg['prefix']}_sa.h5"
                elif m == "splicetransformer": _fn = f"{cfg['prefix']}_spt.h5"
                else: _fn = f"{cfg['prefix']}_{m}.h5"
                _meta_tmp, _ = load_scores_as_deltas(Path(cfg["path"]) / _fn)
                _dpsi_mask = _meta_tmp["delta_psi"].notna().values
                break

    for m, sdf in raw_scores.items():
        if len(sdf) != n_main and _dpsi_mask is not None and len(sdf) == len(_dpsi_mask):
            raw_scores[m] = sdf[_dpsi_mask].reset_index(drop=True)

    raw_dfs[name] = raw_scores
    print(f"{name}: loaded raw scores for {len(raw_scores)} models")

# --- compute combined ref/alt columns (strand-aware) ---
for name in dfs:
    df = dfs[name]
    is_plus = (df["strand"].values == "+")
    raw = raw_dfs[name]

    def _strand_sel(col_start, col_end):
        return np.where(is_plus, col_start, col_end)

    # spliceai ref/alt: strand-aware acceptor + donor
    if "spliceai" in raw:
        rdf = raw["spliceai"]
        for tag in ["ref", "alt"]:
            acc = _strand_sel(rdf[f"spliceai_acceptor_exon_start_{tag}"].values,
                              rdf[f"spliceai_acceptor_exon_end_{tag}"].values)
            don = _strand_sel(rdf[f"spliceai_donor_exon_end_{tag}"].values,
                              rdf[f"spliceai_donor_exon_start_{tag}"].values)
            df[f"spliceai_cls_{tag}"] = (acc + don) / 2

    # splicetransformer ref/alt: strand-aware acceptor + donor
    if "splicetransformer" in raw:
        rdf = raw["splicetransformer"]
        for tag in ["ref", "alt"]:
            acc = _strand_sel(rdf[f"splicetransformer_acceptor_exon_start_{tag}"].values,
                              rdf[f"splicetransformer_acceptor_exon_end_{tag}"].values)
            don = _strand_sel(rdf[f"splicetransformer_donor_exon_end_{tag}"].values,
                              rdf[f"splicetransformer_donor_exon_start_{tag}"].values)
            df[f"splicetransformer_cls_{tag}"] = (acc + don) / 2

    # pangolin / pangolin_v2 ref/alt: avg boundaries then max across tissues
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

    # sphaec ref/alt: strand-aware cls head
    for v in ["ref", "var"]:
        p = f"sphaec_{v}"
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

# ref/alt column mapping (matching MODEL_COLS pattern)
REF_COLS = {
    "pangolin": "pangolin_max_p_splice_ref",
    "pangolin_v2": "pangolin_v2_max_p_splice_ref",
    "spliceai": "spliceai_cls_ref",
    "splicetransformer": "splicetransformer_cls_ref",
    "sphaec_ref": "sphaec_ref_cls_ref",
    "sphaec_var": "sphaec_var_cls_ref",
}
ALT_COLS = {
    "pangolin": "pangolin_max_p_splice_alt",
    "pangolin_v2": "pangolin_v2_max_p_splice_alt",
    "spliceai": "spliceai_cls_alt",
    "splicetransformer": "splicetransformer_cls_alt",
    "sphaec_ref": "sphaec_ref_cls_alt",
    "sphaec_var": "sphaec_var_cls_alt",
}

# verify columns exist
for name, df in dfs.items():
    for m in MODEL_LIST:
        for tag, cols in [("ref", REF_COLS), ("alt", ALT_COLS)]:
            if cols[m] in df.columns:
                n_valid = np.isfinite(df[cols[m]].values).sum()
                print(f"  {name} {m} {tag}: {n_valid:,} valid")

# --- phase 2: merge measured ref inclusion ---

# vex-seq: merge HepG2_ref_psi from original CSV
vex_df = dfs["vexseq"]
# reload original CSVs to get ref_psi
_train = pd.read_csv("vex_seq/data/train.csv")
_test = pd.read_csv("vex_seq/data/test.csv")
_truth = pd.read_csv("vex_seq/data/truth.tsv", sep="\t")

_train["chrom"] = _train["seqnames"].astype(str)
_train["pos"] = _train["hg19_variant_position"].astype(int)
_train["ref"] = _train["reference"].astype(str)
_train["alt"] = _train["variant"].astype(str)

_test["chrom"] = _test["seqnames"].astype(str)
_test["pos"] = _test["hg19_variant_position"].astype(int)
_test["ref"] = _test["reference"].astype(str)
_test["alt"] = _test["variant"].astype(str)

_vex_ref_psi = pd.concat([
    _train[["chrom", "pos", "ref", "alt", "HepG2_ref_psi"]],
    _test[["chrom", "pos", "ref", "alt", "HepG2_ref_psi"]],
], ignore_index=True).drop_duplicates(subset=["chrom", "pos", "ref", "alt"])

_merge_key = ["chrom", "pos", "ref", "alt"]
_before = len(vex_df)
vex_df = vex_df.merge(_vex_ref_psi, on=_merge_key, how="left")
assert len(vex_df) == _before, f"merge changed row count: {_before} -> {len(vex_df)}"
# convert to 0-1 scale
vex_df["measured_ref_psi"] = vex_df["HepG2_ref_psi"].values / 100.0
vex_df.drop(columns=["HepG2_ref_psi"], inplace=True)
n_ref = np.isfinite(vex_df["measured_ref_psi"].values).sum()
print(f"vexseq: merged ref PSI for {n_ref:,} / {len(vex_df):,} variants")
dfs["vexseq"] = vex_df

# mfass: merge nat_v2_index, v2_index, v1_dpsi, v2_dpsi_R1, v2_dpsi_R2
mfass_df = dfs["mfass"]
_mfass_raw = pd.read_csv("mfass/data/snv_data_clean.txt", sep="\t")
_mfass_mut = _mfass_raw[_mfass_raw["category"] == "mutant"].copy()
_mfass_mut = _mfass_mut.rename(columns={"chr": "chrom", "ref_allele": "ref", "alt_allele": "alt", "snp_position": "pos"})
_mfass_mut["pos"] = _mfass_mut["pos"].astype(int)

_mfass_extra_cols = ["nat_v2_index", "v2_index", "v1_dpsi", "v2_dpsi_R1", "v2_dpsi_R2"]
_mfass_merge = _mfass_mut[["chrom", "pos", "ref", "alt"] + _mfass_extra_cols].drop_duplicates(
    subset=["chrom", "pos", "ref", "alt"])

_before = len(mfass_df)
mfass_df = mfass_df.merge(_mfass_merge, on=["chrom", "pos", "ref", "alt"], how="left")
assert len(mfass_df) == _before, f"merge changed row count: {_before} -> {len(mfass_df)}"

# rename for clarity
mfass_df["measured_ref_inclusion"] = mfass_df["nat_v2_index"].values
mfass_df["measured_alt_inclusion"] = mfass_df["v2_index"].values

for col in _mfass_extra_cols:
    n_valid = mfass_df[col].notna().sum()
    print(f"mfass: {col} — {n_valid:,} / {len(mfass_df):,} valid")
dfs["mfass"] = mfass_df

# --- phase 3: correlation tables ---

def compute_corrs(y_true, y_pred):
    """pearson, spearman, r2 for finite pairs"""
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


print("\n" + "=" * 60)
print("CORRELATION TABLES")
print("=" * 60)

# table 1: ref prediction vs measured ref inclusion
ref_corr_rows = []
for name, df in dfs.items():
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
        ref_corr_rows.append({
            "dataset": name, "model": model_names.get(m, m),
            "pearson": r, "spearman": rho, "r2": r2, "n": n
        })

ref_corr_df = pd.DataFrame(ref_corr_rows)
print("\nRef prediction vs measured ref inclusion:")
print(ref_corr_df.to_string(index=False, float_format="%.3f"))

# table 2: alt prediction vs measured alt inclusion (MFASS only)
alt_corr_rows = []
df = dfs["mfass"]
y_alt = df["measured_alt_inclusion"].values
for m in MODEL_LIST:
    col = ALT_COLS.get(m)
    if col is None or col not in df.columns:
        continue
    pred = df[col].values
    r, rho, r2, n = compute_corrs(y_alt, pred)
    alt_corr_rows.append({
        "model": model_names.get(m, m),
        "pearson": r, "spearman": rho, "r2": r2, "n": n
    })

alt_corr_df = pd.DataFrame(alt_corr_rows)
print("\nAlt prediction vs measured alt inclusion (MFASS):")
print(alt_corr_df.to_string(index=False, float_format="%.3f"))

# table 3: delta score vs replicate dpsi (MFASS)
rep_corr_rows = []
df = dfs["mfass"]
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
        rep_corr_rows.append({
            "target": target_name, "model": model_names.get(m, m),
            "pearson": r, "spearman": rho, "n": n
        })

rep_corr_df = pd.DataFrame(rep_corr_rows)
print("\nDelta score vs replicate ΔPSI (MFASS):")
print(rep_corr_df.to_string(index=False, float_format="%.3f"))

# save tables
ref_corr_df.to_csv(f"{out_dir}/ref_inclusion_corr.csv", index=False, float_format="%.4f")
alt_corr_df.to_csv(f"{out_dir}/alt_inclusion_corr.csv", index=False, float_format="%.4f")
rep_corr_df.to_csv(f"{out_dir}/replicate_corr.csv", index=False, float_format="%.4f")
print(f"\nsaved correlation tables to {out_dir}/")

# --- phase 4: figures ---

# figure: ref prediction vs measured ref inclusion (scatter per model)
for name, df in dfs.items():
    if name == "vexseq":
        y_ref = df["measured_ref_psi"].values
        y_label = "Measured ref PSI"
    elif name == "mfass":
        y_ref = df["measured_ref_inclusion"].values
        y_label = "Measured ref inclusion"
    else:
        continue

    avail = [m for m in MODEL_LIST if REF_COLS[m] in df.columns]
    n_mod = len(avail)
    if n_mod == 0:
        continue

    ncols = min(n_mod, 3)
    nrows = (n_mod + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for i, m in enumerate(avail):
        ax = axes[i // ncols, i % ncols]
        pred = df[REF_COLS[m]].values
        mask = np.isfinite(y_ref) & np.isfinite(pred)
        x, y_plot = pred[mask], y_ref[mask]

        # log-density scatter
        try:
            xy = np.vstack([x, y_plot])
            z = gaussian_kde(xy)(xy)
            order = z.argsort()
            ax.scatter(x[order], y_plot[order], c=np.log10(z[order]), s=3, alpha=0.6, cmap="viridis", rasterized=True)
        except Exception:
            ax.scatter(x, y_plot, s=3, alpha=0.3, color=get_color(m), rasterized=True)

        r, rho, r2, n = compute_corrs(y_ref, pred)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(model_names.get(m, m))
        ax.set_xlabel("Predicted ref score")
        ax.set_ylabel(y_label)
        ax.text(0.03, 0.97, f"$r$={r:.3f}\n$\\rho$={rho:.3f}\n$R^2$={r2:.3f}\n$n$={n:,}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # hide unused axes
    for i in range(n_mod, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle(f"{dataset_names[name]} — Ref Prediction vs Measured Ref Inclusion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_ref_inclusion_scatter.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{name}_ref_inclusion_scatter.pdf", bbox_inches="tight")
    plt.show()
    print(f"saved {out_dir}/{name}_ref_inclusion_scatter.png")

# figure: alt prediction vs measured alt inclusion (MFASS only)
df = dfs["mfass"]
y_alt = df["measured_alt_inclusion"].values
avail = [m for m in MODEL_LIST if ALT_COLS[m] in df.columns]
n_mod = len(avail)
if n_mod > 0:
    ncols = min(n_mod, 3)
    nrows = (n_mod + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for i, m in enumerate(avail):
        ax = axes[i // ncols, i % ncols]
        pred = df[ALT_COLS[m]].values
        mask = np.isfinite(y_alt) & np.isfinite(pred)
        x, y_plot = pred[mask], y_alt[mask]

        try:
            xy = np.vstack([x, y_plot])
            z = gaussian_kde(xy)(xy)
            order = z.argsort()
            ax.scatter(x[order], y_plot[order], c=np.log10(z[order]), s=3, alpha=0.6, cmap="viridis", rasterized=True)
        except Exception:
            ax.scatter(x, y_plot, s=3, alpha=0.3, color=get_color(m), rasterized=True)

        r, rho, r2, n = compute_corrs(y_alt, pred)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(model_names.get(m, m))
        ax.set_xlabel("Predicted alt score")
        ax.set_ylabel("Measured alt inclusion")
        ax.text(0.03, 0.97, f"$r$={r:.3f}\n$\\rho$={rho:.3f}\n$R^2$={r2:.3f}\n$n$={n:,}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    for i in range(n_mod, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle("MFASS — Alt Prediction vs Measured Alt Inclusion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mfass_alt_inclusion_scatter.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/mfass_alt_inclusion_scatter.pdf", bbox_inches="tight")
    plt.show()
    print(f"saved {out_dir}/mfass_alt_inclusion_scatter.png")

# figure: replicate consistency bar chart (MFASS)
df = dfs["mfass"]
_rep_targets_bar = [
    ("v2 (main)", df["y"].values),
    ("v2 R1", df["v2_dpsi_R1"].values if "v2_dpsi_R1" in df.columns else None),
    ("v2 R2", df["v2_dpsi_R2"].values if "v2_dpsi_R2" in df.columns else None),
    ("v1", df["v1_dpsi"].values if "v1_dpsi" in df.columns else None),
]
_rep_targets_bar = [(n, y) for n, y in _rep_targets_bar if y is not None]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for mi, metric in enumerate(["Pearson", "Spearman"]):
    ax = axes[mi]
    x_labels = []
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
            ax.bar(x_pos, val, color=get_color(m), edgecolor="black", linewidth=0.5)
            if ti == 0:
                x_labels.append(model_names.get(m, m))

    # group labels
    n_m = len(MODEL_LIST)
    group_centers = [ti * (n_m + 1) + (n_m - 1) / 2 for ti in range(len(_rep_targets_bar))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([t[0] for t in _rep_targets_bar], fontsize=11)
    ax.set_ylabel(metric)
    ax.set_title(metric)

# legend
from matplotlib.patches import Patch as _Patch
_leg = [_Patch(facecolor=get_color(m), edgecolor="black", label=model_names.get(m, m)) for m in MODEL_LIST]
fig.legend(handles=_leg, loc="upper center", ncol=len(MODEL_LIST), fontsize=9, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("MFASS — Delta Score vs Replicate ΔPSI", fontsize=14, fontweight="bold", y=1.06)
plt.tight_layout()
plt.savefig(f"{out_dir}/mfass_replicate_consistency.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_dir}/mfass_replicate_consistency.pdf", bbox_inches="tight")
plt.show()
print(f"saved {out_dir}/mfass_replicate_consistency.png")

# figure: error vs baseline inclusion
for name, df in dfs.items():
    if name == "vexseq":
        baseline = df["measured_ref_psi"].values
        bl_label = "Ref PSI"
    elif name == "mfass":
        baseline = df["measured_ref_inclusion"].values
        bl_label = "Ref inclusion (nat_v2_index)"
    else:
        continue

    avail = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]
    if not avail:
        continue

    ncols = min(len(avail), 3)
    nrows = (len(avail) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    y = df["y"].values
    for i, m in enumerate(avail):
        ax = axes[i // ncols, i % ncols]
        pred = df[MODEL_COLS[m]].values
        err = np.abs(pred - y)
        mask = np.isfinite(baseline) & np.isfinite(err)
        bm, em = baseline[mask], err[mask]

        ax.scatter(bm, em, s=3, alpha=0.2, color=get_color(m), rasterized=True)

        # binned mean trend
        n_bins = 20
        bin_edges = np.linspace(np.nanmin(bm), np.nanmax(bm), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = np.array([
            np.nanmean(em[(bm >= bin_edges[j]) & (bm < bin_edges[j + 1])])
            for j in range(n_bins)
        ])
        valid_bins = np.isfinite(bin_means)
        ax.plot(bin_centers[valid_bins], bin_means[valid_bins], "k-", lw=2)

        rho = stats.spearmanr(bm, em)[0] if len(bm) > 2 else np.nan
        ax.set_title(model_names.get(m, m))
        ax.set_xlabel(bl_label)
        ax.set_ylabel("|Prediction error|")
        ax.text(0.03, 0.97, f"$\\rho$={rho:.3f}\n$n$={mask.sum():,}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    for i in range(len(avail), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle(f"{dataset_names[name]} — Prediction Error vs Baseline Inclusion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}_error_vs_baseline.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/{name}_error_vs_baseline.pdf", bbox_inches="tight")
    plt.show()
    print(f"saved {out_dir}/{name}_error_vs_baseline.png")

print("\n" + "=" * 60)
print("REF/ALT ANALYSIS COMPLETE")
print("=" * 60)


# ==========================================================================
# PER-EXON MODEL PERFORMANCE ANALYSIS
# ==========================================================================
# group variants by exon, compute per-exon metrics, rank models per exon

print("\n" + "=" * 60)
print("PER-EXON MODEL PERFORMANCE")
print("=" * 60)

MIN_VARIANTS_PER_EXON = 5

for name, df in dfs.items():
    y = df["y"].values
    label = df["label"].values
    pos = df["pos"].values.astype(int)
    exon_start = df["exon_start"].values.astype(int)
    exon_end = df["exon_end"].values.astype(int)

    # exon key for grouping
    exon_key = [f"{es}_{ee}" for es, ee in zip(exon_start, exon_end)]
    df["_exon_key"] = exon_key

    mods = [m for m in MODEL_LIST if MODEL_COLS[m] in df.columns]

    # --- per-exon metrics ---
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
    exon_df.to_csv(f"{out_dir}/{name}_per_exon_metrics.csv", index=False, float_format="%.4f")
    print(f"\n{dataset_names[name]}: {len(exon_df):,} exons with >= {MIN_VARIANTS_PER_EXON} variants")
    print(f"  median variants/exon: {exon_df['n_variants'].median():.0f}")
    print(f"  saved {out_dir}/{name}_per_exon_metrics.csv")

    # --- rank models per exon ---
    for metric_suffix, ascending in [("pearson", False), ("spearman", False), ("mae", True)]:
        metric_cols = [f"{m}_{metric_suffix}" for m in mods]
        vals = exon_df[metric_cols].values
        if ascending:
            ranks = np.argsort(np.argsort(vals, axis=1), axis=1) + 1
        else:
            ranks = np.argsort(np.argsort(-vals, axis=1), axis=1) + 1
        for i, m in enumerate(mods):
            exon_df[f"{m}_{metric_suffix}_rank"] = ranks[:, i]

    # --- figure: rank frequency heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(mods) * 0.5 + 1)))
    for ai, metric_suffix in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[ai]
        rank_matrix = np.zeros((len(mods), len(mods)), dtype=int)
        for mi, m in enumerate(mods):
            col = f"{m}_{metric_suffix}_rank"
            for rank_val in range(1, len(mods) + 1):
                rank_matrix[mi, rank_val - 1] = (exon_df[col] == rank_val).sum()

        # normalize to fractions
        rank_frac = rank_matrix / rank_matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(rank_frac, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([f"#{i+1}" for i in range(len(mods))], fontsize=9)
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels([model_names[m] for m in mods], fontsize=9)
        ax.set_xlabel("Rank")
        ax.set_title(metric_suffix.capitalize())
        # annotate cells
        for mi in range(len(mods)):
            for ri in range(len(mods)):
                pct = rank_frac[mi, ri]
                ax.text(ri, mi, f"{100*pct:.0f}%", ha="center", va="center",
                        fontsize=7, color="white" if pct > 0.5 else "black")

    plt.suptitle(f"{dataset_names[name]} — Per-Exon Model Rankings (n={len(exon_df):,} exons)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_per_exon_rank_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_per_exon_rank_heatmap.pdf", bbox_inches="tight")
    plt.show()
    print(f"  saved {fig3_sup}/{name}_per_exon_rank_heatmap.png")

    # --- figure: mean rank bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ai, metric_suffix in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[ai]
        mean_ranks = [exon_df[f"{m}_{metric_suffix}_rank"].mean() for m in mods]
        bars = ax.bar(range(len(mods)), mean_ranks,
                      color=[get_color(m) for m in mods], edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([model_names[m] for m in mods], fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Mean rank")
        ax.set_title(metric_suffix.capitalize())
        ax.set_ylim(0.5, len(mods) + 0.5)
        ax.axhline(len(mods) / 2 + 0.5, color="gray", ls="--", lw=0.5, alpha=0.5)
        for bi, v in enumerate(mean_ranks):
            ax.text(bi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle(f"{dataset_names[name]} — Mean Per-Exon Rank (lower=better, n={len(exon_df):,})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_per_exon_mean_rank.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_per_exon_mean_rank.pdf", bbox_inches="tight")
    plt.show()
    print(f"  saved {fig3_sup}/{name}_per_exon_mean_rank.png")

    # --- figure: win/loss matrix ---
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(mods) * 0.7 + 1)))
    for ai, metric_suffix in enumerate(["pearson", "spearman", "mae"]):
        ax = axes[ai]
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

        # normalize by number of exons
        n_exons = len(exon_df)
        win_frac = win_matrix / n_exons
        im = ax.imshow(win_frac, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(mods)))
        ax.set_xticklabels([model_names[m] for m in mods], fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels([model_names[m] for m in mods], fontsize=8)
        ax.set_xlabel("Column model")
        ax.set_ylabel("Row model wins →")
        ax.set_title(metric_suffix.capitalize())
        for mi in range(len(mods)):
            for mj in range(len(mods)):
                if mi == mj:
                    ax.text(mj, mi, "—", ha="center", va="center", fontsize=7)
                else:
                    ax.text(mj, mi, f"{win_matrix[mi, mj]}", ha="center", va="center",
                            fontsize=7, color="white" if win_frac[mi, mj] > 0.6 else "black")

    plt.suptitle(f"{dataset_names[name]} — Per-Exon Win Counts (row beats column, n={n_exons:,})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_per_exon_win_matrix.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_per_exon_win_matrix.pdf", bbox_inches="tight")
    plt.show()
    print(f"  saved {fig3_sup}/{name}_per_exon_win_matrix.png")

    # --- figure: what makes an exon hard? ---
    # correlate per-exon MAE (avg across models) with exon properties
    avg_mae = exon_df[[f"{m}_mae" for m in mods]].mean(axis=1)
    properties = [
        ("Exon width (bp)", exon_df["width"].values),
        ("Variants tested", exon_df["n_variants"].values),
        ("Mean |ΔPSI|", exon_df["mean_abs_dpsi"].values),
        ("SDV fraction", exon_df["n_sdv"].values / exon_df["n_variants"].values),
    ]

    fig, axes = plt.subplots(1, len(properties), figsize=(4 * len(properties), 4))
    for ax, (prop_name, prop_vals) in zip(axes, properties):
        fin = np.isfinite(prop_vals) & np.isfinite(avg_mae.values)
        ax.scatter(prop_vals[fin], avg_mae.values[fin], s=20, alpha=0.5,
                   color="#4a90d9", edgecolors="none")
        if fin.sum() > 3:
            rho = stats.spearmanr(prop_vals[fin], avg_mae.values[fin])[0]
            ax.text(0.03, 0.97, f"$\\rho$={rho:.2f}", transform=ax.transAxes,
                    fontsize=9, va="top")
        ax.set_xlabel(prop_name)
        ax.set_ylabel("Mean MAE (all models)")
        ax.grid(alpha=0.2)

    plt.suptitle(f"{dataset_names[name]} — Exon Properties vs Model Difficulty",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_exon_difficulty.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_exon_difficulty.pdf", bbox_inches="tight")
    plt.show()
    print(f"  saved {fig3_sup}/{name}_exon_difficulty.png")

    # --- figure: per-exon model concordance ---
    # for each pair of models, scatter per-exon Pearson
    from itertools import combinations as _comb
    pairs = list(_comb(mods, 2))
    n_pairs = len(pairs)
    ncols = min(5, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)
    for pi, (ma, mb) in enumerate(pairs):
        ax = axes[pi // ncols, pi % ncols]
        xa = exon_df[f"{ma}_pearson"].values
        xb = exon_df[f"{mb}_pearson"].values
        fin = np.isfinite(xa) & np.isfinite(xb)
        ax.scatter(xa[fin], xb[fin], s=15, alpha=0.5, color="#555555", edgecolors="none")
        lims = [min(np.nanmin(xa[fin]), np.nanmin(xb[fin])) - 0.05,
                max(np.nanmax(xa[fin]), np.nanmax(xb[fin])) + 0.05]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.4)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(model_names[ma], fontsize=8)
        ax.set_ylabel(model_names[mb], fontsize=8)
        rho = stats.spearmanr(xa[fin], xb[fin])[0] if fin.sum() > 3 else np.nan
        ax.text(0.03, 0.97, f"$\\rho$={rho:.2f}", transform=ax.transAxes, fontsize=8, va="top")
        ax.set_aspect("equal")
    for pi in range(n_pairs, nrows * ncols):
        axes[pi // ncols, pi % ncols].set_visible(False)
    plt.suptitle(f"{dataset_names[name]} — Per-Exon Pearson Concordance (n={len(exon_df):,})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_per_exon_concordance.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_per_exon_concordance.pdf", bbox_inches="tight")
    plt.show()
    print(f"  saved {fig3_sup}/{name}_per_exon_concordance.png")

    # print summary table
    print(f"\n{dataset_names[name]} — Mean rank across exons:")
    print(f"  {'model':<22} {'pearson':>8} {'spearman':>8} {'mae':>8} {'#1 wins (r)':>12}")
    print(f"  {'-'*60}")
    for m in mods:
        mr_p = exon_df[f"{m}_pearson_rank"].mean()
        mr_s = exon_df[f"{m}_spearman_rank"].mean()
        mr_m = exon_df[f"{m}_mae_rank"].mean()
        n_first = (exon_df[f"{m}_pearson_rank"] == 1).sum()
        print(f"  {model_names[m]:<22} {mr_p:>8.2f} {mr_s:>8.2f} {mr_m:>8.2f} {n_first:>12,}")

    df.drop(columns=["_exon_key"], inplace=True)

print("\n" + "=" * 60)
print("PER-EXON ANALYSIS COMPLETE")
print("=" * 60)


# ==========================================================================
# CLS vs SSU/Usage Head Comparison
# ==========================================================================
# for each variant, compare |delta| from two output heads (e.g. CLS vs SSU)
# scatter shows which head detects which SDVs — directly analogous to the
# sQTL margin scatter but adapted for continuous reporter assay data
#
# |delta_A| vs |delta_B| colored by SDV status
# SDVs far from origin = detected; near-diagonal = both heads agree;
# off-diagonal = one head detects what the other misses

print("\n" + "=" * 60)
print("CLS vs SSU/USAGE HEAD COMPARISON")
print("=" * 60)

_head_pairs = [
    ("sphaec_ref", "sphaec_ref_cls_delta", "sphaec_ref_reg_ssu_delta", "SPLAIRE", "CLS", "SSU"),
    ("sphaec_var", "sphaec_var_cls_delta", "sphaec_var_reg_ssu_delta", "SPLAIRE-var", "CLS", "SSU"),
    ("pangolin", "pangolin_max_p_splice", "pangolin_max_usage", "Pangolin", "CLS", "Usage"),
    ("pangolin_v2", "pangolin_v2_max_p_splice", "pangolin_v2_max_usage", "Pangolin v2", "CLS", "Usage"),
]

_loc_colors = {"splice_site": "#D55E00", "splice_region": "#E69F00", "exon": "#0072B2", "intron": "#009E73"}
_loc_labels = {"splice_site": "Splice site", "splice_region": "Splice region", "exon": "Exon", "intron": "Intron"}

for name, df in dfs.items():
    y = df["y"].values
    lab = df["label"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))

    for model_base, col_a, col_b, model_label, a_name, b_name in _head_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue

        pred_a = df[col_a].values
        pred_b = df[col_b].values
        valid = np.isfinite(pred_a) & np.isfinite(pred_b) & np.isfinite(y)

        abs_a = np.abs(pred_a[valid])
        abs_b = np.abs(pred_b[valid])

        fig, ax = plt.subplots(figsize=(7, 7))

        # draw each location category, splice sites on top
        for loc_name in ["intron", "exon", "splice_region", "splice_site"]:
            mask = loc[loc_name][valid]
            n = mask.sum()
            ax.scatter(abs_a[mask], abs_b[mask], s=10, alpha=0.3,
                      color=_loc_colors[loc_name],
                      label=f"{_loc_labels[loc_name]} (n={n:,})",
                      rasterized=True)

        lim = max(abs_a.max(), abs_b.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"|delta| {a_name}")
        ax.set_ylabel(f"|delta| {b_name}")
        ax.set_title(f"{model_label} — {dataset_names[name]}")

        # annotate AUPRC per head
        auprc_a = average_precision_score(lab[valid], abs_a) if lab[valid].sum() > 0 else np.nan
        auprc_b = average_precision_score(lab[valid], abs_b) if lab[valid].sum() > 0 else np.nan
        ax.text(0.03, 0.97, f"{a_name} AUPRC: {auprc_a:.3f}\n{b_name} AUPRC: {auprc_b:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{fig3_sup}/{name}_{model_base}_cls_vs_ssu.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig3_sup}/{name}_{model_base}_cls_vs_ssu.pdf", bbox_inches="tight")
        plt.show()
        plt.close()

        print(f"  {model_label} — {dataset_names[name]}: {a_name} AUPRC={auprc_a:.3f}, {b_name} AUPRC={auprc_b:.3f}")


# ==========================================================================
# Tissue-Specific Head Evaluation
# ==========================================================================
# for each model family with tissue-specific heads, show per-head AUPRC
# vex-seq is HepG2 (liver) — does liver-specific head beat the aggregate?
# mfass is HEK293T — no clear tissue match, just show all ranked

print("\n" + "=" * 60)
print("TISSUE-SPECIFIC HEAD EVALUATION")
print("=" * 60)

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

for name, df in dfs.items():
    lab = df["label"].values
    expected_tissue = _CELL_TISSUE.get(name)

    # add SPT tissues dynamically from columns
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

    fig, axes = plt.subplots(1, n_families, figsize=(5 * n_families, max(4, 0.3 * max(len(c["tissues"]) for c in families.values()) + 2)))
    if n_families == 1:
        axes = [axes]

    for ax, (family, cfg_th) in zip(axes, families.items()):
        agg_col = cfg_th["aggregate"]
        if agg_col not in df.columns:
            ax.text(0.5, 0.5, f"{family}: no data", ha="center", va="center", transform=ax.transAxes)
            continue

        # compute AUPRC for each tissue head
        tissue_auprcs = {}
        for tissue, col in cfg_th["tissues"].items():
            if col in df.columns:
                vals = np.abs(df[col].values)
                tissue_auprcs[tissue] = average_precision_score(lab, vals) if lab.sum() > 0 else np.nan

        # compute aggregate AUPRC
        agg_auprc = average_precision_score(lab, np.abs(df[agg_col].values)) if lab.sum() > 0 else np.nan

        # sort by AUPRC descending
        sorted_tissues = sorted(tissue_auprcs.items(), key=lambda x: x[1])
        bar_labels = [t for t, _ in sorted_tissues] + [cfg_th["agg_label"]]
        bar_vals = [v for _, v in sorted_tissues] + [agg_auprc]

        colors = []
        for t in bar_labels:
            if t == cfg_th["agg_label"]:
                colors.append("#333333")
            elif expected_tissue and t.lower().replace("_", " ") == expected_tissue.lower():
                colors.append("#D55E00")
            else:
                colors.append(get_color(family.lower().split()[0]))

        y_pos = np.arange(len(bar_labels))
        ax.barh(y_pos, bar_vals, color=colors, alpha=0.8, height=0.7)
        for i, v in enumerate(bar_vals):
            if np.isfinite(v):
                ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=9)
        ax.set_xlabel("AUPRC")
        ax.set_title(family)
        ax.set_xlim(0, max(v for v in bar_vals if np.isfinite(v)) * 1.15 if bar_vals else 1)
        ax.grid(alpha=0.2, axis="x")

    cell_name = "HepG2 (liver)" if name == "vexseq" else "HEK293T"
    plt.suptitle(f"{dataset_names[name]} ({cell_name}) — Tissue-Specific Head AUPRC", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_tissue_head_eval.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_tissue_head_eval.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    # print summary
    print(f"\n  {dataset_names[name]} ({cell_name}):")
    for family, cfg_th in families.items():
        agg_col = cfg_th["aggregate"]
        if agg_col not in df.columns:
            continue
        agg_val = average_precision_score(lab, np.abs(df[agg_col].values)) if lab.sum() > 0 else np.nan
        best_tissue = None
        best_val = -1
        for tissue, col in cfg_th["tissues"].items():
            if col in df.columns:
                val = average_precision_score(lab, np.abs(df[col].values)) if lab.sum() > 0 else np.nan
                if np.isfinite(val) and val > best_val:
                    best_val = val
                    best_tissue = tissue
        matched_val = np.nan
        if expected_tissue:
            for tissue, col in cfg_th["tissues"].items():
                if tissue.lower().replace("_", " ") == expected_tissue.lower() and col in df.columns:
                    matched_val = average_precision_score(lab, np.abs(df[col].values)) if lab.sum() > 0 else np.nan
        print(f"    {family}: aggregate={agg_val:.3f}, best tissue={best_tissue} ({best_val:.3f})", end="")
        if expected_tissue and np.isfinite(matched_val):
            print(f", {expected_tissue}={matched_val:.3f}")
        else:
            print()


# ==========================================================================
# Location-Stratified Threshold Sweep
# ==========================================================================
# extends figure 2 (threshold sensitivity) by showing separate curves per
# variant location category (all, splice site, exon, intron)

print("\n" + "=" * 60)
print("LOCATION-STRATIFIED THRESHOLD SWEEP")
print("=" * 60)

for name, df in dfs.items():
    cols = thresh_cols[name]
    y = df["y"].values
    pos = df["pos"].values.astype(int)
    loc = get_location_masks(pos, df["exon_start"].values.astype(int), df["exon_end"].values.astype(int))

    thresholds = np.arange(0.01, 1.01, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
            ax.plot(thresholds, auprcs, lw=2, color=get_color(m), label=model_names[m])

        ax.axvline(sd_thr, color="black", ls=":", lw=1.5, alpha=0.7)
        ax.text(sd_thr + 0.01, 0.95, f"1 SD={sd_thr:.2f}", fontsize=7, va="top", rotation=90, alpha=0.7)
        ax.set_xlabel("|ΔPSI| threshold")
        ax.set_ylabel("AUPRC")
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_ylim(0, 1)
        ax.set_title(f"{LOCATION_LABELS[loc_name]} (n={mask.sum():,})")
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(f"{dataset_names[name]} — Threshold Sensitivity by Location", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{fig3_sup}/{name}_threshold_by_location.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig3_sup}/{name}_threshold_by_location.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"  {dataset_names[name]}: saved {fig3_sup}/{name}_threshold_by_location.png")


# render analysis.md to html with quarto if available
import shutil
if shutil.which("quarto"):
    print("\nrendering analysis.md with quarto...")
    subprocess.run([
        "quarto", "render", "analysis.md", "--to", "html",
        "-M", "code-fold:true",
        "-M", "toc:false",
        "-M", "code-tools:true",
        "-M", "code-copy:true",
        "-M", "embed-resources:true",
        "-M", "theme:cosmo",
    ], check=False)
    if Path("analysis.html").exists():
        print(f"rendered analysis.html ({Path('analysis.html').stat().st_size / 1024:.0f} KB)")
    else:
        print("quarto render failed, check output above")
else:
    print("\nquarto not found, skipping html render")

