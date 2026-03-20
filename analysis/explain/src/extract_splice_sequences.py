#!/usr/bin/env python3
"""
extract splice-site-centered sequences for attribution analysis

reads processed_splicing_matrix.tsv and splicing_matrix.tsv from each tissue
computes stats for each unique splice site:
  - ssu stats: mean, min, max, iqr, var, gini, n_samples
  - read stats: inclusion, total, mean, min, max, iqr, var

filters to:
  - test chromosomes (1, 3, 5, 7)
  - protein-coding genes
  - non-paralogous genes
"""

import os
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import h5py
from pyfaidx import Fasta
from tqdm import tqdm


TEST_CHROMS = {'chr1', 'chr3', 'chr5', 'chr7', '1', '3', '5', '7'}
MISSING_VALUE = 777.0


def gini(vals):
    """gini coefficient - handles zero sum"""
    if len(vals) < 2:
        return 0.0
    vals = np.sort(vals)
    n = len(vals)
    total = np.sum(vals)
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * vals) - (n + 1) * total) / (n * total)


def compute_array_stats(arr, missing=MISSING_VALUE):
    """vectorized stats for 2d array (rows=sites, cols=samples)"""
    # mask missing values
    mask = (arr != missing) & ~np.isnan(arr)

    n = mask.sum(axis=1)

    # replace missing with nan for nanfuncs
    arr_masked = np.where(mask, arr, np.nan)

    with np.errstate(all='ignore'):
        mean = np.nanmean(arr_masked, axis=1)
        min_val = np.nanmin(arr_masked, axis=1)
        max_val = np.nanmax(arr_masked, axis=1)
        var = np.nanvar(arr_masked, axis=1)
        q25 = np.nanpercentile(arr_masked, 25, axis=1)
        q75 = np.nanpercentile(arr_masked, 75, axis=1)
        iqr = q75 - q25

    # gini needs loop but vectorized per-row
    gini_vals = np.zeros(len(arr))
    for i in range(len(arr)):
        valid = arr_masked[i, ~np.isnan(arr_masked[i])]
        if len(valid) >= 2:
            gini_vals[i] = gini(valid)

    return {
        'n': n.astype(np.int32),
        'mean': mean.astype(np.float32),
        'min': min_val.astype(np.float32),
        'max': max_val.astype(np.float32),
        'iqr': iqr.astype(np.float32),
        'var': var.astype(np.float32),
        'gini': gini_vals.astype(np.float32),
    }


def parse_read_matrix(raw_df, sample_cols):
    """parse read count matrix from 'inc/total' format to separate arrays"""
    n_sites = len(raw_df)
    n_samples = len(sample_cols)

    inc = np.full((n_sites, n_samples), np.nan, dtype=np.float32)
    total = np.full((n_sites, n_samples), np.nan, dtype=np.float32)

    for j, col in enumerate(sample_cols):
        for i, val in enumerate(raw_df[col].values):
            if pd.isna(val) or val == '':
                continue
            try:
                parts = str(val).split('/')
                inc[i, j] = int(parts[0])
                total[i, j] = int(parts[1])
            except:
                continue

    return inc, total


def load_protein_coding_genes(path):
    """load protein-coding gene regions for position-based lookup"""
    print(f"loading protein-coding genes: {path}")
    df = pd.read_csv(path, sep='\t')
    # normalize chrom
    df['chrom'] = df['Chromosome'].str.replace('chr', '')
    df['gene'] = df['Gene_ID'].str.split('.').str[0]
    print(f"  {len(df):,} genes")
    return df[['chrom', 'Start', 'End', 'Strand', 'gene']]


def load_paralog_genes(path):
    """load genes with paralogs"""
    print(f"loading paralog genes: {path}")
    df = pd.read_csv(path, sep='\t', compression='gzip')
    has_paralog = df[df['Human paralogue gene stable ID'].notna()]['Gene stable ID'].unique()
    genes = set(has_paralog)
    print(f"  {len(genes):,} genes with paralogs")
    return genes


def map_sites_to_genes(sites_df, gene_df):
    """map splice sites to genes by position overlap"""
    print("mapping sites to genes...")

    # normalize chrom in sites
    sites_df = sites_df.copy()
    sites_df['chrom'] = sites_df['region'].str.replace('chr', '')

    # build gene lookup by chrom
    gene_by_chrom = {}
    for chrom in gene_df['chrom'].unique():
        gdf = gene_df[gene_df['chrom'] == chrom].copy()
        gene_by_chrom[chrom] = gdf

    # map each site
    genes = []
    for _, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc="mapping"):
        chrom = row['chrom']
        pos = row['site']
        strand = row['strand']

        gene = None
        if chrom in gene_by_chrom:
            gdf = gene_by_chrom[chrom]
            # find overlapping gene on same strand
            matches = gdf[(gdf['Start'] <= pos) & (gdf['End'] >= pos) & (gdf['Strand'] == strand)]
            if len(matches) > 0:
                gene = matches.iloc[0]['gene']
        genes.append(gene)

    sites_df['gene_id'] = genes
    sites_df = sites_df.drop(columns=['chrom'])

    n_mapped = sites_df['gene_id'].notna().sum()
    print(f"  mapped: {n_mapped:,} / {len(sites_df):,} sites")
    return sites_df


def filter_to_test_chroms(df):
    """filter dataframe to test chromosomes"""
    df['chrom_norm'] = df['region'].str.replace('chr', '')
    chroms_norm = {c.replace('chr', '') for c in TEST_CHROMS}
    df = df[df['chrom_norm'].isin(chroms_norm)].copy()
    df = df.drop(columns=['chrom_norm'])
    return df


def filter_informative_sites(df, sample_cols):
    """keep sites with at least one non-missing, non-zero ssu"""
    arr = df[sample_cols].values
    # valid = not 777 and not nan and > 0
    valid = (arr != MISSING_VALUE) & ~np.isnan(arr) & (arr > 0)
    has_valid = valid.any(axis=1)
    return df[has_valid].copy()


def load_tissue_data(tissue_dir, tissue_name, filter_chroms=True):
    """load and compute stats for one tissue - vectorized"""
    proc_path = os.path.join(tissue_dir, 'processed_splicing_matrix.tsv')
    raw_path = os.path.join(tissue_dir, 'splicing_matrix.tsv')

    assert os.path.exists(proc_path), f"not found: {proc_path}"
    assert os.path.exists(raw_path), f"not found: {raw_path}"

    print(f"\n  {tissue_name}:")

    # load processed matrix
    proc = pd.read_csv(proc_path, sep='\t')
    print(f"    loaded {len(proc):,} sites")

    # filter to test chroms early
    if filter_chroms:
        proc = filter_to_test_chroms(proc)
        print(f"    after chrom filter: {len(proc):,} sites")

    meta_cols = ['event_id', 'region', 'strand', 'site', 'site_type', 'pop_mean']
    sample_cols = [c for c in proc.columns if c not in meta_cols]
    print(f"    {len(sample_cols)} samples")

    # filter to informative sites (at least one non-missing, non-zero ssu)
    n_before = len(proc)
    proc = filter_informative_sites(proc, sample_cols)
    print(f"    informative sites: {n_before:,} -> {len(proc):,}")

    # ssu stats - vectorized
    print(f"    computing ssu stats...")
    ssu_arr = proc[sample_cols].values.astype(np.float32)
    ssu_stats = compute_array_stats(ssu_arr, missing=MISSING_VALUE)

    # load raw matrix for read counts
    print(f"    loading raw matrix...")
    raw = pd.read_csv(raw_path, sep='\t')
    raw_sample_cols = [c for c in raw.columns if c != 'row_label']

    # filter raw to match proc
    raw = raw[raw['row_label'].isin(proc['event_id'])].copy()
    raw = raw.set_index('row_label').loc[proc['event_id']].reset_index()

    print(f"    parsing read counts...")
    inc_arr, total_arr = parse_read_matrix(raw, raw_sample_cols)

    print(f"    computing read stats...")
    inc_stats = compute_array_stats(inc_arr, missing=np.nan)
    total_stats = compute_array_stats(total_arr, missing=np.nan)

    # build result dataframe
    result = proc[['event_id', 'region', 'strand', 'site', 'site_type']].copy()

    # add ssu stats
    for stat_name, vals in ssu_stats.items():
        result[f'ssu_{stat_name}_{tissue_name}'] = vals

    # add read stats
    for stat_name, vals in inc_stats.items():
        result[f'inc_{stat_name}_{tissue_name}'] = vals
    for stat_name, vals in total_stats.items():
        result[f'total_{stat_name}_{tissue_name}'] = vals

    print(f"    done: {len(result):,} sites")
    return result


def load_all_tissues(tissue_dirs, tissue_names, filter_chroms=True):
    """load and merge data from all tissues"""
    print("=" * 70)
    print("loading tissue data")
    print("=" * 70)

    all_data = []
    for tdir, tname in zip(tissue_dirs, tissue_names):
        df = load_tissue_data(tdir, tname, filter_chroms=filter_chroms)
        all_data.append(df)

    # merge on splice site
    print("\nmerging tissues...")
    merge_cols = ['event_id', 'region', 'strand', 'site', 'site_type']
    merged = all_data[0]
    for df in tqdm(all_data[1:], desc="merging"):
        merged = merged.merge(df, on=merge_cols, how='outer')

    print(f"  merged: {len(merged):,} unique sites across {len(tissue_names)} tissues")
    return merged, tissue_names


def filter_sites(df, gene_df=None, paralogs=None):
    """filter sites by gene type using position overlap"""
    print("\n" + "=" * 70)
    print("filtering sites")
    print("=" * 70)

    n_start = len(df)

    # map sites to genes by position
    if gene_df is not None:
        df = map_sites_to_genes(df, gene_df)

        # keep only sites that map to protein-coding genes
        n_before = len(df)
        df = df[df['gene_id'].notna()]
        print(f"  protein-coding: {n_before:,} -> {len(df):,}")

    # non-paralogous filter
    if paralogs is not None and 'gene_id' in df.columns:
        n_before = len(df)
        df = df[~df['gene_id'].isin(paralogs)]
        print(f"  non-paralogous: {n_before:,} -> {len(df):,}")

    print(f"  final: {len(df):,} sites")
    return df


def reverse_complement(seq):
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]


def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    L = len(seq)
    oh = np.zeros((L, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            oh[i, mapping[base]] = 1.0
    return oh


def get_splice_label(strand, site_type):
    st = (site_type or '').lower()
    if strand == '+':
        if 'start' in st:
            return 'acceptor'
        elif 'end' in st:
            return 'donor'
    else:
        if 'start' in st:
            return 'donor'
        elif 'end' in st:
            return 'acceptor'
    return 'unknown'


def get_motif(seq, splice_label, center):
    if splice_label == 'acceptor':
        return seq[center-2:center].upper()
    elif splice_label == 'donor':
        return seq[center:center+2].upper()
    return ''


def extract_sequences(sites_df, genome_fa, window=5000, skip_log_path=None):
    """extract and one-hot encode splice-centered sequences"""
    print("\n" + "=" * 70)
    print("extracting sequences")
    print("=" * 70)

    genome = Fasta(genome_fa)
    total_len = 2 * window + 1
    center = window
    print(f"  genome: {genome_fa}")
    print(f"  window: +/-{window} bp ({total_len} total)")

    sequences = []
    seq_strings = []
    kept_indices = []
    skipped = {'chr': 0, 'bounds': 0, 'n': 0}
    n_skip_log = []

    for idx, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc="extracting"):
        chrom = str(row['region'])
        pos = int(row['site'])
        strand = str(row['strand'])
        stype = str(row.get('site_type', 'unknown')).lower()

        # check chrom exists
        if chrom not in genome:
            chrom_alt = chrom.replace('chr', '') if chrom.startswith('chr') else f'chr{chrom}'
            if chrom_alt in genome:
                chrom = chrom_alt
            else:
                skipped['chr'] += 1
                continue

        # extract window
        site = pos - 1  # 0-based
        start = site - window
        end = site + window + 1

        chrom_len = len(genome[chrom])
        if start < 0 or end > chrom_len:
            skipped['bounds'] += 1
            continue

        seq = str(genome[chrom][start:end])
        if len(seq) != total_len:
            skipped['bounds'] += 1
            continue

        # skip if contains N
        if 'N' in seq.upper():
            skipped['n'] += 1
            n_skip_log.append({
                'chromosome': chrom, 'position': pos, 'strand': strand,
                'site_type': stype,
            })
            continue

        # reverse complement for - strand
        if strand == '-':
            seq = reverse_complement(seq)

        oh = one_hot_encode(seq)
        sequences.append(oh)
        seq_strings.append(seq.upper())
        kept_indices.append(idx)

    if skip_log_path and n_skip_log:
        print(f"\n  saving skip log: {skip_log_path}")
        pd.DataFrame(n_skip_log).to_csv(skip_log_path, index=False)

    print(f"\n  extracted: {len(sequences):,}")
    print(f"  skipped - chr missing: {skipped['chr']:,}")
    print(f"  skipped - out of bounds: {skipped['bounds']:,}")
    print(f"  skipped - contains N: {skipped['n']:,}")

    if len(sequences) == 0:
        raise ValueError("no sequences extracted - check filters")

    return np.stack(sequences), seq_strings, kept_indices


def save_h5(output_path, X, seq_strings, sites_df, kept_indices, tissues):
    """save sequences and stats to h5"""
    print(f"\nsaving: {output_path}")

    # filter df to kept sites
    df = sites_df.loc[kept_indices].reset_index(drop=True)
    n = len(df)

    with h5py.File(output_path, 'w') as f:
        # sequences
        f.create_dataset('X', data=X, compression='gzip', compression_opts=4)
        f.create_dataset('sequence', data=np.array(seq_strings, dtype='S'))

        # coordinates
        f.create_dataset('chromosome', data=np.array(df['region'].values, dtype='S'))
        f.create_dataset('strand', data=np.array(df['strand'].values, dtype='S'))
        f.create_dataset('position', data=df['site'].values.astype(np.int32))
        f.create_dataset('site_type', data=np.array(df['site_type'].values, dtype='S'))
        f.create_dataset('event_id', data=np.array(df['event_id'].values, dtype='S'))

        # splice label
        labels = [get_splice_label(r['strand'], r['site_type']) for _, r in df.iterrows()]
        f.create_dataset('splice_label', data=np.array(labels, dtype='S'))

        # stats columns
        stat_cols = [c for c in df.columns if any(c.startswith(p) for p in ['ssu_', 'inc_', 'total_'])]
        for col in tqdm(stat_cols, desc="writing stats"):
            vals = df[col].values.astype(np.float32)
            f.create_dataset(col, data=vals)

        # attrs
        f.attrs['n_sequences'] = n
        f.attrs['sequence_length'] = X.shape[1]
        f.attrs['center_index'] = X.shape[1] // 2
        f.attrs['tissues'] = ','.join(tissues)

    # show contents
    print("\nh5 contents:")
    print("-" * 50)
    with h5py.File(output_path, 'r') as f:
        for k in sorted(f.keys()):
            ds = f[k]
            dtype_str = f"{ds.dtype} (string)" if ds.dtype.kind == 'S' else str(ds.dtype)
            print(f"  {k}: {ds.shape} {dtype_str}")


def main():
    ap = argparse.ArgumentParser(
        description="extract splice sequences with comprehensive stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  python extract_splice_sequences.py \\
    --tissue-dirs /scratch/runyan.m/sphaec_out/lung \\
                  /scratch/runyan.m/sphaec_out/brain_cortex \\
                  /scratch/runyan.m/sphaec_out/haec10 \\
    --tissues lung brain haec \\
    --genome /path/to/genome.fa \\
    --protein-coding /path/to/protein_coding_genes.tsv \\
    --paralogs /path/to/paralogs_GRCh38.txt.gz \\
    --output sequences.h5 \\
    --test-chroms
"""
    )

    ap.add_argument('--tissue-dirs', nargs='+', required=True, help='tissue output directories')
    ap.add_argument('--tissues', nargs='+', required=True, help='tissue names')
    ap.add_argument('--genome', required=True, help='genome fasta')
    ap.add_argument('--protein-coding', help='protein-coding genes tsv')
    ap.add_argument('--paralogs', help='paralogs gz file')
    ap.add_argument('--test-chroms', action='store_true', help='filter to chr1,3,5,7')
    ap.add_argument('--window', type=int, default=5000, help='half-window (default: 5000)')
    ap.add_argument('--skip-log', help='csv file to log skipped sequences')
    ap.add_argument('--output', required=True, help='output h5 file')

    args = ap.parse_args()

    assert len(args.tissue_dirs) == len(args.tissues), "tissue-dirs and tissues must match"
    for td in args.tissue_dirs:
        assert os.path.isdir(td), f"not found: {td}"
    assert os.path.exists(args.genome), f"genome not found: {args.genome}"

    print(f"reading files ...", flush=True)

    # load gene filters
    gene_df = None
    paralogs = None
    if args.protein_coding:
        gene_df = load_protein_coding_genes(args.protein_coding)
    if args.paralogs:
        paralogs = load_paralog_genes(args.paralogs)

    # load tissue data (filters to test chroms early if flag set)
    df, tissues = load_all_tissues(args.tissue_dirs, args.tissues, filter_chroms=args.test_chroms)

    # filter by gene type
    df = filter_sites(df, gene_df=gene_df, paralogs=paralogs)

    # extract sequences
    X, seq_strings, kept_indices = extract_sequences(
        df, args.genome, window=args.window, skip_log_path=args.skip_log
    )

    # save
    save_h5(args.output, X, seq_strings, df, kept_indices, tissues)
    print("\ndone.")


if __name__ == "__main__":
    main()
