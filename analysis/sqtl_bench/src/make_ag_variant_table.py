#!/usr/bin/env python3
"""join a VCF with AG's GTF feather to produce a variant table for AG scoring.

for each variant in the input VCF, find protein-coding gene(s) overlapping
the variant position and emit one row per (variant, gene) tuple. the resulting
TSV is the input for analysis/test/src/score_alphagenome_variant.py --mode composite.

usage:
    python make_ag_variant_table.py input.vcf.gz output.tsv \\
        [--gtf-feather $AG_DATA_DIR/gencode.v46.annotation.gtf.gz.feather] \\
        [--biotype protein_coding]
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import pandas as pd


def load_vcf(path):
    """minimal VCF loader (no pysam dependency).
    returns DataFrame with chrom, pos, var_id, ref, alt, var_key + any INFO fields.
    """
    records = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            chrom, pos, vid, ref, alt = cols[0], int(cols[1]), cols[2], cols[3], cols[4]
            info = cols[7] if len(cols) > 7 else ''
            info_dict = {}
            for field in info.split(';'):
                if '=' in field:
                    k, v = field.split('=', 1)
                    info_dict[k] = v
            records.append({
                'chrom': chrom, 'pos': pos, 'var_id': vid,
                'ref': ref, 'alt': alt,
                'var_key': f"{chrom}:{pos}:{ref}:{alt}",
                **info_dict,
            })
    return pd.DataFrame(records)


def load_gene_table(gtf_feather, biotype='protein_coding'):
    """load AG's GTF feather, collapse to one row per gene with min/max coords."""
    df = pd.read_feather(gtf_feather)
    if 'Feature' in df.columns:
        df = df.query("Feature == 'gene'").copy()
    if biotype and 'gene_type' in df.columns:
        df = df[df['gene_type'] == biotype].copy()
    # normalize column names
    rename = {
        'Chromosome': 'chrom',
        'seqname': 'chrom',
        'Start': 'start',
        'End': 'end',
        'Strand': 'strand',
        'gene_id': 'gene_id',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    keep = ['chrom', 'start', 'end', 'strand', 'gene_id']
    if 'gene_name' in df.columns:
        keep.append('gene_name')
    df = df[keep].dropna(subset=['chrom', 'start', 'end', 'strand']).copy()
    df['chrom'] = df['chrom'].astype(str)
    if not df['chrom'].iloc[0].startswith('chr'):
        df['chrom'] = 'chr' + df['chrom']
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_vcf')
    p.add_argument('output_tsv')
    p.add_argument('--gtf-feather', default=None,
                   help='AG GTF feather; default $AG_DATA_DIR/gencode.v46.annotation.gtf.gz.feather')
    p.add_argument('--biotype', default='protein_coding',
                   help='gene_type to filter on (default: protein_coding; "" to disable)')
    args = p.parse_args()

    gtf_path = args.gtf_feather
    if gtf_path is None:
        ag_data = os.environ.get('AG_DATA_DIR')
        if not ag_data:
            sys.exit("ERROR: pass --gtf-feather or set AG_DATA_DIR")
        gtf_path = os.path.join(ag_data, 'gencode.v46.annotation.gtf.gz.feather')

    print(f"loading VCF {args.input_vcf}", flush=True)
    vcf_df = load_vcf(args.input_vcf)
    print(f"  {len(vcf_df):,} variants", flush=True)

    print(f"loading GTF {gtf_path}", flush=True)
    biotype = args.biotype if args.biotype else None
    gene_df = load_gene_table(gtf_path, biotype=biotype)
    print(f"  {len(gene_df):,} genes (biotype={biotype})", flush=True)

    # ensure chrom strings match
    vcf_df['chrom'] = vcf_df['chrom'].astype(str)
    if not vcf_df['chrom'].iloc[0].startswith('chr'):
        vcf_df['chrom'] = 'chr' + vcf_df['chrom']

    # per-chrom join (pandas merge on chrom then filter by interval overlap)
    rows = []
    for chrom, vsub in vcf_df.groupby('chrom'):
        gsub = gene_df[gene_df['chrom'] == chrom]
        if gsub.empty:
            continue
        gstarts = gsub['start'].values
        gends = gsub['end'].values
        gstrands = gsub['strand'].values
        gids = gsub['gene_id'].values
        for vrow in vsub.itertuples(index=False):
            pos = int(vrow.pos)
            mask = (gstarts <= pos) & (gends >= pos)
            if not mask.any():
                continue
            for i in mask.nonzero()[0]:
                rows.append({
                    'variant_id': vrow.var_key,
                    'chrom': chrom,
                    'pos': pos,
                    'ref': vrow.ref,
                    'alt': vrow.alt,
                    'gene_strand': gstrands[i],
                    'tx_start': int(gstarts[i]),
                    'tx_end': int(gends[i]),
                    'gene_id': gids[i],
                })

    out_df = pd.DataFrame(rows)
    print(f"emit {len(out_df):,} (variant, gene) rows "
          f"({len(out_df['variant_id'].unique()):,} unique variants)", flush=True)
    out_df.to_csv(args.output_tsv, sep='\t', index=False)
    print(f"wrote {args.output_tsv}", flush=True)


if __name__ == '__main__':
    main()
