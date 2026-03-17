#!/usr/bin/env python3
"""convert canonical transcript dataset to matrix format for build_h5"""

import argparse
import pandas as pd
from tqdm import tqdm


DONOR_ID = "CANONICAL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('canonical_file', help='canonical dataset file (with corrected paralog flag)')
    parser.add_argument('output_tsv', help='output enriched matrix TSV')
    args = parser.parse_args()

    rows = []
    with open(args.canonical_file) as f:
        for line in tqdm(f, desc="reading"):
            parts = line.strip().split('\t')
            gene, paralog_flag, chrom, strand, start, end, exon_ends_str, exon_starts_str = parts

            # parse comma-separated positions (trailing comma)
            exon_ends = exon_ends_str.rstrip(',')
            exon_starts = exon_starts_str.rstrip(',')
            n_ends = len([x for x in exon_ends.split(',') if x])
            n_starts = len([x for x in exon_starts.split(',') if x])

            # fake SSU values (777.0 = missing, model masks in loss)
            end_ssus = ','.join(['777.0'] * n_ends)
            start_ssus = ','.join(['777.0'] * n_starts)

            rows.append({
                'Unique_ID': f'{DONOR_ID}_{gene}_{chrom}',
                'Gene_ID': gene,
                'Chromosome': chrom,
                'Strand': strand,
                'Start': start,
                'End': end,
                'exon_ends': exon_ends,
                'exon_starts': exon_starts,
                'exon_end_SSUs': end_ssus,
                'exon_start_SSUs': start_ssus,
                'Variants': '',
                'paralog_status': paralog_flag,  # 0=non-paralog, 1=paralog
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_tsv, sep='\t', index=False)

    # summary
    para_counts = df['paralog_status'].value_counts()
    print(f"wrote {len(df):,} rows to {args.output_tsv}")
    print(f"  paralog_status=0 (non-paralog): {para_counts.get('0', 0):,}")
    print(f"  paralog_status=1 (paralog): {para_counts.get('1', 0):,}")


if __name__ == '__main__':
    main()
