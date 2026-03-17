#!/usr/bin/env python3
"""extract splice sites from GENCODE GTF into canonical_dataset format"""

import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gtf', help='GENCODE GTF (plain or gzipped)')
    parser.add_argument('output', help='output canonical_dataset file')
    parser.add_argument('--paralogs', help='paralogs file (tsv with Gene stable ID column)')
    parser.add_argument('--mane-only', action='store_true', help='restrict to MANE_Select transcripts')
    parser.add_argument('--chroms', nargs='+',
                        default=[f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'],
                        help='chromosomes to include')
    args = parser.parse_args()

    chrom_set = set(args.chroms)

    # load paralogs
    paralog_set = set()
    if args.paralogs:
        paralog_set = load_paralogs(args.paralogs)
        print(f"loaded {len(paralog_set):,} paralog gene IDs")

    # parse GTF
    genes = parse_gtf(args.gtf, chrom_set, mane_only=args.mane_only)
    print(f"parsed {len(genes):,} genes from GTF")

    # write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.output, 'w') as f:
        for gene_id in sorted(genes):
            g = genes[gene_id]
            starts = sorted(g['starts'])
            ends = sorted(g['ends'])

            if not starts or not ends:
                continue

            base_id = gene_id.split('.')[0]
            if paralog_set and base_id in paralog_set:
                continue
            paralog_flag = '0'
            gene_start = min(min(starts), min(ends))
            gene_end = max(max(starts), max(ends))

            # trailing comma matches original format
            ends_str = ','.join(str(x) for x in ends) + ','
            starts_str = ','.join(str(x) for x in starts) + ','

            f.write(f"{gene_id}\t{paralog_flag}\t{g['chrom']}\t{g['strand']}\t"
                    f"{gene_start}\t{gene_end}\t{ends_str}\t{starts_str}\n")
            n_written += 1

    print(f"wrote {n_written:,} genes to {args.output}")


def parse_gtf(gtf_path, chrom_set, mane_only=False):
    """parse exon positions from GTF, grouped by gene"""
    genes = defaultdict(lambda: {'starts': set(), 'ends': set(), 'chrom': None, 'strand': None})

    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    n_exons = 0

    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature != 'exon':
                continue

            chrom = parts[0]
            if chrom not in chrom_set:
                continue

            attr = parts[8]

            # filter protein_coding
            gene_type = _extract_attr(attr, 'gene_type')
            if gene_type != 'protein_coding':
                continue

            tx_type = _extract_attr(attr, 'transcript_type')
            if tx_type != 'protein_coding':
                continue

            if mane_only:
                # require MANE_Select tag
                if 'tag "MANE_Select"' not in attr:
                    continue
            else:
                # require TSL=1
                tsl = _extract_attr(attr, 'transcript_support_level')
                if tsl != '1':
                    continue

            gene_id = _extract_attr(attr, 'gene_id')
            strand = parts[6]
            start = int(parts[3])
            end = int(parts[4])

            genes[gene_id]['starts'].add(start)
            genes[gene_id]['ends'].add(end)
            genes[gene_id]['chrom'] = chrom
            genes[gene_id]['strand'] = strand
            n_exons += 1

    mode = "MANE_Select" if mane_only else "TSL=1 protein_coding"
    print(f"parsed {n_exons:,} exons ({mode})")
    return genes


def _extract_attr(attr_str, key):
    """extract value for key from GTF attribute string"""
    m = re.search(rf'{key} "([^"]+)"', attr_str)
    return m.group(1) if m else None


def load_paralogs(path):
    """load paralog gene IDs (base IDs without version)"""
    ids = set()
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        header = f.readline().strip().split('\t')
        col_idx = header.index('Gene stable ID')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > col_idx and parts[col_idx]:
                ids.add(parts[col_idx])
    return ids


if __name__ == '__main__':
    main()
