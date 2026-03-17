#!/usr/bin/env python3
"""convert pangolin splice_table to matrix TSV for build_h5"""

import argparse
import numpy as np

TISSUE_NAMES = ['heart', 'liver', 'brain', 'testis']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('splice_table', help='splice_table_Human.test.txt')
    parser.add_argument('output_tsv', help='output matrix TSV')
    parser.add_argument('--tissue', choices=TISSUE_NAMES,
                        help='use single tissue SSU instead of cross-tissue mean')
    parser.add_argument('--chroms', nargs='+',
                        default=[f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'],
                        help='chromosomes to include')
    parser.add_argument('--paralogs', help='paralogs file (one gene ID per line)')
    args = parser.parse_args()

    # load paralogs file (from pangolin training repo)
    paralog_set = set()
    if args.paralogs:
        paralog_set = load_paralogs(args.paralogs)
        print(f"loaded {len(paralog_set):,} paralog gene IDs")

    tissue_idx = TISSUE_NAMES.index(args.tissue) if args.tissue else None
    if args.tissue:
        print(f"using tissue: {args.tissue} (index {tissue_idx})")

    chrom_set = set(args.chroms)
    genes = parse_splice_table(args.splice_table, chrom_set,
                               paralog_set=paralog_set, tissue_idx=tissue_idx)
    print(f"loaded {len(genes):,} genes on {args.chroms}")

    n_sites = sum(len(g['junctions']) for g in genes)
    n_ssu_valid = sum(1 for g in genes for jn in g['junctions'] if jn['ssu'] != 777.0)
    print(f"total sites: {n_sites:,}, ssu coverage: {n_ssu_valid:,}/{n_sites:,} "
          f"({n_ssu_valid/n_sites:.1%})" if n_sites else "no sites found")

    write_matrix(genes, args.output_tsv)


def parse_splice_table(path, chroms, paralog_set=None, tissue_idx=None):
    """parse pangolin splice_table, returns list of gene dicts"""
    if paralog_set is None:
        paralog_set = set()
    genes = []
    n_paralog_skipped = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue

            gene, paralog, chrom, strand, tx_start, tx_end, jn_raw = parts[:7]

            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom

            if chrom not in chroms:
                continue

            # filter paralogs (strip version suffix)
            base_id = gene.split('.')[0]
            if paralog_set and base_id in paralog_set:
                n_paralog_skipped += 1
                continue

            # parse junctions
            junctions = []
            for entry in jn_raw.split(';'):
                entry = entry.strip()
                if not entry:
                    continue
                pos_str, tissue_str = entry.split(':')
                pos = int(pos_str)
                # empty fields = missing tissue
                tissues = [float(x) if x else -1.0 for x in tissue_str.split(',')]

                if tissue_idx is not None:
                    # single tissue SSU
                    v = tissues[tissue_idx] if tissue_idx < len(tissues) else -1.0
                    ssu = float(v) if v >= 0 else 777.0
                else:
                    # mean of valid (>= 0) tissue values, 777.0 if all invalid
                    valid_vals = [v for v in tissues if v >= 0]
                    ssu = float(np.mean(valid_vals)) if valid_vals else 777.0

                junctions.append({'pos': pos, 'ssu': ssu})

            if not junctions:
                continue

            genes.append({
                'gene': gene,
                'paralog': paralog,
                'chrom': chrom,
                'strand': strand,
                'tx_start': int(tx_start),
                'tx_end': int(tx_end),
                'junctions': junctions,
            })

    if n_paralog_skipped:
        print(f"skipped {n_paralog_skipped:,} paralogous genes")
    return genes


def write_matrix(genes, output_path):
    """write matrix TSV compatible with build_h5.py, all sites in both columns"""
    n_written = 0

    with open(output_path, 'w') as f:
        f.write('\t'.join([
            'Unique_ID', 'Gene_ID', 'Chromosome', 'Strand', 'Start', 'End',
            'exon_ends', 'exon_starts', 'exon_end_SSUs', 'exon_start_SSUs',
            'Variants', 'paralog_status'
        ]) + '\n')

        for g in genes:
            positions = [jn['pos'] for jn in g['junctions']]
            ssus = [jn['ssu'] for jn in g['junctions']]

            pos_str = ','.join(str(p) for p in positions)
            ssu_str = ','.join(f"{s:.6f}" if s != 777.0 else '777.0' for s in ssus)

            uid = f"PANGOLIN_{g['gene']}_{g['chrom']}"

            # both columns get all sites (filter_vars requires both SSU cols non-empty)
            # set_label overwrites, so acceptor (second) wins -> fix_both_labels restores donor
            f.write('\t'.join([
                uid,
                g['gene'],
                g['chrom'],
                g['strand'],
                str(g['tx_start']),
                str(g['tx_end']),
                pos_str,        # exon_ends = all sites
                pos_str,        # exon_starts = all sites (same)
                ssu_str,        # exon_end_SSUs
                ssu_str,        # exon_start_SSUs (same)
                '',             # no variants
                '0',            # non-paralog (paralogs already filtered)
            ]) + '\n')
            n_written += 1

    print(f"wrote {n_written:,} genes to {output_path}")


def load_paralogs(path):
    """load paralog gene IDs, one per line (no version suffix)"""
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


if __name__ == '__main__':
    main()
