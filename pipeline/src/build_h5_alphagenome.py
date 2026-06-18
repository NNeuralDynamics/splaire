#!/usr/bin/env python3
"""per-gene h5 builder for AlphaGenome scoring.

each gene gets one h5 entry with a fixed 1 Mb (2^20) sequence centered on the
reference gene midpoint, plus per-position labels for the gene region only.

sequence is stored in pre-mRNA orientation (RC for - strand).
in-gene donor variants from the splice table are applied. flank regions remain
reference (donor variants outside tx_start..tx_end are not in the splice table).

genes > 1 Mb are tiled: each tile is a separate entry sharing the same unique_id,
with tile_index/n_tiles attrs. center 500 kb of each tile's predictions should
be used at scoring time to avoid edge effects.

schema (h5 datasets, n = entry index, NOT gene index for tiled genes):
  SEQ_{n}    int8        shape (1048576,)            one-hot index, pre-mRNA orientation
  Y_{n}      float32     shape (cov_len, 4)          [neither, acc, don, ssu] per output position
  GC_{n}     structured  shape (cov_len,)            chrom, strand, position, name

attrs on SEQ_{n}:
  tx_start, tx_end       int     adjusted gene span (matches existing pipeline)
  strand                 str     '+' or '-'
  donor                  str
  unique_id              str
  chrom                  int8
  gene_offset_in_seq     int32   offset of cov_start within the 1 Mb sequence
  cov_start, cov_end     int32   genomic span covered by Y_{n}/GC_{n}
  tile_index, n_tiles    int     for single-tile genes: 0, 1

h5 root attrs:
  ag_seq_len             int     1048576
  n_entries              int
  mode                   str     'basic' or 'het'
  donor                  str
  reference_only         bool

usage (single donor / canonical):
    python build_h5_alphagenome.py \
        --donor DONOR_ID --split test \
        --input matrix.tsv \
        --chroms chr1,chr3,chr5,chr7,chr9 \
        --fasta GRCh38.fa \
        --output out.h5 \
        --mode basic --paralog 0
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    GC_DTYPE, IUPAC_HET, parse_int_list, parse_float_list,
    normalize_chrom, seq_to_int_array,
)

AG_SEQ_LEN = 2**20                # 1,048,576
HALF_LEN = AG_SEQ_LEN // 2
TILE_STRIDE = 2**19               # 524,288 — overlap of 50% when tiling
RC_TRANS = str.maketrans("ACGTN", "TGCAN")


def setup_logging(log_file=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def revcomp_str(seq):
    return seq.translate(RC_TRANS)[::-1]


def parse_int_safe(x):
    """parse int, return None on failure or 'nan'/empty."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none'):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_int_list_safe(x):
    """parse comma-separated ints, skipping invalid/nan entries."""
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none'):
        return []
    out = []
    for v in s.split(','):
        n = parse_int_safe(v)
        if n is not None:
            out.append(n)
    return out


def parse_float_list_safe(x):
    """parse comma-separated floats, skipping invalid entries.
    note: float NaN is preserved when valid (e.g. 'nan' parses to math.nan)."""
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() in ('none',):
        return []
    out = []
    for v in s.split(','):
        v = v.strip()
        if not v:
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def apply_variants_in_window(seq, variants_str, win_start, tx_start, tx_end, mode='basic'):
    """apply in-gene variants to a 1 Mb reference window.

    only mutates the gene region of the window so list ops are O(gene_len)
    not O(1 Mb). returns modified seq string (length may differ from input
    if indels present). variants outside the gene span are ignored.
    """
    if not variants_str:
        return seq
    s = str(variants_str).strip()
    if not s or s.lower() in ('', 'none', 'nan'):
        return seq

    # gene region in window (0-based, [lo, hi))
    gene_lo = max(0, tx_start - 1 - win_start)
    gene_hi = min(len(seq), tx_end - win_start)
    if gene_lo >= gene_hi:
        return seq

    left = seq[:gene_lo]
    gene_list = list(seq[gene_lo:gene_hi])
    right = seq[gene_hi:]

    for variant in s.split(','):
        parts = variant.strip().split(':')
        if len(parts) != 4:
            continue
        _vchrom, pos_str, ref, alt = parts
        pos = parse_int_safe(pos_str)
        if pos is None:
            continue
        rel = pos - 1 - win_start - gene_lo
        if rel < 0 or rel >= len(gene_list):
            continue

        extracted_ref = ''.join(gene_list[rel:rel + len(ref)]).upper()
        if extracted_ref != ref.upper():
            logging.debug(f"  ref mismatch at {pos}: expected {ref}, found {extracted_ref}")
            continue

        if mode == 'het' and len(ref) == 1 and len(alt) == 1:
            pair = (ref.upper(), alt.upper())
            if pair in IUPAC_HET:
                gene_list[rel] = IUPAC_HET[pair]
            else:
                gene_list[rel] = alt.upper()
        else:
            if len(ref) > len(alt):
                gene_list = gene_list[:rel + 1] + gene_list[rel + len(ref):]
            elif len(ref) == 1 and len(alt) == 1:
                # fast SNV path: O(1)
                gene_list[rel] = alt.upper()
            else:
                gene_list = gene_list[:rel] + list(alt.upper()) + gene_list[rel + len(ref):]

    return left + ''.join(gene_list) + right


def fetch_window(fasta, chrom, win_start, win_end):
    """fetch reference [win_start, win_end) padded with N at chromosome boundaries."""
    chrom_len = len(fasta[chrom])
    left_pad = max(0, -win_start)
    right_pad = max(0, win_end - chrom_len)
    fs = max(0, win_start)
    fe = min(chrom_len, win_end)
    seq = 'N' * left_pad + str(fasta[chrom][fs:fe]).upper() + 'N' * right_pad
    assert len(seq) == win_end - win_start, f"fetch length mismatch: {len(seq)} vs {win_end - win_start}"
    return seq


def build_labels(row, strand, chrom_int, remove_missing=False):
    """build per-position labels for the gene span [tx_start, tx_end] (adjusted coords).
    matches pipeline/src/utils.create_datapoints labeling convention.
    returns y (gene_len, 4), gc structured (gene_len,) in pre-mRNA orientation.

    remove_missing: if True, treat ssu==777 as 'neither' class. otherwise keep the
    acceptor/donor class label with ssu=777 (canonical benchmark convention).
    """
    tx_start = int(float(row['Start']))
    tx_end = int(float(row['End']))
    gene_len = tx_end - tx_start + 1
    uid = str(row['Unique_ID'])
    name_bytes = uid.encode('ascii', errors='replace')[:200]

    jn_start = parse_int_list_safe(row.get('exon_ends', ''))
    jn_end = parse_int_list_safe(row.get('exon_starts', ''))
    jn_start_ssu = parse_float_list_safe(row.get('exon_end_SSUs', ''))
    jn_end_ssu = parse_float_list_safe(row.get('exon_start_SSUs', ''))

    y = np.zeros((gene_len, 4), dtype='float32')
    y[:, 0] = 1.0  # default "neither"
    gc = np.zeros((gene_len,), dtype=GC_DTYPE)

    def set_label(idx, class_vec, ssu):
        if remove_missing and ssu == 777:
            y[idx] = [1.0, 0.0, 0.0, 777.0]
        else:
            y[idx] = class_vec + [float(ssu)]

    if strand == '+':
        # donors = exon ends, acceptors = exon starts
        for pos, ssu in zip(jn_start, jn_start_ssu):
            if tx_start <= pos <= tx_end:
                set_label(pos - tx_start, [0.0, 0.0, 1.0], ssu)
        for pos, ssu in zip(jn_end, jn_end_ssu):
            if tx_start <= pos <= tx_end:
                set_label(pos - tx_start, [0.0, 1.0, 0.0], ssu)
        for i in range(gene_len):
            gc[i] = (chrom_int, 1, tx_start + i, name_bytes)
    else:
        # - strand: jn_end becomes donor, jn_start becomes acceptor; positions mirrored
        for pos, ssu in zip(jn_end, jn_end_ssu):
            if tx_start <= pos <= tx_end:
                set_label(tx_end - pos, [0.0, 0.0, 1.0], ssu)
        for pos, ssu in zip(jn_start, jn_start_ssu):
            if tx_start <= pos <= tx_end:
                set_label(tx_end - pos, [0.0, 1.0, 0.0], ssu)
        for i in range(gene_len):
            gc[i] = (chrom_int, 0, tx_end - i, name_bytes)

    return y, gc


def build_gene_sequence(fasta, row, strand, mode='basic', reference_only=False):
    """fetch a 1 Mb window centered on the reference gene midpoint, apply in-gene variants,
    pad/trim to exactly AG_SEQ_LEN, RC if minus strand.
    returns: seq_str (length AG_SEQ_LEN), gene_offset_in_seq (int).
    """
    chrom = row['Chromosome']
    tx_start = int(float(row['Start']))                 # = Original_Start
    orig_end = int(float(row.get('Original_End', row['End'])))
    center = (tx_start + orig_end) // 2
    win_start = center - HALF_LEN
    win_end = win_start + AG_SEQ_LEN

    seq = fetch_window(fasta, chrom, win_start, win_end)
    if not reference_only:
        # tx_end is adjusted (variant-induced indels accounted for)
        adj_end_local = int(float(row['End']))
        seq = apply_variants_in_window(
            seq, row.get('Variants', ''), win_start, tx_start, adj_end_local, mode=mode
        )

    # pad/trim to fixed length
    if len(seq) > AG_SEQ_LEN:
        # net insertions — trim from right end (variants near left edge unaffected)
        seq = seq[:AG_SEQ_LEN]
    elif len(seq) < AG_SEQ_LEN:
        seq = seq + 'N' * (AG_SEQ_LEN - len(seq))

    # gene start in window: tx_start (1-based) - 1 - win_start (0-based)
    # left flank of gene is unmodified by in-gene variants, so this stays correct
    gene_offset = tx_start - 1 - win_start
    # gene_len in modified seq = adjusted tx_end - tx_start + 1
    adj_end = int(float(row['End']))
    gene_len = adj_end - tx_start + 1

    if strand == '-':
        seq = revcomp_str(seq)
        # gene region in pre-mRNA orientation: starts at AG_SEQ_LEN - (gene_offset + gene_len)
        gene_offset = AG_SEQ_LEN - (gene_offset + gene_len)

    return seq, gene_offset


def write_entry(h5f, n, seq_str, y, gc, attrs, mode='basic'):
    """encode + write one entry to h5."""
    seq_int = seq_to_int_array(seq_str, mode=mode)
    seq_ds = h5f.create_dataset(f"SEQ_{n}", data=seq_int, compression='gzip', compression_opts=9)
    h5f.create_dataset(f"Y_{n}", data=y, compression='gzip', compression_opts=9)
    h5f.create_dataset(f"GC_{n}", data=gc, compression='gzip', compression_opts=9)
    for k, v in attrs.items():
        seq_ds.attrs[k] = v


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donor", required=True, help="donor ID")
    parser.add_argument("--split", required=True, choices=["train", "valid", "test", "full"])
    parser.add_argument("--input", required=True, help="splice table TSV")
    parser.add_argument("--chroms", required=True, help="comma-separated chromosomes")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="basic", choices=["basic", "het"])
    parser.add_argument("--paralog", default="all", help="paralog filter: 0, 1, or all")
    parser.add_argument("--reference", action="store_true",
                        help="reference-only build (skip variant application)")
    parser.add_argument("--remove-missing", action="store_true",
                        help="treat ssu==777 as 'neither' (default: keep acc/don class with 777)")
    parser.add_argument("--skip-empty-genes", action="store_true",
                        help="skip genes with no labeled splice sites")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    log_file = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"build_ag_{args.donor}.log")
    logger = setup_logging(log_file)
    logger.info(f"build_h5_alphagenome: donor={args.donor} split={args.split}")
    logger.info(f"input={args.input}, output={args.output}, fasta={args.fasta}")
    logger.info(f"mode={args.mode}, paralog={args.paralog}, reference_only={args.reference}")

    logger.info("loading splice table")
    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"  loaded {len(df):,} rows")

    if 'Donor' in df.columns:
        df = df[df['Donor'] == args.donor].copy()
        logger.info(f"  after donor filter: {len(df):,} rows")

    chroms = [c.strip() for c in args.chroms.split(',') if c.strip()]
    df = df[df['Chromosome'].isin(chroms)].copy()
    logger.info(f"  after chrom filter: {len(df):,} rows")

    if args.paralog != 'all' and 'paralog_status' in df.columns:
        df = df[df['paralog_status'].astype(str) == str(args.paralog)].copy()
        logger.info(f"  after paralog filter: {len(df):,} rows")

    if len(df) == 0:
        logger.warning("no rows after filters, writing empty h5")
        with h5py.File(args.output, 'w') as h5f:
            h5f.attrs['ag_seq_len'] = AG_SEQ_LEN
            h5f.attrs['n_entries'] = 0
            h5f.attrs['mode'] = args.mode
            h5f.attrs['donor'] = args.donor
            h5f.attrs['reference_only'] = bool(args.reference)
        return

    logger.info(f"opening FASTA {args.fasta}")
    fasta = Fasta(args.fasta, sequence_always_upper=True)

    logger.info("building h5")
    n_entries = 0
    n_tiled = 0
    n_skipped_empty = 0
    n_errors = 0

    with h5py.File(args.output, 'w') as h5f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="genes",
                           file=sys.stdout, mininterval=5):
            try:
                tx_start = int(float(row['Start']))
                adj_end = int(float(row['End']))
                orig_end = int(float(row.get('Original_End', adj_end)))
                gene_len = adj_end - tx_start + 1
                strand = str(row['Strand'])
                chrom = str(row['Chromosome'])
                chrom_int = normalize_chrom(chrom)
                uid = str(row['Unique_ID'])

                y, gc = build_labels(row, strand, chrom_int, remove_missing=args.remove_missing)

                if args.skip_empty_genes:
                    if (y[:, 1].sum() + y[:, 2].sum()) < 1:
                        n_skipped_empty += 1
                        continue

                if gene_len <= AG_SEQ_LEN:
                    # single tile path
                    seq_str, gene_offset = build_gene_sequence(
                        fasta, row, strand, mode=args.mode, reference_only=args.reference
                    )
                    attrs = {
                        'tx_start': tx_start, 'tx_end': adj_end,
                        'strand': strand, 'donor': args.donor,
                        'unique_id': uid, 'chrom': int(chrom_int),
                        'gene_offset_in_seq': int(gene_offset),
                        'cov_start': tx_start, 'cov_end': adj_end,
                        'tile_index': 0, 'n_tiles': 1,
                    }
                    write_entry(h5f, n_entries, seq_str, y, gc, attrs, mode=args.mode)
                    n_entries += 1
                else:
                    # gene > 1 Mb: tile with TILE_STRIDE; emit one entry per tile
                    # each tile covers TILE_STRIDE bp of labels centered in the 1 Mb seq
                    n_tiled += 1
                    n_tiles = ((gene_len - 1) // TILE_STRIDE) + 1
                    logger.info(f"  tiling {uid}: gene_len={gene_len:,} -> {n_tiles} tiles")
                    for ti in range(n_tiles):
                        # tile covers gene positions [ti*TILE_STRIDE, min((ti+1)*TILE_STRIDE, gene_len))
                        # in the strand's coordinate space (matches y/gc indexing)
                        cov_lo = ti * TILE_STRIDE
                        cov_hi = min((ti + 1) * TILE_STRIDE, gene_len)
                        cov_len = cov_hi - cov_lo
                        if strand == '+':
                            cov_start_genomic = tx_start + cov_lo
                            cov_end_genomic = tx_start + cov_hi - 1
                            tile_center = (cov_start_genomic + cov_end_genomic) // 2
                        else:
                            cov_start_genomic = adj_end - (cov_hi - 1)
                            cov_end_genomic = adj_end - cov_lo
                            tile_center = (cov_start_genomic + cov_end_genomic) // 2

                        # fetch window centered on the tile center
                        win_start = tile_center - HALF_LEN
                        win_end = win_start + AG_SEQ_LEN
                        seq = fetch_window(fasta, chrom, win_start, win_end)

                        if not args.reference:
                            seq = apply_variants_in_window(
                                seq, row.get('Variants', ''), win_start, tx_start, adj_end, mode=args.mode
                            )
                        if len(seq) != AG_SEQ_LEN:
                            seq = (seq + 'N' * AG_SEQ_LEN)[:AG_SEQ_LEN]

                        # offset of cov region start in the window (in reference coords)
                        cov_offset = cov_start_genomic - 1 - win_start

                        if strand == '-':
                            seq = revcomp_str(seq)
                            cov_offset = AG_SEQ_LEN - (cov_offset + cov_len)

                        y_tile = y[cov_lo:cov_hi]
                        gc_tile = gc[cov_lo:cov_hi]

                        attrs = {
                            'tx_start': tx_start, 'tx_end': adj_end,
                            'strand': strand, 'donor': args.donor,
                            'unique_id': uid, 'chrom': int(chrom_int),
                            'gene_offset_in_seq': int(cov_offset),
                            'cov_start': int(cov_start_genomic),
                            'cov_end': int(cov_end_genomic),
                            'tile_index': ti, 'n_tiles': int(n_tiles),
                        }
                        write_entry(h5f, n_entries, seq, y_tile, gc_tile, attrs, mode=args.mode)
                        n_entries += 1
            except Exception as e:
                n_errors += 1
                logger.warning(f"  skip {row.get('Unique_ID', '?')}: {type(e).__name__}: {e}")
                continue

        h5f.attrs['ag_seq_len'] = AG_SEQ_LEN
        h5f.attrs['n_entries'] = int(n_entries)
        h5f.attrs['mode'] = args.mode
        h5f.attrs['donor'] = args.donor
        h5f.attrs['reference_only'] = bool(args.reference)

    logger.info(f"wrote {n_entries:,} entries ({n_tiled:,} tiled genes)")
    if n_skipped_empty:
        logger.info(f"  skipped (no splice sites): {n_skipped_empty}")
    if n_errors:
        logger.info(f"  skipped (errors): {n_errors}")


if __name__ == "__main__":
    main()
