#!/usr/bin/env python3
"""
build h5 dataset for a single donor.

consolidates the 5-step ml data pipeline into one script:
  1. filter_vars - filter variants by donor and remove all-777 rows
  2. select_chrom_and_sample - filter to donor's samples and chromosomes
  3. adjust_sites - adjust transcript positions for indels
  4. mutate_sequences - extract and mutate genomic sequences
  5. create_dataset - build h5 dataset

usage:
    python build_donor_dataset.py \
        --donor GTEX-12WSH \
        --split test \
        --input enriched_matrix.tsv \
        --chroms chr1,chr3,chr5 \
        --fasta genome.fa \
        --output test_GTEX-12WSH.h5 \
        --work-dir ./work \
        --mode basic \
        --paralog 0
"""

import argparse
import os
import sys
import logging
import tempfile
import subprocess
import uuid
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict

# add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from constants import CL_max, SL
from utils import create_datapoints, GC_DTYPE


def setup_logging(log_file):
    """configure logging to file and stdout"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # stdout handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# -----------------------------------------------------------------------------
# step 1: filter_vars
# -----------------------------------------------------------------------------

def filter_vars(df, donor, min_count=1):
    """filter variants: keep only donor's data, filter rare variants

    NOTE: all-777 rows (holder rows from GENCODE filling) are now KEPT.
    The model masks these in the loss but still learns from sequence context.
    """
    initial = len(df)

    def only_empty(val):
        """check if value is empty/missing (not just 777)"""
        if pd.isna(val) or not isinstance(val, str) or not val.strip():
            return True
        return False

    # only remove truly empty rows, keep all-777 rows (holder rows)
    mask_bad = df['exon_end_SSUs'].apply(only_empty) | df['exon_start_SSUs'].apply(only_empty)
    df = df.loc[~mask_bad].copy()
    logging.info(f"filter_vars: removed {mask_bad.sum()} empty rows, {len(df)} remain")

    # filter to donor
    df['Sample'] = df['Unique_ID'].str.split('_').str[0]
    df = df[df['Sample'] == donor].copy()
    logging.info(f"filter_vars: filtered to donor {donor}, {len(df)} rows")

    # count variant|strand pairs and filter by min_count
    if min_count > 1 and len(df) > 0:
        logging.info(f"filter_vars: counting variant|strand pairs (min_count={min_count})")
        var_strand_pairs = []
        for vs, strand in zip(df['Variants'].str.split(','), df['Strand']):
            if isinstance(vs, list):
                var_strand_pairs.extend(f"{v}|{strand}" for v in vs if v)
        counts = Counter(var_strand_pairs)
        logging.info(f"filter_vars: found {len(counts)} unique variant|strand pairs")

        total_before = df['Variants'].dropna().apply(
            lambda x: len(x.split(',')) if isinstance(x, str) and x else 0
        ).sum()

        def filter_row_variants(row):
            vs = row['Variants']
            if not isinstance(vs, str) or not vs:
                return ''
            strand = row['Strand']
            kept = [v for v in vs.split(',') if counts.get(f"{v}|{strand}", 0) >= min_count]
            return ','.join(kept)

        df['Variants'] = df.apply(filter_row_variants, axis=1)

        total_after = df['Variants'].dropna().apply(
            lambda x: len(x.split(',')) if x else 0
        ).sum()
        logging.info(f"filter_vars: filtered {total_before - total_after} variants (min_count={min_count})")

    return df


# -----------------------------------------------------------------------------
# step 2: select_chrom_and_sample
# -----------------------------------------------------------------------------

def select_chrom_samples(df, chromosomes, donor, asymmetric_paralog=False):
    """filter to specific chromosomes and donor.

    if asymmetric_paralog=True, paralogs are kept from ALL chromosomes,
    only non-paralogs are filtered to specified chromosomes.
    """
    if asymmetric_paralog and 'paralog_status' in df.columns:
        is_paralog = df['paralog_status'].astype(str) == '1'
        is_in_chroms = df['Chromosome'].isin(chromosomes)
        df = df[is_paralog | is_in_chroms].copy()
        n_para = is_paralog.sum()
        n_nonpara = (~is_paralog & is_in_chroms).sum()
        logging.info(f"select_chrom_samples: asymmetric mode, {n_para} paralogs (all chroms) + {n_nonpara} non-paralogs ({len(chromosomes)} chroms), {len(df)} total")
    else:
        df = df[df['Chromosome'].isin(chromosomes)].copy()
        logging.info(f"select_chrom_samples: filtered to {len(chromosomes)} chromosomes, {len(df)} rows")
    return df


# -----------------------------------------------------------------------------
# step 3: adjust_sites
# -----------------------------------------------------------------------------

def adjust_row(row, var_counts):
    """adjust transcript coordinates for variants (indels shift downstream positions)"""
    tx_start = int(row["Start"])
    tx_end = int(row["End"])
    original_tx_start = tx_start
    original_tx_end = tx_end
    original_variants = str(row.get("Variants", "")).strip()

    # parse exon coords
    ends_str = str(row.get("exon_ends", "")).strip(",")
    starts_str = str(row.get("exon_starts", "")).strip(",")
    exon_ends = [int(p) for p in ends_str.split(",") if p]
    exon_starts = [int(p) for p in starts_str.split(",") if p]

    cumulative_offset = 0
    adjusted_variants = []

    if original_variants and original_variants not in ("None", "nan"):
        variant_list = [v.strip() for v in original_variants.split(",") if v.strip()]
        try:
            variant_list_sorted = sorted(variant_list, key=lambda x: int(x.split(":")[1]))
        except Exception:
            logging.warning(f"cannot parse variant positions in {row['Unique_ID']}")
            row["Original_Start"] = original_tx_start
            row["Original_End"] = original_tx_end
            row["Original_Variants"] = original_variants
            return row

        for variant in variant_list_sorted:
            chrom, pos_str, ref, alt = variant.split(":")
            pos = int(pos_str)

            if pos < original_tx_start or pos + len(ref) - 1 > original_tx_end:
                var_counts['out_of_bounds'] += 1
                continue

            adjusted_pos = pos + cumulative_offset
            adjusted_variants.append(f"{chrom}:{adjusted_pos}:{ref}:{alt}")
            net_change = len(alt) - len(ref)

            # track variant types
            if len(ref) == 1 and len(alt) == 1:
                var_counts['snv'] += 1
            elif len(ref) > len(alt):
                var_counts['deletion'] += 1
            elif len(ref) < len(alt):
                var_counts['insertion'] += 1
            else:
                var_counts['substitution'] += 1

            # adjust exon coords
            exon_starts = [p + net_change if p >= adjusted_pos else p for p in exon_starts]
            exon_ends = [p + net_change if p >= adjusted_pos else p for p in exon_ends]
            tx_end += net_change
            cumulative_offset += net_change

    row["Start"] = original_tx_start
    row["End"] = tx_end
    row["Variants"] = ",".join(adjusted_variants)
    row["exon_ends"] = ",".join(map(str, exon_ends))
    row["exon_starts"] = ",".join(map(str, exon_starts))
    row["Original_Start"] = original_tx_start
    row["Original_End"] = original_tx_end
    row["Original_Variants"] = original_variants

    return row


def adjust_sites(df):
    """adjust all rows for variant-induced position shifts"""
    logging.info(f"adjust_sites: processing {len(df)} rows")
    var_counts = Counter()
    adjusted_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="adjusting sites"):
        adjusted_rows.append(adjust_row(row, var_counts))

    logging.info(f"adjust_sites: variant types - snv={var_counts['snv']}, "
                 f"deletion={var_counts['deletion']}, insertion={var_counts['insertion']}, "
                 f"substitution={var_counts['substitution']}, out_of_bounds={var_counts['out_of_bounds']}")
    return pd.DataFrame(adjusted_rows)


# -----------------------------------------------------------------------------
# step 4: mutate_sequences
# -----------------------------------------------------------------------------

def extract_sequences(df, fasta_path, is_reference=False):
    """extract sequences from genome fasta and optionally apply variants"""
    unique_id = uuid.uuid4().hex
    bed_file = f"temp_{unique_id}.bed"
    fasta_file = f"temp_{unique_id}.fasta"

    # write bed file
    with open(bed_file, "w") as bed:
        for _, row in df.iterrows():
            chrom = row["Chromosome"]
            uid = row["Unique_ID"]
            if is_reference:
                start = int(row["Start"]) - 1
                end = int(row["End"])
                original_start = int(row["Start"])
                original_end = int(row["End"])
            else:
                start = int(row["Start"]) - 1
                end = int(row["End"])
                original_start = int(row["Original_Start"])
                original_end = int(row["Original_End"])
            bed.write(f"{chrom}\t{start}\t{original_end}\t{uid}::{chrom}:{start}-{end}:{original_start}-{original_end}\n")

    # extract sequences
    subprocess.run(["bedtools", "getfasta", "-fi", fasta_path, "-bed", bed_file, "-fo", fasta_file, "-name"], check=True)
    os.remove(bed_file)

    # process sequences
    sequences = {}
    unique_ids_set = set(df["Unique_ID"].astype(str))

    for record in SeqIO.parse(fasta_file, "fasta"):
        header_parts = record.id.split("::")
        if len(header_parts) < 2:
            continue

        uid = header_parts[0].strip()
        chrom_and_coords = header_parts[1]
        chrom, adjusted_coords, original_coords = chrom_and_coords.split(":")
        original_start, original_end = map(int, original_coords.split("-"))
        start, end = map(int, adjusted_coords.split("-"))

        if uid not in unique_ids_set:
            continue

        sequence = str(record.seq)

        if not is_reference:
            row = df[df["Unique_ID"].astype(str) == uid].iloc[0]
            variants = row["Variants"]
            if pd.notna(variants) and str(variants).strip():
                # find multi-allelic positions (same position appears multiple times)
                variant_list = str(variants).split(",")
                position_counts = {}
                for v in variant_list:
                    parts = v.split(":")
                    if len(parts) == 4:
                        pos = int(parts[1])
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                multiallelic_positions = {p for p, c in position_counts.items() if c > 1}

                for variant in variant_list:
                    parts = variant.split(":")
                    if len(parts) != 4:
                        continue
                    vchrom, pos_str, ref, alt = parts
                    pos = int(pos_str) - 1

                    # skip multi-allelic sites - keep reference at these positions
                    if int(pos_str) in multiallelic_positions:
                        continue

                    relative_pos = pos - start

                    # validate ref
                    extracted_ref = sequence[relative_pos:relative_pos + len(ref)]
                    if extracted_ref != ref:
                        logging.warning(f"ref mismatch in {uid}: expected {ref}, found {extracted_ref}")
                        continue

                    # apply variant
                    if len(ref) > len(alt):  # deletion
                        sequence = sequence[:relative_pos + 1] + sequence[relative_pos + len(ref):]
                    else:  # insertion or SNV
                        sequence = sequence[:relative_pos] + alt + sequence[relative_pos + len(ref):]

        sequences[uid] = (f"{chrom}:{original_start}-{end}", sequence)

    os.remove(fasta_file)
    return sequences


def mutate_sequences(df, fasta_path):
    """generate reference and variant sequences"""
    logging.info("mutate_sequences: extracting reference sequences")
    ref_seqs = extract_sequences(df, fasta_path, is_reference=True)

    logging.info("mutate_sequences: extracting variant sequences")
    var_seqs = extract_sequences(df, fasta_path, is_reference=False)

    return ref_seqs, var_seqs


# -----------------------------------------------------------------------------
# step 5: create_dataset
# -----------------------------------------------------------------------------

CHUNK_SIZE = 2000


def process_record(data1, data2, encoding_mode, remove_missing):
    """process single record for h5 dataset"""
    # data1 = splice table row as list
    # data2 = sequence tuple (coords, sequence)

    coords, seq = data2

    # parse coords
    import re
    coord_parts = re.split(':|-', coords)
    chrom = coord_parts[0]
    start_coord = int(coord_parts[1])
    end_coord = int(coord_parts[2])

    tx_start = int(float(data1['Start']))
    tx_end = int(float(data1['End']))

    # jn_start/jn_end refer to junction boundaries, not exon boundaries
    # jn_start = donor site = exon_end (where exon ends, intron starts)
    # jn_end = acceptor site = exon_start (where intron ends, exon starts)
    kwargs = {
        'seq': seq,
        'strand': data1['Strand'],
        'tx_start': tx_start,
        'tx_end': tx_end,
        'jn_start': [data1['exon_ends']],
        'jn_end': [data1['exon_starts']],
        'jn_start_sse': [data1['exon_end_SSUs']],
        'jn_end_sse': [data1['exon_start_SSUs']],
        'chrom': chrom,
        'name': data1['Unique_ID'],
        'remove_missing': remove_missing,
        'mode': encoding_mode,
    }

    return create_datapoints(**kwargs)


def create_dataset_h5(df, var_seqs, output_path, split_mode, paralog, encoding_mode, make_gc, remove_missing, skip_empty=False):
    """build h5 dataset from dataframe and sequences"""
    logging.info(f"create_dataset: building h5 for {len(df)} records (skip_empty={skip_empty})")

    with h5py.File(output_path, 'w') as h5f:
        buffer_x, buffer_y, buffer_gc = [], [], []
        chunk_idx = 0
        processed = 0
        n_windows_kept = 0
        n_windows_skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="building dataset"):
            uid = row['Unique_ID']

            # filter by paralog
            if paralog != 'all' and str(row.get('paralog_status', '')) != str(paralog):
                continue

            if uid not in var_seqs:
                logging.warning(f"missing sequence for {uid}")
                continue

            try:
                x, y, gc = process_record(row.to_dict(), var_seqs[uid], encoding_mode, remove_missing)

                if skip_empty:
                    # keep only windows with at least one splice site
                    has_site = np.array([yi[:, 1].sum() + yi[:, 2].sum() >= 1 for yi in y])
                    n_windows_skipped += (~has_site).sum()
                    n_windows_kept += has_site.sum()
                    x = x[has_site]
                    y = y[has_site]
                    if make_gc and gc is not None:
                        gc = gc[has_site]
                    if len(x) == 0:
                        continue

                buffer_x.extend(x)
                buffer_y.extend(y)
                if make_gc and gc is not None:
                    buffer_gc.extend(gc)
                processed += 1
            except Exception as e:
                logging.warning(f"error processing {uid}: {e}")
                continue

            # flush chunk
            if len(buffer_x) >= CHUNK_SIZE:
                x_array = np.asarray(buffer_x, dtype='float32')
                y_array = np.asarray(buffer_y, dtype='float32')
                h5f.create_dataset(f"X{chunk_idx}", data=x_array, compression="gzip", compression_opts=4)
                h5f.create_dataset(f"Y{chunk_idx}", data=y_array, compression="gzip", compression_opts=4)
                if make_gc and buffer_gc:
                    gc_array = np.asarray(buffer_gc, dtype=GC_DTYPE)
                    h5f.create_dataset(f"GC{chunk_idx}", data=gc_array, compression="gzip", compression_opts=4)
                chunk_idx += 1
                buffer_x.clear()
                buffer_y.clear()
                buffer_gc.clear()

        # final chunk
        if buffer_x:
            x_array = np.asarray(buffer_x, dtype='float32')
            y_array = np.asarray(buffer_y, dtype='float32')
            h5f.create_dataset(f"X{chunk_idx}", data=x_array, compression="gzip", compression_opts=4)
            h5f.create_dataset(f"Y{chunk_idx}", data=y_array, compression="gzip", compression_opts=4)
            if make_gc and buffer_gc:
                gc_array = np.asarray(buffer_gc, dtype=GC_DTYPE)
                h5f.create_dataset(f"GC{chunk_idx}", data=gc_array, compression="gzip", compression_opts=4)

    if skip_empty:
        logging.info(f"create_dataset: skip_empty kept {n_windows_kept:,} windows, skipped {n_windows_skipped:,}")
    logging.info(f"create_dataset: wrote {processed} records to {output_path}")
    return processed


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="build h5 dataset for a single donor")
    parser.add_argument("--donor", required=True, help="donor ID (e.g. GTEX-12WSH)")
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"], help="dataset split")
    parser.add_argument("--input", required=True, help="input enriched matrix TSV")
    parser.add_argument("--chroms", required=True, help="comma-separated chromosomes")
    parser.add_argument("--fasta", required=True, help="reference genome FASTA")
    parser.add_argument("--output", required=True, help="output h5 file path")
    parser.add_argument("--work-dir", default=".", help="working directory for intermediate files")
    parser.add_argument("--log-dir", default=None, help="log directory (default: work-dir/logs)")
    parser.add_argument("--mode", default="basic", choices=["basic", "het", "pop"], help="encoding mode")
    parser.add_argument("--paralog", default="all", help="paralog filter: 0, 1, or all")
    parser.add_argument("--make-gc", action="store_true", help="include genomic coordinates")
    parser.add_argument("--remove-missing", action="store_true", help="treat SSU==777 as no splice site")
    parser.add_argument("--min-count", type=int, default=1, help="min variant|strand occurrences (default: 1, no filtering)")
    parser.add_argument("--asymmetric-paralog", action="store_true", help="keep paralogs from all chroms, filter non-paralogs by --chroms")
    parser.add_argument("--reference", action="store_true", help="use reference sequences (no donor SNVs inserted)")
    parser.add_argument("--skip-empty-windows", action="store_true", help="skip windows with no splice sites (acceptor or donor)")
    args = parser.parse_args()

    # setup directories
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # setup logging
    log_file = log_dir / f"{args.split}_{args.donor}.log"
    logger = setup_logging(str(log_file))
    logger.info(f"build_donor_dataset: donor={args.donor}, split={args.split}")
    logger.info(f"input={args.input}, output={args.output}")

    # parse chromosomes
    chromosomes = [c.strip() for c in args.chroms.split(",") if c.strip()]
    logger.info(f"chromosomes: {chromosomes}")

    # load input
    logger.info("loading input matrix")
    df = pd.read_csv(args.input, sep="\t")
    logger.info(f"loaded {len(df)} rows")

    # step 1: filter variants
    df = filter_vars(df, args.donor, args.min_count)
    if len(df) == 0:
        logger.warning("no data after filter_vars, creating empty dataset")
        with h5py.File(args.output, 'w') as h5f:
            h5f.create_dataset("X0", data=np.array([], dtype='float32'))
            h5f.create_dataset("Y0", data=np.array([], dtype='float32'))
        return

    # step 2: select chromosomes and samples
    df = select_chrom_samples(df, chromosomes, args.donor, args.asymmetric_paralog)
    if len(df) == 0:
        logger.warning("no data after select_chrom_samples, creating empty dataset")
        with h5py.File(args.output, 'w') as h5f:
            h5f.create_dataset("X0", data=np.array([], dtype='float32'))
            h5f.create_dataset("Y0", data=np.array([], dtype='float32'))
        return

    # step 3: adjust sites
    df = adjust_sites(df)

    # step 4: mutate sequences
    ref_seqs, var_seqs = mutate_sequences(df, args.fasta)
    seqs = ref_seqs if args.reference else var_seqs

    # step 5: create dataset
    n_records = create_dataset_h5(
        df, seqs, args.output,
        args.split, args.paralog, args.mode,
        args.make_gc, args.remove_missing,
        args.skip_empty_windows
    )

    logger.info(f"build_donor_dataset complete: {n_records} records written")


if __name__ == "__main__":
    main()
