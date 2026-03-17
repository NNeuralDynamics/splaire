#!/usr/bin/env python3
"""
extract phased variants from vcf for gene regions.

consolidates 3 steps:
  1. bedtools intersect vcf with gene regions
  2. extract phased variants per gene (paternal/maternal)
  3. filter to SNVs only

outputs both full variants and SNV-only versions with detailed logging.
"""
import argparse
import subprocess
import tempfile
import logging
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import pysam


def setup_logging(log_file):
    """configure logging to file and stdout"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def intersect_vcf(vcf_path, bed_path, output_vcf):
    """step 1: bedtools intersect vcf with gene regions"""
    logging.info(f"intersecting VCF with gene regions BED")
    logging.info(f"  vcf: {vcf_path}")
    logging.info(f"  bed: {bed_path}")

    cmd = ["bedtools", "intersect", "-header", "-a", str(vcf_path), "-b", str(bed_path)]
    bedtools_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if bedtools_proc.stdout is None:
        logging.error("bedtools intersect failed to produce stdout")
        sys.exit(1)

    output_vcf = Path(output_vcf)
    n_variants = 0
    dropped_variants = 0
    header_col_count = None

    with output_vcf.open("wb") as bgzip_out:
        bgzip_proc = subprocess.Popen(
            ["bgzip", "-c"],
            stdin=subprocess.PIPE,
            stdout=bgzip_out,
            stderr=subprocess.PIPE,
        )

        if bgzip_proc.stdin is None:
            logging.error("failed to open bgzip stdin")
            sys.exit(1)

        def write_to_bgzip(line_str):
            try:
                bgzip_proc.stdin.write(line_str.encode("utf-8"))
            except BrokenPipeError:
                logging.error("bgzip terminated while writing intersection output")
                sys.exit(1)

        try:
            for line in bedtools_proc.stdout:
                if line.startswith("#"):
                    if line.startswith("#CHROM"):
                        header_col_count = len(line.rstrip("\n").split("\t"))
                    write_to_bgzip(line)
                    continue

                if header_col_count is None:
                    logging.error("bedtools output missing #CHROM header before variants")
                    sys.exit(1)

                cols = line.rstrip("\n").split("\t")
                if len(cols) < header_col_count:
                    dropped_variants += 1
                    continue

                write_to_bgzip(line)
                n_variants += 1
        finally:
            bedtools_proc.stdout.close()
            bgzip_proc.stdin.close()

        bedtools_stderr = bedtools_proc.stderr.read().strip()
        bedtools_return = bedtools_proc.wait()
        if bedtools_return != 0:
            logging.error(f"bedtools intersect failed: {bedtools_stderr}")
            sys.exit(1)

        bgzip_stderr = bgzip_proc.stderr.read().strip()
        bgzip_return = bgzip_proc.wait()
        if bgzip_return != 0:
            logging.error(f"bgzip failed: {bgzip_stderr}")
            sys.exit(1)

    subprocess.run(["tabix", "-p", "vcf", str(output_vcf)], check=True)
    logging.info(f"  variants in gene regions: {n_variants}")
    if dropped_variants:
        logging.info(f"  variants dropped missing genotype columns: {dropped_variants}")

    return output_vcf


def extract_phased_variants(vcf_path, genes_tsv, donor_id):
    """step 2: extract phased variants per gene region"""
    logging.info(f"extracting phased variants for donor {donor_id}")

    vcf = pysam.VariantFile(vcf_path)
    genes_df = pd.read_csv(genes_tsv, sep="\t")

    if donor_id not in vcf.header.samples:
        logging.error(f"donor '{donor_id}' not found in VCF samples")
        sys.exit(1)

    vcf_chroms = set(vcf.header.contigs.keys())
    expanded_rows = []

    # counters for logging
    stats = Counter()

    for _, row in genes_df.iterrows():
        gene_id = row["Gene_ID"]
        chrom = str(row["Chromosome"])
        start = int(row["Start"])
        end = int(row["End"])

        if chrom not in vcf_chroms:
            stats['genes_skipped_no_chrom'] += 1
            continue

        paternal_variants = []
        maternal_variants = []

        for record in vcf.fetch(chrom, start, end):
            stats['variants_examined'] += 1
            deletion_end = record.pos + len(record.ref) - 1

            # skip variants outside the gene region
            if not (start <= record.pos <= end):
                stats['variants_outside_region'] += 1
                continue
            # skip deletions that extend beyond the gene
            if len(record.ref) > len(record.alts[0]) and deletion_end > end:
                stats['variants_deletion_extends'] += 1
                continue
            # skip multi-allelic records
            if len(record.alts) > 1:
                stats['variants_multiallelic'] += 1
                continue

            genotype = record.samples[donor_id]["GT"]
            if genotype == (0, 0):
                stats['variants_homref'] += 1
                continue

            key = f"{record.chrom}:{record.pos}:{record.ref}:{record.alts[0]}"
            stats['variants_kept'] += 1

            # classify variant type
            ref_len, alt_len = len(record.ref), len(record.alts[0])
            if ref_len == 1 and alt_len == 1:
                stats['type_snv'] += 1
            elif ref_len > alt_len:
                stats['type_deletion'] += 1
            elif ref_len < alt_len:
                stats['type_insertion'] += 1
            else:
                stats['type_mnv'] += 1

            if genotype == (1, 0):
                paternal_variants.append(key)
            elif genotype == (0, 1):
                maternal_variants.append(key)
            elif genotype == (1, 1):
                paternal_variants.append(key)
                maternal_variants.append(key)
                stats['variants_homalt'] += 1

        # remove duplicates while preserving order
        paternal_variants = list(dict.fromkeys(paternal_variants))
        maternal_variants = list(dict.fromkeys(maternal_variants))

        # append one row for each allele
        expanded_rows.append({
            **row,
            "Allele": "Paternal",
            "Unique_ID": f"{donor_id}_{gene_id}_Paternal",
            "Variants": ",".join(paternal_variants) if paternal_variants else "None"
        })
        expanded_rows.append({
            **row,
            "Allele": "Maternal",
            "Unique_ID": f"{donor_id}_{gene_id}_Maternal",
            "Variants": ",".join(maternal_variants) if maternal_variants else "None"
        })

        stats['genes_processed'] += 1

    logging.info(f"  genes processed: {stats['genes_processed']}")
    logging.info(f"  genes skipped (no chrom): {stats['genes_skipped_no_chrom']}")
    logging.info(f"  variants examined: {stats['variants_examined']}")
    logging.info(f"  variants kept: {stats['variants_kept']}")
    logging.info(f"    - SNVs: {stats['type_snv']}")
    logging.info(f"    - deletions: {stats['type_deletion']}")
    logging.info(f"    - insertions: {stats['type_insertion']}")
    logging.info(f"    - MNVs: {stats['type_mnv']}")
    logging.info(f"  variants filtered:")
    logging.info(f"    - outside region: {stats['variants_outside_region']}")
    logging.info(f"    - deletion extends: {stats['variants_deletion_extends']}")
    logging.info(f"    - multi-allelic: {stats['variants_multiallelic']}")
    logging.info(f"    - hom ref: {stats['variants_homref']}")

    return pd.DataFrame(expanded_rows), stats


def filter_to_snvs(df):
    """step 3: filter variants column to SNVs only"""
    logging.info("filtering to SNVs only")

    def keep_snvs(variants_str):
        if pd.isna(variants_str) or not isinstance(variants_str, str):
            return ""
        variants_str = variants_str.strip()
        if variants_str == "None" or variants_str == "":
            return ""
        snv_list = []
        for key in variants_str.split(","):
            parts = key.split(":")
            if len(parts) != 4:
                continue
            chrom, pos, ref, alt = parts
            if len(ref) == 1 and len(alt) == 1:
                snv_list.append(key)
        return ",".join(snv_list) if snv_list else ""

    df_snv = df.copy()
    df_snv["Variants"] = df_snv["Variants"].apply(keep_snvs)

    # count how many rows have variants
    n_with_vars = (df["Variants"] != "None").sum()
    n_with_snvs = (df_snv["Variants"] != "").sum()
    logging.info(f"  rows with any variant: {n_with_vars}")
    logging.info(f"  rows with SNVs: {n_with_snvs}")

    return df_snv


def main():
    parser = argparse.ArgumentParser(description="extract phased variants from VCF for gene regions")
    parser.add_argument("-v", "--vcf", required=True, help="input VCF file (donor-filtered)")
    parser.add_argument("-b", "--bed", required=True, help="gene regions BED file")
    parser.add_argument("-g", "--genes", required=True, help="gene regions TSV file")
    parser.add_argument("-d", "--donor", required=True, help="donor ID")
    parser.add_argument("-o", "--output", required=True, help="output prefix (produces PREFIX.tsv and PREFIX_SNVs.tsv)")
    parser.add_argument("--log", help="log file path")
    args = parser.parse_args()

    # setup logging
    log_file = args.log or f"{args.output}.log"
    setup_logging(log_file)
    logging.info(f"=== extract_variants.py ===")
    logging.info(f"donor: {args.donor}")

    # step 1: intersect vcf with gene regions
    with tempfile.TemporaryDirectory() as tmpdir:
        intersected_vcf = Path(tmpdir) / "intersected.vcf.gz"
        intersect_vcf(args.vcf, args.bed, intersected_vcf)

        # step 2: extract phased variants
        df, stats = extract_phased_variants(intersected_vcf, args.genes, args.donor)

    # step 3: filter to SNVs
    df_snv = filter_to_snvs(df)

    # save outputs
    output_all = f"{args.output}.tsv"
    output_snv = f"{args.output}_SNVs.tsv"

    df.to_csv(output_all, sep="\t", index=False)
    df_snv.to_csv(output_snv, sep="\t", index=False)

    logging.info(f"output (all variants): {output_all}")
    logging.info(f"output (SNVs only): {output_snv}")
    logging.info("=== done ===")


if __name__ == "__main__":
    main()
