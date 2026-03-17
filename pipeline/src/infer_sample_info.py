#!/usr/bin/env python3
"""
infer read length and strandness from bam or fastq files.

usage:
    python infer_sample_info.py sample.bam
    python infer_sample_info.py sample_R1.fastq.gz
"""
import argparse
import subprocess
import sys
from pathlib import Path


def get_read_length_from_bam(bam_path, num_reads=1000):
    """get most common read length from bam file."""
    cmd = f"samtools view {bam_path} | head -{num_reads} | awk '{{print length($10)}}' | sort | uniq -c | sort -rn | head -1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"error reading bam: {result.stderr}", file=sys.stderr)
        return None
    parts = result.stdout.strip().split()
    if len(parts) >= 2:
        return int(parts[1])
    return None


def get_read_length_from_fastq(fastq_path, num_reads=1000):
    """get most common read length from fastq file."""
    # handle gzipped or plain fastq
    cat_cmd = "zcat -f" if str(fastq_path).endswith('.gz') else "cat"
    lines_needed = num_reads * 4
    cmd = f"{cat_cmd} {fastq_path} | head -{lines_needed} | awk 'NR%4==2 {{print length}}' | sort | uniq -c | sort -rn | head -1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"error reading fastq: {result.stderr}", file=sys.stderr)
        return None
    parts = result.stdout.strip().split()
    if len(parts) >= 2:
        return int(parts[1])
    return None


def main():
    parser = argparse.ArgumentParser(description="infer read length from bam or fastq")
    parser.add_argument("input_file", help="path to bam or fastq file")
    parser.add_argument("-n", "--num-reads", type=int, default=1000, help="number of reads to sample (default: 1000)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # detect file type and get read length
    if input_path.suffix == '.bam':
        read_length = get_read_length_from_bam(input_path, args.num_reads)
        file_type = "bam"
    elif input_path.suffix in ['.fastq', '.fq'] or input_path.name.endswith(('.fastq.gz', '.fq.gz')):
        read_length = get_read_length_from_fastq(input_path, args.num_reads)
        file_type = "fastq"
    else:
        print(f"error: unrecognized file type: {input_path.suffix}", file=sys.stderr)
        sys.exit(1)

    if read_length is None:
        print("error: could not determine read length", file=sys.stderr)
        sys.exit(1)

    overhang = read_length - 1

    print(f"file: {input_path}")
    print(f"type: {file_type}")
    print(f"read_length: {read_length}")
    print(f"star_genome_overhang: {overhang}")
    print()
    print("use this when building star index:")
    print(f"  --star_genome_overhang {overhang}")


if __name__ == "__main__":
    main()
