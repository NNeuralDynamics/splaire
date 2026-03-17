#!/usr/bin/env python3
"""create samplesheet from bam or fastq files"""
import argparse
import subprocess
import sys
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_read_length_from_bam(bam_path, num_reads=1000):
    cmd = f"samtools view {bam_path} 2>/dev/null | head -{num_reads} | awk '{{print length($10)}}' | sort | uniq -c | sort -rn | head -1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = result.stdout.strip().split()
    if len(parts) >= 2:
        return int(parts[1])
    return None


def get_read_length_from_fastq(fastq_path, num_reads=1000):
    cat_cmd = "zcat -f" if str(fastq_path).endswith('.gz') else "cat"
    lines_needed = num_reads * 4
    cmd = f"{cat_cmd} {fastq_path} 2>/dev/null | head -{lines_needed} | awk 'NR%4==2 {{print length}}' | sort | uniq -c | sort -rn | head -1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = result.stdout.strip().split()
    if len(parts) >= 2:
        return int(parts[1])
    return None


def get_strandness_from_bam(bam_path, ref_bed):
    cmd = f"infer_experiment.py -r {ref_bed} -i {bam_path} 2>/dev/null"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr

    # parse rseqc output

    output = result.stdout
    fr_match = re.search(r'Fraction of reads explained by "1\+\+,1--,2\+-,2-\+":\s*([\d.]+)', output)
    rf_match = re.search(r'Fraction of reads explained by "1\+-,1-\+,2\+\+,2--":\s*([\d.]+)', output)

    if not fr_match or not rf_match:
        # try alternative format for single-end
        fr_match = re.search(r'Fraction of reads explained by "\+\+":\s*([\d.]+)', output)
        rf_match = re.search(r'Fraction of reads explained by "\+-":\s*([\d.]+)', output)

    if not fr_match or not rf_match:
        return "unknown", output

    fr_frac = float(fr_match.group(1))
    rf_frac = float(rf_match.group(1))

    # determine strandness
    if rf_frac > 0.8:
        return "rf", output
    elif fr_frac > 0.8:
        return "fr", output
    elif abs(fr_frac - rf_frac) < 0.2:
        return "unstranded", output
    else:
        return "unknown", output


def extract_sample_id(filepath):
    name = Path(filepath).name
    # remove common suffixes
    for suffix in ['.Aligned.sortedByCoord.out.patched.md.bam', '.bam', '.sorted.bam',
                   '_1.fastq.gz', '_2.fastq.gz', '_1.fastq', '_2.fastq',
                   '_R1.fastq.gz', '_R2.fastq.gz', '_R1.fastq', '_R2.fastq',
                   '.fastq.gz', '.fastq', '.fq.gz', '.fq']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def extract_donor_id(sample_id):
    # gtex format: GTEX-XXXXX-YYYY-SM-ZZZZZ → GTEX-XXXXX
    parts = sample_id.split('-')
    if len(parts) >= 2 and parts[0] == 'GTEX':
        return f"{parts[0]}-{parts[1]}"
    return sample_id


def process_bam(bam_path, ref_bed, skip_read_length=False):
    sample_id = extract_sample_id(bam_path)
    donor_id = extract_donor_id(sample_id)

    # get strandness
    strandness, _ = get_strandness_from_bam(bam_path, ref_bed)

    # get read length
    read_length = None
    if not skip_read_length:
        read_length = get_read_length_from_bam(bam_path)

    return {
        'sample_id': sample_id,
        'donor_id': donor_id,
        'bam': str(bam_path),
        'strandness': strandness,
        'read_length': read_length
    }


def process_fastq_pair(fastq_1, fastq_2, ref_bed, skip_read_length=False):
    sample_id = extract_sample_id(fastq_1)
    donor_id = extract_donor_id(sample_id)

    # strandness inferred after alignment
    strandness = "unknown"

    # get read length
    read_length = None
    if not skip_read_length:
        read_length = get_read_length_from_fastq(fastq_1)

    return {
        'sample_id': sample_id,
        'donor_id': donor_id,
        'fastq_1': str(fastq_1),
        'fastq_2': str(fastq_2),
        'strandness': strandness,
        'read_length': read_length
    }


def find_files(input_dir):
    input_path = Path(input_dir)

    # check for bams
    bams = list(input_path.glob('*.bam'))
    if bams:
        return 'bam', sorted(bams)

    # check for fastqs
    fastqs = sorted(input_path.glob('*.fastq*')) + sorted(input_path.glob('*.fq*'))
    if fastqs:
        # pair them up
        pairs = {}
        for fq in fastqs:
            name = fq.name
            # find the pair indicator
            if '_1.' in name or '_R1.' in name or '_1_' in name:
                key = re.sub(r'[_.]R?1[_.]', '_X_', name)
                if key not in pairs:
                    pairs[key] = [None, None]
                pairs[key][0] = fq
            elif '_2.' in name or '_R2.' in name or '_2_' in name:
                key = re.sub(r'[_.]R?2[_.]', '_X_', name)
                if key not in pairs:
                    pairs[key] = [None, None]
                pairs[key][1] = fq

        # filter complete pairs
        valid_pairs = [(p[0], p[1]) for p in pairs.values() if p[0] and p[1]]
        return 'fastq', valid_pairs

    return None, []


def print_summary(samples, file_type):
    print("\n" + "=" * 50)
    print("SAMPLE INFO SUMMARY")
    print("=" * 50)
    print(f"\nsamples processed: {len(samples)}")
    print(f"file type: {file_type}")

    # read length summary
    read_lengths = [s['read_length'] for s in samples if s['read_length'] is not None]
    if read_lengths:
        print("\n--- read length ---")
        length_counts = Counter(read_lengths)
        if len(length_counts) == 1:
            length = list(length_counts.keys())[0]
            print(f"  all samples: {length} bp")
            print(f"  recommended star_genome_overhang: {length - 1}")
        else:
            print("  WARNING: mixed read lengths")
            for length, count in sorted(length_counts.items()):
                print(f"    {length} bp: {count} samples")
            min_length = min(length_counts.keys())
            print(f"  recommended star_genome_overhang: {min_length - 1} (use shortest)")

    # strandness summary
    strandness_values = [s['strandness'] for s in samples if s['strandness']]
    if strandness_values:
        print("\n--- strandness ---")
        strand_counts = Counter(strandness_values)
        if len(strand_counts) == 1:
            strand = list(strand_counts.keys())[0]
            strand_desc = {
                'rf': 'reverse-forward (stranded)',
                'fr': 'forward-reverse (stranded)',
                'unstranded': 'unstranded',
                'unknown': 'unknown (will infer after alignment)'
            }.get(strand, strand)
            print(f"  all samples: {strand} ({strand_desc})")
        else:
            print("  WARNING: mixed strandness - check samples")
            for strand, count in sorted(strand_counts.items()):
                print(f"    {strand}: {count} samples")

    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(description="create samplesheet with strandness and read length detection")
    parser.add_argument("--input-dir", required=True, help="directory containing bam or fastq files")
    parser.add_argument("--ref-bed", required=True, help="reference bed file for strandness inference")
    parser.add_argument("--output", "-o", required=True, help="output samplesheet tsv")
    parser.add_argument("--skip-read-length", action="store_true", help="skip read length detection")
    parser.add_argument("--threads", "-t", type=int, default=4, help="number of parallel threads (default: 4)")
    args = parser.parse_args()

    # check ref bed exists
    ref_bed = Path(args.ref_bed)
    if not ref_bed.exists():
        print(f"error: ref-bed not found: {ref_bed}", file=sys.stderr)
        sys.exit(1)

    # find files
    file_type, files = find_files(args.input_dir)
    if not files:
        print(f"error: no bam or fastq files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(files)} {file_type} files")

    # process files
    samples = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        if file_type == 'bam':
            futures = {executor.submit(process_bam, f, ref_bed, args.skip_read_length): f for f in files}
        else:
            futures = {executor.submit(process_fastq_pair, f1, f2, ref_bed, args.skip_read_length): (f1, f2) for f1, f2 in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="processing samples"):
            try:
                result = future.result()
                samples.append(result)
            except Exception as e:
                print(f"error processing {futures[future]}: {e}", file=sys.stderr)

    # sort by sample_id
    samples.sort(key=lambda x: x['sample_id'])

    # check for missing read lengths
    missing_read_length = [s['sample_id'] for s in samples if s['read_length'] is None]

    # write samplesheet
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        if file_type == 'bam':
            f.write("sample_id\tdonor_id\tbam\tstrandness\tread_length\n")
            for s in samples:
                read_len = s['read_length'] if s['read_length'] else ""
                f.write(f"{s['sample_id']}\t{s['donor_id']}\t{s['bam']}\t{s['strandness']}\t{read_len}\n")
        else:
            f.write("sample_id\tdonor_id\tfastq_1\tfastq_2\tstrandness\tread_length\n")
            for s in samples:
                read_len = s['read_length'] if s['read_length'] else ""
                f.write(f"{s['sample_id']}\t{s['donor_id']}\t{s['fastq_1']}\t{s['fastq_2']}\t{s['strandness']}\t{read_len}\n")

    print(f"\nsamplesheet written: {output_path}")

    # warn about missing read lengths
    if missing_read_length:
        print(f"\nWARNING: read_length could not be determined for {len(missing_read_length)} samples:", file=sys.stderr)
        for sid in missing_read_length[:10]:
            print(f"  - {sid}", file=sys.stderr)
        if len(missing_read_length) > 10:
            print(f"  ... and {len(missing_read_length) - 10} more", file=sys.stderr)
        print(f"\nYou must fill in the read_length column before running the pipeline.", file=sys.stderr)

    # print summary
    print_summary(samples, file_type)


if __name__ == "__main__":
    main()
