#!/bin/bash
# submit_fillSamples.sh
# Usage: ./submit_fillSamples.sh /scratch/runyan.m/HAEC-185-pipeline/SamplesFile.tsv /scratch/runyan.m/HAEC-185-pipeline/all_sites_master.tsv /path/to/output_dir

set -euo pipefail

samples_file="$1"
master_list="$2"
outdir="$3"

mkdir -p "$outdir/logs"

while IFS=$'\t' read -r sample processed_tsv bam; do
    # skip header
    [[ "$sample" == "Sample" ]] && continue

    output="${outdir}/${sample}.complete.tsv"
    log="${outdir}/${sample}.fillSample.log"

    sbatch --job-name="fill_${sample}" \
           --output="$log" \
           --time=02:00:00 \
           --mem=8G \
           --cpus-per-task=2 \
           --wrap="python SpliSER.py fillSample \
                   -m $master_list \
                   -i $processed_tsv \
                   -B $bam \
                   -o $output \
                   --isStranded -s fr
done < "$samples_file"
