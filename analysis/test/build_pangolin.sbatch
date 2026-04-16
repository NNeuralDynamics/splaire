#!/bin/bash
#SBATCH --job-name=build_pangolin
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build pangolin benchmark h5 for one tissue (CPU only)
# usage: sbatch build_pangolin.sbatch output_dir tissue
# tissue: heart, liver, brain, or testis

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_pangolin.sbatch output_dir tissue}"
tissue="${2:?usage: sbatch build_pangolin.sbatch output_dir tissue}"
genome_fasta="$repo_root/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa"
paralogs="$repo_root/analysis/test/data/paralogs.txt"
script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"
splice_table="$repo_root/analysis/test/data/splice_table_Human.test.txt"

source ~/.bashrc
conda activate sp

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "tissue: $tissue"
echo "started: $(date)"

if [ ! -f "$splice_table" ]; then
    echo "ERROR: splice_table not found: $splice_table"
    echo "run: curl -L -o data/splice_table_Human.test.txt <url> (see readme)"
    exit 1
fi

# convert splice_table to tissue matrix
tissue_matrix="$output_dir/${tissue}_matrix.tsv"
if [ ! -f "$tissue_matrix" ]; then
    echo "converting splice_table to $tissue matrix..."
    python "$script_dir/pangolin_to_matrix.py" "$splice_table" "$tissue_matrix" \
        --tissue "$tissue" --chroms chr1 chr3 chr5 chr7 --paralogs "$paralogs"
fi

# build h5
tissue_h5="$output_dir/${tissue}.h5"
if [ ! -f "$tissue_h5" ]; then
    echo "building $tissue h5..."
    cd "$pipeline_dir"
    python build_h5.py \
        --donor PANGOLIN --split test \
        --input "$tissue_matrix" \
        --chroms chr1,chr3,chr5,chr7 \
        --fasta "$genome_fasta" \
        --output "$tissue_h5" \
        --mode basic --paralog 0 --make-gc \
        --remove-missing --skip-empty-windows
fi

echo "done: $(date)"
