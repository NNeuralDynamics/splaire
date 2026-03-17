#!/bin/bash
#SBATCH --job-name=build_gencode
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build gencode benchmark h5 (CPU only)
# usage: sbatch build_gencode.sbatch output_dir

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_gencode.sbatch output_dir}"
genome_fasta="$repo_root/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa"
sites="$repo_root/analysis/test/data/canonical_dataset.txt"
script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"

source ~/.bashrc
conda activate sp

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "started: $(date)"

# convert pre-made canonical sites to matrix
matrix="$output_dir/gencode_matrix.tsv"
if [ ! -f "$matrix" ]; then
    echo "converting to matrix..."
    python "$script_dir/canonical_to_matrix.py" "$sites" "$matrix"
fi

# build h5 (note: canonical_dataset.txt uses hg19 coordinates)
h5="$output_dir/gencode.h5"
if [ ! -f "$h5" ]; then
    echo "building h5..."
    cd "$pipeline_dir"
    python build_h5.py \
        --donor CANONICAL --split test \
        --input "$matrix" \
        --chroms chr1,chr3,chr5,chr7 \
        --fasta "$genome_fasta" \
        --output "$h5" \
        --mode basic --paralog 0 --make-gc
fi

echo "done: $(date)"
