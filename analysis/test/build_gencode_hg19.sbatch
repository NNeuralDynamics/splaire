#!/bin/bash
#SBATCH --job-name=build_gencode_hg19
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build the legacy SpliceAI 2019 hg19 canonical bench (CPU only)
# input is data/canonical_dataset.txt, the original SpliceAI test set (hg19 coords)
# does NOT match what the SPLAIRE paper reports — use build_gencode.sbatch for that
# usage: sbatch build_gencode_hg19.sbatch output_dir

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_gencode_hg19.sbatch output_dir}"
genome_fasta="${HG19_FASTA:-$repo_root/pipeline/reference/GRCh37/hg19.fa}"
sites="$repo_root/analysis/test/data/canonical_dataset.txt"
script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "fasta: $genome_fasta"
echo "started: $(date)"

# convert canonical sites to matrix (hg19 coords pass through)
matrix="$output_dir/gencode_hg19_matrix.tsv"
if [ ! -f "$matrix" ]; then
    echo "converting to matrix..."
    python "$script_dir/canonical_to_matrix.py" "$sites" "$matrix"
fi

# build h5 with hg19 fasta
h5="$output_dir/gencode_hg19.h5"
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
