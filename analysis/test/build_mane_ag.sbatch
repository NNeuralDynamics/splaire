#!/bin/bash
#SBATCH --job-name=build_mane_ag
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build mane select per-gene h5 for AlphaGenome (1Mb seq per gene, reference only)
# usage: sbatch build_mane_ag.sbatch output_dir

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_mane_ag.sbatch output_dir}"
genome_fasta="${REF_FASTA:?must export REF_FASTA=/path/to/GRCh38.primary_assembly.genome.fa}"
sites="${SITES:-$repo_root/analysis/test/data/canonical_dataset_mane_select_p.txt}"
script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "started: $(date)"

matrix="$output_dir/mane_select_matrix.tsv"
if [ ! -f "$matrix" ]; then
    echo "converting to matrix..."
    python "$script_dir/canonical_to_matrix.py" "$sites" "$matrix"
fi

h5="$output_dir/mane_select_ag.h5"
if [ ! -f "$h5" ]; then
    echo "building per-gene h5 (AG_SEQ_LEN=1048576, reference only)..."
    cd "$pipeline_dir"
    python build_h5_alphagenome.py \
        --donor CANONICAL --split test \
        --input "$matrix" \
        --chroms chr1,chr3,chr5,chr7,chr9 \
        --fasta "$genome_fasta" \
        --output "$h5" \
        --mode basic --paralog 0 \
        --reference --skip-empty-genes
fi

echo "done: $(date)"
