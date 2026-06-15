#!/bin/bash
#SBATCH --job-name=build_pangolin_ag
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build pangolin per-gene h5 for AlphaGenome (1Mb seq per gene)
# usage: sbatch build_pangolin_ag.sbatch output_dir tissue

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_pangolin_ag.sbatch output_dir tissue}"
tissue="${2:?usage: sbatch build_pangolin_ag.sbatch output_dir tissue}"
genome_fasta="${REF_FASTA:?must export REF_FASTA=/path/to/GRCh38.primary_assembly.genome.fa}"
paralogs="$repo_root/analysis/test/data/paralogs.txt"
script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"
splice_table="$repo_root/analysis/test/data/splice_table_Human.test.txt"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "tissue: $tissue"
echo "started: $(date)"

if [ ! -f "$splice_table" ]; then
    echo "ERROR: splice_table not found: $splice_table"
    exit 1
fi

tissue_matrix="$output_dir/${tissue}_matrix.tsv"
if [ ! -f "$tissue_matrix" ]; then
    echo "converting splice_table to $tissue matrix..."
    python "$script_dir/pangolin_to_matrix.py" "$splice_table" "$tissue_matrix" \
        --tissue "$tissue" --chroms chr1 chr3 chr5 chr7 chr9 --paralogs "$paralogs"
fi

tissue_h5="$output_dir/${tissue}_ag.h5"
if [ ! -f "$tissue_h5" ]; then
    echo "building $tissue per-gene h5..."
    cd "$pipeline_dir"
    python build_h5_alphagenome.py \
        --donor PANGOLIN --split test \
        --input "$tissue_matrix" \
        --chroms chr1,chr3,chr5,chr7,chr9 \
        --fasta "$genome_fasta" \
        --output "$tissue_h5" \
        --mode basic --paralog 0 \
        --reference --remove-missing --skip-empty-genes
fi

echo "done: $(date)"
