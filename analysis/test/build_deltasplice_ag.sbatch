#!/bin/bash
#SBATCH --job-name=build_dsdata_ag
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build deltasplice's own benchmark as per-gene h5 for AlphaGenome (1Mb, hg19)
# usage: sbatch build_deltasplice_ag.sbatch output_dir

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_deltasplice_ag.sbatch output_dir}"
genome_fasta="$repo_root/pipeline/reference/GRCh37/hg19.fa"
ds_data="$repo_root/analysis/other_models/DeltaSplice/data/gene_dataset.tsu.txt"
script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "started: $(date)"

matrix="$output_dir/deltasplice_matrix.tsv"
if [ ! -f "$matrix" ]; then
    echo "converting deltasplice data to matrix..."
    python "$script_dir/deltasplice_to_matrix.py" "$ds_data" "$matrix" \
        --chroms chr1 chr3 chr5 chr7 chr9
fi

h5="$output_dir/deltasplice_ag.h5"
if [ ! -f "$h5" ]; then
    echo "building per-gene h5..."
    cd "$pipeline_dir"
    python build_h5_alphagenome.py \
        --donor DELTASPLICE --split test \
        --input "$matrix" \
        --chroms chr1,chr3,chr5,chr7,chr9 \
        --fasta "$genome_fasta" \
        --output "$h5" \
        --mode basic --paralog 0 \
        --reference --skip-empty-genes
fi

echo "done: $(date)"
