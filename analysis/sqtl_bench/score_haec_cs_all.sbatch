#!/bin/bash
#SBATCH --job-name=splaire_haec_cs_all
#SBATCH --output=logs/splaire_haec_cs_all_%j.out
#SBATCH --error=logs/splaire_haec_cs_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00

# score every (variant, strand) from the haec credible sets file with splaire ref+var
# usage: cd $REPO/analysis/sqtl_bench && sbatch score_haec_cs_all.sbatch $OUT/sqtl_bench

set -e

data_dir="${1:?usage: sbatch score_haec_cs_all.sbatch <data_dir>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
src="$repo_root/analysis/sqtl_bench/src"
fasta="$repo_root/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa"
models_dir="$repo_root/models"

vcf="$data_dir/haec/cs_all.vcf.gz"
score_dir="$data_dir/haec/scores"
out_prefix="$score_dir/cs_all.splaire"

mkdir -p "$score_dir" logs

source ~/.bashrc
conda activate ${SPLAIRE_ENV:-splaire_env}

cd "$src"
python score_splaire.py "$vcf" "$fasta" "$out_prefix" --models-dir "$models_dir"
