#!/bin/bash
#SBATCH --job-name=merlin_mlm_cs
#SBATCH --output=logs/merlin_mlm_cs_%j.out
#SBATCH --error=logs/merlin_mlm_cs_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00

set -e

data_dir="${1:?usage: sbatch score_haec_cs_merlin_mlm.sbatch <data_dir>}"
repo_root="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
src="$repo_root/analysis/sqtl_bench/src"
fasta="${REF_FASTA:?must export REF_FASTA=/path/to/GRCh38.primary_assembly.genome.fa}"

mlm_ckpt="${MERLIN_MLM_CKPT:-$repo_root/analysis/other_models/MERLIN/AdarEditingPrediction/Models/model_MLM_final_data_sp.pth}"

vcf="$data_dir/haec/cs_all.vcf.gz"
score_dir="$data_dir/haec/scores"
out_h5="$score_dir/cs_all.merlin_mlm.h5"

mkdir -p "$score_dir" logs

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"

cd "$src"
python score_merlin_mlm.py "$vcf" "$fasta" "$out_h5" \
    --checkpoint-mlm "$mlm_ckpt"
