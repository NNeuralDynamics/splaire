#!/usr/bin/env bash
#SBATCH --job-name=vex_mlm
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/vex_merlin_mlm_%j.out
#SBATCH --error=scripts/logs/vex_merlin_mlm_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

repo_root="${SPHAEC_REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
mlm_ckpt="${MERLIN_MLM_CKPT:-$repo_root/analysis/other_models/MERLIN/AdarEditingPrediction/Models/model_MLM_final_data_sp.pth}"

mkdir -p scripts/logs

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"

python scripts/score_merlin_mlm.py \
    --input vex_seq/data/vex_seq.h5 \
    --output vex_seq/data/vex_seq_merlin_mlm.h5 \
    --checkpoint-mlm "$mlm_ckpt"
