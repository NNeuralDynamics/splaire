#!/usr/bin/env bash
#SBATCH --job-name=vex_nt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/vex_nt_%j.out
#SBATCH --error=scripts/logs/vex_nt_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

python scripts/score_nt.py \
    --input vex_seq/data/vex_seq.h5 \
    --output vex_seq/data/vex_seq_nt.h5 \
    --batch-size 8
