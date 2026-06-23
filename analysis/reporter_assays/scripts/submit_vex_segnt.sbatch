#!/usr/bin/env bash
#SBATCH --job-name=vex_segnt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/vex_segnt_%j.out
#SBATCH --error=scripts/logs/vex_segnt_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

python scripts/score_segmentnt.py \
    --input vex_seq/data/vex_seq.h5 \
    --output vex_seq/data/vex_seq_segnt.h5 \
    --batch-size 4
