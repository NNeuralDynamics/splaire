#!/usr/bin/env bash
#SBATCH --job-name=mfass_nt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/mfass_nt_%j.out
#SBATCH --error=scripts/logs/mfass_nt_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

# 8 NT variants × ref+alt forward passes per (exon_start, exon_end).
# 2.5B variants dominate runtime — keep batch small.

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

python scripts/score_nt.py \
    --input mfass/data/mfass.h5 \
    --output mfass/data/mfass_nt.h5 \
    --batch-size 8
