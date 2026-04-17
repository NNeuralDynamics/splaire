#!/usr/bin/env bash
#SBATCH --job-name=vex_sa
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/vex_sa_%j.out
#SBATCH --error=scripts/logs/vex_sa_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
conda activate ${SA_ENV:-sa_env}
mkdir -p scripts/logs

python scripts/score_sa.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sa.h5
