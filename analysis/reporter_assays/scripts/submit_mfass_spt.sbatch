#!/usr/bin/env bash
#SBATCH --job-name=mfass_spt
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/mfass_spt_%j.out
#SBATCH --error=scripts/logs/mfass_spt_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
conda activate ${SPT_ENV:-spt-test}
mkdir -p scripts/logs

python scripts/score_spt.py --input mfass/data/mfass.h5 --output mfass/data/mfass_spt.h5
