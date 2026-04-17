#!/usr/bin/env bash
#SBATCH --job-name=vex_sphaec
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/vex_sphaec_%j.out
#SBATCH --error=scripts/logs/vex_sphaec_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
conda activate ${SPLAIRE_ENV:-splaire_env}
mkdir -p scripts/logs

python scripts/score_sphaec.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sphaec_ref.h5 --variant-type ref

python scripts/score_sphaec.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sphaec_var.h5 --variant-type var
