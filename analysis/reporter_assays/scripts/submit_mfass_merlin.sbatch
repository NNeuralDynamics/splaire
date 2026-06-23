#!/usr/bin/env bash
#SBATCH --job-name=mfass_merlin
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/mfass_merlin_%j.out
#SBATCH --error=scripts/logs/mfass_merlin_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

repo_root="${SPHAEC_REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
cls_ckpt="$repo_root/analysis/other_models/MERLIN/Spliceformer/MERLIN/best_model_10k_variant_finetune_headshutoff_clsweighted_biggerhead_cls.pth"
reg_ckpt="$repo_root/analysis/other_models/MERLIN/Spliceformer/MERLIN/best_model_10k_variant_finetune_headshutoff_clsweighted_biggerhead_reg.pth"

mkdir -p scripts/logs

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"

python scripts/score_merlin.py \
    --input mfass/data/mfass.h5 \
    --output mfass/data/mfass_merlin.h5 \
    --checkpoint-cls "$cls_ckpt" \
    --checkpoint-reg "$reg_ckpt"
