#!/usr/bin/env bash
#SBATCH --job-name=cmp_min_merlin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/compass_minigene_merlin_%j.out
#SBATCH --error=scripts/logs/compass_minigene_merlin_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"
mkdir -p scripts/logs

set -e

repo_root="${SPHAEC_REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
cls_ckpt="$repo_root/analysis/other_models/MERLIN/Spliceformer/MERLIN/best_model_10k_variant_finetune_headshutoff_clsweighted_biggerhead_cls.pth"
reg_ckpt="$repo_root/analysis/other_models/MERLIN/Spliceformer/MERLIN/best_model_10k_variant_finetune_headshutoff_clsweighted_biggerhead_reg.pth"

# --- compass_minigene_refs.h5 ---
if [ -f compass/data/compass_minigene_refs.h5 ]; then
    python scripts/score_merlin.py --input compass/data/compass_minigene_refs.h5 --output compass/data/compass_minigene_refs_merlin.h5 --checkpoint-cls "$cls_ckpt" --checkpoint-reg "$reg_ckpt"
else
    echo 'skipping compass/data/compass_minigene_refs.h5 (file not present yet)'
fi

# --- compass_minigene_snvs.h5 ---
if [ -f compass/data/compass_minigene_snvs.h5 ]; then
    python scripts/score_merlin.py --input compass/data/compass_minigene_snvs.h5 --output compass/data/compass_minigene_snvs_merlin.h5 --checkpoint-cls "$cls_ckpt" --checkpoint-reg "$reg_ckpt"
else
    echo 'skipping compass/data/compass_minigene_snvs.h5 (file not present yet)'
fi

# --- compass_minigene_doubles.h5 ---
if [ -f compass/data/compass_minigene_doubles.h5 ]; then
    python scripts/score_merlin.py --input compass/data/compass_minigene_doubles.h5 --output compass/data/compass_minigene_doubles_merlin.h5 --checkpoint-cls "$cls_ckpt" --checkpoint-reg "$reg_ckpt"
else
    echo 'skipping compass/data/compass_minigene_doubles.h5 (file not present yet)'
fi
