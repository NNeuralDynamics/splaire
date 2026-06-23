#!/usr/bin/env bash
#SBATCH --job-name=os_min_merlin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/opensplice_minigene_merlin_%j.out
#SBATCH --error=scripts/logs/opensplice_minigene_merlin_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"
mkdir -p scripts/logs

set -e

repo_root="${SPHAEC_REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
cls_ckpt="$repo_root/analysis/other_models/MERLIN/Spliceformer/MERLIN/best_model_10k_variant_finetune_headshutoff_clsweighted_biggerhead_cls.pth"
reg_ckpt="$repo_root/analysis/other_models/MERLIN/Spliceformer/MERLIN/best_model_10k_variant_finetune_headshutoff_clsweighted_biggerhead_reg.pth"

# --- opensplice_minigene_snvs.h5 ---
if [ -f opensplice/data/opensplice_minigene_snvs.h5 ]; then
    python scripts/score_merlin.py --input opensplice/data/opensplice_minigene_snvs.h5 --output opensplice/data/opensplice_minigene_snvs_merlin.h5 --checkpoint-cls "$cls_ckpt" --checkpoint-reg "$reg_ckpt"
else
    echo 'skipping opensplice/data/opensplice_minigene_snvs.h5 (file not present yet)'
fi

# --- opensplice_minigene_dels.h5 ---
if [ -f opensplice/data/opensplice_minigene_dels.h5 ]; then
    python scripts/score_merlin.py --input opensplice/data/opensplice_minigene_dels.h5 --output opensplice/data/opensplice_minigene_dels_merlin.h5 --checkpoint-cls "$cls_ckpt" --checkpoint-reg "$reg_ckpt"
else
    echo 'skipping opensplice/data/opensplice_minigene_dels.h5 (file not present yet)'
fi
