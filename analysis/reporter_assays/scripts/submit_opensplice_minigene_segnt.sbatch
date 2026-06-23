#!/usr/bin/env bash
#SBATCH --job-name=os_min_segnt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/opensplice_minigene_segnt_%j.out
#SBATCH --error=scripts/logs/opensplice_minigene_segnt_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

# --- opensplice_minigene_snvs.h5 ---
if [ -f opensplice/data/opensplice_minigene_snvs.h5 ]; then
    python scripts/score_segmentnt.py --input opensplice/data/opensplice_minigene_snvs.h5 --output opensplice/data/opensplice_minigene_snvs_segnt.h5 --batch-size 4
else
    echo 'skipping opensplice/data/opensplice_minigene_snvs.h5 (file not present yet)'
fi

# --- opensplice_minigene_dels.h5 ---
if [ -f opensplice/data/opensplice_minigene_dels.h5 ]; then
    python scripts/score_segmentnt.py --input opensplice/data/opensplice_minigene_dels.h5 --output opensplice/data/opensplice_minigene_dels_segnt.h5 --batch-size 4
else
    echo 'skipping opensplice/data/opensplice_minigene_dels.h5 (file not present yet)'
fi
