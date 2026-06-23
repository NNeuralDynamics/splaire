#!/usr/bin/env bash
#SBATCH --job-name=cmp_min_segnt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/compass_minigene_segnt_%j.out
#SBATCH --error=scripts/logs/compass_minigene_segnt_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

# --- compass_minigene_refs.h5 ---
if [ -f compass/data/compass_minigene_refs.h5 ]; then
    python scripts/score_segmentnt.py --input compass/data/compass_minigene_refs.h5 --output compass/data/compass_minigene_refs_segnt.h5 --batch-size 4
else
    echo 'skipping compass/data/compass_minigene_refs.h5 (file not present yet)'
fi

# --- compass_minigene_snvs.h5 ---
if [ -f compass/data/compass_minigene_snvs.h5 ]; then
    python scripts/score_segmentnt.py --input compass/data/compass_minigene_snvs.h5 --output compass/data/compass_minigene_snvs_segnt.h5 --batch-size 4
else
    echo 'skipping compass/data/compass_minigene_snvs.h5 (file not present yet)'
fi

# --- compass_minigene_doubles.h5 ---
if [ -f compass/data/compass_minigene_doubles.h5 ]; then
    python scripts/score_segmentnt.py --input compass/data/compass_minigene_doubles.h5 --output compass/data/compass_minigene_doubles_segnt.h5 --batch-size 4
else
    echo 'skipping compass/data/compass_minigene_doubles.h5 (file not present yet)'
fi
