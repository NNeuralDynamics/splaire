#!/usr/bin/env bash
#SBATCH --job-name=cmp_min_sa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/compass_minigene_sa_%j.out
#SBATCH --error=scripts/logs/compass_minigene_sa_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${SA_ENV_PREFIX:-/scratch/runyan.m/conda_envs/sa_env}"
mkdir -p scripts/logs

# --- compass_minigene_refs.h5 ---
if [ -f compass/data/compass_minigene_refs.h5 ]; then
    python scripts/score_sa.py --input compass/data/compass_minigene_refs.h5 --output compass/data/compass_minigene_refs_sa.h5 --batch-size 256
else
    echo 'skipping compass/data/compass_minigene_refs.h5 (file not present yet)'
fi

# --- compass_minigene_snvs.h5 ---
if [ -f compass/data/compass_minigene_snvs.h5 ]; then
    python scripts/score_sa.py --input compass/data/compass_minigene_snvs.h5 --output compass/data/compass_minigene_snvs_sa.h5 --batch-size 256
else
    echo 'skipping compass/data/compass_minigene_snvs.h5 (file not present yet)'
fi

# --- compass_minigene_doubles.h5 ---
if [ -f compass/data/compass_minigene_doubles.h5 ]; then
    python scripts/score_sa.py --input compass/data/compass_minigene_doubles.h5 --output compass/data/compass_minigene_doubles_sa.h5 --batch-size 256
else
    echo 'skipping compass/data/compass_minigene_doubles.h5 (file not present yet)'
fi
