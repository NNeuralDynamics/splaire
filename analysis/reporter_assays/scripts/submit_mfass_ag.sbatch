#!/usr/bin/env bash
#SBATCH --job-name=mfass_ag
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --output=scripts/logs/mfass_ag_%j.out
#SBATCH --error=scripts/logs/mfass_ag_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"
mkdir -p scripts/logs

python scripts/score_ag.py \
    --input mfass/data/mfass.h5 \
    --output mfass/data/mfass_ag.h5
