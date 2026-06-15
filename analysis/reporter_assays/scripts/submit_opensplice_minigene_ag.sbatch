#!/usr/bin/env bash
#SBATCH --job-name=os_min_ag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_minigene_ag_%j.out
#SBATCH --error=scripts/logs/opensplice_minigene_ag_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"
mkdir -p scripts/logs

# --- opensplice_minigene_snvs.h5 ---
if [ -f opensplice/data/opensplice_minigene_snvs.h5 ]; then
    python scripts/score_ag.py --input opensplice/data/opensplice_minigene_snvs.h5 --output opensplice/data/opensplice_minigene_snvs_ag.h5
else
    echo 'skipping opensplice/data/opensplice_minigene_snvs.h5 (file not present yet)'
fi

# --- opensplice_minigene_dels.h5 ---
if [ -f opensplice/data/opensplice_minigene_dels.h5 ]; then
    python scripts/score_ag.py --input opensplice/data/opensplice_minigene_dels.h5 --output opensplice/data/opensplice_minigene_dels_ag.h5
else
    echo 'skipping opensplice/data/opensplice_minigene_dels.h5 (file not present yet)'
fi
