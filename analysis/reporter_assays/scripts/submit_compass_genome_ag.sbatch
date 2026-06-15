#!/usr/bin/env bash
#SBATCH --job-name=cmp_gen_ag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/compass_genome_ag_%j.out
#SBATCH --error=scripts/logs/compass_genome_ag_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"
mkdir -p scripts/logs

# --- compass_genome_refs.h5 ---
if [ -f compass/data/compass_genome_refs.h5 ]; then
    python scripts/score_ag.py --input compass/data/compass_genome_refs.h5 --output compass/data/compass_genome_refs_ag.h5
else
    echo 'skipping compass/data/compass_genome_refs.h5 (file not present yet)'
fi

# --- compass_genome_snvs.h5 ---
if [ -f compass/data/compass_genome_snvs.h5 ]; then
    python scripts/score_ag.py --input compass/data/compass_genome_snvs.h5 --output compass/data/compass_genome_snvs_ag.h5
else
    echo 'skipping compass/data/compass_genome_snvs.h5 (file not present yet)'
fi

# --- compass_genome_doubles.h5 ---
if [ -f compass/data/compass_genome_doubles.h5 ]; then
    python scripts/score_ag.py --input compass/data/compass_genome_doubles.h5 --output compass/data/compass_genome_doubles_ag.h5
else
    echo 'skipping compass/data/compass_genome_doubles.h5 (file not present yet)'
fi
