#!/usr/bin/env bash
#SBATCH --job-name=cmp_gen_nt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/compass_genome_nt_%j.out
#SBATCH --error=scripts/logs/compass_genome_nt_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

# --- compass_genome_refs.h5 ---
if [ -f compass/data/compass_genome_refs.h5 ]; then
    python scripts/score_nt.py --input compass/data/compass_genome_refs.h5 --output compass/data/compass_genome_refs_nt.h5 --batch-size 8
else
    echo 'skipping compass/data/compass_genome_refs.h5 (file not present yet)'
fi

# --- compass_genome_snvs.h5 ---
if [ -f compass/data/compass_genome_snvs.h5 ]; then
    python scripts/score_nt.py --input compass/data/compass_genome_snvs.h5 --output compass/data/compass_genome_snvs_nt.h5 --batch-size 8
else
    echo 'skipping compass/data/compass_genome_snvs.h5 (file not present yet)'
fi

# --- compass_genome_doubles.h5 ---
if [ -f compass/data/compass_genome_doubles.h5 ]; then
    python scripts/score_nt.py --input compass/data/compass_genome_doubles.h5 --output compass/data/compass_genome_doubles_nt.h5 --batch-size 8
else
    echo 'skipping compass/data/compass_genome_doubles.h5 (file not present yet)'
fi
