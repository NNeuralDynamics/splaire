#!/usr/bin/env bash
#SBATCH --job-name=cmp_min_splaire
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/compass_minigene_splaire_%j.out
#SBATCH --error=scripts/logs/compass_minigene_splaire_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
mkdir -p scripts/logs

# --- compass_minigene_refs.h5 ---
if [ -f compass/data/compass_minigene_refs.h5 ]; then
    python scripts/score_splaire.py --input compass/data/compass_minigene_refs.h5 --output compass/data/compass_minigene_refs_splaire_ref.h5 --variant-type ref
    python scripts/score_splaire.py --input compass/data/compass_minigene_refs.h5 --output compass/data/compass_minigene_refs_splaire_var.h5 --variant-type var
else
    echo 'skipping compass/data/compass_minigene_refs.h5 (file not present yet)'
fi

# --- compass_minigene_snvs.h5 ---
if [ -f compass/data/compass_minigene_snvs.h5 ]; then
    python scripts/score_splaire.py --input compass/data/compass_minigene_snvs.h5 --output compass/data/compass_minigene_snvs_splaire_ref.h5 --variant-type ref
    python scripts/score_splaire.py --input compass/data/compass_minigene_snvs.h5 --output compass/data/compass_minigene_snvs_splaire_var.h5 --variant-type var
else
    echo 'skipping compass/data/compass_minigene_snvs.h5 (file not present yet)'
fi

# --- compass_minigene_doubles.h5 ---
if [ -f compass/data/compass_minigene_doubles.h5 ]; then
    python scripts/score_splaire.py --input compass/data/compass_minigene_doubles.h5 --output compass/data/compass_minigene_doubles_splaire_ref.h5 --variant-type ref
    python scripts/score_splaire.py --input compass/data/compass_minigene_doubles.h5 --output compass/data/compass_minigene_doubles_splaire_var.h5 --variant-type var
else
    echo 'skipping compass/data/compass_minigene_doubles.h5 (file not present yet)'
fi
