#!/usr/bin/env bash
#SBATCH --job-name=os_min_sa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_minigene_sa_%j.out
#SBATCH --error=scripts/logs/opensplice_minigene_sa_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${SA_ENV_PREFIX:-/scratch/runyan.m/conda_envs/sa_env}"
mkdir -p scripts/logs

# --- opensplice_minigene_snvs.h5 ---
if [ -f opensplice/data/opensplice_minigene_snvs.h5 ]; then
    python scripts/score_sa.py --input opensplice/data/opensplice_minigene_snvs.h5 --output opensplice/data/opensplice_minigene_snvs_sa.h5 --batch-size 256
else
    echo 'skipping opensplice/data/opensplice_minigene_snvs.h5 (file not present yet)'
fi

# --- opensplice_minigene_dels.h5 ---
if [ -f opensplice/data/opensplice_minigene_dels.h5 ]; then
    python scripts/score_sa.py --input opensplice/data/opensplice_minigene_dels.h5 --output opensplice/data/opensplice_minigene_dels_sa.h5 --batch-size 256
else
    echo 'skipping opensplice/data/opensplice_minigene_dels.h5 (file not present yet)'
fi
