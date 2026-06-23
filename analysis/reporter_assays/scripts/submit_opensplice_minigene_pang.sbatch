#!/usr/bin/env bash
#SBATCH --job-name=os_min_pang
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_minigene_pang_%j.out
#SBATCH --error=scripts/logs/opensplice_minigene_pang_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${PANG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/pang_env}"
mkdir -p scripts/logs

for stem in opensplice_minigene_snvs opensplice_minigene_dels; do
    in="opensplice/data/${stem}.h5"
    out="opensplice/data/${stem}_pang.h5"
    if [ -f "$in" ] && [ ! -f "$out" ]; then
        python scripts/score_pang.py --input "$in" --output "$out" --batch-size 256
    else
        echo "skipping $in (input missing or output exists)"
    fi
done
