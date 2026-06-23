#!/usr/bin/env bash
#SBATCH --job-name=os_gen_sa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_genome_sa_%j.out
#SBATCH --error=scripts/logs/opensplice_genome_sa_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${SA_ENV_PREFIX:-/scratch/runyan.m/conda_envs/sa_env}"
mkdir -p scripts/logs

for stem in opensplice_genome_snvs opensplice_genome_dels; do
    in="opensplice/data/${stem}.h5"
    out="opensplice/data/${stem}_sa.h5"
    if [ -f "$in" ] && [ ! -f "$out" ]; then
        python scripts/score_sa.py --input "$in" --output "$out" --batch-size 256
    else
        echo "skipping $in (input missing or output exists)"
    fi
done
