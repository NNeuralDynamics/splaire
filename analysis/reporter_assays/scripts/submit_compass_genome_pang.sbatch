#!/usr/bin/env bash
#SBATCH --job-name=cmp_gen_pang
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/compass_genome_pang_%j.out
#SBATCH --error=scripts/logs/compass_genome_pang_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${PANG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/pang_env}"
mkdir -p scripts/logs

for stem in compass_genome_refs compass_genome_snvs compass_genome_doubles; do
    in="compass/data/${stem}.h5"
    out="compass/data/${stem}_pang.h5"
    if [ -f "$in" ] && [ ! -f "$out" ]; then
        python scripts/score_pang.py --input "$in" --output "$out" --batch-size 256
    else
        echo "skipping $in (input missing or output exists)"
    fi
done
