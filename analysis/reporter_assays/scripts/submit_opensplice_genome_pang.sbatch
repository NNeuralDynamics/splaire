#!/usr/bin/env bash
#SBATCH --job-name=os_gen_pang
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_genome_pang_%j.out
#SBATCH --error=scripts/logs/opensplice_genome_pang_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${PANG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/pang_env}"
mkdir -p scripts/logs

# --- opensplice_genome_snvs.h5 ---
if [ -f opensplice/data/opensplice_genome_snvs.h5 ]; then
    python scripts/score_pang.py --input opensplice/data/opensplice_genome_snvs.h5 --output opensplice/data/opensplice_genome_snvs_pang.h5 --batch-size 256
else
    echo 'skipping opensplice/data/opensplice_genome_snvs.h5 (file not present yet)'
fi

# --- opensplice_genome_dels.h5 ---
if [ -f opensplice/data/opensplice_genome_dels.h5 ]; then
    python scripts/score_pang.py --input opensplice/data/opensplice_genome_dels.h5 --output opensplice/data/opensplice_genome_dels_pang.h5 --batch-size 256
else
    echo 'skipping opensplice/data/opensplice_genome_dels.h5 (file not present yet)'
fi
