#!/usr/bin/env bash
#SBATCH --job-name=os_gen_nt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=scripts/logs/opensplice_genome_nt_%j.out
#SBATCH --error=scripts/logs/opensplice_genome_nt_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G

source ~/.bashrc
conda activate "${SEGNT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/segnt_env}"
mkdir -p scripts/logs

# --- opensplice_genome_snvs.h5 ---
if [ -f opensplice/data/opensplice_genome_snvs.h5 ]; then
    python scripts/score_nt.py --input opensplice/data/opensplice_genome_snvs.h5 --output opensplice/data/opensplice_genome_snvs_nt.h5 --batch-size 8
else
    echo 'skipping opensplice/data/opensplice_genome_snvs.h5 (file not present yet)'
fi

# --- opensplice_genome_dels.h5 ---
if [ -f opensplice/data/opensplice_genome_dels.h5 ]; then
    python scripts/score_nt.py --input opensplice/data/opensplice_genome_dels.h5 --output opensplice/data/opensplice_genome_dels_nt.h5 --batch-size 8
else
    echo 'skipping opensplice/data/opensplice_genome_dels.h5 (file not present yet)'
fi
