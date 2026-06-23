#!/usr/bin/env bash
#SBATCH --job-name=os_gen_spt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_genome_spt_%j.out
#SBATCH --error=scripts/logs/opensplice_genome_spt_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${SPT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/spt_env}"
mkdir -p scripts/logs

# --- opensplice_genome_snvs.h5 ---
if [ -f opensplice/data/opensplice_genome_snvs.h5 ]; then
    python scripts/score_spt.py --input opensplice/data/opensplice_genome_snvs.h5 --output opensplice/data/opensplice_genome_snvs_spt.h5 --batch-size 128
else
    echo 'skipping opensplice/data/opensplice_genome_snvs.h5 (file not present yet)'
fi

# --- opensplice_genome_dels.h5 ---
if [ -f opensplice/data/opensplice_genome_dels.h5 ]; then
    python scripts/score_spt.py --input opensplice/data/opensplice_genome_dels.h5 --output opensplice/data/opensplice_genome_dels_spt.h5 --batch-size 128
else
    echo 'skipping opensplice/data/opensplice_genome_dels.h5 (file not present yet)'
fi
