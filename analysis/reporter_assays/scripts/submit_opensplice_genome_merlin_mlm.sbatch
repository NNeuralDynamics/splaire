#!/usr/bin/env bash
#SBATCH --job-name=os_gen_mlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_genome_merlin_mlm_%j.out
#SBATCH --error=scripts/logs/opensplice_genome_merlin_mlm_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"
mkdir -p scripts/logs

set -e

repo_root="${SPHAEC_REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
mlm_ckpt="${MERLIN_MLM_CKPT:-$repo_root/analysis/other_models/MERLIN/AdarEditingPrediction/Models/model_MLM_final_data_sp.pth}"

# --- opensplice_genome_snvs.h5 ---
if [ -f opensplice/data/opensplice_genome_snvs.h5 ]; then
    python scripts/score_merlin_mlm.py --input opensplice/data/opensplice_genome_snvs.h5 --output opensplice/data/opensplice_genome_snvs_merlin_mlm.h5 --checkpoint-mlm "$mlm_ckpt"
else
    echo 'skipping opensplice/data/opensplice_genome_snvs.h5 (file not present yet)'
fi

# --- opensplice_genome_dels.h5 ---
# deletions: scorer writes NaN per variant; sbatch still runs for the meta dump
if [ -f opensplice/data/opensplice_genome_dels.h5 ]; then
    python scripts/score_merlin_mlm.py --input opensplice/data/opensplice_genome_dels.h5 --output opensplice/data/opensplice_genome_dels_merlin_mlm.h5 --checkpoint-mlm "$mlm_ckpt"
else
    echo 'skipping opensplice/data/opensplice_genome_dels.h5 (file not present yet)'
fi
