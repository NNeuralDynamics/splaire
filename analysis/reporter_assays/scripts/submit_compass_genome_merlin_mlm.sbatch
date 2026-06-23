#!/usr/bin/env bash
#SBATCH --job-name=cmp_gen_mlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/compass_genome_merlin_mlm_%j.out
#SBATCH --error=scripts/logs/compass_genome_merlin_mlm_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

source ~/.bashrc
conda activate "${MERLIN_ENV_PREFIX:-/scratch/runyan.m/conda_envs/merlin_env_py311}"
mkdir -p scripts/logs

set -e

repo_root="${SPHAEC_REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
mlm_ckpt="${MERLIN_MLM_CKPT:-$repo_root/analysis/other_models/MERLIN/AdarEditingPrediction/Models/model_MLM_final_data_sp.pth}"

# --- compass_genome_refs.h5 ---
if [ -f compass/data/compass_genome_refs.h5 ]; then
    python scripts/score_merlin_mlm.py --input compass/data/compass_genome_refs.h5 --output compass/data/compass_genome_refs_merlin_mlm.h5 --checkpoint-mlm "$mlm_ckpt"
else
    echo 'skipping compass/data/compass_genome_refs.h5 (file not present yet)'
fi

# --- compass_genome_snvs.h5 ---
if [ -f compass/data/compass_genome_snvs.h5 ]; then
    python scripts/score_merlin_mlm.py --input compass/data/compass_genome_snvs.h5 --output compass/data/compass_genome_snvs_merlin_mlm.h5 --checkpoint-mlm "$mlm_ckpt"
else
    echo 'skipping compass/data/compass_genome_snvs.h5 (file not present yet)'
fi

# --- compass_genome_doubles.h5 ---
if [ -f compass/data/compass_genome_doubles.h5 ]; then
    python scripts/score_merlin_mlm.py --input compass/data/compass_genome_doubles.h5 --output compass/data/compass_genome_doubles_merlin_mlm.h5 --checkpoint-mlm "$mlm_ckpt"
else
    echo 'skipping compass/data/compass_genome_doubles.h5 (file not present yet)'
fi
