#!/usr/bin/env bash
#SBATCH --job-name=pang_missing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/pang_missing_%j.out
#SBATCH --error=scripts/logs/pang_missing_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

# pang v1: covers all 4 missing compass+opensplice files in one job
source ~/.bashrc
conda activate "${PANG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/pang_env}"
mkdir -p scripts/logs

declare -a targets=(
  "compass/data/compass_genome_doubles.h5            compass/data/compass_genome_doubles_pang.h5"
  "opensplice/data/opensplice_genome_snvs.h5         opensplice/data/opensplice_genome_snvs_pang.h5"
  "opensplice/data/opensplice_genome_dels.h5         opensplice/data/opensplice_genome_dels_pang.h5"
  "opensplice/data/opensplice_minigene_dels.h5       opensplice/data/opensplice_minigene_dels_pang.h5"
)
for entry in "${targets[@]}"; do
    read -r in out <<< "$entry"
    if [ -f "$in" ] && [ ! -f "$out" ]; then
        python scripts/score_pang.py --input "$in" --output "$out"
    else
        echo "skipping $in (input missing or output exists)"
    fi
done
