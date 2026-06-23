#!/usr/bin/env bash
#SBATCH --job-name=os_gen_dels_sa_splairev
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/os_gen_dels_sa_splaire_%j.out
#SBATCH --error=scripts/logs/os_gen_dels_sa_splaire_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

# opensplice_genome_dels: missing sa + splaire_var. combined into one job to
# stay under the QOS submit-per-user cap.
source ~/.bashrc
mkdir -p scripts/logs

in_h5="opensplice/data/opensplice_genome_dels.h5"
if [ ! -f "$in_h5" ]; then
    echo "skipping: $in_h5 not found"
    exit 0
fi

# sa
out_sa="opensplice/data/opensplice_genome_dels_sa.h5"
if [ ! -f "$out_sa" ]; then
    conda activate "${SA_ENV_PREFIX:-/scratch/runyan.m/conda_envs/sa_env}"
    python scripts/score_sa.py --input "$in_h5" --output "$out_sa"
    conda deactivate
fi

# splaire --variant-type var
out_sv="opensplice/data/opensplice_genome_dels_splaire_var.h5"
if [ ! -f "$out_sv" ]; then
    conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
    python scripts/score_splaire.py --input "$in_h5" --output "$out_sv" --variant-type var
    conda deactivate
fi
