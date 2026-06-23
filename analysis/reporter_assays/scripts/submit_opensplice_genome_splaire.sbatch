#!/usr/bin/env bash
#SBATCH --job-name=os_gen_splaire
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=scripts/logs/opensplice_genome_splaire_%j.out
#SBATCH --error=scripts/logs/opensplice_genome_splaire_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
mkdir -p scripts/logs

for stem in opensplice_genome_snvs opensplice_genome_dels; do
    in="opensplice/data/${stem}.h5"
    if [ ! -f "$in" ]; then
        echo "skipping $in (input missing)"
        continue
    fi
    for vt in ref var; do
        out="opensplice/data/${stem}_splaire_${vt}.h5"
        if [ ! -f "$out" ]; then
            python scripts/score_splaire.py --input "$in" --output "$out" --variant-type "$vt"
        else
            echo "skipping $out (already exists)"
        fi
    done
done
