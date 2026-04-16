#!/bin/bash
#SBATCH --job-name=splaire_attr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=logs/splaire/%x_%j.out
#SBATCH --error=logs/splaire/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# usage: sbatch src/run_attribution_splaire.sbatch <model>
# example: sbatch src/run_attribution_splaire.sbatch splaire_ref_reg

model=$1

if [ -z "$model" ]; then
    echo "error: model argument required"
    exit 1
fi

source ~/.bashrc
conda activate splaire_env
cd /projects/talisman/mrunyan/paper/SpHAEC/analysis/explain

export SPLAIRE_MODELS_DIR=/projects/talisman/mrunyan/paper/SpHAEC/models

python src/run_attribution.py \
    --model $model \
    --input data/sequences.h5 \
    --output data/attr_${model}_v2.h5 \
    --batch-size 64 \
    --n-shuffles 20 \
    --seed 42 \
    --device cuda
