#!/bin/bash
#SBATCH --job-name=score_gencode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# score gencode benchmark with all models (GPU)
# usage: sbatch score_gencode.sbatch output_dir
# run build_gencode.sbatch first to create the h5

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch score_gencode.sbatch output_dir}"
h5="$output_dir/gencode.h5"
script_dir="$repo_root/analysis/test/src"

source ~/.bashrc

mkdir -p "$output_dir/predictions" logs
echo "output_dir: $output_dir"
echo "started: $(date)"

if [ ! -f "$h5" ]; then
    echo "ERROR: h5 not found: $h5"
    echo "run build_gencode.sbatch first"
    exit 1
fi

pred_dir="$output_dir/predictions"

# optional MODELS env var picks a subset; defaults to all four
MODELS_TO_RUN="${MODELS:-splaire sa pang spt}"
should_run() { [[ " $MODELS_TO_RUN " == *" $1 "* ]]; }

if should_run splaire; then
    echo "scoring with SplAIRE..."
    conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
    python "$script_dir/score_splaire.py" "$h5" "$pred_dir" --ref --var --prefix gencode_splaire --batch-size 6456
fi

if should_run sa; then
    echo "scoring with SpliceAI..."
    conda activate "${SA_ENV_PREFIX:-/scratch/runyan.m/conda_envs/sa_env}"
    python "$script_dir/score_spliceai.py" "$h5" "$pred_dir/gencode_sa.parquet" --batch-size 32
fi

if should_run pang; then
    echo "scoring with Pangolin..."
    conda activate "${PANG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/pang_env}"
    python "$script_dir/score_pangolin.py" "$h5" "$pred_dir/gencode_pang.parquet" --batch-size 6456
fi

if should_run spt; then
    echo "scoring with SpliceTransformer..."
    conda activate "${SPT_ENV_PREFIX:-/scratch/runyan.m/conda_envs/spt_env}"
    python "$script_dir/score_splicetransformer.py" "$h5" "$pred_dir/gencode_spt.parquet" --batch-size 64
fi

# metrics
echo "computing metrics..."
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
python "$script_dir/compute_metrics.py" "$pred_dir" "$output_dir/gencode_metrics.json"

echo "done: $(date)"
