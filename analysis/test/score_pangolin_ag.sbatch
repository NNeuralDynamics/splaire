#!/bin/bash
#SBATCH --job-name=score_pangolin_ag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# score pangolin tissue with AlphaGenome (per-gene 1Mb h5).
# usage: sbatch score_pangolin_ag.sbatch output_dir tissue
#   override AG_FLAGS to keep raw per-fold per-track cols (default: averaged)

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch score_pangolin_ag.sbatch output_dir tissue}"
tissue="${2:?usage: sbatch score_pangolin_ag.sbatch output_dir tissue}"
ag_dir="${output_dir%/}_ag"
h5="$ag_dir/${tissue}_ag.h5"
script_dir="$repo_root/analysis/test/src"
ag_flags="${AG_FLAGS:-}"

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

mkdir -p "$output_dir/predictions" logs
echo "h5: $h5"
echo "tissue: $tissue"
echo "AG flags: $ag_flags"
echo "started: $(date)"

if [ ! -f "$h5" ]; then
    echo "ERROR: $h5 not found (run build_pangolin_ag.sbatch first)"
    exit 1
fi

out_parquet="$output_dir/predictions/${tissue}_ag.parquet"
python "$script_dir/score_alphagenome.py" "$h5" "$out_parquet" $ag_flags

if [[ " $ag_flags " == *" --no-merge "* ]] && [[ " $ag_flags " != *" --folds "* ]]; then
    cd "$_submit_dir"
    sbatch merge_ag.sbatch "$out_parquet"
fi

echo "done: $(date)"
