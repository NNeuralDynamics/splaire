#!/bin/bash
#SBATCH --job-name=warm_ag_cache
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# populate JAX persistent compile cache for AlphaGenome.
# loads each fold, runs warmup at AG_SEQ_LEN, exits.
# after this completes, all future score_alphagenome.py runs skip the ~15-min compile.
#
# WARNING: this job has ~15 min idle GPU per fold (4 × 15 min) during JIT compile.
# the auto-cancel watchdog may warn during these windows; running this once is the cost
# of populating the cache. notify rchelp@northeastern.edu before submitting if needed.
#
# usage: sbatch warm_ag_cache.sbatch [folds]
#   folds default: fold_0,fold_1,fold_2,fold_3

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

folds="${1:-fold_0,fold_1,fold_2,fold_3}"
script_dir="$repo_root/analysis/test/src"

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

mkdir -p "$JAX_COMPILATION_CACHE_DIR" logs
echo "JAX cache dir: $JAX_COMPILATION_CACHE_DIR"
echo "folds: $folds"
echo "started: $(date)"

python "$script_dir/warm_ag_cache.py" --folds "$folds"

echo "done: $(date)"
echo "cache size: $(du -sh "$JAX_COMPILATION_CACHE_DIR" | cut -f1)"
