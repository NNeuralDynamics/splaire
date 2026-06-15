#!/bin/bash
#SBATCH --job-name=merge_ag
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only horizontal stream-merge of per-fold AG temp parquets.
# usage: sbatch merge_ag.sbatch output_parquet [folds]
#   output_parquet is the same path passed to score_alphagenome.py;
#   temp inputs are {basename}_{fold}_temp.parquet sitting next to it.

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_parquet="${1:?usage: sbatch merge_ag.sbatch output_parquet [folds]}"
folds="${2:-fold_0,fold_1,fold_2,fold_3}"
script_dir="$repo_root/analysis/test/src"

source ~/.bashrc
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

mkdir -p logs
echo "merging into $output_parquet"
echo "folds: $folds"
echo "started: $(date)"

python "$script_dir/score_alphagenome.py" /dev/null "$output_parquet" \
    --folds "$folds" --merge-only

echo "done: $(date)"
