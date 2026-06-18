#!/bin/bash
#SBATCH --job-name=score_ag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=score_logs/%x_%j.out
#SBATCH --error=score_logs/%x_%j.err

# per-donor AlphaGenome scoring on per-gene h5s (build_h5_alphagenome.py output).
# script outputs raw per-track per-fold predictions (no aggregation).
# override via AG_FLAGS env var, e.g. AG_FLAGS="--folds fold_0 --with-junctions"
#
# usage: sbatch score_ag.sbatch input_dir_ag output_dir
#   input_dir_ag: e.g. ${OUT}/${tissue}/ml_data_var_ag/individual
#   output_dir:   e.g. ${OUT}/${tissue}/ml_out_var  (parquets go to predictions/{donor}/)

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

input_dir="${1:?usage: sbatch score_ag.sbatch input_dir_ag output_dir}"
output_dir="${2:?usage: sbatch score_ag.sbatch input_dir_ag output_dir}"
script_dir="$repo_root/analysis/test"
ag_flags="${AG_FLAGS:-}"

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

mkdir -p "$output_dir/predictions" score_logs
completed="$output_dir/completed_ag.txt"
touch "$completed"

conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

echo "AG flags: $ag_flags"

for h5 in "$input_dir"/*.h5; do
    [ -f "$h5" ] || continue
    donor=$(basename "$h5" .h5)
    grep -qx "$donor" "$completed" && continue

    echo "$donor: $(date)"
    pred_dir="$output_dir/predictions/$donor"
    mkdir -p "$pred_dir"

    python "$script_dir/src/score_alphagenome.py" "$h5" "$pred_dir/${donor}_ag.parquet" $ag_flags

    echo "$donor" >> "$completed"
done

n_h5=$(ls "$input_dir"/*.h5 2>/dev/null | wc -l)
n_done=$(wc -l < "$completed")
echo "AG scoring complete: $n_done/$n_h5 donors"
