#!/bin/bash
#SBATCH --job-name=score_gencode_ag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# score gencode with AlphaGenome.
# usage: sbatch score_gencode_ag.sbatch output_dir [ag_seq_len]
#   ag_seq_len 1048576 (default): read per-gene 1Mb h5 from sibling _ag dir.
#   ag_seq_len 16384:              read shared 15k h5 from output_dir, N-padded → 16384.
# override AG_FLAGS to keep raw per-fold per-track cols (default: averaged)

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch score_gencode_ag.sbatch output_dir [ag_seq_len]}"
ag_seq_len="${2:-1048576}"
if [ "$ag_seq_len" = "1048576" ]; then
    ag_dir="${output_dir%/}_ag"
    h5="$ag_dir/gencode_ag.h5"
else
    h5="$output_dir/gencode.h5"   # shared 15k h5 used by all classical models
fi
script_dir="$repo_root/analysis/test/src"
ag_flags="${AG_FLAGS:-}"

source ~/.bashrc
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

mkdir -p "$output_dir/predictions" logs
echo "h5: $h5"
echo "AG flags: $ag_flags"
echo "started: $(date)"

if [ ! -f "$h5" ]; then
    if [ "$ag_seq_len" = "1048576" ]; then
        echo "ERROR: $h5 not found (run build_gencode_ag.sbatch first)"
    else
        echo "ERROR: $h5 not found (run build_gencode.sbatch first)"
    fi
    exit 1
fi

pred_dir="$output_dir/predictions"
if [ "$ag_seq_len" = "1048576" ]; then
    out_parquet="$pred_dir/gencode_ag.parquet"           # keep legacy name for native mode
else
    out_parquet="$pred_dir/gencode_ag_${ag_seq_len}.parquet"
fi
python "$script_dir/score_alphagenome.py" "$h5" "$out_parquet" --ag-seq-len "$ag_seq_len" $ag_flags

if [[ " $ag_flags " == *" --no-merge "* ]] && [[ " $ag_flags " != *" --folds "* ]]; then
    cd "$_submit_dir"
    sbatch merge_ag.sbatch "$out_parquet"
fi

echo "done: $(date)"
