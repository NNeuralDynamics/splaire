#!/bin/bash
#SBATCH --job-name=compact_ag
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# compact an alphagenome parquet: null out ssu_*_track_* values at non-splice positions.
# CPU-only, streams row-group by row-group. preserves classification + GT cols at all
# positions; only ssu track cols get nulled at non-splice rows.
#
# usage: sbatch compact_ag.sbatch input.parquet output.parquet

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"
script_dir="$repo_root/analysis/test"

input="${1:?usage: sbatch compact_ag.sbatch input.parquet output.parquet}"
output="${2:?usage: sbatch compact_ag.sbatch input.parquet output.parquet}"

mkdir -p logs "$(dirname "$output")"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"

echo "input:  $input"
echo "output: $output"
echo "started: $(date)"

python "$script_dir/src/compact_ag_parquet.py" "$input" "$output"

echo "done: $(date)"
