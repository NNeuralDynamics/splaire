#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err



_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

pred_dir="${1:?usage: sbatch metrics.sbatch pred_dir output_json [extra args...]}"
output_json="${2:?usage: sbatch metrics.sbatch pred_dir output_json [extra args...]}"
shift 2
script_dir="$repo_root/analysis/test"

source ~/.bashrc
conda activate splaire_env
set -eo pipefail
mkdir -p "$(dirname "$output_json")" logs

echo "pred_dir: $pred_dir"
echo "output: $output_json"
echo "started: $(date)"

python "$script_dir/src/compute_metrics.py" "$pred_dir" "$output_json" "$@"

echo "done: $(date)"
