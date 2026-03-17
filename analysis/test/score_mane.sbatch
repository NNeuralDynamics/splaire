#!/bin/bash
#SBATCH --job-name=score_mane
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# score mane select benchmark with all models (GPU)
# usage: sbatch score_mane.sbatch output_dir
# run build_mane.sbatch first to create the h5

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch score_mane.sbatch output_dir}"
h5="$output_dir/mane_select.h5"
script_dir="$repo_root/analysis/test/src"

source ~/.bashrc

mkdir -p "$output_dir/predictions" logs
echo "output_dir: $output_dir"
echo "started: $(date)"

if [ ! -f "$h5" ]; then
    echo "ERROR: h5 not found: $h5"
    echo "run build_mane.sbatch first"
    exit 1
fi

pred_dir="$output_dir/predictions"

echo "scoring with SplAIRE..."
conda activate splaire_env
python "$script_dir/score_splaire.py" "$h5" "$pred_dir" --ref --var --prefix mane_select_splaire --batch-size 6456

echo "scoring with SpliceAI..."
conda activate sa_env
python "$script_dir/score_spliceai.py" "$h5" "$pred_dir/mane_select_sa.parquet" --batch-size 32

echo "scoring with Pangolin..."
conda activate pang_env
python "$script_dir/score_pangolin.py" "$h5" "$pred_dir/mane_select_pang.parquet" --batch-size 6456

echo "scoring with SpliceTransformer..."
conda activate spt-test
python "$script_dir/score_splicetransformer.py" "$h5" "$pred_dir/mane_select_spt.parquet" --batch-size 64

# metrics
echo "computing metrics..."
conda activate splaire_env
python "$script_dir/compute_metrics.py" "$pred_dir" "$output_dir/mane_select_metrics.json"

echo "done: $(date)"
