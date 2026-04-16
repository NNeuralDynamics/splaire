#!/bin/bash
#SBATCH --job-name=score_pangolin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# score one pangolin test tissue with all models (GPU)
# usage: sbatch score_pangolin.sbatch output_dir tissue
# tissue: heart, liver, brain, or testis
# run build_pangolin.sbatch first to create the h5

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch score_pangolin.sbatch output_dir tissue}"
tissue="${2:?usage: sbatch score_pangolin.sbatch output_dir tissue}"
tissue_h5="$output_dir/${tissue}.h5"
script_dir="$repo_root/analysis/test/src"

source ~/.bashrc

mkdir -p "$output_dir/predictions" logs
echo "output_dir: $output_dir"
echo "tissue: $tissue"
echo "started: $(date)"

if [ ! -f "$tissue_h5" ]; then
    echo "ERROR: h5 not found: $tissue_h5"
    echo "run build_pangolin.sbatch first"
    exit 1
fi

pred_dir="$output_dir/predictions"

echo "scoring with SplAIRE..."
conda activate splaire_env
python "$script_dir/score_splaire.py" "$tissue_h5" "$pred_dir" --ref --var --prefix "${tissue}_splaire" --batch-size 96

echo "scoring with SpliceAI..."
conda activate sa_env
python "$script_dir/score_spliceai.py" "$tissue_h5" "$pred_dir/${tissue}_sa.parquet" --batch-size 32

echo "scoring with Pangolin..."
conda activate pang_env
python "$script_dir/score_pangolin.py" "$tissue_h5" "$pred_dir/${tissue}_pang.parquet" --batch-size 96

echo "scoring with SpliceTransformer..."
conda activate spt-test
python "$script_dir/score_splicetransformer.py" "$tissue_h5" "$pred_dir/${tissue}_spt.parquet" --batch-size 64

# metrics
echo "computing $tissue metrics..."
conda activate splaire_env
python "$script_dir/compute_metrics.py" "$pred_dir" \
    "$output_dir/${tissue}_metrics.json" \
    --splice-combined

echo "done: $(date)"
