#!/bin/bash
#SBATCH --job-name=score_ag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# score h5 with alphagenome splice site classification + usage
#
# default: ensemble all 4 folds (fold_0..fold_3). per-fold and per-track
# outputs stored separately in the parquet — NO averaging at this stage.
#   - fold-specific models use Borzoi cross-validation folds
#   - NOT chromosome-based splits — each chrom is split into 8 sections
#   - ~12.5% of each chromosome is held out per fold
#   - all_folds model trained on everything — only for variant effect prediction
#
# output cols (4 folds × 736 = 2,944 prediction cols):
#   acceptor_fold_<N>, donor_fold_<N>             (per-fold SPLICE_SITES)
#   ssu_fold_<N>_track<T>                         (per-fold, per-tissue-track SPLICE_SITE_USAGE)
#
# usage: sbatch score_alphagenome.sbatch input_h5 output_dir prefix [folds]
# examples:
#   sbatch score_alphagenome.sbatch gencode.h5 /output gencode
#   sbatch score_alphagenome.sbatch gencode.h5 /output gencode fold_0
#   sbatch score_alphagenome.sbatch gencode.h5 /output gencode fold_0,fold_1

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

input_h5="${1:?usage: sbatch score_alphagenome.sbatch input_h5 output_dir prefix [folds]}"
output_dir="${2:?usage: sbatch score_alphagenome.sbatch input_h5 output_dir prefix [folds]}"
prefix="${3:?usage: sbatch score_alphagenome.sbatch input_h5 output_dir prefix [folds]}"
folds="${4:-fold_0,fold_1,fold_2,fold_3}"

script_dir="$repo_root/analysis/test/src"

source ~/.bashrc

mkdir -p "$output_dir/predictions" logs
echo "output_dir: $output_dir"
echo "input_h5:   $input_h5"
echo "folds:      $folds"
echo "started:    $(date)"

if [ ! -f "$input_h5" ]; then
    echo "ERROR: h5 not found: $input_h5"
    exit 1
fi

echo "scoring with AlphaGenome (folds=$folds)..."
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"
python "$script_dir/score_alphagenome.py" \
    "$input_h5" \
    "$output_dir/predictions/${prefix}_ag.parquet" \
    --folds "$folds"

echo "done: $(date)"
