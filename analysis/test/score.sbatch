#!/bin/bash
#SBATCH --job-name=score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=score_logs/%x_%j.out
#SBATCH --error=score_logs/%x_%j.err

# sbatch score.sbatch <input_dir> <output_dir>
# scores all h5 files with all models, skips already-completed donors

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

input_dir="${1:?usage: sbatch score.sbatch input_dir output_dir}"
output_dir="${2:?usage: sbatch score.sbatch input_dir output_dir}"
script_dir="$repo_root/analysis/test"

source ~/.bashrc

mkdir -p "$output_dir/predictions" score_logs
completed="$output_dir/completed.txt"
touch "$completed"

for h5 in "$input_dir"/*.h5; do
    [ -f "$h5" ] || continue
    donor=$(basename "$h5" .h5)
    grep -qx "$donor" "$completed" && continue

    echo "$donor: $(date)"
    pred_dir="$output_dir/predictions/$donor"
    mkdir -p "$pred_dir"

    conda activate ${SPLAIRE_ENV:-splaire_env}
    python "$script_dir/src/score_splaire.py" "$h5" "$pred_dir" --ref --var --prefix "${donor}_splaire" --batch-size 6456

    conda activate ${SA_ENV:-sa_env}
    python "$script_dir/src/score_spliceai.py" "$h5" "$pred_dir/${donor}_sa.parquet" --batch-size 32

    conda activate ${PANG_ENV:-pang_env}
    python "$script_dir/src/score_pangolin.py" "$h5" "$pred_dir/${donor}_pang.parquet" --batch-size 6456

    conda activate ${SPT_ENV:-spt-test}
    python "$script_dir/src/score_splicetransformer.py" "$h5" "$pred_dir/${donor}_spt.parquet" --batch-size 64

    # mark complete after all models succeed
    echo "$donor" >> "$completed"
done

n_h5=$(ls "$input_dir"/*.h5 2>/dev/null | wc -l)
n_done=$(wc -l < "$completed")
echo "scoring complete: $n_done/$n_h5 donors"
