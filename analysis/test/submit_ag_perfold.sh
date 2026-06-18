#!/bin/bash
# submit a benchmark's AG scoring as 4 per-fold jobs + 1 merge job (afterok dep).
# each per-fold job has ONE JIT compile window at start (cached after warm_ag_cache),
# not 4. the merge runs on CPU once all 4 folds succeed.
#
# usage:
#   submit_ag_perfold.sh score_mane_ag.sbatch <output_dir>
#   submit_ag_perfold.sh score_gencode_ag.sbatch <output_dir>
#   submit_ag_perfold.sh score_pangolin_ag.sbatch <output_dir> <tissue>
#   submit_ag_perfold.sh score_deltasplice_ag.sbatch <output_dir>
#
# env overrides:
#   AG_FLAGS_BASE   base flags (default: --no-merge)
#                   --folds and --merge controls are added per-job
#   GPU_GRES        slurm gres (default: gpu:a100:1)

set -euo pipefail

sbatch_script="${1:?usage: submit_ag_perfold.sh <sbatch_script> <output_dir> [tissue]}"
output_dir="${2:?usage: submit_ag_perfold.sh <sbatch_script> <output_dir> [tissue]}"
tissue="${3:-}"

base_flags="${AG_FLAGS_BASE:---no-merge}"
gpu_gres="${GPU_GRES:-gpu:a100:1}"

# infer output parquet path (mirrors how the score_*_ag.sbatch files name it)
case "$sbatch_script" in
    *mane_ag*)        bench=mane_select ;;
    *gencode_ag*)     bench=gencode ;;
    *deltasplice_ag*) bench=deltasplice ;;
    *pangolin_ag*)
        if [ -z "$tissue" ]; then echo "ERROR: pangolin requires tissue arg"; exit 1; fi
        bench="$tissue"
        ;;
    *) echo "ERROR: unknown sbatch script: $sbatch_script"; exit 1 ;;
esac
out_parquet="$output_dir/predictions/${bench}_ag.parquet"

echo "benchmark: $bench"
echo "output parquet: $out_parquet"
echo "base AG_FLAGS: $base_flags"
echo ""

fold_jobs=()
for fold in fold_0 fold_1 fold_2 fold_3; do
    flags="$base_flags --folds $fold"
    if [ -n "$tissue" ]; then
        jid=$(sbatch --parsable --gres="$gpu_gres" --export=ALL,AG_FLAGS="$flags" \
            "$sbatch_script" "$output_dir" "$tissue")
    else
        jid=$(sbatch --parsable --gres="$gpu_gres" --export=ALL,AG_FLAGS="$flags" \
            "$sbatch_script" "$output_dir")
    fi
    echo "  $fold: job $jid"
    fold_jobs+=("$jid")
done

dep=$(IFS=:; echo "${fold_jobs[*]}")
mjid=$(sbatch --parsable --dependency=afterok:$dep merge_ag.sbatch "$out_parquet")
echo "  merge: job $mjid (afterok:$dep)"
