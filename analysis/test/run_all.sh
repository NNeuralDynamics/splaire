#!/bin/bash
# master pipeline submitter for the test bench (mane / gencode / canonical).
# chains build_h5 -> score -> metrics via slurm afterok deps.
# usage: edit the CONFIG block, then `bash run_all.sh`
#
# pipelines per bench:
#   STANDARD : build_${bench}.sbatch   -> score_${bench}.sbatch (splaire ref+var, spliceai, pangolin, spliceformer)
#   AG       : build_${bench}_ag.sbatch -> 4x score_${bench}_ag.sbatch (per fold) -> merge_ag.sbatch
#   then     : metrics.sbatch over all predictions/ outputs

set -euo pipefail

# ============================================================
# CONFIG — edit me
# ============================================================

# where all outputs land (e.g. $OUT/mane_select/, $OUT/gencode/)
OUT="${OUT:-/scratch/$USER/sphaec_out/canonical}"

# reference fasta
export REF_FASTA="${REF_FASTA:-/projects/talisman/mrunyan/paper/SpHAEC/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa}"

# conda env prefixes
export SPLAIRE_ENV_PREFIX="${SPLAIRE_ENV_PREFIX:-/scratch/$USER/conda_envs/splaire_env}"
export SA_ENV_PREFIX="${SA_ENV_PREFIX:-/scratch/$USER/conda_envs/sa_env}"
export PANG_ENV_PREFIX="${PANG_ENV_PREFIX:-/scratch/$USER/conda_envs/pang_env}"
export SPT_ENV_PREFIX="${SPT_ENV_PREFIX:-/scratch/$USER/conda_envs/spt_env}"
export AG_ENV_PREFIX="${AG_ENV_PREFIX:-/scratch/$USER/conda_envs/alphagenome_env}"
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/$USER/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/cache_home/jax_cache}"

# which benchmarks to run (any subset of: mane gencode canonical)
BENCHES=(mane gencode)

# which scoring pipelines to run (subset of: standard ag)
PIPELINES=(standard ag)

# AG GPU partition
GPU_GRES_AG="${GPU_GRES_AG:-gpu:a100:1}"

# ============================================================
# don't edit below
# ============================================================

cd "$(dirname "$0")"
mkdir -p logs "$OUT"

submit() { sbatch --parsable "$@"; }
say() { printf "  %-32s %s\n" "$1" "$2"; }

run_bench() {
    local bench="$1"
    local bench_label="${bench/mane/mane_select}"
    local bench_out="$OUT/$bench_label"
    local pred_dir="$bench_out/predictions"
    mkdir -p "$pred_dir"

    echo ""
    echo "=========== $bench ==========="

    local -a score_jids=()

    if [[ " ${PIPELINES[*]} " == *" standard "* ]]; then
        local std_h5="$bench_out/${bench_label}.h5"
        local build_jid=""
        if [[ ! -f "$std_h5" ]]; then
            build_jid=$(submit "build_${bench}.sbatch" "$bench_out")
            say "build_${bench}" "$build_jid"
        else
            say "build_${bench}" "SKIP (h5 exists)"
        fi
        local dep=""
        [[ -n "$build_jid" ]] && dep="--dependency=afterok:$build_jid"
        local sjid=$(submit $dep "score_${bench}.sbatch" "$bench_out")
        say "score_${bench} (splaire+sa+pang+spt)" "$sjid"
        score_jids+=("$sjid")
    fi

    if [[ " ${PIPELINES[*]} " == *" ag "* ]]; then
        local ag_h5_dir="${bench_out%/}_ag"
        local ag_h5="$ag_h5_dir/${bench_label}_ag.h5"
        local ag_build_jid=""
        if [[ ! -f "$ag_h5" ]]; then
            ag_build_jid=$(submit "build_${bench}_ag.sbatch" "$bench_out")
            say "build_${bench}_ag" "$ag_build_jid"
        else
            say "build_${bench}_ag" "SKIP (h5 exists)"
        fi
        local dep=""
        [[ -n "$ag_build_jid" ]] && dep="--dependency=afterok:$ag_build_jid"
        # 4 per-fold scoring jobs + 1 merge job (afterok all 4)
        local fold_jids=()
        for fold in fold_0 fold_1 fold_2 fold_3; do
            local fjid
            fjid=$(sbatch --parsable $dep --gres="$GPU_GRES_AG" \
                --export=ALL,AG_FLAGS="--no-merge --folds $fold" \
                "score_${bench}_ag.sbatch" "$bench_out")
            say "score_${bench}_ag $fold" "$fjid"
            fold_jids+=("$fjid")
        done
        local fold_dep
        fold_dep=$(IFS=:; echo "${fold_jids[*]}")
        local merged="$pred_dir/${bench_label}_ag.parquet"
        local mjid
        mjid=$(submit --dependency=afterok:$fold_dep merge_ag.sbatch "$merged")
        say "merge_${bench}_ag" "$mjid"
        score_jids+=("$mjid")
    fi

    # metrics
    if [[ ${#score_jids[@]} -gt 0 ]]; then
        local score_dep
        score_dep=$(IFS=:; echo "${score_jids[*]}")
        local mjid
        mjid=$(submit --dependency=afterok:$score_dep \
            metrics.sbatch "$pred_dir" "$bench_out/metrics.json")
        say "metrics_${bench}" "$mjid"
    fi
}

for b in "${BENCHES[@]}"; do
    run_bench "$b"
done

echo ""
echo "all jobs submitted. monitor: squeue -u \$USER"
