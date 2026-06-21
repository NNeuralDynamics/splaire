#!/bin/bash
# submits the whole test bench in one go: build h5s, score, then metrics.
# slurm afterok deps mean each step waits for the previous.
#
# usage: edit the config block below, then `bash run_all.sh`
#
# two pipelines run per benchmark:
#   standard -> build_<bench>.sbatch then score_<bench>.sbatch
#               (one sbatch scores splaire ref+var, spliceai, pangolin, spliceformer)
#   ag       -> build_<bench>_ag.sbatch, then 4 per-fold alphagenome jobs, then merge
# both feed into metrics.sbatch.
#
# to run only alphagenome: set PIPELINES=(ag)
# to run only the standard bundle: set PIPELINES=(standard)
# to run only one of splaire/spliceai/pangolin/spliceformer: skip this script
# and submit that model's individual sbatch instead (score_<bench>.sbatch bundles
# all four sequentially and isn't split per model).

set -euo pipefail

# ============================================================
# config — edit me
# ============================================================

# where outputs land (one subdir per benchmark, e.g. $OUT/mane_select/)
OUT="${OUT:-/scratch/$USER/sphaec_out/canonical}"

# reference genome
export REF_FASTA="${REF_FASTA:-/projects/talisman/mrunyan/paper/SpHAEC/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa}"

# conda envs (must already exist)
export SPLAIRE_ENV_PREFIX="${SPLAIRE_ENV_PREFIX:-/scratch/$USER/conda_envs/splaire_env}"
export SA_ENV_PREFIX="${SA_ENV_PREFIX:-/scratch/$USER/conda_envs/sa_env}"
export PANG_ENV_PREFIX="${PANG_ENV_PREFIX:-/scratch/$USER/conda_envs/pang_env}"
export SPT_ENV_PREFIX="${SPT_ENV_PREFIX:-/scratch/$USER/conda_envs/spt_env}"
export AG_ENV_PREFIX="${AG_ENV_PREFIX:-/scratch/$USER/conda_envs/alphagenome_env}"
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/$USER/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/cache_home/jax_cache}"

# benchmarks to run. options:
#   mane      mane select canonical transcripts (gencode v39, primary chromosomes)
#   gencode   full gencode v39 protein-coding annotation
#   canonical canonical transcripts from gtex v8 expression-defined set
BENCHES=(mane gencode)

# scoring pipelines to run. options:
#   standard  one sbatch that scores splaire ref+var, spliceai, pangolin, spliceformer on the same h5
#   ag        alphagenome (4 per-fold gpu jobs then a cpu merge into one parquet)
PIPELINES=(standard ag)

# alphagenome gpu partition (override if your cluster names differ)
GPU_GRES_AG="${GPU_GRES_AG:-gpu:a100:1}"

# ============================================================
# below this line: script logic
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
            say "build_${bench}" "skip (h5 exists)"
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
            say "build_${bench}_ag" "skip (h5 exists)"
        fi
        local dep=""
        [[ -n "$ag_build_jid" ]] && dep="--dependency=afterok:$ag_build_jid"
        # 4 per-fold jobs, then one merge job waiting on all four
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

    # metrics waits on everything that scored
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
echo "submitted. watch with: squeue -u \$USER"
