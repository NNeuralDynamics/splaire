#!/bin/bash
# submits the whole test bench in one go: build h5s, score, then metrics.
# slurm afterok deps mean each step waits for the previous.
#
# usage: edit the config block below, then `bash run_all.sh`
#
# splaire / sa / pang / spt run inside one bundled sbatch (score_<bench>.sbatch);
# ag runs as 4 per-fold gpu jobs followed by a merge.
# metrics runs after everything finishes.

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
# override from env, e.g. `BENCHES="mane" bash run_all.sh`
read -ra BENCHES <<< "${BENCHES:-mane gencode}"

# models to run. options:
#   splaire   splaire ref + var (cross-tissue mean splicing, both ref-input + var-input)
#   sa        spliceai
#   pang      pangolin (v1)
#   spt       spliceformer
#   ag        alphagenome (4 per-fold gpu jobs then merge)
# override from env, e.g. `MODELS="ag" bash run_all.sh`
read -ra MODELS <<< "${MODELS:-splaire sa pang spt ag}"

# alphagenome gpu partition (override if your cluster names differ)
GPU_GRES_AG="${GPU_GRES_AG:-gpu:a100:1}"

# ============================================================
# below this line: script logic
# ============================================================

cd "$(dirname "$0")"
mkdir -p logs "$OUT"

submit() { sbatch --parsable "$@"; }
say() { printf "  %-32s %s\n" "$1" "$2"; }

want() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

run_bench() {
    local bench="$1"
    local bench_label="${bench/mane/mane_select}"
    local bench_out="$OUT/$bench_label"
    local pred_dir="$bench_out/predictions"
    mkdir -p "$pred_dir"

    echo ""
    echo "=========== $bench ==========="

    local -a score_jids=()

    # standard pipeline (splaire / sa / pang / spt run together in one sbatch)
    local standard_models=()
    for m in splaire sa pang spt; do
        want "$m" && standard_models+=("$m")
    done

    if [[ ${#standard_models[@]} -gt 0 ]]; then
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
        local models_csv="${standard_models[*]}"
        local sjid
        sjid=$(sbatch --parsable $dep "--export=ALL,MODELS=$models_csv" \
            "score_${bench}.sbatch" "$bench_out")
        say "score_${bench} (${models_csv// /,})" "$sjid"
        score_jids+=("$sjid")
    fi

    # alphagenome
    if want ag; then
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
