#!/bin/bash
# submits the whole test bench in one go: build h5s, score, then metrics.
# slurm afterok deps mean each step waits for the previous.
#
# usage: edit the config block below, then `bash run_all.sh`
#
# splaire / sa / pang / spt run inside one bundled sbatch (score_<bench>.sbatch);
# ag runs as 4 per-fold gpu jobs followed by a merge.
# metrics runs after everything finishes.
#
# tiny end-to-end smoke test: `TEST_MODE=1 bash run_all.sh`
# (~3 transcripts per chrom from data/test_subset/, one pangolin tissue)

set -euo pipefail

# ============================================================
# config — edit me
# ============================================================

# where outputs land (one subdir per benchmark, e.g. $OUT/mane_select/)
# TEST_MODE picks a different default below so tiny outputs don't collide
# with stale production prediction files.
if [[ "${TEST_MODE:-0}" == "1" ]]; then
    OUT="${OUT:-/scratch/$USER/sphaec_out_tiny/canonical}"
else
    OUT="${OUT:-/scratch/$USER/sphaec_out/canonical}"
fi

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

# alphagenome gpu partition (override if your cluster names differ)
GPU_GRES_AG="${GPU_GRES_AG:-gpu:a100:1}"

# ============================================================
# TEST_MODE — tiny end-to-end smoke test. ~3 transcripts/chrom.
# overrides default BENCHES to all 5 canonical sets,
# sets MAX_PER_CHROM=3 for gencode (extract path), and points
# SITES at data/test_subset/* for the other benches.
# usage: TEST_MODE=1 bash run_all.sh
# ============================================================
TEST_MODE="${TEST_MODE:-0}"
if [[ "$TEST_MODE" == "1" ]]; then
    DEFAULT_BENCHES="mane gencode gencode_hg19 deltasplice pangolin"
    : "${MAX_PER_CHROM:=3}"
    export MAX_PER_CHROM
    echo "TEST_MODE=1 — tiny inputs from data/test_subset/ (MAX_PER_CHROM=$MAX_PER_CHROM)"
else
    DEFAULT_BENCHES="mane gencode"
fi

# benchmarks to run. options:
#   mane         MANE Select transcripts extracted from gencode v45 GTF (hg38, chr1/3/5/7)
#   gencode      all TSL=1 protein-coding exons from gencode v45 GTF (hg38, chr1/3/5/7) — matches paper
#   gencode_hg19 legacy SpliceAI 2019 canonical_dataset.txt (hg19, chr1/3/5/7) — different from paper
#   deltasplice  deltasplice's own benchmark (hg19, chr1/3/5/7/9)
#   pangolin     pangolin's test splice table, per-tissue (hg38, chr1/3/5/7/9)
# override from env, e.g. `BENCHES="mane" bash run_all.sh`
read -ra BENCHES <<< "${BENCHES:-$DEFAULT_BENCHES}"

# models to run. options:
#   splaire   splaire ref + var (cross-tissue mean splicing, both ref-input + var-input)
#   sa        spliceai
#   pang      pangolin (v1)
#   spt       spliceformer
#   ag        alphagenome (4 per-fold gpu jobs then merge)
# override from env, e.g. `MODELS="ag" bash run_all.sh`
read -ra MODELS <<< "${MODELS:-splaire sa pang spt ag}"

# pangolin tissues to score. defaults to one tissue in test mode, four otherwise.
if [[ "$TEST_MODE" == "1" ]]; then
    PANG_TISSUES_DEFAULT="heart"
else
    PANG_TISSUES_DEFAULT="heart liver brain testis"
fi
read -ra PANG_TISSUES <<< "${PANG_TISSUES:-$PANG_TISSUES_DEFAULT}"

# ============================================================
# below this line: script logic
# ============================================================

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
SUBSET_DIR="$SCRIPT_DIR/data/test_subset"
mkdir -p logs "$OUT"

submit() { sbatch --parsable "$@"; }
say() { printf "  %-32s %s\n" "$1" "$2"; }
want() { [[ " ${MODELS[*]} " == *" $1 "* ]]; }

# build --only-models comma-list from $MODELS so metrics only scores what
# we actually ran. prevents shape mismatches against stale prediction files
# in pred_dir (e.g. old rinalmo or pangolin from prior runs).
metrics_only_models_arg() {
    local out=""
    for m in "${MODELS[@]}"; do
        case "$m" in
            splaire) out+="splaire_ref,splaire_var," ;;
            sa)      out+="spliceai," ;;
            pang)    out+="pangolin," ;;
            spt)     out+="splicetransformer," ;;
            ag)      out+="alphagenome," ;;
        esac
    done
    [[ -n "$out" ]] && echo "--only-models=${out%,}"
}

# point SITES at the right tiny test file for $1, or unset for gencode/unknown.
set_test_sites() {
    local bench="$1"
    case "$bench" in
        mane)         export SITES="$SUBSET_DIR/canonical_dataset_mane_select_p.txt" ;;
        gencode_hg19) export SITES="$SUBSET_DIR/canonical_dataset.txt" ;;
        deltasplice)  export SITES="$SUBSET_DIR/gene_dataset.tsu.txt" ;;
        pangolin)     export SITES="$SUBSET_DIR/splice_table_Human.test.txt" ;;
        gencode)      unset SITES ;;  # gencode uses MAX_PER_CHROM, extracts from GTF
        *)            unset SITES ;;
    esac
}

run_bench() {
    local bench="$1"

    if [[ "$bench" == "pangolin" ]]; then
        run_bench_pangolin
        return
    fi

    local bench_label="${bench/mane/mane_select}"
    local bench_out="$OUT/$bench_label"
    local pred_dir="$bench_out/predictions"
    mkdir -p "$pred_dir"

    echo ""
    echo "=========== $bench ==========="

    [[ "$TEST_MODE" == "1" ]] && set_test_sites "$bench"

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
    if want ag && [[ ! -f "build_${bench}_ag.sbatch" ]]; then
        say "ag" "skip (no build_${bench}_ag.sbatch — AG not supported for this bench)"
    elif want ag; then
        local ag_h5_dir="${bench_out%/}_ag"
        local ag_h5="$ag_h5_dir/${bench_label}_ag.h5"
        local ag_build_jid=""
        mkdir -p "$ag_h5_dir"
        if [[ ! -f "$ag_h5" ]]; then
            ag_build_jid=$(submit "build_${bench}_ag.sbatch" "$ag_h5_dir")
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
            metrics.sbatch "$pred_dir" "$bench_out/metrics.json" $(metrics_only_models_arg))
        say "metrics_${bench}" "$mjid"
    fi

    unset SITES
}

# pangolin runs per tissue — each tissue has its own h5 and parquet set
run_bench_pangolin() {
    local bench_out="$OUT/pangolin"
    mkdir -p "$bench_out"

    echo ""
    echo "=========== pangolin (tissues: ${PANG_TISSUES[*]}) ==========="

    [[ "$TEST_MODE" == "1" ]] && set_test_sites "pangolin"

    local standard_models=()
    for m in splaire sa pang spt; do
        want "$m" && standard_models+=("$m")
    done

    # score_pangolin_ag.sbatch writes temps + final parquet to $bench_out/predictions/
    # (name-keyed by tissue), so all tissues share the same predictions dir.
    local pred_dir="$bench_out/predictions"
    mkdir -p "$pred_dir"

    for tissue in "${PANG_TISSUES[@]}"; do
        echo "----- pangolin/$tissue -----"
        local -a score_jids=()

        if [[ ${#standard_models[@]} -gt 0 ]]; then
            local std_h5="$bench_out/${tissue}.h5"
            local build_jid=""
            if [[ ! -f "$std_h5" ]]; then
                build_jid=$(submit build_pangolin.sbatch "$bench_out" "$tissue")
                say "build_pangolin_$tissue" "$build_jid"
            else
                say "build_pangolin_$tissue" "skip (h5 exists)"
            fi
            local dep=""
            [[ -n "$build_jid" ]] && dep="--dependency=afterok:$build_jid"
            local models_csv="${standard_models[*]}"
            local sjid
            sjid=$(sbatch --parsable $dep "--export=ALL,MODELS=$models_csv" \
                score_pangolin.sbatch "$bench_out" "$tissue")
            say "score_pangolin_$tissue (${models_csv// /,})" "$sjid"
            score_jids+=("$sjid")
        fi

        if want ag; then
            local ag_h5_dir="${bench_out}_ag"
            local ag_h5="$ag_h5_dir/${tissue}_ag.h5"
            local ag_build_jid=""
            mkdir -p "$ag_h5_dir"
            if [[ ! -f "$ag_h5" ]]; then
                ag_build_jid=$(submit build_pangolin_ag.sbatch "$ag_h5_dir" "$tissue")
                say "build_pangolin_ag_$tissue" "$ag_build_jid"
            else
                say "build_pangolin_ag_$tissue" "skip (h5 exists)"
            fi
            local dep=""
            [[ -n "$ag_build_jid" ]] && dep="--dependency=afterok:$ag_build_jid"
            local fold_jids=()
            for fold in fold_0 fold_1 fold_2 fold_3; do
                local fjid
                fjid=$(sbatch --parsable $dep --gres="$GPU_GRES_AG" \
                    --export=ALL,AG_FLAGS="--no-merge --folds $fold" \
                    score_pangolin_ag.sbatch "$bench_out" "$tissue")
                say "score_pangolin_ag $tissue $fold" "$fjid"
                fold_jids+=("$fjid")
            done
            local fold_dep
            fold_dep=$(IFS=:; echo "${fold_jids[*]}")
            local merged="$pred_dir/${tissue}_ag.parquet"
            local mjid
            mjid=$(submit --dependency=afterok:$fold_dep merge_ag.sbatch "$merged")
            say "merge_pangolin_ag_$tissue" "$mjid"
            score_jids+=("$mjid")
        fi

        if [[ ${#score_jids[@]} -gt 0 ]]; then
            local score_dep
            score_dep=$(IFS=:; echo "${score_jids[*]}")
            local mjid
            mjid=$(submit --dependency=afterok:$score_dep \
                metrics.sbatch "$pred_dir" "$bench_out/${tissue}_metrics.json" $(metrics_only_models_arg))
            say "metrics_pangolin_$tissue" "$mjid"
        fi
    done

    unset SITES
}

for b in "${BENCHES[@]}"; do
    run_bench "$b"
done

echo ""
echo "submitted. watch with: squeue -u \$USER"
