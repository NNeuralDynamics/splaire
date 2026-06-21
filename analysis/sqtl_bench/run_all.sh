#!/bin/bash
# master pipeline submitter for sqtl_bench.
# wraps existing run.sh (data prep) + score.sh (scoring) with sbatch deps,
# then submits metrics extraction.
#
# usage: edit CONFIG below, then `bash run_all.sh`

set -euo pipefail

# ============================================================
# CONFIG — edit me
# ============================================================

# data bench: haec | txrevise | leafcutter | ambig | all
DATA_MODE="${DATA_MODE:-haec}"

# scoring mode passed to score.sh (same as DATA_MODE usually)
SCORE_MODE="${SCORE_MODE:-$DATA_MODE}"

# whether to (re)build the data first. set to false if data already exists.
RUN_DATA_PREP="${RUN_DATA_PREP:-true}"

# whether to run scoring + metrics
RUN_SCORING="${RUN_SCORING:-true}"
RUN_METRICS="${RUN_METRICS:-true}"

# ============================================================
# don't edit below
# ============================================================

cd "$(dirname "$0")"

say() { printf "  %-40s %s\n" "$1" "$2"; }

data_jids=()

if [[ "$RUN_DATA_PREP" == "true" ]]; then
    echo ""
    echo "=========== data prep ($DATA_MODE) ==========="
    # run.sh submits but doesn't return JIDs cleanly. capture from squeue afterward.
    pre_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
    bash run.sh "$DATA_MODE" 2>&1 | tee /tmp/sqtl_run_$$.log
    sleep 2
    post_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
    new_jids=$(comm -13 <(echo "$pre_squeue") <(echo "$post_squeue"))
    if [[ -n "$new_jids" ]]; then
        data_jids=($new_jids)
        say "data prep jobs" "${data_jids[*]}"
    fi
    rm -f /tmp/sqtl_run_$$.log
fi

score_jids=()

if [[ "$RUN_SCORING" == "true" ]]; then
    echo ""
    echo "=========== scoring ($SCORE_MODE) ==========="
    if [[ ${#data_jids[@]} -gt 0 ]]; then
        dep=$(IFS=:; echo "${data_jids[*]}")
        pre_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
        bash score.sh "$SCORE_MODE" --after "$dep"
        sleep 2
        post_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
        score_jids=($(comm -13 <(echo "$pre_squeue") <(echo "$post_squeue")))
    else
        pre_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
        bash score.sh "$SCORE_MODE"
        sleep 2
        post_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
        score_jids=($(comm -13 <(echo "$pre_squeue") <(echo "$post_squeue")))
    fi
    if [[ ${#score_jids[@]} -gt 0 ]]; then
        say "score jobs" "${score_jids[*]}"
    fi
fi

if [[ "$RUN_METRICS" == "true" ]]; then
    echo ""
    echo "=========== metrics ==========="
    # metrics for sqtl_bench is computed by extract_cs_scores + the analysis notebook.
    # the existing pipeline doesn't have a single metrics.sbatch — instead, scoring
    # writes per-model parquets, and analysis.qmd loads + scores them.
    if [[ ${#score_jids[@]} -gt 0 ]]; then
        dep=$(IFS=:; echo "${score_jids[*]}")
        echo "  after all scoring (afterok:$dep), run analysis.qmd:"
        echo "    cd $(pwd) && jupyter execute analysis.qmd"
    fi
fi

echo ""
echo "all jobs submitted. monitor: squeue -u \$USER"
