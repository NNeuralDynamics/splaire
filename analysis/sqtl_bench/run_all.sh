#!/bin/bash
# wraps run.sh (data prep) and score.sh (scoring) with slurm afterok deps,
# then prints the metrics step. picks up sbatch job ids by diffing squeue
# before and after each call.
#
# usage: edit the config block below, then `bash run_all.sh`

set -euo pipefail

# ============================================================
# config — edit me
# ============================================================

# which data bench to build. options:
#   haec        primary lung airway epithelial cells, in-house sqtls + credible sets
#   txrevise    txrevise sqtls from gtex (sumstats + credible sets)
#   leafcutter  leafcutter sqtls from gtex (sumstats + credible sets)
#   ambig       ambiguous splice site set for splaire-var stress test
#   all         every bench above
DATA_MODE="${DATA_MODE:-haec}"

# scoring mode — usually same as DATA_MODE. options match score.sh:
#   haec | txrevise | leafcutter | ambig | all
SCORE_MODE="${SCORE_MODE:-$DATA_MODE}"

# models to score. options (space separated):
#   sa pang pang_v2 splaire spt merlin
# leave as default (all 6) or pick a subset e.g. MODELS="splaire" for just splaire,
# MODELS="splaire pang_v2" for two, etc.
export MODELS="${MODELS:-sa pang pang_v2 splaire spt merlin}"

# set to false to skip data prep (e.g. already built)
RUN_DATA_PREP="${RUN_DATA_PREP:-true}"

# scoring + metrics toggles
RUN_SCORING="${RUN_SCORING:-true}"
RUN_METRICS="${RUN_METRICS:-true}"

# ============================================================
# below this line: script logic
# ============================================================

cd "$(dirname "$0")"

say() { printf "  %-40s %s\n" "$1" "$2"; }

data_jids=()

if [[ "$RUN_DATA_PREP" == "true" ]]; then
    echo ""
    echo "=========== data prep ($DATA_MODE) ==========="
    # run.sh prints sbatch lines but no clean jid stream, so diff squeue
    pre_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
    bash run.sh "$DATA_MODE"
    sleep 2
    post_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
    new_jids=$(comm -13 <(echo "$pre_squeue") <(echo "$post_squeue"))
    if [[ -n "$new_jids" ]]; then
        data_jids=($new_jids)
        say "data prep jobs" "${data_jids[*]}"
    fi
fi

score_jids=()

if [[ "$RUN_SCORING" == "true" ]]; then
    echo ""
    echo "=========== scoring ($SCORE_MODE) with MODELS=$MODELS ==========="
    pre_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
    if [[ ${#data_jids[@]} -gt 0 ]]; then
        dep=$(IFS=:; echo "${data_jids[*]}")
        bash score.sh "$SCORE_MODE" --after "$dep"
    else
        bash score.sh "$SCORE_MODE"
    fi
    sleep 2
    post_squeue=$(squeue -u "$USER" -h -o '%i' | sort -u)
    score_jids=($(comm -13 <(echo "$pre_squeue") <(echo "$post_squeue")))
    if [[ ${#score_jids[@]} -gt 0 ]]; then
        say "score jobs" "${score_jids[*]}"
    fi
fi

if [[ "$RUN_METRICS" == "true" ]]; then
    echo ""
    echo "=========== metrics ==========="
    # sqtl metrics happen in analysis.qmd, no single metrics.sbatch
    if [[ ${#score_jids[@]} -gt 0 ]]; then
        dep=$(IFS=:; echo "${score_jids[*]}")
        echo "  after scoring finishes (afterok:$dep) run analysis.qmd:"
        echo "    cd $(pwd) && jupyter execute analysis.qmd"
    fi
fi

echo ""
echo "submitted. watch with: squeue -u \$USER"
