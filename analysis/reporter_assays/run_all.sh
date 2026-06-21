#!/bin/bash
# master pipeline submitter for the reporter_assays bench
# (vex_seq, mfass, compass_genome, compass_minigene, opensplice_genome, opensplice_minigene).
#
# the input h5 ({assay}/data/{assay}.h5) is prepared by prepare_data.ipynb beforehand.
# this script chains: scoring per (assay, model) -> metrics.
#
# usage: edit CONFIG below, then `bash run_all.sh`

set -euo pipefail

# ============================================================
# CONFIG — edit me
# ============================================================

# conda envs
export SPLAIRE_ENV_PREFIX="${SPLAIRE_ENV_PREFIX:-/scratch/$USER/conda_envs/splaire_env}"
export SA_ENV_PREFIX="${SA_ENV_PREFIX:-/scratch/$USER/conda_envs/sa_env}"
export PANG_ENV_PREFIX="${PANG_ENV_PREFIX:-/scratch/$USER/conda_envs/pang_env}"
export SPT_ENV_PREFIX="${SPT_ENV_PREFIX:-/scratch/$USER/conda_envs/spt_env}"
export AG_ENV_PREFIX="${AG_ENV_PREFIX:-/scratch/$USER/conda_envs/alphagenome_env}"
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/$USER/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/cache_home/jax_cache}"
export MERLIN_ENV_PREFIX="${MERLIN_ENV_PREFIX:-/scratch/$USER/conda_envs/merlin_env}"
export NT_ENV_PREFIX="${NT_ENV_PREFIX:-/scratch/$USER/conda_envs/nt_env}"

# which assays to score (subset of: vex mfass compass_genome compass_minigene opensplice_genome opensplice_minigene)
ASSAYS=(vex mfass)

# which models to score (subset of: splaire sa pang pang_v2 spt ag merlin nt segnt)
MODELS=(splaire sa pang pang_v2 spt ag)

# ============================================================
# don't edit below
# ============================================================

cd "$(dirname "$0")"
mkdir -p scripts/logs

submit() { sbatch --parsable "$@"; }
say() { printf "  %-40s %s\n" "$1" "$2"; }

# normalize assay prefix used by the sbatch filenames
sbatch_prefix() {
    case "$1" in
        vex) echo "vex" ;;
        mfass) echo "mfass" ;;
        compass_genome) echo "compass_genome" ;;
        compass_minigene) echo "compass_minigene" ;;
        opensplice_genome) echo "opensplice_genome" ;;
        opensplice_minigene) echo "opensplice_minigene" ;;
    esac
}

declare -a all_score_jids=()

for assay in "${ASSAYS[@]}"; do
    echo ""
    echo "=========== $assay ==========="
    prefix=$(sbatch_prefix "$assay")

    for model in "${MODELS[@]}"; do
        sbatch_file="scripts/submit_${prefix}_${model}.sbatch"
        if [[ ! -f "$sbatch_file" ]]; then
            say "$model" "SKIP (no $sbatch_file)"
            continue
        fi
        jid=$(submit "$sbatch_file")
        say "score $model" "$jid"
        all_score_jids+=("$jid")
    done
done

# metrics computation lives in analysis.qmd (jupyter notebook).
# print pointer rather than submit — running the notebook on the cluster
# is awkward; pull the h5s back and run analysis.qmd locally.
if [[ ${#all_score_jids[@]} -gt 0 ]]; then
    dep=$(IFS=:; echo "${all_score_jids[*]}")
    echo ""
    echo "after all scoring completes (afterok:$dep), run metrics:"
    echo "  cd $(pwd) && jupyter execute analysis.qmd"
    echo "  (or pull {assay}/data/*.h5 locally and run analysis.qmd in your IDE)"
fi

echo ""
echo "all jobs submitted. monitor: squeue -u \$USER"
