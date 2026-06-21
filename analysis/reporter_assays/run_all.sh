#!/bin/bash
# submits scoring for every (assay, model) combo in the reporter_assays bench.
# input h5s ({assay}/data/{assay}.h5) come from prepare_data.ipynb — run that first
# if they don't exist yet.
#
# usage: edit the config block below, then `bash run_all.sh`
#
# to run only one model: set MODELS=(splaire) etc — pick any subset of
#   splaire sa pang pang_v2 spt ag merlin nt segnt
# to run only one assay: set ASSAYS=(vex)
#
# metrics for reporter_assays live in analysis.qmd (jupyter notebook), not an sbatch.
# after all scoring finishes, pull the h5s back local and run the notebook,
# or run `jupyter execute analysis.qmd` on the cluster.

set -euo pipefail

# ============================================================
# config — edit me
# ============================================================

# conda envs (must already exist)
export SPLAIRE_ENV_PREFIX="${SPLAIRE_ENV_PREFIX:-/scratch/$USER/conda_envs/splaire_env}"
export SA_ENV_PREFIX="${SA_ENV_PREFIX:-/scratch/$USER/conda_envs/sa_env}"
export PANG_ENV_PREFIX="${PANG_ENV_PREFIX:-/scratch/$USER/conda_envs/pang_env}"
export SPT_ENV_PREFIX="${SPT_ENV_PREFIX:-/scratch/$USER/conda_envs/spt_env}"
export AG_ENV_PREFIX="${AG_ENV_PREFIX:-/scratch/$USER/conda_envs/alphagenome_env}"
export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/$USER/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/$USER/cache_home/jax_cache}"
export MERLIN_ENV_PREFIX="${MERLIN_ENV_PREFIX:-/scratch/$USER/conda_envs/merlin_env}"
export NT_ENV_PREFIX="${NT_ENV_PREFIX:-/scratch/$USER/conda_envs/nt_env}"

# assays to score. options:
#   vex                  vexseq HepG2 minigene exonic SNVs (Adamson 2018)
#   mfass                massively parallel functional splicing assay, exonic SNVs (Cheung 2019)
#   compass_genome       compass deep mutagenesis on a genomic exon (Braun 2024)
#   compass_minigene     compass deep mutagenesis on a minigene exon (Braun 2024)
#   opensplice_genome    opensplice genomic context (Liao 2024)
#   opensplice_minigene  opensplice minigene context (Liao 2024)
# override from env, e.g. `ASSAYS="vex" bash run_all.sh`
read -ra ASSAYS <<< "${ASSAYS:-vex mfass}"

# models to score. options:
#   splaire   splaire ref + var (cross-tissue mean splicing model, both ref-input + var-input variants)
#   sa        spliceai
#   pang      pangolin v1 (per-tissue p_splice + usage heads)
#   pang_v2   pangolin v2
#   spt       spliceformer (per-tissue usage heads)
#   ag        alphagenome
#   merlin    merlin (mlm-finetuned splicing model)
#   nt        nucleotide transformer
#   segnt     segmentnt
# override from env, e.g. `MODELS="ag" bash run_all.sh`
read -ra MODELS <<< "${MODELS:-splaire sa pang pang_v2 spt ag}"

# ============================================================
# below this line: script logic
# ============================================================

cd "$(dirname "$0")"
mkdir -p scripts/logs

submit() { sbatch --parsable "$@"; }
say() { printf "  %-40s %s\n" "$1" "$2"; }

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
            say "$model" "skip (no $sbatch_file)"
            continue
        fi
        jid=$(submit "$sbatch_file")
        say "score $model" "$jid"
        all_score_jids+=("$jid")
    done
done

# metrics aren't a slurm job — they live in analysis.qmd.
# print the dependency string so you can chain a notebook run manually.
if [[ ${#all_score_jids[@]} -gt 0 ]]; then
    dep=$(IFS=:; echo "${all_score_jids[*]}")
    echo ""
    echo "after scoring finishes (afterok:$dep) run the metrics notebook:"
    echo "  cd $(pwd) && jupyter execute analysis.qmd"
    echo "  or pull {assay}/data/*.h5 back locally and open analysis.qmd"
fi

echo ""
echo "submitted. watch with: squeue -u \$USER"
