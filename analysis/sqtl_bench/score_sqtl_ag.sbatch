#!/bin/bash
#SBATCH --job-name=score_sqtl_ag
#SBATCH --output=logs/score_sqtl_ag_%j.out
#SBATCH --error=logs/score_sqtl_ag_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00

# AlphaGenome scoring (all_folds, 16k context) for an sqtl_bench dataset
# (pos.vcf.gz + neg.vcf.gz). loads the model once and reuses across pos/neg via --extra.
#
# usage: sbatch score_sqtl_ag.sbatch <dataset_dir>
#   dataset_dir e.g. /scratch/runyan.m/sqtl_bench/haec_hungarian
#   expects pos.vcf.gz and neg.vcf.gz in that dir
#   writes pos.ag.{h5,ssu.parquet,sj.parquet} + neg.ag.{...} to <dataset_dir>/scores/

set -e

dataset_dir="${1:?usage: sbatch score_sqtl_ag.sbatch <dataset_dir>}"
repo_root="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
src="$repo_root/analysis/sqtl_bench/src"
variant_src="$repo_root/analysis/test/src"

export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"

score_dir="$dataset_dir/scores"
mkdir -p "$score_dir" logs

source ~/.bashrc
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

# step 1: build variant tables for pos + neg
echo "=== building variant tables ==="
for label in pos neg; do
    vcf="$dataset_dir/${label}.vcf.gz"
    tsv="$score_dir/${label}.ag_variants.tsv"
    if [ ! -f "$vcf" ]; then
        echo "  skip $label: $vcf not found"
        continue
    fi
    python "$src/make_ag_variant_table.py" "$vcf" "$tsv"
done

# step 2: score both tables in one AG session (single model load)
echo "=== scoring with AG ==="
pos_tsv="$score_dir/pos.ag_variants.tsv"
neg_tsv="$score_dir/neg.ag_variants.tsv"
pos_prefix="$score_dir/pos.ag"
neg_prefix="$score_dir/neg.ag"

extra_args=""
if [ -f "$neg_tsv" ]; then
    extra_args="--extra $neg_tsv $neg_prefix"
fi

python "$variant_src/score_alphagenome_variant.py" \
    "$pos_tsv" "$pos_prefix" \
    $extra_args
