#!/bin/bash
#SBATCH --job-name=ag_haec_cs
#SBATCH --output=logs/ag_haec_cs_%j.out
#SBATCH --error=logs/ag_haec_cs_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

# score haec credible-set variants with AlphaGenome (all_folds, 16k context).
# step 1: join VCF with AG GTF to produce a variant table
# step 2: run AG predict_variant per variant; output:
#   <prefix>.h5            dense ref+alt SPLICE_SITES (n, 16384, 5)
#   <prefix>.ssu.parquet   sparse SSU (long format)
#   <prefix>.sj.parquet    SJ junctions (long format)
#
# usage: sbatch score_haec_cs_ag.sbatch $OUT/sqtl_bench

set -e

data_dir="${1:?usage: sbatch score_haec_cs_ag.sbatch <data_dir>}"
repo_root="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
src="$repo_root/analysis/sqtl_bench/src"
variant_src="$repo_root/analysis/test/src"

export AG_DATA_DIR="${AG_DATA_DIR:-/scratch/runyan.m/cache_home/alphagenome_data}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/scratch/runyan.m/cache_home/jax_cache}"

vcf="$data_dir/haec/cs_all.vcf.gz"
score_dir="$data_dir/haec/scores"
tsv="$score_dir/cs_all_ag_variants.tsv"
out_prefix="$score_dir/cs_all_ag"

mkdir -p "$score_dir" logs

source ~/.bashrc

# step 1: join VCF with GTF (CPU only — use alphagenome_env for pandas+feather support)
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"
echo "=== building variant table ==="
python "$src/make_ag_variant_table.py" "$vcf" "$tsv"

# step 2: score with AG (GPU)
echo "=== scoring with AG ==="
python "$variant_src/score_alphagenome_variant.py" \
    "$tsv" "$out_prefix"
