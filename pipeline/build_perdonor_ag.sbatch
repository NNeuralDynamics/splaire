#!/bin/bash
#SBATCH --job-name=build_ag_perdonor
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

tissue="${1:?usage: sbatch build_perdonor_ag.sbatch <tissue>  (tissue in haec10|lung|brain_cortex|testis|whole_blood)}"
repo="${REPO:?must export REPO=/path/to/splaire/repo}"
out="${OUT:?must export OUT=/path/to/output_dir}"
data="${DATA:?must export DATA=/path/to/splice_tables}"

case "$tissue" in
    haec10)
        default_samplesheet="$repo/pipeline/configs/haec/haec182_samples.tsv"
        default_splits="$repo/pipeline/configs/haec/haec10_splits.yaml"
        ;;
    lung|brain_cortex|testis|whole_blood)
        default_samplesheet="$repo/pipeline/configs/gtex/${tissue}_samples.tsv"
        default_splits="$repo/pipeline/configs/gtex/${tissue}_splits.yaml"
        ;;
    *) echo "unknown tissue: $tissue"; exit 1 ;;
esac

# env overrides for tiny test mode
samplesheet="${SAMPLESHEET:-$default_samplesheet}"
splits="${SPLITS:-$default_splits}"
splice_table="${SPLICE_TABLE:-$data/${tissue}_splice_table.tsv}"
run_dir="$out/${tissue}_run_ag"
mkdir -p "$run_dir" logs
cd "$run_dir"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:?must export SPLAIRE_ENV_PREFIX=/path/to/splaire_env}"
export SPLAIRE_CONDA_ENV="$CONDA_PREFIX"

echo "tissue: $tissue"
echo "SPLAIRE_CONDA_ENV: $SPLAIRE_CONDA_ENV"
echo "run_dir: $run_dir"
echo "started: $(date)"

nextflow run "$repo/pipeline/main.nf" -entry build_h5_ag_only \
    --input_matrix "$splice_table" \
    --samplesheet "$samplesheet" \
    --splits_config "$splits" \
    --output_dir "$out/${tissue}" \
    --dataset_out_dir "$out/${tissue}/ml_data_var_ag" \
    -profile slurm -resume

echo "done: $(date)"
