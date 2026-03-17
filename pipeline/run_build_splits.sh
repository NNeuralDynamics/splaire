#!/bin/bash
# build train/validation h5 datasets for 5-way replicate splits
#
# need:
#   - run generate_train_val_splits.py to create split tsv files
#   - set SPLITS_DIR, SAMPLE_FILE, FASTA, OUTPUT_BASE below
#
# usage: ./run_build_splits.sh

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${PIPELINE_DIR}/src"

# paths to set before running
SPLITS_DIR="${SPLITS_DIR:-/scratch/runyan.m/splits}"
SAMPLE_FILE="${SAMPLE_FILE:-${PIPELINE_DIR}/configs/haec/train_samples100.txt}"
FASTA="${FASTA:-${PIPELINE_DIR}/reference/GRCh38/GRCh38.primary_assembly.genome.fa}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/runyan.m/split_h5}"

N_SPLITS=5
CHROMS="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22"

# read donor ids into array
mapfile -t DONORS < "${SAMPLE_FILE}"
N_DONORS=${#DONORS[@]}
echo "loaded ${N_DONORS} donors from ${SAMPLE_FILE}"

for SPLIT in $(seq 1 ${N_SPLITS}); do
    echo "========================================"
    echo "split ${SPLIT}"
    echo "========================================"

    SPLIT_OUT="${OUTPUT_BASE}/split${SPLIT}"
    TRAIN_DIR="${SPLIT_OUT}/train"
    VALID_DIR="${SPLIT_OUT}/valid"
    mkdir -p "${TRAIN_DIR}" "${VALID_DIR}" "${SPLIT_OUT}/logs"

    # concatenate paralogs + non-paralog train into one training tsv
    TRAIN_TSV="${SPLIT_OUT}/split${SPLIT}_train_full.tsv"
    if [ ! -f "${TRAIN_TSV}" ]; then
        echo "  concatenating train tsv..."
        cp "${SPLITS_DIR}/main_train_paralogs.tsv" "${TRAIN_TSV}"
        tail -n +2 "${SPLITS_DIR}/split${SPLIT}_nonparalog_train.tsv" >> "${TRAIN_TSV}"
        echo "  -> ${TRAIN_TSV}"
    else
        echo "  train tsv exists, skipping concatenation"
    fi

    VALID_TSV="${SPLITS_DIR}/split${SPLIT}_validation.tsv"

    # submit build_h5 jobs for training
    echo "  submitting ${N_DONORS} train jobs..."
    TRAIN_JOB=$(sbatch \
        --parsable \
        --job-name="train_s${SPLIT}" \
        --array="0-$((N_DONORS - 1))" \
        --partition=short \
        --mem=16G \
        --time=24:00:00 \
        --cpus-per-task=2 \
        --output="${SPLIT_OUT}/logs/train_%A_%a.out" \
        --error="${SPLIT_OUT}/logs/train_%A_%a.err" \
        --wrap="
            source ~/.bashrc && conda activate sp && module load bedtools
            DONOR=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${SAMPLE_FILE})
            python ${SRC_DIR}/build_h5.py \
                --donor \${DONOR} \
                --split train \
                --input ${TRAIN_TSV} \
                --chroms ${CHROMS} \
                --fasta ${FASTA} \
                --output ${TRAIN_DIR}/train_\${DONOR}.h5 \
                --work-dir ${SPLIT_OUT}/work/train_\${DONOR} \
                --log-dir ${SPLIT_OUT}/logs \
                --paralog all \
                --make-gc
        ")
    echo "  train array job: ${TRAIN_JOB}"

    # submit build_h5 jobs for validation
    echo "  submitting ${N_DONORS} valid jobs..."
    VALID_JOB=$(sbatch \
        --parsable \
        --job-name="valid_s${SPLIT}" \
        --array="0-$((N_DONORS - 1))" \
        --partition=short \
        --mem=16G \
        --time=24:00:00 \
        --cpus-per-task=2 \
        --output="${SPLIT_OUT}/logs/valid_%A_%a.out" \
        --error="${SPLIT_OUT}/logs/valid_%A_%a.err" \
        --wrap="
            source ~/.bashrc && conda activate sp && module load bedtools
            DONOR=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" ${SAMPLE_FILE})
            python ${SRC_DIR}/build_h5.py \
                --donor \${DONOR} \
                --split valid \
                --input ${VALID_TSV} \
                --chroms ${CHROMS} \
                --fasta ${FASTA} \
                --output ${VALID_DIR}/valid_\${DONOR}.h5 \
                --work-dir ${SPLIT_OUT}/work/valid_\${DONOR} \
                --log-dir ${SPLIT_OUT}/logs \
                --paralog all \
                --make-gc
        ")
    echo "  valid array job: ${VALID_JOB}"

    # submit combine jobs (depend on array jobs finishing)
    echo "  submitting combine jobs..."
    sbatch \
        --job-name="combine_train_s${SPLIT}" \
        --dependency="afterok:${TRAIN_JOB}" \
        --partition=short \
        --mem=16G \
        --time=4:00:00 \
        --output="${SPLIT_OUT}/logs/combine_train_%j.out" \
        --error="${SPLIT_OUT}/logs/combine_train_%j.err" \
        --wrap="
            source ~/.bashrc && conda activate sp && module load bedtools
            python ${SRC_DIR}/combine_h5.py \
                --input_dir ${TRAIN_DIR} \
                --pattern '*.h5' \
                --output ${SPLIT_OUT}/combined_train_split${SPLIT}.h5
        "

    sbatch \
        --job-name="combine_valid_s${SPLIT}" \
        --dependency="afterok:${VALID_JOB}" \
        --partition=short \
        --mem=16G \
        --time=4:00:00 \
        --output="${SPLIT_OUT}/logs/combine_valid_%j.out" \
        --error="${SPLIT_OUT}/logs/combine_valid_%j.err" \
        --wrap="
            source ~/.bashrc && conda activate sp && module load bedtools
            python ${SRC_DIR}/combine_h5.py \
                --input_dir ${VALID_DIR} \
                --pattern '*.h5' \
                --output ${SPLIT_OUT}/combined_valid_split${SPLIT}.h5
        "

    echo ""
done

echo "all jobs submitted."
echo "output will be in ${OUTPUT_BASE}/split{1..${N_SPLITS}}/"
echo "  combined_train_split{N}.h5"
echo "  combined_valid_split{N}.h5"
