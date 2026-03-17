#!/usr/bin/env bash
#SBATCH --job-name=splice_data_pipeline
#SBATCH --partition=short
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@northeastern.edu

module load anaconda3/2022.05
eval "$(conda shell.bash hook)"
conda activate proc-rnaseq

set -euo pipefail

usage() {
  cat <<EOF
Usage: sbatch data-pipeline.sh [options]

Options (short | long):
  -i  | --input            INPUT_TSV        path to original v2p_tx_sites_SNVs.dedup.tsv
  -c  | --chrom-file       CHROM_FILE       one‑chromosome‑per‑line txt
  -C  | --chrom            CHROM            single chromosome (e.g. chr1)
  -s  | --sample           SAMPLE_FILE      one‑sample‑per‑line txt
  -f  | --fasta            FASTA            reference genome FASTA
  -I  | --intermediate     INT_DIR          directory for intermediate files
  -F  | --h5dir            H5_DIR           directory for final HDF5 outputs
  -m  | --mode             MODE             create_dataset mode: train,test,all
  -p  | --paralog          PARALOG          paralog filter: 0,1,all
  -n  | --name             NAME             base name suffix (e.g. S10_Chrom24_White)
  -g  | --make-gc          MAKE_GC          include genomic‑coord dataset? (true/false)
  -r  | --remove-missing   REMOVE_MISSING   zero-out SSE==777? (true/false)
EOF
  exit 1
}

# defaults
MODE="all"
PARALOG="all"
NAME="run"
MAKE_GC="false"
REMOVE_MISSING="false"
CHROM_FILE=""
CHROM=""

# parse options
PARSED=$(getopt -n "$0" -o i:c:C:s:f:I:F:m:p:n:g:r: \
  -l input:,chrom-file:,chrom:,sample:,fasta:,intermediate:,h5dir:,mode:,paralog:,name:,make-gc:,remove-missing: -- "$@") || usage

eval set -- "$PARSED"
while true; do
  case "$1" in
    -i|--input)          INPUT_TSV="$2";       shift 2 ;;  
    -c|--chrom-file)     CHROM_FILE="$2";      shift 2 ;;  
    -C|--chrom)          CHROM="$2";           shift 2 ;;  
    -s|--sample)         SAMPLE_FILE="$2";     shift 2 ;;  
    -f|--fasta)          FASTA="$2";           shift 2 ;;  
    -I|--intermediate)   INT_DIR="$2";         shift 2 ;;  
    -F|--h5dir)          H5_DIR="$2";          shift 2 ;;  
    -m|--mode)           MODE="$2";            shift 2 ;;  
    -p|--paralog)        PARALOG="$2";         shift 2 ;;  
    -n|--name)           NAME="$2";            shift 2 ;;  
    -g|--make-gc)        MAKE_GC="$2";         shift 2 ;;  
    -r|--remove-missing) REMOVE_MISSING="$2";  shift 2 ;;  
    --) shift; break ;;  
    *) usage ;;  
  esac
done

# require exactly one of CHROM_FILE or CHROM
if [[ -z "$CHROM_FILE" && -z "$CHROM" ]] || [[ -n "$CHROM_FILE" && -n "$CHROM" ]]; then
  echo "Error: specify exactly one of --chrom-file or --chrom"
  usage
fi

# ensure other required args
if [[ -z "${INPUT_TSV:-}" || -z "${SAMPLE_FILE:-}" || -z "${FASTA:-}" || -z "${INT_DIR:-}" || -z "${H5_DIR:-}" ]]; then
  usage
fi

# optional flags
GC_FLAG=""; [[ "$MAKE_GC" == "true" ]] && GC_FLAG="--make-gc"
RM_FLAG=""; [[ "$REMOVE_MISSING" == "true" ]] && RM_FLAG="--remove-missing"

# create dirs
mkdir -p "$INT_DIR" "$INT_DIR/logs" "$H5_DIR"

# log invocation
INVOKE_LOG="$INT_DIR/logs/${NAME}_${CHROM}_invocation.log"
{
  echo "[$(date)] Command: $0 $*"
  echo "[$(date)] INPUT_TSV=${INPUT_TSV}"
  echo "[$(date)] CHROM_FILE=${CHROM_FILE}"
  echo "[$(date)] CHROM=${CHROM}"
  echo "[$(date)] SAMPLE_FILE=${SAMPLE_FILE}"
  echo "[$(date)] FASTA=${FASTA}"
  echo "[$(date)] INT_DIR=${INT_DIR}"
  echo "[$(date)] H5_DIR=${H5_DIR}"
  echo "[$(date)] MODE=${MODE}"
  echo "[$(date)] PARALOG=${PARALOG}"
  echo "[$(date)] NAME=${NAME}"
  echo "[$(date)] MAKE_GC=${MAKE_GC}"
  echo "[$(date)] REMOVE_MISSING=${REMOVE_MISSING}"
  echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "[$(date)] SLURM_JOB_NAME=$SLURM_JOB_NAME"
  echo "[$(date)] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
} > "$INVOKE_LOG"

# intermediate paths
REF_SITES="$INT_DIR/Ref_${NAME}_sites.txt"
VAR_SITES="$INT_DIR/Var_${NAME}_sites.txt"
REF_SEQ="$INT_DIR/Ref_${NAME}_seqs.txt"
VAR_SEQ="$INT_DIR/Var_${NAME}_seqs.txt"

# Step 1
if [[ -n "$CHROM_FILE" ]]; then
  CHROM_ARG="--chrom-file $CHROM_FILE"
else
  CHROM_ARG="--chrom $CHROM"
fi

echo "[$(date)] Step 1: select chromosomes & samples"
python select_chrom_and_sample.py \
  -i "$INPUT_TSV" $CHROM_ARG \
  -o "$REF_SITES"

#python select_chrom_and_sample.py \
#  -i "$INPUT_TSV" $CHROM_ARG \
#  -s "$SAMPLE_FILE" -o "$REF_SITES"

# Step 2
echo "[$(date)] Step 2: adjust variant positions"
python adjust_sites.py -v "$REF_SITES" -a "$VAR_SITES"

# Step 3
echo "[$(date)] Step 3: extract & mutate sequences"
python mutate_sequences.py \
  -v "$REF_SITES" -a "$VAR_SITES" -f "$FASTA" \
  -o "$VAR_SEQ" -r "$REF_SEQ"


#echo "[$(date)] Step 3.5: sort paired sites & seqs"
#bash sort_paired_datasets.sh \
#  -s "$REF_SITES" \
#  -q "$REF_SEQ" \
#  -o "$INT_DIR"


#REF_SITES="$INT_DIR/$(basename "$REF_SITES" .txt)_sorted.txt"
#REF_SEQ="$INT_DIR/$(basename "$REF_SEQ" .txt)_sorted.txt"


#echo "[$(date)] Step 3.5: sort paired sites & seqs"
#bash sort_paired_datasets.sh \
#  -s "$VAR_SITES" \
#  -q "$VAR_SEQ" \
#  -o "$INT_DIR"


#VAR_SITES="$INT_DIR/$(basename "$VAR_SITES" .txt)_sorted.txt"
#VAR_SEQ="$INT_DIR/$(basename "$VAR_SEQ" .txt)_sorted.txt"



#echo "[$(date)] Step 4a: build VARIANT HDF5 dataset"
#python create_dataset_paired.py \
#  -s "$VAR_SEQ" -st "$VAR_SITES" \
#  -m "all" -p "$PARALOG" -n "Var_${NAME}" -o "$H5_DIR" \
#  $GC_FLAG $RM_FLAG


#echo "[$(date)] Step 4b: build REFERENCE HDF5 dataset"
#python create_dataset_paired.py \
#  -s "$REF_SEQ" -st "$REF_SITES" \
#  -m "all" -p "$PARALOG" -n "Ref_${NAME}" -o "$H5_DIR" \
#  $GC_FLAG $RM_FLAG


echo "[$(date)] Step 4a: build VARIANT HDF5 dataset"
python create_dataset.py \
  -s "$VAR_SEQ" -st "$VAR_SITES" \
  -m "all" -p "$PARALOG" -n "Var_${NAME}" -o "$H5_DIR" \
  $GC_FLAG $RM_FLAG

# Step 4b
#echo "[$(date)] Step 4b: build REFERENCE HDF5 dataset"
#python create_dataset.py \
#  -s "$REF_SEQ" -st "$REF_SITES" \
#  -m "all" -p "$PARALOG" -n "Ref_${NAME}" -o "$H5_DIR" \
#  $GC_FLAG $RM_FLAG




#echo "[$(date)] Step 5: post-process both HDF5 outputs"
#python ../utils/h5/filter_true_h5.py \
# -i "$H5_DIR/dataset_Var_${NAME}_all_${PARALOG}.h5" \
# -o "$H5_DIR/dataset_true_Var_${NAME}_all_${PARALOG}.h5" \
# -r 5000
 
#python ../utils/h5/filter_true_h5.py \
# -i "$H5_DIR/dataset_Ref_${NAME}_all_${PARALOG}.h5" \
# -o "$H5_DIR/dataset_true_Ref_${NAME}_all_${PARALOG}.h5" \
# -r 5000




# done
echo "[$(date)] Pipeline complete"
echo "Intermediate files in: $INT_DIR"
echo "Final H5 files in:    $H5_DIR"
