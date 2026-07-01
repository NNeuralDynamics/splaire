#!/bin/bash
#SBATCH --job-name=build_gencode_ag
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# build gencode per-gene h5 for AlphaGenome (1Mb seq per gene, reference only)
# matches build_gencode.sbatch: extracts TSL=1 protein-coding sites from gencode v45 (hg38)
# usage: sbatch build_gencode_ag.sbatch output_dir

set -eo pipefail

_submit_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
repo_root="$(cd "$_submit_dir/../.." && pwd)"

output_dir="${1:?usage: sbatch build_gencode_ag.sbatch output_dir}"
genome_fasta="${REF_FASTA:?must export REF_FASTA=/path/to/GRCh38.primary_assembly.genome.fa}"
gtf="${GENCODE_GTF:-$repo_root/pipeline/reference/GRCh38/gencode.v45.primary_assembly.annotation.gtf.gz}"
paralogs="${PARALOGS:-$repo_root/pipeline/reference/GRCh38/paralogs_GRCh38.txt.gz}"

script_dir="$repo_root/analysis/test/src"
pipeline_dir="$repo_root/pipeline/src"

source ~/.bashrc
conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"

mkdir -p "$output_dir" logs
echo "output_dir: $output_dir"
echo "gtf: $gtf"
echo "started: $(date)"

# step 1: extract TSL=1 protein-coding sites from v45 GTF on test chroms (hg38)
sites="${SITES:-$output_dir/canonical_dataset_gencode.txt}"
if [ ! -f "$sites" ]; then
    echo "extracting sites from GTF..."
    extra=""
    [ -n "${MAX_PER_CHROM:-}" ] && extra="--max-per-chrom $MAX_PER_CHROM"
    python "$script_dir/extract_canonical_sites.py" "$gtf" "$sites" \
        --paralogs "$paralogs" --chroms chr1 chr3 chr5 chr7 chr9 $extra
fi

# step 2: convert sites to enriched matrix
matrix="$output_dir/gencode_matrix.tsv"
if [ ! -f "$matrix" ]; then
    echo "converting to matrix..."
    python "$script_dir/canonical_to_matrix.py" "$sites" "$matrix"
fi

# step 3: build per-gene h5
h5="$output_dir/gencode_ag.h5"
if [ ! -f "$h5" ]; then
    echo "building per-gene h5..."
    cd "$pipeline_dir"
    python build_h5_alphagenome.py \
        --donor CANONICAL --split test \
        --input "$matrix" \
        --chroms chr1,chr3,chr5,chr7,chr9 \
        --fasta "$genome_fasta" \
        --output "$h5" \
        --mode basic --paralog 0 \
        --reference --skip-empty-genes
fi

echo "done: $(date)"
