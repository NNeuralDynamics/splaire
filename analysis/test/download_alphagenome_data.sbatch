#!/bin/bash
#SBATCH --job-name=dl_ag_data
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# download AG auxiliary data files (FASTA + GTF + splice site/polyA annotations) to scratch
# so AG can find them locally without needing requests+aiohttp + GCS over fsspec.
# overrides OrganismSettings defaults at scoring time via the AG_DATA_DIR env var.
#
# files (~3 GB total, mostly FASTA):
#   GRCh38.p13.genome.fa                               ~3 GB
#   gencode.v46.annotation.gtf.gz.feather              ~MB
#   polyadb_human_v3_exon3_contiguous_gtfv46.feather   ~MB
#   gencode.v46.splice_sites_starts.feather            ~MB
#   gencode.v46.splice_sites_ends.feather              ~MB
#
# usage: sbatch download_alphagenome_data.sbatch [target_dir]

set -eo pipefail

target="${1:-/scratch/runyan.m/cache_home/alphagenome_data}"
mkdir -p "$target" logs
echo "target: $target"
echo "started: $(date)"

cd "$target"

base="https://storage.googleapis.com/alphagenome/reference"

declare -a files=(
    "${base}/gencode/hg38/GRCh38.p13.genome.fa"
    "${base}/gencode/hg38/gencode.v46.annotation.gtf.gz.feather"
    "${base}/exon/hg38/polyadb_human_v3_exon3_contiguous_gtfv46.feather"
    "${base}/gencode/hg38/gencode.v46.splice_sites_starts.feather"
    "${base}/gencode/hg38/gencode.v46.splice_sites_ends.feather"
)

for url in "${files[@]}"; do
    name=$(basename "$url")
    if [ -s "$name" ]; then
        echo "  already have $name ($(du -sh "$name" | cut -f1))"
        continue
    fi
    echo "  downloading $name..."
    curl -L --fail -o "$name" "$url"
    echo "    size: $(du -sh "$name" | cut -f1)"
done

# build FASTA index for pyfaidx
fasta="$target/GRCh38.p13.genome.fa"
if [ ! -f "${fasta}.fai" ]; then
    echo "building .fai index..."
    source ~/.bashrc
    conda activate "${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
    samtools faidx "$fasta"
    echo "  .fai size: $(du -sh "${fasta}.fai" | cut -f1)"
fi

echo "total: $(du -sh "$target" | cut -f1)"
echo "done: $(date)"
