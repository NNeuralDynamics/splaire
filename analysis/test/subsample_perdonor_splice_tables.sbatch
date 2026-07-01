#!/bin/bash
#SBATCH --job-name=subsample_perdonor
#SBATCH --partition=short
#SBATCH --time=0:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# subsample each per-donor tissue splice_table to ~3 genes per chrom (chr1/3/5/7),
# all donors for those genes. writes tiny tables to splice_tables_tiny/.
# usage: sbatch subsample_perdonor_splice_tables.sbatch

set -eo pipefail

src="${SRC:-/scratch/runyan.m/splice_tables}"
dst="${DST:-/scratch/runyan.m/splice_tables_tiny}"
mkdir -p "$dst" logs

echo "src: $src"
echo "dst: $dst"
echo "started: $(date)"

for tissue in haec10 lung brain_cortex testis whole_blood; do
    in_tsv="$src/${tissue}_splice_table.tsv"
    out_tsv="$dst/${tissue}_splice_table.tsv"
    if [ ! -f "$in_tsv" ]; then
        echo "WARN: missing $in_tsv — skipping"
        continue
    fi
    echo "subsampling $tissue..."
    awk -F'\t' '
        NR==1 { print; next }
        {
            gene=$1; chrom=$2
            if (chrom=="chr1" || chrom=="chr3" || chrom=="chr5" || chrom=="chr7") {
                if (!(gene in seen)) {
                    if (chrom_count[chrom] < 3) {
                        seen[gene] = 1
                        chrom_count[chrom]++
                    }
                }
                if (gene in seen) print
            }
        }
    ' "$in_tsv" > "$out_tsv"
    wc -l "$out_tsv"
done

echo "done: $(date)"
