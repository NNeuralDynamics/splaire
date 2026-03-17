#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -s SITES_FILE -q SEQS_FILE -o OUT_DIR

Sort a pair of “sites” and “seqs” files so that:
  - sites file is sorted by Gene_ID (column 1, tab-separated)
  - seqs file is reordered to match the Unique_ID order in sorted sites

Outputs in OUT_DIR:
  <basename>_sites_sorted.txt
  <basename>_seqs_sorted.txt
EOF
  exit 1
}

# parse args
while getopts "s:q:o:" opt; do
  case $opt in
    s) SITES="$OPTARG" ;;
    q) SEQS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    *) usage ;;
  esac
done
[[ -n "${SITES:-}" && -n "${SEQS:-}" && -n "${OUTDIR:-}" ]] || usage

mkdir -p "$OUTDIR"

# derive basenames
base_sites=$(basename "$SITES" .txt)
base_seqs=$(basename "$SEQS" .txt)

sorted_sites="$OUTDIR/${base_sites}_sorted.txt"
sorted_seqs="$OUTDIR/${base_seqs}_sorted.txt"
ids_file="$OUTDIR/${base_sites}.ids.tmp"

# sort sites, preserving header
{
  head -n1 "$SITES"
  tail -n+2 "$SITES" | sort -t$'\t' -k1,1
} > "$sorted_sites"

# extract Unique_IDs (column 8)
tail -n+2 "$sorted_sites" | cut -f8 > "$ids_file"

# reorder seqs to match
awk -F, -v OFS=, '
  NR==FNR { ids[++n]=$1; next }       # read ids into array
  FNR==1  { print; next }             # print header of seqs file
  { seq[$1]=$0 }                      # map each Unique_ID -> full line
  END {
    for(i=1; i<=n; i++) {
      id = ids[i]
      if (id in seq) {
        print seq[id]
      } else {
        # warn if missing
        printf("# MISSING_SEQ_FOR %s\n", id) > "/dev/stderr"
      }
    }
  }
' "$ids_file" "$SEQS" > "$sorted_seqs"

# cleanup
rm -f "$ids_file"

echo "Wrote:"
echo "  sites -> $sorted_sites"
echo "  seqs  -> $sorted_seqs"
