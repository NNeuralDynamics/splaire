"""check whether unmatched leafcutter positives' genes exist in raw GE sumstats

usage: python check_unmatched.py --data-dir /scratch/runyan.m/sqtl_bench
"""
import argparse
import gzip
from pathlib import Path
from collections import defaultdict

import pandas as pd

# genes from the 55 unmatched variants (48 unique genes)
UNMATCHED_GENES = {
    "ENSG00000269179", "ENSG00000241489", "ENSG00000258881", "ENSG00000258924",
    "ENSG00000262633", "ENSG00000283563", "ENSG00000285404", "ENSG00000285952",
    "ENSG00000269891", "ENSG00000273184", "ENSG00000286231", "ENSG00000259060",
    "ENSG00000267426", "ENSG00000267149", "ENSG00000141979", "ENSG00000287542",
    "ENSG00000287505", "ENSG00000273167", "ENSG00000213024", "ENSG00000260861",
    "ENSG00000268533", "ENSG00000288271", "ENSG00000259753", "ENSG00000265794",
    "ENSG00000171570", "ENSG00000286280", "ENSG00000186230", "ENSG00000283321",
    "ENSG00000268193", "ENSG00000284762", "ENSG00000281593", "ENSG00000268279",
    "ENSG00000182093", "ENSG00000287603", "ENSG00000285446", "ENSG00000285959",
    "ENSG00000278419", "ENSG00000288656", "ENSG00000268400", "ENSG00000273331",
    "ENSG00000249209", "ENSG00000288053", "ENSG00000285733", "ENSG00000255872",
    "ENSG00000255054", "ENSG00000264187", "ENSG00000269502", "ENSG00000284505",
}

# versioned prefixes for fast string matching
PREFIXES = {g + "." for g in UNMATCHED_GENES}


def gene_matches(raw_gid):
    """check if versioned gene_id matches any unmatched gene"""
    for p in PREFIXES:
        if raw_gid.startswith(p):
            return p[:-1]  # strip trailing dot
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    data = Path(args.data_dir)
    ge_dir = data / "ge" / "sumstats"
    ge_files = sorted(ge_dir.glob("*.all.tsv.gz"))
    print(f"GE sumstats files: {len(ge_files)}")
    print(f"checking {len(UNMATCHED_GENES)} genes\n")

    gene_found_in = defaultdict(set)

    for f in ge_files:
        tissue = f.stem.split(".")[0].split("_", 1)[1]
        # stream line by line, only track which target genes appear
        found_this_tissue = set()
        with gzip.open(f, "rt") as fh:
            header = fh.readline().strip().split("\t")
            gi = header.index("gene_id")
            for line in fh:
                raw_gid = line.split("\t")[gi]
                g = gene_matches(raw_gid)
                if g:
                    found_this_tissue.add(g)
                    # early exit if all found
                    if len(found_this_tissue) == len(UNMATCHED_GENES):
                        break
        for g in found_this_tissue:
            gene_found_in[g].add(tissue)
        missing = len(UNMATCHED_GENES) - len(found_this_tissue)
        print(f"  {tissue}: {len(found_this_tissue)} found, {missing} missing", flush=True)

    # summary
    print(f"\n{'='*60}")
    never_found = UNMATCHED_GENES - set(gene_found_in.keys())
    sometimes_found = set(gene_found_in.keys())
    print(f"genes never in any GE file: {len(never_found)}/{len(UNMATCHED_GENES)}")
    print(f"genes in >=1 GE file: {len(sometimes_found)}/{len(UNMATCHED_GENES)}")

    if never_found:
        print(f"\nnever found:")
        for g in sorted(never_found):
            print(f"  {g}")

    if sometimes_found:
        print(f"\nfound in some tissues:")
        for g in sorted(sometimes_found):
            print(f"  {g}: {len(gene_found_in[g])} tissues")

    # save results
    out_path = data / "leafcutter" / "unmatched_gene_check.csv"
    rows = []
    for g in sorted(UNMATCHED_GENES):
        tissues = gene_found_in.get(g, set())
        rows.append({"gene_id": g, "n_tissues": len(tissues), "in_ge": len(tissues) > 0})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
