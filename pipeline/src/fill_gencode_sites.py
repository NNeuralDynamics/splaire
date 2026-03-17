#!/usr/bin/env python3
"""fill missing GENCODE splice sites with SSU=777.0"""
import pandas as pd
import argparse
from collections import defaultdict
from tqdm import tqdm


def load_gencode_exons(gtf_path):
    """TSL=1 protein-coding exon positions by gene"""
    gtf = pd.read_csv(
        gtf_path, sep="\t", comment="#", header=None,
        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
        usecols=["seqname", "feature", "start", "end", "strand", "attribute"],
        low_memory=False
    )
    gtf["transcript_type"] = gtf["attribute"].str.extract(r'transcript_type "([^"]+)"')
    gtf["TSL"] = gtf["attribute"].str.extract(r'transcript_support_level "(\d+)"')[0]
    gtf["gene_id"] = gtf["attribute"].str.extract(r'gene_id "([^"]+)"')

    # filter to TSL=1 protein-coding exons
    exons = gtf.query("feature=='exon' and transcript_type=='protein_coding' and TSL=='1'")

    # build lookup by gene
    positions = defaultdict(lambda: {"starts": set(), "ends": set(), "chrom": None, "strand": None})
    for _, r in tqdm(exons.iterrows(), total=len(exons), desc="loading GTF"):
        gene = r.gene_id
        positions[gene]["starts"].add(int(r.start))
        positions[gene]["ends"].add(int(r.end))
        positions[gene]["chrom"] = r.seqname
        positions[gene]["strand"] = r.strand

    return positions


def load_paralogs(paralogs_file):
    par = pd.read_csv(paralogs_file, sep="\t", dtype=str)
    return set(par["Gene stable ID"].dropna())


def parse_list(val):
    if pd.isna(val) or not str(val).strip():
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]


def fill_row(row, gencode_positions, used_by_gene):
    gene = row["Gene_ID"]
    if gene not in gencode_positions:
        return row

    start, end = int(row["Start"]), int(row["End"])
    gc = gencode_positions[gene]

    # parse existing positions
    existing_starts = set(int(x) for x in parse_list(row["exon_starts"]))
    existing_ends = set(int(x) for x in parse_list(row["exon_ends"]))

    # track used positions for Phase 2
    used_by_gene[gene]["starts"] |= existing_starts
    used_by_gene[gene]["ends"] |= existing_ends

    # find missing positions within region
    missing_starts = sorted(p for p in gc["starts"] if start <= p <= end and p not in existing_starts)
    missing_ends = sorted(p for p in gc["ends"] if start <= p <= end and p not in existing_ends)

    if not missing_starts and not missing_ends:
        return row

    # append to existing
    new_starts = parse_list(row["exon_starts"]) + [str(p) for p in missing_starts]
    new_ends = parse_list(row["exon_ends"]) + [str(p) for p in missing_ends]
    new_start_ssus = parse_list(row["exon_start_SSUs"]) + ["777.0"] * len(missing_starts)
    new_end_ssus = parse_list(row["exon_end_SSUs"]) + ["777.0"] * len(missing_ends)

    # track newly used positions
    used_by_gene[gene]["starts"] |= set(missing_starts)
    used_by_gene[gene]["ends"] |= set(missing_ends)

    row["exon_starts"] = ",".join(new_starts)
    row["exon_ends"] = ",".join(new_ends)
    row["exon_start_SSUs"] = ",".join(new_start_ssus)
    row["exon_end_SSUs"] = ",".join(new_end_ssus)

    return row


def main():
    parser = argparse.ArgumentParser(description="fill missing GENCODE splice sites with SSU=777.0")
    parser.add_argument("--input", required=True, help="input TSV with SSU values")
    parser.add_argument("--gtf", required=True, help="GENCODE GTF file")
    parser.add_argument("--paralogs", required=True, help="paralogs file")
    parser.add_argument("--output", required=True, help="output TSV")
    args = parser.parse_args()

    print("loading GENCODE exon positions...")
    gencode = load_gencode_exons(args.gtf)
    print(f"loaded {len(gencode)} genes")

    print("loading paralogs...")
    paralog_set = load_paralogs(args.paralogs)
    print(f"loaded {len(paralog_set)} paralogs")

    print("loading input TSV...")
    df = pd.read_csv(args.input, sep="\t", dtype=str)
    columns = list(df.columns)

    # fill existing rows
    print("filling existing rows...")
    used_by_gene = defaultdict(lambda: {"starts": set(), "ends": set()})
    filled = []
    n_filled = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="rows"):
        orig_starts = len(parse_list(row["exon_starts"]))
        row = fill_row(row, gencode, used_by_gene)
        if len(parse_list(row["exon_starts"])) > orig_starts:
            n_filled += 1
        filled.append(row)
    print(f"filled {n_filled} rows")

    # holder rows for genes with unused GENCODE positions
    print("creating holder rows...")
    holder_rows = []
    for gene, gc in tqdm(gencode.items(), desc="genes"):
        used = used_by_gene.get(gene, {"starts": set(), "ends": set()})
        miss_starts = sorted(gc["starts"] - used["starts"])
        miss_ends = sorted(gc["ends"] - used["ends"])

        # require at least one missing of each type
        if not miss_starts or not miss_ends:
            continue

        # determine paralog status
        base_gene = gene.split(".")[0]
        paralog_status = "1" if base_gene in paralog_set else "0"

        holder = {col: "" for col in columns}
        holder["Gene_ID"] = gene
        holder["Chromosome"] = gc["chrom"] if gc["chrom"] else ""
        holder["Strand"] = gc["strand"] if gc["strand"] else ""
        holder["paralog_status"] = paralog_status
        holder["Allele"] = "holder"
        holder["Unique_ID"] = "holder_holder_holder"
        holder["exon_starts"] = ",".join(map(str, miss_starts))
        holder["exon_ends"] = ",".join(map(str, miss_ends))
        holder["exon_start_SSUs"] = ",".join(["777.0"] * len(miss_starts))
        holder["exon_end_SSUs"] = ",".join(["777.0"] * len(miss_ends))
        holder_rows.append(holder)

    print(f"created {len(holder_rows)} holder rows")

    df_out = pd.concat([pd.DataFrame(filled), pd.DataFrame(holder_rows)], ignore_index=True)
    df_out.to_csv(args.output, sep="\t", index=False)
    print(f"saved {len(df_out)} total rows to {args.output}")


if __name__ == "__main__":
    main()
