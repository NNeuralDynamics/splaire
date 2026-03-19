#!/usr/bin/env python3
"""extract all credible set variants for benchmark positives

for each positive variant used in the sqtl benchmark, finds its credible
set and extracts all variants (including the positive itself). outputs a
vcf for scoring and a cs_map.csv linking variants to their credible sets.

usage:
    python src/make_cs_companions.py txrevise \
        --cs-dir $data_dir/txrevise/raw \
        --pairs $data_dir/txrevise/pairs.csv \
        --gtf $gtf \
        --out-dir $data_dir/cs_txrevise

    python src/make_cs_companions.py leafcutter \
        --cs-dir $data_dir/leafcutter/raw \
        --pairs $data_dir/leafcutter/pairs.csv \
        --gtf $gtf \
        --out-dir $data_dir/cs_leafcutter

    python src/make_cs_companions.py haec \
        --finemapping $haec_finemap \
        --pairs $data_dir/haec/pairs.csv \
        --gtf $gtf \
        --out-dir $data_dir/cs_haec
"""
import argparse
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_splice_sites

# same as benchmark
max_dist = 10_000


# ── shared helpers ────────────────────────────────────────

def load_gene_strands(gtf_path):
    """load gene strand from gtf, returns dict {gene_id: strand}"""
    strands = {}
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if parts[2] != "gene":
                continue
            strand = parts[6]
            for attr in parts[8].split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gene_id = attr.split('"')[1].split(".")[0]
                    strands[gene_id] = strand
                    break
    return strands


def load_gene_bounds_by_chrom(gtf_path):
    """load gene boundaries from gtf, returns dict {chrom: [(gene_id, start, end, strand)]}"""
    bounds = defaultdict(lambda: {"chrom": None, "start": float("inf"), "end": 0, "strand": None})
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if parts[2] != "transcript":
                continue
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            gene_id = None
            for attr in parts[8].split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gene_id = attr.split('"')[1].split(".")[0]
                    break
            if gene_id:
                bounds[gene_id]["chrom"] = chrom
                bounds[gene_id]["start"] = min(bounds[gene_id]["start"], start)
                bounds[gene_id]["end"] = max(bounds[gene_id]["end"], end)
                bounds[gene_id]["strand"] = strand
    by_chrom = defaultdict(list)
    for gene_id, b in bounds.items():
        if b["chrom"]:
            by_chrom[b["chrom"]].append((gene_id, b["start"], b["end"], b["strand"]))
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x[1])
    return dict(by_chrom)


def get_intron_gene(intron_chrom, intron_start, intron_end, gene_bounds_by_chrom):
    """get gene containing this intron, returns (gene_id, strand)"""
    if intron_chrom not in gene_bounds_by_chrom:
        return None, None
    for gene_id, gene_start, gene_end, strand in gene_bounds_by_chrom[intron_chrom]:
        if intron_start >= gene_start and intron_end <= gene_end:
            return gene_id, strand
    return None, None


def compute_splice_dist(chroms, positions, splice_sites):
    """vectorized distance to nearest splice site"""
    if hasattr(chroms, "values"):
        chroms, positions = chroms.values, positions.values
    result = np.full(len(chroms), np.inf)
    for chrom in np.unique(chroms):
        if chrom not in splice_sites:
            continue
        mask = chroms == chrom
        pos = positions[mask]
        sites = splice_sites[chrom]
        idx = np.searchsorted(sites, pos)
        left = np.where(idx > 0, np.abs(pos - sites[np.clip(idx - 1, 0, len(sites) - 1)]), np.inf)
        right = np.where(idx < len(sites), np.abs(pos - sites[np.clip(idx, 0, len(sites) - 1)]), np.inf)
        result[mask] = np.minimum(left, right)
    return result


def parse_intron(mt_id):
    """parse intron coords from molecular_trait_id or phenotype_id"""
    parts = str(mt_id).split(":")
    if len(parts) >= 3:
        try:
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            return chrom, int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None
    return None, None, None


def extract_tissue(path):
    """extract tissue name from eqtl catalogue filename"""
    stem = path.stem.replace(".credible_sets.tsv", "")
    return stem.split("_", 1)[1].lower() if "_" in stem else stem.lower()


def write_vcf(df, path):
    """write vcf with splice dist, strand, and optional intron info"""
    with gzip.open(path, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=SD,Number=1,Type=Integer,Description="Splice distance">\n')
        f.write('##INFO=<ID=ST,Number=1,Type=String,Description="Gene strand">\n')
        f.write('##INFO=<ID=IS,Number=1,Type=Integer,Description="Intron start">\n')
        f.write('##INFO=<ID=IE,Number=1,Type=Integer,Description="Intron end">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, r in df.iterrows():
            vid = r["var_key"].replace(":", "_")
            info = f"SD={int(r['splice_dist'])}"
            if pd.notna(r.get("strand")):
                info += f";ST={r['strand']}"
            if "intron_start" in r.index and pd.notna(r.get("intron_start")):
                info += f";IS={int(r['intron_start'])};IE={int(r['intron_end'])}"
            f.write(f"{r['chrom']}\t{r['pos']}\t{vid}\t{r['ref']}\t{r['alt']}\t.\t.\t{info}\n")


def save_results(result, out_dir, pos_keys):
    """save cs_map.csv and variants.vcf.gz"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result["is_pos"] = result["var_key"].isin(pos_keys)

    # cs_map: all rows including positives
    result.to_csv(out / "cs_map.csv", index=False)
    print(f"saved cs_map: {len(result):,} rows")

    # unique variants for scoring (all CS members)
    unique = result.drop_duplicates("var_key")
    write_vcf(unique, out / "variants.vcf.gz")

    n_pos = unique["is_pos"].sum()
    n_comp = len(unique) - n_pos
    n_cs = result["cs_id"].nunique()
    print(f"vcf: {len(unique):,} unique variants ({n_pos:,} positives, {n_comp:,} companions)")
    print(f"credible sets: {n_cs:,}")


# ── txrevise ──────────────────────────────────────────────

def run_txrevise(args):
    pairs = pd.read_csv(args.pairs)
    pos_keys = set(pairs["pos_var_key"])
    print(f"benchmark positives: {len(pos_keys):,}")

    print("loading splice sites...")
    splice_sites = load_splice_sites(args.gtf)

    print("loading gene strands...")
    gene_strands = load_gene_strands(args.gtf)

    cs_files = sorted(Path(args.cs_dir).glob("*.credible_sets.tsv.gz"))
    print(f"credible set files: {len(cs_files)}")

    all_rows = []
    for path in tqdm(cs_files, desc="scanning credible sets"):
        tissue = extract_tissue(path)
        df = pd.read_csv(path, sep="\t", compression="gzip")

        # txrevise: contained events only
        df = df[df["molecular_trait_id"].str.contains(".contained.")]

        # parse variant
        parts = df["variant"].str.split("_", expand=True)
        df["chrom"] = parts[0].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
        df["pos"] = parts[1].astype(int)
        df["ref"], df["alt"] = parts[2], parts[3]
        df["gene_id"] = df["molecular_trait_id"].str.split(".").str[0]
        df["var_key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]

        # snv only
        df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]

        # find credible sets containing benchmark positives
        cs_with_pos = df[df["var_key"].isin(pos_keys)]["cs_id"].unique()
        if len(cs_with_pos) == 0:
            continue

        # extract all variants from those credible sets
        cs_df = df[df["cs_id"].isin(cs_with_pos)].copy()

        # map each cs to its benchmark positive
        cs_to_pos = (df[df["var_key"].isin(pos_keys)]
                     .groupby("cs_id")["var_key"].first().to_dict())
        cs_df["pos_var_key"] = cs_df["cs_id"].map(cs_to_pos)
        cs_df["tissue"] = tissue
        cs_df["tissue_cs_id"] = tissue + ":" + cs_df["cs_id"].astype(str)

        all_rows.append(cs_df[["var_key", "chrom", "pos", "ref", "alt", "gene_id",
                               "cs_id", "tissue_cs_id", "pip", "tissue", "pos_var_key"]])

    if not all_rows:
        print("no credible sets found containing benchmark positives")
        return

    result = pd.concat(all_rows, ignore_index=True)
    print(f"raw rows: {len(result):,}")

    # splice distance
    result["splice_dist"] = compute_splice_dist(result["chrom"], result["pos"], splice_sites)
    result = result[result["splice_dist"] <= max_dist]

    # strand
    result["strand"] = result["gene_id"].map(gene_strands)

    print(f"after filters: {len(result):,} rows")
    save_results(result, args.out_dir, pos_keys)


# ── leafcutter ────────────────────────────────────────────

def run_leafcutter(args):
    pairs = pd.read_csv(args.pairs)
    pos_keys = set(pairs["pos_var_key"])
    print(f"benchmark positives: {len(pos_keys):,}")

    print("loading gtf...")
    gene_bounds_by_chrom = load_gene_bounds_by_chrom(args.gtf)

    cs_files = sorted(Path(args.cs_dir).glob("*.credible_sets.tsv.gz"))
    print(f"credible set files: {len(cs_files)}")

    all_rows = []
    for path in tqdm(cs_files, desc="scanning credible sets"):
        tissue = extract_tissue(path)
        df = pd.read_csv(path, sep="\t", compression="gzip")

        # parse variant
        parts = df["variant"].str.split("_", expand=True)
        df["chrom"] = parts[0].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
        df["pos"] = parts[1].astype(int)
        df["ref"], df["alt"] = parts[2], parts[3]
        df["var_key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]

        # parse intron
        intron_coords = df["molecular_trait_id"].apply(parse_intron)
        df["intron_chrom"] = intron_coords.apply(lambda x: x[0])
        df["intron_start"] = intron_coords.apply(lambda x: x[1])
        df["intron_end"] = intron_coords.apply(lambda x: x[2])

        # snv + valid intron
        df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
        df = df[df["intron_start"].notna()]

        # find credible sets containing benchmark positives
        cs_with_pos = df[df["var_key"].isin(pos_keys)]["cs_id"].unique()
        if len(cs_with_pos) == 0:
            continue

        cs_df = df[df["cs_id"].isin(cs_with_pos)].copy()

        # splice distance from intron boundaries
        cs_df["splice_dist"] = np.minimum(
            (cs_df["pos"] - cs_df["intron_start"]).abs(),
            (cs_df["pos"] - cs_df["intron_end"]).abs()
        )
        cs_df = cs_df[cs_df["splice_dist"] <= max_dist]

        # gene + strand from intron
        gene_strand = [get_intron_gene(ic, int(ist), int(ie), gene_bounds_by_chrom)
                       for ic, ist, ie in zip(cs_df["intron_chrom"],
                                              cs_df["intron_start"],
                                              cs_df["intron_end"])]
        cs_df["gene_id"] = [gs[0] for gs in gene_strand]
        cs_df["strand"] = [gs[1] for gs in gene_strand]
        cs_df = cs_df[cs_df["gene_id"].notna()]

        # map cs to positive
        cs_to_pos = (df[df["var_key"].isin(pos_keys)]
                     .groupby("cs_id")["var_key"].first().to_dict())
        cs_df["pos_var_key"] = cs_df["cs_id"].map(cs_to_pos)
        cs_df["tissue"] = tissue
        cs_df["tissue_cs_id"] = tissue + ":" + cs_df["cs_id"].astype(str)

        all_rows.append(cs_df[["var_key", "chrom", "pos", "ref", "alt", "gene_id",
                               "intron_start", "intron_end",
                               "cs_id", "tissue_cs_id", "pip", "splice_dist", "strand",
                               "tissue", "pos_var_key"]])

    if not all_rows:
        print("no credible sets found containing benchmark positives")
        return

    result = pd.concat(all_rows, ignore_index=True)
    print(f"after filters: {len(result):,} rows")
    save_results(result, args.out_dir, pos_keys)


# ── haec ──────────────────────────────────────────────────

def run_haec(args):
    pairs = pd.read_csv(args.pairs)
    pos_keys = set(pairs["pos_var_key"])
    print(f"benchmark positives: {len(pos_keys):,}")

    print("loading gtf...")
    gene_bounds_by_chrom = load_gene_bounds_by_chrom(args.gtf)

    # load finemapping
    print("loading finemapping...")
    fm = pd.read_csv(args.finemapping, sep="\t")
    fm["cs_id"] = fm["phenotype"] + "_L" + fm["credible_set_number"].astype(str)
    fm = fm.rename(columns={"posterior_inclusion_probability": "pip"})

    # parse variant_id to get chrom/pos/ref/alt with chr prefix
    parts = fm["variant_id"].str.split(":", expand=True)
    fm["chrom"] = parts[0].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
    fm["pos"] = parts[1].astype(int)
    fm["ref"] = parts[2]
    fm["alt"] = parts[3]
    fm["var_key"] = fm["chrom"] + ":" + fm["pos"].astype(str) + ":" + fm["ref"] + ":" + fm["alt"]

    # snv only
    fm = fm[(fm["ref"].str.len() == 1) & (fm["alt"].str.len() == 1)]

    # parse intron from phenotype
    intron_coords = fm["phenotype"].apply(parse_intron)
    fm["intron_chrom"] = intron_coords.apply(lambda x: x[0])
    fm["intron_start"] = intron_coords.apply(lambda x: x[1])
    fm["intron_end"] = intron_coords.apply(lambda x: x[2])
    fm = fm[fm["intron_start"].notna()]

    # splice distance from intron boundaries
    fm["splice_dist"] = np.minimum(
        (fm["pos"] - fm["intron_start"]).abs(),
        (fm["pos"] - fm["intron_end"]).abs()
    )
    fm = fm[fm["splice_dist"] <= max_dist]

    # gene + strand
    gene_strand = [get_intron_gene(ic, int(ist), int(ie), gene_bounds_by_chrom)
                   for ic, ist, ie in zip(fm["intron_chrom"],
                                          fm["intron_start"],
                                          fm["intron_end"])]
    fm["gene_id"] = [gs[0] for gs in gene_strand]
    fm["strand"] = [gs[1] for gs in gene_strand]
    fm = fm[fm["gene_id"].notna()]

    # find credible sets containing benchmark positives
    cs_with_pos = fm[fm["var_key"].isin(pos_keys)]["cs_id"].unique()
    print(f"credible sets with positives: {len(cs_with_pos):,}")

    result = fm[fm["cs_id"].isin(cs_with_pos)].copy()

    # map cs to positive
    cs_to_pos = (fm[fm["var_key"].isin(pos_keys)]
                 .groupby("cs_id")["var_key"].first().to_dict())
    result["pos_var_key"] = result["cs_id"].map(cs_to_pos)
    result["tissue"] = "HAEC"
    result["tissue_cs_id"] = "HAEC:" + result["cs_id"].astype(str)

    result = result[["var_key", "chrom", "pos", "ref", "alt", "gene_id",
                     "intron_start", "intron_end",
                     "cs_id", "tissue_cs_id", "pip", "splice_dist", "strand",
                     "tissue", "pos_var_key"]]

    print(f"after filters: {len(result):,} rows")
    save_results(result, args.out_dir, pos_keys)


# ── main ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="extract credible set variants for benchmark positives")
    sub = ap.add_subparsers(dest="dataset", required=True)

    tx = sub.add_parser("txrevise")
    tx.add_argument("--cs-dir", required=True, help="txrevise credible set directory")
    tx.add_argument("--pairs", required=True, help="benchmark pairs.csv")
    tx.add_argument("--gtf", required=True, help="gencode gtf for splice sites + strands")
    tx.add_argument("--out-dir", required=True)

    lc = sub.add_parser("leafcutter")
    lc.add_argument("--cs-dir", required=True, help="leafcutter credible set directory")
    lc.add_argument("--pairs", required=True, help="benchmark pairs.csv")
    lc.add_argument("--gtf", required=True, help="gencode gtf for gene bounds")
    lc.add_argument("--out-dir", required=True)

    hc = sub.add_parser("haec")
    hc.add_argument("--finemapping", required=True, help="haec finemapping tsv")
    hc.add_argument("--pairs", required=True, help="benchmark pairs.csv")
    hc.add_argument("--gtf", required=True, help="gencode gtf for gene bounds")
    hc.add_argument("--out-dir", required=True)

    args = ap.parse_args()

    if args.dataset == "txrevise":
        run_txrevise(args)
    elif args.dataset == "leafcutter":
        run_leafcutter(args)
    elif args.dataset == "haec":
        run_haec(args)

    print("\ndone")


if __name__ == "__main__":
    main()
