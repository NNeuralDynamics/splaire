#!/usr/bin/env python3
"""make leafcutter ambig benchmark vcfs

ambiguous credible sets: max pip < 0.9, multiple candidate variants
does model ranking agree with fine-mapping pip ranking

filters:
- max pip < 0.9 (ambiguous)
- 2-10 variants per cs
- intron within gene bounds
- all variants are snvs
- all variants within 5kb of both donor/acceptor
"""
import argparse
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_gene_bounds(gtf_path):
    """load gene boundaries from gtf, returns dict of gene bounds"""
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
    return dict(bounds)


def parse_intron(mt_id):
    """parse intron from molecular_trait_id"""
    parts = mt_id.split(":")
    if len(parts) >= 4:
        try:
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            start, end = int(parts[1]), int(parts[2])
            # strand from cluster part: clu_X_+ or clu_X_-
            clu = parts[3]
            strand = "+" if clu.endswith("+") else "-" if clu.endswith("-") else "+"
            return chrom, start, end, strand
        except ValueError:
            pass
    elif len(parts) >= 3:
        try:
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            return chrom, int(parts[1]), int(parts[2]), "+"
        except ValueError:
            pass
    return None, None, None, "+"


def extract_tissue(path):
    """extract tissue name from filename"""
    stem = path.stem.replace(".credible_sets.tsv", "")
    return "_".join(stem.split("_")[1:]).lower()


def write_vcf(df, path):
    """write vcf with ambig-specific info fields"""
    with gzip.open(path, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=MT,Number=1,Type=String,Description="Molecular trait id">\n')
        f.write('##INFO=<ID=CSID,Number=1,Type=String,Description="Credible set id">\n')
        f.write('##INFO=<ID=ST,Number=1,Type=String,Description="Strand">\n')
        f.write('##INFO=<ID=DIST_DONOR,Number=1,Type=Integer,Description="Distance to donor">\n')
        f.write('##INFO=<ID=DIST_ACCEPTOR,Number=1,Type=Integer,Description="Distance to acceptor">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, r in df.iterrows():
            vid = r["var_key"].replace(":", "_")
            info_parts = [f"MT={r['mt_id']}", f"CSID={r['cs_id']}", f"ST={r['strand']}"]
            if pd.notna(r.get("dist_donor")):
                info_parts.append(f"DIST_DONOR={int(r['dist_donor'])}")
            if pd.notna(r.get("dist_acceptor")):
                info_parts.append(f"DIST_ACCEPTOR={int(r['dist_acceptor'])}")
            info = ";".join(info_parts)
            f.write(f"{r['chrom']}\t{r['pos']}\t{vid}\t{r['ref']}\t{r['alt']}\t.\t.\t{info}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs-dir", required=True, help="credible sets directory")
    ap.add_argument("--gtf", required=True, help="gtf file for gene bounds")
    ap.add_argument("--out-dir", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # load gene bounds
    print("loading gtf...")
    gene_bounds = load_gene_bounds(args.gtf)
    print(f"genes: {len(gene_bounds):,}")

    # load credible sets
    cs_files = list(Path(args.cs_dir).glob("*.credible_sets.tsv.gz"))
    print(f"credible set files: {len(cs_files)}")

    all_cs = []  # list of dicts
    filter_counts = {"total": 0}

    for path in tqdm(cs_files, desc="loading credible sets"):
        tissue = extract_tissue(path)
        df = pd.read_csv(path, sep="\t", compression="gzip")

        # group by cs_id
        for cs_id, cs_df in df.groupby("cs_id"):
            n_vars = len(cs_df)
            max_pip = cs_df["pip"].max()

            # parse intron from first variant
            mt_id = cs_df["molecular_trait_id"].iloc[0]
            intron_chrom, intron_start, intron_end, strand = parse_intron(mt_id)

            # get gene_id
            gene_id = None
            if "gene_id" in cs_df.columns and pd.notna(cs_df["gene_id"].iloc[0]):
                gene_id = str(cs_df["gene_id"].iloc[0]).split(".")[0]

            all_cs.append({
                "tissue": tissue,
                "cs_id": cs_id,
                "mt_id": mt_id,
                "n_variants": n_vars,
                "max_pip": max_pip,
                "gene_id": gene_id,
                "intron_chrom": intron_chrom,
                "intron_start": intron_start,
                "intron_end": intron_end,
                "strand": strand,
                "variants": cs_df.copy(),
            })

    filter_counts["total"] = len(all_cs)
    print(f"total credible sets: {len(all_cs):,}")

    # filter 1: max pip < 0.9 (ambiguous)
    all_cs = [c for c in all_cs if c["max_pip"] < 0.9]
    filter_counts["ambig"] = len(all_cs)
    print(f"after max_pip < 0.9: {len(all_cs):,}")

    # filter 2: 2-10 variants
    all_cs = [c for c in all_cs if 2 <= c["n_variants"] <= 10]
    filter_counts["size_2_10"] = len(all_cs)
    print(f"after 2-10 variants: {len(all_cs):,}")

    # filter 3: intron within gene bounds
    def intron_in_gene(c):
        if c["gene_id"] is None or c["gene_id"] not in gene_bounds:
            return False
        g = gene_bounds[c["gene_id"]]
        if c["intron_start"] is None:
            return False
        return c["intron_start"] >= g["start"] and c["intron_end"] <= g["end"]

    all_cs = [c for c in all_cs if intron_in_gene(c)]
    filter_counts["in_gene"] = len(all_cs)
    print(f"after intron in gene: {len(all_cs):,}")

    # filter 4: all variants are snvs
    def all_snvs(c):
        for var in c["variants"]["variant"]:
            parts = var.split("_")
            if len(parts) >= 4:
                ref, alt = parts[2], parts[3]
                if len(ref) != 1 or len(alt) != 1:
                    return False
            else:
                return False
        return True

    all_cs = [c for c in all_cs if all_snvs(c)]
    filter_counts["snvs"] = len(all_cs)
    print(f"after snvs only: {len(all_cs):,}")

    # compute distances for each variant
    print("computing distances...")
    all_variants = []

    for c in tqdm(all_cs, desc="computing distances"):
        strand = c["strand"]
        intron_start, intron_end = c["intron_start"], c["intron_end"]

        # donor/acceptor based on strand
        if strand == "+":
            donor_pos, acceptor_pos = intron_start, intron_end
        else:
            donor_pos, acceptor_pos = intron_end, intron_start

        for _, row in c["variants"].iterrows():
            # parse variant: chr_pos_ref_alt
            parts = row["variant"].split("_")
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            pos = int(parts[1])
            ref, alt = parts[2], parts[3]

            dist_donor = abs(pos - donor_pos)
            dist_acceptor = abs(pos - acceptor_pos)

            all_variants.append({
                "tissue": c["tissue"],
                "cs_id": c["cs_id"],
                "mt_id": c["mt_id"],
                "var_key": f"{chrom}:{pos}:{ref}:{alt}",
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "pip": row["pip"],
                "strand": strand,
                "intron_start": intron_start,
                "intron_end": intron_end,
                "donor_pos": donor_pos,
                "acceptor_pos": acceptor_pos,
                "dist_donor": dist_donor,
                "dist_acceptor": dist_acceptor,
                "n_variants_in_cs": c["n_variants"],
            })

    variants_df = pd.DataFrame(all_variants)
    print(f"total variant rows: {len(variants_df):,}")

    # filter 5: all variants within 5kb of both donor and acceptor
    # (this is the "all_both" group)
    def cs_all_both(group):
        return (group["dist_donor"] <= 5000).all() and (group["dist_acceptor"] <= 5000).all()

    cs_groups = variants_df.groupby(["tissue", "cs_id"])
    keep_cs = cs_groups.filter(cs_all_both)
    variants_df = keep_cs.copy()
    filter_counts["all_both"] = variants_df[["tissue", "cs_id"]].drop_duplicates().shape[0]
    print(f"after all_both filter: {filter_counts['all_both']:,} cs")

    # save filter funnel
    funnel = pd.DataFrame([
        {"stage": "total", "count": filter_counts["total"]},
        {"stage": "max_pip_lt_0.9", "count": filter_counts["ambig"]},
        {"stage": "2_to_10_variants", "count": filter_counts["size_2_10"]},
        {"stage": "intron_in_gene", "count": filter_counts["in_gene"]},
        {"stage": "snvs_only", "count": filter_counts["snvs"]},
        {"stage": "all_both_5kb", "count": filter_counts["all_both"]},
    ])
    funnel.to_csv(out / "filter_funnel.csv", index=False)

    # cs-level stats
    cs_stats = variants_df.groupby(["tissue", "cs_id"]).agg(
        n_variants=("var_key", "count"),
        max_pip=("pip", "max"),
        pip_range=("pip", lambda x: x.max() - x.min()),
        mt_id=("mt_id", "first"),
        strand=("strand", "first"),
        intron_start=("intron_start", "first"),
        intron_end=("intron_end", "first"),
    ).reset_index()
    cs_stats.to_csv(out / "cs_stats.csv", index=False)
    print(f"credible sets: {len(cs_stats):,}")

    # variants per cs distribution
    size_dist = cs_stats["n_variants"].value_counts().sort_index().reset_index()
    size_dist.columns = ["n_variants", "count"]
    size_dist.to_csv(out / "cs_size_distribution.csv", index=False)

    # unique vcf for scoring
    deduped = variants_df.drop_duplicates("var_key")
    write_vcf(deduped, out / "ambig.vcf.gz")
    print(f"unique variants: {len(deduped):,}")

    # full variant data for analysis
    variants_df.to_csv(out / "cs_data.csv", index=False)

    # variant to tissues mapping
    var_tissues = variants_df.groupby("var_key")["tissue"].apply(list).reset_index()
    var_tissues["n_tissues"] = var_tissues["tissue"].apply(len)
    var_tissues["tissues"] = var_tissues["tissue"].apply(lambda x: ",".join(sorted(set(x))))
    var_tissues = var_tissues[["var_key", "n_tissues", "tissues"]]
    var_tissues.to_csv(out / "variant_tissues.csv", index=False)

    # pip stats per cs
    pip_stats = variants_df.groupby(["tissue", "cs_id"]).apply(
        lambda g: pd.Series({
            "pip_values": ",".join(f"{p:.4f}" for p in sorted(g["pip"], reverse=True)),
            "top_pip": g["pip"].max(),
            "second_pip": sorted(g["pip"], reverse=True)[1] if len(g) > 1 else None,
        }),
        include_groups=False,
    ).reset_index()
    pip_stats.to_csv(out / "pip_stats.csv", index=False)

    print(f"\noutput: {out}")
    print(f"  ambig.vcf.gz - {len(deduped):,} variants for scoring")
    print(f"  cs_data.csv - {len(variants_df):,} rows (all cs/tissue/variant)")
    print(f"  cs_stats.csv - {len(cs_stats):,} credible sets")
    print(f"  filter_funnel.csv - filter stage counts")
    print(f"  cs_size_distribution.csv - variants per cs")
    print(f"  variant_tissues.csv - variant to tissue mapping")
    print(f"  pip_stats.csv - pip values per cs")


if __name__ == "__main__":
    main()
