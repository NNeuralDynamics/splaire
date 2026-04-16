#!/usr/bin/env python3
"""haec sqtl benchmark — 4-tier matched positives/negatives from finemapping"""
import argparse
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# thresholds
neg_pip = 0.01
max_dist = 10_000
tpm_bin_size = 0.4


def write_vcf(df, path):
    """write vcf with intron info"""
    with gzip.open(path, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=SD,Number=1,Type=Integer,Description="Splice distance">\n')
        f.write('##INFO=<ID=IS,Number=1,Type=Integer,Description="Intron start">\n')
        f.write('##INFO=<ID=IE,Number=1,Type=Integer,Description="Intron end">\n')
        f.write('##INFO=<ID=ST,Number=1,Type=String,Description="Gene strand">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, r in df.iterrows():
            vid = r["var_key"].replace(":", "_")
            info = f"SD={int(r['splice_dist'])}"
            if pd.notna(r.get("intron_start")):
                info += f";IS={int(r['intron_start'])};IE={int(r['intron_end'])}"
            if pd.notna(r.get("strand")):
                info += f";ST={r['strand']}"
            f.write(f"{r['chrom']}\t{r['pos']}\t{vid}\t{r['ref']}\t{r['alt']}\t.\t.\t{info}\n")


def parse_intron(phenotype_id):
    """parse intron coords from phenotype_id (chr:start:end:clu_...)"""
    if pd.isna(phenotype_id):
        return None, None, None
    parts = str(phenotype_id).split(":")
    if len(parts) >= 3:
        try:
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            return chrom, int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None
    return None, None, None


def load_gene_bounds(gtf_path):
    """load gene boundaries from gtf"""
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


def get_intron_gene(intron_chrom, intron_start, intron_end, gene_bounds_by_chrom):
    """get gene containing this intron"""
    if intron_chrom not in gene_bounds_by_chrom:
        return None, None
    for gene_id, gene_start, gene_end, strand in gene_bounds_by_chrom[intron_chrom]:
        if intron_start >= gene_start and intron_end <= gene_end:
            return gene_id, strand
    return None, None


def load_intron_sites(intron_dir, gene_bounds_by_chrom):
    """collect all intron boundaries from haec leafcutter phenotype bed files

    reads clusters_perind.counts.gz.qqnorm_chr*.bed.gz, maps introns to genes
    via gene bounds, returns {gene_id: np.array of sorted boundary positions}
    """
    bed_files = sorted(Path(intron_dir).glob("clusters_perind.counts.gz.qqnorm_chr*.bed.gz"))
    if not bed_files:
        bed_files = sorted(Path(intron_dir).glob("*.bed.gz"))
    print(f"intron bed files: {len(bed_files)}")

    gene_introns = defaultdict(set)
    n_introns = 0
    for path in tqdm(bed_files, desc="collecting introns"):
        df = pd.read_csv(path, sep="\t", usecols=[0, 1, 2], compression="gzip",
                         names=["chrom", "start", "end"], skiprows=1,
                         dtype={"chrom": str, "start": int, "end": int})
        df["chrom"] = df["chrom"].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
        for chrom, start, end in zip(df["chrom"], df["start"], df["end"]):
            gene_id, _ = get_intron_gene(chrom, start, end, gene_bounds_by_chrom)
            if gene_id:
                gene_introns[gene_id].add(start)
                gene_introns[gene_id].add(end)
                n_introns += 1

    intron_sites = {g: np.array(sorted(s)) for g, s in gene_introns.items()}
    print(f"introns mapped: {n_introns:,}")
    print(f"genes with intron boundaries: {len(intron_sites):,}")
    avg = np.mean([len(v) for v in intron_sites.values()])
    print(f"avg boundaries per gene: {avg:.1f}")
    return intron_sites


def recompute_splice_dist(df, intron_sites):
    """recompute splice_dist as distance to nearest intron boundary in same gene"""
    new_dists = np.full(len(df), np.inf)
    for gene_id, group_idx in df.groupby("gene_id").groups.items():
        sites = intron_sites.get(gene_id)
        if sites is None or len(sites) == 0:
            continue
        pos_arr = df.loc[group_idx, "pos"].values
        si = np.searchsorted(sites, pos_arr)
        left = np.where(si > 0, np.abs(pos_arr - sites[np.clip(si - 1, 0, len(sites) - 1)]), np.inf)
        right = np.where(si < len(sites), np.abs(pos_arr - sites[np.clip(si, 0, len(sites) - 1)]), np.inf)
        new_dists[group_idx] = np.minimum(left, right)
    df["splice_dist"] = new_dists
    return df


def load_sumstats(input_dir, gene_bounds_by_chrom, fm_pips):
    """load sqtl sumstats from parquet files"""
    parquet_files = sorted(Path(input_dir).glob("sQTL_chr*.cis_qtl_pairs.*.parquet"))
    all_rows = []
    for path in tqdm(parquet_files, desc="loading sumstats"):
        df = pl.read_parquet(path, columns=["phenotype_id", "variant_id", "pval_nominal"])
        # parse variant
        df = df.with_columns([
            pl.col("variant_id").str.split(":").list.get(0).alias("chrom"),
            pl.col("variant_id").str.split(":").list.get(1).cast(pl.Int64).alias("pos"),
            pl.col("variant_id").str.split(":").list.get(2).alias("ref"),
            pl.col("variant_id").str.split(":").list.get(3).alias("alt"),
        ])
        df = df.with_columns(
            pl.when(pl.col("chrom").str.starts_with("chr"))
            .then(pl.col("chrom"))
            .otherwise(pl.lit("chr") + pl.col("chrom"))
            .alias("chrom")
        )
        # parse intron
        df = df.with_columns([
            pl.col("phenotype_id").str.split(":").list.get(0).alias("intron_chrom_raw"),
            pl.col("phenotype_id").str.split(":").list.get(1).cast(pl.Int64, strict=False).alias("intron_start"),
            pl.col("phenotype_id").str.split(":").list.get(2).cast(pl.Int64, strict=False).alias("intron_end"),
        ])
        df = df.with_columns(
            pl.when(pl.col("intron_chrom_raw").str.starts_with("chr"))
            .then(pl.col("intron_chrom_raw"))
            .otherwise(pl.lit("chr") + pl.col("intron_chrom_raw"))
            .alias("intron_chrom")
        )
        # filter valid introns and snvs
        df = df.filter(pl.col("intron_start").is_not_null())
        df = df.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))
        # splice distance
        df = df.with_columns(
            pl.min_horizontal(
                (pl.col("pos") - pl.col("intron_start")).abs(),
                (pl.col("pos") - pl.col("intron_end")).abs()
            ).alias("splice_dist")
        )
        df = df.filter(pl.col("splice_dist") <= max_dist)
        # intron key
        df = df.with_columns(
            (pl.col("intron_chrom") + ":" + pl.col("intron_start").cast(pl.Utf8) + ":" +
             pl.col("intron_end").cast(pl.Utf8)).alias("intron_key")
        )
        df = df.select(["phenotype_id", "chrom", "pos", "ref", "alt", "pval_nominal",
                        "intron_chrom", "intron_start", "intron_end", "intron_key", "splice_dist"])
        all_rows.append(df)

    df = pl.concat(all_rows).to_pandas()
    print(f"sumstats: {len(df):,} rows after filters")

    # get gene
    gene_strand = [get_intron_gene(ic, int(ist), int(ie), gene_bounds_by_chrom)
                   for ic, ist, ie in zip(df["intron_chrom"], df["intron_start"], df["intron_end"])]
    df["gene_id"] = [gs[0] for gs in gene_strand]
    df["strand"] = [gs[1] for gs in gene_strand]
    df = df[df["gene_id"].notna()]
    print(f"  after gene filter: {len(df):,}")

    df["var_key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]
    df["pip"] = df["var_key"].map(fm_pips).fillna(0)

    return df


def build_tpm_bins(tpm_path):
    """build gene->bin mapping from haec salmon tpm"""
    tpm = pd.read_csv(tpm_path, sep="\t")
    tpm["gene_id"] = tpm.iloc[:, 0].str.split(".").str[0]
    tpm = tpm.drop_duplicates("gene_id")
    sample_cols = tpm.select_dtypes(include=[np.number]).columns.tolist()
    tpm["mean_tpm"] = tpm[sample_cols].mean(axis=1)
    vals = tpm["mean_tpm"].values
    log2 = np.where(vals > 0, np.log2(vals), -10)
    bins = np.floor(log2 / tpm_bin_size).astype(int)
    gene_to_bin = dict(zip(tpm["gene_id"], bins))
    print(f"tpm bins: {len(gene_to_bin):,} genes")
    return gene_to_bin


def run_matching(pos, neg, gene_to_bin, max_dist_diff=None, match_scheme="tiered"):
    """4-tier matching (same as leafcutter/txrevise), or hungarian on
    expression-bin + log distance (drops gene stratum).

    tier 1: same gene + same alleles (closest splice_dist)
    tier 2: same gene only
    tier 3: same expression bin + same alleles
    tier 4: same expression bin only
    tier 99 (hungarian): global one-to-one assignment within expression bin
    """
    pos = pos.copy()
    neg = neg.copy()

    if match_scheme == "hungarian":
        from scipy.optimize import linear_sum_assignment
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 99: 0}
        all_pairs = []
        pos["bin"] = pos["gene_id"].map(gene_to_bin).fillna(-1).astype(int)
        neg["bin"] = neg["gene_id"].map(gene_to_bin).fillna(-1).astype(int)
        # dedupe: one row per var_key with min splice_dist
        neg = (neg.sort_values("splice_dist")
                  .drop_duplicates("var_key", keep="first")
                  .reset_index(drop=True))
        pos = (pos.sort_values("splice_dist")
                  .drop_duplicates("var_key", keep="first")
                  .reset_index(drop=True))
        used = set()  # var_keys used
        for b, pos_in_bin in tqdm(pos.groupby("bin"), desc="hungarian matching"):
            neg_in_bin = neg[neg["bin"] == b]
            if len(neg_in_bin) == 0 or len(pos_in_bin) == 0:
                continue
            log_p = np.log10(pos_in_bin["splice_dist"].astype(float).values + 1.0)
            log_n = np.log10(neg_in_bin["splice_dist"].astype(float).values + 1.0)
            cost = np.abs(log_p[:, None] - log_n[None, :])
            if max_dist_diff is not None:
                p_sd = pos_in_bin["splice_dist"].astype(float).values
                n_sd = neg_in_bin["splice_dist"].astype(float).values
                cost = np.where(np.abs(p_sd[:, None] - n_sd[None, :]) > max_dist_diff, 1e9, cost)
            row_ind, col_ind = linear_sum_assignment(cost)
            pos_vals = pos_in_bin.reset_index(drop=True)
            neg_vals = neg_in_bin.reset_index(drop=True)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] >= 1e8:
                    continue
                n = neg_vals.iloc[ci]
                if n["var_key"] in used:
                    continue
                used.add(n["var_key"])
                p = pos_vals.iloc[ri]
                all_pairs.append({
                    "pos_var_key": p["var_key"],
                    "neg_var_key": n["var_key"],
                    "neg_var_key_ideal": n["var_key"],
                    "tier": 99,
                    "neg_splice_dist": n["splice_dist"],
                    "pos_pip": p["pip"],
                })
                tier_counts[99] += 1
        pairs_df = pd.DataFrame(all_pairs)
        print(f"matched {len(pairs_df):,} pairs via hungarian")
        return pairs_df, tier_counts

    all_pairs = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    # build indices
    idx_gene_allele = defaultdict(list)
    idx_gene = defaultdict(list)
    idx_bin_allele = defaultdict(list)
    idx_bin = defaultdict(list)
    neg_arr = neg[["gene_id", "ref", "alt", "splice_dist", "var_key"]].values
    for i, (gene, ref, alt, sd, vk) in enumerate(neg_arr):
        idx_gene_allele[(gene, ref, alt)].append(i)
        idx_gene[gene].append(i)
        if gene in gene_to_bin:
            b = gene_to_bin[gene]
            idx_bin_allele[(b, ref, alt)].append(i)
            idx_bin[b].append(i)

    used = set()

    if max_dist_diff is not None:
        print(f"max splice distance difference: {max_dist_diff}")

    for _, p in tqdm(pos.iterrows(), total=len(pos), desc="matching"):
        gene, ref, alt, p_sd = p["gene_id"], p["ref"], p["alt"], p["splice_dist"]
        best_i, tier = None, None

        # ideal match (ignoring used set)
        ideal_i = None
        for idx_map in [idx_gene_allele.get((gene, ref, alt), []),
                        idx_gene.get(gene, []),
                        idx_bin_allele.get((gene_to_bin.get(gene, None), ref, alt), []) if gene in gene_to_bin else [],
                        idx_bin.get(gene_to_bin.get(gene, None), []) if gene in gene_to_bin else []]:
            if idx_map:
                c = min(idx_map, key=lambda i: abs(neg_arr[i][3] - p_sd))
                if max_dist_diff is None or abs(neg_arr[c][3] - p_sd) <= max_dist_diff:
                    ideal_i = c
                    break

        # tier 1: gene + allele
        cands = [i for i in idx_gene_allele.get((gene, ref, alt), []) if i not in used]
        if cands:
            best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
            tier = 1
            if max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > max_dist_diff:
                best_i, tier = None, None

        # tier 2: gene only
        if best_i is None:
            cands = [i for i in idx_gene.get(gene, []) if i not in used]
            if cands:
                best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
                tier = 2
                if max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > max_dist_diff:
                    best_i, tier = None, None

        # tier 3: bin + allele
        if best_i is None and gene in gene_to_bin:
            b = gene_to_bin[gene]
            cands = [i for i in idx_bin_allele.get((b, ref, alt), []) if i not in used]
            if cands:
                best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
                tier = 3
                if max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > max_dist_diff:
                    best_i, tier = None, None

        # tier 4: bin only
        if best_i is None and gene in gene_to_bin:
            b = gene_to_bin[gene]
            cands = [i for i in idx_bin.get(b, []) if i not in used]
            if cands:
                best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
                tier = 4
                if max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > max_dist_diff:
                    best_i, tier = None, None

        if best_i is not None:
            used.add(best_i)
            tier_counts[tier] += 1
            all_pairs.append({
                "pos_var_key": p["var_key"],
                "neg_var_key": neg_arr[best_i][4],
                "neg_var_key_ideal": neg_arr[ideal_i][4] if ideal_i is not None else neg_arr[best_i][4],
                "tier": tier,
                "neg_splice_dist": neg_arr[best_i][3],
                "pos_pip": p["pip"],
            })

    pairs_df = pd.DataFrame(all_pairs)
    print(f"matched {len(pairs_df):,} pairs: " + ", ".join(f"t{t}={tier_counts[t]}" for t in [1, 2, 3, 4]))
    return pairs_df, tier_counts


# qc

def gen_cs_summary_haec(finemapping_path, out_path):
    """generate per-cs summary from haec finemapping file"""
    fm = pd.read_csv(finemapping_path, sep="\t")
    rows = []
    for (pheno, cs_num), g in fm.groupby(["phenotype", "credible_set_number"]):
        cs_id = f"{pheno}_L{cs_num}"
        pips = g["posterior_inclusion_probability"].values
        rows.append({
            "tissue": "HAEC",
            "cs_id": cs_id,
            "cs_size": len(g),
            "max_pip": pips.max(),
            "pip_sum": pips.sum(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(out):,} rows)")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir with sQTL_chr*.parquet files")
    ap.add_argument("--finemapping", required=True, help="finemapping tsv")
    ap.add_argument("--intron-dir", required=True,
                    help="dir with leafcutter phenotype bed files (clusters_perind.counts.gz.qqnorm_chr*.bed.gz)")
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--tpm", required=True, help="haec salmon tpm file")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--max-dist-diff", type=int, default=None,
                    help="max allowed splice distance difference between pos and neg (0 = exact match)")
    ap.add_argument("--pos-pip", type=float, default=0.9,
                    help="minimum PIP for positive variants (default: 0.9)")
    ap.add_argument("--match-scheme", choices=["tiered", "hungarian"], default="tiered",
                    help="matching algorithm: tiered (4-tier cascade, default) or "
                         "hungarian (bin-stratified, global one-to-one on log-distance)")
    args = ap.parse_args()

    print(f"pos pip threshold: {args.pos_pip}")

    qc_dir = Path(args.out_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # load gene bounds
    print("loading gtf...")
    gene_bounds = load_gene_bounds(args.gtf)
    gene_bounds_by_chrom = defaultdict(list)
    for gene_id, bounds in gene_bounds.items():
        if bounds["chrom"]:
            gene_bounds_by_chrom[bounds["chrom"]].append(
                (gene_id, bounds["start"], bounds["end"], bounds["strand"]))
    for chrom in gene_bounds_by_chrom:
        gene_bounds_by_chrom[chrom].sort(key=lambda x: x[1])
    print(f"genes: {len(gene_bounds):,}")

    # load finemapping pips
    print("loading finemapping...")
    fm = pd.read_csv(args.finemapping, sep="\t")
    fm_pips = fm.groupby("variant_id")["posterior_inclusion_probability"].max().to_dict()
    print(f"variants with pips: {len(fm_pips):,}")

    # collect all intron boundaries from leafcutter phenotype bed files
    print("loading intron boundaries...")
    intron_sites = load_intron_sites(args.intron_dir, gene_bounds_by_chrom)

    # load sumstats
    df = load_sumstats(args.input_dir, gene_bounds_by_chrom, fm_pips)

    # build tpm bins
    gene_to_bin = build_tpm_bins(args.tpm)

    # split pos/neg
    pos_keys = set(k for k, p in fm_pips.items() if p >= args.pos_pip)
    neg_keys = set(df[df["pip"] < neg_pip]["var_key"].unique()) - pos_keys

    # positives: lowest pval row per variant (splice_dist from associated intron)
    # sort by pip descending so high-pip variants get first pick of negatives
    pos = df[df["var_key"].isin(pos_keys)].sort_values("pval_nominal").drop_duplicates("var_key", keep="first").copy()
    pos = pos.sort_values("pip", ascending=False)
    # negatives: all rows in neg_keys
    neg = df[df["var_key"].isin(neg_keys)].drop_duplicates(["var_key", "intron_key"]).copy()

    # recompute negative splice_dist using all intron boundaries per gene
    # (same as leafcutter approach — distance to nearest detected intron boundary)
    n_before = len(neg)
    neg = neg.reset_index(drop=True)
    neg = recompute_splice_dist(neg, intron_sites)
    neg = neg[neg["splice_dist"] <= max_dist]
    print(f"negatives after intron-boundary splice_dist: {len(neg):,} (was {n_before:,})")

    print(f"positives: {len(pos):,}, negatives: {len(neg):,}")

    # shared qc
    sumstats_qc = pd.DataFrame([{
        "n_variants": df["var_key"].nunique(),
        "n_introns": df["intron_key"].nunique(),
        "n_genes": df["gene_id"].nunique()
    }])
    fm_high = fm[fm["posterior_inclusion_probability"] >= args.pos_pip]
    credset_qc = pd.DataFrame([{
        "n_variants": len(fm),
        "n_high_pip": len(fm_high),
        "n_unique_high_pip": fm_high["variant_id"].nunique()
    }])
    pip_qc = fm[["variant_id", "posterior_inclusion_probability"]].rename(
        columns={"variant_id": "var_key", "posterior_inclusion_probability": "pip"})

    # cs summary from finemapping file
    print("\ngenerating cs_summary")
    gen_cs_summary_haec(args.finemapping, qc_dir / "cs_summary.csv")

    # run matching
    print("matching")

    out = qc_dir

    pairs, tier_counts = run_matching(pos, neg, gene_to_bin,
                                      max_dist_diff=args.max_dist_diff,
                                      match_scheme=args.match_scheme)

    # save pairs
    pairs.to_csv(out / "pairs.csv", index=False)

    # save tier counts
    pd.DataFrame([{"tier": t, "count": c} for t, c in tier_counts.items()]).to_csv(
        out / "tiers.csv", index=False)

    # get matched variants
    matched_pos_keys = set(pairs["pos_var_key"])
    matched_neg_keys = set(pairs["neg_var_key"])

    # unique vcfs for scoring
    pos_unique = pos[pos["var_key"].isin(matched_pos_keys)]
    neg_unique = neg[neg["var_key"].isin(matched_neg_keys)].drop_duplicates("var_key")
    write_vcf(pos_unique, out / "pos.vcf.gz")
    write_vcf(neg_unique, out / "neg.vcf.gz")
    print(f"unique variants: {len(pos_unique):,} pos, {len(neg_unique):,} neg")

    # distances
    neg_matched_dist = pairs.groupby("neg_var_key")["neg_splice_dist"].min().reset_index()
    neg_matched_dist.columns = ["var_key", "splice_dist"]
    dists = pd.concat([
        pos_unique[["var_key", "splice_dist"]].assign(type="pos"),
        neg_matched_dist.assign(type="neg")
    ])
    dists.to_csv(out / "distances.csv", index=False)

    # qc
    sumstats_qc.to_csv(out / "sumstats_counts.csv", index=False)
    credset_qc.to_csv(out / "credset_stats.csv", index=False)
    pip_qc.to_csv(out / "pip_values.csv", index=False)

    print(f"output: {out}")
    print("\ndone")


if __name__ == "__main__":
    main()
