#!/usr/bin/env python3
"""make leafcutter sqtl benchmark vcfs

ge-source negatives with intron-boundary splice_dist, 4-tier matching:
- tier 1: same gene + same alleles
- tier 2: same gene only
- tier 3: same expression bin + same alleles
- tier 4: same expression bin only

splice_dist for negatives = distance to nearest leafcutter intron boundary
in same gene (intron boundaries from zenodo phenotype metadata — all
leafcutter-detected introns, not just QTL-tested)

optional --max-dist-diff controls how closely the negative's splice distance
must match the positive's. 0 = exact match (borzoi-style), None = no limit.

usage:
    python src/make_leafcutter.py \
        --cs-dir $data_dir/leafcutter/raw \
        --pheno-dir $data_dir/leafcutter/phenotype_metadata \
        --ge-dir $data_dir/ge/sumstats \
        --gtf $data_dir/reference/gencode.v39.basic.annotation.gtf.gz \
        --tpm $data_dir/reference/GTEx_v8_median_tpm.gct.gz \
        --out-dir $data_dir/leafcutter \
        [--max-dist-diff 0]
"""
import argparse
import gzip
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# thresholds
neg_pip = 0.01
max_dist = 10_000
tpm_bin_size = 0.4

# tissue name mapping
tissue_map = {
    "adipose_subcutaneous": "Adipose - Subcutaneous",
    "adipose_visceral": "Adipose - Visceral (Omentum)",
    "adrenal_gland": "Adrenal Gland",
    "artery_aorta": "Artery - Aorta",
    "artery_coronary": "Artery - Coronary",
    "artery_tibial": "Artery - Tibial",
    "blood": "Whole Blood",
    "brain_amygdala": "Brain - Amygdala",
    "brain_anterior_cingulate_cortex": "Brain - Anterior cingulate cortex (BA24)",
    "brain_caudate": "Brain - Caudate (basal ganglia)",
    "brain_cerebellar_hemisphere": "Brain - Cerebellar Hemisphere",
    "brain_cerebellum": "Brain - Cerebellum",
    "brain_cortex": "Brain - Cortex",
    "brain_frontal_cortex": "Brain - Frontal Cortex (BA9)",
    "brain_hippocampus": "Brain - Hippocampus",
    "brain_hypothalamus": "Brain - Hypothalamus",
    "brain_nucleus_accumbens": "Brain - Nucleus accumbens (basal ganglia)",
    "brain_putamen": "Brain - Putamen (basal ganglia)",
    "brain_spinal_cord": "Brain - Spinal cord (cervical c-1)",
    "brain_substantia_nigra": "Brain - Substantia nigra",
    "breast": "Breast - Mammary Tissue",
    "colon_sigmoid": "Colon - Sigmoid",
    "colon_transverse": "Colon - Transverse",
    "esophagus_gej": "Esophagus - Gastroesophageal Junction",
    "esophagus_mucosa": "Esophagus - Mucosa",
    "esophagus_muscularis": "Esophagus - Muscularis",
    "fibroblast": "Cells - Cultured fibroblasts",
    "heart_atrial_appendage": "Heart - Atrial Appendage",
    "heart_left_ventricle": "Heart - Left Ventricle",
    "kidney_cortex": "Kidney - Cortex",
    "lcl": "Cells - EBV-transformed lymphocytes",
    "liver": "Liver",
    "lung": "Lung",
    "minor_salivary_gland": "Minor Salivary Gland",
    "muscle": "Muscle - Skeletal",
    "nerve_tibial": "Nerve - Tibial",
    "ovary": "Ovary",
    "pancreas": "Pancreas",
    "pituitary": "Pituitary",
    "prostate": "Prostate",
    "skin_not_sun_exposed": "Skin - Not Sun Exposed (Suprapubic)",
    "skin_sun_exposed": "Skin - Sun Exposed (Lower leg)",
    "small_intestine": "Small Intestine - Terminal Ileum",
    "spleen": "Spleen",
    "stomach": "Stomach",
    "testis": "Testis",
    "thyroid": "Thyroid",
    "uterus": "Uterus",
    "vagina": "Vagina",
}

# globals for multiprocessing (set by initializer)
_gene_strands = None
_tissue_bins = None
_neg_dir = None
_max_dist_diff = None


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


def extract_tissue(path):
    """extract tissue name from filename"""
    stem = path.stem.replace(".credible_sets.tsv", "").replace(".all.tsv", "")
    return stem.split("_", 1)[1].lower() if "_" in stem else stem.lower()


def parse_intron(mt_id):
    """parse intron coords from molecular_trait_id"""
    parts = mt_id.split(":")
    if len(parts) >= 3:
        try:
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            return chrom, int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None
    return None, None, None


def load_gene_strands(gtf_path):
    """load gene strand from gtf"""
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


def load_intron_sites(pheno_dir):
    """collect all intron boundaries from zenodo phenotype metadata

    returns {gene_id: np.array of sorted boundary positions}
    """
    pheno_files = sorted(Path(pheno_dir).glob("*.phenotype_metadata.tsv.gz"))
    print(f"phenotype metadata files: {len(pheno_files)}")
    gene_introns = defaultdict(set)
    for f in tqdm(pheno_files, desc="collecting introns"):
        df = pd.read_csv(f, sep="\t", usecols=["gene_id", "intron_start", "intron_end"],
                         compression="gzip")
        df["gene_id"] = df["gene_id"].str.split(".").str[0]
        for gid, ist, ien in zip(df["gene_id"], df["intron_start"], df["intron_end"]):
            gene_introns[gid].add(ist)
            gene_introns[gid].add(ien)

    intron_sites = {g: np.array(sorted(s)) for g, s in gene_introns.items()}
    print(f"unique introns: ~{sum(len(s) for s in gene_introns.values()) // 2:,}")
    print(f"genes with intron boundaries: {len(intron_sites):,}")
    avg = np.mean([len(v) for v in intron_sites.values()])
    print(f"avg boundaries per gene: {avg:.1f}")
    return intron_sites


def load_positives(cs_dir, gene_strands):
    """load positive variants from credible sets

    gene_id taken directly from credible set file (same annotation as GE/phenotype metadata)
    """
    cs_files = list(Path(cs_dir).glob("*.credible_sets.tsv.gz"))
    all_pos = []
    for path in tqdm(cs_files, desc="loading positives"):
        tissue = extract_tissue(path)
        df = pd.read_csv(path, sep="\t", compression="gzip")
        # parse variant
        parts = df["variant"].str.split("_", expand=True)
        df["chrom"] = parts[0].apply(lambda x: x if x.startswith("chr") else f"chr{x}")
        df["pos"] = parts[1].astype(int)
        df["ref"], df["alt"] = parts[2], parts[3]
        df["var_key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]
        # gene_id from credible set (strip version)
        df["gene_id"] = df["gene_id"].str.split(".").str[0]
        df = df[df["gene_id"].notna()]
        # strand from gtf
        df["strand"] = df["gene_id"].map(gene_strands)
        # parse intron
        intron_coords = df["molecular_trait_id"].apply(parse_intron)
        df["intron_chrom"] = intron_coords.apply(lambda x: x[0])
        df["intron_start"] = intron_coords.apply(lambda x: x[1])
        df["intron_end"] = intron_coords.apply(lambda x: x[2])
        df["intron_key"] = df["intron_chrom"] + ":" + df["intron_start"].astype(str) + ":" + df["intron_end"].astype(str)
        # filter snvs
        df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
        # splice distance (to associated intron)
        df["splice_dist"] = df.apply(lambda r: min(abs(r["pos"] - r["intron_start"]),
                                                    abs(r["pos"] - r["intron_end"])), axis=1)
        df = df[df["splice_dist"] <= max_dist]
        df["tissue"] = tissue
        all_pos.append(df[["var_key", "chrom", "pos", "ref", "alt", "gene_id", "intron_key",
                          "intron_start", "intron_end", "splice_dist", "strand", "pip", "tissue"]])
    pos = pd.concat(all_pos, ignore_index=True)
    print(f"positives: {len(pos):,} rows, {pos['var_key'].nunique():,} unique variants")
    return pos


def _make_process_neg_file(intron_sites, pos_pips):
    """create closure for processing ge neg files

    splice_dist = distance to nearest leafcutter intron boundary in same gene
    """
    pip_df = pl.DataFrame({"var_key": list(pos_pips.keys()), "pip": list(pos_pips.values())})
    genes_with_introns = set(intron_sites.keys())

    def process(path):
        tissue = extract_tissue(path)
        cols = ["chromosome", "position", "ref", "alt", "gene_id"]

        df = pl.read_csv(path, separator="\t", columns=cols,
                         schema_overrides={"chromosome": pl.Utf8})

        # snv filter
        df = df.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))

        # build chrom and var_key
        df = df.with_columns([
            ("chr" + pl.col("chromosome")).alias("chrom"),
            pl.col("position").alias("pos"),
        ])
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("pos").cast(pl.Utf8) + ":" +
             pl.col("ref") + ":" + pl.col("alt")).alias("var_key")
        )

        # strip gene_id version
        df = df.with_columns(pl.col("gene_id").str.split(".").list.first().alias("gene_id"))

        # filter to genes with intron boundaries
        df = df.filter(pl.col("gene_id").is_in(genes_with_introns))

        if len(df) == 0:
            return pd.DataFrame(columns=["var_key", "chrom", "pos", "ref", "alt",
                                         "gene_id", "splice_dist", "pip", "tissue"])

        # compute splice_dist per gene (distance to nearest intron boundary)
        df = df.with_row_index("idx")
        results = []
        for gene_id, group in df.group_by("gene_id"):
            gid = gene_id[0] if isinstance(gene_id, tuple) else gene_id
            sites = intron_sites.get(gid)
            if sites is None or len(sites) == 0:
                continue
            pos_arr = group["pos"].to_numpy()
            idx = np.searchsorted(sites, pos_arr)
            left = np.where(idx > 0,
                            np.abs(pos_arr - sites[np.clip(idx - 1, 0, len(sites) - 1)]),
                            np.inf)
            right = np.where(idx < len(sites),
                             np.abs(pos_arr - sites[np.clip(idx, 0, len(sites) - 1)]),
                             np.inf)
            dists = np.minimum(left, right)
            results.append(pl.DataFrame({"idx": group["idx"], "splice_dist": dists}))

        if not results:
            return pd.DataFrame(columns=["var_key", "chrom", "pos", "ref", "alt",
                                         "gene_id", "splice_dist", "pip", "tissue"])

        res = pl.concat(results)
        df = df.join(res, on="idx", how="inner").drop("idx")
        df = df.filter(pl.col("splice_dist") <= max_dist)

        # pip lookup
        df = df.join(pip_df, on="var_key", how="left").with_columns(pl.col("pip").fill_null(0))
        df = df.filter(pl.col("pip") < neg_pip)

        # add tissue and dedup
        df = df.with_columns(pl.lit(tissue).alias("tissue"))
        df = df.unique(subset=["var_key", "gene_id"])

        return df.select(["var_key", "chrom", "pos", "ref", "alt", "gene_id",
                          "splice_dist", "pip", "tissue"]).to_pandas()
    return process


def prepare_negatives(ge_dir, intron_sites, pos_pips, out_dir):
    """process ge negatives with intron-boundary splice_dist, save parquet per tissue"""
    ge_files = list(Path(ge_dir).glob("*.all.tsv.gz"))
    process_fn = _make_process_neg_file(intron_sites, pos_pips)

    tmp_dir = Path(out_dir) / ".tmp_ge"
    tmp_dir.mkdir(exist_ok=True)

    for path in tqdm(ge_files, desc="processing ge negatives"):
        tissue = extract_tissue(path)
        tmp_file = tmp_dir / f"{tissue}.parquet"
        if tmp_file.exists():
            print(f"  skip {path.name} (cached)", flush=True)
            continue
        print(f"  starting {path.name}", flush=True)
        df = process_fn(path)
        df.to_parquet(tmp_file, index=False)
        print(f"  done {path.name}: {len(df):,} rows", flush=True)
        del df

    return tmp_dir


def load_tissue_negatives(tmp_dir, tissue, gene_strands):
    """load negatives for a single tissue from parquet"""
    path = tmp_dir / f"{tissue}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["var_key", "chrom", "pos", "ref", "alt",
                                     "gene_id", "splice_dist", "pip", "tissue", "strand"])
    df = pd.read_parquet(path)
    df["strand"] = df["gene_id"].map(gene_strands)
    return df


def build_tpm_bins(tpm_file):
    """build gene->bin mapping per tissue"""
    tpm = pd.read_csv(tpm_file, sep="\t", skiprows=2)
    tpm["gene_id"] = tpm["Name"].str.split(".").str[0]
    tpm = tpm.drop_duplicates("gene_id")
    tissue_bins = {}
    for tissue, gtex_col in tissue_map.items():
        if gtex_col not in tpm.columns:
            continue
        vals = tpm[gtex_col].values
        log2 = np.where(vals > 0, np.log2(vals), -10)
        bins = np.floor(log2 / tpm_bin_size).astype(int)
        gene_to_bin = dict(zip(tpm["gene_id"], bins))
        bin_to_genes = defaultdict(set)
        for g, b in gene_to_bin.items():
            bin_to_genes[b].add(g)
        tissue_bins[tissue] = {"gene_to_bin": gene_to_bin, "bin_to_genes": dict(bin_to_genes)}
    print(f"tpm bins: {len(tissue_bins)} tissues")
    return tissue_bins


def _init_worker(gene_strands, tissue_bins, neg_dir, max_dist_diff, match_scheme="tiered"):
    """initialize worker process with shared data"""
    global _gene_strands, _tissue_bins, _neg_dir, _max_dist_diff, _match_scheme
    _gene_strands = gene_strands
    _tissue_bins = tissue_bins
    _neg_dir = neg_dir
    _max_dist_diff = max_dist_diff
    _match_scheme = match_scheme


def _match_tissue_hungarian(args):
    """hungarian match within expression bin: drops gene stratum, solves
    global one-to-one assignment minimizing sum of |log10(p+1)-log10(n+1)|
    """
    from scipy.optimize import linear_sum_assignment
    tissue, t_pos_data = args
    t_pos = pd.DataFrame(t_pos_data)

    t_neg = load_tissue_negatives(_neg_dir, tissue, _gene_strands)
    if len(t_neg) == 0:
        return None
    t_neg = t_neg[~t_neg["var_key"].isin(set(t_pos["var_key"]))]
    if len(t_neg) == 0:
        return None

    gene_to_bin = _tissue_bins.get(tissue, {}).get("gene_to_bin", {})

    t_pos = t_pos.copy()
    t_neg = t_neg.copy()
    t_pos["bin"] = t_pos["gene_id"].map(gene_to_bin).fillna(-1).astype(int)
    t_neg["bin"] = t_neg["gene_id"].map(gene_to_bin).fillna(-1).astype(int)
    # dedupe: one row per var_key (min splice_dist)
    t_neg = (t_neg.sort_values("splice_dist")
                   .drop_duplicates("var_key", keep="first")
                   .reset_index(drop=True))
    t_pos = (t_pos.sort_values("splice_dist")
                   .drop_duplicates("var_key", keep="first")
                   .reset_index(drop=True))

    used = set()  # var_keys used
    pairs = []
    for b, pos_in_bin in t_pos.groupby("bin"):
        neg_in_bin = t_neg[t_neg["bin"] == b]
        if len(neg_in_bin) == 0 or len(pos_in_bin) == 0:
            continue
        log_p = np.log10(pos_in_bin["splice_dist"].astype(float).values + 1.0)
        log_n = np.log10(neg_in_bin["splice_dist"].astype(float).values + 1.0)
        cost = np.abs(log_p[:, None] - log_n[None, :])
        if _max_dist_diff is not None:
            p_sd = pos_in_bin["splice_dist"].astype(float).values
            n_sd = neg_in_bin["splice_dist"].astype(float).values
            cost = np.where(np.abs(p_sd[:, None] - n_sd[None, :]) > _max_dist_diff, 1e9, cost)
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
            pairs.append({
                "tissue": tissue,
                "pos_var_key": p["var_key"],
                "neg_var_key": n["var_key"],
                "neg_var_key_ideal": n["var_key"],
                "tier": 99,
                "pos_pip": p["pip"],
            })

    matched_neg = t_neg[t_neg["var_key"].isin(used)]
    return {
        "pairs": pairs,
        "stats": {
            "tissue": tissue,
            "n_pos": len(t_pos),
            "n_matched": len(used),
            **{f"tier_{t}": 0 for t in [1, 2, 3, 4]},
            "tier_99_hungarian": len(used),
        },
        "tiers": {1: 0, 2: 0, 3: 0, 4: 0, 99: len(used)},
        "neg_matched": matched_neg.to_dict("records"),
    }


def _match_tissue(args):
    """match a single tissue - 4-tier ge matching"""
    tissue, t_pos_data = args
    t_pos = pd.DataFrame(t_pos_data)

    t_neg = load_tissue_negatives(_neg_dir, tissue, _gene_strands)
    if len(t_neg) == 0:
        return None

    # exclude positives from negatives
    t_neg = t_neg[~t_neg["var_key"].isin(set(t_pos["var_key"]))]
    if len(t_neg) == 0:
        return None

    gene_to_bin = _tissue_bins.get(tissue, {}).get("gene_to_bin", {})

    # build indices
    idx_gene_allele = defaultdict(list)
    idx_gene = defaultdict(list)
    idx_bin_allele = defaultdict(list)
    idx_bin = defaultdict(list)
    neg_arr = t_neg[["gene_id", "ref", "alt", "splice_dist", "var_key"]].values
    for i, (gene, ref, alt, sd, vk) in enumerate(neg_arr):
        idx_gene_allele[(gene, ref, alt)].append(i)
        idx_gene[gene].append(i)
        if gene in gene_to_bin:
            b = gene_to_bin[gene]
            idx_bin_allele[(b, ref, alt)].append(i)
            idx_bin[b].append(i)

    used = set()
    tissue_tiers = {1: 0, 2: 0, 3: 0, 4: 0}
    pairs = []

    for _, p in t_pos.iterrows():
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
                if _max_dist_diff is None or abs(neg_arr[c][3] - p_sd) <= _max_dist_diff:
                    ideal_i = c
                    break

        # tier 1: gene + allele
        cands = [i for i in idx_gene_allele.get((gene, ref, alt), []) if i not in used]
        if cands:
            best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
            tier = 1
            if _max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > _max_dist_diff:
                best_i, tier = None, None

        # tier 2: gene only
        if best_i is None:
            cands = [i for i in idx_gene.get(gene, []) if i not in used]
            if cands:
                best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
                tier = 2
                if _max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > _max_dist_diff:
                    best_i, tier = None, None

        # tier 3: bin + allele
        if best_i is None and gene in gene_to_bin:
            b = gene_to_bin[gene]
            cands = [i for i in idx_bin_allele.get((b, ref, alt), []) if i not in used]
            if cands:
                best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
                tier = 3
                if _max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > _max_dist_diff:
                    best_i, tier = None, None

        # tier 4: bin only
        if best_i is None and gene in gene_to_bin:
            b = gene_to_bin[gene]
            cands = [i for i in idx_bin.get(b, []) if i not in used]
            if cands:
                best_i = min(cands, key=lambda i: abs(neg_arr[i][3] - p_sd))
                tier = 4
                if _max_dist_diff is not None and abs(neg_arr[best_i][3] - p_sd) > _max_dist_diff:
                    best_i, tier = None, None

        if best_i is not None:
            used.add(best_i)
            tissue_tiers[tier] += 1
            pairs.append({
                "tissue": tissue,
                "pos_var_key": p["var_key"],
                "neg_var_key": neg_arr[best_i][4],
                "neg_var_key_ideal": neg_arr[ideal_i][4] if ideal_i is not None else neg_arr[best_i][4],
                "tier": tier,
                "pos_pip": p["pip"],
            })

    # collect matched negative rows
    matched_keys = {neg_arr[i][4] for i in used}
    matched_neg = t_neg[t_neg["var_key"].isin(matched_keys)]

    return {
        "pairs": pairs,
        "stats": {
            "tissue": tissue,
            "n_pos": len(t_pos),
            "n_matched": len(used),
            **{f"tier_{t}": tissue_tiers[t] for t in [1, 2, 3, 4]}
        },
        "tiers": tissue_tiers,
        "neg_matched": matched_neg.to_dict("records")
    }


def run_matching(pos, neg_dir, tissue_bins, gene_strands, n_workers=8, max_dist_diff=None, pos_pip=0.9):
    """4-tier ge matching, loading negatives tissue-by-tissue in parallel"""
    pos_high = pos[pos["pip"] >= pos_pip].copy()
    # dedup to one row per var_key (highest pip)
    pos_high = pos_high.sort_values("pip", ascending=False).drop_duplicates("var_key", keep="first")
    # track tissues per positive (only tissues where pip >= threshold)
    pos_keys = set(pos_high["var_key"])
    pos_tissues = pos[(pos["var_key"].isin(pos_keys)) & (pos["pip"] >= pos_pip)].groupby("var_key")["tissue"].apply(set).to_dict()

    all_tissues = set()
    for ts in pos_tissues.values():
        all_tissues.update(ts)

    args_list = []
    for tissue in sorted(all_tissues):
        t_pos_keys = {k for k, ts in pos_tissues.items() if tissue in ts}
        t_pos = pos_high[pos_high["var_key"].isin(t_pos_keys)]
        args_list.append((tissue, t_pos.to_dict("records")))

    if max_dist_diff is not None:
        print(f"max splice distance difference: {max_dist_diff}")
    scheme = getattr(run_matching, "_match_scheme", "tiered")
    print(f"matching {len(args_list)} tissues with {n_workers} workers (scheme: {scheme})...")
    worker_fn = _match_tissue_hungarian if scheme == "hungarian" else _match_tissue
    with Pool(n_workers, initializer=_init_worker,
              initargs=(gene_strands, tissue_bins, neg_dir, max_dist_diff, scheme)) as pool:
        results = list(tqdm(pool.imap(worker_fn, args_list),
                            total=len(args_list), desc="matching"))

    # aggregate
    all_pairs = []
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 99: 0}
    tissue_stats = []
    matched_neg_rows = []

    for res in results:
        if res is None:
            continue
        all_pairs.extend(res["pairs"])
        for t in [1, 2, 3, 4, 99]:
            tier_counts[t] += res["tiers"].get(t, 0)
        tissue_stats.append(res["stats"])
        if res["neg_matched"]:
            matched_neg_rows.append(pd.DataFrame(res["neg_matched"]))

    pairs_df = pd.DataFrame(all_pairs)
    stats_df = pd.DataFrame(tissue_stats)
    neg_matched = pd.concat(matched_neg_rows, ignore_index=True) if matched_neg_rows else pd.DataFrame()
    if scheme == "hungarian":
        print(f"matched {len(pairs_df):,} pairs via hungarian (n={tier_counts[99]:,})")
    else:
        print(f"matched {len(pairs_df):,} pairs: " +
              ", ".join(f"t{t}={tier_counts[t]}" for t in [1, 2, 3, 4]))
    return pairs_df, stats_df, tier_counts, pos_high, neg_matched


def gen_cs_summary(raw_dir, out_path):
    """generate per-cs summary from raw credible set files"""
    rows = []
    files = sorted(raw_dir.glob("*.credible_sets.tsv.gz"))
    for f in tqdm(files, desc="loading cs files"):
        tissue = f.stem.replace(".credible_sets.tsv", "").split("_", 1)[1]
        df = pd.read_csv(f, sep="\t", usecols=["cs_id", "cs_size", "pip"])
        for cs_id, g in df.groupby("cs_id"):
            rows.append({
                "tissue": tissue,
                "cs_id": cs_id,
                "cs_size": g["cs_size"].iloc[0],
                "max_pip": g["pip"].max(),
                "pip_sum": g["pip"].sum(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(out):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs-dir", required=True, help="leafcutter credible sets")
    ap.add_argument("--pheno-dir", required=True, help="phenotype metadata (zenodo intron lists)")
    ap.add_argument("--ge-dir", required=True, help="gene expression sumstats (neg source)")
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--tpm", required=True)
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--max-dist-diff", type=int, default=None,
                    help="max allowed splice distance difference between pos and neg (0 = exact match)")
    ap.add_argument("--pos-pip", type=float, default=0.9,
                    help="minimum PIP for positive variants (default: 0.9)")
    ap.add_argument("--match-scheme", choices=["tiered", "hungarian"], default="tiered",
                    help="matching algorithm: tiered (4-tier cascade, default) or "
                         "hungarian (bin-stratified, global one-to-one on log-distance)")
    args = ap.parse_args()
    run_matching._match_scheme = args.match_scheme

    print(f"pos pip threshold: {args.pos_pip}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tissues").mkdir(exist_ok=True)

    # load gene strands
    print("loading gtf...")
    gene_strands = load_gene_strands(args.gtf)
    print(f"genes with strand: {len(gene_strands):,}")

    # collect intron boundaries from phenotype metadata (all detected introns)
    intron_sites = load_intron_sites(args.pheno_dir)

    # load positives (gene_id from credible set, strand from gtf)
    pos = load_positives(args.cs_dir, gene_strands)

    # pip lookup for negative filtering
    pos_pips = pos.groupby("var_key")["pip"].max().to_dict()

    # prepare ge negatives (splice_dist to nearest intron boundary in gene)
    neg_dir = prepare_negatives(args.ge_dir, intron_sites, pos_pips, out)

    # tpm bins
    tissue_bins = build_tpm_bins(args.tpm)

    # run matching
    print("\nge matching")
    pairs, tissue_stats, tier_counts, pos_high, neg_matched = run_matching(
        pos, neg_dir, tissue_bins, gene_strands,
        max_dist_diff=args.max_dist_diff, pos_pip=args.pos_pip
    )

    # save pairs
    pairs.to_csv(out / "pairs.csv", index=False)

    # save tissue stats
    tissue_stats.to_csv(out / "tissue_stats.csv", index=False)

    # save tier counts
    pd.DataFrame([{"tier": t, "count": c} for t, c in tier_counts.items()]).to_csv(
        out / "tiers.csv", index=False)

    # get matched variants
    matched_pos_keys = set(pairs["pos_var_key"])
    matched_neg_keys = set(pairs["neg_var_key"])

    # unique vcfs for scoring
    pos_unique = pos_high[pos_high["var_key"].isin(matched_pos_keys)].drop_duplicates("var_key")
    neg_unique = neg_matched.drop_duplicates("var_key")
    write_vcf(pos_unique, out / "pos.vcf.gz")
    write_vcf(neg_unique, out / "neg.vcf.gz")
    print(f"unique variants: {len(pos_unique):,} pos, {len(neg_unique):,} neg")

    # per-tissue vcfs
    for tissue in pairs["tissue"].unique():
        t_pairs = pairs[pairs["tissue"] == tissue]
        t_pos = pos_high[pos_high["var_key"].isin(t_pairs["pos_var_key"])]
        t_neg = neg_matched[neg_matched["var_key"].isin(t_pairs["neg_var_key"])]
        write_vcf(t_pos, out / "tissues" / f"{tissue}_pos.vcf.gz")
        write_vcf(t_neg, out / "tissues" / f"{tissue}_neg.vcf.gz")

    # distances
    dists = pd.concat([
        pos_unique[["var_key", "splice_dist"]].assign(type="pos"),
        neg_unique[["var_key", "splice_dist"]].assign(type="neg")
    ])
    dists.to_csv(out / "distances.csv", index=False)

    # qc: credset stats
    cs_stats = pos.groupby("tissue").agg(
        n_rows=("var_key", "count"),
        n_variants=("var_key", "nunique"),
        n_genes=("gene_id", "nunique"),
        n_high_pip=("pip", lambda x: (x >= args.pos_pip).sum())
    ).reset_index()
    cs_stats.to_csv(out / "credset_stats.csv", index=False)
    print(f"saved credset_stats: {len(cs_stats)} tissues")

    # qc: pip values
    pos[["var_key", "pip", "tissue"]].to_csv(out / "pip_values.csv", index=False)
    print(f"saved pip_values: {len(pos):,} rows")

    # qc: cs summary
    print("\ngenerating cs_summary")
    gen_cs_summary(Path(args.cs_dir), out / "cs_summary.csv")

    print(f"\noutput: {out}")
    print("done")


if __name__ == "__main__":
    main()
