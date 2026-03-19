"""common utilities for sqtl benchmark"""
import gzip
import numpy as np
import pandas as pd
import pysam


from collections import defaultdict

nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3}
comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def load_splice_sites(gtf_path):
    """load exon boundaries as splice sites"""
    sites = defaultdict(set)
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if cols[2] != "exon":
                continue
            sites[cols[0]].add(int(cols[3]))
            sites[cols[0]].add(int(cols[4]))
    return {c: np.array(sorted(s)) for c, s in sites.items()}


def get_splice_dist(chrom, pos, splice_sites):
    """distance to nearest splice site"""
    if chrom not in splice_sites:
        return np.inf
    sites = splice_sites[chrom]
    idx = np.searchsorted(sites, pos)
    dists = []
    if idx > 0:
        dists.append(abs(pos - sites[idx - 1]))
    if idx < len(sites):
        dists.append(abs(pos - sites[idx]))
    return min(dists) if dists else np.inf


def load_vcf(path):
    """load vcf to dataframe"""
    records = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            chrom, pos, vid, ref, alt = cols[0], int(cols[1]), cols[2], cols[3], cols[4]
            info = cols[7] if len(cols) > 7 else ""

            # parse info field
            info_dict = {}
            for field in info.split(";"):
                if "=" in field:
                    k, v = field.split("=", 1)
                    info_dict[k] = v

            records.append({
                "chrom": chrom,
                "pos": pos,
                "var_id": vid,
                "ref": ref,
                "alt": alt,
                "var_key": f"{chrom}:{pos}:{ref}:{alt}",
                **info_dict,
            })
    return pd.DataFrame(records)


def write_vcf(df, path, info_cols=None):
    """write dataframe to vcf"""
    if info_cols is None:
        info_cols = ["MT", "SD", "PI", "LABEL"]

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=MT,Number=1,Type=String,Description="Molecular trait id">\n')
        f.write('##INFO=<ID=SD,Number=1,Type=Integer,Description="Splice distance">\n')
        f.write('##INFO=<ID=PI,Number=1,Type=String,Description="Positive SNP id">\n')
        f.write('##INFO=<ID=LABEL,Number=1,Type=Integer,Description="1=positive, 0=negative">\n')
        f.write('##INFO=<ID=TISSUE,Number=1,Type=String,Description="Tissue name">\n')
        f.write('##INFO=<ID=GENE,Number=1,Type=String,Description="Gene id">\n')
        f.write('##INFO=<ID=PIP,Number=1,Type=Float,Description="PIP">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for _, row in df.iterrows():
            chrom = row["chrom"]
            pos = int(row["pos"])
            vid = row.get("var_id", f"{chrom}_{pos}_{row['ref']}_{row['alt']}")
            ref = row["ref"]
            alt = row["alt"]

            # build info
            info_parts = []
            for col in info_cols:
                if col in row and pd.notna(row[col]):
                    info_parts.append(f"{col}={row[col]}")
            info = ";".join(info_parts) if info_parts else "."

            f.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t{info}\n")


def extract_sequences(vcf_df, fasta_path, seq_len=20001):
    """extract ref and alt sequences for all variants

    if st column exists, reverse complements - strand sequences so all
    sequences are in transcript direction (5' to 3')

    returns ref_seqs, alt_seqs, strands
    """
    from tqdm import tqdm
    fasta = pysam.FastaFile(fasta_path)
    half = seq_len // 2

    has_strand = "ST" in vcf_df.columns

    ref_seqs, alt_seqs, strands = [], [], []
    for _, row in tqdm(vcf_df.iterrows(), total=len(vcf_df), desc="extracting"):
        chrom, pos = row["chrom"], int(row["pos"])
        ref, alt = row["ref"], row["alt"]
        strand = row.get("ST", "+") if has_strand else "+"

        start = pos - half - 1
        end = pos + half
        seq = fasta.fetch(chrom, start, end).upper()

        if len(seq) != seq_len:
            raise ValueError(f"seq length {len(seq)} != {seq_len} at {chrom}:{pos}")
        center = half
        if seq[center] != ref.upper():
            raise ValueError(f"ref mismatch at {chrom}:{pos}: expected {ref}, got {seq[center]}")

        alt_seq = seq[:center] + alt.upper() + seq[center + 1:]

        # reverse complement for - strand so model sees transcript direction
        if strand == "-":
            seq = revcomp(seq)
            alt_seq = revcomp(alt_seq)

        ref_seqs.append(seq)
        alt_seqs.append(alt_seq)
        strands.append(strand)

    fasta.close()
    return ref_seqs, alt_seqs, strands


def revcomp(seq):
    """reverse complement"""
    return "".join(comp.get(b, "N") for b in seq[::-1])


def onehot(seq):
    """one-hot encode dna sequence"""
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nuc in enumerate(seq.upper()):
        if nuc in nuc_map:
            oh[i, nuc_map[nuc]] = 1.0
    return oh


def batch_onehot(seqs):
    """one-hot encode batch of sequences"""
    n = len(seqs)
    seq_len = len(seqs[0])
    oh = np.zeros((n, seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, nuc in enumerate(seq.upper()):
            if nuc in nuc_map:
                oh[i, j, nuc_map[nuc]] = 1.0
    return oh
