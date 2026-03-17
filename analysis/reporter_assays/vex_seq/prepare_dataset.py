#!/usr/bin/env python3
"""prepare vex-seq dataset for scoring, outputs h5 with one-hot seqs + metadata"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
default_data_dir = script_dir / "data"
default_fasta = script_dir / "hg19.fa"


def revcomp(seq):
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def one_hot(seq):
    seq = seq.upper()
    L = len(seq)
    arr = np.zeros((L, 4), dtype=np.float32)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq):
        if ch in mapping:
            arr[i, mapping[ch]] = 1.0
    return arr


def merge_splits(train, test, truth):
    key_cols = ["chrom", "pos", "ref", "alt"]

    def with_keys(df, chrom, pos, ref, alt):
        out = df.copy()
        out["chrom"] = out[chrom].astype(str)
        out["pos"] = pd.to_numeric(out[pos], errors="coerce").astype("Int64")
        out["ref"] = out[ref].astype(str)
        out["alt"] = out[alt].astype(str)
        return out

    train_k = with_keys(train, "seqnames", "hg19_variant_position", "reference", "variant")
    test_k = with_keys(test, "seqnames", "hg19_variant_position", "reference", "variant")
    truth_k = with_keys(truth, "chromosome", "hg19_variant_position", "reference", "variant")
    truth_k = truth_k[key_cols + ["HepG2_delta_psi"]].drop_duplicates(subset=key_cols)

    # fill test labels from truth table
    merged_test = test_k.merge(truth_k, on=key_cols, how="left", validate="one_to_one")
    missing = merged_test["HepG2_delta_psi"].isna().sum()
    assert missing == 0, f"{missing} test rows missing HepG2_delta_psi"

    train_clean = train_k.drop(columns=key_cols).assign(split="train")
    test_clean = merged_test.drop(columns=key_cols).assign(split="test")

    cols = sorted(set(train_clean.columns) | set(test_clean.columns))
    combined = pd.concat([
        train_clean.reindex(columns=cols),
        test_clean.reindex(columns=cols),
    ], ignore_index=True).drop_duplicates()

    assert not combined["HepG2_delta_psi"].isna().any(), "missing HepG2_delta_psi after merge"
    return combined


class WindowBuilder:

    def __init__(self, fasta, half):
        self.fasta = fasta
        self.half = half

    def get_window(self, chrom, center):
        chrom = str(chrom)
        n = self.fasta.get_reference_length(chrom)
        start1 = int(center) - self.half
        end1 = int(center) + self.half
        left_pad = max(0, 1 - start1)
        right_pad = max(0, end1 - n)
        start0 = max(0, start1 - 1)
        end0 = min(n, end1)
        seq = self.fasta.fetch(chrom, start0, end0).upper()
        if left_pad:
            seq = "N" * left_pad + seq
        if right_pad:
            seq = seq + "N" * right_pad
        return seq, start1, end1

    def extend_head(self, chrom, start1, need):
        if need <= 0:
            return ""
        start0 = max(0, (start1 - 1) - need)
        head = self.fasta.fetch(str(chrom), start0, start1 - 1).upper()
        if len(head) < need:
            head = "N" * (need - len(head)) + head
        return head

    def extend_tail(self, chrom, end1, need):
        if need <= 0:
            return ""
        n = self.fasta.get_reference_length(str(chrom))
        endx = min(n, end1 + need)
        tail = self.fasta.fetch(str(chrom), end1, endx).upper()
        if len(tail) < need:
            tail += "N" * (need - len(tail))
        return tail

    def adjust_alt_length(self, alt_seq, chrom, w_start1, w_end1, var_upstream):
        # indels change length, trim or extend to keep splice site centered
        expected_len = 2 * self.half + 1
        diff = len(alt_seq) - expected_len
        if diff == 0:
            return alt_seq

        if var_upstream:
            # variant is left of center, adjust on left side
            if diff > 0:
                # insertion: trim from left
                return alt_seq[diff:]
            else:
                # deletion: extend from left
                need = -diff
                return self.extend_head(chrom, w_start1, need) + alt_seq
        else:
            # variant is right of center, adjust on right side
            if diff > 0:
                # insertion: trim from right
                return alt_seq[:expected_len]
            else:
                # deletion: extend from right
                need = -diff
                return alt_seq + self.extend_tail(chrom, w_end1, need)

    def build_pair(self, chrom, center, var_pos, ref_allele, alt_allele):
        ref_seq, start1, end1 = self.get_window(chrom, center)
        ref_allele = ref_allele.upper()
        alt_allele = alt_allele.upper()
        idx0 = int(var_pos) - start1

        # variant in window?
        if not (0 <= idx0 < len(ref_seq) - len(ref_allele) + 1):
            return None, None, f"variant out of window at {chrom}:{var_pos}"

        # ref matches genome?
        genome_ref = ref_seq[idx0:idx0 + len(ref_allele)]
        if genome_ref != ref_allele:
            return None, None, f"ref mismatch at {chrom}:{var_pos}: expected {ref_allele}, got {genome_ref}"

        # apply variant
        alt_seq = ref_seq[:idx0] + alt_allele + ref_seq[idx0 + len(ref_allele):]

        var_upstream = int(var_pos) < center

        alt_seq = self.adjust_alt_length(alt_seq, chrom, start1, end1, var_upstream)

        expected_len = 2 * self.half + 1
        assert len(ref_seq) == expected_len
        assert len(alt_seq) == expected_len, f"alt_seq length {len(alt_seq)} != {expected_len}"
        return ref_seq, alt_seq, None


def build_sequences(df, builder):
    required = {"seqnames", "strand", "hg19_variant_position", "reference", "variant", "start", "end"}
    assert required <= set(df.columns), f"missing columns: {required - set(df.columns)}"

    seqs = {
        "exon_start_ref": [],
        "exon_start_alt": [],
        "exon_end_ref": [],
        "exon_end_alt": [],
    }
    meta = {
        "chrom": [],
        "pos": [],
        "ref": [],
        "alt": [],
        "strand": [],
        "exon_start": [],
        "exon_end": [],
        "delta_psi": [],
        "split": [],
    }
    skipped = []

    for row_idx, r in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="building sequences")):
        chrom = str(r.seqnames)
        strand = str(r.strand)
        pos = int(r.hg19_variant_position)
        ref = str(r.reference)
        alt = str(r.variant)

        if strand not in {"+", "-"}:
            skipped.append((row_idx, f"invalid strand: {strand}"))
            continue

        exon_start_1b = int(r.start)
        exon_end_1b = int(r.end)

        # build exon_start sequences
        ref_start, alt_start, err_start = builder.build_pair(chrom, exon_start_1b, pos, ref, alt)
        if err_start:
            skipped.append((row_idx, f"exon_start: {err_start}"))
            continue

        # build exon_end sequences
        ref_end, alt_end, err_end = builder.build_pair(chrom, exon_end_1b, pos, ref, alt)
        if err_end:
            skipped.append((row_idx, f"exon_end: {err_end}"))
            continue

        # revcomp minus strand
        if strand == "-":
            ref_start, alt_start = revcomp(ref_start), revcomp(alt_start)
            ref_end, alt_end = revcomp(ref_end), revcomp(alt_end)

        seqs["exon_start_ref"].append(ref_start)
        seqs["exon_start_alt"].append(alt_start)
        seqs["exon_end_ref"].append(ref_end)
        seqs["exon_end_alt"].append(alt_end)

        meta["chrom"].append(chrom)
        meta["pos"].append(pos)
        meta["ref"].append(ref)
        meta["alt"].append(alt)
        meta["strand"].append(strand)
        meta["exon_start"].append(int(r.start))
        meta["exon_end"].append(int(r.end))
        meta["delta_psi"].append(float(r.HepG2_delta_psi))
        meta["split"].append(str(r.split))

    return seqs, meta, skipped


def save_h5(seqs, meta, path, seq_len):
    n = len(meta["chrom"])

    print("one-hot encoding sequences...")
    encoded = {}
    for key, seq_list in tqdm(seqs.items(), desc="encoding"):
        arr = np.zeros((n, seq_len, 4), dtype=np.float32)
        for i, seq in enumerate(seq_list):
            arr[i] = one_hot(seq)
        encoded[key] = arr

    print(f"writing {path}")
    with h5py.File(path, "w") as f:
        # sequences
        seq_grp = f.create_group("seqs")
        for key, arr in encoded.items():
            seq_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

        # metadata
        meta_grp = f.create_group("meta")
        meta_grp.create_dataset("chrom", data=np.array(meta["chrom"], dtype="S24"))
        meta_grp.create_dataset("pos", data=np.array(meta["pos"], dtype=np.int64))
        meta_grp.create_dataset("ref", data=np.array(meta["ref"], dtype="S256"))
        meta_grp.create_dataset("alt", data=np.array(meta["alt"], dtype="S256"))
        meta_grp.create_dataset("strand", data=np.array(meta["strand"], dtype="S1"))
        meta_grp.create_dataset("exon_start", data=np.array(meta["exon_start"], dtype=np.int64))
        meta_grp.create_dataset("exon_end", data=np.array(meta["exon_end"], dtype=np.int64))
        meta_grp.create_dataset("delta_psi", data=np.array(meta["delta_psi"], dtype=np.float32))
        meta_grp.create_dataset("split", data=np.array(meta["split"], dtype="S8"))

        # attributes
        f.attrs["n_variants"] = n
        f.attrs["seq_len"] = seq_len
        f.attrs["description"] = "VexSeq dataset - HepG2 delta-PSI splice variants"

    print(f"wrote {path} ({n:,} variants, {seq_len}bp sequences)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir,
                    help="directory containing raw csv/tsv files")
    ap.add_argument("--fasta", type=Path, default=default_fasta,
                    help="path to hg19 reference fasta")
    ap.add_argument("--window-half", type=int, default=5000,
                    help="half-width of sequence window (total = 2*half+1)")
    ap.add_argument("--out-h5", type=Path, default=default_data_dir / "vex_seq.h5",
                    help="output h5 file path")
    args = ap.parse_args()

    seq_len = 2 * args.window_half + 1

    assert args.data_dir.is_dir(), f"data dir not found: {args.data_dir}"
    train_path = args.data_dir / "HepG2_delta_PSI_CAGI_training.csv"
    test_path = args.data_dir / "HepG2_delta_PSI_CAGI_testing.csv"
    truth_path = args.data_dir / "Vexseq_HepG2_delta_PSI_CAGI_test_true.tsv"
    for p in [train_path, test_path, truth_path, args.fasta]:
        assert p.exists(), f"missing: {p}"

    print("loading raw data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    truth = pd.read_csv(truth_path, sep="\t")
    combined = merge_splits(train, test, truth)
    print(f"  {len(combined):,} variants after merge")

    print("building sequences...")
    fasta = pysam.FastaFile(str(args.fasta))
    builder = WindowBuilder(fasta, args.window_half)
    seqs, meta, skipped = build_sequences(combined, builder)
    fasta.close()

    if skipped:
        print(f"\nwarning: skipped {len(skipped):,} variants:")
        reasons = {}
        for row_idx, reason in skipped:
            key = reason.split(":")[0] if ":" in reason else reason
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:,} - {reason}")

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    save_h5(seqs, meta, args.out_h5, seq_len)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
