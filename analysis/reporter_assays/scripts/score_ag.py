#!/usr/bin/env python3
"""score reporter assay variants with AlphaGenome (all_folds, 16k context).

handles three input h5 schemas (auto-detected from /meta keys):

  vex_seq, mfass:
    /meta has chrom, pos, ref, alt, strand, exon_start, exon_end (1-based genomic)

  compass minigene + genome:
    /meta has chrom, strand, exon_start_hg38, exon_end_hg38, and variant_hg38
    formatted "chrN:pos:ref>alt" — we parse pos/ref/alt from variant_hg38.

  opensplice minigene + genome:
    /meta has variant_id (e.g. "DTX2_e6_A102C") and ensembl_exon_id only.
    we join open_splice_master.tsv.gz on variant_id to get CHROM, POS, REF, ALT,
    and join the gencode v46 GTF feather on ensembl_exon_id to get
    exon Start, End, Strand.

shape of the SS / SSU / SJ output heads is auto-detected from the first scoreable
variant (warmup); no hardcoded track counts. SS values stored at the canonical
exon_start and exon_end positions with AG's documented -1 splice-site peak shift:

  + strand: idx = boundary_pos - interval.start - 2
  - strand: idx = interval.end - boundary_pos - 1

uses dna_model.create_from_huggingface("all_folds"), ontology_terms=None.

output schema:
  scores/ss_<site>_<allele>   (n, n_ss_channels)
  scores/ssu_<site>_<allele>  (n, n_ssu_tracks)
  scores/sj_<site>_<allele>   (n, n_sj_tracks)     sum of SJ values across
                                                    junctions touching site
  site ∈ {exon_start, exon_end}, allele ∈ {ref, alt}
  meta/<copied from input>
  attrs: model, window_len, n_ss_channels, n_ssu_tracks, n_sj_tracks, n_variants

usage:
    python score_ag.py --input vex_seq/data/vex_seq.h5 \\
                       --output vex_seq/data/vex_seq_ag.h5
"""
import argparse
import gc as gc_mod
import os
import re
import signal
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test" / "src"))
from _ag_common import (  # noqa: E402
    make_organism_settings, load_model, build_interval, site_idx,
    REQUESTED_OUTPUTS, dna_model, genome,
)


_term_received = False
def _sigterm(signum, frame):
    global _term_received
    _term_received = True
    print(f"received signal {signum}; exiting at next variant boundary", flush=True)
signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


SITES = ("exon_start", "exon_end")
ALLELES = ("ref", "alt")
HEADS = ("ss", "ssu", "sj")


_OS_MASTER_CACHE = {"df": None}
_GENCODE_EXON_CACHE = {"df": None}


def _to_str(arr):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def _ensure_chr(c):
    return c if c.startswith("chr") else f"chr{c}"


_VARIANT_HG38_RE = re.compile(r"^(chr[\w]+):(\d+):([ACGTN-]+)>([ACGTN-]+)$")


def _h5_to_variants_vex_mfass(meta):
    chrom = [_ensure_chr(c) for c in _to_str(meta["chrom"])]
    pos = np.asarray(meta["pos"], dtype=np.int64)
    ref = _to_str(meta["ref"])
    alt = _to_str(meta["alt"])
    strand = _to_str(meta["strand"])
    exon_start = np.asarray(meta["exon_start"], dtype=np.int64)
    exon_end = np.asarray(meta["exon_end"], dtype=np.int64)
    return list(zip(chrom, pos.tolist(), ref, alt, strand, exon_start.tolist(), exon_end.tolist()))


def _h5_to_variants_compass(meta):
    chrom = [_ensure_chr(c) for c in _to_str(meta["chrom"])]
    strand = _to_str(meta["strand"])
    exon_start = np.asarray(meta["exon_start_hg38"], dtype=np.int64).tolist()
    exon_end = np.asarray(meta["exon_end_hg38"], dtype=np.int64).tolist()
    variant_hg38 = _to_str(meta["variant_hg38"])
    out = []
    for i, vh in enumerate(variant_hg38):
        m = _VARIANT_HG38_RE.match(vh)
        if not m:
            out.append(None)
            continue
        _ch, pos_str, r, a = m.groups()
        out.append((chrom[i], int(pos_str), r, a, strand[i], exon_start[i], exon_end[i]))
    return out


def _load_opensplice_master():
    if _OS_MASTER_CACHE["df"] is not None:
        return _OS_MASTER_CACHE["df"]
    candidates = [
        Path("opensplice/data/open_splice_master.tsv.gz"),
        Path(__file__).parent.parent / "opensplice" / "data" / "open_splice_master.tsv.gz",
    ]
    master_path = next((p for p in candidates if p.exists()), None)
    if master_path is None:
        raise FileNotFoundError(
            "open_splice_master.tsv.gz not found in expected locations; needed for opensplice scoring"
        )
    print(f"loading opensplice master tsv ({master_path.stat().st_size/1e6:.0f} MB)...", flush=True)
    df = pd.read_csv(master_path, sep="\t",
                     usecols=["variant_id", "CHROM", "POS", "REF", "ALT"],
                     low_memory=False)
    df = df.drop_duplicates(subset=["variant_id"], keep="first")
    df = df.dropna(subset=["CHROM", "POS", "REF", "ALT"])
    df["CHROM"] = df["CHROM"].astype(str).apply(_ensure_chr)
    # POS may be "<start>_<end>del" for deletion rows; strip the suffix and keep start
    df["POS"] = (df["POS"].astype(str)
                          .str.extract(r"^(\d+)", expand=False)
                          .astype("Int64"))
    df = df.dropna(subset=["POS"])
    df["POS"] = df["POS"].astype(np.int64)
    print(f"  master labels: {len(df):,} unique variants with genomic coords", flush=True)
    _OS_MASTER_CACHE["df"] = df.set_index("variant_id")
    return _OS_MASTER_CACHE["df"]


def _load_gencode_exons():
    if _GENCODE_EXON_CACHE["df"] is not None:
        return _GENCODE_EXON_CACHE["df"]
    ag_data_dir = os.environ.get("AG_DATA_DIR")
    if not ag_data_dir:
        raise RuntimeError("AG_DATA_DIR not set; gencode feather needed for opensplice scoring")
    feather = Path(ag_data_dir) / "gencode.v46.annotation.gtf.gz.feather"
    print(f"loading gencode exons from {feather}...", flush=True)
    import pyarrow.feather as ft
    tbl = ft.read_table(feather, columns=["Feature", "Chromosome", "Start", "End", "Strand", "exon_id"])
    df = tbl.to_pandas()
    df = df[df["Feature"] == "exon"].copy()
    df["exon_id_base"] = df["exon_id"].astype(str).str.split(".").str[0]
    df = df.drop_duplicates(subset=["exon_id_base"], keep="first")
    df = df[["exon_id_base", "Chromosome", "Start", "End", "Strand"]].set_index("exon_id_base")
    df["Chromosome"] = df["Chromosome"].astype(str).apply(_ensure_chr)
    df["Start"] = df["Start"].astype(np.int64)
    df["End"] = df["End"].astype(np.int64)
    print(f"  {len(df):,} unique exon_id (base, no version)", flush=True)
    _GENCODE_EXON_CACHE["df"] = df
    return df


def _h5_to_variants_opensplice(meta):
    master = _load_opensplice_master()
    exons = _load_gencode_exons()
    variant_id = _to_str(meta["variant_id"])
    ensembl_exon_id = _to_str(meta["ensembl_exon_id"])
    out = []
    n_no_master = n_no_exon = 0
    for vid, eid in zip(variant_id, ensembl_exon_id):
        if vid not in master.index:
            n_no_master += 1
            out.append(None)
            continue
        eid_base = eid.split(".")[0]
        if eid_base not in exons.index:
            n_no_exon += 1
            out.append(None)
            continue
        m = master.loc[vid]
        e = exons.loc[eid_base]
        out.append((str(m.CHROM), int(m.POS), str(m.REF), str(m.ALT),
                    str(e.Strand), int(e.Start), int(e.End)))
    print(f"  opensplice join: {sum(1 for x in out if x is not None):,}/{len(out):,} resolved "
          f"({n_no_master:,} missing in master, {n_no_exon:,} missing exon)", flush=True)
    return out


def detect_mode(h5_path):
    """read fh.attrs['mode'] (set by compass + opensplice prep notebooks); default 'genome'."""
    with h5py.File(h5_path, "r") as f:
        mode = f.attrs.get("mode")
    if isinstance(mode, bytes):
        mode = mode.decode()
    return (mode or "genome")


def h5_to_variants(h5_path):
    """genome-mode: detect /meta schema and dispatch to the right parser."""
    with h5py.File(h5_path, "r") as f:
        build = f["meta"].attrs.get("genome_build")
        if build is None:
            sys.stderr.write(
                f"WARNING: {h5_path}: /meta has no genome_build attribute; assuming hg38 (AG's reference). "
                f"Run fix_meta_hg38.py if this is wrong.\n"
            )
        elif (build.decode() if isinstance(build, bytes) else build) != "hg38":
            sys.exit(f"ERROR: {h5_path}: /meta is {build!r}; AG requires hg38. "
                     "Run fix_meta_hg38.py to lift it first.")
        meta_keys = set(f["meta"].keys())
        meta = {k: f[f"meta/{k}"][:] for k in meta_keys}
    if {"pos", "exon_start", "exon_end"}.issubset(meta_keys):
        print(f"  schema: vex/mfass", flush=True)
        variants = _h5_to_variants_vex_mfass(meta)
    elif {"variant_hg38", "exon_start_hg38", "exon_end_hg38"}.issubset(meta_keys):
        print(f"  schema: compass", flush=True)
        variants = _h5_to_variants_compass(meta)
    elif {"variant_id", "ensembl_exon_id"}.issubset(meta_keys):
        print(f"  schema: opensplice (using master tsv + gencode feather)", flush=True)
        variants = _h5_to_variants_opensplice(meta)
    else:
        raise ValueError(f"unknown reporter h5 schema; /meta keys: {sorted(meta_keys)}")
    return variants, meta


# -- minigene mode --------------------------------------------------------------
# precomputed seqs/ windows are 10001 nt with the SS at index 5000. center-pad to
# 16,384: pad_left = (16384 - 10001)//2 = 3191, pad_right = 3192. SS in padded
# frame is at 3191+5000 = 8191. AG SS peak shift -1 -> idx 8190.
MG_PAD_LEFT = 3191
MG_PAD_RIGHT = 3192
MG_SS_IDX = MG_PAD_LEFT + 5000 - 1   # 8190
MG_SS_IDX_PLUS1 = MG_SS_IDX + 1      # 8191
_IDX_TO_BASE = np.array(["A", "C", "G", "T"])


def _decode_onehot_window(onehot):
    """one-hot (L, 4) -> uppercase ACGTN string of length L."""
    has_base = onehot.sum(axis=1) > 0
    idx = onehot.argmax(axis=1)
    bases = _IDX_TO_BASE[idx]
    bases = np.where(has_base, bases, "N")
    return "".join(bases.tolist())


def _pad_construct(seq_10001):
    """symmetric N-pad to 16384."""
    return "N" * MG_PAD_LEFT + seq_10001 + "N" * MG_PAD_RIGHT


def score_one_minigene(model, seqs_row, _window=16384):
    """4 predict_sequence calls (acc_ref, acc_alt, don_ref, don_alt) per variant.

    seqs_row: dict with one-hot (10001, 4) arrays under keys exon_start_ref/alt
    and exon_end_ref/alt — pre-encoded construct windows from the input h5.
    """
    sites = (("exon_start", "exon_start_ref", "exon_start_alt"),
             ("exon_end",   "exon_end_ref",   "exon_end_alt"))
    rec = {}
    for site, ref_key, alt_key in sites:
        for allele, h5_key in (("ref", ref_key), ("alt", alt_key)):
            seq_str = _decode_onehot_window(seqs_row[h5_key])
            padded = _pad_construct(seq_str)
            out = model.predict_sequence(
                sequence=padded,
                organism=dna_model.Organism.HOMO_SAPIENS,
                requested_outputs=REQUESTED_OUTPUTS,
                ontology_terms=None,
            )
            ss_vals = out.splice_sites.values
            if MG_SS_IDX >= ss_vals.shape[0]:
                return None
            rec[f"ss_{site}_{allele}"] = ss_vals[MG_SS_IDX, :].astype(np.float32)
            rec[f"ssu_{site}_{allele}"] = out.splice_site_usage.values[MG_SS_IDX, :].astype(np.float32)

            sj = out.splice_junctions
            if sj is None or sj.values.shape[0] == 0:
                rec[f"sj_{site}_{allele}"] = None
            else:
                j_starts = np.array([int(j.start) for j in sj.junctions])
                j_ends = np.array([int(j.end) for j in sj.junctions])
                touch = ((j_starts == MG_SS_IDX) | (j_ends == MG_SS_IDX)
                         | (j_ends == MG_SS_IDX_PLUS1))
                if touch.any():
                    rec[f"sj_{site}_{allele}"] = sj.values[touch].sum(axis=0).astype(np.float32)
                else:
                    rec[f"sj_{site}_{allele}"] = np.zeros(sj.values.shape[1], dtype=np.float32)
    return rec


def _ag_predict_seq(model, padded_seq):
    """run predict_sequence; return (ss_vec, ssu_vec, sj_starts, sj_ends, sj_values)
    extracted around MG_SS_IDX/MG_SS_IDX_PLUS1. ss_vec/ssu_vec are at MG_SS_IDX.
    sj fields are full arrays so caller can sum across touching junctions.
    """
    out = model.predict_sequence(
        sequence=padded_seq,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=REQUESTED_OUTPUTS,
        ontology_terms=None,
    )
    ss_vec = out.splice_sites.values[MG_SS_IDX, :].astype(np.float32)
    ssu_vec = out.splice_site_usage.values[MG_SS_IDX, :].astype(np.float32)
    sj = out.splice_junctions
    if sj is None or sj.values.shape[0] == 0:
        return ss_vec, ssu_vec, None
    j_starts = np.array([int(j.start) for j in sj.junctions])
    j_ends = np.array([int(j.end) for j in sj.junctions])
    touch = ((j_starts == MG_SS_IDX) | (j_ends == MG_SS_IDX)
             | (j_ends == MG_SS_IDX_PLUS1))
    if touch.any():
        sj_vec = sj.values[touch].sum(axis=0).astype(np.float32)
    else:
        sj_vec = np.zeros(sj.values.shape[1], dtype=np.float32)
    return ss_vec, ssu_vec, sj_vec


def score_one_with_cache(model, seqs_row, exon_id, ref_cache):
    """score 2 alt windows (acc + donor) and reuse cached ref predictions.

    seqs_row: dict with one-hot (10001, 4) arrays under keys
      exon_start_ref/alt and exon_end_ref/alt.
    exon_id: hashable key for the cache (event_id / ensembl_exon_id).
    ref_cache: dict {exon_id: {acc_ref: (ss, ssu, sj), don_ref: (ss, ssu, sj)}}.
    """
    if exon_id not in ref_cache:
        acc_ref_padded = _pad_construct(_decode_onehot_window(seqs_row["exon_start_ref"]))
        don_ref_padded = _pad_construct(_decode_onehot_window(seqs_row["exon_end_ref"]))
        ref_cache[exon_id] = {
            "acc_ref": _ag_predict_seq(model, acc_ref_padded),
            "don_ref": _ag_predict_seq(model, don_ref_padded),
        }
    acc_ref_ss, acc_ref_ssu, acc_ref_sj = ref_cache[exon_id]["acc_ref"]
    don_ref_ss, don_ref_ssu, don_ref_sj = ref_cache[exon_id]["don_ref"]

    acc_alt_padded = _pad_construct(_decode_onehot_window(seqs_row["exon_start_alt"]))
    don_alt_padded = _pad_construct(_decode_onehot_window(seqs_row["exon_end_alt"]))
    acc_alt_ss, acc_alt_ssu, acc_alt_sj = _ag_predict_seq(model, acc_alt_padded)
    don_alt_ss, don_alt_ssu, don_alt_sj = _ag_predict_seq(model, don_alt_padded)

    return {
        "ss_exon_start_ref": acc_ref_ss, "ss_exon_start_alt": acc_alt_ss,
        "ss_exon_end_ref": don_ref_ss, "ss_exon_end_alt": don_alt_ss,
        "ssu_exon_start_ref": acc_ref_ssu, "ssu_exon_start_alt": acc_alt_ssu,
        "ssu_exon_end_ref": don_ref_ssu, "ssu_exon_end_alt": don_alt_ssu,
        "sj_exon_start_ref": acc_ref_sj, "sj_exon_start_alt": acc_alt_sj,
        "sj_exon_end_ref": don_ref_sj, "sj_exon_end_alt": don_alt_sj,
    }


def warmup_minigene(model, seqs_handles, window):
    """try rows until one scores; return its (idx, rec). None if none scoreable."""
    n = seqs_handles["exon_start_ref"].shape[0]
    for i in range(n):
        seqs_row = {k: h[i] for k, h in seqs_handles.items()}
        try:
            rec = score_one_minigene(model, seqs_row, window)
        except Exception as e:
            sys.stderr.write(f"  warmup skip row {i}: {type(e).__name__}: {e}\n")
            continue
        if rec is not None:
            return i, rec
    return None, None


def score_one_genome(model, chrom, pos, ref_b, alt_b, strand, exon_start, exon_end, window):
    """run predict_variant; extract ref+alt SS/SSU/SJ at exon_start and exon_end.

    returns dict of 12 numpy arrays (per head × site × allele), or None on
    out-of-window index.
    """
    interval = build_interval(chrom, int(pos), strand, window)
    variant = genome.Variant(
        chromosome=chrom, position=int(pos),
        reference_bases=str(ref_b), alternate_bases=str(alt_b),
    )
    out = model.predict_variant(
        interval=interval, variant=variant,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=REQUESTED_OUTPUTS,
        ontology_terms=None,
    )
    ref_out = out.reference
    alt_out = out.alternate

    site_pos = {"exon_start": int(exon_start), "exon_end": int(exon_end)}
    site_indices = {s: site_idx(p, strand, interval.start, interval.end)
                    for s, p in site_pos.items()}
    seq_len = ref_out.splice_sites.values.shape[0]
    if any(i < 0 or i >= seq_len for i in site_indices.values()):
        return None

    rec = {}
    for site, idx in site_indices.items():
        rec[f"ss_{site}_ref"] = ref_out.splice_sites.values[idx, :].astype(np.float32)
        rec[f"ss_{site}_alt"] = alt_out.splice_sites.values[idx, :].astype(np.float32)
        rec[f"ssu_{site}_ref"] = ref_out.splice_site_usage.values[idx, :].astype(np.float32)
        rec[f"ssu_{site}_alt"] = alt_out.splice_site_usage.values[idx, :].astype(np.float32)

    ref_j = ref_out.splice_junctions
    alt_j = alt_out.splice_junctions
    have_sj = (ref_j is not None and alt_j is not None
               and ref_j.values.shape == alt_j.values.shape
               and ref_j.values.shape[0] > 0)
    if have_sj:
        j_starts = np.array([int(j.start) for j in ref_j.junctions])
        j_ends = np.array([int(j.end) for j in ref_j.junctions])
        ref_v = ref_j.values
        alt_v = alt_j.values

        def site_sj(idx, arr):
            touch = (j_starts == idx) | (j_ends == idx) | (j_ends == idx + 1)
            if not touch.any():
                return np.zeros(arr.shape[1], dtype=np.float32)
            return arr[touch].sum(axis=0).astype(np.float32)

        for site, idx in site_indices.items():
            rec[f"sj_{site}_ref"] = site_sj(idx, ref_v)
            rec[f"sj_{site}_alt"] = site_sj(idx, alt_v)
    else:
        # fill with zeros, width learned at warmup
        for site in SITES:
            rec[f"sj_{site}_ref"] = None
            rec[f"sj_{site}_alt"] = None
    return rec


def warmup_genome(model, variants, window):
    """try variants until one scores; return its record + index. None on no luck."""
    for i, v in enumerate(variants):
        if v is None:
            continue
        chrom, pos, ref_b, alt_b, strand, es, ee = v
        try:
            rec = score_one_genome(model, chrom, pos, ref_b, alt_b, strand, es, ee, window)
        except Exception as e:
            sys.stderr.write(f"  warmup skip {chrom}:{pos}:{ref_b}>{alt_b}: {type(e).__name__}: {e}\n")
            continue
        if rec is not None:
            # if SJ was empty for this warmup, retry up to ~50 more to learn SJ width
            if rec[f"sj_exon_start_ref"] is None:
                # keep this rec but try to find one with SJ for shape
                pass
            return i, rec
    return None, None


def allocate_datasets(f_out, n, shapes):
    scores = f_out.create_group("scores")
    for head in HEADS:
        width = shapes[head]
        if width <= 0:
            width = 1
        for site in SITES:
            for allele in ALLELES:
                name = f"{head}_{site}_{allele}"
                scores.create_dataset(
                    name, shape=(n, width), dtype=np.float32, fillvalue=np.nan,
                    chunks=(min(64, n), width),
                    compression="gzip", compression_opts=4,
                )
    f_out.create_dataset(
        "progress/done", shape=(n,), dtype=bool, fillvalue=False,
        chunks=(min(64, n),),
    )


def finalize(f_out, meta, n, model_name, window, shapes):
    if "meta" not in f_out:
        meta_grp = f_out.create_group("meta")
        for k, v in meta.items():
            meta_grp.create_dataset(k, data=v)
    f_out.attrs["model"] = model_name
    f_out.attrs["window_len"] = int(window)
    f_out.attrs["n_ss_channels"] = int(shapes["ss"])
    f_out.attrs["n_ssu_tracks"] = int(shapes["ssu"])
    f_out.attrs["n_sj_tracks"] = int(shapes["sj"])
    f_out.attrs["n_variants"] = int(n)
    if "progress" in f_out:
        del f_out["progress"]


def write_record(f_out, i, rec, shapes):
    for k in [f"{h}_{s}_{a}" for h in HEADS for s in SITES for a in ALLELES]:
        v = rec.get(k)
        ds = f_out[f"scores/{k}"]
        width = ds.shape[1]
        if v is None:
            continue  # leave NaN
        if len(v) == width:
            ds[i] = v
        elif len(v) < width:
            buf = np.full(width, np.nan, dtype=np.float32)
            buf[:len(v)] = v
            ds[i] = buf
        else:
            ds[i] = v[:width]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=16384)
    p.add_argument("--model", default="all_folds")
    p.add_argument("--source", default="huggingface", choices=["huggingface", "kaggle"])
    args = p.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        sys.exit(f"output already exists: {args.output}")
    partial_path = Path(str(output_path) + ".partial")

    mode = detect_mode(args.input)
    # compass + opensplice get the ref-cached path (huge speedup: ref predicted
    # once per exon instead of once per variant). vex/mfass have no per-exon
    # clustering metadata and stay on the predict_variant genome path.
    with h5py.File(args.input, "r") as f:
        meta_keys = set(f["meta"].keys())
    if "event_id" in meta_keys:
        exon_id_key, dataset = "event_id", "compass"
    elif "ensembl_exon_id" in meta_keys:
        exon_id_key, dataset = "ensembl_exon_id", "opensplice"
    else:
        exon_id_key, dataset = None, None
    print(f"mode: {mode}; dataset: {dataset or 'vex/mfass'}", flush=True)

    settings = make_organism_settings()
    if settings is None:
        sys.exit("ERROR: set AG_DATA_DIR")
    print(f"loading AG {args.model} from {args.source}", flush=True)
    model = load_model(args.model, source=args.source, organism_settings=settings)

    if exon_id_key is not None:
        run_with_ref_cache(args, model, partial_path, output_path, exon_id_key)
    elif mode == "minigene":
        run_minigene(args, model, partial_path, output_path)
    else:
        run_genome(args, model, partial_path, output_path)


def run_genome(args, model, partial_path, output_path):
    print(f"loading variants from {args.input}", flush=True)
    variants, meta = h5_to_variants(args.input)
    n = len(variants)
    n_resolved = sum(1 for v in variants if v is not None)
    print(f"  {n:,} variants total, {n_resolved:,} resolved with coords", flush=True)
    if n_resolved == 0:
        sys.exit("ERROR: no variants resolvable to genomic coords")

    print("warmup: scoring first variant to detect output shapes...", flush=True)
    warmup_idx, warmup_rec = warmup_genome(model, variants, args.window)
    if warmup_rec is None:
        sys.exit("ERROR: no variants scoreable in warmup")
    shapes = {
        "ss": len(warmup_rec["ss_exon_start_ref"]),
        "ssu": len(warmup_rec["ssu_exon_start_ref"]),
        "sj": len(warmup_rec["sj_exon_start_ref"]) if warmup_rec["sj_exon_start_ref"] is not None else 0,
    }
    if shapes["sj"] == 0:
        for j in range(warmup_idx + 1, min(warmup_idx + 50, n)):
            if variants[j] is None:
                continue
            try:
                rec = score_one_genome(model, *variants[j], args.window)
            except Exception:
                continue
            if rec is not None and rec["sj_exon_start_ref"] is not None:
                shapes["sj"] = len(rec["sj_exon_start_ref"])
                break
    print(f"  shapes: ss={shapes['ss']}, ssu={shapes['ssu']}, sj={shapes['sj']}", flush=True)

    os.makedirs(output_path.parent or ".", exist_ok=True)
    f_out = h5py.File(partial_path, "w")
    allocate_datasets(f_out, n, shapes)
    write_record(f_out, warmup_idx, warmup_rec, shapes)
    f_out["progress/done"][warmup_idx] = True

    pbar = tqdm(total=n - 1, desc="scoring", file=sys.stdout, mininterval=10)
    flush_every = 100
    steps = 0
    try:
        for i in range(n):
            if i == warmup_idx:
                continue
            if _term_received:
                print("graceful exit; partial preserved for resume", flush=True)
                f_out.flush()
                f_out.close()
                del model
                gc_mod.collect()
                sys.exit(0)
            v = variants[i]
            if v is None:
                f_out["progress/done"][i] = True
                pbar.update(1)
                continue
            chrom, pos, ref_b, alt_b, strand, es, ee = v
            try:
                rec = score_one_genome(model, chrom, pos, ref_b, alt_b, strand, es, ee, args.window)
            except Exception as e:
                sys.stderr.write(f"  skip {chrom}:{pos}:{ref_b}>{alt_b}: {type(e).__name__}: {e}\n")
                rec = None
            if rec is not None:
                write_record(f_out, i, rec, shapes)
            f_out["progress/done"][i] = True
            steps += 1
            if steps % flush_every == 0:
                f_out.flush()
            pbar.update(1)
    finally:
        pbar.close()

    finalize(f_out, meta, n, args.model, args.window, shapes)
    f_out.close()
    partial_path.rename(output_path)
    print(f"wrote {output_path}", flush=True)
    del model
    gc_mod.collect()


def run_minigene(args, model, partial_path, output_path):
    """score a minigene-mode h5: decode pre-encoded 10001-bp seqs/ windows, N-pad
    to 16384, call predict_sequence per (ref|alt) x (acc|donor) window."""
    print(f"loading minigene constructs from {args.input}", flush=True)
    with h5py.File(args.input, "r") as f_in:
        meta = {k: f_in[f"meta/{k}"][:] for k in f_in["meta"].keys()}
        n = f_in["seqs/exon_start_ref"].shape[0]
    print(f"  {n:,} variants (minigene mode)", flush=True)

    SEQ_KEYS = ("exon_start_ref", "exon_start_alt", "exon_end_ref", "exon_end_alt")
    f_in = h5py.File(args.input, "r")
    seqs_handles = {k: f_in[f"seqs/{k}"] for k in SEQ_KEYS}

    print("warmup: scoring first variant to detect output shapes...", flush=True)
    warmup_idx, warmup_rec = warmup_minigene(model, seqs_handles, args.window)
    if warmup_rec is None:
        sys.exit("ERROR: no variants scoreable in warmup")
    shapes = {
        "ss": len(warmup_rec["ss_exon_start_ref"]),
        "ssu": len(warmup_rec["ssu_exon_start_ref"]),
        "sj": (len(warmup_rec["sj_exon_start_ref"])
               if warmup_rec["sj_exon_start_ref"] is not None else 0),
    }
    if shapes["sj"] == 0:
        for j in range(warmup_idx + 1, min(warmup_idx + 50, n)):
            seqs_row = {k: h[j] for k, h in seqs_handles.items()}
            try:
                rec = score_one_minigene(model, seqs_row, args.window)
            except Exception:
                continue
            if rec is not None and rec["sj_exon_start_ref"] is not None:
                shapes["sj"] = len(rec["sj_exon_start_ref"])
                break
    print(f"  shapes: ss={shapes['ss']}, ssu={shapes['ssu']}, sj={shapes['sj']}", flush=True)

    os.makedirs(output_path.parent or ".", exist_ok=True)
    f_out = h5py.File(partial_path, "w")
    allocate_datasets(f_out, n, shapes)
    write_record(f_out, warmup_idx, warmup_rec, shapes)
    f_out["progress/done"][warmup_idx] = True

    pbar = tqdm(total=n - 1, desc="scoring", file=sys.stdout, mininterval=10)
    flush_every = 100
    steps = 0
    try:
        for i in range(n):
            if i == warmup_idx:
                continue
            if _term_received:
                print("graceful exit; partial preserved for resume", flush=True)
                f_out.flush()
                f_out.close()
                f_in.close()
                del model
                gc_mod.collect()
                sys.exit(0)
            seqs_row = {k: h[i] for k, h in seqs_handles.items()}
            try:
                rec = score_one_minigene(model, seqs_row, args.window)
            except Exception as e:
                sys.stderr.write(f"  skip row {i}: {type(e).__name__}: {e}\n")
                rec = None
            if rec is not None:
                write_record(f_out, i, rec, shapes)
            f_out["progress/done"][i] = True
            steps += 1
            if steps % flush_every == 0:
                f_out.flush()
            pbar.update(1)
    finally:
        pbar.close()
        f_in.close()

    finalize(f_out, meta, n, args.model, args.window, shapes)
    f_out.attrs["mode"] = "minigene"
    f_out.close()
    partial_path.rename(output_path)
    print(f"wrote {output_path}", flush=True)
    del model
    gc_mod.collect()


def run_with_ref_cache(args, model, partial_path, output_path, exon_id_key):
    """compass + opensplice (genome or minigene): use pre-encoded seqs/ + a
    per-exon reference-prediction cache. ~2x fewer predict_sequence calls
    vs naive per-variant scoring."""
    print(f"loading constructs from {args.input}", flush=True)
    with h5py.File(args.input, "r") as f_in_meta:
        meta = {k: f_in_meta[f"meta/{k}"][:] for k in f_in_meta["meta"].keys()}
        n = f_in_meta["seqs/exon_start_ref"].shape[0]
    exon_ids_raw = meta[exon_id_key]
    if exon_ids_raw.dtype.kind in ("S", "O"):
        exon_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in exon_ids_raw]
    else:
        exon_ids = [int(v) for v in exon_ids_raw]
    n_unique = len(set(exon_ids))
    print(f"  {n:,} variants across {n_unique:,} unique exons "
          f"({n/max(1,n_unique):.0f} variants/exon avg)", flush=True)

    SEQ_KEYS = ("exon_start_ref", "exon_start_alt", "exon_end_ref", "exon_end_alt")
    f_in = h5py.File(args.input, "r")
    seqs_handles = {k: f_in[f"seqs/{k}"] for k in SEQ_KEYS}

    ref_cache = {}

    print("warmup: scoring first variant to detect output shapes...", flush=True)
    warmup_idx, warmup_rec = None, None
    for i in range(n):
        seqs_row = {k: h[i] for k, h in seqs_handles.items()}
        try:
            rec = score_one_with_cache(model, seqs_row, exon_ids[i], ref_cache)
        except Exception as e:
            sys.stderr.write(f"  warmup skip row {i}: {type(e).__name__}: {e}\n")
            continue
        if rec is not None:
            warmup_idx, warmup_rec = i, rec
            break
    if warmup_rec is None:
        sys.exit("ERROR: no variants scoreable in warmup")
    shapes = {
        "ss": len(warmup_rec["ss_exon_start_ref"]),
        "ssu": len(warmup_rec["ssu_exon_start_ref"]),
        "sj": (len(warmup_rec["sj_exon_start_ref"])
               if warmup_rec["sj_exon_start_ref"] is not None else 0),
    }
    if shapes["sj"] == 0:
        for j in range(warmup_idx + 1, min(warmup_idx + 50, n)):
            seqs_row = {k: h[j] for k, h in seqs_handles.items()}
            try:
                rec = score_one_with_cache(model, seqs_row, exon_ids[j], ref_cache)
            except Exception:
                continue
            if rec is not None and rec["sj_exon_start_ref"] is not None:
                shapes["sj"] = len(rec["sj_exon_start_ref"])
                break
    print(f"  shapes: ss={shapes['ss']}, ssu={shapes['ssu']}, sj={shapes['sj']}", flush=True)

    os.makedirs(output_path.parent or ".", exist_ok=True)
    f_out = h5py.File(partial_path, "w")
    allocate_datasets(f_out, n, shapes)
    write_record(f_out, warmup_idx, warmup_rec, shapes)
    f_out["progress/done"][warmup_idx] = True

    pbar = tqdm(total=n - 1, desc="scoring", file=sys.stdout, mininterval=10)
    flush_every = 100
    steps = 0
    try:
        for i in range(n):
            if i == warmup_idx:
                continue
            if _term_received:
                print("graceful exit; partial preserved for resume", flush=True)
                f_out.flush()
                f_out.close()
                f_in.close()
                del model
                gc_mod.collect()
                sys.exit(0)
            seqs_row = {k: h[i] for k, h in seqs_handles.items()}
            try:
                rec = score_one_with_cache(model, seqs_row, exon_ids[i], ref_cache)
            except Exception as e:
                sys.stderr.write(f"  skip row {i}: {type(e).__name__}: {e}\n")
                rec = None
            if rec is not None:
                write_record(f_out, i, rec, shapes)
            f_out["progress/done"][i] = True
            steps += 1
            if steps % flush_every == 0:
                f_out.flush()
            pbar.update(1)
    finally:
        pbar.close()
        f_in.close()

    finalize(f_out, meta, n, args.model, args.window, shapes)
    f_out.attrs["ref_cache_size"] = len(ref_cache)
    f_out.close()
    partial_path.rename(output_path)
    print(f"wrote {output_path}  (ref cache hit {len(ref_cache):,} unique exons)", flush=True)
    del model
    gc_mod.collect()


if __name__ == "__main__":
    main()
