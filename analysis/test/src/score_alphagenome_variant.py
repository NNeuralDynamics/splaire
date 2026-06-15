#!/usr/bin/env python3
"""score sQTL-style variants with AlphaGenome (all_folds, 16k context).

reads a variant table (TSV) with variant_id, chrom, pos, ref, alt, gene_strand and writes:

  <out_prefix>.h5            dense ref + alt SPLICE_SITES across full 16k window
                              ref         (n, 16384, 5)   float32
                              alt         (n, 16384, 5)   float32
                              var_key     (n,)            str
                              gene_strand (n,)            str
  <out_prefix>.ssu.parquet   sparse SPLICE_SITE_USAGE long format
                              variant_id, allele ('ref'/'alt'),
                              pos_in_window (genomic-frame index),
                              track_idx (0..735), value (float32)
                              only rows with value > SSU_SPARSE_THRESHOLD written
  <out_prefix>.sj.parquet    SPLICE_JUNCTIONS long format (junction × track)
                              variant_id, allele, j_start, j_end (genomic-frame
                              array indices), track_idx (0..524), value (float32)

outputs in genomic frame: minus-strand intervals are flipped before write so the
position-in-window indexing matches other sQTL scorers.

usage:
  python score_alphagenome_variant.py variants.tsv out_prefix --fasta GRCh38.fa
"""
import argparse
import gc as gc_mod
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _ag_common import (  # noqa: E402
    make_organism_settings, load_model, build_interval,
    flip_to_genomic, junction_idx_to_genomic,
    REQUESTED_OUTPUTS, SS_N_CHANNELS, SSU_N_TRACKS, SJ_N_TRACKS,
    dna_model, genome,
)


SSU_SPARSE_THRESHOLD = 1e-3   # keep SSU values above this in the sparse parquet
BATCH_SIZE = 64               # flush h5 + write parquet shard + progress every N variants


def load_variants(path):
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, dtype={"chrom": str})
    if not df["chrom"].iloc[0].startswith("chr"):
        df["chrom"] = "chr" + df["chrom"].astype(str)
    needed = ["variant_id", "chrom", "pos", "ref", "alt", "gene_strand"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"variant table missing columns: {missing}")
    # one inference per variant; (variant, gene) duplicates in input collapse to first
    df = df.drop_duplicates(subset=["variant_id"], keep="first").reset_index(drop=True)
    return df


def score_one(model, chrom, pos, ref_b, alt_b, strand, window):
    """run predict_variant on one variant, return outputs flipped to genomic frame.

    returns (ss_ref, ss_alt, ssu_ref, ssu_alt, j_starts, j_ends, sj_ref, sj_alt):
      ss arrays  (window, 5)
      ssu arrays (window, 736)
      sj_starts / sj_ends (n_junctions,) genomic-frame array indices
      sj arrays  (n_junctions, 525)
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

    ss_ref = flip_to_genomic(ref_out.splice_sites.values.astype(np.float32), strand)
    ss_alt = flip_to_genomic(alt_out.splice_sites.values.astype(np.float32), strand)
    ssu_ref = flip_to_genomic(ref_out.splice_site_usage.values.astype(np.float32), strand)
    ssu_alt = flip_to_genomic(alt_out.splice_site_usage.values.astype(np.float32), strand)

    ref_j = ref_out.splice_junctions
    alt_j = alt_out.splice_junctions
    if (ref_j is not None and alt_j is not None
            and ref_j.values.shape == alt_j.values.shape
            and ref_j.values.shape[0] > 0):
        j_starts_raw = np.array([int(j.start) for j in ref_j.junctions])
        j_ends_raw = np.array([int(j.end) for j in ref_j.junctions])
        if strand == "-":
            j_starts = np.array([junction_idx_to_genomic(e, "-", window) for e in j_ends_raw])
            j_ends = np.array([junction_idx_to_genomic(s, "-", window) for s in j_starts_raw])
        else:
            j_starts = j_starts_raw
            j_ends = j_ends_raw
        sj_ref = ref_j.values.astype(np.float32)
        sj_alt = alt_j.values.astype(np.float32)
    else:
        j_starts = np.zeros(0, dtype=np.int32)
        j_ends = np.zeros(0, dtype=np.int32)
        sj_ref = np.zeros((0, SJ_N_TRACKS), dtype=np.float32)
        sj_alt = np.zeros((0, SJ_N_TRACKS), dtype=np.float32)

    return ss_ref, ss_alt, ssu_ref, ssu_alt, j_starts, j_ends, sj_ref, sj_alt


_SSU_SCHEMA = pa.schema([
    ("variant_id", pa.string()),
    ("allele", pa.string()),
    ("pos_in_window", pa.int32()),
    ("track_idx", pa.int16()),
    ("value", pa.float32()),
])

_SJ_SCHEMA = pa.schema([
    ("variant_id", pa.string()),
    ("allele", pa.string()),
    ("j_start", pa.int32()),
    ("j_end", pa.int32()),
    ("track_idx", pa.int16()),
    ("value", pa.float32()),
])


def ssu_rows(ssu_ref, ssu_alt, variant_id):
    """yield (table) for each non-empty allele's sparse SSU rows."""
    for allele, mat in (("ref", ssu_ref), ("alt", ssu_alt)):
        pos_idx, track_idx = np.where(mat > SSU_SPARSE_THRESHOLD)
        if len(pos_idx) == 0:
            continue
        vals = mat[pos_idx, track_idx]
        n = len(pos_idx)
        yield pa.table({
            "variant_id": pa.array([variant_id] * n, type=pa.string()),
            "allele":     pa.array([allele] * n, type=pa.string()),
            "pos_in_window": pa.array(pos_idx, type=pa.int32()),
            "track_idx":  pa.array(track_idx, type=pa.int16()),
            "value":      pa.array(vals, type=pa.float32()),
        }, schema=_SSU_SCHEMA)


def sj_rows(j_starts, j_ends, sj_ref, sj_alt, variant_id):
    """yield (table) for each non-empty allele's SJ rows (junction × tissue)."""
    for allele, mat in (("ref", sj_ref), ("alt", sj_alt)):
        nj, nt = mat.shape
        if nj == 0:
            continue
        ji, ti = np.indices((nj, nt))
        yield pa.table({
            "variant_id": pa.array([variant_id] * (nj * nt), type=pa.string()),
            "allele":     pa.array([allele] * (nj * nt), type=pa.string()),
            "j_start":    pa.array(np.repeat(j_starts, nt).astype(np.int32), type=pa.int32()),
            "j_end":      pa.array(np.repeat(j_ends, nt).astype(np.int32), type=pa.int32()),
            "track_idx":  pa.array(ti.ravel().astype(np.int16), type=pa.int16()),
            "value":      pa.array(mat.ravel().astype(np.float32), type=pa.float32()),
        }, schema=_SJ_SCHEMA)


def _write_progress(progress_file, next_batch):
    """atomically write next-batch-to-process so resume picks up there."""
    tmp = progress_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"next_batch": int(next_batch)}))
    tmp.rename(progress_file)


def _read_progress(progress_file):
    if not progress_file.exists():
        return 0
    try:
        return int(json.loads(progress_file.read_text())["next_batch"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0


def _flush_batch(f_h5, ssu_tables, sj_tables, ssu_dir, sj_dir, batch_idx,
                 progress_file, next_batch_idx):
    """commit h5 to disk, write batch parquet shards, advance progress.

    order matters for crash-safety:
      1. h5.flush — h5 rows for this batch hit disk
      2. write parquet shards (overwrites existing if any)
      3. progress.json points to next_batch
    if killed between 1 and 3 we re-do this batch on resume (idempotent overwrite).
    """
    f_h5.flush()
    if ssu_tables:
        pq.write_table(pa.concat_tables(ssu_tables),
                       ssu_dir / f"batch_{batch_idx:06d}.parquet",
                       compression="zstd")
    if sj_tables:
        pq.write_table(pa.concat_tables(sj_tables),
                       sj_dir / f"batch_{batch_idx:06d}.parquet",
                       compression="zstd")
    _write_progress(progress_file, next_batch_idx)


def _concat_shards(shard_dir, out_path, schema):
    """stream parquet shards into single output parquet."""
    shards = sorted(shard_dir.glob("batch_*.parquet"))
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    try:
        if not shards:
            writer.write_table(pa.table({c.name: [] for c in schema}, schema=schema))
        else:
            for shard in shards:
                writer.write_table(pq.read_table(shard))
    finally:
        writer.close()


def score_table(model, df, out_prefix, window, model_name):
    """score one variant table into <out_prefix>.{h5, ssu.parquet, sj.parquet}.

    resume-capable: writes to <out_prefix>.partial/ with per-batch parquet shards
    and a progress.json marker. on walltime kill, next invocation picks up at the
    first batch whose shards aren't written yet.
    """
    n = len(df)
    out_prefix = Path(out_prefix)
    out_h5 = Path(str(out_prefix) + ".h5")
    out_ssu = Path(str(out_prefix) + ".ssu.parquet")
    out_sj = Path(str(out_prefix) + ".sj.parquet")

    if out_h5.exists():
        print(f"skip: {out_h5} exists", flush=True)
        return

    os.makedirs(out_prefix.parent or ".", exist_ok=True)

    partial_dir = Path(str(out_prefix) + ".partial")
    partial_h5 = partial_dir / "scores.h5"
    ssu_dir = partial_dir / "ssu_shards"
    sj_dir = partial_dir / "sj_shards"
    progress_file = partial_dir / "progress.json"
    partial_dir.mkdir(exist_ok=True)
    ssu_dir.mkdir(exist_ok=True)
    sj_dir.mkdir(exist_ok=True)

    next_batch = _read_progress(progress_file)
    start = next_batch * BATCH_SIZE
    if start >= n:
        start = n   # everything done — fall through to finalize

    if partial_h5.exists() and start > 0:
        print(f"resuming from row {start:,}/{n:,} (batch {next_batch})", flush=True)
        f_h5 = h5py.File(partial_h5, "a")
        if f_h5["ref"].shape != (n, window, SS_N_CHANNELS):
            f_h5.close()
            sys.exit(f"ERROR: partial h5 shape {f_h5['ref'].shape} != expected "
                     f"{(n, window, SS_N_CHANNELS)} — delete {partial_dir} to restart")
        existing_keys = f_h5["var_key"][:].astype(str)
        new_keys = df["variant_id"].values.astype(str)
        if not np.array_equal(existing_keys, new_keys):
            f_h5.close()
            sys.exit(f"ERROR: partial var_key mismatch — input table changed since "
                     f"resume started. delete {partial_dir} to restart")
    else:
        # fresh start (or partial h5 missing while progress.json says >0 — treat as fresh)
        if partial_h5.exists():
            partial_h5.unlink()
        for shard_dir in (ssu_dir, sj_dir):
            for p in shard_dir.glob("batch_*.parquet"):
                p.unlink()
        if progress_file.exists():
            progress_file.unlink()
        start = 0
        next_batch = 0

        f_h5 = h5py.File(partial_h5, "w")
        chunks_ss = (min(8, n), window, SS_N_CHANNELS)
        f_h5.create_dataset("ref", shape=(n, window, SS_N_CHANNELS), dtype=np.float32,
                            fillvalue=np.nan, chunks=chunks_ss,
                            compression="gzip", compression_opts=4)
        f_h5.create_dataset("alt", shape=(n, window, SS_N_CHANNELS), dtype=np.float32,
                            fillvalue=np.nan, chunks=chunks_ss,
                            compression="gzip", compression_opts=4)
        str_dt = h5py.special_dtype(vlen=str)
        f_h5.create_dataset("var_key", data=df["variant_id"].values.astype(object), dtype=str_dt)
        f_h5.create_dataset("gene_strand", data=df["gene_strand"].values.astype(object), dtype=str_dt)

    pbar = tqdm(total=n, initial=start, desc="scoring", file=sys.stdout, mininterval=10)
    batch_ssu, batch_sj = [], []
    batch_idx = next_batch

    for i in range(start, n):
        row = df.iloc[i]
        try:
            ss_ref, ss_alt, ssu_ref, ssu_alt, j_starts, j_ends, sj_ref, sj_alt = score_one(
                model, row["chrom"], int(row["pos"]), row["ref"], row["alt"],
                row["gene_strand"], window,
            )
            f_h5["ref"][i] = ss_ref
            f_h5["alt"][i] = ss_alt
            for tab in ssu_rows(ssu_ref, ssu_alt, row["variant_id"]):
                batch_ssu.append(tab)
            for tab in sj_rows(j_starts, j_ends, sj_ref, sj_alt, row["variant_id"]):
                batch_sj.append(tab)
        except Exception as e:
            sys.stderr.write(f"  skip {row['variant_id']}: {type(e).__name__}: {e}\n")
        pbar.update(1)

        if (i + 1) % BATCH_SIZE == 0 or i == n - 1:
            _flush_batch(f_h5, batch_ssu, batch_sj, ssu_dir, sj_dir,
                         batch_idx, progress_file, batch_idx + 1)
            batch_ssu, batch_sj = [], []
            batch_idx += 1

    pbar.close()

    # finalize: count actual scored rows (rows with ref[i, 0, 0] not NaN)
    n_scored = int(np.sum(~np.isnan(f_h5["ref"][:, 0, 0])))
    f_h5.attrs["model"] = model_name
    f_h5.attrs["window_len"] = int(window)
    f_h5.attrs["n_ss_channels"] = SS_N_CHANNELS
    f_h5.attrs["n_ssu_tracks"] = SSU_N_TRACKS
    f_h5.attrs["n_sj_tracks"] = SJ_N_TRACKS
    f_h5.attrs["n_variants"] = int(n)
    f_h5.attrs["n_scored"] = n_scored
    f_h5.close()

    print(f"concatenating parquet shards", flush=True)
    _concat_shards(ssu_dir, out_ssu, _SSU_SCHEMA)
    _concat_shards(sj_dir, out_sj, _SJ_SCHEMA)
    partial_h5.rename(out_h5)
    shutil.rmtree(partial_dir)

    print(f"wrote {out_h5}  ({n_scored:,}/{n:,} scored)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_table")
    p.add_argument("out_prefix", help="output prefix; writes .h5, .ssu.parquet, .sj.parquet")
    p.add_argument("--extra", action="append", nargs=2, default=[],
                   metavar=("INPUT", "OUT_PREFIX"),
                   help="extra (input_table, out_prefix) pairs to score in same model session")
    p.add_argument("--fasta", default=None)
    p.add_argument("--window", type=int, default=16384,
                   help="AG context length (16384/131072/524288/1048576). default 16384.")
    p.add_argument("--model", default="all_folds")
    p.add_argument("--source", default="huggingface", choices=["huggingface", "kaggle"])
    args = p.parse_args()

    pairs = [(args.input_table, args.out_prefix)] + list(args.extra)
    print(f"will score {len(pairs)} table(s):", flush=True)
    for inp, out in pairs:
        print(f"  {inp} -> {out}.{{h5,ssu.parquet,sj.parquet}}", flush=True)

    pair_dfs = []
    for inp, out in pairs:
        print(f"loading {inp}", flush=True)
        df = load_variants(inp)
        print(f"  {len(df):,} variants", flush=True)
        pair_dfs.append((df, out))

    settings = make_organism_settings(fasta_override=args.fasta)
    if settings is None:
        sys.exit("ERROR: pass --fasta or set AG_DATA_DIR")

    print(f"loading AG {args.model} from {args.source}", flush=True)
    model = load_model(args.model, source=args.source, organism_settings=settings)

    for df, out_prefix in pair_dfs:
        score_table(model, df, out_prefix, args.window, args.model)
    del model
    gc_mod.collect()


if __name__ == "__main__":
    main()
