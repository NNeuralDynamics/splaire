#!/usr/bin/env python3
"""score per-gene h5 with AlphaGenome — streaming version (one job, all folds).

streams per-entry predictions directly to per-fold parquets via pyarrow, then
stream-merges per-fold parquets horizontally (row-group by row-group) into the
final output. avoids accumulating large per-entry arrays in memory.

input h5 schema (from build_h5_alphagenome.py):
  SEQ_{n}    int8        shape (1048576,)            int-encoded one-hot
  Y_{n}      float32     shape (cov_len, 4)          [neither, acc, don, ssu]
  GC_{n}     structured  shape (cov_len,)            per-bp chrom/strand/position/name

per-fold parquet columns:
  chrom, pos, strand, y_acceptor, y_donor, y_ssu    (ground truth, in every per-fold file)
  acceptor_{fold}, donor_{fold}                     (SS softmax, ch1/ch0 in our pre-mRNA orientation)
  ssu_{fold}_track_{t}                              (SSU sigmoid per tissue track, t=0..n_ssu-1)

final merged parquet: GT once + each fold's pred cols horizontally concatenated.

junctions are written to a sidecar parquet (variable n_junctions per gene) only
when --with-junctions is set. dense + large; off by default.

usage:
    python score_alphagenome.py input.h5 output.parquet \\
        [--folds fold_0,fold_1,fold_2,fold_3] [--with-junctions] [--keep-temp]
"""

import argparse
import gc as gc_mod
import os
import queue
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

script_dir = Path(__file__).parent
ag_path = script_dir.parent.parent / "other_models" / "alphagenome_research"
sys.path.insert(0, str(ag_path / "src"))

from alphagenome_research.model import dna_model
try:
    from alphagenome.models import dna_output
except ImportError:
    from alphagenome.data import dna_output

AG_SEQ_LEN = 2**20            # native 1 Mb (per-gene h5)
AG_SEQ_LEN_SHORT = 16384      # 8 * 2048, shared 15k h5 mode (N-padded 692 each side)
INPUT_LEN_SHORT = 15000       # shared h5 window
OUTPUT_LEN_SHORT = 5000       # center label window
PAD_SHORT = (AG_SEQ_LEN_SHORT - INPUT_LEN_SHORT) // 2                       # 692
CENTER_START_SHORT = PAD_SHORT + (INPUT_LEN_SHORT - OUTPUT_LEN_SHORT) // 2  # 5692
CENTER_END_SHORT = CENTER_START_SHORT + OUTPUT_LEN_SHORT                    # 10692

INT_TO_BASE = np.array(['N', 'A', 'C', 'G', 'T'])
ACC_CH = 1
DON_CH = 0
GT_COLS = ['chrom', 'pos', 'strand', 'y_acceptor', 'y_donor', 'y_ssu']


def make_organism_settings():
    ag_data_dir = os.environ.get('AG_DATA_DIR')
    if not ag_data_dir:
        return None
    d = Path(ag_data_dir)
    return {
        dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
            fasta_path=str(d / 'GRCh38.p13.genome.fa'),
            gtf_feather_path=str(d / 'gencode.v46.annotation.gtf.gz.feather'),
            pas_feather_path=str(d / 'polyadb_human_v3_exon3_contiguous_gtfv46.feather'),
            splice_site_starts_feather_path=str(d / 'gencode.v46.splice_sites_starts.feather'),
            splice_site_ends_feather_path=str(d / 'gencode.v46.splice_sites_ends.feather'),
        ),
    }


def decode_seq(seq_int):
    return ''.join(INT_TO_BASE[seq_int])


def compact_ssu_to_splice_sites(table):
    """null out ssu_*_track_* values at non-splice-site rows. classification + GT
    cols are kept at all positions. parquet's def-level encoding compresses the
    null runs to almost nothing. saves ~99% of disk for the 736-track ssu cols."""
    mask = pc.or_(
        pc.greater(table.column('y_acceptor'), 0),
        pc.greater(table.column('y_donor'), 0),
    )
    arrays = []
    for name in table.column_names:
        col = table.column(name)
        if name.startswith('ssu_') and '_track_' in name:
            null_scalar = pa.scalar(None, type=col.type)
            col = pc.if_else(mask, col, null_scalar)
        arrays.append(col)
    return pa.table(arrays, names=table.column_names)


class AsyncParquetWriter:
    """parquet writer running in a background thread so writes don't block the gpu loop.
    main thread calls write_table(t) → enqueues; worker pulls and writes.
    queue is bounded (maxsize) so memory stays predictable if gpu outruns disk."""

    def __init__(self, path, compression='zstd', maxsize=8):
        self.path = path
        self.compression = compression
        self._q = queue.Queue(maxsize=maxsize)
        self._exc = None
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        try:
            writer = None
            while True:
                item = self._q.get()
                if item is None:
                    break
                if writer is None:
                    writer = pq.ParquetWriter(self.path, item.schema, compression=self.compression)
                writer.write_table(item)
            if writer is not None:
                writer.close()
        except Exception as e:
            self._exc = e

    def write_table(self, table):
        if self._exc is not None:
            raise RuntimeError(f"async parquet writer failed: {self._exc}") from self._exc
        self._q.put(table)

    def depth(self):
        return self._q.qsize()

    def close(self):
        self._q.put(None)
        self._t.join()
        if self._exc is not None:
            raise RuntimeError(f"async parquet writer failed: {self._exc}") from self._exc


def score_one_fold_streaming(model, h5_path, entry_keys, fold, fold_path,
                             requested, include_junctions, junctions_path):
    """stream per-entry predictions for one fold to disk; returns (n_ssu, n_sj, names)."""
    writer = AsyncParquetWriter(fold_path)
    jx_writer = None
    n_ssu = 0
    n_sj = 0
    ssu_names = None
    jx_names = None

    # phase timing — accumulated, logged periodically so we can see where time goes
    t_h5 = t_predict = t_post = t_write = t_junc = 0.0
    t_fold_start = time.perf_counter()
    last_log = t_fold_start
    LOG_INTERVAL_S = 30  # phase breakdown every 30s

    with h5py.File(h5_path, "r") as fin:
        pbar = tqdm(entry_keys, desc=f"  {fold}", file=sys.stdout, mininterval=10)
        for i, key in enumerate(pbar):
            t0 = time.perf_counter()
            seq_ds = fin[f"SEQ_{key}"]
            attrs = dict(seq_ds.attrs)
            gene_offset = int(attrs['gene_offset_in_seq'])
            uid = str(attrs.get('unique_id', f'entry_{key}'))

            seq_int = seq_ds[:]
            y = fin[f"Y_{key}"][:]
            gc_arr = fin[f"GC_{key}"][:]
            cov_len = y.shape[0]
            seq_str = decode_seq(seq_int)
            t1 = time.perf_counter()

            out = model.predict_sequence(
                sequence=seq_str,
                organism=dna_model.Organism.HOMO_SAPIENS,
                requested_outputs=requested,
                ontology_terms=None,
            )
            t2 = time.perf_counter()

            ss = out.splice_sites.values
            su = out.splice_site_usage.values
            lo, hi = gene_offset, gene_offset + cov_len
            acc = ss[lo:hi, ACC_CH].astype(np.float32)
            don = ss[lo:hi, DON_CH].astype(np.float32)
            ssu = su[lo:hi].astype(np.float32)

            if n_ssu == 0:
                n_ssu = ssu.shape[1]
                ssu_names = list(out.splice_site_usage.metadata['name'].values)

            data = {
                'chrom':      gc_arr['chrom'].astype(np.int8),
                'pos':        gc_arr['position'].astype(np.int32),
                'strand':     gc_arr['strand'].astype(np.int8),
                'y_acceptor': y[:, 1].astype(np.float32),
                'y_donor':    y[:, 2].astype(np.float32),
                'y_ssu':      y[:, 3].astype(np.float32),
                f'acceptor_{fold}': acc,
                f'donor_{fold}':    don,
            }
            for t in range(n_ssu):
                data[f'ssu_{fold}_track_{t}'] = ssu[:, t]
            t3 = time.perf_counter()

            table = pa.Table.from_pydict(data)
            table = compact_ssu_to_splice_sites(table)  # null ssu cols at non-splice rows
            writer.write_table(table)  # non-blocking unless background queue is full
            t4 = time.perf_counter()

            if include_junctions and out.splice_junctions is not None and len(out.splice_junctions) > 0:
                j_data = out.splice_junctions
                jx = j_data.junctions
                jv = j_data.values.astype(np.float32)
                if n_sj == 0:
                    n_sj = jv.shape[1]
                    jx_names = list(j_data.metadata['name'].values)
                n_j = len(jx)
                jd = {
                    'unique_id': np.array([uid] * n_j, dtype=object),
                    'fold':      np.array([fold] * n_j, dtype=object),
                    'j_start':   np.array([int(j.start) for j in jx], dtype=np.int32),
                    'j_end':     np.array([int(j.end)   for j in jx], dtype=np.int32),
                    'j_strand':  np.array([str(j.strand) for j in jx], dtype=object),
                    'k':         np.array(
                        [j.k if j.k is not None else -1 for j in jx], dtype=np.int32),
                }
                for t in range(n_sj):
                    jd[f'j_track_{t}'] = jv[:, t]
                jt = pa.Table.from_pydict(jd)
                if jx_writer is None:
                    jx_writer = pq.ParquetWriter(junctions_path, jt.schema, compression='zstd')
                jx_writer.write_table(jt)
                del j_data, jx, jv, jd, jt
            t5 = time.perf_counter()

            t_h5      += t1 - t0
            t_predict += t2 - t1
            t_post    += t3 - t2
            t_write   += t4 - t3
            t_junc    += t5 - t4

            del ss, su, acc, don, ssu, out, data, table, seq_int, seq_str
            gc_mod.collect()

            # periodic phase breakdown — first iter always, then every LOG_INTERVAL_S
            now = time.perf_counter()
            if i == 0 or now - last_log >= LOG_INTERVAL_S:
                n = i + 1
                total = t_h5 + t_predict + t_post + t_write + t_junc
                print(
                    f"  [{fold} phase] iter={n}/{len(entry_keys)} "
                    f"h5={t_h5/n*1000:.0f}ms "
                    f"predict={t_predict/n*1000:.0f}ms "
                    f"post={t_post/n*1000:.0f}ms "
                    f"enqueue={t_write/n*1000:.0f}ms "
                    f"junc={t_junc/n*1000:.0f}ms "
                    f"total={total/n*1000:.0f}ms/it "
                    f"qdepth={writer.depth()}",
                    flush=True,
                )
                last_log = now

    writer.close()
    if jx_writer:
        jx_writer.close()

    n = len(entry_keys)
    if n > 0:
        total = t_h5 + t_predict + t_post + t_write + t_junc
        elapsed = time.perf_counter() - t_fold_start
        print(
            f"  [{fold} done] {n} entries in {elapsed:.0f}s wall, mean: "
            f"h5={t_h5/n*1000:.0f}ms "
            f"predict={t_predict/n*1000:.0f}ms "
            f"post={t_post/n*1000:.0f}ms "
            f"enqueue={t_write/n*1000:.0f}ms "
            f"junc={t_junc/n*1000:.0f}ms "
            f"total={total/n*1000:.0f}ms/it",
            flush=True,
        )
    return n_ssu, n_sj, ssu_names, jx_names


def score_one_fold_chunked(model, h5_path, chunk_ids, fold, fold_path, requested):
    """16384 mode: read shared h5 X{n}/Y{n}/GC{n} chunks, N-pad 15000→16384,
    predict, slice center 5000 [5692:10692], stream to parquet.
    junctions disabled — too little context at 15k for pairing to make sense."""
    writer = AsyncParquetWriter(fold_path)
    n_ssu = 0
    ssu_names = None

    t_h5 = t_predict = t_post = t_write = 0.0
    t_fold_start = time.perf_counter()
    last_log = t_fold_start
    LOG_INTERVAL_S = 30

    with h5py.File(h5_path, "r") as fin:
        pbar = tqdm(chunk_ids, desc=f"  {fold}", file=sys.stdout, mininterval=10)
        it_seen = 0
        for chunk in pbar:
            x_chunk = fin[f"X{chunk}"][:]     # (nw, 15000, 4)
            y_chunk = fin[f"Y{chunk}"][:]     # (nw, 5000, 4)
            gc_chunk = fin[f"GC{chunk}"][:]   # (nw, 5000) structured
            nw = x_chunk.shape[0]

            for wi in range(nw):
                t0 = time.perf_counter()
                # (15000, 4) one-hot → char string, then N-pad to 16384
                idx = np.argmax(x_chunk[wi], axis=-1)
                # non-hot positions (all-zero rows) → N; but argmax on all-zero returns 0 → 'N' via INT_TO_BASE
                has_base = x_chunk[wi].sum(axis=-1) > 0
                idx = np.where(has_base, idx + 1, 0)  # shift so 0=N, 1=A..
                seq_str = ''.join(INT_TO_BASE[idx])
                seq_str = 'N' * PAD_SHORT + seq_str + 'N' * PAD_SHORT
                t1 = time.perf_counter()

                out = model.predict_sequence(
                    sequence=seq_str,
                    organism=dna_model.Organism.HOMO_SAPIENS,
                    requested_outputs=requested,
                    ontology_terms=None,
                )
                t2 = time.perf_counter()

                ss = out.splice_sites.values                                       # (16384, 2)
                su = out.splice_site_usage.values                                  # (16384, n_ssu)
                acc = ss[CENTER_START_SHORT:CENTER_END_SHORT, ACC_CH].astype(np.float32)
                don = ss[CENTER_START_SHORT:CENTER_END_SHORT, DON_CH].astype(np.float32)
                ssu = su[CENTER_START_SHORT:CENTER_END_SHORT].astype(np.float32)

                if n_ssu == 0:
                    n_ssu = ssu.shape[1]
                    ssu_names = list(out.splice_site_usage.metadata['name'].values)

                gc_w = gc_chunk[wi]           # (5000,) structured
                y_w = y_chunk[wi]             # (5000, 4)
                data = {
                    'chrom':      gc_w['chrom'].astype(np.int8),
                    'pos':        gc_w['position'].astype(np.int32),
                    'strand':     gc_w['strand'].astype(np.int8),
                    'y_acceptor': y_w[:, 1].astype(np.float32),
                    'y_donor':    y_w[:, 2].astype(np.float32),
                    'y_ssu':      y_w[:, 3].astype(np.float32),
                    f'acceptor_{fold}': acc,
                    f'donor_{fold}':    don,
                }
                for t in range(n_ssu):
                    data[f'ssu_{fold}_track_{t}'] = ssu[:, t]
                t3 = time.perf_counter()

                table = pa.Table.from_pydict(data)
                table = compact_ssu_to_splice_sites(table)
                writer.write_table(table)
                t4 = time.perf_counter()

                t_h5      += t1 - t0
                t_predict += t2 - t1
                t_post    += t3 - t2
                t_write   += t4 - t3

                del ss, su, acc, don, ssu, out, data, table, idx, seq_str
                gc_mod.collect()
                it_seen += 1

                now = time.perf_counter()
                if it_seen == 1 or now - last_log >= LOG_INTERVAL_S:
                    total = t_h5 + t_predict + t_post + t_write
                    print(
                        f"  [{fold} phase] iter={it_seen} "
                        f"decode={t_h5/it_seen*1000:.0f}ms "
                        f"predict={t_predict/it_seen*1000:.0f}ms "
                        f"post={t_post/it_seen*1000:.0f}ms "
                        f"enqueue={t_write/it_seen*1000:.0f}ms "
                        f"total={total/it_seen*1000:.0f}ms/it "
                        f"qdepth={writer.depth()}",
                        flush=True,
                    )
                    last_log = now

    writer.close()

    if it_seen > 0:
        elapsed = time.perf_counter() - t_fold_start
        total = t_h5 + t_predict + t_post + t_write
        print(
            f"  [{fold} done] {it_seen} windows in {elapsed:.0f}s wall, mean: "
            f"decode={t_h5/it_seen*1000:.0f}ms "
            f"predict={t_predict/it_seen*1000:.0f}ms "
            f"post={t_post/it_seen*1000:.0f}ms "
            f"enqueue={t_write/it_seen*1000:.0f}ms "
            f"total={total/it_seen*1000:.0f}ms/it",
            flush=True,
        )
    return n_ssu, 0, ssu_names, None


def stream_merge_folds(fold_paths, output_path):
    """row-group horizontal merge of per-fold parquets.
    each per-fold parquet shares GT cols and adds its own fold-specific pred cols.
    """
    readers = [pq.ParquetFile(fp) for fp in fold_paths]
    n_rg = readers[0].num_row_groups
    print(f"merging {len(readers)} fold parquets ({n_rg} row groups)", flush=True)

    # merged schema = fold_0's full schema + each subsequent fold's non-GT cols
    fields = list(readers[0].schema_arrow)
    for r in readers[1:]:
        for f in r.schema_arrow:
            if f.name not in GT_COLS:
                fields.append(f)
    merged_schema = pa.schema(fields)

    with pq.ParquetWriter(output_path, merged_schema, compression='zstd') as out:
        for rg in tqdm(range(n_rg), desc='merge', file=sys.stdout, mininterval=5):
            tables = [r.read_row_group(rg) for r in readers]
            merged = tables[0]
            for t in tables[1:]:
                for col in t.column_names:
                    if col not in GT_COLS:
                        merged = merged.append_column(col, t.column(col))
            out.write_table(merged)
    print(f"wrote merged {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5")
    parser.add_argument("output_parquet")
    parser.add_argument("--folds", default="fold_0,fold_1,fold_2,fold_3")
    parser.add_argument("--source", default="huggingface",
                        choices=["huggingface", "kaggle"])
    parser.add_argument("--with-junctions", action="store_true",
                        help="enable junctions output (off by default; dense + large)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="do not delete per-fold temp parquets after merge")
    parser.add_argument("--no-merge", action="store_true",
                        help="exit after scoring; do not merge per-fold parquets (run merge in CPU job)")
    parser.add_argument("--merge-only", action="store_true",
                        help="skip scoring, just merge existing per-fold temp parquets")
    parser.add_argument("--ag-seq-len", type=int, default=AG_SEQ_LEN,
                        choices=[AG_SEQ_LEN, AG_SEQ_LEN_SHORT],
                        help=f"AG input length: {AG_SEQ_LEN} (native per-gene h5) "
                             f"or {AG_SEQ_LEN_SHORT} (shared 15k h5, N-padded)")
    args = parser.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    include_junctions = args.with_junctions
    seq_len = args.ag_seq_len
    is_short = seq_len == AG_SEQ_LEN_SHORT
    if is_short and include_junctions:
        print("WARN: junctions ignored in 16384 mode (context too short)", flush=True)
        include_junctions = False
    print(f"folds={folds}, ag_seq_len={seq_len}, include_junctions={include_junctions}", flush=True)

    out_base = args.output_parquet[:-len('.parquet')] if args.output_parquet.endswith('.parquet') \
               else args.output_parquet
    fold_paths = [f"{out_base}_{fold}_temp.parquet" for fold in folds]
    jx_paths = [f"{out_base}_{fold}_junctions_temp.parquet" for fold in folds] if include_junctions else []

    if args.merge_only:
        existing_folds = [fp for fp in fold_paths if os.path.exists(fp)]
        if not existing_folds:
            sys.exit("ERROR: --merge-only requires existing per-fold temp parquets")
        stream_merge_folds(existing_folds, args.output_parquet)
        existing_jx = [jp for jp in jx_paths if os.path.exists(jp)]
        if existing_jx:
            jx_out = f"{out_base}_junctions.parquet"
            print(f"merging {len(existing_jx)} junctions parquets → {jx_out}", flush=True)
            first = pq.ParquetFile(existing_jx[0])
            with pq.ParquetWriter(jx_out, first.schema_arrow, compression='zstd') as w:
                for jp in existing_jx:
                    pf = pq.ParquetFile(jp)
                    for rg in range(pf.num_row_groups):
                        w.write_table(pf.read_row_group(rg))
        if not args.keep_temp:
            for fp in existing_folds + existing_jx:
                os.remove(fp)
            print("cleaned temp parquets", flush=True)
        return

    with h5py.File(args.input_h5, "r") as fin:
        if is_short:
            chunk_ids = sorted(
                int(k[1:]) for k in fin.keys() if k.startswith("X") and k[1:].isdigit()
            )
            n_windows = sum(fin[f"X{c}"].shape[0] for c in chunk_ids)
            print(f"  16384 mode: {len(chunk_ids)} chunks, {n_windows:,} windows", flush=True)
            entry_keys = chunk_ids
        else:
            n_entries = int(fin.attrs.get('n_entries', 0))
            if n_entries == 0:
                n_entries = sum(1 for k in fin.keys() if k.startswith("SEQ_"))
            entry_keys = list(range(n_entries))
            print(f"  1Mb mode: {n_entries} gene entries", flush=True)

    requested = [
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
    ]
    if include_junctions:
        requested.append(dna_output.OutputType.SPLICE_JUNCTIONS)

    settings = make_organism_settings()
    print(f"organism_settings: "
          f"{'local AG_DATA_DIR' if settings is not None else 'AG defaults (GCS)'}",
          flush=True)

    fold_paths_done = []
    jx_paths_done = []
    ssu_names = jx_names = None

    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '<not set>')
    print(f"JAX_COMPILATION_CACHE_DIR={cache_dir}", flush=True)
    print(f"requested outputs: {[r.name for r in requested]}", flush=True)

    for fi, fold in enumerate(folds):
        print(f"[{fi+1}/{len(folds)}] loading {fold}", flush=True)
        t_l0 = time.perf_counter()
        if args.source == "huggingface":
            model = dna_model.create_from_huggingface(fold, organism_settings=settings)
        else:
            model = dna_model.create_from_kaggle(fold, organism_settings=settings)
        print(f"  model loaded in {time.perf_counter()-t_l0:.1f}s", flush=True)

        # warm-up at scoring length — hits jax compile cache if matching signature already cached.
        # this call IS the phase the watchdog flags as "GPU idle" on a cache miss
        # (XLA compile runs cpu-side; nvidia-smi reads 0% during it).
        t_w0 = time.perf_counter()
        _ = model.predict_sequence(
            sequence="A" * seq_len,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=requested,
            ontology_terms=None,
        )
        t_warmup = time.perf_counter() - t_w0
        cache_state = 'HIT' if t_warmup < 30 else 'MISS — compiled fresh; watchdog may fire'
        print(f"  JIT warmup: {t_warmup:.1f}s — cache {cache_state}", flush=True)

        fold_path = f"{out_base}_{fold}_temp.parquet"
        jx_path = f"{out_base}_{fold}_junctions_temp.parquet" if include_junctions else None
        print(f"  scoring → {fold_path}", flush=True)

        if is_short:
            _, _, ssu_names_, jx_names_ = score_one_fold_chunked(
                model, args.input_h5, entry_keys, fold, fold_path, requested,
            )
        else:
            _, _, ssu_names_, jx_names_ = score_one_fold_streaming(
                model, args.input_h5, entry_keys, fold,
                fold_path, requested, include_junctions, jx_path,
            )
        if ssu_names is None:
            ssu_names = ssu_names_
        if jx_names is None and jx_names_:
            jx_names = jx_names_

        fold_paths_done.append(fold_path)
        if jx_path and os.path.exists(jx_path):
            jx_paths_done.append(jx_path)

        del model
        gc_mod.collect()

    if ssu_names:
        with open(f"{out_base}_ssu_track_names.txt", "w") as f:
            for nm in ssu_names:
                f.write(f"{nm}\n")
        print(f"wrote {len(ssu_names)} ssu track names", flush=True)
    if jx_names:
        with open(f"{out_base}_junctions_track_names.txt", "w") as f:
            for nm in jx_names:
                f.write(f"{nm}\n")
        print(f"wrote {len(jx_names)} junction track names", flush=True)

    if args.no_merge:
        print("--no-merge set; skipping merge (run with --merge-only in CPU job)", flush=True)
        return

    stream_merge_folds(fold_paths_done, args.output_parquet)

    if jx_paths_done:
        jx_out = f"{out_base}_junctions.parquet"
        print(f"merging {len(jx_paths_done)} junctions parquets → {jx_out}", flush=True)
        first = pq.ParquetFile(jx_paths_done[0])
        with pq.ParquetWriter(jx_out, first.schema_arrow, compression='zstd') as w:
            for jp in jx_paths_done:
                pf = pq.ParquetFile(jp)
                for rg in range(pf.num_row_groups):
                    w.write_table(pf.read_row_group(rg))

    if not args.keep_temp:
        for fp in fold_paths_done + jx_paths_done:
            if os.path.exists(fp):
                os.remove(fp)
        print("cleaned temp parquets", flush=True)


if __name__ == "__main__":
    main()
