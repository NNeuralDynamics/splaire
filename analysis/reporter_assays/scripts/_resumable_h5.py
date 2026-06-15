"""shared helpers for chunked + resumable h5 scoring.

usage in a score_*.py:

    from _resumable_h5 import ResumableOutput, stream_input, SEQ_KEYS

    score_names = ["acceptor", "donor", ...]
    score_keys = [f"{name}_{site}_{allele}"
                   for name in score_names
                   for site in ("exon_start", "exon_end")
                   for allele in ("ref", "alt")]

    with ResumableOutput(args.output, args.input, n, score_keys,
                          args.batch_size, attrs={"model": "spliceai"}) as (f_out, start_idx):
        for i, j, chunk in stream_input(args.input, args.batch_size, start_idx):
            # chunk = {seq_key: np.ndarray(j-i, L, 4)}
            for site, allele in COMBOS:
                X = chunk[f"{site}_{allele}"]
                preds = predict_fn(X)  # one row per variant per score_name
                for name, vals in preds.items():
                    f_out[f"scores/{name}_{site}_{allele}"][i:j] = vals
            f_out["progress/done"][i:j] = True

each batch is written to disk before the next is fetched, so a job killed mid-run
leaves a `<output>.partial` file with `progress/done[k]==True` for completed
variants. on restart the script finds the first not-done index and continues.
once the loop finishes the partial is finalized (meta copied, progress dropped)
and renamed to the final output path.
"""
import os
import signal
import sys
from pathlib import Path

import h5py
import numpy as np


SEQ_KEYS = ("exon_start_ref", "exon_start_alt", "exon_end_ref", "exon_end_alt")
COMBOS = [("exon_start", "ref"), ("exon_start", "alt"),
          ("exon_end", "ref"), ("exon_end", "alt")]


class GracefulShutdown(Exception):
    """raised after SIGTERM - tells ResumableOutput to keep the .partial file"""


_term_received = False


def _sigterm_handler(signum, frame):
    global _term_received
    _term_received = True
    print(f"received signal {signum}, will exit cleanly at next batch boundary", flush=True)


def install_signal_handlers():
    """call once at startup. catches SIGTERM (slurm walltime hit) and SIGINT (ctrl-c)."""
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)


def should_stop():
    return _term_received


def run_main(main_fn):
    """wrap main() so a GracefulShutdown exits 0 cleanly (slurm sees success, .partial preserved)"""
    install_signal_handlers()
    try:
        main_fn()
    except GracefulShutdown:
        print("graceful shutdown - partial output preserved for resume", flush=True)
        sys.exit(0)


class ResumableOutput:
    """context manager for an output h5 that supports resume after job kill.

    creates `<output>.partial` while running. on clean exit:
      - copies meta/* from the input h5
      - sets attrs (caller-supplied + n_variants + seq_len)
      - drops the `progress/` group
      - renames `.partial` -> final output path

    if an exception propagates, the partial is left in place for resume.
    """

    def __init__(self, output_path, input_path, n, score_keys, batch_size, attrs=None):
        self.output_path = Path(output_path)
        self.partial = Path(str(output_path) + ".partial")
        self.input_path = input_path
        self.n = n
        self.score_keys = list(score_keys)
        self.batch_size = batch_size
        self.attrs = dict(attrs or {})
        self.f_out = None
        self.start_idx = 0

    def __enter__(self):
        if self.output_path.exists():
            raise RuntimeError(f"output already exists: {self.output_path}")

        chunks = (min(self.batch_size, self.n),)

        if self.partial.exists():
            self.f_out = h5py.File(self.partial, "a")
            # validate schema matches; if not, refuse to resume
            missing = [k for k in self.score_keys if f"scores/{k}" not in self.f_out]
            if missing or "progress/done" not in self.f_out:
                self.f_out.close()
                raise RuntimeError(
                    f"existing {self.partial} has incompatible schema "
                    f"(missing scores/{missing[0]} or progress/done). "
                    f"delete it to start fresh."
                )
            done = self.f_out["progress/done"][:]
            if done.all():
                self.start_idx = self.n
                print(f"all {self.n:,} variants already scored, finalizing", flush=True)
            else:
                self.start_idx = int(np.argmin(done))
                print(f"resuming from variant {self.start_idx:,} of {self.n:,}", flush=True)
        else:
            self.f_out = h5py.File(self.partial, "w")
            scores_grp = self.f_out.create_group("scores")
            for name in self.score_keys:
                scores_grp.create_dataset(
                    name, shape=(self.n,), dtype=np.float32,
                    fillvalue=np.nan, chunks=chunks,
                    compression="gzip", compression_opts=4,
                )
            self.f_out.create_dataset(
                "progress/done", shape=(self.n,), dtype=bool,
                fillvalue=False, chunks=chunks,
                compression="gzip", compression_opts=4,
            )
            self.start_idx = 0
            print(f"starting fresh: {self.partial}  n={self.n:,}", flush=True)

        return self.f_out, self.start_idx

    def __exit__(self, exc_type, exc, tb):
        if self.f_out is None:
            return False

        if exc_type is not None:
            # error during run - flush and keep partial for resume
            try:
                self.f_out.flush()
            except Exception:
                pass
            self.f_out.close()
            print(f"error during scoring; partial output preserved at {self.partial}", flush=True)
            return False

        # finalize: copy meta + attrs, drop progress, rename
        if "meta" not in self.f_out:
            meta_grp = self.f_out.create_group("meta")
            with h5py.File(self.input_path, "r") as f_in:
                for k in f_in["meta"]:
                    meta_grp.create_dataset(k, data=f_in["meta"][k][:])
                input_attrs = dict(f_in.attrs)
        else:
            with h5py.File(self.input_path, "r") as f_in:
                input_attrs = dict(f_in.attrs)

        for k, v in self.attrs.items():
            self.f_out.attrs[k] = v
        self.f_out.attrs["n_variants"] = self.n
        if "seq_len" in input_attrs:
            self.f_out.attrs["seq_len"] = input_attrs["seq_len"]

        if "progress" in self.f_out:
            del self.f_out["progress"]

        self.f_out.close()
        self.partial.rename(self.output_path)
        print(f"wrote {self.output_path}", flush=True)
        return False


def load_exon_ids(input_path):
    """return per-row exon_id array (event_id for compass, ensembl_exon_id for
    opensplice), or None if neither is present. used by score_sa/pang/spt/splaire
    to cache reference predictions per exon (variants sharing an exon have
    identical ref windows so the model only needs to score each ref once)."""
    with h5py.File(input_path, "r") as f:
        if "meta" not in f:
            return None
        if "event_id" in f["meta"]:
            return f["meta/event_id"][:]
        if "ensembl_exon_id" in f["meta"]:
            return f["meta/ensembl_exon_id"][:]
    return None


def stream_input(input_path, batch_size, start_idx):
    """yield (i, j, chunk_dict) tuples - lazy h5 reads, never loads the full
    dataset. chunk_dict has all four SEQ_KEYS sliced to [i:j].
    """
    with h5py.File(input_path, "r") as f:
        seqs = {k: f[f"seqs/{k}"] for k in SEQ_KEYS}
        n = seqs[SEQ_KEYS[0]].shape[0]
        for i in range(start_idx, n, batch_size):
            j = min(i + batch_size, n)
            chunk = {k: seqs[k][i:j] for k in SEQ_KEYS}
            yield i, j, chunk
