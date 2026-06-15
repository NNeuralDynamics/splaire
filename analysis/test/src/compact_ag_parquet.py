#!/usr/bin/env python3
"""compact alphagenome parquet — null out ssu_*_track_* values at non-splice-site rows.

we only ever use ssu predictions at splice sites (where y_acceptor>0 OR y_donor>0),
so storing the 736-track ssu values at every position in the coverage window is
wasteful. this script reads a source parquet row-group by row-group, nulls out
ssu_*_track_* columns wherever the position is NOT a splice site, and writes a
compact parquet. parquet's def-level encoding compresses the resulting null runs
to almost nothing.

classification cols (acceptor_fold_X, donor_fold_X) and GT cols (chrom/pos/strand/y_*)
are preserved at ALL positions — we need them for classification metrics.

works on both per-fold _temp.parquet and merged _ag.parquet (same schema family).

usage:
    python compact_ag_parquet.py input.parquet output.parquet
"""
import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


def is_ssu_track_col(name):
    # matches ssu_fold_0_track_0 .. ssu_fold_3_track_735
    return name.startswith("ssu_") and "_track_" in name


def compact(in_path, out_path):
    reader = pq.ParquetFile(in_path)
    n_rg = reader.num_row_groups
    print(f"input: {in_path}  ({n_rg} row groups)", flush=True)
    print(f"output: {out_path}", flush=True)

    schema_names = reader.schema_arrow.names
    ssu_cols = [n for n in schema_names if is_ssu_track_col(n)]
    print(f"ssu track cols: {len(ssu_cols)}", flush=True)
    if not ssu_cols:
        print("WARNING: no ssu_*_track_* cols found — nothing to compact. copying as-is.", flush=True)

    writer = None
    t0 = time.perf_counter()
    n_rows_total = 0
    n_ss_total = 0

    for rg in tqdm(range(n_rg), desc="compact", file=sys.stdout, mininterval=10):
        table = reader.read_row_group(rg)
        if ssu_cols:
            mask = pc.or_(
                pc.greater(table.column("y_acceptor"), 0),
                pc.greater(table.column("y_donor"), 0),
            )
            new_arrays = []
            for name in table.column_names:
                col = table.column(name)
                if is_ssu_track_col(name):
                    null_scalar = pa.scalar(None, type=col.type)
                    col = pc.if_else(mask, col, null_scalar)
                new_arrays.append(col)
            new_table = pa.table(new_arrays, names=table.column_names)
            n_ss_total += pc.sum(pc.cast(mask, pa.int64())).as_py()
        else:
            new_table = table

        if writer is None:
            writer = pq.ParquetWriter(out_path, new_table.schema, compression="zstd")
        writer.write_table(new_table)
        n_rows_total += new_table.num_rows

    if writer:
        writer.close()
    elapsed = time.perf_counter() - t0

    in_size = Path(in_path).stat().st_size
    out_size = Path(out_path).stat().st_size
    print(
        f"\ndone: {n_rows_total:,} rows total, {n_ss_total:,} splice-site rows "
        f"({100*n_ss_total/max(n_rows_total,1):.3f}%)",
        flush=True,
    )
    print(f"elapsed: {elapsed:.0f}s", flush=True)
    print(
        f"size: {in_size/1e9:.1f} GB -> {out_size/1e9:.3f} GB "
        f"({100*out_size/in_size:.2f}% of input)",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_parquet")
    p.add_argument("output_parquet")
    args = p.parse_args()
    compact(args.input_parquet, args.output_parquet)
