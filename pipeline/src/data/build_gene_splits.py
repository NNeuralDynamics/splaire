#!/usr/bin/env python3
"""generate transcript-level splits from config"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from pipeline.scripts.config_blocks import default_data_dir, load_yaml_block


def _resolve_path(base: pathlib.Path, path_str: str) -> pathlib.Path:
    return (base / path_str).expanduser().resolve()


def _load_table(path: pathlib.Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    return pd.read_csv(path, sep=sep)


def _normalize_chrom(series: pd.Series, prefix: str) -> pd.Series:
    core = series.astype(str)
    if prefix:
        core = core.str.replace(prefix, "", regex=False)
    return core


def _filter_autosomes(df: pd.DataFrame, chrom_col: str) -> pd.DataFrame:
    def is_auto(value: str) -> bool:
        try:
            val = int(str(value).replace("chr", ""))  # tolerate literal "chr"
            return 1 <= val <= 22
        except ValueError:
            return False

    mask = df[chrom_col].apply(is_auto)
    return df[mask].copy().reset_index(drop=True)


def _nonparalog_folds(job: Dict, base: pathlib.Path) -> None:
    table = _load_table(_resolve_path(base, job["input_tsv"]), job.get("sep", "\t"))
    chrom_col = job.get("chromosome_column", "Chromosome")
    paralog_col = job.get("paralog_column", "paralog_status")

    if job.get("autosomes_only", True):
        table = _filter_autosomes(table, chrom_col)

    chrom_core = _normalize_chrom(table[chrom_col], job.get("strip_prefix", "chr"))
    table = table.assign(__chrom_core=chrom_core)

    excl = set(str(x) for x in job.get("exclude_nonparalog_chromosomes", []))
    if excl:
        mask = ~((table[paralog_col] == 0) & (table["__chrom_core"].isin(excl)))
        table = table[mask].copy()

    output_dir = _resolve_path(base, job.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    paralog_df = table[table[paralog_col] == 1].drop(columns="__chrom_core").reset_index(drop=True)
    non_para_df = table[table[paralog_col] == 0].drop(columns="__chrom_core").reset_index(drop=True)

    paralog_path = output_dir / job.get("paralog_output", "main_train_paralogs.tsv")
    paralog_df.to_csv(paralog_path, sep=job.get("sep", "\t"), index=False)
    print(f"[{job['name']}] Saved {len(paralog_df)} paralog rows → {paralog_path}")

    n_nonpara = len(non_para_df)
    val_frac = job.get("validation_fraction", 0.1)
    target_val = int(n_nonpara * val_frac)

    n_splits = job.get("n_splits", 5)
    seed_base = job.get("seed_base", 42)

    val_tpl = job.get("validation_template", "split{split}_validation.tsv")
    train_tpl = job.get("train_template", "split{split}_nonparalog_train.tsv")

    for split in range(1, n_splits + 1):
        rng = np.random.default_rng(seed_base + split)
        idx = np.arange(n_nonpara)
        val_idx = rng.choice(idx, size=target_val, replace=False)
        mask = np.ones(n_nonpara, dtype=bool)
        mask[val_idx] = False

        val_df = non_para_df.iloc[val_idx].reset_index(drop=True)
        train_df = non_para_df.iloc[mask].reset_index(drop=True)

        val_path = output_dir / val_tpl.format(split=split)
        train_path = output_dir / train_tpl.format(split=split)

        val_df.to_csv(val_path, sep=job.get("sep", "\t"), index=False)
        train_df.to_csv(train_path, sep=job.get("sep", "\t"), index=False)

        print(f"[{job['name']}] split {split}: {len(val_df)} val rows → {val_path}")
        print(f"[{job['name']}] split {split}: {len(train_df)} train rows → {train_path}")


def _chromosome_holdout(job: Dict, base: pathlib.Path) -> None:
    table = _load_table(_resolve_path(base, job["input_tsv"]), job.get("sep", "\t"))
    chrom_col = job.get("chromosome_column", "Chromosome")
    paralog_col = job.get("paralog_column", "paralog_status")

    if job.get("autosomes_only", True):
        table = _filter_autosomes(table, chrom_col)

    chrom_core = _normalize_chrom(table[chrom_col], job.get("strip_prefix", "chr"))
    table = table.assign(__chrom_core=chrom_core)

    output_dir = _resolve_path(base, job.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    test_chroms = {str(x) for x in job.get("test_chromosomes", [])}
    test_mask = (table["__chrom_core"].isin(test_chroms)) & (table[paralog_col] == 0)
    test_df = table[test_mask].drop(columns="__chrom_core").reset_index(drop=True)

    remaining = table[~test_mask].reset_index(drop=True)
    target_val = int(len(remaining) * job.get("validation_fraction", 0.1))

    rng = np.random.default_rng(job.get("seed", 42))
    non_para = remaining[remaining[paralog_col] == 0]
    val_indices: List[int] = []

    if job.get("ensure_validation_per_chromosome", False):
        for chrom in sorted(non_para["__chrom_core"].unique()):
            chrom_rows = non_para[non_para["__chrom_core"] == chrom]
            if not chrom_rows.empty:
                chosen = rng.choice(chrom_rows.index.to_numpy())
                val_indices.append(int(chosen))

    needed = max(0, target_val - len(val_indices))
    pool = [idx for idx in non_para.index.tolist() if idx not in val_indices]
    if needed > len(pool):
        raise ValueError("Not enough non-paralog rows to satisfy validation fraction")
    extra = rng.choice(pool, size=needed, replace=False).tolist()
    val_indices.extend(extra)

    val_df = remaining.loc[val_indices].drop(columns="__chrom_core").reset_index(drop=True)
    train_df = remaining.drop(index=val_indices).drop(columns="__chrom_core").reset_index(drop=True)

    test_path = output_dir / job.get("test_filename", "sample100_test_set.tsv")
    val_path = output_dir / job.get("validation_filename", "sample100_validation_set.tsv")
    train_path = output_dir / job.get("train_filename", "sample100_training_set.tsv")

    test_df.to_csv(test_path, sep=job.get("sep", "\t"), index=False)
    val_df.to_csv(val_path, sep=job.get("sep", "\t"), index=False)
    train_df.to_csv(train_path, sep=job.get("sep", "\t"), index=False)

    print(f"[{job['name']}] test set: {len(test_df)} rows → {test_path}")
    print(f"[{job['name']}] validation set: {len(val_df)} rows → {val_path}")
    print(f"[{job['name']}] training set: {len(train_df)} rows → {train_path}")


def run_from_config(cfg: Dict, base: pathlib.Path) -> None:

    for job in cfg.get("jobs", []):
        mode = job["mode"]
        if mode == "nonparalog_folds":
            _nonparalog_folds(job, base)
        elif mode == "chromosome_holdout":
            _chromosome_holdout(job, base)
        else:
            raise ValueError(f"Unknown job mode: {mode}")


def run(config_path: pathlib.Path) -> None:
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    run_from_config(cfg, config_path.parent)


def _load_nextflow_config(path: pathlib.Path, block: str, base_dir: pathlib.Path | None = None) -> tuple[Dict, pathlib.Path]:
    cfg = load_yaml_block(path, block)
    base = base_dir or default_data_dir(path)
    return cfg, base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build transcript-level splits from config or pipeline/nextflow.config."
    )
    parser.add_argument("--config", help="Path to gene_split_config.yaml")
    parser.add_argument(
        "--nextflow-config",
        default=str(pathlib.Path(__file__).resolve().parents[2] / "nextflow.config"),
        help="Path to nextflow.config containing gene_split_config_yaml.",
    )
    parser.add_argument(
        "--config-block",
        default="gene_split_config_yaml",
        help="Name of the $/.../$ block holding the gene split jobs.",
    )
    parser.add_argument(
        "--base-dir",
        help="Override the relative-path base directory when loading from nextflow.config (defaults to datasets/data).",
    )
    args = parser.parse_args()

    if args.config:
        run(pathlib.Path(args.config))
        return

    nf_path = pathlib.Path(args.nextflow_config).resolve()
    base_dir = pathlib.Path(args.base_dir).resolve() if args.base_dir else None
    cfg, base = _load_nextflow_config(nf_path, args.config_block, base_dir)
    run_from_config(cfg, base)


if __name__ == "__main__":
    main()
