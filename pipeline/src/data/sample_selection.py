#!/usr/bin/env python3
"""
Reproduce the pick_samples*.ipynb workflow using a YAML configuration.

The canonical configuration now lives inside ``pipeline/nextflow.config`` under
``params.sample_selection_config_yaml``.  This script can still ingest a
standalone YAML file via ``--config`` for ad-hoc experiments, but publication
runs should invoke it as ``python sample_selection.py --nextflow-config
pipeline/nextflow.config`` so that the per-paper selections stay in sync with
Nextflow parameters.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import yaml

from pipeline.scripts.config_blocks import default_data_dir, load_yaml_block


def _read_sample_file(path: pathlib.Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Sample list not found: {path}")
    data = pd.read_csv(path, header=None, names=["sample"])
    return set(data["sample"].astype(str).str.strip())


def _write_sample_file(samples: List[str], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(samples, name="sample").to_csv(path, index=False, header=False)


def _resolve_path(base: pathlib.Path, maybe_path: str) -> pathlib.Path:
    return (base / maybe_path).expanduser().resolve()


def _load_membership_lists(base: pathlib.Path, cfg: Dict) -> Dict[str, Set[str]]:
    lists = {}
    for name, entry in cfg.get("membership_lists", {}).items():
        lists[name] = _read_sample_file(_resolve_path(base, entry["path"]))
    return lists


def _union_sets(sample_sets: Dict[str, Set[str]], names: List[str]) -> Set[str]:
    out: Set[str] = set()
    for name in names:
        if name not in sample_sets:
            raise KeyError(f"Unknown sample set referenced: {name}")
        out.update(sample_sets[name])
    return out


def _apply_membership_filter(df: pd.DataFrame, column: str, lists: Dict[str, Set[str]], names: List[str]) -> pd.DataFrame:
    if not names:
        return df
    mask = pd.Series(True, index=df.index)
    for name in names:
        if name not in lists:
            raise KeyError(f"Unknown membership list '{name}' in config")
        mask &= df[column].isin(lists[name])
    return df[mask].copy()


def _ensure_enough(pool: pd.DataFrame, strat_col: str, quotas: Dict[str, int]) -> None:
    for label, count in quotas.items():
        available = (pool[strat_col] == label).sum()
        if available < count:
            raise ValueError(f"Not enough samples for stratum '{label}': requested {count}, only {available} available")


def run_from_config(cfg: Dict, base_dir: pathlib.Path) -> None:
    metadata_path = _resolve_path(base_dir, cfg["metadata"]["csv"])
    df = pd.read_csv(metadata_path)
    sample_column = cfg["metadata"]["sample_column"]
    strata_column = cfg["metadata"].get("strata_column")

    df[sample_column] = df[sample_column].astype(str).str.strip()

    sample_sets: Dict[str, Set[str]] = {}
    if "existing_sets" in cfg:
        for name, entry in cfg["existing_sets"].items():
            sample_sets[name] = _read_sample_file(_resolve_path(base_dir, entry["path"]))

    membership_lists = _load_membership_lists(base_dir, cfg)

    for action in cfg.get("actions", []):
        action_type = action["type"]
        if action_type == "stratified_pick":
            if not strata_column:
                raise ValueError("stratified_pick requires 'strata_column' in metadata config")
            pool = df.copy()
            exclude = _union_sets(sample_sets, action.get("exclude_sets", []))
            if exclude:
                pool = pool[~pool[sample_column].isin(exclude)]

            for col in action.get("drop_missing_values", []):
                pool = pool[pool[col].notna()]

            if action.get("allowed_strata"):
                pool = pool[pool[strata_column].isin(action["allowed_strata"])]

            pool = _apply_membership_filter(pool, sample_column, membership_lists, action.get("require_membership", []))

            quotas = action["quotas"]
            _ensure_enough(pool, strata_column, quotas)

            rng = np.random.default_rng(action.get("seed", 42))
            selections = []
            for label, count in quotas.items():
                subset = pool[pool[strata_column] == label]
                idx = rng.choice(subset.index.to_numpy(), size=count, replace=False)
                selections.append(subset.loc[idx])

            pick_df = pd.concat(selections).reset_index(drop=True)
            samples = sorted(pick_df[sample_column].tolist())
            output = _resolve_path(base_dir, action["output"])
            _write_sample_file(samples, output)
            sample_sets[action["name"]] = set(samples)
            print(f"[stratified_pick] Wrote {len(samples)} samples → {output}")

        elif action_type == "train_from_list":
            source_name = action["source_list"]
            if source_name in membership_lists:
                pool = set(membership_lists[source_name])
            else:
                pool = _read_sample_file(_resolve_path(base_dir, source_name))

            exclude = _union_sets(sample_sets, action.get("exclude_sets", []))
            pool -= exclude

            for name in action.get("require_membership", []):
                if name not in membership_lists:
                    raise KeyError(f"Unknown membership list '{name}'")
                pool &= membership_lists[name]

            for drop in action.get("manual_remove", []):
                pool.discard(drop)

            pool_list = sorted(pool)
            target_size = action.get("target_size", len(pool_list))
            if target_size > len(pool_list):
                raise ValueError(f"Not enough candidates in {source_name}: need {target_size}, only {len(pool_list)} available")

            rng = np.random.default_rng(action.get("seed", 42))
            idx = rng.choice(len(pool_list), size=target_size, replace=False)
            selected = sorted(pool_list[i] for i in idx)

            output = _resolve_path(base_dir, action["output"])
            _write_sample_file(selected, output)
            sample_sets[action["name"]] = set(selected)
            print(f"[train_from_list] Wrote {len(selected)} samples → {output}")

        else:
            raise ValueError(f"Unsupported action type: {action_type}")

    for export in cfg.get("tsv_exports", []):
        source = _resolve_path(base_dir, export["source"])
        if not source.exists():
            raise FileNotFoundError(f"TSV source not found: {source}")
        df = pd.read_csv(source, sep=export.get("sep", "\t"))

        sample_set_name = export["sample_set"]
        if sample_set_name not in sample_sets:
            raise KeyError(f"Unknown sample set '{sample_set_name}' in TSV export")
        keep = sample_sets[sample_set_name]

        sample_column_name = export.get("sample_column", "Unique_ID")
        splitter = export.get("sample_parser", {"type": "split", "sep": "_", "index": 0})
        if splitter["type"] == "split":
            col = df[sample_column_name].astype(str).str.split(splitter.get("sep", "_")).str[splitter.get("index", 0)]
        else:
            raise ValueError(f"Unsupported sample_parser type: {splitter['type']}")

        df = df.assign(__sample=col)
        subset = df[df["__sample"].isin(keep)].copy()
        if not export.get("keep_helper_column", False):
            subset = subset.drop(columns="__sample")

        output = _resolve_path(base_dir, export["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(output, sep=export.get("sep", "\t"), index=False)
        print(f"[tsv_export] {len(subset)} rows → {output}")


def run(config_path: pathlib.Path) -> None:
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    base_dir = config_path.parent
    run_from_config(cfg, base_dir)


def _load_nextflow_config(path: pathlib.Path, block: str, base_dir: pathlib.Path | None = None) -> tuple[Dict, pathlib.Path]:
    cfg = load_yaml_block(path, block)
    base = base_dir or default_data_dir(path)
    return cfg, base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate the HAEC sample selection pipeline from YAML or the embedded Nextflow config."
    )
    parser.add_argument("--config", help="Path to sample_selection_config.yaml (optional when using --nextflow-config).")
    parser.add_argument(
        "--nextflow-config",
        help="Path to pipeline/nextflow.config containing sample_selection_config_yaml.",
        default=str(pathlib.Path(__file__).resolve().parents[2] / "nextflow.config"),
    )
    parser.add_argument(
        "--config-block",
        default="sample_selection_config_yaml",
        help="Name of the $/.../$ block inside nextflow.config to read.",
    )
    parser.add_argument(
        "--base-dir",
        help="Override the base directory for relative paths when consuming nextflow.config (defaults to datasets/data).",
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
