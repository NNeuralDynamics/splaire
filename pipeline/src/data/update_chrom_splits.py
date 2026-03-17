#!/usr/bin/env python3
"""
Synchronize ``chroms_*.txt`` with the canonical definitions in nextflow.config.

This replaces the manual editing step from the runbook.  Chromosomes for the
train/valid/test cohorts now live under ``params.chromosome_splits_config_yaml``
and this utility materializes those lists into the text files consumed by the
Nextflow ML jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from pipeline.scripts.config_blocks import default_data_dir, load_yaml_block


def _resolve_path(base: Path, filename: str) -> Path:
    return (base / filename).expanduser().resolve()


def _write_split(name: str, chroms: List[str], path: Path) -> None:
    if not chroms:
        raise ValueError(f"Chromosome list for split '{name}' is empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chroms) + "\n")
    print(f"[{name}] wrote {len(chroms)} chromosomes → {path}")


def run_from_config(cfg: Dict[str, Dict], base_dir: Path) -> None:
    for name, entry in cfg.items():
        file_rel = entry.get("file") or f"chroms_{name}.txt"
        chroms = entry.get("chromosomes", [])
        path = _resolve_path(base_dir, file_rel)
        _write_split(name, chroms, path)


def _load_nextflow_block(config_path: Path, block: str, base_dir: Path | None = None) -> tuple[Dict, Path]:
    cfg = load_yaml_block(config_path, block)
    base = base_dir or default_data_dir(config_path)
    return cfg, base


def main() -> None:
    parser = argparse.ArgumentParser(description="Write chroms_*.txt based on nextflow.config.")
    parser.add_argument("--config", help="Optional manual YAML block instead of nextflow.config.")
    parser.add_argument(
        "--nextflow-config",
        default=str(Path(__file__).resolve().parents[2] / "nextflow.config"),
        help="Path to nextflow.config containing chromosome_splits_config_yaml.",
    )
    parser.add_argument(
        "--config-block",
        default="chromosome_splits_config_yaml",
        help="Name of the $/.../$ block holding the chromosome splits.",
    )
    parser.add_argument(
        "--base-dir",
        help="Override the datasets/data directory when loading from nextflow.config.",
    )
    args = parser.parse_args()

    if args.config:
        import yaml  # Local import to keep dependency optional.

        cfg_path = Path(args.config)
        with cfg_path.open() as handle:
            cfg = yaml.safe_load(handle) or {}
        run_from_config(cfg, cfg_path.parent)
        return

    nf_path = Path(args.nextflow_config).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else None
    cfg, base = _load_nextflow_block(nf_path, args.config_block, base_dir)
    run_from_config(cfg, base)


if __name__ == "__main__":
    main()
