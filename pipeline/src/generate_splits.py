#!/usr/bin/env python3
"""generate train/valid/test splits from a splits config yaml file."""

import argparse
import shutil
from pathlib import Path

import yaml


def expand_path(path, project_dir):
    if not path:
        return path
    return path.replace("${projectDir}", project_dir)


def main():
    parser = argparse.ArgumentParser(description="generate train/valid/test splits")
    parser.add_argument("--config", required=True, help="path to splits config yaml")
    parser.add_argument("--output-dir", default=".", help="output directory for split files")
    parser.add_argument("--project-dir", default=".", help="project directory for expanding paths")
    args = parser.parse_args()

    # freeze config by copying to output dir
    out_dir = Path(args.output_dir)
    frozen_config = out_dir / "splits_config.yaml"
    if Path(args.config).resolve() != frozen_config.resolve():
        shutil.copy(args.config, frozen_config)
        print(f"frozen config: {frozen_config}")

    # load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    samples_cfg = config.get("samples", {})
    chroms_cfg = config.get("chromosomes", {})
    generate_mode = config.get("generate", "all")

    # determine which splits to create
    make_train = generate_mode in ["all", "train_valid", "train_only"]
    make_valid = generate_mode in ["all", "train_valid"]
    make_test = generate_mode in ["all", "test_only"]

    # read sample files (expand ${projectDir} in paths)
    train_file = expand_path(samples_cfg.get("train_file"), args.project_dir)
    valid_file = expand_path(samples_cfg.get("valid_file"), args.project_dir)
    test_file = expand_path(samples_cfg.get("test_file"), args.project_dir)

    train_samples = []
    valid_samples = []
    test_samples = []

    if train_file and make_train:
        with open(train_file) as f:
            train_samples = [s.strip() for s in f.read().strip().split("\n") if s.strip()]

    if valid_file and make_valid:
        with open(valid_file) as f:
            valid_samples = [s.strip() for s in f.read().strip().split("\n") if s.strip()]

    if test_file and make_test:
        with open(test_file) as f:
            test_samples = [s.strip() for s in f.read().strip().split("\n") if s.strip()]

    # write sample files
    out_dir = args.output_dir
    with open(f"{out_dir}/train_samples.txt", "w") as f:
        if train_samples:
            f.write("\n".join(train_samples) + "\n")

    with open(f"{out_dir}/valid_samples.txt", "w") as f:
        if valid_samples:
            f.write("\n".join(valid_samples) + "\n")

    with open(f"{out_dir}/test_samples.txt", "w") as f:
        if test_samples:
            f.write("\n".join(test_samples) + "\n")

    # chromosome splits
    train_chroms = chroms_cfg.get("train", []) if make_train else []
    valid_chroms = chroms_cfg.get("valid", []) if make_valid else []
    test_chroms = chroms_cfg.get("test", []) if make_test else []

    with open(f"{out_dir}/train_chroms.txt", "w") as f:
        if train_chroms:
            f.write("\n".join(train_chroms) + "\n")

    with open(f"{out_dir}/valid_chroms.txt", "w") as f:
        if valid_chroms:
            f.write("\n".join(valid_chroms) + "\n")

    with open(f"{out_dir}/test_chroms.txt", "w") as f:
        if test_chroms:
            f.write("\n".join(test_chroms) + "\n")

    # write summary
    with open(f"{out_dir}/splits_summary.txt", "w") as f:
        f.write(f"generate_mode: {generate_mode}\n")
        f.write(f"train_samples: {len(train_samples)}\n")
        f.write(f"valid_samples: {len(valid_samples)}\n")
        f.write(f"test_samples: {len(test_samples)}\n")
        f.write(f"train_chroms: {len(train_chroms)}\n")
        f.write(f"valid_chroms: {len(valid_chroms)}\n")
        f.write(f"test_chroms: {len(test_chroms)}\n")

    # parse dataset options
    dataset_cfg = config.get("dataset", {})
    variant = dataset_cfg.get("variant", "single")
    paralog = dataset_cfg.get("paralog", "all")
    make_gc = dataset_cfg.get("make_gc", False)
    remove_missing = dataset_cfg.get("remove_missing", False)
    reference = dataset_cfg.get("reference", False)
    asymmetric_paralog_chroms = dataset_cfg.get("asymmetric_paralog_chroms", False)

    # parse fill_gencode options (per-split)
    fill_gencode_cfg = config.get("fill_gencode", {})
    fill_gencode_train = fill_gencode_cfg.get("train", True)
    fill_gencode_valid = fill_gencode_cfg.get("valid", True)
    fill_gencode_test = fill_gencode_cfg.get("test", True)

    # write dataset options
    with open(f"{out_dir}/dataset_options.txt", "w") as f:
        f.write(f"variant={variant}\n")
        f.write(f"paralog={paralog}\n")
        f.write(f"make_gc={make_gc}\n")
        f.write(f"remove_missing={remove_missing}\n")
        f.write(f"reference={reference}\n")
        f.write(f"asymmetric_paralog_chroms={asymmetric_paralog_chroms}\n")
        f.write(f"fill_gencode_train={fill_gencode_train}\n")
        f.write(f"fill_gencode_valid={fill_gencode_valid}\n")
        f.write(f"fill_gencode_test={fill_gencode_test}\n")

    # parse parallel options
    parallel_cfg = config.get("parallel", {})
    parallel_by = parallel_cfg.get("by", "donor")
    save_individual = parallel_cfg.get("save_individual", True)

    # write parallel options
    with open(f"{out_dir}/parallel_options.txt", "w") as f:
        f.write(f"parallel_by={parallel_by}\n")
        f.write(f"save_individual={save_individual}\n")

    print(f"generate mode: {generate_mode}")
    print(f"train: {len(train_samples)} samples, {len(train_chroms)} chromosomes")
    print(f"valid: {len(valid_samples)} samples, {len(valid_chroms)} chromosomes")
    print(f"test: {len(test_samples)} samples, {len(test_chroms)} chromosomes")
    print(f"dataset: variant={variant}, paralog={paralog}, make_gc={make_gc}, remove_missing={remove_missing}")
    print(f"fill_gencode: train={fill_gencode_train}, valid={fill_gencode_valid}, test={fill_gencode_test}")
    print(f"parallel: by={parallel_by}, save_individual={save_individual}")


if __name__ == "__main__":
    main()
