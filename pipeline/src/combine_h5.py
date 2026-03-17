#!/usr/bin/env python3
import os
import glob
import argparse
import h5py
import logging
from tqdm import tqdm


def combine_h5(input_dir, pattern, output_path):
    # configure logging
    log_path = output_path + ".log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting combine_h5: input_dir={input_dir!r}, pattern={pattern!r}, output={output_path!r}")

    input_pattern = os.path.join(input_dir, pattern)
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        logging.error(f"No files match pattern {input_pattern!r}")
        raise ValueError(f"No files match pattern {input_pattern!r}")

    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Removed existing output file {output_path!r}")

    chunk_idx = 0
    try:
        with h5py.File(output_path, "w") as out_f:
            for in_path in tqdm(input_files, desc="Files", unit="file"):
                logging.info(f"Opening input file {in_path!r}")
                with h5py.File(in_path, "r") as in_f:
                    # detect chunk suffixes from X* keys
                    suffixes = sorted({name[1:] for name in in_f.keys() if name.startswith("X")})
                    for suf in tqdm(suffixes, desc=os.path.basename(in_path), unit="chunk", leave=False):
                        for dset in ("X", "Y", "GC"):
                            old_name = f"{dset}{suf}"
                            if old_name in in_f:
                                new_name = f"{dset}{chunk_idx}"
                                in_f.copy(old_name, out_f, new_name)
                                logging.info(f"Copied {old_name!r} -> {new_name!r}")
                        chunk_idx += 1
        logging.info(f"Done, combined {chunk_idx} chunks into {output_path!r}")
    except Exception:
        logging.exception("Fatal error during combine_h5")
        raise

    print(f"Done, combined {chunk_idx} chunks into '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-chromosome .h5 files into one HDF5")
    parser.add_argument("--input_dir", required=True, help="Directory containing .h5 files")
    parser.add_argument("-p", "--pattern", default="*.h5", help="Glob pattern (default: *.h5)")
    parser.add_argument("--output", required=True, help="Output .h5 file path")
    args = parser.parse_args()

    try:
        combine_h5(args.input_dir, args.pattern, args.output)
    except Exception:
        exit(1)
