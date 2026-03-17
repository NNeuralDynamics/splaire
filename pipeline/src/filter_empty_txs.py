#!/usr/bin/env python3
"""filter rows where either SSU column is all-777, run before fill_gencode_sites"""
import argparse
import pandas as pd


def is_all_777(val):
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return True
    parts = [x.strip() for x in str(val).split(",") if x.strip()]
    return all(p in {"777", "777.0"} for p in parts)


def main():
    parser = argparse.ArgumentParser(description="filter rows with all-777 SSU values")
    parser.add_argument("--input", required=True, help="input TSV with SSU values")
    parser.add_argument("--output", required=True, help="output filtered TSV")
    args = parser.parse_args()

    print(f"loading {args.input}")
    df = pd.read_csv(args.input, sep="\t")
    n_before = len(df)

    # filter rows where EITHER SSU column is all-777
    donor_all_777 = df["exon_end_SSUs"].apply(is_all_777)
    acc_all_777 = df["exon_start_SSUs"].apply(is_all_777)
    mask_valid = ~(donor_all_777 | acc_all_777)

    df = df[mask_valid]
    n_after = len(df)
    n_dropped = n_before - n_after

    print(f"filtered {n_dropped} rows with all-777 SSUs ({n_before} -> {n_after})")

    df.to_csv(args.output, sep="\t", index=False)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
