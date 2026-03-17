import os
import argparse
import pandas as pd


def combine_tsv_files(root_dir, suffix):
    # allow suffix with or without ".tsv"
    suf = suffix[:-4] if suffix.endswith(".tsv") else suffix
    needle = f"{suf}.tsv"

    tsv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(needle):
                print("found:", os.path.join(dirpath, filename))
                tsv_files.append(os.path.join(dirpath, filename))

    assert tsv_files, f"no tsv files found in '{root_dir}' ending with '{needle}'"

    # read and concatenate all files
    dfs = [pd.read_csv(f, sep="\t", dtype=str) for f in tsv_files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"combined {len(tsv_files)} files, {len(combined)} total rows")
    return combined


def add_paralog_status(df, paralog_file):
    assert os.path.exists(paralog_file), f"paralogs file not found: {paralog_file}"

    # load paralog file, get set of genes with paralogs
    par = pd.read_csv(paralog_file, sep="\t", dtype=str)
    par = par.dropna(subset=["Human paralogue gene stable ID"])
    paralog_set = set(par["Gene stable ID"])
    print(f"loaded {len(paralog_set)} genes with paralogs from {paralog_file}")

    # strip version from gene id, check membership
    base_ids = df["Gene_ID"].str.split(".", n=1).str[0]
    df["paralog_status"] = base_ids.isin(paralog_set).astype(int)

    # reorder so paralog_status is after Chromosome
    cols = list(df.columns)
    cols.remove("paralog_status")
    insert_idx = cols.index("Chromosome") + 1
    cols.insert(insert_idx, "paralog_status")

    n_para = (df["paralog_status"] == 1).sum()
    print(f"paralog_status: {n_para} rows with paralogs, {len(df) - n_para} without")
    return df[cols]


def main():
    p = argparse.ArgumentParser(description="combine tsv files by suffix and add paralog status")
    p.add_argument("-s", "--suffix", required=True,
                   help="suffix to match (e.g. 'gene_variants')")
    p.add_argument("-d", "--directory", default=".",
                   help="root directory to search")
    p.add_argument("-o", "--output",
                   help="output filename (defaults to combined_${suffix}.tsv)")
    p.add_argument("--paralogs", required=True,
                   help="path to paralogs_GRCh38.txt file")
    args = p.parse_args()

    suf = args.suffix[:-4] if args.suffix.endswith(".tsv") else args.suffix
    out = args.output or f"combined_{suf}.tsv"

    # combine all matching files
    df = combine_tsv_files(args.directory, suf)

    # add paralog status
    df = add_paralog_status(df, args.paralogs)

    # write output
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
