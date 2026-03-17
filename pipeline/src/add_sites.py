#!/usr/bin/env python3
import sys
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

matrix_df_global = None
sample_cols_global = None
matrix_groups_global = None

def parse_matrix_index(index_str):
    """
    Given an index string of the form:
       Region_Strand_Site_site_type
    return a tuple: (region, strand, site (int), site_type)
    """
    try:
        region, strand, site, site_type = index_str.split("_", 3)
        return region, strand, int(site), site_type
    except Exception as e:
        raise ValueError(f"Error parsing matrix index '{index_str}': {e}")

def init_worker(matrix_df, sample_cols):
    """
    Initializes global variables for the worker processes.
    Additionally, pre-group the matrix DataFrame by ('region', 'strand')
    to speed up filtering in each process.
    """
    global matrix_df_global, sample_cols_global, matrix_groups_global
    matrix_df_global = matrix_df
    sample_cols_global = sample_cols
    matrix_groups_global = {
        key: group for key, group in matrix_df_global.groupby(['region', 'strand'])
    }

def process_variant(row):
    chrom     = row["Chromosome"]
    start     = row["Start"]
    end       = row["End"]
    strand    = row["Strand"]
    unique_id = row["Unique_ID"]
    short_id  = unique_id.split("_")[0]

    # find the real sample column
    matches = [c for c in sample_cols_global if c.startswith(short_id)]
    if len(matches) == 1:
        sample = matches[0]
    elif len(matches) > 1:
        sample = matches[0]
        print(f"Warning: multiple columns match '{short_id}': {matches}; using '{sample}'")
    else:
        print(f"Warning: no column starts with '{short_id}'; leaving empty exon values for {unique_id}.")
        return "", "", "", ""

    key = (chrom, strand)
    if key not in matrix_groups_global:
        return "", "", "", ""
    group = matrix_groups_global[key]

    subset = group[(group['site'] >= start) & (group['site'] <= end)]

    exon_end_df = subset[subset['site_type'] == "exon_end"]
    exon_end_positions = exon_end_df['site'].astype(str).tolist()
    exon_end_ssu_values = exon_end_df[sample].astype(str).tolist()

    exon_start_df = subset[subset['site_type'] == "exon_start"]
    exon_start_positions = exon_start_df['site'].astype(str).tolist()
    exon_start_ssu_values = exon_start_df[sample].astype(str).tolist()

    return (
        ",".join(exon_end_positions),
        ",".join(exon_start_positions),
        ",".join(exon_end_ssu_values),
        ",".join(exon_start_ssu_values),
    )


def main(matrix_file, gene_variants_file, output_file):
    print("Loading matrix file...")
    matrix_df = pd.read_csv(matrix_file, sep="\t", index_col=0)

    print("Parsing matrix index and adding new columns (region, strand, site, site_type)...")
    parsed = matrix_df.index.to_series().apply(parse_matrix_index)
    matrix_df['region'] = parsed.apply(lambda x: x[0])
    matrix_df['strand'] = parsed.apply(lambda x: x[1])
    matrix_df['site']   = parsed.apply(lambda x: x[2])
    matrix_df['site_type'] = parsed.apply(lambda x: x[3])

    matrix_df = matrix_df.reset_index().rename(columns={'index': 'event_id'})

    print("Determining sample columns from the matrix...")
    static_cols = {'event_id', 'region', 'strand', 'site', 'site_type'}
    sample_cols = [c for c in matrix_df.columns if c not in static_cols]
    print(f"Found sample columns: {sample_cols}")

    print("Loading gene variants file...")
    gv_df = pd.read_csv(gene_variants_file, sep="\t")

    print("Removing original exon columns if they exist...")
    for col in ["exon_ends", "exon_starts", "exon_end_SSUs", "exon_start_SSUs"]:
        if col in gv_df.columns:
            gv_df.drop(columns=[col], inplace=True)

    print("Converting gene variants DataFrame to a list of records for parallel processing...")
    gv_records = gv_df.to_dict(orient='records')

    print(f"Starting parallel processing for {len(gv_records)} gene variants...")
    results = []
    # limit workers to avoid OOM - each worker gets copy of matrix_df
    n_workers = min(4, len(gv_records) // 10000 + 1)
    print(f"Using {n_workers} worker processes")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker, initargs=(matrix_df, sample_cols)) as executor:
        for res in tqdm(executor.map(process_variant, gv_records, chunksize=1000), total=len(gv_records), desc="Processing gene variants"):
            results.append(res)

    print("Parallel processing complete. Unpacking results...")
    new_exon_ends, new_exon_starts, new_exon_end_SSUs, new_exon_start_SSUs = zip(*results)

    print("Adding new exon columns to gene variants DataFrame...")
    gv_df["exon_ends"] = new_exon_ends
    gv_df["exon_starts"] = new_exon_starts
    gv_df["exon_end_SSUs"] = new_exon_end_SSUs
    gv_df["exon_start_SSUs"] = new_exon_start_SSUs

    print("Saving the updated gene variants file...")
    gv_df.to_csv(output_file, sep="\t", index=False)
    print(f"Updated gene variants saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python add_sites.py <matrix_file.tsv> <gene_variants.tsv> <output_file.tsv>")

    matrix_file = sys.argv[1]
    gene_variants_file = sys.argv[2]
    output_file = sys.argv[3]
    main(matrix_file, gene_variants_file, output_file)
