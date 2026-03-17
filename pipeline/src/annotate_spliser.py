import pandas as pd
import sys
import os


AMBIGUOUS_TYPES = {"exon_start,both", "exon_end,exon_start,both", "exon_end,exon_start", "both"}

def load_master_list(master_file):
    """
    Loads the master list from a TSV file with columns: Region, Site, Strand, SiteType.
    Returns a dictionary keyed by (Region, Site, Strand) with value SiteType.
    """
    try:
        master_df = pd.read_csv(master_file, sep="\t")
    except Exception as e:
        print(f"Error reading master list file {master_file}: {e}")
        return {}
    
    required_columns = {"Region", "Site", "Strand", "SiteType"}
    if not required_columns.issubset(master_df.columns):
        print(f"Error: Master list file {master_file} is missing required columns. Expected {required_columns}")
        return {}
    
    # Ensure Site column is integer for consistent key matching
    master_df["Site"] = master_df["Site"].astype(int)
    master_dict = {}
    for _, row in master_df.iterrows():
        key = (row["Region"], row["Site"], row["Strand"])
        master_dict[key] = row["SiteType"].strip()
    return master_dict

def annotate_splice_sites(splice_df, master_dict):
    """
    For each splice site, looks up its classification directly in the master list 
    (using the key: (Region, Site, Strand)). 
    If the master list classification is ambiguous (i.e. one of the values in AMBIGUOUS_TYPES),
    this function returns None for that row.
    
    Rows with None in the site_type will later be dropped.
    
    Args:
        splice_df (pd.DataFrame): DataFrame with columns ["Region", "Site", "Strand", ...]
        master_dict (dict): Dictionary mapping (Region, Site, Strand) to SiteType.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a new "site_type" column.
    """
    def classify_site(row):
        key = (row["Region"], row["Site"], row["Strand"])
        classification = master_dict.get(key, None)
        if classification in AMBIGUOUS_TYPES:
            return None
        return classification

    splice_df["site_type"] = splice_df.apply(classify_site, axis=1)
    return splice_df

def adjust_splice_positions(annotated_df):
    """
    Adjusts splice site positions so that:
      - Donor sites (exon ends) always represent the first base of the intron.
      - Acceptor sites (exon starts) always represent the last base of the exon.
      - Sites labeled as "both" remain unchanged.
    
    Args:
        annotated_df (pd.DataFrame): DataFrame with "Site" and "site_type"
    
    Returns:
        pd.DataFrame: Updated DataFrame with adjusted positions.
    """
    adjusted_df = annotated_df.copy()
    adjusted_df.loc[adjusted_df["site_type"] == "exon_start", "Site"] += 1
    return adjusted_df

def filter_duplicates(df):
    """
    Removes any duplicate rows altogether. For each (Region, Site, Strand) group,
    if there is more than one row the entire group is dropped. Only groups with exactly
    one row (unique key) are kept.
    
    Returns the filtered DataFrame.
    """
    filtered_df = df.groupby(["Region", "Site", "Strand"]).filter(lambda g: len(g) == 1)
    return filtered_df

def save_log_file(out_filename, annotated_df, drop_stats, ambiguous_sites, duplicate_sites):
    """
    Saves a log file summarizing site counts and dropped sites.

    Args:
        out_filename (str): The base name of the output file (without extension)
        annotated_df (pd.DataFrame): The DataFrame containing annotated sites.
        drop_stats (dict): Counts of initial, ambiguous, duplicate, and final sites.
        ambiguous_sites (pd.DataFrame): Sites dropped due to ambiguous classification.
        duplicate_sites (pd.DataFrame): Sites dropped due to duplicate keys.
    """
    log_filename = f"{out_filename}.log"
    site_counts = annotated_df["site_type"].value_counts(dropna=True).to_dict()
    strand_counts = annotated_df.groupby(["Strand", "site_type"]).size().unstack(fill_value=0)

    with open(log_filename, "w") as log_file:
        log_file.write("=== Splice Site Annotation Summary ===\n\n")

        # filtering summary
        log_file.write("--- filtering ---\n")
        log_file.write(f"  initial sites: {drop_stats['initial_count']}\n")
        log_file.write(f"  dropped (ambiguous): {drop_stats['ambiguous_count']}\n")
        log_file.write(f"  dropped (duplicate): {drop_stats['duplicate_count']}\n")
        log_file.write(f"  final sites: {drop_stats['final_count']}\n\n")

        # final site type counts
        log_file.write("--- final site types ---\n")
        for site_type, count in site_counts.items():
            log_file.write(f"  {site_type}: {count}\n")
        log_file.write("\n")

        log_file.write("--- breakdown by strand ---\n")
        log_file.write(str(strand_counts))
        log_file.write("\n\n")

        # log dropped ambiguous sites
        if len(ambiguous_sites) > 0:
            log_file.write("--- dropped ambiguous sites ---\n")
            for _, row in ambiguous_sites.head(100).iterrows():
                log_file.write(f"  {row['Region']}:{row['Site']} ({row['Strand']})\n")
            if len(ambiguous_sites) > 100:
                log_file.write(f"  ... and {len(ambiguous_sites) - 100} more\n")
            log_file.write("\n")

        # log dropped duplicate sites
        if len(duplicate_sites) > 0:
            log_file.write("--- dropped duplicate sites ---\n")
            for _, row in duplicate_sites.head(100).iterrows():
                log_file.write(f"  {row['Region']}:{row['Site']} ({row['Strand']})\n")
            if len(duplicate_sites) > 100:
                log_file.write(f"  ... and {len(duplicate_sites) - 100} more\n")
            log_file.write("\n")

    print(f"Log file saved: {log_filename}")

def main(input_file, master_file):
    """
    Reads a SpliSER TSV file, looks up each site's classification from the master list,
    removes ambiguous rows (those with ambiguous classification) and duplicate groups,
    adjusts the splice positions, and then saves the final output.

    Args:
        input_file (str): Path to the SpliSER TSV file.
        master_file (str): Path to the master list TSV file.
    """
    master_dict = load_master_list(master_file)

    try:
        df = pd.read_csv(input_file, sep="\t")
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        sys.exit(1)

    required_columns = {"Region", "Site", "Strand", "Partners"}
    if not required_columns.issubset(df.columns):
        print(f"Error: Missing required columns in {input_file}. Expected {required_columns}")
        sys.exit(1)

    df["Site"] = df["Site"].astype(int)
    initial_count = len(df)

    # annotate and track ambiguous sites
    annotated_df = annotate_splice_sites(df, master_dict)
    ambiguous_mask = annotated_df["site_type"].isna()
    ambiguous_sites = annotated_df[ambiguous_mask].copy()
    annotated_df = annotated_df[~ambiguous_mask]

    adjusted_df = adjust_splice_positions(annotated_df)

    # track duplicate sites before filtering
    dup_counts = adjusted_df.groupby(["Region", "Site", "Strand"]).size()
    duplicate_keys = dup_counts[dup_counts > 1].index.tolist()
    duplicate_sites = adjusted_df.set_index(["Region", "Site", "Strand"]).loc[duplicate_keys].reset_index() if duplicate_keys else pd.DataFrame()

    final_df = filter_duplicates(adjusted_df)

    output_file = os.path.splitext(input_file)[0] + "_annotated.tsv"
    final_df.to_csv(output_file, sep="\t", index=False)
    print(f"Annotated file saved: {output_file}")

    # build drop stats
    drop_stats = {
        "initial_count": initial_count,
        "ambiguous_count": len(ambiguous_sites),
        "duplicate_count": len(duplicate_sites),
        "final_count": len(final_df),
    }
    save_log_file(output_file, final_df, drop_stats, ambiguous_sites, duplicate_sites)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python annotate_spliser.py <SpliSER_TSV_File> <Master_List_TSV_File>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    master_list_filename = sys.argv[2]
    main(input_filename, master_list_filename)

"""
Example usage:
python annotate_spliser.py /path/to/sampele.SpliSER.tsv /path/to/all_sites_master_classed.tsv
"""
