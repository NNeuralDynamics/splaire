#!/usr/bin/env python3
"""download eqtl catalogue data for sqtl benchmark"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

max_retries = 10
retry_delay = 10

urls = {
    "tabix": "https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/tabix/tabix_ftp_paths.tsv",
    "gencode": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.basic.annotation.gtf.gz",
    "gtex_tpm": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
}

# zenodo phenotype metadata (all leafcutter-detected introns per tissue)
zenodo_base = "https://zenodo.org/records/7850746/files"

files = {
    "tabix": "tabix_ftp_paths.tsv",
    "gencode": "gencode.v39.basic.annotation.gtf.gz",
    "gtex_tpm": "GTEx_v8_median_tpm.gct.gz",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="base output directory")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = ap.parse_args()

    out = Path(args.out_dir)
    ref = out / "reference"
    ref.mkdir(parents=True, exist_ok=True)

    print(f"output dir: {out}")
    print(f"workers: {args.workers}")
    print()

    # reference files (small, download sequentially first)
    print("reference files")
    for key in urls:
        download(urls[key], ref / files[key])
    print()

    # check tabix exists
    tabix = ref / files["tabix"]
    if not tabix.exists():
        print("error: tabix paths not found, cannot download eqtl data")
        return

    # build download list
    df = pd.read_csv(tabix, sep="\t")
    downloads = []

    # datasets to download: (quant_method, output_dir)
    datasets = [
        ("txrev", "txrevise"),
        ("leafcutter", "leafcutter"),
        ("ge", "ge"),
    ]

    for quant_method, dir_name in datasets:
        mask = df["study_label"].str.contains("GTEx", case=False) & (df["quant_method"] == quant_method)
        subset = df[mask]
        print(f"found {len(subset)} GTEx {quant_method} tissues")

        for _, row in subset.iterrows():
            tissue = row["sample_group"]
            dataset_id = row["dataset_id"]

            # credible sets (always download - small)
            cs_url = row.get("ftp_cs_path")
            if pd.notna(cs_url) and cs_url:
                downloads.append((
                    cs_url,
                    out / dir_name / "raw" / f"{dataset_id}_{tissue}.credible_sets.tsv.gz"
                ))

            # summary stats
            ss_url = row.get("ftp_path")
            if pd.notna(ss_url) and ss_url:
                downloads.append((
                    ss_url,
                    out / dir_name / "sumstats" / f"{dataset_id}_{tissue}.all.tsv.gz"
                ))

            # phenotype metadata from zenodo (leafcutter only - complete intron lists)
            if quant_method == "leafcutter":
                pheno_url = f"{zenodo_base}/leafcutter_{dataset_id}_Ensembl_105_phenotype_metadata.tsv.gz"
                downloads.append((
                    pheno_url,
                    out / dir_name / "phenotype_metadata" / f"{dataset_id}_{tissue}.phenotype_metadata.tsv.gz"
                ))

    print()

    # filter to pending downloads
    pending = [(url, path) for url, path in downloads if not path.exists()]
    existing = len(downloads) - len(pending)

    print("download summary")
    print(f"total files: {len(downloads)}")
    print(f"already exist: {existing}")
    print(f"to download: {len(pending)}")

    if not pending:
        print("\nall files already downloaded!")
        return

    # create all directories
    for _, path in pending:
        path.parent.mkdir(parents=True, exist_ok=True)

    # parallel download with retry loop for failures
    start = time.time()
    round_num = 1

    while pending:
        print(f"\nround {round_num}: downloading {len(pending)} files with {args.workers} workers")

        failed_this_round = []
        success = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download, url, path): (url, path) for url, path in pending}
            for future in as_completed(futures):
                url, path = futures[future]
                try:
                    if future.result():
                        success += 1
                    else:
                        failed_this_round.append((url, path))
                except Exception as e:
                    print(f"  error {path.name}: {e}")
                    failed_this_round.append((url, path))

        print(f"\nround {round_num} complete: {success} success, {len(failed_this_round)} failed")

        if failed_this_round:
            print(f"retrying {len(failed_this_round)} failed files in 30 seconds...")
            time.sleep(30)
            pending = failed_this_round
            round_num += 1
        else:
            pending = []

    elapsed = time.time() - start
    print(f"\nall done in {elapsed/60:.1f} min")


def download(url, out_path):
    """download file with retry"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return True

    for attempt in range(max_retries):
        result = subprocess.run(
            ["curl", "-f", "-L", "-s", "--progress-bar", "-o", str(out_path), url],
            capture_output=False
        )
        if result.returncode == 0:
            size_mb = out_path.stat().st_size / 1e6
            print(f"  {out_path.name} ({size_mb:.1f} MB)")
            return True

        out_path.unlink(missing_ok=True)
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    print(f"  failed: {out_path.name}")
    return False


if __name__ == "__main__":
    main()
