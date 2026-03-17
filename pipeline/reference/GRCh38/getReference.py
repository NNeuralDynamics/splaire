import os
import requests
import argparse

def download_file(url, local_filename):
    """
    Download a file from a URL to a local directory.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def main():
    parser = argparse.ArgumentParser(description="Download GTF and/or reference genome files.")
    parser.add_argument("--version", required=True, help="GTF version number (e.g., '29').")
    parser.add_argument("--download", choices=['gtf', 'ref', 'both'], default='both', help="Specify which files to download: 'gtf', 'ref', or 'both'.")
    parser.add_argument("--output_dir", default='.', help="Directory to save downloaded files. Defaults to the current directory.")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fasta_url = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{args.version}/GRCh38.primary_assembly.genome.fa.gz"
    gtf_url = f"https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{args.version}/gencode.v{args.version}.primary_assembly.annotation.gtf.gz"

    fasta_file = os.path.join(output_dir, f"GRCh38.primary_assembly.genome.fa.gz")
    gtf_file = os.path.join(output_dir, f"gencode.v{args.version}.primary_assembly.annotation.gtf.gz")

    if args.download in ['both', 'ref']:
        print("Downloading FASTA file...")
        download_file(fasta_url, fasta_file)
        print(f"FASTA file downloaded: {fasta_file}")

    if args.download in ['both', 'gtf']:
        print("Downloading GTF file...")
        download_file(gtf_url, gtf_file)
        print(f"GTF file downloaded: {gtf_file}")

if __name__ == "__main__":
    main()

# python getReference.py --version 29 --download both --output_dir /path/to/save/files
# python getReference.py --version 46 --download both --output_dir GRCh38



