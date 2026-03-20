# SPLAIRE

## setup

```bash
conda env create -f envs/splaire_env.yml
conda activate splaire_env
```

the genome FASTA must be at `pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa`. if not present:

```bash
cd pipeline/reference/GRCh38
python getReference.py --version 45 --download ref --output_dir .
gunzip GRCh38.primary_assembly.genome.fa.gz
```

see individual subfolder readmes for detailed usage.

## data

splice tables and model weights are available on [Zenodo](https://zenodo.org/records/19136478?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBjMDExYjFhLTA3MWItNGYwMC05ZTZhLTZiMjc2ZDMwMTkxNiIsImRhdGEiOnt9LCJyYW5kb20iOiJhMmU1OGI4MjFiYTk1N2NjNzZkMTYxN2VmNDc1ZDJkOSJ9.f_lt_JU-VqH_Z137xby98oSLu6aRl3OpX2-Wg3Gi4p4XGc5sXMF2FIdTFoRVyEN2WazT7RXM24CL3_Rbq6BJFA).
