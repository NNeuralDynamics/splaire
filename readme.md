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
