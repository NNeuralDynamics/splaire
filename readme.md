# SPLAIRE

**SPL**icing in **AIR**way **E**pithelium

splice site usage prediction from pre-mRNA sequence. dilated CNN with classification (acceptor/donor/neither) and regression (SSU) heads, trained on HAEC RNA-seq.

## setup

```bash
conda env create -f envs/splaire_env.yml
conda activate splaire_env
```

genome FASTA at `pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa`:

```bash
cd pipeline/reference/GRCh38
python getReference.py --version 45 --download ref --output_dir .
gunzip GRCh38.primary_assembly.genome.fa.gz
```

## data

splice tables and model weights on [Zenodo](https://zenodo.org/records/19136478).

model weights are also in `models/` in this repo.

## repo layout

- `pipeline/` — nextflow pipeline for splice site quantification and dataset building
- `train/` — keras training scripts
- `models/` — trained weights (`{Ref,Var}_100_v{1-5}_{cls,reg}_best.keras`)
- `analysis/` — evaluation, benchmarks, figures. see `analysis/readme.md`
- `envs/` — conda environments

see `analysis/readme.md` for figure reproduction commands.
