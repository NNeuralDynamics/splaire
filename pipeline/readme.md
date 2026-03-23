# pipeline

nextflow pipeline to quantify splice site usage (SSU) from RNA-seq and build ML-ready HDF5 datasets. most users will start from existing splice tables and build h5 datasets directly.

## setup

```bash
conda env create -f envs/splaire_env.yml
conda activate splaire_env
export SPLAIRE_CONDA_ENV=${CONDA_PREFIX}
```

set paths for all commands below

```bash
REPO=<path_to_cloned_repo>
DATA=<dir_with_splice_tables>
OUT=<output_dir>
```

`DATA` should contain the splice tables.

**northeastern cluster** — files are already at `/scratch/runyan.m/splice_tables/`:

```bash
DATA=/scratch/runyan.m/splice_tables
```

**external users** — download from Zenodo (link available after publication)

this extracts:

```
splice_tables/
├── haec_train_splice_table.tsv    # HAEC 182 individuals (training)
├── haec10_splice_table.tsv        # HAEC 10 held-out individuals (test)
├── brain_cortex_splice_table.tsv  # GTEx tissues (test)
├── lung_splice_table.tsv
├── testis_splice_table.tsv
└── whole_blood_splice_table.tsv
```

the genome FASTA must be at `pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa`. if not present:

```bash
cd ${REPO}/pipeline/reference/GRCh38
python getReference.py --version 45 --download ref --output_dir .
gunzip GRCh38.primary_assembly.genome.fa.gz
```

## train

build training h5 datasets from the HAEC splice table (182 individuals, 100 train / 10 test)

```bash
mkdir -p ${OUT}/haec_train_run && cd ${OUT}/haec_train_run
nextflow run ${REPO}/pipeline/main.nf \
    -entry build_h5_only \
    --input_matrix ${DATA}/splice_tables/haec_train_splice_table.tsv \
    --samplesheet ${REPO}/pipeline/configs/haec/haec182_samples.tsv \
    --splits_config ${REPO}/pipeline/configs/haec/haec_splits.yaml \
    --output_dir ${OUT}/haec_train \
    --dataset_out_dir ${OUT}/haec_train/ml_data \
    -profile slurm
```

training splits config (`configs/haec/haec_splits.yaml`):

```yaml
samples:
  test_file: ${projectDir}/configs/haec/haec10_test.txt
  train_file: ${projectDir}/configs/haec/train_samples100.txt

chromosomes:
  test: [chr1, chr3, chr5, chr7]
  train: [chr2, chr4, chr6, chr8, chr10-chr22]

generate: all

dataset:
  variant: single
  paralog: all
  make_gc: true
  reference: true                 # use reference sequences (no donor variants)
  asymmetric_paralog_chroms: true

fill_gencode:
  train: true   # add GENCODE TSL=1 splice sites not observed in RNA-seq
  valid: true
  test: false

# optional: transcript-level train/validation splitting
# creates N replicate train/valid datasets for ensemble training
# frac is relative to total training rows, validation drawn from non-paralogs only
validation:
  frac: 0.10      # 10% of total training rows held out for validation
  n_splits: 5
  seed: 42        # base seed, split N uses seed + N
```

without the `validation:` block, the pipeline builds a single `combined_train.h5` and `combined_test.h5`. with it, the pipeline automatically splits the training data at the transcript level and produces replicate datasets:

output (with validation):

```
${OUT}/haec_train/ml_data/
├── combined_test.h5
├── combined_train_split1.h5
├── combined_valid_split1.h5
├── combined_train_split2.h5
├── combined_valid_split2.h5
├── ... (through split5)
└── individual/
    ├── test_DONOR1.h5
    ├── train_split1_DONOR1.h5
    ├── valid_split1_DONOR1.h5
    └── ...
```

output (without validation):

```
${OUT}/haec_train/ml_data/
├── combined_train.h5
├── combined_test.h5
└── individual/
    ├── train_DONOR1.h5
    └── ...
```

## test

build test h5 datasets for evaluation. one run per tissue.

### GTEx tissues

```bash
for tissue in brain_cortex whole_blood testis lung; do
    mkdir -p ${OUT}/${tissue}_run && cd ${OUT}/${tissue}_run
    nextflow run ${REPO}/pipeline/main.nf \
        -entry build_h5_only \
        --input_matrix ${DATA}/${tissue}_splice_table.tsv \
        --samplesheet ${REPO}/pipeline/configs/gtex/${tissue}_samples.tsv \
        --splits_config ${REPO}/pipeline/configs/gtex/${tissue}_splits.yaml \
        --output_dir ${OUT}/${tissue} \
        --dataset_out_dir ${OUT}/${tissue}/ml_data \
        -profile slurm
done
```

### HAEC10 (held-out test individuals)

```bash
mkdir -p ${OUT}/haec10_run && cd ${OUT}/haec10_run
nextflow run ${REPO}/pipeline/main.nf \
    -entry build_h5_only \
    --input_matrix ${DATA}/haec10_splice_table.tsv \
    --samplesheet ${REPO}/pipeline/configs/haec/haec182_samples.tsv \
    --splits_config ${REPO}/pipeline/configs/haec/haec10_splits.yaml \
    --output_dir ${OUT}/haec10 \
    --dataset_out_dir ${OUT}/haec10/ml_data \
    -profile slurm
```

test splits config (`configs/gtex/lung_splits.yaml`, same structure for all GTEx tissues):

```yaml
samples:
  test_file: ${projectDir}/configs/gtex/lung_test_samples.txt

chromosomes:
  test: [chr1, chr3, chr5, chr7]

generate: test_only

dataset:
  variant: single
  paralog: 0          # non-paralogs only
  make_gc: true
  reference: false    # include donor variants

fill_gencode:
  test: false

parallel:
  by: donor
  save_individual: true
```

output per tissue:

```
${OUT}/${tissue}/ml_data/
├── combined_test.h5
└── individual/
    ├── test_DONOR1.h5
    ├── test_DONOR2.h5
    └── ...
```

## full pipeline (BAM to datasets)

for processing raw BAM/FASTQ files from scratch. see [readme_detailed.md](readme_detailed.md) for full documentation including samplesheet format, VCF requirements, and cluster-specific commands.

```bash
nextflow run ${REPO}/pipeline/main.nf \
    --samplesheet <samplesheet.tsv> \
    --vcf <phased.vcf.gz> \
    --splits_config <splits_config.yaml> \
    --output_dir ${OUT}/<tissue> \
    -profile slurm
```

## modules

| module | description |
|--------|-------------|
| ALIGN_PASS1 | BAM/FASTQ -> fastp -> STAR pass 1 |
| ALIGN_PASS2 | STAR pass 2 -> filter -> regtools -> SpliSER |
| COLLECT_SITES | build master splice site catalog |
| FILL_SSU | fill missing sites per sample |
| ANNOTATE_SSU | classify splice site types |
| EXTRACT_VARIANTS | extract variants from VCF |
| MERGE_VARIANTS | combine variants across donors |
| MAKE_MATRIX | build SSU matrix |
| ADD_SITES | merge SSU with variant table |
| FILL_GENCODE | add GENCODE splice sites |
| BUILD_H5 | create per-donor HDF5 |
| COMBINE_H5 | merge into final datasets |
