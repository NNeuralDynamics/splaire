# test

score test-set donors with all models and compute metrics

## setup

```bash
conda env create -f envs/splaire_env.yml
conda activate splaire_env
export SPLAIRE_CONDA_ENV=$(conda info --base)/envs/splaire_env
```

set paths for all commands below

```bash
REPO=<path_to_cloned_repo>
DATA=<dir_with_splice_tables>
OUT=<output_dir>
```

`DATA` should contain the splice tables:

```
brain_cortex_splice_table.tsv
whole_blood_splice_table.tsv
testis_splice_table.tsv
lung_splice_table.tsv
haec10_splice_table.tsv
```

## build h5 datasets

nextflow builds per-donor h5 files from the splice tables. one run per tissue.

```bash
for tissue in brain_cortex whole_blood testis lung; do
    mkdir -p ${OUT}/${tissue}_run && cd ${OUT}/${tissue}_run
    nextflow run ${REPO}/pipeline/main.nf \
        -entry build_h5_only \
        --input_matrix ${DATA}/${tissue}_splice_table.tsv \
        --samplesheet ${REPO}/pipeline/configs/gtex/${tissue}_samples.tsv \
        --splits_config ${REPO}/pipeline/configs/gtex/${tissue}_splits.yaml \
        --output_dir ${OUT}/${tissue} \
        --dataset_out_dir ${OUT}/${tissue}/ml_data_var \
        -profile slurm
done

# haec10
mkdir -p ${OUT}/haec10_run && cd ${OUT}/haec10_run
nextflow run ${REPO}/pipeline/main.nf \
    -entry build_h5_only \
    --input_matrix ${DATA}/haec10_splice_table.tsv \
    --samplesheet ${REPO}/pipeline/configs/haec/haec182_samples.tsv \
    --splits_config ${REPO}/pipeline/configs/haec/haec10_splits.yaml \
    --output_dir ${OUT}/haec10 \
    --dataset_out_dir ${OUT}/haec10/ml_data_var \
    -profile slurm
```

output tree per tissue:

```
${OUT}/${tissue}/ml_data_var/
├── combined_test.h5
└── individual/
    ├── test_DONOR1.h5
    ├── test_DONOR2.h5
    └── ...
```

## score (per donor, all models)

runs splaire, spliceai, pangolin, and splicetransformer on each donor h5

```bash
cd ${REPO}/analysis/test

for tissue in brain_cortex whole_blood testis lung haec10; do
    sbatch -J ${tissue} score.sbatch \
        ${OUT}/${tissue}/ml_data_var/individual \
        ${OUT}/${tissue}/ml_out_var
done
```

output tree per tissue:

```
${OUT}/${tissue}/ml_out_var/predictions/
├── test_DONOR1/
│   ├── test_DONOR1_splaire_ref.parquet
│   ├── test_DONOR1_splaire_var.parquet
│   ├── test_DONOR1_sa.parquet
│   ├── test_DONOR1_pang.parquet
│   └── test_DONOR1_spt.parquet
├── test_DONOR2/
│   └── ...
└── ...
```

## scoring your own model

each scoring script reads an h5 file and writes a parquet with predictions. to add your own model, write a script that produces a parquet with these columns:

**required columns** (ground truth + coordinates, same for all models):

| column | type | description |
|---|---|---|
| chrom | int8 | chromosome (1-22, X=23, Y=24) |
| pos | int32 | genomic position |
| strand | int8 | 0=minus, 1=plus |
| y_acceptor | float32 | true acceptor label (0 or 1) |
| y_donor | float32 | true donor label (0 or 1) |
| y_ssu | float32 | true splice site usage (0-1, 777=missing) |

**prediction columns** (model-specific, at least one):

| column | type | what it means |
|---|---|---|
| acceptor | float32 | predicted acceptor probability |
| donor | float32 | predicted donor probability |
| ssu | float32 | predicted splice site usage |

the filename must end with a suffix that identifies the model: `{donor_name}_{suffix}.parquet`

to register it with `compute_metrics.py`, add an entry to the `models` dict at the top of `src/compute_metrics.py`:

```python
models = {
    ...
    "_mymodel": ("my_model", ["acceptor", "donor"]),
}
```

the dict key is the filename suffix, the value is `(display_name, list_of_prediction_columns)`.

see `src/score_spliceai.py` for a minimal example.

## compute metrics

```bash
cd ${REPO}/analysis/test

for tissue in brain_cortex whole_blood testis lung haec10; do
    for ind in ${OUT}/${tissue}/ml_out_var/predictions/*/; do
        name=$(basename "$ind")
        sbatch -J ${tissue}_${name} metrics.sbatch \
            "$ind" \
            ${OUT}/${tissue}/ml_out_var/metrics/${name}.json \
            --splicing-matrix ${DATA}/${tissue}_splice_table.tsv
    done
done
```

## scoring only your model (skip others)

you can either comment out the other models in `score.sbatch` and add your own, or run your scoring script directly on the individual h5 files:

```bash
cd ${REPO}/analysis/test

for tissue in brain_cortex whole_blood testis lung haec10; do
    for h5 in ${OUT}/${tissue}/ml_data_var/individual/*.h5; do
        name=$(basename "$h5" .h5)
        out_dir=${OUT}/${tissue}/ml_out_var/predictions/${name}
        mkdir -p ${out_dir}
        python src/score_mymodel.py ${h5} ${out_dir}/${name}_mymodel.parquet
    done
done
```

then run `compute_metrics.py` as above. it will only compute metrics for models whose parquet files exist in the predictions directory.

## benchmarks

### gencode

scores all TSL=1 protein-coding splice sites on test chromosomes. uses `data/canonical_dataset.txt` (hg19 coordinates from SpliceAI).

requires `pipeline/reference/GRCh38/` (genome FASTA, GTF, paralogs)

```bash
cd ${REPO}/analysis/test
sbatch score_gencode.sbatch ${OUT}/canonical/gencode
```

### mane select

same but uses MANE Select transcripts. input is `data/canonical_dataset_mane_select_p.txt` (hg38)

```bash
sbatch score_mane.sbatch ${OUT}/canonical/mane_select
```

### pangolin

uses pangolin's test splice table for tissue-specific comparison. download first:

```bash
cd ${REPO}/analysis/test
curl -L -o data/splice_table_Human.test.txt \
    https://raw.githubusercontent.com/tkzeng/Pangolin_train/main/preprocessing/splice_table_Human.test.txt

for tissue in heart liver brain testis; do
    sbatch -J pang_${tissue} score_pangolin.sbatch \
        ${OUT}/canonical/pangolin $tissue
done
```
