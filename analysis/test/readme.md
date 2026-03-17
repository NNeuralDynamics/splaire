# test

> *Note: Shell commands and sbatch scripts reference paths on the Northeastern Explorer cluster. Adapt paths for your environment.*

score test-set donors with all models and compute metrics. includes benchmarks against gencode canonical sites, mane select, and pangolin test data.

## 0. download data

download splice tables from Zenodo into a data directory:

```
<data_dir>/
├── brain_cortex_splice_table.tsv
├── whole_blood_splice_table.tsv
├── testis_splice_table.tsv
├── lung_splice_table.tsv
└── haec10_splice_table.tsv
```

set these variables for all commands below

```bash
REPO=<path_to_repo>
DATA=<data_dir>
OUT=<output_dir>
```

## 1. build individual h5 datasets

build per-donor HDF5 files from splice tables

```bash
conda activate sp
module load nextflow

# gtex tissues
for tissue in brain_cortex whole_blood testis lung; do
    mkdir -p ${OUT}/${tissue}_run
    cd ${OUT}/${tissue}_run
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
mkdir -p ${OUT}/haec10_run
cd ${OUT}/haec10_run
nextflow run ${REPO}/pipeline/main.nf \
    -entry build_h5_only \
    --input_matrix ${DATA}/haec10_splice_table.tsv \
    --samplesheet ${REPO}/pipeline/configs/haec/haec182_samples.tsv \
    --splits_config ${REPO}/pipeline/configs/haec/haec10_splits.yaml \
    --output_dir ${OUT}/haec10 \
    --dataset_out_dir ${OUT}/haec10/ml_data_var \
    -profile slurm
```

## 2. individual predictions (per donor, all models)

scores each donor's h5 with splaire, spliceai, pangolin, and splicetransformer.

```bash
cd ${REPO}/analysis/test

for tissue in brain_cortex whole_blood testis lung haec10; do
    sbatch -J ${tissue} score.sbatch \
        ${OUT}/${tissue}/ml_data_var/individual \
        ${OUT}/${tissue}/ml_out_var
done
```

## 3. individual metrics (per donor)

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

## 4. gencode benchmark

scores all TSL=1 protein-coding splice sites on test chromosomes. converts `data/canonical_dataset.txt` (from SpliceAI, hg19) → matrix → h5 → scores with all models → metrics, all in one sbatch.

requires reference files in `pipeline/reference/GRCh38/` (genome FASTA, GTF, paralogs).

```bash
cd ${REPO}/analysis/test
sbatch score_gencode.sbatch ${OUT}/canonical/gencode
```

## 5. mane select benchmark

same as gencode but uses MANE Select transcripts. input is `data/canonical_dataset_mane_select_p.txt` (hg38, with paralog column).

```bash
cd ${REPO}/analysis/test
sbatch score_mane.sbatch ${OUT}/canonical/mane_select
```

## 6. pangolin benchmark (per tissue)

uses pangolin's test splice table to compare models on tissue-specific splice sites.

### setup — download pangolin test data

the pangolin benchmark requires two files from [Pangolin_train](https://github.com/tkzeng/Pangolin_train). `paralogs.txt` is included in `data/`. download the splice table:

```bash
cd ${REPO}/analysis/test
curl -L -o data/splice_table_Human.test.txt \
    https://raw.githubusercontent.com/tkzeng/Pangolin_train/main/preprocessing/splice_table_Human.test.txt
```

### run

```bash
cd ${REPO}/analysis/test

for tissue in heart liver brain testis; do
    sbatch -J pang_${tissue} score_pangolin.sbatch \
        ${OUT}/canonical/pangolin $tissue
done
```

## 7. metrics only

if predictions exist and you only need to recompute metrics:

```bash
cd ${REPO}/analysis/test

# gencode
sbatch metrics.sbatch \
    ${OUT}/canonical/gencode/predictions \
    ${OUT}/canonical/gencode/gencode_metrics.json

# mane select
sbatch metrics.sbatch \
    ${OUT}/canonical/mane_select/predictions \
    ${OUT}/canonical/mane_select/mane_select_metrics.json

# pangolin tissues
for tissue in heart liver brain testis; do
    sbatch -J metrics_${tissue} metrics.sbatch \
        ${OUT}/canonical/pangolin/predictions \
        ${OUT}/canonical/pangolin/${tissue}_metrics.json \
        --splice-combined
done
```

---

<details>
<summary>cluster-specific commands (northeastern explorer)</summary>

### build h5 datasets

```bash
# gtex tissues
for tissue in lung testis brain_cortex whole_blood; do
    mkdir -p /scratch/runyan.m/splaire_run/${tissue}_var
    sbatch \
        --job-name="nf_${tissue}_var" \
        --partition=short --mem=4G --time=12:00:00 \
        --output="/scratch/runyan.m/splaire_run/${tissue}_var/nf_%j.out" \
        --error="/scratch/runyan.m/splaire_run/${tissue}_var/nf_%j.err" \
        --wrap="
            source ~/.bashrc && module load nextflow && conda activate sp && unset JAVA_HOME
            cd /scratch/runyan.m/splaire_run/${tissue}_var
            nextflow run /projects/talisman/mrunyan/paper/splaire/pipeline/main.nf \
                -entry build_h5_only \
                --input_matrix /scratch/runyan.m/splaire_out/${tissue}/combined/combined_gene_variants_SNVs_sites.tsv \
                --samplesheet /projects/talisman/mrunyan/paper/splaire/pipeline/configs/gtex/${tissue}_samples.tsv \
                --splits_config /projects/talisman/mrunyan/paper/splaire/pipeline/configs/gtex/${tissue}_splits.yaml \
                --output_dir /scratch/runyan.m/splaire_out/${tissue} \
                --dataset_out_dir /scratch/runyan.m/splaire_out/${tissue}/ml_data_var \
                -profile slurm -resume
        "
done

# haec10
mkdir -p /scratch/runyan.m/splaire_run/haec10_var
sbatch \
    --job-name="nf_haec10_var" \
    --partition=short --mem=4G --time=12:00:00 \
    --output="/scratch/runyan.m/splaire_run/haec10_var/nf_%j.out" \
    --error="/scratch/runyan.m/splaire_run/haec10_var/nf_%j.err" \
    --wrap="
        source ~/.bashrc && module load nextflow && conda activate sp && unset JAVA_HOME
        cd /scratch/runyan.m/splaire_run/haec10_var
        nextflow run /projects/talisman/mrunyan/paper/splaire/pipeline/main.nf \
            -entry build_h5_only \
            --input_matrix /scratch/runyan.m/splaire_out/haec10/combined/combined_gene_variants_SNVs_sites.tsv \
            --samplesheet /projects/talisman/mrunyan/paper/splaire/pipeline/configs/haec/haec182_samples.tsv \
            --splits_config /projects/talisman/mrunyan/paper/splaire/pipeline/configs/haec/haec10_splits.yaml \
            --output_dir /scratch/runyan.m/splaire_out/haec10 \
            --dataset_out_dir /scratch/runyan.m/splaire_out/haec10/ml_data_var \
            -profile slurm
    "
```

### score all tissues

```bash
cd /projects/talisman/mrunyan/paper/splaire/analysis/test

for tissue in brain_cortex whole_blood testis lung haec10; do
    sbatch -J ${tissue} score.sbatch \
        /scratch/runyan.m/splaire_out/${tissue}/ml_data_var/individual \
        /scratch/runyan.m/splaire_out/${tissue}/ml_out_var
done
```

### metrics all tissues

```bash
cd /projects/talisman/mrunyan/paper/splaire/analysis/test

for tissue in brain_cortex whole_blood testis lung haec10; do
    for ind in /scratch/runyan.m/splaire_out/${tissue}/ml_out_var/predictions/*/; do
        name=$(basename "$ind")
        sbatch -J ${tissue}_${name} metrics.sbatch \
            "$ind" \
            /scratch/runyan.m/splaire_out/${tissue}/ml_out_var/metrics/${name}.json \
            --splicing-matrix /scratch/runyan.m/splaire_out/${tissue}/processed_splicing_matrix.tsv
    done
done
```

### benchmarks

```bash
cd /projects/talisman/mrunyan/paper/splaire/analysis/test

# gencode
sbatch score_gencode.sbatch /scratch/runyan.m/splaire_out/canonical/gencode

# mane select
sbatch score_mane.sbatch /scratch/runyan.m/splaire_out/canonical/mane_select

# pangolin
for tissue in heart liver brain testis; do
    sbatch -J pang_${tissue} score_pangolin.sbatch \
        /scratch/runyan.m/splaire_out/canonical/pangolin $tissue
done
```

</details>
