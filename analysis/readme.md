# analysis

figures for the SPLAIRE paper. each figure has its own subfolder with a readme covering setup and scoring. below is the full chain for each.

## setup

```bash
conda env create -f ../envs/splaire_env.yml
conda activate splaire_env
```

set paths once, everything below uses these.
 `OUT` needs at least 400GB for full analysis.

```bash
REPO=/full/path/to/cloned_repo
OUT=/full/path/to/output_dir
```

## figures 1 & 2 — splice sites & model comparison

build h5 datasets from splice tables, score with all models, compute metrics, make figures.

```bash
# download splice tables from zenodo
mkdir -p $OUT/splice_tables && cd $OUT/splice_tables
wget https://zenodo.org/records/19136478/files/splice_tables.tar
tar xf splice_tables.tar && gunzip splice_tables/*.gz

# build per-donor h5 datasets (one nextflow run per tissue, see pipeline/readme.md)
for tissue in haec10 lung brain_cortex testis whole_blood; do
    mkdir -p $OUT/${tissue}_run && cd $OUT/${tissue}_run
    nextflow run $REPO/pipeline/main.nf \
        -entry build_h5_only \
        --input_matrix $OUT/splice_tables/${tissue}_splice_table.tsv \
        --samplesheet $REPO/pipeline/configs/gtex/${tissue}_samples.tsv \
        --splits_config $REPO/pipeline/configs/gtex/${tissue}_splits.yaml \
        --output_dir $OUT/${tissue} \
        --dataset_out_dir $OUT/${tissue}/ml_data \
        -profile slurm
done

# score with all models (gpu, ~24 hrs per tissue for 5 models)
cd $REPO/analysis/test
for tissue in haec10 lung brain_cortex testis whole_blood; do
    sbatch score.sbatch $OUT/${tissue}/ml_data/individual $OUT/${tissue}/ml_out
done

# compute metrics per donor
for tissue in haec10 lung brain_cortex testis whole_blood; do
    for ind in $OUT/${tissue}/ml_out/predictions/*/; do
        name=$(basename "$ind")
        sbatch metrics.sbatch "$ind" $OUT/${tissue}/ml_out/metrics/${name}.json
    done
done

# generate figures
cd $REPO/analysis/test
python analysis.py
```

output: `figures/fig1/{main,sup}/` and `figures/fig2/{main,sup}/`

see `test/readme.md` for h5 build details and scoring your own model.

## figure 3 — reporter assays (vex-seq + mfass)

download variant data, build sequence h5s, score, analyze.

```bash
# get hg19 reference
cd $REPO/analysis/reporter_assays/vex_seq
curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz
gunzip hg19.fa.gz && samtools faidx hg19.fa

# run analysis.py to download source csvs and builds h5s 
cd $REPO/analysis/reporter_assays
python analysis.py


# score with all models (gpu)
for dataset in vex_seq mfass; do
    sbatch scripts/submit_${dataset}_sphaec.sbatch
    sbatch scripts/submit_${dataset}_sa.sbatch
    sbatch scripts/submit_${dataset}_pang.sbatch
    sbatch scripts/submit_${dataset}_spt.sbatch
done

# rerun analysis.py to load scores and make figures
python analysis.py
```

output: `figures/fig3/{main,sup}/`

see `reporter_assays/readme.md` for scoring details.

## figure 4 — sqtl benchmark

download eqtl catalogue data, build matched variant pairs, score, analyze.

```bash
# download credible sets + summary stats
cd $REPO/analysis/sqtl_bench
python src/download.py --out-dir $OUT/sqtl_bench

# build matched positive/negative pairs
bash run.sh all

# score variants (gpu, run one at a time — explorer limit is 8 gpu jobs)
bash score.sh leafcutter
bash score.sh txrevise
bash score.sh haec
bash score.sh ambig

# generate figures
python analysis.py
```

output: `figures/fig4/{main,sup}/`

see `sqtl_bench/readme.md` for per-dataset commands and config.

## figure 5 — attribution analysis

extract splice site sequences, run deeplift-shap, analyze.

```bash
# extract test-set splice site sequences
cd $REPO/analysis/explain
python src/extract_splice_sequences.py \
    --tissue-dirs $OUT/brain_cortex $OUT/haec10 $OUT/lung $OUT/testis $OUT/whole_blood \
    --tissues brain_cortex haec lung testis whole_blood \
    --genome $REPO/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa \
    --protein-coding $REPO/pipeline/reference/GRCh38/protein_coding_genes.tsv \
    --paralogs $REPO/pipeline/reference/GRCh38/paralogs_GRCh38.txt.gz \
    --test-chroms \
    --output data/sequences.h5

# run attribution (gpu, one per model/head)
sbatch src/run_attribution_splaire.sbatch splaire_ref_reg
sbatch src/run_attribution_splaire.sbatch splaire_ref_cls

# normalize
python src/add_normalized_attributions.py --attr data/attr_splaire_*.h5 --seq data/sequences.h5

# generate figures
python analysis.py
```

output: `figures/fig5/{main,supp}/`

see `explain/readme.md` for model options and parameters.

## environment variables

scripts default to cluster paths. override for external use:

| variable | what |
|---|---|
| `SPLAIRE_DATA_DIR` | root for splice tables, h5s, predictions |
| `SPLAIRE_CANONICAL_DIR` | canonical benchmark data |
| `SPLAIRE_SQTL_DIR` | sqtl benchmark data |
