# sqtl benchmark

## setup

edit the config block at the top of `run.sh` and `score.sh`:

`run.sh`:
- `data_dir` — output/working directory for benchmark data
- `tpm_haec`, `haec_sumstats`, `haec_finemap` — haec-specific inputs

`score.sh`:
- `data_dir` — same as above
- `spt_dir` — path to SpliceTransformer repo

`fasta`, `gtf_haec`, and `models_dir` are derived from the repo root automatically.

## 1. download raw data

```bash
sbatch --job-name=download --partition=short --time=12:00:00 --mem=8G \
    --wrap="source ~/.bashrc && conda activate splaire_env && python src/download.py \
        --out-dir /scratch/runyan.m/sqtl_bench"
```

## 2. generate matching + vcfs

```bash
bash run.sh all
```

or individually:

```bash
bash run.sh txrevise
bash run.sh leafcutter
bash run.sh haec
bash run.sh ambig
```

## 3. score variants (gpu)

after step 2 jobs complete:

```bash
bash score.sh all
```

or individually:

```bash
bash score.sh txrevise
bash score.sh leafcutter
bash score.sh haec
bash score.sh ambig
```

## 4. analyze

```bash
python analysis.py
```

## credible set companions (separate from main benchmark)

extract all variants from credible sets that contributed benchmark positives,
then score them. requires step 2 to have completed (needs `pairs.csv`).

### generate vcfs

```bash
bash run.sh cs_all
```

or individually:

```bash
bash run.sh cs_txrevise
bash run.sh cs_leafcutter
bash run.sh cs_haec
```

### score variants (gpu)

```bash
bash score.sh cs_all
```

outputs go to `$data_dir/cs_txrevise/`, `$data_dir/cs_leafcutter/`, `$data_dir/cs_haec/`,
each with `variants.vcf.gz`, `cs_map.csv`, and `scores/`.
