# reporter assays

benchmarking splice prediction models on Vex-seq and MFASS reporter assay datasets.

## pipeline

`analysis.py` handles everything end-to-end: downloads raw data, builds sequence H5s, loads model scores, computes metrics, and generates all figures at `figures/fig3/`.

```
analysis.py
├── download vex-seq csvs (from MMSplice github)
├── download mfass snv data (from KosuriLab github)
├── build vex_seq/data/vex_seq.h5 (±5kb windows, one-hot encoded)
├── build mfass/data/mfass.h5
├── load scored H5s from scripts/score_*.py
├── compute pearson, spearman, auprc, r²
└── generate figures/fig3/{main,sup}/*.pdf
```

## setup

hg19 reference genome (needed for sequence extraction):

```bash
cd vex_seq
curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz
gunzip hg19.fa.gz
samtools faidx hg19.fa
```

## scoring

score with each model before running analysis.py (GPU, submit via sbatch):

```bash
# splaire (ref + var)
python scripts/score_sphaec.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sphaec_ref.h5 --variant-type ref
python scripts/score_sphaec.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sphaec_var.h5 --variant-type var

# spliceai
python scripts/score_sa.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sa.h5

# pangolin
python scripts/score_pang.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_pang.h5

# splicetransformer
python scripts/score_spt.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_spt.h5
```

same pattern for mfass (replace `vex_seq` with `mfass` in paths).

## analysis

```bash
python analysis.py
```

downloads raw data if not present, builds H5s if not present, loads scores, computes all metrics, saves figures to `figures/fig3/main/` and `figures/fig3/sup/`.

## datasets

| dataset | source | n variants | phenotype | reference |
|---------|--------|------------|-----------|-----------|
| Vex-seq | MMSplice/CAGI5 | 2,055 | HepG2 delta-PSI | hg19 |
| MFASS | Cheung et al. 2019 | 27,733 | v2_dpsi | hg19 |

## data layout (created at runtime by analysis.py)

```
vex_seq/
├── hg19.fa              # user downloads this
└── data/
    ├── train.csv        # downloaded from MMSplice github
    ├── test.csv
    ├── truth.tsv
    ├── vex_seq.h5       # built by analysis.py
    ├── vex_seq_sa.h5    # scored by scripts/score_sa.py
    ├── vex_seq_pang.h5
    ├── vex_seq_spt.h5
    ├── vex_seq_sphaec_ref.h5
    └── vex_seq_sphaec_var.h5

mfass/
└── data/
    ├── snv_data_clean.txt  # downloaded from MFASS github
    ├── mfass.h5            # built by analysis.py
    └── mfass_*.h5          # scored
```
