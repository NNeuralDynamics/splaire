# Reporter Assays

Benchmarking splice prediction models against reporter assay datasets.

## Overview

```
raw data → prepare_data.ipynb → data H5 → score_*.py → scores H5 → analysis notebook
```

## Datasets

| dataset | source | n variants | phenotype | reference |
|---------|--------|------------|-----------|-----------|
| VexSeq | MMSplice/CAGI | ~2k | HepG2 delta-PSI | hg19 |
| MFASS | Cheung et al. 2019 | ~28k | v2_dpsi | hg19 |

## Data Model

Each variant is associated with an exon, create 4 sequences per variant:

```
         intron              exon               intron
    ──────────────┬──────────────────────┬──────────────
                  │                      │
              exon_start             exon_end
```

- `exon_start_ref` - window centered on exon start, reference allele
- `exon_start_alt` - window centered on exon start, alternate allele
- `exon_end_ref` - window centered on exon end, reference allele
- `exon_end_alt` - window centered on exon end, alternate allele

All sequences are 10,001bp centered on the boundary position. 


### Download Data

**VexSeq data (on MMSplice GitHub from CAGI challenge):**
```bash
cd vex_seq/data
# training set
curl -O https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/refs/heads/master/data/vexseq/HepG2_delta_PSI_CAGI_training.csv
# test set (without labels)
curl -O https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/refs/heads/master/data/vexseq/HepG2_delta_PSI_CAGI_testing.csv
# test set ground truth labels
curl -O https://raw.githubusercontent.com/gagneurlab/MMSplice_paper/refs/heads/master/data/vexseq/Vexseq_HepG2_delta_PSI_CAGI_test_true.tsv
```

**MFASS data:**
```bash
cd mfass/data
curl -O https://raw.githubusercontent.com/KosuriLab/MFASS/refs/heads/master/processed_data/snv/snv_data_clean.txt
```

**hg19 reference genome:**
```bash
cd vex_seq
curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz
gunzip hg19.fa.gz
samtools faidx hg19.fa
# symlink for mfass to use same reference
cd ../mfass && ln -s ../vex_seq/hg19.fa .
```

### Prepare Data

Run `prepare_dataset.py` to generate data H5 files:
```bash
# VexSeq
cd vex_seq
python prepare_dataset.py --data-dir data --fasta hg19.fa --out-h5 data/vex_seq.h5

# MFASS
cd ../mfass
python prepare_dataset.py --data-dir data --fasta ../vex_seq/hg19.fa --out-h5 data/mfass.h5
```

The `prepare_data.ipynb` notebooks do the same thing.



### Score with Models (scripts/score_*.py)

```bash
# SpliceAI
python scripts/score_sa.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sa.h5

# Pangolin
python scripts/score_pang.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_pang.h5

# SpliceTransformer
python scripts/score_spt.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_spt.h5

# SpHAEC (two runs - ref and var models)
python scripts/score_sphaec.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sphaec_ref.h5 --variant-type ref
python scripts/score_sphaec.py --input vex_seq/data/vex_seq.h5 --output vex_seq/data/vex_seq_sphaec_var.h5 --variant-type var
```





(base) [runyan.m@c0586 data]$ h5ls vex_seq_sa.h5
meta                     Group
scores                   Group
(base) [runyan.m@c0586 data]$ h5ls vex_seq_sa.h5/scores
acceptor_exon_end_alt    Dataset {2055}
acceptor_exon_end_ref    Dataset {2055}
acceptor_exon_start_alt  Dataset {2055}
acceptor_exon_start_ref  Dataset {2055}
donor_exon_end_alt       Dataset {2055}
donor_exon_end_ref       Dataset {2055}
donor_exon_start_alt     Dataset {2055}
donor_exon_start_ref     Dataset {2055}
neither_exon_end_alt     Dataset {2055}
neither_exon_end_ref     Dataset {2055}
neither_exon_start_alt   Dataset {2055}
neither_exon_start_ref   Dataset {2055}

make: 
acceptor_exon_end_delta    Dataset {2055}
acceptor_exon_start_delta  Dataset {2055}
donor_exon_end_delta       Dataset {2055}
donor_exon_start_delta     Dataset {2055}



(base) [runyan.m@c0586 data]$ h5ls vex_seq_spt.h5
meta                     Group
scores                   Group
(base) [runyan.m@c0586 data]$ h5ls vex_seq_spt.h5/scores
acceptor_exon_end_alt    Dataset {2055}
acceptor_exon_end_ref    Dataset {2055}
acceptor_exon_start_alt  Dataset {2055}
acceptor_exon_start_ref  Dataset {2055}
donor_exon_end_alt       Dataset {2055}
donor_exon_end_ref       Dataset {2055}
donor_exon_start_alt     Dataset {2055}
donor_exon_start_ref     Dataset {2055}
neither_exon_end_alt     Dataset {2055}
neither_exon_end_ref     Dataset {2055}
neither_exon_start_alt   Dataset {2055}
neither_exon_start_ref   Dataset {2055}
usage_Adipose_Tissue_exon_end_alt Dataset {2055}
usage_Adipose_Tissue_exon_end_ref Dataset {2055}
usage_Adipose_Tissue_exon_start_alt Dataset {2055}
usage_Adipose_Tissue_exon_start_ref Dataset {2055}
usage_Blood_Vessel_exon_end_alt Dataset {2055}
usage_Blood_Vessel_exon_end_ref Dataset {2055}
usage_Blood_Vessel_exon_start_alt Dataset {2055}
usage_Blood_Vessel_exon_start_ref Dataset {2055}
usage_Blood_exon_end_alt Dataset {2055}
usage_Blood_exon_end_ref Dataset {2055}
usage_Blood_exon_start_alt Dataset {2055}
usage_Blood_exon_start_ref Dataset {2055}
usage_Brain_exon_end_alt Dataset {2055}
usage_Brain_exon_end_ref Dataset {2055}
usage_Brain_exon_start_alt Dataset {2055}
usage_Brain_exon_start_ref Dataset {2055}
usage_Colon_exon_end_alt Dataset {2055}
usage_Colon_exon_end_ref Dataset {2055}
usage_Colon_exon_start_alt Dataset {2055}
usage_Colon_exon_start_ref Dataset {2055}
usage_Heart_exon_end_alt Dataset {2055}
usage_Heart_exon_end_ref Dataset {2055}
usage_Heart_exon_start_alt Dataset {2055}
usage_Heart_exon_start_ref Dataset {2055}
usage_Kidney_exon_end_alt Dataset {2055}
usage_Kidney_exon_end_ref Dataset {2055}
usage_Kidney_exon_start_alt Dataset {2055}
usage_Kidney_exon_start_ref Dataset {2055}
usage_Liver_exon_end_alt Dataset {2055}
usage_Liver_exon_end_ref Dataset {2055}
usage_Liver_exon_start_alt Dataset {2055}
usage_Liver_exon_start_ref Dataset {2055}
usage_Lung_exon_end_alt  Dataset {2055}
usage_Lung_exon_end_ref  Dataset {2055}
usage_Lung_exon_start_alt Dataset {2055}
usage_Lung_exon_start_ref Dataset {2055}
usage_Muscle_exon_end_alt Dataset {2055}
usage_Muscle_exon_end_ref Dataset {2055}
usage_Muscle_exon_start_alt Dataset {2055}
usage_Muscle_exon_start_ref Dataset {2055}
usage_Nerve_exon_end_alt Dataset {2055}
usage_Nerve_exon_end_ref Dataset {2055}
usage_Nerve_exon_start_alt Dataset {2055}
usage_Nerve_exon_start_ref Dataset {2055}
usage_Skin_exon_end_alt  Dataset {2055}
usage_Skin_exon_end_ref  Dataset {2055}
usage_Skin_exon_start_alt Dataset {2055}
usage_Skin_exon_start_ref Dataset {2055}
usage_Small_Intestine_exon_end_alt Dataset {2055}
usage_Small_Intestine_exon_end_ref Dataset {2055}
usage_Small_Intestine_exon_start_alt Dataset {2055}
usage_Small_Intestine_exon_start_ref Dataset {2055}
usage_Spleen_exon_end_alt Dataset {2055}
usage_Spleen_exon_end_ref Dataset {2055}
usage_Spleen_exon_start_alt Dataset {2055}
usage_Spleen_exon_start_ref Dataset {2055}
usage_Stomach_exon_end_alt Dataset {2055}
usage_Stomach_exon_end_ref Dataset {2055}
usage_Stomach_exon_start_alt Dataset {2055}
usage_Stomach_exon_start_ref Dataset {2055}

make:
acceptor_exon_end_delta    Dataset {2055}
acceptor_exon_start_delta  Dataset {2055}
donor_exon_end_delta       Dataset {2055}
donor_exon_start_delta     Dataset {2055}
usage_Adipose_Tissue_exon_end_delta Dataset {2055}
usage_Adipose_Tissue_exon_start_delta Dataset {2055}
usage_Blood_Vessel_exon_end_delta Dataset {2055}
usage_Blood_Vessel_exon_start_delta Dataset {2055}
usage_Blood_exon_end_delta Dataset {2055}
usage_Blood_exon_start_delta Dataset {2055}
usage_Brain_exon_end_delta Dataset {2055}
usage_Brain_exon_start_delta Dataset {2055}
usage_Colon_exon_end_delta Dataset {2055}
usage_Colon_exon_start_delta Dataset {2055}
usage_Heart_exon_end_delta Dataset {2055}
usage_Heart_exon_start_delta Dataset {2055}
usage_Kidney_exon_end_delta Dataset {2055}
usage_Kidney_exon_start_delta Dataset {2055}
usage_Liver_exon_end_delta Dataset {2055}
usage_Liver_exon_start_delta Dataset {2055}
usage_Lung_exon_end_delta  Dataset {2055}
usage_Lung_exon_start_delta Dataset {2055}
usage_Muscle_exon_end_delta Dataset {2055}
usage_Muscle_exon_start_delta Dataset {2055}
usage_Nerve_exon_end_delta Dataset {2055}
usage_Nerve_exon_start_delta Dataset {2055}
usage_Skin_exon_end_delta  Dataset {2055}
usage_Skin_exon_start_delta Dataset {2055}
usage_Small_Intestine_exon_end_delta Dataset {2055}
usage_Small_Intestine_exon_start_delta Dataset {2055}
usage_Spleen_exon_end_delta Dataset {2055}
usage_Spleen_exon_start_delta Dataset {2055}
usage_Stomach_exon_end_delta Dataset {2055}
usage_Stomach_exon_start_delta Dataset {2055}



(base) [runyan.m@c0586 data]$ h5ls vex_seq_pang.h5
meta                     Group
scores                   Group
(base) [runyan.m@c0586 data]$ h5ls vex_seq_pang.h5/scores
brain_p_splice_exon_end_alt Dataset {2055}
brain_p_splice_exon_end_ref Dataset {2055}
brain_p_splice_exon_start_alt Dataset {2055}
brain_p_splice_exon_start_ref Dataset {2055}
brain_usage_exon_end_alt Dataset {2055}
brain_usage_exon_end_ref Dataset {2055}
brain_usage_exon_start_alt Dataset {2055}
brain_usage_exon_start_ref Dataset {2055}
heart_p_splice_exon_end_alt Dataset {2055}
heart_p_splice_exon_end_ref Dataset {2055}
heart_p_splice_exon_start_alt Dataset {2055}
heart_p_splice_exon_start_ref Dataset {2055}
heart_usage_exon_end_alt Dataset {2055}
heart_usage_exon_end_ref Dataset {2055}
heart_usage_exon_start_alt Dataset {2055}
heart_usage_exon_start_ref Dataset {2055}
liver_p_splice_exon_end_alt Dataset {2055}
liver_p_splice_exon_end_ref Dataset {2055}
liver_p_splice_exon_start_alt Dataset {2055}
liver_p_splice_exon_start_ref Dataset {2055}
liver_usage_exon_end_alt Dataset {2055}
liver_usage_exon_end_ref Dataset {2055}
liver_usage_exon_start_alt Dataset {2055}
liver_usage_exon_start_ref Dataset {2055}
testis_p_splice_exon_end_alt Dataset {2055}
testis_p_splice_exon_end_ref Dataset {2055}
testis_p_splice_exon_start_alt Dataset {2055}
testis_p_splice_exon_start_ref Dataset {2055}
testis_usage_exon_end_alt Dataset {2055}
testis_usage_exon_end_ref Dataset {2055}
testis_usage_exon_start_alt Dataset {2055}
testis_usage_exon_start_ref Dataset {2055}


make:
brain_p_splice_exon_end_delta Dataset {2055}
brain_p_splice_exon_start_delta Dataset {2055}
brain_usage_exon_end_delta Dataset {2055}
brain_usage_exon_start_delta Dataset {2055}
heart_p_splice_exon_end_delta Dataset {2055}
heart_p_splice_exon_start_delta Dataset {2055}
heart_usage_exon_end_delta Dataset {2055}
heart_usage_exon_start_delta Dataset {2055}
liver_p_splice_exon_end_delta Dataset {2055}
liver_p_splice_exon_start_delta Dataset {2055}
liver_usage_exon_end_delta Dataset {2055}
liver_usage_exon_start_delta Dataset {2055}
testis_p_splice_exon_end_delta Dataset {2055}
testis_p_splice_exon_start_delta Dataset {2055}
testis_usage_exon_end_delta Dataset {2055}
testis_usage_exon_start_delta Dataset {2055}




(base) [runyan.m@c0586 data]$ h5ls  vex_seq_sphaec_ref.h5 
meta                     Group
scores                   Group
(base) [runyan.m@c0586 data]$ h5ls  vex_seq_sphaec_ref.h5 /scores
meta                     Group
scores                   Group
/scores: unable to open file
(base) [runyan.m@c0586 data]$ h5ls  vex_seq_sphaec_ref.h5/scores
cls_acceptor_exon_end_alt Dataset {2055}
cls_acceptor_exon_end_ref Dataset {2055}
cls_acceptor_exon_start_alt Dataset {2055}
cls_acceptor_exon_start_ref Dataset {2055}
cls_donor_exon_end_alt   Dataset {2055}
cls_donor_exon_end_ref   Dataset {2055}
cls_donor_exon_start_alt Dataset {2055}
cls_donor_exon_start_ref Dataset {2055}
cls_neither_exon_end_alt Dataset {2055}
cls_neither_exon_end_ref Dataset {2055}
cls_neither_exon_start_alt Dataset {2055}
cls_neither_exon_start_ref Dataset {2055}
reg_ssu_exon_end_alt     Dataset {2055}
reg_ssu_exon_end_ref     Dataset {2055}
reg_ssu_exon_start_alt   Dataset {2055}
reg_ssu_exon_start_ref   Dataset {2055}

make:
cls_acceptor_exon_end_delta    Dataset {2055}
cls_acceptor_exon_start_delta  Dataset {2055}
cls_donor_exon_end_delta       Dataset {2055}
cls_donor_exon_start_delta     Dataset {2055}
reg_ssu_exon_end_delta     Dataset {2055}
reg_ssu_exon_start_delta   Dataset {2055}

(base) [runyan.m@c0586 data]$ h5ls  vex_seq_sphaec_var.h5/scores
cls_acceptor_exon_end_alt Dataset {2055}
cls_acceptor_exon_end_ref Dataset {2055}
cls_acceptor_exon_start_alt Dataset {2055}
cls_acceptor_exon_start_ref Dataset {2055}
cls_donor_exon_end_alt   Dataset {2055}
cls_donor_exon_end_ref   Dataset {2055}
cls_donor_exon_start_alt Dataset {2055}
cls_donor_exon_start_ref Dataset {2055}
cls_neither_exon_end_alt Dataset {2055}
cls_neither_exon_end_ref Dataset {2055}
cls_neither_exon_start_alt Dataset {2055}
cls_neither_exon_start_ref Dataset {2055}
reg_ssu_exon_end_alt     Dataset {2055}
reg_ssu_exon_end_ref     Dataset {2055}
reg_ssu_exon_start_alt   Dataset {2055}
reg_ssu_exon_start_ref   Dataset {2055}

make:
cls_acceptor_exon_end_delta    Dataset {2055}
cls_acceptor_exon_start_delta  Dataset {2055}
cls_donor_exon_end_delta       Dataset {2055}
cls_donor_exon_start_delta     Dataset {2055}


if strand is + then keep - cls_acceptor_exon_start_delta and cls_donor_exon_end_delta and column should be acceptor_delta and donor delta
if strand is - the keep -cls_acceptor_exon_end_delta and cls_donor_exon_start_delta and these also go in should be acceptor_delta and donor delta

do this for all models except pangolin

reg_ssu_exon_end_delta     Dataset {2055}
reg_ssu_exon_start_delta   Dataset {2055}



