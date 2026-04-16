# Attribution Analysis Report

> Auto-generated from `analysis.ipynb` — figures produced by `analysis_script.py`
> 
> All figures saved as PNG (300 dpi) and PDF in `figures/` subdirectories.
> Key numerical results in `figures/results.txt` and `figures/results.json`.
> Full output log in `figures/analysis.log`.

## Table of Contents

- [attribution analysis](#attribution-analysis)
  - [sequence for attributions](#sequence-for-attributions)
  - [extraction command that produced sequences.h5](#extraction-command-that-produced-sequencesh5)
  - [annotation command](#annotation-command)
  - [model conversion](#model-conversion)
  - [attribution computation](#attribution-computation)
  - [attribution command](#attribution-command)
    - [attribution normalization](#attribution-normalization)
      - [completeness check](#completeness-check)
      - [why normalize](#why-normalize)
      - [what normalization loses](#what-normalization-loses)
      - [summary](#summary)
    - [Mean Attribution Profiles by Splice Type](#mean-attribution-profiles-by-splice-type)
  - [Attribution Profiles](#attribution-profiles)
  - [Standalone Attribution Profiles by SSU](#standalone-attribution-profiles-by-ssu)
- [Pangolin Attribution Profiles](#pangolin-attribution-profiles)
  - [Nucleotide Frequency at Splice Sites](#nucleotide-frequency-at-splice-sites)
  - [Brain](#brain)
  - [Heart](#heart)
  - [Liver](#liver)
  - [Testis](#testis)
- [SPLAIRE Classification Attribution Profiles](#splaire-classification-attribution-profiles)
- [Pangolin P(splice) Attribution Profiles](#pangolin-psplice-attribution-profiles)
  - [Brain (p_splice)](#brain-psplice)
  - [Heart (p_splice)](#heart-psplice)
  - [Liver (p_splice)](#liver-psplice)
  - [Testis (p_splice)](#testis-psplice)

---

# attribution analysis

this notebook documents how we produced attribution interpretability results for SPLAIRE models.

## sequence for attributions

each tissue directory contains two files from the pipeline

- `processed_splicing_matrix.tsv` - ssu values per splice site per sample 
- `splicing_matrix.tsv` - raw read counts in inc/total format

tissues

- brain_cortex
- lung  
- testis
- whole_blood
- haec


sites were filtered to

- chr1, chr3, chr5, chr7
- protein-coding, non-paralogous genes
- no N's bases in the genomic sequence

## extraction command that produced sequences.h5
```
python src/extract_splice_sequences.py \
    --tissue-dirs ${SPLAIRE_DATA_DIR}/brain_cortex \
                  ${SPLAIRE_DATA_DIR}/haec10 \
                  ${SPLAIRE_DATA_DIR}/lung \
                  ${SPLAIRE_DATA_DIR}/testis \
                  ${SPLAIRE_DATA_DIR}/whole_blood \
    --tissues brain_cortex haec lung testis whole_blood \
    --genome /ref/GRCh38/GRCh38.primary_assembly.genome.fa \
    --protein-coding /ref/gencode/protein_coding_genes.tsv \
    --paralogs /ref/ensembl/paralogs_GRCh38.txt.gz \
    --test-chroms \
    --window 5000 \
    --output data/sequences.h5
```

![data overview — splice type distribution](figures/data_overview/splice_type_distribution.png)

**Figure 1.** Distribution of splice site types in the attribution analysis dataset. Bar heights indicate the number of acceptor and donor splice sites from test chromosomes (chr1, 3, 5, 7) in non-paralogous protein-coding genes.


**SSU per tissue**: Each sequence has a SSU measurement from one of the 5 tissues (brain_cortex, haec, lung, testis, whole_blood). NaN indicates the site was not observed in that tissue. This is the mean SSU across samples from each tissue.

![data overview — ssu distributions](figures/data_overview/ssu_distributions.png)

**Figure 2.** Splice site usage (SSU) distributions across five tissues. Each panel shows the distribution of mean SSU values for one tissue. SSU ranges from 0 (never used) to 1 (constitutively used). NaN values indicate splice sites not observed in that tissue.


## annotation command
**Nearby splice sites**: After extraction, each sequence is annotated with other known splice sites within +/-5kb from GENCODE (v45) and GTEx (v8) junction data.
```
python src/annotate_splice_sites.py \
    --h5 data/sequences.h5 \
    --gtf /ref/gencode/gencode.v45.annotation.gtf \
    --gtex /ref/gtex/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz \
    --window 5000
```

![data overview — nearby sites histogram](figures/data_overview/nearby_sites_histogram.png)

**Figure 3.** Distribution of annotated nearby splice sites within ±5 kb of each query splice site. Nearby sites were identified from GENCODE v45 annotations and GTEx v8 splice junction data.


## model conversion

SPLAIRE models were trained in tensorflow/keras. converted to pytorch for use with tangermeme.

**conversion process**
- parse keras model config to build equivalent pytorch architecture
- copy weights layer by layer matching names
- verify outputs on real splice sites match (atol=1e-5)

the conversion script is `../../models/keras_to_torch.py`.

## attribution computation

we use deeplift-shap to compute per-nucleotide attribution scores. the script `run_attribution.py` wraps tangermeme's `deep_lift_shap` function.

**what our script does**
- loads pytorch model and sequences
- batches sequences for gpu efficiency
- calls tangermeme deep_lift_shap per batch
- saves results to h5

**what tangermeme deep_lift_shap does**
- generates reference sequences by dinucleotide shuffling each input
- computes deeplift attributions comparing input to each reference
- averages attributions across all references (shap averaging)
- checks convergence (attributions should sum to prediction difference)
- returns hypothetical attributions for all 4 bases at each position

**reference generation**

dinucleotide shuffling preserves GC content and dinucleotide frequencies while breaking higher-order patterns. we generate 20 references per sequence which gives stable attributions (cv < 5% at most positions).

**what we get back**

tangermeme returns hypothetical attributions with shape (batch, 4, seq_len). these show what each nucleotide would contribute if it were present. we then compute

- observed attributions: multiply by one-hot to get actual contribution at each position
- L1 normalized: divide by sum of absolute values so each sequence sums to 1

**parameters**
- 20 shuffled references per sequence
- warning threshold 0.001 for convergence
- computed on v100 gpu

## attribution command
```
python src/run_attribution.py \
    --model sphaec_ref_reg \
    --input data/sequences.h5 \
    --output data/attr_sphaec_ref_reg.h5 \
    --batch-size 64 \
    --n-shuffles 20 \
    --shuffle-method dinucleotide \
    --device cuda
```

### attribution normalization

#### completeness check

DeepLIFT-SHAP satisfies the **completeness axiom**: the sum of all attributions equals the difference between the model's prediction and its expected prediction over shuffled references.

$$\sum_i \text{attr}_i = f(x) - \mathbb{E}[f(x_{\text{ref}})]$$

this is a correctness test — if signed raw attributions don't sum to the prediction, something is wrong with the attribution computation. we expect r=1.0 on the diagonal.

**interpreting deviations**: most points fall on x=y because shuffled references predict near zero for most sequences. the small deviations arise from two sources:

- **upper right (off-diagonal)**: sequences where shuffled references retain some predictive signal from dinucleotide composition alone (e.g., pyrimidine-rich regions). the attribution sum equals prediction minus the non-zero reference baseline, shifting points slightly off the diagonal.
- **bottom left (negative sums)**: sequences where the model predicts near zero but the shuffled references predict slightly positive. the attribution sum is negative because the real sequence is *less splice-like than random* — repressive elements push the prediction below the shuffled baseline. these are rare and small in magnitude.

![attribution magnitude — completeness check](figures/attribution_magnitude/completeness_check.png)

**Figure 4.** DeepLIFT-SHAP completeness verification. Scatter plot of model prediction versus the sum of signed raw attributions across the 10,001 bp input sequence. Points falling on the diagonal (r ≈ 1.0) confirm that attributions satisfy the completeness axiom.


#### why normalize

raw attribution magnitude scales with prediction — high-SSU sites get more total attribution simply because the model's output is larger. this makes cross-sequence comparison unfair: a position with attribution 0.01 in a high-SSU sequence is not comparable to 0.01 in a low-SSU sequence.

**L1 normalization** divides each position's attribution by the total absolute attribution for that sequence, so every sequence's absolute attributions sum to 1.0.

**interpretation**: after normalization, a value of **+0.02** at position k means "this nucleotide contributes **2% of the model's reasoning** and pushes the prediction **higher** (vs shuffled references)." a value of **-0.02** means it contributes 2% but pushes the prediction **lower**.

![attribution magnitude — raw vs norm magnitude](figures/attribution_magnitude/raw_vs_norm_magnitude.png)

**Figure 5.** Effect of L1 normalization on attribution magnitude. Left: total absolute raw attribution versus model prediction, showing magnitude scales with prediction. Right: after L1 normalization, every sequence has equal total absolute attribution (1.0), enabling cross-sequence comparison.


#### what normalization loses

after L1 normalization, the signed sum no longer equals the prediction. instead, the **signed normalized sum** reflects the balance of positive vs negative attribution — how much of the model's attention is activating vs repressive.

a moderate correlation (r ~ 0.5) with prediction is expected: high-SSU sites have more net-positive attribution because the model uses more activating than repressive signals for strong splice sites. this is not a problem — it's an interpretable property of how the model works.

![attribution magnitude — signed norm vs prediction](figures/attribution_magnitude/signed_norm_vs_prediction.png)

**Figure 6.** Signed sum of L1-normalized attributions versus model prediction. The moderate positive correlation reflects the balance of activating versus repressive attribution: high-SSU sites have proportionally more positive (activating) attribution.


#### summary

- **completeness verified** — signed raw attributions sum to the prediction (r ~ 1.0), confirming DeepLIFT-SHAP correctness
- **L1 normalization removes magnitude confound** — all sequences now have equal attribution budget for cross-sequence comparison
- **interpretation**: positive = activating (increases SSU vs shuffled reference), negative = repressive (decreases SSU vs shuffled reference)

### Mean Attribution Profiles by Splice Type

**Data**: L1-normalized observed attributions (1D) for all sequences, separated by splice type (acceptor vs donor). Extract ±100bp window around center splice site. Compute mean and standard deviation across all sequences of each splice type.

Two line plots (acceptor, donor) showing mean attribution by position with +/- 1 standard deviation shaded region. Red dashed line marks the splice site position (0). Shows average attribution patterns for acceptor vs donor sites. Width of shaded region indicates variability across sequences.

![attribution magnitude — mean profiles by splice type](figures/attribution_magnitude/mean_profiles_by_splice_type.png)

**Figure 7.** Mean L1-normalized attribution profiles at acceptor (left) and donor (right) splice sites. Solid lines show the mean across all sequences; shaded regions indicate ±1 standard deviation. The dashed red line marks the splice junction (position 0). Window spans ±100 bp.



## Standalone Attribution Profiles by SSU

Mean attribution by true SSU bin (classification and regression heads), plotted as standalone 2x2 panels.

![attribution by true ssu](figures/attribution_by_true_ssu.png)

**Figure 8.** Mean L1-normalized attribution profiles by true SSU bin, shown as standalone 2×2 panels (rows: classification, regression; columns: acceptor, donor). Lines show mean attribution at each position; colors indicate SSU bin (low, mid, high).


---

# Pangolin Attribution Profiles

Pangolin predicts tissue-specific splice site usage with separate output heads for brain, heart, liver, and testis. We computed DeepLIFT-SHAP attributions for the usage head of each tissue using the same 72,537 splice sites and ±5 kb sequences as the SPLAIRE analysis. Attributions are L1-normalized. Mean attribution matrices and predictions are cached in `figures/pangolin/cache/` to skip reloading the ~2.7 GB obs_norm arrays on re-runs.

## Nucleotide Frequency at Splice Sites

frequency heatmaps are model-independent — same sequences, same dinucleotide grouping regardless of which model produced the attributions.

![donor frequency](figures/pangolin/donor_freq.png)

**Figure 9a.** Nucleotide frequency at donor splice site positions grouped by dinucleotide (GU, GC, AU). Dashed line marks the exon-intron boundary.

![acceptor frequency](figures/pangolin/acceptor_freq.png)

**Figure 9b.** Nucleotide frequency at acceptor splice site positions grouped by dinucleotide (AG, AC). Dashed line marks the intron-exon boundary.


## Brain

![brain donor logos](figures/pangolin/brain_donor_logos.png)

**Figure 10a.** Pangolin brain usage attribution logos at donor splice sites (GU, GC, AU).

![brain donor kde](figures/pangolin/brain_donor_kde.png)

**Figure 10b.** Pangolin brain predicted vs true SSU density for donor dinucleotides.

![brain acceptor logos](figures/pangolin/brain_acceptor_logos.png)

**Figure 10c.** Pangolin brain usage attribution logos at acceptor splice sites (AG, AC).

![brain acceptor kde](figures/pangolin/brain_acceptor_kde.png)

**Figure 10d.** Pangolin brain predicted vs true SSU density for acceptor dinucleotides.


## Heart

![heart donor logos](figures/pangolin/heart_donor_logos.png)

**Figure 11a.** Pangolin heart usage attribution logos at donor splice sites.

![heart donor kde](figures/pangolin/heart_donor_kde.png)

**Figure 11b.** Pangolin heart predicted vs true SSU density for donor dinucleotides.

![heart acceptor logos](figures/pangolin/heart_acceptor_logos.png)

**Figure 11c.** Pangolin heart usage attribution logos at acceptor splice sites.

![heart acceptor kde](figures/pangolin/heart_acceptor_kde.png)

**Figure 11d.** Pangolin heart predicted vs true SSU density for acceptor dinucleotides.


## Liver

![liver donor logos](figures/pangolin/liver_donor_logos.png)

**Figure 12a.** Pangolin liver usage attribution logos at donor splice sites.

![liver donor kde](figures/pangolin/liver_donor_kde.png)

**Figure 12b.** Pangolin liver predicted vs true SSU density for donor dinucleotides.

![liver acceptor logos](figures/pangolin/liver_acceptor_logos.png)

**Figure 12c.** Pangolin liver usage attribution logos at acceptor splice sites.

![liver acceptor kde](figures/pangolin/liver_acceptor_kde.png)

**Figure 12d.** Pangolin liver predicted vs true SSU density for acceptor dinucleotides.


## Testis

![testis donor logos](figures/pangolin/testis_donor_logos.png)

**Figure 13a.** Pangolin testis usage attribution logos at donor splice sites.

![testis donor kde](figures/pangolin/testis_donor_kde.png)

**Figure 13b.** Pangolin testis predicted vs true SSU density for donor dinucleotides.

![testis acceptor logos](figures/pangolin/testis_acceptor_logos.png)

**Figure 13c.** Pangolin testis usage attribution logos at acceptor splice sites.

![testis acceptor kde](figures/pangolin/testis_acceptor_kde.png)

**Figure 13d.** Pangolin testis predicted vs true SSU density for acceptor dinucleotides.


---

# SPLAIRE Classification Attribution Profiles

SPLAIRE's classification head predicts the probability that a position is an acceptor or donor splice site (matched by splice type). Attribution logos and prediction density are computed using the same sequences and windows as the regression analysis. Nucleotide frequency heatmaps are model-independent and shown in Figure 9.

![splaire cls donor logos](figures/splaire_cls/donor_logos.png)

**Figure 14a.** SPLAIRE classification attribution logos at donor splice sites (GU, GC, AU).

![splaire cls donor kde](figures/splaire_cls/donor_kde.png)

**Figure 14b.** SPLAIRE classification predicted vs true SSU density for donor dinucleotides.

![splaire cls acceptor logos](figures/splaire_cls/acceptor_logos.png)

**Figure 14c.** SPLAIRE classification attribution logos at acceptor splice sites (AG, AC).

![splaire cls acceptor kde](figures/splaire_cls/acceptor_kde.png)

**Figure 14d.** SPLAIRE classification predicted vs true SSU density for acceptor dinucleotides.


---

# Pangolin P(splice) Attribution Profiles

Pangolin's p_splice output head predicts the probability of a position being a splice site, analogous to SPLAIRE's classification head. Attributions and predictions are computed per tissue. Frequency heatmaps are model-independent (see Figure 9).

## Brain (p_splice)

![brain psplice donor logos](figures/pangolin/brain_psplice_donor_logos.png)

**Figure 15a.** Pangolin brain P(splice) attribution logos at donor splice sites.

![brain psplice donor kde](figures/pangolin/brain_psplice_donor_kde.png)

**Figure 15b.** Pangolin brain P(splice) vs true SSU density for donor dinucleotides.

![brain psplice acceptor logos](figures/pangolin/brain_psplice_acceptor_logos.png)

**Figure 15c.** Pangolin brain P(splice) attribution logos at acceptor splice sites.

![brain psplice acceptor kde](figures/pangolin/brain_psplice_acceptor_kde.png)

**Figure 15d.** Pangolin brain P(splice) vs true SSU density for acceptor dinucleotides.


## Heart (p_splice)

![heart psplice donor logos](figures/pangolin/heart_psplice_donor_logos.png)

**Figure 16a.** Pangolin heart P(splice) attribution logos at donor splice sites.

![heart psplice donor kde](figures/pangolin/heart_psplice_donor_kde.png)

**Figure 16b.** Pangolin heart P(splice) vs true SSU density for donor dinucleotides.

![heart psplice acceptor logos](figures/pangolin/heart_psplice_acceptor_logos.png)

**Figure 16c.** Pangolin heart P(splice) attribution logos at acceptor splice sites.

![heart psplice acceptor kde](figures/pangolin/heart_psplice_acceptor_kde.png)

**Figure 16d.** Pangolin heart P(splice) vs true SSU density for acceptor dinucleotides.


## Liver (p_splice)

![liver psplice donor logos](figures/pangolin/liver_psplice_donor_logos.png)

**Figure 17a.** Pangolin liver P(splice) attribution logos at donor splice sites.

![liver psplice donor kde](figures/pangolin/liver_psplice_donor_kde.png)

**Figure 17b.** Pangolin liver P(splice) vs true SSU density for donor dinucleotides.

![liver psplice acceptor logos](figures/pangolin/liver_psplice_acceptor_logos.png)

**Figure 17c.** Pangolin liver P(splice) attribution logos at acceptor splice sites.

![liver psplice acceptor kde](figures/pangolin/liver_psplice_acceptor_kde.png)

**Figure 17d.** Pangolin liver P(splice) vs true SSU density for acceptor dinucleotides.


## Testis (p_splice)

![testis psplice donor logos](figures/pangolin/testis_psplice_donor_logos.png)

**Figure 18a.** Pangolin testis P(splice) attribution logos at donor splice sites.

![testis psplice donor kde](figures/pangolin/testis_psplice_donor_kde.png)

**Figure 18b.** Pangolin testis P(splice) vs true SSU density for donor dinucleotides.

![testis psplice acceptor logos](figures/pangolin/testis_psplice_acceptor_logos.png)

**Figure 18c.** Pangolin testis P(splice) attribution logos at acceptor splice sites.

![testis psplice acceptor kde](figures/pangolin/testis_psplice_acceptor_kde.png)

**Figure 18d.** Pangolin testis P(splice) vs true SSU density for acceptor dinucleotides.

