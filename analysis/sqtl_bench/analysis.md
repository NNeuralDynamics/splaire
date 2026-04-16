---
title: "sQTL Benchmark Analysis"
toc: true
toc-depth: 3
toc-expand: false
toc-location: right
embed-resources: true
theme: cosmo
---

# sQTL Benchmark Analysis

## Data Sources

Three sQTL datasets are benchmarked: **GTEx txrevise** (49 tissues, exon-level splicing events from eQTL Catalogue), **GTEx leafcutter** (49 tissues, intron-level splicing from eQTL Catalogue), and **HAEC185** (single endothelial cell tissue, leafcutter-based). All use the same evaluation framework: high-PIP positives matched 1:1 to low-PIP negatives via a 4-tier cascade (same gene + matching alleles → same gene → same TPM bin + matching alleles → same TPM bin), with splice distance as the within-tier tiebreaker.

**Scoring:** Each variant's delta score = max |alt - ref| within ±2 kb of the variant center, across all output positions and channels for the given head. Scores are tissue-independent (the same variant gets the same score regardless of tissue context).

---

# QC

## GTEx Txrevise

**Positives:** For each txrevise sQTL credible set, the variant with the highest PIP is selected. Credible sets are filtered to "contained" events only (within gene body), max PIP >= 0.9, SNVs only, splice distance <= 10 kb.

**Negatives:** Variants from GTEx gene expression summary statistics with PIP < 0.01, SNVs only, splice distance <= 10 kb (distance to nearest exon boundary within the same gene).

| | |
|:---:|:---:|
| ![](figures/txrevise/qc/07_credsets_counts.png) | ![](figures/txrevise/qc/03_sumstats_ge.png) |
| **a.** Per-tissue counts from txrevise credible sets (variants, genes, high-PIP) | **b.** Per-tissue variant and gene counts from GE summary statistics |

**Figure 1.** Txrevise per-tissue QC.

| | |
|:---:|:---:|
| ![](figures/txrevise/qc/08_credsets_pip_dist.png) | ![](figures/txrevise/qc/05_gtex_tpm.png) |
| **a.** PIP distribution across all credible set variants | **b.** GTEx v8 median TPM distribution across tissues |

**Figure 2.** Txrevise PIP and expression distributions.

| | |
|:---:|:---:|
| ![](figures/txrevise/qc/09_credsets_pip_sum.png) | ![](figures/txrevise/qc/10_credsets_size_vs_pip.png) |
| **a.** PIP sum per credible set (log scale, median annotated) | **b.** Credible set size vs maximum PIP |

**Figure 3.** Txrevise credible set calibration.

## GTEx Leafcutter

**Positives:** For each leafcutter sQTL credible set, the variant with highest PIP is selected. Only credible sets with max PIP >= 0.9 are included. Introns must fall within a gene body.

**Negatives:** Same source as txrevise (GE summary statistics, PIP < 0.01), but splice distance = distance to nearest leafcutter intron boundary within the same gene (using all leafcutter-detected introns per tissue from Zenodo phenotype metadata).

| | |
|:---:|:---:|
| ![](figures/leafcutter/qc/05_credsets_counts.png) | ![](figures/leafcutter/qc/09_credsets_structure.png) |
| **a.** Per-tissue counts from leafcutter credible sets | **b.** Credible set structure: per-tissue counts and size distribution |

**Figure 4.** Leafcutter per-tissue QC.

| | |
|:---:|:---:|
| ![](figures/leafcutter/qc/06_credsets_pip_dist.png) | ![](figures/leafcutter/qc/07_credsets_pip_sum.png) |
| **a.** PIP distribution across all credible set variants | **b.** PIP sum per credible set |

**Figure 5.** Leafcutter PIP distributions.

![](figures/leafcutter/qc/08_credsets_size_vs_pip.png)

**Figure 6.** Leafcutter credible set size vs maximum PIP.

## HAEC185

**Positives:** Variants from HAEC185 sQTL fine-mapping credible sets. Max PIP >= 0.9. Single tissue.

**Negatives:** Variants from HAEC sQTL summary statistics with PIP < 0.01. Splice distance = distance to nearest leafcutter-detected intron boundary in the same gene.

| | |
|:---:|:---:|
| ![](figures/haec/qc/02_sumstats_qc.png) | ![](figures/haec/qc/04_credsets_qc.png) |
| **a.** Summary statistics quality metrics | **b.** Credible set statistics (counts, PIP dist, PIP sum, CS size) |

**Figure 7.** HAEC QC.

---

# Matching

## Tier Distributions

| | | |
|:---:|:---:|:---:|
| ![](figures/txrevise/matching/ge_set1/01_tier_distribution.png) | ![](figures/leafcutter/matching/set1/01_tier_distribution.png) | ![](figures/haec/matching/set1/01_tier_distribution.png) |
| **a.** Txrevise | **b.** Leafcutter | **c.** HAEC |

**Figure 8.** Matching tier distribution for each dataset. Each positive is matched 1:1 to a negative using a 4-tier cascade: tier 1 = same gene + matching alleles, tier 2 = same gene, tier 3 = same TPM bin + matching alleles, tier 4 = same TPM bin. Within each tier, the negative with the closest splice distance is selected.

## Splice Distance Distributions

| | | |
|:---:|:---:|:---:|
| ![](figures/txrevise/matching/ge_set1/05_distance_dist.png) | ![](figures/leafcutter/matching/set1/05_distance_dist.png) | ![](figures/haec/matching/set1/05_distance_dist.png) |
| **a.** Txrevise | **b.** Leafcutter | **c.** HAEC |

**Figure 9.** Splice distance distributions for positive and negative variants after matching. For txrevise, splice distance = distance to nearest exon boundary within the same gene. For leafcutter and HAEC, splice distance = distance to the nearest intron boundary.

## Per-Tissue Composition (GTEx only)

| | |
|:---:|:---:|
| ![](figures/txrevise/matching/ge_set1/06_per_tissue.png) | ![](figures/leafcutter/matching/set1/06_per_tissue.png) |
| **a.** Txrevise | **b.** Leafcutter |

**Figure 10.** Per-tissue breakdown of tissue-specific vs shared positives. A shared positive is a variant that appears as positive in multiple tissues. Stacked bars show unique (single-tissue) vs shared variants per tissue.

---

# Results

## Delta Score Distributions

| | |
|:---:|:---:|
| ![](figures/txrevise/results/set1/00a_delta_distributions.png) | ![](figures/leafcutter/results/set1/00a_delta_distributions.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/results/set1/00a_delta_distributions.png)

**c.** HAEC

**Figure 11.** Delta score distributions for positive and negative variants across models. Each variant's delta score = max |alt - ref| within ±2 kb. Positives and negatives are not deduplicated across tissues — a variant appearing in multiple tissues contributes one score per tissue pair. Median delta annotated with dashed line. Y-axis log scale. HAEC is single-tissue so each variant appears exactly once.

## Classification Performance (Swarm Plots)

| | |
|:---:|:---:|
| ![](figures/txrevise/results/set1/01_swarm.png) | ![](figures/leafcutter/results/set1/01_swarm.png) |
| **a.** Txrevise (49 tissues) | **b.** Leafcutter (49 tissues) |

**Figure 12.** Per-tissue AUPRC swarm plots for GTEx datasets. Each dot is one tissue's AUPRC. Within each tissue, every matched positive-negative pair contributes independently: the positive's delta score is labeled 1 and the matched negative's is labeled 0, then AUPRC is computed over the concatenated scores. A variant appearing as positive in multiple tissues uses the same delta score each time (scores are tissue-independent). Horizontal bar = median across 49 tissues.

![](figures/haec/results/set1/01_overall_auprc.png)

**Figure 13.** AUPRC for HAEC (single tissue). Same scoring procedure as Figure 12 but pooled over all pairs since there is only one tissue. Each variant appears in exactly one pair.

## All Output Heads Comparison

| | |
|:---:|:---:|
| ![](figures/txrevise/results/set1/01b_hbar.png) | ![](figures/leafcutter/results/set1/01b_hbar.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/results/set1/01b_hbar.png)

**c.** HAEC

**Figure 14.** Median AUPRC across all output heads. Shows CLS, usage/SSU, per-tissue, and combined heads for each model. Computed as in Figure 12 (median of per-tissue AUPRCs for GTEx, pooled for HAEC). Allows comparison of which output representation best separates positives from negatives.

## PR and ROC Curves

| | |
|:---:|:---:|
| ![](figures/txrevise/results/set1/01b_pr_curves.png) | ![](figures/txrevise/results/set1/01d_roc_curves.png) |
| **a.** Txrevise PR | **b.** Txrevise ROC |
| ![](figures/leafcutter/results/set1/01b_pr_curves.png) | ![](figures/leafcutter/results/set1/01d_roc_curves.png) |
| **c.** Leafcutter PR | **d.** Leafcutter ROC |
| ![](figures/haec/results/set1/01b_pr_curves.png) | ![](figures/haec/results/set1/01b_roc_curves.png) |
| **e.** HAEC PR | **f.** HAEC ROC |

**Figure 15.** Precision-recall (left) and ROC (right) curves for all three datasets. For GTEx (49 tissues), per-tissue curves are interpolated onto a shared grid (recall for PR, FPR for ROC) and the median across tissues is plotted. AUROC computed from the median ROC curve via trapezoidal integration. HAEC curves are computed directly over all pairs (single tissue).

## Cross-Dataset Summary

Per-tissue median AUPRC and AUROC across all three datasets for the primary classification benchmark. Summary table saved to `figures/cross_dataset_classification_summary.csv`.

## AUPRC by Splice Distance (Paired Filtering)

| | |
|:---:|:---:|
| ![](figures/txrevise/results/set1/02_distance.png) | ![](figures/leafcutter/results/set1/02_distance.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/results/set1/02_distance.png)

**c.** HAEC

**Figure 16.** AUPRC by splice distance with paired filtering. Pairs are filtered by the positive variant's splice distance only — at each threshold, all pairs where the positive's distance to its nearest splice site is within the threshold are included, regardless of the negative's distance. The matched negative stays paired with its positive. For GTEx, AUPRC is the median across per-tissue AUPRCs. For HAEC, AUPRC is pooled. Grey bars show unique positive and negative variant counts at each threshold.

## AUPRC by Splice Distance (Unique Positives)

| | |
|:---:|:---:|
| ![](figures/txrevise/results/set1/02b_distance_unique.png) | ![](figures/leafcutter/results/set1/02b_distance_unique.png) |
| **a.** Txrevise | **b.** Leafcutter |

**Figure 17.** AUPRC by splice distance with unique positives. Same paired filtering as Figure 16, but after filtering, pairs are deduplicated by `pos_var_key` (keeping the first occurrence), so each positive variant is scored exactly once with one matched negative, ignoring tissue. This removes the effect of a variant appearing in many tissues and yields a single pooled AUPRC rather than a tissue median.

## AUPRC by Splice Distance (Independent Filtering)

| | |
|:---:|:---:|
| ![](figures/leafcutter/results/set1/02c_distance_independent.png) | ![](figures/txrevise/results/set1/02c_distance_independent.png) |
| **a.** Leafcutter | **b.** Txrevise |

**Figure 18.** AUPRC by splice distance with independent filtering. Unlike the paired method (Figure 16), positives and negatives are filtered independently by their own splice distance at each threshold — a positive passes if its distance <= threshold, and a negative passes if its distance <= threshold, regardless of the other member of the pair. This breaks the 1:1 pairing: the number of positives and negatives can differ. AUPRC is the median across per-tissue AUPRCs. Matches the evaluation in Linder et al. (2025).

## Supplemental: ROC with Youden's J Threshold

![](figures/leafcutter/roc_youden.png)

**Figure 19.** ROC curves for GTEx leafcutter with Youden's J optimal threshold annotated per model.

---

# Tissue Specificity Analysis

For each positive variant, tissue fraction = (n tissues where PIP >= 0.9) / (n tissues where the variant appears in any credible set). A tissue fraction of 1.0 means the variant is a constitutive sQTL; a low fraction means it is tissue-specific. GTEx datasets only (HAEC is single tissue).

## Cumulative AUPRC by Tissue Fraction

| | |
|:---:|:---:|
| ![](figures/leafcutter/results/set1/03a_tissue_fraction_cumulative.png) | ![](figures/txrevise/results/set1/03a_tissue_fraction_cumulative.png) |
| **a.** Leafcutter | **b.** Txrevise |

**Figure 20.** Cumulative AUPRC by tissue specificity. At each x-axis threshold, only pairs whose positive has tissue fraction at or below that value are included; AUPRC is the median across per-tissue AUPRCs. Tissue-specific variants (low fraction) are hardest to classify; constitutive variants (fraction = 1.0) are easiest. Grey bars show unique positive and negative variant counts at each threshold.

## Per-Tissue AUPRC vs Variant Composition

| | |
|:---:|:---:|
| ![](figures/leafcutter/results/set1/03b_tissue_composition_scatter.png) | ![](figures/txrevise/results/set1/03b_tissue_composition_scatter.png) |
| **a.** Leafcutter | **b.** Txrevise |

**Figure 21.** Per-tissue AUPRC versus fraction of positives that are constitutive (tissue fraction = 1.0). Each point is one tissue. X-axis = fraction of that tissue's positive variants that appear as positive in every tissue they were tested in. Spearman correlation quantifies how much per-tissue AUPRC variance is explained by variant composition.

## Tissue Fraction Distributions

| | |
|:---:|:---:|
| ![](figures/leafcutter/results/set1/03c_tissue_fraction_hist.png) | ![](figures/txrevise/results/set1/03c_tissue_fraction_hist.png) |
| **a.** Leafcutter | **b.** Txrevise |

**Figure 22.** Distribution of tissue specificity (n positive / n tested) across all positive variants.

---

# Txrevise Comprehensive GTF

Rerun of the txrevise benchmark using GENCODE v39 **comprehensive** annotation (`gencode.v39.annotation.gtf.gz`, ~250k transcripts) instead of the basic subset (`gencode.v39.basic.annotation.gtf.gz`, ~114k transcripts). The eQTL Catalogue uses the comprehensive GTF for txrevise event quantification, so this is the matched annotation. More transcripts means more exon boundaries per gene, which changes splice distances and therefore which variants pass the 10 kb filter and which negatives get picked by the 4-tier matching cascade.

## QC

![](figures/txrevise_comp/qc/07_credsets_counts.png)

**Figure 22a.** Per-tissue counts from txrevise credible sets with comprehensive GTF.

## Matching

| | |
|:---:|:---:|
| ![](figures/txrevise_comp/matching/ge_set1/01_tier_distribution.png) | ![](figures/txrevise_comp/matching/ge_set1/05_distance_dist.png) |
| **a.** Tier distribution | **b.** Splice distance distribution |

**Figure 22b.** Matching results with comprehensive GTF. (a) 4-tier cascade tier counts. (b) Splice distance distributions for positives and negatives — should show more variants near splice sites than the basic-GTF run since additional exon boundaries reduce per-gene splice distances.

## Results

| | |
|:---:|:---:|
| ![](figures/txrevise_comp/results/set1/00a_delta_distributions.png) | ![](figures/txrevise_comp/results/set1/01_swarm.png) |
| **a.** Delta score distributions | **b.** Per-tissue AUPRC swarm |

**Figure 22c.** Classification performance on the comprehensive-GTF txrevise benchmark. (a) Delta distributions for positives and negatives across models. (b) Per-tissue AUPRC swarm (49 GTEx tissues). Horizontal bar = median. Compare to Figure 12a (basic-GTF run) to see the effect of annotation completeness.

| | |
|:---:|:---:|
| ![](figures/txrevise_comp/results/set1/01b_pr_curves.png) | ![](figures/txrevise_comp/results/set1/01d_roc_curves.png) |
| **a.** PR curves | **b.** ROC curves |

**Figure 22d.** Precision-recall and ROC curves for the comprehensive-GTF txrevise benchmark (median over 49 tissues, interpolated onto a shared grid).

| | |
|:---:|:---:|
| ![](figures/txrevise_comp/results/set1/02_distance.png) | ![](figures/txrevise_comp/results/set1/02b_distance_unique.png) |
| **a.** AUPRC by distance (paired filtering) | **b.** AUPRC by distance (unique positives) |

**Figure 22e.** AUPRC as a function of splice distance threshold. (a) Paired filtering: pairs are filtered by the positive's splice distance only. (b) Unique positives: pairs are deduplicated by `pos_var_key` after filtering so each positive is scored exactly once.

![](figures/txrevise_comp/results/set1/01b_hbar.png)

**Figure 22f.** Median AUPRC across all output heads for the comprehensive-GTF txrevise benchmark (only generated when run without `--nohbar`).

---

# Hungarian Matching

Rerun of all three sQTL benchmarks with **Hungarian (global one-to-one optimal assignment)** matching. The tiered cascade (gene+allele → gene → expression bin+allele → expression bin) leaves 33–35% of GTEx pairs with $|\text{pos\_dist} - \text{neg\_dist}| > 500$ bp because within-gene negative pools are often too small to produce a close-distance match. Hungarian drops the gene stratum entirely, groups candidates by expression bin only, and solves the one-to-one assignment that minimizes total $|\log_{10}(\text{pos\_dist}+1) - \log_{10}(\text{neg\_dist}+1)|$. It produces near-perfect distance balance (median `dist_diff` ≈ 0 bp, >99% of pairs within 10 bp) at the same N as tiered. PIP thresholds: txrevise `>= 0.9` (comprehensive GTF, matches `txrevise_comp` baseline); leafcutter `>= 0.5` matched, filtered to `>= 0.9` in results (same as `leafcutter_pip50`); HAEC `>= 0.9`.

## Txrevise Hungarian

| | |
|:---:|:---:|
| ![](figures/txrevise_hungarian/matching/01_tier_distribution.png) | ![](figures/txrevise_hungarian/matching/05_distance_dist.png) |
| **a.** Tier distribution (all tier 99 = hungarian) | **b.** Splice distance histogram for pos and neg |

**Figure 23a.** Txrevise hungarian matching diagnostic. All pairs are produced by a single global-optimum assignment within expression bin (no gene stratum).

| | |
|:---:|:---:|
| ![](figures/txrevise_hungarian/results/00a_delta_distributions.png) | ![](figures/txrevise_hungarian/results/01_swarm.png) |
| **a.** Delta distributions | **b.** Per-tissue AUPRC swarm (49 GTEx tissues) |

**Figure 23b.** Classification performance on the hungarian-matched txrevise benchmark.

| | |
|:---:|:---:|
| ![](figures/txrevise_hungarian/results/01b_pr_curves.png) | ![](figures/txrevise_hungarian/results/01d_roc_curves.png) |
| **a.** PR curves | **b.** ROC curves |

**Figure 23c.** PR and ROC curves (tissue median over 49 GTEx tissues).

| | |
|:---:|:---:|
| ![](figures/txrevise_hungarian/results/02_distance.png) | ![](figures/txrevise_hungarian/results/02b_distance_unique.png) |
| **a.** AUPRC by distance (paired) | **b.** AUPRC by distance (unique positives) |

**Figure 23d.** AUPRC stratified by splice distance on the hungarian-matched benchmark.

## Leafcutter Hungarian

| | |
|:---:|:---:|
| ![](figures/leafcutter_hungarian/matching/01_tier_distribution.png) | ![](figures/leafcutter_hungarian/matching/05_distance_dist.png) |
| **a.** Tier distribution | **b.** Splice distance histogram |

**Figure 24a.** Leafcutter hungarian matching diagnostic (positives filtered to PIP >= 0.9 in results, matched at PIP >= 0.5).

| | |
|:---:|:---:|
| ![](figures/leafcutter_hungarian/results/00a_delta_distributions.png) | ![](figures/leafcutter_hungarian/results/01_swarm.png) |
| **a.** Delta distributions | **b.** Per-tissue AUPRC swarm |

**Figure 24b.** Classification performance on hungarian-matched leafcutter. Note: SpT negative scoring may be incomplete — if so, SpT is skipped automatically.

| | |
|:---:|:---:|
| ![](figures/leafcutter_hungarian/results/01b_pr_curves.png) | ![](figures/leafcutter_hungarian/results/01d_roc_curves.png) |
| **a.** PR curves | **b.** ROC curves |

**Figure 24c.** PR and ROC curves.

| | |
|:---:|:---:|
| ![](figures/leafcutter_hungarian/results/02_distance.png) | ![](figures/leafcutter_hungarian/results/02b_distance_unique.png) |
| **a.** AUPRC by distance (paired) | **b.** AUPRC by distance (unique positives) |

**Figure 24d.** AUPRC stratified by splice distance on hungarian-matched leafcutter.

## HAEC Hungarian

| | |
|:---:|:---:|
| ![](figures/haec_hungarian/matching/01_tier_distribution.png) | ![](figures/haec_hungarian/matching/05_distance_dist.png) |
| **a.** Tier distribution | **b.** Splice distance histogram |

**Figure 25a.** HAEC hungarian matching diagnostic.

| | |
|:---:|:---:|
| ![](figures/haec_hungarian/results/00a_delta_distributions.png) | ![](figures/haec_hungarian/results/01_overall_auprc.png) |
| **a.** Delta distributions | **b.** Overall AUPRC per model |

**Figure 25b.** Classification performance on hungarian-matched HAEC (single tissue).

| | |
|:---:|:---:|
| ![](figures/haec_hungarian/results/01b_pr_curves.png) | ![](figures/haec_hungarian/results/01d_roc_curves.png) |
| **a.** PR curves | **b.** ROC curves |

**Figure 25c.** PR and ROC curves.

| | |
|:---:|:---:|
| ![](figures/haec_hungarian/results/02_distance.png) | ![](figures/haec_hungarian/results/02b_distance_unique.png) |
| **a.** AUPRC by distance (paired) | **b.** AUPRC by distance (unique positives) |

**Figure 25d.** AUPRC stratified by splice distance on hungarian-matched HAEC.

## Hungarian vs Tiered Comparison

Side-by-side median AUPRC per model, tiered vs hungarian. The full numbers are saved to `figures/hungarian_vs_tiered_auprc.csv` (one row per dataset × model with `tiered_auprc`, `hungarian_auprc`, and `delta`). The key questions:

1. **Does the hungarian AUPRC match the tiered `dist_diff <= 50` bp filter result?** If yes, it confirms that the filter analysis was correctly isolating the well-matched subset; if hungarian is higher, it means the filter was losing power by discarding pairs rather than rematching them.
2. **Does the hungarian ranking differ from the tiered ranking?** Flips on the `dist_diff <= 50` filter (leafcutter Pangolin > SpliceAI; HAEC SpliceAI drops to #4) should reappear under hungarian since hungarian achieves the same distance balance at full N.
3. **Does SPLAIRE's relative position improve?** If SPLAIRE was the model most penalized by distance-confound (smallest drop when the filter was tightened), hungarian should narrow the gap to SpliceAI on txrevise and stay competitive on leafcutter / HAEC.

Values printed at stdout of `analysis.py`, and by row/model in `figures/hungarian_vs_tiered_auprc.csv`.

---

# Cross-Dataset Positive Variant Overlap

Comparison of positive variant sets across txrevise, leafcutter, and HAEC. Variants are deduplicated across all tissues before computing overlap.

## Overall Overlap

| | |
|:---:|:---:|
| ![](figures/overlap/overall_overlap.png) | ![](figures/overlap/overall_overlap_neg.png) |
| **a.** Positive variants | **b.** Negative variants |

**Figure 23.** Unique variant overlap across all three datasets. (a) Positive variants: txrevise and leafcutter share some positives (same genomic variant is an sQTL in both exon-level and intron-level analyses). HAEC uses a different cohort so overlap with GTEx is expected to be small. (b) Negative variants: drawn from GE summary statistics for both GTEx datasets, so substantial overlap is expected.

## Per-Tissue Overlap (GTEx)

| | |
|:---:|:---:|
| ![](figures/overlap/per_tissue_overlap.png) | ![](figures/overlap/per_tissue_overlap_neg.png) |
| **a.** Positive variants | **b.** Negative variants |

**Figure 24.** Per-tissue variant overlap between txrevise and leafcutter (49 GTEx tissues). Shared = variant appears in both datasets for that tissue. (a) Positives: overlap reflects variants that are sQTLs in both splicing quantification methods. (b) Negatives: both datasets draw negatives from GE summary statistics using the same 4-tier matching cascade, so shared negatives indicate variants matched to positives in both datasets.

## Dataset-Exclusive Swarm Plots

| | |
|:---:|:---:|
| ![](figures/overlap/swarm_txrevise_only.png) | ![](figures/overlap/swarm_leafcutter_only.png) |
| **a.** Txrevise-only positives | **b.** Leafcutter-only positives |

| | |
|:---:|:---:|
| ![](figures/overlap/swarm_shared_txrevise.png) | ![](figures/overlap/swarm_shared_leafcutter.png) |
| **c.** Shared positives (txrevise pairs) | **d.** Shared positives (leafcutter pairs) |

**Figure 25.** Per-tissue AUPRC for dataset-exclusive and shared positive variants. (a, b) Variants that are positive in only one dataset. (c, d) The same shared variants evaluated with their respective dataset's matched negatives. Differences between (c) and (d) reflect different negative matching (GE-sourced for txrevise vs sQTL-sourced for leafcutter).

---

# Tissue-Specific Output Head Evaluation

For GTEx tissues that map to a model's tissue-specific output head (SpliceTransformer: 15 tissues, Pangolin v1/v2: 4 tissues each), we compare the tissue-matched head's AUPRC against the best AUPRC from any other available output head. Points above the diagonal indicate the tissue-specific head is not the best choice for that tissue.

## Option A: Grid Scatter (3×2)

| | |
|:---:|:---:|
| ![](figures/tissue_heads/A_grid_scatter.png) | ![](figures/tissue_heads/A_grid_scatter_no_agg.png) |
| **a.** All heads | **b.** No aggregates |

**Figure 26a.** Tissue-specific head AUPRC (x) vs best alternative head AUPRC (y). 3 rows (SPT, Pangolin v1, Pangolin v2) × 2 columns (txrevise, leafcutter). Points above diagonal = tissue head loses. Left: aggregates included (always >= tissue head since they contain it). Right: aggregates excluded — only individual tissue heads, CLS/SSU, and SpliceAI as alternatives.

## Option B: Single Panel Overlay

| | |
|:---:|:---:|
| ![](figures/tissue_heads/B_single_scatter.png) | ![](figures/tissue_heads/B_single_scatter_no_agg.png) |
| **a.** All heads | **b.** No aggregates |

**Figure 26b.** Same data as 26a but all families overlaid in a single panel. Shape = model family (circle=SPT, triangle=Pangolin v1, square=Pangolin v2). Color = dataset (blue=txrevise, orange=leafcutter).

## Option C: Delta Swarm

| | |
|:---:|:---:|
| ![](figures/tissue_heads/C_delta_swarm.png) | ![](figures/tissue_heads/C_delta_swarm_no_agg.png) |
| **a.** All heads | **b.** No aggregates |

**Figure 26c.** AUPRC difference (tissue head - best other) per tissue. Each dot is a GTEx tissue. Negative = tissue head loses. Red dashed line at 0. Median annotated. Most direct visualization of whether tissue heads help.

## Option D: Paired Bar

| | |
|:---:|:---:|
| ![](figures/tissue_heads/D_paired_bar.png) | ![](figures/tissue_heads/D_paired_bar_no_agg.png) |
| **a.** All heads | **b.** No aggregates |

**Figure 26d.** Median AUPRC comparison: tissue-matched head (blue) vs best alternative (orange). Grouped by model family and dataset.

## Option E: Heatmap

| | |
|:---:|:---:|
| ![](figures/tissue_heads/E_heatmap_txrevise.png) | ![](figures/tissue_heads/E_heatmap_leafcutter.png) |
| **a.** Txrevise (all heads) | **b.** Leafcutter (all heads) |

| | |
|:---:|:---:|
| ![](figures/tissue_heads/E_heatmap_txrevise_no_agg.png) | ![](figures/tissue_heads/E_heatmap_leafcutter_no_agg.png) |
| **c.** Txrevise (no aggregates) | **d.** Leafcutter (no aggregates) |

**Figure 26e.** AUPRC heatmap: rows = GTEx tissues, columns = model tissue heads + best-other. Shows the full matrix of tissue × head performance.

---

# CLS vs SSU/Usage Margin Scatter

For each pos-neg pair, compute margin = pos_delta - neg_delta for both output heads (CLS and SSU/Usage). The scatter shows which head gets each pair right. Quadrant I = both win, II = head B rescues, IV = head A rescues, III = both fail.

## SPLAIRE (CLS vs SSU)

| | |
|:---:|:---:|
| ![](figures/cls_vs_ssu/sphaec_ref_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/sphaec_ref_leafcutter_margin_scatter.png) |
| **a.** Txrevise | **b.** Leafcutter |

**Figure 38.** SPLAIRE CLS vs SSU margin scatter.

## SPLAIRE-var (CLS vs SSU)

| | |
|:---:|:---:|
| ![](figures/cls_vs_ssu/sphaec_var_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/sphaec_var_leafcutter_margin_scatter.png) |
| **a.** Txrevise | **b.** Leafcutter |

**Figure 39.** SPLAIRE-var CLS vs SSU margin scatter.

## Pangolin v1 (CLS vs Usage)

| | |
|:---:|:---:|
| ![](figures/cls_vs_ssu/pangolin_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_leafcutter_margin_scatter.png) |
| **a.** Txrevise | **b.** Leafcutter |

**Figure 40.** Pangolin v1 CLS vs Usage margin scatter.

## Pangolin v2 (CLS vs Usage)

| | |
|:---:|:---:|
| ![](figures/cls_vs_ssu/pangolin_v2_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_v2_leafcutter_margin_scatter.png) |
| **a.** Txrevise | **b.** Leafcutter |

**Figure 41.** Pangolin v2 CLS vs Usage margin scatter.

## Pangolin v1 Per-Tissue (CLS vs Usage)

| | |
|:---:|:---:|
| ![](figures/cls_vs_ssu/pangolin_brain_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_brain_leafcutter_margin_scatter.png) |
| **a.** Brain — Txrevise | **b.** Brain — Leafcutter |
| ![](figures/cls_vs_ssu/pangolin_heart_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_heart_leafcutter_margin_scatter.png) |
| **c.** Heart — Txrevise | **d.** Heart — Leafcutter |
| ![](figures/cls_vs_ssu/pangolin_liver_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_liver_leafcutter_margin_scatter.png) |
| **e.** Liver — Txrevise | **f.** Liver — Leafcutter |
| ![](figures/cls_vs_ssu/pangolin_testis_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_testis_leafcutter_margin_scatter.png) |
| **g.** Testis — Txrevise | **h.** Testis — Leafcutter |

**Figure 42.** Pangolin v1 per-tissue CLS vs Usage margin scatter.

## Pangolin v2 Per-Tissue (CLS vs Usage)

| | |
|:---:|:---:|
| ![](figures/cls_vs_ssu/pangolin_v2_brain_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_v2_brain_leafcutter_margin_scatter.png) |
| **a.** Brain — Txrevise | **b.** Brain — Leafcutter |
| ![](figures/cls_vs_ssu/pangolin_v2_heart_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_v2_heart_leafcutter_margin_scatter.png) |
| **c.** Heart — Txrevise | **d.** Heart — Leafcutter |
| ![](figures/cls_vs_ssu/pangolin_v2_liver_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_v2_liver_leafcutter_margin_scatter.png) |
| **e.** Liver — Txrevise | **f.** Liver — Leafcutter |
| ![](figures/cls_vs_ssu/pangolin_v2_testis_txrevise_margin_scatter.png) | ![](figures/cls_vs_ssu/pangolin_v2_testis_leafcutter_margin_scatter.png) |
| **g.** Testis — Txrevise | **h.** Testis — Leafcutter |

**Figure 43.** Pangolin v2 per-tissue CLS vs Usage margin scatter.

---

# PIP Threshold Analysis

Classification performance as a function of the minimum PIP threshold used to select positive variants. The pip50 datasets include all variants with PIP >= 0.5, matched to negatives using the same 4-tier cascade. High-PIP variants get first pick of negatives (sorted by PIP descending before matching). By filtering pairs by `pos_pip` after scoring, we evaluate how label confidence affects benchmark results.

The `neg_var_key_ideal` column records the best negative each positive would have received if negatives were shared (ignoring exclusive matching). Comparing `neg_var_key` to `neg_var_key_ideal` quantifies whether lower-PIP positives got degraded negatives due to competition. Plots show AUPRC with actual negatives (solid lines) and ideal negatives (dashed lines).

## PIP Distributions

![](figures/haec_pip50/pip_distribution.png)

**a.** HAEC

![](figures/txrevise_pip50/pip_distribution.png)

**b.** Txrevise

![](figures/leafcutter_pip50/pip_distribution.png)

**c.** Leafcutter

**Figure 27.** Distribution of PIP values for pip50 matched positive variants (PIP >= 0.5). Dashed line indicates the standard PIP >= 0.9 threshold.

## AUPRC by PIP Threshold

![](figures/haec_pip50/auprc_by_pip_threshold.png)

**a.** HAEC

![](figures/txrevise_pip50/auprc_by_pip_threshold.png)

**b.** Txrevise

![](figures/leafcutter_pip50/auprc_by_pip_threshold.png)

**c.** Leafcutter

**Figure 28.** AUPRC and AUROC as a function of minimum PIP threshold. Solid lines = actual matched negatives, dashed lines = ideal negatives (what each positive would have gotten without exclusive matching). Performance at PIP >= 0.9 should match the original benchmark; lower thresholds include noisier positives. For GTEx datasets, AUPRC is the per-tissue median. Grey bars show pair counts at each threshold.

---

# Leafcutter Ambiguous Analysis

**Task:** Within-credible-set ranking. Given multiple variants in a credible set, does the model rank them by their fine-mapping probability (PIP)?

**Variants:** From GTEx leafcutter credible sets where no single variant dominates (max PIP < 0.9). All variants within each credible set are included. Filtered to credible sets with 2+ variants where all variants are near the same intron.

**Unit of analysis:** Each (credible set, tissue) pair.

## QC

| | |
|:---:|:---:|
| ![](figures/leafcutter_ambig/qc/02_cs_filtering_funnel.png) | ![](figures/leafcutter_ambig/qc/03_variants_per_cs.png) |
| **a.** Credible set filtering funnel | **b.** Distribution of CS sizes after filtering |

**Figure 29.** Leafcutter ambig QC: how many credible sets pass each filter and the resulting size distribution.

## Results

![](figures/leafcutter_ambig/results/02_delta_distributions.png)

**Figure 30.** Delta score distributions stratified by credible set size for the within-CS ranking task.

![](figures/leafcutter_ambig/results/03_pip_vs_delta.png)

**Figure 31.** Correlation between variant PIP and model delta score in ambiguous credible sets.

![](figures/leafcutter_ambig/results/09b_top1_by_cs_size.png)

**Figure 32.** Top-1 accuracy by credible set size. For each CS, does the highest-scoring variant match the highest-PIP variant?

### Binary Threshold Analysis

Youden's J optimal threshold is computed from the leafcutter pos/neg classification task and applied to within-CS variants.

| | |
|:---:|:---:|
| ![](figures/leafcutter_ambig/results/09c_exactly1_by_cs_size.png) | ![](figures/leafcutter_ambig/results/17_threshold_sweep.png) |
| **a.** Exactly-1 positive call rate by CS size | **b.** Precision and recall vs threshold sweep |

**Figure 33.** Binary threshold analysis: (a) percentage of credible sets with exactly one variant called positive using Youden's J threshold, stratified by CS size; (b) precision/recall at various delta score thresholds.

### Distribution of Positive Calls per CS

![](figures/leafcutter_ambig/results/09d_ncalled_dist_by_cs_size.png)

**a.** Youden's J threshold

![](figures/leafcutter_ambig/results/09e_ncalled_dist_by_cs_size_t02.png)

**b.** Fixed threshold = 0.2

**Figure 34.** Distribution of positive calls per CS by size: how many variants per credible set exceed the threshold.

![](figures/leafcutter_ambig/results/09f_ncalled_bucketed_by_cs_size.png)

**a.** Youden's J threshold (bucketed 0/1/2/3+)

![](figures/leafcutter_ambig/results/09g_ncalled_bucketed_by_cs_size_t02.png)

**b.** Fixed threshold = 0.2 (bucketed 0/1/2/3+)

**Figure 35.** Same as Figure 34 but with bucketed categories (0, 1, 2, 3+ positive calls).

| | |
|:---:|:---:|
| ![](figures/leafcutter_ambig/results/09h_ncalled_bucketed_vertical.png) | ![](figures/leafcutter_ambig/results/09i_ncalled_bucketed_vertical_t02.png) |
| **a.** Youden's J threshold (vertical layout) | **b.** Fixed threshold = 0.2 (vertical layout) |

**Figure 36.** Vertical layout of bucketed positive calls per CS by size.

| | |
|:---:|:---:|
| ![](figures/leafcutter_ambig/results/09j_ncalled_sphaec_ref.png) | ![](figures/leafcutter_ambig/results/09k_ncalled_sphaec_ref_t02.png) |
| **a.** SPLAIRE-ref, Youden's J threshold | **b.** SPLAIRE-ref, fixed threshold = 0.2 |

**Figure 37.** SPLAIRE-ref only: bucketed positive calls per CS by size.

---

# Delta Position Analysis

For each variant the delta score is max |alt - ref| within ±2 kb. The **offset** is the distance (in bp) from the variant center to the position where that maximum occurs. Offset = 0 means the model's strongest predicted effect is at the variant itself. Large offsets indicate the model places the effect at a nearby splice site or other sequence feature.

## Offset Distributions (Primary Models)

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/01_offset_distribution.png) | ![](figures/leafcutter/delta_position/01_offset_distribution.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/01_offset_distribution.png)

**c.** HAEC

**Figure 44.** Distribution of delta position offset for positive vs negative variants. Each unique variant contributes one offset value per model. Log-scale y-axis. Offset = 0 means the model's max delta is at the variant itself; offset = ±2000 means it is at the edge of the ±2 kb window. Differences between pos and neg distributions reflect whether models preferentially find effects near positive variants.

## Median |Offset| by Output Head (All Heads)

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/02_median_offset_all_heads.png) | ![](figures/leafcutter/delta_position/02_median_offset_all_heads.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/02_median_offset_all_heads.png)

**c.** HAEC

**Figure 45.** Median absolute offset for positive variants, ranked across all output heads. Heads with smaller median |offset| tend to place the max delta closer to the variant; larger values indicate the head tends to find effects at more distant positions (e.g. nearby splice sites). Includes per-tissue, aggregate, CLS, SSU/usage, and combined heads.

## Fraction at/near Variant (All Heads)

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/03_frac_at_variant.png) | ![](figures/leafcutter/delta_position/03_frac_at_variant.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/03_frac_at_variant.png)

**c.** HAEC

**Figure 46.** Percentage of variants with |offset| ≤ 10 bp (max delta at or very near the variant) for positive (red) and negative (green) variants, per output head. Higher values mean the head more often finds its strongest effect at the variant itself. Comparing pos vs neg reveals whether models are more likely to find effects at positive variant positions.

## |Offset| vs Splice Distance (Positives)

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/04_offset_vs_splice_dist.png) | ![](figures/leafcutter/delta_position/04_offset_vs_splice_dist.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/04_offset_vs_splice_dist.png)

**c.** HAEC

**Figure 47.** Absolute offset vs annotated splice distance for positive variants. If the model correctly identifies the nearby splice site, points should fall near the y = x diagonal. Points far below the diagonal mean the model places the max delta closer to the variant than the annotated splice site. Log-log scale. Spearman ρ annotated.

## AUPRC by Offset Bin (Primary Models)

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/05_auprc_by_offset.png) | ![](figures/leafcutter/delta_position/05_auprc_by_offset.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/05_auprc_by_offset.png)

**c.** HAEC

**Figure 48.** AUPRC stratified by |offset| of the positive variant's delta position. Pairs are binned by the positive's absolute offset from center. Variants where the model places the max delta near the variant (0-10 bp) may be easier to classify than those where the max occurs hundreds of bp away. For GTEx, AUPRC is tissue median. Grey bars show variant counts per bin.

## AUPRC: At-Variant vs Elsewhere (All Heads)

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/06_auprc_near_vs_far.png) | ![](figures/leafcutter/delta_position/06_auprc_near_vs_far.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/06_auprc_near_vs_far.png)

**c.** HAEC

**Figure 49.** AUPRC comparison for all output heads when the positive's max delta is at/near the variant (|offset| ≤ 10 bp, blue) vs elsewhere (|offset| > 10 bp, orange). Shows whether finding the effect at the variant itself correlates with better classification accuracy.

## Cross-Model Offset Concordance

| | |
|:---:|:---:|
| ![](figures/txrevise/delta_position/07_cross_model_concordance.png) | ![](figures/leafcutter/delta_position/07_cross_model_concordance.png) |
| **a.** Txrevise | **b.** Leafcutter |

![](figures/haec/delta_position/07_cross_model_concordance.png)

**c.** HAEC

**Figure 50.** Cross-model concordance of delta position offsets for positive variants. Each point is one variant; axes show the offset from each model. Points on the diagonal = both models place the max delta at the same position. Spearman ρ and exact-match percentage annotated.
