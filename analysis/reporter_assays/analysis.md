# Reporter Assay Benchmarks

Benchmarking splice variant effect prediction on Vex-seq and MFASS reporter assays. Reference genome: hg19.

## Assay comparison

| | Vex-seq (Adamson 2018) | MFASS (Chong 2019) |
|---|---|---|
| **Variants** | 2,055 (from 2,059 designed) | 27,733 ExAC SNVs |
| **Exons** | 110 alternatively spliced | 2,198 constitutive (<100 bp, inclusion ≥0.8) |
| **Readout** | RNA-seq PSI (continuous, %) | FACS + DNA-seq (bimodal inclusion index, 0–1) |
| **Delivery** | Episomal transfection | Single-copy genomic integration |
| **Cell line** | HepG2 (used here; K562 also available) | HEK293T |
| **Intron context** | 50 bp upstream, 20 bp downstream | ≥40 bp upstream, ≥30 bp downstream |
| **SDV threshold** | \|ΔPSI\| > 0.10 (our convention) | \|ΔPSI\| > 0.50 (Chong: Δinclusion ≤ -0.5) |
| **SDV rate** | — | 3.8% (1,050 / 27,733) |

Key difference: Vex-seq tests alternatively spliced exons (graded effects), MFASS tests constitutive exons (large disruptions only).

## Vex-seq data

Source: [MMSplice GitHub](https://github.com/gagneurlab/MMSplice_paper/tree/master/data/vexseq) — 957 train (chr1-8) + 1,098 test (chr9-22+X) = 2,055 unique variants across 110 exons. CAGI5 split. HepG2 ΔPSI in percentage units (÷100 for 0–1 scale).

| Property | Train | Test | Combined |
|----------|-------|------|----------|
| Variants | 957 | 1,098 | 2,055 |
| SNVs / indels | 906 / 51 | 1,054 / 44 | 1,960 / 95 |
| ΔPSI mean (std) | -2.12 (14.33) | -1.84 (13.44) | -1.97 (13.86) |

## MFASS data

Source: [MFASS GitHub](https://github.com/KosuriLab/MFASS/tree/master/processed_data/snv) — `snv_data_clean.txt`. Filtering: 32,669 raw → 28,972 mutant → 27,733 after removing 1,239 with no `v2_dpsi`. 100% SNVs. Exon sizes 18–99 bp (median 81). Strand validated against hg19.

| Property | Value |
|----------|-------|
| Variants | 27,733 |
| Exons | 2,185 unique (of 2,198 backgrounds) |
| ΔPSI range | -0.998 to 0.476 |
| ΔPSI mean (std) | 0.004 (0.181) |
| SDVs | 1,050 (3.8%) — all loss-of-inclusion (max ΔPSI = 0.476 < 0.5) |
| By location | 14,130 exon, 7,595 upstream intron, 6,008 downstream intron |

**Replicate consistency.** MFASS has two biological replicates (`v1_dpsi`, `v2_dpsi`). We use v2 (27,733 variants measured) as do all other papers. Only 2,043 variants have both replicates. Between-replicate correlation: Pearson $r$ = 0.85, Spearman $\rho$ = 0.47. SDV concordance: 197 of 234 v1 SDVs (84%) are also SDVs in v2. The high Pearson but low Spearman indicates replicates agree on large effects but diverge on ranking small effects — consistent with the bimodal readout having limited sensitivity to small changes.

## Variant count verification

### Splice site definition

We use the Mount et al. (2019) CAGI5 definition: **3 exonic + 8 intronic nt** at each junction. This differs from Chong's narrower "splice site" (2 bp intron only) — their "splice region" (3 exon + 8 intron) is closer to ours. Our `get_location_masks()` correctly implements the Mount definition.

### Cross-paper variant counts

| Paper | MFASS $n$ | SDVs | Notes |
|-------|----------|------|-------|
| Chong 2019 | 27,733 | 1,050 | original; Δinclusion ≤ -0.5 |
| Pangolin (Zeng 2022) | 27,733 | ~1,050 | v2 weights; AUPRC = 0.56 |
| DeltaSplice (Xu 2024) | — | 1,048 | strict < -0.5 (2 variants at boundary excluded) |
| AlphaGenome (Avsec 2026) | **28,256** | — | **523 extra variants** from v1 replicate |
| **Our pipeline** | **27,733** | **1,050** | matches Chong/Pangolin exactly |

**AlphaGenome's 28,256:** The raw data has 28,972 mutant rows. 27,733 have `v2_dpsi`; 1,239 are missing it but 1,121 have `v1_dpsi`. AlphaGenome included 523 of these v1-only variants. Their AUPRC numbers (0.54 AlphaGenome, 0.51 Pangolin) are on this larger set and not directly comparable to Pangolin's own 0.56.

### Cross-paper benchmark numbers

| Paper | Model | MFASS AUPRC | $n$ |
|-------|-------|-------------|-----|
| Zeng & Li 2022 | Pangolin (v2) | **0.56** | 27,733 |
| Zeng & Li 2022 | SpliceAI | 0.47 | 27,733 |
| Avsec et al. 2026 | AlphaGenome | 0.54 | 28,256 |
| Avsec et al. 2026 | Pangolin | 0.51 | 28,256 |
| Avsec et al. 2026 | SpliceAI | 0.49 | 28,256 |
| Avsec et al. 2026 | DeltaSplice | 0.49 | 28,256 |

Pangolin v1 vs v2 and variant set size both contribute to the 0.51 vs 0.56 gap.

## Data overview

**Distribution of measured ΔPSI, variant distance to nearest splice site, and variant counts by location category (Mount et al. 2019).** **(a)** Histogram of measured ΔPSI. Dotted vertical lines mark ±1 SD. The distributions are centered near zero and right-skewed (more loss-of-inclusion than gain). **(b)** Distance from each variant to its nearest splice site (acceptor or donor). Most variants are within 50 bp of a splice site due to the short flanking intronic context in both assays. **(c)** Variant counts by location category: exon (excluding splice sites), intron (excluding splice sites), and splice site (3 exonic + 8 intronic nt per junction, following Mount et al. 2019).

![Vex-seq data overview](output/vexseq_data_overview.png)

![MFASS data overview](output/mfass_data_overview.png)

# Model Comparison

Both datasets were scored with six models (SpliceAI, Pangolin, Pangolin v2, SpliceTransformer, SPLAIRE, and SPLAIRE-var). Each model produces scores at both exon boundaries (acceptor and donor splice sites) for reference and alternative alleles. We compute delta scores (alt − ref) and average over the acceptor and donor scores of the given exon. Performance is evaluated using Pearson correlation, Spearman correlation, and AUPRC against measured ΔPSI. For MFASS, variants with |ΔPSI| > 0.5 define the positive class; for Vex-seq, |ΔPSI| > 0.10.

---

## Figures

### Figure 1. Precision-recall curves by variant location

**Precision-recall curves for splice-disrupting variant classification, stratified by variant location.** Variants are classified as splice-disrupting if |ΔPSI| exceeds 0.10 for Vex-seq or 0.50 for MFASS. Variant locations follow the categories of Mount et al. (2019): splice site (3 exonic and 8 intronic nucleotides immediately adjacent to each splice junction), exon (remaining exonic positions), and intron (remaining intronic positions). **(a)** All variants. **(b)** Exonic variants only. **(c)** Intronic variants only. **(d)** Splice site variants only. Dashed gray line indicates the positive class prevalence (baseline). Each model uses its best single output head as determined by Figure S1: Pangolin and Pangolin v2 use testis $P$(splice), SpliceTransformer uses the best tissue-specific usage head, and SPLAIRE, SPLAIRE-var, and SpliceAI use their classification outputs. Top: Vex-seq ($n$ = 2,055 variants, 431 positives). Bottom: MFASS ($n$ = 27,733 variants, 1,050 positives).

![Vex-seq PR curves](output/vexseq_pr.png)

![MFASS PR curves](output/mfass_pr.png)

---

### Figure 2. Threshold sensitivity

**Sensitivity of AUPRC to the splice-disrupting variant threshold.** The |ΔPSI| threshold defining the positive class is swept from 0.01 to 1.0 and AUPRC is recomputed at each value. The dotted vertical line marks the 1-SD threshold (standard deviation of measured ΔPSI: 0.139 for Vex-seq, 0.181 for MFASS). **(a)** AUPRC versus threshold for each model. At high thresholds the positive class shrinks and AUPRC estimates become unstable. **(b)** Number of positive and negative variants as a function of threshold, showing how class balance shifts across the sweep. Relative model rankings are consistent across the full range of reasonable thresholds. Vex-seq: $n$ = 2,055. MFASS: $n$ = 27,733.

![Vex-seq threshold sensitivity](output/vexseq_threshold.png)

![MFASS threshold sensitivity](output/mfass_threshold.png)

---

### Figure 3. Model performance on reporter assays

**Density scatter plots of measured versus predicted ΔPSI and grouped bar charts of model performance metrics, stratified by variant location.** **(a, c)** Log-density scatter plots of measured versus predicted ΔPSI for each model, colored by kernel density estimate (log-transformed). $R^2$ (coefficient of determination) is annotated per panel. **(b, d)** Pearson correlation, Spearman correlation, and AUPRC stratified by variant location (All, Splice site, Exon, Intron), following the categories of Mount et al. (2019). Significance brackets denote pairwise comparisons by paired bootstrap resampling ($n$ = 10,000) with Holm–Bonferroni correction ($k$ = 10 comparisons per metric). Panels (a, b): Vex-seq ($n$ = 2,055). Panels (c, d): MFASS ($n$ = 27,733). All variants included regardless of measured effect size.

![Full grouped bars](../../figures/fig3/main/reporter_full_grouped_bars.png)

---

### Figure 3 — Layout alternatives

Five alternative bar chart layouts for Figure 3 (MFASS data shown). "All" is redundant (weighted average of other categories). Location categories follow Chong et al. (2019).

**Option A** — 3 panels (Exon, Intron, SS combined). More room for stars.

![Option A](../../figures/fig3/main/layout_option_A.png)

**Option B** — 4 Chong categories (Exon, Intron, Splice region, Splice site). Matches Chong exactly.

![Option B](../../figures/fig3/main/layout_option_B.png)

**Option C** — Metrics as panels, locations as grouped bars within. Reviewers compare models within a metric.

![Option C](../../figures/fig3/main/layout_option_C.png)

**Option D** — Dot/forest plot. Models on y-axis, metric on x-axis, shapes = location. Most compact.

![Option D](../../figures/fig3/main/layout_option_D.png)

**Option E** — 4 Chong categories, no significance stars (moved to supplement). Cleanest.

![Option E](../../figures/fig3/main/layout_option_E.png)

---

### Figure 4. All pairwise model comparisons

**Pairwise statistical comparisons of model performance across datasets and variant locations.** Grid of bar charts (2 datasets × 4 location subsets). For each location subset, Pearson correlation, Spearman correlation, and AUPRC are shown as grouped bars for all models. Significance brackets are drawn for all ${6 \choose 2}$ = 15 model pairs. $P$-values were computed by paired bootstrap resampling ($n$ = 10,000, seed = 42) with Holm–Bonferroni correction applied separately within each metric. Asterisks denote significance levels: \* $p$ < 0.05, \*\* $p$ < 0.01, \*\*\* $p$ < 0.001.

![Full all pairs](../../figures/fig3/main/reporter_full_allpairs.png)

---

## Supplemental Figures

### Figure S1. All model outputs ranked by metric

**Horizontal bar charts ranking every model output by Pearson correlation, Spearman correlation, and AUPRC.** Each model produces multiple outputs (tissue-specific scores, classification heads, cross-tissue max-aggregated scores). All outputs are evaluated against measured ΔPSI and sorted by AUPRC (row order shared across all three panels). Bars are colored by parent model. This figure is used to identify the best-performing output head for each model in downstream analyses. **(a)** Vex-seq ($n$ = 2,055). **(b)** MFASS ($n$ = 27,733).

![Vex-seq all outputs](../../figures/fig3/sup/vexseq_all_outputs.png)

![MFASS all outputs](../../figures/fig3/sup/mfass_all_outputs.png)

---

### Figure S2. Measured ΔPSI by variant position

**Measured ΔPSI for each variant as a function of position relative to the nearest splice site.** Each variant is assigned to the nearest splice site (acceptor or donor) based on genomic distance. Negative x-values indicate intronic positions; positive values indicate exonic positions. An exon schematic is drawn below each panel. Splice site variants produce the largest measured effects, while exonic and deep intronic variants cluster near zero ΔPSI. **(a)** Vex-seq ($n$ = 2,055). **(b)** MFASS ($n$ = 27,733).

![Vex-seq measured ΔPSI by position](../../figures/fig3/sup/vexseq_dpsi_by_position.png)

![MFASS measured ΔPSI by position](../../figures/fig3/sup/mfass_dpsi_by_position.png)

---

### Figure S3. Prediction error by variant position

**Signed prediction error (predicted − measured ΔPSI) at each position relative to the nearest splice site.** Layout as in Figure S2. Only SpliceAI and SPLAIRE are shown as representative models to illustrate position-dependent error patterns. Systematic overprediction at canonical splice sites (where true effects are large) and near-zero errors in the exon interior are visible in both models. **(a, b)** Vex-seq. **(c, d)** MFASS.

![Vex-seq SpliceAI error by position](../../figures/fig3/sup/vexseq_spliceai_error_by_position.png)

![Vex-seq SPLAIRE error by position](../../figures/fig3/sup/vexseq_sphaec_ref_error_by_position.png)

![MFASS SpliceAI error by position](../../figures/fig3/sup/mfass_spliceai_error_by_position.png)

![MFASS SPLAIRE error by position](../../figures/fig3/sup/mfass_sphaec_ref_error_by_position.png)

---

### Figure S4. Measured effect size by position

**Mean (solid line) and median (dashed line) |ΔPSI| at each position relative to the nearest splice site, binned at single-nucleotide resolution.** This establishes the baseline difficulty for prediction: positions near the canonical AG acceptor and GT donor dinucleotides show the largest measured effects, with a sharp decay into the exon interior and flanking intron. The mean exceeds the median at most positions, reflecting the right-skewed distribution of effect sizes. Vex-seq: $n$ = 2,055 variants across 110 exons. MFASS: $n$ = 27,733 variants across 2,198 exons.

![Vex-seq measured PSI by position](../../figures/fig3/sup/vexseq_measured_psi_by_position.png)

![MFASS measured PSI by position](../../figures/fig3/sup/mfass_measured_psi_by_position.png)

---

### Figure S4b. Rigorous effect size by position (LOWESS + exon normalization)

**LOWESS-smoothed effect size as a function of position relative to the nearest splice site, with bootstrap 95% confidence intervals.** 2 × 2 panels per dataset: top row shows raw |ΔPSI|, bottom row shows exon-normalized |$z$|-scores. Exon normalization computes a $z$-score within each exon ($z_i = (y_i - \bar{y}_{\text{exon}}) / \sigma_{\text{exon}}$), removing the per-exon baseline inclusion level so that the positional profile reflects purely within-exon positional effects rather than differences between exons. LOWESS smoothing (fraction = 0.15) adapts bandwidth to local data density — narrower near splice sites where variants are dense, wider in the exon interior and deep intron where variants are sparse — providing a more statistically principled curve than fixed 1-nt binning. Shaded bands show 95% bootstrap confidence intervals ($n$ = 200 resamples). Individual variants are shown as gray scatter points (subsampled to 2,000 for readability where needed). If the raw and $z$-scored rows show the same positional profile, the effect is driven by variant position within the exon rather than by a few high-inclusion exons dominating certain positions. Vex-seq: $n$ = 2,055 variants across 110 exons. MFASS: $n$ = 27,733 variants across 2,198 exons.

![Vex-seq rigorous effect by position](../../figures/fig3/sup/vexseq_effect_by_position_rigorous.png)

![MFASS rigorous effect by position](../../figures/fig3/sup/mfass_effect_by_position_rigorous.png)

---

### Figure S4c. %SDV by normalized exon position

**Comprehensive positional analysis of splice-disrupting variants and model performance (cf. Chong et al. 2019, Figure 4C).** Exon positions are normalized to a [0, 1] fraction so exons of different widths align; intron positions are in raw base pairs. From top to bottom:

1. **%SDV** — Percentage of splice-disrupting variants at each position (symlog y-axis). Splice donor and acceptor regions show the highest SDV rates with sharp decay into the exon interior.
2. **SDV recall by position** — For each model, the percentage of true SDVs correctly flagged (|Δscore| above model-specific threshold) at each position bin. Directly shows whether each model's recall tracks the spatial distribution of SDVs — models whose curves follow the %SDV profile are finding SDVs wherever they occur; models with recall gaps at specific positions have positional blind spots.
3. **Per-region AUPRC** — Grouped bar chart of AUPRC for each model across the four Chong categories (Splice site, Splice region, Exon, Intron). Variant counts and SDV counts annotated below each group. Places model classification performance in direct spatial context alongside the positional analysis above.
4. **Per-region summary table** — Compact table showing $n$, number of SDVs, %SDV rate, best model by AUPRC, and best model's Pearson correlation for each region. Provides a quick reference for the key numbers underlying the figure.
5. **%SDV by exon width** — Same as row 1 but stratified by exon width (split at median), with the overall curve in gray. Tests whether short exons (where acceptor/donor signals may interact) have a different positional profile than long exons.
6. **Nucleotide substitution heatmap** — %SDV for each of the 12 possible single-nucleotide substitutions at each position. Reveals which substitution types are most disruptive at specific locations (e.g., G→T at the +1 donor position).
7. **Recall by distance from nearest splice site** — For each model, the cumulative fraction of true SDVs correctly flagged (|Δscore| above model-specific threshold) within increasing distance from the nearest splice site. Answers: "if I only trust model predictions within $X$ bp of a splice site, what percentage of SDVs do I capture?" A gray dotted line shows the total fraction of SDVs within each distance regardless of model prediction. Models whose curves track the gray line are capturing SDVs wherever they occur; models that plateau early are effective only near splice sites.
8. **Precision by region** — Grouped bar chart of model precision (fraction of flagged variants that are true SDVs) across six coarse location categories: canonical splice site (2 bp intronic), extended splice site (remaining Mount positions), near-exon (≤10 bp from SS), exon interior (>10 bp from SS), near-intron (≤10 bp from SS), and deep intron (>10 bp from SS). Coarse bins ensure sufficient $n$ for stable estimates. Variant counts and SDV counts are annotated below each group.
9. **MAE** — Mean absolute error by position with all models overlaid. Shows where models collectively fail, typically at canonical splice sites where true effects are large.
10. **Model disagreement** — Standard deviation of |Δscore| across all models at each position. High disagreement identifies positions where model architectures diverge — these are the positions where model choice matters most.
11. **phyloP conservation** — Mean mammalian phyloP score by position (included only if the hg19 phyloP bigWig is available). Confirms whether high-%SDV positions are under evolutionary constraint.
12. **SNV density** — Variant count per bin, establishing statistical power at each position.

Prediction thresholds for cumulative recall and precision are set per model as the median |Δscore| among true SDVs. Summary statistics comparing Chong's narrow splice site definition (2 bp intronic) with the Mount et al. (2019) definition (3 exonic + 8 intronic nt) are printed below each plot.

![Vex-seq %SDV by position](../../figures/fig3/sup/vexseq_pct_sdv_by_position.png)

![MFASS %SDV by position](../../figures/fig3/sup/mfass_pct_sdv_by_position.png)

---

### Figure S5. MAE by position for all models

**Per-model mean absolute error at each position relative to the nearest splice site, binned at single-nucleotide resolution.** All six models are plotted on the same axes. On MFASS, all models produce nearly identical error profiles, indicating that position-dependent difficulty dominates over model choice. On Vex-seq, model-to-model differences are more apparent at the donor splice site, where SPLAIRE and SPLAIRE-var show lower MAE than SpliceAI and Pangolin.

![Vex-seq MAE by position](../../figures/fig3/sup/vexseq_mae_by_position.png)

![MFASS MAE by position](../../figures/fig3/sup/mfass_mae_by_position.png)

---

### Figure S6. Prediction error vs measured effect size

**Scatter plot of |prediction error| versus |measured ΔPSI| with binned trend line (black).** Each subpanel shows one model. All models show a positive relationship between measured effect size and prediction error magnitude, reflecting the limited dynamic range of model predictions relative to the full range of measured ΔPSI values. The trend is steeper on Vex-seq (ΔPSI in percentage units, range up to ~90%) than MFASS (inclusion index, range up to ~1.0).

![Vex-seq error vs effect](../../figures/fig3/sup/vexseq_error_vs_effect.png)

![MFASS error vs effect](../../figures/fig3/sup/mfass_error_vs_effect.png)

---

### Figure S7. Prediction error vs flanking intron length

**Scatter plot of |prediction error| versus flanking intron length (bp) with binned trend line.** For each tested exon, the nearest neighboring exon on the same strand upstream and downstream was identified from GENCODE v19 annotations (strand-matched). Intron length is the distance from the tested exon boundary to the nearest annotated neighboring exon. This measures how much real genomic splicing context exists beyond the reporter construct — shorter introns mean the model's 10 kb input window can reach a neighboring exon and its splice signals. Three versions are shown using: **(a)** the minimum of the upstream and downstream intron, **(b)** the maximum, and **(c)** the mean. Each subpanel shows one model. X-axis is in base pairs (linear scale).

**Min flanking intron:**

![Vex-seq error vs min intron length](../../figures/fig3/sup/vexseq_error_vs_min_intron_length.png)

![MFASS error vs min intron length](../../figures/fig3/sup/mfass_error_vs_min_intron_length.png)

**Max flanking intron:**

![Vex-seq error vs max intron length](../../figures/fig3/sup/vexseq_error_vs_max_intron_length.png)

![MFASS error vs max intron length](../../figures/fig3/sup/mfass_error_vs_max_intron_length.png)

**Mean flanking intron:**

![Vex-seq error vs mean intron length](../../figures/fig3/sup/vexseq_error_vs_mean_intron_length.png)

![MFASS error vs mean intron length](../../figures/fig3/sup/mfass_error_vs_mean_intron_length.png)

---

### Figure S8. Prediction error vs allele frequency

**Scatter plot of |prediction error| versus $\log_{10}$(gnomAD allele frequency) with binned trend line.** Each subpanel shows one model. Only variants present in gnomAD are included. Sequence-based models do not use population frequency information, so any observed trend reflects the correlation between allele frequency and the nature of the variant (e.g., rare variants are more likely to disrupt splice sites and have larger effects, producing larger errors).

![Vex-seq error vs MAF](../../figures/fig3/sup/vexseq_error_vs_maf.png)

![MFASS error vs MAF](../../figures/fig3/sup/mfass_error_vs_maf.png)

---

### Figure S9. Measured effect size vs allele frequency

**Scatter plot of |measured ΔPSI| versus $\log_{10}$(gnomAD allele frequency) with median trend line (black).** Spearman $\rho$ is annotated. Rarer variants tend to have larger measured splice effects, consistent with purifying selection against splice-disrupting alleles. This relationship is expected: strongly deleterious splice variants are maintained at low frequency by natural selection. Only variants present in gnomAD are included.

![Vex-seq effect vs MAF](../../figures/fig3/sup/vexseq_effect_vs_maf.png)

![MFASS effect vs MAF](../../figures/fig3/sup/mfass_effect_vs_maf.png)

---

## Bootstrap Significance Testing

### Statistical comparison methods

Variants were classified as splice-altering if |ΔPSI| exceeded 0.10 for Vex-seq or 0.50 for MFASS. Model performance was assessed using three metrics: Pearson correlation ($r$), Spearman rank correlation ($\rho$), and area under the precision-recall curve (AUPRC) between predicted delta scores and measured ΔPSI.

Statistical significance was determined by paired bootstrap resampling ($n$ = 10,000, seed = 42). In each bootstrap iteration, variant indices were resampled with replacement and applied identically across all models, preserving the paired structure of model comparisons. For each pair of models, the difference in the metric of interest was computed on each bootstrap sample. Two-sided $p$-values were calculated as $2 \times \min(p_{\text{left}},\ 1 - p_{\text{left}})$, where $p_{\text{left}}$ is the fraction of bootstrap differences $\leq 0$. Holm–Bonferroni correction was applied separately within each metric, with $k$ = 10 (or 15 with Pangolin v2) pairwise comparisons.

---

### Figure S11. Correction sensitivity

**Heatmap comparing significance levels across multiple-testing correction methods.** 3 × 4 grid: rows correspond to metrics (Pearson, Spearman, AUPRC); columns correspond to correction methods (Holm $k$ = 4, Holm $k$ = 10 standard bootstrap, Bonferroni $k$ = 4, Holm $k$ = 10 cluster bootstrap). Each cell indicates the significance level (ns, \*, \*\*, \*\*\*) for one model pair in one context (dataset × filter state × location subset = 16 contexts). Black borders mark cells where significance differs from the reference method (Holm $k$ = 4, column 1). The overall pattern of significant and non-significant comparisons is robust to the choice of correction method.

![Correction sensitivity](../../figures/fig3/sup/correction_sensitivity.png)
