# analysis

benchmarking, evaluation, attribution, and figures for the SPLAIRE paper.

## reproducing figures

each figure in the paper maps to one script. all scripts read pre-computed predictions from disk (scored on GPU via the sbatch scripts in each subfolder).

```bash
# figures 1-2: splice site landscape + model comparison
cd analysis/test && python analysis.py

# figure 3: vex-seq + mfass reporter assays
cd analysis/reporter_assays && python analysis.py

# figure 4: sqtl fine-mapping benchmark
cd analysis/sqtl_bench && python analysis.py

# figure 5: deeplift-shap attribution
cd analysis/explain && python analysis.py
```

each script downloads external data if needed, loads scored predictions, computes metrics, and saves figures to `figures/fig{1-5}/{main,sup}/`.

## environment variables

scripts default to cluster paths. external users override with:

| variable | what |
|---|---|
| `SPLAIRE_DATA_DIR` | root for splice tables, h5s, predictions |
| `SPLAIRE_CANONICAL_DIR` | canonical benchmark data (gencode, mane, pangolin test) |
| `SPLAIRE_SQTL_DIR` | sqtl benchmark data |

## subfolders

### test
test-set scoring and metrics. scores held-out donors with SPLAIRE, SpliceAI, Pangolin, and SpliceTransformer. includes GENCODE and MANE Select canonical benchmarks. see `test/readme.md`.

### sqtl_bench
sQTL fine-mapping benchmark. downloads eQTL Catalogue credible sets, builds matched positive/negative variant sets (txrevise, leafcutter, HAEC), scores with all models. see `sqtl_bench/readme.md`.

### explain
DeepLIFT-SHAP attribution analysis for SPLAIRE models. extracts splice-site-centered sequences, runs attribution, normalizes. see `explain/readme.md`.

### reporter_assays
VexSeq and MFASS reporter assay evaluation. scores with all models, compares predicted delta-PSI to measured values. see `reporter_assays/readme.md`.
