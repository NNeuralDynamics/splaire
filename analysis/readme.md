# analysis

benchmarking, evaluation, attribution, and figures for the SPLAIRE paper

## subfolders

### test
test-set scoring and metrics. scores held-out donors with SPLAIRE, SpliceAI, Pangolin, and SpliceTransformer. includes GENCODE and MANE Select canonical benchmarks. see `test/readme.md`.

### sqtl_bench
sQTL fine-mapping benchmark. downloads eQTL Catalogue credible sets, builds matched positive/negative variant sets (txrevise, leafcutter, HAEC), scores with all models. see `sqtl_bench/readme.md`.

### explain
DeepLIFT-SHAP attribution analysis for SPLAIRE models. extracts splice-site-centered sequences, runs attribution, normalizes. see `explain/readme.md`.

### reporter_assays
VexSeq and MFASS reporter assay evaluation. prepares variant datasets from published reporter assay data, scores with all models, compares predicted delta-PSI to measured values.
