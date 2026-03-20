# explain

deeplift-shap attribution analysis for splaire models

## setup

set paths:

```bash
REPO=<path_to_cloned_repo>
TISSUE_DIRS=<space-separated paths to pipeline output dirs per tissue>
```

reference files must be in `pipeline/reference/GRCh38/` (genome fasta, protein_coding_genes.tsv, paralogs)

## extract sequences

```bash
python src/extract_splice_sequences.py \
    --tissue-dirs $TISSUE_DIRS \
    --tissues whole_blood brain_cortex haec lung testis \
    --genome ${REPO}/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa \
    --protein-coding ${REPO}/pipeline/reference/GRCh38/protein_coding_genes.tsv \
    --paralogs ${REPO}/pipeline/reference/GRCh38/paralogs_GRCh38.txt.gz \
    --test-chroms \
    --output data/sequences.h5
```

## attribution

```bash
python src/run_attribution.py \
    --model splaire_ref_reg \
    --input data/sequences.h5 \
    --output data/attr_splaire_ref_reg.h5

python src/run_attribution.py \
    --model splaire_ref_cls \
    --input data/sequences.h5 \
    --output data/attr_splaire_ref_cls.h5
```

for large datasets, split first:

```bash
python src/split_h5.py data/sequences.h5 4
# runs attribution on each split, then:
python src/combine_h5.py data/attr_splaire_ref_reg.h5 4
```

## normalize attributions

```bash
python src/add_normalized_attributions.py \
    --attr data/attr_splaire_*.h5 \
    --seq data/sequences.h5
```

## available splaire models

| model | heads |
|-------|-------|
| splaire_ref_reg | reg |
| splaire_ref_cls | neither, acceptor, donor |
| splaire_var_reg | reg |
| splaire_var_cls | neither, acceptor, donor |

## tangermeme deep_lift_shap parameters

| parameter | default | ours | note |
|-----------|---------|------|------|
| batch_size | 32 | 64 | increased for speed |
| hypothetical | False | **True** | get all 4 channels |
| warning_threshold | 0.001 | **0.05** | suppress minor convergence warnings |
| random_state | None | **42** | reproducibility |

## output files

```
data/attr_splaire_ref_reg.h5
├── attr_reg: (N, 10001, 4)     # hypothetical attributions
├── pred_reg: (N,)              # predictions at center
└── attrs: model, input, n_sequences, n_shuffles, shuffle_method, seed

data/attr_splaire_ref_cls.h5
├── attr_neither: (N, 10001, 4)
├── attr_acceptor: (N, 10001, 4)
├── attr_donor: (N, 10001, 4)
├── pred_neither/acceptor/donor: (N,)
└── attrs: model, input, n_sequences, n_shuffles, shuffle_method, seed
```

indices match sequences.h5 row order.
