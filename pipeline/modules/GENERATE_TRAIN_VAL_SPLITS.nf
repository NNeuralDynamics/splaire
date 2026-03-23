nextflow.enable.dsl=2

// split master tsv into N replicate train/validation sets at transcript level

process GENERATE_TRAIN_VAL_SPLITS {
    tag "train_val_splits"
    label 'process_high'
    publishDir "${params.output_dir}/validation_splits", mode: 'copy'

    input:
    path input_matrix
    val val_frac
    val n_splits
    val seed
    val exclude_chroms

    output:
    path "split*_train.tsv", emit: train_tsvs
    path "split*_validation.tsv", emit: valid_tsvs

    script:
    """
    set -euo pipefail
    python ${projectDir}/src/generate_train_val_splits.py \
        --input ${input_matrix} \
        --output-dir . \
        --n-splits ${n_splits} \
        --val-frac ${val_frac} \
        --seed-base ${seed} \
        --exclude-chroms "${exclude_chroms}"
    """
}
