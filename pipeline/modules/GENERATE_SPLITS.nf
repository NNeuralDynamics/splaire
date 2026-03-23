nextflow.enable.dsl=2

process GENERATE_SPLITS {
    tag "generate_splits"
    label 'process_single'
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path samplesheet
    path splits_config

    output:
    path "splits_config.yaml", emit: frozen_config
    path "train_samples.txt", emit: train_samples
    path "valid_samples.txt", emit: valid_samples
    path "test_samples.txt", emit: test_samples
    path "train_chroms.txt", emit: train_chroms
    path "valid_chroms.txt", emit: valid_chroms
    path "test_chroms.txt", emit: test_chroms
    path "splits_summary.txt", emit: summary
    path "dataset_options.txt", emit: dataset_options
    path "parallel_options.txt", emit: parallel_options
    path "validation_options.txt", emit: validation_options

    script:
    """
    python ${projectDir}/src/generate_splits.py \
        --config ${splits_config} \
        --output-dir . \
        --project-dir ${projectDir}
    """
}
