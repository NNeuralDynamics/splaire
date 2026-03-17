nextflow.enable.dsl=2

process MERGE_VARIANTS {
    tag "merge_variants"
    label 'process_medium'
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    tuple val(suffix), val(directory), path(paralogs_file)

    output:
    path "combined_${suffix}.tsv"

    script:
    """
    python ${projectDir}/src/merge_variants.py \
        -s ${suffix} \
        -d ${directory} \
        -o combined_${suffix}.tsv \
        --paralogs ${paralogs_file}
    """
}
