nextflow.enable.dsl=2

process ADD_SITES {
    tag "add_sites"
    label 'process_high'
    publishDir "${params.output_dir}/combined", mode: 'copy'

    input:
    path matrix_file
    path merged_variants

    output:
    path "${merged_variants.simpleName}_sites.tsv"

    script:
    """
    python ${projectDir}/src/add_sites.py \
        ${matrix_file} \
        ${merged_variants} \
        ${merged_variants.simpleName}_sites.tsv
    """
}
