nextflow.enable.dsl=2

process MAKE_MATRIX {
    tag "make_matrix"
    label 'process_high'
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path annotated_files

    output:
    path "${params.matrix_output}", emit: raw_matrix
    path "processed_${params.matrix_output}", emit: processed_matrix

    script:
    """
    python ${projectDir}/src/build_matrix.py \
        -d ${params.output_dir} \
        -s complete_annotated.tsv \
        -o ${params.matrix_output} \
        -t ${params.matrix_denom_thresh}
    """
}
