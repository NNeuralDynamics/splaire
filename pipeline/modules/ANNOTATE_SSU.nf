nextflow.enable.dsl=2

process ANNOTATE_SSU {
    tag "$meta.id"
    label 'process_low'
    publishDir "${params.output_dir}/${meta.id}/spliser", mode: 'link'

    input:
    tuple val(meta), path(spliser_tsv), path(master_sites)

    output:
    tuple val(meta), path("${meta.id}.complete_annotated.tsv"), path("${meta.id}.complete_annotated.tsv.log")

    script:
    """
    python ${projectDir}/src/annotate_spliser.py $spliser_tsv $master_sites
    """
}
