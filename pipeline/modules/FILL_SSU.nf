nextflow.enable.dsl=2

process FILL_SSU {
    tag "$meta.id"
    label 'process_medium'
    publishDir "${params.output_dir}/${meta.id}/spliser", mode: 'move'

    input:
    tuple val(meta), path(spliser_tsv), path(filtered_bam), path(filtered_bam_index), path(master_sites)

    output:
    tuple val(meta), path("${meta.id}.complete.tsv")

    script:
    def strand_args = Utils.strandArgs(meta.strandness)
    """
    python ${projectDir}/SpliSER/SpliSER_v0_1_8_pysam.py fillSample \
        -m $master_sites \
        -i $spliser_tsv \
        -B $filtered_bam \
        -o ${meta.id}.complete.tsv \
        ${strand_args}
    """
}
