nextflow.enable.dsl=2

process STAR_GENOME_INDEX {
    tag "star_index"
    label 'process_star'
    scratch false
    publishDir "${params.star_index_dir ?: projectDir}", mode: 'copy'

    input:
    path fasta

    output:
    path "STAR_index/"

    script:
    """
    mkdir -p STAR_index
    STAR --runMode genomeGenerate \
         --genomeDir STAR_index \
         --genomeFastaFiles ${fasta} \
         --runThreadN ${params.threads}
    """
}
