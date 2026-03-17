nextflow.enable.dsl=2

process FILL_GENCODE {
    tag "fill_gencode"
    label 'process_high'
    publishDir "${params.output_dir}/combined", mode: 'copy'

    input:
    path sites_tsv

    output:
    path "${sites_tsv.simpleName}_filled.tsv"

    script:
    """
    python ${projectDir}/src/fill_gencode_sites.py \
        --input ${sites_tsv} \
        --gtf ${params.genome_gtf} \
        --paralogs ${params.paralogs_file} \
        --output ${sites_tsv.simpleName}_filled.tsv
    """
}
