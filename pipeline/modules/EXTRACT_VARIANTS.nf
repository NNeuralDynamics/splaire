nextflow.enable.dsl=2

// consolidated variant extraction
// combines: bedtools intersect + phased extraction + SNV filtering

process EXTRACT_VARIANTS {
    tag "$donor"
    label 'process_medium'
    publishDir "${params.output_dir}/${donor}/variants", mode: 'copy'

    input:
    tuple val(donor), path(filtered_vcf), path(genes_bed), path(genes_tsv)

    output:
    tuple val(donor), path("${donor}_gene_variants.tsv"), path("${donor}_gene_variants_SNVs.tsv"), path("${donor}_gene_variants.log")

    script:
    """
    python ${projectDir}/src/extract_variants.py \
        -v ${filtered_vcf} \
        -b ${genes_bed} \
        -g ${genes_tsv} \
        -d ${donor} \
        -o ${donor}_gene_variants \
        --log ${donor}_gene_variants.log
    """
}
