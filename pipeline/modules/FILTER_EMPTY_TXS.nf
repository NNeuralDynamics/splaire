nextflow.enable.dsl=2

// filter transcripts where either SSU column is all-777 (no valid data)
// run BEFORE fill_gencode to keep only rows with real splice data

process FILTER_EMPTY_TXS {
    tag "filter_empty_txs"
    label 'process_medium'
    publishDir "${params.output_dir}/combined", mode: 'copy'

    input:
    path input_tsv

    output:
    path "${input_tsv.simpleName}_filtered.tsv"

    script:
    """
    python ${projectDir}/src/filter_empty_txs.py \
        --input ${input_tsv} \
        --output ${input_tsv.simpleName}_filtered.tsv
    """
}
