nextflow.enable.dsl=2

process COLLECT_SITES {
    tag 'collect_sites'
    label 'process_medium'
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path samples_file

    output:
    path 'master_sites.tsv', emit: master_sites

    script:
    """
    set -euo pipefail
    python ${params.spliser_path} collectSites \
        -S ${samples_file} \
        -o master_sites.tsv
    """
}
