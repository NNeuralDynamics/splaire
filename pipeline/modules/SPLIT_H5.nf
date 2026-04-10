nextflow.enable.dsl=2

// split a donor's full h5 into per-fold train/valid h5s
// reads gc dataset to match windows to transcripts, writes one output per fold

process SPLIT_H5 {
    tag "${donor}"
    label 'process_low'
    publishDir "${params.dataset_out_dir ?: params.output_dir + '/ml_data'}/individual", mode: 'move', pattern: "*.h5"

    input:
    tuple val(donor), path(full_h5), path(fold_tsvs)

    output:
    tuple val(donor), path("*_split*_${donor}.h5")

    script:
    def logDir = "${params.logs_dir}/ml_data"
    """
    set -euo pipefail
    mkdir -p ${logDir}
    python ${projectDir}/src/split_h5.py \
        --input ${full_h5} \
        --fold-tsvs ${fold_tsvs} \
        --output-dir . \
        --donor ${donor} \
        2>&1 | tee ${logDir}/split_${donor}.log
    """
}
