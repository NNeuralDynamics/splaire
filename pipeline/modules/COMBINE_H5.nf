nextflow.enable.dsl=2

process COMBINE_H5 {
    tag "$output_name"
    label 'process_high'
    publishDir "${params.dataset_out_dir ?: params.output_dir + '/ml_data'}", mode: 'move'

    input:
    path h5_files
    val output_name

    output:
    path "combined_${output_name}.h5"

    script:
    """
    set -euo pipefail
    python ${projectDir}/src/combine_h5.py \
        --input_dir . \
        --pattern "*.h5" \
        --output combined_${output_name}.h5
    """
}
