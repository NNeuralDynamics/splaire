nextflow.enable.dsl=2

// per-donor per-gene h5 builder for AlphaGenome (1 Mb sequences, per-gene format).
// thin variant of BUILD_H5.nf that calls build_h5_alphagenome.py instead of build_h5.py.

process BUILD_H5_AG {
    tag "${job.output_prefix ?: job.split}_${job.donor}_ag"
    label 'process_high'
    publishDir "${params.dataset_out_dir ?: params.output_dir + '/ml_data_ag'}/individual", mode: 'link', pattern: "*.h5"

    input:
    tuple val(job), val(input_tsv)

    output:
    tuple val(job), path("${job.output_prefix ?: job.split}_${job.donor}.h5")

    script:
    def build_split = job.build_split ?: job.split
    def prefix = job.output_prefix ?: job.split
    def chromArg = job.chrom_file ? file(job.chrom_file).text.trim().split('\n').join(',') : job.chroms
    def fastaPath = params.dataset_fasta ?: params.genome_fasta
    def logDir = "${params.logs_dir}/ml_data_ag"
    def rmFlag = job.remove_missing ? "--remove-missing" : ""
    def referenceFlag = job.reference ? "--reference" : ""
    def skipEmptyFlag = job.skip_empty_genes ? "--skip-empty-genes" : ""
    def encoding_mode = job.encoding_mode ?: 'basic'
    def paralog = job.paralog ?: 'all'
    """
    set -euo pipefail
    mkdir -p ${logDir}
    python ${projectDir}/src/build_h5_alphagenome.py \
        --donor ${job.donor} \
        --split ${build_split} \
        --input ${input_tsv} \
        --chroms "${chromArg}" \
        --fasta ${fastaPath} \
        --output ${prefix}_${job.donor}.h5 \
        --log-dir ${logDir} \
        --mode ${encoding_mode} \
        --paralog ${paralog} \
        ${rmFlag} \
        ${referenceFlag} \
        ${skipEmptyFlag}
    """
}
