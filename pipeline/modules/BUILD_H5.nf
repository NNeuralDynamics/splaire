nextflow.enable.dsl=2

// per-donor h5 dataset builder
// consolidates: filter_vars -> select_chrom_samples -> adjust_sites -> mutate_sequences -> create_dataset

process BUILD_H5 {
    tag "${job.output_prefix ?: job.split}_${job.donor}"
    label 'process_high'
    publishDir "${params.dataset_out_dir ?: params.output_dir + '/ml_data'}/individual", mode: 'link', pattern: "*.h5"

    input:
    tuple val(job), val(input_tsv)

    output:
    tuple val(job), path("${job.output_prefix ?: job.split}_${job.donor}.h5")

    script:
    // build_split = train/valid/test (for build_h5.py --split arg)
    // output_prefix = train_split1, valid_split2, etc (for filenames + grouping)
    def build_split = job.build_split ?: job.split
    def prefix = job.output_prefix ?: job.split
    def chromArg = job.chrom_file ? file(job.chrom_file).text.trim().split('\n').join(',') : job.chroms
    def fastaPath = params.dataset_fasta ?: params.genome_fasta
    def workDir = "."
    def logDir = "${params.logs_dir}/ml_data"
    def makeGcFlag = job.make_gc ? "--make-gc" : ""
    def rmFlag = job.remove_missing ? "--remove-missing" : ""
    def asymmetricFlag = job.asymmetric_paralog ? "--asymmetric-paralog" : ""
    def referenceFlag = job.reference ? "--reference" : ""
    def encoding_mode = job.encoding_mode ?: 'basic'
    def paralog = job.paralog ?: 'all'
    """
    set -euo pipefail
    mkdir -p ${workDir} ${logDir}
    python ${projectDir}/src/build_h5.py \
        --donor ${job.donor} \
        --split ${build_split} \
        --input ${input_tsv} \
        --chroms "${chromArg}" \
        --fasta ${fastaPath} \
        --output ${prefix}_${job.donor}.h5 \
        --work-dir ${workDir} \
        --log-dir ${logDir} \
        --mode ${encoding_mode} \
        --paralog ${paralog} \
        ${makeGcFlag} \
        ${rmFlag} \
        ${asymmetricFlag} \
        ${referenceFlag}
    """
}
