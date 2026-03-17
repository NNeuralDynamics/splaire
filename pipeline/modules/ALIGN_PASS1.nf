// per-sample job: [bam or fastq] -> fastp -> star pass1
// supports both BAM input (converts to fastq) and direct FASTQ input

nextflow.enable.dsl=2

process ALIGN_PASS1 {
    tag "$meta.id"
    label 'process_star'
    // only publish fastp metrics, not FASTQs (6GB+ per sample)
    publishDir "${params.output_dir}/${meta.id}/fastp", mode: 'copy', pattern: "fastp/*.{html,json}"

    input:
    tuple val(meta), path(input_files), path(vcf), path(genome_dir)

    output:
    tuple val(meta), path("star_1st_pass/${meta.id}_SJ.out.tab"), emit: sj
    tuple val(meta), path("fastp/${meta.id}_1.trimmed.fastq.gz"), path("fastp/${meta.id}_2.trimmed.fastq.gz"), emit: reads
    tuple val(meta), path(vcf), emit: vcf
    path("fastp/*.{html,json}"), emit: metrics

    script:
    def is_bam = meta.input_type == 'bam'
    """
    mkdir -p fastq fastp star_1st_pass tmpdir

    if [ "${is_bam}" = "true" ]; then
        # bam to fastq (name-sorted for proper pair extraction)
        samtools sort -n -@ ${task.cpus} -T tmpdir/sort -o sorted.bam ${input_files[0]}
        bedtools bamtofastq -i sorted.bam -fq fastq/${meta.id}_1.fastq -fq2 fastq/${meta.id}_2.fastq
        rm -rf sorted.bam tmpdir
        R1=fastq/${meta.id}_1.fastq
        R2=fastq/${meta.id}_2.fastq
    else
        # direct fastq input - handle gzipped or uncompressed
        R1=${input_files[0]}
        R2=${input_files[1]}
    fi

    # fastp trimming
    fastp -i \$R1 -I \$R2 \
        -o fastp/${meta.id}_1.trimmed.fastq.gz -O fastp/${meta.id}_2.trimmed.fastq.gz \
        --detect_adapter_for_pe --qualified_quality_phred 20 --length_required 30 \
        --thread ${task.cpus} \
        --html fastp/${meta.id}.fastp.html --json fastp/${meta.id}.fastp.json

    if [ "${is_bam}" = "true" ]; then
        rm -rf fastq
    fi

    # star pass 1
    STAR --runThreadN ${task.cpus} \
        --genomeDir ${genome_dir} \
        --readFilesIn fastp/${meta.id}_1.trimmed.fastq.gz fastp/${meta.id}_2.trimmed.fastq.gz \
        --readFilesCommand zcat \
        --outFileNamePrefix star_1st_pass/${meta.id}_ \
        --varVCFfile ${vcf} \
        ${params.star_align_params}
    """
}
