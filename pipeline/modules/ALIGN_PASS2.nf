// per-donor job: star pass2 -> filter_bam -> regtools -> spliser -> featurecounts
// one slurm job runs entire pass2 + quantification pipeline for a sample

nextflow.enable.dsl=2

process ALIGN_PASS2 {
    tag "$meta.id"
    label 'process_star'
    // publish filtered BAM, spliser, featurecounts, and star_2nd_pass (with metrics)
    publishDir "${params.output_dir}/${meta.id}", mode: 'copy', pattern: "{filtered_bam,spliser,featurecounts,star_2nd_pass}/*"

    input:
    tuple val(meta), path(r1), path(r2), path(vcf), path(genome_dir), path(sj_files, stageAs: '*.SJ.out.tab')

    output:
    tuple val(meta), path("filtered_bam/${meta.id}_filtered.bam"), path("filtered_bam/${meta.id}_filtered.bam.bai"), emit: bam
    tuple val(meta), path("spliser/${meta.id}.SpliSER.tsv"), emit: ssu
    path("featurecounts/*"), emit: counts
    path("star_2nd_pass/${meta.id}_metrics.txt"), emit: bam_metrics

    script:
    def strand_flag = Utils.featureCountsStrand(meta.strandness)
    def strand_args = Utils.strandArgs(meta.strandness)
    """
    mkdir -p star_2nd_pass filtered_bam regtools spliser featurecounts

    # star pass 2 with collected junctions
    STAR --runThreadN ${task.cpus} \
        --genomeDir ${genome_dir} \
        --readFilesIn ${r1} ${r2} \
        --readFilesCommand zcat \
        --outFileNamePrefix star_2nd_pass/${meta.id}_ \
        --varVCFfile ${vcf} \
        --sjdbFileChrStartEnd ${sj_files} \
        ${params.star_align_params}

    # filter step 1: unique mappers (-q 255) + proper pairs (-f 2) + primary only (-F 256)
    samtools view -h -b -q 255 -f 2 -F 256 \
        star_2nd_pass/${meta.id}_Aligned.sortedByCoord.out.bam \
        > filtered_bam/${meta.id}_quality.bam

    # filter step 2: wasp filter (keep vW=1 or no vW tag)
    samtools view -h -b -e '[vW]==1 || ![vW]' \
        filtered_bam/${meta.id}_quality.bam \
        > filtered_bam/${meta.id}_filtered.bam
    samtools index filtered_bam/${meta.id}_filtered.bam
    rm filtered_bam/${meta.id}_quality.bam

    # bam metrics: raw_total, final_total, wasp_failed
    raw_total=\$(samtools view -c star_2nd_pass/${meta.id}_Aligned.sortedByCoord.out.bam)
    final_total=\$(samtools view -c filtered_bam/${meta.id}_filtered.bam)
    wasp_failed=\$(samtools view -c -e '[vW] && [vW]!=1' star_2nd_pass/${meta.id}_Aligned.sortedByCoord.out.bam)
    echo -e "\${raw_total}\\t\${final_total}\\t\${wasp_failed}" > star_2nd_pass/${meta.id}_metrics.txt

    # regtools junction extraction
    regtools junctions extract \
        ${params.regtools_args} \
        -o regtools/${meta.id}_junctions.bed \
        filtered_bam/${meta.id}_filtered.bam

    # spliser quantification
    python ${params.spliser_path} process \
        -B filtered_bam/${meta.id}_filtered.bam \
        -b regtools/${meta.id}_junctions.bed \
        -A ${params.genome_gtf} \
        ${strand_args} \
        ${params.spliser_process_args} \
        -o spliser/${meta.id}

    # featurecounts
    featureCounts -T ${task.cpus} -p -B -C \
        -s ${strand_flag} -t exon -g gene_id \
        -a ${params.genome_gtf} \
        -o featurecounts/${meta.id}.counts.txt \
        filtered_bam/${meta.id}_filtered.bam
    """
}
