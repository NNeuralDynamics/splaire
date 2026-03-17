nextflow.enable.dsl=2

process FILTER_VCF {
    tag "$donor"
    label 'process_low'
    // no publishDir - VCF stays in work dir only (30GB+ per donor)

    input:
    tuple val(donor), path(vcf_file)

    output:
    tuple val(donor), path("${donor}_filtered.vcf")

    when:
    params.filter_vcf

    script:
    """
    # extract only sites where donor has alt allele (skip 0/0 and ./.)
    bcftools view -s "${donor}" -i 'GT="alt"' --threads ${task.cpus} -o ${donor}_filtered.vcf ${vcf_file}

    # validate vcf for star --varVCFfile compatibility
    vcf_out="${donor}_filtered.vcf"

    # check not gzipped - first two bytes of gzip are 1f 8b
    if head -c 2 "\$vcf_out" | od -An -tx1 | grep -q '1f 8b'; then
        echo "error: vcf is gzipped but star requires uncompressed vcf" >&2
        exit 1
    fi

    # check single sample - count columns after FORMAT (column 9)
    sample_count=\$(grep -m1 '^#CHROM' "\$vcf_out" | awk -F'\\t' '{print NF - 9}')
    if [ "\$sample_count" -ne 1 ]; then
        echo "error: vcf has \$sample_count samples but star requires exactly 1 sample" >&2
        exit 1
    fi

    # check genotype field exists in first variant line
    first_variant=\$(grep -v '^#' "\$vcf_out" | head -1)
    if [ -n "\$first_variant" ]; then
        format_field=\$(echo "\$first_variant" | cut -f9)
        if [ "\${format_field:0:2}" != "GT" ]; then
            echo "error: vcf format field does not start with GT genotype" >&2
            exit 1
        fi

        gt_value=\$(echo "\$first_variant" | cut -f10 | cut -d':' -f1)
        if ! echo "\$gt_value" | grep -q '[0-9][/|][0-9]'; then
            echo "error: vcf genotype '\$gt_value' is not valid format like 0/1 or 1|1" >&2
            exit 1
        fi
    fi

    echo "vcf validation passed: single-sample, uncompressed, valid genotype format"
    """
}
