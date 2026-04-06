#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// alignment -> quantification -> annotation -> datasets

include { STAR_GENOME_INDEX } from './modules/STAR_GENOME_INDEX.nf'
include { FILTER_VCF }        from './modules/FILTER_VCF.nf'
include { ALIGN_PASS1 }       from './modules/ALIGN_PASS1.nf'
include { ALIGN_PASS2 }       from './modules/ALIGN_PASS2.nf'
include { COLLECT_SITES }     from './modules/COLLECT_SITES.nf'
include { FILL_SSU }          from './modules/FILL_SSU.nf'
include { ANNOTATE_SSU }      from './modules/ANNOTATE_SSU.nf'
include { EXTRACT_VARIANTS }  from './modules/EXTRACT_VARIANTS.nf'
include { MERGE_VARIANTS }    from './modules/MERGE_VARIANTS.nf'
include { MAKE_MATRIX }       from './modules/MAKE_MATRIX.nf'
include { ADD_SITES }         from './modules/ADD_SITES.nf'
include { FILTER_EMPTY_TXS }  from './modules/FILTER_EMPTY_TXS.nf'
include { FILL_GENCODE }      from './modules/FILL_GENCODE.nf'
include { GENERATE_SPLITS }           from './modules/GENERATE_SPLITS.nf'
include { GENERATE_TRAIN_VAL_SPLITS } from './modules/GENERATE_TRAIN_VAL_SPLITS.nf'
include { BUILD_H5 }                  from './modules/BUILD_H5.nf'
include { SPLIT_H5 }                  from './modules/SPLIT_H5.nf'
include { COMBINE_H5 }                from './modules/COMBINE_H5.nf'


workflow {
    if (params.make_star_index) {
        STAR_GENOME_INDEX(file(params.genome_fasta))
        return
    }

    assert params.samplesheet : "provide --samplesheet"
    assert params.vcf : "provide --vcf"

    samples = parse_samplesheet()

    // alignment
    donor_vcfs = filter_vcf_per_donor(samples)
    pass1 = run_pass1(samples, donor_vcfs)

    // quantification
    pass2 = run_pass2(pass1.junctions, pass1.trimmed_reads, pass1.donor_vcfs)

    // annotation
    annotated = annotate_samples(pass2.bam, pass2.ssu, donor_vcfs)

    // datasets
    build_datasets(annotated.sites, annotated.filled)
}



// alignment

workflow filter_vcf_per_donor {
    take: samples
    main:
    donor_vcfs = samples
        .map { meta -> tuple(meta.donor, file(params.vcf)) }
        .unique { it[0] }
        | FILTER_VCF
    emit: donor_vcfs
}

workflow run_pass1 {
    take:
    samples
    donor_vcfs

    main:
    genome = file(params.star_genome_dir)

    aligned = samples
        .map { meta -> tuple(meta.donor, meta, meta.input_files) }
        .combine(donor_vcfs, by: 0)
        .map { donor, meta, input_files, vcf -> tuple(meta, input_files, vcf, genome) }
        | ALIGN_PASS1

    emit:
    junctions = aligned.sj
    trimmed_reads = aligned.reads
    donor_vcfs = aligned.vcf
}



// quantification

workflow run_pass2 {
    take:
    junctions
    trimmed_reads
    donor_vcfs

    main:
    genome = file(params.star_genome_dir)

    // collect all splice junction files for STAR 2nd pass (staged as actual files)
    // toList() keeps them as a single list element when combined
    all_junctions = junctions.map { meta, sj -> sj }.collect().map { files -> [files] }

    // join reads with their vcf, add genome and junctions
    aligned = trimmed_reads
        .join(donor_vcfs, by: [0])
        .combine(Channel.value(genome))
        .combine(all_junctions)
        .map { meta, r1, r2, vcf, genome_dir, sj_list -> tuple(meta, r1, r2, vcf, genome_dir, sj_list) }
        | ALIGN_PASS2

    emit:
    bam = aligned.bam
    ssu = aligned.ssu
}



// annotation

workflow annotate_samples {
    take:
    alignments
    splice_sites
    donor_vcfs

    main:
    genes_bed = file(params.gene_regions_bed)
    genes_tsv = file(params.gene_regions_tsv)
    paralogs = file(params.paralogs_file)

    // build sample list for COLLECT_SITES: sample_id \t ssu.tsv \t bam
    sample_manifest = splice_sites
        .join(alignments, by: [0])
        .map { meta, tsv, bam, bai -> "${meta.id}\t${tsv}\t${bam}" }
        .collect()
        .map { lines ->
            def f = file("${params.output_dir}/samples_for_collect.tsv")
            f.parent.toFile().mkdirs()
            f.text = lines.join('\n') + '\n'
            f
        }
    master_sites = COLLECT_SITES(sample_manifest)

    // fill missing sites per sample
    filled_ssu = splice_sites
        .join(alignments, by: [0])
        .combine(master_sites)
        .map { meta, tsv, bam, bai, sites -> tuple(meta, tsv, bam, bai, sites) }
        | FILL_SSU

    // annotate splice site types
    annotated_ssu = filled_ssu
        .combine(master_sites)
        .map { meta, tsv, sites -> tuple(meta, tsv, sites) }
        | ANNOTATE_SSU

    // extract variants per donor
    donor_variants = donor_vcfs
        .map { donor, vcf -> tuple(donor, vcf, genes_bed, genes_tsv) }
        | EXTRACT_VARIANTS

    // merge variants across all donors (wait for annotation to finish)
    merged_variants = annotated_ssu.collect()
        .map { params.output_dir }
        .flatMap { dir ->
            [tuple('gene_variants', dir, paralogs), tuple('gene_variants_SNVs', dir, paralogs)]
        }
        | MERGE_VARIANTS

    ssu_matrix = annotated_ssu.map { meta, tsv, log -> tsv }.collect() | MAKE_MATRIX

    // combine variants with ssu
    variant_file = merged_variants.collect().map { file("${params.output_dir}/combined_gene_variants_SNVs.tsv") }
    sites_with_variants = ADD_SITES(ssu_matrix.processed_matrix, variant_file)

    // filter transcripts with all-777 SSUs before adding GENCODE placeholders
    sites_filtered = FILTER_EMPTY_TXS(sites_with_variants).tsv
    sites_with_gencode = FILL_GENCODE(sites_filtered)

    emit:
    sites = sites_filtered  // use filtered version (not raw ADD_SITES output)
    filled = sites_with_gencode
}



// datasets

workflow build_datasets {
    take:
    sites_unfilled
    sites_filled

    main:
    split_config = GENERATE_SPLITS(file(params.samplesheet), file(params.splits_config))

    // parse configs
    val_opts_ch = split_config.validation_options.map { f -> Utils.parseValidationOpts(f) }

    // standard jobs: test + train + valid (from sample-level config)
    standard_jobs = split_config.train_samples
        .combine(split_config.valid_samples)
        .combine(split_config.test_samples)
        .combine(split_config.train_chroms)
        .combine(split_config.valid_chroms)
        .combine(split_config.test_chroms)
        .combine(split_config.dataset_options)
        .combine(split_config.validation_options)
        .combine(sites_unfilled)
        .combine(sites_filled)
        .flatMap { train_samp, valid_samp, test_samp, train_chr, valid_chr, test_chr, opts_file, val_opts_file, unfilled, filled ->
            def opts = Utils.parseDatasetOpts(opts_file)
            def val_opts = Utils.parseValidationOpts(val_opts_file)
            def encoding = Utils.getEncodingMode(opts.variant)
            def jobs = []

            // test split (always from original matrix)
            if (test_samp.size() > 0) {
                def chrom_str = Utils.chromsToString(test_chr)
                def input_tsv = opts.fill_gencode_test ? filled : unfilled
                Utils.extractDonors(test_samp).collect { donor ->
                    jobs << tuple([
                        donor: donor, split: 'test', build_split: 'test',
                        output_prefix: 'test', chroms: chrom_str,
                        encoding_mode: encoding, paralog: opts.paralog,
                        make_gc: opts.make_gc, remove_missing: opts.remove_missing,
                        asymmetric_paralog: opts.asymmetric_paralog,
                        reference: opts.reference
                    ], input_tsv)
                }
            }

            // train/valid from sample-level config (skip if validation splitting is enabled)
            if (!val_opts.has_validation) {
                [['valid', valid_samp, valid_chr, opts.fill_gencode_valid],
                 ['train', train_samp, train_chr, opts.fill_gencode_train]].each { split, samples, chroms, use_gencode ->
                    if (samples.size() > 0) {
                        def chrom_str = Utils.chromsToString(chroms)
                        def input_tsv = use_gencode ? filled : unfilled
                        Utils.extractDonors(samples).collect { donor ->
                            jobs << tuple([
                                donor: donor, split: split, build_split: split,
                                output_prefix: split, chroms: chrom_str,
                                encoding_mode: encoding, paralog: opts.paralog,
                                make_gc: opts.make_gc, remove_missing: opts.remove_missing,
                                asymmetric_paralog: opts.asymmetric_paralog,
                                reference: opts.reference
                            ], input_tsv)
                        }
                    }
                }
            }
            jobs
        }

    // validation-split jobs: build full h5 per donor, then split by fold
    // only runs when validation config is present
    val_split_jobs = split_config.validation_options
        .combine(split_config.train_samples)
        .combine(split_config.dataset_options)
        .combine(sites_filled)
        .flatMap { val_opts_file, train_samp, opts_file, filled ->
            def val_opts = Utils.parseValidationOpts(val_opts_file)
            if (!val_opts.has_validation) return []
            [tuple(val_opts, train_samp, opts_file, filled)]
        }

    // build one full h5 per train donor (all transcripts, all train chroms)
    // always uses filled matrix since fold tsvs come from the same filled matrix
    full_build_jobs = val_split_jobs
        .combine(split_config.train_chroms)
        .flatMap { val_opts, train_samp, opts_file, filled, train_chr ->
            def opts = Utils.parseDatasetOpts(opts_file)
            def encoding = Utils.getEncodingMode(opts.variant)
            def chrom_str = Utils.chromsToString(train_chr)

            Utils.extractDonors(train_samp).collect { donor ->
                tuple([
                    donor: donor, split: 'full', build_split: 'train',
                    output_prefix: 'full', chroms: chrom_str,
                    encoding_mode: encoding, paralog: 'all',
                    make_gc: true,
                    remove_missing: opts.remove_missing,
                    asymmetric_paralog: opts.asymmetric_paralog,
                    reference: opts.reference
                ], filled)
            }
        }

    // run transcript-level splitting (produces N train + N valid tsvs)
    val_input = val_split_jobs.map { val_opts, train_samp, opts_file, filled ->
        tuple(filled, val_opts.frac, val_opts.n_splits, val_opts.seed, val_opts.exclude_chroms)
    }
    .multiMap { matrix, frac, n, seed, excl ->
        matrix: matrix
        frac: frac
        n_splits: n
        seed: seed
        exclude: excl
    }
    val_tsvs = GENERATE_TRAIN_VAL_SPLITS(
        val_input.matrix, val_input.frac, val_input.n_splits, val_input.seed, val_input.exclude
    )

    // collect all fold tsvs into a single list for split_h5
    all_fold_tsvs = val_tsvs.train_tsvs.flatten()
        .mix(val_tsvs.valid_tsvs.flatten())
        .collect()

    // build full h5s, then split by fold
    all_jobs = standard_jobs.mix(full_build_jobs)
    donor_h5s = BUILD_H5(all_jobs)

    // separate full builds from standard (test) builds
    full_h5s = donor_h5s.filter { job, h5 -> job.split == 'full' }
    standard_h5s = donor_h5s.filter { job, h5 -> job.split != 'full' }

    // pair each donor's full h5 with the fold tsvs and split
    split_input = full_h5s
        .map { job, h5 -> tuple(job.donor, h5) }
        .combine(all_fold_tsvs)
    split_h5s = SPLIT_H5(split_input)

    // flatten split outputs and group by fold name for combine
    val_by_split = split_h5s
        .flatMap { donor, h5s ->
            h5s.collect { h5 ->
                // strip _DONOR.h5 suffix to get fold name like "train_split1"
                def prefix = h5.name.replace("_${donor}.h5", '')
                tuple(prefix, h5)
            }
        }
        .groupTuple()

    // group standard builds (test) for combine
    standard_by_split = standard_h5s
        .map { job, h5 -> tuple(job.output_prefix ?: job.split, h5) }
        .groupTuple()

    // merge and combine
    by_split = standard_by_split.mix(val_by_split)
        .multiMap { split, files -> h5s: files; name: split }

    COMBINE_H5(by_split.h5s, by_split.name)
}



// standalone annotation from existing MAKE_MATRIX and MERGE_VARIANTS outputs
// usage: nextflow run main.nf -entry annotate_only --input_matrix /path/to/splicing_matrix_processed.tsv --input_variants /path/to/combined_gene_variants_SNVs.tsv

workflow annotate_only {
    assert params.input_matrix : "provide --input_matrix (path to splicing_matrix_processed.tsv from MAKE_MATRIX)"
    assert params.input_variants : "provide --input_variants (path to combined_gene_variants_SNVs.tsv from MERGE_VARIANTS)"
    assert params.samplesheet : "provide --samplesheet"

    ssu_matrix = Channel.fromPath(params.input_matrix)
    variant_file = Channel.fromPath(params.input_variants)

    // ADD_SITES -> FILTER_EMPTY_TXS -> FILL_GENCODE
    sites_with_variants = ADD_SITES(ssu_matrix, variant_file)
    sites_filtered = FILTER_EMPTY_TXS(sites_with_variants).tsv
    sites_with_gencode = FILL_GENCODE(sites_filtered)

    // build datasets
    build_datasets(sites_filtered, sites_with_gencode)
}


// standalone dataset building from existing processed files
// usage: nextflow run main.nf -entry build_h5_only --input_matrix /path/to/master_sites_variants.tsv

workflow build_h5_only {
    assert params.input_matrix : "provide --input_matrix (path to master_sites_variants.tsv)"
    assert params.samplesheet : "provide --samplesheet"

    sites_unfilled = Channel.fromPath(params.input_matrix)
    sites_filled = Channel.fromPath(params.input_matrix)  // same file when no gencode filling

    build_datasets(sites_unfilled, sites_filled)
}



// samplesheet parsing

workflow parse_samplesheet {
    main:
    ch = Channel.fromPath(params.samplesheet)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            assert row.sample_id?.trim() : "sample_id missing"
            assert row.donor_id?.trim() : "donor_id missing"

            // support both bam and fastq input formats
            def has_bam = row.bam?.trim()
            def has_fastq = row.fastq_1?.trim() && row.fastq_2?.trim()
            assert has_bam || has_fastq : "must provide bam or fastq_1/fastq_2 for ${row.sample_id}"

            def input_type = has_bam ? 'bam' : 'fastq'
            def input_files = has_bam ? [file(row.bam)] : [file(row.fastq_1), file(row.fastq_2)]

            [
                id: row.sample_id,
                donor: row.donor_id,
                input_type: input_type,
                input_files: input_files,
                strandness: row.strandness ?: 'rf',
                read_length: (row.read_length ?: 76) as int
            ]
        }
    emit: ch
}
