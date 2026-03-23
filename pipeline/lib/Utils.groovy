// helper functions for splaire pipeline

class Utils {

    // strandness -> spliser args: "--isStranded -s rf" or ""
    // whitelist approach: only accept rf/fr, everything else is unstranded
    static String strandArgs(strandness) {
        if (strandness == 'rf' || strandness == 'fr') {
            return "--isStranded -s ${strandness}"
        }
        return ''
    }

    // strandness -> featurecounts flag: rf=2, fr=1, else=0
    static String featureCountsStrand(strandness) {
        if (strandness == 'rf') return '2'
        if (strandness == 'fr') return '1'
        return '0'
    }

    // sample file -> unique donor ids
    // "GTEX-1234-lung" -> "GTEX-1234"
    static List extractDonors(sample_file) {
        if (sample_file.size() == 0) return []
        sample_file.text.trim().split('\n').findAll { it.trim() }.collect { s ->
            def p = s.split('-')
            p.size() >= 2 ? p[0..1].join('-') : s
        }.unique()
    }

    // chromosome file -> comma-separated string
    static String chromsToString(chrom_file) {
        return chrom_file.text.trim().split('\n').findAll { it.trim() }.join(',')
    }

    // parse dataset_options.txt from GENERATE_SPLITS
    static Map parseDatasetOpts(file) {
        def m = [:]
        file.text.trim().split('\n').each { line ->
            def p = line.split('=', 2)
            if (p.size() == 2) m[p[0]] = p[1]
        }
        return [
            variant: m.variant ?: 'single',
            paralog: m.paralog ?: 'all',
            make_gc: m.make_gc == 'True',
            remove_missing: m.remove_missing == 'True',
            asymmetric_paralog: m.asymmetric_paralog_chroms == 'True',
            reference: m.reference == 'True',
            fill_gencode_train: m.fill_gencode_train == 'True',
            fill_gencode_valid: m.fill_gencode_valid == 'True',
            fill_gencode_test: m.fill_gencode_test == 'True'
        ]
    }

    // variant type -> encoding mode for build_h5.py
    static String getEncodingMode(variant_type) {
        def modes = [single: 'basic', het: 'het', paired: 'basic', pop: 'pop']
        return modes[variant_type] ?: 'basic'
    }

    // parse validation_options.txt from GENERATE_SPLITS
    static Map parseValidationOpts(file) {
        def m = [:]
        file.text.trim().split('\n').each { line ->
            def p = line.split('=', 2)
            if (p.size() == 2) m[p[0]] = p[1]
        }
        return [
            has_validation: m.has_validation == 'True',
            frac: (m.frac ?: '0.10') as float,
            n_splits: (m.n_splits ?: '5') as int,
            seed: (m.seed ?: '42') as int,
            exclude_chroms: m.exclude_chroms ?: ''
        ]
    }
}
