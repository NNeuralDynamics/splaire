#!/bin/bash
# submit data generation jobs to slurm
# usage: bash run.sh [txrevise|leafcutter|haec|ambig|txrevise_pip50|leafcutter_pip50|haec_pip50|pip50|all]
set -e

mode=${1:-all}

# config — set these before running
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$(cd "$(dirname "$0")" && pwd)/src"
data_dir="/scratch/runyan.m/sqtl_bench"
tpm_haec="/projects/talisman/shared-data/HAEC-185/bams/salmon.merged.gene_tpm.tsv"
haec_sumstats="/projects/talisman/shared-data/HAEC-185/sQTL/sQTL_redo"
haec_finemap="/scratch/runyan.m/sqtl_bench/haec/raw/HAEC185_sQTL_credible_sets_120825.tsv"
haec_intron_dir="/projects/talisman/shared-data/HAEC-185/sQTL"
env="splaire_env"


fasta="$repo_root/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa"
gtf_haec="$repo_root/pipeline/reference/GRCh38/gencode.v45.primary_assembly.annotation.gtf"

if [ -z "$data_dir" ]; then
    echo "error: set data_dir in run.sh before running"
    exit 1
fi
logs="$data_dir/logs"
mkdir -p "$logs"


gtf_gtex="$data_dir/reference/gencode.v39.basic.annotation.gtf.gz"
tpm_gtex="$data_dir/reference/GTEx_v8_median_tpm.gct.gz"


submit_txrevise() {
    sbatch \
        --job-name=make_txrevise \
        --output="$logs/make_txrevise_%j.out" \
        --error="$logs/make_txrevise_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=120G \
        --cpus-per-task=8 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_txrevise.py \
            --cs-dir $data_dir/txrevise/raw \
            --ge-dir $data_dir/ge/sumstats \
            --gtf $gtf_gtex \
            --tpm $tpm_gtex \
            --out-dir $data_dir/txrevise"
}

submit_leafcutter() {
    sbatch \
        --job-name=make_leafcutter \
        --output="$logs/make_leafcutter_%j.out" \
        --error="$logs/make_leafcutter_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=120G \
        --cpus-per-task=8 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_leafcutter.py \
            --cs-dir $data_dir/leafcutter/raw \
            --pheno-dir $data_dir/leafcutter/phenotype_metadata \
            --ge-dir $data_dir/ge/sumstats \
            --gtf $gtf_gtex \
            --tpm $tpm_gtex \
            --out-dir $data_dir/leafcutter"
}

submit_haec() {
    sbatch \
        --job-name=make_haec \
        --output="$logs/make_haec_%j.out" \
        --error="$logs/make_haec_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=80G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_haec.py \
            --input-dir $haec_sumstats \
            --finemapping $haec_finemap \
            --intron-dir $haec_intron_dir \
            --gtf $gtf_haec \
            --tpm $tpm_haec \
            --out-dir $data_dir/haec"
}

submit_txrevise_pip50() {
    sbatch \
        --job-name=make_txrevise_pip50 \
        --output="$logs/make_txrevise_pip50_%j.out" \
        --error="$logs/make_txrevise_pip50_%j.err" \
        --partition=short \
        --time=12:00:00 \
        --mem=20G \
        --cpus-per-task=8 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_txrevise.py \
            --cs-dir $data_dir/txrevise/raw \
            --ge-dir $data_dir/ge/sumstats \
            --gtf $gtf_gtex \
            --tpm $tpm_gtex \
            --pos-pip 0.5 \
            --out-dir $data_dir/txrevise_pip50"
}

submit_leafcutter_pip50() {
    sbatch \
        --job-name=make_leafcutter_pip50 \
        --output="$logs/make_leafcutter_pip50_%j.out" \
        --error="$logs/make_leafcutter_pip50_%j.err" \
        --partition=short \
        --time=12:00:00 \
        --mem=20G \
        --cpus-per-task=8 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_leafcutter.py \
            --cs-dir $data_dir/leafcutter/raw \
            --pheno-dir $data_dir/leafcutter/phenotype_metadata \
            --ge-dir $data_dir/ge/sumstats \
            --gtf $gtf_gtex \
            --tpm $tpm_gtex \
            --pos-pip 0.5 \
            --out-dir $data_dir/leafcutter_pip50"
}

submit_haec_pip50() {
    sbatch \
        --job-name=make_haec_pip50 \
        --output="$logs/make_haec_pip50_%j.out" \
        --error="$logs/make_haec_pip50_%j.err" \
        --partition=short \
        --time=4:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_haec.py \
            --input-dir $haec_sumstats \
            --finemapping $haec_finemap \
            --intron-dir $haec_intron_dir \
            --gtf $gtf_haec \
            --tpm $tpm_haec \
            --pos-pip 0.5 \
            --out-dir $data_dir/haec_pip50"
}

submit_txrevise_hungarian() {
    sbatch \
        --job-name=make_txrevise_hungarian \
        --output="$logs/make_txrevise_hungarian_%j.out" \
        --error="$logs/make_txrevise_hungarian_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=120G \
        --cpus-per-task=8 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_txrevise.py \
            --cs-dir $data_dir/txrevise/raw \
            --ge-dir $data_dir/ge/sumstats \
            --gtf $data_dir/reference/gencode.v39.annotation.gtf.gz \
            --tpm $tpm_gtex \
            --match-scheme hungarian \
            --out-dir $data_dir/txrevise_hungarian"
}

submit_leafcutter_hungarian() {
    sbatch \
        --job-name=make_leafcutter_hungarian \
        --output="$logs/make_leafcutter_hungarian_%j.out" \
        --error="$logs/make_leafcutter_hungarian_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=120G \
        --cpus-per-task=8 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_leafcutter.py \
            --cs-dir $data_dir/leafcutter/raw \
            --pheno-dir $data_dir/leafcutter/phenotype_metadata \
            --ge-dir $data_dir/ge/sumstats \
            --gtf $gtf_gtex \
            --tpm $tpm_gtex \
            --pos-pip 0.5 \
            --match-scheme hungarian \
            --out-dir $data_dir/leafcutter_hungarian"
}

submit_haec_hungarian() {
    sbatch \
        --job-name=make_haec_hungarian \
        --output="$logs/make_haec_hungarian_%j.out" \
        --error="$logs/make_haec_hungarian_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=80G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_haec.py \
            --input-dir $haec_sumstats \
            --finemapping $haec_finemap \
            --intron-dir $haec_intron_dir \
            --gtf $gtf_haec \
            --tpm $tpm_haec \
            --match-scheme hungarian \
            --out-dir $data_dir/haec_hungarian"
}

submit_ambig() {
    sbatch \
        --job-name=make_ambig \
        --output="$logs/make_ambig_%j.out" \
        --error="$logs/make_ambig_%j.err" \
        --partition=short \
        --time=24:00:00 \
        --mem=80G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_leafcutter_ambig.py \
            --cs-dir $data_dir/leafcutter/raw \
            --gtf $gtf_gtex \
            --out-dir $data_dir/leafcutter_ambig"
}


case "$mode" in
    txrevise)      submit_txrevise ;;
    leafcutter)    submit_leafcutter ;;
    haec)          submit_haec ;;
    ambig)         submit_ambig ;;
    txrevise_pip50)   submit_txrevise_pip50 ;;
    leafcutter_pip50) submit_leafcutter_pip50 ;;
    haec_pip50)       submit_haec_pip50 ;;
    txrevise_hungarian)   submit_txrevise_hungarian ;;
    leafcutter_hungarian) submit_leafcutter_hungarian ;;
    haec_hungarian)       submit_haec_hungarian ;;
    hungarian)
        echo "submitting hungarian matching jobs..."
        submit_txrevise_hungarian
        submit_leafcutter_hungarian
        submit_haec_hungarian
        ;;
    pip50)
        echo "submitting pip50 matching jobs..."
        jid_tx=$(submit_txrevise_pip50 | grep -o '[0-9]*')
        jid_lc=$(submit_leafcutter_pip50 | grep -o '[0-9]*')
        jid_hc=$(submit_haec_pip50 | grep -o '[0-9]*')
        echo "matching jobs: tx=$jid_tx lc=$jid_lc hc=$jid_hc"
        echo ""
        echo "run scoring with:"
        echo "  GPU_PARTITION=gpu-short GPU_TIME=2:00:00 bash score.sh haec_pip50 --after $jid_hc"
        echo "  bash score.sh txrevise_pip50 --after $jid_tx"
        echo "  bash score.sh leafcutter_pip50 --after $jid_lc"
        ;;
    all)
        echo "submitting all jobs..."
        submit_txrevise
        submit_leafcutter
        submit_haec
        submit_ambig
        echo ""
        ;;
    *)
        echo "usage: bash run.sh [txrevise|leafcutter|haec|ambig|txrevise_pip50|leafcutter_pip50|haec_pip50|pip50|all]"
        exit 1
        ;;
esac
