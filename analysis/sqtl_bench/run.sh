#!/bin/bash
# submit data generation jobs to slurm
# usage: bash run.sh [txrevise|leafcutter|haec|ambig|cs_txrevise|cs_leafcutter|cs_haec|cs_all|all]
set -e

mode=${1:-all}

# config — set these before running
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$(cd "$(dirname "$0")" && pwd)/src"
data_dir=""                  # output/working directory for benchmark data
tpm_haec=""                  # haec salmon tpm file
haec_sumstats=""             # haec sqtl sumstats directory
haec_finemap=""              # haec finemapping credible sets file
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
            --gtf $gtf_haec \
            --tpm $tpm_haec \
            --out-dir $data_dir/haec"
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


# credible set companion extraction (requires pairs.csv from step 2)
submit_cs_txrevise() {
    sbatch \
        --job-name=cs_txrevise \
        --output="$logs/cs_txrevise_%j.out" \
        --error="$logs/cs_txrevise_%j.err" \
        --partition=short \
        --time=12:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_cs_companions.py txrevise \
            --cs-dir $data_dir/txrevise/raw \
            --pairs $data_dir/txrevise/pairs.csv \
            --gtf $gtf_gtex \
            --out-dir $data_dir/cs_txrevise"
}

submit_cs_leafcutter() {
    sbatch \
        --job-name=cs_leafcutter \
        --output="$logs/cs_leafcutter_%j.out" \
        --error="$logs/cs_leafcutter_%j.err" \
        --partition=short \
        --time=12:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_cs_companions.py leafcutter \
            --cs-dir $data_dir/leafcutter/raw \
            --pairs $data_dir/leafcutter/pairs.csv \
            --gtf $gtf_gtex \
            --out-dir $data_dir/cs_leafcutter"
}

submit_cs_haec() {
    sbatch \
        --job-name=cs_haec \
        --output="$logs/cs_haec_%j.out" \
        --error="$logs/cs_haec_%j.err" \
        --partition=short \
        --time=4:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --wrap="source ~/.bashrc && conda activate $env && python $src/make_cs_companions.py haec \
            --finemapping $haec_finemap \
            --pairs $data_dir/haec/pairs.csv \
            --gtf $gtf_haec \
            --out-dir $data_dir/cs_haec"
}


case "$mode" in
    txrevise)      submit_txrevise ;;
    leafcutter)    submit_leafcutter ;;
    haec)          submit_haec ;;
    ambig)         submit_ambig ;;
    cs_txrevise)   submit_cs_txrevise ;;
    cs_leafcutter) submit_cs_leafcutter ;;
    cs_haec)       submit_cs_haec ;;
    cs_all)
        echo "submitting cs companion jobs..."
        submit_cs_txrevise
        submit_cs_leafcutter
        submit_cs_haec
        echo ""
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
        echo "usage: bash run.sh [txrevise|leafcutter|haec|ambig|cs_txrevise|cs_leafcutter|cs_haec|cs_all|all]"
        exit 1
        ;;
esac
