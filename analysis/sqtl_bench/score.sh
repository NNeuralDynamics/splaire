#!/bin/bash
# submit gpu scoring jobs to slurm
# usage: bash score.sh [txrevise|leafcutter|haec|ambig|cs_txrevise|cs_leafcutter|cs_haec|cs_all|all]
set -e

mode=${1:-all}

# config — set these before running
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$(cd "$(dirname "$0")" && pwd)/src"
data_dir=""                  # output/working directory for benchmark data
spt_dir=""                   # path to SpliceTransformer repo

# derived from repo
fasta="$repo_root/pipeline/reference/GRCh38/GRCh38.primary_assembly.genome.fa"
models_dir="$repo_root/models"

if [ -z "$data_dir" ]; then
    echo "error: set data_dir in score.sh before running"
    exit 1
fi
logs="$data_dir/logs"
mkdir -p "$logs"

# conda environments per model
declare -A envs=([sa]=sa_env [pang]=pang_env [splaire]=splaire_env [spt]=spt-test)

score_dataset() {
    local dataset="$1"
    shift
    local vcf_bases=("$@")

    local score_dir="$data_dir/$dataset/scores"
    mkdir -p "$score_dir"

    for model in sa pang splaire spt; do
        local env="${envs[$model]}"
        local cmds=""
        local need_run=false

        for vcf_base in "${vcf_bases[@]}"; do
            local vcf="$data_dir/$dataset/${vcf_base}.vcf.gz"
            [ ! -f "$vcf" ] && continue

            if [ "$model" = "splaire" ]; then
                local out_file="$score_dir/${vcf_base}.splaire"
                local check_file="${out_file}.ref.h5"
            else
                local out_file="$score_dir/${vcf_base}.${model}.h5"
                local check_file="$out_file"
            fi

            if [ ! -f "$check_file" ]; then
                need_run=true
                [ -n "$cmds" ] && cmds="$cmds && "

                # model-specific extra args
                local extra=""
                [ "$model" = "splaire" ] && extra="--models-dir $models_dir"
                [ "$model" = "spt" ] && extra="--spt-dir $spt_dir"

                cmds="${cmds}python score_${model}.py $vcf $fasta $out_file $extra"
            fi
        done

        if [ "$need_run" = true ]; then
            sbatch --job-name="${model}_${dataset}" \
                --output="$logs/${model}_${dataset}_%j.out" \
                --error="$logs/${model}_${dataset}_%j.err" \
                --partition=gpu --gres=gpu:v100-sxm2:1 --mem=64G --time=8:00:00 --cpus-per-task=4 \
                --wrap="source ~/.bashrc && conda activate $env && cd $src && $cmds"
            echo "  submitted ${model} for ${dataset}"
        else
            echo "  skip ${model} for ${dataset} (all exist)"
        fi
    done
}

score_txrevise() {
    echo "txrevise"
    score_dataset txrevise pos neg
}

score_leafcutter() {
    echo "leafcutter"
    score_dataset leafcutter pos neg
}

score_haec() {
    echo "haec"
    score_dataset haec pos neg
}

score_ambig() {
    echo "leafcutter_ambig"
    score_dataset leafcutter_ambig ambig
}

# credible set companion scoring
score_cs_txrevise() {
    echo "cs_txrevise"
    score_dataset cs_txrevise variants
}

score_cs_leafcutter() {
    echo "cs_leafcutter"
    score_dataset cs_leafcutter variants
}

score_cs_haec() {
    echo "cs_haec"
    score_dataset cs_haec variants
}


case "$mode" in
    txrevise)      score_txrevise ;;
    leafcutter)    score_leafcutter ;;
    haec)          score_haec ;;
    ambig)         score_ambig ;;
    cs_txrevise)   score_cs_txrevise ;;
    cs_leafcutter) score_cs_leafcutter ;;
    cs_haec)       score_cs_haec ;;
    cs_all)
        score_cs_txrevise
        echo ""
        score_cs_leafcutter
        echo ""
        score_cs_haec
        ;;
    all)
        score_txrevise
        echo ""
        score_leafcutter
        echo ""
        score_haec
        echo ""
        score_ambig
        ;;
    *)
        echo "usage: bash score.sh [txrevise|leafcutter|haec|ambig|cs_txrevise|cs_leafcutter|cs_haec|cs_all|all]"
        exit 1
        ;;
esac

echo ""
echo "done"
