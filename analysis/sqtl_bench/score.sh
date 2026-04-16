#!/bin/bash
# submit gpu scoring jobs to slurm
# usage: bash score.sh [mode] [--after JOBID]
#   e.g.: bash score.sh haec --after 12345
#         bash score.sh pip50 --after 12345:12346:12347
set -e

mode=${1:-all}
DEP=""
if [ "$2" = "--after" ] && [ -n "$3" ]; then
    DEP="--dependency=afterok:$3"
    echo "scoring will wait for job(s): $3"
fi

# config — set these before running
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$(cd "$(dirname "$0")" && pwd)/src"
data_dir="/scratch/runyan.m/sqtl_bench"
spt_dir="/projects/talisman/mrunyan/other_models/SpliceTransformer"

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
declare -A envs=([sa]=sa_env [pang]=pang_env [pang_v2]=pang_env [splaire]=splaire_env [spt]=spt-test)

score_dataset() {
    local dataset="$1"
    shift
    local vcf_bases=("$@")

    local score_dir="$data_dir/$dataset/scores"
    mkdir -p "$score_dir"

    for model in sa pang pang_v2 splaire spt; do
        local env="${envs[$model]}"
        local cmds=""
        local need_run=false

        if [ "$model" = "splaire" ]; then
            # splaire: single invocation with --extra for additional vcfs
            # load models once, score all vcfs without gpu idle gaps
            local first_vcf="" first_out=""
            local extras=""
            for vcf_base in "${vcf_bases[@]}"; do
                local vcf="$data_dir/$dataset/${vcf_base}.vcf.gz"
                [ -z "$DEP" ] && [ ! -f "$vcf" ] && continue
                local out_file="$score_dir/${vcf_base}.splaire"
                local check_file="${out_file}.ref.h5"
                if [ ! -f "$check_file" ]; then
                    need_run=true
                    if [ -z "$first_vcf" ]; then
                        first_vcf="$vcf"
                        first_out="$out_file"
                    else
                        extras="$extras --extra $vcf $out_file"
                    fi
                fi
            done
            if [ "$need_run" = true ] && [ -n "$first_vcf" ]; then
                cmds="python score_splaire.py $first_vcf $fasta $first_out --models-dir $models_dir$extras"
            fi
        else
            for vcf_base in "${vcf_bases[@]}"; do
                local vcf="$data_dir/$dataset/${vcf_base}.vcf.gz"
                [ -z "$DEP" ] && [ ! -f "$vcf" ] && continue
                local out_file="$score_dir/${vcf_base}.${model}.h5"
                local check_file="$out_file"

                if [ ! -f "$check_file" ]; then
                    need_run=true
                    [ -n "$cmds" ] && cmds="$cmds && "

                    local extra=""
                    local script="score_${model}.py"
                    [ "$model" = "spt" ] && extra="--spt-dir $spt_dir"
                    if [ "$model" = "pang_v2" ]; then
                        script="score_pang.py"
                        extra="--v2"
                    fi

                    cmds="${cmds}python $script $vcf $fasta $out_file $extra"
                fi
            done
        fi

        if [ "$need_run" = true ]; then
            sbatch $DEP --job-name="${model}_${dataset}" \
                --output="$logs/${model}_${dataset}_%j.out" \
                --error="$logs/${model}_${dataset}_%j.err" \
                --partition=${GPU_PARTITION:-gpu} --gres=gpu:v100-sxm2:1 --mem=64G --time=${GPU_TIME:-8:00:00} --cpus-per-task=4 \
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


score_txrevise_pip50() {
    echo "txrevise_pip50"
    score_dataset txrevise_pip50 pos neg
}

score_leafcutter_pip50() {
    echo "leafcutter_pip50"
    score_dataset leafcutter_pip50 pos neg
}

score_haec_pip50() {
    echo "haec_pip50"
    score_dataset haec_pip50 pos neg
}

score_txrevise_hungarian() {
    echo "txrevise_hungarian"
    score_dataset txrevise_hungarian pos neg
}

score_leafcutter_hungarian() {
    echo "leafcutter_hungarian"
    score_dataset leafcutter_hungarian pos neg
}

score_haec_hungarian() {
    echo "haec_hungarian (gpu-short)"
    GPU_PARTITION=gpu-short GPU_TIME=2:00:00 score_dataset haec_hungarian pos neg
}


case "$mode" in
    txrevise)      score_txrevise ;;
    leafcutter)    score_leafcutter ;;
    haec)          score_haec ;;
    ambig)         score_ambig ;;
    txrevise_pip50)   score_txrevise_pip50 ;;
    leafcutter_pip50) score_leafcutter_pip50 ;;
    haec_pip50)       score_haec_pip50 ;;
    txrevise_hungarian)   score_txrevise_hungarian ;;
    leafcutter_hungarian) score_leafcutter_hungarian ;;
    haec_hungarian)       score_haec_hungarian ;;
    hungarian)
        score_txrevise_hungarian
        echo ""
        score_leafcutter_hungarian
        echo ""
        score_haec_hungarian
        ;;
    pip50)
        score_txrevise_pip50
        echo ""
        score_leafcutter_pip50
        echo ""
        score_haec_pip50
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
        echo "usage: bash score.sh [txrevise|leafcutter|haec|ambig|txrevise_pip50|leafcutter_pip50|haec_pip50|pip50|txrevise_hungarian|leafcutter_hungarian|haec_hungarian|hungarian|all]"
        exit 1
        ;;
esac

echo ""
echo "done"
