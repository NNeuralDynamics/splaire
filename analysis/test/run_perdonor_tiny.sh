#!/bin/bash
# tiny end-to-end smoke test for the per-donor AG track.
# uses subsampled splice tables (12 genes/tissue) + 2-donor configs from
# /scratch/runyan.m/{splice_tables_tiny,configs_tiny}/.
#
# builds h5 per donor per tissue via nextflow build_h5_ag_only,
# then scores AlphaGenome on each donor h5, merges per-fold, and computes metrics.
#
# usage: bash run_perdonor_tiny.sh
#        TISSUES="lung" bash run_perdonor_tiny.sh   # one tissue only
#
# QOS note: GPU partition is max 8 submit / 4 running. nextflow build spawns
# up to 2 donor-h5 sub-jobs per tissue (we use 2-donor tiny configs).
# scoring is 1 AG sbatch per donor h5. don't submit more than 2-3 tissues at once.

set -euo pipefail

REPO="${REPO:-/projects/talisman/mrunyan/paper/SpHAEC}"
OUT="${OUT:-/scratch/runyan.m/sphaec_out_tiny/perdonor}"
DATA_TINY="${DATA_TINY:-/scratch/runyan.m/splice_tables_tiny}"
CONFIG_TINY="${CONFIG_TINY:-/scratch/runyan.m/configs_tiny}"
SPLAIRE_ENV_PREFIX="${SPLAIRE_ENV_PREFIX:-/scratch/runyan.m/conda_envs/splaire_env}"
AG_ENV_PREFIX="${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

export REPO OUT SPLAIRE_ENV_PREFIX AG_ENV_PREFIX

read -ra TISSUES <<< "${TISSUES:-haec10 lung brain_cortex testis whole_blood}"

cd "$(dirname "$0")"
mkdir -p logs "$OUT"

say() { printf "  %-32s %s\n" "$1" "$2"; }

for tissue in "${TISSUES[@]}"; do
    echo ""
    echo "=========== perdonor $tissue ==========="

    # config subdir for this tissue
    if [[ "$tissue" == "haec10" ]]; then
        ss="$REPO/pipeline/configs/haec/haec182_samples.tsv"
        splits="$CONFIG_TINY/haec/haec10_splits.yaml"
    else
        ss="$REPO/pipeline/configs/gtex/${tissue}_samples.tsv"
        splits="$CONFIG_TINY/gtex/${tissue}_splits.yaml"
    fi

    splice_table="$DATA_TINY/${tissue}_splice_table.tsv"
    if [[ ! -f "$splice_table" ]]; then
        echo "ERROR: tiny splice_table not found: $splice_table"
        continue
    fi
    if [[ ! -f "$splits" ]]; then
        echo "ERROR: tiny splits not found: $splits"
        continue
    fi

    # build per-donor AG h5 via nextflow (CPU, 2-donor tiny config)
    bjid=$(sbatch --parsable \
        --export=ALL,SPLICE_TABLE="$splice_table",SAMPLESHEET="$ss",SPLITS="$splits",DATA="$DATA_TINY" \
        "$REPO/pipeline/build_perdonor_ag.sbatch" "$tissue")
    say "build_perdonor_${tissue}" "$bjid"

    # score AG on each per-donor h5 — wait until build finishes, then loop
    # we wrap the per-donor scoring in a "driver" sbatch that submits 1 score_alphagenome
    # per donor h5 after the build is done.
    driver_script="$OUT/${tissue}_score_driver.sh"
    cat > "$driver_script" << EOF
#!/bin/bash
set -eo pipefail
ind_dir="$OUT/${tissue}/ml_data_var_ag/individual"
if [[ ! -d "\$ind_dir" ]]; then
    echo "ERROR: \$ind_dir missing — build failed?"
    exit 1
fi
for h5 in "\$ind_dir"/*.h5; do
    name=\$(basename "\$h5" .h5)
    out_dir="$OUT/${tissue}/ml_out_var_ag/predictions/\$name"
    mkdir -p "\$out_dir"
    sbatch --parsable --gres=gpu:a100:1 \\
        "$REPO/analysis/test/score_alphagenome.sbatch" "\$h5" "\$out_dir/\${name}_ag.parquet"
done
EOF
    chmod +x "$driver_script"

    djid=$(sbatch --parsable --dependency=afterok:$bjid \
        --partition=short --time=0:10:00 --mem=2G \
        --output="$REPO/analysis/test/logs/%x_%j.out" \
        --error="$REPO/analysis/test/logs/%x_%j.err" \
        --job-name="${tissue}_score_driver" \
        --wrap="bash $driver_script")
    say "score_driver_${tissue}" "$djid"
done

echo ""
echo "submitted. watch with: squeue -u \$USER"
