#!/bin/bash
#SBATCH --job-name=dl_ag
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# download alphagenome weights for all 4 folds (and optionally all_folds)
# into scratch. CPU only — pure huggingface snapshot_download.
#
# usage:
#   sbatch download_alphagenome.sbatch                # 4 folds
#   sbatch download_alphagenome.sbatch fold_0         # one fold
#   sbatch download_alphagenome.sbatch fold_0,fold_1,all_folds

set -eo pipefail

folds="${1:-fold_0,fold_1,fold_2,fold_3}"

export HF_HOME=/scratch/runyan.m/huggingface_cache
mkdir -p "$HF_HOME" logs

source ~/.bashrc
conda activate "${AG_ENV_PREFIX:-/scratch/runyan.m/conda_envs/alphagenome_env}"

echo "HF_HOME: $HF_HOME"
echo "folds:   $folds"
echo "started: $(date)"

python - "$folds" <<'PYEOF'
import os, sys, time, subprocess
import huggingface_hub

folds = [f.strip() for f in sys.argv[1].split(",") if f.strip()]
print(f"will download {len(folds)} fold(s): {folds}", flush=True)

for fold in folds:
    repo_id = f"google/alphagenome-{fold.replace('_', '-').lower()}"
    print(f"\n=== {fold} -> {repo_id} ===", flush=True)
    t0 = time.time()
    path = huggingface_hub.snapshot_download(repo_id=repo_id)
    elapsed = time.time() - t0
    size = subprocess.check_output(["du", "-sh", path], text=True).strip()
    print(f"  path:    {path}", flush=True)
    print(f"  size:    {size}", flush=True)
    print(f"  elapsed: {elapsed:.1f}s", flush=True)

print(f"\n=== total HF_HOME size ===", flush=True)
total = subprocess.check_output(["du", "-sh", os.environ["HF_HOME"]], text=True).strip()
print(total, flush=True)
PYEOF

echo "done: $(date)"
