# other_models

## SpliceAI
- dir: `SpliceAI/`
- paper: Jaganathan et al. 2019, *Cell* — https://doi.org/10.1016/j.cell.2018.12.015
- clone: `git clone git@github.com:Illumina/SpliceAI.git`
- env: `envs/sa_env.yml`

## Pangolin
- dir: `Pangolin/`
- paper: Zeng & Li 2022, *Genome Biology* — https://doi.org/10.1186/s13059-022-02664-4
- clone: `git clone https://github.com/tkzeng/Pangolin.git`
- env: `envs/pang_env.yml`

## Pangolin (training)
- dir: `Pangolin_train/`
- paper: Zeng & Li 2022, *Genome Biology* — https://doi.org/10.1186/s13059-022-02664-4
- clone: `git clone git@github.com:tkzeng/Pangolin_train.git`
- env: `envs/pang_env.yml`

## SpliceTransformer
- dir: `SpliceTransformer/`
- paper: You et al. 2024, *Nat Commun* — https://doi.org/10.1038/s41467-024-46824-5
- clone: `git clone https://github.com/ShenLab-Genomics/SpliceTransformer.git`
- env: `envs/spt_env.yml`

## DeltaSplice
- dir: `DeltaSplice/`
- paper: Xu et al. 2024, *Nat Commun* — https://doi.org/10.1038/s41467-024-53088-6
- clone: `git clone git@github.com:chaolinzhanglab/DeltaSplice.git`
- env: `envs/splaire_env.yml`

## RiNALMo
- dir: `RiNALMo/`
- paper: Penić et al. 2025, *Nat Commun* — https://doi.org/10.1038/s41467-025-60872-5
- clone: `git clone git@github.com:lbcb-sci/RiNALMo.git`
- env: `envs/rinalmo_env.yml`

## Nucleotide Transformer
- dir: `nucleotide-transformer/`
- paper: Dalla-Torre et al. 2025, *Nat Methods* — https://doi.org/10.1038/s41592-024-02523-z
- clone: `git clone git@github.com:instadeepai/nucleotide-transformer.git`
- env: `envs/rinalmo_env.yml`

## AlphaGenome
- dir: `alphagenome_research/`
- paper: Avsec et al. 2025, DeepMind technical report — https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/
- clone: `git clone https://github.com/google-deepmind/alphagenome_research.git`
- env: `envs/alphagenome_env.yml`

**setup (one-time):**

1. create the conda env and pip-install the research repo editable inside it (their scoring code lives there, not on pypi):

```bash
conda env create -f envs/alphagenome_env.yml
conda activate alphagenome_env
git clone https://github.com/google-deepmind/alphagenome_research.git analysis/other_models/alphagenome_research
pip install -e analysis/other_models/alphagenome_research
```

2. huggingface access — sign in at huggingface.co and request approved access to all 5 alphagenome model cards:
   - `google/alphagenome-all-folds`
   - `google/alphagenome-fold-0`
   - `google/alphagenome-fold-1`
   - `google/alphagenome-fold-2`
   - `google/alphagenome-fold-3`

3. create a personal access token at https://huggingface.co/settings/tokens and save it (chmod 600):

```bash
mkdir -p ~/.cache/huggingface
echo "hf_..." > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

4. request an `ALPHAGENOME_API_KEY` from DeepMind (used by the alphagenome python client for live inference).

5. download model weights and aux data (one-time, ~3 GB) via the sbatches in `analysis/test/`:

```bash
sbatch analysis/test/download_alphagenome.sbatch         # weights -> HF_HOME
sbatch analysis/test/download_alphagenome_data.sbatch    # fasta + gtfs + feathers -> AG_DATA_DIR
sbatch analysis/test/warm_ag_cache.sbatch                # warm the JAX compile cache
```

**env vars:** fill in your local paths in `envs/slurm_config`, then `source envs/slurm_config` once per shell (or add to your `~/.bashrc`). every AG sbatch reads these via `${VAR:?...}` guards that fail fast if any is unset.

to stop git from tracking your local edits: `git update-index --skip-worktree envs/slurm_config`

scoring instructions are in `analysis/test/readme.md` (`alphagenome` section) and `analysis/sqtl_bench/` (sQTL-specific sbatches).

## MERLIN
- dir: `MERLIN/`
- paper: Gupta et al. (in preparation), Small Models for Splicing manuscript
- clone: `git clone git@github.com:NNeuralDynamics/splaire.git`
- env: `envs/splaire_env.yml`

## RBPNet
- dir: `RBPNet/`
- paper: Her et al. 2026, *Cell Systems* — https://doi.org/10.1016/j.cels.2026.101588
- clone: `git clone git@github.com:algaebrown/RBPNet.git`
- data: `curl -L -o RBPNet/zenodo_18080544/zenodo_18080544.zip "https://zenodo.org/api/records/18080544/files/YeoLab/encode_analysis-Publish.zip/content" && unzip RBPNet/zenodo_18080544/zenodo_18080544.zip -d RBPNet/zenodo_18080544/`
- env: `RBPNet/zenodo_18080544/YeoLab-encode_analysis-0e58789/env.yaml`

## other_papers
- dir: `other_papers/`
- reference PDFs, no code
