#!/usr/bin/env python3
"""populate JAX persistent compile cache for AlphaGenome.

loads each fold, runs one warmup predict_sequence call at the full AG_SEQ_LEN
input shape (which is what our scoring uses), then drops the model. the JIT
compile artifact gets persisted to $JAX_COMPILATION_CACHE_DIR so subsequent
scoring jobs skip the ~15-min compile step.

usage:
    python warm_ag_cache.py [--folds fold_0,fold_1,fold_2,fold_3] [--source huggingface|kaggle]
"""

import argparse
import gc as gc_mod
import os
import sys
import time
from pathlib import Path

script_dir = Path(__file__).parent
ag_path = script_dir.parent.parent / "other_models" / "alphagenome_research"
sys.path.insert(0, str(ag_path / "src"))

from alphagenome_research.model import dna_model
try:
    from alphagenome.models import dna_output
except ImportError:
    from alphagenome.data import dna_output

AG_SEQ_LEN = 2**20


def make_organism_settings():
    ag_data_dir = os.environ.get('AG_DATA_DIR')
    if not ag_data_dir:
        return None
    d = Path(ag_data_dir)
    return {
        dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
            fasta_path=str(d / 'GRCh38.p13.genome.fa'),
            gtf_feather_path=str(d / 'gencode.v46.annotation.gtf.gz.feather'),
            pas_feather_path=str(d / 'polyadb_human_v3_exon3_contiguous_gtfv46.feather'),
            splice_site_starts_feather_path=str(d / 'gencode.v46.splice_sites_starts.feather'),
            splice_site_ends_feather_path=str(d / 'gencode.v46.splice_sites_ends.feather'),
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="fold_0,fold_1,fold_2,fold_3")
    p.add_argument("--source", default="huggingface", choices=["huggingface", "kaggle"])
    p.add_argument("--junctions", action="store_true",
                   help="include SPLICE_JUNCTIONS in warmup (default: off — matches scoring default)")
    args = p.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    requested = [
        dna_output.OutputType.SPLICE_SITES,
        dna_output.OutputType.SPLICE_SITE_USAGE,
    ]
    if args.junctions:
        requested.append(dna_output.OutputType.SPLICE_JUNCTIONS)

    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '<not set>')
    print(f"JAX_COMPILATION_CACHE_DIR={cache_dir}", flush=True)
    settings = make_organism_settings()
    print(f"organism_settings: {'local' if settings is not None else 'AG defaults (GCS)'}", flush=True)
    print(f"folds: {folds}", flush=True)
    print(f"requested outputs: {[r.name for r in requested]}", flush=True)

    for fi, fold in enumerate(folds):
        t0 = time.time()
        print(f"[{fi+1}/{len(folds)}] loading {fold}", flush=True)
        if args.source == "huggingface":
            model = dna_model.create_from_huggingface(fold, organism_settings=settings)
        else:
            model = dna_model.create_from_kaggle(fold, organism_settings=settings)
        t_load = time.time() - t0

        t1 = time.time()
        print(f"  running warmup at length {AG_SEQ_LEN}", flush=True)
        _ = model.predict_sequence(
            sequence="A" * AG_SEQ_LEN,
            organism=dna_model.Organism.HOMO_SAPIENS,
            requested_outputs=requested,
            ontology_terms=None,
        )
        t_warm = time.time() - t1

        print(f"  {fold}: load={t_load:.1f}s warmup={t_warm:.1f}s total={(t_load+t_warm):.1f}s", flush=True)

        del model
        gc_mod.collect()

    print("done warming cache", flush=True)


if __name__ == "__main__":
    main()
