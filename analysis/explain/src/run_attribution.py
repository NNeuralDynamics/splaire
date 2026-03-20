#!/usr/bin/env python3
"""
deeplift-shap attribution for splice prediction models

usage:
    python run_attribution.py --model splaire_ref_reg --input sequences.h5 --output attr.h5
    python run_attribution.py --model spliceai --heads acceptor donor --input sequences.h5 --output attr.h5
    python run_attribution.py --model splicetransformer_cls --predictions-only --input seq.h5 --output pred.h5

supported models:
    splaire: splaire_ref_reg, splaire_ref_cls, splaire_var_reg, splaire_var_cls
    spliceai: spliceai (5 replicate ensemble)
    splicetransformer: splicetransformer_cls, splicetransformer_<tissue>
    pangolin: pangolin_<tissue>_not_splice, pangolin_<tissue>_p_splice, pangolin_<tissue>_usage

conda environments:
    splaire: splaire_env
    spliceai: sa_env
    splicetransformer: spt-test
    pangolin: pang_env
"""

# SpHAEC models now use _full.pt files which don't need TensorFlow
# TensorFlow is only imported as fallback if _full.pt not found

import os
import sys
import argparse
import time
import warnings

import numpy as np
import torch
import h5py
from tqdm import tqdm

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.ersatz import dinucleotide_shuffle, shuffle as mono_shuffle


shuffle_methods = {
    "dinucleotide": dinucleotide_shuffle,
    "mononucleotide": mono_shuffle,
}

batch_size_default = 32
n_shuffles_default = 20
seed_default = 42
warning_threshold_default = 0.001  # tangermeme default
shuffle_method_default = "dinucleotide"

models_dir = os.environ.get("SPLAIRE_MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models"))


# model registry: model_name -> (model_type, model_file, default_heads)
# model_file is None for models that load from packages

# splaire models
splaire_cls_heads = ["neither", "acceptor", "donor"]
splaire_reg_heads = ["reg"]

# spliceai heads
spliceai_heads = ["neither", "acceptor", "donor"]

# splicetransformer heads
spt_cls_heads = ["neither", "acceptor", "donor"]
spt_tissue_list = [
    "adipose", "blood", "blood_vessel", "brain", "colon",
    "heart", "kidney", "liver", "lung", "muscle",
    "nerve", "small_intestine", "skin", "spleen", "stomach"
]
spt_all_heads = spt_cls_heads + spt_tissue_list  # all 18 heads

# pangolin heads
pang_tissues = ["heart", "liver", "brain", "testis"]
pang_tasks = ["not_splice", "p_splice", "usage"]

all_models = {
    # splaire models - load .keras files directly
    "splaire_ref_reg": ("splaire", "Ref_100_v1_reg_best_sigmoid.keras", splaire_reg_heads),
    "splaire_ref_cls": ("splaire", "Ref_100_v1_cls_best.keras", splaire_cls_heads),
    "splaire_var_reg": ("splaire", "Var_100_v1_reg_best_sigmoid.keras", splaire_reg_heads),
    "splaire_var_cls": ("splaire", "Var_100_v1_cls_best.keras", splaire_cls_heads),

    # spliceai - loads from spliceai package
    "spliceai": ("spliceai", None, spliceai_heads),

    # splicetransformer - default runs all 18 heads (3 cls + 15 tissue)
    "splicetransformer": ("splicetransformer", None, spt_all_heads),
}

# add splicetransformer variants
all_models["splicetransformer_cls"] = ("splicetransformer", None, spt_cls_heads)  # just cls heads
for tissue in spt_tissue_list:
    all_models[f"splicetransformer_{tissue}"] = ("splicetransformer", None, [tissue])

# add pangolin models (one per tissue-task combo)
for tissue in pang_tissues:
    for task in pang_tasks:
        head_name = f"{tissue}_{task}"
        all_models[f"pangolin_{head_name}"] = ("pangolin", head_name, [head_name])


def load_model_heads(model_name, device, heads=None):
    """load model and wrap requested heads for attribution"""
    assert model_name in all_models, f"unknown model: {model_name}, available: {list(all_models.keys())}"
    model_type, model_file, default_heads = all_models[model_name]

    if heads is None:
        heads = default_heads

    if model_type == "splaire":
        from model_wrappers.sphaec import load_sphaec_model, get_head
        keras_path = os.path.join(models_dir, model_file)
        model = load_sphaec_model(keras_path, device)
        return {h: get_head(model, h).to(device).eval() for h in heads}

    elif model_type == "spliceai":
        from model_wrappers.spliceai import load_spliceai_ensemble, get_head
        models = load_spliceai_ensemble(device)
        return {h: get_head(models, h).to(device).eval() for h in heads}

    elif model_type == "splicetransformer":
        from model_wrappers.splicetransformer import load_splicetransformer_model, get_head
        model = load_splicetransformer_model(device)
        return {h: get_head(model, h).to(device).eval() for h in heads}

    elif model_type == "pangolin":
        from model_wrappers.pangolin import load_pangolin_model, get_head
        # pangolin loads a separate model per task
        result = {}
        for h in heads:
            model = load_pangolin_model(h, device)
            result[h] = get_head(model, h).to(device).eval()
        return result


def load_sequences(h5_path):
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]
    print(f"loaded {len(X):,} sequences, shape {X.shape}")
    return X


def compute_attributions(model_head, X_batch, n_shuffles, seed, device, warning_threshold, references):
    # warnings go to stderr naturally
    attr = deep_lift_shap(
        model_head,
        X_batch,
        target=0,
        device=device,
        references=references,
        n_shuffles=n_shuffles,
        hypothetical=True,
        verbose=False,
        print_convergence_deltas=False,
        warning_threshold=warning_threshold,
        random_state=seed,
    )
    return attr.permute(0, 2, 1).cpu().numpy()


def run_attribution(
    model_name,
    input_path,
    output_path,
    heads=None,
    batch_size=batch_size_default,
    n_shuffles=n_shuffles_default,
    seed=seed_default,
    device="cuda",
    warning_threshold=warning_threshold_default,
    shuffle_method=shuffle_method_default,
    predictions_only=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        print("=" * 60, file=sys.stderr)
        print("WARNING: cuda requested but not available, falling back to cpu", file=sys.stderr)
        print("this will be ~100x slower", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        device = "cpu"

    print(f"loading {model_name}...")
    model_heads = load_model_heads(model_name, device, heads)
    available_heads = list(model_heads.keys())
    print(f"available heads: {available_heads}")

    # verify model is on correct device
    first_head = model_heads[available_heads[0]]
    actual_device = next(first_head.parameters()).device
    print(f"model device: {actual_device}")
    if device == "cuda" and actual_device.type != "cuda":
        print("WARNING: model is on CPU despite cuda requested", file=sys.stderr)

    print(f"running heads: {available_heads}")
    if predictions_only:
        print(f"mode: PREDICTIONS ONLY (no attributions)")
        print(f"device: {device}, batch: {batch_size}")
    else:
        print(f"mode: attributions + predictions")
        print(f"device: {device}, batch: {batch_size}, shuffles: {n_shuffles}, shuffle_method: {shuffle_method}")

    references = shuffle_methods[shuffle_method] if not predictions_only else None

    X = load_sequences(input_path)
    n = len(X)

    # create output file
    with h5py.File(output_path, "w") as out:
        out.attrs["model"] = model_name
        out.attrs["input"] = os.path.abspath(input_path)
        out.attrs["n_sequences"] = n
        out.attrs["predictions_only"] = predictions_only
        out.attrs["seed"] = seed
        if not predictions_only:
            out.attrs["n_shuffles"] = n_shuffles
            out.attrs["shuffle_method"] = shuffle_method
            out.attrs["warning_threshold"] = warning_threshold

    # optimized path: splicetransformer with predictions-only
    # runs model once per batch and extracts all heads from single forward pass
    model_type = all_models[model_name][0]
    if model_type == "splicetransformer" and predictions_only:
        print(f"\nusing optimized all-heads prediction (single forward pass per batch)")
        from model_wrappers.splicetransformer import get_all_heads_model
        all_heads_model = get_all_heads_model(device, available_heads)
        t0 = time.time()

        # collect predictions for all heads
        pred_batches = {h: [] for h in available_heads}

        for i in tqdm(range(0, n, batch_size), desc="all_heads", file=sys.stdout):
            Xb = X[i : i + batch_size]
            Xt = torch.from_numpy(Xb).permute(0, 2, 1).float().to(device)

            with torch.inference_mode():
                preds_dict = all_heads_model(Xt)
            for h in available_heads:
                pred_batches[h].append(preds_dict[h].cpu().numpy())

            if device == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"computed all {len(available_heads)} heads in {elapsed / 60:.1f} min")

        # save all predictions
        with h5py.File(output_path, "a") as out:
            for h in available_heads:
                predictions = np.concatenate(pred_batches[h], axis=0)
                out.create_dataset(f"pred_{h}", data=predictions)
                print(f"saved pred_{h}")
            out.attrs["total_elapsed_sec"] = elapsed

    else:
        # standard path: run each head separately (required for attributions)
        for head_name in available_heads:
            print(f"\n--- {head_name} ---")
            model_head = model_heads[head_name]
            t0 = time.time()

            attr_batches = []
            pred_batches = []

            for i in tqdm(range(0, n, batch_size), desc=head_name, file=sys.stdout):
                Xb = X[i : i + batch_size]
                Xt = torch.from_numpy(Xb).permute(0, 2, 1).float().to(device)

                with torch.inference_mode():
                    pred = model_head(Xt).cpu().numpy()
                pred_batches.append(pred)

                if not predictions_only:
                    attr = compute_attributions(model_head, Xt, n_shuffles, seed, device, warning_threshold, references)
                    attr_batches.append(attr)

                if device == "cuda":
                    torch.cuda.empty_cache()

            predictions = np.concatenate(pred_batches, axis=0)[:, 0]

            # pangolin outputs logits for DeepLIFT stability, convert to prob for saving
            if model_name.startswith("pangolin"):
                predictions = 1 / (1 + np.exp(-predictions))

            elapsed = time.time() - t0
            print(f"computed in {elapsed / 60:.1f} min")

            with h5py.File(output_path, "a") as out:
                if not predictions_only:
                    attributions = np.concatenate(attr_batches, axis=0)
                    out.create_dataset(f"attr_{head_name}", data=attributions,
                                       compression="gzip", compression_opts=4)
                out.create_dataset(f"pred_{head_name}", data=predictions)
                out.attrs[f"{head_name}_elapsed_sec"] = elapsed

            if predictions_only:
                print(f"saved pred_{head_name}")
            else:
                print(f"saved attr_{head_name}, pred_{head_name}")

    print(f"\noutput: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="DeepLIFT-SHAP attribution for splice prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
available models:
  splaire:            splaire_ref_reg, splaire_ref_cls, splaire_var_reg, splaire_var_cls
  spliceai:           spliceai
  splicetransformer:  splicetransformer (all 18 heads), splicetransformer_cls (3 cls only),
                      splicetransformer_<tissue> (single tissue)
  pangolin:           pangolin_<tissue>_<task>

tissues (splicetransformer): adipose, blood, blood_vessel, brain, colon, heart,
                             kidney, liver, lung, muscle, nerve, small_intestine,
                             skin, spleen, stomach
tissues (pangolin): heart, liver, brain, testis
tasks (pangolin): not_splice, p_splice, usage

examples:
  python run_attribution.py --model splaire_ref_cls --input seq.h5 --output attr.h5
  python run_attribution.py --model spliceai --heads acceptor donor --input seq.h5 --output attr.h5
  python run_attribution.py --model splicetransformer_heart --input seq.h5 --output attr.h5
  python run_attribution.py --model pangolin_brain_p_splice --input seq.h5 --output attr.h5
  python run_attribution.py --model pangolin_brain_not_splice --input seq.h5 --output attr.h5

  # predictions only (no attributions) - useful for SpliceTransformer which is incompatible with DeepLIFT
  python run_attribution.py --model splicetransformer --predictions-only --input seq.h5 --output pred.h5
""")
    p.add_argument("--model", required=True, choices=list(all_models.keys()),
                   help="model to use for attribution")
    p.add_argument("--input", required=True, help="input h5 with sequences")
    p.add_argument("--output", required=True, help="output h5 for attributions")
    p.add_argument("--heads", nargs="+", help="specific heads to run (default: all for model)")
    p.add_argument("--batch-size", type=int, default=batch_size_default)
    p.add_argument("--n-shuffles", type=int, default=n_shuffles_default)
    p.add_argument("--seed", type=int, default=seed_default)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--warning-threshold", type=float, default=warning_threshold_default,
                   help="convergence delta threshold (default: 0.001)")
    p.add_argument("--shuffle-method", choices=list(shuffle_methods.keys()),
                   default=shuffle_method_default,
                   help="reference shuffle method: dinucleotide (default) or mononucleotide")
    p.add_argument("--predictions-only", action="store_true",
                   help="skip attributions, only compute predictions (useful for models incompatible with DeepLIFT)")
    args = p.parse_args()

    assert os.path.exists(args.input), f"input not found: {args.input}"

    run_attribution(
        model_name=args.model,
        input_path=args.input,
        output_path=args.output,
        heads=args.heads,
        batch_size=args.batch_size,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        device=args.device,
        warning_threshold=args.warning_threshold,
        shuffle_method=args.shuffle_method,
        predictions_only=args.predictions_only,
    )


if __name__ == "__main__":
    main()
