import os, warnings, json, logging, sys, subprocess
_no_plot = "--no-plot" in sys.argv
_only_tau = "--only-tau" in sys.argv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from pathlib import Path

# output directories
fig_main = Path("data/figures/main")
fig_sup = Path("data/figures/sup")
results_dir = Path("data/results")
for _d in [fig_main, fig_sup, results_dir]:
    _d.mkdir(parents=True, exist_ok=True)

# logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("data/figures.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _save_fig(fig_or_plt, path_no_ext, **kwargs):
    """save figure as png + pdf"""
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    p = Path(path_no_ext)
    fig_or_plt.savefig(p.with_suffix(".png"), **kwargs)
    fig_or_plt.savefig(p.with_suffix(".pdf"), **kwargs)
    plt.close("all")

ANNOT_SIZE = 12
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 14, "figure.titlesize": 18,
    "axes.titlesize": 16, "axes.labelsize": 14, "axes.titleweight": "normal",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "legend.title_fontsize": 14, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

GRAY_DARK, GRAY_MED, GRAY_LIGHT = "#404040", "#666666", "#888888"
GRAY_FAINT, GRAY_BG, GRAY_LINE = "#cccccc", "#e0e0e0", "#5a5a5a"

model_colors = {
    "pangolin": "#009E73", "spliceai": "#D55E00", "splicetransformer": "#E69F00",
    "splaire_ref": "#56B4E9", "splaire_var": "#0072B2", "splaire_avg": "#CC79A7",
    "gencode": "#666666",
}
tissue_colors = {
    "haec": "#7570b3", "haec10": "#7570b3", "lung": "#1b9e77", "testis": "#d95f02",
    "brain_cortex": "#e7298a", "whole_blood": "#66a61e",
}
tissue_display = {
    "haec": "HAEC", "haec10": "HAEC", "lung": "Lung", "testis": "Testis",
    "brain_cortex": "Brain", "whole_blood": "Blood", "gencode": "GENCODE", "mane_select": "MANE",
}
model_display = {
    "pangolin": "Pangolin", "spliceai": "SpliceAI", "splicetransformer": "SpliceTransformer",
    "splaire_ref": "SPLAIRE", "splaire_var": "SPLAIRE (var)", "splaire_avg": "SPLAIRE (avg)",
}
r2_outputs = ["splaire_ref_ssu", "spliceai_cls", "pangolin_avg_usage", "splicetransformer_avg_tissue"]

def _get_base_model(key):
    key = key.lower().replace("-", "_").replace(" ", "_")
    if key in model_colors: return key
    for b in sorted(model_colors, key=len, reverse=True):
        if key.startswith(b): return b
    return None

def get_color(key):
    b = _get_base_model(key)
    return model_colors[b] if b else "#999999"

def get_name(key):
    b = _get_base_model(key)
    return model_display.get(b, key)

def _match_tissue(desired, available):
    for t in available:
        if desired in t.lower(): return t
    return None

tissues_fig1 = ["haec10", "lung", "testis", "brain_cortex", "whole_blood"]
tissues = list(tissues_fig1)
# data roots — override via env vars for external reproducers
base = Path(os.environ.get("SPLAIRE_DATA_DIR", "/scratch/runyan.m/sphaec_out"))
canonical_base = Path(os.environ.get("SPLAIRE_CANONICAL_DIR", "/scratch/runyan.m/splaire_out/canonical"))

# canonical benchmarks (classification only, no per-individual variation)
gencode_tissues = ["gencode", "mane_select"]

# pangolin test set (4 tissues, each scored separately)
pang_test_tissues = ["heart", "liver", "brain", "testis"]
pang_tissue_display = {"heart": "Heart", "liver": "Liver", "brain": "Brain", "testis": "Testis"}

fig2_main = fig_main
fig2_sup = fig_sup

exclude_bases = {"splaire_var", "splaire_avg"}
model_order = ["splaire_ref", "spliceai", "pangolin", "splicetransformer"]
metrics_subdir = "ml_out_var/metrics"

log.info("config loaded")

# load all metrics from JSON files
cls_rows, reg_rows, bin_cls_ratio_rows, bin_reg_rows = [], [], [], []
n_files_loaded = 0

cls_subsets = ["all", "ssu_valid", "ssu_valid_nonzero", "ssu_shared", "ssu_shared_nonzero"]
reg_subsets_list = ["ssu_valid", "ssu_valid_nonzero", "ssu_shared", "ssu_shared_nonzero"]


def _load_json_metrics(jp, tissue, sid):
    """parse metrics json into cls/reg/binned rows"""
    with open(jp) as fp:
        d = json.load(fp)["overall"]

    # classification
    for subset in cls_subsets:
        for target in ["acceptor", "donor"]:
            grp = d.get("classification", {}).get(subset, {}).get(target, {})
            for output, m in grp.items():
                row = {"tissue": tissue, "sample_id": sid, "subset": subset,
                       "target": target, "output": output,
                       "auprc": float(m["auprc"]), "auroc": float(m["auroc"]),
                       "topk": float(m["topk"]),
                       "n_pos": int(m["n_pos"]), "n_neg": int(m.get("n_neg", 0))}
                if "f1_max" in m:
                    row["f1_max"] = float(m["f1_max"])
                cls_rows.append(row)

    # regression
    for subset in reg_subsets_list:
        grp = d.get("regression", {}).get(subset, {})
        for output, m in grp.items():
            p = m.get("pearson")
            if p is None or (isinstance(p, float) and p != p):
                continue  # skip NaN
            row = {"tissue": tissue, "sample_id": sid, "subset": subset,
                   "output": output,
                   "pearson": float(p), "spearman": float(m["spearman"]),
                   "r2": float(m["r2"]), "mse": float(m["mse"]), "n": int(m["n"])}
            if "mae" in m: row["mae"] = float(m["mae"])
            reg_rows.append(row)

    # binned classification (ratio-preserving)
    for subset in ["ssu_valid", "ssu_valid_nonzero"]:
        for target in ["acceptor", "donor"]:
            grp = d.get("binned", {}).get("classification_ratio", {}).get(subset, {}).get(target, {})
            for output, bins in grp.items():
                for bk, m in bins.items():
                    bin_cls_ratio_rows.append({
                        "tissue": tissue, "sample_id": sid, "subset": subset,
                        "target": target, "output": output, "bin": int(bk.replace("bin_", "")),
                        "auprc": float(m["auprc"]), "auroc": float(m["auroc"]),
                        "topk": float(m["topk"]),
                        "n_pos": int(m["n_pos"]), "n_neg": int(m.get("n_neg", 0))})

    # binned regression
    for subset in reg_subsets_list:
        grp = d.get("binned", {}).get("regression", {}).get(subset, {})
        for output, bins in grp.items():
            for bk, m in bins.items():
                p = m.get("pearson")
                if p is None or (isinstance(p, float) and p != p):
                    continue
                row = {"tissue": tissue, "sample_id": sid, "subset": subset,
                       "output": output, "bin": int(bk.replace("bin_", "")),
                       "pearson": float(p), "spearman": float(m["spearman"]),
                       "r2": float(m["r2"]), "mse": float(m["mse"]), "n": int(m["n"])}
                if "mae" in m: row["mae"] = float(m["mae"])
                bin_reg_rows.append(row)


# --- expression tissues (per-individual JSON files) ---
for tissue in tissues:
    json_dir = base / tissue / metrics_subdir
    if not json_dir.exists():
        log.info(f"{tissue}: not found")
        continue
    json_files = sorted(json_dir.glob("test_*.json"))
    log.info(f"{tissue}: {len(json_files)} files")

    for jp in json_files:
        sid = jp.stem.replace("test_", "")
        n_files_loaded += 1
        _load_json_metrics(jp, tissue, sid)

# --- canonical benchmarks (gencode, mane_select) ---
canonical_metrics = {
    "gencode": canonical_base / "gencode" / "gencode_metrics.json",
    "mane_select": canonical_base / "mane_select" / "mane_select_metrics.json",
}
for tissue, jp in canonical_metrics.items():
    if not jp.exists():
        log.info(f"{tissue}: not found at {jp}")
        continue
    n_files_loaded += 1
    _load_json_metrics(jp, tissue, "reference")
    log.info(f"{tissue}: {jp.name}")

# --- pangolin test set (per-tissue JSON files) ---
pang_cls_rows, pang_reg_rows = [], []
pang_test_dir = canonical_base / "pangolin"
for pt in pang_test_tissues:
    jp = pang_test_dir / f"{pt}_metrics.json"
    if not jp.exists():
        log.info(f"pangolin {pt}: not found at {jp}")
        continue
    with open(jp) as fp:
        d = json.load(fp)["overall"]
    log.info(f"pangolin {pt}: {jp.name}")

    # classification (all subset)
    for target in ["acceptor", "donor"]:
        grp = d.get("classification", {}).get("all", {}).get(target, {})
        for output, m in grp.items():
            pang_cls_rows.append({
                "tissue": pt, "output": output,
                "auprc": float(m["auprc"]), "auroc": float(m["auroc"]),
                "topk": float(m["topk"]),
                "n_pos": int(m["n_pos"]), "n_neg": int(m.get("n_neg", 0)),
                "f1": float(m.get("f1_max", m["topk"])),
            })

    # regression
    for subset in ["ssu_valid", "ssu_valid_nonzero"]:
        grp = d.get("regression", {}).get(subset, {})
        for output, m in grp.items():
            p = m.get("pearson")
            if p is None or (isinstance(p, float) and p != p):
                continue
            pang_reg_rows.append({
                "tissue": pt, "subset": subset, "output": output,
                "pearson": float(p), "spearman": float(m["spearman"]),
                "r2": float(m["r2"]), "mse": float(m["mse"]), "n": int(m["n"]),
            })

df_pang_cls = pd.DataFrame(pang_cls_rows)
df_pang_reg = pd.DataFrame(pang_reg_rows)

# --- build dataframes ---
df_cls = pd.DataFrame(cls_rows)
df_reg = pd.DataFrame(reg_rows)
df_bin_ratio = pd.DataFrame(bin_cls_ratio_rows) if bin_cls_ratio_rows else pd.DataFrame(
    columns=["tissue","sample_id","subset","target","output","bin","auprc","auroc","topk","n_pos","n_neg"])
df_bin_reg = pd.DataFrame(bin_reg_rows) if bin_reg_rows else pd.DataFrame(
    columns=["tissue","sample_id","subset","output","bin","mse","n"])
all_tissues = tissues + [t for t in gencode_tissues if t in df_cls.tissue.unique()]

log.info(f"\nloaded {n_files_loaded} files")
log.info(f"df_cls: {df_cls.shape}, df_reg: {df_reg.shape}")
log.info(f"df_bin_ratio: {df_bin_ratio.shape}, df_bin_reg: {df_bin_reg.shape}")
log.info(f"all_tissues: {all_tissues}")
if len(df_cls) > 0: log.info(f"cls outputs: {sorted(df_cls.output.unique())}")
if len(df_reg) > 0: log.info(f"reg outputs: {sorted(df_reg.output.unique())}")
if len(df_pang_cls) > 0: log.info(f"pangolin cls: {df_pang_cls.shape}, tissues: {sorted(df_pang_cls.tissue.unique())}")
if len(df_pang_reg) > 0: log.info(f"pangolin reg: {df_pang_reg.shape}, tissues: {sorted(df_pang_reg.tissue.unique())}")

# save classification metrics summary
if len(df_cls) > 0:
    _cls_summary = (df_cls.groupby(["tissue", "subset", "output"])
                    .agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), topk=("topk", "mean"))
                    .reset_index())
    _cls_summary.to_csv(results_dir / "cls_metrics.csv", index=False)
    log.info(f"saved cls_metrics.csv: {_cls_summary.shape}")

# save regression metrics summary
if len(df_reg) > 0:
    _reg_summary = (df_reg.groupby(["tissue", "subset", "output"])
                    .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                         r2=("r2", "mean"), mse=("mse", "mean"))
                    .reset_index())
    _reg_summary.to_csv(results_dir / "reg_metrics.csv", index=False)
    log.info(f"saved reg_metrics.csv: {_reg_summary.shape}")

# --- supplemental LaTeX tables ---
tables_dir = Path("data/tables")
tables_dir.mkdir(parents=True, exist_ok=True)

# all cls/reg output names from data
_all_cls_outputs = sorted(df_cls.output.unique()) if len(df_cls) > 0 else []
_all_reg_outputs = sorted(df_reg.output.unique()) if len(df_reg) > 0 else []
_all_pang_cls_outputs = sorted(df_pang_cls.output.unique()) if len(df_pang_cls) > 0 else []
_all_pang_reg_outputs = sorted(df_pang_reg.output.unique()) if len(df_pang_reg) > 0 else []

# short display names for outputs
_output_display = {
    "splaire_ref": "SPLAIRE", "splaire_ref_ssu": "SPLAIRE (ssu)",
    "splaire_ref_cls": "SPLAIRE (cls)",
    "splaire_var": "SPLAIRE-var", "splaire_var_ssu": "SPLAIRE-var (ssu)",
    "splaire_var_cls": "SPLAIRE-var (cls)",
    "spliceai": "SpliceAI", "spliceai_cls": "SpliceAI (cls)",
    "pangolin_avg_p_splice": "Pang (avg cls)", "pangolin_avg_usage": "Pang (avg reg)",
    "pangolin_heart_p_splice": "Pang (heart cls)", "pangolin_heart_usage": "Pang (heart reg)",
    "pangolin_liver_p_splice": "Pang (liver cls)", "pangolin_liver_usage": "Pang (liver reg)",
    "pangolin_brain_p_splice": "Pang (brain cls)", "pangolin_brain_usage": "Pang (brain reg)",
    "pangolin_testis_p_splice": "Pang (testis cls)", "pangolin_testis_usage": "Pang (testis reg)",
    "splicetransformer": "SpT", "splicetransformer_cls": "SpT (cls)",
    "splicetransformer_avg_tissue": "SpT (avg)",
    "splicetransformer_adipose": "SpT (adipose)", "splicetransformer_blood": "SpT (blood)",
    "splicetransformer_blood_vessel": "SpT (bv)", "splicetransformer_brain": "SpT (brain)",
    "splicetransformer_colon": "SpT (colon)", "splicetransformer_heart": "SpT (heart)",
    "splicetransformer_kidney": "SpT (kidney)", "splicetransformer_liver": "SpT (liver)",
    "splicetransformer_lung": "SpT (lung)", "splicetransformer_muscle": "SpT (muscle)",
    "splicetransformer_nerve": "SpT (nerve)", "splicetransformer_skin": "SpT (skin)",
    "splicetransformer_small_intestine": "SpT (sm int)", "splicetransformer_spleen": "SpT (spleen)",
    "splicetransformer_stomach": "SpT (stomach)",
    "merlin_cls": "Merlin (cls)", "merlin_ssu": "Merlin (ssu)",
}

_exp_tissues = ["haec10", "lung", "testis", "brain_cortex", "whole_blood"]
_cls_tissues = _exp_tissues + ["gencode", "mane_select"]

_cls_metrics_list = [("auprc", "AUPRC", True), ("auroc", "AUROC", True), ("topk", "Top-$k$", True)]
_reg_metrics_list = [
    ("pearson", r"Pearson $r$", True), ("spearman", r"Spearman $\rho$", True),
    ("r2", r"$R^2$", True), ("mse", "MSE", False),
]
_reg_subsets = [
    ("ssu_valid", "all valid sites"),
    ("ssu_valid_nonzero", "valid, SSU > 0"),
    ("ssu_shared", "shared sites"),
    ("ssu_shared_nonzero", "shared, SSU > 0"),
]


def _fmt_row(vals, maximize=True):
    """format floats with best value bolded"""
    valid = [v for v in vals if not np.isnan(v)]
    best = (max(valid) if maximize else min(valid)) if valid else None
    out = []
    for v in vals:
        if np.isnan(v):
            out.append("--")
        elif best is not None and abs(v - best) < 1e-9:
            out.append(rf"\textbf{{{v:.3f}}}")
        else:
            out.append(f"{v:.3f}")
    return out


def _make_md_table(df, outputs, tissues, metrics, caption):
    """generate markdown pipe table, bold best per row"""
    names = [_output_display.get(o, o) for o in outputs]
    lines = [f"**{caption}**", "", '<div style="overflow-x: auto">', ""]
    header = "| Metric | Tissue | " + " | ".join(names) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(outputs)) + "|"
    lines += [header, sep]
    for mcol, mname, maximize in metrics:
        for tissue in tissues:
            rdf = df[df.tissue == tissue]
            vals = []
            for o in outputs:
                match = rdf[rdf.output == o]
                vals.append(float(match[mcol].values[0]) if len(match) > 0 else float("nan"))
            # bold best
            valid = [v for v in vals if not np.isnan(v)]
            best = (max(valid) if maximize else min(valid)) if valid else None
            cells = []
            for v in vals:
                if np.isnan(v):
                    cells.append("--")
                elif best is not None and abs(v - best) < 1e-9:
                    cells.append(f"**{v:.3f}**")
                else:
                    cells.append(f"{v:.3f}")
            tname = tissue_display.get(tissue, tissue)
            lines.append(f"| {mname} | {tname} | " + " | ".join(cells) + " |")
    lines += ["", "</div>", ""]
    return "\n".join(lines)


def _make_table(df, outputs, tissues, metrics, label, caption, split_at=13):
    """generate latex tables for metric comparisons"""
    names = [_output_display.get(o, o) for o in outputs]
    n_t = len(tissues)

    parts = [(outputs, names)]
    if len(outputs) > split_at:
        parts = [
            (outputs[:split_at], names[:split_at]),
            (outputs[split_at:], names[split_at:]),
        ]

    all_lines = []
    for pi, (part_out, part_names) in enumerate(parts):
        lines = [
            r"\begin{landscape}",
            r"\begin{table}[htbp]",
        ]
        if pi > 0:
            lines.append(r"\ContinuedFloat")
        lines += [r"\small", r"\centering"]
        if pi == 0:
            lines.append(rf"\caption{{{caption}}}")
        else:
            lines.append(r"\caption[]{continued}")
        part_label = f"{label}_{chr(97 + pi)}" if len(parts) > 1 else label
        lines.append(rf"\label{{{part_label}}}")
        lines += [
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{ll" + "r" * len(part_out) + "}",
            r"\toprule",
            r"Metric & Tissue & " + " & ".join(part_names) + r" \\",
            r"\midrule",
        ]
        for mi, (mcol, mname, maximize) in enumerate(metrics):
            for ti, tissue in enumerate(tissues):
                rdf = df[df.tissue == tissue]
                vals = []
                for o in part_out:
                    match = rdf[rdf.output == o]
                    vals.append(float(match[mcol].values[0]) if len(match) > 0 else float("nan"))
                cells = _fmt_row(vals, maximize=maximize)
                tname = tissue_display.get(tissue, tissue)
                mrcell = rf"\multirow{{{n_t}}}{{*}}{{{mname}}}" if ti == 0 else ""
                lines.append(f"{mrcell} & {tname} & " + " & ".join(cells) + r" \\")
            if mi < len(metrics) - 1:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", r"\end{landscape}"]
        all_lines.extend(lines)
    return "\n".join(all_lines)


all_md_tables = []  # collect markdown tables for combined output

# --- 1. classification tables (5 tissues + gencode + mane, all cls outputs, per subset) ---
if len(df_cls) > 0:
    all_md_tables.append("## Classification metrics\n")
    for subset in cls_subsets:
        agg = (df_cls[df_cls.subset == subset]
               .groupby(["tissue", "output"])
               .agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), topk=("topk", "mean"))
               .reset_index())
        avail = [o for o in _all_cls_outputs if o in agg.output.values]
        cap = (f"Classification metrics ({subset.replace('_', ' ')}) across tissues and models. "
               "Values are means across held-out individuals and acceptor/donor targets; "
               "GENCODE and MANE use a single reference sample. Bold indicates best per row.")
        tex = _make_table(agg, avail, _cls_tissues, _cls_metrics_list, f"tab:cls_{subset}", cap)
        (tables_dir / f"table_cls_{subset}.tex").write_text(tex)
        all_md_tables.append(_make_md_table(agg, avail, _cls_tissues, _cls_metrics_list, cap))
    log.info(f"saved {len(cls_subsets)} classification tables")

# --- 2. regression tables (5 expression tissues, all reg outputs, per subset) ---
if len(df_reg) > 0:
    all_md_tables.append("## Regression metrics\n")
    for sub_key, sub_label in _reg_subsets:
        agg = (df_reg[df_reg.subset == sub_key]
               .groupby(["tissue", "output"])
               .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                    r2=("r2", "mean"), mse=("mse", "mean"))
               .reset_index())
        avail = [o for o in _all_reg_outputs if o in agg.output.values]
        cap = (f"Regression metrics ({sub_label}) across tissues and models. "
               "Values are means across held-out individuals. Bold indicates best per row.")
        tex = _make_table(agg, avail, _exp_tissues, _reg_metrics_list, f"tab:reg_{sub_key}", cap)
        (tables_dir / f"table_reg_{sub_key}.tex").write_text(tex)
        all_md_tables.append(_make_md_table(agg, avail, _exp_tissues, _reg_metrics_list, cap))
    log.info(f"saved {len(_reg_subsets)} regression tables")

# --- 3. cls outputs evaluated as regression (cross-task) ---
if len(df_reg) > 0:
    cls_on_reg = [o for o in _all_reg_outputs
                  if any(o.endswith(s) for s in ("_cls",)) or o in ("spliceai_cls", "splicetransformer_cls")]
    if cls_on_reg:
        all_md_tables.append("## Classification outputs as regression\n")
        for sub_key, sub_label in _reg_subsets:
            agg = (df_reg[df_reg.subset == sub_key]
                   .groupby(["tissue", "output"])
                   .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                        r2=("r2", "mean"), mse=("mse", "mean"))
                   .reset_index())
            avail = [o for o in cls_on_reg if o in agg.output.values]
            if not avail:
                continue
            cap = (f"Classification outputs evaluated as regression ({sub_label}). "
                   "Classification scores used as SSU proxies. Bold indicates best per row.")
            tex = _make_table(agg, avail, _exp_tissues, _reg_metrics_list, f"tab:cls_as_reg_{sub_key}", cap)
            (tables_dir / f"table_cls_as_reg_{sub_key}.tex").write_text(tex)
            all_md_tables.append(_make_md_table(agg, avail, _exp_tissues, _reg_metrics_list, cap))
        log.info("saved cls-as-regression tables")

# --- 4. reg outputs evaluated as classification (cross-task) ---
if len(df_cls) > 0:
    reg_on_cls = [o for o in _all_cls_outputs
                  if any(o.endswith(s) for s in ("_ssu", "_usage", "_avg_tissue"))
                  or "splicetransformer_" in o and o != "splicetransformer"]
    if reg_on_cls:
        all_md_tables.append("## Regression outputs as classification\n")
        for subset in cls_subsets:
            agg = (df_cls[df_cls.subset == subset]
                   .groupby(["tissue", "output"])
                   .agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), topk=("topk", "mean"))
                   .reset_index())
            avail = [o for o in reg_on_cls if o in agg.output.values]
            if not avail:
                continue
            cap = (f"Regression outputs evaluated as classification ({subset.replace('_', ' ')}). "
                   "Regression scores thresholded for splice site identification. "
                   "Bold indicates best per row.")
            tex = _make_table(agg, avail, _cls_tissues, _cls_metrics_list, f"tab:reg_as_cls_{subset}", cap)
            (tables_dir / f"table_reg_as_cls_{subset}.tex").write_text(tex)
            all_md_tables.append(_make_md_table(agg, avail, _cls_tissues, _cls_metrics_list, cap))
        log.info("saved reg-as-classification tables")

# --- 5. pangolin benchmark classification table ---
if len(df_pang_cls) > 0:
    all_md_tables.append("## Pangolin benchmark — classification\n")
    agg = (df_pang_cls
           .groupby(["tissue", "output"])
           .agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), topk=("topk", "mean"))
           .reset_index())
    avail = [o for o in _all_pang_cls_outputs if o in agg.output.values]
    cap = ("Classification metrics on the Pangolin independent test set. "
           "Single reference genome per tissue, no per-individual variation. "
           "Bold indicates best per row.")
    tex = _make_table(agg, avail, sorted(df_pang_cls.tissue.unique()), _cls_metrics_list, "tab:pang_cls", cap)
    (tables_dir / "table_pang_cls.tex").write_text(tex)
    all_md_tables.append(_make_md_table(agg, avail, sorted(df_pang_cls.tissue.unique()), _cls_metrics_list, cap))
    log.info("saved pangolin classification table")

# --- 6. pangolin benchmark regression tables (per subset) ---
if len(df_pang_reg) > 0:
    all_md_tables.append("## Pangolin benchmark — regression\n")
    pang_reg_subsets = sorted(df_pang_reg.subset.unique())
    for sub_key in pang_reg_subsets:
        sub_label = sub_key.replace("_", " ")
        agg = (df_pang_reg[df_pang_reg.subset == sub_key]
               .groupby(["tissue", "output"])
               .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                    r2=("r2", "mean"), mse=("mse", "mean"))
               .reset_index())
        avail = [o for o in _all_pang_reg_outputs if o in agg.output.values]
        cap = (f"Regression metrics ({sub_label}) on the Pangolin independent test set. "
               "Bold indicates best per row.")
        tex = _make_table(agg, avail, sorted(df_pang_reg.tissue.unique()), _reg_metrics_list,
                          f"tab:pang_reg_{sub_key}", cap)
        (tables_dir / f"table_pang_reg_{sub_key}.tex").write_text(tex)
        all_md_tables.append(_make_md_table(agg, avail, sorted(df_pang_reg.tissue.unique()), _reg_metrics_list, cap))
    log.info(f"saved {len(pang_reg_subsets)} pangolin regression tables")

# write combined markdown tables
(tables_dir / "all_tables.md").write_text("\n".join(all_md_tables))
log.info(f"saved combined markdown tables to {tables_dir / 'all_tables.md'}")

log.info(f"all supplemental tables saved to {tables_dir}")

# unified parquet loading — used by density, calibration, tissue variability, diagnostics
from tqdm.auto import tqdm
from scipy import stats

SENTINEL = 777.0
pred_base = Path(os.environ.get("SPLAIRE_DATA_DIR", "/scratch/runyan.m/sphaec_out"))
tissues_load = list(tissues_fig1)

# all prediction columns to load per model
_load_models = {
    "splaire_ref": ("_splaire_ref", ["acceptor", "donor", "ssu"]),
    "spliceai": ("_sa", ["acceptor", "donor"]),
    "pangolin": ("_pang", ["avg_usage", "avg_p_splice"]),
    "splicetransformer": ("_spt", ["acceptor", "donor", "avg_tissue"]),
}

# build both tissue_data (concatenated arrays) and df_all (per-site with coords)
tissue_data = {}
all_site_rows = []  # for tissue variability (needs coords)

for tissue in tissues_load:
    pred_dir = pred_base / tissue / "ml_out_var" / "predictions"
    sample_dirs = sorted(pred_dir.glob("test_*"))
    log.info(f"{tissue}: {len(sample_dirs)} samples")

    # for tissue_data (density/calibration): concatenated arrays
    all_ssu = []
    all_preds = {k: [] for k in ["splaire_reg", "splaire_cls", "spliceai_cls",
                                  "pangolin_reg", "pangolin_cls", "spt_reg", "spt_cls"]}

    for sdir in tqdm(sample_dirs, desc=tissue):
        sid = sdir.name

        # use pre-filtered splice_sites.parquet (splice sites only, both alleles)
        slim = sdir / "splice_sites.parquet"
        if not slim.exists():
            log.warning(f"  {sid}: splice_sites.parquet not found, skipping")
            continue
        df_ss = pd.read_parquet(slim)
        y_ssu_arr = df_ss["y_ssu"].values

        # for tissue_data
        all_ssu.append(y_ssu_arr)
        all_preds["splaire_reg"].append(df_ss["splaire_ref_ssu"].values)
        all_preds["splaire_cls"].append(np.maximum(df_ss["splaire_ref_acceptor"].values, df_ss["splaire_ref_donor"].values))
        all_preds["spliceai_cls"].append(np.maximum(df_ss["spliceai_acceptor"].values, df_ss["spliceai_donor"].values))
        all_preds["pangolin_reg"].append(df_ss["pangolin_avg_usage"].values)
        all_preds["pangolin_cls"].append(df_ss["pangolin_avg_p_splice"].values)
        all_preds["spt_reg"].append(df_ss["splicetransformer_avg_tissue"].values)
        all_preds["spt_cls"].append(np.maximum(df_ss["splicetransformer_acceptor"].values, df_ss["splicetransformer_donor"].values))

        # for df_all (tissue variability)
        site_row = df_ss[["chrom", "pos", "strand"]].copy()
        site_row["tissue"] = tissue
        site_row["sample"] = sid
        site_row["y_ssu"] = y_ssu_arr
        # regression outputs (SSU-scale predictions)
        site_row["pred_splaire_ref_reg"] = df_ss["splaire_ref_ssu"].values
        site_row["pred_pangolin_reg"] = df_ss["pangolin_avg_usage"].values
        site_row["pred_splicetransformer_reg"] = df_ss["splicetransformer_avg_tissue"].values
        # classification outputs (splice probability, max of acceptor/donor)
        site_row["pred_splaire_ref_cls"] = np.maximum(df_ss["splaire_ref_acceptor"].values, df_ss["splaire_ref_donor"].values)
        site_row["pred_spliceai_cls"] = np.maximum(df_ss["spliceai_acceptor"].values, df_ss["spliceai_donor"].values)
        site_row["pred_pangolin_cls"] = df_ss["pangolin_avg_p_splice"].values
        site_row["pred_splicetransformer_cls"] = np.maximum(df_ss["splicetransformer_acceptor"].values, df_ss["splicetransformer_donor"].values)
        # backwards compat: old names used by tissue variability section
        site_row["pred_splaire_ref"] = site_row["pred_splaire_ref_reg"]
        site_row["pred_spliceai"] = site_row["pred_spliceai_cls"]
        site_row["pred_pangolin"] = site_row["pred_pangolin_reg"]
        site_row["pred_splicetransformer"] = site_row["pred_splicetransformer_reg"]

        all_site_rows.append(site_row)
        del df_ss

    # build tissue_data for density/calibration
    y_ssu_cat = np.concatenate(all_ssu)
    preds_cat = {k: np.concatenate(v) for k, v in all_preds.items()}
    valid = y_ssu_cat != SENTINEL
    nonzero = y_ssu_cat > 0

    tissue_data[tissue] = {
        "y_ssu": y_ssu_cat,
        "preds": preds_cat,
        "masks": {
            "ssu_valid": valid,
            "ssu_valid_nonzero": valid & nonzero,
        },
    }
    for sname, smask in tissue_data[tissue]["masks"].items():
        log.info(f"  {sname}: {smask.sum():,}")

# build df_all for tissue variability
df_all = pd.concat(all_site_rows, ignore_index=True)
log.info(f"\ntotal: {len(df_all):,} site-sample-tissue observations")
log.info(f"unique sites: {df_all.groupby(['chrom','pos','strand']).ngroups:,}")
log.info(f"tissues: {df_all['tissue'].nunique()}, samples: {df_all['sample'].nunique()}")

# tissue variability: cross-tissue stats (inner join)
tissue_means = (df_all[df_all.y_ssu != SENTINEL]
                .groupby(["chrom", "pos", "strand", "tissue"])["y_ssu"]
                .mean()
                .unstack("tissue"))
tissue_means = tissue_means.dropna()
log.info(f"\nsites with valid SSU in all {len(tissues_load)} tissues: {len(tissue_means):,}")

tissue_means["ssu_tissue_mean"] = tissue_means[tissues_load].mean(axis=1)
tissue_means["ssu_tissue_var"] = tissue_means[tissues_load].var(axis=1)
tissue_means["ssu_tissue_range"] = tissue_means[tissues_load].max(axis=1) - tissue_means[tissues_load].min(axis=1)
tissue_means["ssu_tissue_std"] = tissue_means[tissues_load].std(axis=1)

within_var = (df_all[df_all.y_ssu != SENTINEL]
              .groupby(["chrom", "pos", "strand", "tissue"])["y_ssu"]
              .var()
              .unstack("tissue"))
tissue_means["within_tissue_var"] = within_var.reindex(tissue_means.index).mean(axis=1)

var_cols = tissue_means[["ssu_tissue_mean", "ssu_tissue_var", "ssu_tissue_range",
                          "ssu_tissue_std", "within_tissue_var"]].reset_index()
var_cols.to_csv(results_dir / "tissue_variability.csv", index=False)
log.info(f"saved tissue_variability.csv: {var_cols.shape}")
df_eval = df_all.merge(var_cols, on=["chrom", "pos", "strand"])

pred_cols = [c for c in df_eval.columns if c.startswith("pred_")]
for pc in pred_cols:
    model = pc.replace("pred_", "")
    df_eval[f"resid_{model}"] = (df_eval[pc] - df_eval["y_ssu"]).abs()

resid_cols = [c for c in df_eval.columns if c.startswith("resid_")]
id_cols = ["chrom", "pos", "strand", "tissue", "sample", "y_ssu",
           "ssu_tissue_var", "ssu_tissue_range", "ssu_tissue_mean", "ssu_tissue_std", "within_tissue_var"]
df_resid = df_eval[id_cols + resid_cols].copy()
df_resid = df_resid.melt(id_vars=id_cols, value_vars=resid_cols,
                          var_name="model", value_name="abs_resid")
df_resid["model"] = df_resid["model"].str.replace("resid_", "", regex=False)

log.info(f"\nevaluation table: {len(df_eval):,} rows")
log.info(f"residual table: {len(df_resid):,} rows")
log.info("done loading")

# --only-tau: skip all plotting/tables, run only tau analysis + html render
if _only_tau:
    from scipy.stats import pearsonr as _pearsonr
    _reg_models = ["splaire_ref", "pangolin", "splicetransformer"]
    _cls_models = ["splaire_ref", "spliceai", "pangolin", "splicetransformer"]
    _model_display = {"splaire_ref": "SPLAIRE", "spliceai": "SpliceAI",
                      "pangolin": "Pangolin", "splicetransformer": "SpliceTransformer"}
    _reg_pred_cols = {"splaire_ref": "pred_splaire_ref_reg",
                      "pangolin": "pred_pangolin_reg",
                      "splicetransformer": "pred_splicetransformer_reg"}
    _cls_pred_cols = {"splaire_ref": "pred_splaire_ref_cls",
                      "spliceai": "pred_spliceai_cls",
                      "pangolin": "pred_pangolin_cls",
                      "splicetransformer": "pred_splicetransformer_cls"}
    # backwards compat for tissue variability section
    _pred_cols = {"splaire_ref": "pred_splaire_ref", "spliceai": "pred_spliceai",
                  "pangolin": "pred_pangolin", "splicetransformer": "pred_splicetransformer"}

if not _only_tau:  # skip to tau analysis
    bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    cls_order = ["splaire_ref", "pangolin_avg_p_splice", "splicetransformer", "spliceai"]
    reg_order = ["splaire_ref_ssu", "pangolin_avg_usage", "splicetransformer_avg_tissue"]
    all_order = cls_order + reg_order
    
    _desired_v = ["haec", "lung", "brain", "testis", "blood"]
    tissues_v_order = [m for d in _desired_v if (m := _match_tissue(d, tissues_fig1)) is not None]
    
    # horizontal combined — binned AUPRC
    nt_h = len(tissues_v_order)
    fig, axes = plt.subplots(1, nt_h, figsize=(6 * nt_h, 5), sharey=True)
    for col, tissue in enumerate(tissues_v_order):
        ax = axes[col]
        sub = df_bin_ratio.query("tissue == @tissue & subset == 'ssu_valid_nonzero'").copy()
        if len(sub) == 0:
            ax.set_title(tissue_display.get(tissue, tissue)); continue
        avail = [o for o in all_order if o in sub.output.unique()]
        sub = sub[sub.output.isin(avail)]
        agg = sub.groupby(["output", "bin"]).agg(auprc=("auprc", "mean"), n_pos=("n_pos", "mean")).reset_index()
        counts = agg[agg.output == avail[0]][["bin", "n_pos"]].sort_values("bin")
        ax2 = ax.twinx()
        ax2.bar(counts["bin"], counts["n_pos"], width=0.5, color=GRAY_FAINT, alpha=0.4, zorder=1)
        ax2.tick_params(axis="y", labelcolor=GRAY_MED); ax2.set_yscale("log")
        ax2.set_ylim(counts["n_pos"].min() * 0.5, counts["n_pos"].max() * 3)
        if col == nt_h - 1:
            ax2.set_ylabel("Splice Sites", color=GRAY_MED)
        else:
            ax2.set_yticklabels([])
        for out in avail:
            od = agg[agg.output == out].sort_values("bin")
            ax.plot(od["bin"], od["auprc"], marker="o", color=get_color(out), lw=2, markersize=6, zorder=3,
                    ls="--" if out in reg_order else "-",
                    label=get_name(out) if col == 0 else None)
        ax.set_ylim(0, 1)
        if col == 0:
            ax.set_ylabel("AUPRC")
        ax.set_title(tissue_display.get(tissue, tissue))
        ax.set_xticks(range(10)); ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_xlabel("SSU bin")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(all_order), frameon=False)
    fig.suptitle("AUPRC stratified by SSU", fontsize=16, y=1.10)
    plt.tight_layout()
    _save_fig(plt, fig2_main / "binned_auprc_h")
    
    output_order_spec = [
        "splaire_ref_cls","splaire_ref_ssu","pangolin_avg_p_splice","pangolin_avg_usage",
        "splicetransformer_cls","splicetransformer_avg_tissue","spliceai_cls",
    ]
    reg_outputs = ["splaire_ref_ssu","pangolin_avg_usage","splicetransformer_avg_tissue"]
    _mc = "mae" if "mae" in df_bin_reg.columns else "mse"
    
    # horizontal combined — binned regression MSE/MAE
    nt_h = len(tissues_v_order)
    fig, axes = plt.subplots(1, nt_h, figsize=(6 * nt_h, 5), sharey=True)
    for col, tissue in enumerate(tissues_v_order):
        ax = axes[col]
        sub = df_bin_reg.query("tissue == @tissue & subset == 'ssu_valid_nonzero'").copy()
        if len(sub) == 0:
            ax.set_title(tissue_display.get(tissue, tissue)); continue
        avail = [o for o in output_order_spec if o in sub.output.unique()]
        sub = sub[sub.output.isin(avail)]
        agg = sub.groupby(["output", "bin"]).agg(metric=(_mc, "mean"), n=("n", "mean")).reset_index()
        counts = agg[agg.output == avail[0]][["bin", "n"]].sort_values("bin")
        ax2 = ax.twinx()
        ax2.bar(counts["bin"], counts["n"], width=0.5, color=GRAY_FAINT, alpha=0.4, zorder=1)
        ax2.tick_params(axis="y", labelcolor=GRAY_MED); ax2.set_yscale("log")
        ax2.set_ylim(counts["n"].min() * 0.5, counts["n"].max() * 3)
        if col == nt_h - 1:
            ax2.set_ylabel("Splice Sites", color=GRAY_MED)
        else:
            ax2.set_yticklabels([])
        for out in avail:
            od = agg[agg.output == out].sort_values("bin")
            ax.plot(od["bin"], od["metric"], marker="o", color=get_color(out), lw=2, markersize=6, zorder=3,
                    ls="--" if out in reg_outputs else "-",
                    label=get_name(out) if col == 0 else None)
        if col == 0:
            ax.set_ylabel("Mean Squared Error")
        ax.set_title(tissue_display.get(tissue, tissue))
        ax.set_xticks(range(10)); ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_xlabel("SSU bin")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(output_order_spec), frameon=False)
    fig.suptitle("MSE stratified by SSU", fontsize=16, y=1.10)
    plt.tight_layout()
    _save_fig(plt, fig2_main / "binned_regression_h")
    
    r2_display = {
        "splaire_ref_ssu":"SPLAIRE","spliceai_cls":"SpliceAI",
        "pangolin_avg_usage":"Pangolin","splicetransformer_avg_tissue":"SpliceTransformer",
    }
    bar_outputs = ["splaire_ref","spliceai","pangolin_avg_p_splice","splicetransformer"]
    bar_display = {"splaire_ref":"SPLAIRE","spliceai":"SpliceAI",
                   "pangolin_avg_p_splice":"Pangolin","splicetransformer":"SpliceTransformer"}
    r2_to_bar = {"splaire_ref_ssu":"splaire_ref","spliceai_cls":"spliceai",
                 "pangolin_avg_usage":"pangolin_avg_p_splice","splicetransformer_avg_tissue":"splicetransformer"}
    
    swarm_subsets = [("ssu_valid","all sites","all"),("ssu_valid_nonzero","SSU > 0","nonzero"),
                     ("ssu_shared","shared sites","shared"),("ssu_shared_nonzero","shared, SSU > 0","shared_nonzero")]
    
    # all regression metrics to plot
    _reg_metrics = [("r2", "$R^2$"), ("pearson", "Pearson r"), ("spearman", "Spearman r"), ("mse", "MSE")]
    if "mae" in df_reg.columns and df_reg["mae"].notna().any():
        _reg_metrics.append(("mae", "MAE"))
    
    # --- swarm plots: grouped by subset, all metrics per subset ---
    for subset_name, subset_label, suffix in swarm_subsets:
        sub = df_reg.query("subset == @subset_name").copy()
        sub["base_model"] = sub.output.apply(_get_base_model)
        sub = sub[~sub.base_model.isin(exclude_bases)]
        avail = [o for o in r2_outputs if o in sub.output.unique()]
        sub = sub[sub.output.isin(avail)]
        if len(sub) == 0: continue
    
        log.info(f"\n{'='*60}")
        log.info(f"subset: {subset_label}")
    
        for metric_col, metric_label in _reg_metrics:
            if metric_col not in sub.columns or sub[metric_col].isna().all():
                continue
            # sort by mean (descending for r2/pearson/spearman, ascending for mse/mae)
            ascending = metric_col in {"mse", "mae"}
            output_order = list(sub.groupby("output")[metric_col].mean()
                               .sort_values(ascending=ascending).index)
    
            fig, axes = plt.subplots(1, len(tissues_fig1), figsize=(3 * len(tissues_fig1), 4), sharey=True)
            for col, tissue in enumerate(tissues_fig1):
                ax = axes[col]; tdf = sub[sub.tissue == tissue]
                sns.stripplot(data=tdf, x="output", y=metric_col, hue="output", order=output_order,
                              palette={o: get_color(o) for o in output_order},
                              size=5, jitter=0.2, alpha=0.7, ax=ax, legend=False)
                for i, out in enumerate(output_order):
                    vals = tdf[tdf.output == out][metric_col]
                    if len(vals) == 0: continue
                    ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=get_color(out), lw=2)
                    ax.hlines(vals.median(), i - 0.3, i + 0.3, color=get_color(out), lw=1.5, ls="--")
                ax.set_xlabel("")
                ax.set_ylabel(metric_label if col == 0 else "")
                ax.set_title(tissue_display.get(tissue, tissue))
                ax.set_xticks(range(len(output_order)))
                ax.set_xticklabels([r2_display.get(o, o) for o in output_order], rotation=45, ha="right")
            fig.suptitle(f"{metric_label} ({subset_label})", y=1.05)
            fig.legend(handles=[Line2D([0], [0], color="gray", lw=2, label="mean"),
                                Line2D([0], [0], color="gray", lw=1.5, ls="--", label="median")],
                       loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"{metric_col}_swarm_{suffix}")
    
        log.info(f"  {len(_reg_metrics)} metric plots for {subset_label}")
    
    # --- bar plots ---
    output_order_bar = [r2_to_bar[o] for o in r2_outputs if r2_to_bar.get(o) in bar_outputs]
    
    def make_bar_plots(df, metric_col, ylabel, output_list, display_dict, filename_suffix, tissue_list=None):
        if tissue_list is None: tissue_list = tissues_fig1
        bdf = df.groupby(["tissue", "sample_id", "output"]).agg(val=(metric_col, "mean")).reset_index()
        bdf["base_model"] = bdf.output.apply(_get_base_model)
        bdf = bdf[~bdf.base_model.isin(exclude_bases)]
        avail = [o for o in output_list if o in bdf.output.unique()]
        bdf = bdf[bdf.output.isin(avail)]
        order = [o for o in output_order_bar if o in avail]
        nt = len(tissue_list)
        fig, axes = plt.subplots(1, nt, figsize=(3 * nt, 4), sharey=True)
        if nt == 1: axes = [axes]
        for col, tissue in enumerate(tissue_list):
            ax = axes[col]; tdf = bdf[bdf.tissue == tissue]
            agg = tdf.groupby("output").agg(mean=("val", "mean")).reset_index()
            agg = agg.set_index("output").reindex(order).reset_index()
            x = np.arange(len(order))
            bars = ax.bar(x, agg["mean"], color=[get_color(o) for o in order],
                          edgecolor="black", linewidth=0.5, alpha=0.8)
            for bar, val in zip(bars, agg["mean"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5, f"{val:.2f}",
                        ha="center", va="center", fontweight="bold", rotation=90, color="white")
            ax.set_xlabel(""); ax.set_ylabel(ylabel)
            ax.set_title(tissue_display.get(tissue, tissue))
            ax.set_xticks(x)
            ax.set_xticklabels([display_dict.get(o, o) for o in order], rotation=45, ha="right")
            ax.set_ylim(0, 1.1)
        plt.tight_layout()
        _save_fig(plt, fig2_main / f"{filename_suffix}")
    
    make_bar_plots(df_cls.query("subset == 'all'"), "topk", "Top-k", bar_outputs, bar_display, "topk_bars")
    make_bar_plots(df_cls.query("subset == 'all'"), "auprc", "AUPRC", bar_outputs, bar_display, "auprc_bars")
    
    # --- combined figure: AUPRC bars | binned AUPRC | binned MSE | R² swarm ---
    tissues_comb = [m for d in _desired_v if (m := _match_tissue(d, tissues_fig1)) is not None]
    if not tissues_comb: tissues_comb = tissues_fig1
    nt = len(tissues_comb)
    
    _orig_params = {k: plt.rcParams[k] for k in ["font.size", "axes.titlesize", "axes.labelsize",
                    "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "legend.title_fontsize", "figure.titlesize"]}
    _scale = 1.4
    for k, v in _orig_params.items(): plt.rcParams[k] = v * _scale
    _gap_01, _gap_12, _gap_23 = 0.35, 0.65, 0.65
    col_titles = ["AUPRC", "Binned AUPRC", "Binned MSE", r"$R^2$"]
    
    _bar_df = df_cls.query("subset == 'all'").copy()
    _bar_df = _bar_df.groupby(["tissue", "sample_id", "output"]).agg(val=("auprc", "mean")).reset_index()
    _bar_df["base_model"] = _bar_df.output.apply(_get_base_model)
    _bar_df = _bar_df[~_bar_df.base_model.isin(exclude_bases)]
    _bar_avail = [o for o in bar_outputs if o in _bar_df.output.unique()]
    _bar_df = _bar_df[_bar_df.output.isin(_bar_avail)]
    _bar_order = [o for o in output_order_bar if o in _bar_avail]
    
    _r2_sub = df_reg.query("subset == 'ssu_valid'").copy()
    _r2_sub["base_model"] = _r2_sub.output.apply(_get_base_model)
    _r2_sub = _r2_sub[~_r2_sub.base_model.isin(exclude_bases)]
    _r2_avail = [o for o in r2_outputs if o in _r2_sub.output.unique()]
    _r2_sub = _r2_sub[_r2_sub.output.isin(_r2_avail)]
    _r2_order = list(_r2_sub.groupby("output").r2.mean().sort_values(ascending=False).index)
    
    _mc = "mae" if "mae" in df_bin_reg.columns else "mse"
    _bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    _mse_vals = []
    for t in tissues_comb:
        sm = df_bin_reg.query("tissue == @t & subset == 'ssu_valid_nonzero'")
        if len(sm) > 0:
            av = [o for o in output_order_spec if o in sm.output.unique()]
            sm = sm[sm.output.isin(av)]
            _mse_vals.extend(sm.groupby(["output", "bin"]).agg(metric=(_mc, "mean")).reset_index()["metric"].dropna().tolist())
    _mse_ylim = (0, max(_mse_vals) * 1.1) if _mse_vals else (0, 1)
    _r2_vals = []
    for t in tissues_comb:
        tr = _r2_sub[_r2_sub.tissue == t]
        if len(tr) > 0: _r2_vals.extend(tr.r2.tolist())
    _r2_ylim = (min(_r2_vals) - 0.02, max(_r2_vals) + 0.02) if _r2_vals else (0, 1)
    _sec_fs = plt.rcParams["font.size"] * 0.75
    
    def _plot_row(fig, gs, row, tissue, show_titles=False, show_xlabels=False):
        _dc = [0, 2, 4, 6]
        ax = fig.add_subplot(gs[row, _dc[0]])
        tdf = _bar_df[_bar_df.tissue == tissue]
        agg = tdf.groupby("output").agg(mean=("val", "mean")).reset_index()
        agg = agg.set_index("output").reindex(_bar_order).reset_index()
        x = np.arange(len(_bar_order))
        bars = ax.bar(x, agg["mean"], color=[get_color(o) for o in _bar_order],
                      edgecolor="black", linewidth=0.5, alpha=0.8)
        for b, v in zip(bars, agg["mean"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.5, f"{v:.2f}",
                    ha="center", va="center", fontweight="bold", rotation=90, color="white",
                    fontsize=plt.rcParams["font.size"] * 1.25)
        ax.set_ylim(0, 1.1); ax.set_xticks(x)
        ax.set_xticklabels([bar_display.get(o, o) for o in _bar_order] if show_xlabels else [], rotation=45, ha="right")
        ax.set_ylabel("AUPRC")
        ax.text(-0.35, 0.5, tissue_display.get(tissue, tissue), transform=ax.transAxes,
                ha="right", va="center", fontweight="bold", fontsize=plt.rcParams["axes.labelsize"], rotation=90)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if show_titles: ax.set_title(col_titles[0], pad=15)
    
        ax = fig.add_subplot(gs[row, _dc[1]])
        sub = df_bin_ratio.query("tissue==@tissue & subset=='ssu_valid_nonzero'").copy()
        if len(sub) > 0:
            avail = [o for o in cls_order if o in sub.output.unique()]
            sub = sub[sub.output.isin(avail)]
            ag = sub.groupby(["output", "bin"]).agg(auprc=("auprc", "mean"), n_pos=("n_pos", "mean")).reset_index()
            ct = ag[ag.output == avail[0]][["bin", "n_pos"]].sort_values("bin")
            ax2 = ax.twinx()
            ax2.bar(ct["bin"], ct["n_pos"], width=0.5, color=GRAY_FAINT, alpha=0.4, zorder=1)
            ax2.tick_params(axis="y", labelcolor=GRAY_MED); ax2.set_yscale("log")
            ax2.set_ylim(ct["n_pos"].min() * 0.5, ct["n_pos"].max() * 3)
            ax2.set_ylabel("Splice Sites", color=GRAY_MED, fontsize=_sec_fs)
            for out in avail:
                od = ag[ag.output == out].sort_values("bin")
                ax.plot(od["bin"], od["auprc"], marker="o", color=get_color(out), lw=2, markersize=4, zorder=3)
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_ylim(0, 1); ax.set_ylabel("AUPRC"); ax.set_xticks(range(10))
        if show_xlabels: ax.set_xticklabels(_bin_labels, rotation=45, ha="right"); ax.set_xlabel("SSU bin")
        else: ax.set_xticklabels([])
        if show_titles: ax.set_title(col_titles[1], pad=15)
    
        ax = fig.add_subplot(gs[row, _dc[2]])
        sub = df_bin_reg.query("tissue==@tissue & subset=='ssu_valid_nonzero'").copy()
        if len(sub) > 0:
            avail = [o for o in output_order_spec if o in sub.output.unique()]
            sub = sub[sub.output.isin(avail)]
            ag = sub.groupby(["output", "bin"]).agg(metric=(_mc, "mean"), n=("n", "mean")).reset_index()
            ct = ag[ag.output == avail[0]][["bin", "n"]].sort_values("bin")
            ax2 = ax.twinx()
            ax2.bar(ct["bin"], ct["n"], width=0.5, color=GRAY_FAINT, alpha=0.4, zorder=1)
            ax2.tick_params(axis="y", labelcolor=GRAY_MED); ax2.set_yscale("log")
            ax2.set_ylim(ct["n"].min() * 0.5, ct["n"].max() * 3)
            ax2.set_ylabel("Splice Sites", color=GRAY_MED, fontsize=_sec_fs)
            for out in avail:
                od = ag[ag.output == out].sort_values("bin")
                ax.plot(od["bin"], od["metric"], marker="o", color=get_color(out), lw=2, markersize=4, zorder=3,
                        ls="--" if out in reg_outputs else "-")
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_ylabel("MSE"); ax.set_ylim(*_mse_ylim); ax.set_xticks(range(10))
        if show_xlabels: ax.set_xticklabels(_bin_labels, rotation=45, ha="right"); ax.set_xlabel("SSU bin")
        else: ax.set_xticklabels([])
        if show_titles: ax.set_title(col_titles[2], pad=15)
    
        ax = fig.add_subplot(gs[row, _dc[3]])
        tdf = _r2_sub[_r2_sub.tissue == tissue]
        if len(tdf) > 0:
            sns.stripplot(data=tdf, x="output", y="r2", hue="output", order=_r2_order,
                          palette={o: get_color(o) for o in _r2_order},
                          size=4, jitter=0.2, alpha=0.7, ax=ax, legend=False)
            for i, out in enumerate(_r2_order):
                vals = tdf[tdf.output == out].r2
                if len(vals) == 0: continue
                ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=get_color(out), lw=2)
                ax.hlines(vals.median(), i - 0.3, i + 0.3, color=get_color(out), lw=1.5, ls="--")
        ax.set_xlabel(""); ax.set_ylabel(r"$R^2$"); ax.set_ylim(*_r2_ylim)
        ax.set_xticks(range(len(_r2_order)))
        if show_xlabels: ax.set_xticklabels([r2_display.get(o, o) for o in _r2_order], rotation=45, ha="right")
        else: ax.set_xticklabels([])
        if show_titles: ax.set_title(col_titles[3], pad=15)
    
    fig = plt.figure(figsize=(28, 4.5 * nt))
    gs = fig.add_gridspec(nt, 7, width_ratios=[1, _gap_01, 1.5, _gap_12, 1.5, _gap_23, 1], hspace=0.35, wspace=0.05)
    for row, tissue in enumerate(tissues_comb):
        _plot_row(fig, gs, row, tissue, show_titles=(row == 0), show_xlabels=(row == nt - 1))
    for k, v in _orig_params.items(): plt.rcParams[k] = v
    _save_fig(plt, fig2_main / "combined_vertical")
    
    # standalone legend
    from collections import OrderedDict
    _all_outputs_leg = [o for o in output_order_spec
                        if o in df_bin_reg[df_bin_reg.subset == "ssu_valid_nonzero"].output.unique().tolist()]
    _legend_groups = OrderedDict()
    for o in _all_outputs_leg:
        bm = _get_base_model(o)
        if bm not in _legend_groups: _legend_groups[bm] = []
        ls = "--" if o in reg_outputs else "-"
        _legend_groups[bm].append((o, ls))
    _base_display = {"splaire_ref": "SPLAIRE", "pangolin": "Pangolin",
                     "splicetransformer": "SpliceTransformer", "spliceai": "SpliceAI"}
    _lh, _ll = [], []
    for i, (bm, entries) in enumerate(_legend_groups.items()):
        if i > 0: _lh.append(Line2D([0], [0], color="none", lw=0)); _ll.append("")
        for o, ls in entries:
            name = _base_display.get(bm, bm)
            _lh.append(Line2D([0], [0], color=get_color(o), lw=2.5, ls=ls, marker="o", markersize=8))
            _ll.append(f"{name} - SSU" if ls == "--" else name)
    fig_leg, ax_leg = plt.subplots(figsize=(3, 0.5 * len(_ll)))
    ax_leg.axis("off")
    ax_leg.legend(handles=_lh, labels=_ll, loc="center", frameon=True, edgecolor="lightgray", fancybox=True,
                  fontsize=plt.rcParams["font.size"] * _scale, handlelength=3, handleheight=1.5, labelspacing=0.6)
    _save_fig(fig_leg, fig2_main / "combined_legend")
    
    # sample diagnostics — uses df_all and tissue_data from unified loading
    # no additional parquet loading needed except for plot 2 (allele pairs need raw parquet rows)
    
    tissues_diag = list(tissues_fig1)
    
    # --- plot 1: n valid sites per individual per tissue ---
    n_rows = []
    for subset_name, subset_label in [("ssu_valid", "SSU Valid"), ("ssu_valid_nonzero", "SSU > 0")]:
        sub = df_reg.query("subset == @subset_name & output == @r2_outputs[0]")
        for _, row in sub.iterrows():
            n_rows.append({"tissue": row["tissue"], "sample_id": row["sample_id"],
                           "subset": subset_label, "n": row["n"]})
    df_n = pd.DataFrame(n_rows)
    
    if len(df_n) > 0:
        fig, axes = plt.subplots(1, len(tissues_diag), figsize=(3 * len(tissues_diag), 4), sharey=True)
        for col, tissue in enumerate(tissues_diag):
            ax = axes[col]
            tdf = df_n[df_n.tissue == tissue]
            sns.stripplot(data=tdf, x="subset", y="n", hue="subset", size=6, jitter=0.15, alpha=0.7, ax=ax, legend=False)
            for i, subset in enumerate(["SSU Valid", "SSU > 0"]):
                vals = tdf[tdf.subset == subset]["n"]
                if len(vals) > 0:
                    ax.hlines(np.median(vals), i - 0.3, i + 0.3, color="black", lw=2)
            ax.set_xlabel("")
            ax.set_ylabel("n valid splice sites" if col == 0 else "")
            ax.set_title(tissue_display.get(tissue, tissue))
        fig.suptitle("valid site count per individual (both alleles)", y=1.02)
        plt.tight_layout()
        _save_fig(plt, fig2_sup / "valid_sites_per_individual")
        log.info("plot 1: n valid sites per individual")
    
    # --- plot 2: maternal vs paternal prediction differences ---
    # this still needs raw parquets since tissue_data doesn't distinguish alleles
    allele_rows = []
    for tissue in tissues_diag:
        pred_dir = pred_base / tissue / "ml_out_var" / "predictions"
        sample_dirs = sorted(pred_dir.glob("test_*"))
        log.info(f"\n{tissue}: checking {len(sample_dirs)} individuals for allele differences")
    
        for sdir in tqdm(sample_dirs, desc=tissue):
            sid = sdir.name
            slim = sdir / "splice_sites.parquet"
            if not slim.exists():
                continue
            df = pd.read_parquet(slim, columns=["chrom", "pos", "strand", "y_ssu", "splaire_ref_ssu"])
            df = df.rename(columns={"splaire_ref_ssu": "ssu"})
            df["site_key"] = df["chrom"].astype(str) + "_" + df["pos"].astype(str) + "_" + df["strand"].astype(str)
            dups = df[df.duplicated("site_key", keep=False)].copy()
            if len(dups) == 0:
                continue
            site_groups = dups.groupby("site_key").agg(
                pred_range=("ssu", lambda x: x.max() - x.min()),
                n_alleles=("ssu", "count"),
                y_ssu=("y_ssu", "first"),
            ).reset_index()
            pairs = site_groups.query("n_alleles == 2 & y_ssu != @SENTINEL")
            if len(pairs) == 0:
                continue
            delta = pairs["pred_range"].values
            allele_rows.append({
                "tissue": tissue, "sample": sid,
                "n_pairs": len(pairs),
                "frac_diff_001": (delta > 0.01).mean(),
                "frac_diff_005": (delta > 0.05).mean(),
                "frac_diff_010": (delta > 0.10).mean(),
                "median_delta": np.median(delta),
                "mean_delta": np.mean(delta),
            })
    
    df_allele = pd.DataFrame(allele_rows)
    if len(df_allele) > 0:
        fig, axes = plt.subplots(1, len(tissues_diag), figsize=(4 * len(tissues_diag), 4), sharey=True)
        for col, tissue in enumerate(tissues_diag):
            ax = axes[col]
            tdf = df_allele[df_allele.tissue == tissue]
            if len(tdf) == 0: continue
            ax.bar(range(len(tdf)), tdf["frac_diff_001"].values, alpha=0.5, label="|delta| > 0.01")
            ax.bar(range(len(tdf)), tdf["frac_diff_005"].values, alpha=0.5, label="|delta| > 0.05")
            ax.bar(range(len(tdf)), tdf["frac_diff_010"].values, alpha=0.5, label="|delta| > 0.10")
            ax.set_xlabel("individual")
            ax.set_ylabel("fraction of sites" if col == 0 else "")
            ax.set_title(tissue_display.get(tissue, tissue))
            if col == 0: ax.legend(fontsize=9)
        fig.suptitle("fraction of sites where maternal != paternal prediction (SPLAIRE SSU)", y=1.02)
        plt.tight_layout()
        _save_fig(plt, fig2_sup / "allele_differences")
    
        log.info("\nper-tissue summary:")
        for tissue in tissues_diag:
            tdf = df_allele[df_allele.tissue == tissue]
            if len(tdf) == 0: continue
            tname = tissue_display.get(tissue, tissue)
            log.info(f"  {tname}: median |delta|={tdf['median_delta'].median():.4f}, "
                  f"frac>0.01={tdf['frac_diff_001'].median():.3f}, "
                  f"frac>0.05={tdf['frac_diff_005'].median():.3f}, "
                  f"frac>0.10={tdf['frac_diff_010'].median():.3f}, "
                  f"n_pairs={int(tdf['n_pairs'].median()):,}")
    else:
        log.info("no allele pairs found")
    
    # --- plot 2b: allele differences aggregated per tissue, all models ---
    allele_model_cols = {
        "splaire_ref_ssu": "SPLAIRE",
        "pangolin_avg_usage": "Pangolin",
        "splicetransformer_avg_tissue": "SpliceTransformer",
        "spliceai_cls": "SpliceAI",  # max(acceptor, donor)
    }
    allele_model_rows = []
    for tissue in tissues_diag:
        pred_dir = pred_base / tissue / "ml_out_var" / "predictions"
        sample_dirs = sorted(pred_dir.glob("test_*"))
        log.info(f"\n{tissue}: allele diffs all models, {len(sample_dirs)} individuals")
    
        for sdir in tqdm(sample_dirs, desc=f"{tissue} allele-all"):
            slim = sdir / "splice_sites.parquet"
            if not slim.exists():
                continue
            cols_needed = ["chrom", "pos", "strand", "y_ssu",
                           "splaire_ref_ssu", "pangolin_avg_usage",
                           "splicetransformer_avg_tissue",
                           "spliceai_acceptor", "spliceai_donor"]
            df = pd.read_parquet(slim, columns=cols_needed)
            # derive spliceai max
            df["spliceai_cls"] = np.maximum(df["spliceai_acceptor"].values,
                                            df["spliceai_donor"].values)
            df["site_key"] = (df["chrom"].astype(str) + "_" +
                              df["pos"].astype(str) + "_" +
                              df["strand"].astype(str))
            dups = df[df.duplicated("site_key", keep=False)].copy()
            if len(dups) == 0:
                continue
            for mcol, mname in allele_model_cols.items():
                grp = dups.groupby("site_key").agg(
                    pred_range=(mcol, lambda x: x.max() - x.min()),
                    n_alleles=(mcol, "count"),
                    y_ssu=("y_ssu", "first"),
                ).reset_index()
                pairs = grp.query("n_alleles == 2 & y_ssu != @SENTINEL")
                if len(pairs) == 0:
                    continue
                delta = pairs["pred_range"].values
                allele_model_rows.append({
                    "tissue": tissue, "model": mname,
                    "frac_diff_001": (delta > 0.01).mean(),
                    "frac_diff_005": (delta > 0.05).mean(),
                    "frac_diff_010": (delta > 0.10).mean(),
                    "median_delta": np.median(delta),
                    "mean_delta": np.mean(delta),
                })
    
    df_allele_models = pd.DataFrame(allele_model_rows)
    if len(df_allele_models) > 0:
        # aggregate across individuals: median per tissue-model
        agg = (df_allele_models
               .groupby(["tissue", "model"])
               .agg(frac_diff_001=("frac_diff_001", "median"),
                    frac_diff_005=("frac_diff_005", "median"),
                    frac_diff_010=("frac_diff_010", "median"),
                    median_delta=("median_delta", "median"),
                    mean_delta=("mean_delta", "median"))
               .reset_index())
    
        thresholds = [("frac_diff_001", r"|$\Delta$| > 0.01"),
                      ("frac_diff_005", r"|$\Delta$| > 0.05"),
                      ("frac_diff_010", r"|$\Delta$| > 0.10")]
        model_names = list(allele_model_cols.values())
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=False)
        if n_models == 1:
            axes = [axes]
        x_labels = [tissue_display.get(t, t) for t in tissues_diag]
        x = np.arange(len(tissues_diag))
        bar_w = 0.25
    
        for mi, mname in enumerate(model_names):
            ax = axes[mi]
            sub = agg.query("model == @mname")
            # reindex to tissues_diag order
            sub = sub.set_index("tissue").reindex(tissues_diag).reset_index()
            for ti, (col, label) in enumerate(thresholds):
                vals = sub[col].fillna(0).values
                ax.bar(x + (ti - 1) * bar_w, vals, bar_w, alpha=0.7, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=30, ha="right")
            ax.set_title(mname)
            ax.set_ylabel("fraction of sites")
            if mi == 0:
                ax.legend(fontsize=9)
    
        fig.suptitle("allele prediction differences aggregated across individuals", y=1.02)
        plt.tight_layout()
        _save_fig(plt, fig2_sup / "allele_differences_by_model")
    
        log.info("\nallele diffs by model (median across individuals):")
        for _, row in agg.iterrows():
            tname = tissue_display.get(row["tissue"], row["tissue"])
            log.info(f"  {row['model']:20s} {tname:8s}: "
                     f"frac>0.01={row['frac_diff_001']:.3f}, "
                     f"frac>0.05={row['frac_diff_005']:.3f}, "
                     f"frac>0.10={row['frac_diff_010']:.3f}, "
                     f"median|delta|={row['median_delta']:.4f}")
    else:
        log.info("no allele model pairs found")
    
    # --- plot 3: per-site SSU variability across individuals ---
    # uses df_all — deduplicate alleles, compute SD across individuals
    for tissue in tissues_diag:
        sub = df_all.query("tissue == @tissue & y_ssu != @SENTINEL").copy()
        # deduplicate: take first allele per site per individual
        sub = sub.drop_duplicates(subset=["chrom", "pos", "strand", "sample"], keep="first")
        site_sd = sub.groupby(["chrom", "pos", "strand"])["y_ssu"].agg(["std", "count"])
        site_sd = site_sd[site_sd["count"] >= 2]
        tname = tissue_display.get(tissue, tissue)
        log.info(f"\n{tname}: {len(site_sd):,} sites with >=2 individuals")
        log.info(f"  median SD={site_sd['std'].median():.4f}, "
              f"frac SD>0.01={((site_sd['std'] > 0.01).mean()):.3f}, "
              f"frac SD>0.05={((site_sd['std'] > 0.05).mean()):.3f}")
    
    # histogram
    fig, axes = plt.subplots(1, len(tissues_diag), figsize=(4 * len(tissues_diag), 4), sharey=True)
    for col, tissue in enumerate(tissues_diag):
        ax = axes[col]
        sub = df_all.query("tissue == @tissue & y_ssu != @SENTINEL").copy()
        sub = sub.drop_duplicates(subset=["chrom", "pos", "strand", "sample"], keep="first")
        site_sd = sub.groupby(["chrom", "pos", "strand"])["y_ssu"].std()
        site_sd = site_sd.dropna()
        ax.hist(site_sd.values, bins=100, color=tissue_colors.get(tissue, "gray"), alpha=0.7)
        ax.axvline(site_sd.median(), color="red", ls="--", lw=1.5, label=f"median={site_sd.median():.4f}")
        ax.set_xlabel("SD of y_ssu across individuals")
        ax.set_ylabel("n sites" if col == 0 else "")
        ax.set_title(tissue_display.get(tissue, tissue))
        ax.legend(fontsize=9)
        ax.set_yscale("log")
    fig.suptitle("per-site SSU variability across individuals (deduplicated by site)", y=1.02)
    plt.tight_layout()
    _save_fig(plt, fig2_sup / "ssu_variability_histogram")
    
    # --- plot 4: R² spread on shared vs valid ---
    fig, axes = plt.subplots(1, len(tissues_diag), figsize=(4 * len(tissues_diag), 4), sharey=True)
    for col, tissue in enumerate(tissues_diag):
        ax = axes[col]
        for i, (subset_name, subset_label, color) in enumerate([
            ("ssu_valid", "valid", "#56B4E9"),
            ("ssu_shared", "shared", "#D55E00"),
        ]):
            sub = df_reg.query("subset == @subset_name & output == 'splaire_ref_ssu' & tissue == @tissue")
            if len(sub) == 0: continue
            vals = sub["r2"].values
            ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.15, 0.15, len(vals)),
                       vals, s=30, alpha=0.7, color=color, label=subset_label)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color="black", lw=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["valid", "shared"])
        ax.set_ylabel("$R^2$ (SPLAIRE)" if col == 0 else "")
        ax.set_title(tissue_display.get(tissue, tissue))
        if col == 0: ax.legend(fontsize=9)
    fig.suptitle("R² spread: ssu_valid vs ssu_shared (same sites for all individuals)", y=1.02)
    plt.tight_layout()
    _save_fig(plt, fig2_sup / "r2_spread_valid_vs_shared")
    log.info("plot 4: R² spread comparison")
    
    # --- plot 5: slopegraph — paired valid→shared R² per individual, all models ---
    _slope_outputs = r2_outputs  # splaire_ref_ssu, spliceai_cls, pangolin_avg_usage, splicetransformer_avg_tissue
    _slope_display = {
        "splaire_ref_ssu": "SPLAIRE", "spliceai_cls": "SpliceAI",
        "pangolin_avg_usage": "Pangolin", "splicetransformer_avg_tissue": "SpliceTransformer",
    }
    
    n_out = len(_slope_outputs)
    fig, axes = plt.subplots(n_out, len(tissues_diag),
                             figsize=(3.5 * len(tissues_diag), 3.5 * n_out),
                             sharey="row")
    
    for ri, output in enumerate(_slope_outputs):
        for ci, tissue in enumerate(tissues_diag):
            ax = axes[ri, ci]
            valid = df_reg.query("subset == 'ssu_valid' & output == @output & tissue == @tissue")
            shared = df_reg.query("subset == 'ssu_shared' & output == @output & tissue == @tissue")
            if len(valid) == 0 or len(shared) == 0:
                ax.set_visible(False)
                continue
            merged = valid[["sample_id", "r2"]].merge(
                shared[["sample_id", "r2"]], on="sample_id", suffixes=("_valid", "_shared"))
            for _, row in merged.iterrows():
                ax.plot([0, 1], [row["r2_valid"], row["r2_shared"]],
                        "o-", color=GRAY_MED, alpha=0.5, markersize=4, lw=1)
            # median lines
            ax.hlines(merged["r2_valid"].median(), -0.15, 0.15, color="black", lw=2)
            ax.hlines(merged["r2_shared"].median(), 0.85, 1.15, color="black", lw=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["valid", "shared"])
            if ci == 0:
                ax.set_ylabel(f"{_slope_display[output]}\n$R^2$")
            if ri == 0:
                ax.set_title(tissue_display.get(tissue, tissue))
            ax.grid(axis="y", alpha=0.2)
    
    fig.suptitle("per-individual R² shift: valid sites → shared sites", y=1.01)
    plt.tight_layout()
    _save_fig(plt, fig2_sup / "r2_slopegraph_valid_shared")
    log.info("plot 5: R² slopegraph valid→shared")
    
    # --- plot 6: site dropout — valid vs shared site counts per individual ---
    fig, axes = plt.subplots(1, len(tissues_diag), figsize=(4 * len(tissues_diag), 4), sharey=True)
    for ci, tissue in enumerate(tissues_diag):
        ax = axes[ci]
        # use n from splaire_ref_ssu (same sites regardless of model)
        valid = (df_reg.query("subset == 'ssu_valid' & output == 'splaire_ref_ssu' & tissue == @tissue")
                 [["sample_id", "n"]].rename(columns={"n": "n_valid"}))
        shared = (df_reg.query("subset == 'ssu_shared' & output == 'splaire_ref_ssu' & tissue == @tissue")
                  [["sample_id", "n"]].rename(columns={"n": "n_shared"}))
        if len(valid) == 0 or len(shared) == 0:
            continue
        merged = valid.merge(shared, on="sample_id").sort_values("n_valid", ascending=False)
        x = np.arange(len(merged))
        w = 0.35
        ax.bar(x - w / 2, merged["n_valid"].values, w, label="valid", color="#56B4E9", alpha=0.8)
        ax.bar(x + w / 2, merged["n_shared"].values, w, label="shared", color="#D55E00", alpha=0.8)
        # dropout percentage annotation
        for xi, (_, row) in enumerate(merged.iterrows()):
            pct = (1 - row["n_shared"] / row["n_valid"]) * 100
            ax.text(xi, row["n_valid"] + 200, f"{pct:.0f}%", ha="center", fontsize=8, color=GRAY_MED)
        ax.set_xlabel("individual")
        ax.set_ylabel("n splice sites" if ci == 0 else "")
        ax.set_title(tissue_display.get(tissue, tissue))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i+1}" for i in x], fontsize=8)
        if ci == 0:
            ax.legend(fontsize=9)
    
    fig.suptitle("valid vs shared splice site counts per individual (% dropped)", y=1.02)
    plt.tight_layout()
    _save_fig(plt, fig2_sup / "site_dropout_valid_shared")
    log.info("plot 6: site dropout valid vs shared")
    
    # calibration data — computed from tissue_data (no separate parquet loading)
    
    n_cal_bins = 20
    cal_edges = np.linspace(0, 1, n_cal_bins + 1)
    cal_tissues = list(tissues_fig1)
    
    # output key -> (base model for color, display name)
    _cal_outputs = {
        "splaire_cls": ("splaire_ref", "SPLAIRE"),
        "splaire_reg": ("splaire_ref", "SPLAIRE"),
        "spliceai_cls": ("spliceai", "SpliceAI"),
        "pangolin_cls": ("pangolin", "Pangolin"),
        "pangolin_reg": ("pangolin", "Pangolin"),
        "spt_cls": ("splicetransformer", "SpliceTransformer"),
        "spt_reg": ("splicetransformer", "SpliceTransformer"),
    }
    
    def _cal_from_arrays(pred, true, edges, n_bins):
        bin_idx = np.clip(np.digitize(pred, edges) - 1, 0, n_bins - 1)
        pm, tm = [], []
        for b in range(n_bins):
            m = bin_idx == b
            n = m.sum()
            pm.append(float(pred[m].mean()) if n > 0 else np.nan)
            tm.append(float(true[m].mean()) if n > 0 else np.nan)
        return np.array(pm), np.array(tm)
    
    # compute calibration from tissue_data
    cal_cls_keys = ["spliceai_cls", "splaire_cls", "pangolin_cls", "spt_cls"]
    cal_reg_keys = ["splaire_reg", "pangolin_reg", "spt_reg"]
    
    cal_data = {}
    for tissue in cal_tissues:
        if tissue not in tissue_data:
            continue
        td = tissue_data[tissue]
        mask = td["masks"]["ssu_valid_nonzero"]
        y = td["y_ssu"][mask]
        cal_data[tissue] = {}
        for key in cal_cls_keys + cal_reg_keys:
            if key not in td["preds"]:
                continue
            p = td["preds"][key][mask]
            pm, tm = _cal_from_arrays(p, y, cal_edges, n_cal_bins)
            cal_data[tissue][key] = {"pred_mean": pm, "true_mean": tm}
    
    log.info("calibration computed from tissue_data")
    for tissue in cal_data:
        log.info(f"  {tissue}: {len(cal_data[tissue])} outputs")
    
    # calibration plots — uses cal_data from cell above
    
    nt = len(cal_tissues)
    
    fig, axes = plt.subplots(2, nt, figsize=(6 * nt, 12), sharey=True, sharex=True)
    
    # top row: classification
    for ti, tissue in enumerate(cal_tissues):
        ax = axes[0, ti]
        td = cal_data.get(tissue, {})
        for key in cal_cls_keys:
            if key not in td: continue
            pm, tm = td[key]["pred_mean"], td[key]["true_mean"]
            valid = np.isfinite(pm) & np.isfinite(tm)
            base, display = _cal_outputs[key]
            ax.plot(pm[valid], tm[valid], "o-", markersize=6, color=get_color(base),
                    lw=2.5, label=display)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        if ti == 0: ax.set_ylabel("Actual SSU", fontsize=14)
        ax.set_title(tissue_display.get(tissue, tissue), fontsize=15, fontweight="bold")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    
    fig.text(0.02, 0.73, "Classification", va="center", ha="center",
             fontsize=14, fontweight="bold", rotation=90)
    
    # bottom row: regression
    for ti, tissue in enumerate(cal_tissues):
        ax = axes[1, ti]
        td = cal_data.get(tissue, {})
        for key in cal_reg_keys:
            if key not in td: continue
            pm, tm = td[key]["pred_mean"], td[key]["true_mean"]
            valid = np.isfinite(pm) & np.isfinite(tm)
            base, display = _cal_outputs[key]
            ax.plot(pm[valid], tm[valid], "o-", markersize=6, color=get_color(base),
                    lw=2.5, label=display)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("Predicted SSU", fontsize=14)
        if ti == 0: ax.set_ylabel("Actual SSU", fontsize=14)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    
    fig.text(0.02, 0.28, "Regression", va="center", ha="center",
             fontsize=14, fontweight="bold", rotation=90)
    
    fig.suptitle("SSU Calibration", fontsize=17, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(plt, fig2_main / "calibration_combined")
    log.info("calibration: combined cls + reg")
    
    # heatmaps showing ALL outputs (not just best per base model)
    # red solid = best overall per tissue, orange dashed = best per base model family
    # includes canonical benchmarks
    
    lower_better = {"mse", "mae"}
    
    # tissue lists for heatmaps
    _heatmap_tissues = list(tissues_fig1)
    _heatmap_tissues_cls = _heatmap_tissues + [t for t in gencode_tissues if t in df_cls.tissue.unique()]
    
    _output_display = {
        "splaire_ref": "SPLAIRE CLS",
        "splaire_ref_ssu": "SPLAIRE SSU",
        "spliceai": "SpliceAI CLS",
        "splicetransformer": "SpliceTransformer CLS",
        "splicetransformer_avg_tissue": "SpliceTransformer Avg Tissue",
    }
    
    def _output_name(code):
        """pretty name for model output"""
        if code in _output_display:
            return _output_display[code]
        if code.startswith("pangolin_"):
            rest = code.replace("pangolin_", "")
            if rest.endswith("_p_splice"):
                tissue = rest.replace("_p_splice", "").replace("_", " ").title()
                return f"Pangolin {tissue} CLS"
            if rest.endswith("_usage"):
                tissue = rest.replace("_usage", "").replace("_", " ").title()
                return f"Pangolin {tissue} Usage"
            return f"Pangolin {rest.replace('_', ' ').title()}"
        if code.startswith("splicetransformer_"):
            tissue = code.replace("splicetransformer_", "").replace("_", " ").title()
            return f"SpliceTransformer {tissue}"
        return code
    
    
    def plot_heatmap_all_outputs(data, metric, title_suffix="", fname_suffix="", out_dir=None, tissue_list=None):
        """heatmap with all outputs, highlighting best per base and best overall"""
        if tissue_list is None:
            tissue_list = all_tissues
        
        sub = data[data.metric == metric]
        if len(sub) == 0:
            log.info(f"no data for {metric}")
            return
        
        pivot = sub.pivot(index="output", columns="tissue", values="score")
        pivot = pivot[[t for t in tissue_list if t in pivot.columns]]
        
        # sort by mean performance (best at top)
        ascending = metric in lower_better
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=ascending).index]
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.4)))
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([tissue_display.get(t, t) for t in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([_output_name(o) for o in pivot.index])
        
        # add vertical line to separate expression vs gencode
        n_expr = len([t for t in pivot.columns if t in set(tissues_fig1)])
        if n_expr < len(pivot.columns):
            ax.axvline(x=n_expr - 0.5, color="white", lw=2, ls="--")
        
        # build output -> base model mapping
        output_to_base = {out: _get_base_model(out) for out in pivot.index}
        base_models = set(output_to_base.values()) - {None}
        
        for j in range(len(pivot.columns)):
            col_vals = pivot.iloc[:, j]
            
            # best per base model (orange dashed)
            for base in base_models:
                base_outputs = [out for out, b in output_to_base.items() if b == base]
                base_vals = col_vals.loc[base_outputs].dropna()
                if len(base_vals) == 0: continue
                best_out = base_vals.idxmin() if metric in lower_better else base_vals.idxmax()
                i = list(pivot.index).index(best_out)
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="orange", lw=1.5, ls="--")
                ax.add_patch(rect)
            
            # best overall (red solid)
            col_vals_clean = col_vals.dropna()
            if len(col_vals_clean) == 0: continue
            best_i = col_vals_clean.idxmin() if metric in lower_better else col_vals_clean.idxmax()
            i = list(pivot.index).index(best_i)
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5)
            ax.add_patch(rect)
        
        # add values
        vmin, vmax = pivot.values.min(), pivot.values.max()
        mid = (vmin + vmax) / 2
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.iloc[i, j]
                color = "white" if v > mid else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=ANNOT_SIZE)
        
        plt.colorbar(im, ax=ax, shrink=0.5)
        ax.set_title(f"{metric.upper()}{title_suffix}")
        
        # legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="none", edgecolor="red", lw=2.5, label="best overall"),
            Patch(facecolor="none", edgecolor="orange", lw=1.5, ls="--", label="best per model"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        if out_dir:
            _save_fig(fig, out_dir / f"heatmap_{metric}{fname_suffix}")
        else:
            plt.close("all")
    
    
    # --- classification heatmaps ---
    if len(df_cls) > 0:
        log.info("=== classification (all outputs, including gencode) ===")
        sub = df_cls[df_cls.subset == "all"].copy()
        sub["base_model"] = sub.output.apply(_get_base_model)
        sub = sub[~sub.base_model.isin(exclude_bases)]
        
        # average across acceptor/donor and samples
        agg = sub.groupby(["tissue", "output"]).agg(
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            topk=("topk", "mean"),
        ).reset_index()
        
        # melt to long format for plotting function
        cls_long = agg.melt(id_vars=["tissue", "output"], var_name="metric", value_name="score")
        
        for m in ["auprc", "auroc", "topk"]:
            plot_heatmap_all_outputs(cls_long, m, tissue_list=_heatmap_tissues_cls, out_dir=fig2_sup)
    else:
        log.info("no classification data")
    
    # --- regression heatmaps (all subsets) - expression tissues only ---
    subset_labels = {
        "ssu_valid": "all sites",
        "ssu_valid_nonzero": "SSU > 0",
        "ssu_shared": "shared sites",
        "ssu_shared_nonzero": "shared, SSU > 0",
    }
    
    if len(df_reg) > 0:
        for subset, subset_label in subset_labels.items():
            log.info(f"\n=== regression - {subset_label} (all outputs, expression only) ===")
            
            sub = df_reg[df_reg.subset == subset].copy()
            sub["base_model"] = sub.output.apply(_get_base_model)
            sub = sub[~sub.base_model.isin(exclude_bases)]
            
            if len(sub) == 0:
                log.info(f"no data for {subset}")
                continue
            
            # average across samples
            agg_cols = ["pearson", "spearman", "r2", "mse"]
            if "mae" in sub.columns:
                agg_cols.append("mae")
            
            agg = sub.groupby(["tissue", "output"]).agg(
                {col: "mean" for col in agg_cols}
            ).reset_index()
            
            # melt to long format
            reg_long = agg.melt(id_vars=["tissue", "output"], var_name="metric", value_name="score")
            
            title_suffix = f" ({subset_label})"
            fname_suffix = f"_{subset}"
            
            for m in agg_cols:
                plot_heatmap_all_outputs(reg_long, m, title_suffix=title_suffix, fname_suffix=fname_suffix, tissue_list=_heatmap_tissues, out_dir=fig2_sup)
    else:
        log.info("no regression data")
    
    # prediction error vs tissue variability — data processing
    # average within site first so each site is one observation
    
    _model_display = {
        "splaire_ref": "SPLAIRE", "spliceai": "SpliceAI",
        "pangolin": "Pangolin", "splicetransformer": "SpliceTransformer",
    }
    _model_order = ["splaire_ref", "spliceai", "pangolin", "splicetransformer"]
    
    # collapse to one value per site per model: median absolute error across tissues and samples
    df_site = (df_resid.groupby(["chrom", "pos", "strand", "model"])
               .agg(abs_resid=("abs_resid", "median"),
                    ssu_tissue_std=("ssu_tissue_std", "first"))
               .reset_index())
    
    log.info(f"site-level table: {len(df_site):,} rows ({len(df_site) // len(_model_order):,} sites x {len(_model_order)} models)")
    
    # quantile bins on unique sites
    n_qbins = 8
    df_site["sd_qbin"] = pd.qcut(df_site["ssu_tissue_std"], q=n_qbins, duplicates="drop")
    
    bin_stats = (df_site.groupby("sd_qbin", observed=True)["ssu_tissue_std"]
                 .agg(["mean", "min", "max", "count"]))
    n_per_bin = bin_stats["count"].values[0] // len(_model_order)
    
    # summary stats
    median_sd = df_site["ssu_tissue_std"].median()
    df_site["var_group"] = np.where(df_site["ssu_tissue_std"] <= median_sd, "constitutive", "variable")
    
    log.info(f"quantile bins: {n_qbins}, ~{n_per_bin:,} unique sites per bin")
    log.info(f"SD range per bin:")
    for iv, row in bin_stats.iterrows():
        log.info(f"  {iv}: mean SD={row['mean']:.4f}, n={int(row['count'] // len(_model_order)):,} sites")
    
    log.info(f"\nmedian SD split: {median_sd:.4f}")
    log.info(f"\n=== median absolute error by group (site-level) ===")
    summary = (df_site.groupby(["var_group", "model"])["abs_resid"]
               .agg(["mean", "median", "count"])
               .round(4))
    log.info("\n%s", summary)
    
    log.info("\n=== SPLAIRE improvement (median absolute error, site-level) ===")
    for group in ["constitutive", "variable"]:
        sp_med = df_site.query("model == 'splaire_ref' and var_group == @group")["abs_resid"].median()
        log.info(f"\n{group} (SD {'<=' if group == 'constitutive' else '>'} {median_sd:.4f}):")
        log.info(f"  SPLAIRE median |resid|: {sp_med:.4f}")
        for comp in ["spliceai", "pangolin", "splicetransformer"]:
            comp_med = df_site.query("model == @comp and var_group == @group")["abs_resid"].median()
            pct = (comp_med - sp_med) / comp_med * 100
            log.info(f"  vs {_model_display[comp]:20s}: {comp_med:.4f} -> {sp_med:.4f} ({pct:+.1f}%)")
    
    # metrics vs tissue variability — R², Pearson r, MSE, MAE
    # split into regression outputs (left) and classification outputs (right)
    # computes metrics per tissue×individual within each SD bin, then averages
    
    from scipy.stats import pearsonr as _pearsonr
    
    _reg_models = ["splaire_ref", "pangolin", "splicetransformer"]
    _cls_models = ["splaire_ref", "spliceai", "pangolin", "splicetransformer"]
    _model_display = {
        "splaire_ref": "SPLAIRE", "spliceai": "SpliceAI",
        "pangolin": "Pangolin", "splicetransformer": "SpliceTransformer",
    }
    
    _pred_cols = {
        "splaire_ref": "pred_splaire_ref",
        "spliceai": "pred_spliceai",
        "pangolin": "pred_pangolin",
        "splicetransformer": "pred_splicetransformer",
    }
    _reg_pred_cols = {
        "splaire_ref": "pred_splaire_ref_reg",
        "pangolin": "pred_pangolin_reg",
        "splicetransformer": "pred_splicetransformer_reg",
    }
    _cls_pred_cols = {
        "splaire_ref": "pred_splaire_ref_cls",
        "spliceai": "pred_spliceai_cls",
        "pangolin": "pred_pangolin_cls",
        "splicetransformer": "pred_splicetransformer_cls",
    }
    
    # add sd_qbin to df_eval
    site_bins = df_site[["chrom", "pos", "strand", "sd_qbin"]].drop_duplicates(
        subset=["chrom", "pos", "strand"])
    df_eval_binned = df_eval.merge(site_bins, on=["chrom", "pos", "strand"], how="inner")
    log.info(f"binned evaluation table: {len(df_eval_binned):,} rows")
    
    bin_labels_ordered = sorted(df_eval_binned["sd_qbin"].unique())
    n_qbins = len(bin_labels_ordered)
    bin_centers = np.arange(n_qbins)
    bin_xlabels = [f"{iv.left:.3f}-{iv.right:.3f}" for iv in bin_labels_ordered]
    
    # compute metrics per bin — average across tissue×individual combinations
    def _compute_binned_metrics(df, models, pred_col_map):
        results = {metric: {m: [] for m in models}
                   for metric in ["r2", "pearson", "mse", "mae"]}
        for qbin in bin_labels_ordered:
            sub = df.query("sd_qbin == @qbin")
            for model in models:
                col = pred_col_map[model]
                if col not in sub.columns:
                    for metric in results:
                        results[metric][model].append(np.nan)
                    continue
    
                # compute per tissue×individual, then average
                r2_vals, pearson_vals, mse_vals, mae_vals = [], [], [], []
                for (tissue, sample), grp in sub.groupby(["tissue", "sample"]):
                    y = grp["y_ssu"].values
                    p = grp[col].values
                    valid = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
                    yv, pv = y[valid], p[valid]
                    if len(yv) < 10:
                        continue
                    ss_tot = np.sum((yv - yv.mean())**2)
                    if ss_tot > 0:
                        r2_vals.append(1 - np.sum((yv - pv)**2) / ss_tot)
                    r, _ = _pearsonr(pv, yv)
                    pearson_vals.append(r)
                    mse_vals.append(np.mean((yv - pv)**2))
                    mae_vals.append(np.median(np.abs(yv - pv)))
    
                results["r2"][model].append(np.mean(r2_vals) if r2_vals else np.nan)
                results["pearson"][model].append(np.mean(pearson_vals) if pearson_vals else np.nan)
                results["mse"][model].append(np.mean(mse_vals) if mse_vals else np.nan)
                results["mae"][model].append(np.mean(mae_vals) if mae_vals else np.nan)
        return results
    
    reg_metrics = _compute_binned_metrics(df_eval_binned, _reg_models, _pred_cols)
    cls_metrics = _compute_binned_metrics(df_eval_binned, _cls_models, _pred_cols)
    
    # combined plot: 4 metrics × 2 columns (regression | classification)
    _metric_labels = [
        ("r2", "$R^2$"),
        ("pearson", "Pearson r"),
        ("mse", "MSE"),
        ("mae", "Median Absolute Error"),
    ]
    
    fig, axes = plt.subplots(len(_metric_labels), 2, figsize=(14, 4 * len(_metric_labels)),
                             sharex=True)
    
    for row_i, (metric, ylabel) in enumerate(_metric_labels):
        ax_reg, ax_cls = axes[row_i, 0], axes[row_i, 1]
    
        for model in _reg_models:
            vals = np.array(reg_metrics[metric][model])
            lw = 2.5 if model == "splaire_ref" else 1.8
            ms = 6 if model == "splaire_ref" else 4
            ax_reg.plot(bin_centers, vals, "o-", color=get_color(model),
                        label=_model_display[model], linewidth=lw, markersize=ms)
        ax_reg.set_ylabel(ylabel)
        if row_i == 0:
            ax_reg.set_title("Regression Outputs")
        ax_reg.legend(frameon=True, edgecolor="lightgray", fontsize=9)
        ax_reg.grid(alpha=0.2)
        ax_reg.set_xticks(bin_centers)
        if row_i == len(_metric_labels) - 1:
            ax_reg.set_xticklabels(bin_xlabels, rotation=45, ha="right")
            ax_reg.set_xlabel("cross-tissue SD of mean SSU")
    
        for model in _cls_models:
            vals = np.array(cls_metrics[metric][model])
            lw = 2.5 if model == "splaire_ref" else 1.8
            ms = 6 if model == "splaire_ref" else 4
            ax_cls.plot(bin_centers, vals, "o-", color=get_color(model),
                        label=_model_display[model], linewidth=lw, markersize=ms)
        if row_i == 0:
            ax_cls.set_title("Classification Outputs")
        ax_cls.legend(frameon=True, edgecolor="lightgray", fontsize=9)
        ax_cls.grid(alpha=0.2)
        ax_cls.set_xticks(bin_centers)
        if row_i == len(_metric_labels) - 1:
            ax_cls.set_xticklabels(bin_xlabels, rotation=45, ha="right")
            ax_cls.set_xlabel("cross-tissue SD of mean SSU")
    
    fig.suptitle("prediction metrics vs tissue variability", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(plt, fig2_main / "tissue_variability_combined")
    log.info("tissue variability combined: done\n")
    

# --- tissue specificity analysis ---
log.info("\n" + "=" * 60)
log.info("tissue specificity analysis")

# deduplicate alleles (splice_sites.parquet already excludes 777 sentinels)
_df_ts = df_all.query("y_ssu != @SENTINEL").copy()
_df_ts = _df_ts.drop_duplicates(
    subset=["chrom", "pos", "strand", "tissue", "sample"], keep="first")

# per site x tissue summary: n_detected, mean, median, sd, min, max
_site_tissue_all = (_df_ts.groupby(["chrom", "pos", "strand", "tissue"])["y_ssu"]
                    .agg(n_detected="count", mean_ssu="mean", median_ssu="median",
                         sd_ssu="std", min_ssu="min", max_ssu="max")
                    .reset_index())
_site_tissue_all["sd_ssu"] = _site_tissue_all["sd_ssu"].fillna(0)

# how many tissues each site appears in
_nt_per_site = (_site_tissue_all.groupby(["chrom", "pos", "strand"])["tissue"]
                .nunique().reset_index(name="n_tissues_detected"))
_site_tissue_all = _site_tissue_all.merge(_nt_per_site, on=["chrom", "pos", "strand"])

_n_total_sites = len(_nt_per_site)
log.info(f"total unique sites: {_n_total_sites:,}")
for nt in range(1, len(tissues_fig1) + 1):
    n = (_nt_per_site.n_tissues_detected == nt).sum()
    log.info(f"  in {nt} tissues: {n:,}")

# save full per-site × per-tissue summary (all sites, not just 5-tissue)
_site_tissue_all.to_csv(results_dir / "site_tissue_summary.csv", index=False)
log.info(f"saved site_tissue_summary.csv: {len(_site_tissue_all):,} rows")

# =====================================================================
# DIAGNOSTIC PLOTS — data quality and detection patterns
# =====================================================================

# --- figure: n_detected per tissue (how many individuals per site) ---
fig, axes = plt.subplots(1, len(tissues_fig1), figsize=(3.5 * len(tissues_fig1), 4), sharey=True)
for ci, tissue in enumerate(tissues_fig1):
    ax = axes[ci]
    sub = _site_tissue_all[_site_tissue_all.tissue == tissue]
    ax.hist(sub.n_detected.values, bins=range(1, 13), color=tissue_colors.get(tissue, "#999"),
            edgecolor="white", alpha=0.8, align="left")
    ax.set_xlabel("n individuals detected")
    if ci == 0: ax.set_ylabel("n splice sites")
    ax.set_title(f"{tissue_display.get(tissue, tissue)} ({len(sub):,} sites)")
    ax.set_xticks(range(1, 11))
plt.suptitle("individuals per site per tissue", fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_sup / "n_detected_per_tissue")
log.info("n_detected per tissue: done")

# --- figure: mean SSU distribution per tissue ---
fig, axes = plt.subplots(1, len(tissues_fig1), figsize=(3.5 * len(tissues_fig1), 4), sharey=True)
for ci, tissue in enumerate(tissues_fig1):
    ax = axes[ci]
    sub = _site_tissue_all[_site_tissue_all.tissue == tissue]
    ax.hist(sub.mean_ssu.values, bins=50, color=tissue_colors.get(tissue, "#999"),
            edgecolor="white", alpha=0.8)
    med = sub.mean_ssu.median()
    ax.axvline(med, color="red", ls="--", lw=1.5, label=f"median={med:.3f}")
    ax.set_xlabel("mean SSU")
    if ci == 0: ax.set_ylabel("n splice sites")
    ax.set_title(tissue_display.get(tissue, tissue))
    ax.legend(fontsize=8)
plt.suptitle("mean SSU distribution per tissue (all sites)", fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_sup / "mean_ssu_per_tissue")
log.info("mean SSU per tissue: done")

# --- figure: median SSU distribution per tissue ---
fig, axes = plt.subplots(1, len(tissues_fig1), figsize=(3.5 * len(tissues_fig1), 4), sharey=True)
for ci, tissue in enumerate(tissues_fig1):
    ax = axes[ci]
    sub = _site_tissue_all[_site_tissue_all.tissue == tissue]
    ax.hist(sub.median_ssu.values, bins=50, color=tissue_colors.get(tissue, "#999"),
            edgecolor="white", alpha=0.8)
    med = sub.median_ssu.median()
    ax.axvline(med, color="red", ls="--", lw=1.5, label=f"median={med:.3f}")
    ax.set_xlabel("median SSU")
    if ci == 0: ax.set_ylabel("n splice sites")
    ax.set_title(tissue_display.get(tissue, tissue))
    ax.legend(fontsize=8)
plt.suptitle("median SSU distribution per tissue (all sites)", fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_sup / "median_ssu_per_tissue")
log.info("median SSU per tissue: done")

# --- figure: n_tissues_detected histogram ---
fig, ax = plt.subplots(figsize=(8, 4))
_ntd_counts = _nt_per_site.n_tissues_detected.value_counts().sort_index()
bars = ax.bar(_ntd_counts.index, _ntd_counts.values, color="#56B4E9", edgecolor="white")
for bar, val in zip(bars, _ntd_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("n tissues with at least 1 valid individual")
ax.set_ylabel("n splice sites")
ax.set_title(f"tissue detection breadth ({_n_total_sites:,} total sites)")
ax.set_xticks(range(1, len(tissues_fig1) + 1))
plt.tight_layout()
_save_fig(plt, fig2_main / "n_tissues_detected")
log.info("n_tissues_detected: done")

# =====================================================================
# APPROACH D: n_tissues_detected as tissue specificity metric
# prediction error stratified by how many tissues a site appears in
# =====================================================================
log.info("\n--- approach D: n_tissues_detected ---")

# merge n_tissues_detected with df_all (not df_eval — df_eval is filtered to 5-tissue sites)
# df_all has predictions for all sites in all tissues they appear in
df_eval_ntd = df_all.merge(_nt_per_site, on=["chrom", "pos", "strand"], how="inner")
log.info(f"evaluation rows with n_tissues_detected: {len(df_eval_ntd):,}")
log.info(f"  unique sites: {df_eval_ntd.groupby(['chrom','pos','strand']).ngroups:,}")
log.info(f"  n_tissues_detected distribution: {dict(df_eval_ntd.drop_duplicates(['chrom','pos','strand']).n_tissues_detected.value_counts().sort_index())}")

# --- figure: prediction error vs n_tissues_detected ---
_ntd_met_labels = [
    ("r2", "$R^2$"), ("pearson", "Pearson r"),
    ("mse", "MSE"), ("mae", "Median Absolute Error"),
]
_ntd_vals = sorted(df_eval_ntd["n_tissues_detected"].unique())

def _compute_ntd_metrics(df, models, pcols):
    """pooled metrics per n_tissues_detected value"""
    res = {m: {mo: [] for mo in models} for m in ["r2", "pearson", "mse", "mae"]}
    for ntd in _ntd_vals:
        sub = df[df["n_tissues_detected"] == ntd]
        for mo in models:
            col = pcols[mo]
            y = sub["y_ssu"].values
            p = sub[col].values
            ok = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
            yv, pv = y[ok], p[ok]
            if len(yv) < 10:
                for m in res: res[m][mo].append(np.nan)
                continue
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            res["r2"][mo].append(
                1 - np.sum((yv - pv) ** 2) / ss_tot if ss_tot > 0 else np.nan)
            res["pearson"][mo].append(
                _pearsonr(pv, yv)[0] if ss_tot > 0 else np.nan)
            res["mse"][mo].append(float(np.mean((yv - pv) ** 2)))
            res["mae"][mo].append(float(np.median(np.abs(yv - pv))))
    return res

_ntd_reg = _compute_ntd_metrics(df_eval_ntd, _reg_models, _reg_pred_cols)
_ntd_cls = _compute_ntd_metrics(df_eval_ntd, _cls_models, _cls_pred_cols)

_ntd_site_counts = {ntd: (_nt_per_site.n_tissues_detected == ntd).sum() for ntd in _ntd_vals}
_ntd_xlabels = [f"{ntd}\n(n={_ntd_site_counts[ntd]:,})" for ntd in _ntd_vals]

_n_met_rows = len(_ntd_met_labels)
from matplotlib.gridspec import GridSpec as _GS_ntd
fig = plt.figure(figsize=(12, 4 * _n_met_rows + 3))
gs = _GS_ntd(_n_met_rows + 1, 2, figure=fig,
             height_ratios=[1] * _n_met_rows + [0.6], hspace=0.3)

for row_i, (met, ylabel) in enumerate(_ntd_met_labels):
    ax_reg = fig.add_subplot(gs[row_i, 0])
    ax_cls = fig.add_subplot(gs[row_i, 1])
    for mo in _reg_models:
        vals = np.array(_ntd_reg[met][mo])
        lw = 2.5 if mo == "splaire_ref" else 1.8
        ms = 8 if mo == "splaire_ref" else 6
        ax_reg.plot(_ntd_vals, vals, "o-", color=get_color(mo),
                    label=_model_display[mo], lw=lw, markersize=ms)
    ax_reg.set_ylabel(ylabel)
    if row_i == 0: ax_reg.set_title("Regression Outputs")
    ax_reg.legend(frameon=True, edgecolor="lightgray", fontsize=9)
    ax_reg.grid(alpha=0.2)
    ax_reg.set_xticks(_ntd_vals)
    ax_reg.set_xticklabels([])

    for mo in _cls_models:
        vals = np.array(_ntd_cls[met][mo])
        lw = 2.5 if mo == "splaire_ref" else 1.8
        ms = 8 if mo == "splaire_ref" else 6
        ax_cls.plot(_ntd_vals, vals, "o-", color=get_color(mo),
                    label=_model_display[mo], lw=lw, markersize=ms)
    if row_i == 0: ax_cls.set_title("Classification Outputs")
    ax_cls.legend(frameon=True, edgecolor="lightgray", fontsize=9)
    ax_cls.grid(alpha=0.2)
    ax_cls.set_xticks(_ntd_vals)
    ax_cls.set_xticklabels([])

# bottom row: SSU violin (shared across both columns)
ax_vio = fig.add_subplot(gs[_n_met_rows, :])
_vdata_ntd_inline = []
for ntd in _ntd_vals:
    _sites_ntd = _site_tissue_all[_site_tissue_all.n_tissues_detected == ntd]
    _vdata_ntd_inline.append(_sites_ntd.mean_ssu.values)
parts = ax_vio.violinplot(_vdata_ntd_inline, positions=_ntd_vals,
                           showmedians=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("#56B4E9")
    pc.set_alpha(0.7)
parts["cmedians"].set_color("red")
ax_vio.set_xticks(_ntd_vals)
ax_vio.set_xticklabels(_ntd_xlabels)
ax_vio.set_xlabel("n tissues detected")
ax_vio.set_ylabel("mean SSU")
ax_vio.set_title("SSU distribution", fontsize=11)

fig.suptitle(f"prediction metrics by tissue detection breadth ({_n_total_sites:,} sites)",
             fontsize=14, y=1.01)
plt.tight_layout()
_save_fig(plt, fig2_main / "ntd_metrics")
log.info("n_tissues_detected metrics: done")

# --- figure: SSU distribution per n_tissues_detected (violin) ---
fig, ax = plt.subplots(figsize=(10, 5))
_vdata_ntd = []
for ntd in _ntd_vals:
    _sites_ntd = _site_tissue_all[_site_tissue_all.n_tissues_detected == ntd]
    _vdata_ntd.append(_sites_ntd.mean_ssu.values)
parts = ax.violinplot(_vdata_ntd, positions=_ntd_vals, showmedians=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor("#56B4E9")
    pc.set_alpha(0.7)
parts["cmedians"].set_color("red")
ax.set_xticks(_ntd_vals)
ax.set_xticklabels(_ntd_xlabels)
ax.set_xlabel("n tissues detected")
ax.set_ylabel("mean SSU (per site × tissue)")
ax.set_title("SSU distribution by tissue detection breadth")
plt.tight_layout()
_save_fig(plt, fig2_main / "ntd_ssu_violin")
log.info("n_tissues_detected SSU violin: done")

# --- figure: which tissues contribute single-tissue sites? ---
_single = _site_tissue_all[_site_tissue_all.n_tissues_detected == 1]
_single_tissue_counts = _single.tissue.value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
_st_order = [t for t in tissues_fig1 if t in _single_tissue_counts.index]
_st_vals = [_single_tissue_counts.get(t, 0) for t in _st_order]
_st_colors = [tissue_colors.get(t, "#999") for t in _st_order]
_st_names = [tissue_display.get(t, t) for t in _st_order]
bars = ax.bar(_st_names, _st_vals, color=_st_colors, edgecolor="white")
for bar, val in zip(bars, _st_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("n single-tissue sites")
ax.set_title(f"tissue origin of single-tissue sites ({len(_single):,} total)")
plt.tight_layout()
_save_fig(plt, fig2_main / "single_tissue_origin")
log.info("single tissue origin: done")

# save n_tissues_detected metrics csv
_ntd_rows = []
for ni, ntd in enumerate(_ntd_vals):
    for mo in _reg_models + _cls_models:
        _src = _ntd_reg if mo in _reg_models else _ntd_cls
        _ntd_rows.append({
            "n_tissues_detected": ntd, "model": _model_display[mo],
            "output_type": "regression" if mo in _reg_models else "classification",
            "r2": _src["r2"][mo][ni], "pearson": _src["pearson"][mo][ni],
            "mse": _src["mse"][mo][ni], "mae": _src["mae"][mo][ni],
        })
pd.DataFrame(_ntd_rows).to_csv(results_dir / "ntd_metrics.csv", index=False)
log.info("saved ntd_metrics.csv")

# =====================================================================
# APPROACH C: tau stratified by n_tissues_detected
# compute tau only over the tissues where each site is detected
# =====================================================================
log.info("\n--- approach C: tau stratified by n_tissues_detected ---")

# for each n_tissues group, compute tau from the detected tissues only
_df_tau_all = []
for ntd in range(2, len(tissues_fig1) + 1):  # tau needs >=2 tissues
    _sites_grp = _nt_per_site[_nt_per_site.n_tissues_detected == ntd]
    if len(_sites_grp) == 0:
        continue
    _st_grp = _site_tissue_all.merge(
        _sites_grp[["chrom", "pos", "strand"]], on=["chrom", "pos", "strand"])
    # pivot: one row per site, only detected tissues as columns
    _pv = _st_grp.pivot_table(
        index=["chrom", "pos", "strand"], columns="tissue", values="mean_ssu")
    # tau over detected tissues only (no NaN fill — use only observed values)
    _vals = _pv.values  # (n_sites, n_tissues) with NaN for missing
    _maxs = np.nanmax(_vals, axis=1, keepdims=True)
    _maxs_safe = np.where(_maxs > 0, _maxs, 1)
    _xhat = _vals / _maxs_safe
    _tau = np.nansum(1 - _xhat, axis=1) / (ntd - 1)
    _tau[_maxs.ravel() == 0] = 0

    _df_grp = _pv.reset_index()[["chrom", "pos", "strand"]].copy()
    _df_grp["tau"] = _tau
    _df_grp["n_tissues_detected"] = ntd
    _df_grp["max_ssu"] = _maxs.ravel()
    _df_grp["mean_ssu"] = np.nanmean(_vals, axis=1)
    _df_tau_all.append(_df_grp)
    log.info(f"  ntd={ntd}: {len(_df_grp):,} sites, tau median={np.median(_tau):.3f}")

_df_tau_strat = pd.concat(_df_tau_all, ignore_index=True)
_df_tau_strat.to_csv(results_dir / "site_tau_stratified.csv", index=False)
log.info(f"saved site_tau_stratified.csv: {len(_df_tau_strat):,} sites")

# --- figure: tau distribution per n_tissues_detected stratum ---
_ntd_for_tau = sorted(_df_tau_strat.n_tissues_detected.unique())
fig, axes = plt.subplots(1, len(_ntd_for_tau), figsize=(4.5 * len(_ntd_for_tau), 4), sharey=True)
if len(_ntd_for_tau) == 1: axes = [axes]
for ci, ntd in enumerate(_ntd_for_tau):
    ax = axes[ci]
    sub = _df_tau_strat[_df_tau_strat.n_tissues_detected == ntd]
    ax.hist(sub.tau.values, bins=30, color="#56B4E9", edgecolor="white", alpha=0.8)
    med = sub.tau.median()
    ax.axvline(med, color="red", ls="--", lw=1.5, label=f"median={med:.2f}")
    ax.set_xlabel("tau")
    if ci == 0: ax.set_ylabel("n splice sites")
    ax.set_title(f"{ntd} tissues (n={len(sub):,})")
    ax.legend(fontsize=8)
plt.suptitle("tau distribution by n tissues detected", fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_main / "tau_by_ntd_histograms")
log.info("tau by ntd histograms: done")

# --- figure: tau vs mean SSU per stratum ---
fig, axes = plt.subplots(1, len(_ntd_for_tau), figsize=(4.5 * len(_ntd_for_tau), 4), sharey=True)
if len(_ntd_for_tau) == 1: axes = [axes]
_rng_strat = np.random.default_rng(42)
for ci, ntd in enumerate(_ntd_for_tau):
    ax = axes[ci]
    sub = _df_tau_strat[_df_tau_strat.n_tissues_detected == ntd]
    _ns = min(3000, len(sub))
    _si = _rng_strat.choice(len(sub), _ns, replace=False)
    ax.scatter(sub.tau.values[_si], sub.mean_ssu.values[_si],
               s=3, alpha=0.3, color="#D55E00")
    ax.set_xlabel("tau")
    if ci == 0: ax.set_ylabel("mean SSU (detected tissues)")
    ax.set_title(f"{ntd} tissues (n={len(sub):,})")
plt.suptitle("tau vs mean SSU by n tissues detected", fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_main / "tau_vs_ssu_by_ntd")
log.info("tau vs SSU by ntd: done")

# --- figure: prediction error vs tau within each stratum ---
# merge stratified tau with df_eval
df_eval_strat = df_eval.merge(
    _df_tau_strat[["chrom", "pos", "strand", "tau", "n_tissues_detected"]],
    on=["chrom", "pos", "strand"], how="inner")
log.info(f"stratified evaluation rows: {len(df_eval_strat):,}")

# for each stratum with enough sites, bin tau and compute metrics
fig, axes = plt.subplots(2, len(_ntd_for_tau),
                         figsize=(5 * len(_ntd_for_tau), 8), squeeze=False)

for ci, ntd in enumerate(_ntd_for_tau):
    sub_eval = df_eval_strat[df_eval_strat.n_tissues_detected == ntd]
    sub_sites = _df_tau_strat[_df_tau_strat.n_tissues_detected == ntd]
    if len(sub_sites) < 50:
        for ri in range(2): axes[ri, ci].set_visible(False)
        continue

    # bin tau within this stratum
    _edges = np.quantile(sub_sites.tau.values, np.linspace(0, 1, 7))
    _edges = np.unique(np.round(_edges, 3))
    if len(_edges) < 3:
        for ri in range(2): axes[ri, ci].set_visible(False)
        continue
    sub_eval = sub_eval.copy()
    sub_eval["_tbin"] = pd.cut(sub_eval.tau, bins=_edges, include_lowest=True, duplicates="drop")
    _bins_s = sorted(sub_eval["_tbin"].dropna().unique())
    _bc_s = np.arange(len(_bins_s))
    _bl_s = [f"{b.left:.2f}-{b.right:.2f}" for b in _bins_s]

    for ri, (met, ylabel, pcols_m, models) in enumerate([
        ("mae", "Median AE", _reg_pred_cols, _reg_models),
        ("mae", "Median AE", _cls_pred_cols, _cls_models),
    ]):
        ax = axes[ri, ci]
        for mo in models:
            col = pcols_m[mo]
            vals = []
            for b in _bins_s:
                bsub = sub_eval[sub_eval["_tbin"] == b]
                y = bsub["y_ssu"].values
                p = bsub[col].values
                ok = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
                yv, pv = y[ok], p[ok]
                vals.append(float(np.median(np.abs(yv - pv))) if len(yv) >= 10 else np.nan)
            lw = 2.5 if mo == "splaire_ref" else 1.5
            ax.plot(_bc_s, vals, "o-", color=get_color(mo),
                    label=_model_display[mo], lw=lw, markersize=5)
        ax.set_xticks(_bc_s)
        ax.set_xticklabels(_bl_s, fontsize=8, rotation=45, ha="right")
        ax.set_xlabel("tau bin" if ci == 0 else "")
        if ci == 0: ax.set_ylabel(ylabel)
        title = f"{ntd} tissues — {'reg' if ri == 0 else 'cls'}"
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, frameon=True, edgecolor="lightgray")
        ax.grid(alpha=0.15)

plt.suptitle("prediction error vs tau within each tissue-count stratum", fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_main / "tau_error_by_ntd_stratum")
log.info("tau error by ntd stratum: done")

# =====================================================================
# TAU ANALYSIS — 5-tissue sites only (original, for comparison)
# =====================================================================
log.info("\n--- 5-tissue tau analysis ---")

_site_tissue_5 = _site_tissue_all[_site_tissue_all.n_tissues_detected == len(tissues_fig1)]
_n_sites_5 = _site_tissue_5.groupby(["chrom", "pos", "strand"]).ngroups
log.info(f"sites in all {len(tissues_fig1)} tissues: {_n_sites_5:,}")

_pv_mean = _site_tissue_5.pivot_table(
    index=["chrom", "pos", "strand"], columns="tissue",
    values="mean_ssu", fill_value=0)
_tissue_means = _pv_mean.values
_max_ssu = _tissue_means.max(axis=1, keepdims=True)
_max_ssu_safe = np.where(_max_ssu > 0, _max_ssu, 1)
_x_hat = _tissue_means / _max_ssu_safe
_tau_vals = np.sum(1 - _x_hat, axis=1) / (len(tissues_fig1) - 1)
_tau_vals[_max_ssu.ravel() == 0] = 0

_df_tau_5 = _pv_mean.reset_index()[["chrom", "pos", "strand"]].copy()
_df_tau_5["tau"] = _tau_vals
_df_tau_5["max_ssu"] = _max_ssu.ravel()
_df_tau_5["cross_tissue_sd"] = _tissue_means.std(axis=1)
_df_tau_5["cross_tissue_range"] = _tissue_means.max(axis=1) - _tissue_means.min(axis=1)
_df_tau_5["n_active"] = (_tissue_means > 0.1).sum(axis=1)
_df_tau_5["mean_across_tissues"] = _tissue_means.mean(axis=1)

log.info(f"5-tissue tau: range {_df_tau_5.tau.min():.3f}-{_df_tau_5.tau.max():.3f}, "
         f"median={_df_tau_5.tau.median():.3f}")

_df_tau_5.to_csv(results_dir / "site_tau_5tissue.csv", index=False)

# --- 3 tissue specificity groups ---
# interpretable thresholds based on tau distribution
_tau_group_thresh = [0.1, 0.5]  # constitutive < 0.1, intermediate 0.1-0.5, tissue-specific >= 0.5
_df_tau_5["specificity_group"] = pd.cut(
    _df_tau_5["tau"],
    bins=[-0.001, _tau_group_thresh[0], _tau_group_thresh[1], 1.001],
    labels=["constitutive", "intermediate", "tissue-specific"])
_grp_counts = _df_tau_5["specificity_group"].value_counts()
for g in ["constitutive", "intermediate", "tissue-specific"]:
    log.info(f"  {g}: {_grp_counts.get(g, 0):,} sites")

# save group assignments (used by figures_pub.py for paper figure)
_df_tau_5[["chrom", "pos", "strand", "tau", "specificity_group", "max_ssu",
           "cross_tissue_sd", "mean_across_tissues"]].to_csv(
    results_dir / "site_specificity_groups.csv", index=False)
log.info("saved site_specificity_groups.csv")

# merge groups with df_eval for metric computation
df_eval_grp = df_eval.merge(
    _df_tau_5[["chrom", "pos", "strand", "specificity_group"]],
    on=["chrom", "pos", "strand"], how="inner")

# compute metrics per group
_grp_order = ["constitutive", "intermediate", "tissue-specific"]
_grp_display = {"constitutive": "Constitutive", "intermediate": "Intermediate",
                "tissue-specific": "Tissue-specific"}
_grp_colors = {"constitutive": "#56B4E9", "intermediate": "#E69F00", "tissue-specific": "#D55E00"}

def _compute_grp_metrics(df, models, pcols, groups):
    """pooled metrics per specificity group per tissue"""
    rows = []
    for tissue in tissues_fig1:
        for grp in groups:
            sub = df[(df.tissue == tissue) & (df.specificity_group == grp)]
            for mo in models:
                col = pcols[mo]
                y = sub["y_ssu"].values
                p = sub[col].values
                ok = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
                yv, pv = y[ok], p[ok]
                if len(yv) < 10:
                    continue
                ss_tot = np.sum((yv - yv.mean()) ** 2)
                rows.append({
                    "tissue": tissue, "group": grp, "model": _model_display[mo],
                    "model_key": mo,
                    "r2": 1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan,
                    "pearson": float(_pearsonr(pv, yv)[0]),
                    "mse": float(np.mean((yv - pv)**2)),
                    "mae": float(np.median(np.abs(yv - pv))),
                    "n": len(yv),
                })
    return pd.DataFrame(rows)

_grp_reg = _compute_grp_metrics(df_eval_grp, _reg_models, _reg_pred_cols, _grp_order)
_grp_cls = _compute_grp_metrics(df_eval_grp, _cls_models, _cls_pred_cols, _grp_order)
_grp_all = pd.concat([
    _grp_reg.assign(output_type="regression"),
    _grp_cls.assign(output_type="classification"),
], ignore_index=True)
_grp_all.to_csv(results_dir / "specificity_group_metrics.csv", index=False)
log.info(f"saved specificity_group_metrics.csv: {len(_grp_all)} rows")

# --- supplemental figure: SSU distributions by specificity group ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# left: SSU histogram per group (all tissues pooled)
ax = axes[0]
for grp in _grp_order:
    sub = _df_tau_5[_df_tau_5.specificity_group == grp]
    vals = _pv_mean.loc[sub.set_index(["chrom", "pos", "strand"]).index].values.ravel()
    ax.hist(vals, bins=50, alpha=0.5, color=_grp_colors[grp],
            label=f"{_grp_display[grp]} (n={len(sub):,})", histtype="stepfilled")
ax.set_xlabel("mean SSU (per tissue)")
ax.set_ylabel("count")
ax.set_title("SSU distribution by specificity group")
ax.legend(fontsize=9)

# right: violin per group
ax = axes[1]
_vdata_grp = []
for grp in _grp_order:
    sub = _df_tau_5[_df_tau_5.specificity_group == grp]
    vals = _pv_mean.loc[sub.set_index(["chrom", "pos", "strand"]).index].values.ravel()
    _vdata_grp.append(vals)
parts = ax.violinplot(_vdata_grp, positions=range(len(_grp_order)),
                       showmedians=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(list(_grp_colors.values())[i])
    pc.set_alpha(0.7)
parts["cmedians"].set_color("red")
ax.set_xticks(range(len(_grp_order)))
_grp_xl = [f"{_grp_display[g]}\n(n={_grp_counts.get(g, 0):,})" for g in _grp_order]
ax.set_xticklabels(_grp_xl, fontsize=9)
ax.set_ylabel("mean SSU (per tissue)")
ax.set_title("SSU by specificity group")

plt.tight_layout()
_save_fig(plt, fig2_sup / "specificity_group_ssu")
log.info("specificity group SSU distributions: done")

# --- supplemental figure: metrics by specificity group ---
for met, ylabel in [("mae", "MAE"), ("pearson", "Pearson r"), ("r2", "$R^2$")]:
    fig, axes = plt.subplots(1, len(tissues_fig1), figsize=(4 * len(tissues_fig1), 5), sharey=True)
    for ci, tissue in enumerate(tissues_fig1):
        ax = axes[ci]
        for oi, (otype, models) in enumerate([("regression", _reg_models), ("classification", _cls_models)]):
            sub = _grp_all[((_grp_all.tissue == tissue) & (_grp_all.output_type == otype))]
            if len(sub) == 0:
                continue
            n_models = len(models)
            n_grps = len(_grp_order)
            _w = 0.8 / n_models
            for mi, mo in enumerate(models):
                msub = sub[sub.model_key == mo]
                vals = [msub[msub.group == g][met].values[0]
                        if len(msub[msub.group == g]) > 0 else np.nan
                        for g in _grp_order]
                x = np.arange(n_grps) + oi * (n_grps + 1)
                ax.bar(x + mi * _w - 0.4 + _w / 2, vals, _w,
                       color=get_color(mo), alpha=0.8, edgecolor="white")
        # x-axis
        _all_x = list(range(n_grps)) + list(range(n_grps + 1, 2 * n_grps + 1))
        _all_labels = [_grp_display[g] for g in _grp_order] + [_grp_display[g] for g in _grp_order]
        ax.set_xticks(_all_x)
        ax.set_xticklabels(_all_labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(tissue_display.get(tissue, tissue), fontsize=11, fontweight="bold")
        if ci == 0: ax.set_ylabel(ylabel)
        ax.grid(alpha=0.1, axis="y")
        # section labels
        ax.text(1, -0.22, "reg", ha="center", transform=ax.get_xaxis_transform(), fontsize=8, color=GRAY_MED)
        ax.text(n_grps + 1 + 1, -0.22, "cls", ha="center", transform=ax.get_xaxis_transform(), fontsize=8, color=GRAY_MED)

    # legend
    from matplotlib.patches import Patch as _PatchG
    _leg = [_PatchG(facecolor=get_color(mo), label=_model_display[mo])
            for mo in list(dict.fromkeys(_reg_models + _cls_models))]
    axes[-1].legend(handles=_leg, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"{ylabel} by tissue specificity group", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(plt, fig2_sup / f"specificity_group_{met}")
log.info("specificity group metrics: done")

# --- figure: 5-tissue tau overview (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
ax.hist(_df_tau_5.tau.values, bins=50, color="#56B4E9", edgecolor="white", alpha=0.8)
ax.set_xlabel("tau (tissue specificity)")
ax.set_ylabel("n splice sites")
ax.set_title(f"tau — 5-tissue sites ({_n_sites_5:,})")
ax.axvline(_df_tau_5.tau.median(), color="red", ls="--", lw=1.5,
           label=f"median={_df_tau_5.tau.median():.2f}")
ax.legend(fontsize=9)

ax = axes[0, 1]
_na5 = _df_tau_5.n_active.value_counts().sort_index()
ax.bar(_na5.index, _na5.values, color="#E69F00", edgecolor="white")
ax.set_xlabel("n tissues with mean SSU > 0.1")
ax.set_ylabel("n splice sites")
ax.set_title("tissue activity (5-tissue sites)")
ax.set_xticks(range(0, len(tissues_fig1) + 1))

_n_sc5 = min(5000, len(_df_tau_5))
_idx5 = np.random.default_rng(42).choice(len(_df_tau_5), _n_sc5, replace=False)

ax = axes[1, 0]
ax.scatter(_df_tau_5.tau.values[_idx5], _df_tau_5.cross_tissue_sd.values[_idx5],
           s=3, alpha=0.3, color=GRAY_MED)
ax.set_xlabel("tau"); ax.set_ylabel("cross-tissue SD"); ax.set_title("tau vs SD")

ax = axes[1, 1]
ax.scatter(_df_tau_5.tau.values[_idx5], _df_tau_5.mean_across_tissues.values[_idx5],
           s=3, alpha=0.3, color="#D55E00")
ax.set_xlabel("tau"); ax.set_ylabel("mean SSU across tissues"); ax.set_title("tau vs mean SSU")

plt.tight_layout()
_save_fig(plt, fig2_main / "tau_distribution_5tissue")
log.info("5-tissue tau distribution: done")

# --- figure: 5-tissue tau binned prediction metrics ---
_tau5_nbins = 8
_tau5_edges = np.quantile(_df_tau_5.tau.values, np.linspace(0, 1, _tau5_nbins + 1))
_tau5_edges = np.unique(np.round(_tau5_edges, 3))
_tau5_nbins = len(_tau5_edges) - 1

_df_tau_5["tau_bin"] = pd.cut(_df_tau_5.tau, bins=_tau5_edges,
                               include_lowest=True, duplicates="drop")
df_eval_tau5 = df_eval.merge(
    _df_tau_5[["chrom", "pos", "strand", "tau", "tau_bin", "n_active",
               "cross_tissue_sd", "cross_tissue_range", "max_ssu", "mean_across_tissues"]],
    on=["chrom", "pos", "strand"], how="inner")
_tau5_bins = sorted(_df_tau_5.tau_bin.dropna().unique())
_tau5_nbins = len(_tau5_bins)
_tau5_labels = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in _tau5_bins]
_tau5_counts = [(_df_tau_5.tau_bin == b).sum() for b in _tau5_bins]
_tau5_xl = [f"{_tau5_labels[i]}\n(n={_tau5_counts[i]:,})" for i in range(_tau5_nbins)]

def _compute_binned_tau(df, models, pcols, bins):
    """pooled metrics per tau bin"""
    res = {m: {mo: [] for mo in models} for m in ["r2", "pearson", "mse", "mae"]}
    for b in bins:
        sub = df[df["tau_bin"] == b]
        for mo in models:
            col = pcols[mo]
            y = sub["y_ssu"].values; p = sub[col].values
            ok = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
            yv, pv = y[ok], p[ok]
            if len(yv) < 10:
                for m in res: res[m][mo].append(np.nan); continue
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            res["r2"][mo].append(1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan)
            res["pearson"][mo].append(_pearsonr(pv, yv)[0] if ss_tot > 0 else np.nan)
            res["mse"][mo].append(float(np.mean((yv - pv)**2)))
            res["mae"][mo].append(float(np.median(np.abs(yv - pv))))
    return res

_t5_reg = _compute_binned_tau(df_eval_tau5, _reg_models, _reg_pred_cols, _tau5_bins)
_t5_cls = _compute_binned_tau(df_eval_tau5, _cls_models, _cls_pred_cols, _tau5_bins)

_tau_met_labels = [("r2", "$R^2$"), ("pearson", "Pearson r"),
                   ("mse", "MSE"), ("mae", "Median Absolute Error")]
_tau5_centers = np.arange(_tau5_nbins)

fig, axes = plt.subplots(len(_tau_met_labels), 2,
                         figsize=(14, 4 * len(_tau_met_labels)), sharex=True)
for row_i, (met, ylabel) in enumerate(_tau_met_labels):
    for col_i, (models, src, title) in enumerate([
        (_reg_models, _t5_reg, "Regression Outputs"),
        (_cls_models, _t5_cls, "Classification Outputs"),
    ]):
        ax = axes[row_i, col_i]
        for mo in models:
            vals = np.array(src[met][mo])
            lw = 2.5 if mo == "splaire_ref" else 1.8
            ax.plot(_tau5_centers, vals, "o-", color=get_color(mo),
                    label=_model_display[mo], lw=lw, markersize=5)
        ax.set_ylabel(ylabel if col_i == 0 else "")
        if row_i == 0: ax.set_title(title)
        ax.legend(fontsize=8, frameon=True, edgecolor="lightgray")
        ax.grid(alpha=0.2)
        ax.set_xticks(_tau5_centers)
        if row_i == len(_tau_met_labels) - 1:
            ax.set_xticklabels(_tau5_xl, fontsize=8, rotation=45, ha="right")
            ax.set_xlabel("tau bin")
fig.suptitle(f"prediction metrics vs tau — 5-tissue sites ({_n_sites_5:,})",
             fontsize=14, y=1.01)
plt.tight_layout()
_save_fig(plt, fig2_main / "tau_metrics_binned_5tissue")
log.info("5-tissue tau metrics binned: done")

# --- figure: site-level scatter (tau vs error) for 5-tissue sites ---
_se5 = df_eval_tau5.groupby(["chrom", "pos", "strand"]).agg(
    tau=("tau", "first"), y_mean=("y_ssu", "mean"),
    **{f"p_{mo}_reg": (_reg_pred_cols[mo], "mean") for mo in _reg_models},
    **{f"p_{mo}_cls": (_cls_pred_cols[mo], "mean") for mo in _cls_models},
).reset_index()
for mo in _reg_models:
    _se5[f"ae_{mo}_reg"] = np.abs(_se5["y_mean"] - _se5[f"p_{mo}_reg"])
for mo in _cls_models:
    _se5[f"ae_{mo}_cls"] = np.abs(_se5["y_mean"] - _se5[f"p_{mo}_cls"])

_rng5 = np.random.default_rng(42)
_ns5 = min(5000, len(_se5))
_si5 = _rng5.choice(len(_se5), _ns5, replace=False)
_ss5 = _se5.iloc[_si5]

for output_type, models in [("reg", _reg_models), ("cls", _cls_models)]:
    nm = len(models)
    fig, axes = plt.subplots(1, nm, figsize=(5 * nm, 4.5), sharey=True)
    if nm == 1: axes = [axes]
    _be = np.linspace(0, 1, 31)
    _bc = (_be[:-1] + _be[1:]) / 2
    for ci, mo in enumerate(models):
        ae_col = f"ae_{mo}_{output_type}"
        ax = axes[ci]
        ax.scatter(_ss5.tau.values, _ss5[ae_col].values,
                   s=3, alpha=0.2, color=get_color(mo))
        _bi = np.clip(np.digitize(_se5.tau.values, _be) - 1, 0, 29)
        _med = [np.median(_se5[ae_col].values[_bi == i])
                if (_bi == i).sum() > 10 else np.nan for i in range(30)]
        ax.plot(_bc, _med, "-", color=get_color(mo), lw=2.5, label="median trend")
        ax.set_xlabel("tau")
        if ci == 0: ax.set_ylabel("|predicted - true SSU|")
        ax.set_title(f"{_model_display[mo]} ({output_type})")
        ax.set_xlim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.15)
    fig.suptitle(f"site-level error vs tau — {output_type} outputs (5-tissue sites)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(plt, fig2_main / f"tau_scatter_{output_type}_5tissue")
log.info("5-tissue tau scatter: done")

# --- figure: SSU violin per tau bin (5-tissue) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
_vdata = []
for b in _tau5_bins:
    _sb = _df_tau_5[_df_tau_5.tau_bin == b]
    _bm = _pv_mean.loc[_sb.set_index(["chrom", "pos", "strand"]).index]
    _vdata.append(_bm.values.ravel())
ax = axes[0]
parts = ax.violinplot(_vdata, positions=range(_tau5_nbins), showmedians=True, showextrema=False)
for pc in parts["bodies"]: pc.set_facecolor("#56B4E9"); pc.set_alpha(0.7)
parts["cmedians"].set_color("red")
ax.set_xticks(range(_tau5_nbins))
ax.set_xticklabels(_tau5_xl, fontsize=8, rotation=45, ha="right")
ax.set_xlabel("tau bin"); ax.set_ylabel("SSU (tissue means)")
ax.set_title("SSU by tau bin (5-tissue)")

ax = axes[1]
_bdata = [_df_tau_5[_df_tau_5.tau_bin == b]["max_ssu"].values for b in _tau5_bins]
parts2 = ax.violinplot(_bdata, positions=range(_tau5_nbins), showmedians=True, showextrema=False)
for pc in parts2["bodies"]: pc.set_facecolor("#E69F00"); pc.set_alpha(0.7)
parts2["cmedians"].set_color("red")
ax.set_xticks(range(_tau5_nbins))
ax.set_xticklabels(_tau5_xl, fontsize=8, rotation=45, ha="right")
ax.set_xlabel("tau bin"); ax.set_ylabel("max SSU"); ax.set_title("max SSU by tau bin (5-tissue)")
plt.tight_layout()
_save_fig(plt, fig2_main / "tau_ssu_violin_5tissue")
log.info("5-tissue SSU violin: done")

# --- outlier tissue analysis: which tissue makes each site specific? ---
# a high-tau site could be specific because:
#   - one tissue has HIGH SSU while others are low ("inclusion-specific")
#   - one tissue has LOW SSU while others are high ("exclusion-specific")
# we identify both the outlier (max) and depleted (min) tissue

_max_tissue = _pv_mean.idxmax(axis=1).reset_index()
_max_tissue.columns = ["chrom", "pos", "strand", "_max_tissue"]
_min_tissue = _pv_mean.idxmin(axis=1).reset_index()
_min_tissue.columns = ["chrom", "pos", "strand", "_min_tissue"]
_df_tau_5 = _df_tau_5.merge(_max_tissue, on=["chrom", "pos", "strand"], how="left")
_df_tau_5 = _df_tau_5.merge(_min_tissue, on=["chrom", "pos", "strand"], how="left")

# specificity direction: is the outlier the high tissue or the low tissue?
# compare distance of max from mean vs distance of min from mean
_site_mean = _tissue_means.mean(axis=1)
_dist_max = _tissue_means.max(axis=1) - _site_mean  # how far max is above mean
_dist_min = _site_mean - _tissue_means.min(axis=1)  # how far min is below mean
# "inclusion" = max is the bigger outlier, "exclusion" = min is the bigger outlier
_df_tau_5["specificity_direction"] = np.where(
    _dist_max >= _dist_min, "inclusion", "exclusion")
# outlier tissue: whichever tissue is furthest from the mean
_df_tau_5["outlier_tissue"] = np.where(
    _dist_max >= _dist_min,
    _df_tau_5["_max_tissue"],   # max tissue is the outlier
    _df_tau_5["_min_tissue"])   # min tissue is the outlier

# specificity ratio: max / second_max (measures concentration at top)
_sorted_vals = np.sort(_pv_mean.values, axis=1)[:, ::-1]
_second_max = _sorted_vals[:, 1]
_spec_ratio = np.where(_second_max > 0,
                        _sorted_vals[:, 0] / _second_max, np.inf)
_df_tau_5["specificity_ratio"] = _spec_ratio

# depletion ratio: second_min / min (measures concentration at bottom)
_second_min = _sorted_vals[:, -2]
_dep_ratio = np.where(_sorted_vals[:, -1] > 0,
                       _second_min / _sorted_vals[:, -1], np.inf)
_df_tau_5["depletion_ratio"] = _dep_ratio

log.info(f"outlier tissue distribution:")
for t in tissues_fig1:
    n = (_df_tau_5.outlier_tissue == t).sum()
    log.info(f"  {tissue_display.get(t, t)}: {n:,}")
log.info(f"depleted tissue distribution:")
for t in tissues_fig1:
    n = (_df_tau_5._min_tissue == t).sum()
    log.info(f"  {tissue_display.get(t, t)}: {n:,}")
_dir_counts = _df_tau_5.specificity_direction.value_counts()
log.info(f"specificity direction: {dict(_dir_counts)}")
log.info(f"outlier tissue distribution:")
for t in tissues_fig1:
    n = (_df_tau_5.outlier_tissue == t).sum()
    log.info(f"  {tissue_display.get(t, t)}: {n:,}")

# --- figure: outlier tissue by tau bin ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# stacked bar: outlier tissue (highest SSU) by tau bin
ax = axes[0, 0]
_dom_counts = pd.crosstab(_df_tau_5["tau_bin"], _df_tau_5["outlier_tissue"])
_dom_counts = _dom_counts.reindex(columns=tissues_fig1, fill_value=0)
_dom_pct = _dom_counts.div(_dom_counts.sum(axis=1), axis=0) * 100
_tau5_bin_x = np.arange(_tau5_nbins)
_bottom = np.zeros(_tau5_nbins)
for t in tissues_fig1:
    vals = _dom_pct[t].values if t in _dom_pct.columns else np.zeros(_tau5_nbins)
    ax.bar(_tau5_bin_x, vals, bottom=_bottom, color=tissue_colors.get(t, "#999"),
           label=tissue_display.get(t, t), edgecolor="white", linewidth=0.5)
    _bottom += vals
ax.set_xticks(_tau5_bin_x)
ax.set_xticklabels(_tau5_xl, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("% of sites"); ax.set_title("outlier tissue (highest SSU) by tau bin")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_ylim(0, 100)

# stacked bar: depleted tissue (lowest SSU) by tau bin
ax = axes[0, 1]
_dep_counts = pd.crosstab(_df_tau_5["tau_bin"], _df_tau_5["_min_tissue"])
_dep_counts = _dep_counts.reindex(columns=tissues_fig1, fill_value=0)
_dep_pct = _dep_counts.div(_dep_counts.sum(axis=1), axis=0) * 100
_bottom = np.zeros(_tau5_nbins)
for t in tissues_fig1:
    vals = _dep_pct[t].values if t in _dep_pct.columns else np.zeros(_tau5_nbins)
    ax.bar(_tau5_bin_x, vals, bottom=_bottom, color=tissue_colors.get(t, "#999"),
           label=tissue_display.get(t, t), edgecolor="white", linewidth=0.5)
    _bottom += vals
ax.set_xticks(_tau5_bin_x)
ax.set_xticklabels(_tau5_xl, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("% of sites"); ax.set_title("depleted tissue (lowest SSU) by tau bin")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_ylim(0, 100)

# specificity direction by tau bin
ax = axes[1, 0]
_dir_counts = pd.crosstab(_df_tau_5["tau_bin"], _df_tau_5["specificity_direction"])
for d in ["inclusion", "exclusion"]:
    if d not in _dir_counts.columns:
        _dir_counts[d] = 0
_dir_pct = _dir_counts.div(_dir_counts.sum(axis=1), axis=0) * 100
ax.bar(_tau5_bin_x, _dir_pct["inclusion"].values, color="#0072B2",
       label="inclusion-specific (max is outlier)", edgecolor="white")
ax.bar(_tau5_bin_x, _dir_pct["exclusion"].values,
       bottom=_dir_pct["inclusion"].values, color="#D55E00",
       label="exclusion-specific (min is outlier)", edgecolor="white")
ax.set_xticks(_tau5_bin_x)
ax.set_xticklabels(_tau5_xl, fontsize=7, rotation=45, ha="right")
ax.set_ylabel("% of sites"); ax.set_title("specificity direction by tau bin")
ax.legend(fontsize=8); ax.set_ylim(0, 100)

# tau distribution colored by outlier tissue
ax = axes[1, 1]
for t in tissues_fig1:
    sub = _df_tau_5[_df_tau_5.outlier_tissue == t]
    ax.hist(sub.tau.values, bins=50, alpha=0.5, color=tissue_colors.get(t, "#999"),
            label=f"{tissue_display.get(t, t)} ({len(sub):,})", histtype="stepfilled")
ax.set_xlabel("tau"); ax.set_ylabel("n splice sites")
ax.set_title("tau distribution by outlier tissue")
ax.legend(fontsize=8)

plt.tight_layout()
_save_fig(plt, fig2_main / "outlier_tissue_by_tau")
log.info("outlier tissue by tau: done")

# high-tau threshold and eval dataframes (used by heatmap and specificity direction)
_high_tau_thresh = _df_tau_5.tau.quantile(0.75)
_high_tau_sites = _df_tau_5[_df_tau_5.tau >= _high_tau_thresh]
log.info(f"high-tau sites (>= {_high_tau_thresh:.3f}): {len(_high_tau_sites):,}")

df_eval_dom = df_eval_tau5.merge(
    _df_tau_5[["chrom", "pos", "strand", "outlier_tissue", "specificity_ratio"]],
    on=["chrom", "pos", "strand"], how="inner")

# --- figure: heatmap of mean SSU across tissues for high-tau sites ---
_hm_tau_thresh = _df_tau_5.tau.quantile(0.90)
_hm_sites_all = _df_tau_5[_df_tau_5.tau >= _hm_tau_thresh].copy()
log.info(f"heatmap: {len(_hm_sites_all):,} sites with tau >= {_hm_tau_thresh:.3f}")
_tissue_cols = [t for t in tissues_fig1 if t in _pv_mean.columns]


def _plot_tau_heatmap(sites_df, title_extra, save_name):
    """plot z-score + raw SSU heatmap for a set of high-tau sites"""
    profiles = sites_df.merge(_pv_mean.reset_index(), on=["chrom", "pos", "strand"])
    # sort by tau descending within each outlier tissue group
    parts = []
    for t in tissues_fig1:
        grp = profiles[profiles.outlier_tissue == t].copy()
        if len(grp) == 0:
            continue
        grp = grp.sort_values("tau", ascending=False)
        parts.append(grp)
    if not parts:
        return
    profiles = pd.concat(parts, ignore_index=True)
    raw = profiles[_tissue_cols].values
    dom = profiles["outlier_tissue"].values

    # z-score per row
    rmean = raw.mean(axis=1, keepdims=True)
    rstd = raw.std(axis=1, keepdims=True)
    rstd[rstd == 0] = 1
    zdata = (raw - rmean) / rstd
    vmax_z = 2.5

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(raw) * 0.008)),
                              gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})

    ax = axes[0]
    im = ax.imshow(np.clip(zdata, -vmax_z, vmax_z), aspect="auto",
                   cmap="RdBu_r", vmin=-vmax_z, vmax=vmax_z, interpolation="nearest")
    ax.set_xticks(range(len(_tissue_cols)))
    ax.set_xticklabels([tissue_display.get(t, t) for t in _tissue_cols], fontsize=10)
    ax.set_ylabel("splice sites (grouped by outlier tissue)")
    ax.set_title("row-normalized SSU (z-score)")
    plt.colorbar(im, ax=ax, label="z-score", shrink=0.6, pad=0.02)

    ax2 = axes[1]
    im2 = ax2.imshow(raw, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                     interpolation="nearest")
    ax2.set_xticks(range(len(_tissue_cols)))
    ax2.set_xticklabels([tissue_display.get(t, t) for t in _tissue_cols], fontsize=10)
    ax2.set_title("raw mean SSU")
    plt.colorbar(im2, ax=ax2, label="mean SSU", shrink=0.6, pad=0.02)

    n_sites_hm = len(profiles)
    fig.suptitle(f"SSU profiles of high-tau sites (top 10%, tau >= {_hm_tau_thresh:.2f})"
                 f"{title_extra}\n{n_sites_hm:,} sites", fontsize=13, y=1.02)

    for _ax in [axes[0], axes[1]]:
        _cum = 0
        _labels_y = []
        for t in tissues_fig1:
            n = (dom == t).sum()
            if n == 0:
                continue
            _labels_y.append((_cum + n / 2, f"{tissue_display.get(t, t)}\n({n})"))
            if _cum > 0:
                _ax.axhline(_cum - 0.5, color="black", lw=2)
            _cum += n
        if _ax is axes[0]:
            for ypos, label in _labels_y:
                _ax.text(-0.08, ypos, label, ha="right", va="center", fontsize=9,
                         transform=_ax.get_yaxis_transform(), fontweight="bold")
        _ax.set_yticks([])
        _ax.set_xlim(-0.5, len(_tissue_cols) - 0.5)

    plt.tight_layout()
    _save_fig(plt, fig2_main / save_name)


# version 1: all high-tau sites (no filter)
_plot_tau_heatmap(_hm_sites_all, "", "high_tau_ssu_heatmap")
log.info("high-tau SSU heatmap (all): done")

# version 2: filtered to max SSU >= 0.1
_hm_sites_filt = _hm_sites_all[_hm_sites_all["max_ssu"] >= 0.1]
log.info(f"heatmap after max_ssu >= 0.1 filter: {len(_hm_sites_filt):,} sites "
         f"(removed {len(_hm_sites_all) - len(_hm_sites_filt):,})")
_plot_tau_heatmap(_hm_sites_filt, " — max SSU >= 0.1", "high_tau_ssu_heatmap_filtered")
log.info("high-tau SSU heatmap (filtered): done")

# =====================================================================
# MODEL OUTPUT PERFORMANCE ON HIGH-TAU SITES (options A-E)
# same sites as the heatmaps, all model outputs
# =====================================================================
import pyarrow.parquet as pq

_pang_tissue_heads = {
    "heart": ("heart_usage", "heart_p_splice"),
    "liver": ("liver_usage", "liver_p_splice"),
    "brain": ("brain_usage", "brain_p_splice"),
    "testis": ("testis_usage", "testis_p_splice"),
}
_spt_tissue_heads = [
    "adipose", "blood", "blood_vessel", "brain", "colon", "heart", "kidney",
    "liver", "lung", "muscle", "nerve", "small_intestine", "skin", "spleen", "stomach",
]

# load tissue head cache (built later if missing — run full script first)
_head_cache = Path("data/tissue_head_cache.parquet")
if not _head_cache.exists():
    log.warning(f"tissue head cache not found at {_head_cache}, "
                "run full analysis first to build it. skipping performance plots.")
    _perf_base = None
else:
    _df_heads_early = pd.read_parquet(_head_cache)
    log.info(f"loaded tissue head cache: {len(_df_heads_early):,} rows")
    # merge with tau
    _df_heads_tau_early = _df_heads_early.merge(
        _df_tau_5[["chrom", "pos", "strand", "tau", "outlier_tissue",
                    "specificity_direction"]],
        on=["chrom", "pos", "strand"], how="inner")

    # build merged prediction frame: df_all cols + tissue head cols for heatmap sites
    _perf_base = _df_heads_tau_early.merge(
    df_all[["chrom", "pos", "strand", "tissue", "sample",
            "pred_splaire_ref_reg", "pred_splaire_ref_cls",
            "pred_spliceai_cls", "pred_pangolin_reg", "pred_pangolin_cls",
            "pred_splicetransformer_reg", "pred_splicetransformer_cls"]],
    on=["chrom", "pos", "strand", "tissue", "sample"], how="left")

    # also add splaire_var if available
    if "pred_splaire_var_reg" in df_all.columns:
        _var_cols = df_all[["chrom", "pos", "strand", "tissue", "sample",
                            "pred_splaire_var_reg", "pred_splaire_var_cls"]]
        _perf_base = _perf_base.merge(_var_cols, on=["chrom", "pos", "strand", "tissue", "sample"],
                                       how="left")
    
    # all model outputs to evaluate
    _all_heads_list = []
    
    # splaire
    _all_heads_list.append(("SPLAIRE (SSU)", "pred_splaire_ref_reg", "SPLAIRE"))
    _all_heads_list.append(("SPLAIRE (cls)", "pred_splaire_ref_cls", "SPLAIRE"))
    if "pred_splaire_var_reg" in _perf_base.columns:
        _all_heads_list.append(("SPLAIRE-var (SSU)", "pred_splaire_var_reg", "SPLAIRE"))
        _all_heads_list.append(("SPLAIRE-var (cls)", "pred_splaire_var_cls", "SPLAIRE"))

    # spliceai
    _all_heads_list.append(("SpliceAI", "pred_spliceai_cls", "SpliceAI"))

    # pangolin avg
    _all_heads_list.append(("Pangolin (avg usage)", "pred_pangolin_reg", "Pangolin"))
    _all_heads_list.append(("Pangolin (avg p_splice)", "pred_pangolin_cls", "Pangolin"))

    # pangolin tissue heads
    for pt in ["heart", "liver", "brain", "testis"]:
        for suffix, col_sfx in [("usage", "usage"), ("p_splice", "cls")]:
            col = f"pang_{pt}_{col_sfx}"
            if col in _perf_base.columns:
                _all_heads_list.append((f"Pangolin ({pt} {suffix})", col, "Pangolin"))

    # splicetransformer avg
    _all_heads_list.append(("SpliceTransformer (avg)", "pred_splicetransformer_reg", "SpliceTransformer"))
    _all_heads_list.append(("SpliceTransformer (cls)", "pred_splicetransformer_cls", "SpliceTransformer"))

    # splicetransformer tissue heads
    for st in _spt_tissue_heads:
        col = f"spt_{st}"
        if col in _perf_base.columns:
            _all_heads_list.append((f"SpliceTransformer ({st})", col, "SpliceTransformer"))
    
    log.info(f"model outputs for performance analysis: {len(_all_heads_list)}")
    
    _family_colors = {"SPLAIRE": "#56B4E9", "SpliceAI": "#D55E00",
                      "Pangolin": "#009E73", "SpliceTransformer": "#E69F00"}
    
    
    def _compute_head_perf(df, heads_list):
        """compute R², pearson, MSE, MAE per eval tissue per head"""
        rows = []
        for tissue in tissues_fig1:
            sub = df[df.tissue == tissue]
            y = sub["y_ssu"].values
            ok_base = np.isfinite(y) & (y != SENTINEL)
            for name, col, family in heads_list:
                if col not in sub.columns:
                    continue
                p = sub[col].values
                ok = ok_base & np.isfinite(p)
                yv, pv = y[ok], p[ok]
                if len(yv) < 50:
                    continue
                ss_tot = np.sum((yv - yv.mean()) ** 2)
                rows.append({
                    "tissue": tissue, "head": name, "col": col, "family": family,
                    "r2": 1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan,
                    "pearson": float(_pearsonr(pv, yv)[0]),
                    "mse": float(np.mean((yv - pv)**2)),
                    "mae": float(np.median(np.abs(yv - pv))),
                    "n": len(yv),
                })
        return pd.DataFrame(rows)
    
    
    def _plot_all_options(site_df, perf_base_df, label, save_suffix):
        """generate options A-E for a given site set"""
        # filter perf_base to these sites
        site_keys = site_df[["chrom", "pos", "strand"]].drop_duplicates()
        df_perf = perf_base_df.merge(site_keys, on=["chrom", "pos", "strand"])
        n_sites = len(site_keys)
        log.info(f"\n--- {label}: {n_sites:,} sites, {len(df_perf):,} observations ---")
    
        perf = _compute_head_perf(df_perf, _all_heads_list)
        if len(perf) == 0:
            log.info(f"  no data, skipping")
            return
        perf.to_csv(results_dir / f"head_perf_{save_suffix}.csv", index=False)
    
        # --- option A: heatmap (tissue × head) ---
        for met, ylabel, cmap in [("mae", "MAE", "YlOrRd_r"), ("pearson", "Pearson r", "YlOrRd")]:
            _heads_ordered = []
            for fam in ["SPLAIRE", "SpliceAI", "Pangolin", "SpliceTransformer"]:
                fam_heads = perf[perf.family == fam]["head"].unique()
                _heads_ordered.extend(sorted(fam_heads))
            _eval_ts = [t for t in tissues_fig1 if t in perf.tissue.unique()]
            mat = np.full((len(_eval_ts), len(_heads_ordered)), np.nan)
            for ri, t in enumerate(_eval_ts):
                for ci, h in enumerate(_heads_ordered):
                    row = perf[(perf.tissue == t) & (perf["head"] == h)]
                    if len(row) > 0:
                        mat[ri, ci] = row[met].values[0]
    
            fig, ax = plt.subplots(figsize=(max(10, len(_heads_ordered) * 0.5),
                                            max(3, len(_eval_ts) * 0.9)))
            im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
            ax.set_xticks(range(len(_heads_ordered)))
            ax.set_xticklabels(_heads_ordered, rotation=60, ha="right", fontsize=6)
            ax.set_yticks(range(len(_eval_ts)))
            ax.set_yticklabels([tissue_display.get(t, t) for t in _eval_ts], fontsize=9)
            # annotate and highlight best per row
            for ri in range(len(_eval_ts)):
                row_vals = mat[ri]
                valid = ~np.isnan(row_vals)
                if not valid.any():
                    continue
                best_ci = np.nanargmin(row_vals) if met == "mae" else np.nanargmax(row_vals)
                for ci in range(len(_heads_ordered)):
                    if np.isnan(mat[ri, ci]):
                        continue
                    color = "white" if mat[ri, ci] > np.nanmedian(row_vals[valid]) else "black"
                    weight = "bold" if ci == best_ci else "normal"
                    ax.text(ci, ri, f"{mat[ri, ci]:.3f}", ha="center", va="center",
                            fontsize=5, color=color, fontweight=weight)
                from matplotlib.patches import Rectangle as _R
                ax.add_patch(_R((best_ci - 0.5, ri - 0.5), 1, 1,
                                fill=False, edgecolor="red", lw=2))
            # family separators
            _ci = 0
            for fam in ["SPLAIRE", "SpliceAI", "Pangolin", "SpliceTransformer"]:
                n_fam = sum(1 for _, _, f in _all_heads_list if f == fam and
                            any(h == _ for h in _heads_ordered) is False) or 0
                n_fam = len([h for h in _heads_ordered if
                             any(name == h and f == fam for name, _, f in _all_heads_list)])
                if _ci > 0 and n_fam > 0:
                    ax.axvline(_ci - 0.5, color="black", lw=1, alpha=0.5)
                _ci += n_fam
    
            plt.colorbar(im, ax=ax, label=ylabel, shrink=0.7)
            ax.set_title(f"A: {ylabel} — all heads × tissues ({label})")
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"perf_A_heatmap_{met}_{save_suffix}")
        log.info(f"  option A (heatmap): done")
    
        # --- option B: ranked horizontal bars per tissue ---
        # per-tissue matched heads (starred on y-axis)
        # key = eval tissue, value = set of head display names that are matched
        _matched_heads = {
            "haec10": {"SPLAIRE (SSU)", "SPLAIRE (cls)"},
            "lung": {"SPLAIRE (SSU)", "SPLAIRE (cls)", "SpliceTransformer (lung)"},
            "testis": {"SPLAIRE (SSU)", "SPLAIRE (cls)",
                       "Pangolin (testis usage)", "Pangolin (testis p_splice)"},
            "brain_cortex": {"SPLAIRE (SSU)", "SPLAIRE (cls)",
                             "SpliceTransformer (brain)",
                             "Pangolin (brain usage)", "Pangolin (brain p_splice)"},
            "whole_blood": {"SPLAIRE (SSU)", "SPLAIRE (cls)", "SpliceTransformer (blood)"},
        }
        # also match splaire-var if present
        for t in _matched_heads:
            _matched_heads[t] |= {"SPLAIRE-var (SSU)", "SPLAIRE-var (cls)"}

        for met, ylabel in [("mae", "MAE"), ("pearson", "Pearson r")]:
            ascending = (met == "mae")
            n_heads = perf["head"].nunique()
            fig, axes = plt.subplots(1, len(tissues_fig1),
                                     figsize=(4.5 * len(tissues_fig1), max(7, n_heads * 0.28)),
                                     sharey=False)
            for ci, tissue in enumerate(tissues_fig1):
                ax = axes[ci]
                sub = perf[perf.tissue == tissue].copy()
                if len(sub) == 0:
                    ax.set_visible(False)
                    continue
                sub = sub.sort_values(met, ascending=not ascending)
                ypos = np.arange(len(sub))

                matched_set = _matched_heads.get(tissue, set())

                bar_colors = []
                alphas = []
                is_matched_list = []
                for _, row in sub.iterrows():
                    head_name = row["head"]
                    if "var" in head_name.lower() and row["family"] == "SPLAIRE":
                        base = "#0072B2"
                    elif "avg" in head_name.lower() and row["family"] == "SPLAIRE":
                        base = "#CC79A7"
                    else:
                        base = _family_colors.get(row["family"], "#999")
                    is_matched = head_name in matched_set
                    bar_colors.append(base)
                    alphas.append(1.0 if is_matched else 0.2)
                    is_matched_list.append(is_matched)

                for i, (y_i, v, c, a) in enumerate(zip(ypos, sub[met].values, bar_colors, alphas)):
                    ax.barh(y_i, v, color=c, alpha=a, height=0.75, edgecolor="white", linewidth=0.3)

                # annotate values
                for i, v in enumerate(sub[met].values):
                    ax.text(v + 0.002 if met == "mae" else max(v - 0.002, 0.002), i,
                            f"{v:.3f}", va="center", fontsize=5.5,
                            ha="left" if met == "mae" else ("right" if v > 0.1 else "left"),
                            color=GRAY_DARK)

                # y-axis labels: * prefix for matched, normal weight
                ax.set_yticks(ypos)
                star_labels = [f"\u2605 {h}" if m else f"  {h}"
                               for h, m in zip(sub["head"].values, is_matched_list)]
                ax.set_yticklabels(star_labels, fontsize=7)
                for i, (lbl, matched) in enumerate(zip(ax.get_yticklabels(), is_matched_list)):
                    if matched:
                        lbl.set_fontweight("bold")
                        lbl.set_color("black")
                    else:
                        lbl.set_color("#aaaaaa")

                ax.set_xlabel(ylabel, fontsize=10)
                ax.set_title(tissue_display.get(tissue, tissue), fontsize=12, fontweight="bold")
                ax.grid(alpha=0.1, axis="x")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # legend
            from matplotlib.patches import Patch as _Patch
            _leg_handles = [_Patch(facecolor=c, alpha=1.0, label=f) for f, c in _family_colors.items()]
            _leg_handles.append(_Patch(facecolor=GRAY_FAINT, alpha=0.2, label="no tissue match"))
            _leg_handles.append(plt.Line2D([], [], marker="*", color="black", lw=0,
                                           markersize=8, label="\u2605 = tissue-matched"))
            axes[-1].legend(handles=_leg_handles, fontsize=7,
                           bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True,
                           edgecolor="lightgray")

            fig.suptitle(f"{ylabel} — all model outputs on high-tau sites ({label})",
                         fontsize=14, y=1.01)
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"perf_B_bars_{met}_{save_suffix}")
        log.info(f"  option B (ranked bars): done")
    
        # --- option C: faceted scatter (predicted vs true, best head per family per tissue) ---
        _fam_order = ["SPLAIRE", "SpliceAI", "Pangolin", "SpliceTransformer"]
        _avail_fams = [f for f in _fam_order if f in perf.family.unique()]
        _avail_ts = [t for t in tissues_fig1 if t in perf.tissue.unique()]
        fig, axes = plt.subplots(len(_avail_fams), len(_avail_ts),
                                 figsize=(3.5 * len(_avail_ts), 3.5 * len(_avail_fams)),
                                 squeeze=False)
        _rng_c = np.random.default_rng(42)
        for ri, fam in enumerate(_avail_fams):
            for ci, tissue in enumerate(_avail_ts):
                ax = axes[ri, ci]
                fam_sub = perf[(perf.family == fam) & (perf.tissue == tissue)]
                if len(fam_sub) == 0:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                    if ri == 0: ax.set_title(tissue_display.get(tissue, tissue))
                    if ci == 0: ax.set_ylabel(fam)
                    continue
                best = fam_sub.sort_values("mae", ascending=True).iloc[0]
                col = best["col"]
                tsub = df_perf[(df_perf.tissue == tissue)]
                y = tsub["y_ssu"].values
                p = tsub[col].values if col in tsub.columns else np.full(len(y), np.nan)
                ok = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
                yv, pv = y[ok], p[ok]
                # subsample
                n_sc = min(3000, len(yv))
                idx = _rng_c.choice(len(yv), n_sc, replace=False)
                ax.scatter(pv[idx], yv[idx], s=1, alpha=0.15, color=_family_colors.get(fam, "#999"))
                ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.text(0.05, 0.95, f"r={best['pearson']:.3f}\nMAE={best['mae']:.3f}\n{best['head']}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
                if ri == 0: ax.set_title(tissue_display.get(tissue, tissue), fontsize=10)
                if ci == 0: ax.set_ylabel(f"{fam}\ntrue SSU", fontsize=9)
                if ri == len(_avail_fams) - 1: ax.set_xlabel("predicted SSU", fontsize=9)
                ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
                ax.tick_params(labelsize=7)
        fig.suptitle(f"C: predicted vs true SSU — best head per family ({label})",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        _save_fig(plt, fig2_main / f"perf_C_scatter_{save_suffix}")
        log.info(f"  option C (scatter): done")
    
        # --- option D: parallel coordinates (lines across tissues) ---
        for met, ylabel in [("mae", "MAE"), ("pearson", "Pearson r")]:
            fig, ax = plt.subplots(figsize=(10, 6))
            _x_pc = np.arange(len(_avail_ts))
            for _, row in perf.drop_duplicates("head").iterrows():
                h = row["head"]
                fam = row["family"]
                vals = []
                for t in _avail_ts:
                    match = perf[(perf["head"] == h) & (perf.tissue == t)]
                    vals.append(match[met].values[0] if len(match) > 0 else np.nan)
                ax.plot(_x_pc, vals, "-o", color=_family_colors.get(fam, "#999"),
                        alpha=0.3, lw=1, markersize=3)
            # highlight best per family
            for fam in _avail_fams:
                fam_perf = perf[perf.family == fam]
                if len(fam_perf) == 0:
                    continue
                best_head = (fam_perf.groupby("head")[met].mean().idxmin()
                            if met == "mae" else
                            fam_perf.groupby("head")[met].mean().idxmax())
                vals = []
                for t in _avail_ts:
                    match = perf[(perf["head"] == best_head) & (perf.tissue == t)]
                    vals.append(match[met].values[0] if len(match) > 0 else np.nan)
                ax.plot(_x_pc, vals, "-o", color=_family_colors.get(fam, "#999"),
                        alpha=1, lw=2.5, markersize=6, label=f"{fam}: {best_head}")
            ax.set_xticks(_x_pc)
            ax.set_xticklabels([tissue_display.get(t, t) for t in _avail_ts])
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(alpha=0.2)
            ax.set_title(f"D: {ylabel} across tissues — all heads ({label})")
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"perf_D_parallel_{met}_{save_suffix}")
        log.info(f"  option D (parallel): done")
    
        # --- option E: grouped bars (best head per family) ---
        for met, ylabel in [("mae", "MAE"), ("pearson", "Pearson r")]:
            # select representative heads per family
            _rep_heads = []
            for fam in _fam_order:
                fam_perf = perf[perf.family == fam]
                if len(fam_perf) == 0:
                    continue
                # best overall head for this family
                best = (fam_perf.groupby("head")[met].mean().idxmin()
                        if met == "mae" else
                        fam_perf.groupby("head")[met].mean().idxmax())
                _rep_heads.append((best, fam))
                # also add avg head if different
                avg_heads = [n for n, c, f in _all_heads_list if f == fam and "avg" in n.lower()]
                for ah in avg_heads:
                    if ah != best and ah in fam_perf["head"].values:
                        _rep_heads.append((ah, fam))
    
            fig, ax = plt.subplots(figsize=(max(10, len(_avail_ts) * 2.5), 5))
            n_reps = len(_rep_heads)
            _w = 0.8 / n_reps
            for hi, (head_name, fam) in enumerate(_rep_heads):
                vals = []
                for t in _avail_ts:
                    match = perf[(perf["head"] == head_name) & (perf.tissue == t)]
                    vals.append(match[met].values[0] if len(match) > 0 else np.nan)
                x = np.arange(len(_avail_ts))
                ax.bar(x + hi * _w - 0.4 + _w / 2, vals, _w,
                       color=_family_colors.get(fam, "#999"), label=head_name,
                       edgecolor="white", alpha=0.8)
            ax.set_xticks(np.arange(len(_avail_ts)))
            ax.set_xticklabels([tissue_display.get(t, t) for t in _avail_ts], fontsize=10)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(alpha=0.15, axis="y")
            ax.set_title(f"E: {ylabel} — best + avg heads per family ({label})")
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"perf_E_grouped_{met}_{save_suffix}")
        log.info(f"  option E (grouped bars): done")
    
    
    # run for both heatmap site sets
    _plot_all_options(_hm_sites_all, _perf_base, "all high-tau", "all")
    _plot_all_options(_hm_sites_filt, _perf_base, "high-tau filtered (max SSU >= 0.1)", "filt")
    log.info("model output performance plots: done")

    # --- composed figure: heatmap (z-score only) + bar plot (MAE), tissues aligned ---
    from matplotlib.image import imread as _imread_c
    from matplotlib.gridspec import GridSpec as _GS_comp

    for suffix, site_label in [("all", "all high-tau"), ("filt", "filtered (max SSU >= 0.1)")]:
        hm_path = fig2_main / f"high_tau_ssu_heatmap{'_filtered' if suffix == 'filt' else ''}.png"
        bar_path = fig2_main / f"perf_B_bars_mae_{suffix}.png"
        if not hm_path.exists() or not bar_path.exists():
            log.info(f"  composed {suffix}: missing panels, skipping")
            continue

        hm_img = _imread_c(str(hm_path))
        bar_img = _imread_c(str(bar_path))

        # figure with 2 rows: heatmap on top, bars below
        hm_h, hm_w = hm_img.shape[:2]
        bar_h, bar_w = bar_img.shape[:2]
        # scale to same width, compute relative heights
        hm_aspect = hm_h / hm_w
        bar_aspect = bar_h / bar_w
        fig_w = 18  # inches
        hm_h_in = fig_w * hm_aspect
        bar_h_in = fig_w * bar_aspect
        total_h = hm_h_in + bar_h_in + 0.8  # padding

        fig = plt.figure(figsize=(fig_w, total_h))
        gs = _GS_comp(2, 1, figure=fig, height_ratios=[hm_h_in, bar_h_in],
                       hspace=0.05)

        ax_hm = fig.add_subplot(gs[0])
        ax_hm.imshow(hm_img)
        ax_hm.axis("off")

        ax_bar = fig.add_subplot(gs[1])
        ax_bar.imshow(bar_img)
        ax_bar.axis("off")

        # panel labels
        fig.text(0.02, 0.98, "a", fontsize=16, fontweight="bold", va="top",
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor="lightgrey", edgecolor="none"))
        fig.text(0.02, bar_h_in / total_h + 0.02, "b", fontsize=16, fontweight="bold", va="top",
                 bbox=dict(boxstyle="circle,pad=0.3", facecolor="lightgrey", edgecolor="none"))

        plt.tight_layout()
        _save_fig(plt, fig2_main / f"composed_tau_perf_{suffix}")
        log.info(f"composed tau+perf ({suffix}): done")

# --- figure: error by specificity direction (inclusion vs exclusion) ---
df_eval_dir = df_eval_dom.copy()
df_eval_dir = df_eval_dir.merge(
    _df_tau_5[["chrom", "pos", "strand", "specificity_direction", "_min_tissue"]],
    on=["chrom", "pos", "strand"], how="left")
df_eval_dir_ht = df_eval_dir[df_eval_dir.tau >= _high_tau_thresh]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for col_i, (models, pcols_d, title) in enumerate([
    (_reg_models, _reg_pred_cols, "Regression"),
    (_cls_models, _cls_pred_cols, "Classification"),
]):
    ax = axes[col_i]
    _dirs = ["inclusion", "exclusion"]
    _x_dir = np.arange(len(_dirs))
    _w = 0.8 / len(models)
    for mi, mo in enumerate(models):
        col = pcols_d[mo]
        vals = []
        for d in _dirs:
            sub = df_eval_dir_ht[df_eval_dir_ht.specificity_direction == d]
            y = sub["y_ssu"].values; p = sub[col].values
            ok = np.isfinite(p) & np.isfinite(y) & (y != SENTINEL)
            yv, pv = y[ok], p[ok]
            vals.append(float(np.median(np.abs(yv - pv))) if len(yv) >= 10 else np.nan)
        ax.bar(_x_dir + mi * _w - 0.4 + _w / 2, vals, _w,
               color=get_color(mo), label=_model_display[mo], alpha=0.8)
    ax.set_xticks(_x_dir)
    ax.set_xticklabels(["inclusion\n(high in 1)", "exclusion\n(low in 1)"])
    ax.set_ylabel("Median AE" if col_i == 0 else "")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=True, edgecolor="lightgray")
    ax.grid(alpha=0.15, axis="y")
fig.suptitle(f"prediction error by specificity direction (high-tau, tau >= {_high_tau_thresh:.2f})",
             fontsize=13, y=1.02)
plt.tight_layout()
_save_fig(plt, fig2_main / "error_by_specificity_direction")
log.info("error by specificity direction: done")

# save site-level info
_df_tau_5[["chrom", "pos", "strand", "tau", "outlier_tissue",
           "outlier_tissue", "_min_tissue",
           "specificity_direction", "specificity_ratio", "depletion_ratio",
           "max_ssu", "n_active"]].to_csv(results_dir / "site_tau_outlier.csv", index=False)
log.info("saved site_tau_outlier.csv")

# save all metrics CSVs
_tau5_rows = []
for bi, b in enumerate(_tau5_bins):
    for mo in _reg_models + _cls_models:
        _src = _t5_reg if mo in _reg_models else _t5_cls
        _tau5_rows.append({
            "tau_bin": str(b), "model": _model_display[mo],
            "output_type": "regression" if mo in _reg_models else "classification",
            "r2": _src["r2"][mo][bi], "pearson": _src["pearson"][mo][bi],
            "mse": _src["mse"][mo][bi], "mae": _src["mae"][mo][bi],
        })
pd.DataFrame(_tau5_rows).to_csv(results_dir / "tau_metrics_5tissue.csv", index=False)
log.info("saved tau_metrics_5tissue.csv")

# =====================================================================
# TISSUE-MATCHED HEAD ANALYSIS
# do tissue-specific model heads outperform avg heads on matched sites?
# =====================================================================
log.info("\n" + "=" * 60)
log.info("tissue-matched head analysis")

import pyarrow.parquet as pq

# tissue-specific columns to extract from full parquets
_pang_tissue_heads = {
    "heart": ("heart_usage", "heart_p_splice"),
    "liver": ("liver_usage", "liver_p_splice"),
    "brain": ("brain_usage", "brain_p_splice"),
    "testis": ("testis_usage", "testis_p_splice"),
}
_spt_tissue_heads = [
    "adipose", "blood", "blood_vessel", "brain", "colon", "heart", "kidney",
    "liver", "lung", "muscle", "nerve", "small_intestine", "skin", "spleen", "stomach",
]

# map evaluation tissues to closest model heads
_eval_to_pang = {
    "haec10": "heart",       # HAEC ~ heart-derived endothelial
    "lung": None,            # no lung head in pangolin
    "testis": "testis",
    "brain_cortex": "brain",
    "whole_blood": None,     # no blood head in pangolin
}
_eval_to_spt = {
    "haec10": "heart",       # closest match
    "lung": "lung",
    "testis": None,          # no testis head in splicetransformer
    "brain_cortex": "brain",
    "whole_blood": "blood",
}

# cache path
_head_cache = Path("data/tissue_head_cache.parquet")

if _head_cache.exists():
    log.info(f"loading cached tissue head predictions from {_head_cache}")
    _df_heads = pd.read_parquet(_head_cache)
    log.info(f"  {len(_df_heads):,} rows, {len(_df_heads.columns)} columns")
else:
    log.info("building tissue head predictions (first run — will cache)")
    _head_rows = []

    for tissue in tissues_fig1:
        pred_dir = pred_base / tissue / "ml_out_var" / "predictions"
        sample_dirs = sorted(pred_dir.glob("test_*"))
        log.info(f"  {tissue}: {len(sample_dirs)} samples")

        for sdir in tqdm(sample_dirs, desc=tissue):
            # find pangolin and splicetransformer full parquets
            pang_pq = [p for p in sdir.glob("*_pang.parquet")]
            spt_pq = [p for p in sdir.glob("*_spt.parquet")]

            if not pang_pq or not spt_pq:
                continue

            # read splice site positions from slim parquet
            slim = sdir / "splice_sites.parquet"
            if not slim.exists():
                continue
            df_slim = pd.read_parquet(slim, columns=["chrom", "pos", "strand", "y_ssu"])

            # pangolin: read tissue-specific heads
            ptbl = pq.read_table(pang_pq[0])
            p_gc = ptbl.column("chrom").to_numpy() if "chrom" in ptbl.schema.names else None

            # need to find splice site indices in the full parquet
            # use y_ssu to build the mask (same logic as extract_splice_sites)
            p_y_acc = ptbl.column("y_acceptor").to_numpy()
            p_y_don = ptbl.column("y_donor").to_numpy()
            p_y_ssu = ptbl.column("y_ssu").to_numpy()
            mask = ((p_y_acc == 1) | (p_y_don == 1)) & (p_y_ssu != SENTINEL)
            idx = np.where(mask)[0]

            row = df_slim.copy()
            row["tissue"] = tissue
            row["sample"] = sdir.name

            # extract pangolin tissue heads at splice site positions
            for pang_tissue, (usage_col, cls_col) in _pang_tissue_heads.items():
                if usage_col in ptbl.schema.names:
                    row[f"pang_{pang_tissue}_usage"] = ptbl.column(usage_col).to_numpy()[idx]
                if cls_col in ptbl.schema.names:
                    row[f"pang_{pang_tissue}_cls"] = ptbl.column(cls_col).to_numpy()[idx]
            del ptbl

            # splicetransformer tissue heads
            stbl = pq.read_table(spt_pq[0])
            s_y_acc = stbl.column("y_acceptor").to_numpy()
            s_y_don = stbl.column("y_donor").to_numpy()
            s_y_ssu = stbl.column("y_ssu").to_numpy()
            s_mask = ((s_y_acc == 1) | (s_y_don == 1)) & (s_y_ssu != SENTINEL)
            s_idx = np.where(s_mask)[0]

            for spt_tissue in _spt_tissue_heads:
                if spt_tissue in stbl.schema.names:
                    row[f"spt_{spt_tissue}"] = stbl.column(spt_tissue).to_numpy()[s_idx]
            del stbl

            _head_rows.append(row)

    _df_heads = pd.concat(_head_rows, ignore_index=True)
    _df_heads.to_parquet(_head_cache, index=False)
    log.info(f"  cached {len(_df_heads):,} rows to {_head_cache}")

# merge with tau info (5-tissue sites)
_df_heads_tau = _df_heads.merge(
    _df_tau_5[["chrom", "pos", "strand", "tau", "outlier_tissue",
               "specificity_direction"]],
    on=["chrom", "pos", "strand"], how="inner")
log.info(f"head predictions with tau: {len(_df_heads_tau):,} rows")
log.info(f"  unique sites: {_df_heads_tau.groupby(['chrom','pos','strand']).ngroups:,}")

# --- compute matched vs avg head metrics by outlier tissue ---
# for each eval tissue, compare: avg head vs matched tissue head vs best available head

_head_results = []

for eval_tissue in tissues_fig1:
    sub = _df_heads_tau[_df_heads_tau.tissue == eval_tissue]
    if len(sub) == 0:
        continue
    y = sub["y_ssu"].values
    ok = np.isfinite(y) & (y != SENTINEL)

    # pangolin heads
    pang_match = _eval_to_pang.get(eval_tissue)
    pang_avg_col = sub.get("pang_heart_usage")  # placeholder, use from slim
    # avg is in slim parquet as pangolin_avg_usage — but we need it from _df_heads
    # it's not in _df_heads, pull from df_all
    _sub_avg = df_all[(df_all.tissue == eval_tissue)].copy()
    _sub_avg = _sub_avg.drop_duplicates(["chrom", "pos", "strand", "sample"])

    # simpler: just compute per-head metrics directly
    for pang_t, (usage_col, cls_col) in _pang_tissue_heads.items():
        u_col = f"pang_{pang_t}_usage"
        c_col = f"pang_{pang_t}_cls"
        if u_col not in sub.columns:
            continue
        p = sub[u_col].values
        ok_p = ok & np.isfinite(p)
        yv, pv = y[ok_p], p[ok_p]
        if len(yv) < 50:
            continue
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        _head_results.append({
            "eval_tissue": eval_tissue,
            "model": "Pangolin",
            "head": f"{pang_t} (usage)",
            "matched": pang_t == pang_match,
            "r2": 1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan,
            "pearson": float(_pearsonr(pv, yv)[0]),
            "mse": float(np.mean((yv - pv)**2)),
            "mae": float(np.median(np.abs(yv - pv))),
            "n": len(yv),
        })

    for spt_t in _spt_tissue_heads:
        s_col = f"spt_{spt_t}"
        if s_col not in sub.columns:
            continue
        p = sub[s_col].values
        ok_p = ok & np.isfinite(p)
        yv, pv = y[ok_p], p[ok_p]
        if len(yv) < 50:
            continue
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        spt_match = _eval_to_spt.get(eval_tissue)
        _head_results.append({
            "eval_tissue": eval_tissue,
            "model": "SpliceTransformer",
            "head": spt_t,
            "matched": spt_t == spt_match,
            "r2": 1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan,
            "pearson": float(_pearsonr(pv, yv)[0]),
            "mse": float(np.mean((yv - pv)**2)),
            "mae": float(np.median(np.abs(yv - pv))),
            "n": len(yv),
        })

_df_head_results = pd.DataFrame(_head_results)
_df_head_results.to_csv(results_dir / "tissue_head_metrics.csv", index=False)
log.info(f"saved tissue_head_metrics.csv: {len(_df_head_results)} rows")

# --- same analysis but restricted to high-tau sites grouped by outlier tissue ---
_head_results_tau = []

for out_tissue in tissues_fig1:
    sub_out = _df_heads_tau[
        (_df_heads_tau.outlier_tissue == out_tissue) &
        (_df_heads_tau.tau >= _df_tau_5.tau.quantile(0.5))  # top 50% tau
    ]
    if len(sub_out) < 100:
        continue

    y = sub_out["y_ssu"].values
    ok = np.isfinite(y) & (y != SENTINEL)

    for pang_t, (usage_col, cls_col) in _pang_tissue_heads.items():
        u_col = f"pang_{pang_t}_usage"
        if u_col not in sub_out.columns:
            continue
        p = sub_out[u_col].values
        ok_p = ok & np.isfinite(p)
        yv, pv = y[ok_p], p[ok_p]
        if len(yv) < 50:
            continue
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        _head_results_tau.append({
            "outlier_tissue": out_tissue,
            "model": "Pangolin",
            "head": f"{pang_t} (usage)",
            "matched": pang_t == out_tissue or (out_tissue == "haec10" and pang_t == "heart"),
            "r2": 1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan,
            "pearson": float(_pearsonr(pv, yv)[0]),
            "mse": float(np.mean((yv - pv)**2)),
            "mae": float(np.median(np.abs(yv - pv))),
            "n": len(yv),
        })

    for spt_t in _spt_tissue_heads:
        s_col = f"spt_{spt_t}"
        if s_col not in sub_out.columns:
            continue
        p = sub_out[s_col].values
        ok_p = ok & np.isfinite(p)
        yv, pv = y[ok_p], p[ok_p]
        if len(yv) < 50:
            continue
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        _head_results_tau.append({
            "outlier_tissue": out_tissue,
            "model": "SpliceTransformer",
            "head": spt_t,
            "matched": (spt_t == _eval_to_spt.get(out_tissue)),
            "r2": 1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan,
            "pearson": float(_pearsonr(pv, yv)[0]),
            "mse": float(np.mean((yv - pv)**2)),
            "mae": float(np.median(np.abs(yv - pv))),
            "n": len(yv),
        })

_df_head_tau = pd.DataFrame(_head_results_tau)
_df_head_tau.to_csv(results_dir / "tissue_head_metrics_by_outlier.csv", index=False)
log.info(f"saved tissue_head_metrics_by_outlier.csv: {len(_df_head_tau)} rows")

# --- figure: per-outlier-tissue detail (all heads + SSU + counts) ---
# one figure per outlier tissue: all Pangolin and SpT heads as horizontal bars,
# with SSU distribution and site count inset

for out_tissue in tissues_fig1:
    sub_out = _df_heads_tau[
        (_df_heads_tau.outlier_tissue == out_tissue) &
        (_df_heads_tau.tau >= _df_tau_5.tau.quantile(0.5))
    ]
    if len(sub_out) < 50:
        log.info(f"  {out_tissue}: too few high-tau sites ({len(sub_out)}), skipping")
        continue

    n_sites_dom = sub_out.groupby(["chrom", "pos", "strand"]).ngroups
    y = sub_out["y_ssu"].values
    ok_base = np.isfinite(y) & (y != SENTINEL)

    # collect all head R² values
    _bars = []
    # pangolin heads
    for pt, (ucol, ccol) in _pang_tissue_heads.items():
        for col_type, suffix in [(f"pang_{pt}_usage", f"{pt} (usage)"),
                                  (f"pang_{pt}_cls", f"{pt} (p_splice)")]:
            if col_type not in sub_out.columns:
                continue
            p = sub_out[col_type].values
            ok = ok_base & np.isfinite(p)
            yv, pv = y[ok], p[ok]
            if len(yv) < 50:
                continue
            ss = np.sum((yv - yv.mean())**2)
            r2 = 1 - np.sum((yv - pv)**2) / ss if ss > 0 else np.nan
            pear = float(_pearsonr(pv, yv)[0])
            mse = float(np.mean((yv - pv)**2))
            mae = float(np.median(np.abs(yv - pv)))
            matched = (pt == _eval_to_pang.get(out_tissue))
            _bars.append({"model": "Pangolin", "head": suffix,
                          "r2": r2, "pearson": pear, "mse": mse, "mae": mae,
                          "matched": matched, "n": len(yv)})

    # splicetransformer heads
    for st in _spt_tissue_heads:
        col = f"spt_{st}"
        if col not in sub_out.columns:
            continue
        p = sub_out[col].values
        ok = ok_base & np.isfinite(p)
        yv, pv = y[ok], p[ok]
        if len(yv) < 50:
            continue
        ss = np.sum((yv - yv.mean())**2)
        r2 = 1 - np.sum((yv - pv)**2) / ss if ss > 0 else np.nan
        pear = float(_pearsonr(pv, yv)[0])
        mse = float(np.mean((yv - pv)**2))
        mae = float(np.median(np.abs(yv - pv)))
        matched = (st == _eval_to_spt.get(out_tissue))
        _bars.append({"model": "SpliceTransformer", "head": st,
                      "r2": r2, "pearson": pear, "mse": mse, "mae": mae,
                      "matched": matched, "n": len(yv)})

    if not _bars:
        continue
    _df_bars = pd.DataFrame(_bars)

    # plot: 3 columns — Pangolin heads | SpliceTransformer heads | SSU distribution
    from matplotlib.gridspec import GridSpec as _GS
    fig = plt.figure(figsize=(20, 8))
    gs = _GS(1, 3, figure=fig, width_ratios=[1, 1.5, 0.8], wspace=0.35)

    # pangolin panel
    ax_p = fig.add_subplot(gs[0])
    pang = _df_bars[_df_bars.model == "Pangolin"].sort_values("r2", ascending=True)
    if len(pang) > 0:
        ypos = np.arange(len(pang))
        colors = ["#D55E00" if m else GRAY_FAINT for m in pang.matched]
        ax_p.barh(ypos, pang.r2.values, color=colors, edgecolor="white", height=0.7)
        for i, (r2v, maev) in enumerate(zip(pang.r2.values, pang.mae.values)):
            ax_p.text(max(r2v, 0) + 0.01, i, f"R²={r2v:.3f}  MAE={maev:.3f}",
                      va="center", fontsize=7)
        ax_p.set_yticks(ypos)
        ax_p.set_yticklabels(pang["head"].values, fontsize=8)
        ax_p.set_xlabel("$R^2$")
        ax_p.set_title("Pangolin")
        ax_p.axvline(0, color="black", lw=0.5)

    # splicetransformer panel
    ax_s = fig.add_subplot(gs[1])
    spt = _df_bars[_df_bars.model == "SpliceTransformer"].sort_values("r2", ascending=True)
    if len(spt) > 0:
        ypos = np.arange(len(spt))
        colors = ["#D55E00" if m else GRAY_FAINT for m in spt.matched]
        ax_s.barh(ypos, spt.r2.values, color=colors, edgecolor="white", height=0.7)
        for i, (r2v, maev) in enumerate(zip(spt.r2.values, spt.mae.values)):
            ax_s.text(max(r2v, 0) + 0.01, i, f"R²={r2v:.3f}  MAE={maev:.3f}",
                      va="center", fontsize=7)
        ax_s.set_yticks(ypos)
        ax_s.set_yticklabels(spt["head"].values, fontsize=8)
        ax_s.set_xlabel("$R^2$")
        ax_s.set_title("SpliceTransformer")
        ax_s.axvline(0, color="black", lw=0.5)

    # SSU distribution panel
    ax_d = fig.add_subplot(gs[2])
    _ssu_dom = sub_out.drop_duplicates(["chrom", "pos", "strand"])["y_ssu"]
    # get tissue-specific SSU (mean SSU in the outlier tissue only)
    _out_ssu = _site_tissue_all[
        (_site_tissue_all.tissue == out_tissue)
    ].merge(
        sub_out[["chrom", "pos", "strand"]].drop_duplicates(),
        on=["chrom", "pos", "strand"]
    )["mean_ssu"]
    ax_d.hist(_out_ssu.values, bins=40, color=tissue_colors.get(out_tissue, "#999"),
              edgecolor="white", alpha=0.8, orientation="horizontal")
    ax_d.set_ylabel("mean SSU in " + tissue_display.get(out_tissue, out_tissue))
    ax_d.set_xlabel("n sites")
    ax_d.set_title(f"SSU dist (n={n_sites_dom:,})")

    tname = tissue_display.get(out_tissue, out_tissue)
    fig.suptitle(f"all model heads — sites dominated by {tname} "
                 f"(tau ≥ {_df_tau_5.tau.quantile(0.5):.2f}, n={n_sites_dom:,} sites)\n"
                 f"red = tissue-matched head",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    _save_fig(plt, fig2_main / f"all_heads_{out_tissue}")
    log.info(f"all heads {out_tissue}: done ({len(_bars)} heads)")

# --- option A: grid scatter (matched R² vs best other R²) ---
if len(_df_head_tau) > 0:
    _families = ["Pangolin", "SpliceTransformer"]
    _metrics_th = [("mae", "MAE"), ("pearson", "Pearson r")]

    for met, ylabel in _metrics_th:
        fig, axes = plt.subplots(1, len(_families), figsize=(7 * len(_families), 6), squeeze=False)
        for mi, model in enumerate(_families):
            ax = axes[0, mi]
            msub = _df_head_tau[_df_head_tau.model == model]
            if len(msub) == 0:
                ax.set_visible(False)
                continue
            out_tissues = sorted(msub.outlier_tissue.unique())
            _tc = {t: tissue_colors.get(t, "#999") for t in out_tissues}
            for _, row in msub.iterrows():
                # matched value
                if not row["matched"]:
                    continue
                dt = row["outlier_tissue"]
                matched_val = row[met]
                # best other for same outlier tissue
                others = msub[(msub.outlier_tissue == dt) & (~msub.matched)]
                if met == "mae":
                    best_other = others[met].min() if len(others) > 0 else np.nan
                else:
                    best_other = others[met].max() if len(others) > 0 else np.nan
                ax.scatter(matched_val, best_other, color=_tc[dt], s=60, alpha=0.8,
                           edgecolors="white", linewidths=0.5, zorder=3)
                ax.annotate(tissue_display.get(dt, dt), (matched_val, best_other),
                            fontsize=8, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")

            all_vals = msub[met].dropna()
            if len(all_vals) > 0:
                lo, hi = all_vals.min(), all_vals.max()
                buf = (hi - lo) * 0.1
                lims = [lo - buf, hi + buf]
                ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
                ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f"matched head {ylabel}")
            ax.set_ylabel(f"best other head {ylabel}")
            ax.set_title(model)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            # count above diagonal
            _above = "below" if met == "mae" else "above"

        fig.suptitle(f"tissue-matched vs best other head ({ylabel}, high-tau sites)",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        _save_fig(plt, fig2_main / f"th_scatter_{met}")
    log.info("tissue head scatter: done")

    # --- option C: delta swarm (matched - best other) ---
    for met, ylabel in _metrics_th:
        fig, axes = plt.subplots(1, len(_families), figsize=(6 * len(_families), 5), squeeze=False)
        _rng_sw = np.random.default_rng(42)
        for mi, model in enumerate(_families):
            ax = axes[0, mi]
            msub = _df_head_tau[_df_head_tau.model == model]
            if len(msub) == 0:
                ax.set_visible(False)
                continue
            out_tissues = [t for t in tissues_fig1 if t in msub.outlier_tissue.unique()]
            for di, dt in enumerate(out_tissues):
                dt_sub = msub[msub.outlier_tissue == dt]
                matched = dt_sub[dt_sub.matched]
                others = dt_sub[~dt_sub.matched]
                if len(matched) == 0:
                    continue
                m_val = matched[met].values[0]
                # delta vs each other head
                for _, orow in others.iterrows():
                    if met == "mae":
                        delta = orow[met] - m_val  # positive = matched is better (lower mae)
                    else:
                        delta = m_val - orow[met]  # positive = matched is better (higher pearson)
                    jx = di + _rng_sw.normal(0, 0.06)
                    ax.scatter(jx, delta, color=tissue_colors.get(dt, "#999"),
                               s=25, alpha=0.6, edgecolors="white", linewidths=0.3)
                # median delta
                if met == "mae":
                    deltas = others[met].values - m_val
                else:
                    deltas = m_val - others[met].values
                med = np.median(deltas)
                ax.hlines(med, di - 0.3, di + 0.3, colors="black", lw=2)
                ax.text(di + 0.35, med, f"{med:.3f}", va="center", fontsize=8)

            ax.axhline(0, color="red", ls="--", lw=1, alpha=0.5)
            ax.set_xticks(range(len(out_tissues)))
            ax.set_xticklabels([tissue_display.get(t, t) for t in out_tissues], fontsize=9)
            ax.set_ylabel(f"Δ{ylabel} (matched advantage)")
            ax.set_title(model)
            ax.grid(alpha=0.2, axis="y")

        fig.suptitle(f"tissue head advantage ({ylabel}, high-tau sites)\n"
                     f"positive = matched head is better", fontsize=13, y=1.03)
        plt.tight_layout()
        _save_fig(plt, fig2_main / f"th_delta_swarm_{met}")
    log.info("tissue head delta swarm: done")

    # --- option E: heatmap (eval tissue × all heads) ---
    for model in _families:
        msub_all = _df_head_results[_df_head_results.model == model]
        if len(msub_all) == 0:
            continue
        _eval_ts = [t for t in tissues_fig1 if t in msub_all.eval_tissue.unique()]
        _heads = sorted(msub_all["head"].unique())
        if not _eval_ts or not _heads:
            continue

        for met, ylabel in _metrics_th:
            mat = np.full((len(_eval_ts), len(_heads)), np.nan)
            for ri, et in enumerate(_eval_ts):
                for ci, h in enumerate(_heads):
                    row = msub_all[(msub_all.eval_tissue == et) & (msub_all["head"] == h)]
                    if len(row) > 0:
                        mat[ri, ci] = row[met].values[0]

            fig, ax = plt.subplots(figsize=(max(8, len(_heads) * 0.6),
                                            max(3, len(_eval_ts) * 0.8)))
            im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r" if met == "mae" else "YlOrRd",
                           interpolation="nearest")
            ax.set_xticks(range(len(_heads)))
            ax.set_xticklabels(_heads, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(_eval_ts)))
            ax.set_yticklabels([tissue_display.get(t, t) for t in _eval_ts], fontsize=9)
            # annotate values and highlight matched
            for ri, et in enumerate(_eval_ts):
                matched_head_map = _eval_to_pang if model == "Pangolin" else _eval_to_spt
                matched_t = matched_head_map.get(et)
                for ci, h in enumerate(_heads):
                    v = mat[ri, ci]
                    if np.isnan(v):
                        continue
                    # check if this is the matched head
                    is_matched = False
                    if matched_t is not None:
                        if model == "Pangolin" and matched_t in h:
                            is_matched = True
                        elif model == "SpliceTransformer" and h == matched_t:
                            is_matched = True
                    color = "white" if v > np.nanmedian(mat) else "black"
                    weight = "bold" if is_matched else "normal"
                    ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                            fontsize=6, color=color, fontweight=weight)
                    if is_matched:
                        from matplotlib.patches import Rectangle as _Rect
                        ax.add_patch(_Rect((ci - 0.5, ri - 0.5), 1, 1,
                                           fill=False, edgecolor="red", lw=2))
            plt.colorbar(im, ax=ax, label=ylabel, shrink=0.8)
            ax.set_title(f"{model} — {ylabel} per eval tissue × head")
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"th_heatmap_{model.lower()}_{met}")
        log.info(f"tissue head heatmap {model}: done")

log.info("tissue-matched head analysis: done\n")
log.info("tissue specificity analysis: done\n")

if not _only_tau:  # skip post-tau analyses
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize
    
    # density plots — one figure per tissue per subset (skip with --no-plot)
    grid = [
        [("SPLAIRE", "splaire_reg"), ("Pangolin", "pangolin_reg"), ("SpliceTransformer", "spt_reg"), None],
        [("SPLAIRE", "splaire_cls"), ("Pangolin", "pangolin_cls"), ("SpliceTransformer", "spt_cls"), ("SpliceAI", "spliceai_cls")],
    ]
    row_labels = ["Regression", "Classification"]
    
    _title_names = {
        "haec10": "HAEC", "lung": "Lung", "testis": "Testis",
        "brain_cortex": "Brain Cortex", "whole_blood": "Whole Blood",
    }
    _subset_labels = {
        "ssu_valid": "All Splice Sites",
        "ssu_valid_nonzero": "Nonzero SSU",
    }
    
    sns.set(font_scale=1.2)
    vmax = 11
    _ticks = [0, 0.25, 0.5, 0.75, 1.0]
    _ticklabels = ["0", ".25", ".5", ".75", "1"]
    KDE_MAX = 50_000  # subsample for hist2d display, stats on full data
    rng = np.random.default_rng(42)
    
    if _no_plot:
        log.info("skipping density plots (--no-plot)")
    
    for tissue in ([] if _no_plot else tissues):
        td = tissue_data[tissue]
        y_ssu = td["y_ssu"]
        p = td["preds"]
        tname = _title_names.get(tissue, tissue)
    
        for subset_name in ["ssu_valid", "ssu_valid_nonzero"]:
            subset_mask = td["masks"][subset_name]
            y_true = y_ssu[subset_mask]
            slabel = _subset_labels[subset_name]
    
            # shared subsample index for all panels
            n = len(y_true)
            if n > KDE_MAX:
                kde_idx = rng.choice(n, KDE_MAX, replace=False)
            else:
                kde_idx = np.arange(n)
            y_true_kde = y_true[kde_idx]
    
            fig = plt.figure(figsize=(16, 7.5))
            gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.25, hspace=0.35)
    
            for row_i, row in enumerate(grid):
                for col_i, entry in enumerate(row):
                    if entry is None:
                        continue
                    title, key = entry
                    ax = fig.add_subplot(gs[row_i, col_i])
                    pred_full = p[key][subset_mask]
                    pred_kde = pred_full[kde_idx]
                    sns.kdeplot(x=pred_kde, y=y_true_kde, ax=ax, fill=True, cmap="turbo", cbar=False, thresh=0)
                    norm = Normalize(0, vmax)
                    for collection in ax.collections:
                        collection.set_norm(norm)
                    ax.plot([0, 1], [0, 1], "--", color="white", lw=0.8, alpha=0.7)
                    # stats on full data
                    r = stats.pearsonr(pred_full, y_true)[0]
                    r2 = 1 - np.sum((y_true - pred_full)**2) / np.sum((y_true - y_true.mean())**2)
                    ax.text(0.05, 0.92, f"r = {r:.3f}\nR² = {r2:.3f}", transform=ax.transAxes,
                            fontsize=10, color="white", va="top")
                    ax.set_title(title, fontsize=12)
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.set_xticks(_ticks); ax.set_yticks(_ticks)
                    ax.tick_params(which="both", direction="out", length=4)
                    if row_i == 1:
                        ax.set_xticklabels(_ticklabels)
                        ax.set_xlabel("Predicted SSU")
                    else:
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
                    if col_i == 0:
                        ax.set_yticklabels(_ticklabels)
                        ax.set_ylabel("True SSU")
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
    
            # row labels
            for row_i, label in enumerate(row_labels):
                fig.text(0.02, 0.72 - row_i * 0.44, label, va="center", ha="center",
                         fontsize=14, fontweight="bold", rotation=90)
    
            # colorbar
            cax = fig.add_subplot(gs[:, 4])
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=Normalize(0, vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Density", fontsize=12, rotation=270, labelpad=14)
    
            n_sites = subset_mask.sum()
            n_samples = len(list((pred_base / tissue / "ml_out_var" / "predictions").glob("test_*")))
            fig.suptitle(f"{tname} \u2014 {slabel} \u2014 {n_samples} individuals, {n_sites:,} sites",
                         fontsize=14, y=1.01)
            _save_fig(fig, fig2_sup / f"density_{tissue}_{subset_name}")
            log.info(f"{tname} / {slabel}: {n_sites:,} sites\n")
    
    sns.set(font_scale=1.0)
    
    # pangolin test set: load metrics from JSON files
    # replaces old h5-based loading
    
    pang_metric_base = Path(os.environ.get("SPLAIRE_CANONICAL_DIR", "/scratch/runyan.m/splaire_out/canonical")) / "pangolin"
    pang_tissues = ["heart", "liver", "brain", "testis"]
    pang_tissue_display = {"heart": "Heart", "liver": "Liver", "brain": "Brain", "testis": "Testis"}
    
    pang_cls_rows = []
    pang_reg_rows = []
    
    for tissue in pang_tissues:
        jpath = pang_metric_base / f"{tissue}_metrics.json"
        if not jpath.exists():
            log.info(f"  {tissue}: metrics not found at {jpath}")
            continue
    
        with open(jpath) as f:
            d = json.load(f)
    
        ov = d.get("overall", {})
    
        # classification metrics
        cls = ov.get("classification", {})
        for subset, targets in cls.items():
            for target, outputs in targets.items():
                for output, metrics in outputs.items():
                    pang_cls_rows.append({
                        "tissue": tissue, "subset": subset, "target": target,
                        "output": output,
                        "auprc": metrics.get("auprc", np.nan),
                        "auroc": metrics.get("auroc", np.nan),
                        "topk": metrics.get("topk", np.nan),
                        "f1_max": metrics.get("f1_max", np.nan),
                        "n_pos": metrics.get("n_pos", 0),
                        "n_neg": metrics.get("n_neg", 0),
                    })
    
        # regression metrics
        reg = ov.get("regression", {})
        for subset, outputs in reg.items():
            for output, metrics in outputs.items():
                pang_reg_rows.append({
                    "tissue": tissue, "subset": subset, "output": output,
                    "pearson": metrics.get("pearson", np.nan),
                    "spearman": metrics.get("spearman", np.nan),
                    "r2": metrics.get("r2", np.nan),
                    "mse": metrics.get("mse", np.nan),
                    "mae": metrics.get("mae", np.nan),
                    "n": metrics.get("n", 0),
                })
    
    df_pang_cls = pd.DataFrame(pang_cls_rows)
    df_pang_reg = pd.DataFrame(pang_reg_rows)
    
    log.info(f"pangolin benchmark metrics:")
    log.info(f"  classification: {df_pang_cls.shape}, tissues: {sorted(df_pang_cls.tissue.unique())}")
    log.info(f"  regression: {df_pang_reg.shape}, tissues: {sorted(df_pang_reg.tissue.unique())}")
    if len(df_pang_cls) > 0:
        log.info(f"  cls outputs: {sorted(df_pang_cls.output.unique())}")
    if len(df_pang_reg) > 0:
        log.info(f"  reg outputs: {sorted(df_pang_reg.output.unique())}")
    
    # save pangolin benchmark summary
    if len(df_pang_cls) > 0 or len(df_pang_reg) > 0:
        _pang_parts = []
        if len(df_pang_cls) > 0:
            _pc = (df_pang_cls.groupby(["tissue", "output"])
                   .agg(auprc=("auprc", "mean"), auroc=("auroc", "mean"), topk=("topk", "mean"))
                   .reset_index())
            _pc["task"] = "classification"
            _pang_parts.append(_pc)
        if len(df_pang_reg) > 0:
            _pr = (df_pang_reg.groupby(["tissue", "subset", "output"])
                   .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"),
                        r2=("r2", "mean"), mse=("mse", "mean"))
                   .reset_index())
            _pr["task"] = "regression"
            _pang_parts.append(_pr)
        _pang_summary = pd.concat(_pang_parts, ignore_index=True)
        _pang_summary.to_csv(results_dir / "pangolin_benchmark.csv", index=False)
        log.info(f"saved pangolin_benchmark.csv: {_pang_summary.shape}")
    
    sns.set_style("white")
    
    # pangolin test set bar plots — horizontal only
    
    if len(df_pang_cls) == 0:
        log.info("no pangolin test data loaded — skipping bar plots")
    else:
        _pbar_outputs = ["splaire_ref", "spliceai", "pangolin_avg_p_splice", "splicetransformer"]
        _pbar_display = {"splaire_ref": "SPLAIRE", "spliceai": "SpliceAI",
                         "pangolin_avg_p_splice": "Pangolin", "splicetransformer": "SpliceTransformer"}
        _pr2_display = {"splaire_ref_ssu": "SPLAIRE", "spliceai_cls": "SpliceAI",
                        "pangolin_avg_usage": "Pangolin", "splicetransformer_avg_tissue": "SpliceTransformer"}
    
        pang_cls_order = [o for o in _pbar_outputs if o in df_pang_cls.output.unique()]
        pang_reg_order = [o for o in r2_outputs if o in df_pang_reg.output.unique()]
        avail_pang_bar_tissues = [t for t in pang_tissues if t in df_pang_cls.tissue.unique()]
        log.info(f"cls outputs: {pang_cls_order}")
        log.info(f"reg outputs: {pang_reg_order}")
        log.info(f"tissues: {avail_pang_bar_tissues}")
    
        def make_pang_bar(df, metric, ylabel, output_order, display_dict, tissues, suffix):
            # aggregate across subsets/targets to one value per tissue x output
            sub = df[df.tissue.isin(tissues) & df.output.isin(output_order)].copy()
            agg = sub.groupby(["tissue", "output"])[metric].mean().reset_index()
            nt = len(tissues)
            fig, axes = plt.subplots(1, nt, figsize=(3 * nt, 4), sharey=True)
            if nt == 1: axes = [axes]
            for col, tissue in enumerate(tissues):
                ax = axes[col]
                tdf = agg[agg.tissue == tissue].set_index("output").reindex(output_order)
                x = np.arange(len(output_order))
                colors = [get_color(o) for o in output_order]
                vals = tdf[metric].values
                bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                                f"{v:.2f}", ha="center", va="center", fontweight="bold",
                                rotation=90, color="white", fontsize=ANNOT_SIZE)
                ax.set_xlabel("")
                ax.set_ylabel(ylabel if col == 0 else "")
                ax.set_title(pang_tissue_display.get(tissue, tissue))
                ax.set_xticks(x)
                ax.set_xticklabels([display_dict.get(o, o) for o in output_order],
                                   rotation=45, ha="right")
                if metric in {"auprc", "topk", "f1_max"}: ax.set_ylim(0, 1.1)
            plt.tight_layout()
            _save_fig(plt, fig2_main / f"pang_{suffix}")
    
        # classification bar plots
        log.info("\n=== pangolin test classification ===")
        for metric, ylabel in [("topk", "Top-k"), ("auprc", "AUPRC"), ("f1_max", "F1-max")]:
            if metric not in df_pang_cls.columns: continue
            make_pang_bar(df_pang_cls, metric, ylabel, pang_cls_order, _pbar_display,
                          avail_pang_bar_tissues, f"{metric}_bars")
    
        # regression bar plots
        if len(df_pang_reg) > 0:
            log.info("\n=== pangolin test regression ===")
            for subset_name, suffix in [("ssu_valid", "all"), ("ssu_valid_nonzero", "nonzero")]:
                sub_reg = df_pang_reg[df_pang_reg.subset == subset_name]
                if sub_reg.empty: continue
                subset_label = "all sites" if suffix == "all" else "SSU > 0"
                for metric, ylabel in [("r2", f"$R^2$ ({subset_label})"),
                                       ("pearson", f"Pearson r ({subset_label})"),
                                       ("spearman", f"Spearman r ({subset_label})"),
                                       ("mse", f"MSE ({subset_label})")]:
                    if metric not in sub_reg.columns: continue
                    make_pang_bar(sub_reg, metric, ylabel, pang_reg_order, _pr2_display,
                                  avail_pang_bar_tissues, f"{metric}_bars_{suffix}")
    
        log.info(f"\nsaved pangolin bar plots to {fig2_main}")
    
    # pangolin benchmark: all-output heatmaps
    # uses plot_heatmap_all_outputs from section 9 and df_pang_cls/df_pang_reg
    
    if len(df_pang_cls) > 0 or len(df_pang_reg) > 0:
        _pang_heatmap_tissues = [t for t in pang_tissues if t in
                                  (df_pang_cls.tissue.unique().tolist() if len(df_pang_cls) > 0
                                   else df_pang_reg.tissue.unique().tolist())]
    
        # classification heatmaps
        if len(df_pang_cls) > 0:
            sub = df_pang_cls.copy()
            sub["base_model"] = sub.output.apply(_get_base_model)
            # aggregate across subsets/targets
            agg = sub.groupby(["tissue", "output"]).agg(
                auprc=("auprc", "mean"), auroc=("auroc", "mean"),
                topk=("topk", "mean"), f1_max=("f1_max", "mean"),
            ).reset_index()
            cl = agg.melt(id_vars=["tissue", "output"], var_name="metric", value_name="score")
            for m in ["auprc", "auroc", "topk", "f1_max"]:
                plot_heatmap_all_outputs(cl, m, title_suffix=" (Pangolin Benchmark)",
                                         fname_suffix="_pang", tissue_list=_pang_heatmap_tissues,
                                         out_dir=fig2_sup)
    
        # regression heatmaps
        if len(df_pang_reg) > 0:
            for subset, slabel in [("ssu_valid", "all sites"), ("ssu_valid_nonzero", "SSU > 0")]:
                sub = df_pang_reg.query("subset == @subset").copy()
                if len(sub) == 0: continue
                sub["base_model"] = sub.output.apply(_get_base_model)
                ac = ["pearson", "spearman", "r2", "mse"]
                if "mae" in sub.columns: ac.append("mae")
                agg = sub.groupby(["tissue", "output"]).agg({c: "mean" for c in ac}).reset_index()
                rl = agg.melt(id_vars=["tissue", "output"], var_name="metric", value_name="score")
                for m in ac:
                    plot_heatmap_all_outputs(rl, m, title_suffix=f" ({slabel}, Pangolin Benchmark)",
                                             fname_suffix=f"_{subset}_pang",
                                             tissue_list=_pang_heatmap_tissues, out_dir=fig2_sup)
    else:
        log.info("no pangolin metrics data for heatmaps")
    
    # pangolin benchmark — load predictions for all 4 tissues
    SENTINEL = 777.0
    pang_base = Path(os.environ.get("SPLAIRE_CANONICAL_DIR", "/scratch/runyan.m/splaire_out/canonical")) / "pangolin"
    pang_pred_dir = pang_base / "predictions"
    pang_tissues = ["heart", "liver", "brain", "testis"]
    
    # tissue-matched pangolin columns
    _pang_tissue_cols = {
        "heart": ("heart_usage", "heart_p_splice"),
        "liver": ("liver_usage", "liver_p_splice"),
        "brain": ("brain_usage", "brain_p_splice"),
        "testis": ("testis_usage", "testis_p_splice"),
    }
    
    pang_data = {}
    for tissue in pang_tissues:
        log.info(f"loading {tissue}...")
        prefix = tissue
    
        # ground truth from splaire ref
        gt = pd.read_parquet(pang_pred_dir / tissue / f"{prefix}_splaire_ref.parquet",
                             columns=["y_acceptor", "y_donor", "y_ssu"])
        is_splice = (gt["y_acceptor"] == 1) | (gt["y_donor"] == 1)
        mask = is_splice.values
        y_ssu = gt.loc[mask, "y_ssu"].values
    
        preds = {}
    
        # splaire
        df = pd.read_parquet(pang_pred_dir / tissue / f"{prefix}_splaire_ref.parquet",
                             columns=["acceptor", "donor", "ssu"])
        preds["splaire_reg"] = df["ssu"].values[mask]
        preds["splaire_cls"] = np.maximum(df["acceptor"].values[mask], df["donor"].values[mask])
    
        # spliceai
        df = pd.read_parquet(pang_pred_dir / tissue / f"{prefix}_sa.parquet",
                             columns=["acceptor", "donor"])
        preds["spliceai_cls"] = np.maximum(df["acceptor"].values[mask], df["donor"].values[mask])
    
        # pangolin — tissue-matched columns
        reg_col, cls_col = _pang_tissue_cols[tissue]
        df = pd.read_parquet(pang_pred_dir / tissue / f"{prefix}_pang.parquet",
                             columns=[reg_col, cls_col])
        preds["pangolin_reg"] = df[reg_col].values[mask]
        preds["pangolin_cls"] = df[cls_col].values[mask]
    
        # splicetransformer
        df = pd.read_parquet(pang_pred_dir / tissue / f"{prefix}_spt.parquet",
                             columns=["acceptor", "donor", "avg_tissue"])
        preds["spt_reg"] = df["avg_tissue"].values[mask]
        preds["spt_cls"] = np.maximum(df["acceptor"].values[mask], df["donor"].values[mask])
    
        valid = y_ssu != SENTINEL
        nonzero = y_ssu > 0
    
        pang_data[tissue] = {
            "y_ssu": y_ssu,
            "preds": preds,
            "masks": {
                "ssu_valid": valid,
                "ssu_valid_nonzero": valid & nonzero,
            },
        }
        for sname, smask in pang_data[tissue]["masks"].items():
            log.info(f"  {sname}: {smask.sum():,}")
    
    log.info("\ndone loading pangolin benchmark")
    
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize
    from scipy import stats
    
    # pangolin benchmark density plots — one figure per tissue per subset
    grid = [
        [("SPLAIRE", "splaire_reg"), ("Pangolin", "pangolin_reg"), ("SpliceTransformer", "spt_reg"), None],
        [("SPLAIRE", "splaire_cls"), ("Pangolin", "pangolin_cls"), ("SpliceTransformer", "spt_cls"), ("SpliceAI", "spliceai_cls")],
    ]
    row_labels = ["Regression", "Classification"]
    
    _pang_title_names = {
        "heart": "Heart", "liver": "Liver", "brain": "Brain", "testis": "Testis",
    }
    _subset_labels = {
        "ssu_valid": "All Splice Sites",
        "ssu_valid_nonzero": "Nonzero SSU",
    }
    
    sns.set(font_scale=1.2)
    vmax = 11
    _ticks = [0, 0.25, 0.5, 0.75, 1.0]
    _ticklabels = ["0", ".25", ".5", ".75", "1"]
    
    if _no_plot:
        log.info("skipping pangolin density plots (--no-plot)")
    
    for tissue in ([] if _no_plot else pang_tissues):
        td = pang_data[tissue]
        y_ssu = td["y_ssu"]
        p = td["preds"]
        tname = _pang_title_names[tissue]
    
        for subset_name in ["ssu_valid"]:
            subset_mask = td["masks"][subset_name]
            y_true = y_ssu[subset_mask]
            slabel = _subset_labels[subset_name]
    
            fig = plt.figure(figsize=(16, 7.5))
            gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.25, hspace=0.35)
    
            for row_i, row in enumerate(grid):
                for col_i, entry in enumerate(row):
                    if entry is None:
                        continue
                    title, key = entry
                    ax = fig.add_subplot(gs[row_i, col_i])
                    pred = p[key][subset_mask]
                    sns.kdeplot(x=pred, y=y_true, ax=ax, fill=True, cmap="turbo", cbar=False, thresh=0)
                    norm = Normalize(0, vmax)
                    for collection in ax.collections:
                        collection.set_norm(norm)
                    ax.plot([0, 1], [0, 1], "--", color="white", lw=0.8, alpha=0.7)
                    r = stats.pearsonr(pred, y_true)[0]
                    r2 = 1 - np.sum((y_true - pred)**2) / np.sum((y_true - y_true.mean())**2)
                    ax.text(0.05, 0.92, f"r = {r:.3f}\nR² = {r2:.3f}", transform=ax.transAxes,
                            fontsize=10, color="white", va="top")
                    ax.set_title(title, fontsize=12)
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.set_xticks(_ticks); ax.set_yticks(_ticks)
                    ax.tick_params(which="both", direction="out", length=4)
                    if row_i == 1:
                        ax.set_xticklabels(_ticklabels)
                        ax.set_xlabel("Predicted SSU")
                    else:
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
                    if col_i == 0:
                        ax.set_yticklabels(_ticklabels)
                        ax.set_ylabel("True SSU")
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
    
            # row labels
            for row_i, label in enumerate(row_labels):
                fig.text(0.02, 0.72 - row_i * 0.44, label, va="center", ha="center",
                         fontsize=14, fontweight="bold", rotation=90)
    
            # colorbar
            cax = fig.add_subplot(gs[:, 4])
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=Normalize(0, vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Density", fontsize=12, rotation=270, labelpad=14)
    
            n_sites = subset_mask.sum()
            fig.suptitle(f"Pangolin Benchmark \u2014 {tname} \u2014 {slabel} \u2014 {n_sites:,} sites",
                         fontsize=14, y=1.01)
            _save_fig(fig, fig2_sup / f"pang_density_{tissue}_{subset_name}")
            log.info(f"Pangolin {tname} / {slabel}: {n_sites:,} sites\n")
    
    sns.set(font_scale=1.0)
    
    # pangolin benchmark: binned metrics from JSON
    # loads binned AUPRC and binned MSE from the metrics JSONs
    
    pang_bin_ratio_rows = []
    pang_bin_reg_rows = []
    bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    
    for tissue in pang_tissues:
        jpath = pang_metric_base / f"{tissue}_metrics.json"
        if not jpath.exists():
            continue
        with open(jpath) as f:
            d = json.load(f)
        binned = d.get("overall", {}).get("binned", {})
    
        # binned classification (ratio-preserving)
        bcr = binned.get("classification_ratio", {})
        for subset, targets in bcr.items():
            for target, outputs in targets.items():
                for output, bins in outputs.items():
                    for bname, metrics in bins.items():
                        b = int(bname.replace("bin_", ""))
                        pang_bin_ratio_rows.append({
                            "tissue": tissue, "subset": subset, "target": target,
                            "output": output, "bin": b,
                            "auprc": metrics.get("auprc", np.nan),
                            "n_pos": metrics.get("n_pos", 0),
                            "n_neg": metrics.get("n_neg", 0),
                        })
    
        # binned regression
        br = binned.get("regression", {})
        for subset, outputs in br.items():
            for output, bins in outputs.items():
                for bname, metrics in bins.items():
                    b = int(bname.replace("bin_", ""))
                    pang_bin_reg_rows.append({
                        "tissue": tissue, "subset": subset, "output": output, "bin": b,
                        "pearson": metrics.get("pearson", np.nan),
                        "r2": metrics.get("r2", np.nan),
                        "mse": metrics.get("mse", np.nan),
                        "n": metrics.get("n", 0),
                    })
    
    df_pang_bin_ratio = pd.DataFrame(pang_bin_ratio_rows)
    df_pang_bin_reg = pd.DataFrame(pang_bin_reg_rows)
    log.info(f"pangolin binned: cls={df_pang_bin_ratio.shape}, reg={df_pang_bin_reg.shape}")
    log.info(f"  tissues: {sorted(df_pang_bin_ratio.tissue.unique()) if len(df_pang_bin_ratio) > 0 else 'none'}")
    
    # pangolin benchmark: binned AUPRC + binned MSE plots
    # one row per tissue, horizontal
    
    _pang_cls_order = ["splaire_ref", "pangolin_avg_p_splice", "splicetransformer", "spliceai"]
    _pang_reg_order = ["splaire_ref_ssu", "pangolin_avg_usage", "splicetransformer_avg_tissue"]
    _pang_all_order = _pang_cls_order + _pang_reg_order
    _pang_reg_set = set(_pang_reg_order)
    
    _pang_display = {
        "haec10": "HAEC", "lung": "Lung", "testis": "Testis",
        "brain_cortex": "Brain Cortex", "whole_blood": "Whole Blood",
        "heart": "Heart", "liver": "Liver", "brain": "Brain",
    }
    
    avail_pang_tissues = sorted(df_pang_bin_ratio.tissue.unique()) if len(df_pang_bin_ratio) > 0 else []
    
    if len(avail_pang_tissues) > 0:
        # binned AUPRC
        nt = len(avail_pang_tissues)
        fig, axes = plt.subplots(1, nt, figsize=(6 * nt, 5), sharey=True)
        if nt == 1: axes = [axes]
        for col, tissue in enumerate(avail_pang_tissues):
            ax = axes[col]
            sub = df_pang_bin_ratio.query("tissue == @tissue & subset == 'ssu_valid_nonzero'")
            if len(sub) == 0: continue
            avail = [o for o in _pang_all_order if o in sub.output.unique()]
            sub = sub[sub.output.isin(avail)]
            agg = sub.groupby(["output", "bin"]).agg(auprc=("auprc", "mean"), n_pos=("n_pos", "mean")).reset_index()
            counts = agg[agg.output == avail[0]][["bin", "n_pos"]].sort_values("bin")
            ax2 = ax.twinx()
            ax2.bar(counts["bin"], counts["n_pos"], width=0.5, color=GRAY_FAINT, alpha=0.4, zorder=1)
            ax2.tick_params(axis="y", labelcolor=GRAY_MED); ax2.set_yscale("log")
            ax2.set_ylim(counts["n_pos"].min() * 0.5, counts["n_pos"].max() * 3)
            if col == nt - 1: ax2.set_ylabel("Splice Sites", color=GRAY_MED)
            else: ax2.set_yticklabels([])
            for out in avail:
                od = agg[agg.output == out].sort_values("bin")
                ax.plot(od["bin"], od["auprc"], marker="o", color=get_color(out), lw=2, markersize=6,
                        zorder=3, ls="--" if out in _pang_reg_set else "-",
                        label=get_name(out) if col == 0 else None)
            ax.set_ylim(0, 1)
            if col == 0: ax.set_ylabel("AUPRC")
            ax.set_title(_pang_display.get(tissue, tissue.capitalize()))
            ax.set_xticks(range(10)); ax.set_xticklabels(bin_labels, rotation=45, ha="right")
            ax.set_xlabel("SSU bin")
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(_pang_all_order), frameon=False)
        fig.suptitle("Pangolin Benchmark — AUPRC stratified by SSU", fontsize=16, y=1.10)
        plt.tight_layout()
        _save_fig(plt, fig2_main / "pang_binned_auprc")
    
        # binned MSE
        fig, axes = plt.subplots(1, nt, figsize=(6 * nt, 5), sharey=True)
        if nt == 1: axes = [axes]
        for col, tissue in enumerate(avail_pang_tissues):
            ax = axes[col]
            sub = df_pang_bin_reg.query("tissue == @tissue & subset == 'ssu_valid_nonzero'")
            if len(sub) == 0: continue
            avail = [o for o in _pang_all_order if o in sub.output.unique()]
            sub = sub[sub.output.isin(avail)]
            agg = sub.groupby(["output", "bin"]).agg(mse=("mse", "mean"), n=("n", "mean")).reset_index()
            counts = agg[agg.output == avail[0]][["bin", "n"]].sort_values("bin")
            ax2 = ax.twinx()
            ax2.bar(counts["bin"], counts["n"], width=0.5, color=GRAY_FAINT, alpha=0.4, zorder=1)
            ax2.tick_params(axis="y", labelcolor=GRAY_MED); ax2.set_yscale("log")
            ax2.set_ylim(counts["n"].min() * 0.5, counts["n"].max() * 3)
            if col == nt - 1: ax2.set_ylabel("Splice Sites", color=GRAY_MED)
            else: ax2.set_yticklabels([])
            for out in avail:
                od = agg[agg.output == out].sort_values("bin")
                ax.plot(od["bin"], od["mse"], marker="o", color=get_color(out), lw=2, markersize=6,
                        zorder=3, ls="--" if out in _pang_reg_set else "-",
                        label=get_name(out) if col == 0 else None)
            if col == 0: ax.set_ylabel("MSE")
            ax.set_title(_pang_display.get(tissue, tissue.capitalize()))
            ax.set_xticks(range(10)); ax.set_xticklabels(bin_labels, rotation=45, ha="right")
            ax.set_xlabel("SSU bin")
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=len(_pang_all_order), frameon=False)
        fig.suptitle("Pangolin Benchmark — MSE stratified by SSU", fontsize=16, y=1.10)
        plt.tight_layout()
        _save_fig(plt, fig2_main / "pang_binned_mse")
    else:
        log.info("no pangolin binned data available")
    
    # pangolin benchmark: calibration curves — same layout as main calibration
    # uses pang_data from density loading cell, same _cal_from_arrays helper
    
    _pang_cal_cls_keys = ["spliceai_cls", "splaire_cls", "pangolin_cls", "spt_cls"]
    _pang_cal_reg_keys = ["splaire_reg", "pangolin_reg", "spt_reg"]
    _pang_cal_base = {
        "splaire_cls": "splaire_ref", "splaire_reg": "splaire_ref",
        "spliceai_cls": "spliceai",
        "pangolin_cls": "pangolin", "pangolin_reg": "pangolin",
        "spt_cls": "splicetransformer", "spt_reg": "splicetransformer",
    }
    _pang_cal_display = {
        "splaire_cls": "SPLAIRE", "splaire_reg": "SPLAIRE",
        "spliceai_cls": "SpliceAI",
        "pangolin_cls": "Pangolin", "pangolin_reg": "Pangolin",
        "spt_cls": "SpliceTransformer", "spt_reg": "SpliceTransformer",
    }
    
    n_cal_bins = 20
    cal_edges = np.linspace(0, 1, n_cal_bins + 1)
    
    avail_pang_cal = [t for t in pang_tissues if t in pang_data]
    nt = len(avail_pang_cal)
    
    if nt > 0:
        _pang_display = {"heart": "Heart", "liver": "Liver", "brain": "Brain", "testis": "Testis"}
    
        fig, axes = plt.subplots(2, nt, figsize=(6 * nt, 12), sharey=True, sharex=True)
        if nt == 1:
            axes = axes.reshape(2, 1)
    
        # top row: classification
        for ti, tissue in enumerate(avail_pang_cal):
            ax = axes[0, ti]
            td = pang_data[tissue]
            mask = td["masks"]["ssu_valid_nonzero"]
            y = td["y_ssu"][mask]
            for key in _pang_cal_cls_keys:
                if key not in td["preds"]: continue
                p = td["preds"][key][mask]
                pm, tm = _cal_from_arrays(p, y, cal_edges, n_cal_bins)
                valid = np.isfinite(pm) & np.isfinite(tm)
                base = _pang_cal_base[key]
                ax.plot(pm[valid], tm[valid], "o-", markersize=6, color=get_color(base),
                        lw=2.5, label=_pang_cal_display[key])
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            if ti == 0: ax.set_ylabel("Actual SSU", fontsize=14)
            ax.set_title(_pang_display.get(tissue, tissue), fontsize=15, fontweight="bold")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    
        fig.text(0.02, 0.73, "Classification", va="center", ha="center",
                 fontsize=14, fontweight="bold", rotation=90)
    
        # bottom row: regression
        for ti, tissue in enumerate(avail_pang_cal):
            ax = axes[1, ti]
            td = pang_data[tissue]
            mask = td["masks"]["ssu_valid_nonzero"]
            y = td["y_ssu"][mask]
            for key in _pang_cal_reg_keys:
                if key not in td["preds"]: continue
                p = td["preds"][key][mask]
                pm, tm = _cal_from_arrays(p, y, cal_edges, n_cal_bins)
                valid = np.isfinite(pm) & np.isfinite(tm)
                base = _pang_cal_base[key]
                ax.plot(pm[valid], tm[valid], "o-", markersize=6, color=get_color(base),
                        lw=2.5, label=_pang_cal_display[key])
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xlabel("Predicted SSU", fontsize=14)
            if ti == 0: ax.set_ylabel("Actual SSU", fontsize=14)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    
        fig.text(0.02, 0.28, "Regression", va="center", ha="center",
                 fontsize=14, fontweight="bold", rotation=90)
    
        fig.suptitle("Pangolin Benchmark — SSU Calibration", fontsize=17, fontweight="bold", y=1.01)
        plt.tight_layout()
        _save_fig(plt, fig2_sup / "pang_calibration")
    else:
        log.info("no pangolin prediction data for calibration")
    
    # pangolin benchmark: tissue variability analysis
    # each site has y_ssu from 4 tissues — compute cross-tissue SD and bin
    
    from scipy.stats import pearsonr as _pearsonr
    
    avail_pang_var = [t for t in pang_tissues if t in pang_data]
    
    if len(avail_pang_var) >= 2:
        # collect y_ssu and predictions per site across tissues
        # sites are at same positions across tissues (same h5 build)
        # merge on (chrom, pos, strand) — need to load coordinates
    
        # reload with coordinates
        site_dfs = {}
        for tissue in avail_pang_var:
            td = pang_data[tissue]
            # we need coordinates — reload from parquet
            gt = pd.read_parquet(pang_pred_dir / tissue / f"{tissue}_splaire_ref.parquet",
                                 columns=["chrom", "pos", "strand", "y_acceptor", "y_donor", "y_ssu"])
            is_splice = (gt["y_acceptor"] == 1) | (gt["y_donor"] == 1)
            # filter to splice sites first (matches pang_data loading)
            gt_splice = gt.loc[is_splice.values].reset_index(drop=True)
            # then filter to valid (non-sentinel)
            valid = (gt_splice["y_ssu"] != SENTINEL).values
            df_t = gt_splice.loc[valid, ["chrom", "pos", "strand", "y_ssu"]].copy()
            df_t = df_t.rename(columns={"y_ssu": f"ssu_{tissue}"})
    
            # add predictions — td["preds"] already filtered to splice sites,
            # so apply same valid mask to align
            for key in ["splaire_reg", "spliceai_cls", "pangolin_reg", "spt_reg"]:
                if key in td["preds"]:
                    df_t[f"pred_{key}_{tissue}"] = td["preds"][key][valid]
            site_dfs[tissue] = df_t
    
        # inner join across tissues
        df_var = site_dfs[avail_pang_var[0]]
        for tissue in avail_pang_var[1:]:
            df_var = df_var.merge(site_dfs[tissue], on=["chrom", "pos", "strand"], how="inner")
    
        ssu_cols = [f"ssu_{t}" for t in avail_pang_var]
        df_var["ssu_mean"] = df_var[ssu_cols].mean(axis=1)
        df_var["ssu_std"] = df_var[ssu_cols].std(axis=1)
    
        log.info(f"pangolin cross-tissue: {len(df_var):,} sites in {len(avail_pang_var)} tissues")
    
        # bin by SD
        n_qbins = 8
        df_var["sd_qbin"] = pd.qcut(df_var["ssu_std"], q=n_qbins, duplicates="drop")
        bin_labels_var = [f"{iv.left:.3f}-{iv.right:.3f}" for iv in sorted(df_var["sd_qbin"].unique())]
        bin_centers_var = np.arange(len(bin_labels_var))
    
        # compute per-site median absolute error across tissues
        _var_models = {
            "splaire_ref": ("SPLAIRE", "splaire_reg"),
            "spliceai": ("SpliceAI", "spliceai_cls"),
            "pangolin": ("Pangolin", "pangolin_reg"),
            "splicetransformer": ("SpliceTransformer", "spt_reg"),
        }
    
        # for each site: median |pred - true| across all tissues
        site_errors = {}
        for model_key, (display, pred_key) in _var_models.items():
            pred_cols_t = [f"pred_{pred_key}_{t}" for t in avail_pang_var]
            true_cols_t = [f"ssu_{t}" for t in avail_pang_var]
            # check all columns exist
            if not all(c in df_var.columns for c in pred_cols_t):
                log.info(f"  skipping {model_key}: missing prediction columns")
                continue
            errors = np.abs(df_var[pred_cols_t].values - df_var[true_cols_t].values)
            site_errors[model_key] = np.median(errors, axis=1)
    
        # R², pearson, MSE, MAE per bin — combined 4×2 figure (matches main)
        _reg_models_pv = ["splaire_ref", "pangolin", "splicetransformer"]
        _cls_models_pv = ["splaire_ref", "spliceai", "pangolin", "splicetransformer"]
    
        def _pang_var_metrics(models, pred_key_map):
            results = {m: {met: [] for met in ["r2", "pearson", "mse", "mae"]} for m in models}
            for qbin in sorted(df_var["sd_qbin"].unique()):
                sub = df_var[df_var["sd_qbin"] == qbin]
                for model_key in models:
                    _, pk = _var_models[model_key]
                    pred_cols_t = [f"pred_{pk}_{t}" for t in avail_pang_var]
                    true_cols_t = [f"ssu_{t}" for t in avail_pang_var]
                    if not all(c in sub.columns for c in pred_cols_t):
                        for met in results[model_key]: results[model_key][met].append(np.nan)
                        continue
                    p = sub[pred_cols_t].values.ravel()
                    y = sub[true_cols_t].values.ravel()
                    valid = np.isfinite(p) & np.isfinite(y)
                    pv, yv = p[valid], y[valid]
                    if len(yv) < 10:
                        for met in results[model_key]: results[model_key][met].append(np.nan)
                        continue
                    ss_tot = np.sum((yv - yv.mean())**2)
                    results[model_key]["r2"].append(1 - np.sum((yv - pv)**2) / ss_tot if ss_tot > 0 else np.nan)
                    results[model_key]["pearson"].append(_pearsonr(pv, yv)[0])
                    results[model_key]["mse"].append(np.mean((yv - pv)**2))
                    results[model_key]["mae"].append(np.median(np.abs(yv - pv)))
            return results
    
        reg_m = _pang_var_metrics(_reg_models_pv, _var_models)
        cls_m = _pang_var_metrics(_cls_models_pv, _var_models)
    
        _pang_var_metrics_list = [
            ("r2", "$R^2$"),
            ("pearson", "Pearson r"),
            ("mse", "MSE"),
            ("mae", "Median Absolute Error"),
        ]
    
        fig, axes = plt.subplots(len(_pang_var_metrics_list), 2,
                                 figsize=(14, 4 * len(_pang_var_metrics_list)), sharex=True)
    
        for row_i, (metric, ylabel) in enumerate(_pang_var_metrics_list):
            ax_reg, ax_cls = axes[row_i, 0], axes[row_i, 1]
    
            for model_key in _reg_models_pv:
                display, _ = _var_models[model_key]
                vals = np.array(reg_m[model_key][metric])
                lw = 2.5 if model_key == "splaire_ref" else 1.8
                ms = 6 if model_key == "splaire_ref" else 4
                ax_reg.plot(bin_centers_var, vals, "o-", color=get_color(model_key),
                            label=display, linewidth=lw, markersize=ms)
            ax_reg.set_ylabel(ylabel)
            if row_i == 0:
                ax_reg.set_title("Regression Outputs")
            ax_reg.legend(frameon=True, edgecolor="lightgray", fontsize=9)
            ax_reg.grid(alpha=0.2)
            ax_reg.set_xticks(bin_centers_var)
            if row_i == len(_pang_var_metrics_list) - 1:
                ax_reg.set_xticklabels(bin_labels_var, rotation=45, ha="right")
                ax_reg.set_xlabel("cross-tissue SD of mean SSU")
    
            for model_key in _cls_models_pv:
                display, _ = _var_models[model_key]
                vals = np.array(cls_m[model_key][metric])
                lw = 2.5 if model_key == "splaire_ref" else 1.8
                ms = 6 if model_key == "splaire_ref" else 4
                ax_cls.plot(bin_centers_var, vals, "o-", color=get_color(model_key),
                            label=display, linewidth=lw, markersize=ms)
            if row_i == 0:
                ax_cls.set_title("Classification Outputs")
            ax_cls.legend(frameon=True, edgecolor="lightgray", fontsize=9)
            ax_cls.grid(alpha=0.2)
            ax_cls.set_xticks(bin_centers_var)
            if row_i == len(_pang_var_metrics_list) - 1:
                ax_cls.set_xticklabels(bin_labels_var, rotation=45, ha="right")
                ax_cls.set_xlabel("cross-tissue SD of mean SSU")
    
        fig.suptitle("Pangolin Benchmark — prediction metrics vs tissue variability", fontsize=14, y=1.01)
        plt.tight_layout()
        _save_fig(plt, fig2_sup / "pang_variability_combined")
    else:
        log.info("need at least 2 pangolin tissues for variability analysis")
    

# --- render analysis.md to html with quarto if available ---
import shutil
if shutil.which("quarto"):
    log.info("quarto found, rendering analysis.md")
    subprocess.run([
        "quarto", "render", "analysis.md", "--to", "html",
        "-M", "code-fold:true",
        "-M", "toc:true",
        "-M", "toc-depth:3",
        "-M", "toc-location:body",
        "-M", "code-tools:true",
        "-M", "code-copy:true",
        "-M", "embed-resources:true",
        "-M", "theme:cosmo",
    ], check=True)
    log.info("quarto render complete")
else:
    log.info("quarto not found, skipping html render")
