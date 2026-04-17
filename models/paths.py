import os
from pathlib import Path

models_dir = Path(os.environ.get("SPLAIRE_MODELS_DIR", Path(__file__).parent))


def get_keras(variant_type, head_type, versions=range(1, 6)):
    return [models_dir / f"{variant_type}_100_v{v}_{head_type}_best.keras" for v in versions]


def get_pytorch(variant_type, head_type, versions=range(1, 6)):
    # reg models have _sigmoid suffix in pytorch
    suffix = "_sigmoid" if head_type == "reg" else ""
    return [models_dir / f"{variant_type}_100_v{v}_{head_type}_best{suffix}.pt" for v in versions]


def ref_cls(framework="keras"):
    return get_keras("Ref", "cls") if framework == "keras" else get_pytorch("Ref", "cls")


def ref_reg(framework="keras"):
    return get_keras("Ref", "reg") if framework == "keras" else get_pytorch("Ref", "reg")


def var_cls(framework="keras"):
    return get_keras("Var", "cls") if framework == "keras" else get_pytorch("Var", "cls")


def var_reg(framework="keras"):
    return get_keras("Var", "reg") if framework == "keras" else get_pytorch("Var", "reg")
