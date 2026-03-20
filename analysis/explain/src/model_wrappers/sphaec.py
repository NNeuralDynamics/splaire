"""
sphaec model wrapper for attribution analysis

loads full pytorch models (_full.pt) - no TensorFlow needed.
falls back to keras conversion if full model not found.

input: (batch, 4, length) one-hot dna
output: (batch, 1) for attribution
"""

import os
import sys
import torch
import torch.nn as nn

models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models")
sys.path.insert(0, models_dir)


def load_sphaec_model(model_path, device="cuda"):
    """load sphaec model - prefers _full.pt (no TensorFlow needed)"""
    device = device if torch.cuda.is_available() else "cpu"

    # check for full model first (no TensorFlow needed)
    if model_path.endswith(".keras"):
        full_path = model_path.replace(".keras", "_full.pt")
    else:
        full_path = model_path.replace(".pt", "_full.pt")

    if os.path.exists(full_path):
        print(f"  loading full model: {os.path.basename(full_path)}")
        model = torch.load(full_path, map_location="cpu", weights_only=False)
        model.eval()
        return model.to(device)

    # fallback to keras conversion (requires TensorFlow)
    print(f"  full model not found, falling back to keras conversion")
    keras_path = model_path if model_path.endswith(".keras") else model_path.replace(".pt", ".keras")
    assert os.path.exists(keras_path), f"keras not found: {keras_path}"

    from tensorflow.keras.models import load_model
    from keras_to_torch import FunctionalConv1DModel

    keras_model = load_model(keras_path, compile=False)
    model = FunctionalConv1DModel(keras_model.get_config())
    model.load_weights_from(keras_model)
    del keras_model

    return model.to(device).eval()


class SpHAECRegHead(nn.Module):
    """wrap regression model to output center as (batch, 1)"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        if y.ndim == 3 and y.shape[1] == 1:
            y = y[:, 0, :]
        center = y.shape[-1] // 2
        return y[:, center].unsqueeze(1)


class SpHAECClsHead(nn.Module):
    """wrap classification model for specific class"""

    def __init__(self, model, class_idx):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, x):
        y = self.model(x)
        y = y[:, self.class_idx, :]
        center = y.shape[-1] // 2
        return y[:, center].unsqueeze(1)


cls_idx = {"neither": 0, "acceptor": 1, "donor": 2}


def get_head(model, head_name):
    """wrap model to output (batch, 1)"""
    if head_name == "reg":
        return SpHAECRegHead(model)
    assert head_name in cls_idx, f"unknown head: {head_name}"
    return SpHAECClsHead(model, cls_idx[head_name])
