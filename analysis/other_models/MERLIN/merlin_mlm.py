"""MLM-pretrained MERLIN: encoder + nucleotide decoder for zero-shot variant scoring

decoder architecture mirrors Spliceformer/MLM.py:18-32 (3-layer MLP head over
the encoder embedding, predicts A/C/G/T logits per position).

usage:
    sys.path.insert(0, ".../other_models/MERLIN")
    from merlin_mlm import load_mlm_from_ckpt, score_llr

ckpt path (cluster):
  /projects/talisman/mrunyan/paper/SpHAEC/analysis/other_models/MERLIN/AdarEditingPrediction/Models/model_MLM_final_data_sp.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from merlin_model import (
    GenerateEmbeddings,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EMBEDDING_LENGTH,
    DEFAULT_TRANSFORMER_DEPTH,
    strip_ddp_prefix,
)


class MaskedNucleotideModel(nn.Module):
    """encoder + MLM head. predicts 4-channel logits per input position."""
    def __init__(self, transformer_block_depth, embedding_length,
                 dropout_rate=0.3, attn_dropout=0.1, input_channels=4):
        super().__init__()
        self.model = GenerateEmbeddings(
            dropout_rate=dropout_rate,
            attn_dropout=attn_dropout,
            embedding_length=embedding_length,
            transformer_block_depth=transformer_block_depth,
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_length, embedding_length // 8),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_length // 8, embedding_length // 16),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_length // 16, input_channels),
        )

    def forward(self, x):
        # x: (B, L, 4) one-hot. encoder prepends a cls token → (B, L+1, D).
        # drop the cls token before decoding so output positions align with input positions.
        emb = self.model(x)
        emb = emb[:, 1:, :]
        return self.decoder(emb)


def load_mlm_from_ckpt(ckpt_path, device, *,
                       embedding_length=DEFAULT_EMBEDDING_LENGTH,
                       transformer_block_depth=DEFAULT_TRANSFORMER_DEPTH,
                       dropout_rate=0.3,
                       attn_dropout=0.1):
    m = MaskedNucleotideModel(
        transformer_block_depth=transformer_block_depth,
        embedding_length=embedding_length,
        dropout_rate=dropout_rate,
        attn_dropout=attn_dropout,
    ).to(device)
    m = torch.compile(m)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd["model_state_dict"] if "model_state_dict" in sd else sd
    sd = strip_ddp_prefix(sd)
    res = m.load_state_dict(sd, strict=True)
    print(f"loaded mlm: {ckpt_path}", flush=True)
    print(f"  missing: {res.missing_keys}", flush=True)
    print(f"  unexpected: {res.unexpected_keys}", flush=True)
    m.eval()
    return m


@torch.no_grad()
def score_llr(model, oh, var_offsets, ref_idx, alt_idx, device, bs=8,
              autocast_dtype=torch.bfloat16):
    """zero-shot llr scoring at variant positions

    oh:          (n, L, 4) ref one-hot; we mask oh[:, var_offsets[i], :] to zero
    var_offsets: (n,) int — position within L to mask + score
    ref_idx:     (n,) int in {0,1,2,3} — base index for ref allele
    alt_idx:     (n,) int in {0,1,2,3} — base index for alt allele
    returns:     (n,) float32 — log P(alt) - log P(ref) at the masked position
    """
    n = oh.shape[0]
    out = torch.empty(n, dtype=torch.float32)
    for i in range(0, n, bs):
        j = min(i + bs, n)
        x = torch.from_numpy(oh[i:j].astype("float32")).to(device)
        offs = var_offsets[i:j]
        # mask the variant row (all four channels → 0) so the decoder predicts from context
        idx = torch.arange(x.size(0), device=device)
        x[idx, torch.from_numpy(offs).to(device), :] = 0.0
        with torch.autocast(device.type if device.type == "cuda" else "cpu",
                            dtype=autocast_dtype, enabled=(device.type == "cuda")):
            logits = model(x)  # (b, L, 4)
        logp = F.log_softmax(logits.float(), dim=-1)
        # gather at variant positions
        row = torch.arange(x.size(0), device=device)
        pos = torch.from_numpy(offs).to(device)
        logp_at = logp[row, pos, :]                                # (b, 4)
        r = torch.from_numpy(ref_idx[i:j]).long().to(device)
        a = torch.from_numpy(alt_idx[i:j]).long().to(device)
        llr = logp_at.gather(1, a[:, None]).squeeze(1) - logp_at.gather(1, r[:, None]).squeeze(1)
        out[i:j] = llr.cpu()
    return out.numpy()
