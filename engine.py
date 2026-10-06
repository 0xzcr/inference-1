import math
import torch
import torch.nn.functional as F
from safetensors import safe_open
from huggingface_hub import hf_hub_download

NAME = "HuggingFaceTB/SmolLM2-135M"
N_LAYERS, N_HEADS, N_KV, HEAD_DIM = 30, 9, 3, 64
THETA, EPS = 100000.0, 1e-5

# ---- weights (loaded once, as float32; roughly 0.5 GB of memory) ----
_f = safe_open(hf_hub_download(NAME, "model.safetensors"), framework="pt")
def W(name): return _f.get_tensor(name).float()

embed = W("model.embed_tokens.weight")          # also the output matrix (tied)
final_norm = W("model.norm.weight")
layers = []
for i in range(N_LAYERS):
    p = f"model.layers.{i}."
    layers.append(dict(
        ln1=W(p + "input_layernorm.weight"),
        ln2=W(p + "post_attention_layernorm.weight"),
        Wq=W(p + "self_attn.q_proj.weight"), Wk=W(p + "self_attn.k_proj.weight"),
        Wv=W(p + "self_attn.v_proj.weight"), Wo=W(p + "self_attn.o_proj.weight"),
        Wg=W(p + "mlp.gate_proj.weight"), Wu=W(p + "mlp.up_proj.weight"),
        Wd=W(p + "mlp.down_proj.weight"),
    ))

# ---- building blocks (all verified earlier) ----
def rmsnorm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight

def rope_cos_sin(positions):
    inv_freq = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    angles = positions[:, None].float() * inv_freq[None, :]
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]

def attention(h, L, cos, sin):
    T = h.shape[0]
    q = apply_rope((h @ L["Wq"].T).view(T, N_HEADS, HEAD_DIM), cos, sin)
    k = apply_rope((h @ L["Wk"].T).view(T, N_KV, HEAD_DIM), cos, sin)
    v = (h @ L["Wv"].T).view(T, N_KV, HEAD_DIM)
    rep = N_HEADS // N_KV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(HEAD_DIM)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
    out = torch.einsum("hts,shd->thd", weights, v).reshape(T, N_HEADS * HEAD_DIM)
    return out @ L["Wo"].T

def mlp(h, L):
    return (F.silu(h @ L["Wg"].T) * (h @ L["Wu"].T)) @ L["Wd"].T

def layer(x, L, cos, sin):
    x = x + attention(rmsnorm(x, L["ln1"]), L, cos, sin)
    x = x + mlp(rmsnorm(x, L["ln2"]), L)
    return x

# ---- the whole model ----
def forward(ids):                      # ids: 1D tensor of token IDs -> logits [T, vocab]
    cos, sin = rope_cos_sin(torch.arange(len(ids)))
    x = embed[ids]
    for L in layers:
        x = layer(x, L, cos, sin)
    return rmsnorm(x, final_norm) @ embed.T
