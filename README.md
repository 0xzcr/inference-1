# Building an LLM Inference Engine from Scratch — Notes for Phases 1–4

These notes cover everything we did up to the end of Phase 4: using an open model from HuggingFace, then rebuilding its forward pass ourselves until our logits matched the real model's. Phases 5–8 (KV cache, sampling, batching, serving) are not covered here.

All numbers quoted as "your result" come from runs on your own machine during the session.

---

## 0. Where this started

**Blog 1 — "Build Your Own Inference Engine: From Scratch to '7'" (C++, MNIST).** A tiny engine that loads a trained network from an ONNX file and runs it. Its four steps were: load the model, build a graph of nodes, topologically sort the nodes (so every node runs after the nodes it depends on), then run inference node by node. The model used only four operations: Flatten, Gemm, ReLU and Add, and it had two branches that merge, which is why the sort mattered.

**Blog 2 — "I built an LLM inference engine from scratch" (Python).** This one is a *serving layer* around the HuggingFace `pipeline`, not an engine that does the math itself. Its ideas were: continuous batching with two timeouts (50 ms to wait for a first request, 10 ms to let a batch fill), running a whole batch in one model call instead of one call per prompt, a slot-based KV-cache allocator that rejects requests when full, and an async FastAPI front end. I read it through a summarizing tool, so treat its exact numbers as approximate.

**What you wanted:** take an open LLM from HuggingFace and build the inference engine for it yourself, learning each piece as we go. That is what the phases below do.

---

## 1. Roadmap

| Phase | Topic | Status |
|---|---|---|
| 1 | Meet the model | done |
| 2 | The generation loop by hand | done |
| 3 | Load the weights ourselves | done |
| 4 | Rebuild the transformer, layer by layer | done |
| 5 | KV cache | next |
| 6 | Sampling (temperature, top-k, top-p) | |
| 7 | Batching and scheduling | |
| 8 | A server with streaming | |

Phase 4 was split into smaller steps: 4a embeddings, 4b RMSNorm, 4c RoPE, 4d attention, 4e MLP and residuals, 4f the whole model.

---

## 2. The model: SmolLM2-135M

A small open model with the same architecture family as Llama, so everything here transfers to bigger models. It is small enough to run on a laptop CPU.

| Fact | Value |
|---|---|
| Vocabulary size | 49,152 tokens |
| Width (numbers per token vector) | 576 |
| Layers | 30 |
| Query heads | 9 |
| Key/value heads | 3 (grouped-query attention) |
| Head size | 64 (9 × 64 = 576) |
| MLP hidden size | 1,536 |
| RoPE theta | 100,000 |
| RMSNorm eps | 1e-5 |
| Tensors in the file | 272 |
| Stored dtype | bfloat16 |
| Output layer | tied to the embedding table |

**Sanity check on the parameter count** (this adds up exactly):

- Attention per layer: q 576×576 + o 576×576 + k 192×576 + v 192×576 = 884,736
- MLP per layer: 3 × 576 × 1,536 = 2,654,208
- Two norms per layer: 2 × 576 = 1,152
- One layer total: 3,540,096 → 30 layers = 106,202,880
- Embedding table: 49,152 × 576 = 28,311,552
- Final norm: 576
- **Total: 134,515,008 ≈ 135 M parameters.**

The MLP holds roughly three times as many parameters per layer as attention does. The embedding table is about a fifth of the whole model, which is why tying it to the output layer matters.

---

## 3. The big picture: what a forward pass is

```
token IDs [T]
   │  embedding lookup (a row lookup, no math)
   ▼
x  [T, 576]
   │
   ├─ repeat for each of the 30 layers ───────────────────────────┐
   │     h = RMSNorm(x, ln1)                                       │
   │     x = x + Attention(h)     # tokens look at earlier tokens  │
   │     h = RMSNorm(x, ln2)                                       │
   │     x = x + MLP(h)           # each token "thinks" alone      │
   └───────────────────────────────────────────────────────────────┘
   │  RMSNorm(x, final_norm)
   ▼
logits = x @ embed.T      [T, 49152]   (one score per possible next token)
```

The model does one thing: given a list of token IDs, it produces a score for every possible next token. Generating text is just picking a token, appending it, and repeating. Everything else is making that step correct and fast.

---

## 4. Phase 1 — Meet the model

Using the library exactly as shipped:

- `tok("The capital of France is")` produced the token IDs **[504, 3575, 282, 4649, 314]**, five tokens for five words. Each ID is just a row number in the embedding table. The model never sees letters.
- `model.generate(...)` produced text.
- `print(model)` showed the structure we then rebuilt: `embed_tokens`, 30 × `LlamaDecoderLayer` (each with `self_attn` = q/k/v/o projections, `mlp` = gate/up/down projections, and two RMSNorms), a final `norm`, and an `lm_head`.

**Detail worth noticing:** `q_proj` outputs 576 numbers but `k_proj` and `v_proj` output only 192. That is grouped-query attention: 9 query heads share 3 key/value heads.

---

## 5. Phase 2 — The generation loop by hand

`generate()` hides a simple loop:

1. Run the model on the current token list. It returns scores for every position.
2. Take the scores at the **last** position (they describe what comes next).
3. Pick the highest-scoring token (`argmax`, called greedy decoding).
4. Append it and repeat.

Your experiment showed two things.

**The shape grows each iteration.** `logits shape` went `[1, 5, 49152]`, `[1, 6, ...]`, `[1, 7, ...]`, and so on. We feed the *whole* sequence back in every step, and we only use the last row of the output. Earlier positions would have produced the same numbers as last time (a token only looks at tokens before it), so most of each step is repeated work that also grows with length. Phase 5, the KV cache, removes it.

**Greedy decoding can loop.** With a longer run, your output turned into "...of the department of department of department...". That was not a bug in the loop: the 10-token run had matched `generate()` exactly. Greedy always picks the single top token, repeated text makes repeated text more likely, and with no randomness the model can't escape. Small models fall into this more easily. The usual remedies are sampling (temperature, top-k, top-p) and repetition penalties, which is Phase 6.

---

## 6. Phase 3 — Load the weights ourselves

The model's knowledge is a pile of numbers saved in a **safetensors** file: named arrays, each with a shape and a dtype, and no code inside. We opened it directly with `safe_open` and listed the 272 tensors.

What you saw:

- Names follow the model printout, e.g. `model.layers.0.self_attn.q_proj.weight`.
- A `Linear(in=576, out=192)` layer is stored as shape **`[out, in]`**, so `k_proj.weight` is `(192, 576)`. When we multiply, we use `h @ W.T`.
- Everything is **bfloat16**, a 16-bit number format that halves the file size. For our own math we convert to float32 with `.float()`.
- The count: 30 layers × 9 tensors (q, k, v, o, gate, up, down, two norms) = 270, plus `embed_tokens` and the final `norm` = 272. **There is no `lm_head` tensor in the file.**

**Weight tying.** The final step ("turn a vector into 49,152 scores") reuses the embedding table as its weights. Embedding maps token → vector, and the output layer maps vector → token, so sharing one table saves about 28 M numbers. You confirmed it: `tie_word_embeddings` was `True` and `lm_head.weight` was identical to `embed_tokens.weight`.

---

## 7. Phase 4 — Rebuild the transformer

The method for every step was the same: write the operation ourselves from the weights file, then compare against the real HuggingFace layer. If the numbers match, the piece is right.

### 4a. Embeddings

Token ID 504 means "take row 504 of the embedding table". There is no arithmetic. Five tokens become a `[5, 576]` grid, one 576-number vector per token, and that grid is the input to layer 0. The shape is `[5, 576]` and not `[576]` because every token gets its own vector.

*Your result:* `our embeddings: torch.Size([5, 576])`, `match: True`.

### 4b. RMSNorm

**Problem.** After many multiplications and additions, vector sizes drift: some tokens get huge values, some tiny. That makes the numbers unstable. Before each major step, the model rescales each token's vector to a consistent size.

**Recipe (per token, over its 576 numbers):**

1. `ms = mean(x²)`, the mean square.
2. `x = x / sqrt(ms + eps)` (the tiny eps avoids dividing by zero).
3. `x = x * weight`, a learned list of 576 numbers from the file.

*Your result:*

```
mean square before: [0.0066, 0.0107, 0.0144, 0.0088, 0.0114]
mean square after:  [0.9985, 0.9991, 0.9993, 0.9989, 0.9991]
```

Before, the token vectors had different sizes (the third carried over twice the "energy" of the first). After, every token sits near 1. It is 0.9985 and not exactly 1 because of eps: for the first token, 0.0066 / (0.0066 + 0.00001) ≈ 0.9985. The "after" line leaves out step 3, the learned weight; the full operation matched layer 0's `input_layernorm` (`match: True`).

### 4c. RoPE (rotary position embeddings)

**Problem.** Nothing so far knows word order. "The capital of France is" and the same words shuffled would give the same set of vectors.

**Idea.** Split each 64-number head vector into 32 pairs, treat each pair as a point in 2D, and **rotate** it by an angle that grows with the token's position. Different pairs spin at different speeds: fast ones notice small distances, slow ones track long distances. The speeds are `1 / theta^(2i/64)` for pair `i`.

**Why it works.** When a rotated query is compared with a rotated key by a dot product, the result depends only on the *difference* of the two angles, which is the distance between the tokens. The model gets relative position even though we only feed in absolute positions.

**Details that matter:**

- RoPE is applied to **queries and keys only**, not values. Position decides who to look at, not what is passed along.
- HuggingFace's Llama code pairs number `i` with number `i+32` (first half with second half), not neighbouring numbers. The weights were trained that way, so we must match it. In code this is `rotate_half`, which returns `(−second half, first half)`.
- The rotation formula is `x * cos + rotate_half(x) * sin`.

*Your results:* `rope theta: 100000`, `cos match: True`, `sin match: True`, `position 0 unchanged: True` (angle 0 means no rotation), `length preserved: True` (a rotation never changes a vector's length).

The distance test, using one fixed random query and key placed at different positions:

```
distance 2 at (3,1): 1.1427770    distance 2 at (3,1): -6.2146602
distance 2 at (9,7): 1.1427758    distance 2 at (9,7): -6.2146597
distance 5 at (9,4): 0.7320932    distance 5 at (9,4): -1.0806379
      (first run)                        (second run)
```

The two distance-2 scores agree to about 6 decimal places (the leftover is float rounding) and the distance-5 score differs. The values change between runs only because the test vectors are random.

### 4d. Attention

Each token's normalized vector `h` becomes three things: a **query** (what I'm looking for), a **key** (what I contain), and a **value** (what I hand over if chosen).

**Steps:**

1. **Project:** `q = h @ Wq.T` → `[T, 9, 64]`; `k = h @ Wk.T` → `[T, 3, 64]`; `v = h @ Wv.T` → `[T, 3, 64]`.
2. **RoPE** on q and k.
3. **Grouped-query attention:** repeat each K/V head 3 times (`repeat_interleave`) so K/V head 0 serves query heads 0–2, head 1 serves 3–5, head 2 serves 6–8. This is why `k_proj` and `v_proj` are smaller, and it shrinks the KV cache 3×.
4. **Score:** `scores = q · k / sqrt(64)`. Dividing by 8 keeps scores from growing so large that softmax becomes extremely spiky.
5. **Causal mask:** set scores for later tokens to `-inf`, so a token can only look at itself and earlier tokens.
6. **Softmax** along each row: positive weights that sum to 1.
7. **Mix:** output = weighted average of the value vectors.
8. **Combine heads:** concatenate the 9 heads (back to 576 numbers) and apply `o_proj`.

*Your result:* `match: True` against the real layer-0 attention, and head 0's attention weights:

```
[[1.00, 0.00, 0.00, 0.00, 0.00],
 [0.51, 0.49, 0.00, 0.00, 0.00],
 [0.62, 0.36, 0.02, 0.00, 0.00],
 [0.22, 0.47, 0.04, 0.27, 0.00],
 [0.33, 0.23, 0.03, 0.23, 0.18]]
row sums: [1.00, 1.00, 1.00, 1.00, 1.00]
```

How to read it: row `t` is token `t`, and column `s` is how much it looks at token `s`. The upper-right triangle is zero because of the causal mask. Row 0 is `[1, 0, 0, 0, 0]` because the first token can only see itself, and softmax over a single option is 1. Every row sums to 1 because that is what softmax guarantees. Column 0 gets a large share in every row, which is a common model behaviour sometimes called an "attention sink"; it is only one head of one layer here, so don't read too much into it.

### 4e. The MLP, and residual connections

**MLP (SwiGLU).** Attention is where tokens communicate. The MLP is where each token processes information on its own, never looking at other tokens. Steps:

1. Two parallel projections expand 576 → 1,536: `gate_proj` and `up_proj`.
2. The gate branch goes through SiLU (`x * sigmoid(x)`, a smooth ReLU).
3. Multiply the two branches element by element. The gate works like a dial deciding how much of each `up` value gets through.
4. `down_proj` shrinks 1,536 → 576.

In one line: `mlp(h) = down( silu(gate(h)) * up(h) )`.

**Residual connections.** Neither attention nor the MLP *replaces* a token's vector; each **adds** its result to it:

```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Think of the 576-number vector as a running notebook that every layer writes small additions into. The normalized copies (`h`, `h_post`) are scratch inputs for each block. The result is added to the real stream `x`, not to the scratch copy, otherwise the content and magnitude built up so far would be thrown away. This is also why every layer has two norms: one before attention (`input_layernorm`) and one before the MLP (`post_attention_layernorm`).

*Your result:* `mlp match: True` and `layer 0 match: True`, so your code reproduced a complete transformer layer.

### 4f. The whole model

Wrap the layer in a loop over all 30 layers, apply the final RMSNorm, and multiply by the transposed embedding table (the tied output weights) to get 49,152 scores per token. The full code is in section 10.

*Your results:*

```
logits shape: torch.Size([5, 49152])
max abs diff: 4.77e-05
same next token: True
```

The diff is not exactly 0 because float32 rounding accumulates across 30 layers, but it is tiny. We then checked at every sequence length from 5 to 15 tokens, feeding identical tokens to both models. The max difference stayed between 4.6e-05 and 5.2e-05 everywhere.

It stays at exactly 4.63e-05 from length 8 onward because the largest gap comes from an early token, and later tokens can't change earlier ones (the causal mask again). That same fact is what the KV cache in Phase 5 exploits.

---

## 8. A lesson on precision: bfloat16 vs float32

Your Phase 2 text ("...the city of Paris. It is the largest city") did not match the text from the float32 engine ("...the capital of the country."). We tested it directly:

```
HF float32 : 'The capital of France is the capital of the country.\n\nThe capital'
HF bfloat16: 'The capital of France is the city of Paris. It is the largest city'
ours       : (same as HF float32 when run in a fresh process)
```

The explanation: newer `transformers` versions load a model in the checkpoint's own dtype (bfloat16) when you don't specify one, so the early experiments ran in bfloat16. bfloat16 keeps about 3 significant digits and float32 about 7. Greedy decoding picks the single top token, so when two candidates are close (like " city" and " capital") tiny rounding differences can flip the winner. After one token differs, the feedback loop makes the two texts drift apart. Neither output is wrong; it is the same model at two precisions.

This matters for inference engines in general: precision (bf16, fp16, int8 quantization) is one of the main knobs for trading a little accuracy for speed and memory, and it changes outputs slightly.

Tip: use `dtype=torch.float32` explicitly (the old name `torch_dtype=` is deprecated) whenever you compare against your own float32 code.

---

## 9. One open note

One script (`check_again.py`) that loaded two HuggingFace models before running our generation produced garbage from our engine (`'...the capital of t((((((((.i(.(.i...'`). Running our generation first in a fresh process, and the per-length logit comparison, both came out correct (`check_one_more_time.py`), so the engine itself looks right at every length tested. We did **not** find the cause of that one bad run; it could be a typo in that script or an interaction from loading a bfloat16 model in the same process. Practical rule for now: run each experiment in its own script, and if garbage ever appears in a fresh run, treat it as a real bug.

---

## 10. The engine as of the end of Phase 4 (`engine.py`)

```python
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

# ---- building blocks (each verified against HuggingFace) ----
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
```

A minimal check against HuggingFace:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine import forward, NAME

tok = AutoTokenizer.from_pretrained(NAME)
ref = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32)
ids = tok("The capital of France is", return_tensors="pt")["input_ids"]

with torch.no_grad():
    ours = forward(ids[0])
    theirs = ref(ids).logits[0]

print("max abs diff:", (ours - theirs).abs().max().item())
print("same next token:", ours[-1].argmax().item() == theirs[-1].argmax().item())
```

---

## 11. Glossary

- **Token:** a chunk of text (a word or part of one) mapped to an integer ID.
- **Embedding:** the learned vector (576 numbers) for each token; the first thing the model looks up.
- **Logits:** the raw scores for every possible next token; the largest is the model's top guess.
- **Weights / parameters:** the learned numbers stored in the model file.
- **Layer / decoder block:** one repeat of attention + MLP; this model has 30.
- **Residual stream:** the running vector each token carries through the layers; blocks add to it.
- **RMSNorm:** rescales a vector to a consistent size, with a learned per-number multiplier.
- **RoPE:** encodes position by rotating query and key vectors; gives relative position for free.
- **Q, K, V:** query, key and value vectors used by attention.
- **Head:** one independent attention computation on a 64-number slice; there are 9 query heads here.
- **GQA (grouped-query attention):** several query heads share one key/value head; here 3 query heads per K/V head, which makes the KV cache 3× smaller.
- **Causal mask:** forbids a token from looking at later tokens.
- **Softmax:** turns a row of scores into positive weights that sum to 1.
- **MLP / SwiGLU:** the per-token feed-forward block: `down(silu(gate(x)) * up(x))`.
- **Weight tying:** the output layer reuses the embedding table.
- **Greedy decoding:** always pick the single highest-scoring token.
- **bfloat16 / float32:** 16-bit and 32-bit number formats; bfloat16 is smaller and faster but less precise.
- **safetensors:** a simple, safe file format for named arrays of weights.

---

## 12. What comes next

- **Phase 5, KV cache:** store each layer's K and V for tokens already processed, so each new token only computes its own Q/K/V and attends to the stored ones. Prefill (whole prompt, once) and decode (one token at a time) become two different modes. Cost: about 45 KB of cache per token in float32 for this model (30 layers × 2 × 3 heads × 64 numbers × 4 bytes).
- **Phase 6, sampling:** temperature, top-k, top-p, and repetition handling, to fix the looping seen in Phase 2.
- **Phase 7, batching and scheduling:** the ideas from Blog 2: queueing requests, running several sequences in one model call, and managing cache memory per request.
- **Phase 8, a server:** an API with streaming of tokens.
