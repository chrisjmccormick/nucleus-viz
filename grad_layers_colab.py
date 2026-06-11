# Colab GPU cell for nucleus-viz/grad_layers.py — self-contained, no math-rollouts.
#
# Paste this whole file into one cell of a Colab GPU notebook (T4 is plenty) and
# run it. Three experiments:
#
#   1. RL gradients: backprop the single-token GRPO loss (loss = -A * log p,
#      A = +1.81) for five tokens of the correct geometry/627 rollout — the
#      singleton/branch comparison (steps 7/8/9) plus the probability-matched
#      deep branch tokens (steps 180/339) for the depth comparison.
#      Every measurement also records a per-component breakdown inside each
#      layer: 12 query heads, 2 K / 2 V heads (GQA), 12 o_proj column slices,
#      MLP gate/up/down, and the two RMSNorms — for the full-model heatmap.
#   2. SFT gradients: backprop the plain cross-entropy loss (loss = -log p,
#      no advantage) for four tokens of the teacher-forced dataset solution —
#      'Name' (OOD opener), ' first' (OOD, the 10.5-nat peak), '8' (OOD, the
#      "(8, 9)" moment), and '7' (p = 0.999 singleton control).
#   3. Nucleus probe at solution steps 140/142 (what the model wanted to write
#      instead of "(8, 9)").
#
# Outputs (download or paste back; both drop into nucleus-viz/data/):
#   geometry_627_grad_layers.json       RL results + probe
#   geometry_627_grad_layers_sft.json   SFT results
# Render locally with grad_layers.py (no GPU needed).
#
# All token ids below were extracted from the cached traces so the numbers
# match the blog figures exactly (bf16, same prompt, same completions).

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
ADVANTAGE = 1.807392228230128          # (1 - 15/64) / sqrt(p(1-p)), group 15/64 correct

# Qwen-Math chat prompt for MATH-500 geometry/627 (78 tokens).
PROMPT_IDS = [151644, 8948, 198, 5501, 2874, 3019, 553, 3019, 11, 323, 2182, 697,
    1590, 4226, 2878, 1124, 79075, 46391, 151645, 198, 151644, 872, 198, 785,
    13934, 315, 264, 57030, 839, 2396, 525, 320, 20, 11, 220, 18, 701, 320, 21,
    11, 220, 23, 701, 320, 22, 11, 220, 19, 8, 323, 4930, 87, 11, 379, 15087,
    323, 400, 87, 861, 220, 22, 12947, 3555, 374, 279, 897, 315, 400, 87, 488,
    379, 3, 30, 151645, 198, 151644, 77091, 198]

# First 340 tokens of the correct rollout (sample 10).
ROLLOUT_IDS = [1249, 11625, 419, 3491, 11, 582, 1184, 311, 3535, 279, 5888,
    315, 264, 57030, 839, 2396, 13, 758, 264, 57030, 839, 2396, 11, 14002,
    11067, 525, 6144, 304, 3084, 323, 15279, 13, 1096, 3363, 429, 279, 22879,
    14064, 279, 11067, 315, 279, 57030, 839, 2396, 1969, 387, 6144, 304, 25621,
    323, 5106, 382, 22043, 279, 13934, 315, 2326, 17228, 315, 279, 57030, 839,
    2396, 25, 1124, 1188, 20, 11, 220, 18, 10699, 701, 1124, 1188, 21, 11, 220,
    23, 10699, 701, 323, 1124, 1188, 22, 11, 220, 19, 10699, 701, 582, 646,
    78064, 1493, 3501, 438, 17767, 32, 59, 701, 17767, 33, 59, 701, 323, 17767,
    34, 57758, 15576, 13, 6771, 279, 11737, 11936, 387, 17767, 35, 284, 320,
    87, 11, 379, 10699, 3593, 1249, 1477, 279, 13934, 315, 17767, 35, 59, 701,
    582, 646, 990, 279, 3343, 429, 279, 4621, 1124, 11520, 1975, 19491, 90,
    1867, 11035, 8, 374, 6144, 311, 279, 4621, 1124, 11520, 1975, 19491, 90,
    6484, 11035, 8, 320, 11284, 17767, 1867, 57758, 323, 17767, 6484, 57758,
    525, 14002, 11067, 315, 279, 57030, 839, 2396, 3593, 5338, 11, 1077, 594,
    11047, 279, 4621, 1124, 11520, 1975, 19491, 90, 1867, 11035, 982, 59, 9640,
    59, 1975, 19491, 90, 1867, 92, 284, 320, 21, 12, 20, 11, 220, 23, 12, 18,
    8, 284, 320, 16, 11, 220, 20, 340, 59, 2533, 1986, 4621, 1124, 1188, 16,
    11, 220, 20, 10699, 8, 1969, 387, 6144, 311, 279, 4621, 1124, 11520, 1975,
    19491, 90, 6484, 11035, 982, 59, 9640, 59, 1975, 19491, 90, 6484, 92, 284,
    320, 87, 12, 22, 11, 379, 12, 19, 8, 284, 320, 16, 11, 220, 20, 340, 59,
    2533, 3830, 419, 11, 582, 633, 279, 2701, 1849, 315, 37906, 510, 59, 9640,
    87, 481, 220, 22, 284, 220, 16, 1124, 6383, 550, 856, 284, 220, 23, 198,
    59, 921, 59, 9640, 88, 481, 220, 19, 284, 220, 20, 1124, 6383, 550, 379,
    284, 220, 24, 198, 59, 2533, 4416, 279, 13934, 315, 17767, 35, 57758, 525,
    1124, 1188, 23, 11, 220, 24, 10699, 568, 4695]
RL_TARGETS = [7, 8, 9, 180, 339]
RL_NUC = {7: (1, True), 8: (3, True), 9: (1, True), 180: (2, True), 339: (3, True)}

# All 161 tokens of the dataset reference solution ([asy] stripped, EOS appended).
SOLUTION_IDS = [675, 279, 3501, 400, 32, 7, 20, 11, 18, 15087, 11, 400, 33, 7,
    21, 11, 23, 15087, 11, 400, 34, 7, 22, 11, 19, 15087, 11, 323, 400, 35,
    2075, 7358, 15087, 323, 25529, 279, 1156, 2326, 13, 220, 1205, 1477, 429,
    1052, 525, 2326, 3204, 10468, 369, 400, 35, 3, 320, 4060, 7071, 568, 220,
    8278, 279, 825, 311, 279, 1290, 702, 458, 400, 87, 3, 12, 62526, 7046,
    1091, 220, 22, 13, 220, 8704, 400, 1706, 3, 374, 15279, 311, 400, 9548, 3,
    323, 6144, 304, 3084, 311, 432, 11, 400, 35, 3, 374, 1378, 8153, 311, 279,
    1290, 323, 825, 4982, 705, 504, 400, 33, 54876, 1101, 438, 400, 34, 3, 374,
    1378, 8153, 311, 279, 1290, 323, 825, 4982, 705, 504, 400, 32, 12947, 220,
    15277, 11, 279, 13934, 315, 400, 35, 3, 525, 4930, 23, 11, 24, 15087, 11,
    323, 400, 87, 43010, 28, 23, 10, 24, 34433, 79075, 90, 16, 22, 92, 12947,
    151643]
SFT_TARGETS = [0, 36, 140, 157]
SFT_NUC = {0: (4, False), 36: (2, False), 140: (1, False), 157: (1, True)}
PROBE_STEPS = [140, 142]

device = "cuda"
dtype = torch.bfloat16                 # match the sampler / the blog's figures
print(f"Loading {MODEL_ID} ({dtype}) ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
table = model.model.embed_tokens.weight
assert table is model.lm_head.weight, "expected tied embeddings"


def _gnorm(*tensors):
    return sum(float((t.float() ** 2).sum()) for t in tensors if t is not None) ** 0.5


def layer_breakdown(layer, n_heads, n_kv_heads, head_dim):
    """Per-component grad norms inside one decoder layer. Q/K/V slice by output
    rows (head dims, bias included); o_proj slices by input columns (per-head
    contributions). MLP: gate/up (same plane) and down. Plus the two RMSNorms."""
    attn, mlp = layer.self_attn, layer.mlp
    qg, qb = attn.q_proj.weight.grad, attn.q_proj.bias.grad
    kg, kb = attn.k_proj.weight.grad, attn.k_proj.bias.grad
    vg, vb = attn.v_proj.weight.grad, attn.v_proj.bias.grad
    og = attn.o_proj.weight.grad
    sl = lambda h: slice(h * head_dim, (h + 1) * head_dim)
    return {
        "q": [_gnorm(qg[sl(h)], qb[sl(h)]) for h in range(n_heads)],
        "k": [_gnorm(kg[sl(h)], kb[sl(h)]) for h in range(n_kv_heads)],
        "v": [_gnorm(vg[sl(h)], vb[sl(h)]) for h in range(n_kv_heads)],
        "o": [_gnorm(og[:, sl(h)]) for h in range(n_heads)],
        "gate": _gnorm(mlp.gate_proj.weight.grad),
        "up": _gnorm(mlp.up_proj.weight.grad),
        "down": _gnorm(mlp.down_proj.weight.grad),
        "ln1": _gnorm(layer.input_layernorm.weight.grad),
        "ln2": _gnorm(layer.post_attention_layernorm.weight.grad),
    }


def measure(completion_ids, step, advantage, nuc):
    """Backprop one token's loss and collect per-weight-group gradient norms."""
    target = completion_ids[step]
    input_ids = torch.tensor([PROMPT_IDS + completion_ids[:step]], device=device)
    model.zero_grad(set_to_none=True)
    hidden = model.model(input_ids).last_hidden_state[0, -1]    # post final-norm
    logits = model.lm_head(hidden).float()
    logp = torch.log_softmax(logits, dim=-1)
    loss = -advantage * logp[target]
    loss.backward()

    layer_norms = [
        (sum(float((p.grad.float() ** 2).sum())
             for p in layer.parameters() if p.grad is not None)) ** 0.5
        for layer in model.model.layers
    ]
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    components = [
        layer_breakdown(layer, cfg.num_attention_heads,
                        cfg.num_key_value_heads, head_dim)
        for layer in model.model.layers
    ]
    probs = torch.softmax(logits.detach(), dim=-1)
    u = advantage * probs.clone(); u[target] -= advantage
    v = hidden.detach().float()
    out_norm = float(u.norm() * v.norm())
    total_sq, gv = 0.0, []
    for chunk in table.grad.split(16384, dim=0):
        c = chunk.float(); total_sq += float((c * c).sum()); gv.append(c @ v)
    gv = torch.cat(gv)
    in_sq = total_sq - 2.0 * float(u @ gv) + (float(u.norm()) * float(v.norm())) ** 2
    top = torch.topk(u.abs() * v.norm(), 5)
    nuc_size, in_nucleus = nuc
    return {
        "step": step, "token": tok.decode([target]),
        "prob": float(probs[target]), "nuc_size": nuc_size,
        "in_nucleus": in_nucleus, "loss": float(loss.detach()),
        "embed_in_norm": max(0.0, in_sq) ** 0.5, "head_out_norm": out_norm,
        "table_total_norm": total_sq ** 0.5,
        "final_norm_grad": float(model.model.norm.weight.grad.float().norm()),
        "layer_norms": layer_norms,
        "components": components,
        "top_head_rows": [
            {"token": tok.decode([int(i)]), "row_grad_norm": float(n),
             "softmax_p": float(probs[int(i)])}
            for n, i in zip(top.values, top.indices)],
    }


# ------------------------------------------------------------ RL gradients --
rl = {"advantage": ADVANTAGE, "model_id": MODEL_ID, "loss_kind": "grpo",
      "num_layers": len(model.model.layers), "tokens": []}
for t in RL_TARGETS:
    tk = measure(ROLLOUT_IDS, t, ADVANTAGE, RL_NUC[t])
    rl["tokens"].append(tk)
    print(f"RL  step {t:>3} {tk['token']!r}: loss={tk['loss']:.4f}, "
          f"layers max {max(tk['layer_norms']):.4g}, head {tk['head_out_norm']:.4g}")

# ----------------------------------------------------------- SFT gradients --
sft = {"advantage": 1.0, "model_id": MODEL_ID, "loss_kind": "sft",
       "num_layers": len(model.model.layers), "tokens": []}
for t in SFT_TARGETS:
    tk = measure(SOLUTION_IDS, t, 1.0, SFT_NUC[t])
    sft["tokens"].append(tk)
    print(f"SFT step {t:>3} {tk['token']!r}: loss={tk['loss']:.4f}, "
          f"layers max {max(tk['layer_norms']):.4g}, head {tk['head_out_norm']:.4g}")

# -------------------------------------------------- nucleus probe (140/142) --
model.zero_grad(set_to_none=True)
rl["probe"] = []
with torch.no_grad():
    for t in PROBE_STEPS:
        input_ids = torch.tensor([PROMPT_IDS + SOLUTION_IDS[:t]], device=device)
        logits = model(input_ids).logits[0, -1].float()
        p06 = torch.softmax(logits / 0.6, dim=-1)
        vals, idx = torch.topk(p06, 20)
        keep = int((torch.cumsum(vals, 0) < 0.95).sum()) + 1
        p10 = torch.softmax(logits, dim=-1)
        rl["probe"].append({
            "step": t, "forced_token": tok.decode([SOLUTION_IDS[t]]),
            "nucleus": [{"token": tok.decode([int(i)]), "p_T06": float(v),
                         "p_T10": float(p10[int(i)])}
                        for v, i in zip(vals[:keep], idx[:keep])],
            "top10_T10": [{"token": tok.decode([int(i)]), "p": float(p10[int(i)])}
                          for i in torch.topk(p10, 10).indices],
        })

with open("geometry_627_grad_layers.json", "w") as f:
    json.dump(rl, f, indent=2)
with open("geometry_627_grad_layers_sft.json", "w") as f:
    json.dump(sft, f, indent=2)
print("\nsaved geometry_627_grad_layers.json + geometry_627_grad_layers_sft.json")
print("\n--- RL JSON ---\n" + json.dumps(rl))
print("\n--- SFT JSON ---\n" + json.dumps(sft))
