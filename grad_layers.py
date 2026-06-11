"""Per-layer gradient norms for single-token GRPO losses (Qwen2.5-Math-1.5B).

Companion to ``html_token_losses.py``: where that script shows per-token *loss*
values, this one shows what a single token's loss actually does to the model.
For a few example tokens of the correct geometry/627 rollout we backprop
``loss = -A * log p(token)`` (the same advantage-scaled loss as the chips) and
record the total gradient norm of every weight group, bottom to top:

    embeddings (input side) -> decoder layers 0..27 -> LM head (output side)

The result renders via Graphviz as side-by-side vertical stacks of the model,
one per token, every node colored by its gradient norm on a scale shared across
all stacks. Expectation: a branch token lights up the top of the model; a
singleton (p ~ 1.0) leaves the whole stack grey.

Tied-embedding caveat: Qwen2.5-1.5B has ``tie_word_embeddings=True`` — the input
embedding table IS the LM head matrix, so ``weight.grad`` mixes both roles. The
output-side gradient is exactly rank-1, ``A * (softmax(z) - onehot) (x) h``
(h = final hidden state), so the two contributions are separated analytically:
the head node shows the rank-1 output-side norm, the embeddings node shows the
norm of (total grad - output-side grad). The per-row output-side norms also
answer "which vocab rows get gradient?": every row with non-negligible softmax
mass, not just the target (the top competitors are pushed down).

Default tokens (steps into the completion, from the cached loss trace):
    7  'to'          singleton, p ~ 1.0   (loss ~ 1e-3)
    8  'understand'  branch,    p ~ 0.18  (loss ~ 3.1)
    9  'the'         singleton, p ~ 0.95  (loss ~ 0.09)

Gradient norms are cached to ``data/`` (JSON); re-run with ``--regen`` to
recompute. Rendering tweaks therefore never touch the model.

Usage:
    python grad_layers.py             # cached norms if present, else compute
    python grad_layers.py --regen
"""
from __future__ import annotations

import argparse
import colorsys
import json
import shutil
import subprocess
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
UNIQUE_ID = "math500/geometry/9467"      # canonical id (display name: geometry/627)
TARGET_STEPS = [7, 8, 9]                 # singleton / branch / singleton

DOT_EXE_FALLBACK = (Path.home() / "AppData" / "Local" / "Graphviz"
                    / "Graphviz-12.2.1-win64" / "bin" / "dot.exe")

# Node color: same "hot" ramp as the SFT loss chips in html_token_losses.py —
# pale yellow through orange — with light grey for ~no gradient.
GREY_RGB = (0.92, 0.93, 0.94)            # hsl(210, 10%, 92%)
GREY_THRESHOLD = 0.02                    # fraction of the global max


def _hot_rgb(intensity: float) -> tuple[float, float, float]:
    intensity = max(0.0, min(1.0, intensity))
    hue = (48 - 26 * intensity) / 360.0
    sat = 0.55 + 0.40 * intensity
    light = 0.94 - 0.52 * intensity
    return colorsys.hls_to_rgb(hue, light, sat)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb)


def norm_to_colors(norm: float, global_max: float) -> tuple[str, str]:
    """(fillcolor, fontcolor) for a node with this gradient norm."""
    frac = norm / global_max if global_max > 0 else 0.0
    if frac < GREY_THRESHOLD:
        return _hex(GREY_RGB), "#94a3b8"
    intensity = frac ** 0.5
    fill = _hex(_hot_rgb(intensity))
    font = "#ffffff" if intensity > 0.75 else "#431407"
    return fill, font


# --------------------------------------------------------------------------- #
# Gradient measurement.
# --------------------------------------------------------------------------- #
def _chunked_table_stats(grad, v, chunk_rows: int = 16384):
    """For the [V, H] tied-table grad G: return (||G||^2, G @ v) without
    materializing a float32 copy of the whole table."""
    import torch

    sq = 0.0
    gv = []
    for chunk in grad.split(chunk_rows, dim=0):
        c = chunk.float()
        sq += float((c * c).sum())
        gv.append(c @ v)
    return sq, torch.cat(gv)


def measure_gradients(steps: list[int]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from math_rollouts.adapters.qwen_math import QwenMathAdapter
    from math_rollouts.data.problems import load_problems_by_ids

    here = Path(__file__).resolve().parent
    cache = here / "data" / "geometry_627_loss_trace_correct.parquet"
    import pandas as pd
    trace_df = pd.read_parquet(cache)
    meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    advantage = float(meta["advantage"])
    completion_ids = [int(x) for x in trace_df["chosen_id"]]

    problem = load_problems_by_ids([UNIQUE_ID])[0]

    # bf16 to match the rest of the post's figures (the sampler's precision).
    dtype = torch.bfloat16
    print(f"Loading {MODEL_ID} on cpu ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval()
    prompt_ids = QwenMathAdapter(MODEL_ID).prompt_ids(problem, tok)
    table = model.model.embed_tokens.weight        # tied: also the LM head matrix
    assert table is model.lm_head.weight, "expected tied embeddings"

    results = {"advantage": advantage, "model_id": MODEL_ID,
               "num_layers": len(model.model.layers), "tokens": []}
    for t in steps:
        target = completion_ids[t]
        row = trace_df.iloc[t]
        # Inputs up to (not including) the target token predict it.
        input_ids = torch.tensor([prompt_ids + completion_ids[:t]])
        model.zero_grad(set_to_none=True)
        hidden = model.model(input_ids).last_hidden_state[0, -1]   # post final-norm
        logits = model.lm_head(hidden).float()
        logp = torch.log_softmax(logits, dim=-1)
        loss = -advantage * logp[target]
        print(f"step {t} {row['chosen_str']!r}: p={row['prob']:.4f} "
              f"loss={float(loss):.4f} — backward ...")
        loss.backward()

        layer_norms = []
        for layer in model.model.layers:
            sq = sum(float((p.grad.float() ** 2).sum())
                     for p in layer.parameters() if p.grad is not None)
            layer_norms.append(sq ** 0.5)
        final_norm_grad = float(model.model.norm.weight.grad.float().norm())

        # Split the tied table: output side is rank-1, u (x) v.
        probs = torch.softmax(logits.detach(), dim=-1)
        u = advantage * probs.clone()
        u[target] -= advantage
        v = hidden.detach().float()
        out_norm = float(u.norm() * v.norm())
        total_sq, gv = _chunked_table_stats(table.grad, v)
        inner = float(u @ gv)                       # <G, u (x) v>
        in_sq = total_sq - 2.0 * inner + (float(u.norm()) * float(v.norm())) ** 2
        in_norm = max(0.0, in_sq) ** 0.5

        # Which vocab rows carry the output-side gradient? |u_i| * ||v||.
        row_norms = u.abs() * v.norm()
        top = torch.topk(row_norms, 5)
        top_rows = [
            {"token": tok.decode([int(i)]), "row_grad_norm": float(n),
             "softmax_p": float(probs[int(i)])}
            for n, i in zip(top.values, top.indices)
        ]

        results["tokens"].append({
            "step": t,
            "token": str(row["chosen_str"]),
            "prob": float(row["prob"]),
            "nuc_size": int(row["nuc_size"]),
            "loss": float(loss),
            "embed_in_norm": in_norm,
            "head_out_norm": out_norm,
            "table_total_norm": total_sq ** 0.5,
            "final_norm_grad": final_norm_grad,
            "layer_norms": layer_norms,
            "top_head_rows": top_rows,
        })
    return results


# --------------------------------------------------------------------------- #
# Graphviz rendering.
# --------------------------------------------------------------------------- #
def build_dot(results: dict, render_steps: list[int] | None = None) -> str:
    num_layers = results["num_layers"]
    tokens = [tk for tk in results["tokens"]
              if render_steps is None or tk["step"] in render_steps]
    results = {**results, "tokens": tokens}
    all_norms = []
    for tk in results["tokens"]:
        all_norms += tk["layer_norms"] + [tk["embed_in_norm"], tk["head_out_norm"]]
    global_max = max(all_norms)

    lines = [
        "digraph grads {",
        "  rankdir=TB;",
        "  bgcolor=\"#f8fafc\";",
        "  node [shape=box, style=\"filled,rounded\", fontname=\"Helvetica\","
        " fontsize=11, width=1.7, height=0.28, fixedsize=true,"
        " color=\"#cbd5e1\", penwidth=0.8];",
        "  edge [color=\"#94a3b8\", arrowsize=0.5, penwidth=0.8];",
        ("  label=\"Gradient norm per weight group from a single token's SFT"
         " loss (loss = -log p, teacher-forced dataset solution)."
         f" Shared color scale; grey = < {GREY_THRESHOLD:.0%} of max.\";"
         if results.get("loss_kind") == "sft" else
         f"  label=\"Gradient norm per weight group from a single token's loss"
         f" (loss = -A · log p, A = {results['advantage']:+.2f})."
         f" Shared color scale; grey = < {GREY_THRESHOLD:.0%} of max.\";"),
        "  labelloc=b; fontname=\"Helvetica\"; fontsize=11; fontcolor=\"#475569\";",
    ]
    for col, tk in enumerate(results["tokens"]):
        if not tk.get("in_nucleus", True):
            kind = "OUTSIDE nucleus"
        else:
            kind = "branch" if tk["nuc_size"] > 1 else "singleton"
        head = (f"\\\"{tk['token'].strip()}\\\" — {kind}\\n"
                f"p = {tk['prob']:.2f}   loss = {tk['loss']:.2f}"
                if tk["loss"] >= 0.005 else
                f"\\\"{tk['token'].strip()}\\\" — {kind}\\n"
                f"p = {tk['prob']:.2f}   loss = {tk['loss']:.0e}")
        lines.append(f"  subgraph cluster_{col} {{")
        lines.append("    peripheries=0;")
        lines.append("    label=\"\";")            # don't inherit the graph caption
        lines.append(f"    title_{col} [shape=plaintext, style=\"\","
                     f" fixedsize=false, fontsize=12, label=\"{head}\"];")

        def node(nid: str, text: str, norm: float) -> str:
            fill, font = norm_to_colors(norm, global_max)
            val = f"{norm:.3g}" if norm >= 1e-4 else f"{norm:.1e}"
            return (f"    {nid} [label=\"{text}  ·  {val}\","
                    f" fillcolor=\"{fill}\", fontcolor=\"{font}\"];")

        lines.append(node(f"head_{col}", "LM head (out)", tk["head_out_norm"]))
        for i in range(num_layers - 1, -1, -1):
            lines.append(node(f"L{i}_{col}", f"layer {i}", tk["layer_norms"][i]))
        lines.append(node(f"emb_{col}", "embeddings (in)", tk["embed_in_norm"]))

        chain = [f"title_{col}", f"head_{col}"]
        chain += [f"L{i}_{col}" for i in range(num_layers - 1, -1, -1)]
        chain.append(f"emb_{col}")
        lines.append("    " + " -> ".join(chain) + ";")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def render(dot_source: str, out_base: Path) -> None:
    dot = shutil.which("dot") or str(DOT_EXE_FALLBACK)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_base.with_suffix(".dot")
    dot_path.write_text(dot_source, encoding="utf-8")
    for fmt in ("png", "svg"):
        out = out_base.with_suffix(f".{fmt}")
        subprocess.run([dot, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
        print(f"Wrote {out}")


# --------------------------------------------------------------------------- #
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=TARGET_STEPS)
    ap.add_argument("--cache", type=Path,
                    default=here / "data" / "geometry_627_grad_layers.json")
    ap.add_argument("--out", type=Path,
                    default=here / "figures" / "geometry_627_grad_layers")
    ap.add_argument("--render-steps", type=int, nargs="+", default=None,
                    help="render only these steps from the cached norms "
                         "(default: all)")
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    if args.cache.exists() and not args.regen:
        print(f"loading cached gradient norms from {args.cache}")
        results = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        results = measure_gradients(args.steps)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.cache}")

    shown = (results["tokens"] if args.render_steps is None else
             [tk for tk in results["tokens"] if tk["step"] in args.render_steps])
    for tk in shown:
        print(f"\nstep {tk['step']} {tk['token']!r} (p={tk['prob']:.4f}, "
              f"loss={tk['loss']:.4g}):")
        print(f"  head (out side) {tk['head_out_norm']:.4g} | "
              f"embeddings (in side) {tk['embed_in_norm']:.4g} | "
              f"layers max {max(tk['layer_norms']):.4g}")
        print("  top head rows: " + ", ".join(
            f"{r['token']!r} {r['row_grad_norm']:.3g} (p={r['softmax_p']:.3f})"
            for r in tk["top_head_rows"]))

    render(build_dot(results, args.render_steps), args.out)


if __name__ == "__main__":
    main()
