"""LM-head row-gradient heatmaps in the part-1 nucleus-zoom style.

Companion to ``grad_layers.py``: that script shows how much gradient each
*layer* receives from a single token's loss; this one zooms into the LM head
and shows how the gradient lands across *vocabulary rows*. The per-row head
gradient is analytic — ``dL/dW_row_i = u_i · h`` with ``u = A(p − onehot)`` —
so a single forward pass (no backward) gives the exact picture.

The figure reuses part 1's Chebyshev-ring layout (``first_token_logits_black``):
the position's top tokens by logit rank, highest at center, each cell labeled
with the token and its gradient norm. Color is signed: green = the update
pushes that row's logit UP, orange = DOWN, black = effectively no gradient.
For every row except the target the magnitude is just ``|A| · p_i · ||h||`` — a
rescaled copy of part 1's probability heatmap. The target row is the one cell
that switches teams.

Three panels:
  1. ' understand' — RL branch token, A = +1.81 (correct rollout, step 8):
     target up, its competitor ' use' (the model's favorite) down.
  2. 'Let'         — RL with A = −0.55 (incorrect rollout, step 0): every
     arrow flips — the sampled token is pushed down, competitors up.
  3. '8'           — SFT (dataset solution, step 140): cross-entropy pushes
     the forced '8' up and the model's wrong singleton '9' down.

Forward passes run locally on CPU (bf16, no gradients); results are cached to
``data/`` so the rendering can be tweaked freely. ``--regen`` recomputes.

Usage:
    python grad_head_rows.py            # cached panels if present, else compute
    python grad_head_rows.py --regen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from first_token_logits_black import ring_side, ring_zoom_crop

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
UNIQUE_ID = "math500/geometry/9467"      # canonical id (display name: geometry/627)
ZOOM_RINGS = 3                            # rings 0..3 -> 7x7 = 49 cells, like part 1

# (panel name, trace cache stem, step, figure subtitle)
PANELS = [
    ("rl-branch", "correct", 8,
     "RL branch token — sampled ' understand' rewarded (A = +1.81)"),
    ("rl-negative", "incorrect", 0,
     "RL negative advantage — sampled 'Let' penalized (A = −0.55)"),
    ("sft-ood", "solution", 140,
     "SFT out-of-distribution — teacher-forced '8' (plain cross-entropy)"),
]

GREY_FLOOR = 0.01                         # |signed value| below this renders black


# --------------------------------------------------------------------------- #
# Compute: one forward pass per panel; per-row gradient is analytic.
# --------------------------------------------------------------------------- #
def compute_panels() -> dict:
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from math_rollouts.adapters.qwen_math import QwenMathAdapter
    from math_rollouts.data.problems import load_problems_by_ids

    here = Path(__file__).resolve().parent
    problem = load_problems_by_ids([UNIQUE_ID])[0]

    dtype = torch.bfloat16               # match the rest of the post's figures
    print(f"Loading {MODEL_ID} on cpu ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval()
    prompt_ids = QwenMathAdapter(MODEL_ID).prompt_ids(problem, tok)

    out = {"model_id": MODEL_ID, "zoom_rings": ZOOM_RINGS, "panels": []}
    for name, stem, step, subtitle in PANELS:
        cache = here / "data" / f"geometry_627_loss_trace_{stem}.parquet"
        df = pd.read_parquet(cache, columns=["chosen_id"])
        meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        completion_ids = [int(x) for x in df["chosen_id"]]
        target = completion_ids[step]
        advantage = 1.0 if meta.get("loss_kind") == "sft" else float(meta["advantage"])

        input_ids = torch.tensor([prompt_ids + completion_ids[:step]])
        print(f"[{name}] forward over {input_ids.shape[-1]} tokens ...")
        with torch.inference_mode():
            hidden = model.model(input_ids).last_hidden_state[0, -1]
            logits = model.lm_head(hidden).float()

        p = torch.softmax(logits, dim=-1)
        u = advantage * p.clone()
        u[target] -= advantage           # u = A(p - onehot): dLoss/dlogits
        h_norm = float(hidden.float().norm())
        # Signed per-row value: gradient-descent direction of the row's logit
        # (-u) times the row-gradient magnitude |u_i| * ||h||.
        signed = (-u.sign() * u.abs() * h_norm).numpy().astype(np.float32)

        rank_order = torch.argsort(logits, descending=True).numpy()
        side = ring_side(signed.shape[0])
        grid, token_grid = ring_zoom_crop(signed, rank_order, side, ZOOM_RINGS)

        labels = {}
        for tid in token_grid.flatten():
            if int(tid) >= 0:
                labels[str(int(tid))] = tok.decode([int(tid)])   # str keys: JSON-stable

        out["panels"].append({
            "name": name, "subtitle": subtitle, "step": step,
            "target_id": int(target), "target_str": tok.decode([target]),
            "advantage": advantage, "h_norm": h_norm,
            "target_prob": float(p[target]),
            "grid": grid.tolist(), "token_grid": token_grid.tolist(),
            "token_strs": labels,
        })
    return out


# --------------------------------------------------------------------------- #
# Render: diverging colormap with a black floor around zero.
# --------------------------------------------------------------------------- #
def _diverging_black_floor(floor: float, n: int = 512) -> ListedColormap:
    """Green for pushed-up, orange for pushed-down, black for ~zero.
    Input range is [-1, 1] (signed, normalized per panel)."""
    down = [(0.49, 0.18, 0.07), (0.92, 0.35, 0.05), (0.99, 0.73, 0.45)]  # dark->bright
    up = [(0.08, 0.33, 0.18), (0.09, 0.64, 0.29), (0.53, 0.94, 0.67)]

    def ramp(stops, t):
        t = t * (len(stops) - 1)
        i = min(int(t), len(stops) - 2)
        f = t - i
        return tuple((1 - f) * a + f * b for a, b in zip(stops[i], stops[i + 1]))

    colors = []
    for i in range(n):
        t = -1.0 + 2.0 * i / (n - 1)
        if abs(t) <= floor:
            colors.append((0.0, 0.0, 0.0, 1.0))
        elif t > 0:
            colors.append((*ramp(up, (t - floor) / (1 - floor)), 1.0))
        else:
            colors.append((*ramp(down, (-t - floor) / (1 - floor)), 1.0))
    return ListedColormap(colors, name="grad_diverging_black")


def _fmt_grad(v: float) -> str:
    av = abs(v)
    if av >= 100:
        return f"{av:.0f}"
    if av >= 1:
        return f"{av:.1f}"
    if av >= 0.01:
        return f"{av:.2f}"
    return f"{av:.0e}"


def render_panel(panel: dict, out_path: Path) -> None:
    grid = np.array(panel["grid"], dtype=np.float64)
    token_grid = np.array(panel["token_grid"], dtype=np.int64)
    max_abs = float(np.nanmax(np.abs(grid)))
    norm_grid = grid / max_abs

    cmap = _diverging_black_floor(GREY_FLOOR)
    crop_side = norm_grid.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(norm_grid, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest")
    title = (
        "LM-Head Row Gradients From One Token's Loss\n"
        f"{panel['subtitle']}\n"
        f"{MODEL_ID.split('/')[-1]} | MATH-500 geometry/627 | top "
        f"{crop_side * crop_side} of 151,936 rows, by logit rank | "
        f"green = logit pushed up, orange = pushed down"
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(crop_side):
        for col in range(crop_side):
            tid = int(token_grid[row, col])
            if tid < 0:
                continue
            val = float(grid[row, col])
            t = float(norm_grid[row, col])
            text = repr(panel["token_strs"][str(tid)])[1:-1]
            if len(text) > 12:
                text = text[:11] + "…"
            arrow = "" if abs(t) <= GREY_FLOOR else ("▲" if val > 0 else "▼")
            label = f"{text}\n{arrow}{_fmt_grad(val)}"
            rgba = cmap((t + 1.0) / 2.0)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if lum < 0.45 else "black"
            weight = "bold" if tid == panel["target_id"] else "normal"
            ax.text(col, row, label, ha="center", va="center", fontsize=7,
                    color=color, clip_on=True, parse_math=False, weight=weight)
            if tid == panel["target_id"]:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                           edgecolor="white", linewidth=2.0))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.set_yticklabels([f"▼{max_abs:.0f}", f"▼{max_abs / 2:.0f}", "0",
                             f"▲{max_abs / 2:.0f}", f"▲{max_abs:.0f}"])
    cbar.set_label("row gradient norm (▲ logit up, ▼ logit down)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")


# --------------------------------------------------------------------------- #
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=here / "data" / "geometry_627_grad_head_rows.json")
    ap.add_argument("--out-dir", type=Path,
                    default=here / "figures" / "geometry_627")
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    if args.cache.exists() and not args.regen:
        print(f"loading cached panels from {args.cache}")
        data = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        data = compute_panels()
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(data), encoding="utf-8")
        print(f"wrote {args.cache}")

    for i, panel in enumerate(data["panels"], start=1):
        print(f"\n[{panel['name']}] target {panel['target_str']!r} "
              f"(p={panel['target_prob']:.4f}, A={panel['advantage']:+.2f}, "
              f"||h||={panel['h_norm']:.1f})")
        render_panel(panel, args.out_dir / f"Grad-Rows-{i:02d}-{panel['name']}.png")


if __name__ == "__main__":
    main()
