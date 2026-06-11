"""Full-model component-level gradient heatmap (Graphviz).

The indulgent version of ``grad_layers.py``: instead of one node per decoder
layer, every layer is drawn as its internal circuit and every component is
colored by the gradient norm it received from a single token's loss:

  - 12 query heads in two GQA groups of 6, funneling into the 2 key heads,
    then the 2 value heads, then back out to the 12 o_proj column slices
    (two groups of 6),
  - the two RMSNorms in their actual positions,
  - the MLP's W_gate and W_up on one plane, feeding W_down,
  - embeddings (input side) at the bottom, final norm + LM head (output side)
    at the top. Forward direction flows up the page.

Needs the ``components`` field produced by the updated grad_layers_colab.py
cell (per-layer per-head norms). Color is the same hot ramp as the other
gradient figures, normalized to the rendered token's max component; grey
below the floor.

Usage:
    python grad_components.py                      # ' understand' (step 8), RL json
    python grad_components.py --step 9
    python grad_components.py --cache data/geometry_627_grad_layers_sft.json --step 140
    python grad_components.py --layers 24 27       # render only layers 24..27
"""
from __future__ import annotations

import argparse
import colorsys
import json
import shutil
import subprocess
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
DOT_EXE_FALLBACK = (Path.home() / "AppData" / "Local" / "Graphviz"
                    / "Graphviz-12.2.1-win64" / "bin" / "dot.exe")

GREY_THRESHOLD = 0.02                    # fraction of the figure max -> grey


def _hot_rgb(intensity: float) -> tuple[float, float, float]:
    intensity = max(0.0, min(1.0, intensity))
    hue = (48 - 26 * intensity) / 360.0
    sat = 0.55 + 0.40 * intensity
    light = 0.94 - 0.52 * intensity
    return colorsys.hls_to_rgb(hue, light, sat)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb)


def colors_for(norm: float, global_max: float) -> tuple[str, str]:
    frac = norm / global_max if global_max > 0 else 0.0
    if frac < GREY_THRESHOLD:
        return "#e8eaee", "#94a3b8"
    intensity = frac ** 0.5
    fill = _hex(_hot_rgb(intensity))
    font = "#ffffff" if intensity > 0.75 else "#431407"
    return fill, font


def _fmt(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    if v >= 0.01:
        return f"{v:.2f}"
    return f"{v:.0e}"


def build_dot(token: dict, num_layers: int, layer_range: tuple[int, int]) -> str:
    comps = token["components"]
    lo, hi = layer_range
    all_norms = []
    for i in range(lo, hi + 1):
        c = comps[i]
        all_norms += c["q"] + c["k"] + c["v"] + c["o"]
        all_norms += [c["gate"], c["up"], c["down"], c["ln1"], c["ln2"]]
    full_stack = lo == 0 and hi == num_layers - 1
    if full_stack:
        all_norms += [token["head_out_norm"], token["embed_in_norm"],
                      token["final_norm_grad"]]
    gmax = max(all_norms)

    n_heads = len(comps[0]["q"])
    n_kv = len(comps[0]["k"])
    group = n_heads // n_kv

    L: list[str] = []

    def node(nid: str, text: str, norm: float, *, width: float = 0.62,
             fontsize: int = 8) -> None:
        fill, font = colors_for(norm, gmax)
        L.append(
            f"    {nid} [label=\"{text}\\n{_fmt(norm)}\", fillcolor=\"{fill}\","
            f" fontcolor=\"{font}\", width={width}, fontsize={fontsize}];")

    L.append("digraph components {")
    L.append("  rankdir=BT;")             # forward direction flows up the page
    L.append("  bgcolor=\"#f8fafc\";")
    L.append("  ranksep=0.22; nodesep=0.12;")
    L.append("  node [shape=box, style=\"filled,rounded\", fontname=\"Helvetica\","
             " height=0.32, fixedsize=true, color=\"#cbd5e1\", penwidth=0.6];")
    L.append("  edge [color=\"#b6c2d2\", arrowsize=0.4, penwidth=0.6];")
    tok_disp = token["token"].strip() or repr(token["token"])
    loss_txt = f"{token['loss']:.2f}" if abs(token["loss"]) >= 0.005 else f"{token['loss']:.0e}"
    L.append(f"  label=\"Gradient norm of every weight group from one token's"
             f" loss — '{tok_disp}' (p = {token['prob']:.2f}, loss = {loss_txt})\\n"
             f"{MODEL_ID.split('/')[-1]} | 12 Q heads in 2 GQA groups of 6 ->"
             f" 2 K -> 2 V -> 12 o_proj slices; MLP gate/up -> down."
             f" Grey = < {GREY_THRESHOLD:.0%} of max ({_fmt(gmax)}).\";")
    L.append("  labelloc=t; fontname=\"Helvetica\"; fontsize=12;"
             " fontcolor=\"#334155\";")

    if full_stack:
        node("emb", "embeddings (in)", token["embed_in_norm"],
             width=1.7, fontsize=9)

    prev_top = "emb" if full_stack else None
    for i in range(lo, hi + 1):
        c = comps[i]
        L.append(f"  subgraph cluster_L{i} {{")
        L.append(f"    label=\"layer {i}\"; labeljust=l; fontsize=10;"
                 " fontname=\"Helvetica\"; fontcolor=\"#64748b\";"
                 " color=\"#dbe2ea\"; style=rounded;")
        node(f"ln1_{i}", "ln1", c["ln1"], width=0.55)
        for h in range(n_heads):
            node(f"q{h}_{i}", f"q{h}", c["q"][h], width=0.55)
        for h in range(n_kv):
            node(f"k{h}_{i}", f"k{h}", c["k"][h], width=0.55)
            node(f"v{h}_{i}", f"v{h}", c["v"][h], width=0.55)
        for h in range(n_heads):
            node(f"o{h}_{i}", f"o{h}", c["o"][h], width=0.55)
        node(f"ln2_{i}", "ln2", c["ln2"], width=0.55)
        node(f"gate_{i}", "W gate", c["gate"], width=1.0)
        node(f"up_{i}", "W in", c["up"], width=1.0)
        node(f"down_{i}", "W out", c["down"], width=1.0)

        # rows
        L.append("    { rank=same; " + "; ".join(f"q{h}_{i}" for h in range(n_heads)) + "; }")
        L.append("    { rank=same; " + "; ".join(f"k{h}_{i}" for h in range(n_kv)) + "; }")
        L.append("    { rank=same; " + "; ".join(f"v{h}_{i}" for h in range(n_kv)) + "; }")
        L.append("    { rank=same; " + "; ".join(f"o{h}_{i}" for h in range(n_heads)) + "; }")
        L.append(f"    {{ rank=same; gate_{i}; up_{i}; }}")

        # wiring (forward direction): ln1 -> Q; Q groups funnel into their K
        # head; K -> V; V fans back out to that group's o_proj slices; o -> ln2
        # -> gate/up -> down.
        for h in range(n_heads):
            L.append(f"    ln1_{i} -> q{h}_{i};")
        for g in range(n_kv):
            for h in range(g * group, (g + 1) * group):
                L.append(f"    q{h}_{i} -> k{g}_{i};")
            L.append(f"    k{g}_{i} -> v{g}_{i};")
            for h in range(g * group, (g + 1) * group):
                L.append(f"    v{g}_{i} -> o{h}_{i};")
        for h in range(n_heads):
            L.append(f"    o{h}_{i} -> ln2_{i};")
        L.append(f"    ln2_{i} -> gate_{i}; ln2_{i} -> up_{i};")
        L.append(f"    gate_{i} -> down_{i}; up_{i} -> down_{i};")
        L.append("  }")
        if prev_top is not None:
            L.append(f"  {prev_top} -> ln1_{i} [penwidth=1.0];")
        prev_top = f"down_{i}"

    if full_stack:
        node("fnorm", "final norm", token["final_norm_grad"], width=1.2, fontsize=9)
        node("head", "LM head (out)", token["head_out_norm"], width=1.7, fontsize=9)
        L.append(f"  {prev_top} -> fnorm [penwidth=1.0];")
        L.append("  fnorm -> head [penwidth=1.0];")

    L.append("}")
    return "\n".join(L)


def render(dot_source: str, out_base: Path) -> None:
    dot = shutil.which("dot") or str(DOT_EXE_FALLBACK)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_base.with_suffix(".dot")
    dot_path.write_text(dot_source, encoding="utf-8")
    for fmt in ("png", "svg"):
        out = out_base.with_suffix(f".{fmt}")
        subprocess.run([dot, f"-T{fmt}", str(dot_path), "-o", str(out)], check=True)
        print(f"Wrote {out}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=here / "data" / "geometry_627_grad_layers.json")
    ap.add_argument("--step", type=int, default=8,
                    help="token step within the cached results (default: 8,"
                         " ' understand')")
    ap.add_argument("--layers", type=int, nargs=2, metavar=("LO", "HI"),
                    default=None, help="render only layers LO..HI")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.cache.read_text(encoding="utf-8"))
    token = next((tk for tk in data["tokens"] if tk["step"] == args.step), None)
    if token is None:
        raise SystemExit(f"step {args.step} not in {args.cache}")
    if "components" not in token:
        raise SystemExit("cache has no per-component breakdown — re-run the "
                         "updated grad_layers_colab.py cell and refresh the json")

    num_layers = data["num_layers"]
    layer_range = tuple(args.layers) if args.layers else (0, num_layers - 1)
    out = args.out
    if out is None:
        kind = "sft" if data.get("loss_kind") == "sft" else "rl"
        suffix = (f"_L{layer_range[0]}-{layer_range[1]}"
                  if args.layers else "")
        out = (here / "figures" /
               f"geometry_627_grad_components_{kind}_step{args.step}{suffix}")

    print(f"rendering step {args.step} ({token['token']!r}, "
          f"loss {token['loss']:.4g}) layers {layer_range[0]}..{layer_range[1]}")
    render(build_dot(token, num_layers, layer_range), out)


if __name__ == "__main__":
    main()
