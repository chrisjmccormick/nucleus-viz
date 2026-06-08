"""Token-probability chip strip for one MATH-500 rollout (Qwen2.5-Math-1.5B).

Pulls a real, naturally-sampled rollout for ``geometry/627`` from the
``math-rollouts`` HF dataset, recomputes the per-token nucleus along it
(teacher-forced HF forward pass, T=0.6 / top_p=0.95 / top_k=20 — the dataset's
canonical config), and writes standalone HTML where each generated token is a chip
showing the chosen token plus its nucleus alternates, colored by probability.

The expensive step is the forward pass; its output (the per-token trace) is cached
to ``data/`` so tweaking the visual never re-runs the model. Re-run it with
``--regen``.

Default rollout: the shortest CORRECT, cleanly-reasoned sample for geometry/627 —
``math500_passK`` sample_idx=10, which opens "To solve this problem ..." and works
through to ``\\boxed{17}``.

Data comes through the math-rollouts interface (``pip install -e ../math-rollouts``):
    from math_rollouts.data.hf import load_generation_parquet
    from math_rollouts.nucleus import trace_nuclei

Usage:
    python html_tokens.py                 # uses the cached trace if present
    python html_tokens.py --regen         # recompute the trace (loads the model)
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
PASSK_NAME = "math500_passK"             # naturally-sampled pool in the HF dataset
PROBLEM_ID = "test/geometry/627.json"
DEFAULT_SAMPLE_IDX = 10                   # shortest correct "To solve ..." rollout

DEFAULT_MAX_ROW_WIDTH_PX = 1024
MAX_PER_COL = 10                          # chosen + up to 9 alternates per column
_NL = "↵"                            # ↵ shown in place of a newline


# --------------------------------------------------------------------------- #
# Chip rendering (unchanged recipe — fed by the recomputed trace).
# --------------------------------------------------------------------------- #
def _blue_at(intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    lightness = 96 - 64 * intensity
    saturation = 25 + 55 * intensity
    return f"hsl(210, {saturation:.0f}%, {lightness:.0f}%)"


def prob_to_blue_style(p: float) -> str:
    p = max(0.0, min(1.0, float(p)))
    if p == 0.0:
        return "background-color: hsl(0, 75%, 88%); color: #7f1d1d;"
    intensity = p ** 0.5
    bg = _blue_at(intensity)
    text_color = "#ffffff" if intensity > 0.6 else "#1e293b"
    return f"background-color: {bg}; color: {text_color};"


def render_token_chip(token_str: str, prob: float, is_chosen: bool) -> str:
    tok = html.escape(token_str).replace(" ", "&nbsp;")
    if tok == "":
        tok = "&nbsp;"
    bg = prob_to_blue_style(prob)
    border = (
        "border: 2px solid #1e3a8a;"
        if is_chosen
        else "border: 1px solid rgba(15, 23, 42, 0.18);"
    )
    prob_text = "" if prob == 0.0 else (f"{prob:.2f}" if prob >= 0.005 else f"{prob:.0e}")
    return (
        f"<span title=\"p={prob:.4f}\" style=\"{bg} {border} "
        f"display:inline-flex; flex-direction:column; align-items:center;"
        f" border-radius:6px; padding:3px 6px;"
        f" font-family:monospace; line-height:1.15; min-width:36px;\">"
        f"<span style=\"font-size:13px;\">{tok}</span>"
        f"<span style=\"font-size:8px; opacity:0.75; margin-top:1px;\">"
        f"{prob_text}</span>"
        f"</span>"
    )


def _write_html_file(path: Path, body_fragment: str, title: str = "Token probabilities") -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_title = html.escape(title, quote=True)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #f8fafc; }}
</style>
</head>
<body>
{body_fragment}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def _build_colorbar_html() -> str:
    stops = []
    for i in range(11):
        p = i / 10.0
        stops.append(f"{_blue_at(p ** 0.5)} {i * 10}%")
    gradient = ", ".join(stops)
    red_swatch = (
        "<span style=\"display:inline-block; width:14px; height:12px;"
        " border-radius:3px; border:1px solid rgba(15, 23, 42, 0.18);"
        " background-color: hsl(0, 75%, 88%);\"></span>"
    )
    return (
        "<div style='display:flex; align-items:center; gap:8px;"
        " margin:8px 0 14px 0; font-size:12px; color:#334155;'>"
        "<span>Token probability</span>"
        "<span>0.0</span>"
        f"<span style=\"display:inline-block; width:240px; height:12px;"
        f" border-radius:3px; border:1px solid rgba(15, 23, 42, 0.18);"
        f" background: linear-gradient(to right, {gradient});\"></span>"
        "<span>1.0</span>"
        f"<span style='margin-left:16px;'>{red_swatch}</span>"
        f"<span>Outside nucleus</span>"
        "</div>"
    )


def visualize_columns(
    cols: list,
    *,
    start: int = 0,
    count: int | None = None,
    max_row_width_px: int = DEFAULT_MAX_ROW_WIDTH_PX,
    output_path: Path,
    title: str,
) -> Path:
    """Build the token-column HTML and write a standalone UTF-8 document."""
    end = len(cols) if count is None else start + count
    sliced = cols[start:end]
    if not sliced:
        fragment = "<p><em>No tokens in this slice.</em></p>"
        actual_end = start
    else:
        column_htmls = []
        for column in sliced:
            chips = "".join(
                render_token_chip(tok_str, p, is_chosen)
                for tok_str, p, is_chosen in column
            )
            column_htmls.append(
                "<div style='display:flex; flex-direction:column; gap:3px;"
                " align-items:stretch; flex:0 0 auto;'>"
                + chips
                + "</div>"
            )
        row_html = (
            f"<div style='display:flex; flex-flow:row wrap; gap:16px 6px;"
            f" align-items:flex-start; max-width:{max_row_width_px}px;'>"
            + "".join(column_htmls)
            + "</div>"
        )
        actual_end = start + len(sliced)
        header_html = (
            "<div style='margin:0 0 6px 0; font-size:12px; color:#475569;'>"
            f"Tokens [{start}, {actual_end})"
            "</div>"
        )
        fragment = header_html + _build_colorbar_html() + row_html

    _write_html_file(output_path, fragment, title=title)
    print(f"Wrote HTML: {output_path.resolve()}")
    return output_path


def build_columns_from_trace(trace: list[dict], *, max_per_col: int = MAX_PER_COL) -> list:
    """Per-step column = chosen chip first, then nucleus alternates (descending
    prob, the chosen token removed), capped at ``max_per_col``. Strings are taken
    straight from the cached trace — no tokenizer needed to re-render."""
    def disp(s: str) -> str:
        return s.replace("\n", _NL)

    columns = []
    for step in trace:
        chosen_id = step["chosen_id"]
        column = [(disp(step["chosen_str"]), float(step["chosen_prob"]), True)]
        for tid, tstr, p in zip(step["nuc_ids"], step["nuc_strs"], step["nuc_probs"]):
            if int(tid) == int(chosen_id):
                continue
            column.append((disp(tstr), float(p), False))
            if len(column) >= max_per_col:
                break
        columns.append(column)
    return columns


# --------------------------------------------------------------------------- #
# Rollout selection + trace (compute) + cache.
# --------------------------------------------------------------------------- #
def select_rollout(sample_idx: int) -> dict:
    """Pick the geometry/627 rollout from the naturally-sampled passK pool."""
    from math_rollouts.data.hf import load_generation_parquet

    df = load_generation_parquet(MODEL_ID, PASSK_NAME)
    g = df[(df.math500_native_id == PROBLEM_ID) & (df.sample_idx == sample_idx)]
    if len(g) != 1:
        raise SystemExit(
            f"expected exactly 1 rollout for {PROBLEM_ID} sample_idx={sample_idx}, "
            f"got {len(g)}"
        )
    r = g.iloc[0]
    if not bool(r.is_correct):
        print(f"WARNING: selected rollout (sample_idx={sample_idx}) is NOT correct")
    return {
        "completion_token_ids": [int(t) for t in r.completion_token_ids],
        "completion_text": str(r.completion_text),
        "answer": str(r.answer),
        "num_tokens": int(r.num_tokens),
        "sample_idx": int(r.sample_idx),
        "is_correct": bool(r.is_correct),
    }


def compute_trace(rollout: dict, *, device: str) -> tuple[list[dict], dict]:
    """Load the HF model, build the Qwen-Math prompt, and teacher-force the rollout
    to recover the per-token nucleus. Decodes nucleus strings here so the cache is
    self-contained for re-rendering."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from math_rollouts.adapters.qwen_math import QwenMathAdapter
    from math_rollouts.config import GenConfig
    from math_rollouts.data.problems import load_math500_by_ids
    from math_rollouts.nucleus import trace_nuclei

    problem = load_math500_by_ids([PROBLEM_ID])
    if not problem:
        raise SystemExit(f"{PROBLEM_ID} not found in MATH-500")
    problem = problem[0]

    # bfloat16 ON BOTH cpu/cuda — deliberately, not fp32. The rollouts were sampled
    # by vLLM in bf16 and the blog's nucleus figure is bf16; for geometry/627 the
    # 'To'/'Please'/'Let' first-token logits are nearly tied, so fp32 reshuffles the
    # nucleus (promotes 'Please', pulls in 'Given') and the first chip would not
    # match the blog. Match the generation engine's precision. (See trace.py's
    # ENGINE CAVEAT.)
    dtype = torch.bfloat16
    print(f"Loading {MODEL_ID} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()

    cfg = GenConfig()
    adapter = QwenMathAdapter(MODEL_ID)
    prompt_ids = adapter.prompt_ids(problem, tok)
    print(f"Prompt tokens: {len(prompt_ids)} | completion tokens: "
          f"{len(rollout['completion_token_ids'])}")

    trace = trace_nuclei(model, tok, prompt_ids,
                         rollout["completion_token_ids"], cfg, device=device)
    for step in trace:
        step["nuc_strs"] = [tok.decode([i]) for i in step["nuc_ids"]]

    # Sanity: the first response token's nucleus should match the blog figure
    # (nucleus {The, To, Please, Let}; this rollout takes "To" at ~0.27).
    s0 = trace[0]
    head = ", ".join(f"{repr(t)}={p:.3f}"
                     for t, p in zip(s0["nuc_strs"], s0["nuc_probs"]))
    print(f"first-token nucleus: {head}")
    print(f"  chosen {repr(s0['chosen_str'])} p={s0['chosen_prob']:.4f}")

    meta = {
        "model_id": MODEL_ID,
        "problem_id": PROBLEM_ID,
        "answer": rollout["answer"],
        "sample_idx": rollout["sample_idx"],
        "is_correct": rollout["is_correct"],
        "num_tokens": rollout["num_tokens"],
        "gen_config": cfg.as_dict(),
        "n_prompt_tokens": len(prompt_ids),
        "eos_id": int(tok.eos_token_id),
        "completion_text": rollout["completion_text"],
    }
    return trace, meta


def save_trace(cache_path: Path, trace: list[dict], meta: dict) -> None:
    import pandas as pd

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_parquet(cache_path, index=False)
    cache_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote trace cache: {cache_path.resolve()} ({len(trace)} steps)")


def load_trace(cache_path: Path) -> tuple[list[dict], dict]:
    import pandas as pd

    df = pd.read_parquet(cache_path)
    trace = []
    for _, r in df.iterrows():
        trace.append({
            "step": int(r["step"]),
            "chosen_id": int(r["chosen_id"]),
            "chosen_str": str(r["chosen_str"]),
            "chosen_prob": float(r["chosen_prob"]),
            "nuc_ids": [int(x) for x in r["nuc_ids"]],
            "nuc_probs": [float(x) for x in r["nuc_probs"]],
            "nuc_strs": [str(x) for x in r["nuc_strs"]],
        })
    meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    return trace, meta


# --------------------------------------------------------------------------- #
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-idx", type=int, default=DEFAULT_SAMPLE_IDX)
    ap.add_argument("--cache", type=Path,
                    default=here / "data" / "geometry_627_to_solve_trace.parquet")
    ap.add_argument("--out", type=Path,
                    default=here / "html" / "math500_geometry_627_to_solve.html")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="render only the first N steps (default: the whole rollout)")
    ap.add_argument("--max-per-col", type=int, default=MAX_PER_COL)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--regen", action="store_true",
                    help="recompute the trace even if the cache exists")
    args = ap.parse_args()

    if args.cache.exists() and not args.regen:
        print(f"loading cached trace from {args.cache}")
        trace, meta = load_trace(args.cache)
    else:
        device = args.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        rollout = select_rollout(args.sample_idx)
        trace, meta = compute_trace(rollout, device=device)
        save_trace(args.cache, trace, meta)

    # Drop a trailing EOS step (present when the rollout finished on "stop"): it
    # sits after the boxed answer and renders as an ugly "<|endoftext|>" chip.
    eos_id = meta.get("eos_id")
    while trace and eos_id is not None and trace[-1]["chosen_id"] == eos_id:
        trace = trace[:-1]

    cols = build_columns_from_trace(trace, max_per_col=args.max_per_col)
    pid = meta["problem_id"].replace("test/", "").replace(".json", "")
    cfg = meta["gen_config"]
    verdict = "correct" if meta.get("is_correct") else "incorrect"
    title = (
        f"MATH-500 {pid} — naturally-sampled rollout (sample {meta['sample_idx']}, "
        f"{verdict}, answer {meta['answer']}) — {MODEL_ID.split('/')[-1]} "
        f"(T={cfg['temperature']}, top_p={cfg['top_p']}, top_k={cfg['top_k']})"
    )
    visualize_columns(cols, count=args.max_tokens, output_path=args.out, title=title)


if __name__ == "__main__":
    main()
