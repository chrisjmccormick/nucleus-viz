"""Token-probability chip strip for one MATH-500 rollout (Qwen2.5-Math-1.5B).

Pulls a real, naturally-sampled rollout for ``geometry/627`` from the
``math-rollouts`` HF dataset and writes standalone HTML where each generated token
is a chip showing the chosen token plus its nucleus alternates, colored by
probability (T=0.6 / top_p=0.95 / top_k=20 — the dataset's canonical config).

The per-token nuclei come from one of two sources (``--source``):
  * **store** (preferred): the precomputed per-token nucleus shards in the
    math-rollouts dataset (``<pool>_token_nuclei/``) — no model, no GPU.
  * **generate**: a teacher-forced bf16 HF forward pass (``trace_nuclei``).
``auto`` (default) tries the store and falls back to generating if it isn't there
yet. Either way the resulting trace is cached to ``data/`` so tweaking the visual
never recomputes; rebuild with ``--regen``.

Default rollout: the shortest CORRECT, cleanly-reasoned sample for geometry/627 —
``math500_passK`` sample_idx=10, which opens "To solve this problem ..." and works
through to ``\\boxed{17}``.

Data comes through the math-rollouts interface (``pip install -e ../math-rollouts``).

Usage:
    python html_tokens.py                 # cached trace if present, else store/generate
    python html_tokens.py --regen         # rebuild (store if available, else model)
    python html_tokens.py --source generate --regen   # force the model path

Optional chip-strip modes (all opt-in; the base render is unchanged without them):
    python html_tokens.py --newline-breaks            # one non-wrapping row per source
                                                      #   line (newline / display-math \\[),
                                                      #   scrolling horizontally
    python html_tokens.py --show-prompt               # prompt-template reference chips
    python html_tokens.py --show-positions            # token position above each column
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
PASSK_NAME = "math500_passK"             # naturally-sampled pool in the HF dataset
UNIQUE_ID = "math500/geometry/9467"      # split-aware id (HF MATH-500 geometry/627)
PROBLEM_ID = "test/geometry/627.json"    # HF MATH-500 native id (for prompt text + title)
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


# --------------------------------------------------------------------------- #
# Optional chip-strip modes (promoted from
# agent-ops/nucleus-viz/2026-07-25_0750am_chip-linebreaks-prompt-ref):
#   --newline-breaks  one row per source line (newline / display-math \[)
#   --show-prompt     a prompt-template reference strip after the legend
#   --show-positions  a token-position label above each column
# All opt-in; with no flags the render is unchanged.
# --------------------------------------------------------------------------- #
_BSL = "\\"


def render_ref_chip(token_str: str) -> str:
    """A small, deliberately-quiet reference chip: white bg, soft-grey border + text,
    token text only (no probability, no shading). Used for the prompt-template strip."""
    tok = html.escape(token_str).replace(" ", "&nbsp;")
    if tok == "":
        tok = "&nbsp;"
    return (
        "<span style=\"background:#ffffff; border:1px solid #d8dee6; color:#9aa4b1;"
        " border-radius:5px; padding:2px 5px; font-family:monospace; font-size:11px;"
        " line-height:1.25; display:inline-flex; align-items:center;\">"
        f"{tok}</span>"
    )


def _pos_label(idx: int) -> str:
    """A small, muted 0-based token-position marker above a column (the alternates below
    share the same position, so it labels the column, not each alternate)."""
    return ("<div style='font-size:8px; color:#94a3b8; font-family:monospace;"
            f" line-height:1; text-align:center; margin-bottom:2px;'>{idx}</div>")


def compute_break_before(strs: list[str]) -> list[bool]:
    """``break_before[i]`` — should column ``i`` start a new visual line?

    * newline: the previous token's text contains ``\\n`` -> break before this one.
    * display math: a ``\\[`` opener begins here -> break before it. Qwen tokenises ``\\[``
      as a lone ``\\`` token then a ``[…`` token, so we break before the backslash (keeping
      ``\\[`` at the start of the equation's own line); the single-token form is handled too.
    """
    n = len(strs)
    bb = [False] * n
    for i in range(1, n):
        if "\n" in strs[i - 1]:
            bb[i] = True
    for j in range(n):
        s = strs[j]
        if (_BSL + "[") in s:
            bb[j] = True
        elif s.startswith("[") and j > 0 and strs[j - 1].endswith(_BSL):
            bb[j - 1] = True
    if n:
        bb[0] = False
    return bb


def _group_lines(items: list, break_before: list[bool]) -> list[list]:
    """Split ``items`` into runs (logical lines) at each ``break_before`` boundary."""
    lines: list[list] = []
    cur: list = []
    for i, it in enumerate(items):
        if break_before[i] and cur:
            lines.append(cur)
            cur = []
        cur.append(it)
    if cur:
        lines.append(cur)
    return lines


def _column_html(idx: int, column: list, show_pos: bool) -> str:
    chips = "".join(render_token_chip(t, p, c) for t, p, c in column)
    head = _pos_label(idx) if show_pos else ""
    return ("<div style='display:flex; flex-direction:column; gap:3px;"
            " align-items:stretch; flex:0 0 auto;'>" + head + chips + "</div>")


def _render_flow(cols: list, break_before, start: int, max_w: int,
                 show_pos: bool, wrap_lines: bool) -> str:
    """Lay out the (already-sliced) columns. ``break_before is None`` -> the legacy single
    wrap-row. Otherwise one row per source line; ``wrap_lines`` wraps each line at ``max_w``,
    else each line is non-wrapping and the whole strip scrolls horizontally."""
    if break_before is None:
        col_htmls = [_column_html(start + i, c, show_pos) for i, c in enumerate(cols)]
        return ("<div style='display:flex; flex-flow:row wrap; gap:16px 6px;"
                f" align-items:flex-start; max-width:{max_w}px;'>" + "".join(col_htmls) + "</div>")
    lines = _group_lines(list(enumerate(cols)), break_before)
    flow = "row wrap" if wrap_lines else "row nowrap"
    cap = f" max-width:{max_w}px;" if wrap_lines else ""
    line_htmls = [
        f"<div style='display:flex; flex-flow:{flow}; gap:6px; align-items:flex-start;{cap}'>"
        + "".join(_column_html(start + i, c, show_pos) for i, c in line) + "</div>"
        for line in lines
    ]
    if wrap_lines:
        return (f"<div style='display:flex; flex-direction:column; gap:10px; max-width:{max_w}px;'>"
                + "".join(line_htmls) + "</div>")
    inner = ("<div style='display:inline-flex; flex-direction:column; gap:10px;"
             " align-items:flex-start;'>" + "".join(line_htmls) + "</div>")
    return f"<div style='overflow-x:auto; padding-bottom:8px;'>{inner}</div>"


def prompt_template_block(strs: list[str], max_w: int, wrap: bool = False) -> str:
    """The prompt template as quiet reference chips, one row per template line. ``wrap=False``
    (default) keeps each line non-wrapping in an ``overflow-x:auto`` strip."""
    bb = compute_break_before(strs)
    disp = [s.replace("\n", _NL) for s in strs]
    lines = _group_lines(disp, bb)
    flow = "row wrap" if wrap else "row nowrap"
    cap = f" max-width:{max_w}px;" if wrap else ""
    line_htmls = [
        f"<div style='display:flex; flex-flow:{flow}; gap:3px; align-items:center;{cap}'>"
        + "".join(render_ref_chip(s) for s in line) + "</div>"
        for line in lines
    ]
    label = ("<div style='font-size:11px; letter-spacing:0.06em; text-transform:uppercase;"
             " color:#94a3b8; margin:2px 0 5px 0;'>prompt template</div>")
    if wrap:
        strip = (f"<div style='display:flex; flex-direction:column; gap:3px; max-width:{max_w}px;'>"
                 + "".join(line_htmls) + "</div>")
    else:
        inner = ("<div style='display:inline-flex; flex-direction:column; gap:3px;"
                 " align-items:flex-start;'>" + "".join(line_htmls) + "</div>")
        strip = f"<div style='overflow-x:auto; padding-bottom:6px;'>{inner}</div>"
    return f"<div style='margin:4px 0 16px 0;'>{label}{strip}</div>"


def prompt_template_strs(meta: dict, cache_json: Path) -> list[str]:
    """Tokenise the Qwen-Math prompt template (boxed-instruction system turn + the problem in
    the user turn + the assistant opener) into per-token strings, cached into the trace's json
    (``prompt_strs``) so re-renders stay offline."""
    cached = meta.get("prompt_strs")
    if cached:
        return list(cached)
    from transformers import AutoTokenizer

    from math_rollouts.adapters.qwen_math import apply_qwen_math_template

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ids = tok(apply_qwen_math_template(meta["problem_text"]),
              add_special_tokens=False).input_ids
    strs = [tok.decode([i]) for i in ids]
    meta["prompt_strs"] = strs
    cache_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return strs


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
    return (
        "<div style='display:flex; align-items:center; gap:8px;"
        " margin:8px 0 14px 0; font-size:12px; color:#334155;'>"
        "<span>Token probability</span>"
        "<span>0.0</span>"
        f"<span style=\"display:inline-block; width:240px; height:12px;"
        f" border-radius:3px; border:1px solid rgba(15, 23, 42, 0.18);"
        f" background: linear-gradient(to right, {gradient});\"></span>"
        "<span>1.0</span>"
        "</div>"
    )


# Problem card + stats strip ported from html_token_losses.py so both figures
# open the same way (problem text, then length/singleton/verdict cards).
def _build_problem_card(meta: dict) -> str:
    # Display-only: this problem's TeX is light enough that dropping the $
    # delimiters reads as plain text.
    problem = html.escape(meta["problem_text"].replace("$", ""))
    pid = meta["problem_id"].replace("test/", "").replace(".json", "")
    return (
        "<div style='background:#ffffff; border:1px solid rgba(15, 23, 42, 0.12);"
        " border-left:4px solid #1e3a8a; border-radius:8px; padding:10px 16px;"
        f" max-width:{DEFAULT_MAX_ROW_WIDTH_PX - 20}px; margin:0 0 10px 0;'>"
        "<div style='font-size:11px; letter-spacing:0.08em; text-transform:uppercase;"
        f" color:#64748b; margin-bottom:4px;'>MATH-500 &middot; {pid}</div>"
        "<div style='font-size:15px; font-family:Georgia, \"Times New Roman\", serif;"
        f" color:#1e293b;'>{problem}</div>"
        "</div>"
    )


def _build_stats_html(trace: list[dict], meta: dict) -> str:
    n = len(trace)
    singles = sum(1 for s in trace if len(s["nuc_ids"]) == 1)
    branches = n - singles

    def stat_block(value: str, label: str) -> str:
        return (
            "<div style='background:#ffffff; border:1px solid rgba(15, 23, 42, 0.12);"
            " border-radius:8px; padding:6px 16px; text-align:center;'>"
            f"<div style='font-size:18px; font-weight:600; color:#1e293b;'>{value}</div>"
            f"<div style='font-size:11px; color:#64748b;'>{label}</div>"
            "</div>"
        )

    if meta["is_correct"]:
        verdict = (
            "<div style='background:hsl(140, 45%, 92%); border:1px solid"
            " hsl(140, 35%, 78%); border-radius:8px; padding:6px 16px;"
            " text-align:center;'>"
            "<div style='font-size:18px; font-weight:600; color:#14532d;'>&#10003;"
            " correct</div>"
            f"<div style='font-size:11px; color:#3f6212;'>answer {meta['answer']}</div>"
            "</div>"
        )
    else:
        verdict = (
            "<div style='background:hsl(0, 55%, 94%); border:1px solid"
            " hsl(0, 45%, 82%); border-radius:8px; padding:6px 16px;"
            " text-align:center;'>"
            "<div style='font-size:18px; font-weight:600; color:#7f1d1d;'>&#10007;"
            " incorrect</div>"
            f"<div style='font-size:11px; color:#9f1239;'>answer is {meta['answer']}</div>"
            "</div>"
        )

    return (
        "<div style='display:flex; gap:10px; margin:0 0 12px 0; flex-wrap:wrap;'>"
        + stat_block(f"{n}", "tokens")
        + stat_block(f"{singles} ({singles / n:.0%})", "singletons")
        + stat_block(f"{branches} ({branches / n:.0%})", "branch tokens")
        + verdict
        + "</div>"
    )


def visualize_columns(
    cols: list,
    *,
    start: int = 0,
    count: int | None = None,
    max_row_width_px: int = DEFAULT_MAX_ROW_WIDTH_PX,
    output_path: Path,
    title: str,
    heading_html: str = "",
    break_before: list[bool] | None = None,
    show_pos: bool = False,
    prompt_html: str = "",
    wrap_lines: bool = False,
) -> Path:
    """Build the token-column HTML and write a standalone UTF-8 document.

    Optional modes (all off by default, so the base render is unchanged):
      * ``break_before`` — a per-column list (aligned to ``cols``); lay out one row per
        source line instead of a single wrap-column (see ``compute_break_before``).
      * ``show_pos`` — print the 0-based token position above each column.
      * ``prompt_html`` — a reference strip inserted right after the colour bar.
      * ``wrap_lines`` — with ``break_before``, wrap each line at ``max_row_width_px``
        rather than letting it run on in a horizontally-scrolling strip.
    """
    end = len(cols) if count is None else start + count
    sliced = cols[start:end]
    bb = None
    if break_before is not None:
        bb = list(break_before[start:end])
        if bb:
            bb[0] = False                 # the first shown column never leads with a break
    if not sliced:
        fragment = "<p><em>No tokens in this slice.</em></p>"
    else:
        flow = _render_flow(sliced, bb, start, max_row_width_px, show_pos, wrap_lines)
        fragment = _build_colorbar_html() + prompt_html + flow

    _write_html_file(output_path, heading_html + fragment, title=title)
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
    g = df[(df.unique_id == UNIQUE_ID) & (df.sample_idx == sample_idx)]
    if len(g) != 1:
        raise SystemExit(
            f"expected exactly 1 rollout for {UNIQUE_ID} sample_idx={sample_idx}, "
            f"got {len(g)}"
        )
    r = g.iloc[0]
    if not bool(r.answer_matches):
        print(f"WARNING: selected rollout (sample_idx={sample_idx}) is NOT correct")
    return {
        "unique_id": str(r.unique_id),               # math12k id, for the store lookup
        "completion_token_ids": [int(t) for t in r.completion_token_ids],
        "completion_text": str(r.completion_text),
        "answer": str(r.answer),
        "num_tokens": int(r.completion_num_tokens),
        "sample_idx": int(r.sample_idx),
        "is_correct": bool(r.answer_matches),
    }


def problem_text() -> str:
    from math_rollouts.data.problems import load_problems_by_ids

    problems = load_problems_by_ids([UNIQUE_ID])
    if not problems:
        raise SystemExit(f"{UNIQUE_ID} not found in math_problems")
    return str(problems[0]["problem"])


def pull_trace_from_store(rollout: dict, *, pool: str = PASSK_NAME) -> tuple[list[dict], dict]:
    """Rebuild the trace from the math-rollouts per-token nucleus store — no model.

    The store keeps, per position, the kept token ids + raw logits (nucleus members
    first, ``nuc_sizes`` of them, plus a few out-of-nucleus alternates). The chip
    probabilities are the nucleus renormalized — ``softmax(logit/T)`` over the
    nucleus members — which needs only the stored nucleus logits. The chosen token
    comes from the rollout; its prob is 0 if it fell outside the recomputed nucleus.
    Raises if the store shard isn't present (caller falls back to generating)."""
    import math

    from transformers import AutoTokenizer

    from math_rollouts.config import GenConfig
    from math_rollouts.data.hf import load_token_nuclei

    df = load_token_nuclei(MODEL_ID, pool, rollout["unique_id"])
    g = df[df.sample_idx == rollout["sample_idx"]]
    if len(g) != 1:
        raise FileNotFoundError(
            f"no stored nuclei row for {rollout['unique_id']} sample_idx="
            f"{rollout['sample_idx']}")
    row = g.iloc[0]

    cfg = GenConfig()
    T = cfg.temperature
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    nuc_sizes = [int(x) for x in row["nuc_sizes"]]
    keep_counts = [int(x) for x in row["keep_counts"]]
    kept_ids = [int(x) for x in row["kept_ids"]]
    kept_logits = [float(x) for x in row["kept_logits"]]
    chosen_ids = rollout["completion_token_ids"]
    if len(nuc_sizes) != len(chosen_ids):
        raise ValueError(f"store ({len(nuc_sizes)}) / rollout ({len(chosen_ids)}) "
                         "length mismatch")

    trace, off = [], 0
    for t, kc in enumerate(keep_counts):
        size = nuc_sizes[t]
        ids = kept_ids[off:off + size]              # nucleus members come first
        logits = kept_logits[off:off + size]
        off += kc
        mx = max(logits)
        exps = [math.exp((l - mx) / T) for l in logits]
        Z = sum(exps)
        probs = [e / Z for e in exps]               # renormalized within the nucleus
        chosen = int(chosen_ids[t])
        chosen_prob = probs[ids.index(chosen)] if chosen in ids else 0.0
        trace.append({
            "step": t, "chosen_id": chosen, "chosen_str": tok.decode([chosen]),
            "chosen_prob": float(chosen_prob), "nuc_ids": ids,
            "nuc_probs": [float(p) for p in probs],
            "nuc_strs": [tok.decode([i]) for i in ids],
        })

    meta = {
        "model_id": MODEL_ID, "problem_id": PROBLEM_ID, "answer": rollout["answer"],
        "sample_idx": rollout["sample_idx"], "is_correct": rollout["is_correct"],
        "num_tokens": rollout["num_tokens"], "gen_config": cfg.as_dict(),
        "eos_id": int(tok.eos_token_id), "completion_text": rollout["completion_text"],
        "problem_text": problem_text(), "source": "store",
    }
    return trace, meta


def compute_trace(rollout: dict, *, device: str) -> tuple[list[dict], dict]:
    """Load the HF model, build the Qwen-Math prompt, and teacher-force the rollout
    to recover the per-token nucleus. Decodes nucleus strings here so the cache is
    self-contained for re-rendering."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from math_rollouts.adapters.qwen_math import QwenMathAdapter
    from math_rollouts.config import GenConfig
    from math_rollouts.data.problems import load_problems_by_ids
    from math_rollouts.nucleus import trace_nuclei

    problem = load_problems_by_ids([UNIQUE_ID])
    if not problem:
        raise SystemExit(f"{UNIQUE_ID} not found in math_problems")
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
        "problem_text": str(problem["problem"]),
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
    ap.add_argument("--source", choices=["auto", "store", "generate"], default="auto",
                    help="auto (default): pull the per-token nuclei from the "
                         "math-rollouts store, fall back to generating them via the "
                         "model; store: require the store; generate: always recompute")
    ap.add_argument("--regen", action="store_true",
                    help="rebuild the trace even if the cache exists")
    ap.add_argument("--newline-breaks", action="store_true",
                    help="lay out one row per source line (break after a newline token and "
                         "before a display-math \\[ opener) instead of a single wrap-column; "
                         "each line is non-wrapping and the strip scrolls horizontally")
    ap.add_argument("--wrap-lines", action="store_true",
                    help="with --newline-breaks, wrap each line at the max row width instead "
                         "of scrolling horizontally")
    ap.add_argument("--show-prompt", action="store_true",
                    help="show the prompt template as small grey reference chips after the legend")
    ap.add_argument("--show-positions", action="store_true",
                    help="print the 0-based token position above each column")
    args = ap.parse_args()

    if args.cache.exists() and not args.regen:
        print(f"loading cached trace from {args.cache}")
        trace, meta = load_trace(args.cache)
    else:
        rollout = select_rollout(args.sample_idx)
        trace = meta = None
        if args.source in ("auto", "store"):
            try:
                trace, meta = pull_trace_from_store(rollout)
                print(f"pulled per-token nuclei from the math-rollouts store "
                      f"({len(trace)} steps)")
            except Exception as e:               # store missing / not yet generated
                if args.source == "store":
                    raise
                print(f"store unavailable ({type(e).__name__}: {e}); generating instead")
        if trace is None:
            device = args.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            trace, meta = compute_trace(rollout, device=device)
        save_trace(args.cache, trace, meta)

    # Older caches predate the problem-card heading; backfill so the render
    # stays offline next time.
    if "problem_text" not in meta:
        meta["problem_text"] = problem_text()
        args.cache.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

    cols = build_columns_from_trace(trace, max_per_col=args.max_per_col)

    break_before = None
    if args.newline_breaks:
        break_before = compute_break_before([step["chosen_str"] for step in trace])
    prompt_html = ""
    if args.show_prompt:
        prompt_html = prompt_template_block(
            prompt_template_strs(meta, args.cache.with_suffix(".json")),
            DEFAULT_MAX_ROW_WIDTH_PX, wrap=args.wrap_lines)

    pid = meta["problem_id"].replace("test/", "").replace(".json", "")
    cfg = meta["gen_config"]
    verdict = "correct" if meta.get("is_correct") else "incorrect"
    title = (
        f"MATH-500 {pid} — naturally-sampled rollout (sample {meta['sample_idx']}, "
        f"{verdict}, answer {meta['answer']}) — {MODEL_ID.split('/')[-1]} "
        f"(T={cfg['temperature']}, top_p={cfg['top_p']}, top_k={cfg['top_k']})"
    )
    heading = _build_problem_card(meta) + _build_stats_html(trace, meta)
    visualize_columns(cols, count=args.max_tokens, output_path=args.out, title=title,
                      heading_html=heading, break_before=break_before,
                      show_pos=args.show_positions, prompt_html=prompt_html,
                      wrap_lines=args.wrap_lines)


if __name__ == "__main__":
    main()
