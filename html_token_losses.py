"""Per-token GRPO loss chip strip for MATH-500 rollouts (Qwen2.5-Math-1.5B).

Companion to ``html_tokens.py`` (post 1's token-probability strip). Renders the
quantities that RL training actually sees: for every generated token, a column of
two chips —

  1. the token — light grey when its sampling nucleus is a singleton (post 1's
     sense: only one possible token at T=0.6 / top_p=0.95), otherwise colored by
     the training policy's probability of it (full-vocab softmax at T=1.0),
  2. the per-token loss ``-A * log p(token)`` (yellow by magnitude,
     light grey when there is essentially no learning signal).

The advantage A is one scalar shared by every token of the rollout; the full-
sequence figure leaves it to the blog text entirely. Each page opens with the
math problem and rollout stats (length, singletons, verdict) instead.

A second, zoomed-in figure per rollout shows the loss arithmetic on the opening
tokens: rows of token / -log p (nats) / x A / = loss, with the green (or red)
advantage chips that the full-sequence figure drops.

A third figure illustrates fine-tuning: the dataset's reference solution is
teacher-forced as the model's response (plain cross-entropy, ``loss = -log p``,
no advantage). Unlike RL rollouts, this text did not come from the model's own
sampling tree — tokens that fall outside the model's T=0.6/top_p=0.95 nucleus
render in burgundy, and the loss ramp extends from yellow into deep orange to
carry the much larger losses that out-of-distribution tokens produce.

Two figures, from the same naturally-sampled ``math500_passK`` group for
geometry/627: the correct rollout used throughout post 1 (sample 10, positive
advantage) and a clean incorrect rollout (negative advantage).

The probabilities come from a teacher-forced bf16 forward pass over the cached
rollout token ids — the math-rollouts nucleus store only keeps nucleus-member
logits, which can't reconstruct the full softmax denominator. The resulting loss
trace is cached to ``data/`` so tweaking the visual never touches the model;
rebuild with ``--regen``.

The advantage is the GRPO/DAPO group-relative one, ``A = (r - mean) / std`` over
the group's binary rewards — with r in {0, 1} it is fully determined by the
group's size and number of correct rollouts, which are baked in below.

Usage:
    python html_token_losses.py                     # render both from cache
    python html_token_losses.py --example correct   # just the positive-A figure
    python html_token_losses.py --regen             # teacher-force again
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
PASSK_NAME = "math500_passK"             # naturally-sampled pool in the HF dataset
# Problem ids: 9467 is the canonical integer id (the math12k-style numbering the
# dataset standardized on; test problems are > 7500). "627" is the legacy id from
# the separate HF MATH-500 dataset — kept only for display/file names because
# post 1's figures already use it.
UNIQUE_ID = "math500/geometry/9467"
PROBLEM_ID = "test/geometry/627.json"

# Group stats for UNIQUE_ID in math500_passK (counted from the pool 2026-06-10):
# 64 rollouts, 15 with answer_matches. Binary rewards make these two numbers the
# whole advantage calculation: mean = 15/64, std = sqrt(mean * (1 - mean)).
GROUP_SIZE = 64
NUM_CORRECT = 15

CORRECT_SAMPLE_IDX = 10                  # post 1's rollout: shortest correct "To solve ..."
# Shortest *clean* incorrect rollout. Samples 22/25/59 are shorter but degenerate
# (echo the question / emit garbage); sample 48 also opens by echoing the question.
# Sample 36 is a genuine wrong attempt: reasons step-by-step to \boxed{5} (answer: 17).
INCORRECT_SAMPLE_IDX = 36

LOSS_GREY_THRESHOLD = 0.05               # |loss| below this renders grey ("nothing to learn")
DEFAULT_MAX_ROW_WIDTH_PX = 1024
_NL = "↵"                            # ↵ shown in place of a newline

# The MATH reference solution carries an [asy] diagram block; strip it so the
# SFT figure shows the prose (which ends at the boxed answer), not Asymptote code.
SOLUTION_STRIP_ASY = True


# --------------------------------------------------------------------------- #
# Advantage (GRPO/DAPO group-relative, binary rewards).
# --------------------------------------------------------------------------- #
def group_advantage(reward: float) -> float:
    p_bar = NUM_CORRECT / GROUP_SIZE
    std = math.sqrt(p_bar * (1.0 - p_bar))
    if std == 0.0:
        raise SystemExit("group is all-correct or all-incorrect; advantage undefined")
    return (reward - p_bar) / std


# --------------------------------------------------------------------------- #
# Chip rendering. Branch-token chips reuse the post-1 blue recipe, singletons go
# grey so the branches stand out; the loss chip is yellow by |loss| with grey
# for ~0.
# --------------------------------------------------------------------------- #
def _blue_at(intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    lightness = 96 - 64 * intensity
    saturation = 25 + 55 * intensity
    return f"hsl(210, {saturation:.0f}%, {lightness:.0f}%)"


def prob_to_blue_style(p: float) -> str:
    p = max(0.0, min(1.0, float(p)))
    intensity = p ** 0.5
    bg = _blue_at(intensity)
    text_color = "#ffffff" if intensity > 0.6 else "#1e293b"
    return f"background-color: {bg}; color: {text_color};"


def _yellow_at(intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    lightness = 94 - 39 * intensity
    saturation = 55 + 40 * intensity
    return f"hsl(45, {saturation:.0f}%, {lightness:.0f}%)"


def _hot_at(intensity: float) -> str:
    """Extended loss ramp for the SFT figure: same pale yellow at the bottom,
    but running through orange into a deep burnt hue at the top, so the huge
    losses of out-of-distribution tokens read as a different magnitude class."""
    intensity = max(0.0, min(1.0, intensity))
    hue = 48 - 26 * intensity
    saturation = 55 + 40 * intensity
    lightness = 94 - 52 * intensity
    return f"hsl({hue:.0f}, {saturation:.0f}%, {lightness:.0f}%)"


def loss_to_style(loss: float, loss_max: float, *, hot: bool = False) -> str:
    mag = abs(loss)
    if mag < LOSS_GREY_THRESHOLD:
        return "background-color: hsl(210, 10%, 92%); color: #94a3b8;"
    intensity = (mag / loss_max) ** 0.5 if loss_max > 0 else 0.0
    if hot:
        text = "#ffffff" if intensity > 0.75 else "#431407"
        return f"background-color: {_hot_at(intensity)}; color: {text};"
    return f"background-color: {_yellow_at(intensity)}; color: #422006;"


_SINGLETON_STYLE = "background-color: hsl(210, 10%, 92%); color: #64748b;"
_OOD_STYLE = "background-color: #660033; color: #ffffff;"   # outside the nucleus


def _fmt_prob(p: float) -> str:
    return f"{p:.2f}" if p >= 0.005 else f"{p:.0e}"


def _fmt_loss(loss: float) -> str:
    if loss == 0.0:          # incl. -0.0: a singleton at p=1.0 has exactly no loss
        return "0.00"
    return f"{loss:.2f}" if abs(loss) >= 0.005 else f"{loss:.0e}"


_CHIP_BASE = (
    " display:inline-flex; flex-direction:column; align-items:center;"
    " border-radius:6px; padding:3px 6px;"
    " font-family:monospace; line-height:1.15; min-width:36px;"
)


def render_token_chip(token_str: str, prob: float, nuc_size: int,
                      in_nucleus: bool = True) -> str:
    tok = html.escape(token_str).replace(" ", "&nbsp;")
    if tok == "":
        tok = "&nbsp;"
    if not in_nucleus:
        bg = _OOD_STYLE
        tip = f"p={prob:.2e} — OUTSIDE the sampling nucleus (size {nuc_size})"
    else:
        bg = _SINGLETON_STYLE if nuc_size == 1 else prob_to_blue_style(prob)
        tip = f"p={prob:.4f}, nucleus size {nuc_size}"
    return (
        f"<span title=\"{tip}\" style=\"{bg}"
        f" border: 1px solid rgba(15, 23, 42, 0.18);{_CHIP_BASE}\">"
        f"<span style=\"font-size:13px;\">{tok}</span>"
        f"<span style=\"font-size:8px; opacity:0.75; margin-top:1px;\">"
        f"{_fmt_prob(prob)}</span>"
        f"</span>"
    )


def render_value_chip(text: str, style: str, tooltip: str) -> str:
    return (
        f"<span title=\"{html.escape(tooltip, quote=True)}\" style=\"{style}"
        f" border: 1px solid rgba(15, 23, 42, 0.12);{_CHIP_BASE}\">"
        f"<span style=\"font-size:10px;\">{html.escape(text)}</span>"
        f"</span>"
    )


def build_columns(trace: list[dict], advantage: float | None, loss_max: float,
                  *, hot: bool = False) -> list[str]:
    columns = []
    for step in trace:
        prob = float(step["prob"])
        logp = float(step["logp"])
        if advantage is None:                       # SFT: plain cross-entropy
            loss = -logp
            tip = f"loss = -log p = {loss:.6f}"
        else:
            loss = -advantage * logp
            tip = f"loss = -A * log p = {loss:.6f} (log p = {logp:.6f})"
        token_chip = render_token_chip(step["chosen_str"].replace("\n", _NL), prob,
                                       int(step["nuc_size"]),
                                       bool(step.get("in_nucleus", True)))
        loss_chip = render_value_chip(_fmt_loss(loss),
                                      loss_to_style(loss, loss_max, hot=hot), tip)
        columns.append(
            "<div style='display:flex; flex-direction:column; gap:3px;"
            " align-items:stretch; flex:0 0 auto;'>"
            + token_chip + loss_chip
            + "</div>"
        )
    return columns


def _write_html_file(path: Path, body_fragment: str, title: str) -> Path:
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


def _legend_chip(text: str, style: str) -> str:
    return (
        f"<span style=\"{style} border: 1px solid rgba(15, 23, 42, 0.18);"
        " display:inline-flex; align-items:center; border-radius:6px;"
        f" padding:2px 8px; font-family:monospace; font-size:11px;\">{text}</span>"
    )


def _gradient_bar(stops: str) -> str:
    return (
        f"<span style=\"display:inline-block; width:240px; height:12px;"
        f" border-radius:3px; border:1px solid rgba(15, 23, 42, 0.18);"
        f" background: linear-gradient(to right, {stops});\"></span>"
    )


def _build_legend_html(loss_max: float, *, hot: bool = False,
                       show_ood: bool = False) -> str:
    blue_stops = ", ".join(f"{_blue_at((i / 10.0) ** 0.5)} {i * 10}%"
                           for i in range(11))
    ramp = _hot_at if hot else _yellow_at
    yellow_stops = ", ".join(f"{ramp((i / 10.0) ** 0.5)} {i * 10}%"
                             for i in range(11))
    label = "display:inline-block; width:92px;"
    row = "display:flex; align-items:center; gap:8px; font-size:12px; color:#334155;"
    ood_chip = (
        f"<span style='margin-left:4px;'>"
        f"{_legend_chip('outside nucleus', _OOD_STYLE)}</span>"
        if show_ood else ""
    )
    prob_row = (
        f"<div style='{row}'>"
        f"<span style='{label}'>Probabilities</span>"
        f"<span>0.0</span>{_gradient_bar(blue_stops)}<span>1.0</span>"
        f"<span style='margin-left:12px;'>"
        f"{_legend_chip('singleton', _SINGLETON_STYLE)}</span>"
        f"{ood_chip}"
        "</div>"
    )
    loss_row = (
        f"<div style='{row}'>"
        f"<span style='{label}'>Loss</span>"
        f"<span>0.0</span>{_gradient_bar(yellow_stops)}<span>{loss_max:.2f}</span>"
        f"<span style='margin-left:12px;'>"
        + _legend_chip(f"loss &lt; {LOSS_GREY_THRESHOLD:g}",
                       "background-color: hsl(210, 10%, 92%); color: #94a3b8;")
        + "</span>"
        "</div>"
    )
    return (
        "<div style='display:flex; flex-direction:column; gap:6px;"
        f" margin:0 0 14px 0;'>{prob_row}{loss_row}</div>"
    )


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
    singles = sum(1 for s in trace if int(s["nuc_size"]) == 1)
    branches = n - singles

    def stat_block(value: str, label: str, extra_style: str = "") -> str:
        return (
            "<div style='background:#ffffff; border:1px solid rgba(15, 23, 42, 0.12);"
            f" border-radius:8px; padding:6px 16px; text-align:center;{extra_style}'>"
            f"<div style='font-size:18px; font-weight:600; color:#1e293b;'>{value}</div>"
            f"<div style='font-size:11px; color:#64748b;'>{label}</div>"
            "</div>"
        )

    if meta.get("loss_kind") == "sft":
        ood = sum(1 for s in trace if not s.get("in_nucleus", True))
        verdict = (
            "<div style='background:#660033; border:1px solid #4c0226;"
            " border-radius:8px; padding:6px 16px; text-align:center;'>"
            f"<div style='font-size:18px; font-weight:600; color:#ffffff;'>"
            f"{ood} ({ood / n:.0%})</div>"
            "<div style='font-size:11px; color:#e7b3cc;'>outside nucleus</div>"
            "</div>"
            "<div style='background:#ffffff; border:1px solid rgba(15, 23, 42, 0.12);"
            " border-radius:8px; padding:6px 16px; text-align:center;'>"
            "<div style='font-size:18px; font-weight:600; color:#1e293b;'>&#9998;"
            " dataset solution</div>"
            "<div style='font-size:11px; color:#64748b;'>teacher-forced (SFT)</div>"
            "</div>"
        )
    elif meta["is_correct"]:
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


def visualize_losses(
    trace: list[dict],
    meta: dict,
    *,
    count: int | None = None,
    max_row_width_px: int = DEFAULT_MAX_ROW_WIDTH_PX,
    output_path: Path,
    title: str,
) -> Path:
    sliced = trace[: len(trace) if count is None else count]
    is_sft = meta.get("loss_kind") == "sft"
    advantage = None if is_sft else float(meta["advantage"])
    losses = [(-1.0 if advantage is None else -advantage) * float(s["logp"])
              for s in sliced]
    loss_max = max((abs(l) for l in losses), default=0.0)

    row_html = (
        f"<div style='display:flex; flex-flow:row wrap; gap:16px 6px;"
        f" align-items:flex-start; max-width:{max_row_width_px}px;'>"
        + "".join(build_columns(sliced, advantage, loss_max, hot=is_sft))
        + "</div>"
    )
    # Stats describe the whole rollout even when --max-tokens slices the strip.
    fragment = (_build_problem_card(meta) + _build_stats_html(trace, meta)
                + _build_legend_html(loss_max, hot=is_sft, show_ood=is_sft)
                + row_html)
    _write_html_file(output_path, fragment, title=title)
    print(f"Wrote HTML: {output_path.resolve()}")

    top = sorted(zip(sliced, losses), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("top-10 |loss| tokens:")
    for step, loss in top:
        print(f"  step {step['step']:>4}  {step['chosen_str']!r:<16} "
              f"p={step['prob']:.4f}  loss={loss:+.4f}")
    return output_path


# --------------------------------------------------------------------------- #
# Zoomed-in opening figure: the loss arithmetic, one row per quantity.
# --------------------------------------------------------------------------- #
def _advantage_style(advantage: float) -> str:
    if advantage >= 0:
        return "background-color: hsl(140, 45%, 88%); color: #14532d;"
    return "background-color: hsl(0, 55%, 91%); color: #7f1d1d;"


_NATS_STYLE = "background-color: #ffffff; color: #334155;"


def visualize_opening(
    trace: list[dict],
    meta: dict,
    *,
    n: int,
    output_path: Path,
    title: str,
) -> Path:
    sliced = trace[:n]
    advantage = float(meta["advantage"])
    losses = [-advantage * float(s["logp"]) for s in sliced]
    loss_max = max((abs(l) for l in losses), default=0.0)

    def cell(chip: str) -> str:
        return f"<td style='padding:0; text-align:center;'>{chip}</td>"

    def label(text: str) -> str:
        return ("<td style='padding:0 10px 0 0; font-size:12px; color:#475569;"
                f" text-align:right; white-space:nowrap;'>{text}</td>")

    rows = {"token": [], "nats": [], "adv": [], "loss": []}
    for s, loss in zip(sliced, losses):
        nats = -float(s["logp"])
        if nats == 0.0:      # avoid "-0.00" when log p is exactly zero
            nats = 0.0
        rows["token"].append(cell(render_token_chip(
            s["chosen_str"].replace("\n", _NL), float(s["prob"]),
            int(s["nuc_size"]))))
        rows["nats"].append(cell(render_value_chip(
            f"{nats:.2f}", _NATS_STYLE, f"-log p = {nats:.6f}")))
        rows["adv"].append(cell(render_value_chip(
            f"{advantage:+.2f}", _advantage_style(advantage),
            f"advantage A = {advantage:+.4f} (same on every token)")))
        rows["loss"].append(cell(render_value_chip(
            _fmt_loss(loss), loss_to_style(loss, loss_max),
            f"loss = A * -log p = {loss:.6f}")))

    table = (
        "<table style='border-collapse:separate; border-spacing:4px 3px;'>"
        f"<tr>{label('token')}{''.join(rows['token'])}</tr>"
        f"<tr>{label('&minus;log p &nbsp;(nats)')}{''.join(rows['nats'])}</tr>"
        f"<tr>{label('&times; A &nbsp;(advantage)')}{''.join(rows['adv'])}</tr>"
        f"<tr>{label('= loss')}{''.join(rows['loss'])}</tr>"
        "</table>"
    )
    caption = (
        "<div style='margin:0 0 8px 0; font-size:12px; color:#475569;'>"
        f"first {len(sliced)} of {len(trace)} tokens</div>"
    )
    _write_html_file(output_path, caption + table, title=title)
    print(f"Wrote HTML: {output_path.resolve()}")
    return output_path


# --------------------------------------------------------------------------- #
# Rollout token ids: the correct example comes straight from post 1's cached
# trace (its chosen_id sequence IS the completion); the incorrect one is fetched
# from the pool once and then lives in this script's own loss-trace cache.
# --------------------------------------------------------------------------- #
def rollout_from_post1_cache(trace_cache: Path) -> dict:
    import pandas as pd

    df = pd.read_parquet(trace_cache, columns=["chosen_id"])
    meta = json.loads(trace_cache.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("sample_idx") != CORRECT_SAMPLE_IDX:
        raise SystemExit(f"{trace_cache} holds sample {meta.get('sample_idx')}, "
                         f"expected {CORRECT_SAMPLE_IDX}")
    return {
        "completion_token_ids": [int(t) for t in df["chosen_id"]],
        "answer": str(meta["answer"]),
        "sample_idx": CORRECT_SAMPLE_IDX,
        "is_correct": True,
        "completion_text": meta["completion_text"],
    }


def rollout_from_solution() -> dict:
    """The dataset's reference solution, teacher-forced as the model's response.
    Out-of-distribution by construction: this text never came from the model's
    sampling tree. Token ids are produced in compute_loss_trace (needs the
    tokenizer); EOS is appended there as the SFT target's end-of-sequence."""
    from math_rollouts.data.problems import load_problems_by_ids

    problems = load_problems_by_ids([UNIQUE_ID])
    if not problems:
        raise SystemExit(f"{UNIQUE_ID} not found in math_problems")
    p = problems[0]
    text = str(p["solution"])
    if SOLUTION_STRIP_ASY:
        text = re.sub(r"\s*\[asy\].*?\[/asy\]", "", text, flags=re.DOTALL).strip()
    return {
        "kind": "solution",
        "solution_text": text,
        "answer": str(p["answer"]),
        "sample_idx": None,
        "is_correct": None,
        "completion_text": text,
    }


def rollout_from_pool(sample_idx: int) -> dict:
    from math_rollouts.data.hf import load_generation_parquet

    df = load_generation_parquet(MODEL_ID, PASSK_NAME)
    g = df[(df.unique_id == UNIQUE_ID) & (df.sample_idx == sample_idx)]
    if len(g) != 1:
        raise SystemExit(f"expected exactly 1 rollout for {UNIQUE_ID} "
                         f"sample_idx={sample_idx}, got {len(g)}")
    r = g.iloc[0]
    return {
        "completion_token_ids": [int(t) for t in r.completion_token_ids],
        "answer": str(r.answer),
        "sample_idx": int(r.sample_idx),
        "is_correct": bool(r.answer_matches),
        "completion_text": str(r.completion_text),
    }


def problem_text() -> str:
    from math_rollouts.data.problems import load_problems_by_ids

    problems = load_problems_by_ids([UNIQUE_ID])
    if not problems:
        raise SystemExit(f"{UNIQUE_ID} not found in math_problems")
    return str(problems[0]["problem"])


def nucleus_sizes(sample_idx: int, n_tokens: int) -> list[int]:
    """Per-position sampling-nucleus sizes (post 1's T=0.6 / top_p=0.95 sense),
    from the math-rollouts per-token nucleus store. Singleton = size 1."""
    from math_rollouts.data.hf import load_token_nuclei

    df = load_token_nuclei(MODEL_ID, PASSK_NAME, UNIQUE_ID)
    g = df[df.sample_idx == sample_idx]
    if len(g) != 1:
        raise SystemExit(f"no stored nuclei row for {UNIQUE_ID} "
                         f"sample_idx={sample_idx}")
    sizes = [int(x) for x in g.iloc[0]["nuc_sizes"]]
    if len(sizes) != n_tokens:
        raise SystemExit(f"nucleus store ({len(sizes)}) / rollout ({n_tokens}) "
                         "length mismatch")
    return sizes


# --------------------------------------------------------------------------- #
# Loss trace: teacher-forced forward pass, full-vocab log-softmax at T=1.0.
# --------------------------------------------------------------------------- #
def compute_loss_trace(rollout: dict, *, model_bundle: tuple) -> tuple[list[dict], dict]:
    import torch

    from math_rollouts.config import GenConfig
    from math_rollouts.nucleus import compute_nucleus

    tok, model, prompt_ids, device = model_bundle
    is_sft = rollout.get("kind") == "solution"
    if is_sft:
        completion_ids = tok.encode(rollout["solution_text"],
                                    add_special_tokens=False)
        completion_ids.append(int(tok.eos_token_id))
    else:
        completion_ids = rollout["completion_token_ids"]
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long,
                             device=device)
    print(f"teacher-forcing {'dataset solution' if is_sft else 'sample ' + str(rollout['sample_idx'])}: "
          f"{len(prompt_ids)} prompt + {len(completion_ids)} completion tokens")
    with torch.no_grad():
        logits = model(input_ids).logits[0]

    p = len(prompt_ids)
    cfg = GenConfig()
    # Sampled rollouts are in-nucleus by construction; their sizes come from the
    # math-rollouts store. The forced solution isn't in the store (it was never
    # sampled), so its nucleus is recomputed per position from these same logits.
    nuc_sizes = None if is_sft else nucleus_sizes(rollout["sample_idx"],
                                                  len(completion_ids))
    trace = []
    for t, chosen in enumerate(completion_ids):
        # logits at position i predict token i+1, so completion token t (input
        # index p+t) is predicted by row p+t-1. float32 log-softmax row by row
        # keeps the peak memory at one vocab row instead of the whole sequence.
        row = logits[p + t - 1].float()
        logp = torch.log_softmax(row, dim=-1)[int(chosen)].item()
        if is_sft:
            nuc_ids, _ = compute_nucleus(row, temperature=cfg.temperature,
                                         top_p=cfg.top_p, top_k=cfg.top_k)
            nuc_size = len(nuc_ids)
            in_nucleus = int(chosen) in nuc_ids
        else:
            nuc_size = nuc_sizes[t]
            in_nucleus = True
        trace.append({
            "step": t,
            "chosen_id": int(chosen),
            "chosen_str": tok.decode([chosen]),
            "prob": float(math.exp(logp)),
            "logp": float(logp),
            "nuc_size": nuc_size,
            "in_nucleus": in_nucleus,
        })

    meta = {
        "model_id": MODEL_ID, "problem_id": PROBLEM_ID,
        "problem_text": problem_text(), "answer": rollout["answer"],
        "sample_idx": rollout["sample_idx"], "is_correct": rollout["is_correct"],
        "num_tokens": len(completion_ids), "n_prompt_tokens": p,
        "loss_temperature": 1.0,
        "sampling_config": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "eos_id": int(tok.eos_token_id),
        "completion_text": rollout["completion_text"],
    }
    if is_sft:
        meta["loss_kind"] = "sft"
    else:
        reward = 1.0 if rollout["is_correct"] else 0.0
        meta.update({
            "loss_kind": "grpo",
            "group_size": GROUP_SIZE, "num_correct": NUM_CORRECT,
            "reward": reward, "advantage": group_advantage(reward),
        })
    return trace, meta


def load_model_bundle(device: str | None) -> tuple:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from math_rollouts.adapters.qwen_math import QwenMathAdapter
    from math_rollouts.data.problems import load_problems_by_ids

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    problems = load_problems_by_ids([UNIQUE_ID])
    if not problems:
        raise SystemExit(f"{UNIQUE_ID} not found in math_problems")
    problem = problems[0]

    # bfloat16 ON BOTH cpu/cuda — deliberately, not fp32. The rollouts were
    # sampled by vLLM in bf16; match the engine's precision so the probabilities
    # are the ones the sampler actually saw. (See html_tokens.py / trace.py.)
    dtype = torch.bfloat16
    print(f"Loading {MODEL_ID} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()

    adapter = QwenMathAdapter(MODEL_ID)
    prompt_ids = adapter.prompt_ids(problem, tok)
    return tok, model, prompt_ids, device


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
    has_in_nuc = "in_nucleus" in df.columns      # older RL caches predate it
    trace = [
        {
            "step": int(r["step"]),
            "chosen_id": int(r["chosen_id"]),
            "chosen_str": str(r["chosen_str"]),
            "prob": float(r["prob"]),
            "logp": float(r["logp"]),
            "nuc_size": int(r["nuc_size"]),
            "in_nucleus": bool(r["in_nucleus"]) if has_in_nuc else True,
        }
        for _, r in df.iterrows()
    ]
    meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    return trace, meta


# --------------------------------------------------------------------------- #
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example",
                    choices=["correct", "incorrect", "solution", "all"],
                    default="all")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="render only the first N steps (default: the whole rollout)")
    ap.add_argument("--opening", type=int, default=10,
                    help="tokens in the zoomed-in opening figure (default: 10)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--regen", action="store_true",
                    help="rebuild the loss traces even if the caches exist")
    args = ap.parse_args()

    examples = (["correct", "incorrect", "solution"] if args.example == "all"
                else [args.example])
    model_bundle = None
    for name in examples:
        cache = here / "data" / f"geometry_627_loss_trace_{name}.parquet"
        out = here / "html" / f"math500_geometry_627_losses_{name}.html"

        if cache.exists() and not args.regen:
            print(f"loading cached loss trace from {cache}")
            trace, meta = load_trace(cache)
        else:
            if name == "correct":
                rollout = rollout_from_post1_cache(
                    here / "data" / "geometry_627_to_solve_trace.parquet")
            elif name == "incorrect":
                rollout = rollout_from_pool(INCORRECT_SAMPLE_IDX)
            else:
                rollout = rollout_from_solution()
            if model_bundle is None:
                model_bundle = load_model_bundle(args.device)
            trace, meta = compute_loss_trace(rollout, model_bundle=model_bundle)
            save_trace(cache, trace, meta)

        # The trailing EOS step (rollouts that finished on "stop") is kept:
        # confirming the EOS is itself a singleton is part of the story.
        pid = meta["problem_id"].replace("test/", "").replace(".json", "")
        if meta.get("loss_kind") == "sft":
            title = (
                f"MATH-500 {pid} — per-token SFT loss — dataset reference solution "
                f"(teacher-forced, answer {meta['answer']}) — "
                f"{MODEL_ID.split('/')[-1]} (loss at T=1.0)"
            )
            ood = sum(1 for s in trace if not s.get("in_nucleus", True))
            print(f"[{name}] SFT cross-entropy (no advantage); "
                  f"{ood}/{len(trace)} tokens outside the sampling nucleus")
        else:
            verdict = "correct" if meta["is_correct"] else "incorrect"
            title = (
                f"MATH-500 {pid} — per-token GRPO loss — naturally-sampled rollout "
                f"(sample {meta['sample_idx']}, {verdict}, answer {meta['answer']}) — "
                f"{MODEL_ID.split('/')[-1]} (loss at T=1.0; sampled at T=0.6, top_p=0.95)"
            )
            print(f"[{name}] advantage = {meta['advantage']:+.4f} "
                  f"(reward {meta['reward']:g}, group {meta['num_correct']}/"
                  f"{meta['group_size']} correct)")
        visualize_losses(trace, meta, count=args.max_tokens, output_path=out,
                         title=title)
        if meta.get("loss_kind") == "sft":
            continue                     # no x A row to illustrate for plain CE
        opening_out = here / "html" / f"math500_geometry_627_losses_opening_{name}.html"
        visualize_opening(trace, meta, n=args.opening, output_path=opening_out,
                          title=title.replace("per-token GRPO loss",
                                              f"per-token GRPO loss, first "
                                              f"{args.opening} tokens"))


if __name__ == "__main__":
    main()
