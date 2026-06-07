#!/usr/bin/env python3
"""Per-opener bar plot for a single MATH-500 problem (default: geometry/627).

Mirrors the unguided arm of the openers experiment
(guided-rollouts/math-random/openers/scripts/openings_k16.py) but for ONE
problem, and plots, side by side for each first-token opener (branch):

  * the probability the model assigns to that branch (first-token nucleus prob,
    renormalized within the nucleus), and
  * the observed accuracy along that branch (fraction of K forced rollouts that
    reach the correct \\boxed{} answer).

Both quantities live on [0, 1], so the y-axis is fixed to 0-1. No guided
rollouts — base model only.

Self-contained: it generates the rollouts itself (HF forward pass for the
nucleus, vLLM for the K forced rollouts), caches them to a parquet, and reads
the cache on subsequent runs. Re-generate with --regen.

Run:
  source ~/env.sh && \
  micromamba run -n guided-rollouts python branch_accuracy_vs_prob.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# FlashInfer's top-p/top-k sampler JIT-compiles and needs nvcc, which isn't on
# this box; the project disables it the same way. Must be set before vLLM import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import matplotlib

matplotlib.use("Agg")
# opener token strings / answers contain '$' — keep them literal, not mathtext
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

MODEL = "Qwen/Qwen2.5-Math-1.5B"
PROBLEM_ID = "test/geometry/627.json"
QWEN_MATH_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# Generation config — identical to the canonical unguided_k16 opener run.
T, TOP_P = 0.6, 0.95
K = 64            # forced rollouts per opener
TOPK = 20         # cap on nucleus size
MAX_TOKENS = 3000

# Two-color scheme, deliberately not the grey/orange base-vs-guided pair.
C_PROB = "#4C72B0"   # model branch probability (indigo-blue)
C_ACC = "#55A868"    # observed branch accuracy (green)


def apply_qwen_math_template(user_body: str) -> str:
    return (
        "<|im_start|>system\n" + QWEN_MATH_SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n" + user_body + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def check_correct(completion: str, answer: str) -> bool:
    """math_verify check on the full completion (matches the project's scorer)."""
    from math_verify import parse, verify

    gold = parse(f"\\boxed{{{answer}}}")
    try:
        return bool(verify(gold, parse(completion)))
    except Exception:
        return False


def load_problem(problem_id: str) -> dict:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for row in ds:
        if row["unique_id"] == problem_id:
            return dict(row)
    raise RuntimeError(f"{problem_id} not found in MATH-500")


def first_token_nucleus(prompt_ids: list[int]) -> tuple[list[int], list[str], list[float]]:
    """First-token nucleus (top-p on temperature-scaled probs, capped at TOPK),
    with probabilities renormalized within the nucleus. Same recipe as
    openings_k16.py phase 1."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    with torch.no_grad():
        logits = model(torch.tensor([prompt_ids], device="cuda")).logits[0, -1]
    probs = torch.softmax(logits.float() / T, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    keep = (torch.cumsum(sp, 0) - sp) < TOP_P
    keep[0] = True
    nuc_ids = si[keep][:TOPK]
    nuc_p = probs[nuc_ids]
    nuc_p = (nuc_p / nuc_p.sum()).tolist()
    nuc_ids = [int(t) for t in nuc_ids.tolist()]
    nuc_strs = [tok.decode([t]) for t in nuc_ids]
    del model
    torch.cuda.empty_cache()
    return nuc_ids, nuc_strs, [round(float(x), 6) for x in nuc_p]


def generate_rollouts(cache_path: Path) -> pd.DataFrame:
    """Generate the nucleus + K forced rollouts per opener, cache, and return
    a per-rollout DataFrame."""
    from transformers import AutoTokenizer

    problem = load_problem(PROBLEM_ID)
    answer = problem["answer"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(
        apply_qwen_math_template(problem["problem"]), add_special_tokens=False
    ).input_ids

    nuc_ids, nuc_strs, nuc_probs = first_token_nucleus(prompt_ids)
    print(f"nucleus size {len(nuc_ids)}: " + ", ".join(
        f"{repr(s)}={p:.3f}" for s, p in zip(nuc_strs, nuc_probs)))

    # vLLM is imported after the HF model is freed so they don't fight for VRAM.
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096)
    sp = SamplingParams(n=K, temperature=T, top_p=TOP_P, max_tokens=MAX_TOKENS,
                        stop=["<|im_end|>"])
    prompts = [{"prompt_token_ids": prompt_ids + [tid]} for tid in nuc_ids]
    outs = llm.generate(prompts, sp)

    rows = []
    for tid, tstr, pr, o in zip(nuc_ids, nuc_strs, nuc_probs, outs):
        for j, c in enumerate(o.outputs):
            text = tstr + c.text                      # include the forced opener
            fin = c.finish_reason
            rows.append(dict(
                problem_id=PROBLEM_ID, answer=answer, token_id=tid, token_str=tstr,
                nuc_prob=pr, sample_idx=j, num_tokens=len([tid] + list(c.token_ids)),
                finish_reason=fin,
                is_correct=bool(fin == "stop" and check_correct(text, answer)),
            ))
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"wrote {cache_path}  ({len(df)} rollouts, "
          f"{int(df.is_correct.sum())} correct)")
    return df


def opener_label(s: str) -> str:
    """Readable x label: show repr for whitespace-only tokens, else stripped."""
    return repr(s)[1:-1][:10] if s.strip() == "" else s.strip()[:10]


def plot(df: pd.DataFrame, out_path: Path) -> None:
    # one row per opener, ordered by model-assigned branch probability (desc)
    g = (df.groupby(["token_id", "token_str", "nuc_prob"], as_index=False)
           .agg(correct=("is_correct", "sum"), n=("is_correct", "size"))
           .sort_values("nuc_prob", ascending=False)
           .reset_index(drop=True))
    n_per = int(g["n"].iloc[0])

    # Twin axes: blue (model probability) on the left, green (correct count) on the
    # right. Both span the full height (left 0-1, right 0-K), so the bars stay
    # visually comparable while each carries its own natural scale + label.
    x = np.arange(len(g))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(g) + 2), 5))
    ax2 = ax.twinx()
    ax2.patch.set_visible(False)   # let the left-axis (blue) bars show through

    b1 = ax.bar(x - w / 2, g["nuc_prob"], w, color=C_PROB, label="Model's prediction")
    b2 = ax2.bar(x + w / 2, g["correct"], w, color=C_ACC,
                 label=f"Correct rollouts at K={n_per}")
    ax.bar_label(b1, labels=[f"{p*100:.0f}%" for p in g["nuc_prob"]], fontsize=8, padding=2)
    ax2.bar_label(b2, labels=[f"{int(c)}" for c in g["correct"]], fontsize=8, padding=2)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 5))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("model probability", color=C_PROB)
    ax.tick_params(axis="y", colors=C_PROB)

    ax2.set_ylim(0, n_per)
    ax2.set_yticks(np.linspace(0, n_per, 5))
    ax2.set_ylabel(f"number correct (out of {n_per})", color=C_ACC)
    ax2.tick_params(axis="y", colors=C_ACC)

    ax.set_xlabel("first-token opener, ordered by model probability")
    ax.set_xticks(x)
    ax.set_xticklabels([opener_label(s) for s in g["token_str"]],
                       rotation=0, fontsize=9)
    pid = PROBLEM_ID.replace("test/", "").replace(".json", "")
    ax.set_title(f"First Token Probability vs. Accuracy\n"
                 f"{MODEL.split('/')[-1]} | Math500 - {pid}", fontsize=12)
    ax.legend([b1, b2], [t.get_label() for t in (b1, b2)],
              frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.resolve()}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path,
                    default=here / "data" / f"geometry_627_openers_k{K}.parquet")
    ap.add_argument("--out", type=Path,
                    default=here / "figures" / "geometry_627_branch_acc_vs_prob.png")
    ap.add_argument("--regen", action="store_true",
                    help="re-generate rollouts even if the cache exists")
    args = ap.parse_args()

    if args.cache.exists() and not args.regen:
        print(f"loading cached rollouts from {args.cache}")
        df = pd.read_parquet(args.cache)
    else:
        df = generate_rollouts(args.cache)
    plot(df, args.out)


if __name__ == "__main__":
    main()
