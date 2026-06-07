#!/usr/bin/env python3
"""First-response-token logit heatmaps for Qwen2.5-Math-1.5B.

Loads geometry/627 from MATH-500, applies the literal Qwen-Math chat template
used in guided-rollouts, runs one forward pass, and writes Chebyshev-ring
layouts (highest raw-logit rank at center, expanding outward):

  Vocab-01-Raw_Logits          raw logits (scale locked to post-T logit range)
  Vocab-02-Temp{T}             logits / T (same scale as raw)
  Vocab-03-Softmax             softmax(logits / T), 0–1, black floor
  Vocab-04-Softmax-Nucleus     zoomed rings 0..N, per-cell token labels
  Vocab-05-Top-p{p}-Nucleus    as above, but probabilities shown only for
                               the top-p nucleus ("-" elsewhere)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
GEOMETRY_627_ID = "test/geometry/627.json"
GEOMETRY_627_PROBLEM = (
    "The coordinates of a parallelogram are (5, 3), (6, 8), (7, 4) and $(x, y)$ "
    "and $x > 7$. What is the value of $x + y$?"
)

# Literal Qwen-Math template from guided-rollouts/math-random/run_random_nothink.py
QWEN_MATH_SYSTEM = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def apply_qwen_math_template(user_body: str) -> str:
    im_end = "\u003c|im_end|\u003e"
    return (
        "<|im_start|>system\n" + QWEN_MATH_SYSTEM + im_end + "\n"
        "<|im_start|>user\n" + user_body + im_end + "\n"
        "<|im_start|>assistant\n"
    )


def load_geometry_627() -> dict:
    try:
        from datasets import load_dataset
    except ImportError:
        return {
            "unique_id": GEOMETRY_627_ID,
            "problem": GEOMETRY_627_PROBLEM,
            "answer": "17",
            "subject": "Geometry",
            "level": 4,
        }

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for row in ds:
        if row["unique_id"] == GEOMETRY_627_ID:
            return dict(row)
    raise RuntimeError(f"{GEOMETRY_627_ID} not found in MATH-500")


def first_response_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits[0, -1]


def topp_nucleus_mask(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Bool mask: minimal top-p set on temperature-scaled logits (guided-rollouts)."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = cumulative - sorted_probs < top_p
    keep_sorted[0] = True
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[sorted_idx[keep_sorted]] = True
    return mask


def ring_side(vocab: int) -> int:
    side = int(math.ceil(math.sqrt(vocab)))
    if side % 2 == 0:
        side += 1
    return side


def chebyshev_positions(side: int) -> list[tuple[int, int]]:
    """Center-out Chebyshev rings; ring k has 1 cell (k=0) else 8k cells."""
    center = side // 2
    positions = [(center, center)]
    ring = 1
    while len(positions) < side * side:
        top = center - ring
        bottom = center + ring
        left = center - ring
        right = center + ring
        ring_cells: list[tuple[int, int]] = []
        for col in range(left, right + 1):
            ring_cells.append((top, col))
        for row in range(top + 1, bottom + 1):
            ring_cells.append((row, right))
        for col in range(right - 1, left - 1, -1):
            ring_cells.append((bottom, col))
        for row in range(bottom - 1, top, -1):
            ring_cells.append((row, left))
        positions.extend(ring_cells)
        ring += 1
    return positions


def place_values_on_rings(
    values: np.ndarray, rank_order: np.ndarray, side: int,
) -> np.ndarray:
    """Place values in Chebyshev rings; rank_order is raw-logit rank (best first)."""
    positions = chebyshev_positions(side)
    grid = np.full((side, side), np.nan, dtype=np.float32)
    vocab = values.shape[0]
    for rank, pos in enumerate(positions[:vocab]):
        token_id = int(rank_order[rank])
        grid[pos] = values[token_id]
    return grid


def ring_zoom_crop(
    values: np.ndarray,
    rank_order: np.ndarray,
    full_side: int,
    max_ring: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to Chebyshev rings 0..max_ring; return (value grid, token-id grid)."""
    center = full_side // 2
    crop_side = 2 * max_ring + 1
    grid = np.full((crop_side, crop_side), np.nan, dtype=np.float32)
    token_grid = np.full((crop_side, crop_side), -1, dtype=np.int64)
    positions = chebyshev_positions(full_side)
    vocab = values.shape[0]
    for rank, pos in enumerate(positions[:vocab]):
        ring = max(abs(pos[0] - center), abs(pos[1] - center))
        if ring > max_ring:
            continue
        token_id = int(rank_order[rank])
        crop_row = pos[0] - center + max_ring
        crop_col = pos[1] - center + max_ring
        grid[crop_row, crop_col] = values[token_id]
        token_grid[crop_row, crop_col] = token_id
    return grid, token_grid


def format_token_label(
    tokenizer, token_id: int, prob: float, max_chars: int = 14, *, show_prob: bool = True,
) -> str:
    text = tokenizer.decode([int(token_id)])
    visible = repr(text)[1:-1]
    if len(visible) > max_chars:
        visible = visible[: max_chars - 1] + "…"
    prob_str = f"{prob:.4f}" if show_prob else "-"
    return f"{visible}\n{prob_str}"


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 20) -> list[tuple[str, int, float]]:
    vals, ids = torch.topk(logits, k)
    out = []
    for tid, val in zip(ids.tolist(), vals.tolist()):
        text = tokenizer.decode([tid])
        out.append((repr(text), tid, val))
    return out


def find_viridis_trip_norm(rgb_threshold: float = 0.01) -> float:
    """Normalized viridis position where RGB first differs from t=0 by rgb_threshold."""
    viridis = plt.cm.viridis
    base = np.array(viridis(0.0)[:3])
    for i in range(1, 256):
        t = i / 255.0
        rgb = np.array(viridis(t)[:3])
        if float(np.linalg.norm(rgb - base)) >= rgb_threshold:
            return t
    return 1.0


def trip_value(trip_norm: float, vmin: float, vmax: float) -> float:
    return vmin + trip_norm * (vmax - vmin)


def black_floor_cmap(trip_norm: float, n: int = 256) -> ListedColormap:
    """viridis with [0, trip_norm] flattened to black; (trip_norm, 1] stretched to full viridis."""
    viridis = plt.cm.viridis
    colors = []
    span = 1.0 - trip_norm
    for i in range(n):
        t = i / (n - 1)
        if t <= trip_norm:
            colors.append((0.0, 0.0, 0.0, 1.0))
        else:
            remapped = (t - trip_norm) / span
            colors.append(viridis(remapped))
    return ListedColormap(colors, name="viridis_black_floor")


def save_ring_heatmap(
    grid: np.ndarray,
    title: str,
    color_label: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    *,
    black_floor: bool = False,
    trip_norm: float | None = None,
    rgb_threshold: float = 0.01,
) -> float | None:
    """Save a ring heatmap. Returns trip value in data units if black_floor is enabled."""
    cmap = "viridis"
    trip_data: float | None = None
    if black_floor:
        trip_norm = trip_norm if trip_norm is not None else find_viridis_trip_norm(rgb_threshold)
        trip_data = trip_value(trip_norm, vmin, vmax)
        cmap = black_floor_cmap(trip_norm)
        title += f"\nBlack if $\\leq$ {trip_data:.3f}"

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=color_label)
    if black_floor and trip_norm is not None:
        cbar.ax.axhline(trip_norm, color="white", linestyle="--", linewidth=1.0)
        cbar.ax.text(
            1.15, trip_norm, f"≤{trip_data:.6g}\n→ black",
            transform=cbar.ax.transAxes, va="center", ha="left",
            fontsize=8, color="0.85",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return trip_data


def save_ring_zoom_annotated(
    grid: np.ndarray,
    token_grid: np.ndarray,
    tokenizer,
    title: str,
    out_path: Path,
    *,
    trip_norm: float,
    nucleus_ids: set[int] | None = None,
) -> None:
    vmin, vmax = 0.0, 1.0
    trip_data = trip_value(trip_norm, vmin, vmax)
    cmap = black_floor_cmap(trip_norm)
    title += f"\nBlack if $\\leq$ {trip_data:.3f}"

    crop_side = grid.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, crop_side * 0.9), max(10, crop_side * 0.9)))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(crop_side):
        for col in range(crop_side):
            token_id = int(token_grid[row, col])
            if token_id < 0:
                continue
            prob = float(grid[row, col])
            show_prob = nucleus_ids is None or token_id in nucleus_ids
            label = format_token_label(tokenizer, token_id, prob, show_prob=show_prob)
            rgba = cmap((prob - vmin) / (vmax - vmin))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.45 else "black"
            ax.text(
                col, row, label, ha="center", va="center",
                fontsize=7, color=text_color, clip_on=True,
                parse_math=False,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probability")
    cbar.ax.axhline(trip_norm, color="white", linestyle="--", linewidth=1.0)
    cbar.ax.text(
        1.15, trip_norm, f"≤{trip_data:.6g}\n→ black",
        transform=cbar.ax.transAxes, va="center", ha="left",
        fontsize=8, color="0.85",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="directory for output PNGs",
    )
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--trip-rgb-threshold",
        type=float,
        default=0.005,
        help="RGB L2 distance on viridis that defines the black-floor trip point",
    )
    ap.add_argument(
        "--zoom-rings",
        type=int,
        default=3,
        help="max Chebyshev ring index for the annotated softmax zoom (0..N => N+1 rings)",
    )
    ap.add_argument("--show", action="store_true", help="open interactive window")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    problem = load_geometry_627()
    prompt = apply_qwen_math_template(problem["problem"])

    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(device)
    print(f"Prompt tokens: {input_ids.shape[-1]}")

    logits = first_response_logits(model, input_ids)
    vocab = logits.shape[0]
    rank_order = torch.argsort(logits, descending=True).cpu().numpy()

    raw = logits.detach().float().cpu().numpy()
    scaled = (logits.float() / args.temperature).detach().cpu().numpy()
    full_softmax = torch.softmax(logits.float() / args.temperature, dim=-1).cpu().numpy()
    nucleus_mask = topp_nucleus_mask(logits, args.temperature, args.top_p)
    nucleus_size = int(nucleus_mask.sum().item())

    scaled_vmin, scaled_vmax = float(scaled.min()), float(scaled.max())
    side = ring_side(vocab)
    zoom_cells = 1 + 8 * sum(range(1, args.zoom_rings + 1)) if args.zoom_rings > 0 else 1

    top = top_tokens(tokenizer, logits, k=args.top_k)
    print(f"Vocab size: {vocab:,}")
    print(f"Top-p nucleus size (T={args.temperature}, p={args.top_p}): {nucleus_size}")
    print("Top logits:")
    for rank, (text, tid, val) in enumerate(top[:10], start=1):
        print(f"  {rank:2d}. id={tid:6d}  logit={val:8.3f}  {text}")

    out_dir = args.out_dir
    stem = "geometry_627"
    fig_dir = out_dir / stem

    model_name = args.model.split("/")[-1]
    pid = problem["unique_id"].replace("test/", "").replace(".json", "")
    subtitle = f"{model_name} | MATH-500 {pid}"

    trip_norm = find_viridis_trip_norm(args.trip_rgb_threshold)
    trip_prob = trip_value(trip_norm, 0.0, 1.0)
    print(
        f"Viridis black-floor trip (RGB Δ={args.trip_rgb_threshold}): "
        f"norm={trip_norm:.6f} → {trip_prob:.6g} on 0–1 scale"
    )

    raw_path = fig_dir / "Vocab-01-Raw_Logits.png"
    save_ring_heatmap(
        place_values_on_rings(raw, rank_order, side),
        f"Raw First-Token Logits Across the Full Vocabulary\n"
        f"{subtitle}",
        "raw logit", raw_path, scaled_vmin, scaled_vmax,
    )
    print(f"Wrote {raw_path.resolve()}")

    temp_path = fig_dir / f"Vocab-02-Temp{args.temperature}.png"
    save_ring_heatmap(
        place_values_on_rings(scaled, rank_order, side),
        f"First-Token Logits After Temperature Scaling (T={args.temperature})\n"
        f"{subtitle}",
        f"logit / T  (T={args.temperature})", temp_path, scaled_vmin, scaled_vmax,
    )
    print(f"Wrote {temp_path.resolve()}")

    softmax_path = fig_dir / "Vocab-03-Softmax.png"
    save_ring_heatmap(
        place_values_on_rings(full_softmax, rank_order, side),
        f"First-Token Probability Collapses Onto a Few Tokens\n"
        f"{subtitle} | softmax over {vocab:,} tokens, highest at center",
        "probability", softmax_path, 0.0, 1.0,
        black_floor=True,
        trip_norm=trip_norm,
        rgb_threshold=args.trip_rgb_threshold,
    )
    print(f"Wrote {softmax_path.resolve()}")

    zoom_grid, zoom_tokens = ring_zoom_crop(full_softmax, rank_order, side, args.zoom_rings)
    nucleus_ids = set(torch.nonzero(nucleus_mask).flatten().tolist())

    nucleus_renorm = np.where(nucleus_mask.cpu().numpy(), full_softmax, 0.0)
    nucleus_renorm = (nucleus_renorm / nucleus_renorm.sum()).astype(np.float32)
    nuc_zoom_grid, nuc_zoom_tokens = ring_zoom_crop(
        nucleus_renorm, rank_order, side, args.zoom_rings
    )

    nucleus_zoom_path = fig_dir / "Vocab-04-Softmax-Nucleus.png"
    save_ring_zoom_annotated(
        zoom_grid,
        zoom_tokens,
        tokenizer,
        f"The Few Tokens Carrying Real First-Token Probability\n"
        f"{subtitle} | top {zoom_cells} of {vocab:,} tokens, highest at center",
        nucleus_zoom_path,
        trip_norm=trip_norm,
    )
    print(f"Wrote {nucleus_zoom_path.resolve()}")

    topp_zoom_path = fig_dir / f"Vocab-05-Top-p{args.top_p}-Nucleus.png"
    save_ring_zoom_annotated(
        nuc_zoom_grid,
        nuc_zoom_tokens,
        tokenizer,
        f"The Top-p={args.top_p} Nucleus — the Only Tokens Sampling Can Pick\n"
        f"{subtitle}",
        topp_zoom_path,
        trip_norm=trip_norm,
        nucleus_ids=nucleus_ids,
    )
    print(f"Wrote {topp_zoom_path.resolve()}")

    if args.show:
        img = plt.imread(softmax_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
