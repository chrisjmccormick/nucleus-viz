# nucleus-viz

Figures and visualizations for the **"Branch Tokens in Reasoning Traces"** blog
series — exploring how deterministic reasoning-model rollouts are (singletons
vs. branch tokens, nucleus size, per-token losses, etc.).

> **Note:** this README is a work in progress. It currently documents only the
> wrapping nucleus visualization — the rest of the repo (vocab heatmaps, gradient
> plots, loss tables) isn't covered yet.

## The wrapping nucleus visualization

`html/math500_geometry_627_to_solve.html` is a self-contained (inline-styled,
no external assets) view of a single rollout: Qwen2.5-Math-1.5B answering a
MATH-500 geometry problem at `T=0.6, top_p=0.95`. Each token position is a
chip showing the sampled token and its probability; **branch tokens** (positions
with more than one nucleus candidate) stack their alternatives underneath. A
header card shows the problem and summary stats (540 tokens, 86% singletons,
14% branches, correct). The chips wrap to fill the page width, so the full
response is many screens tall.

`html_tokens.py` regenerates it (from a cached trace, so no GPU is needed), and
has a few opt-in display modes — with no flags the output is unchanged:

- `--newline-breaks` — lay out **one row per source line** (break after a newline
  token and before a display-math `\[` opener) instead of wrapping. Each line is
  non-wrapping and the strip scrolls horizontally, which keeps a line intact and
  makes the **newline ↔ branch-token** relationship legible (branch tokens cluster
  at line-starts / sentence openers).
- `--show-prompt` — render the model's prompt template (system boxed-answer
  instruction + the problem + the assistant opener) as small grey reference chips
  after the legend.
- `--show-positions` — print the 0-based token position above each column.

## Blog hosting

For the published post, the figures are copied into the blog repo so they're
frozen alongside the post rather than hot-linked from GitHub raw:
`chrisjmccormick.github.io/assets/BranchTokens/` →
`http://www.mccormickml.com/assets/BranchTokens/` — a short cropped preview of
the visualization above, the full interactive HTML, and the `Vocab-*.png` figures.
