# nucleus-viz

Figures and visualizations for the **"Branch Tokens in Reasoning Traces"** blog
series — exploring how deterministic reasoning-model rollouts are (singletons
vs. branch tokens, nucleus size, per-token losses, etc.).

> **Note:** this README is a work in progress. It currently documents only the
> wrapping nucleus visualization and its screenshot / preview pipeline — the
> rest of the repo (vocab heatmaps, gradient plots, loss tables) isn't covered
> yet.

## The wrapping nucleus visualization

`html/math500_geometry_627_to_solve.html` is a self-contained (inline-styled,
no external assets) view of a single rollout: Qwen2.5-Math-1.5B answering a
MATH-500 geometry problem at `T=0.6, top_p=0.95`. Each token position is a
chip showing the sampled token and its probability; **branch tokens** (positions
with more than one nucleus candidate) stack their alternatives underneath. A
header card shows the problem and summary stats (540 tokens, 86% singletons,
14% branches, correct). The chips wrap to fill the page width, so the full
response is many screens tall.

## Rendering PNGs — `make_screenshots.py`

Renders `html/*.html` to crisp PNGs in `screenshots/` via headless Chrome at
`--scale 2`, trims the uniform page background to the content (a Pillow pass),
and adds a soft drop shadow (gdrive-tools' `DriveImage`) so each sits on the
blog as a floating card.

```bash
python make_screenshots.py                         # all html/*.html
python make_screenshots.py html/foo.html --width 800
```

`--width` is the layout width in CSS px. The published full `*_to_solve` image
(`screenshots/math500_geometry_627_to_solve.png`, 1624×8600) was rendered at
**`--width 800`** — wide enough to keep the problem statement on one line and
avoid clipping the first row of chips. (The script's current default of 760 is
narrower and reflows differently.)

## Blog preview crop — `make_preview.py`

The full `*_to_solve` PNG is too tall to drop inline in a post, so this script
renders the page and crops it to the header plus the first few rows of chips,
cutting cleanly inside one of the 16px row-gaps (so no chip is sliced) and
re-applying the same drop shadow. Host the full HTML alongside it so readers
can scroll the whole response.

```bash
# defaults reproduce the crop used in the post: header + ~5 rows, 800px wide
python make_preview.py
# -> screenshots/math500_geometry_627_to_solve_preview.png  (1624 x 1362)
```

The defaults (`--width 800 --rows-height 664`) match the published full image.
`--rows-height` is the approximate CSS-px height of chip rows to keep; the cut
is snapped to the nearest row-gap. Use the **same `--width`** as the full image
so the layout (and the cut row) match.

**Gotcha:** both scripts launch headless Chrome. If your interactive Chrome is
already running, a plain `--headless` invocation fails with
`ERR_FILE_NOT_FOUND` (locked profile). `make_preview.py` works around this by
pointing Chrome at a fresh temp `--user-data-dir`; `make_screenshots.py` does
not, so close Chrome (or add the flag) if it errors.

## Blog hosting

For the published post, these are copied into the blog repo so they're frozen
alongside the post rather than hot-linked from GitHub:
`chrisjmccormick.github.io/assets/BranchTokens/` →
`http://www.mccormickml.com/assets/BranchTokens/` (the cropped preview PNG, the
full interactive HTML, and the `Vocab-*.png` figures).

## For my agents — related ops folders

Scratch scripts, run logs, and working notes for this repo's figures live in the
private `experiment-ops` repo (kept out of this public repo to reduce clutter).

In experiment-ops, see `2026-06-14_1036am_branch-tokens-blog-figures/`.
