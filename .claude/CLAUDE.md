# nucleus-viz — notes for agents

Scratch scripts, run logs, and working notes for this repo live in the **private
`agent-ops` repo**, under `agent-ops/nucleus-viz/` — one timestamped folder per
session. (Kept out of this public repo to reduce clutter; promote a utility back
here only if it's broadly useful.)

The blog-publishing scripts were moved there as publishing refinements:
- `make_screenshots.py` — renders `html/*.html` → `screenshots/*.png` (headless
  Chrome, trim, drop shadow).
- `make_preview.py` — crops the `to_solve` viz to a short header+rows preview for
  inline blog use.

They read this repo's `html/` and write its `screenshots/` (override the repo
location with `NUCLEUS_VIZ_DIR`). Latest session:
`agent-ops/nucleus-viz/2026-06-14_1036am_branch-tokens-blog-figures/`.
