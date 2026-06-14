"""Render a short "preview" crop of the wrapping nucleus viz for the blog.

The full ``*_to_solve`` visualization is many screens tall — great as an
interactive HTML file, but unwieldy as an inline blog image. This renders the
page at the blog width, then crops it down to the header plus the first few
rows of token chips, cutting cleanly inside one of the 16px row-gaps so no chip
is sliced in half. The same soft drop shadow as ``make_screenshots.py`` is
applied so the preview matches the other figure cards.

Link the full HTML alongside this PNG so readers can scroll the whole response.

Usage:
    python make_preview.py                  # default crop (~4 rows)
    python make_preview.py --rows-height 520 --out screenshots/preview.png
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from gdrive_tools import DriveImage
from PIL import Image

from make_screenshots import find_chrome, trim

HTML_DIR = Path(__file__).parent / "html"
DEFAULT_HTML = HTML_DIR / "math500_geometry_627_to_solve.html"
DEFAULT_OUT = Path(__file__).parent / "screenshots" / "math500_geometry_627_to_solve_preview.png"
DEFAULT_WIDTH = 800       # CSS px; matches the published full *_to_solve screenshot
RENDER_HEIGHT = 5000      # CSS px; tall enough that content never scrolls (no scrollbar)
TARGET_HEIGHT = 664       # CSS px; ~where to cut — snapped to the nearest row-gap
MIN_GAP_CSS = 8           # CSS px; ignore the tiny within-row gaps when finding cut points


def render_full(html: Path, out: Path, *, width: int, scale: int) -> None:
    """Render the whole page (no scrollbar) via headless Chrome."""
    chrome = find_chrome()
    prof = tempfile.mkdtemp(prefix="cr_")  # own profile dir: works while Chrome is open
    subprocess.run(
        [str(chrome), "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--user-data-dir={prof}", "--hide-scrollbars",
         f"--force-device-scale-factor={scale}",
         f"--window-size={width},{RENDER_HEIGHT}",
         f"--screenshot={out}", html.resolve().as_uri()],
        check=True, capture_output=True)
    if not out.exists():
        raise SystemExit(f"Chrome produced no screenshot for {html.name}")


def cut_row(img: Image.Image, *, target_px: int, min_gap_px: int) -> int:
    """Return a y (px) inside the row-gap nearest ``target_px`` to crop at."""
    a = np.asarray(img.convert("RGB"))
    bg = a[0, 0]
    ink = np.abs(a.astype(int) - bg.astype(int)).sum(axis=2) > 12
    blank = ~ink.any(axis=1)  # rows with no content at all = gap bands

    bands, y, H = [], 0, a.shape[0]
    while y < H:
        if blank[y]:
            y0 = y
            while y < H and blank[y]:
                y += 1
            if y - y0 >= min_gap_px:
                bands.append((y0 + y) // 2)  # band center
        else:
            y += 1
    # skip the top margin band (center near 0); pick the gap closest to target
    candidates = [c for c in bands if c > min_gap_px]
    if not candidates:
        return H
    return min(candidates, key=lambda c: abs(c - target_px))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--rows-height", type=int, default=TARGET_HEIGHT,
                    help="approx CSS-px height of the chip rows to keep "
                         "(snapped to the nearest row-gap; default %(default)s)")
    ap.add_argument("--no-shadow", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_full(args.html, args.out, width=args.width, scale=args.scale)
    trim(args.out, scale=args.scale)  # crop the uniform page margin to content + pad

    img = Image.open(args.out).convert("RGB")
    cut = cut_row(img, target_px=args.rows_height * args.scale,
                  min_gap_px=MIN_GAP_CSS * args.scale)
    img.crop((0, 0, img.width, cut)).save(args.out)

    w, h = Image.open(args.out).size
    if not args.no_shadow:
        # offset/blur are device px; scale them to keep CSS-pixel softness.
        di = DriveImage.open(args.out).shadow(offset=10 * args.scale,
                                              blur=8 * args.scale)
        di.save(args.out)
        w, h = di.size
    print(f"{args.out.name}: {w} x {h} (cut at y={cut})")


if __name__ == "__main__":
    main()
