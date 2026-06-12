"""Render the html/ visualizations to crisp PNGs via headless Chrome.

Each page is rendered at ``--scale``x device pixels (default 2, retina-sharp),
with a deliberately generous capture height. A Pillow pass then trims the
uniform page background down to the content plus a small margin, so per-file
heights never need hand-tuning.

``--width`` is the layout width in CSS pixels. For the wrapping nucleus viz
(``*_to_solve``) it controls how tall vs. wide the chips reflow; for the loss
tables it just needs to exceed the table width — the script warns if content
got clipped at the right edge.

Each trimmed capture then gets a soft drop shadow (gdrive-tools' DriveImage),
so the pngs sit on the blog page as floating cards; skip with ``--no-shadow``.

Usage:
    python make_screenshots.py                       # every html/*.html
    python make_screenshots.py html/foo.html ...     # specific files
    python make_screenshots.py --width 900 --scale 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

from gdrive_tools import DriveImage
from PIL import Image, ImageChops

CHROME_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

HTML_DIR = Path(__file__).parent / "html"
DEFAULT_OUT_DIR = Path(__file__).parent / "screenshots"
DEFAULT_WIDTH = 760      # CSS px; matches the blog content column
CAPTURE_HEIGHT = 4000    # CSS px starting height; doubled until content fits
MAX_HEIGHT = 32000       # CSS px; give up past this
PAD_CSS = 12             # margin (CSS px) kept around the trimmed content


def find_chrome() -> Path:
    for p in CHROME_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("no Chrome/Edge executable found; edit CHROME_CANDIDATES")


def render(chrome: Path, html_path: Path, out_path: Path, *,
           width: int, height: int, scale: int) -> None:
    subprocess.run(
        [str(chrome), "--headless=new", "--disable-gpu",
         f"--force-device-scale-factor={scale}",
         f"--window-size={width},{height}",
         f"--screenshot={out_path}",
         html_path.resolve().as_uri()],
        check=True, capture_output=True)
    if not out_path.exists():
        raise SystemExit(f"Chrome produced no screenshot for {html_path.name}")


def trim(out_path: Path, *, scale: int) -> tuple[int, int, bool]:
    """Crop to content + margin. Returns (w, h, content_reaches_bottom)."""
    img = Image.open(out_path).convert("RGB")
    bg = Image.new("RGB", img.size, img.getpixel((0, 0)))
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox is None:  # blank page; leave as-is
        return img.width, img.height, False
    hit_bottom = bbox[3] >= img.height
    pad = PAD_CSS * scale
    cropped = img.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                        min(bbox[2] + pad, img.width),
                        min(bbox[3] + pad, img.height)))
    cropped.save(out_path)
    return cropped.width, cropped.height, hit_bottom


def capture(chrome: Path, html_path: Path, out_path: Path, *,
            width: int, scale: int) -> tuple[int, int]:
    """Render and trim, doubling the capture height until the content fits."""
    height = CAPTURE_HEIGHT
    while True:
        render(chrome, html_path, out_path, width=width, height=height,
               scale=scale)
        w, h, hit_bottom = trim(out_path, scale=scale)
        if not hit_bottom:
            return w, h
        if height >= MAX_HEIGHT:
            print(f"  WARNING: {html_path.name} still clipped at "
                  f"{MAX_HEIGHT} CSS px tall", file=sys.stderr)
            return w, h
        height = min(height * 2, MAX_HEIGHT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path,
                    help="html files to render (default: all of html/*.html)")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                    help="layout width in CSS px (default %(default)s)")
    ap.add_argument("--scale", type=int, default=2,
                    help="device scale factor (default %(default)s)")
    ap.add_argument("--no-shadow", action="store_true",
                    help="skip the drop shadow")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    files = args.files or sorted(HTML_DIR.glob("*.html"))
    if not files:
        raise SystemExit(f"no html files found in {HTML_DIR}")
    chrome = find_chrome()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for html_path in files:
        out_path = args.out_dir / (html_path.stem + ".png")
        w, h = capture(chrome, html_path, out_path, width=args.width,
                       scale=args.scale)
        if not args.no_shadow:
            # offset/blur are device pixels; scale them so the shadow has the
            # same CSS-pixel softness at any --scale.
            img = DriveImage.open(out_path).shadow(offset=10 * args.scale,
                                                   blur=8 * args.scale)
            img.save(out_path)
            w, h = img.size
        print(f"{out_path.name}: {w} x {h}")


if __name__ == "__main__":
    main()
