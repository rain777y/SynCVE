"""
Rasterize public/favicon.svg into favicon.ico, logo192.png, logo512.png.

Uses Playwright (Chromium) so we don't need cairosvg.
"""
import io
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image

PUB = Path(__file__).resolve().parents[1] / "src" / "frontend" / "public"
SVG = PUB / "favicon.svg"
svg_text = SVG.read_text(encoding="utf-8")

# Rendered HTML wrapper: a single SVG sized to the target
HTML = """<!doctype html><html><head><style>
html,body{{margin:0;padding:0;background:transparent}}
svg{{display:block;width:{n}px;height:{n}px}}
</style></head><body>{svg}</body></html>"""


def render_png(size: int) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": size, "height": size})
        page.set_content(HTML.format(n=size, svg=svg_text), wait_until="domcontentloaded")
        png = page.screenshot(omit_background=True, clip={"x": 0, "y": 0, "width": size, "height": size})
        browser.close()
    return png


def main():
    print(f"source SVG: {SVG}")

    # PNG icons used by manifest / Apple touch
    for size, name in [(192, "logo192.png"), (512, "logo512.png")]:
        png = render_png(size)
        out = PUB / name
        out.write_bytes(png)
        print(f"  · wrote {out.name}  ({size}x{size}, {len(png)} bytes)")

    # Multi-resolution ICO: 16, 32, 48
    images = []
    for size in (16, 32, 48):
        png = render_png(size)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        images.append(img)
        print(f"  · rendered {size}x{size}")

    ico_path = PUB / "favicon.ico"
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48)],
        append_images=images[1:],
    )
    print(f"  · wrote {ico_path.name}  ({ico_path.stat().st_size} bytes)")

    print("\ndone.")


if __name__ == "__main__":
    main()
