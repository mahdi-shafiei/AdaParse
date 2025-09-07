#!/usr/bin/env python3
# Extract specific pages from PDFs and save as single-page PDFs under ./data
# Optionally watermark each page, then merge all outputs into ./data/test_tmp.pdf

import argparse
from pathlib import Path
import pymupdf  # PyMuPDF

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = Path('/lus/flare/projects/FoundEpidem/siebenschuh/adaparse_data/input/small-pdf-dataset')
tasks = [
    (BASE / "1_of_20.pdf",  1),
    (BASE / "2_of_20.pdf", 23),
    (BASE / "3_of_20.pdf", 9),
    (BASE / "4_of_20.pdf", 8),
    (BASE / "5_of_20.pdf", 1),
    (BASE / "6_of_20.pdf", 8),
    (BASE / "7_of_20.pdf", 5),
    (BASE / "8_of_20.pdf", 3),
    (BASE / "9_of_20.pdf", 8),
    (BASE / "10_of_20.pdf", 11),
    (BASE / "11_of_20.pdf", 3),
    (BASE / "12_of_20.pdf", 4),
    (BASE / "13_of_20.pdf", 11),
    (BASE / "14_of_20.pdf", 14),
    (BASE / "15_of_20.pdf", 7),
    (BASE / "16_of_20.pdf", 12),
    (BASE / "17_of_20.pdf", 9),
    (BASE / "18_of_20.pdf", 11),
    (BASE / "19_of_20.pdf", 3),
]

def _add_big_mark(page: "pymupdf.Page", text: str) -> None:
    """
    Draw a very large, semi-transparent grey label across the page.
    Uses TextWriter so we can set fill_opacity. Rotates text for watermark vibe.
    """
    rect = page.rect
    # Pick a big font size relative to page size
    fontsize = max(rect.width, rect.height) * 0.12  # ~12% of longer side
    fontsize = max(24, int(fontsize))

    # Center horizontally using text length
    try:
        text_width = pymupdf.get_text_length(text, fontname="helv", fontsize=fontsize)
    except Exception:
        # Fallback if font metrics unavailable
        text_width = rect.width * 0.8

    x = (rect.width - text_width) / 2.0
    y = rect.height / 2.0

    tw = pymupdf.TextWriter(rect, color=(0.5, 0.5, 0.5))  # grey
    tw.fill_opacity = 0.18  # semi-transparent
    # Rotate ~30 degrees; overlay so it’s on top of page content
    tw.append(pymupdf.Point(x, y), text, fontsize=fontsize, fontname="helv", rotate=30)
    tw.write_text(page, overlay=True)

def extract_single_page(pdf_path: Path, page_idx: int, out_dir: Path, mark: bool=False) -> Path:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input not found: {pdf_path}")

    with pymupdf.open(pdf_path) as src:
        if page_idx < 0 or page_idx >= src.page_count:
            raise IndexError(
                f"Page index {page_idx} out of range for {pdf_path.name} "
                f"(page_count={src.page_count})"
            )

        stem = pdf_path.stem
        out_path = out_dir / f"{stem}_page_{page_idx}.pdf"

        new_doc = pymupdf.open()  # empty PDF
        new_doc.insert_pdf(src, from_page=page_idx, to_page=page_idx)

        if mark:
            label = f"{pdf_path.name} — page {page_idx + 1} of {src.page_count}"
            _add_big_mark(new_doc[0], label)

        new_doc.save(out_path)
        new_doc.close()

    return out_path

def merge_pdfs(pdf_paths: list[Path], out_path: Path) -> None:
    """Merge the given single-page PDFs in order into out_path."""
    merged = pymupdf.open()
    try:
        for p in pdf_paths:
            with pymupdf.open(p) as doc:
                merged.insert_pdf(doc)
        if merged.page_count == 0:
            print("No pages merged; nothing to save.")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(out_path)
        print(f"✅ Merged {len(pdf_paths)} pages into: {out_path.resolve()}")
    finally:
        merged.close()

def main():
    parser = argparse.ArgumentParser(description="Extract specific pages; optionally watermark; then merge.")
    parser.add_argument("--mark", action="store_true",
                        help="If set, print source filename and page count on each output PDF page.")
    args = parser.parse_args()

    written: list[Path] = ['data/01_of_X_2.pdf']
    for src, idx in tasks:
        try:
            out = extract_single_page(src, idx, OUTPUT_DIR, mark=args.mark)
            written.append(out)
            print(f"✅ Wrote: {out}")
        except Exception as e:
            print(f"❌ Failed for {src} (page_idx={idx}): {e}")

    # Merge all extracted pages (in produced order)
    if written:
        merge_pdfs(written, OUTPUT_DIR / "test_tmp.pdf")

if __name__ == "__main__":
    main()
