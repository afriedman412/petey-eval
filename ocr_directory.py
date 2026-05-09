"""
ocr_directory.py — Standalone Tesseract OCR for a directory of PDFs.

Saves one .txt per PDF so runs are resumable. Skips files whose output
already exists. Runs pages in parallel within each PDF.

Edit INPUT_DIR / OUTPUT_DIR / DPI below, then run:
    python ocr_directory.py
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


INPUT_DIR = "/Users/user/Documents/code/petey-master/benchmarks/par_decision"
OUTPUT_DIR = "/Users/user/Documents/code/petey-master/benchmarks/par_tesseract"
DPI = 300


def ocr_page(args):
    pdf_path, page_num, dpi = args
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    text = pytesseract.image_to_string(img)
    doc.close()
    return page_num, text


def ocr_pdf(pdf_path: Path, out_path: Path, dpi: int):
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    page_args = [(str(pdf_path), i, dpi) for i in range(n_pages)]
    pages = [None] * n_pages

    with ProcessPoolExecutor(max_workers=min(4, n_pages)) as ex:
        for fut in as_completed([ex.submit(ocr_page, a) for a in page_args]):
            i, text = fut.result()
            pages[i] = text

    out_path.write_text("\n\n--- PAGE BREAK ---\n\n".join(pages))


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    todo = [p for p in pdfs if not (out_dir / f"{p.stem}.txt").exists()]
    done = len(pdfs) - len(todo)

    print(f"Found {len(pdfs)} PDFs in {in_dir}")
    print(f"  {done} already done, {len(todo)} to process")

    start = time.time()
    for i, pdf in enumerate(todo, 1):
        out_path = out_dir / f"{pdf.stem}.txt"
        t0 = time.time()
        try:
            ocr_pdf(pdf, out_path, DPI)
            elapsed = time.time() - t0
            print(f"  [{i}/{len(todo)}] {pdf.name}  ({elapsed:.1f}s)")
        except Exception as e:
            msg = f"  [{i}/{len(todo)}] {pdf.name}  FAILED: {e}"
            print(msg, file=sys.stderr)

    total = time.time() - start
    print(f"\nDone. Processed {len(todo)} files in {total:.1f}s")


if __name__ == "__main__":
    main()
