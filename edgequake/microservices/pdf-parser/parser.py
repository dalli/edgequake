import os
import logging
from typing import Tuple

import fitz  # PyMuPDF
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Resolution for rendering PDF pages to images.
# 150 DPI is a good balance between VLM accuracy and memory use.
# Increase to 200+ for dense tables with small text.
_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "150"))

# Maximum number of pages to process (0 = unlimited).
_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "0"))


def _pdf_page_to_image(page: fitz.Page, dpi: int = _RENDER_DPI) -> Image.Image:
    """Render a single PDF page to a PIL Image at the given DPI."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def process_pdf(file_path: str) -> Tuple[str, int, list]:
    """
    Process a PDF file using VLM page-by-page extraction.

    Pipeline:
    1. Each page is rendered to an image via PyMuPDF at PDF_RENDER_DPI resolution.
    2. The VLM extracts structured content (text, tables with colspan/rowspan, charts)
       and returns JSON per the SYSTEM_PROMPT schema defined in vlm.py.
    3. Cross-page table continuation is tracked: if a page ended with an incomplete
       table, the next page's VLM call is told to treat its first element as a
       potential continuation.
    4. All pages are assembled into a single Markdown document.

    Environment variables:
      PDF_RENDER_DPI       - Page image resolution (default: 150)
      PDF_MAX_PAGES        - Max pages to process; 0 = unlimited (default: 0)
      VLM_RESPONSE_LANGUAGE - Output language (default: Korean)
    """
    from vlm import analyze_page_with_vlm

    logger.info(f"Opening PDF: {file_path} (DPI={_RENDER_DPI})")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    max_pages = _MAX_PAGES if _MAX_PAGES > 0 else total_pages
    pages_to_process = min(total_pages, max_pages)

    logger.info(f"PDF has {total_pages} pages, processing {pages_to_process}")

    markdown_parts = [f"# Document\n\n*Total pages: {pages_to_process}*\n"]

    # Track whether the previous page ended with a table that continues
    prev_page_continues = False

    for page_num in range(pages_to_process):
        page = doc[page_num]
        logger.info(f"Processing page {page_num + 1}/{pages_to_process} (prev_continues={prev_page_continues})")

        try:
            img = _pdf_page_to_image(page, dpi=_RENDER_DPI)
            page_content, continues_to_next = analyze_page_with_vlm(
                img,
                page_number=page_num + 1,
                prev_page_continues=prev_page_continues,
            )
            prev_page_continues = continues_to_next
            markdown_parts.append(f"\n\n---\n\n<!-- page {page_num + 1} -->\n\n{page_content}")

        except Exception as e:
            logger.warning(f"Failed to process page {page_num + 1}: {e}")
            markdown_parts.append(
                f"\n\n---\n\n<!-- page {page_num + 1} -->\n\n*[페이지 추출 실패: {e}]*"
            )
            prev_page_continues = False

    doc.close()

    markdown = "\n".join(markdown_parts)
    logger.info(f"VLM processing complete: {pages_to_process} pages")
    return markdown, pages_to_process, []
