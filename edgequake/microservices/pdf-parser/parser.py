import os
import logging
from typing import Tuple

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult

logger = logging.getLogger(__name__)

# Language configuration for OCR.
# Set OCR_LANGUAGES env var to a comma-separated list of language codes.
# Examples: "ko,en" (Korean + English), "ja,en" (Japanese + English), "zh,en" (Chinese + English)
# Default: "ko,en" to support Korean documents out of the box.
_OCR_LANGUAGES = [
    lang.strip()
    for lang in os.getenv("OCR_LANGUAGES", "ko,en").split(",")
    if lang.strip()
]


def get_converter() -> DocumentConverter:
    """
    Initializes and returns a configured Docling DocumentConverter.
    Configured for high precision table extraction and multi-language OCR.

    OCR language is controlled by the OCR_LANGUAGES environment variable.
    Default: "ko,en" (Korean + English).
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # Configure OCR languages for multi-language support (including Korean)
    try:
        from docling.datamodel.pipeline_options import EasyOcrOptions
        pipeline_options.ocr_options = EasyOcrOptions(lang=_OCR_LANGUAGES)
        logger.info(f"OCR configured with languages: {_OCR_LANGUAGES}")
    except (ImportError, AttributeError):
        # Older versions of docling may not support EasyOcrOptions
        logger.warning(
            "EasyOcrOptions not available in this docling version. "
            f"OCR language may default to English only. "
            f"Requested: {_OCR_LANGUAGES}"
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def process_pdf(file_path: str) -> Tuple[str, int, list]:
    """
    Processes a PDF file using Docling, extracts images, runs them through the VLM,
    and injects the VLM descriptions back into the final Markdown.
    """
    logger.info(f"Starting Docling conversion for: {file_path}")
    converter = get_converter()

    # Run conversion
    result = converter.convert(file_path)
    document = result.document
    page_count = len(result.input.pages) if getattr(result.input, 'pages', None) else 0

    # Enable image exporting in docling to extract bounding box images
    vlm_descriptions = {}

    # Lazy load VLM function
    from vlm import analyze_image_with_vlm

    # Process each picture found in the document
    if hasattr(document, 'pictures'):
        for pic in document.pictures:
            try:
                # Some docling versions support extracting PIL image directly
                img = pic.get_image(document)
                if img:
                    logger.info(f"Analyzing image/chart at page {pic.prov[0].page_no} with VLM...")
                    vlm_text = analyze_image_with_vlm(img)
                    vlm_descriptions[pic.get_ref().id_] = vlm_text
            except Exception as e:
                logger.warning(f"Failed to process image with VLM: {str(e)}")

    # Export to markdown.
    markdown = document.export_to_markdown()

    # Inject VLM descriptions into the markdown
    if vlm_descriptions:
        markdown += "\n\n## Visual Elements Analysis (VLM)\n\n"
        for ref_id, desc in vlm_descriptions.items():
            markdown += f"### Element {ref_id}\n{desc}\n\n"

    logger.info(f"Two-Track conversion complete. Pages: {page_count}, VLM Elements processed: {len(vlm_descriptions)}")

    elements = getattr(document, "texts", []) + getattr(document, "pictures", []) + getattr(document, "tables", [])
    return markdown, page_count, elements
