import os
import logging
from typing import Tuple

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult

logger = logging.getLogger(__name__)

def get_converter() -> DocumentConverter:
    """
    Initializes and returns a configured Docling DocumentConverter.
    Configured for high precision table extraction.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    
    # Docling > 2.0 handles multi-page table merging natively when possible,
    # but we ensure table structure recognition is fully enabled.

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
                # If get_image is not available or fails, it might need crop from pdf coords
                img = pic.get_image(document)
                if img:
                    logger.info(f"Analyzing image/chart at page {pic.prov[0].page_no} with VLM...")
                    vlm_text = analyze_image_with_vlm(img)
                    vlm_descriptions[pic.get_ref().id_] = vlm_text
            except Exception as e:
                logger.warning(f"Failed to process image with VLM: {str(e)}")

    # Export to markdown. 
    # Current docling versions might not have simple placeholder replacement natively for all items,
    # so we iterate through items and generate custom markdown if needed, OR we just append 
    # VLM output to the end of the markdown for missing links. 
    # A robust way is to export markdown and append the VLM context.
    markdown = document.export_to_markdown()
    
    # Inject VLM descriptions into the markdown
    # Assuming docling puts image placeholders or we can just append them explicitly
    if vlm_descriptions:
        markdown += "\n\n## Visual Elements Analysis (VLM)\n\n"
        for ref_id, desc in vlm_descriptions.items():
            markdown += f"### Element {ref_id}\n{desc}\n\n"
            
    logger.info(f"Two-Track conversion complete. Pages: {page_count}, VLM Elements processed: {len(vlm_descriptions)}")
    
    elements = getattr(document, "texts", []) + getattr(document, "pictures", []) + getattr(document, "tables", [])
    return markdown, page_count, elements

