import os
import json
import logging
import re
from typing import Optional
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_vlm_engine = None
_vlm_type = None

_VLM_RESPONSE_LANGUAGE = os.getenv("VLM_RESPONSE_LANGUAGE", "Korean")

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a high-precision PDF document parser that analyzes PDF page images.
Your job is to extract ALL content from each page and return it as structured JSON.

## Output Format
Return a single valid JSON object with the following schema:
{
  "elements": [
    // One element per logical block, in top-to-bottom, left-to-right reading order
  ],
  "table_continues_to_next_page": false  // true if the last element is a table that likely continues
}

## Element Types

### 1. text
For headings, paragraphs, captions, footnotes, and any regular prose.
{
  "type": "text",
  "subtype": "heading" | "paragraph" | "caption" | "footnote" | "list_item" | "other",
  "heading_level": 1,  // 1–4, only when subtype is "heading"
  "content": "Full text content here"
}

### 2. table
For any tabular data. CRITICAL rules:
- Detect and represent colspan and rowspan for merged cells
- A cell that spans multiple columns has "colspan" > 1
- A cell that spans multiple rows has "rowspan" > 1
- For cells that are "shadowed" by a colspan/rowspan from a previous cell, emit a special {"spanned": true} cell
- Detect if this table is a CONTINUATION of a table from the previous page:
  - "table_continues_from_prev" = true when: the table has no visible header row, OR the column structure matches what a cross-page continuation would look like
- Detect if this table likely CONTINUES to the next page:
  - "table_continues_to_next" = true when: the table appears to be cut off at the bottom of the page

{
  "type": "table",
  "table_continues_from_prev": false,   // true if this is a continuation of the previous page's table
  "table_continues_to_next": false,     // true if this table likely continues on the next page
  "has_header": true,
  "headers": [
    { "text": "Column A", "colspan": 1, "rowspan": 1 },
    { "text": "Column B", "colspan": 2, "rowspan": 1 },
    { "spanned": true }
  ],
  "rows": [
    [
      { "text": "Row 1, Col 1", "colspan": 1, "rowspan": 2 },
      { "text": "Row 1, Col 2", "colspan": 1, "rowspan": 1 },
      { "text": "Row 1, Col 3", "colspan": 1, "rowspan": 1 }
    ],
    [
      { "spanned": true },
      { "text": "Row 2, Col 2", "colspan": 1, "rowspan": 1 },
      { "text": "Row 2, Col 3", "colspan": 1, "rowspan": 1 }
    ]
  ]
}

### 3. chart
For any chart, graph, or data visualization.
If an image is a chart: extract all quantitative data and axis information.
Include both the data table AND a description.
{
  "type": "chart",
  "chart_type": "bar" | "line" | "pie" | "scatter" | "area" | "radar" | "heatmap" | "other",
  "title": "Chart title if visible",
  "x_axis": {
    "label": "X axis label",
    "unit": "unit if any (e.g., %, million, year)",
    "values": ["2020", "2021", "2022"]
  },
  "y_axis": {
    "label": "Y axis label",
    "unit": "unit if any",
    "range": [0, 100]
  },
  "series": [
    {
      "name": "Series name (e.g., legend label)",
      "values": [10.5, 23.1, 41.8]
    }
  ],
  "description": "A natural language description of what the chart shows and key insights"
}

### 4. image
For non-chart images (photos, diagrams, logos, etc.)
{
  "type": "image",
  "alt_text": "Description of what the image shows",
  "caption": "Caption text if visible near the image"
}

## Critical Rules
1. ALL text must be in the correct response language specified in the user prompt
2. Do NOT summarize or paraphrase text content — reproduce it exactly
3. Numbers in tables and charts must be extracted precisely — do not approximate
4. If a table cell is empty, use {"text": "", "colspan": 1, "rowspan": 1}
5. Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
6. If a page has no content, return {"elements": [], "table_continues_to_next_page": false}
"""

# ─────────────────────────────────────────────────────────────────────────────
# JSON → Markdown converter
# ─────────────────────────────────────────────────────────────────────────────

def _table_to_markdown(table: dict) -> str:
    """Convert a structured table element to GFM Markdown table format."""
    lines = []

    if table.get("has_header") and table.get("headers"):
        headers = table["headers"]
        header_cells = []
        for cell in headers:
            if cell.get("spanned"):
                header_cells.append("↑")
            else:
                txt = cell.get("text", "")
                cs = cell.get("colspan", 1)
                if cs > 1:
                    header_cells.extend([txt] + ["→"] * (cs - 1))
                else:
                    header_cells.append(txt)
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

    for row in table.get("rows", []):
        row_cells = []
        for cell in row:
            if cell.get("spanned"):
                row_cells.append("↑")
            else:
                txt = cell.get("text", "")
                cs = cell.get("colspan", 1)
                rs = cell.get("rowspan", 1)
                # Annotate merges inline
                if cs > 1 or rs > 1:
                    txt = f"{txt} [span:{cs}×{rs}]"
                row_cells.append(txt)
        lines.append("| " + " | ".join(row_cells) + " |")

    meta = []
    if table.get("table_continues_from_prev"):
        meta.append("*↑ 이전 페이지에서 이어지는 표*")
    if table.get("table_continues_to_next"):
        meta.append("*↓ 다음 페이지로 이어지는 표*")

    return ("\n".join(meta) + "\n" if meta else "") + "\n".join(lines)


def _chart_to_markdown(chart: dict) -> str:
    """Convert a structured chart element to Markdown with data table."""
    parts = []

    title = chart.get("title", "")
    if title:
        parts.append(f"**[차트: {title}]**")

    desc = chart.get("description", "")
    if desc:
        parts.append(f"\n{desc}\n")

    # Render data table
    x_axis = chart.get("x_axis", {})
    y_axis = chart.get("y_axis", {})
    series_list = chart.get("series", [])
    x_values = x_axis.get("values", [])
    x_label = x_axis.get("label", "X")
    x_unit = x_axis.get("unit", "")
    y_label = y_axis.get("label", "Y")
    y_unit = y_axis.get("unit", "")

    if x_values and series_list:
        x_header = f"{x_label} ({x_unit})" if x_unit else x_label
        series_headers = [
            f"{s.get('name', f'값{i+1}')} ({y_unit})" if y_unit else s.get('name', f'값{i+1}')
            for i, s in enumerate(series_list)
        ]
        headers = [x_header] + series_headers
        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for col_idx, x_val in enumerate(x_values):
            row = [str(x_val)]
            for s in series_list:
                vals = s.get("values", [])
                row.append(str(vals[col_idx]) if col_idx < len(vals) else "")
            parts.append("| " + " | ".join(row) + " |")

    return "\n".join(parts)


def _elements_to_markdown(page_data: dict) -> str:
    """Convert the structured JSON page data to Markdown."""
    parts = []
    for el in page_data.get("elements", []):
        el_type = el.get("type", "text")

        if el_type == "text":
            st = el.get("subtype", "paragraph")
            content = el.get("content", "")
            level = el.get("heading_level", 1)
            if st == "heading":
                parts.append("#" * level + " " + content)
            elif st == "footnote":
                parts.append(f"*{content}*")
            elif st == "list_item":
                parts.append(f"- {content}")
            else:
                parts.append(content)

        elif el_type == "table":
            parts.append(_table_to_markdown(el))

        elif el_type == "chart":
            parts.append(_chart_to_markdown(el))

        elif el_type == "image":
            alt = el.get("alt_text", "이미지")
            caption = el.get("caption", "")
            parts.append(f"![{alt}]({{}})") # placeholder – actual image embedding handled separately
            if caption:
                parts.append(f"*{caption}*")

    return "\n\n".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────────────────────────────────────
# VLM Engine
# ─────────────────────────────────────────────────────────────────────────────

def get_vlm_engine():
    global _vlm_engine, _vlm_type
    if _vlm_engine is None:
        model_name = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

        if torch.cuda.is_available():
            try:
                from vllm import LLM
                logger.info("CUDA available. Initializing vLLM Engine...")
                _vlm_engine = LLM(
                    model=model_name,
                    trust_remote_code=True,
                    max_model_len=8192,
                    limit_mm_per_prompt={"image": 1}
                )
                _vlm_type = "vllm"
                logger.info("vLLM Engine initialized.")
                return _vlm_engine, _vlm_type
            except Exception as e:
                logger.warning(f"vLLM init failed: {e}. Falling back to CPU.")
        else:
            logger.info("CUDA not available. Using CPU via Transformers.")

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            logger.info(f"Loading {model_name} on CPU (no GPU detected)...")
            # WHY: Use dtype= instead of deprecated torch_dtype=
            # WHY: Avoid device_map="cpu" which requires accelerate; use .to("cpu") instead
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float32,
            ).to("cpu")
            processor = AutoProcessor.from_pretrained(model_name)
            _vlm_engine = {"model": model, "processor": processor}
            _vlm_type = "transformers"
            logger.info("Transformers engine initialized on CPU.")
        except Exception as e:
            logger.error(f"Transformers CPU init failed: {e}", exc_info=True)
            _vlm_engine = "mock"
            _vlm_type = "mock"
            logger.warning(
                "VLM running in MOCK mode — parsed output will be placeholders. "
                "Install accelerate or provide a GPU to enable real VLM inference."
            )

    return _vlm_engine, _vlm_type


def _run_vlm_inference(engine, engine_type: str, image: Image.Image, user_prompt: str) -> str:
    """Run VLM inference. System prompt is prepended for models that support it."""
    if engine_type == "mock":
        mock_json = {
            "elements": [{"type": "text", "subtype": "paragraph", "content": f"[Mock: {image.width}x{image.height}]"}],
            "table_continues_to_next_page": False
        }
        return json.dumps(mock_json, ensure_ascii=False)

    if engine_type == "vllm":
        from vllm import SamplingParams
        # Combine system + user prompt (vLLM may not support system role for all models)
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"
        sampling_params = SamplingParams(temperature=0.05, max_tokens=4096, stop=None)
        outputs = engine.generate(
            {"prompt": full_prompt, "multi_modal_data": {"image": image}},
            sampling_params=sampling_params
        )
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()

    elif engine_type == "transformers":
        model = engine["model"]
        processor = engine["processor"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    return '{"elements": [], "table_continues_to_next_page": false}'


def _parse_vlm_response(raw: str) -> dict:
    """Extract and parse the JSON object from a VLM response string."""
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)

    # Find first { ... } JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Raw response snippet: {raw[:200]}")

    return {"elements": [], "table_continues_to_next_page": False}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _build_page_user_prompt(page_number: int, is_continuation: bool, language: str) -> str:
    """Build the per-page user prompt."""
    continuation_hint = (
        "\n⚠️ HINT: The FIRST element on this page may be a table that continues from the previous page. "
        "If so, set table_continues_from_prev=true on that table element."
        if is_continuation else ""
    )
    return (
        f"Extract ALL content from this PDF page (page {page_number}) and respond in {language}.{continuation_hint}\n"
        f"Return structured JSON exactly as specified in the system instructions."
    )


def analyze_page_with_vlm(
    image: Image.Image,
    page_number: int = 1,
    prev_page_continues: bool = False,
) -> tuple[str, bool]:
    """
    Extract all content from a PDF page image.

    Returns:
        (markdown_content, table_continues_to_next) tuple.
        `table_continues_to_next` is True when the page ends with an incomplete table.
    """
    engine, engine_type = get_vlm_engine()
    user_prompt = _build_page_user_prompt(page_number, prev_page_continues, _VLM_RESPONSE_LANGUAGE)

    logger.info(f"VLM page {page_number} ({engine_type}), lang={_VLM_RESPONSE_LANGUAGE}, continuation={prev_page_continues}")
    raw = _run_vlm_inference(engine, engine_type, image, user_prompt)

    page_data = _parse_vlm_response(raw)
    markdown = _elements_to_markdown(page_data)
    continues = page_data.get("table_continues_to_next_page", False)

    return markdown, continues


def analyze_image_with_vlm(image: Image.Image) -> str:
    """Analyze a standalone chart or image element (legacy interface)."""
    engine, engine_type = get_vlm_engine()
    user_prompt = (
        f"Analyze this chart or image and respond in {_VLM_RESPONSE_LANGUAGE}. "
        f"Return structured JSON with a single chart or image element as specified."
    )
    raw = _run_vlm_inference(engine, engine_type, image, user_prompt)
    page_data = _parse_vlm_response(raw)
    return _elements_to_markdown(page_data)
