import os
import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# To prevent loading VLM unconditionally on import, we load it lazily
_vlm_engine = None
_vlm_type = None  # Tracks if we initialized 'vllm', 'transformers', or 'mock'

# Language for VLM responses.
# Set VLM_RESPONSE_LANGUAGE to the desired language name (e.g., "Korean", "English", "Japanese").
# Default: "Korean" to ensure Korean documents produce Korean VLM descriptions.
_VLM_RESPONSE_LANGUAGE = os.getenv("VLM_RESPONSE_LANGUAGE", "Korean")


def get_vlm_engine():
    global _vlm_engine, _vlm_type
    if _vlm_engine is None:
        model_name = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

        # 1. Attempt GPU/vLLM Initialization
        if torch.cuda.is_available():
            try:
                from vllm import LLM
                logger.info("CUDA is available. Initializing vLLM Engine for Qwen2-VL...")
                _vlm_engine = LLM(
                    model=model_name,
                    trust_remote_code=True,
                    max_model_len=4096,
                    limit_mm_per_prompt={"image": 1}
                )
                _vlm_type = "vllm"
                logger.info("vLLM Engine initialized successfully.")
                return _vlm_engine, _vlm_type
            except Exception as e:
                logger.warning(f"Failed to initialize vLLM (GPU available but encountered error): {e}. Falling back to CPU.")
        else:
            logger.info("CUDA is NOT available. Skipping vLLM and falling back to CPU via Transformers.")

        # 2. CPU Fallback via HuggingFace Transformers
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            logger.info(f"Initializing {model_name} on CPU. This may take significant RAM and be slower.")
            _vlm_engine = {
                "model": Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                ),
                "processor": AutoProcessor.from_pretrained(model_name)
            }
            _vlm_type = "transformers"
            logger.info("Transformers Engine initialized successfully on CPU.")
        except Exception as e:
            logger.error(f"Failed to initialize Transformers CPU fallback: {e}")
            _vlm_engine = "mock"
            _vlm_type = "mock"

    return _vlm_engine, _vlm_type


def _build_vlm_prompt(language: str = _VLM_RESPONSE_LANGUAGE) -> str:
    """Build a language-aware VLM analysis prompt.

    Args:
        language: Target language for the response (e.g., "Korean", "English", "Japanese").
                  Controlled by VLM_RESPONSE_LANGUAGE environment variable.

    Returns:
        Prompt string instructing the VLM to respond in the given language.
    """
    return (
        f"Please analyze this chart or image and respond in {language}. "
        f"Extract any numerical data into a table format and explain the core insights clearly. "
        f"If there is text in the image, transcribe it accurately."
    )


def analyze_image_with_vlm(image: Image.Image) -> str:
    """
    Analyzes an image containing a chart or complex structure using the Vision Language Model.

    The response language is controlled by the VLM_RESPONSE_LANGUAGE environment variable
    (default: "Korean"). This ensures consistency between the document language and VLM output.
    """
    engine, engine_type = get_vlm_engine()
    prompt = _build_vlm_prompt(_VLM_RESPONSE_LANGUAGE)
    logger.debug(f"VLM prompt language: {_VLM_RESPONSE_LANGUAGE}")

    if engine_type == "mock":
        return f"[Mock VLM Response]: Chart analyzed successfully. Dimensions: {image.width}x{image.height}"

    if engine_type == "vllm":
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)
        outputs = engine.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params
        )
        if outputs and len(outputs) > 0 and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()

    elif engine_type == "transformers":
        model = engine["model"]
        processor = engine["processor"]

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]

        # Format for Qwen2-VL using transformers
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Inference on CPU
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

    return "[VLM Analysis Failed]"
