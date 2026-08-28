# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from vllm.logger import init_logger

logger = init_logger(__file__)

CHAT_TEMPLATES_DIR = Path(__file__).parent

ChatTemplatePath: TypeAlias = Path | Callable[[str], Path | None]


def _get_minicpmv_chat_template_fallback(tokenizer_name_or_path: str) -> Path | None:
    # MiniCPM-V-4.5 version uses a dedicated template
    if "4.5" in tokenizer_name_or_path or "4_5" in tokenizer_name_or_path:
        return CHAT_TEMPLATES_DIR / "template_minicpmv45.jinja"

    # Other versions use chatml template
    return CHAT_TEMPLATES_DIR / "template_chatml.jinja"


_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK: dict[str, ChatTemplatePath] = {
    "blip-2": CHAT_TEMPLATES_DIR / "template_blip2.jinja",
    "chameleon": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "clip": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "colpali": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "deepseek_ocr": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
    "deepseek_ocr2": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
    "unlimited-ocr": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
    "deepseek_vl_v2": CHAT_TEMPLATES_DIR / "template_deepseek_vl2.jinja",
    "minicpmv": _get_minicpmv_chat_template_fallback,
    "minicpmv4_6": _get_minicpmv_chat_template_fallback,
    "paligemma": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "siglip": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "siglip2": CHAT_TEMPLATES_DIR / "template_basic.jinja",
}


_MODEL_TYPE_TO_MM_CHAT_TEMPLATE_OVERRIDE: dict[str, tuple[str, Path]] = {
    # model_type -> (image placeholder the chat template must emit, canonical
    # multimodal template substituted when the checkpoint's template lacks it).
    # Some conversions of multimodal checkpoints package the text-only
    # release's chat template, which renders image/video content parts as a
    # refusal reminder instead of placeholder tokens; multimodal prompt
    # replacement then fails on every image request.
    "glm5_next": ("<|image|>", CHAT_TEMPLATES_DIR / "template_glm5next_mm.jinja"),
}


def get_mm_chat_template_override(model_type: str) -> tuple[str, Path] | None:
    """(placeholder, template path) for multimodal models whose checkpoint
    may ship a text-only chat template; ``None`` for everything else."""
    return _MODEL_TYPE_TO_MM_CHAT_TEMPLATE_OVERRIDE.get(model_type)


def register_chat_template_fallback_path(
    model_type: str,
    chat_template: ChatTemplatePath,
) -> None:
    if model_type in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Model type %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.",
            model_type,
            chat_template,
        )

    _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK[model_type] = chat_template


def get_chat_template_fallback_path(
    model_type: str,
    tokenizer_name_or_path: str,
) -> Path | None:
    chat_template = _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK.get(model_type)
    if callable(chat_template):
        chat_template = chat_template(tokenizer_name_or_path)

    if chat_template is None:
        return None

    return chat_template
