# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash multimodal chat template repair.

Some GLM-5.3-Flash conversions (e.g. EXL3) package the text-only release's
chat template, which renders image/video content parts as a refusal reminder
instead of ``<|begin_of_image|><|image|><|end_of_image|>``; multimodal prompt
replacement then fails on every image request. ``resolve_chat_template``
substitutes the bundled canonical template whenever a glm5_next multimodal
model's template never emits the image placeholder.
"""

from types import SimpleNamespace

import jinja2
import pytest

from vllm.transformers_utils.chat_templates.registry import (
    get_mm_chat_template_override,
)

TEXT_ONLY_SNIPPET = (
    "{%- for m in messages -%}{{ m.content }}{%- endfor -%}"
    "<reminder>You are unable to process this image</reminder>"
)


def _render_bundled_template(messages, **kwargs):
    _, path = get_mm_chat_template_override("glm5_next")
    env = jinja2.Environment(
        extensions=["jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.from_string(path.read_text())
    return template.render(
        messages=messages, add_generation_prompt=True, **kwargs
    )


def test_override_registered_for_glm5_next():
    override = get_mm_chat_template_override("glm5_next")
    assert override is not None
    placeholder, path = override
    assert placeholder == "<|image|>"
    assert path.is_file()
    assert placeholder in path.read_text()

    assert get_mm_chat_template_override("llama") is None


@pytest.mark.parametrize("image_type", ["image", "image_url"])
def test_bundled_template_emits_image_placeholder(image_type):
    rendered = _render_bundled_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": image_type},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
    )
    assert "<|begin_of_image|><|image|><|end_of_image|>" in rendered
    assert "Describe this image." in rendered
    assert rendered.rstrip().endswith("<|assistant|><think>")


def test_bundled_template_emits_video_placeholder():
    rendered = _render_bundled_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video_url"},
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
    )
    assert "<|begin_of_video|><|video|><|end_of_video|>" in rendered


def _model_config(model_type: str, is_multimodal: bool):
    return SimpleNamespace(
        model="dummy-model",
        hf_config=SimpleNamespace(model_type=model_type),
        is_multimodal_model=is_multimodal,
    )


def test_repair_replaces_text_only_template():
    from vllm.renderers.hf import _maybe_repair_text_only_mm_chat_template

    repaired = _maybe_repair_text_only_mm_chat_template(
        TEXT_ONLY_SNIPPET, _model_config("glm5_next", is_multimodal=True)
    )
    assert repaired is not None
    assert "<|image|>" in repaired


def test_repair_keeps_template_that_already_emits_placeholder():
    from vllm.renderers.hf import _maybe_repair_text_only_mm_chat_template

    good = "{{ messages }}<|begin_of_image|><|image|><|end_of_image|>"
    assert (
        _maybe_repair_text_only_mm_chat_template(
            good, _model_config("glm5_next", is_multimodal=True)
        )
        is good
    )


def test_repair_leaves_text_only_models_alone():
    from vllm.renderers.hf import _maybe_repair_text_only_mm_chat_template

    assert (
        _maybe_repair_text_only_mm_chat_template(
            TEXT_ONLY_SNIPPET, _model_config("glm5_next", is_multimodal=False)
        )
        is TEXT_ONLY_SNIPPET
    )
    assert (
        _maybe_repair_text_only_mm_chat_template(
            TEXT_ONLY_SNIPPET, _model_config("llama", is_multimodal=True)
        )
        is TEXT_ONLY_SNIPPET
    )
