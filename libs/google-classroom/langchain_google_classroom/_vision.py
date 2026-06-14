"""Shared vision LLM helper for image description.

Centralises the duplicated ``_describe_image`` logic that was present
in ``PDFParser``, ``DocxParser``, and ``ImageParser``.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def describe_image(
    image_data: bytes,
    vision_model: BaseChatModel,
    *,
    prompt: str = (
        "Describe this image in detail for a student studying this course material."
    ),
    name: str = "image",
) -> Optional[str]:
    """Use a vision LLM to produce a text description of an image.

    Args:
        image_data: Raw image bytes (PNG, JPEG, GIF, WebP, or BMP).
        vision_model: A LangChain chat model with vision support
            (e.g. ``ChatGoogleGenerativeAI``, ``ChatOpenAI`` with GPT-4V).
        prompt: Text prompt sent alongside the image.
        name: Human-readable image name for error logging.

    Returns:
        A text description from the vision model, or ``None`` on failure.
    """
    from langchain_core.messages import HumanMessage

    from langchain_google_classroom.parsers.image_parser import detect_image_mime

    b64 = base64.b64encode(image_data).decode("utf-8")
    mime = detect_image_mime(image_data)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                },
            },
        ]
    )

    try:
        response = vision_model.invoke([message])
        content: Any = response.content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            content = " ".join(parts)
        return str(content)
    except Exception as exc:
        logger.warning("Vision LLM failed for image %s: %s", name, exc)
        return None
