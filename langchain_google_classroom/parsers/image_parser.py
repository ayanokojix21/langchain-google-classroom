"""Image file parser with optional vision LLM description."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.documents.base import Blob
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image MIME detection from magic bytes
# ---------------------------------------------------------------------------

_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),
    (b"BM", "image/bmp"),
]


def detect_image_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes.

    Args:
        data: Raw image bytes.

    Returns:
        MIME type string, defaults to ``"image/png"`` if unknown.
    """
    for sig, mime in _SIGNATURES:
        if data[: len(sig)] == sig:
            return mime
    return "image/png"


class ImageParser(BaseBlobParser):
    """Parse image files into ``Document`` objects.

    When a *vision_model* is provided, the image is sent to the LLM
    for a textual description.  Otherwise, the image bytes are stored
    as a base64 string in ``Document.metadata["image_base64"]`` for
    downstream multimodal pipelines.

    Conforms to LangChain's :class:`BaseBlobParser` interface.
    """

    def __init__(
        self,
        *,
        vision_model: Optional[BaseChatModel] = None,
        image_prompt: str = (
            "Describe this image in detail for a student studying this course material."
        ),
    ) -> None:
        """Initialise the image parser.

        Args:
            vision_model: Optional LangChain chat model with vision
                support (e.g. ``ChatGoogleGenerativeAI``,
                ``ChatOpenAI`` with GPT-4V).
            image_prompt: Prompt sent alongside the image to the
                vision model.
        """
        self.vision_model = vision_model
        self.image_prompt = image_prompt

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse an image blob into a Document.

        Args:
            blob: A LangChain ``Blob`` containing image bytes.

        Yields:
            One ``Document`` with either a vision-generated
            description or a base64 placeholder.
        """
        data = blob.as_bytes()
        if not data:
            return

        source = str(blob.source) if blob.source else "unnamed"

        if self.vision_model:
            description = self._describe_image(data, source)
            page_content = (
                f"[Image: {source}]\n{description}"
                if description
                else f"[Image: {source}]"
            )
            metadata: dict[str, Any] = {
                "source": source,
                "mime_type": blob.mimetype or "",
            }
        else:
            page_content = f"[Image: {source}]"
            metadata = {
                "source": source,
                "mime_type": blob.mimetype or "",
                "image_base64": base64.b64encode(data).decode("utf-8"),
            }

        yield Document(page_content=page_content, metadata=metadata)

    def _describe_image(self, image_bytes: bytes, name: str) -> Optional[str]:
        """Send image to vision LLM and return description.

        Args:
            image_bytes: Raw image bytes.
            name: Image name for logging.

        Returns:
            Text description from the vision model, or ``None``
            on failure.
        """
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = detect_image_mime(image_bytes)

        message = HumanMessage(
            content=[
                {"type": "text", "text": self.image_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                },
            ]
        )
        try:
            response = self.vision_model.invoke([message])  # type: ignore[union-attr]
            content = response.content
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
