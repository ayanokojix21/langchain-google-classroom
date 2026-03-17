"""DOCX file parser using python-docx (BaseBlobParser interface)."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from langchain_core.documents.base import Blob
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class DocxParser(BaseBlobParser):
    """Parse DOCX files into ``Document`` objects using ``python-docx``.

    Conforms to LangChain's :class:`BaseBlobParser` interface.

    !!! note "Optional Dependency"

        Requires ``python-docx``.  Install via:

        ```bash
        pip install "langchain-google-classroom[parsers]"
        ```

    When a ``vision_model`` is provided, embedded DOCX images are
    described and appended to the final extracted text.
    """

    def __init__(
        self,
        *,
        vision_model: Optional[BaseChatModel] = None,
        image_prompt: str = (
            "Describe this image in detail for a student studying this course material."
        ),
    ) -> None:
        """Initialise the DOCX parser.

        Args:
            vision_model: Optional LangChain chat model with vision support.
            image_prompt: Prompt sent alongside each embedded image.
        """
        self.vision_model = vision_model
        self.image_prompt = image_prompt

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse a DOCX blob into a single ``Document``.

        All paragraphs are concatenated into one document.

        Args:
            blob: A LangChain ``Blob`` containing DOCX bytes.

        Yields:
            A single ``Document`` with all paragraph text.
        """
        DocxDocument = guard_import(
            module_name="docx",
            pip_name="python-docx",
        ).Document

        from io import BytesIO

        doc = DocxDocument(BytesIO(blob.as_bytes()))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)

        if self.vision_model:
            image_notes = self._extract_docx_image_descriptions(doc)
            if image_notes:
                if text:
                    text += "\n\n"
                text += "\n\n".join(image_notes)

        if text:
            yield Document(
                page_content=text,
                metadata={"source": blob.source or ""},
            )

    def _extract_docx_image_descriptions(self, doc: Any) -> list[str]:
        """Extract embedded images and return vision understanding context.

        Args:
            doc: A ``python-docx`` document object.

        Returns:
            A list of formatted image understanding blocks.
        """
        part = getattr(doc, "part", None)
        rels = getattr(part, "rels", None)
        if rels is None or not hasattr(rels, "values"):
            return []

        descriptions: list[str] = []
        for index, rel in enumerate(rels.values(), start=1):
            reltype = str(getattr(rel, "reltype", ""))
            if "image" not in reltype:
                continue

            target_part = getattr(rel, "target_part", None)
            image_blob = getattr(target_part, "blob", None)
            if not isinstance(image_blob, (bytes, bytearray)) or not image_blob:
                continue

            image_name = str(getattr(target_part, "partname", ""))
            if not image_name:
                image_name = f"image_{index}"

            description = self._describe_image(bytes(image_blob), image_name)
            if description:
                descriptions.append(f"[Image: {image_name}]\n{description}")

        return descriptions

    def _describe_image(self, image_bytes: bytes, name: str) -> Optional[str]:
        """Send image bytes to vision model and return description text."""
        from langchain_core.messages import HumanMessage

        from langchain_google_classroom.parsers.image_parser import detect_image_mime

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
            logger.warning("Vision LLM failed for DOCX image %s: %s", name, exc)
            return None
