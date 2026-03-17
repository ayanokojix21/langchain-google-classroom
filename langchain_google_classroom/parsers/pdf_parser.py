"""PDF file parser using pypdf (BaseBlobParser interface)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from langchain_core.documents.base import Blob
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class PDFParser(BaseBlobParser):
    """Parse PDF files into ``Document`` objects using ``pypdf``.

    Conforms to LangChain's :class:`BaseBlobParser` interface so it can be
    composed with any blob loader in the ecosystem.

    When a *vision_model* is provided, embedded images on each page are
    extracted via ``pypdf``'s ``page.images`` API, sent to the vision
    LLM for description, and the description is appended to the page
    text.

    !!! note "Optional Dependency"

        Requires ``pypdf``.  Install via:

        ```bash
        pip install "langchain-google-classroom[parsers]"
        ```

    ??? example "Basic Usage"

        ```python
        from langchain_google_classroom.parsers import PDFParser

        parser = PDFParser()
        docs = list(parser.lazy_parse(blob))
        ```

    ??? example "With Vision LLM"

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        vision = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        parser = PDFParser(vision_model=vision)
        docs = list(parser.lazy_parse(blob))
        ```
    """

    def __init__(
        self,
        *,
        vision_model: Optional[BaseChatModel] = None,
        image_prompt: str = (
            "Describe this image in detail for a student studying this course material."
        ),
    ) -> None:
        """Initialise the PDF parser.

        Args:
            vision_model: Optional LangChain chat model with vision
                support (e.g. ``ChatGoogleGenerativeAI``,
                ``ChatOpenAI`` with GPT-4V).  When set, embedded
                images in the PDF are sent to this model and the
                returned understanding context is appended to the page text.
            image_prompt: Prompt sent alongside each embedded image
                to the vision model.
        """
        self.vision_model = vision_model
        self.image_prompt = image_prompt

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse a PDF blob into Document objects.

        Each page is emitted as a separate ``Document`` whose metadata
        includes ``source`` and ``page`` number.  When a vision model
        is configured, image understanding context is appended to the page
        text.

        Args:
            blob: A LangChain ``Blob`` containing PDF bytes.

        Yields:
            One ``Document`` per page.
        """
        PdfReader = guard_import(
            module_name="pypdf",
            pip_name="pypdf",
        ).PdfReader

        from io import BytesIO

        reader = PdfReader(BytesIO(blob.as_bytes()))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            # Extract and describe embedded images if vision is on
            if self.vision_model:
                text = self._process_page_images(page, text)

            if text.strip():
                yield Document(
                    page_content=text,
                    metadata={
                        "source": blob.source or "",
                        "page": i + 1,
                    },
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_page_images(self, page: Any, text: str) -> str:
        """Extract images from a page and append understanding context.

        Args:
            page: A ``pypdf`` page object.
            text: Existing text extracted from the page.

        Returns:
            Text with image understanding context appended.
        """
        try:
            images = page.images
        except Exception:
            return text

        for img_obj in images:
            try:
                description = self._describe_image(img_obj.data, img_obj.name)
                if description:
                    text += f"\n\n[Image: {img_obj.name}]\n{description}"
            except Exception as exc:
                logger.warning(
                    "Failed to process image %s: %s",
                    getattr(img_obj, "name", "unknown"),
                    exc,
                )
        return text

    def _describe_image(self, image_bytes: bytes, name: str) -> Optional[str]:
        """Send image to vision LLM and return description.

        Args:
            image_bytes: Raw image bytes.
            name: Image filename for logging.

        Returns:
            Text description from the vision model, or ``None``
            on failure.
        """
        import base64

        from langchain_core.messages import HumanMessage

        from langchain_google_classroom.parsers.image_parser import (
            detect_image_mime,
        )

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
