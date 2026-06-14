"""Unit tests for the shared _vision.describe_image helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_google_classroom._vision import describe_image


class TestDescribeImage:
    """Tests for describe_image."""

    def test_returns_description(self) -> None:
        """Vision model response is returned as a string."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A bar chart showing student performance."
        mock_model.invoke.return_value = mock_response

        result = describe_image(
            b"\x89PNG fake image data",
            mock_model,
            prompt="Describe this image.",
            name="chart.png",
        )
        assert result == "A bar chart showing student performance."
        mock_model.invoke.assert_called_once()

    def test_list_content_joined(self) -> None:
        """When model returns list of blocks, text parts are joined."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            "First part.",
            {"type": "text", "text": "Second part."},
        ]
        mock_model.invoke.return_value = mock_response

        result = describe_image(b"\x89PNG fake", mock_model)
        assert result is not None
        assert "First part." in result
        assert "Second part." in result

    def test_returns_none_on_error(self) -> None:
        """Vision model failure returns None, not an exception."""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("API down")

        result = describe_image(b"\x89PNG fake", mock_model, name="broken.png")
        assert result is None

    def test_custom_prompt_forwarded(self) -> None:
        """Custom prompt is passed through to the HumanMessage."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "description"
        mock_model.invoke.return_value = mock_response

        describe_image(b"\x89PNG fake", mock_model, prompt="Custom prompt here.")

        # Check the message content includes our prompt
        call_args = mock_model.invoke.call_args
        messages = call_args[0][0]
        assert any("Custom prompt here." in str(block) for block in messages[0].content)
