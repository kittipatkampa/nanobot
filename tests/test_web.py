"""Tests for web tools: web_search and web_fetch.

Run: pytest tests/test_web.py -v
(Integration test for Winter Olympics 2026 requires BRAVE_API_KEY or tools.web.search.apiKey.)
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.web import (
    WebFetchTool,
    WebSearchTool,
    _normalize,
    _strip_tags,
    _validate_url,
)


class TestValidateUrl:
    """Test _validate_url helper."""

    def test_valid_https(self):
        assert _validate_url("https://example.com") == (True, "")
        assert _validate_url("https://example.com/path?q=1") == (True, "")

    def test_valid_http(self):
        assert _validate_url("http://example.com") == (True, "")

    def test_invalid_scheme(self):
        valid, msg = _validate_url("ftp://example.com")
        assert not valid
        assert "ftp" in msg or "http" in msg

    def test_invalid_no_domain(self):
        valid, msg = _validate_url("https://")
        assert not valid
        assert "domain" in msg.lower() or "Missing" in msg


class TestStripTags:
    """Test _strip_tags helper."""

    def test_removes_simple_tags(self):
        assert _strip_tags("<p>hello</p>") == "hello"

    def test_removes_script_tags(self):
        assert _strip_tags("<script>alert(1)</script>foo") == "foo"

    def test_unescapes_entities(self):
        assert _strip_tags("&amp; &lt;test&gt;") == "& <test>"


class TestNormalize:
    """Test _normalize helper."""

    def test_collapses_spaces(self):
        assert _normalize("a   b   c") == "a b c"

    def test_collapses_newlines(self):
        assert _normalize("a\n\n\n\nb") == "a\n\nb"


class TestWebSearchTool:
    """Test WebSearchTool."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_error(self):
        """Without API key, returns error message."""
        tool = WebSearchTool(api_key="")
        result = await tool.execute(query="test")
        assert "BRAVE_API_KEY" in result
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_returns_formatted_results(self):
        """With mocked API, returns formatted search results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "description": "Description one",
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                    },
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_context):
            tool = WebSearchTool(api_key="test-key")
            result = await tool.execute(query="python", count=2)

        assert "Results for: python" in result
        assert "Test Result 1" in result
        assert "https://example.com/1" in result
        assert "Description one" in result
        assert "Test Result 2" in result
        assert "https://example.com/2" in result

    @pytest.mark.asyncio
    async def test_execute_empty_results(self):
        """Empty API results return no-results message."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_context):
            tool = WebSearchTool(api_key="test-key")
            result = await tool.execute(query="xyznonexistent123")

        assert "No results for:" in result

    @pytest.mark.asyncio
    async def test_execute_api_error_returns_error_message(self):
        """API errors are caught and returned as error message."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_context):
            tool = WebSearchTool(api_key="test-key")
            result = await tool.execute(query="test")

        assert result.startswith("Error:")
        assert "Connection failed" in result

    @pytest.mark.asyncio
    async def test_execute_uses_env_api_key_when_passed_none(self):
        """When api_key is None, falls back to BRAVE_API_KEY env."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": [{"title": "OK", "url": "https://x.com"}]}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_context):
            with patch.dict("os.environ", {"BRAVE_API_KEY": "env-key"}):
                tool = WebSearchTool(api_key=None)
                result = await tool.execute(query="test")

        assert "OK" in result
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        assert call_kwargs["headers"]["X-Subscription-Token"] == "env-key"

    @pytest.mark.asyncio
    async def test_web_search_winter_olympics_2026_host(self):
        """Integration test: search for 2026 Winter Olympics host returns Italy."""
        from nanobot.config import load_config

        config = load_config()
        api_key = config.tools.web.search.api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            pytest.skip("BRAVE_API_KEY or tools.web.search.apiKey not configured")

        tool = WebSearchTool(api_key=api_key)
        result = await tool.execute(
            query="What country host winter olympic games in 2026?",
            count=5,
        )

        assert "Results for:" in result
        assert "Italy" in result
        assert "Error" not in result


class TestWebFetchTool:
    """Test WebFetchTool."""

    @pytest.mark.asyncio
    async def test_invalid_url_returns_validation_error(self):
        """Invalid URL returns validation error."""
        tool = WebFetchTool()
        result = await tool.execute(url="ftp://example.com")
        data = json.loads(result)
        assert "error" in data
        assert "validation" in data["error"].lower() or "url" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_url_missing_domain(self):
        """URL without domain returns validation error."""
        tool = WebFetchTool()
        result = await tool.execute(url="https://")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_fetch_json_content(self):
        """Fetches JSON URL and returns parsed content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://api.example.com/data"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"key": "value"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_context):
            tool = WebFetchTool()
            result = await tool.execute(url="https://api.example.com/data")

        data = json.loads(result)
        assert data["url"] == "https://api.example.com/data"
        assert data["status"] == 200
        assert data["extractor"] == "json"
        assert '"key": "value"' in data["text"]

    @pytest.mark.asyncio
    async def test_fetch_network_error_returns_error(self):
        """Network errors are caught and returned."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_context):
            tool = WebFetchTool()
            result = await tool.execute(url="https://example.com")

        data = json.loads(result)
        assert "error" in data
        assert "Connection refused" in data["error"]
