"""
tests/test_agent.py — Unit tests for the DeepfakeInvestigationAgent.

All external calls (Anthropic, Tavily, Supabase) are fully mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Tests — DeepfakeInvestigationAgent._parse_output
# =============================================================================


class TestAgentParseOutput:
    """Tests for the static JSON parser used on agent output."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from agents.investigation_agent import DeepfakeInvestigationAgent

        self.parse = DeepfakeInvestigationAgent._parse_output

    def test_valid_json(self):
        """Valid JSON with all keys is returned as-is."""
        payload = {
            "origin_found": True,
            "confidence_assessment": "confirmed_fake",
            "sources": ["https://example.com"],
            "exif_flags": ["Software: Stable Diffusion"],
            "summary": "This is fake.",
        }
        result = self.parse(json.dumps(payload))
        assert result["origin_found"] is True
        assert result["confidence_assessment"] == "confirmed_fake"
        assert result["sources"] == ["https://example.com"]

    def test_json_with_markdown_fences(self):
        """JSON wrapped in ```json ... ``` fences is cleaned and parsed."""
        payload = {
            "origin_found": False,
            "confidence_assessment": "inconclusive",
            "sources": [],
            "exif_flags": [],
            "summary": "Could not determine.",
        }
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = self.parse(raw)
        assert result["origin_found"] is False
        assert result["confidence_assessment"] == "inconclusive"

    def test_malformed_json_returns_fallback(self):
        """If LLM returns garbage, parser returns the safe fallback dict."""
        result = self.parse("not json at all {{{ broken")
        assert result["origin_found"] is False
        assert result["confidence_assessment"] == "inconclusive"
        assert result["sources"] == []
        assert result["exif_flags"] == []
        assert "could not be completed" in result["summary"].lower()

    def test_empty_output_returns_fallback(self):
        """Empty string returns the safe fallback dict."""
        result = self.parse("")
        assert result["confidence_assessment"] == "inconclusive"

    def test_missing_keys_filled_from_fallback(self):
        """Missing keys are filled from fallback, present keys are kept."""
        partial = json.dumps(
            {
                "origin_found": True,
                "confidence_assessment": "likely_fake",
            }
        )
        result = self.parse(partial)
        assert result["origin_found"] is True
        assert result["confidence_assessment"] == "likely_fake"
        assert result["sources"] == []
        assert result["exif_flags"] == []

    def test_invalid_assessment_defaults_to_inconclusive(self):
        """Invalid confidence_assessment is replaced with 'inconclusive'."""
        payload = {
            "origin_found": True,
            "confidence_assessment": "DEFINITELY_FAKE",
            "sources": [],
            "exif_flags": [],
            "summary": "Bad enum.",
        }
        result = self.parse(json.dumps(payload))
        assert result["confidence_assessment"] == "inconclusive"


# =============================================================================
# Tests — investigate()
# =============================================================================


class TestInvestigate:
    """Tests for DeepfakeInvestigationAgent.investigate() with mocked LLM."""

    def test_agent_returns_valid_schema(self):
        """investigate() always returns a dict with all 5 required keys."""
        valid_output = json.dumps(
            {
                "origin_found": False,
                "confidence_assessment": "likely_fake",
                "sources": ["https://example.com/report"],
                "exif_flags": [],
                "summary": "Test summary.",
            }
        )

        with patch(
            "agents.investigation_agent.create_tool_calling_agent"
        ) as mock_create, patch("agents.investigation_agent.ChatAnthropic"):
            mock_executor = MagicMock()
            mock_executor.invoke.return_value = {
                "output": valid_output,
                "intermediate_steps": [],
            }
            with patch(
                "agents.investigation_agent.AgentExecutor", return_value=mock_executor
            ):
                from agents.investigation_agent import DeepfakeInvestigationAgent

                agent = DeepfakeInvestigationAgent(confidence=0.95)
                result = agent.investigate("test-job-123")

        required_keys = {
            "origin_found",
            "confidence_assessment",
            "sources",
            "exif_flags",
            "summary",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_agent_handles_parse_failure(self):
        """If LLM returns malformed JSON, agent returns safe fallback dict."""
        with patch("agents.investigation_agent.create_tool_calling_agent"), patch(
            "agents.investigation_agent.ChatAnthropic"
        ):
            mock_executor = MagicMock()
            mock_executor.invoke.return_value = {
                "output": "totally broken output {{{}",
                "intermediate_steps": [],
            }
            with patch(
                "agents.investigation_agent.AgentExecutor", return_value=mock_executor
            ):
                from agents.investigation_agent import DeepfakeInvestigationAgent

                agent = DeepfakeInvestigationAgent(confidence=0.9)
                result = agent.investigate("fail-job")

        assert result["origin_found"] is False
        assert result["confidence_assessment"] == "inconclusive"

    def test_agent_handles_executor_exception(self):
        """If the executor itself throws, we get the fallback dict."""
        with patch("agents.investigation_agent.create_tool_calling_agent"), patch(
            "agents.investigation_agent.ChatAnthropic"
        ):
            mock_executor = MagicMock()
            mock_executor.invoke.side_effect = RuntimeError("LLM exploded")
            with patch(
                "agents.investigation_agent.AgentExecutor", return_value=mock_executor
            ):
                from agents.investigation_agent import DeepfakeInvestigationAgent

                agent = DeepfakeInvestigationAgent(confidence=0.9)
                result = agent.investigate("crash-job")

        assert result["confidence_assessment"] == "inconclusive"


# =============================================================================
# Tests — run_investigation
# =============================================================================


class TestRunInvestigation:
    """Tests for the run_investigation() entry point."""

    def test_saves_to_supabase(self):
        """run_investigation persists results via retry_insert to agent_logs."""
        valid_output = json.dumps(
            {
                "origin_found": True,
                "confidence_assessment": "confirmed_fake",
                "sources": [],
                "exif_flags": [],
                "summary": "Test.",
            }
        )

        with patch("agents.investigation_agent.create_tool_calling_agent"), patch(
            "agents.investigation_agent.ChatAnthropic"
        ), patch("agents.investigation_agent.retry_insert") as mock_insert, patch(
            "agents.investigation_agent.retry_fetch"
        ):
            mock_executor = MagicMock()
            mock_executor.invoke.return_value = {
                "output": valid_output,
                "intermediate_steps": [],
            }
            with patch(
                "agents.investigation_agent.AgentExecutor", return_value=mock_executor
            ):
                from agents.investigation_agent import run_investigation

                result = run_investigation("save-test-job", {"confidence": 0.95})

        # Assert retry_insert was called once with the "agent_logs" table
        mock_insert.assert_called_once()
        call_args = mock_insert.call_args
        assert call_args[0][0] == "agent_logs"
        assert call_args[0][1]["job_id"] == "save-test-job"
