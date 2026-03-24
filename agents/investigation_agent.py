"""
agents/investigation_agent.py — LangChain-powered deepfake investigation agent.

When a detection is flagged as fake with high confidence, this agent
autonomously:
  1. Fetches detection details from Supabase.
  2. Reads EXIF metadata from the media to flag AI-generation software.
  3. Searches the web (Tavily) for the original source or known reports.
  4. Synthesises a structured JSON investigation report.

Custom tools
------------
exif_metadata_reader  — Downloads image from URL, extracts EXIF via Pillow.
get_detection_result  — Fetches a detection row from Supabase by job_id.

Main entry point
----------------
run_investigation(job_id, detection_payload) — called from FastAPI background tasks.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger
from PIL import Image
from PIL.ExifTags import TAGS

import config
from utils.supabase_utils import retry_fetch, retry_insert, SupabaseWriteError

# ── LangChain imports ─────────────────────────────────────────────────────────
# AgentExecutor moved between langchain versions — try multiple import paths.
try:
    from langchain.agents import AgentExecutor
except ImportError:
    from langchain.agents.agent import AgentExecutor  # type: ignore[no-redef]

try:
    from langchain.agents import create_tool_calling_agent
except ImportError:
    from langchain_core.agents import create_tool_calling_agent  # type: ignore[no-redef]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# ── Suspicious EXIF software patterns ─────────────────────────────────────────
_SUSPICIOUS_SOFTWARE = [
    "stable diffusion",
    "dall-e",
    "dall·e",
    "midjourney",
    "gan",
    "comfyui",
    "automatic1111",
    "invoke ai",
    "novelai",
    "deepdream",
]


# =============================================================================
# Tool 1 — EXIF Metadata Reader
# =============================================================================


@tool
def exif_metadata_reader(image_url: str) -> str:
    """Extracts EXIF metadata from an image URL. Returns camera model, GPS,
    software, creation date, and any suspicious fields that indicate the
    image was AI-generated (e.g. Stable Diffusion, DALL-E, Midjourney, GAN)."""
    try:
        # ── Download image ────────────────────────────────────────────────────
        resp = httpx.get(image_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    except Exception as exc:
        logger.warning("exif_metadata_reader: failed to download/open image — {}", exc)
        return f"Error: could not download or open image from URL: {exc}"

    # ── Extract EXIF ──────────────────────────────────────────────────────────
    try:
        raw_exif = img._getexif()
    except Exception:
        raw_exif = None

    if not raw_exif:
        return "No EXIF data found"

    # ── Decode tags ───────────────────────────────────────────────────────────
    decoded: dict[str, str] = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
        try:
            decoded[tag_name] = str(value)[:500]  # cap long values
        except Exception:
            decoded[tag_name] = "<unreadable>"

    # ── Flag suspicious fields ────────────────────────────────────────────────
    suspicious: list[str] = []
    for field_name, field_value in decoded.items():
        lower_val = field_value.lower()
        for pattern in _SUSPICIOUS_SOFTWARE:
            if pattern in lower_val:
                suspicious.append(
                    f"{field_name}: {field_value} [SUSPICIOUS — matches '{pattern}']"
                )

    # ── Build output string ───────────────────────────────────────────────────
    lines = ["=== EXIF Metadata ==="]
    for k, v in sorted(decoded.items()):
        lines.append(f"  {k}: {v}")

    if suspicious:
        lines.append("")
        lines.append("⚠️ SUSPICIOUS FIELDS:")
        for s in suspicious:
            lines.append(f"  • {s}")
    else:
        lines.append("")
        lines.append("No suspicious AI-generation software detected in EXIF.")

    return "\n".join(lines)


# =============================================================================
# Tool 2 — Supabase Detection Result Fetcher
# =============================================================================


@tool
def get_detection_result(job_id: str) -> str:
    """Fetches the deepfake detection result for a given job_id from the
    database. Returns is_fake, confidence, media_url, and media_type as
    a formatted string."""
    try:
        row = retry_fetch("detections", {"job_id": job_id}, single=True)
    except SupabaseWriteError as exc:
        logger.error("get_detection_result: Supabase fetch failed — {}", exc)
        return f"Error: could not fetch detection result — {exc}"

    if row is None:
        return f"No detection result found for job_id '{job_id}'."

    fields = {
        "job_id": row.get("job_id"),
        "is_fake": row.get("is_fake"),
        "confidence": row.get("confidence"),
        "media_type": row.get("media_type"),
        "media_url": row.get("media_url", row.get("heatmap_url", "N/A")),
        "status": row.get("status"),
        "created_at": row.get("created_at"),
    }

    # Include video sub-scores if present
    for key in ("vision_score", "temporal_score", "audio_score", "fused_score"):
        val = row.get(key)
        if val is not None:
            fields[key] = val

    lines = [f"=== Detection Result for {job_id} ==="]
    for k, v in fields.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# =============================================================================
# System prompt template
# =============================================================================

_SYSTEM_PROMPT_TEMPLATE = """\
You are a deepfake investigation agent. You are given a media file flagged as \
potentially fake with confidence {confidence:.0%}.

Your job:
1. Use get_detection_result to fetch full detection details
2. Use exif_metadata_reader to check image metadata for AI generation software
3. Use web_search to look for the original source of this face or scene
4. Synthesize findings into a clear investigation report

Your final response MUST be valid JSON with these exact keys:
- origin_found: boolean
- confidence_assessment: string (one of: "confirmed_fake", "likely_fake", "inconclusive", "likely_real")
- sources: list of URL strings
- exif_flags: list of suspicious EXIF field strings (empty list if none)
- summary: 2-3 sentence plain English summary for a non-technical user

Do not include any text outside the JSON."""


# ── Default fallback result ───────────────────────────────────────────────────

_FALLBACK_RESULT: dict[str, Any] = {
    "origin_found": False,
    "confidence_assessment": "inconclusive",
    "sources": [],
    "exif_flags": [],
    "summary": "Investigation could not be completed.",
}

_REQUIRED_KEYS = set(_FALLBACK_RESULT.keys())


# =============================================================================
# DeepfakeInvestigationAgent
# =============================================================================


class DeepfakeInvestigationAgent:
    """
    LangChain agent that autonomously investigates flagged deepfakes.

    Uses Claude (claude-sonnet-4-20250514) as the LLM, with three tools:
    * TavilySearchResults (web search, max 3 results)
    * exif_metadata_reader (EXIF extraction from image URL)
    * get_detection_result (Supabase row fetcher)

    Parameters
    ----------
    confidence : float
        Detection confidence score — injected into the system prompt.
    """

    def __init__(self, confidence: float = 0.0) -> None:
        os.environ.setdefault("TAVILY_API_KEY", config.TAVILY_API_KEY)

        # ── LLM ──────────────────────────────────────────────────────────────
        self._llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            max_tokens=2048,
        )

        # ── Tools ─────────────────────────────────────────────────────────────
        tavily_search = TavilySearchResults(max_results=3)
        self._tools = [tavily_search, exif_metadata_reader, get_detection_result]

        # ── Prompt ────────────────────────────────────────────────────────────
        system_text = _SYSTEM_PROMPT_TEMPLATE.format(confidence=confidence)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # ── Agent + executor ──────────────────────────────────────────────────
        agent = create_tool_calling_agent(self._llm, self._tools, prompt)
        self._executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=10,
            handle_parsing_errors=True,
        )

        self._confidence = confidence

    # ── Core investigation method ─────────────────────────────────────────────

    def investigate(self, job_id: str) -> dict[str, Any]:
        """
        Run the full investigation workflow for a flagged detection.

        Steps
        -----
        1. Invoke the LangChain agent executor with the job context.
        2. Parse the agent's final output as JSON.
        3. Validate that all required keys are present.
        4. Return the parsed result, falling back to a safe default on error.

        Parameters
        ----------
        job_id : str
            Detection job identifier to investigate.

        Returns
        -------
        dict
            Investigation report with keys: ``origin_found``,
            ``confidence_assessment``, ``sources``, ``exif_flags``, ``summary``.
        """
        user_input = (
            f"Investigate detection job '{job_id}'. "
            f"The media was flagged as fake with {self._confidence:.0%} confidence."
        )

        try:
            result = self._executor.invoke({"input": user_input})
        except Exception as exc:
            logger.error("Agent executor failed for job {}: {}", job_id, exc)
            return dict(_FALLBACK_RESULT)

        raw_output = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        logger.info(
            "Agent finished for job {} — {} tool calls, output length {}",
            job_id,
            len(steps),
            len(raw_output),
        )

        # ── Parse JSON output ─────────────────────────────────────────────────
        parsed = self._parse_output(raw_output)

        # Attach tool call metadata (not part of the agent's JSON, for our logs)
        parsed["_tool_calls_count"] = len(steps)
        parsed["_raw_tool_calls"] = self._extract_tool_calls(steps)

        return parsed

    # ── Output parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_output(raw: str) -> dict[str, Any]:
        """
        Parse the agent's final text output into a validated dict.

        Strips markdown code fences, parses JSON, and checks that every
        required key is present.  Returns the fallback result on any failure.
        """
        if not raw or not raw.strip():
            logger.warning("Agent returned empty output — using fallback result.")
            return dict(_FALLBACK_RESULT)

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Agent output is not valid JSON: {} — raw[:200]={}", exc, raw[:200]
            )
            return dict(_FALLBACK_RESULT)

        if not isinstance(parsed, dict):
            logger.warning(
                "Agent output parsed to {} instead of dict.", type(parsed).__name__
            )
            return dict(_FALLBACK_RESULT)

        # Validate required keys
        missing = _REQUIRED_KEYS - set(parsed.keys())
        if missing:
            logger.warning(
                "Agent JSON missing keys {} — using fallback for those.", missing
            )
            for key in missing:
                parsed[key] = _FALLBACK_RESULT[key]

        # Validate confidence_assessment enum
        valid_assessments = {
            "confirmed_fake",
            "likely_fake",
            "inconclusive",
            "likely_real",
        }
        if parsed.get("confidence_assessment") not in valid_assessments:
            logger.warning(
                "Invalid confidence_assessment '{}' — defaulting to 'inconclusive'.",
                parsed.get("confidence_assessment"),
            )
            parsed["confidence_assessment"] = "inconclusive"

        return parsed

    @staticmethod
    def _extract_tool_calls(steps: list) -> list[dict]:
        """Extract serialisable tool call info from intermediate_steps."""
        calls: list[dict] = []
        for action, observation in steps:
            calls.append(
                {
                    "tool": getattr(action, "tool", str(action)),
                    "tool_input": getattr(action, "tool_input", ""),
                    "output": str(observation)[:2000],
                }
            )
        return calls


# =============================================================================
# Public entry point — called from main.py background tasks
# =============================================================================


def run_investigation(job_id: str, detection_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Instantiate the investigation agent, run it, and persist results.

    This function is designed to be called as a FastAPI ``BackgroundTasks``
    callback or from a Celery worker — it is synchronous (LangChain's
    ``AgentExecutor.invoke`` is sync).

    Parameters
    ----------
    job_id             : Detection job identifier.
    detection_payload  : Dict with at least ``confidence`` (float) and
                         ``media_type`` (str).

    Returns
    -------
    dict
        The parsed investigation report.
    """
    confidence = detection_payload.get("confidence", 0.0)

    logger.info(
        "=== Investigation START — job_id={} confidence={:.4f} ===", job_id, confidence
    )
    start = time.perf_counter()

    # ── Run agent ─────────────────────────────────────────────────────────────
    agent = DeepfakeInvestigationAgent(confidence=confidence)
    report = agent.investigate(job_id)

    elapsed_s = round(time.perf_counter() - start, 2)
    tool_calls_used = report.pop("_tool_calls_count", 0)
    raw_tool_calls = report.pop("_raw_tool_calls", [])

    logger.info(
        "=== Investigation COMPLETE — job_id={} elapsed={}s tool_calls={} assessment={} ===",
        job_id,
        elapsed_s,
        tool_calls_used,
        report.get("confidence_assessment"),
    )

    # ── Persist to agent_logs table ───────────────────────────────────────────
    log_row = {
        "job_id": job_id,
        "origin_found": report.get("origin_found", False),
        "confidence_assessment": report.get("confidence_assessment", "inconclusive"),
        "sources": report.get("sources", []),
        "exif_flags": report.get("exif_flags", []),
        "summary": report.get("summary", ""),
        "tool_calls": raw_tool_calls,
        "tool_calls_count": tool_calls_used,
        "elapsed_seconds": elapsed_s,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        retry_insert("agent_logs", log_row)
        logger.info("Agent log persisted for job {}.", job_id)
    except SupabaseWriteError as exc:
        logger.error("Failed to persist agent log for job {}: {}", job_id, exc)

    return report
