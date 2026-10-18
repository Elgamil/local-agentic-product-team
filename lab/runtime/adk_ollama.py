"""ADK + LiteLLM → Ollama runtime implementation.

This module is the only place that knows about google-adk and litellm.
All other lab code interacts with the ModelRuntime ABC only.

Tested against: google-adk 2.12.0, litellm 1.106.0
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from lab.runtime.base import ModelRuntime
from lab.runtime.config import RuntimeConfig, default_config
from lab.schemas.poc import StructuredResponse, ToolCallResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured-output prompt helpers
# ---------------------------------------------------------------------------

_STRUCTURED_SYSTEM = (
    "You are a precise assistant. "
    "Always reply with a single JSON object and nothing else — no markdown fences, "
    "no commentary, no explanation outside the JSON. "
    "The object must contain exactly three keys: "
    '"answer" (string), "confidence" (float between 0 and 1), "reasoning" (string).'
)

_STRUCTURED_PROMPT_TEMPLATE = (
    "{user_prompt}\n\n"
    "Reply ONLY with a JSON object matching this schema:\n"
    '{{"answer": "<your answer>", "confidence": <0.0-1.0>, "reasoning": "<brief reasoning>"}}'
)


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from text, stripping markdown fences."""
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = stripped.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response: {text!r}")
    depth = 0
    for i, ch in enumerate(stripped[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : i + 1])
    raise ValueError(f"Unbalanced JSON in model response: {text!r}")


# ---------------------------------------------------------------------------
# Main runtime class
# ---------------------------------------------------------------------------


class OllamaAdkRuntime(ModelRuntime):
    """Concrete ModelRuntime using Google ADK → LiteLLM → Ollama.

    ADK version: 2.12.0
    LiteLLM model string: ollama_chat/<model_id>
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self._config = config or default_config()
        # Set env var that LiteLLM reads for the Ollama base URL
        os.environ.setdefault("OLLAMA_API_BASE", self._config.ollama_base_url)
        logger.info(
            "OllamaAdkRuntime initialised: model=%s  litellm_model=%s",
            self._config.model_id,
            self._config.litellm_model,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def runtime_label(self) -> str:
        return "google-adk+litellm+ollama"

    # ------------------------------------------------------------------
    # Internal: create an ADK agent + runner and stream one turn
    # ------------------------------------------------------------------

    async def _run_adk_agent(
        self,
        instruction: str,
        tools: list[Any],
        user_message: str,
        session_id: str = "probe",
    ) -> tuple[str, list[dict]]:
        """Run one agent turn and return (final_text, event_summaries).

        Uses ADK 2.x API:
          - LlmAgent(name, model, instruction, tools)
          - Runner(agent, app_name, session_service)
          - InMemorySessionService
          - runner.run_async(user_id, session_id, new_message)

        Events are google.adk.events.Event instances with .content (genai Content),
        .author, .actions, etc.
        """
        from google.adk.agents import LlmAgent
        from google.adk.events import Event
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types as genai_types

        model = LiteLlm(model=self._config.litellm_model)
        agent = LlmAgent(
            name="lab_probe_agent",
            model=model,
            instruction=instruction,
            tools=tools,
        )

        app_name = "lab_probe"
        user_id = "probe_user"
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

        runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        )

        final_text = ""
        event_summaries: list[dict] = []

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            summary: dict = {
                "author": getattr(event, "author", None),
                "type": type(event).__name__,
            }

            if event.content:
                parts_info = []
                for part in event.content.parts:
                    part_d: dict = {}
                    if hasattr(part, "text") and part.text:
                        final_text += part.text
                        part_d["text_snippet"] = part.text[:100]
                    if hasattr(part, "function_call") and part.function_call:
                        part_d["function_call"] = {
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args or {}),
                        }
                    if hasattr(part, "function_response") and part.function_response:
                        part_d["function_response"] = {
                            "name": part.function_response.name,
                        }
                    if part_d:
                        parts_info.append(part_d)
                if parts_info:
                    summary["parts"] = parts_info

            event_summaries.append(summary)

        logger.debug("ADK events: %s", event_summaries)
        return final_text.strip(), event_summaries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, prompt: str) -> str:
        t0 = time.perf_counter()
        text, _ = await self._run_adk_agent(
            instruction="You are a helpful assistant. Answer concisely.",
            tools=[],
            user_message=prompt,
            session_id="complete",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("complete() finished in %.0f ms, chars=%d", elapsed, len(text))
        if not text:
            raise RuntimeError("Model returned an empty response for plain completion.")
        return text

    async def complete_structured(self, prompt: str) -> StructuredResponse:
        user_msg = _STRUCTURED_PROMPT_TEMPLATE.format(user_prompt=prompt)
        t0 = time.perf_counter()
        text, _ = await self._run_adk_agent(
            instruction=_STRUCTURED_SYSTEM,
            tools=[],
            user_message=user_msg,
            session_id="structured",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("complete_structured() %.0f ms | raw: %.120s", elapsed, text)

        try:
            data = _extract_json(text)
            return StructuredResponse(**data)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"Model response could not be parsed into StructuredResponse: {exc}\n"
                f"Raw text: {text!r}"
            ) from exc

    async def call_tool(
        self,
        tool_fn: Any,
        prompt: str,
    ) -> ToolCallResult:
        tool_name = getattr(tool_fn, "__name__", str(tool_fn))
        t0 = time.perf_counter()

        text, events = await self._run_adk_agent(
            instruction=(
                f"You are a helpful assistant with access to the '{tool_name}' tool. "
                f"When the user asks for workspace information, you MUST call the "
                f"'{tool_name}' tool to obtain the answer. "
                "Do not make up or guess values. Call the tool."
            ),
            tools=[tool_fn],
            user_message=prompt,
            session_id="tool_call",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("call_tool() finished in %.0f ms", elapsed)
        logger.debug("tool call events: %s", events)

        # Determine whether the model itself requested the tool
        # by inspecting the event stream for a function_call part
        agent_called_tool = any(
            any(
                "function_call" in part and part["function_call"].get("name") == tool_name
                for part in evt.get("parts", [])
            )
            for evt in events
        )

        # Obtain ground-truth payload: call the real function to get its return value.
        # This is NOT the invocation that counts as the tool call (that was the ADK runner's).
        # We call it here only to have a validated payload to attach to the result record.
        payload: dict = {}
        if agent_called_tool:
            try:
                payload = tool_fn()
            except Exception as exc:
                payload = {"error": str(exc)}

        return ToolCallResult(
            tool_name=tool_name,
            success=agent_called_tool,
            payload=payload,
            agent_used_tool=agent_called_tool,
        )
