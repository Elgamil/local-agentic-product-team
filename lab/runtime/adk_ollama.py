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
from typing import Any, TypeVar

from pydantic import BaseModel

from lab.runtime.base import ModelRuntime
from lab.runtime.config import RuntimeConfig, default_config
from lab.schemas.poc import StructuredResponse, ToolCallResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Prevent LiteLLM from making outbound network calls to fetch remote cost maps
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


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
    """Extract a valid JSON object from text, stripping fences and handling thinking output."""
    trimmed = text.strip()

    # 1. Try direct JSON parse on stripped text
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 2. Try markdown code block extraction
    fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. String-aware balanced brace search to find valid JSON candidates
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response: {text!r}")

    depth = 0
    in_string = False
    escape = False
    block_start = -1
    candidates: list[dict] = []

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                if depth == 0:
                    block_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and block_start != -1:
                    candidate = text[block_start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    block_start = -1

    if candidates:
        # Return the candidate with the most content (largest JSON representation)
        candidates.sort(key=lambda d: len(json.dumps(d)), reverse=True)
        return candidates[0]

    # If depth > 0, the JSON braces were never closed (truncated/unbalanced output)
    if depth > 0:
        raise ValueError(
            f"Incomplete or truncated JSON object in model response (unclosed braces, depth={depth}): {text[:300]!r}"
        )

    raise ValueError(f"No valid JSON object found in model response: {text!r}")


#: Public name for the shared structured-output extractor.
extract_json = _extract_json


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
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
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
        response_format: dict | None = None,
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
        if hasattr(self, "_logger") and self._logger:
            self._logger.log_model_request(self._config.model_id, instruction, user_message)

        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types as genai_types

        model_kwargs: dict[str, Any] = {
            "model": self._config.litellm_model,
            "max_tokens": self._config.max_output_tokens,
            "num_predict": self._config.max_output_tokens,
        }
        if response_format:
            model_kwargs["response_format"] = response_format
        if self._config.num_ctx is not None:
            model_kwargs["num_ctx"] = self._config.num_ctx
        if self._config.keep_alive is not None:
            model_kwargs["keep_alive"] = self._config.keep_alive
        if self._config.think is not None:
            model_kwargs["think"] = self._config.think
        if self._config.request_timeout_seconds is not None:
            model_kwargs["timeout"] = self._config.request_timeout_seconds

        model = LiteLlm(**model_kwargs)
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

        run_kwargs: dict[str, Any] = {}
        if self._config.max_llm_calls is not None:
            from google.adk.agents.run_config import RunConfig

            run_kwargs["run_config"] = RunConfig(max_llm_calls=self._config.max_llm_calls)

        t0 = time.perf_counter()
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
            **run_kwargs,
        ):
            summary: dict = {
                "author": getattr(event, "author", None),
                "type": type(event).__name__,
            }

            if event.content:
                parts_info = []
                for part in event.content.parts or []:
                    # Thinking-capable models (gemma4) deliver reasoning as thought
                    # parts.  It is not part of the answer and must never be parsed
                    # as output or persisted to a transcript.
                    if getattr(part, "thought", False) is True:
                        continue
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

        duration_ms = (time.perf_counter() - t0) * 1000

        logger.debug("ADK events: %s", event_summaries)

        final_response_text = final_text.strip()
        if hasattr(self, "_logger") and self._logger:
            self._logger.log_model_response(final_response_text, duration_ms, event_summaries)

        return final_response_text, event_summaries

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
            response_format={"type": "json_object"},
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

    async def complete_structured_schema(
        self,
        prompt: str,
        schema: type[T],
        instruction: str | None = None,
        session_id: str = "structured_schema",
        max_retries: int = 2,
    ) -> T:
        schema_dict = schema.model_json_schema()
        schema_json = json.dumps(schema_dict, indent=2)
        required_fields = schema_dict.get("required", [])

        system_instruction = instruction or (
            "You are a precise software engineering assistant. "
            "Analyze the request and reply ONLY with a single JSON object conforming "
            "strictly to the requested JSON schema. No explanations, no markdown fences."
        )

        current_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL: Reply ONLY with a single valid JSON object adhering strictly to this JSON schema.\n"
            f"Do not include ANY commentary, planning text, or explanation before or after the JSON.\n"
            f"Begin immediately with '{{' and end with '}}'.\n\n"
            f"Required JSON Schema:\n{schema_json}"
        )

        max_attempts = max_retries + 1
        last_error: Exception | None = None
        last_text: str = ""

        for attempt in range(1, max_attempts + 1):
            curr_session = f"{session_id}_try_{attempt}" if attempt > 1 else session_id
            t0 = time.perf_counter()
            text, _ = await self._run_adk_agent(
                instruction=system_instruction,
                tools=[],
                user_message=current_prompt,
                session_id=curr_session,
                response_format={"type": "json_object"},
            )
            elapsed = (time.perf_counter() - t0) * 1000
            last_text = text
            logger.info(
                "complete_structured_schema(%s) attempt %d/%d completed in %.0f ms",
                schema.__name__,
                attempt,
                max_attempts,
                elapsed,
            )

            try:
                data = _extract_json(text)
                validated = schema.model_validate(data)
                if attempt > 1:
                    logger.info(
                        "complete_structured_schema(%s) succeeded on retry attempt %d",
                        schema.__name__,
                        attempt,
                    )
                return validated
            except Exception as exc:
                last_error = exc
                err_summary = str(exc)
                logger.warning(
                    "complete_structured_schema(%s) validation failed on attempt %d/%d: %s",
                    schema.__name__,
                    attempt,
                    max_attempts,
                    err_summary,
                )

                if hasattr(self, "_logger") and self._logger:
                    self._logger.log_validation_failed(
                        schema_name=schema.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_message=err_summary,
                        raw_snippet=text,
                    )

                if attempt < max_attempts:
                    correction = (
                        f"Your previous response could not be parsed or validated against the {schema.__name__} schema.\n\n"
                        f"Validation Error:\n{err_summary}\n\n"
                        f"Mandatory Schema Fields: {required_fields}\n\n"
                        f"Please output a COMPLETE, valid JSON object that corrects these errors and includes ALL mandatory fields. "
                        f"Do not truncate. Do not include explanatory text outside the JSON. "
                        f"Start immediately with '{{' and end with '}}'."
                    )
                    if hasattr(self, "_logger") and self._logger:
                        self._logger.log_retry_attempt(
                            schema_name=schema.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            correction_message=correction,
                        )
                    current_prompt = f"{prompt}\n\n{correction}"

        raise ValueError(
            f"Model response could not be parsed into {schema.__name__} after {max_attempts} attempts. "
            f"Final validation error: {last_error}\n"
            f"Raw text: {last_text[:500]!r}"
        ) from last_error

    async def run_agent(
        self,
        instruction: str,
        user_message: str,
        tools: list[Any] | None = None,
        session_id: str = "agent",
    ) -> tuple[str, list[dict]]:
        return await self._run_adk_agent(
            instruction=instruction,
            tools=tools or [],
            user_message=user_message,
            session_id=session_id,
        )

    async def generate(
        self,
        instruction: str,
        user_message: str,
        tools: list[Any] | None = None,
        json_mode: bool = False,
        session_id: str = "generate",
    ) -> tuple[str, list[dict]]:
        return await self._run_adk_agent(
            instruction=instruction,
            tools=tools or [],
            user_message=user_message,
            session_id=session_id,
            response_format={"type": "json_object"} if json_mode else None,
        )

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
