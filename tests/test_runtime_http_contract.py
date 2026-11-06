"""The real ADK + LiteLLM stack against a stub Ollama HTTP server.

No model runs.  The stub returns canned JSON, but everything on our side is real,
so these tests check what the mocked tests cannot: that our options reach the wire
and that ADK's behaviour matches what the platform assumes.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lab.runtime.adk_ollama import OllamaAdkRuntime
from lab.runtime.config import RuntimeConfig
from lab.runtime.lifecycle import OllamaLifecycle
from lab.tools.developer_tools import ProjectWorkspace, build_developer_tools
from tests.fakes import ScriptedTeam, ScriptedTerminal


class StubOllama:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.chat_replies: list[dict] = []
        self.loaded: list[str] = []
        self.peak_loaded = 0
        self.reply_delay = 0.0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _reply(self, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                stub.requests.append((self.path, {}))
                if self.path == "/api/ps":
                    self._reply({"models": [{"name": name} for name in stub.loaded]})
                else:
                    self._reply({"models": [{"name": "gemma4:latest"}]})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append((self.path, body))
                if self.path == "/api/generate":
                    if body.get("keep_alive") == 0 and body["model"] in stub.loaded:
                        stub.loaded.remove(body["model"])
                    self._reply({"model": body["model"], "done": True, "response": ""})
                elif self.path == "/api/chat":
                    if body["model"] not in stub.loaded:
                        stub.loaded.append(body["model"])
                    stub.peak_loaded = max(stub.peak_loaded, len(stub.loaded))
                    time.sleep(stub.reply_delay)
                    message = stub.chat_replies.pop(0) if stub.chat_replies else {"content": ""}
                    self._reply(
                        {
                            "model": body["model"],
                            "created_at": "2026-11-05T00:00:00Z",
                            "message": {"role": "assistant", **message},
                            "done": True,
                            "done_reason": "stop",
                            "prompt_eval_count": 1,
                            "eval_count": 1,
                        }
                    )
                else:
                    self._reply({})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def chats(self) -> list[dict]:
        return [body for path, body in self.requests if path == "/api/chat"]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub(monkeypatch):
    server = StubOllama()
    monkeypatch.setenv("OLLAMA_API_BASE", server.base_url)
    yield server
    server.close()


def runtime_for(stub: StubOllama, **options) -> OllamaAdkRuntime:
    return OllamaAdkRuntime(
        RuntimeConfig(
            model_id="gemma4:latest",
            ollama_base_url=stub.base_url,
            max_output_tokens=512,
            num_ctx=8192,
            keep_alive="5m",
            request_timeout_seconds=30,
            max_llm_calls=4,
            **options,
        )
    )


async def test_memory_options_reach_ollama(stub):
    stub.chat_replies.append({"content": "hello"})
    text, _ = await runtime_for(stub).generate("You help.", "Say hello.")

    assert text == "hello"
    body = stub.chats()[0]
    assert body["keep_alive"] == "5m"
    assert body["options"]["num_ctx"] == 8192
    assert body["options"]["num_predict"] == 512
    assert body["model"] == "gemma4:latest"


async def test_disabling_thinking_reaches_ollama(stub):
    stub.chat_replies.append({"content": "hello"})
    await runtime_for(stub, think=False).generate("You help.", "Say hello.")
    assert stub.chats()[0]["think"] is False


async def test_thinking_is_left_to_ollama_when_unset(stub):
    stub.chat_replies.append({"content": "hello"})
    await runtime_for(stub).generate("You help.", "Say hello.")
    assert "think" not in stub.chats()[0]


async def test_json_mode_reaches_ollama(stub):
    stub.chat_replies.append({"content": '{"a": 1}'})
    await runtime_for(stub).generate("You help.", "Reply.", json_mode=True)
    assert stub.chats()[0].get("format") == "json"


async def test_defaults_leave_existing_callers_unchanged(stub):
    stub.chat_replies.append({"content": "hello"})
    runtime = OllamaAdkRuntime(
        RuntimeConfig(model_id="gemma4:latest", ollama_base_url=stub.base_url)
    )
    await runtime.generate("You help.", "Say hello.")
    body = stub.chats()[0]
    assert "keep_alive" not in body
    assert "num_ctx" not in body.get("options", {})
    assert body["options"]["num_predict"] == 8192


async def test_model_thinking_is_excluded_from_the_answer(stub):
    stub.chat_replies.append({"content": "the answer", "thinking": "SECRET private reasoning"})
    text, events = await runtime_for(stub).generate("You help.", "Question?")
    assert text == "the answer"
    assert "SECRET" not in text
    assert "SECRET" not in json.dumps(events)


async def test_a_product_bound_tool_is_executed_through_real_adk(stub, tmp_path):
    workspace = ProjectWorkspace(tmp_path / "product")
    stub.chat_replies.extend(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_project_file",
                            "arguments": {"relative_path": "app.py", "content": "x = 1\n"},
                        }
                    }
                ],
            },
            {"content": "Written."},
        ]
    )

    text, events = await runtime_for(stub).generate(
        "You build software.",
        "Create app.py.",
        tools=build_developer_tools(workspace),
    )

    assert (workspace.root / "app.py").read_text() == "x = 1\n"
    assert text == "Written."
    assert any(
        part.get("function_call", {}).get("name") == "write_project_file"
        for event in events
        for part in event.get("parts", [])
    )
    declared = {item["function"]["name"] for item in stub.chats()[0]["tools"]}
    assert declared == {"write_project_file", "read_project_file", "list_project_files"}


async def test_a_tool_refusal_reaches_the_model_without_killing_the_run(stub, tmp_path):
    workspace = ProjectWorkspace(tmp_path / "product")
    stub.chat_replies.extend(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_project_file",
                            "arguments": {"relative_path": "../escape.py", "content": "x"},
                        }
                    }
                ],
            },
            {"content": "Understood."},
        ]
    )

    await runtime_for(stub).generate(
        "You build software.", "Try it.", tools=build_developer_tools(workspace)
    )

    assert not (tmp_path / "escape.py").exists()
    tool_message = next(
        message
        for message in stub.chats()[1]["messages"]
        if message.get("role") == "tool" or "ERROR" in str(message.get("content"))
    )
    assert "ERROR" in str(tool_message.get("content"))


async def test_a_runaway_agent_is_stopped_by_the_llm_call_limit(stub, tmp_path):
    workspace = ProjectWorkspace(tmp_path / "product")
    loop_reply = {
        "content": "",
        "tool_calls": [{"function": {"name": "list_project_files", "arguments": {}}}],
    }
    stub.chat_replies.extend([loop_reply] * 20)

    runtime = OllamaAdkRuntime(
        RuntimeConfig(
            model_id="gemma4:latest",
            ollama_base_url=stub.base_url,
            max_llm_calls=3,
        )
    )
    with pytest.raises(Exception, match="(?i)limit|llm_calls|exceeded"):
        await runtime.generate(
            "You build software.", "Go.", tools=build_developer_tools(workspace)
        )
    assert len(stub.chats()) <= 3


async def test_unload_uses_ollamas_keep_alive_zero_and_confirms(stub):
    stub.loaded.append("gemma4:latest")
    lifecycle = OllamaLifecycle(stub.base_url, unload_wait_seconds=2)

    assert await lifecycle.running() == ["gemma4:latest"]
    assert await lifecycle.unload("gemma4:latest") is True

    request = next(body for path, body in stub.requests if path == "/api/generate")
    assert request == {"model": "gemma4:latest", "keep_alive": 0}, "no prompt, so no inference"
    assert await lifecycle.running() == []


async def test_unload_reports_when_ollama_keeps_the_model(stub):
    stub.loaded.append("gemma4:latest")
    lifecycle = OllamaLifecycle(stub.base_url, unload_wait_seconds=0)

    async def stays_loaded() -> list[str]:
        return ["gemma4:latest"]

    lifecycle.running = stays_loaded
    assert await lifecycle.unload("gemma4:latest") is False


class TestWholePlatform:
    """Session, workflow, router, lifecycle and the real runtime, all wired for real."""

    def _replies(self):
        team = ScriptedTeam()
        kinds = [
            ("discovery", "discovery-1"),
            ("brief", "brief-1"),
            ("roadmap", "roadmap-1"),
            ("step_brief", "step-01-brief-v1"),
            ("architect", "step-01-architect-v1"),
        ]
        return [{"content": json.dumps(team._default(kind, task))} for kind, task in kinds]

    async def _run(self, stub, tmp_path, monkeypatch):
        from lab.cli import build_router
        from lab.config import load_model_policy
        from lab.playbooks import PlaybookLibrary
        from lab.product.manager import ProductManager
        from lab.product.session import ProductSession

        monkeypatch.setenv("OLLAMA_API_BASE", stub.base_url)
        policy = load_model_policy()
        policy.memory.unload_wait_seconds = 2
        playbooks = PlaybookLibrary()
        manager = ProductManager(tmp_path / "products_log", playbooks)
        stub.chat_replies.extend(self._replies())
        terminal = ScriptedTerminal("1", "I want a small calculator", "yes", "yes", "yes")
        session = ProductSession(manager, build_router(policy), policy, playbooks, terminal)
        await session.run()
        return manager, terminal

    async def test_a_conversation_reaches_the_architect_on_the_right_models(
        self, stub, tmp_path, monkeypatch
    ):
        manager, terminal = await self._run(stub, tmp_path, monkeypatch)

        models = [body["model"] for body in stub.chats()]
        assert models[:5] == ["gemma4:latest"] * 4 + ["qwen2.5:7b"]
        assert any("I suggest the product name" in text for text in terminal.said)
        assert any("Proposed roadmap" in text for text in terminal.said)

    async def test_policy_options_are_applied_per_model(self, stub, tmp_path, monkeypatch):
        await self._run(stub, tmp_path, monkeypatch)
        gemma, qwen = stub.chats()[0], stub.chats()[4]
        assert gemma["options"]["num_ctx"] == 8192
        assert gemma["think"] is False
        assert qwen["options"]["num_ctx"] == 16384
        assert "think" not in qwen

    async def test_the_real_lifecycle_unloads_before_the_switch(
        self, stub, tmp_path, monkeypatch
    ):
        await self._run(stub, tmp_path, monkeypatch)
        unloads = [body for path, body in stub.requests if path == "/api/generate"]
        assert {"model": "gemma4:latest", "keep_alive": 0} in unloads
        assert stub.peak_loaded == 1, "the two models were never resident together"

    async def test_models_are_released_when_the_session_ends(self, stub, tmp_path, monkeypatch):
        await self._run(stub, tmp_path, monkeypatch)
        assert stub.loaded == []

    async def test_model_traffic_is_logged_without_reasoning(self, stub, tmp_path, monkeypatch):
        manager, _ = await self._run(stub, tmp_path, monkeypatch)
        log = (manager.root / "pocket-calc" / "logs" / "model_io.md").read_text()
        assert "## Request to gemma4:latest" in log
        assert "## Response" in log

    async def test_litellm_banners_stay_out_of_the_terminal(
        self, stub, tmp_path, monkeypatch, capsys
    ):
        import litellm

        monkeypatch.setattr(litellm, "suppress_debug_info", False)
        from lab.cli import build_router
        from lab.config import load_model_policy

        policy = load_model_policy()
        router = build_router(policy)
        router._runtime("gemma4:latest")
        assert litellm.suppress_debug_info is True
