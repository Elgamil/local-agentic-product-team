# Local Agentic Software Delivery Lab

**Milestone 1 — Runtime Proof of Concept**

A local-first, model-independent agent workflow skeleton. This milestone proves
that Google ADK + LiteLLM + Ollama can run `gemma4:latest` locally and satisfy
the five acceptance checks: endpoint health, basic completion, structured output,
real tool calling, and report integrity.

---

## System requirements

| Component | Required | Verified value |
|---|---|---|
| macOS | Apple Silicon recommended | Darwin 27.0.0 (M4) |
| Python | ≥ 3.11 | 3.11.14 (`/opt/homebrew/bin/python3.11`) |
| Ollama | Running on `localhost:11434` | `ollama 0.20.0+` |
| Disk | ~10 GB free for `gemma4:latest` | 9.6 GB (Q4\_K\_M) |

---

## Model

```
gemma4:latest   c6eb396dbd59   9.6 GB
```

Capabilities reported by `ollama show gemma4:latest`:
- Architecture: `gemma4`, 8B parameters, Q4\_K\_M quantisation
- Capabilities: **completion, vision, audio, tools, thinking**
- Context length: 131 072 tokens

---

## Setup (exact commands)

```bash
# 1. Clone / enter the workspace
cd ~/Desktop/LocalAgenticSoftwareDeliveryLab

# 2. Create an isolated virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate   # or: .venv/bin/python3.11 for explicit invocation

# 3. Install all dependencies (network required)
pip install \
    "google-adk>=1.0.0" \
    "litellm>=1.40.0" \
    "mcp>=2.0.0" \
    "ollama>=0.3.0" \
    "pydantic>=2.0.0" \
    "httpx>=0.27.0" \
    "pytest>=8.0.0" \
    "pytest-asyncio>=0.23.0"

# 4. Confirm Ollama is running and gemma4 is available
ollama list   # should show gemma4:latest

# 5. (Optional) verify the endpoint directly
curl http://localhost:11434/api/tags | python3.11 -m json.tool
```

> **Note on `mcp`:** `litellm 1.106.0` unconditionally imports the `mcp` package
> in its tool-calling code path, even when MCP tools are not used. This is a
> [known litellm bug](https://github.com/BerriAI/litellm/issues). Installing `mcp`
> resolves it. This dependency is listed in `pyproject.toml`.

### Verified installed versions

```
google-adk      2.12.0
litellm         1.106.0
mcp             2.3.0
ollama          0.6.3
pydantic        2.14.1
httpx           0.28.1
pytest          9.1.1
pytest-asyncio  1.4.0
```

---

## Running the probe

```bash
# Set environment (or use defaults)
export OLLAMA_API_BASE=http://localhost:11434
export LAB_MODEL_ID=gemma4:latest

# Run all five acceptance checks
PYTHONPATH=. python -m lab.probe
```

The probe writes a JSON report to `runs/<ISO-timestamp>.json`.

### Override model or endpoint

```bash
PYTHONPATH=. python -m lab.probe --model gemma4:latest --base-url http://localhost:11434
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LAB_MODEL_ID` | `gemma4:latest` | Model identifier |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama endpoint |
| `LAB_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `LAB_LITELLM_PROVIDER` | `ollama_chat` | LiteLLM provider prefix |

---

## Running the tests

```bash
# Offline unit tests only (no Ollama required)
PYTHONPATH=. python -m pytest tests/test_schemas.py tests/test_runtime_adapter.py -v

# Live integration tests (Ollama must be running)
PYTHONPATH=. python -m pytest tests/test_probe_live.py -v -m live

# Full suite
PYTHONPATH=. python -m pytest -v
```

---

## Observed results (Milestone 1)

Run ID: `20261018T212130Z`  
Timestamp: 2026-10-18 21:21:30 UTC  
Total duration: **19 419 ms**

| # | Check | Result | Duration |
|---|---|---|---|
| 1 | Endpoint health | **PASS** | 49 ms |
| 2 | Basic completion | **PASS** | 4 177 ms |
| 3 | Structured output (ADK path) | **PASS** | 4 333 ms |
| 4 | Tool call (`agent_used_tool=True`) | **PASS** | 10 857 ms |
| 5 | Report integrity (JSON round-trip) | **PASS** | 1 ms |

Unit tests: **33 / 33 passed** (offline, no Ollama)  
Live integration tests: **5 / 5 passed**

See [`runs/example_report.json`](runs/example_report.json) for the full
serialised `ProbeResult`.

---

## Acceptance criteria status

| Criterion | Status | Evidence |
|---|---|---|
| Local Ollama endpoint responds | **PASS** | `GET /api/tags` → HTTP 200, `gemma4:latest` listed |
| Basic completion returns non-empty string | **PASS** | 194–528 chars returned across runs |
| Structured output validates against schema | **PASS** | `StructuredResponse(answer='Paris', confidence=1.0, reasoning=…)` validated by Pydantic v2 |
| Tool call: model requested the tool | **PASS** | `agent_used_tool=True`; ADK event stream shows `function_call` part before `function_response` |
| Tool call: ADK runner executed the real function | **PASS** | ADK made two LiteLLM calls (one for tool request, one for final answer); payload contains ground-truth values from the Python function |
| Report serialises and deserialises correctly | **PASS** | 1 291-byte JSON round-trips with identical field values |
| Offline unit tests pass | **PASS** | 33 / 33 |
| Live integration tests pass | **PASS** | 5 / 5 |

---

## Limitations and observations

### Gemma 4 thinking mode
`gemma4:latest` has thinking mode **on by default** (`temperature=1`). The model
emits internal reasoning text before its final answer. The `complete()` path
accumulates all text parts from the event stream including thinking output — the
visible answer appears at the end. The `complete_structured()` path uses
`_extract_json()` which finds the first valid `{…}` block regardless of position,
so thinking preamble does not break structured parsing.

### LiteLLM `mcp` import bug
`litellm 1.106.0` imports the `mcp` package unconditionally when entering the
tool-calling code path, even when MCP tools are not used. This raises
`ModuleNotFoundError: No module named 'mcp'` without the `mcp` package installed.
Workaround: install `mcp>=2.0.0`. This is a LiteLLM bug, not a Gemma or Ollama
issue. Recorded in [`runs/20261018T212046Z.json`](runs/) (the failing run) for
reference.

### ADK experimental JSON schema for function declarations
ADK 2.12.0 emits a `UserWarning` when registering plain Python function tools:
```
[EXPERIMENTAL] feature FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled.
```
This is cosmetic and does not affect functionality. Gemma 4 successfully calls the
tool despite the experimental flag.

### Tool call latency
The tool call check takes ~10 s because ADK runs two model inference passes: one
for the tool-selection decision and one for the final response after the tool
returns. Basic completion and structured output each take 1–8 s depending on
thinking depth.

### `qwen2.5:7b` not used
`qwen2.5:7b` is installed on the machine but was not used in any acceptance test.
All checks were completed with `gemma4:latest`.

### PYTHONPATH
Because the project is not installed as a package (editable install from
`pyproject.toml` fails with the venv's `setuptools`), all commands prefix
`PYTHONPATH=.` to make `lab` importable. This is the only workaround needed and
does not affect production behaviour.

---

## Repository structure

```
LocalAgenticSoftwareDeliveryLab/
├── .gitignore
├── README.md
├── pyproject.toml
│
├── lab/
│   ├── __init__.py
│   ├── probe.py                   # CLI: runs all five checks, writes run log
│   │
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract ModelRuntime interface
│   │   ├── config.py              # RuntimeConfig (env vars, no secrets)
│   │   └── adk_ollama.py          # ADK + LiteLLM → Ollama implementation
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── poc.py                 # Pydantic v2: CheckResult, StructuredResponse,
│   │                              #              ToolCallResult, ProbeResult
│   │
│   └── tools/
│       ├── __init__.py
│       └── workspace_info.py      # Permitted tool: safe workspace metadata
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Fixtures, live marker, Ollama health check
│   ├── test_schemas.py            # Unit tests — no Ollama needed
│   ├── test_runtime_adapter.py    # Unit tests — mocked ADK
│   └── test_probe_live.py         # Integration tests — @pytest.mark.live
│
└── runs/
    ├── .gitkeep
    └── example_report.json        # Sanitised sample run report (committed)
```

---

## What is deferred to later milestones

- Full specialist-agent roles (business analyst, architect, developer, test engineer)
- Multi-agent collaboration and routing
- Domain logic (e-commerce application)
- Persistent session storage
- Role instruction files (stubs in `lab/roles/` are intentionally empty)
