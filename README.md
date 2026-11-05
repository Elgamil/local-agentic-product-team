# Local Agentic Software Delivery Lab

A local-first, model-independent agentic software delivery laboratory demonstrating an end-to-end multi-agent software engineering team powered by local models running via **Google ADK + LiteLLM + Ollama**.

---

## 1. System Requirements & Environment

| Component | Target Specification | Verified Environment |
|---|---|---|
| Platform | macOS (Apple Silicon M-series) | MacBook Air M4, 16 GB unified memory |
| Python | ≥ 3.11 | 3.11.14 (`.venv/bin/python`) |
| Local Inference | Ollama | Running at `http://localhost:11434` |
| Local Models | `gemma4:latest` (Default, 9.6 GB, Q4_K_M) | Verified in Ollama model list |
| Alternate Models | `qwen2.5:7b` (4.7 GB, Q4_K_M) | Verified in Ollama model list |

---

## 2. Clean Mac Setup & Installation

The project uses a standard, modern PEP 517/660 `setuptools.build_meta` build backend and is configured for clean editable installation on macOS.

```bash
# 1. Navigate to the project root directory
cd /Users/ahmedelgamil/Desktop/LocalAgenticSoftwareDeliveryLabUpdated/LocalAgenticSoftwareDeliveryLab

# 2. Activate the macOS virtual environment
source .venv/bin/activate

# 3. Install the project in editable mode
python -m pip install -e . --no-build-isolation

# 4. Verify package installation
python -c "import lab; import minishop; print('Local Agentic Lab successfully installed.')"
```

---

## 3. Milestone Overview

### Milestone 1 — Local Model Runtime PoC: COMPLETE
- Validates local runtime connectivity through Google ADK 2.12.0 + LiteLLM 1.106.0 + Ollama.
- 5/5 Acceptance Checks verified live with `gemma4:latest`:
  1. Endpoint health check
  2. Basic plain-text completion
  3. Structured output via ADK
  4. Real tool calling (`agent_used_tool=True`)
  5. Report integrity (JSON round-trip)

### Milestone 2 — Four-Agent Software Delivery & MiniShop Lite
Deterministic, sequential multi-agent software delivery workflow:
1. **Business Analyst**: Analyzes feature request; produces validated `RequirementsArtifact` with Functional Requirements (FRs), Given/When/Then Acceptance Criteria (ACs), and edge cases.
2. **Software Architect**: Translates requirements into technical design; produces validated `ArchitectureArtifact` with Architectural Decisions (ADRs), component specifications, SQLite data model, and API endpoints.
3. **Test Engineer**: Independently defines verification criteria; produces validated `TestPlanArtifact` with concrete test cases covering all acceptance criteria.
4. **Senior Developer**: Implements MiniShop Lite (`database.py`, `app.py`, `templates/`, `static/`) via controlled project tools (`write_project_file`, `read_project_file`, `list_project_files`).
5. **Authoritative Acceptance Testing**: Test runner executes pytest against the actual implementation.
6. **Bounded Repair**: If acceptance tests fail, conducts up to 2 repair iterations.
7. **Comprehensive Audit & Reporting**: Records full agent transcripts, events log, JSON report, and Markdown report.

### Milestone 3 — Evaluation & Model Comparison Benchmark
- Automated benchmark comparing models (e.g. `gemma4:latest` vs `qwen2.5:7b`) on the **price sorting** catalog task.
- Isolates each run into its own temporary workspace copy (`runs/evaluations/eval_<task>_<model>_<timestamp>/`).
- Generates a consolidated Markdown comparison report (`runs/evaluations/comparison_<task>_<timestamp>.md`).

---

## 4. MiniShop Lite Application

MiniShop Lite is a lightweight, responsive e-commerce product catalog built with FastAPI, Jinja2, and SQLite:
- Displays seeded products (name, description, formatted price, in/out of stock badges).
- Case-insensitive substring search by product name.
- User-friendly empty-state message when no products match.
- Local SQLite database persistence (`minishop/minishop.db`).
- Health check endpoint (`GET /health`).

To launch the web application manually:
```bash
python -m uvicorn minishop.app:app --host 127.0.0.1 --port 8000
```
- Catalog UI: `http://127.0.0.1:8000/`
- Search Example: `http://127.0.0.1:8000/?q=keyboard`
- Health Check: `http://127.0.0.1:8000/health`

---

## 5. Execution Guide (Terminal Commands)

### A. Run Offline Unit & Acceptance Tests (No Ollama Required)
```bash
python -m pytest -m "not live" tests -v
```
*Expected output: All offline tests PASS with 5 live tests deselected.*

### B. Run Milestone 1 Live Probe (Requires Local Ollama)
```bash
python -m lab.probe --model gemma4:latest --base-url http://localhost:11434
```
*Expected output: 5/5 PASS.*

### C. Run Milestone 2 Multi-Agent Delivery Workflow
```bash
# Default run with gemma4:latest
python -m lab.workflow.orchestrator

# Or override model / base URL
python -m lab.workflow.orchestrator --model gemma4:latest --base-url http://localhost:11434
python -m lab.workflow.orchestrator --model qwen2.5:7b --base-url http://localhost:11434
```
*Output: Saves complete run directory under `runs/<timestamp>/` containing `report.json`, `report.md`, `events.jsonl`, `transcript.md`, and `artifacts/`.*

### D. Run Milestone 3 Benchmark Evaluation
```bash
# Evaluate both models sequentially with consolidated comparison
python -m lab.evaluation.benchmark --models gemma4:latest qwen2.5:7b --repetitions 1
```
*Output: Saves isolated runs and consolidated comparison under `runs/evaluations/`.*

---

## 6. Repository Structure

```
LocalAgenticSoftwareDeliveryLab/
├── README.md
├── pyproject.toml
│
├── artifacts/                         # Active workflow artifacts
│   ├── requirements_*.json            # BA requirements
│   ├── architecture_*.json            # Architect specifications
│   ├── test_plan_*.json               # Test Engineer test plan
│   └── developer_*.json               # Developer file operations
│
├── lab/
│   ├── probe.py                       # Milestone 1 runtime probe CLI
│   ├── evaluation/
│   │   └── benchmark.py               # Milestone 3 benchmark harness
│   ├── roles/                         # Specialist agent prompt instructions
│   │   ├── __init__.py
│   │   └── prompts/
│   │       ├── business_analyst.md
│   │       ├── software_architect.md
│   │       ├── test_engineer.md
│   │       └── developer.md
│   ├── runtime/                       # Model runtime adapters
│   │   ├── base.py                    # ModelRuntime ABC
│   │   ├── config.py                  # RuntimeConfig & environment loading
│   │   └── adk_ollama.py              # Google ADK + LiteLLM + Ollama adapter
│   ├── schemas/                       # Pydantic v2 data contracts
│   │   ├── poc.py                     # Milestone 1 probe schemas
│   │   └── artifacts.py               # Milestone 2 artifact schemas
│   ├── tools/                         # Safe, bounded developer & test tools
│   │   ├── developer_tools.py         # Controlled workspace file operations
│   │   ├── test_runner.py             # Subprocess pytest runner
│   │   └── workspace_info.py          # Safe workspace metadata probe
│   └── workflow/
│       ├── __init__.py
│       ├── logger.py                  # EventLogger (events.jsonl & transcript.md)
│       └── orchestrator.py            # Workflow orchestration & reporting
│
├── minishop/                          # MiniShop Lite application
│   ├── __init__.py
│   ├── app.py                         # FastAPI web application
│   ├── database.py                    # SQLite database layer & seed data
│   ├── minishop.db                    # SQLite database file
│   ├── static/
│   │   └── style.css                  # UI styling
│   └── templates/
│       ├── base.html                  # Jinja2 layout
│       └── catalog.html               # Catalog template & search
│
├── runs/                              # Execution logs, reports & transcripts
│   └── evaluations/                   # Milestone 3 benchmark evaluations
│
└── tests/                             # Comprehensive test suite
    ├── acceptance/
    │   └── test_catalog_acceptance.py # Authoritative acceptance tests
    ├── test_evaluation_benchmark.py   # Benchmark harness unit tests
    ├── test_m2_schemas.py             # Schema & alias validation unit tests
    ├── test_m2_tools.py               # Tool boundary & safety unit tests
    ├── test_probe_live.py             # Live Ollama integration tests
    ├── test_runtime_adapter.py        # Runtime adapter unit tests
    ├── test_schemas.py                # Milestone 1 schema unit tests
    ├── test_structured_retry.py       # JSON extraction & retry unit tests
    └── test_workflow_orchestrator.py  # Orchestrator flow & diagnostics tests
```
