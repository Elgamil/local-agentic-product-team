# Local AI Product Development Team

An interactive, local-first product development team you talk to in your terminal. Describe a product to the Business Analyst, approve its roadmap, approve each step, and four specialist agents plan, build and verify it, one step at a time. Everything runs on local models through **Google ADK + LiteLLM + Ollama**, and everything is saved to disk, so a product survives a restart, a model failure, or months on pause.

```bash
python -m lab
```

```text
Hi! What product can I help you with today?

  1. Start a new product
  2. Continue an existing product
  3. List existing products
  4. Exit
```

**You are the approver.** The agents propose. Nothing is built, and no step starts, until you say so, or until you authorize a stretch of the roadmap in advance with [Autonomous Mode](#autonomous-mode). See [The Product Team](#the-product-team) for how it works. The original Milestone 1 to 3 lab (probe, benchmark, one-shot MiniShop workflow) still works and is documented further down.

---

## The Product Team

### Commands

```bash
python -m lab                       # Talk to the Business Analyst
python -m lab new "a mini shop for dairy products"
python -m lab list                  # Every product, its status and next action
python -m lab status "Dairy Corner"
python -m lab resume "Dairy Corner"
python -m lab doctor                # Check Ollama and the models, with no inference
python -m lab migrate-minishop      # Preserve the old MiniShop (safe to repeat)
```

Inside a conversation you can talk normally. These shortcuts always work:

| Command | Meaning |
|---|---|
| `/status` | Where this product stands |
| `/team` | What the specialists decided recently |
| `/products` | List your products |
| `/approve` | Approve what is on the table |
| `/reject [reason]` | Reject the proposal |
| `/change <request>` | Ask for a change |
| `/ask <question>` | Ask the team a question |
| `/retry` | Authorize another AI attempt after a blocked task |
| `/verify` | I fixed it by hand: check it and continue |
| `/report` | Show what is blocked and how to fix it |
| `/accept` | Accept a step whose tests passed but which has known limitations |
| `/done` | Finish the discussion and move on |
| `/autonomy on` | Run the rest of the roadmap without asking at each step |
| `/autonomy until <point>` | Run until a step or milestone, then stop and ask |
| `/autonomy off` | Return to Manual Mode |
| `/autonomy status` | Show the current execution mode |
| `/pause`, `/exit` | Leave. Your progress is saved |
| `/stop` | Leave, and end any autonomous run |

Approvals are read deterministically, never by a model. Only plain affirmations count: "yes, but make it blue" is a change request, not an approval, and "ok?" is a question.

An empty line never reaches a model. You are asked to type a response, and nothing is saved or changed.

When the Business Analyst suggests a product name, replies such as "yes", "keep it" or "ok for the name" accept it. A name only changes when you state one, for example `call it Milk Market` or a name in quotes. If you type just `Milk Market`, you are asked to confirm first. Anything unclear gets a clarifying question.

### Autonomous Mode

Manual Mode is the default for every product: you approve each step before it is built. Autonomous Mode changes one thing only, which is whether the workflow waits for you at the step-start gate and the proposal gate. Every agent still works in the same order, the tests are still sealed and run by Python, reviews still decide, repairs stay bounded, and model failover is unchanged.

| You type | What happens |
|---|---|
| `/autonomy on` | The team works through the remaining roadmap |
| `/autonomy until backend complete` | The team works until the Backend milestone is complete |
| `/autonomy until step 3` | The team works until Step 3 is complete |
| `/autonomy until current module` | The team works to the end of the current step's milestone |
| `/autonomy off` | Manual Mode, from the next gate |

**Checkpoints map to the roadmap or not at all.** `until` accepts a step number, a step's name, a milestone, `current step`, `next step` or `current module`. A milestone is the `milestone` the Business Analyst gave its steps in the roadmap, and it is complete when its last step is. Text that fits several steps, or that appears only in a step's description, is never guessed at: you are shown the choices and nothing starts. A checkpoint that leaves the roadmap after a change returns the product to Manual Mode.

**You stay in control during a run.** While the team works you can type `/status`, `/team`, `/report`, `/pause`, `/stop`, `/exit`, `/autonomy off` or `/autonomy until <point>`. Nothing interrupts a running model call or file write. A mode change is saved straight away and read at the next gate, and a pause takes effect at the next safe point, which includes the gap between two planned files. A product you paused, or stopped with Ctrl-C, waits for your word when you reopen it, whereas one that crashed picks up by itself.

**Some things always need you.** Autonomy never approves the name or the roadmap: `/autonomy on` at the roadmap gate arms the mode, and it starts once you approve. Anything that blocks stops the run. A proposal you changed or rejected comes back to you, because you authorized what the team would propose, not a new direction. Revised tests are only ever approved by a person.

**The audit trail is the same.** Each proposal and agent output is kept exactly as in Manual Mode. Your authorization is recorded in `logs/approvals.jsonl`, and each gate the policy passes is recorded as `decided_by: autonomy_policy`, citing that authorization. Implementation refuses to start if the authorization it cites does not exist.

The mode belongs to the product, lives in `state.json` and survives restarts and crashes. `/status` always shows it. It ends by itself at its checkpoint, and when the roadmap is complete.

### When the team gets stuck

Each task still has its four model attempts. When they run out, or tests keep failing, or something else blocks the work, the product enters a persistent **BLOCKED / HUMAN INTERVENTION REQUIRED** state instead of simply stopping. Nothing is reset: files, failed model outputs, the attempt ledger and the sealed tests all stay as they were, and you can close the application.

The report is at `products_log/<slug>/interventions/INT-001/report.md`, and `/report` shows a summary. No model writes it, because the models have just failed and a report must not invent a cause. It covers the context, the exact problem (files, errors, test names, expected against actual, traceback, and the acceptance criteria involved), a root-cause section that keeps confirmed facts, the agents' opinions and unconfirmed possibilities strictly apart, and what to do about it.

To recover:

1. Read the report and fix the problem in your editor.
2. Reopen the product with `python -m lab resume "<name>"`.
3. Type `/verify`.

`/verify` finds what you changed by comparing hashes with the failure checkpoint, records it, and runs the same authoritative acceptance and regression tests. It uses no model attempt and resets no budget.

* **If they pass**, the workflow continues from its saved checkpoint: the Test Engineer assesses, the Architect reviews, and the Business Analyst reports. Autonomous runs carry on to their checkpoint.
* **If they fail**, you get an updated report, your changes stay, and nothing else happens. No model repair starts unless you say `/retry`.

A task that failed before the tests were sealed is fixed the same way: write the file it could not produce, and `/verify` holds it to the task's own schema and rules.

**Your fix cannot weaken the acceptance criteria.** Changing the product code is always allowed. A sealed test is read-only, so to correct one you first run `chmod u+w <file>`. `/verify` then shows you the differences and the old and new hashes, and nothing runs against the edited suite until you approve it, at which point it is sealed as a new revision. If the edit removes tests, removes assertions or adds skips, a plain yes is refused: approve with `/approve <reason>`, which is recorded, or `/reject` to restore the sealed tests, which keeps your version beside the report. A test correction may change how a criterion is tested but not what it requires. Editing an approved document, or the seal's manifest, is refused outright. To change a requirement, use `/change`, which produces a new proposal that needs your approval even in Autonomous Mode.

### The conversation display

Every message has a speaker label, so the conversation stays readable without color.

```text
Business Analyst:
I understand that we're building a browser-based Tic-Tac-Toe game.

You:
I want a clean, modern interface.
```

| Output | Color | Label |
|---|---|---|
| Agents | Cyan | `Business Analyst:`, `Software Architect:`, `Developer:`, `Test Engineer:` |
| You | Green, including what you type | `You:` |
| Warnings | Yellow | `Warning:` |
| Errors | Red | `Error:` |
| System status | Faint | `  …` |

Color uses standard ANSI codes and turns off automatically when output is redirected, when `TERM` is `dumb` or unset, and when the [`NO_COLOR`](https://no-color.org) environment variable is set:

```bash
NO_COLOR=1 python -m lab
```

### Progress while you wait

Local models can take minutes to answer, so every long operation says what it is doing. On a live terminal the "still waiting" line updates in place about every 10 seconds:

```text
  … Developer — Repair 1 of 2
  … Model: qwen2.5:7b
  … Waiting for model response... (time limit: 15 minutes)
  … Still waiting for model response — 20 seconds elapsed.
```

When the wait ends, that line is replaced by the outcome, such as `Model response received — 43 seconds.`

These messages report only what is known. There are no percentages, because a local model gives no sign of progress until its answer arrives. "Still waiting" means the time limit has not been reached and no answer has come back, and the time limit, retries and failover work exactly as they would with no display attached.

| What is happening | What you see |
|---|---|
| Waiting for a model | `Waiting for model response...` |
| Running tests | `Running test process...` and `Test process still running` |
| Checking an answer | `Still processing the response` (only if it takes a while) |
| Retrying | `Warning: ... Trying once more.` and `(attempt 2 of 4)` |
| Switching models | `Warning: ... switching to the backup model.` |
| Hitting a time limit | `Warning: Timed out: ...` |
| A failed attempt | `Error: Failed after 12 seconds: ...` |

When output is redirected, or on a dumb terminal, heartbeats become timestamped log lines every 30 seconds instead, so each line makes sense on its own in a log:

```text
[14:03:22] Developer — Repair 1 of 2 (qwen2.5:7b): Still waiting for model response — 30 seconds elapsed.
```

Set `LAB_HEARTBEAT_SECONDS` to change the interval. Heartbeats are display only: they are not written to the audit log.

### How a product moves

```
Discovery → Brief → [name] → Roadmap → [roadmap approval] → for each step:

  [start step]
  A  Business Analyst prepares the step
  B  Architect designs → Developer plans → Test Engineer plans and writes the tests
  C  Business Analyst consolidates a proposal
     [proposal approval]  ← approve, change, ask, or reject. Nothing is built before this.
  D  Developer implements through journaled file tools
  E  Python runs the sealed tests → Test Engineer assesses → Architect reviews
     Failures go to bounded Developer repair, with the real test output
  F  Business Analyst reports; Python decides from the evidence
```

Brackets are human gates. The workflow cannot leave one without your decision.

A step completes only if Python confirms all of the following: the sealed tests ran and passed with at least one real passing test, the tests are byte-identical to what you approved, the code on disk is the code that was tested, the Test Engineer and Architect did not object, and regression suites from earlier steps still pass. The Developer saying it worked counts for nothing, and the Business Analyst can hold a step back but cannot push one through.

### Who decides what

Agents return validated artifacts and hold no reference to the workflow, so they cannot change state, approve anything, or choose a file location.

| | Decided by |
|---|---|
| Workflow phase and transitions | Python, against `ALLOWED_TRANSITIONS` |
| Approvals | You, recorded with the hash of what you approved |
| Folder names | Python (`slugify`); a model only suggests a display name |
| Test commands | The allow-list in `lab/config/verification_profiles.yaml`; a model picks a profile id |
| Whether a step is complete | Python, from recorded evidence |

### Products on disk

The filesystem is the source of truth. `products_log/` is gitignored, so products stay on this machine.

```
products_log/<product-slug>/
├── product.json            Metadata
├── state.json              The checkpoint (state.prev.json is the one before)
├── brief.md  roadmap.md  decisions.md
├── workspace/              Product source. The only place the Developer can write
├── verification/<step>/    Sealed authoritative tests and their hash manifest
├── artifacts/              Validated product artifacts; every roadmap version
├── interventions/INT-001/  A blocked task: report.md (the latest), report.json,
│                           earlier versions, baseline hashes, your recorded changes
├── steps/<step-id>/
│   ├── brief.md  architect.md  developer.md  test_plan.md  proposal.md
│   ├── implementation.md  review.md  test_results.md  closure.md
│   ├── proposals/v1/, v2/  Every proposal, with its tests, kept for audit
│   ├── test_runs/          Actual commands and output
│   ├── handoffs.jsonl  transcript.md  events.jsonl  file_ops.jsonl
└── logs/
    ├── events.jsonl  approvals.jsonl  handoffs.jsonl  conversation.jsonl
    ├── model_io.md         Model traffic, with any reasoning removed
    └── tasks/<task-id>/    Exact prompt of each attempt, attempt log, result
```

### Model routing and failover

`lab/config/model_policy.yaml` is the single source for routing:

| Role | Preferred | Fallback |
|---|---|---|
| Business Analyst | `gemma4:latest` | `qwen2.5:7b` |
| Software Architect | `qwen2.5:7b` | `gemma4:latest` |
| Developer | `qwen2.5:7b` | `gemma4:latest` |
| Test Engineer | Opposite of the Developer's model | The Developer's model |

For each logical task: two attempts on the preferred model, then two on the alternate, four at most. Retries include corrective feedback. A switched model gets a prompt rebuilt from the last valid saved context plus exact diagnostics, and never sees the failed output. Filesystem and dependency failures stop immediately, because another model cannot fix them. When both models are exhausted the product persists as BLOCKED and tells you why. Attempt budgets are saved, so a restart cannot reset them.

### Memory on a 16 GB Mac

Only one inference runs at a time. The resident model is unloaded (`keep_alive: 0`, then confirmed through `/api/ps`) before a different one loads, models left by a crashed session are released at startup, and everything is released on exit. `num_ctx` is set per model, prompts are sized to the receiving model's window, and context comes from bounded sections of saved files rather than whole transcripts.

For a server-side guarantee as well, start Ollama with `OLLAMA_MAX_LOADED_MODELS=1`.

### Crash recovery

State is written atomically after every meaningful transition, and history is append-only. A task's result is recorded only once its artifact is safely on disk, so on restart:

- Finished work is recognised and reused, never repeated.
- An interrupted task resumes from its saved inputs. A crash does not use up its attempts.
- A pending approval is asked again from saved files, with no agent work.
- Developer writes use a write-ahead journal with hashes. Resume confirms what landed, redoes what did not, and reports any file changed outside the platform without overwriting it.

### Adding to the platform

- **A product category:** drop a Markdown file in `lab/playbooks/` with the same sections as the others. No code changes.
- **A test stack:** add a profile to `lab/config/verification_profiles.yaml`, or to `products_log/<slug>/verification/profiles.yaml` for one product.

### Limitations

- **Not yet proven against real models.** Every test uses scripted model responses or a stub server. Whether `gemma4` and `qwen2.5` produce good results through this workflow is unverified until you run it.
- **`think: false` for gemma4 is unverified live.** It is confirmed to reach Ollama, not confirmed against the real model. If gemma4 misbehaves, remove that line from `model_policy.yaml`.
- **Tests are not sandboxed.** The runner uses no shell, a scrubbed environment, timeouts, process-group kill and bounded output, but the tests are still LLM-written code running as you. Read what you approve.
- **No dependency installation.** Products are limited to what is already installed. The Architect is told so, and names anything that would need your help.
- **Sealed regression tests are strict.** If a later step legitimately changes earlier behaviour, its regression failure needs a decision from you, because the Developer cannot change those tests.
- **Mini Shop Lite is read-only** in the product list. It has no roadmap to continue.

---

# The Original Lab (Milestones 1 to 3)

Everything below describes the original lab, which still works. Its one-shot workflow targets the `minishop/` package directly. A copy of MiniShop is also preserved as a historical product at `products_log/mini-shop-lite/`.

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
cd /path/to/LocalAgenticSoftwareDeliveryLab

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
*Expected output: All offline tests PASS with 5 live tests deselected. No Ollama inference runs.*

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
├── products_log/                      # Your products (gitignored)
│
├── lab/
│   ├── __main__.py  cli.py            # python -m lab
│   ├── config/                        # model_policy.yaml, verification_profiles.yaml
│   ├── playbooks/                     # One Markdown file per product category
│   ├── product/                       # The product team platform
│   │   ├── workflow.py                # Deterministic workflow and human gates
│   │   ├── session.py                 # Interactive Business Analyst terminal
│   │   ├── autonomy.py                # Execution modes and checkpoints, deterministic
│   │   ├── intervention.py            # Failure diagnosis and intervention reports
│   │   ├── console.py                 # Speaker labels, colors and no-color fallback
│   │   ├── progress.py                # What is shown during long operations
│   │   ├── decisions.py               # Deterministic reading of human input
│   │   ├── manager.py                 # Creating, naming, discovering products
│   │   ├── store.py                   # Checkpoints, events, attempt log, lock
│   │   ├── context.py                 # Bounded prompts and failover handover
│   │   ├── objectives.py              # Task text for each agent
│   │   ├── render.py                  # Markdown for every artifact
│   │   ├── presentation.py            # Plain-language status wording
│   │   └── migration.py               # Preserving the original MiniShop
│   ├── probe.py                       # Milestone 1 runtime probe CLI
│   ├── evaluation/
│   │   └── benchmark.py               # Milestone 3 benchmark harness
│   ├── roles/                         # Specialist agent prompt instructions
│   │   ├── __init__.py
│   │   └── prompts/
│   │       ├── team/                  # Product team role prompts
│   │       ├── business_analyst.md
│   │       ├── software_architect.md
│   │       ├── test_engineer.md
│   │       └── developer.md
│   ├── runtime/                       # Model runtime adapters
│   │   ├── base.py                    # ModelRuntime ABC
│   │   ├── config.py                  # RuntimeConfig & environment loading
│   │   ├── adk_ollama.py              # Google ADK + LiteLLM + Ollama adapter
│   │   ├── router.py                  # Routing, retry, failover, sequential execution
│   │   ├── heartbeat.py               # Background ticker for blocking operations
│   │   └── lifecycle.py               # Ollama model unloading
│   ├── schemas/                       # Pydantic v2 data contracts
│   │   ├── poc.py                     # Milestone 1 probe schemas
│   │   ├── artifacts.py               # Milestone 2 artifact schemas
│   │   └── product.py                 # Product team contracts and phase machine
│   ├── tools/                         # Safe, bounded developer & test tools
│   │   ├── developer_tools.py         # Controlled workspace file operations
│   │   ├── journal.py                 # Write-ahead journal for Developer writes
│   │   ├── verification.py            # Sealed tests and profile-based runner
│   │   ├── fsutil.py                  # Atomic writes and durable appends
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
    ├── fakes.py                       # Scripted models; everything else is real
    ├── test_product_end_to_end.py     # Whole products through the real workflow
    ├── test_workflow_gates.py         # Approval gates, repair, evidence, blocking
    ├── test_workflow_recovery.py      # Crashes at each phase, real process kill
    ├── test_model_router.py           # Retry, failover, memory policy
    ├── test_context_reconstruction.py # Prompts, budgets, failover handover
    ├── test_workspace_isolation.py    # Boundaries, sealing, tamper detection
    ├── test_verification_runner.py    # Real pytest, unittest and Node runs
    ├── test_runtime_http_contract.py  # Real ADK + LiteLLM against a stub server
    ├── test_product_manager.py        # Creating, naming, listing, selecting
    ├── test_product_store.py          # Atomic writes, events, locking
    ├── test_playbooks_and_policy.py   # Playbooks and configuration
    ├── test_session_commands.py       # Input parsing and the terminal
    ├── test_terminal_ux.py            # Labels, colors, empty input, resume
    ├── test_progress_heartbeat.py     # Heartbeats, timeouts, interruption, logs
    ├── test_name_confirmation.py      # Accepting, changing and clarifying names
    ├── test_minishop_migration.py     # Preserving the original MiniShop
    ├── test_evaluation_benchmark.py   # Benchmark harness unit tests
    ├── test_m2_schemas.py             # Schema & alias validation unit tests
    ├── test_m2_tools.py               # Tool boundary & safety unit tests
    ├── test_probe_live.py             # Live Ollama integration tests
    ├── test_runtime_adapter.py        # Runtime adapter unit tests
    ├── test_schemas.py                # Milestone 1 schema unit tests
    ├── test_structured_retry.py       # JSON extraction & retry unit tests
    └── test_workflow_orchestrator.py  # Orchestrator flow & diagnostics tests
```
