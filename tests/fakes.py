"""Scripted stand-ins for the local models, with no inference anywhere.

Only the models are fake.  File writes, journals, checkpoints, sealing and test
execution are all real, so these tests exercise the actual machinery.  They prove
how the platform behaves, and say nothing about how good a real model's output is.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lab.config import ModelPolicy, ModelSpec, load_model_policy
from lab.playbooks import PlaybookLibrary
from lab.product.decisions import ParsedInput  # noqa: F401  (re-exported for tests)
from lab.product.manager import ProductManager
from lab.product.store import ProductStore
from lab.product.workflow import HumanDecision, ProductWorkflow
from lab.runtime.base import ModelRuntime
from lab.runtime.router import ModelRouter
from lab.schemas.poc import StructuredResponse, ToolCallResult
from lab.schemas.product import ProductPhase

GEMMA = "gemma4:latest"
QWEN = "qwen2.5:7b"

CALC_SOURCE = "def add(a, b):\n    return a + b\n"
CALC_BUGGY = "def add(a, b):\n    return a - b\n"
STATS_SOURCE = "def mean(values):\n    return sum(values) / len(values)\n"
CALC_TEST = (
    "import calc\n\n\n"
    "def test_add_positive():\n    assert calc.add(2, 3) == 5\n\n\n"
    "def test_add_negative_numbers():\n    assert calc.add(-2, -3) == -5\n"
)
STATS_TEST = (
    "import stats\n\n\n"
    "def test_mean():\n    assert stats.mean([1, 2, 3]) == 2\n\n\n"
    "def test_mean_of_one():\n    assert stats.mean([4]) == 4\n"
)


class SimulatedPowerLoss(BaseException):
    """Stands in for the Mac shutting down.

    It derives from BaseException so nothing can catch it as an ordinary failure:
    the process simply stops, and only what is already on disk survives.
    """


class Text:
    """Return this text from the model instead of the scripted default."""

    def __init__(self, text: str) -> None:
        self.text = text


class Fail:
    """Raise this exception from the model call."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class Silent:
    """Answer without calling any tool, as a model that only describes code would."""


_KINDS = (
    (r"-route-change-", "route_modification"),
    (r"-question-", "team_question"),
    (r"^roadmap-question-", "team_question"),
    (r"^discovery-", "discovery"),
    (r"^brief-", "brief"),
    (r"^roadmap-", "roadmap"),
    (r"-brief-v", "step_brief"),
    (r"-architect-v", "architect"),
    (r"-developer-plan-v", "developer_plan"),
    (r"-test-plan-v", "test_plan"),
    (r"-test-file-v", "test_file"),
    (r"-proposal-v", "proposal"),
    (r"-impl-v", "implement"),
    (r"-assess-v", "assess"),
    (r"-review-v", "review"),
    (r"-repair-", "repair"),
    (r"-closure-v", "closure"),
)


def kind_of(session_id: str) -> str:
    for pattern, kind in _KINDS:
        if re.search(pattern, session_id):
            return kind
    raise AssertionError(f"Unrecognised task: {session_id}")


@dataclass
class Call:
    model: str
    kind: str
    task_id: str
    instruction: str
    prompt: str
    tools: list[str]
    json_mode: bool


@dataclass
class ScriptedTeam:
    """Both local models, scripted.  Unscripted calls get a valid default answer."""

    calls: list[Call] = field(default_factory=list)
    scripts: dict[str, list[Any]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=lambda: {"calc.py": CALC_SOURCE})
    repaired_sources: dict[str, str] = field(default_factory=lambda: {"calc.py": CALC_SOURCE})
    discovery_ready: bool = True
    product_name: str = "Pocket Calc"
    active: int = 0
    max_active: int = 0
    delay: float = 0.0
    lifecycle: "FakeLifecycle | None" = None
    max_resident: int = 0

    def script(self, kind: str, *responses: Any) -> None:
        self.scripts.setdefault(kind, []).extend(responses)

    def kinds(self) -> list[str]:
        return [call.kind for call in self.calls]

    def calls_of(self, kind: str) -> list[Call]:
        return [call for call in self.calls if call.kind == kind]

    def models_for(self, kind: str) -> list[str]:
        return [call.model for call in self.calls_of(kind)]

    def runtime_factory(self, spec: ModelSpec) -> "ScriptedRuntime":
        return ScriptedRuntime(spec.id, self)

    # ------------------------------------------------------------------
    # Default artifacts for a small two-step product
    # ------------------------------------------------------------------

    @staticmethod
    def _number(task_id: str) -> int:
        match = re.match(r"step-(\d+)", task_id)
        return int(match.group(1)) if match else 1

    def _default(self, kind: str, task_id: str) -> dict[str, Any]:
        number = self._number(task_id)
        module = "calc" if number == 1 else "stats"
        return {
            "discovery": {
                "reply": "Thanks, that is clear. Who will use it?",
                "understanding": "A small calculator library.",
                "target_users": "Students",
                "category": "cli_tool",
                "questions": [],
                "ready_for_brief": self.discovery_ready,
            },
            "brief": {
                "product_name": self.product_name,
                "one_line_summary": "A tiny calculator.",
                "description": "A tiny calculator library.",
                "category": "cli_tool",
                "target_users": ["Students"],
                "problem_statement": "Quick sums are fiddly.",
                "goals": ["Add numbers"],
                "key_features": ["Addition"],
            },
            "roadmap": {
                "summary": "Two small steps.",
                "steps": [
                    {
                        "number": 1,
                        "name": "Core arithmetic",
                        "objective": "Add numbers.",
                        "user_visible_outcome": "Numbers can be added.",
                        "deliverables": ["calc module"],
                        "dependencies": [],
                        "completion_conditions": ["Addition works"],
                    },
                    {
                        "number": 2,
                        "name": "Statistics",
                        "objective": "Compute a mean.",
                        "user_visible_outcome": "A mean can be computed.",
                        "deliverables": ["stats module"],
                        "dependencies": [1],
                        "completion_conditions": ["Mean works"],
                    },
                ],
            },
            "step_brief": {
                "objective": "Deliver the step.",
                "business_outcome": "The feature works.",
                "requirements": [
                    {"id": "FR-1", "title": "Feature", "description": "It works."}
                ],
            },
            "architect": {
                "summary": f"A plain {module} module.",
                "design": "One Python module.",
                "components": [
                    {"name": module, "file_path": f"{module}.py", "purpose": "The logic."}
                ],
                "interfaces": [
                    {
                        "name": "add" if number == 1 else "mean",
                        "kind": "function",
                        "location": f"{module}.py",
                        "contract": "Returns the result.",
                    }
                ],
                "technology_choices": [
                    {"name": "Python", "purpose": "Logic", "reason": "Already available."}
                ],
                "decisions": [
                    {
                        "id": "ADR-1",
                        "decision": f"Use a single {module} module",
                        "rationale": "It is the simplest thing that works.",
                    }
                ],
                "handoff_summary": f"Build {module}.py with a plain function.",
            },
            "developer_plan": {
                "summary": f"Create {module}.py.",
                "approach": "Write one function.",
                "file_changes": [
                    {"path": f"{module}.py", "action": "create", "purpose": "The logic."}
                ],
                "handoff_summary": f"I will create {module}.py.",
            },
            "test_plan": {
                "summary": "Check the function.",
                "verification_profile": "python_pytest",
                "acceptance_criteria": [
                    {"id": "AC-1", "scenario": "Works", "given": "Inputs", "when": "Called",
                     "then": "Correct result"}
                ],
                "test_cases": [
                    {"id": "TC-1", "name": "Happy", "target_acceptance_criterion": "AC-1",
                     "description": "Normal input.", "steps": ["Call"], "expected_result": "OK",
                     "scenario_type": "positive"},
                    {"id": "TC-2", "name": "Sad", "target_acceptance_criterion": "AC-1",
                     "description": "Unusual input.", "steps": ["Call"], "expected_result": "OK",
                     "scenario_type": "negative"},
                ],
                "test_files": [{"path": f"test_{module}.py", "purpose": "All cases."}],
                "handoff_summary": "Two automated tests are planned.",
            },
            "test_file": {
                "path": f"test_{module}.py",
                "content": CALC_TEST if number == 1 else STATS_TEST,
            },
            "proposal": {
                "what_this_step_delivers": "A working feature.",
                "why_this_design": "It is simple.",
                "technology_choices_explained": ["Python, because it is already installed."],
                "what_the_developer_will_do": "Write one small file.",
                "how_it_will_be_verified": "Automated tests.",
                "completion_conditions": ["It works."],
            },
            "assess": {"verdict": "pass", "summary": "All tests passed."},
            "review": {
                "verdict": "approved",
                "summary": "Matches the design.",
                "consistency_with_design": "Consistent.",
            },
            "closure": {
                "message": "This step is done.",
                "what_was_delivered": "The feature.",
                "what_was_tested": "The automated suite.",
                "conditions": [
                    {
                        "condition": "Addition works" if number == 1 else "Mean works",
                        "met": True,
                        "evidence": "Tests passed.",
                    }
                ],
            },
            "route_modification": {
                "restated_request": "Apply the requested change.",
                "affected_agents": ["software_architect"],
            },
            "team_question": {"answer": "Here is a plain answer."},
        }[kind]

    def _developer_writes(self, kind: str, task_id: str) -> dict[str, str]:
        number = self._number(task_id)
        if number >= 2:
            return {"stats.py": STATS_SOURCE}
        return dict(self.repaired_sources if kind == "repair" else self.sources)


class ScriptedRuntime(ModelRuntime):
    def __init__(self, model: str, team: ScriptedTeam) -> None:
        self._model = model
        self._team = team

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def runtime_label(self) -> str:
        return "scripted"

    async def generate(
        self,
        instruction: str,
        user_message: str,
        tools: list[Any] | None = None,
        json_mode: bool = False,
        session_id: str = "generate",
    ) -> tuple[str, list[dict]]:
        team = self._team
        task_id = session_id.rsplit("_r", 1)[0]
        kind = kind_of(task_id)
        team.calls.append(
            Call(
                model=self._model,
                kind=kind,
                task_id=task_id,
                instruction=instruction,
                prompt=user_message,
                tools=[tool.__name__ for tool in (tools or [])],
                json_mode=json_mode,
            )
        )
        team.active += 1
        team.max_active = max(team.max_active, team.active)
        if team.lifecycle is not None:
            team.lifecycle.loaded.add(self._model)
            team.max_resident = max(team.max_resident, len(team.lifecycle.loaded))
        try:
            if team.delay:
                await asyncio.sleep(team.delay)
            queue = team.scripts.get(kind)
            step = queue.pop(0) if queue else None

            if isinstance(step, Fail):
                raise step.exc
            if isinstance(step, Text):
                return step.text, []
            if callable(step):
                return step(self._model, user_message, tools), []
            if isinstance(step, Silent):
                return "Here is the code you asked for.", []
            if tools:
                writer = tools[0]
                for path, content in team._developer_writes(kind, task_id).items():
                    writer(path, content)
                return "Done.", []
            return json.dumps(team._default(kind, task_id)), []
        finally:
            team.active -= 1

    async def complete(self, prompt: str) -> str:
        raise NotImplementedError

    async def complete_structured(self, prompt: str) -> StructuredResponse:
        raise NotImplementedError

    async def complete_structured_schema(self, prompt, schema, instruction=None,
                                         session_id="structured", max_retries=2):
        raise NotImplementedError

    async def run_agent(self, instruction, user_message, tools=None, session_id="agent"):
        raise NotImplementedError

    async def call_tool(self, tool_fn, prompt) -> ToolCallResult:
        raise NotImplementedError


class FakeLifecycle:
    """Records residency instead of talking to Ollama."""

    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.unloads: list[str] = []

    async def running(self) -> list[str]:
        return sorted(self.loaded)

    async def unload(self, model_id: str) -> bool:
        self.unloads.append(model_id)
        self.loaded.discard(model_id)
        return True


class ScriptedTerminal:
    """A person at the keyboard: answers in order, and records what was shown."""

    heartbeat_seconds = 10.0

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        # Lines typed while the team works.  A callable decides when each one appears.
        self.typed: list[str] = []
        self.on_poll: Callable[[], str | None] | None = None
        self.heartbeats: list[str] = []
        self.finished: list[str] = []
        self.prompts: list[str] = []
        self.said: list[str] = []
        self.agent_messages: list[tuple[str, str]] = []
        self.statuses: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def poll_line(self) -> str | None:
        if self.typed:
            return self.typed.pop(0)
        return self.on_poll() if self.on_poll is not None else None

    def say(self, text: str) -> None:
        self.said.append(text)

    def agent(self, role: str, text: str) -> None:
        self.agent_messages.append((role, text))

    def status(self, text: str) -> None:
        self.statuses.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def error(self, text: str) -> None:
        self.errors.append(text)

    def progress(self, text: str, context: str = "") -> None:
        self.heartbeats.append(text)

    def progress_done(self, text: str, context: str = "") -> None:
        self.finished.append(text)

    @property
    def transcript(self) -> str:
        spoken = [text for _, text in self.agent_messages]
        return "\n".join(
            [*self.said, *spoken, *self.prompts, *self.statuses, *self.warnings, *self.errors]
        )


class Harness:
    """Builds the platform around a ScriptedTeam, and can simulate a restart."""

    def __init__(self, root: Path, team: ScriptedTeam | None = None,
                 policy: ModelPolicy | None = None) -> None:
        self.root = root / "products_log"
        self.team = team or ScriptedTeam()
        self.policy = policy or load_model_policy()
        self.playbooks = PlaybookLibrary()
        self.manager = ProductManager(self.root, self.playbooks)
        self.lifecycle = FakeLifecycle()
        self.team.lifecycle = self.lifecycle
        self.router = self.new_router()

    def new_router(self) -> ModelRouter:
        return ModelRouter(self.policy, self.team.runtime_factory, self.lifecycle)

    def workflow(self, store: ProductStore, reporter: Any = None) -> ProductWorkflow:
        return ProductWorkflow(
            store, self.manager, self.router, self.policy, self.playbooks, reporter=reporter
        )

    def new_workflow(self) -> ProductWorkflow:
        return self.workflow(self.manager.create())

    def restart(self, workflow: ProductWorkflow) -> ProductWorkflow:
        """Build everything again from disk, as a fresh process would."""
        self.router = self.new_router()
        store = ProductStore(self.root / workflow.store.slug)
        fresh = self.workflow(store)
        fresh.resume()
        return fresh


APPROVE = HumanDecision("yes")


async def to_name_gate(workflow: ProductWorkflow) -> None:
    await workflow.discovery_message("I want a small calculator.")
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.AWAITING_NAME


async def to_roadmap_gate(workflow: ProductWorkflow) -> None:
    await to_name_gate(workflow)
    workflow.confirm_name(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL


async def to_step_start_gate(workflow: ProductWorkflow) -> None:
    await to_roadmap_gate(workflow)
    workflow.approve_roadmap(APPROVE)
    assert workflow.state.phase == ProductPhase.AWAITING_STEP_START


async def to_proposal_gate(workflow: ProductWorkflow) -> None:
    await to_step_start_gate(workflow)
    workflow.start_step(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL


async def complete_first_step(workflow: ProductWorkflow) -> None:
    await to_proposal_gate(workflow)
    workflow.approve_step_proposal(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.AWAITING_STEP_START


Responder = Callable[[str, str, list[Any] | None], str]


def roadmap_of(*steps: tuple[str, str]) -> str:
    """A scripted roadmap of (name, milestone) pairs, each a one-condition step."""
    return json.dumps(
        {
            "summary": "A scripted plan.",
            "steps": [
                {
                    "number": number,
                    "name": name,
                    "objective": f"Deliver {name}.",
                    "user_visible_outcome": f"{name} works.",
                    "deliverables": [f"{name} module"],
                    "dependencies": [number - 1] if number > 1 else [],
                    "completion_conditions": ["Mean works" if number > 1 else "Addition works"],
                    "milestone": milestone,
                }
                for number, (name, milestone) in enumerate(steps, start=1)
            ],
        }
    )


BACKEND_THEN_FRONTEND = (
    ("Core arithmetic", "Backend"),
    ("Statistics", "Backend"),
    ("Screens", "Frontend"),
)


def enable_autonomy(workflow: ProductWorkflow, text: str = "on") -> str:
    return workflow.change_autonomy(HumanDecision(text))


async def to_failing_verification(harness: Harness, workflow: ProductWorkflow) -> None:
    """Reach a block with real failing tests and no repair budget left."""
    harness.team.sources = {"calc.py": CALC_BUGGY}
    harness.policy.workflow.max_repair_cycles = 0
    await to_proposal_gate(workflow)
    workflow.approve_step_proposal(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.BLOCKED
