"""Bounded prompt context, rebuilt from saved files on every attempt.

Nothing here reads process memory, and no transcript is copied wholesale into a
request.  Each task names the sections it needs, each section is capped, and the
total is sized to the context window of whichever model will receive it.  A
fallback model with a smaller window therefore gets a smaller prompt, not a
silently truncated one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lab.config import ModelPolicy, VerificationProfile
from lab.playbooks import PlaybookLibrary
from lab.product import render
from lab.product.manager import latest_roadmap
from lab.product.store import ProductStore
from lab.schemas.product import (
    DeveloperPlan,
    HandoffRecord,
    ModelAttempt,
    ProductMetadata,
    Roadmap,
    StepArchitecture,
    StepBrief,
    StepStatus,
    StepTestPlan,
    TaskResult,
    WorkflowState,
)
from lab.tools.developer_tools import ProjectWorkspace
from lab.tools.fsutil import read_jsonl

_WORD = re.compile(r"[a-z0-9_]{3,}")
_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "that", "this", "from", "will", "are", "not", "you", "can"}
)
_RETRIEVABLE_SUFFIXES = frozenset(
    {
        ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".html", ".css", ".json", ".md", ".txt",
        ".yaml", ".yml", ".toml", ".sql", ".sh",
    }
)


def proposal_artifact(step_id: str, version: int, name: str) -> str:
    return f"steps/{step_id}/proposals/v{version}/{name}.json"


@dataclass
class ContextRequest:
    """Which saved material a task needs."""

    sections: tuple[str, ...]
    step_id: str | None = None
    proposal_version: int | None = None
    query: str = ""
    paths: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)


class ContextBuilder:
    def __init__(
        self,
        store: ProductStore,
        policy: ModelPolicy,
        playbooks: PlaybookLibrary,
        profiles: dict[str, VerificationProfile],
    ) -> None:
        self.store = store
        self.policy = policy
        self.playbooks = playbooks
        self.profiles = profiles

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def budget_chars(self, model_id: str) -> int:
        spec = self.policy.spec_for(model_id)
        tokens = spec.num_ctx - spec.max_output_tokens - self.policy.context.reserved_tokens
        return max(int(tokens * self.policy.context.chars_per_token), 2000)

    def _clip(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[: max(limit - 40, 0)].rstrip()}\n[... shortened to fit ...]"

    # ------------------------------------------------------------------
    # Saved state
    # ------------------------------------------------------------------

    def _state(self) -> WorkflowState:
        return self.store.load_state()

    def _metadata(self) -> ProductMetadata:
        return self.store.load_metadata()

    def _roadmap(self) -> Roadmap | None:
        return latest_roadmap(self.store, self._state())

    def _artifact(self, request: ContextRequest, name: str, schema):
        if request.step_id is None or request.proposal_version is None:
            return None
        relative = proposal_artifact(request.step_id, request.proposal_version, name)
        return self.store.load_model(relative, schema) if self.store.exists(relative) else None

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _section(self, name: str, request: ContextRequest) -> tuple[str, str] | None:
        if name in request.extra:
            return name.replace("_", " ").title(), request.extra[name]
        if name.startswith("playbook:"):
            return self._playbook_section(name.split(":", 1)[1])
        provider = getattr(self, f"_section_{name}", None)
        return provider(request) if provider else None

    def _playbook_section(self, key: str) -> tuple[str, str] | None:
        category = self._metadata().category
        playbook = self.playbooks.get(category)
        if playbook is None:
            return None
        return (
            f"Internal {playbook.title} guidance (adapt it, do not copy it to the user)",
            playbook.section(key),
        )

    def _section_brief(self, request: ContextRequest) -> tuple[str, str] | None:
        text = self.store.read_text("brief.md")
        return ("Product brief", text) if text else None

    def _section_conversation(self, request: ContextRequest) -> tuple[str, str] | None:
        turns = self.store.read_conversation()[-self.policy.context.max_conversation_turns :]
        if not turns:
            return None
        labels = {"user": "User", "business_analyst": "Business Analyst"}
        lines = [f"{labels.get(turn['speaker'], turn['speaker'])}: {turn['text']}" for turn in turns]
        return "Conversation so far", "\n".join(lines)

    def _section_roadmap(self, request: ContextRequest) -> tuple[str, str] | None:
        roadmap = self._roadmap()
        if roadmap is None:
            return None
        state = self._state()
        lines = [f"Approved baseline version: {state.approved_roadmap_version}", roadmap.summary]
        for step in roadmap.steps:
            progress = state.steps.get(step.step_id)
            status = progress.status.value if progress else "pending"
            marker = " <- current step" if step.step_id == request.step_id else ""
            lines.append(f"{step.number}. {step.name} [{status}]{marker}")
        return "Roadmap", "\n".join(lines)

    def _section_step(self, request: ContextRequest) -> tuple[str, str] | None:
        roadmap = self._roadmap()
        if roadmap is None or request.step_id is None:
            return None
        step = roadmap.step(request.step_id)
        return (
            "Current roadmap step",
            "\n".join(
                [
                    f"Step {step.number}: {step.name}",
                    f"Objective: {step.objective}",
                    f"User-visible outcome: {step.user_visible_outcome}",
                    f"Deliverables: {'; '.join(step.deliverables)}",
                    f"Completion conditions: {'; '.join(step.completion_conditions)}",
                ]
            ),
        )

    def _section_step_brief(self, request: ContextRequest) -> tuple[str, str] | None:
        brief = self._artifact(request, "step_brief", StepBrief)
        return ("Step requirements", render.step_brief_markdown(brief)) if brief else None

    def _section_architecture(self, request: ContextRequest) -> tuple[str, str] | None:
        design = self._artifact(request, "architect", StepArchitecture)
        return ("Approved design", render.architecture_markdown(design)) if design else None

    def _section_interfaces(self, request: ContextRequest) -> tuple[str, str] | None:
        design = self._artifact(request, "architect", StepArchitecture)
        return ("Public contracts", render.interfaces_markdown(design)) if design else None

    def _section_developer_plan(self, request: ContextRequest) -> tuple[str, str] | None:
        plan = self._artifact(request, "developer", DeveloperPlan)
        return ("Developer plan", render.developer_plan_markdown(plan)) if plan else None

    def _section_test_plan(self, request: ContextRequest) -> tuple[str, str] | None:
        plan = self._artifact(request, "test_plan", StepTestPlan)
        return ("Test plan", render.test_plan_markdown(plan)) if plan else None

    def _section_decisions(self, request: ContextRequest) -> tuple[str, str] | None:
        text = self.store.read_text("decisions.md")
        if not text:
            return None
        return "Earlier decisions (most recent)", text[-self.policy.context.max_section_chars :]

    def _section_earlier_steps(self, request: ContextRequest) -> tuple[str, str] | None:
        roadmap = self._roadmap()
        if roadmap is None:
            return None
        state = self._state()
        lines: list[str] = []
        for step in roadmap.steps:
            progress = state.steps.get(step.step_id)
            if progress is None or progress.status != StepStatus.COMPLETED:
                continue
            lines.append(f"### Step {step.number}: {step.name}")
            version = progress.approved_version
            if version is None:
                continue
            relative = proposal_artifact(step.step_id, version, "architect")
            if self.store.exists(relative):
                design = self.store.load_model(relative, StepArchitecture)
                lines.append(design.summary)
                lines.append(render.interfaces_markdown(design))
        return ("Existing architecture from completed steps", "\n".join(lines)) if lines else None

    def _section_modifications(self, request: ContextRequest) -> tuple[str, str] | None:
        if request.step_id is None:
            return None
        progress = self._state().steps.get(request.step_id)
        if progress is None or not progress.modification_requests:
            return None
        return "Changes the human requested", "\n".join(
            f"- {item}" for item in progress.modification_requests
        )

    def _section_workspace_tree(self, request: ContextRequest) -> tuple[str, str]:
        files = ProjectWorkspace(self.store.workspace_dir).list_files()
        body = "\n".join(files[:200]) if files else "The workspace is empty."
        return "Files currently in the product workspace", body

    def _section_relevant_files(self, request: ContextRequest) -> tuple[str, str] | None:
        chosen = self._retrieve(request)
        if not chosen:
            return None
        limit = self.policy.context.retrieval_file_chars
        blocks = [
            f"--- {path} ---\n{self._clip(self._read_workspace(path), limit)}" for path in chosen
        ]
        return "Relevant existing files", "\n\n".join(blocks)

    def _read_workspace(self, relative_path: str) -> str:
        path = self.store.workspace_dir / relative_path
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""

    def _retrieve(self, request: ContextRequest) -> list[str]:
        """Pick files by keyword overlap with the task.  Named paths always come first."""
        files = ProjectWorkspace(self.store.workspace_dir).list_files()
        named = [path for path in request.paths if path in files]
        terms = set(_WORD.findall(request.query.lower())) - _STOPWORDS
        scored: list[tuple[int, str]] = []
        for path in files:
            if path in named or f".{path.rsplit('.', 1)[-1]}" not in _RETRIEVABLE_SUFFIXES:
                continue
            words = set(_WORD.findall(path.lower()))
            words |= set(_WORD.findall(self._read_workspace(path)[:20000].lower()))
            score = len(terms & words)
            if score:
                scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = named + [path for _, path in scored]
        return ranked[: max(self.policy.context.retrieval_top_k, len(named))]

    def _section_verification_profiles(self, request: ContextRequest) -> tuple[str, str]:
        from lab.tools.verification import profile_available

        lines = [
            f"- {profile.profile_id}: {profile.description}"
            for profile in self.profiles.values()
            if profile_available(profile)
        ]
        return "Verification profiles available on this machine", "\n".join(lines)

    def _section_implementation_files(self, request: ContextRequest) -> tuple[str, str] | None:
        files = ProjectWorkspace(self.store.workspace_dir).list_files()
        if not files:
            return None
        ordered = [path for path in request.paths if path in files]
        ordered += [path for path in files if path not in ordered]
        limit = self.policy.context.retrieval_file_chars
        blocks = [
            f"--- {path} ---\n{self._clip(self._read_workspace(path), limit)}"
            for path in ordered[:12]
        ]
        return "Actual implementation files", "\n\n".join(blocks)

    def _section_last_handoff(self, request: ContextRequest) -> tuple[str, str] | None:
        records, _ = read_jsonl(self.store.logs_dir / "handoffs.jsonl")
        if not records:
            return None
        last = HandoffRecord.model_validate(records[-1])
        return (
            "Last successful handoff",
            f"{last.sender} to {last.receiver} on model {last.model}: "
            f"{last.handoff_summary or last.response_excerpt}",
        )

    def _completed_work(self, step_id: str | None) -> str:
        tasks_dir = self.store.logs_dir / "tasks"
        if not tasks_dir.is_dir():
            return "Nothing yet."
        lines = []
        for path in sorted(tasks_dir.glob("*/result.json")):
            result = TaskResult.model_validate_json(path.read_text(encoding="utf-8"))
            if result.step_id == step_id:
                lines.append(f"- {result.kind} ({result.role}) finished: {result.task_id}")
        return "\n".join(lines) or "Nothing yet."

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _render_sections(self, request: ContextRequest, budget: int) -> tuple[str, list[str]]:
        cap = self.policy.context.max_section_chars
        blocks: list[str] = []
        used: list[str] = []
        remaining = budget
        for name in request.sections:
            if remaining < 200:
                break
            section = self._section(name, request)
            if section is None or not section[1].strip():
                continue
            title, body = section
            block = f"## {title}\n\n{self._clip(body.strip(), min(cap, remaining))}"
            blocks.append(block)
            used.append(name)
            remaining -= len(block)
        return "\n\n".join(blocks), used

    def sections_used(self, request: ContextRequest, model_id: str) -> list[str]:
        return self._render_sections(request, self.budget_chars(model_id))[1]

    def task_prompt(self, objective: str, request: ContextRequest, model_id: str) -> str:
        budget = self.budget_chars(model_id) - len(objective)
        body, _ = self._render_sections(request, budget)
        return f"# Context\n\n{body}\n\n# Your task\n\n{objective}"

    def failover_prompt(
        self,
        objective: str,
        request: ContextRequest,
        model_id: str,
        failures: list[ModelAttempt],
    ) -> str:
        """Rebuild an unfinished task for a different model.

        The failed output is deliberately absent.  The new model gets the original
        task, the last valid saved context, and diagnostics describing what went
        wrong, so it starts clean rather than continuing from broken text.
        """
        last = failures[-1]
        attempts = "\n".join(
            f"- Attempt {item.overall_attempt} on {item.model}: "
            f"{item.failure_class.value if item.failure_class else 'failed'}. {item.error}"
            for item in failures
        )
        header = (
            "# Handover: resume an unfinished task\n\n"
            "Another model was working on this task and could not finish it. You are "
            "taking it over. Resume this one task. Do NOT restart the product, and do "
            "not redo finished work. Treat the context below as correct and final.\n\n"
            f"## Original task\n\n{objective}\n\n"
            f"## Exact failure to avoid\n\n"
            f"{last.failure_class.value if last.failure_class else 'failed'}: {last.error}\n\n"
            f"## Previous attempts\n\n{attempts}\n\n"
            f"## Already completed for this step\n\n{self._completed_work(request.step_id)}\n\n"
            "## What remains\n\nProduce the complete result for the original task above, "
            "from the beginning. The failed output is not available to you."
        )
        mandatory = ("brief", "roadmap", "step")
        extra = tuple(name for name in request.sections if name not in mandatory)
        handover = ContextRequest(
            sections=(*mandatory, *extra, "decisions", "last_handoff"),
            step_id=request.step_id,
            proposal_version=request.proposal_version,
            query=request.query,
            paths=request.paths,
            extra=request.extra,
        )
        budget = self.budget_chars(model_id) - len(header)
        body, _ = self._render_sections(handover, budget)
        return f"{header}\n\n# Saved context\n\n{body}"
