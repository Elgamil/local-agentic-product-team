"""Reading autonomy commands and checkpoints.  Deterministic on purpose.

A checkpoint is where a person's authorization ends, so no model interprets it.  It
maps to one roadmap step or to one explicit milestone, or it does not map at all:
text that fits several steps, or that only appears in a step's description, comes
back as a question for the person rather than as a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lab.schemas.product import (
    AutonomyPolicy,
    ExecutionMode,
    Roadmap,
    RoadmapStep,
    StepStatus,
    WorkflowState,
)

USAGE = (
    "Use /autonomy on, /autonomy until <milestone or step>, /autonomy off or /autonomy status."
)

_ON_WORDS = frozenset({"on", "enable", "enabled", "full", "start", "product", "all"})
_OFF_WORDS = frozenset({"off", "disable", "disabled", "manual"})
_STATUS_WORDS = frozenset({"", "status", "show", "mode"})
_UNTIL = re.compile(r"^(?:run\s+)?(?:until|till|through|up\s+to)\s+(?P<target>.+)$", re.IGNORECASE)

# Words that describe finishing something, not which thing.
_FILLER = frozenset(
    {
        "a", "all", "an", "and", "are", "complete", "completed", "completion", "done",
        "entire", "finish", "finished", "for", "full", "is", "layer", "of", "phase",
        "stage", "the", "through", "to", "until", "whole", "work",
    }
)
_RELATIVE = frozenset({"current", "this", "next"})
_GROUP_WORDS = frozenset({"module", "milestone"})


@dataclass(frozen=True)
class AutonomyRequest:
    action: Literal["status", "on", "off", "until"]
    target: str = ""


def parse_autonomy_request(text: str) -> AutonomyRequest:
    """Read what follows /autonomy.  Raises ValueError on anything else."""
    raw = text.strip()
    lowered = raw.lower()
    if lowered in _STATUS_WORDS:
        return AutonomyRequest("status")
    if lowered in _ON_WORDS:
        return AutonomyRequest("on")
    if lowered in _OFF_WORDS:
        return AutonomyRequest("off")
    match = _UNTIL.match(raw)
    if match:
        return AutonomyRequest("until", match.group("target").strip().rstrip(".!"))
    raise ValueError(USAGE)


@dataclass(frozen=True)
class CheckpointResolution:
    """Either one step to stop after, or the reason there is not exactly one."""

    step: RoadmapStep | None = None
    milestone: str = ""
    candidates: tuple[RoadmapStep, ...] = ()
    problem: str = ""


def _tokens(text: str) -> frozenset[str]:
    """Words that identify something, with a plural 's' removed for comparison."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return frozenset(
        word[:-1] if word.endswith("s") and not word.endswith(("ss", "is", "us")) else word
        for word in words
        if word not in _FILLER
    )


def _is_complete(state: WorkflowState, step: RoadmapStep) -> bool:
    progress = state.steps.get(step.step_id)
    return progress is not None and progress.status == StepStatus.COMPLETED


def _pending(roadmap: Roadmap, state: WorkflowState) -> list[RoadmapStep]:
    return [step for step in roadmap.steps if not _is_complete(state, step)]


def _current_or_next(roadmap: Roadmap, state: WorkflowState) -> RoadmapStep | None:
    if state.current_step_id is not None:
        for step in roadmap.steps:
            if step.step_id == state.current_step_id:
                return step
    pending = _pending(roadmap, state)
    return pending[0] if pending else None


def _chosen(step: RoadmapStep, state: WorkflowState, milestone: str = "") -> CheckpointResolution:
    if _is_complete(state, step):
        return CheckpointResolution(
            problem=f"Step {step.number}: {step.name} is already complete, so autonomous "
            "execution would have nowhere to stop."
        )
    return CheckpointResolution(step=step, milestone=milestone)


def milestone_end(roadmap: Roadmap, milestone: str) -> RoadmapStep | None:
    """The last step of a milestone.  The milestone is complete when this one is."""
    steps = [
        step for step in roadmap.steps if step.milestone.casefold() == milestone.casefold()
    ]
    return steps[-1] if steps else None


def _by_position(
    tokens: frozenset[str], text: str, roadmap: Roadmap, state: WorkflowState
) -> CheckpointResolution | None:
    """Handle 'step 3', 'current step', 'next step' and 'current module'."""
    number = re.search(r"\bstep[\s-]*(\d+)\b", text.casefold())
    bare_number = len(tokens) == 1 and next(iter(tokens)).isdigit()
    if bare_number or (number and tokens <= {"step", number.group(1)}):
        wanted = int(number.group(1) if number else next(iter(tokens)))
        for step in roadmap.steps:
            if step.number == wanted:
                return _chosen(step, state)
        return CheckpointResolution(
            problem=f"The roadmap has no step {wanted}. It has {len(roadmap.steps)} step(s)."
        )

    if not tokens & _RELATIVE or not tokens <= (_RELATIVE | _GROUP_WORDS | {"step"}):
        return None
    current = _current_or_next(roadmap, state)
    if current is None:
        return CheckpointResolution(problem="Every roadmap step is already complete.")
    if tokens & _GROUP_WORDS:
        if not current.milestone:
            return CheckpointResolution(
                problem=f"Step {current.number}: {current.name} does not belong to a named "
                "milestone on this roadmap, so I cannot tell where the current module ends."
            )
        end = milestone_end(roadmap, current.milestone)
        assert end is not None
        return _chosen(end, state, current.milestone)
    if "next" in tokens and state.current_step_id is not None:
        pending = _pending(roadmap, state)
        following = [step for step in pending if step.number > current.number]
        if not following:
            return CheckpointResolution(problem="There is no step after the current one.")
        return _chosen(following[0], state)
    return _chosen(current, state)


def _by_milestone(
    tokens: frozenset[str], roadmap: Roadmap, state: WorkflowState
) -> CheckpointResolution | None:
    labels = list(dict.fromkeys(step.milestone for step in roadmap.steps if step.milestone))
    exact = [label for label in labels if _tokens(label) == tokens]
    partial = [label for label in labels if tokens <= _tokens(label)]
    matches = exact or partial
    if len(matches) == 1:
        end = milestone_end(roadmap, matches[0])
        assert end is not None
        return _chosen(end, state, matches[0])
    if matches:
        ends = [milestone_end(roadmap, label) for label in matches]
        return CheckpointResolution(candidates=tuple(step for step in ends if step))
    return None


def _by_name(
    tokens: frozenset[str], roadmap: Roadmap, state: WorkflowState
) -> CheckpointResolution | None:
    exact = [step for step in roadmap.steps if _tokens(step.name) == tokens]
    partial = [step for step in roadmap.steps if tokens <= _tokens(step.name)]
    matches = exact or partial
    if len(matches) == 1:
        return _chosen(matches[0], state)
    if matches:
        return CheckpointResolution(candidates=tuple(matches))
    return None


def _mentioned(tokens: frozenset[str], roadmap: Roadmap) -> tuple[RoadmapStep, ...]:
    """Steps that only mention the words.  Worth suggesting, never worth assuming."""
    found = []
    for step in roadmap.steps:
        body = " ".join([step.objective, step.user_visible_outcome, *step.deliverables])
        if tokens <= _tokens(body):
            found.append(step)
    return tuple(found)


def resolve_checkpoint(
    text: str, roadmap: Roadmap, state: WorkflowState
) -> CheckpointResolution:
    """Map a person's words to exactly one roadmap step, or say why that is impossible."""
    tokens = _tokens(text)
    if not tokens:
        return CheckpointResolution(problem="That does not name a step or a milestone.")
    return (
        _by_position(tokens, text, roadmap, state)
        or _by_milestone(tokens, roadmap, state)
        or _by_name(tokens, roadmap, state)
        or CheckpointResolution(candidates=_mentioned(tokens, roadmap))
    )


def clarification(text: str, resolution: CheckpointResolution, roadmap: Roadmap) -> str:
    """The question to ask when a checkpoint could not be mapped.  Nothing has started."""
    if resolution.problem:
        return resolution.problem
    lines = [f"I cannot map '{text}' to one point on the roadmap, so nothing has started."]
    if resolution.candidates:
        lines.append("It could mean any of these:")
        shown = resolution.candidates
    else:
        lines.append("These are the steps on the roadmap:")
        shown = tuple(roadmap.steps)
    lines.extend(f"  Step {step.number}: {step.name}" for step in shown)
    lines.append("Which one should I stop after? For example: /autonomy until step 2")
    return "\n".join(lines)


def describe_mode(policy: AutonomyPolicy, roadmap: Roadmap | None) -> str:
    """The execution mode in plain language, for /status and for progress lines."""
    if policy.mode == ExecutionMode.MANUAL:
        return "Manual (your approval is required at every step)"
    if policy.mode == ExecutionMode.AUTONOMOUS_PRODUCT:
        return "Autonomous, complete product"
    checkpoint = policy.checkpoint
    if checkpoint is None:
        return "Autonomous, until a checkpoint"
    target = checkpoint.step_id
    if roadmap is not None:
        for step in roadmap.steps:
            if step.step_id == checkpoint.step_id:
                target = f"Step {step.number}: {step.name}"
    if checkpoint.milestone:
        return f"Autonomous, until the {checkpoint.milestone} milestone (ends with {target})"
    return f"Autonomous, until {target}"
