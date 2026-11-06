"""Plain-language wording for workflow state.  All deterministic, no model involved."""

from __future__ import annotations

from lab.schemas.product import ProductPhase, Roadmap, StepStatus, WorkflowState

_ROLE_LABELS = {
    "business_analyst": "Business Analyst",
    "software_architect": "Software Architect",
    "developer": "Developer",
    "test_engineer": "Test Engineer",
}


def role_label(role: str | None) -> str:
    return _ROLE_LABELS.get(role or "", role or "the team")


def _current_number(state: WorkflowState, roadmap: Roadmap | None) -> int | None:
    if roadmap is None or state.current_step_id is None:
        return None
    for step in roadmap.steps:
        if step.step_id == state.current_step_id:
            return step.number
    return None


def _next_step(state: WorkflowState, roadmap: Roadmap | None):
    if roadmap is None:
        return None
    for step in roadmap.steps:
        progress = state.steps.get(step.step_id)
        if progress is None or progress.status != StepStatus.COMPLETED:
            return step
    return None


def completed_count(state: WorkflowState, roadmap: Roadmap | None) -> int:
    if roadmap is None:
        return 0
    return sum(
        1
        for step in roadmap.steps
        if (progress := state.steps.get(step.step_id)) is not None
        and progress.status == StepStatus.COMPLETED
    )


def _blocked_label(state: WorkflowState) -> str:
    message = state.blocked.message if state.blocked else "needs attention"
    if state.intervention is not None:
        return f"BLOCKED / HUMAN INTERVENTION REQUIRED: {message}"
    return f"Blocked: {message}"


def status_label(state: WorkflowState, roadmap: Roadmap | None) -> str:
    number = _current_number(state, roadmap)
    step = f"Step {number}" if number else "Current step"
    labels = {
        ProductPhase.DISCOVERY: "Discovering requirements",
        ProductPhase.BRIEF_DRAFTING: "Preparing the product brief",
        ProductPhase.AWAITING_NAME: "Waiting for your name decision",
        ProductPhase.ROADMAP_DRAFTING: "Preparing the roadmap",
        ProductPhase.AWAITING_ROADMAP_APPROVAL: "Roadmap waiting for approval",
        ProductPhase.STEP_PREPARING: f"{step}: preparing",
        ProductPhase.STEP_ARCHITECT: f"{step}: consulting the Architect",
        ProductPhase.STEP_DEVELOPER_PLAN: f"{step}: consulting the Developer",
        ProductPhase.STEP_TEST_PLAN: f"{step}: consulting the Test Engineer",
        ProductPhase.STEP_PROPOSAL: f"{step}: preparing your proposal",
        ProductPhase.AWAITING_STEP_APPROVAL: f"{step}: proposal waiting for approval",
        ProductPhase.IMPLEMENTING: f"{step}: implementing",
        ProductPhase.VERIFYING_TESTS: f"{step}: running tests",
        ProductPhase.REVIEWING: f"{step}: reviewing the implementation",
        ProductPhase.REPAIRING: f"{step}: repairing test failures",
        ProductPhase.STEP_CLOSING: f"{step}: closing",
        ProductPhase.PRODUCT_COMPLETE: "All roadmap steps completed",
        ProductPhase.HISTORICAL: "Historical product",
        ProductPhase.BLOCKED: _blocked_label(state),
    }
    if state.phase == ProductPhase.AWAITING_STEP_START:
        done = completed_count(state, roadmap)
        if done == 0:
            label = "Roadmap approved"
        else:
            label = f"Step {done} completed"
    else:
        label = labels[state.phase]
    return f"Paused — {label}" if state.paused else label


def progress_label(state: WorkflowState, roadmap: Roadmap | None) -> str:
    if state.phase == ProductPhase.HISTORICAL:
        return "Built before this platform"
    if roadmap is None:
        return "Not started"
    total = len(roadmap.steps)
    done = completed_count(state, roadmap)
    if done >= total:
        return f"{total} of {total} steps completed"
    number = _current_number(state, roadmap)
    if number is None:
        following = _next_step(state, roadmap)
        number = following.number if following else done + 1
    return f"Step {number} of {total}"


def next_action(state: WorkflowState, roadmap: Roadmap | None) -> str:
    """The single next thing the product is waiting for, in plain language."""
    number = _current_number(state, roadmap)
    step = f"Step {number}" if number else "this step"
    following = _next_step(state, roadmap)
    actions = {
        ProductPhase.DISCOVERY: "Continue the product discussion.",
        ProductPhase.BRIEF_DRAFTING: "Prepare the product brief.",
        ProductPhase.AWAITING_NAME: "Confirm or change the product name.",
        ProductPhase.ROADMAP_DRAFTING: "Prepare the proposed roadmap.",
        ProductPhase.AWAITING_ROADMAP_APPROVAL: "Review and approve the proposed roadmap.",
        ProductPhase.STEP_PREPARING: f"Prepare {step}.",
        ProductPhase.STEP_ARCHITECT: f"Finish the Architect's proposal for {step}.",
        ProductPhase.STEP_DEVELOPER_PLAN: f"Finish the Developer's plan for {step}.",
        ProductPhase.STEP_TEST_PLAN: f"Finish the Test Engineer's plan for {step}.",
        ProductPhase.STEP_PROPOSAL: f"Prepare the proposal for {step}.",
        ProductPhase.AWAITING_STEP_APPROVAL: f"Approve implementation proposal for {step}.",
        ProductPhase.IMPLEMENTING: f"Finish implementing {step}.",
        ProductPhase.VERIFYING_TESTS: f"Run the tests for {step}.",
        ProductPhase.REVIEWING: f"Finish the Architect's review of {step}.",
        ProductPhase.REPAIRING: f"Repair the findings for {step}.",
        ProductPhase.STEP_CLOSING: f"Close {step}.",
        ProductPhase.PRODUCT_COMPLETE: "Nothing pending. The roadmap is complete.",
        ProductPhase.HISTORICAL: "Nothing pending. This product is preserved for reference.",
        ProductPhase.BLOCKED: "Decide how to continue from the blocked state.",
    }
    if state.phase == ProductPhase.BLOCKED and state.intervention is not None:
        if state.intervention.pending_revision is not None:
            return "Review your revised tests, then approve or reject them."
        return f"Read {state.intervention.report_path}, fix the problem, then use /verify."
    if state.phase == ProductPhase.AWAITING_STEP_START:
        if following is None:
            return "Nothing pending. The roadmap is complete."
        return f"Start Step {following.number}: {following.name}."
    return actions[state.phase]
