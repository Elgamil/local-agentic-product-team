"""The task text given to each agent.  Kept apart from the state machine it serves."""

from __future__ import annotations

from lab.config import VerificationProfile
from lab.schemas.product import PlannedFileChange, PlannedTestFile, RoadmapStep


def discovery(categories: str, *, must_conclude: bool) -> str:
    closing = (
        "You have asked enough questions. Set ready_for_brief to true now, and tell the "
        "user which sensible assumptions you are making for anything still unknown."
        if must_conclude
        else "Set ready_for_brief to true once you know what the product is, who it is for "
        "and its main capabilities. Sensible assumptions can cover the rest."
    )
    return (
        "Continue the product discussion by responding to the user's latest message.\n"
        "Identify the most suitable category, choosing an id from this list:\n"
        f"{categories}\n\n"
        "Ask only the questions still needed, three at most, inside your reply. "
        f"{closing}"
    )


def brief(categories: str) -> str:
    return (
        "Write the product brief from the conversation. Suggest a short, descriptive, "
        "memorable product name, for example 'Dairy Corner' for a dairy shop. Include "
        "the user's own wording for scope, record what is out of scope, and list your "
        "assumptions. Use one of these category ids:\n"
        f"{categories}"
    )


def roadmap(first_number: int, *, change_request: str | None) -> str:
    text = (
        "Prepare the complete proposed roadmap for this specific product, in simple "
        "business language. Each step must deliver something the user can see and be "
        "small enough to build and verify on its own. Base the steps on this product's "
        "actual needs, not on a template, and include a final review step.\n"
        f"Number the steps consecutively starting at {first_number}. A step may depend "
        "only on lower-numbered steps. Where steps form a larger stage of the product, give "
        "them the same short milestone name, such as Backend or Frontend. Otherwise leave "
        "the milestone empty."
    )
    if first_number > 1:
        text += (
            f"\nSteps before {first_number} are finished and locked. Plan only the "
            "remaining work and do not repeat them."
        )
    if change_request:
        text += f"\n\nApply this change requested by the user: {change_request}"
    return text


def step_brief(step: RoadmapStep) -> str:
    return (
        f"Prepare Step {step.number}, '{step.name}', for the specialist team. State its "
        "objective, the expected business outcome, concrete functional requirements with "
        "ids such as FR-1, the existing work it builds on, its dependencies and any "
        "assumptions. Stay at the business level. Do not design the solution."
    )


def architect(step: RoadmapStep) -> str:
    return (
        f"Propose the technical design for Step {step.number}, '{step.name}'. Cover "
        "components with file paths relative to the workspace root, public interfaces, "
        "data changes if any, technology choices with reasons, decisions with rationale "
        "and alternatives, extensibility, and risks. Build on the existing architecture. "
        "Keep the design proportionate to this step."
    )


def developer_plan(step: RoadmapStep) -> str:
    return (
        f"Plan the implementation of Step {step.number}, '{step.name}', from the design. "
        "List every file to create or modify with its purpose, your approach, "
        "dependencies, and technical risks. This is a plan only: write no code and "
        "implement nothing yet."
    )


def test_plan(step: RoadmapStep) -> str:
    return (
        f"Independently plan how to verify Step {step.number}, '{step.name}'. Define "
        "acceptance criteria in Given/When/Then form with ids such as AC-1, concrete test "
        "cases each targeting one criterion, positive and negative scenarios, edge cases "
        "and regression checks. Select one verification profile by id, and plan one to "
        "four automated test files."
    )


def test_file(planned: PlannedTestFile, profile: VerificationProfile) -> str:
    return (
        f"Write the complete automated test file '{planned.path}'.\n"
        f"Purpose: {planned.purpose}\n\n"
        f"Verification profile rules:\n{profile.authoring_notes}\n"
        "Rely only on the public contracts in the context. Set the 'path' field to "
        f"exactly '{planned.path}'."
    )


def proposal(step: RoadmapStep) -> str:
    return (
        f"Consolidate the specialists' work on Step {step.number}, '{step.name}', into a "
        "proposal for the user, in plain language. Explain what the step delivers, why "
        "the Architect recommends this design, each technology choice in simple terms, "
        "what the Developer intends to implement, how the Test Engineer will verify the "
        "result, and the completion conditions."
    )


def implement_file(change: PlannedFileChange) -> str:
    verb = "Create" if change.action == "create" else "Update"
    notes = f"\nNotes: {change.implementation_notes}" if change.implementation_notes else ""
    return (
        f"{verb} the file '{change.path}'.\nPurpose: {change.purpose}{notes}\n\n"
        f"You MUST call write_project_file with relative_path='{change.path}' and the "
        "complete, working file content. Use read_project_file first if you need to see "
        "an existing file. Writing code in your reply does not create the file."
    )


def assess(step: RoadmapStep) -> str:
    return (
        f"The automated tests for Step {step.number}, '{step.name}', have run. Compare the "
        "actual results with your test plan. Give a verdict of pass only if the real "
        "output supports it, record whether each test case was covered, and note any "
        "concerns."
    )


def review(step: RoadmapStep) -> str:
    return (
        f"Review the actual implementation of Step {step.number}, '{step.name}', against "
        "the approved design. Judge the files you are shown, not the plan. Use "
        "changes_requested only for a real blocker or major problem. Minor suggestions "
        "belong in findings on an approved review."
    )


def repair(details: str) -> str:
    return (
        "The implementation has problems that must be fixed. Here is the real evidence:\n\n"
        f"{details}\n\n"
        "Find the root cause, then fix the product code by calling write_project_file "
        "with the complete corrected content of each file you change. The authoritative "
        "tests cannot be changed. Keep working behaviour intact."
    )


def closure(step: RoadmapStep) -> str:
    conditions = "\n".join(f"- {item}" for item in step.completion_conditions)
    return (
        f"Write the closing report for Step {step.number}, '{step.name}', for the user. "
        "Use only the evidence in the context. Say what was delivered, what was tested, "
        "and any remaining limitations. Judge each completion condition below, using "
        f"exactly this wording, with the evidence for your judgement:\n{conditions}"
    )


def team_question(question: str) -> str:
    return (
        "The user asked this about the proposal. Answer in plain language, based only on "
        f"the context, and be honest about anything unknown:\n\n{question}"
    )


def route_modification(request: str) -> str:
    return (
        "The user asked for this change to the proposal:\n\n"
        f"{request}\n\n"
        "Restate it precisely, and say which specialists it affects: software_architect "
        "for design or technology, developer for implementation approach, test_engineer "
        "for how it is verified."
    )
