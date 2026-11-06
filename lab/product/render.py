"""Human-readable Markdown for every artifact.

Keeping these as pure functions means the same text serves three readers: the
files under ``products_log/``, the terminal, and the context given to the models.
"""

from __future__ import annotations

from lab.product.presentation import role_label
from lab.schemas.product import (
    DeveloperPlan,
    ExecutionResult,
    HandoffRecord,
    ImplementationReview,
    ProductBrief,
    Roadmap,
    StepArchitecture,
    StepBrief,
    StepClosureReport,
    StepProgress,
    StepProposal,
    StepStatus,
    StepTestPlan,
    TestAssessment,
    VerificationEvidence,
)


def _bullets(items: list[str], empty: str = "None.") -> str:
    return "\n".join(f"- {item}" for item in items) if items else empty


def brief_markdown(brief: ProductBrief) -> str:
    return "\n\n".join(
        [
            f"# {brief.product_name}",
            f"_{brief.one_line_summary}_",
            f"## Description\n\n{brief.description}",
            f"## Problem\n\n{brief.problem_statement}",
            f"## Target users\n\n{_bullets(brief.target_users)}",
            f"## Goals\n\n{_bullets(brief.goals)}",
            f"## Key features\n\n{_bullets(brief.key_features)}",
            f"## Out of scope\n\n{_bullets(brief.out_of_scope)}",
            f"## Constraints\n\n{_bullets(brief.constraints)}",
            f"## Assumptions\n\n{_bullets(brief.assumptions)}",
        ]
    )


def roadmap_for_people(roadmap: Roadmap, product_name: str) -> str:
    """The roadmap as the Business Analyst presents it."""
    lines = [f"Proposed roadmap for {product_name}", "", roadmap.summary, ""]
    for step in roadmap.steps:
        lines.append(f"{step.number}. {step.name}")
        if step.milestone:
            lines.append(f"   Milestone: {step.milestone}")
        lines.append(f"   Objective: {step.objective}")
        lines.append(f"   You will see: {step.user_visible_outcome}")
        lines.append(f"   Deliverables: {'; '.join(step.deliverables)}")
        if step.dependencies:
            needs = ", ".join(str(number) for number in step.dependencies)
            lines.append(f"   Needs step(s): {needs}")
        lines.append(f"   Done when: {'; '.join(step.completion_conditions)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def roadmap_markdown(
    roadmap: Roadmap,
    product_name: str,
    *,
    approved_version: int | None,
    progress: dict[str, StepProgress],
) -> str:
    if approved_version == roadmap.version:
        banner = f"Approved baseline, version {roadmap.version}"
    else:
        banner = f"Proposed version {roadmap.version}, not yet approved"
    lines = [f"# Roadmap: {product_name}", "", f"**{banner}**", "", roadmap.summary, ""]
    for step in roadmap.steps:
        state = progress.get(step.step_id)
        status = state.status.value.replace("_", " ") if state else StepStatus.PENDING.value
        lines.extend(
            [
                f"## {step.number}. {step.name}",
                "",
                f"Status: {status} (`{step.step_id}`)",
                "",
                *([f"**Milestone:** {step.milestone}", ""] if step.milestone else []),
                f"**Objective:** {step.objective}",
                "",
                f"**User-visible outcome:** {step.user_visible_outcome}",
                "",
                f"**Deliverables:**\n{_bullets(step.deliverables)}",
                "",
                f"**Depends on:** {', '.join(map(str, step.dependencies)) or 'nothing'}",
                "",
                f"**Completion conditions:**\n{_bullets(step.completion_conditions)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def step_brief_markdown(brief: StepBrief) -> str:
    requirements = "\n".join(
        f"- **{item.id} {item.title}:** {item.description}" for item in brief.requirements
    )
    return "\n\n".join(
        [
            f"## Objective\n\n{brief.objective}",
            f"## Expected outcome\n\n{brief.business_outcome}",
            f"## Requirements\n\n{requirements}",
            f"## Existing work this builds on\n\n{_bullets(brief.relevant_existing_work)}",
            f"## Dependencies\n\n{_bullets(brief.dependencies)}",
            f"## Assumptions\n\n{_bullets(brief.assumptions)}",
        ]
    )


def interfaces_markdown(design: StepArchitecture) -> str:
    """Only the public contracts, which is what independent tests may rely on."""
    components = "\n".join(
        f"- `{item.file_path}` ({item.name}): {item.purpose}" for item in design.components
    )
    interfaces = "\n".join(
        f"- `{item.name}` [{item.kind}] in `{item.location}`: {item.contract}"
        for item in design.interfaces
    )
    return (
        f"## Components\n\n{components}\n\n"
        f"## Public interfaces\n\n{interfaces or 'None declared.'}"
    )


def architecture_markdown(design: StepArchitecture) -> str:
    technology = "\n".join(
        f"- **{item.name}** ({item.purpose}): {item.reason}" for item in design.technology_choices
    )
    decisions = "\n".join(
        f"- **{item.id}: {item.decision}** Reason: {item.rationale}"
        + (f" Trade-offs: {item.trade_offs}" if item.trade_offs else "")
        + (
            f" Alternatives: {', '.join(item.alternatives_considered)}."
            if item.alternatives_considered
            else ""
        )
        for item in design.decisions
    )
    data = "\n".join(
        f"- **{item.table_name}:** {item.description} "
        f"({', '.join(f'{key}: {value}' for key, value in item.fields.items())})"
        for item in design.data_changes
    )
    return "\n\n".join(
        [
            f"## Summary\n\n{design.summary}",
            f"## Design\n\n{design.design}",
            interfaces_markdown(design),
            f"## Data changes\n\n{data or 'None.'}",
            f"## Technology choices\n\n{technology}",
            f"## Decisions\n\n{decisions}",
            f"## Extensibility\n\n{_bullets(design.extensibility)}",
            f"## Risks\n\n{_bullets(design.risks)}",
        ]
    )


def developer_plan_markdown(plan: DeveloperPlan) -> str:
    changes = "\n".join(
        f"- `{item.path}` ({item.action}): {item.purpose}"
        + (f" {item.implementation_notes}" if item.implementation_notes else "")
        for item in plan.file_changes
    )
    return "\n\n".join(
        [
            f"## Summary\n\n{plan.summary}",
            f"## Approach\n\n{plan.approach}",
            f"## Files\n\n{changes}",
            f"## Dependencies\n\n{_bullets(plan.dependencies)}",
            f"## Risks\n\n{_bullets(plan.risks)}",
        ]
    )


def test_plan_markdown(plan: StepTestPlan) -> str:
    criteria = "\n".join(
        f"- **{item.id} {item.scenario}:** Given {item.given}, when {item.when}, then {item.then}."
        for item in plan.acceptance_criteria
    )
    cases = "\n".join(
        f"- **{item.id} {item.name}** ({item.scenario_type}, covers "
        f"{item.target_acceptance_criterion}): {item.description} "
        f"Expected: {item.expected_result}"
        for item in plan.test_cases
    )
    files = "\n".join(f"- `{item.path}`: {item.purpose}" for item in plan.test_files)
    return "\n\n".join(
        [
            f"## Summary\n\n{plan.summary}",
            f"## Verification profile\n\n`{plan.verification_profile}`",
            f"## Acceptance criteria\n\n{criteria}",
            f"## Test cases\n\n{cases}",
            f"## Edge cases\n\n{_bullets(plan.edge_cases)}",
            f"## Regression checks\n\n{_bullets(plan.regression_checks)}",
            f"## Automated test files\n\n{files}",
        ]
    )


def proposal_for_people(
    step_number: int,
    step_name: str,
    proposal: StepProposal,
    plan: DeveloperPlan,
    test_plan: StepTestPlan,
    version: int,
) -> str:
    """Phase C: what the human reads before deciding."""
    heading = f"Proposal for Step {step_number}: {step_name}"
    if version > 1:
        heading += f" (revision {version})"
    return "\n\n".join(
        [
            heading,
            f"What this step will deliver\n{proposal.what_this_step_delivers}",
            f"Why the Architect recommends this design\n{proposal.why_this_design}",
            "Technology choices\n" + _bullets(proposal.technology_choices_explained),
            "What the Developer intends to implement\n"
            f"{proposal.what_the_developer_will_do}\n"
            f"({len(plan.file_changes)} file(s) to create or change)",
            f"How the Test Engineer will verify it\n{proposal.how_it_will_be_verified}\n"
            f"({len(test_plan.test_cases)} test case(s) in "
            f"{len(test_plan.test_files)} automated test file(s))",
            "Completion conditions\n" + _bullets(proposal.completion_conditions),
        ]
    )


def implementation_markdown(result: ExecutionResult) -> str:
    written = "\n".join(
        f"- `{path}` (sha256 `{digest[:12]}`)" for path, digest in sorted(result.written_files.items())
    )
    return "\n\n".join(
        [
            f"# Implementation of {result.step_id}",
            f"Approved proposal version: {result.proposal_version}",
            f"Developer model(s): {', '.join(result.developer_models) or 'none recorded'}",
            f"## Files written\n\n{written or 'None.'}",
            f"## Files outside the approved plan\n\n{_bullets(result.unplanned_files)}",
        ]
    )


def test_results_markdown(evidence: VerificationEvidence) -> str:
    lines = [
        f"# Test results for {evidence.step_id}, cycle {evidence.cycle}",
        "",
        f"Overall: {'PASSED' if evidence.passed else 'FAILED'}",
        "",
        f"Tested workspace fingerprint: `{evidence.tested_workspace_fingerprint}`",
        "",
        f"Sealed manifest: `{evidence.manifest_sha256}`",
        "",
    ]
    if evidence.failure_summary:
        lines.extend([f"Failure summary: {evidence.failure_summary}", ""])
    for run in evidence.runs:
        output = run.stdout.strip() or run.stderr.strip()
        lines.extend(
            [
                f"## {run.scope.title()}: {run.step_id} ({run.profile_id})",
                "",
                f"- Command: `{run.command}`",
                f"- Working directory: `{run.cwd}`",
                f"- Exit code: {run.exit_code}",
                f"- Passed: {run.passed_count}, failed: {run.failed_count}",
                f"- Timed out: {run.timed_out}",
                "",
                "```text",
                output[-4000:],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def assessment_markdown(assessment: TestAssessment) -> str:
    coverage = "\n".join(
        f"- {item.test_case_id}: {item.status}" + (f" ({item.note})" if item.note else "")
        for item in assessment.plan_coverage
    )
    return "\n\n".join(
        [
            f"## Test Engineer assessment: {assessment.verdict}",
            assessment.summary,
            f"### Plan coverage\n\n{coverage or 'Not itemised.'}",
            f"### Concerns\n\n{_bullets(assessment.concerns)}",
        ]
    )


def review_markdown(review: ImplementationReview) -> str:
    findings = "\n".join(
        f"- **{item.severity}** `{item.file_path or 'general'}`: {item.description} "
        f"Recommendation: {item.recommendation}"
        for item in review.findings
    )
    return "\n\n".join(
        [
            f"# Architect review: {review.verdict}",
            review.summary,
            f"## Consistency with the approved design\n\n{review.consistency_with_design}",
            f"## Findings\n\n{findings or 'None.'}",
        ]
    )


def closure_for_people(
    step_number: int,
    step_name: str,
    report: StepClosureReport,
    evidence: VerificationEvidence | None,
    *,
    verified: bool,
) -> str:
    """Phase F.  Test numbers come from evidence, never from the model's wording."""
    parts = [f"Step {step_number}: {step_name}", report.message]
    parts.append(f"What was delivered\n{report.what_was_delivered}")
    parts.append(f"What was tested\n{report.what_was_tested}")
    if evidence is None:
        parts.append("Test results\nNo test evidence was recorded.")
    else:
        lines = [
            f"- {run.scope} ({run.step_id}): {run.passed_count} passed, "
            f"{run.failed_count} failed"
            for run in evidence.runs
        ]
        parts.append("Test results (from the actual runs)\n" + "\n".join(lines))
    conditions = "\n".join(
        f"- [{'met' if item.met else 'NOT met'}] {item.condition} ({item.evidence})"
        for item in report.conditions
    )
    parts.append(f"Completion conditions\n{conditions}")
    parts.append("Remaining limitations\n" + _bullets(report.remaining_limitations))
    parts.append(
        "Verification verdict: PASSED" if verified else "Verification verdict: NOT PASSED"
    )
    return "\n\n".join(parts)


def handoff_markdown(record: HandoffRecord) -> str:
    lines = [
        f"## {role_label(record.sender)} → {role_label(record.receiver)}",
        "",
        f"- Task: `{record.task_id}`",
        f"- Time: {record.timestamp.isoformat()}",
        f"- Model: `{record.model}`",
        f"- Validation: {record.validation_status}",
        f"- Attempts: {record.attempts}"
        + (f" (failed over across {', '.join(record.models_tried)})" if record.failed_over else ""),
        f"- Artifact: `{record.response_artifact or 'none'}`",
        "",
        f"**Summary:** {record.handoff_summary or record.response_excerpt}",
    ]
    if record.decisions:
        lines.extend(["", "**Decisions:**", _bullets(record.decisions)])
    if record.rationale:
        lines.extend(["", "**Rationale:**", _bullets(record.rationale)])
    if record.relevant_context:
        lines.extend(["", f"**Context used:** {', '.join(record.relevant_context)}"])
    return "\n".join(lines)
