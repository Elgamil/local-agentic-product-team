"""Human Intervention Reports: what failed, and what a person should do about it.

Everything here is deterministic.  No model runs, because this code is reached when
the models have just failed, and because a report must not invent a root cause.  So
the three kinds of statement are kept strictly apart:

* **Confirmed facts** come from recorded test runs, the attempt ledger and files.
* **The agents' diagnosis** quotes what the agents said, and is labelled as opinion.
* **Unconfirmed causes** are possibilities that fit the evidence, labelled as such.

The functions take plain data and return plain data.  The workflow gathers the inputs
and owns all state changes.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

from lab.config import VerificationProfile
from lab.product.presentation import role_label
from lab.schemas.product import (
    AttemptSummary,
    BlockedInfo,
    BlockKind,
    FailedTest,
    FailureClass,
    InterventionReport,
    ManualFileChange,
    ModelAttempt,
    RecommendedAction,
    RoadmapStep,
    StepBrief,
    StepProgress,
    StepTestPlan,
    SuiteDiagnosis,
    VerificationEvidence,
    VerificationRun,
)

MAX_FAILED_TESTS = 25
MAX_ERROR_LINES = 12
DIAGNOSTIC_CHARS = 4000
DIFF_EXCERPT_LINES = 120

# ---------------------------------------------------------------------------
# Reading test source
# ---------------------------------------------------------------------------

_PYTHON_TEST = re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)\s*\(", re.MULTILINE)
_NODE_TEST = re.compile(
    r"""(?:^|[\s;(])(?:test|it)(?:\.only)?\s*\(\s*(?:["'`])(.*?)(?:["'`])""", re.MULTILINE
)
_TEST_DEFINITIONS = {"pytest": _PYTHON_TEST, "unittest": _PYTHON_TEST, "tap": _NODE_TEST}
_ASSERTIONS = {
    "pytest": re.compile(r"^\s*assert\b|pytest\.raises", re.MULTILINE),
    "unittest": re.compile(r"self\.assert\w+"),
    "tap": re.compile(r"\bassert(?:\.\w+)?\s*\(|\bexpect\s*\("),
}
_SKIPS = re.compile(
    r"pytest\.mark\.(?:skip|skipif|xfail)|pytest\.(?:skip|xfail)\(|unittest\.skip|skipTest\(|"
    r"\.skip\s*\(|\.todo\s*\(|\{\s*skip\s*:|\{\s*todo\s*:"
)


def defined_tests(source: str, parser: str) -> list[str]:
    """The test cases a source file defines, found by reading it, not by running it."""
    return _TEST_DEFINITIONS[parser].findall(source)


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(Path(name).name, pattern) for pattern in patterns)


# ---------------------------------------------------------------------------
# Reading test output
# ---------------------------------------------------------------------------

_ERROR_LINE = re.compile(
    r"^[\s#E>|]*((?:[A-Za-z_][\w.]*)?(?:Error|Exception)\b(?::.*)?)\s*$", re.MULTILINE
)
_PYTEST_SUMMARY = re.compile(
    r"^(FAILED|ERROR)\s+(?P<target>[^\s]+?)(?:::(?P<name>\S+))?(?:\s+-\s+(?P<message>.*))?$",
    re.MULTILINE,
)
_PYTEST_ASSERT = re.compile(r"^E\s+assert\s+(?P<actual>.+?)\s*==\s*(?P<expected>.+?)\s*$", re.MULTILINE)
_UNITTEST_HEADER = re.compile(r"^(?:FAIL|ERROR): (?P<name>\S+)(?: \((?P<where>[^)]*)\))?", re.MULTILINE)
_TAP_FAILURE = re.compile(r"^\s*not ok \d+ - (?P<name>.+)$", re.MULTILINE)


def _unique(items: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))[:limit]


def _error_lines(output: str) -> list[str]:
    return _unique(_ERROR_LINE.findall(output), MAX_ERROR_LINES)


def _pytest_failures(output: str) -> list[FailedTest]:
    found = []
    for match in _PYTEST_SUMMARY.finditer(output):
        name = match.group("name") or f"{Path(match.group('target')).name} (could not be loaded)"
        found.append(
            FailedTest(
                name=name,
                file=Path(match.group("target")).name,
                message=(match.group("message") or "").strip(),
            )
        )
    comparisons = list(_PYTEST_ASSERT.finditer(output))
    for item, comparison in zip(found, comparisons):
        item.actual = comparison.group("actual")
        item.expected = comparison.group("expected")
    return found


def _unittest_failures(output: str) -> list[FailedTest]:
    blocks = re.split(r"^={40,}$", output, flags=re.MULTILINE)
    found = []
    for block in blocks:
        header = _UNITTEST_HEADER.search(block)
        if header is None:
            continue
        errors = _error_lines(block)
        name = header.group("name")
        where = header.group("where") or ""
        # A module that cannot be imported is reported as a test named after the module.
        module = name if "_FailedTest" in where else where.split(".")[0]
        found.append(
            FailedTest(
                name=name,
                file=f"{module}.py" if module else "",
                message=errors[-1] if errors else "",
            )
        )
    return found


def _tap_field(body: str, name: str) -> str:
    match = re.search(
        rf"^\s*{name}:\s*(?P<value>\|-?\n(?:\s{{4,}}.*\n?)+|.*)$", body, re.MULTILINE
    )
    if match is None:
        return ""
    value = match.group("value").strip()
    if value.startswith("|"):
        value = " ".join(line.strip() for line in value.splitlines()[1:] if line.strip())
    return value.strip("'\"")


def _tap_failures(output: str) -> list[FailedTest]:
    found = []
    matches = list(_TAP_FAILURE.finditer(output))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        body = output[match.end() : end]
        body = body.split("\n  ...")[0] if "\n  ..." in body else body
        name = match.group("name").strip()
        whole_file = name.startswith("/") or "\\" in name
        location = _tap_field(body, "location")
        source = name if whole_file else location.split(":")[0]
        found.append(
            FailedTest(
                name=f"{Path(name).name} (the file did not run)" if whole_file else name,
                file=Path(source).name if source else "",
                message="" if whole_file else _tap_field(body, "error"),
                expected=_tap_field(body, "expected"),
                actual=_tap_field(body, "actual"),
            )
        )
    return found


_FAILURE_READERS = {
    "pytest": _pytest_failures,
    "unittest": _unittest_failures,
    "tap": _tap_failures,
}


def _suite_could_not_run(run: VerificationRun, parser: str, output: str) -> bool:
    """True when no test got far enough to pass or fail on its own merits."""
    if run.timed_out or (run.passed_count == 0 and run.failed_count == 0):
        return True
    if parser == "pytest":
        return bool(re.search(r"ERROR collecting|errors? during collection", output))
    if parser == "unittest":
        return "_FailedTest" in output or "Failed to import test module" in output
    if run.passed_count:
        return False
    # Node reports a file that cannot even load as one failure named after the file.
    names = [match.group("name").strip() for match in _TAP_FAILURE.finditer(output)]
    return bool(names) and all(name.startswith("/") or "\\" in name for name in names)


def diagnose_run(
    run: VerificationRun,
    parser: str,
    suite_files: dict[str, str],
    patterns: list[str],
    root_label: str,
) -> SuiteDiagnosis:
    """Turn one real test run into facts.  *suite_files* maps paths to their source."""
    output = f"{run.stdout}\n{run.stderr}".strip()
    failures = _FAILURE_READERS[parser](output)[:MAX_FAILED_TESTS]
    executable = not _suite_could_not_run(run, parser, output)
    errors = _error_lines(output)

    # Verbose runners print every file, so only what actually failed counts as involved.
    named = {item.file for item in failures if item.file}
    involved = [path for path in suite_files if Path(path).name in named]
    if not involved:
        involved = [path for path in suite_files if _matches(path, patterns)]

    findings = []
    for path, source in suite_files.items():
        if not _matches(path, patterns):
            continue
        if not defined_tests(source, parser):
            findings.append(f"{root_label}/{path} defines no test cases.")
            if path not in involved:
                involved.append(path)
    return SuiteDiagnosis(
        step_id=run.step_id,
        scope=run.scope,
        command=run.command,
        exit_code=run.exit_code,
        passed=run.passed_count,
        failed=run.failed_count,
        timed_out=run.timed_out,
        executable=executable,
        failed_tests=failures,
        errors=errors,
        files=[f"{root_label}/{path}" for path in involved],
        static_findings=findings,
        diagnostic_output=output[-DIAGNOSTIC_CHARS:],
    )


# ---------------------------------------------------------------------------
# Comparing an edited test suite with the sealed one
# ---------------------------------------------------------------------------


def unified_diff(path: str, old: str | None, new: str | None) -> str:
    return "\n".join(
        difflib.unified_diff(
            (old or "").splitlines(),
            (new or "").splitlines(),
            f"sealed/{path}",
            f"edited/{path}",
            lineterm="",
        )
    )


def compare_suites(
    old: dict[str, str | None], new: dict[str, str], parser: str, patterns: list[str]
) -> list[str]:
    """Signs that an edited suite may test less than the sealed one.

    These are warnings for a person, not verdicts.  A real repair can trigger them, so
    they raise the bar for approval rather than forbid it.  ``None`` as a value means
    the original content is no longer available.
    """
    warnings: list[str] = []
    if any(content is None for content in old.values()):
        warnings.append(
            "The original content of a sealed file is not available, so I could not compare "
            "it with your version. Only its hash is known."
        )
    for path in sorted(set(old) - set(new)):
        warnings.append(f"{path} was removed from the suite.")
    for path in sorted(set(new) - set(old)):
        if not _matches(path, patterns):
            warnings.append(f"{path} is a new file that is not itself a test.")

    def collected(files: dict[str, str | None]) -> tuple[list[str], int, int]:
        names: list[str] = []
        assertions = skips = 0
        for path, source in files.items():
            if source is None or not _matches(path, patterns):
                continue
            names.extend(defined_tests(source, parser))
            assertions += len(_ASSERTIONS[parser].findall(source))
            skips += len(_SKIPS.findall(source))
        return names, assertions, skips

    old_names, old_assertions, old_skips = collected(old)
    new_names, new_assertions, new_skips = collected(dict(new))
    missing = sorted(set(old_names) - set(new_names))
    if missing:
        warnings.append(f"These tests no longer exist: {', '.join(missing)}.")
    if new_assertions < old_assertions:
        warnings.append(
            f"The suite has fewer assertions: {old_assertions} before, {new_assertions} now."
        )
    if new_skips > old_skips:
        warnings.append(
            f"More tests are skipped or expected to fail: {old_skips} before, {new_skips} now."
        )
    if old_names and not new_names:
        warnings.append("The revised suite defines no test cases at all.")
    return warnings


# ---------------------------------------------------------------------------
# Building the report
# ---------------------------------------------------------------------------

_UNCONFIRMED_BY_FAILURE = {
    FailureClass.INVALID_JSON: [
        "The answer may be too long for the model's output limit, so the JSON was cut off.",
    ],
    FailureClass.SCHEMA_VALIDATION: [
        "The task may be too large for this model to return as one complete structure.",
    ],
    FailureClass.INVALID_RESPONSE: [
        "The prompt may be too long for the model's context window.",
    ],
    FailureClass.TOOL_EXECUTION: [
        "The model may have described the change instead of calling the file tool.",
    ],
    FailureClass.INFERENCE_TIMEOUT: [
        "Memory pressure from other applications may have slowed the model down.",
        "The prompt may be too large to finish within the time limit on this machine.",
    ],
    FailureClass.MODEL_CONNECTION: [
        "Ollama may not be running, or it may have stopped while loading the model.",
    ],
}
_ACTION_BY_FAILURE = {
    FailureClass.MODEL_CONNECTION: "Check that Ollama is running ('python -m lab doctor' will "
    "confirm it), then use /retry.",
    FailureClass.INFERENCE_TIMEOUT: "Close other applications to free memory, then use /retry.",
    FailureClass.DEPENDENCY: "Install the missing tool, then use /verify.",
    FailureClass.FILESYSTEM: "Fix the file or disk problem shown above, then use /retry.",
}
_HOW_SEALED_FILES_OPEN = (
    "Sealed tests are read-only on purpose. To correct one, make it writable first with "
    "`chmod u+w <file>`. /verify will then show you the differences and ask you to approve "
    "the revised suite before it is used."
)


@dataclass
class ReportInputs:
    """Everything the workflow gathered for one report."""

    intervention_id: str
    version: int
    product: str
    slug: str
    blocked: BlockedInfo
    execution_mode: str
    step: RoadmapStep | None = None
    progress: StepProgress | None = None
    chain: list[str] = field(default_factory=list)
    attempts: list[ModelAttempt] = field(default_factory=list)
    attempts_allowed: int = 0
    preserved_outputs: list[str] = field(default_factory=list)
    evidence: VerificationEvidence | None = None
    suites: dict[str, dict[str, str]] = field(default_factory=dict)
    profiles: dict[str, VerificationProfile] = field(default_factory=dict)
    test_plan: StepTestPlan | None = None
    step_brief: StepBrief | None = None
    agent_notes: list[str] = field(default_factory=list)
    planned_files: list[str] = field(default_factory=list)
    manual_changes: list[ManualFileChange] = field(default_factory=list)
    changed_sealed_files: list[str] = field(default_factory=list)
    note: str = ""


def _acceptance_lines(plan: StepTestPlan | None) -> list[str]:
    if plan is None:
        return []
    return [
        f"{item.id}: {item.scenario}. Given {item.given}, when {item.when}, then {item.then}"
        for item in plan.acceptance_criteria
    ]


def _test_case_lines(plan: StepTestPlan | None) -> list[str]:
    if plan is None:
        return []
    return [
        f"{item.id} ({item.scenario_type}, covers {item.target_acceptance_criterion}): "
        f"{item.name}. {item.description}"
        for item in plan.test_cases
    ]


def _diagnose_evidence(inputs: ReportInputs) -> list[SuiteDiagnosis]:
    if inputs.evidence is None:
        return []
    diagnoses = []
    for run in inputs.evidence.runs:
        if run.success:
            continue
        profile = inputs.profiles.get(run.profile_id)
        if profile is None:
            continue
        diagnoses.append(
            diagnose_run(
                run,
                profile.result_parser,
                inputs.suites.get(run.step_id, {}),
                profile.test_file_patterns,
                f"verification/{run.step_id}",
            )
        )
    return diagnoses


def _attempt_facts(inputs: ReportInputs) -> list[str]:
    failed = [item for item in inputs.attempts if item.status == "failed"]
    if not failed:
        return []
    models = ", ".join(dict.fromkeys(item.model for item in failed))
    limit = f" of {inputs.attempts_allowed}" if inputs.attempts_allowed else ""
    return [f"{len(failed)}{limit} model attempts failed, on {models}."]


def _task_label(blocked: BlockedInfo) -> str:
    return (blocked.task_kind or "task").replace("_", " ")


def _unrunnable_action(
    diagnosis: SuiteDiagnosis, inputs: ReportInputs, profile: VerificationProfile | None
) -> RecommendedAction:
    errors = diagnosis.errors or [
        f"exit code {diagnosis.exit_code}, {diagnosis.passed} passed, {diagnosis.failed} failed"
    ]
    action = ["Correct the test file so that it runs."]
    if diagnosis.timed_out:
        action = ["Find out why the suite never finished, and correct the test or the code."]
    if profile is not None and profile.authoring_notes:
        action.append(f"Rules for this suite: {' '.join(profile.authoring_notes.split())}")
    if inputs.test_plan is not None:
        count = len(inputs.test_plan.test_cases)
        action.append(
            f"It must implement the {count} approved test case(s) listed in this report, "
            "without changing what they require."
        )
    action.append(_HOW_SEALED_FILES_OPEN)
    return RecommendedAction(
        problem=", ".join(diagnosis.files) or f"The {diagnosis.scope} suite for {diagnosis.step_id}",
        error="; ".join(dict.fromkeys(errors[:2] + diagnosis.static_findings)),
        why_it_matters=(
            "The authoritative acceptance suite cannot execute, so the implementation cannot "
            "be evaluated. Repairing the product code cannot change this."
        ),
        action=" ".join(action),
        files=diagnosis.files,
    )


def _failing_tests_action(diagnosis: SuiteDiagnosis, inputs: ReportInputs) -> RecommendedAction:
    names = [item.name for item in diagnosis.failed_tests][:6]
    files = [f"workspace/{path}" for path in inputs.planned_files]
    first = diagnosis.failed_tests[0] if diagnosis.failed_tests else None
    regression = diagnosis.scope == "regression"
    return RecommendedAction(
        problem=(
            f"{diagnosis.failed} test(s) failed in the {diagnosis.scope} suite for "
            f"{diagnosis.step_id}: {', '.join(names) or 'see the output below'}."
        ),
        error=(first.message if first and first.message else "; ".join(diagnosis.errors[:2])),
        why_it_matters=(
            "A step that already worked no longer does, so this change broke earlier behaviour."
            if regression
            else "The step cannot complete until its approved acceptance tests pass."
        ),
        action=(
            "Compare each failing test with the acceptance criteria in this report, then fix "
            "the product code. If a test itself is wrong, you can correct how it checks the "
            f"criterion, but not what the criterion requires. {_HOW_SEALED_FILES_OPEN}"
        ),
        files=files + [path for path in diagnosis.files if path not in files],
    )


def _missing_deliverable_action(inputs: ReportInputs) -> RecommendedAction | None:
    blocked = inputs.blocked
    agent = role_label(blocked.role)
    if blocked.expected_files and inputs.step is not None:
        files = [f"verification/{inputs.step.step_id}/{path}" for path in blocked.expected_files]
        cases = len(inputs.test_plan.test_cases) if inputs.test_plan else 0
        return RecommendedAction(
            problem=f"{agent} could not produce a usable file: {', '.join(files)}",
            why_it_matters=(
                "These tests decide whether the step is finished, so the step cannot go "
                "forward without them."
            ),
            action=(
                f"Create the file yourself, implementing the {cases} approved test case(s) "
                "listed in this report, then use /verify. Or use /retry to give the models "
                "another attempt, which would replace a file written there."
            ),
            files=files,
        )
    if blocked.artifact:
        fields = ", ".join(blocked.artifact_fields)
        action = (
            f"Use /retry to give the models another attempt. Or write the result yourself as "
            f"JSON in {blocked.artifact}"
        )
        action += f", with the fields {fields}, " if fields else ", "
        return RecommendedAction(
            problem=f"{agent} did not produce the {_task_label(blocked)}.",
            why_it_matters="The next agent in the workflow needs this result to continue.",
            action=f"{action}then use /verify.",
            files=[blocked.artifact],
        )
    return None


def build_report(inputs: ReportInputs) -> InterventionReport:
    """Assemble the report from recorded facts only."""
    blocked = inputs.blocked
    agent = role_label(blocked.role) if blocked.role else "The team"
    diagnoses = _diagnose_evidence(inputs)

    facts: list[str] = [*_attempt_facts(inputs)]
    errors: list[str] = []
    deliverables: list[str] = []
    failed_tests: list[FailedTest] = []
    actions: list[RecommendedAction] = []
    unconfirmed: list[str] = []
    files: list[str] = []

    if blocked.failure_class in _UNCONFIRMED_BY_FAILURE:
        unconfirmed.extend(_UNCONFIRMED_BY_FAILURE[blocked.failure_class])

    for diagnosis in diagnoses:
        facts.append(
            f"The {diagnosis.scope} suite for {diagnosis.step_id} ran with exit code "
            f"{diagnosis.exit_code}: {diagnosis.passed} passed, {diagnosis.failed} failed"
            + (", and it timed out." if diagnosis.timed_out else ".")
        )
        facts.extend(diagnosis.static_findings)
        facts.extend(f"Reported error: {line}" for line in diagnosis.errors[:3])
        errors.extend(diagnosis.errors)
        failed_tests.extend(diagnosis.failed_tests)
        files.extend(diagnosis.files)
        profile = inputs.profiles.get(
            next(
                (run.profile_id for run in inputs.evidence.runs if run.step_id == diagnosis.step_id),
                "",
            )
            if inputs.evidence
            else ""
        )
        if diagnosis.executable:
            actions.append(_failing_tests_action(diagnosis, inputs))
            unconfirmed.append(
                "The fault may be in the product code, or a test's expectation may be wrong. "
                "The test output alone cannot tell which."
            )
        else:
            facts.append(
                f"The {diagnosis.scope} suite for {diagnosis.step_id} could not run its tests."
            )
            actions.append(_unrunnable_action(diagnosis, inputs, profile))

    for item in inputs.attempts:
        if item.status == "failed" and item.error:
            errors.append(f"Attempt {item.overall_attempt} on {item.model}: {item.error}")

    progress = inputs.progress
    if progress is not None and blocked.kind == BlockKind.VERIFICATION_FAILED:
        facts.append(
            f"{progress.repair_cycles_used} of {progress.repair_cycle_limit} repair cycles were used."
        )
    if blocked.kind == BlockKind.CONDITIONS_UNMET:
        facts.append(blocked.message)
        actions.append(
            RecommendedAction(
                problem="The tests passed, but a completion condition was judged unmet.",
                why_it_matters="The roadmap step promised this condition.",
                action=(
                    "Check whether the condition is really unmet. Fix the code and use /verify, "
                    "use /accept to take the step as it is, or use /change to alter the plan."
                ),
                files=[f"workspace/{path}" for path in inputs.planned_files],
            )
        )
    if blocked.kind == BlockKind.INTEGRITY:
        facts.append(blocked.message)
        files.extend(inputs.changed_sealed_files)
        actions.append(
            RecommendedAction(
                problem="Approved material no longer matches its recorded hash.",
                error=blocked.message,
                why_it_matters=(
                    "The team may only build and verify against exactly what was approved."
                ),
                action=(
                    "If you changed a sealed test on purpose, use /verify to review the "
                    "differences. If you changed an approved document, restore it, or use "
                    "/change to ask for a new proposal."
                ),
                files=inputs.changed_sealed_files,
            )
        )
    if blocked.failure_class == FailureClass.DEPENDENCY and not diagnoses:
        facts.append(blocked.message)
        errors.append(blocked.message)
        actions.append(
            RecommendedAction(
                problem="A tool the verification needs is missing.",
                error=blocked.message,
                why_it_matters="The authoritative tests cannot run without it.",
                action=_ACTION_BY_FAILURE[FailureClass.DEPENDENCY],
            )
        )

    if not diagnoses and blocked.task_id:
        missing = _missing_deliverable_action(inputs)
        if missing is not None:
            actions.append(missing)
            files.extend(missing.files)
            deliverables.append(missing.problem)
        else:
            advice = _ACTION_BY_FAILURE.get(blocked.failure_class)  # type: ignore[arg-type]
            actions.append(
                RecommendedAction(
                    problem=f"{agent} could not finish the {_task_label(blocked)}.",
                    why_it_matters="The workflow cannot continue past this task.",
                    action=advice
                    or (
                        "Change the file named in the errors, then use /verify to run the "
                        "authoritative tests, or use /retry for another model attempt."
                    ),
                    files=[f"workspace/{path}" for path in inputs.planned_files],
                )
            )
            files.extend(f"workspace/{path}" for path in inputs.planned_files)
    if blocked.failure_class in _ACTION_BY_FAILURE and actions:
        advice = _ACTION_BY_FAILURE[blocked.failure_class]
        if advice not in actions[0].action:
            actions[0].action = f"{advice} {actions[0].action}"

    if not actions:
        facts.append(blocked.message)
        actions.append(
            RecommendedAction(
                problem=blocked.message,
                why_it_matters="The workflow cannot continue until this is resolved.",
                action="Use /retry for another attempt, or /verify after fixing it yourself.",
            )
        )

    commands = ["/verify  after your fix, to run the checks and continue"]
    if blocked.kind != BlockKind.INTEGRITY:
        commands.append("/retry   to authorize another AI attempt instead")
    if blocked.kind in (BlockKind.VERIFICATION_FAILED, BlockKind.CONDITIONS_UNMET):
        commands.append("/change  to ask for a new proposal")
    if blocked.kind == BlockKind.CONDITIONS_UNMET:
        commands.append("/accept  to take the step with its limitations")

    return InterventionReport(
        intervention_id=inputs.intervention_id,
        version=inputs.version,
        product=inputs.product,
        slug=inputs.slug,
        step_number=inputs.step.number if inputs.step else None,
        step_name=inputs.step.name if inputs.step else None,
        phase=blocked.resume_phase.value,
        block_kind=blocked.kind.value,
        responsible_agent=agent,
        preferred_model=inputs.chain[0] if inputs.chain else None,
        fallback_model=inputs.chain[1] if len(inputs.chain) > 1 else None,
        attempts_used=len([item for item in inputs.attempts if item.status == "failed"]),
        attempts_allowed=inputs.attempts_allowed,
        repair_cycles_used=progress.repair_cycles_used if progress else 0,
        repair_cycles_allowed=progress.repair_cycle_limit if progress else 0,
        task_id=blocked.task_id,
        execution_mode=inputs.execution_mode,
        headline=inputs.note or blocked.message,
        files=list(dict.fromkeys(files + [path for item in actions for path in item.files])),
        errors=_unique(errors, MAX_ERROR_LINES),
        attempts=[
            AttemptSummary(
                model=item.model,
                attempt=item.overall_attempt,
                status=item.status,
                failure_class=item.failure_class.value if item.failure_class else None,
                error=item.error or "",
            )
            for item in inputs.attempts
        ],
        failed_tests=failed_tests[:MAX_FAILED_TESTS],
        suites=diagnoses,
        deliverable_problems=deliverables,
        requirements=[
            f"{item.id}: {item.title}. {item.description}"
            for item in (inputs.step_brief.requirements if inputs.step_brief else [])
        ],
        acceptance_criteria=_acceptance_lines(inputs.test_plan),
        test_cases=_test_case_lines(inputs.test_plan),
        preserved_outputs=inputs.preserved_outputs,
        manual_changes=inputs.manual_changes,
        confirmed_facts=list(dict.fromkeys(facts)),
        agent_diagnosis=inputs.agent_notes,
        unconfirmed_causes=list(dict.fromkeys(unconfirmed)),
        actions=actions,
        next_commands=commands,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _list(items: list[str], empty: str) -> str:
    return "\n".join(f"- {item}" for item in items) if items else empty


def _failed_test_lines(tests: list[FailedTest]) -> list[str]:
    lines = []
    for item in tests:
        line = item.name + (f" ({item.file})" if item.file else "")
        if item.message:
            line += f": {item.message}"
        if item.expected or item.actual:
            line += f" Expected {item.expected or 'unknown'}, got {item.actual or 'unknown'}."
        lines.append(line)
    return lines


def _attempts_label(report: InterventionReport) -> str:
    if report.attempts_allowed:
        return f"{report.attempts_used} of {report.attempts_allowed} model attempts"
    if report.repair_cycles_allowed:
        return f"{report.repair_cycles_used} of {report.repair_cycles_allowed} repair cycles"
    return ""


def report_markdown(report: InterventionReport) -> str:
    """The full report, as saved in the product folder."""
    step = (
        f"Step {report.step_number}: {report.step_name}" if report.step_number else "None yet"
    )
    models = report.preferred_model or "not applicable"
    if report.fallback_model:
        models = f"{report.preferred_model}, then {report.fallback_model}"
    attempts = _attempts_label(report) or "not applicable"

    sections = [
        f"# Human Intervention Report {report.intervention_id}",
        f"**BLOCKED / HUMAN INTERVENTION REQUIRED**, report version {report.version}.\n\n"
        f"{report.headline}",
        "## A. Context\n\n"
        + "\n".join(
            [
                f"- Product: {report.product}",
                f"- Roadmap step: {step}",
                f"- Workflow phase: {report.phase}",
                f"- Responsible agent: {report.responsible_agent}",
                f"- Models, preferred then fallback: {models}",
                f"- Attempts used: {attempts}",
                f"- Execution mode: {report.execution_mode}",
            ]
        ),
        "## B. Exact problem",
        f"### Files involved\n\n{_list(report.files, 'None identified.')}",
        f"### Errors\n\n{_list(report.errors, 'None recorded.')}",
    ]
    if report.failed_tests:
        sections.append(
            "### Failed tests\n\n" + _list(_failed_test_lines(report.failed_tests), "")
        )
    if report.deliverable_problems:
        sections.append(
            "### Missing or invalid deliverables\n\n" + _list(report.deliverable_problems, "")
        )
    if report.requirements:
        sections.append("### Requirements\n\n" + _list(report.requirements, ""))
    if report.acceptance_criteria:
        sections.append(
            "### Approved acceptance criteria\n\n" + _list(report.acceptance_criteria, "")
        )
    if report.test_cases:
        sections.append("### Approved test cases\n\n" + _list(report.test_cases, ""))
    for suite in report.suites:
        if suite.diagnostic_output:
            sections.append(
                f"### Diagnostic output, {suite.scope} suite {suite.step_id}\n\n"
                f"Command: `{suite.command}`\n\n```text\n{suite.diagnostic_output}\n```"
            )
    if report.preserved_outputs:
        sections.append(
            "### Preserved model outputs\n\n" + _list(report.preserved_outputs, "")
        )
    if report.manual_changes:
        sections.append(
            "### Your changes so far\n\n"
            + _list([f"{item.path} ({item.status})" for item in report.manual_changes], "")
        )
    sections.extend(
        [
            "## C. Root-cause assessment",
            "### Confirmed facts\n\nFrom test results, runtime errors and recorded attempts.\n\n"
            + _list(report.confirmed_facts, "None recorded."),
            "### The agents' diagnosis\n\nWhat the agents said. This is their opinion and "
            "has not been verified.\n\n"
            + _list(report.agent_diagnosis, "The agents offered no diagnosis."),
            "### Possible causes, not confirmed\n\nThese fit the evidence, but nothing has "
            "shown them to be true.\n\n"
            + _list(report.unconfirmed_causes, "None suggested by the evidence."),
            "## D. Recommended manual intervention",
        ]
    )
    for index, item in enumerate(report.actions, start=1):
        block = [f"### {index}. Problem\n\n{item.problem}"]
        if item.error:
            block.append(f"**Error:** {item.error}")
        block.append(f"**Why it matters:** {item.why_it_matters}")
        block.append(f"**Recommended action:** {item.action}")
        if item.files:
            block.append("**Files that need attention:**\n" + _list(item.files, ""))
        sections.append("\n\n".join(block))
    sections.append("## When you are ready\n\n" + _list(report.next_commands, ""))
    return "\n\n".join(sections) + "\n"


def terminal_summary(report: InterventionReport, location: str) -> str:
    """The useful part of the report, short enough to read in the terminal."""
    step = f"Step {report.step_number}: {report.step_name}" if report.step_number else "Before Step 1"
    lines = [
        "BLOCKED / HUMAN INTERVENTION REQUIRED",
        f"{step}, {report.responsible_agent}.",
    ]
    if _attempts_label(report):
        lines.append(f"Used: {_attempts_label(report)}.")
    for item in report.actions[:3]:
        lines.append("")
        lines.append(f"Problem: {item.problem}")
        if item.error:
            lines.append(f"Error: {item.error}")
        lines.append(f"Why it matters: {item.why_it_matters}")
        lines.append(f"Recommended action: {item.action}")
        if item.files:
            lines.append("Files that need attention:")
            lines.extend(f"  {path}" for path in item.files)
    lines.append("")
    lines.append(f"Full report: {location}")
    lines.append("You can leave now and come back: everything is saved exactly as it is.")
    return "\n".join(lines)


def revision_summary(diff_excerpt: str, location: str, warnings: list[str],
                     criteria: list[str], changes: list[str]) -> str:
    """What a person must see before a revised suite can become authoritative."""
    lines = ["You changed sealed tests, so the suite needs your approval before it is used.", ""]
    lines.append("Changed files, with their old and new hashes:")
    lines.extend(f"  {item}" for item in changes)
    lines.extend(["", "The differences:", diff_excerpt.rstrip(), f"(Full diff: {location})"])
    if criteria:
        lines.extend(["", "These approved acceptance criteria stay authoritative:"])
        lines.extend(f"  {item}" for item in criteria)
    if warnings:
        lines.extend(["", "This edit may weaken the tests:"])
        lines.extend(f"  - {item}" for item in warnings)
    return "\n".join(lines)
