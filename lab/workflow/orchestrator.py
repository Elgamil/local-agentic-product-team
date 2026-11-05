"""Milestone 2 Workflow Orchestrator.

Implements a deterministic, sequential multi-agent software delivery workflow:
  1. Receive feature request.
  2. Run Business Analyst (produces RequirementsArtifact).
  3. Validate and save requirements artifact.
  4. Run Software Architect (produces ArchitectureArtifact).
  5. Validate and save architecture artifact.
  6. Run Test Engineer (produces TestPlanArtifact).
  7. Validate and save test plan artifact.
  8. Run Developer (implements MiniShop Lite product catalog via controlled tools).
  9. Execute automated acceptance tests via controlled test runner.
 10. If tests fail, execute bounded repair attempts (at most 2).
 11. Generate and save comprehensive run report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.roles import get_role_prompt
from lab.runtime.adk_ollama import OllamaAdkRuntime
from lab.runtime.base import ModelRuntime
from lab.runtime.config import RuntimeConfig, default_config
from lab.schemas.artifacts import (
    ArchitectureArtifact,
    DeveloperRunArtifact,
    FileOperationRecord,
    RequirementsArtifact,
    TestPlanArtifact,
    TestRunResult,
    WorkflowRunReport,
    WorkflowStageResult,
)
from lab.tools.developer_tools import (
    DEFAULT_WORKSPACE_ROOT,
    ProjectWorkspace,
    get_active_workspace,
    list_project_files,
    read_project_file,
    set_active_workspace,
    write_project_file,
)
from lab.tools.test_runner import run_controlled_tests
from lab.workflow.logger import EventLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("lab.workflow.orchestrator")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_FEATURE_REQUEST = (
    "Build MiniShop Lite: a clean, local e-commerce demo implementing the product catalog feature.\n"
    "Requirements:\n"
    "- Display a list of sample products.\n"
    "- Show each product's name, description, price, and stock status (In Stock / Out of Stock).\n"
    "- Search products by name (case-insensitive substring match).\n"
    "- Display a clear, user-friendly message when no products match.\n"
    "- Store products in a local SQLite database ('minishop.db').\n"
    "- Include sample seed data with diverse products so the app works immediately on startup.\n"
    "- Technology stack: Python, FastAPI, Jinja2, SQLite, and pytest.\n"
    "- Provide clean, accessible UI styling with CSS and responsive layout."
)


class WorkflowOrchestrator:
    """Coordinates the 4-agent development lifecycle and test execution."""

    def __init__(
        self,
        runtime: ModelRuntime | None = None,
        workspace_dir: Path | str | None = None,
        test_target: str | Path | None = None,
        runs_dir: Path | str | None = None,
    ) -> None:
        self.runtime = runtime or OllamaAdkRuntime()
        self.workspace_dir = Path(workspace_dir or DEFAULT_WORKSPACE_ROOT).resolve()
        self.test_target = str(test_target or "tests/acceptance")
        self.runs_dir = Path(runs_dir or RUNS_DIR).resolve()

        # Build list of protected paths so the agent cannot modify test files
        protected_paths: list[Path] = []
        raw_target = Path(self.test_target)
        if not raw_target.is_absolute():
            protected_paths.append((PROJECT_ROOT / raw_target).resolve())
        else:
            protected_paths.append(raw_target.resolve())

        set_active_workspace(self.workspace_dir, protected_paths=protected_paths)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: Business Analyst
    # ------------------------------------------------------------------

    async def run_business_analyst(
        self,
        feature_request: str,
        run_id: str,
    ) -> tuple[RequirementsArtifact, WorkflowStageResult]:
        """Execute Agent 1 — Business Analyst to produce RequirementsArtifact."""
        logger.info(">>> Stage 1: Running Business Analyst agent...")
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        instruction = get_role_prompt("business_analyst")

        user_prompt = (
            f"Feature Request:\n{feature_request}\n\n"
            "Analyze the above feature request. Produce a complete RequirementsArtifact with "
            "functional requirements (FRs), acceptance criteria (ACs) in Given/When/Then format, "
            "and identified edge cases."
        )

        artifact = await self.runtime.complete_structured_schema(
            prompt=user_prompt,
            schema=RequirementsArtifact,
            instruction=instruction,
            session_id=f"ba_{run_id}",
        )

        finished_at = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - t0) * 1000

        artifact_path = ARTIFACTS_DIR / f"requirements_{run_id}.json"
        artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved RequirementsArtifact to %s", artifact_path)

        try:
            rel_path = str(artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = str(artifact_path.absolute())

        stage_result = WorkflowStageResult(
            stage_name="business_analyst",
            agent_role="Business Analyst",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            success=True,
            artifact_path=rel_path,
            summary=f"Defined {len(artifact.functional_requirements)} FRs, {len(artifact.acceptance_criteria)} ACs, and {len(artifact.edge_cases)} edge cases.",
        )
        return artifact, stage_result

    # ------------------------------------------------------------------
    # Stage 2: Software Architect
    # ------------------------------------------------------------------

    async def run_software_architect(
        self,
        requirements: RequirementsArtifact,
        run_id: str,
    ) -> tuple[ArchitectureArtifact, WorkflowStageResult]:
        """Execute Agent 2 — Software Architect to produce ArchitectureArtifact."""
        logger.info(">>> Stage 2: Running Software Architect agent...")
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        instruction = get_role_prompt("software_architect")

        req_json = requirements.model_dump_json(indent=2)
        user_prompt = (
            f"Validated Requirements Artifact:\n{req_json}\n\n"
            "Design the complete technical architecture for the required system. "
            "Specify modular components (files relative to workspace root), data model entities, "
            "API endpoints or interface contracts, architectural decisions (ADRs), and "
            "an ordered sequence of implementation steps for the Developer.\n"
            "Ensure the JSON strictly includes: system_overview, decisions, components, data_model, api_endpoints, and implementation_steps."
        )

        artifact = await self.runtime.complete_structured_schema(
            prompt=user_prompt,
            schema=ArchitectureArtifact,
            instruction=instruction,
            session_id=f"architect_{run_id}",
        )

        finished_at = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - t0) * 1000

        artifact_path = ARTIFACTS_DIR / f"architecture_{run_id}.json"
        artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved ArchitectureArtifact to %s", artifact_path)

        try:
            rel_path = str(artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = str(artifact_path.absolute())

        stage_result = WorkflowStageResult(
            stage_name="software_architect",
            agent_role="Software Architect",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            success=True,
            artifact_path=rel_path,
            summary=f"Designed {len(artifact.components)} components, {len(artifact.data_model)} entities, and {len(artifact.api_endpoints)} endpoints.",
        )
        return artifact, stage_result

    # ------------------------------------------------------------------
    # Stage 3: Test Engineer
    # ------------------------------------------------------------------

    async def run_test_engineer(
        self,
        requirements: RequirementsArtifact,
        run_id: str,
    ) -> tuple[TestPlanArtifact, WorkflowStageResult]:
        """Execute Agent 3 — Test Engineer to produce TestPlanArtifact."""
        logger.info(">>> Stage 3: Running Test Engineer agent...")
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        instruction = get_role_prompt("test_engineer")

        req_json = requirements.model_dump_json(indent=2)
        user_prompt = (
            f"Validated Requirements Artifact:\n{req_json}\n\n"
            "Read these requirements independently. Define 4 to 6 concise, authoritative test cases covering all "
            "acceptance criteria, core workflows, edge cases, and failure modes described in the requirements.\n"
            "Keep test case descriptions and steps concise."
        )

        artifact = await self.runtime.complete_structured_schema(
            prompt=user_prompt,
            schema=TestPlanArtifact,
            instruction=instruction,
            session_id=f"test_engineer_{run_id}",
        )

        finished_at = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - t0) * 1000

        artifact_path = ARTIFACTS_DIR / f"test_plan_{run_id}.json"
        artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved TestPlanArtifact to %s", artifact_path)

        try:
            rel_path = str(artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = str(artifact_path.absolute())

        stage_result = WorkflowStageResult(
            stage_name="test_engineer",
            agent_role="Test Engineer",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            success=True,
            artifact_path=rel_path,
            summary=f"Created test plan with {len(artifact.test_cases)} test cases covering all criteria.",
        )
        return artifact, stage_result

    # ------------------------------------------------------------------
    # Stage 4: Developer
    # ------------------------------------------------------------------

    async def run_developer(
        self,
        requirements: RequirementsArtifact,
        architecture: ArchitectureArtifact,
        test_plan: TestPlanArtifact,
        run_id: str,
    ) -> tuple[DeveloperRunArtifact, WorkflowStageResult]:
        """Execute Agent 4 — Developer to implement MiniShop Lite using controlled tools."""
        logger.info(">>> Stage 4: Running Developer agent...")
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        instruction = get_role_prompt("developer")

        # Standard components needed for MiniShop Lite
        target_files = [
            (
                "database.py",
                "SQLite database layer for MiniShop Lite.\n"
                "- Define DB_PATH = 'minishop.db'\n"
                "- Function init_db(db_path=DB_PATH): creates 'products' table with columns: "
                "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, description TEXT, price REAL NOT NULL, in_stock BOOLEAN NOT NULL. "
                "If table is empty, seeds at least 4 products: Quantum Laptop X1 ($1299.99, In Stock), 4K Ultra Monitor Z ($450.50, In Stock), "
                "Mechanical Keyboard K9 ($120.00, In Stock), and WebCam Pro Mini ($79.99, Out of Stock, in_stock=0).\n"
                "- Function get_all_products(db_path=DB_PATH) -> list: queries all products ordered by id.\n"
                "- Function search_products(query: str, db_path=DB_PATH) -> list: searches products WHERE name LIKE ? OR description LIKE ? ordered by name.\n"
                "Call write_project_file('database.py', content)."
            ),
            (
                "app.py",
                "FastAPI application for MiniShop Lite.\n"
                "- Imports os, from pathlib import Path, FastAPI, Request, HTMLResponse, StaticFiles, Jinja2Templates.\n"
                "- BASE_DIR = Path(__file__).resolve().parent\n"
                "- DB_PATH = str(BASE_DIR / 'minishop.db')\n"
                "- Imports init_db, get_all_products, search_products from minishop.database.\n"
                "- app = FastAPI(title='MiniShop Lite')\n"
                "- templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))\n"
                "- Mounts '/static' to StaticFiles(directory=str(BASE_DIR / 'static'), name='static')\n"
                "- On startup event or lifespan, calls init_db(DB_PATH).\n"
                "- Route GET '/' (response_class=HTMLResponse): accepts q: str | None = None. "
                "If q: products = search_products(q, db_path=DB_PATH). Else: products = get_all_products(db_path=DB_PATH). "
                "Renders 'catalog.html' with context {'request': request, 'products': products, 'query': q or ''}.\n"
                "- Route GET '/health': returns {'status': 'ok'}.\n"
                "Call write_project_file('app.py', content)."
            ),
            (
                "templates/base.html",
                "Jinja2 base layout template.\n"
                "- HTML5 document structure with head containing <title>MiniShop Lite</title> and <link rel='stylesheet' href='/static/style.css'>.\n"
                "- Header with <h1>MiniShop Lite</h1>.\n"
                "- Main container with {% block content %}{% endblock %}.\n"
                "- Footer with copyright notice.\n"
                "Call write_project_file('templates/base.html', content)."
            ),
            (
                "templates/catalog.html",
                "Jinja2 catalog template extending base.html.\n"
                "- {% extends 'base.html' %}\n"
                "- {% block content %}\n"
                "- Search form with method='GET' action='/' containing an input name='q' value='{{ query }}' and search button.\n"
                "- If products: render product grid. For each product, show name ({{ prod[1] if prod is sequence else prod.name }}), "
                "description ({{ prod[2] if prod is sequence else prod.description }}), price (${{ '%.2f'|format(prod[3] if prod is sequence else prod.price) }}), "
                "and stock badge: if (prod[4] if prod is sequence else prod.in_stock): <span class='badge in-stock'>In Stock</span> "
                "else: <span class='badge out-of-stock'>Out of Stock</span>.\n"
                "- Else: <div class='no-products'><p>No products match your search.</p></div>.\n"
                "- {% endblock %}\n"
                "Call write_project_file('templates/catalog.html', content)."
            ),
            (
                "static/style.css",
                "Modern, responsive CSS styling for MiniShop Lite.\n"
                "- Clean typography, layout container, header with gradient or solid accent.\n"
                "- Search form styling with styled input and button.\n"
                "- Product grid (grid/flex layout) with card shadows, hover effects.\n"
                "- Product title, description, bold price.\n"
                "- Stock badges: .in-stock (green text/background) and .out-of-stock (red text/background).\n"
                "- .no-products empty state alert styling.\n"
                "Call write_project_file('static/style.css', content)."
            ),
        ]

        implemented_files: list[str] = []
        workspace = get_active_workspace()

        for filename, description in target_files:
            logger.info("Developer agent implementing: %s", filename)
            dev_prompt = (
                f"Architecture Specification:\n{architecture.model_dump_json(indent=2)}\n\n"
                f"Task: Implement the file '{filename}' inside the project workspace.\n"
                f"Description: {description}\n\n"
                f"You MUST call the write_project_file tool with relative_path='{filename}' and the complete, "
                f"fully functional code/content."
            )

            await self.runtime.run_agent(
                instruction=instruction,
                user_message=dev_prompt,
                tools=[write_project_file, read_project_file, list_project_files],
                session_id=f"dev_{filename.replace('/', '_')}_{run_id}",
            )
            implemented_files.append(filename)

        finished_at = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - t0) * 1000

        dev_artifact = DeveloperRunArtifact(
            implemented_files=implemented_files,
            file_operations=list(workspace.operations_log),
            summary=f"Developer agent implemented {len(implemented_files)} files using controlled write_project_file tool.",
        )

        artifact_path = ARTIFACTS_DIR / f"developer_{run_id}.json"
        artifact_path.write_text(dev_artifact.model_dump_json(indent=2), encoding="utf-8")

        try:
            rel_path = str(artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = str(artifact_path.absolute())

        stage_result = WorkflowStageResult(
            stage_name="developer",
            agent_role="Senior Developer",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            success=True,
            artifact_path=rel_path,
            summary=f"Implemented {len(implemented_files)} files: {', '.join(implemented_files)}.",
        )
        return dev_artifact, stage_result

    # ------------------------------------------------------------------
    # Stage 5 & 6: Test Execution & Bounded Repair
    # ------------------------------------------------------------------

    async def execute_tests_with_repair(
        self,
        run_id: str,
        max_repairs: int = 2,
        run_logger: EventLogger | None = None,
    ) -> tuple[list[TestRunResult], int, bool]:
        """Run acceptance tests; if failing, conduct at most max_repairs repair cycles."""
        logger.info(">>> Stage 5: Executing authoritative acceptance tests...")
        test_runs: list[TestRunResult] = []
        repairs_conducted = 0

        workspace_cwd = get_active_workspace().root.parent
        initial_run = run_controlled_tests(test_target="tests/acceptance", cwd=workspace_cwd)
        test_runs.append(initial_run)
        logger.info(
            "Initial test run: success=%s, passed=%d, failed=%d",
            initial_run.success,
            initial_run.passed_count,
            initial_run.failed_count,
        )
        if run_logger:
            run_logger.log_test_run(
                initial_run.command,
                initial_run.exit_code,
                initial_run.passed_count,
                initial_run.failed_count,
                initial_run.stdout,
                initial_run.stderr,
            )

        if initial_run.success:
            return test_runs, repairs_conducted, True

        instruction = get_role_prompt("developer")

        # Enter bounded repair loop
        for attempt in range(1, max_repairs + 1):
            repairs_conducted += 1
            logger.warning(
                ">>> Stage 6: Bounded repair attempt %d of %d...",
                attempt,
                max_repairs,
            )
            if run_logger:
                run_logger.log_agent_start(
                    "Senior Developer (Repair)",
                    f"Repair attempt {attempt} due to test failures.",
                )

            latest_failure = test_runs[-1]
            repair_prompt = (
                f"The authoritative acceptance tests FAILED with exit code {latest_failure.exit_code}.\n"
                f"Stdout:\n{latest_failure.stdout}\n"
                f"Stderr:\n{latest_failure.stderr}\n\n"
                "Please inspect the errors and the project files. Use write_project_file to fix "
                "the defects in minishop so the acceptance tests pass. You cannot modify the tests."
            )

            t0 = time.perf_counter()
            await self.runtime.run_agent(
                instruction=instruction,
                user_message=repair_prompt,
                tools=[write_project_file, read_project_file, list_project_files],
                session_id=f"repair_{attempt}_{run_id}",
            )
            if run_logger:
                duration_ms = (time.perf_counter() - t0) * 1000
                run_logger.log_agent_end(
                    "Senior Developer (Repair)", duration_ms, True, "repair_applied"
                )

            # Re-run automated tests
            retest_run = run_controlled_tests(test_target="tests/acceptance", cwd=workspace_cwd)
            test_runs.append(retest_run)
            logger.info(
                "Retest run %d: success=%s, passed=%d, failed=%d",
                attempt,
                retest_run.success,
                retest_run.passed_count,
                retest_run.failed_count,
            )
            if run_logger:
                run_logger.log_test_run(
                    retest_run.command,
                    retest_run.exit_code,
                    retest_run.passed_count,
                    retest_run.failed_count,
                    retest_run.stdout,
                    retest_run.stderr,
                )

            if retest_run.success:
                logger.info("Repairs succeeded on attempt %d!", attempt)
                return test_runs, repairs_conducted, True

        return test_runs, repairs_conducted, False

    # ------------------------------------------------------------------
    # Complete Workflow Execution
    # ------------------------------------------------------------------

    async def execute_workflow(
        self,
        feature_request: str = DEFAULT_FEATURE_REQUEST,
    ) -> WorkflowRunReport:
        """Run the end-to-end multi-agent delivery workflow."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        started_at = datetime.now(timezone.utc)
        logger.info("========== Starting Workflow Run: %s ==========", run_id)

        run_dir = self.runs_dir / run_id
        run_logger = EventLogger(run_dir)
        self.runtime._logger = run_logger

        run_logger.log_workflow_start(run_id, feature_request)

        # Ensure artifacts go into runs/<run_id>/artifacts
        global ARTIFACTS_DIR
        old_artifacts_dir = ARTIFACTS_DIR
        ARTIFACTS_DIR = run_dir / "artifacts"
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        report = WorkflowRunReport(
            run_id=run_id,
            model_id=self.runtime.model_id,
            runtime=self.runtime.runtime_label,
            feature_request=feature_request,
            started_at=started_at,
            summary="Workflow in progress",
        )

        requirements: RequirementsArtifact | None = None
        architecture: ArchitectureArtifact | None = None
        test_plan: TestPlanArtifact | None = None
        dev_artifact: DeveloperRunArtifact | None = None

        try:
            # Stage 1: Business Analyst
            stage_t0 = time.perf_counter()
            stage_start = datetime.now(timezone.utc)
            run_logger.log_agent_start("Business Analyst", "Generate RequirementsArtifact")
            try:
                requirements, ba_stage = await self.run_business_analyst(feature_request, run_id)
                report.add_stage(ba_stage)
                run_logger.log_agent_end(
                    "Business Analyst",
                    ba_stage.duration_ms,
                    ba_stage.success,
                    ba_stage.artifact_path or "",
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - stage_t0) * 1000
                ba_stage = WorkflowStageResult(
                    stage_name="business_analyst",
                    agent_role="Business Analyst",
                    started_at=stage_start,
                    finished_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(exc),
                    summary=f"Failed to produce RequirementsArtifact: {exc}",
                )
                report.add_stage(ba_stage)
                run_logger.log_stage_error("Business Analyst", "business_analyst", str(exc))
                run_logger.log_agent_end("Business Analyst", duration_ms, False)
                raise

            # Stage 2: Software Architect
            stage_t0 = time.perf_counter()
            stage_start = datetime.now(timezone.utc)
            run_logger.log_agent_start("Software Architect", "Generate ArchitectureArtifact")
            try:
                architecture, arch_stage = await self.run_software_architect(requirements, run_id)
                report.add_stage(arch_stage)
                run_logger.log_agent_end(
                    "Software Architect",
                    arch_stage.duration_ms,
                    arch_stage.success,
                    arch_stage.artifact_path or "",
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - stage_t0) * 1000
                arch_stage = WorkflowStageResult(
                    stage_name="software_architect",
                    agent_role="Software Architect",
                    started_at=stage_start,
                    finished_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(exc),
                    summary=f"Failed to produce ArchitectureArtifact: {exc}",
                )
                report.add_stage(arch_stage)
                run_logger.log_stage_error("Software Architect", "software_architect", str(exc))
                run_logger.log_agent_end("Software Architect", duration_ms, False)
                raise

            # Stage 3: Test Engineer
            stage_t0 = time.perf_counter()
            stage_start = datetime.now(timezone.utc)
            run_logger.log_agent_start("Test Engineer", "Generate TestPlanArtifact")
            try:
                test_plan, te_stage = await self.run_test_engineer(requirements, run_id)
                report.add_stage(te_stage)
                run_logger.log_agent_end(
                    "Test Engineer",
                    te_stage.duration_ms,
                    te_stage.success,
                    te_stage.artifact_path or "",
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - stage_t0) * 1000
                te_stage = WorkflowStageResult(
                    stage_name="test_engineer",
                    agent_role="Test Engineer",
                    started_at=stage_start,
                    finished_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(exc),
                    summary=f"Failed to produce TestPlanArtifact: {exc}",
                )
                report.add_stage(te_stage)
                run_logger.log_stage_error("Test Engineer", "test_engineer", str(exc))
                run_logger.log_agent_end("Test Engineer", duration_ms, False)
                raise

            # Stage 4: Developer
            stage_t0 = time.perf_counter()
            stage_start = datetime.now(timezone.utc)
            run_logger.log_agent_start("Senior Developer", "Implement MiniShop Lite")
            try:
                dev_artifact, dev_stage = await self.run_developer(
                    requirements, architecture, test_plan, run_id
                )
                report.add_stage(dev_stage)
                run_logger.log_agent_end(
                    "Senior Developer",
                    dev_stage.duration_ms,
                    dev_stage.success,
                    dev_stage.artifact_path or "",
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - stage_t0) * 1000
                dev_stage = WorkflowStageResult(
                    stage_name="developer",
                    agent_role="Senior Developer",
                    started_at=stage_start,
                    finished_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    success=False,
                    error=str(exc),
                    summary=f"Failed to implement project files: {exc}",
                )
                report.add_stage(dev_stage)
                run_logger.log_stage_error("Senior Developer", "developer", str(exc))
                run_logger.log_agent_end("Senior Developer", duration_ms, False)
                raise

            # Stage 5 & 6: Acceptance Testing & Bounded Repairs
            test_runs, repair_count, test_success = await self.execute_tests_with_repair(
                run_id, max_repairs=2, run_logger=run_logger
            )
            report.test_runs = test_runs
            report.repair_attempts = repair_count

            finished_at = datetime.now(timezone.utc)
            report.finished_at = finished_at
            report.total_duration_ms = (finished_at - started_at).total_seconds() * 1000

            if test_success:
                report.final_status = "SUCCESS"
                report.summary = (
                    f"Milestone 2 multi-agent workflow succeeded. "
                    f"All 4 roles executed via {self.runtime.model_id}. "
                    f"MiniShop Lite implemented and passed all {test_runs[-1].passed_count} acceptance tests."
                )
            else:
                report.final_status = "FAILED"
                report.summary = (
                    f"MiniShop Lite implementation completed but failed acceptance tests "
                    f"after {repair_count} repair attempts."
                )

        except Exception as exc:
            logger.exception("Workflow halted due to error: %s", exc)
            finished_at = datetime.now(timezone.utc)
            report.finished_at = finished_at
            report.total_duration_ms = (finished_at - started_at).total_seconds() * 1000
            report.final_status = "FAILED"
            report.summary = f"Workflow aborted due to error: {exc}"

        finally:
            run_logger.log_workflow_end(report.final_status, report.total_duration_ms or 0.0)

            # Save run report
            report_path = run_dir / "report.json"
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

            # Save Markdown report
            md_report_path = run_dir / "report.md"
            self._write_markdown_report(
                report,
                md_report_path,
                requirements,
                architecture,
                test_plan,
                dev_artifact,
            )

            logger.info("Saved final workflow run report to %s", report_path)

            # Also write to old path for backward compatibility
            old_report_path = self.runs_dir / f"milestone2_{run_id}.json"
            try:
                old_report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                pass

            ARTIFACTS_DIR = old_artifacts_dir

        return report

    def _write_markdown_report(
        self,
        report: WorkflowRunReport,
        path: Path,
        req: RequirementsArtifact | None,
        arch: ArchitectureArtifact | None,
        plan: TestPlanArtifact | None,
        dev: DeveloperRunArtifact | None,
    ) -> None:
        duration_s = (report.total_duration_ms or 0) / 1000
        md = [
            f"# Workflow Run Report: {report.run_id}",
            "",
            "## 1. Executive Summary",
            "",
            "| Attribute | Value |",
            "|---|---|",
            f"| **Run ID** | `{report.run_id}` |",
            f"| **Final Status** | **{report.final_status}** |",
            f"| **Model** | `{report.model_id}` |",
            f"| **Runtime** | `{report.runtime}` |",
            f"| **Duration** | {duration_s:.1f} s |",
            f"| **Acceptance Tests** | {report.test_runs[-1].passed_count if report.test_runs else 0} passed / {report.test_runs[-1].failed_count if report.test_runs else 0} failed |",
            f"| **Repair Attempts** | {report.repair_attempts} |",
            "",
            f"> **Summary:** {report.summary}",
            "",
            "## 2. Stage Execution Summary",
            "",
            "| Stage | Role | Status | Duration | Artifact / Summary |",
            "|---|---|---|---|---|",
        ]

        failed_stage_details = None
        for st in report.stages:
            st_status = "✅ PASS" if st.success else "❌ FAIL"
            st_info = st.artifact_path or st.summary or (f"Error: {st.error}" if st.error else "")
            md.append(f"| `{st.stage_name}` | {st.agent_role} | {st_status} | {st.duration_ms / 1000:.1f} s | {st_info} |")
            if not st.success and st.error:
                failed_stage_details = (st.agent_role, st.stage_name, st.error)

        md.append("")

        if failed_stage_details:
            role, stage_name, err = failed_stage_details
            md.extend([
                "## ⚠️ Stage Failure Diagnostics",
                "",
                f"- **Failed Role:** {role}",
                f"- **Stage Identifier:** `{stage_name}`",
                f"- **Error Message:**",
                "```text",
                err,
                "```",
                "",
            ])

        if req:
            md.extend([
                "## 3. Requirements & Acceptance Criteria (Business Analyst)",
                "",
                f"**Feature Title:** {req.feature_title}",
                "",
                f"**Summary:** {req.summary}",
                "",
                "### Functional Requirements",
                "| ID | Title | Description |",
                "|---|---|---|",
            ])
            for fr in req.functional_requirements:
                md.append(f"| `{fr.id}` | {fr.title} | {fr.description} |")

            md.extend([
                "",
                "### Acceptance Criteria (Given/When/Then)",
                "| ID | Scenario | Given | When | Then |",
                "|---|---|---|---|---|",
            ])
            for ac in req.acceptance_criteria:
                md.append(f"| `{ac.id}` | {ac.scenario} | {ac.given} | {ac.when} | {ac.then} |")

            if req.edge_cases:
                md.extend(["", "### Edge Cases Identified"])
                for ec in req.edge_cases:
                    md.append(f"- {ec}")
            md.append("")

        if arch:
            md.extend([
                "## 4. Architectural Specification (Software Architect)",
                "",
                f"**System Overview:**\n{arch.system_overview}",
                "",
            ])
            if arch.decisions:
                md.extend([
                    "### Architectural Decisions",
                    "| ID | Decision | Rationale | Trade-Offs |",
                    "|---|---|---|---|",
                ])
                for dec in arch.decisions:
                    md.append(f"| `{dec.id}` | {dec.decision} | {dec.rationale} | {dec.trade_offs} |")
                md.append("")

            md.extend([
                "### Components",
                "| Name | File Path | Purpose |",
                "|---|---|---|",
            ])
            for comp in arch.components:
                md.append(f"| {comp.name} | `{comp.file_path}` | {comp.purpose} |")
            md.append("")

            md.extend([
                "### Data Model",
                "| Table | Description | Fields |",
                "|---|---|---|",
            ])
            for ent in arch.data_model:
                fields_str = ", ".join(f"`{k}`: {v}" for k, v in ent.fields.items())
                md.append(f"| `{ent.table_name}` | {ent.description} | {fields_str} |")
            md.append("")

            md.extend([
                "### API Endpoints",
                "| Method | Path | Description | Query Params |",
                "|---|---|---|---|",
            ])
            for ep in arch.api_endpoints:
                params_str = ", ".join(ep.query_params) if ep.query_params else "None"
                md.append(f"| `{ep.method}` | `{ep.path}` | {ep.description} | {params_str} |")
            md.append("")

        if plan:
            md.extend([
                "## 5. Test Plan (Test Engineer)",
                "",
                f"**Strategy:** {plan.test_strategy}",
                "",
                f"**Summary:** {plan.test_plan_summary}",
                "",
                "### Test Cases",
                "| ID | Name | Target AC | Description | Expected Result |",
                "|---|---|---|---|---|",
            ])
            for tc in plan.test_cases:
                md.append(f"| `{tc.id}` | {tc.name} | `{tc.target_acceptance_criterion}` | {tc.description} | {tc.expected_result} |")
            md.append("")

        if dev:
            md.extend([
                "## 6. Implementation Summary (Senior Developer)",
                "",
                f"{dev.summary}",
                "",
                "### Implemented Files",
            ])
            for f in dev.implemented_files:
                md.append(f"- `{f}`")
            md.append("")

        if report.test_runs:
            md.extend([
                "## 7. Independent Test Execution",
                "",
            ])
            for idx, tr in enumerate(report.test_runs, 1):
                badge = "PASS" if tr.success else "FAIL"
                md.extend([
                    f"### Run {idx} — [{badge}]",
                    f"- **Command:** `{tr.command}`",
                    f"- **Exit Code:** `{tr.exit_code}`",
                    f"- **Results:** {tr.passed_count} passed, {tr.failed_count} failed ({tr.duration_ms:.0f} ms)",
                    "```text",
                    tr.stdout.strip()[:1000] if tr.stdout else tr.stderr.strip()[:1000],
                    "```",
                    "",
                ])

        path.write_text("\n".join(md), encoding="utf-8")


def print_workflow_summary(report: WorkflowRunReport) -> None:
    """Print an ASCII summary table of the workflow run."""
    print("\n" + "=" * 70)
    print(f"  Milestone 2 Multi-Agent Software Delivery Run: {report.run_id}")
    print(f"  Model ID:     {report.model_id}")
    print(f"  Runtime:      {report.runtime}")
    print(f"  Duration:     {(report.total_duration_ms or 0) / 1000:.1f} s")
    print(f"  Final Status: {report.final_status}")
    print("=" * 70)
    print("  STAGES:")
    for st in report.stages:
        status_str = "PASS" if st.success else "FAIL"
        summary_str = st.summary or st.error or ""
        print(f"    [{status_str:4s}] {st.agent_role:<22} ({st.duration_ms / 1000:.1f} s) — {summary_str}")
    print("-" * 70)
    print(f"  ACCEPTANCE TESTS: ({len(report.test_runs)} run(s), {report.repair_attempts} repair attempt(s))")
    for idx, tr in enumerate(report.test_runs, 1):
        status_str = "PASS" if tr.success else "FAIL"
        print(f"    Run {idx}: [{status_str:4s}] {tr.passed_count} passed, {tr.failed_count} failed ({tr.duration_ms:.0f} ms)")
    print("=" * 70)
    print(f"  SUMMARY: {report.summary}")
    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 2 Multi-Agent Delivery Workflow")
    parser.add_argument("--model", default=None, help="Override LAB_MODEL_ID")
    parser.add_argument("--base-url", default=None, help="Override OLLAMA_API_BASE")
    args = parser.parse_args()

    if args.model:
        os.environ["LAB_MODEL_ID"] = args.model
    if args.base_url:
        os.environ["OLLAMA_API_BASE"] = args.base_url

    config = default_config()
    runtime = OllamaAdkRuntime(config)
    orchestrator = WorkflowOrchestrator(runtime=runtime)

    report = asyncio.run(orchestrator.execute_workflow())
    print_workflow_summary(report)

    return 0 if report.final_status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
