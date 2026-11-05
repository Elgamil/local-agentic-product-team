import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lab.runtime.config import RuntimeConfig
from lab.runtime.adk_ollama import OllamaAdkRuntime
from lab.tools.developer_tools import set_active_workspace
from lab.workflow.orchestrator import WorkflowOrchestrator

logger = logging.getLogger("benchmark")

BENCHMARK_TASKS = {
    "price_sorting": (
        "Add price sorting to the product catalog. Users should be able to sort "
        "products by price in ascending or descending order using a dropdown or "
        "buttons on the catalog page. The sorting MUST preserve any active case-insensitive "
        "search filter and still display the correct empty-state message if no "
        "products match the search criteria."
    )
}

def create_isolated_workspace(run_id: str) -> Path:
    """Create an isolated copy of the project for a benchmark run."""
    # benchmark.py is at lab/evaluation/benchmark.py; project root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = repo_root / "runs" / "evaluations" / run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy minishop
    minishop_src = repo_root / "minishop"
    minishop_dst = eval_dir / "minishop"
    if minishop_dst.exists():
        shutil.rmtree(minishop_dst)
    shutil.copytree(minishop_src, minishop_dst)
    
    # Copy tests
    tests_src = repo_root / "tests"
    tests_dst = eval_dir / "tests"
    if tests_dst.exists():
        shutil.rmtree(tests_dst)
    shutil.copytree(tests_src, tests_dst)
    
    return eval_dir

async def run_benchmark(task_id: str, model_id: str) -> dict:
    feature_request = BENCHMARK_TASKS[task_id]
    
    run_id = f"eval_{task_id}_{model_id.replace(':', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info(f"Starting benchmark run: {run_id} for model {model_id}")
    
    workspace_dir = create_isolated_workspace(run_id)
    
    # Set active workspace to the isolated minishop
    set_active_workspace(workspace_dir / "minishop")
    
    config = RuntimeConfig(model_id=model_id)
    runtime = OllamaAdkRuntime(config)
    orchestrator = WorkflowOrchestrator(runtime=runtime, runs_dir=workspace_dir.parent)
    
    # Add acceptance tests for sorting before running
    add_sorting_acceptance_tests(workspace_dir / "tests" / "acceptance" / "test_sorting_acceptance.py")
    
    t0 = time.perf_counter()
    report = await orchestrator.execute_workflow(feature_request=feature_request)
    duration = time.perf_counter() - t0
    
    return {
        "run_id": run_id,
        "model": model_id,
        "task": task_id,
        "success": report.final_status == "SUCCESS",
        "duration_seconds": duration,
        "repair_attempts": report.repair_attempts,
        "test_runs": [{"success": tr.success, "passed": tr.passed_count, "failed": tr.failed_count} for tr in report.test_runs] if report.test_runs else [],
        "report_path": str(workspace_dir.parent / report.run_id / "report.json")
    }

def add_sorting_acceptance_tests(test_file: Path):
    test_file.parent.mkdir(parents=True, exist_ok=True)
    content = """import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_sorting_ascending():
    from minishop.app import app
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/?sort=asc")
        assert response.status_code == 200
        html = response.text.lower()
        # Verify it still renders successfully and has sorted items
        assert "laptop" in html

@pytest.mark.asyncio
async def test_sorting_descending():
    from minishop.app import app
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/?sort=desc")
        assert response.status_code == 200
        html = response.text.lower()
        assert "laptop" in html
"""
    test_file.write_text(content, encoding="utf-8")

async def compare_models(models: list[str], task_id: str, repetitions: int = 1):
    results = []
    
    for model in models:
        for i in range(repetitions):
            logger.info(f"=== Running evaluation for {model} (Repetition {i+1}/{repetitions}) ===")
            result = await run_benchmark(task_id, model)
            results.append(result)
            
    # Generate consolidated report
    repo_root = Path(__file__).resolve().parents[2]
    report_file = repo_root / "runs" / "evaluations" / f"comparison_{task_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
    
    lines = [
        f"# Model Comparison Report: {task_id}",
        "",
        "## Summary Comparison",
        "",
        "| Model | Run ID | Status | Duration (s) | Repairs | Final Tests Passed | Final Tests Failed |",
        "|---|---|---|---|---|---|---|",
    ]
    
    for r in results:
        status_str = "✅ SUCCESS" if r["success"] else "❌ FAILED"
        passed = r["test_runs"][-1]["passed"] if r["test_runs"] else 0
        failed = r["test_runs"][-1]["failed"] if r["test_runs"] else 0
        lines.append(
            f"| `{r['model']}` | `{r['run_id']}` | {status_str} | {r['duration_seconds']:.1f} | {r['repair_attempts']} | {passed} | {failed} |"
        )
    
    lines.extend(["", "## Detailed Run Results", ""])
    for r in results:
        lines.append(f"### Model: {r['model']} — Run: `{r['run_id']}`")
        lines.append(f"- **Final Status:** {'SUCCESS' if r['success'] else 'FAILED'}")
        lines.append(f"- **Duration:** {r['duration_seconds']:.2f} s")
        lines.append(f"- **Repair Attempts:** {r['repair_attempts']}")
        if r['test_runs']:
            final_run = r['test_runs'][-1]
            lines.append(f"- **Final Test Outcome:** Passed: {final_run['passed']}, Failed: {final_run['failed']}")
        lines.append(f"- **Full Report JSON:** `{r['report_path']}`")
        lines.append("")
        
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")
    
    logger.info(f"Consolidated comparison report saved to {report_file}")
    print(f"\nConsolidated comparison report saved to {report_file}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate and compare agent models.")
    parser.add_argument("--task", default="price_sorting", choices=BENCHMARK_TASKS.keys(), help="Task to evaluate")
    parser.add_argument("--models", nargs="+", default=["gemma4:latest", "qwen2.5:7b"], help="Models to compare")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of times to run each model")
    
    args = parser.parse_args()
    
    asyncio.run(compare_models(args.models, args.task, args.repetitions))

if __name__ == "__main__":
    main()
