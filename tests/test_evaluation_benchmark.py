import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lab.evaluation.benchmark import create_isolated_workspace, run_benchmark

@pytest.fixture
def temp_workspace(tmp_path):
    repo_root = tmp_path / "repo"
    minishop = repo_root / "minishop"
    minishop.mkdir(parents=True)
    tests = repo_root / "tests"
    tests.mkdir(parents=True)
    (minishop / "app.py").write_text("print('hello')", encoding="utf-8")
    return repo_root

def test_create_isolated_workspace(temp_workspace):
    with patch("lab.evaluation.benchmark.Path") as mock_path:
        # Mock Path(__file__).resolve().parents[2] to return our temp_workspace
        mock_path.return_value.resolve.return_value.parents = [None, None, temp_workspace]
        
        eval_dir = create_isolated_workspace("test_run_123")
        
        assert eval_dir == temp_workspace / "runs" / "evaluations" / "test_run_123"
        assert (eval_dir / "minishop" / "app.py").exists()
        assert (eval_dir / "tests").exists()

@pytest.mark.asyncio
async def test_run_benchmark(temp_workspace):
    with patch("lab.evaluation.benchmark.create_isolated_workspace") as mock_create, \
         patch("lab.evaluation.benchmark.set_active_workspace") as mock_set, \
         patch("lab.workflow.orchestrator.WorkflowOrchestrator.execute_workflow") as mock_exec, \
         patch("lab.evaluation.benchmark.Path") as mock_path:
        
        eval_dir = temp_workspace / "runs" / "evaluations" / "test_run_123"
        (eval_dir / "tests" / "acceptance").mkdir(parents=True)
        mock_create.return_value = eval_dir
        
        class DummyReport:
            final_status = "SUCCESS"
            repair_attempts = 0
            test_runs = []
            run_id = "test_run_123"
            
        mock_exec.return_value = DummyReport()
        
        result = await run_benchmark("price_sorting", "dummy_model")
        
        assert result["model"] == "dummy_model"
        assert result["task"] == "price_sorting"
        assert result["success"] is True
        assert "duration_seconds" in result
        
        mock_set.assert_called_once_with(eval_dir / "minishop")
