"""pytest fixtures, markers, and shared utilities."""

from __future__ import annotations

import pytest
import httpx


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: integration tests requiring a running Ollama instance",
    )


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    return "http://localhost:11434"


@pytest.fixture(scope="session")
def ollama_available(ollama_base_url: str) -> bool:
    """Return True if Ollama is reachable; skip live tests if not."""
    try:
        resp = httpx.get(f"{ollama_base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=False)
def require_ollama(ollama_available: bool):
    """Session-scoped fixture that skips the test if Ollama is not reachable."""
    if not ollama_available:
        pytest.skip("Ollama is not reachable at localhost:11434 — skipping live tests")


@pytest.fixture(scope="session")
def runtime_config():
    from lab.runtime.config import RuntimeConfig
    return RuntimeConfig(
        model_id="gemma4:latest",
        ollama_base_url="http://localhost:11434",
        timeout_seconds=120,
    )
