"""Ollama model residency: see what is loaded, and release it.

Unloading uses Ollama's documented mechanism: a generate request naming the model,
with no prompt and ``keep_alive: 0``, expires the loaded runner.  It never performs
inference, and it never loads a model that is not already resident.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class ModelLifecycle(Protocol):
    async def running(self) -> list[str]:
        """Return the ids of models currently loaded."""

    async def unload(self, model_id: str) -> bool:
        """Release *model_id*.  Return True once it is confirmed gone."""


class OllamaLifecycle:
    """ModelLifecycle backed by Ollama's HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        unload_wait_seconds: float = 20.0,
        client_timeout_seconds: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._unload_wait = unload_wait_seconds
        self._timeout = client_timeout_seconds

    async def running(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/api/ps")
            response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]

    async def unload(self, model_id: str) -> bool:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={"model": model_id, "keep_alive": 0},
            )
            response.raise_for_status()

        deadline = time.monotonic() + self._unload_wait
        while True:
            if model_id not in await self.running():
                return True
            if time.monotonic() >= deadline:
                logger.warning("Ollama still reports %s as loaded after unload", model_id)
                return False
            await asyncio.sleep(0.25)

    async def installed(self) -> list[str]:
        """Return the installed model ids (metadata only)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]
