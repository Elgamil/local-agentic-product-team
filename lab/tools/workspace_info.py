"""Permitted local tool: workspace information.

Safety constraints:
- Returns only lightweight runtime and workspace-level metadata.
- Does NOT enumerate the home directory.
- Does NOT inspect files outside the project workspace.
- Does NOT execute shell commands.
- Does NOT make network calls.
- Does NOT write files or modify the repository.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def get_workspace_info() -> dict:
    """Return a small, safe snapshot of the current runtime and workspace.

    This is the only tool exposed to the ADK agent in Milestone 1.

    Returns:
        A dict with keys:
            python_version  – e.g. "3.11.14"
            platform        – OS name and version
            workspace_path  – absolute path of the project workspace
            file_count      – number of *.py files directly under workspace
            venv_active     – True if running inside a virtual environment
    """
    workspace = Path(__file__).resolve().parents[2]  # project root

    # Count only .py files at the top two levels of the workspace.
    # Never recurse into the home directory or outside the workspace.
    py_files = list(workspace.glob("*.py")) + list(workspace.glob("*/*.py")) + list(workspace.glob("*/*/*.py"))

    return {
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "workspace_path": str(workspace),
        "file_count": len(py_files),
        "venv_active": sys.prefix != sys.base_prefix,
    }
