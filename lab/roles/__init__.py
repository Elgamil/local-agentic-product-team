"""Role definitions and prompt loaders for the four specialist agents."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def get_role_prompt(role_name: str) -> str:
    """Load the markdown system prompt for a specified role.

    Args:
        role_name: One of 'business_analyst', 'software_architect', 'test_engineer', 'developer'
    """
    prompt_file = PROMPTS_DIR / f"{role_name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Role prompt not found for: {role_name} at {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")
