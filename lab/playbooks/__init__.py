"""Category playbooks: reusable product guidance kept out of the orchestrator.

A playbook is one Markdown file with YAML front matter.  Adding a product category
means dropping a new file in this directory; no code changes.

Playbooks are internal guidance for the agents.  They are never shown to the user,
and they are not feature checklists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PLAYBOOKS_DIR = Path(__file__).resolve().parent
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")

REQUIRED_SECTIONS = (
    "considerations",
    "development_phases",
    "quality_requirements",
    "edge_cases",
    "acceptance_checks",
)


def _section_key(title: str) -> str:
    aliases = {
        "typical development phases": "development_phases",
        "common edge cases": "edge_cases",
        "suggested acceptance checks": "acceptance_checks",
    }
    normalized = title.strip().lower()
    return aliases.get(normalized, normalized.replace(" ", "_"))


@dataclass(frozen=True)
class Playbook:
    category_id: str
    title: str
    label: str
    keywords: tuple[str, ...]
    body: str
    sections: dict[str, str] = field(default_factory=dict)

    def section(self, key: str) -> str:
        return self.sections.get(key, "")


def parse_playbook(path: Path) -> Playbook:
    match = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Playbook {path.name} has no front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()

    pieces = _SECTION_HEADING.split(body)
    sections = {
        _section_key(pieces[index]): pieces[index + 1].strip()
        for index in range(1, len(pieces) - 1, 2)
    }
    missing = [key for key in REQUIRED_SECTIONS if key not in sections]
    if missing:
        raise ValueError(f"Playbook {path.name} is missing sections: {missing}")

    return Playbook(
        category_id=str(meta.get("id") or path.stem),
        title=str(meta["title"]),
        label=str(meta.get("label") or meta["title"]),
        keywords=tuple(str(word).lower() for word in meta.get("keywords", [])),
        body=body,
        sections=sections,
    )


class PlaybookLibrary:
    """Every playbook found in a directory."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory or PLAYBOOKS_DIR)
        self._playbooks = {
            playbook.category_id: playbook
            for playbook in (parse_playbook(path) for path in sorted(self.directory.glob("*.md")))
        }

    def ids(self) -> list[str]:
        return sorted(self._playbooks)

    def get(self, category_id: str | None) -> Playbook | None:
        return self._playbooks.get(category_id or "")

    def label_for(self, category_id: str | None) -> str:
        playbook = self.get(category_id)
        return playbook.label if playbook else "Category to be decided"

    def catalog(self) -> str:
        """Category ids and titles for model prompts."""
        return "\n".join(
            f"- {playbook.category_id}: {playbook.title}"
            for playbook in self._playbooks.values()
        )

    def suggest(self, text: str) -> str | None:
        """Best keyword match, used to pick guidance for the very first question."""
        words = set(_WORD.findall(text.lower()))
        lowered = text.lower()
        best_id, best_score = None, 0
        for playbook in self._playbooks.values():
            score = sum(
                1
                for keyword in playbook.keywords
                if (keyword in lowered if " " in keyword else keyword in words)
            )
            if score > best_score:
                best_id, best_score = playbook.category_id, score
        return best_id
