"""Preserve the original MiniShop Lite as a historical product.

Copies only.  The originals are left in place, because the legacy Milestone 2
workflow, the benchmark and the existing acceptance tests still import them.  Each
copied file is verified by hash, and the import is recorded in a manifest.

No model runs here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from lab.product.manager import ProductManager
from lab.product.store import ProductStore
from lab.schemas.product import ProductPhase, WorkflowState, utcnow
from lab.tools.fsutil import atomic_write_text, sha256_file

SLUG = "mini-shop-lite"
DISPLAY_NAME = "Mini Shop Lite"
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache"})
_SKIP_NAMES = frozenset({".DS_Store"})

_BRIEF = """# Mini Shop Lite

_A small local product catalog, built by the earlier milestones of this lab._

## Description

A lightweight, responsive e-commerce catalog built with FastAPI, Jinja2 and SQLite.
It lists seeded products with name, description, price and stock status, supports
case-insensitive search, shows a clear message when nothing matches, and exposes a
health check.

## History

Imported from the original `minishop/` package and `tests/acceptance/` directory.
It predates the product team workflow, so it has no roadmap or step history. Its
authoritative acceptance tests are preserved under `verification/legacy/`.
"""


@dataclass(frozen=True)
class MigrationReport:
    status: str
    product_root: Path
    files_copied: int
    detail: str


def _collect(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.name not in _SKIP_NAMES
        and not _SKIP_DIRS.intersection(path.relative_to(source).parts)
    )


def _copy_verified(source: Path, destination: Path, record: dict[str, str], root: Path) -> None:
    """Copy a tree, proving every file arrived intact."""
    for path in _collect(source):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        original = sha256_file(path)
        if original is None or sha256_file(target) != original:
            raise OSError(f"Verification failed for copied file {target}")
        record[target.relative_to(root).as_posix()] = original


def migrate_minishop(manager: ProductManager, repo_root: Path | str) -> MigrationReport:
    """Create ``products_log/mini-shop-lite``.  Safe to run more than once."""
    repo = Path(repo_root).resolve()
    minishop = repo / "minishop"
    acceptance = repo / "tests" / "acceptance"
    product_root = manager.root / SLUG

    if (product_root / "product.json").is_file():
        return MigrationReport("already_migrated", product_root, 0, "Nothing to do.")
    if not minishop.is_dir():
        return MigrationReport("source_missing", product_root, 0, f"{minishop} does not exist.")
    if product_root.exists():
        return MigrationReport(
            "conflict",
            product_root,
            0,
            f"{product_root} exists but is not a product. It was left untouched.",
        )

    store = manager.create(DISPLAY_NAME, origin="imported")
    copied: dict[str, str] = {}
    _copy_verified(minishop, store.workspace_dir / "minishop", copied, store.root)
    if acceptance.is_dir():
        legacy_tests = store.verification_dir / "legacy"
        _copy_verified(acceptance, legacy_tests, copied, store.root)
        atomic_write_text(legacy_tests / "pytest.ini", "[pytest]\n")

    for name in ("runs", "artifacts"):
        source = repo / name
        if source.is_dir() and _collect(source):
            _copy_verified(source, store.logs_dir / f"legacy_{name}", copied, store.root)

    meta = store.load_metadata()
    meta.category = "ecommerce"
    meta.category_label = "Website / E-commerce"
    meta.description = "Historical catalog demo from the lab's earlier milestones."
    meta.name_confirmed = True
    store.save_metadata(meta)

    store.write_text("brief.md", _BRIEF)
    state = WorkflowState(phase=ProductPhase.HISTORICAL)
    state.next_expected_action = "Nothing pending. This product is preserved for reference."
    store.save_state(state)
    atomic_write_text(
        store.logs_dir / "migration_manifest.json",
        json.dumps(
            {
                "migrated_at": utcnow().isoformat(),
                "source": str(repo),
                "originals_kept": True,
                "files": copied,
            },
            indent=2,
        ),
    )
    store.append_event("ProductMigrated", payload={"files": len(copied)})
    store.append_decision(
        f"Imported the original MiniShop Lite. {len(copied)} files were copied and verified "
        "by hash. The originals were left in place."
    )
    return MigrationReport("migrated", store.root, len(copied), "Every copy verified by hash.")


def open_migrated(manager: ProductManager) -> ProductStore:
    return manager.open(SLUG)
