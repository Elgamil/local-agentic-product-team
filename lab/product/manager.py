"""Product creation, naming and discovery under ``products_log/``.

Folder names come from Python, never from a model: a model may suggest a display
name, and :func:`slugify` turns it into a safe directory name.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lab.playbooks import PlaybookLibrary
from lab.product.autonomy import describe_mode
from lab.product.presentation import next_action, progress_label, status_label
from lab.product.store import ProductStore, StateCorruptedError
from lab.schemas.product import (
    ExecutionMode,
    ProductMetadata,
    ProductPhase,
    Roadmap,
    WorkflowState,
    utcnow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTS_ROOT = PROJECT_ROOT / "products_log"
MAX_SLUG_LENGTH = 60
UNNAMED_DISPLAY_NAME = "New product (not named yet)"


def default_products_root() -> Path:
    return Path(os.environ.get("LAB_PRODUCTS_ROOT", DEFAULT_PRODUCTS_ROOT)).resolve()


def slugify(name: str) -> str:
    """Convert a display name into a lowercase, filesystem-safe folder name."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")
    slug = slug[:MAX_SLUG_LENGTH].strip("-")
    return slug or "product"


@dataclass(frozen=True)
class ProductSummary:
    """One row of the product list."""

    slug: str
    display_name: str
    category_label: str
    description: str
    status: str
    progress: str
    last_activity: datetime
    next_action: str
    phase: ProductPhase
    interrupted_task: str | None
    paused: bool
    execution_mode: str = ""
    intervention_report: str | None = None

    @property
    def finished(self) -> bool:
        return self.phase in (ProductPhase.PRODUCT_COMPLETE, ProductPhase.HISTORICAL)

    def render(self) -> str:
        heading = f"{self.display_name} — {self.category_label}"
        lines = [
            heading,
            f"  Status: {self.status}",
            f"  Progress: {self.progress}",
            f"  Last activity: {format_day(self.last_activity)}",
            f"  Next action: {self.next_action}",
        ]
        if self.description:
            lines.insert(1, f"  About: {self.description}")
        if self.interrupted_task:
            lines.append(f"  Interrupted: {self.interrupted_task}")
        if self.execution_mode:
            lines.append(f"  Execution mode: {self.execution_mode}")
        if self.intervention_report:
            lines.append(f"  Intervention report: {self.intervention_report}")
        return "\n".join(lines)


@dataclass(frozen=True)
class InvalidProduct:
    slug: str
    reason: str


def format_day(moment: datetime) -> str:
    return f"{moment.strftime('%B')} {moment.day}, {moment.year}"


def latest_roadmap(store: ProductStore, state: WorkflowState) -> Roadmap | None:
    if state.roadmap_version < 1:
        return None
    relative = f"artifacts/roadmap.v{state.roadmap_version}.json"
    if not store.exists(relative):
        return None
    return store.load_model(relative, Roadmap)


class ProductManager:
    def __init__(self, root: Path | str | None, playbooks: PlaybookLibrary) -> None:
        self.root = Path(root).resolve() if root else default_products_root()
        self.playbooks = playbooks

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def unique_slug(self, base: str) -> str:
        """Pick a free folder name.  Claiming it is left to :meth:`_claim`."""
        candidate = base
        counter = 2
        while (self.root / candidate).exists():
            suffix = f"-{counter}"
            candidate = f"{base[: MAX_SLUG_LENGTH - len(suffix)].rstrip('-')}{suffix}"
            counter += 1
        return candidate

    def _claim(self, base: str) -> Path:
        """Create the folder atomically, so two claimants can never share one."""
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            path = self.root / self.unique_slug(base)
            try:
                os.mkdir(path)
            except FileExistsError:
                continue
            return path

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create(
        self,
        display_name: str | None = None,
        *,
        origin: str = "interactive",
    ) -> ProductStore:
        """Create a product folder.  An existing folder is never touched.

        Without a name the product gets a temporary ``draft-…`` folder, so the
        discovery conversation is durable from its very first message.
        """
        if display_name:
            base = slugify(display_name)
        else:
            base = f"draft-{utcnow().strftime('%Y%m%d-%H%M%S')}"
        path = self._claim(base)
        store = ProductStore(path)
        store.create_layout()
        meta = ProductMetadata(
            product_id=uuid.uuid4().hex,
            display_name=display_name or UNNAMED_DISPLAY_NAME,
            slug=path.name,
            origin=origin,  # type: ignore[arg-type]
        )
        store.save_metadata(meta)
        store.save_state(WorkflowState())
        store.append_event("ProductCreated", payload={"slug": path.name})
        return store

    def rename(self, store: ProductStore, display_name: str) -> str:
        """Move a product to the folder for *display_name* and return the new slug.

        The destination is claimed first and only that empty claim is replaced, so
        renaming can never overwrite another product.
        """
        desired = slugify(display_name)
        if desired != store.root.name:
            target = self._claim(desired)
            os.rename(store.root, target)
            store.root = target
        meta = store.load_metadata()
        meta.display_name = display_name
        meta.slug = store.root.name
        store.save_metadata(meta)
        store.append_event(
            "ProductRenamed", payload={"display_name": display_name, "slug": store.root.name}
        )
        return store.root.name

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _summarize(self, store: ProductStore) -> ProductSummary:
        meta = store.load_metadata()
        state = store.load_state()
        roadmap = latest_roadmap(store, state)
        interrupted = None
        if state.active_task is not None:
            interrupted = f"{state.active_task.kind} ({state.active_task.task_id})"
        mode = ""
        if state.autonomy.mode != ExecutionMode.MANUAL:
            mode = describe_mode(state.autonomy, roadmap)
        report = None
        if state.intervention is not None:
            report = f"products_log/{store.slug}/{state.intervention.report_path}"
        return ProductSummary(
            slug=store.slug,
            display_name=meta.display_name,
            category_label=meta.category_label or self.playbooks.label_for(meta.category),
            description=meta.description,
            status=status_label(state, roadmap),
            progress=progress_label(state, roadmap),
            last_activity=state.updated_at,
            next_action=next_action(state, roadmap),
            phase=state.phase,
            interrupted_task=interrupted,
            paused=state.paused,
            execution_mode=mode,
            intervention_report=report,
        )

    def discover(self) -> tuple[list[ProductSummary], list[InvalidProduct]]:
        """Scan for products.  Damage to one product never hides the others."""
        summaries: list[ProductSummary] = []
        invalid: list[InvalidProduct] = []
        if not self.root.is_dir():
            return summaries, invalid
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            store = ProductStore(path)
            if not store.metadata_path.is_file():
                continue
            try:
                summaries.append(self._summarize(store))
            except (StateCorruptedError, OSError, ValueError, json.JSONDecodeError) as exc:
                invalid.append(InvalidProduct(slug=path.name, reason=str(exc)[:200]))
        summaries.sort(key=lambda item: item.last_activity, reverse=True)
        return summaries, invalid

    def open(self, slug: str) -> ProductStore:
        store = ProductStore(self.root / slug)
        if not store.metadata_path.is_file():
            raise FileNotFoundError(f"No product found at {store.root}")
        return store

    @staticmethod
    def select(selection: str, summaries: list[ProductSummary]) -> ProductSummary | None:
        """Pick a product by list number, exact name, slug, or unique partial name."""
        text = selection.strip().casefold()
        if not text:
            return None
        if text.isdigit():
            index = int(text) - 1
            return summaries[index] if 0 <= index < len(summaries) else None
        exact = [
            item
            for item in summaries
            if text in (item.display_name.casefold(), item.slug.casefold())
        ]
        if len(exact) == 1:
            return exact[0]
        partial = [
            item
            for item in summaries
            if text in item.display_name.casefold() or text in item.slug.casefold()
        ]
        return partial[0] if len(partial) == 1 else None
