"""Command-line entry point for the Local AI Product Development Team.

    python -m lab                 Talk to the Business Analyst
    python -m lab new "an idea"   Start a product straight away
    python -m lab list            List products
    python -m lab status NAME     Show one product's status
    python -m lab resume NAME     Continue a product
    python -m lab doctor          Check Ollama and the configured models (no inference)
    python -m lab migrate-minishop   Preserve the old MiniShop as a historical product
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

from lab.config import ModelPolicy, load_model_policy
from lab.playbooks import PlaybookLibrary
from lab.product.manager import ProductManager, default_products_root
from lab.product.migration import migrate_minishop
from lab.product.session import ProductSession
from lab.runtime.lifecycle import OllamaLifecycle
from lab.runtime.router import ModelRouter, default_runtime_factory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_logging(products_root: Path) -> None:
    """Send internal logs to a file, so the conversation stays readable."""
    products_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=products_root / ".session.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Library warnings belong in the log file too, not in the conversation.
    logging.captureWarnings(True)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ.setdefault("LITELLM_LOG", "ERROR")


def _base_url() -> str:
    return os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")


def build_router(policy: ModelPolicy) -> ModelRouter:
    base_url = _base_url()
    return ModelRouter(
        policy,
        default_runtime_factory(policy, base_url),
        OllamaLifecycle(base_url, policy.memory.unload_wait_seconds),
    )


def build_session(products_root: Path) -> ProductSession:
    policy = load_model_policy()
    playbooks = PlaybookLibrary()
    return ProductSession(
        ProductManager(products_root, playbooks), build_router(policy), policy, playbooks
    )


def _find(manager: ProductManager, selection: str) -> str | None:
    summaries, _ = manager.discover()
    chosen = manager.select(selection, summaries)
    return chosen.slug if chosen else None


def command_list(manager: ProductManager) -> int:
    summaries, invalid = manager.discover()
    if not summaries:
        print("There are no products yet.")
    for index, item in enumerate(summaries, start=1):
        print(f"{index}. {item.render()}\n")
    for item in invalid:
        print(f"Needs attention: {item.slug} ({item.reason})")
    return 0


def command_status(manager: ProductManager, selection: str) -> int:
    slug = _find(manager, selection)
    if slug is None:
        print(f"No single product matches '{selection}'.", file=sys.stderr)
        return 1
    summaries, _ = manager.discover()
    print(next(item for item in summaries if item.slug == slug).render())
    return 0


def command_doctor(policy: ModelPolicy) -> int:
    """Report readiness.  Reads metadata only; performs no inference."""

    async def check() -> int:
        lifecycle = OllamaLifecycle(_base_url())
        print(f"Ollama endpoint: {_base_url()}")
        try:
            installed = await lifecycle.installed()
            running = await lifecycle.running()
        except Exception as exc:
            print(f"  NOT REACHABLE: {exc}")
            print("  Start the Ollama app, then run this again.")
            return 1
        print(f"  Reachable. Loaded right now: {running or 'nothing'}")
        missing = 0
        for alias, spec in policy.models.items():
            present = spec.id in installed
            missing += 0 if present else 1
            print(f"  {alias}: {spec.id} {'installed' if present else 'NOT INSTALLED'}")
        return 1 if missing else 0

    return asyncio.run(check())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lab", description="Local AI Product Development Team"
    )
    parser.add_argument(
        "--products-dir", default=None, help="Where products live (default: products_log/)"
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("start", help="Talk to the Business Analyst (the default)")
    new = commands.add_parser("new", help="Start a new product")
    new.add_argument("idea", nargs="+", help="Your product idea")
    commands.add_parser("list", help="List products")
    status = commands.add_parser("status", help="Show one product's status")
    status.add_argument("product", help="Product name, or its number in the list")
    resume = commands.add_parser("resume", help="Continue a product")
    resume.add_argument("product", help="Product name, or its number in the list")
    commands.add_parser("doctor", help="Check Ollama and the models, without inference")
    commands.add_parser("migrate-minishop", help="Preserve the old MiniShop as a product")
    args = parser.parse_args(argv)

    products_root = Path(args.products_dir).resolve() if args.products_dir else default_products_root()
    configure_logging(products_root)

    if args.command == "doctor":
        return command_doctor(load_model_policy())

    manager = ProductManager(products_root, PlaybookLibrary())
    if args.command == "list":
        return command_list(manager)
    if args.command == "status":
        return command_status(manager, args.product)
    if args.command == "migrate-minishop":
        report = migrate_minishop(manager, PROJECT_ROOT)
        print(f"{report.status}: {report.detail} ({report.files_copied} files)")
        print(f"Location: {report.product_root}")
        return 0 if report.status in ("migrated", "already_migrated") else 1

    session = build_session(products_root)

    async def run() -> None:
        if args.command == "new":
            await session.run_product(session.start_new(" ".join(args.idea)))
        elif args.command == "resume":
            slug = _find(manager, args.product)
            if slug is None:
                print(f"No single product matches '{args.product}'.", file=sys.stderr)
                return
            await session.run_product(session.open_product(slug))
        else:
            await session.run()

    run_interruptibly(run(), session.release_models)
    return 0


def run_interruptibly(
    main_coroutine: Coroutine[Any, Any, None],
    cleanup: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    """Run the session so that one Ctrl-C stops it cleanly.

    asyncio.run() only cancels the main task on the first Ctrl-C, and a task blocked
    in input() cannot see that, so the prompt would ignore the key.  With a plain
    event loop the interrupt is raised where the program is.  The interrupted task
    then simply stays interrupted on disk, ready to resume.
    """
    loop = asyncio.new_event_loop()
    task = loop.create_task(main_coroutine)
    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        # A task that itself ended in KeyboardInterrupt must not be waited on again:
        # asyncio never stops the loop for it, so that wait would hang forever.
        if not task.done():
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
        # Printed last, so any heartbeat has stopped and cannot land after it.
        print("\nStopped. Everything saved so far is safe, and the task can be resumed.")
    finally:
        try:
            loop.run_until_complete(cleanup())
        finally:
            loop.close()
