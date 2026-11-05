"""Authoritative acceptance tests for MiniShop Lite product catalog.

Evaluation integrity rules:
- These tests are independent of the Developer agent and cannot be modified by the model.
- Tests verify actual application behavior via HTTP requests (FastAPI TestClient)
  and direct SQLite verification.
- Covers:
    1. Sample product listing (name, description, price, stock status).
    2. Search products by name (case-insensitive substring match).
    3. Clear message displayed when no products match.
    4. Stock status indicators (In Stock vs Out of Stock).
    5. Database seed data and schema integrity.
    6. Health check endpoint.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest
from starlette.testclient import TestClient

@pytest.fixture(scope="module")
def minishop_pkg():
    """Dynamically import minishop modules."""
    try:
        import minishop.app as app_mod
        import minishop.database as db_mod
        return app_mod, db_mod
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"MiniShop implementation files missing: {exc}")


@pytest.fixture(scope="module", autouse=True)
def ensure_database(minishop_pkg):
    """Ensure the database is initialized with seed data before running acceptance tests."""
    _, db_mod = minishop_pkg
    db_mod.init_db(db_mod.DB_PATH)


@pytest.fixture
def client(minishop_pkg):
    """FastAPI test client."""
    app_mod, _ = minishop_pkg
    return TestClient(app_mod.app)


@pytest.fixture
def db(minishop_pkg):
    """Database module fixture."""
    _, db_mod = minishop_pkg
    return db_mod


# ---------------------------------------------------------------------------
# Test 1: Seed Product Display
# ---------------------------------------------------------------------------


def test_catalog_displays_seeded_products(client, db):
    """Verify that visiting '/' loads the catalog and renders all seeded products."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Verify key structural elements
    assert "MiniShop" in html
    assert "Catalog" in html or "Products" in html

    # Verify that products from the database are present in the HTML
    products = db.get_all_products(db.DB_PATH)
    assert len(products) >= 3, "Expected at least 3 seed products in the database"

    for prod in products:
        # Tuple format: (id, name, description, price, in_stock) or dict
        if isinstance(prod, (list, tuple)):
            p_name = prod[1]
            p_desc = prod[2]
            p_price = prod[3]
        else:
            p_name = prod["name"]
            p_desc = prod["description"]
            p_price = prod["price"]

        assert p_name in html, f"Product name {p_name!r} not rendered in catalog"
        if p_desc:
            # Check snippet of description
            assert p_desc[:20] in html, f"Product description {p_desc!r} not rendered"
        assert f"{p_price:.2f}" in html or f"{p_price}" in html, f"Price for {p_name} not found"


# ---------------------------------------------------------------------------
# Test 2: Search Products by Name
# ---------------------------------------------------------------------------


def test_search_products_by_name(client, db):
    """Verify searching for a product returns matching results."""
    # Search for 'keyboard'
    response = client.get("/?q=keyboard")
    assert response.status_code == 200
    html = response.text

    assert "keyboard" in html.lower()
    # Ensure unrelated products are filtered out if they don't match
    # Example: 'Laptop' should not appear if searching for 'keyboard' unless description matches
    products = db.get_all_products(db.DB_PATH)
    matching = [p for p in products if "keyboard" in (p[1] if isinstance(p, (tuple, list)) else p["name"]).lower()]
    non_matching = [p for p in products if "keyboard" not in (p[1] if isinstance(p, (tuple, list)) else p["name"]).lower() and "keyboard" not in (p[2] if isinstance(p, (tuple, list)) else p.get("description", "")).lower()]

    assert len(matching) > 0, "Seed data must contain a keyboard product for test"
    for m in matching:
        m_name = m[1] if isinstance(m, (tuple, list)) else m["name"]
        assert m_name in html

    for nm in non_matching:
        nm_name = nm[1] if isinstance(nm, (tuple, list)) else nm["name"]
        assert nm_name not in html, f"Non-matching product {nm_name!r} should be filtered out"


def test_search_case_insensitive(client):
    """Verify search is case-insensitive ('KEYBOARD' vs 'keyboard')."""
    res_lower = client.get("/?q=keyboard")
    res_upper = client.get("/?q=KEYBOARD")
    assert res_lower.status_code == 200
    assert res_upper.status_code == 200
    assert "keyboard" in res_lower.text.lower()
    assert "keyboard" in res_upper.text.lower()


# ---------------------------------------------------------------------------
# Test 3: No Match Message
# ---------------------------------------------------------------------------


def test_search_no_match_displays_clear_message(client):
    """Verify that a search yielding no results shows a clear user message."""
    query = "NonExistentProductXYZ9999"
    response = client.get(f"/?q={query}")
    assert response.status_code == 200
    html = response.text

    # Must state clearly that no products were found
    lowered = html.lower()
    assert any(phrase in lowered for phrase in [
        "no products match",
        "no products found",
        "no items found",
        "nothing found",
        "0 products found",
    ]), f"Clear 'no match' message missing in response: {html[:400]}"


# ---------------------------------------------------------------------------
# Test 4: Stock Status Indicators
# ---------------------------------------------------------------------------


def test_stock_status_displayed_for_all_items(client):
    """Verify both in-stock and out-of-stock items have clear stock status indicators."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Seed data contains both in-stock and out-of-stock items
    assert "in stock" in html.lower(), "Expected 'In Stock' badge or text"
    assert "out of stock" in html.lower(), "Expected 'Out of Stock' badge or text"


# ---------------------------------------------------------------------------
# Test 5: SQLite Database Persistence & Schema
# ---------------------------------------------------------------------------


def test_database_schema_and_persistence(db):
    """Directly inspect SQLite database to verify tables, columns, and seed rows."""
    db_file = Path(db.DB_PATH)
    assert db_file.exists(), f"Database file {db.DB_PATH} does not exist"

    conn = sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()

    # Check products table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
    assert cursor.fetchone() is not None, "Table 'products' does not exist in SQLite DB"

    # Check table columns: id, name, description, price, in_stock
    cursor.execute("PRAGMA table_info(products)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    assert "id" in columns
    assert "name" in columns
    assert "description" in columns
    assert "price" in columns
    assert "in_stock" in columns

    # Check seed data count
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    assert count >= 3, f"Expected at least 3 seed products, found {count}"

    conn.close()


# ---------------------------------------------------------------------------
# Test 6: Health Check
# ---------------------------------------------------------------------------


def test_health_check_endpoint(client):
    """Verify GET /health returns HTTP 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
