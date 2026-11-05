import sqlite3
from pathlib import Path
from typing import List, Dict

# Define the canonical database path
DB_PATH = str(Path(__file__).resolve().parent / "minishop.db")

def init_db(db_path: str = DB_PATH):
    """
    Initializes the database: creates the 'products' table and seeds initial data 
    if the table is empty.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create the products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            in_stock BOOLEAN NOT NULL
        );
    """)
    
    # 2. Check if data needs seeding
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        print("Database is empty. Seeding initial product data...")
        
        # Seed data structure: (name, description, price, in_stock)
        products_data = [
            ("Quantum Laptop X1", "High performance laptop for demanding tasks.", 1299.99, 1),
            ("4K Ultra Monitor Z", "Stunning clarity monitor for professional work.", 450.50, 1),
            ("Mechanical Keyboard K9", "Tactile and durable mechanical keyboard.", 120.00, 1),
            ("WebCam Pro Mini", "Crystal clear webcam perfect for video conferencing.", 79.99, 0)
        ]
        
        # Insert all products
        cursor.executemany("""
            INSERT INTO products (name, description, price, in_stock) 
            VALUES (?, ?, ?, ?)
        """, products_data)
        
    conn.commit()
    conn.close()
    print(f"Database '{db_path}' initialized successfully.")


def get_all_products(db_path: str = DB_PATH) -> List[Dict]:
    """
    Retrieves all products from the database, ordered by ID.
    Returns a list of dictionaries, where each dict represents a product.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query all products
    cursor.execute("SELECT id, name, description, price, in_stock FROM products ORDER BY id")
    rows = cursor.fetchall()
    
    # Get column names for dictionary keys
    columns = [desc[0] for desc in cursor.description]
    
    results = []
    for row in rows:
        product = dict(zip(columns, row))
        results.append(product)
        
    conn.close()
    return results


def search_products(query: str, db_path: str = DB_PATH) -> List[Dict]:
    """
    Searches products where the name or description matches the query (case-insensitive).
    Returns a list of dictionaries.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Use LIKE for case-insensitive search
    search_term = f"%{query}%"
    
    # Note: SQL LIKE is usually case-insensitive by default for many SQLite configurations, 
    # but we use the placeholder approach for safety.
    cursor.execute("""
        SELECT id, name, description, price, in_stock 
        FROM products 
        WHERE name LIKE ? OR description LIKE ? 
        ORDER BY name
    """, (search_term, search_term))
    
    rows = cursor.fetchall()
    
    # Get column names for dictionary keys
    columns = [desc[0] for desc in cursor.description]
    
    results = []
    for row in rows:
        product = dict(zip(columns, row))
        results.append(product)
        
    conn.close()
    return results

if __name__ == '__main__':
    # Example usage: Initialize and test
    init_db()
    
    # Test 1: Get all
    all_products = get_all_products()
    print("\\n--- All Products ---")
    for p in all_products:
        print(f"{p['name']} (${p['price']:.2f}) - Stock: {'In' if p['in_stock'] else 'Out'}")
        
    # Test 2: Search
    search_query = "Monitor"
    search_results = search_products(search_query)
    print(f"\\n--- Search Results for '{search_query}' ---")
    if search_results:
        for p in search_results:
            print(f"{p['name']} (${p['price']:.2f}) - Stock: {'In' if p['in_stock'] else 'Out'}")
    else:
        print("No products found.")