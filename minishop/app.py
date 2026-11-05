import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from contextlib import asynccontextmanager

# Assuming database.py is accessible in the project scope
# Note: The prompt suggests importing from minishop.database, but since we are implementing app.py
try:
    from minishop.database import init_db, get_all_products, search_products
except ImportError:
    from database import init_db, get_all_products, search_products

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / 'minishop.db')
init_db(DB_PATH)

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    yield

app = FastAPI(
    title="MiniShop Lite",
    lifespan=lifespan
)

# --- Template and Static File Mounting ---
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, q: str | None = None):
    """
    Displays the main product catalog page. 
    Handles search queries via 'q' parameter.
    """
    query = q if q else ""
    
    try:
        if query:
            products = search_products(query, db_path=DB_PATH)
        else:
            products = get_all_products(db_path=DB_PATH)
    except Exception as e:
        print(f"Database error during product fetching: {e}")
        products = [] 
        
    return templates.TemplateResponse(
        request=request,
        name="catalog.html", 
        context={
            "products": products, 
            "query": query,
        },
    )

# End of app.py