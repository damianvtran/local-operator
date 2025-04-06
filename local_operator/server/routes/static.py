"""
Static file serving endpoints for the Local Operator API.

This module contains the FastAPI route handlers for serving static files.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["Static"])
logger = logging.getLogger("local_operator.server.routes.static")

# Path to the static directory
STATIC_DIR = Path(__file__).parent.parent / "static"
HTML_DIR = STATIC_DIR / "html"


@router.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Serve the index page.
    
    Returns:
        HTMLResponse: The index page HTML
    """
    try:
        with open(HTML_DIR / "index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        # If index.html doesn't exist, return a simple HTML page
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Local Operator</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }
                h1 {
                    color: #2c3e50;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                }
                li {
                    margin-bottom: 10px;
                }
                a {
                    color: #3498db;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Local Operator</h1>
            <p>Welcome to the Local Operator API server.</p>
            <h2>Available Pages:</h2>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/redoc">ReDoc API Documentation</a></li>
                <li><a href="/documents">Document Upload Interface</a></li>
            </ul>
        </body>
        </html>
        """


@router.get("/documents", response_class=HTMLResponse)
async def get_document_upload_page():
    """
    Serve the document upload page.
    
    Returns:
        HTMLResponse: The document upload page HTML
    """
    try:
        with open(HTML_DIR / "document_upload.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document upload page not found")


@router.get("/static/{file_path:path}")
async def get_static_file(file_path: str):
    """
    Serve static files.
    
    Args:
        file_path: Path to the static file
        
    Returns:
        FileResponse: The requested file
    """
    file = STATIC_DIR / file_path
    if not file.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(file)
