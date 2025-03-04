"""
OpenAPI schema generation utilities for the Local Operator API.

This module provides utilities for generating and saving the OpenAPI schema
from the FastAPI application.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

logger = logging.getLogger("local_operator.server.openapi")


def generate_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate the OpenAPI schema from the FastAPI application.

    Args:
        app: The FastAPI application instance

    Returns:
        Dict: The OpenAPI schema as a dictionary
    """
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=getattr(app, "servers", None),
    )


def save_openapi_schema(app: FastAPI, output_path: Union[str, Path], pretty: bool = True) -> None:
    """
    Generate and save the OpenAPI schema to a file.

    Args:
        app: The FastAPI application instance
        output_path: Path where the schema will be saved
        pretty: Whether to format the JSON with indentation for readability

    Raises:
        IOError: If there's an error writing to the file
    """
    schema = generate_openapi_schema(app)

    # Convert Path to string if needed
    if isinstance(output_path, Path):
        output_path = str(output_path)

    # Create parent directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w") as f:
            if pretty:
                json.dump(schema, f, indent=2)
            else:
                json.dump(schema, f)
        logger.info(f"OpenAPI schema saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving OpenAPI schema: {e}")
        raise


def get_openapi_schema_path(output_dir: Optional[Path] = None) -> Path:
    """
    Get the default path for the OpenAPI schema file.

    Args:
        config_dir: Optional configuration directory. If not provided,
                   defaults to ./docs

    Returns:
        Path: The path where the OpenAPI schema should be saved
    """
    if output_dir is None:
        output_dir = Path.cwd() / "docs"

    return output_dir / "openapi.json"
