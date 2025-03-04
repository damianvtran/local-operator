#!/usr/bin/env python
"""
Script to generate OpenAPI schema from the Local Operator API.

This script can be used to generate the OpenAPI schema from the FastAPI application
without starting the server. It saves the schema to a file that can be used with
other tools or for documentation purposes.
"""

import argparse
import logging
import sys

from local_operator.server.app import app
from local_operator.server.openapi import get_openapi_schema_path, save_openapi_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("local_operator.server.generate_openapi")


def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Generate OpenAPI schema for Local Operator API")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the OpenAPI schema (default: ~/.local-operator/openapi.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Format the JSON with indentation for readability (default: True)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    output_path = args.output
    if not output_path:
        output_path = get_openapi_schema_path()

    try:
        save_openapi_schema(app, output_path, pretty=args.pretty)
        logger.info(f"OpenAPI schema saved to {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Error generating OpenAPI schema: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
