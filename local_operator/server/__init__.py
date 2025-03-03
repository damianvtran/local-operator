"""
Local Operator Server Module

This module provides a FastAPI-based server implementation for the Local Operator API,
allowing users to interact with the Local Operator agent through HTTP requests instead
of the command-line interface.

The server exposes REST endpoints for:
- Chat generation with AI models
- Agent management (create, read, update, delete)
- Health monitoring

The server maintains state for credential management, configuration, and agent registry
across requests, while creating isolated model instances for each chat request.
"""
