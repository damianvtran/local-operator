"""
Tests for the health endpoint of the FastAPI server.

This module contains tests for the health check endpoint to ensure
the server is responding correctly.
"""

import pytest


@pytest.mark.asyncio
async def test_health_check(test_app_client):
    """Test the health check endpoint using the test_app_client fixture."""
    response = await test_app_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "ok"
