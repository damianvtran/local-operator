import pytest

from local_operator.server.app import app


# Test that the CORS middleware is properly configured
@pytest.mark.asyncio
async def test_cors_headers(test_app_client):
    """Test that CORS headers are properly set in the response."""
    # Make a request with an Origin header to simulate a cross-origin request
    response = await test_app_client.get("/v1/agents", headers={"Origin": "http://localhost:3000"})

    # Verify that the CORS headers are present in the response
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
    assert response.headers.get("access-control-allow-credentials") == "true"


# Test that the app state is properly initialized using the test_app_client fixture
@pytest.mark.asyncio
async def test_app_state_initialization(test_app_client):
    """Test that the app state is properly initialized with the test_app_client fixture."""
    # Verify that the app state has been initialized with the expected managers
    assert app.state.credential_manager is not None
    assert app.state.config_manager is not None
    assert app.state.agent_registry is not None
    assert app.state.whisper_model is not None

    # Test a request that depends on the app state
    response = await test_app_client.get("/v1/agents")
    assert response.status_code == 200
