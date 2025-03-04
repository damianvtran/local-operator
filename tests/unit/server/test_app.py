import pytest

from local_operator.server.app import app


# Test that the app state is properly initialized using the test_app_client fixture
@pytest.mark.asyncio
async def test_app_state_initialization(test_app_client):
    """Test that the app state is properly initialized with the test_app_client fixture."""
    # Verify that the app state has been initialized with the expected managers
    assert app.state.credential_manager is not None
    assert app.state.config_manager is not None
    assert app.state.agent_registry is not None

    # Test a request that depends on the app state
    response = await test_app_client.get("/v1/agents")
    assert response.status_code == 200
