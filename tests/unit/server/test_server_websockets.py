"""
Tests for the WebSocket endpoints in the Local Operator API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import (
    CodeExecutionResult,
    ConversationRole,
    ExecutionType,
    ProcessResponseStatus,
)


@pytest.fixture
def websocket_manager():
    """Create a WebSocket manager for testing."""
    return WebSocketManager()


# Mock the get_websocket_manager_ws dependency
@pytest.fixture
def mock_get_websocket_manager(monkeypatch):
    """Mock the get_websocket_manager_ws dependency."""
    mock_manager = MagicMock(spec=WebSocketManager)

    # Mock the async methods
    mock_manager.connect = AsyncMock()
    mock_manager.disconnect = AsyncMock()
    mock_manager.subscribe = AsyncMock()
    mock_manager.unsubscribe = AsyncMock()
    mock_manager.broadcast = AsyncMock()
    mock_manager.broadcast_update = AsyncMock()

    # Apply the patch
    with patch(
        "local_operator.server.routes.websockets.get_websocket_manager_ws",
        return_value=mock_manager,
    ):
        yield mock_manager


@pytest.mark.asyncio
async def test_websocket_manager_connect(websocket_manager):
    """Test the WebSocket manager connect method."""
    # Create a mock WebSocket with async methods
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()

    # Connect the WebSocket to a message ID
    message_id = "test-message-id"
    await websocket_manager.connect(websocket, message_id)

    # Check that the WebSocket was accepted
    websocket.accept.assert_called_once()

    # Check that the WebSocket was added to the message connections
    assert websocket in websocket_manager.message_connections[message_id]

    # Check that the message ID was added to the connection messages
    assert message_id in websocket_manager.connection_messages[websocket]


@pytest.mark.asyncio
async def test_websocket_manager_disconnect(websocket_manager):
    """Test the WebSocket manager disconnect method."""
    # Create a mock WebSocket with async methods
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()

    # Connect the WebSocket to a message ID
    message_id = "test-message-id"
    await websocket_manager.connect(websocket, message_id)

    # Disconnect the WebSocket
    await websocket_manager.disconnect(websocket)

    # Check that the WebSocket was removed from the message connections
    assert (
        message_id not in websocket_manager.message_connections
        or websocket not in websocket_manager.message_connections[message_id]
    )

    # Check that the message ID was removed from the connection messages
    assert websocket not in websocket_manager.connection_messages


@pytest.mark.asyncio
async def test_websocket_manager_subscribe_unsubscribe(websocket_manager):
    """Test the WebSocket manager subscribe and unsubscribe methods."""
    # Create a mock WebSocket with async methods
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()

    # Connect the WebSocket to a message ID
    message_id1 = "test-message-id-1"
    await websocket_manager.connect(websocket, message_id1)

    # Subscribe to another message ID
    message_id2 = "test-message-id-2"
    await websocket_manager.subscribe(websocket, message_id2)

    # Check that the WebSocket was added to the message connections for both message IDs
    assert websocket in websocket_manager.message_connections[message_id1]
    assert websocket in websocket_manager.message_connections[message_id2]

    # Check that both message IDs were added to the connection messages
    assert message_id1 in websocket_manager.connection_messages[websocket]
    assert message_id2 in websocket_manager.connection_messages[websocket]

    # Unsubscribe from the first message ID
    await websocket_manager.unsubscribe(websocket, message_id1)

    # Check that the WebSocket was removed from the message connections for the first message ID
    assert (
        message_id1 not in websocket_manager.message_connections
        or websocket not in websocket_manager.message_connections[message_id1]
    )

    # Check that the first message ID was removed from the connection messages
    assert message_id1 not in websocket_manager.connection_messages[websocket]

    # Check that the WebSocket is still subscribed to the second message ID
    assert websocket in websocket_manager.message_connections[message_id2]
    assert message_id2 in websocket_manager.connection_messages[websocket]


@pytest.mark.asyncio
async def test_websocket_manager_broadcast(websocket_manager):
    """Test the WebSocket manager broadcast method."""
    # Create a mock WebSocket
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()

    # Connect the WebSocket to a message ID
    message_id = "test-message-id"
    await websocket_manager.connect(websocket, message_id)

    # Reset the mock to clear the connection message
    websocket.send_text.reset_mock()

    # Create a test message
    test_data = {"test": "data"}

    # Broadcast the message
    await websocket_manager.broadcast(message_id, test_data)

    # Check that the WebSocket received the message
    websocket.send_text.assert_called_once()

    # Check that the message contains the test data and message ID
    called_args = websocket.send_text.call_args[0][0]
    called_data = json.loads(called_args)
    assert called_data["test"] == "data"
    assert called_data["message_id"] == message_id


@pytest.mark.asyncio
async def test_websocket_manager_broadcast_update(websocket_manager):
    """Test the WebSocket manager broadcast_update method."""
    # Create a mock WebSocket
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()

    # Connect the WebSocket to a message ID
    message_id = "test-message-id"
    await websocket_manager.connect(websocket, message_id)

    # Reset the mock to clear the connection message
    websocket.send_text.reset_mock()

    # Create a test execution result
    execution_result = CodeExecutionResult(
        id=message_id,
        stdout="test stdout",
        stderr="test stderr",
        logging="test logging",
        message="test message",
        code="test code",
        formatted_print="test formatted print",
        role=ConversationRole.ASSISTANT,
        status=ProcessResponseStatus.SUCCESS,
        files=[],
        execution_type=ExecutionType.PLAN,
        is_streamable=True,
        is_complete=True,
    )

    # Broadcast the update
    await websocket_manager.broadcast_update(message_id, execution_result)

    # Check that the WebSocket received the message
    websocket.send_text.assert_called_once()

    # Check that the message contains the execution result and message ID
    called_args = websocket.send_text.call_args[0][0]
    called_data = json.loads(called_args)
    assert called_data["id"] == message_id
    assert called_data["stdout"] == "test stdout"
    assert called_data["stderr"] == "test stderr"
    assert called_data["logging"] == "test logging"
    assert called_data["message"] == "test message"
    assert called_data["code"] == "test code"
    assert called_data["formatted_print"] == "test formatted print"
    assert called_data["status"] == "success"
    assert called_data["execution_type"] == "plan"
    assert called_data["is_streamable"] is True
    assert called_data["is_complete"] is True


@pytest.mark.asyncio
async def test_websocket_endpoint(mock_get_websocket_manager):
    """Test the WebSocket endpoint."""
    # Create a mock WebSocket
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock(return_value=json.dumps({"type": "ping"}))
    websocket.send_text = AsyncMock()

    # Import the websocket_endpoint function
    from local_operator.server.routes.websockets import websocket_endpoint

    # Call the endpoint with our mocked objects
    # Set up the mock to raise WebSocketDisconnect after one iteration
    websocket.receive_text.side_effect = [
        json.dumps({"type": "ping"}),
        json.dumps({"type": "subscribe", "message_id": "new-message-id"}),
        json.dumps({"type": "unsubscribe", "message_id": "new-message-id"}),
        WebSocketDisconnect(),
    ]

    await websocket_endpoint(websocket, "test-message-id", mock_get_websocket_manager)

    # Verify the WebSocket was connected
    mock_get_websocket_manager.connect.assert_called_once_with(websocket, "test-message-id")

    # Verify ping was responded to with pong
    websocket.send_text.assert_any_call(json.dumps({"type": "pong"}))

    # Verify disconnect was called
    mock_get_websocket_manager.disconnect.assert_called_once_with(websocket)


@pytest.mark.asyncio
async def test_websocket_error_handling(mock_get_websocket_manager):
    """Test error handling in the WebSocket endpoint."""
    # Create a mock WebSocket
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.send_text = AsyncMock()

    # Import the websocket_endpoint function
    from local_operator.server.routes.websockets import websocket_endpoint

    # Set up the mock to raise an exception after one iteration
    websocket.receive_text.side_effect = ["invalid json", WebSocketDisconnect()]

    # Call the endpoint with our mocked objects
    await websocket_endpoint(websocket, "test-message-id", mock_get_websocket_manager)

    # Verify the WebSocket was connected
    mock_get_websocket_manager.connect.assert_called_once_with(websocket, "test-message-id")

    # Verify disconnect was called
    mock_get_websocket_manager.disconnect.assert_called_once_with(websocket)
