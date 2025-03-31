"""
WebSocket manager for the Local Operator API.

This module provides a WebSocketManager class for managing WebSocket connections
and broadcasting updates to connected clients.
"""

import json
import logging
from typing import Any, Dict, Set

from fastapi import WebSocket

from local_operator.types import CodeExecutionResult

logger = logging.getLogger("local_operator.server.utils.websocket_manager")


class WebSocketManager:
    """
    WebSocket manager for the Local Operator API.

    This class manages WebSocket connections and provides methods for broadcasting
    updates to connected clients.

    Attributes:
        message_connections (Dict[str, Set[WebSocket]]): Maps message IDs to a set of
            connected WebSockets.
        connection_messages (Dict[WebSocket, Set[str]]): Maps WebSockets to a set of
            message IDs they are subscribed to.
    """

    def __init__(self):
        """Initialize the WebSocketManager."""
        # Maps message IDs to a set of connected WebSockets
        self.message_connections: Dict[str, Set[WebSocket]] = {}
        # Maps WebSockets to a set of message IDs they are subscribed to
        self.connection_messages: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket, message_id: str) -> None:
        """
        Connect a WebSocket to a message ID.

        Args:
            websocket (WebSocket): The WebSocket to connect.
            message_id (str): The message ID to connect to.
        """
        await websocket.accept()

        # Add the WebSocket to the message connections
        if message_id not in self.message_connections:
            self.message_connections[message_id] = set()
        self.message_connections[message_id].add(websocket)

        # Add the message ID to the connection messages
        if websocket not in self.connection_messages:
            self.connection_messages[websocket] = set()
        self.connection_messages[websocket].add(message_id)

        # Send a connection established message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection_established",
                    "message_id": message_id,
                    "status": "connected",
                }
            )
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket.

        Args:
            websocket (WebSocket): The WebSocket to disconnect.
        """
        # Remove the WebSocket from all message connections
        for message_id in self.connection_messages.get(websocket, set()):
            if message_id in self.message_connections:
                self.message_connections[message_id].discard(websocket)
                if not self.message_connections[message_id]:
                    del self.message_connections[message_id]

        # Remove the WebSocket from the connection messages
        if websocket in self.connection_messages:
            del self.connection_messages[websocket]

    async def subscribe(self, websocket: WebSocket, message_id: str) -> None:
        """
        Subscribe a WebSocket to a message ID.

        Args:
            websocket (WebSocket): The WebSocket to subscribe.
            message_id (str): The message ID to subscribe to.
        """
        # Add the WebSocket to the message connections
        if message_id not in self.message_connections:
            self.message_connections[message_id] = set()
        self.message_connections[message_id].add(websocket)

        # Add the message ID to the connection messages
        if websocket not in self.connection_messages:
            self.connection_messages[websocket] = set()
        self.connection_messages[websocket].add(message_id)

        # Send a subscription confirmation message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "subscription",
                    "message_id": message_id,
                    "status": "subscribed",
                }
            )
        )

    async def unsubscribe(self, websocket: WebSocket, message_id: str) -> None:
        """
        Unsubscribe a WebSocket from a message ID.

        Args:
            websocket (WebSocket): The WebSocket to unsubscribe.
            message_id (str): The message ID to unsubscribe from.
        """
        # Remove the WebSocket from the message connections
        if message_id in self.message_connections:
            self.message_connections[message_id].discard(websocket)
            if not self.message_connections[message_id]:
                del self.message_connections[message_id]

        # Remove the message ID from the connection messages
        if websocket in self.connection_messages:
            self.connection_messages[websocket].discard(message_id)
            if not self.connection_messages[websocket]:
                del self.connection_messages[websocket]

        # Send an unsubscription confirmation message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "unsubscription",
                    "message_id": message_id,
                    "status": "unsubscribed",
                }
            )
        )

    async def broadcast(self, message_id: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a message to all WebSockets subscribed to a message ID.

        Args:
            message_id (str): The message ID to broadcast to.
            data (Dict[str, Any]): The data to broadcast.
        """
        if message_id not in self.message_connections:
            logger.debug(f"No connections for message ID: {message_id}")
            return

        # Add the message ID to the data
        data["message_id"] = message_id

        # Convert the data to JSON
        json_data = json.dumps(data)

        # Broadcast the message to all WebSockets subscribed to the message ID
        disconnected_websockets = set()
        for websocket in self.message_connections[
            message_id
        ].copy():  # Use copy to avoid modification during iteration
            try:
                await websocket.send_text(json_data)
            except Exception as e:
                logger.error(f"Failed to broadcast message to WebSocket: {e}")
                disconnected_websockets.add(websocket)

        # Disconnect any WebSockets that failed to receive the message
        for websocket in disconnected_websockets:
            logger.debug(f"Disconnecting failed WebSocket for message ID: {message_id}")
            await self.disconnect(websocket)

    async def broadcast_update(
        self, message_id: str, execution_result: CodeExecutionResult
    ) -> None:
        """
        Broadcast an execution result update to all WebSockets subscribed to a message ID.

        Args:
            message_id (str): The message ID to broadcast to.
            execution_result (CodeExecutionResult): The execution result to broadcast.
        """
        # Convert the execution result to a dictionary
        data = execution_result.model_dump()

        # Broadcast the update
        logger.debug(f"Broadcasting update for message ID: {message_id}")
        await self.broadcast(message_id, data)
