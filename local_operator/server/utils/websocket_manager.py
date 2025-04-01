"""
WebSocket manager for the Local Operator API.

This module provides a WebSocketManager class for managing WebSocket connections
and broadcasting updates to connected clients.
"""

import json
import logging
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

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

    async def connect(self, websocket: WebSocket, message_id: str) -> bool:
        """
        Connect a WebSocket to a message ID.

        Args:
            websocket (WebSocket): The WebSocket to connect.
            message_id (str): The message ID to connect to.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        # The websocket should already be accepted by the caller
        # This method just registers the connection in our tracking system
        try:
            # Add the WebSocket to the message connections
            if message_id not in self.message_connections:
                self.message_connections[message_id] = set()
            self.message_connections[message_id].add(websocket)

            logger.info(
                f"Connected WebSocket to message ID: {message_id} "
                f"({len(self.message_connections[message_id])} connections)"
            )

            # Add the message ID to the connection messages
            if websocket not in self.connection_messages:
                self.connection_messages[websocket] = set()
            self.connection_messages[websocket].add(message_id)

            # Send a connection established message
            try:
                # Check if the WebSocket is still connected before sending
                try:
                    # Try a simple ping to check connection
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "connection_established",
                                "message_id": message_id,
                                "status": "connected",
                            }
                        )
                    )
                    return True
                except WebSocketDisconnect:
                    # Client disconnected immediately after connecting
                    logger.warning(
                        "Client disconnected immediately after connecting "
                        f"for message ID: {message_id}"
                    )
                    # Clean up the connection state
                    self._remove_connection_tracking(websocket, message_id)
                    return False
            except Exception as e:
                logger.warning(f"Failed to send connection confirmation: {e}")
                # Log the full exception for debugging
                logger.warning(
                    f"Connection confirmation exception details: {type(e).__name__}: {str(e)}"
                )

                # Clean up the connection state but don't try to close the websocket
                self._remove_connection_tracking(websocket, message_id)
                return False
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            # Ensure we clean up any partial connection state
            self._remove_connection_tracking(websocket, message_id)
            return False

    def _remove_connection_tracking(
        self, websocket: WebSocket, message_id: Optional[str] = None
    ) -> None:
        """
        Remove a WebSocket from our tracking system without trying to close it.

        This is a non-async method that just cleans up our internal tracking state.

        Args:
            websocket (WebSocket): The WebSocket to remove from tracking.
            message_id (str, optional): A specific message ID to remove. If None,
                                        removes from all message IDs.
        """
        try:
            # If a specific message ID is provided, only remove from that one
            if message_id is not None:
                if message_id in self.message_connections:
                    self.message_connections[message_id].discard(websocket)
                    if not self.message_connections[message_id]:
                        del self.message_connections[message_id]

                if websocket in self.connection_messages:
                    self.connection_messages[websocket].discard(message_id)
                    if not self.connection_messages[websocket]:
                        del self.connection_messages[websocket]
            else:
                # Remove the WebSocket from all message connections
                for msg_id in list(self.connection_messages.get(websocket, set())):
                    if msg_id in self.message_connections:
                        self.message_connections[msg_id].discard(websocket)
                        if not self.message_connections[msg_id]:
                            del self.message_connections[msg_id]

                # Remove the WebSocket from the connection messages
                if websocket in self.connection_messages:
                    del self.connection_messages[websocket]
        except Exception as e:
            logger.error(f"Error removing WebSocket from tracking: {e}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket.

        Args:
            websocket (WebSocket): The WebSocket to disconnect.
        """
        try:
            # Try to close the WebSocket if it's still open
            try:
                await websocket.close()
            except Exception as e:
                # It's okay if this fails, the client might already be disconnected
                logger.debug(f"Error closing WebSocket during disconnect: {e}")
        except Exception as outer_e:
            # Log any unexpected errors during the close attempt
            logger.error(f"Unexpected error during WebSocket close: {outer_e}")
        finally:
            # Always clean up the connection state regardless of whether close() succeeds
            self._remove_connection_tracking(websocket)

    async def subscribe(self, websocket: WebSocket, message_id: str) -> None:
        """
        Subscribe a WebSocket to a message ID.

        Args:
            websocket (WebSocket): The WebSocket to subscribe.
            message_id (str): The message ID to subscribe to.
        """
        try:
            # Add the WebSocket to the message connections
            if message_id not in self.message_connections:
                self.message_connections[message_id] = set()
            self.message_connections[message_id].add(websocket)

            # Add the message ID to the connection messages
            if websocket not in self.connection_messages:
                self.connection_messages[websocket] = set()
            self.connection_messages[websocket].add(message_id)

            # Send a subscription confirmation message
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "subscription",
                            "message_id": message_id,
                            "status": "subscribed",
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to send subscription confirmation: {e}")
                # If we can't send the confirmation, the client might have disconnected
                # We'll keep the subscription active but log the issue
        except Exception as e:
            logger.error(f"Error subscribing WebSocket to {message_id}: {e}")

    async def unsubscribe(self, websocket: WebSocket, message_id: str) -> None:
        """
        Unsubscribe a WebSocket from a message ID.

        Args:
            websocket (WebSocket): The WebSocket to unsubscribe.
            message_id (str): The message ID to unsubscribe from.
        """
        try:
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
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "unsubscription",
                            "message_id": message_id,
                            "status": "unsubscribed",
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to send unsubscription confirmation: {e}")
                # If we can't send the confirmation, the client might have disconnected
                # The unsubscription was still processed, so we just log the issue
        except Exception as e:
            logger.error(f"Error unsubscribing WebSocket from {message_id}: {e}")

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
        try:
            json_data = json.dumps(data)
        except Exception as e:
            logger.error(f"Failed to serialize broadcast data: {e}")
            return

        # Broadcast the message to all WebSockets subscribed to the message ID
        disconnected_websockets = set()

        # Check if there are any connections for this message ID
        if message_id not in self.message_connections:
            logger.debug(f"No connections found for message ID: {message_id}")
            return

        # Make a copy to avoid modification during iteration
        connections = self.message_connections[message_id].copy()

        if not connections:
            logger.debug(f"Empty connection set for message ID: {message_id}")
            return

        for websocket in connections:
            try:
                # Check if the WebSocket is still in our tracking
                if (
                    websocket not in self.connection_messages
                    or message_id not in self.connection_messages.get(websocket, set())
                ):
                    logger.debug(f"WebSocket no longer tracked for message ID: {message_id}")
                    disconnected_websockets.add(websocket)
                    continue

                # Check if we can send a message
                await websocket.send_text(json_data)
            except Exception as e:
                logger.warning(f"Failed to broadcast message to WebSocket: {e}")
                disconnected_websockets.add(websocket)

        # Disconnect any WebSockets that failed to receive the message
        for websocket in disconnected_websockets:
            logger.debug(f"Disconnecting failed WebSocket for message ID: {message_id}")
            try:
                await self.disconnect(websocket)
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")
                # Even if disconnect fails, make sure we remove the connection from our tracking
                try:
                    # Remove the WebSocket from all message connections
                    for msg_id in self.connection_messages.get(websocket, set()):
                        if msg_id in self.message_connections:
                            self.message_connections[msg_id].discard(websocket)
                            if not self.message_connections[msg_id]:
                                del self.message_connections[msg_id]

                    # Remove the WebSocket from the connection messages
                    if websocket in self.connection_messages:
                        del self.connection_messages[websocket]
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up WebSocket state: {cleanup_error}")

    async def broadcast_update(
        self, message_id: str, execution_result: CodeExecutionResult
    ) -> None:
        """
        Broadcast an execution result update to all WebSockets subscribed to a message ID.

        Args:
            message_id (str): The message ID to broadcast to.
            execution_result (CodeExecutionResult): The execution result to broadcast.
        """
        try:
            # Convert the execution result to a dictionary
            data = execution_result.model_dump()

            # Broadcast the update
            logger.debug(f"Broadcasting update for message ID: {message_id}")
            await self.broadcast(message_id, data)
        except Exception as e:
            logger.error(f"Error broadcasting update for message ID {message_id}: {e}")
