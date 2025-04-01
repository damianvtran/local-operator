"""
WebSocket routes for the Local Operator API.

This module contains the FastAPI route handlers for WebSocket-related endpoints.
"""

import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from local_operator.server.dependencies import get_websocket_manager_ws
from local_operator.server.models.schemas import WebsocketConnectionType
from local_operator.server.utils.websocket_manager import WebSocketManager

router = APIRouter(prefix="/v1/ws", tags=["WebSockets"])
logger = logging.getLogger("local_operator.server.routes.websockets")


@router.websocket("/messages/{message_id}")
async def websocket_message_endpoint(
    websocket: WebSocket,
    message_id: str,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager_ws),
):
    """
    WebSocket endpoint for subscribing to message updates for a specific message ID.

    Args:
        websocket (WebSocket): The WebSocket connection.
        message_id (str): The message ID to subscribe to.
        websocket_manager (WebSocketManager): The WebSocket manager.
    """
    connection_established = False
    connection_accepted = False
    try:
        # First accept the connection at the transport level
        await websocket.accept()
        connection_accepted = True
        logger.info(f"WebSocket connection accepted for message ID: {message_id}")

        # Then register the connection with the WebSocket manager
        try:
            connection_established = await websocket_manager.connect(
                websocket, message_id, WebsocketConnectionType.MESSAGE
            )

            if not connection_established:
                logger.warning(
                    f"Failed to establish WebSocket connection for message ID: {message_id}"
                )
                # Don't return immediately - try to keep the connection alive
                # The client might still be able to receive messages
                logger.info(
                    "Attempting to continue with WebSocket connection despite initial failure "
                    f"for message ID: {message_id}"
                )
                connection_established = (
                    True  # Set to true to ensure cleanup happens in finally block
                )
        except Exception as e:
            logger.error(
                f"Exception during WebSocket connection establishment "
                f"for message ID {message_id}: {e}"
            )
            # Still set connection_established to ensure cleanup in finally block
            connection_established = True

        logger.info(f"WebSocket connection established for message ID: {message_id}")

        # Main message processing loop
        while True:
            try:
                # Use a timeout to detect disconnected clients
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "ping":
                        try:
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        except Exception as e:
                            logger.warning(f"Failed to send pong response: {e}")
                            # Client might have disconnected, break the loop
                            break
                    elif message_type == "subscribe":
                        subscribe_message_id = message.get("message_id")
                        connection_type_str = message.get("connection_type", "message")
                        try:
                            connection_type = WebsocketConnectionType(connection_type_str)
                        except ValueError:
                            connection_type = WebsocketConnectionType.MESSAGE

                        if subscribe_message_id:
                            try:
                                await websocket_manager.subscribe(
                                    websocket, subscribe_message_id, connection_type
                                )
                            except Exception as e:
                                logger.error(f"Error subscribing to {subscribe_message_id}: {e}")
                                # Don't break the loop for subscription errors
                    elif message_type == "unsubscribe":
                        unsubscribe_message_id = message.get("message_id")
                        connection_type_str = message.get("connection_type", "message")
                        try:
                            connection_type = WebsocketConnectionType(connection_type_str)
                        except ValueError:
                            connection_type = WebsocketConnectionType.MESSAGE

                        if unsubscribe_message_id:
                            try:
                                await websocket_manager.unsubscribe(
                                    websocket, unsubscribe_message_id, connection_type
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error unsubscribing from {unsubscribe_message_id}: {e}"
                                )
                                # Don't break the loop for unsubscription errors
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode WebSocket message: {data}")
            except WebSocketDisconnect:
                logger.debug(
                    f"WebSocket disconnected during message processing for message ID: {message_id}"
                )
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                # Log the error and break the loop for connection errors
                if "WebSocket is not connected" in str(e) or "not connected" in str(e).lower():
                    logger.debug(f"WebSocket connection error detected: {e}")
                    logger.debug(f"Breaking WebSocket loop due to connection error: {e}")
                    break
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected during connection for message ID: {message_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Always ensure we clean up the connection
        if connection_established:
            try:
                await websocket_manager.disconnect(websocket)
                logger.debug(f"WebSocket disconnected and cleaned up for message ID: {message_id}")
            except Exception as e:
                logger.error(f"Error during WebSocket disconnect cleanup: {e}")
        elif connection_accepted:
            # If we accepted the connection but didn't establish it with the manager,
            # we still need to close the raw WebSocket
            try:
                await websocket.close()
                logger.debug(f"Raw WebSocket closed for message ID: {message_id}")
            except Exception as e:
                logger.error(f"Error closing raw WebSocket: {e}")


@router.websocket("/health")
async def websocket_health_endpoint(
    websocket: WebSocket,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager_ws),
):
    """
    WebSocket health check endpoint.

    Args:
        websocket (WebSocket): The WebSocket connection.
        websocket_manager (WebSocketManager): The WebSocket manager.
    """
    connection_established = False
    connection_accepted = False
    try:
        # First accept the connection at the transport level
        await websocket.accept()
        connection_accepted = True
        connection_established = True
        logger.debug("Health check WebSocket connection accepted")

        while True:
            try:
                # Use a timeout to detect disconnected clients
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "ping":
                        try:
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        except Exception as e:
                            logger.warning(f"Failed to send health pong response: {e}")
                            # Client might have disconnected, break the loop
                            break
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode health WebSocket message: {data}")
            except WebSocketDisconnect:
                logger.debug("Health check WebSocket disconnected during message processing")
                break
            except Exception as e:
                logger.error(f"Error processing health WebSocket message: {e}")
                # Log the error and break the loop for connection errors
                if "WebSocket is not connected" in str(e) or "not connected" in str(e).lower():
                    logger.debug(f"WebSocket health connection error detected: {e}")
                    logger.debug(f"Breaking health WebSocket loop due to connection error: {e}")
                    break
    except WebSocketDisconnect:
        logger.debug("Health check WebSocket disconnected during connection")
    except Exception as e:
        logger.error(f"WebSocket health check error: {e}")
    finally:
        # Always ensure we clean up the connection
        if connection_established:
            try:
                await websocket.close()
                logger.debug("Health check WebSocket closed")
            except Exception as e:
                logger.error(f"Error during health WebSocket close: {e}")
        elif connection_accepted:
            # If we accepted the connection but didn't fully establish it,
            # we still need to close the raw WebSocket
            try:
                await websocket.close()
                logger.debug("Raw health WebSocket closed")
            except Exception as e:
                logger.error(f"Error closing raw health WebSocket: {e}")
