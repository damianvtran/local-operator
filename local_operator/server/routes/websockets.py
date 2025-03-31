"""
WebSocket routes for the Local Operator API.

This module contains the FastAPI route handlers for WebSocket-related endpoints.
"""

import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from local_operator.server.dependencies import get_websocket_manager
from local_operator.server.utils.websocket_manager import WebSocketManager

router = APIRouter(prefix="/v1/ws", tags=["WebSockets"])
logger = logging.getLogger("local_operator.server.routes.websockets")


@router.websocket("/{message_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    message_id: str,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """
    WebSocket endpoint for subscribing to updates for a specific message ID.

    Args:
        websocket (WebSocket): The WebSocket connection.
        message_id (str): The message ID to subscribe to.
        websocket_manager (WebSocketManager): The WebSocket manager.
    """
    await websocket_manager.connect(websocket, message_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message_type == "subscribe":
                    subscribe_message_id = message.get("message_id")
                    if subscribe_message_id:
                        await websocket_manager.subscribe(websocket, subscribe_message_id)
                elif message_type == "unsubscribe":
                    unsubscribe_message_id = message.get("message_id")
                    if unsubscribe_message_id:
                        await websocket_manager.unsubscribe(websocket, unsubscribe_message_id)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode WebSocket message: {data}")
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for message ID: {message_id}")
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(websocket)


@router.websocket("/health")
async def websocket_health_endpoint(
    websocket: WebSocket,
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """
    WebSocket health check endpoint.

    Args:
        websocket (WebSocket): The WebSocket connection.
        websocket_manager (WebSocketManager): The WebSocket manager.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                logger.error(f"Failed to decode WebSocket message: {data}")
    except WebSocketDisconnect:
        logger.debug("Health check WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket health check error: {e}")
        await websocket.close()
