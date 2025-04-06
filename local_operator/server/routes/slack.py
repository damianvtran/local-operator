"""
Slack integration endpoints for the Local Operator API.

This module contains the FastAPI route handlers for Slack-related endpoints.
"""

import logging
import json
import re
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.jobs import JobManager
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
    get_job_manager,
    get_websocket_manager,
)
from local_operator.server.models.schemas import (
    ChatRequest,
    ChatResponse,
    CRUDResponse,
    SlackChallenge,
    SlackEventWrapper,
    SlackImplementationRequest,
    SlackResponse,
)
from local_operator.server.utils.job_processor_queue import (
    run_job_in_process_with_queue,
)
from local_operator.server.utils.operator import create_operator
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.tools import execute_wsl_command

router = APIRouter(tags=["Slack"])
logger = logging.getLogger("local_operator.server.routes.slack")


@router.post(
    "/v1/slack/events",
    response_model=Union[Dict[str, str], CRUDResponse[SlackResponse]],
)
async def slack_events(
    request: Request,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    job_manager: JobManager = Depends(get_job_manager),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
) -> JSONResponse:
    """
    Handle Slack events API requests.
    
    This endpoint handles both URL verification challenges and message events from Slack.
    
    Args:
        request: The incoming request
        credential_manager: The credential manager instance
        config_manager: The config manager instance
        job_manager: The job manager instance
        websocket_manager: The websocket manager instance
        
    Returns:
        JSONResponse: Response to Slack
    """
    try:
        # Parse the request body
        body = await request.json()
        
        # Handle URL verification challenge
        if body.get("type") == "url_verification":
            challenge = SlackChallenge(**body)
            return JSONResponse(content={"challenge": challenge.challenge})
        
        # Handle event callbacks
        if body.get("type") == "event_callback":
            event_data = SlackEventWrapper(**body)
            
            # Only process message events that don't come from bots
            if event_data.event.type == "message" and not body.get("event", {}).get("bot_id"):
                # Process the message
                logger.info(f"Received message from Slack: {event_data.event.text}")
                
                # Create a response
                return JSONResponse(
                    content={
                        "status": 200,
                        "message": "Event received",
                    }
                )
            
        # Return a 200 OK for any other event types
        return JSONResponse(content={"status": 200, "message": "Event received"})
    
    except Exception as e:
        logger.error(f"Error processing Slack event: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "message": f"Error processing event: {str(e)}"},
        )


def extract_git_commands(text: str) -> List[str]:
    """
    Extract Git commands from the implementation request text.
    
    Args:
        text: The implementation request text
        
    Returns:
        List[str]: List of Git commands to execute
    """
    # Look for git commands in code blocks
    git_commands = []
    
    # Find all code blocks with bash/shell commands
    code_blocks = re.findall(r'```(?:bash|shell)?\s*(.*?)```', text, re.DOTALL)
    
    for block in code_blocks:
        # Extract git commands from the code block
        lines = block.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('git '):
                git_commands.append(line)
    
    return git_commands


def extract_wsl_config(text: str) -> Dict[str, str]:
    """
    Extract WSL2 configuration from the implementation request text.
    
    Args:
        text: The implementation request text
        
    Returns:
        Dict[str, str]: WSL2 configuration parameters
    """
    wsl_config = {
        "distribution": "Ubuntu",  # Default distribution
        "username": None,
        "password": None,
    }
    
    # Look for WSL2 configuration in the text
    distribution_match = re.search(r'use wsl2 instance named ["\']([^"\']+)["\']', text, re.IGNORECASE)
    if distribution_match:
        wsl_config["distribution"] = distribution_match.group(1)
    
    # Look for username
    username_match = re.search(r'["\']username[^"\']*["\'] ["\']([^"\']+)["\']', text, re.IGNORECASE)
    if username_match:
        wsl_config["username"] = username_match.group(1)
    
    # Look for password
    password_match = re.search(r'["\']password[^"\']*["\'] ["\']([^"\']+)["\']', text, re.IGNORECASE)
    if password_match:
        wsl_config["password"] = password_match.group(1)
    
    return wsl_config


@router.post(
    "/v1/slack/implementation",
    response_model=CRUDResponse[SlackResponse],
)
async def implementation_request(
    request: SlackImplementationRequest,
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    job_manager: JobManager = Depends(get_job_manager),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> JSONResponse:
    """
    Process implementation requests from Slack.
    
    This endpoint receives implementation requests from Slack and processes them
    using the local operator. It can also execute Git commands in WSL2 if specified.
    
    Args:
        request: The implementation request
        credential_manager: The credential manager instance
        config_manager: The config manager instance
        job_manager: The job manager instance
        agent_registry: The agent registry instance
        
    Returns:
        JSONResponse: Response with acknowledgment or error
    """
    try:
        logger.info(f"Received implementation request: {request.text}")
        
        # Check if this is a WSL2 command request
        if "use wsl2 instance" in request.text.lower():
            # Extract WSL2 configuration
            wsl_config = extract_wsl_config(request.text)
            
            # Extract Git commands
            git_commands = extract_git_commands(request.text)
            
            # Execute Git commands in WSL2
            results = []
            for cmd in git_commands:
                result = execute_wsl_command(
                    command=cmd,
                    distribution=wsl_config["distribution"],
                    username=wsl_config["username"],
                    password=wsl_config["password"]
                )
                results.append({
                    "command": cmd,
                    "success": result["success"],
                    "output": result["output"],
                    "error": result["error"] if not result["success"] else None
                })
            
            # Create a response with the results
            response_text = "## WSL2 Command Execution Results\n\n"
            for result in results:
                response_text += f"### Command: `{result['command']}`\n"
                response_text += f"**Success**: {result['success']}\n\n"
                
                if result["output"]:
                    response_text += "**Output**:\n```\n" + result["output"] + "\n```\n\n"
                
                if not result["success"] and result["error"]:
                    response_text += "**Error**:\n```\n" + result["error"] + "\n```\n\n"
            
            response = SlackResponse(
                text=response_text,
                thread_ts=request.thread_ts,
            )
            
            return JSONResponse(
                content={
                    "status": 200,
                    "message": "WSL2 commands executed",
                    "result": response.dict(),
                }
            )
        
        # If not a WSL2 command, process as a regular implementation request
        # Get configuration
        config = config_manager.get_config()
        hosting = config.values.get("hosting", "openrouter")
        model_name = config.values.get("model_name", "openai/gpt-4o-mini")
        
        # Create a chat request
        chat_request = ChatRequest(
            hosting=hosting,
            model=model_name,
            prompt=request.text,
            stream=False,
        )
        
        # Create an operator
        operator = create_operator(credential_manager, config_manager)
        
        # Process the request
        job_id = job_manager.create_job(
            prompt=chat_request.prompt,
            model=chat_request.model,
            hosting=chat_request.hosting,
        )
        
        # Run the job in a separate process
        result = await run_job_in_process_with_queue(
            job_id=job_id,
            job_manager=job_manager,
            operator=operator,
            prompt=chat_request.prompt,
            model=chat_request.model,
            hosting=chat_request.hosting,
            stream=chat_request.stream,
            context=None,
            options=None,
            attachments=None,
        )
        
        # Create a response
        response = SlackResponse(
            text=f"Implementation request received and processed. Response: {result.response}",
            thread_ts=request.thread_ts,
        )
        
        return JSONResponse(
            content={
                "status": 200,
                "message": "Implementation request processed successfully",
                "result": response.dict(),
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing implementation request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "message": f"Error processing implementation request: {str(e)}",
            },
        )