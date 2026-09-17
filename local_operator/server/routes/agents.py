"""
Agent management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for agent-related endpoints.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    HTTPException,
    Path,
    Query,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.agent_profiles import (
    SEED_ORIGIN_PREFIX,
    is_specialist,
    profile_from_agent,
)
from local_operator.agents import AgentData, AgentEditFields, AgentRegistry
from local_operator.clients._http import APIError
from local_operator.clients.radient import (
    InstructionSetError,
    RadientClient,
    build_instruction_set_document,
    validate_document_overrides,
)
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig, get_env_config
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_agent_registry,
    get_credential_manager,
    get_provider_auth_store,
)
from local_operator.server.models.schemas import (
    Agent,
    AgentCreate,
    AgentExecutionHistoryResult,
    AgentGetConversationResult,
    AgentListResult,
    AgentUpdate,
    CRUDResponse,
    ExecutionVariable,
    ExecutionVariablesResponse,
    ImportedAgent,
)
from local_operator.types import AgentState

router = APIRouter(tags=["Agents"])
logger = logging.getLogger("local_operator.server.routes.agents")


@router.get(
    "/v1/agents",
    response_model=CRUDResponse[AgentListResult],
    summary="List agents",
    description="Retrieve a paginated list of agents with their details. Optionally filter "
    "by agent name and sort by various fields.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agents list retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agents retrieved successfully",
                            "result": {
                                "total": 20,
                                "page": 1,
                                "per_page": 10,
                                "agents": [
                                    {
                                        "id": "agent123",
                                        "name": "Example Agent",
                                        "created_date": "2024-01-01T00:00:00",
                                        "version": "0.2.16",
                                        "security_prompt": "Example security prompt",
                                        "hosting": "openrouter",
                                        "model": "openai/gpt-4o-mini",
                                        "description": "An example agent",
                                        "last_message": "Hello, how can I help?",
                                        "last_message_datetime": "2024-01-01T12:00:00",
                                        "temperature": 0.7,
                                        "top_p": 1.0,
                                        "top_k": 20,
                                        "max_tokens": 2048,
                                        "stop": None,
                                        "frequency_penalty": 0.0,
                                        "presence_penalty": 0.0,
                                        "seed": None,
                                    }
                                ],
                            },
                        }
                    }
                },
            }
        },
    },
)
async def list_agents(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, description="Number of agents per page"),
    name: str = Query(None, description="Filter agents by name (case-insensitive)"),
    sort: str = Query(
        "last_message_datetime",
        description="Sort field (name, created_date, last_message_datetime)",
    ),
    direction: str = Query("desc", description="Sort direction (asc, desc)"),
):
    """
    Retrieve a paginated list of agents.

    Optionally filter the list by agent name using the 'name' query parameter.
    The filter is case-insensitive and matches agents whose names contain the provided string.

    Supports sorting by name, created_date, or last_message_datetime in ascending or
    descending order.
    Default sort is by last_message_datetime in descending order.
    """
    try:
        agents_list = agent_registry.list_agents()

        # Filter by name if provided
        if name:
            agents_list = [agent for agent in agents_list if name.lower() in agent.name.lower()]

        # Validate sort field
        valid_sort_fields = ["name", "created_date", "last_message_datetime"]
        if sort not in valid_sort_fields:
            sort = "last_message_datetime"

        # Validate direction
        is_ascending = direction.lower() == "asc"

        # Sort the agents list
        if sort == "name":
            agents_list.sort(key=lambda agent: agent.name.lower(), reverse=not is_ascending)
        elif sort == "created_date":
            # Sort directly using the datetime object, fallback to min datetime
            agents_list.sort(
                key=lambda agent: (
                    agent.created_date
                    if isinstance(agent.created_date, datetime)
                    else datetime.min.replace(tzinfo=timezone.utc)
                ),
                reverse=not is_ascending,
            )
        else:  # last_message_datetime (default)
            # Sort directly using the datetime object, fallback to created_date, then min datetime
            def get_sort_key(agent: AgentData) -> datetime:
                # Prefer last_message_datetime if it's a valid datetime
                last_msg_dt = agent.last_message_datetime
                if isinstance(last_msg_dt, datetime):
                    # Ensure timezone awareness for comparison
                    return (
                        last_msg_dt
                        if last_msg_dt.tzinfo
                        else last_msg_dt.replace(tzinfo=timezone.utc)
                    )

                # Fallback to created_date if it's a valid datetime
                created_dt = agent.created_date
                if isinstance(created_dt, datetime):
                    # Ensure timezone awareness for comparison
                    return (
                        created_dt if created_dt.tzinfo else created_dt.replace(tzinfo=timezone.utc)
                    )

                # Absolute fallback if neither date is valid
                return datetime.min.replace(tzinfo=timezone.utc)

            agents_list.sort(key=get_sort_key, reverse=not is_ascending)

    except Exception as e:
        logger.exception("Error retrieving agents")
        raise HTTPException(status_code=500, detail=f"Error retrieving agents: {e}")

    total = len(agents_list)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_agents = agents_list[start_idx:end_idx]

    # Explicitly construct Agent objects from AgentData fields for the response
    agents_for_response = [Agent.model_validate(agent.model_dump()) for agent in paginated_agents]

    result = AgentListResult(
        total=total,
        page=page,
        per_page=per_page,
        agents=agents_for_response,  # Pass the list of Agent objects
    )

    return CRUDResponse(
        status=200,
        message="Agents retrieved successfully",
        result=result.model_dump(),
    )


@router.post(
    "/v1/agents",
    response_model=CRUDResponse[Agent],
    summary="Create a new agent",
    description="Create a new agent with the provided details.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Create Agent Example",
                            "value": {
                                "name": "New Agent",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "A helpful assistant",
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Agent created successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 201,
                            "message": "Agent created successfully",
                            "result": {
                                "id": "agent123",
                                "name": "New Agent",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "A helpful assistant",
                                "last_message": "",
                                "last_message_datetime": "2024-01-01T00:00:00",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def create_agent(
    agent: AgentCreate,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> JSONResponse:
    """
    Create a new agent.
    """
    try:
        agent_edit_metadata = AgentEditFields.model_validate(agent.model_dump(exclude_unset=True))
        new_agent = agent_registry.create_agent(agent_edit_metadata)
    except ValidationError as e:
        logger.exception("Validation error creating agent")
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Error type: {type(e).__name__}")
        logger.exception("Error creating agent")
        raise HTTPException(status_code=400, detail=f"Failed to create agent: {e}")

    new_agent_serialized = new_agent.model_dump()

    response = CRUDResponse(
        status=201,
        message="Agent created successfully",
        result=new_agent_serialized,
    )
    return JSONResponse(status_code=201, content=jsonable_encoder(response))


@router.get(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse[Agent],
    summary="Retrieve an agent",
    description="Retrieve details for an agent by its ID.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent retrieved successfully",
                            "result": {
                                "id": "agent123",
                                "name": "Example Agent",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "An example agent",
                                "last_message": "Hello, how can I help?",
                                "last_message_datetime": "2024-01-01T12:00:00",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def get_agent(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to retrieve", examples=["agent123"]),
):
    """
    Retrieve an agent by ID.
    """
    try:
        agent_obj = agent_registry.get_agent(agent_id)
    except KeyError as e:
        logger.exception("Agent not found")
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    except Exception as e:
        logger.exception("Error retrieving agent")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {e}")

    if not agent_obj:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_serialized = agent_obj.model_dump()

    return CRUDResponse(
        status=200,
        message="Agent retrieved successfully",
        result=agent_serialized,
    )


@router.patch(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse[Agent],
    summary="Update an agent",
    description="Update an existing agent with new details. Only provided fields will be updated.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "example": {
                            "summary": "Update Agent Example",
                            "value": {
                                "name": "Updated Agent Name",
                                "security_prompt": "Updated security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "Updated description",
                            },
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Agent updated successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent updated successfully",
                            "result": {
                                "id": "agent123",
                                "name": "Updated Agent Name",
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Updated security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "Updated description",
                                "last_message": "Hello, how can I help?",
                                "last_message_datetime": "2024-01-01T12:00:00",
                            },
                        }
                    }
                },
            }
        },
    },
)
async def update_agent(
    agent_data: AgentUpdate,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to update", examples=["agent123"]),
):
    """
    Update an existing agent.
    """
    try:
        agent_edit_data = AgentEditFields.model_validate(agent_data.model_dump(exclude_unset=True))
        updated_agent = agent_registry.update_agent(agent_id, agent_edit_data)
    except KeyError as e:
        logger.exception("Agent not found")
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    except Exception as e:
        logger.exception("Error updating agent")
        raise HTTPException(status_code=400, detail=f"Failed to update agent: {e}")

    if not updated_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    updated_agent_serialized = updated_agent.model_dump()

    return CRUDResponse(
        status=200,
        message="Agent updated successfully",
        result=updated_agent_serialized,
    )


@router.post(
    "/v1/agents/{agent_id}/upload",
    response_model=CRUDResponse,
    summary="Upload (push) an agent to Radient Agent Hub",
    description=(
        "Upload (push) the agent with the given ID to the Radient agents marketplace. "
        "Requires RADIENT_API_KEY."
    ),
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent uploaded to Radient successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent uploaded to Radient successfully",
                            "result": {"agent_id": "radient-agent-id"},
                        }
                    }
                },
            },
            "400": {
                "description": "Bad request",
                "content": {
                    "application/json": {"example": {"detail": "Error uploading agent to Radient"}}
                },
            },
            "401": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {"example": {"detail": "RADIENT_API_KEY is required"}}
                },
            },
        },
    },
)
async def upload_agent_to_radient(
    agent_id: str = Path(..., description="ID of the agent to upload", examples=["agent123"]),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
    credential_manager: CredentialManager = Depends(get_credential_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """
    Upload (push) the agent with the given ID to the Radient agents marketplace.
    Requires RADIENT_API_KEY.
    """
    try:
        # Get config and credentials
        from local_operator.providers.radient_credentials import (
            resolve_radient_credential,
        )

        api_key = await resolve_radient_credential(
            credential_manager, env_config.radient_api_base_url, store=provider_auth_store
        )
        if not api_key:
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")
        base_url = env_config.radient_api_base_url
        radient_client = RadientClient(api_key=api_key, base_url=base_url)

        # Get agent and export as zip
        try:
            agent = agent_registry.get_agent(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        # The archive is consumed entirely within this handler, so the context
        # manager owns its temp directory. Using the bare export_agent() here
        # leaked a directory holding a full agent zip on every upload.
        with agent_registry.exported_agent_archive(agent.id) as (zip_path, _):
            # Upload to Radient
            try:
                agent_registry.upload_agent_to_radient(radient_client, agent_id, zip_path)
            except Exception as e:
                logger.exception("Error uploading agent to Radient")
                raise HTTPException(
                    status_code=400, detail=f"Error uploading agent to Radient: {e}"
                )

        return CRUDResponse(
            status=200,
            message="Agent uploaded to Radient successfully",
            result={"agent_id": agent_id},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error uploading agent to Radient")
        # If the error is about missing API key, return a clear message
        if "RADIENT_API_KEY" in str(e) or "credential" in str(e):
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")
        raise HTTPException(status_code=400, detail=f"Error uploading agent to Radient: {e}")


@router.get(
    "/v1/agents/{agent_id}/download",
    # ``ImportedAgent``, not ``Agent``: FastAPI coerces the response into the
    # declared model and DROPS unknown keys, so declaring the narrower model
    # here silently discards ``renamed_from`` — the field that tells a caller a
    # pull landed under a suffixed name (contract §3.6).
    response_model=CRUDResponse[ImportedAgent],
    summary="Download (pull) an agent from Radient Agent Hub",
    description="Download (pull) an agent from the Radient agents marketplace by agent ID.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent downloaded from Radient successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent downloaded from Radient successfully",
                            "result": {
                                "id": "imported-agent-123",
                                "name": "Imported Agent",
                                "renamed_from": None,
                                "created_date": "2024-01-01T00:00:00",
                                "version": "0.2.16",
                                "security_prompt": "Example security prompt",
                                "hosting": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "description": "An imported agent",
                                "last_message": "",
                                "last_message_datetime": "2024-01-01T00:00:00",
                            },
                        }
                    }
                },
            },
            "400": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "example": {"detail": "Error downloading agent from Radient"}
                    }
                },
            },
        },
    },
)
async def download_agent_from_radient(
    agent_id: str = Path(
        ..., description="ID of the agent to download from Radient", examples=["radient-agent-id"]
    ),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
):
    """
    Download (pull) an agent from the Radient agents marketplace by agent ID.
    """
    try:
        base_url = env_config.radient_api_base_url
        # API key not required for download
        radient_client = RadientClient(api_key=None, base_url=base_url)

        # Download from Radient
        try:
            imported_agent, renamed_from = agent_registry.download_agent_from_radient(
                radient_client, agent_id
            )
        except Exception as e:
            logger.exception("Error downloading agent from Radient")
            raise HTTPException(
                status_code=400, detail=f"Error downloading agent from Radient: {e}"
            )

        agent_serialized = imported_agent.model_dump()
        # A pull whose name is already held locally lands under a suffixed name;
        # the note is what lets the caller explain that instead of leaving the
        # user looking for an agent under the name they asked for (contract §3.6).
        agent_serialized["renamed_from"] = renamed_from
        return CRUDResponse(
            status=200,
            message="Agent downloaded from Radient successfully",
            result=agent_serialized,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error downloading agent from Radient")
        raise HTTPException(status_code=400, detail=f"Error downloading agent from Radient: {e}")


# --- Instruction-set publication ---------------------------------------------
#
# These routes carry the DOCUMENT, not the id, and they are the reason the zip
# upload path above can stay untouched: a published agent is an instruction set
# (contract §1), and an archive is how an agent published under the old standard
# is still updated and pulled. Every failure here answers with a STRUCTURE --
# ``{code, message, details}`` -- because a duplicate name, a reserved built-in
# name, a moderation refusal and an oversized document are indistinguishable
# behind one prose sentence, and each needs a different next step from the user.

#: The hub's refusal codes mapped onto the HTTP status the desktop app's error
#: surface switches on (contract §2.4, §6.2). The hub's own status travels through
#: unchanged; these are the fallback when a code arrives without one, and they are
#: the mapping the tests pin per code.
#:
#: ``name_claim_in_flight`` is the ninth code and the newest (agent-server #31): no
#: row holds the name, but a concurrent write holds it transiently. It is a 409
#: like ``name_taken`` and it needs a different next step — the caller RETRIES,
#: where a taken name is answered by choosing another — which is why the hub gives
#: it a code of its own and carries ``details.retryable``. It is listed here so the
#: status is right even if a hub ever sends it without one; the payload itself is
#: passed through untouched, which is what tells the renderer what to do.
PUBLICATION_STATUS_BY_CODE: Dict[str, int] = {
    "invalid_instruction_set": 422,
    "moderation_rejected": 422,
    "payload_too_large": 413,
    "name_taken": 409,
    "name_claim_in_flight": 409,
    "name_reserved_builtin": 409,
    "not_owner": 403,
    "agent_not_found": 404,
    "moderation_unavailable": 503,
}

#: The two codes this proxy adds to the hub's vocabulary. They exist because the
#: proxy can fail in ways the hub never sees and never describes: it can fail to
#: REACH the hub (``hub_unavailable``), and it can fail on this machine before the
#: hub is asked anything (``local_failure``). Reporting either as a hub code would
#: blame a dependency that was never involved.
HUB_UNAVAILABLE_CODE = "hub_unavailable"
LOCAL_FAILURE_CODE = "local_failure"

#: Local registry tags that ENCODE a profile field rather than tag the agent.
#: ``role`` marks the row as a delegation role, and ``tools:``/``effort:``/
#: ``delegate:`` carry fields the document publishes in their own right (the keys
#: ``profile_from_agent`` decodes). Publishing them as tags as well would put this
#: machine's registry encoding on the hub.
PROFILE_ENCODING_TAG_KEYS = frozenset({"role", "tools", "effort", "delegate"})


class AgentPublicationRequest(BaseModel):
    """The request body of a publish or republish (contract §6.4).

    ``document`` is deliberately open rather than a pydantic model: the closed
    field set of a version-1 document is enforced by
    :func:`validate_document_overrides`, so an undefined key is refused with
    ``invalid_instruction_set`` naming the field -- exactly as the hub refuses it.
    A pydantic model would instead drop the key, which is how a publisher comes to
    believe it published something it did not.
    """

    model_config = ConfigDict(extra="forbid")

    #: The fields the caller is overriding. Anything omitted is read from the
    #: local agent row, which is where the instruction body actually lives.
    document: Dict[str, Any] = Field(default_factory=dict)
    #: The hub listing to update. Required to republish: the local registry keeps
    #: no link to the listing an agent was published as, so a republish that did
    #: not name one would have to guess which row to overwrite.
    hub_agent_id: Optional[str] = Field(default=None, max_length=128)


def _publication_detail(
    code: str, message: str, details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """The structured ``detail`` of a publication failure.

    ``code`` is the key the renderer switches on and the only stable part;
    ``message`` is the sentence a human reads (the hub composes its own, so it is
    used unchanged); ``details`` carries the machine-readable rest -- which field
    was wrong, which built-in reserved the name, which moderation categories were
    cited -- and holds no prose, so a client that renders only ``message`` still
    reads correctly.
    """

    return {"code": code, "message": message, "details": dict(details or {})}


def _publication_http_error(exc: APIError) -> HTTPException:
    """Map a hub refusal onto the local response.

    A code the vocabulary defines keeps its meaning and its status; anything else
    is the hub failing to describe a failure, which is reported as
    :data:`HUB_UNAVAILABLE_CODE` with a 502 rather than as a rejection of the
    document the user sent. The upstream BODY never travels: a response shape we do
    not recognise is the case where the body may be a proxy's HTML page, and that
    belongs in neither the response nor the log.
    """

    if exc.code in PUBLICATION_STATUS_BY_CODE:
        return HTTPException(
            status_code=exc.status_code or PUBLICATION_STATUS_BY_CODE[exc.code],
            detail=_publication_detail(exc.code, str(exc), exc.details),
        )
    logger.warning(
        "Radient Agent Hub returned an unrecognised publication failure: HTTP %s",
        exc.status_code,
    )
    return HTTPException(
        status_code=502,
        detail=_publication_detail(HUB_UNAVAILABLE_CODE, str(exc)),
    )


def _invalid_document_error(exc: InstructionSetError) -> HTTPException:
    """Refuse a document this machine can already see is invalid.

    The rule text is the hub's own, so the user reads the same sentence whichever
    side refused it and the renderer needs one branch for both.
    """

    return HTTPException(
        status_code=422,
        detail=_publication_detail("invalid_instruction_set", str(exc), exc.details),
    )


def _instruction_set_fields(
    agent_registry: AgentRegistry, agent: AgentData, overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """The document fields a local registry row publishes, before overrides.

    WHY THIS MACHINE BUILDS THE DOCUMENT AND NOT THE DESKTOP APP: the instruction
    body lives here -- in the agent's ``system_prompt.md`` -- so an app-assembled
    document would be a second, drifting copy of the local-to-hub mapping. The
    renderer supplies through ``document`` only what it actually edits, and
    everything else comes from the row the user is looking at.
    """

    profile = profile_from_agent(agent_registry, agent)
    # The profile's instructions are the delegation PREAMBLE: capped, because they
    # ride in front of every turn of a child. A publication must not be truncated,
    # because a body silently cut short is a document the author never wrote,
    # published under their name. So the body is read unbounded here and the
    # document's own cap refuses it with the rule the hub would use.
    instructions = agent_registry.get_agent_system_prompt(agent.id) or ""

    fields: Dict[str, Any] = {
        "name": agent.name,
        "description": str(agent.description or ""),
        "instructions": instructions,
        # Locally, role and specialist are a registry tag and a category; the hub
        # has one explicit `kind`. A row that is neither is published as a role: a
        # published agent IS a role to whoever pulls it, and refusing it would
        # leave the user unable to publish an agent for a reason the dialog cannot
        # explain or offer a fix for.
        "kind": "specialist" if is_specialist(agent) else "role",
        # The AUTHOR's content version. Deliberately not the row's `version`, which
        # records the local-operator release that wrote agent.yml: reusing it would
        # publish an application version as the author's own.
        "version": "1.0.0",
        "tags": [
            str(tag)
            for tag in (agent.tags or [])
            if str(tag).partition(":")[0].strip().lower() not in PROFILE_ENCODING_TAG_KEYS
            and not str(tag).startswith(SEED_ORIGIN_PREFIX)
        ],
    }
    # `when_to_use` and `categories` are NOT derived. Locally a role stores its
    # routing text AS the description (``profile_from_agent``), so sending both
    # would publish one sentence twice; and the local category vocabulary is not
    # the hub's -- `specialist` is a local kind marker, not one of the hub's
    # categories -- so translating between them silently is the one thing neither
    # side is allowed to do. The caller supplies them when it means them.
    if profile.tools is not None:
        fields["tools"] = list(profile.tools)
    if profile.effort:
        fields["effort"] = profile.effort
    fields["delegate"] = profile.may_delegate

    fields.update(overrides)
    return fields


@router.post(
    "/v1/agents/{agent_id}/publish",
    response_model=CRUDResponse,
    summary="Publish an agent's instruction set to the Radient Agent Hub",
    description=(
        "Publish the agent with the given ID to the Radient Agent Hub as a version-1 "
        "instruction-set document. Requires RADIENT_API_KEY. Failures answer with a "
        "structured detail carrying the hub's own error code."
    ),
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent published to Radient",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent published to Radient successfully",
                            "result": {
                                "agent_id": "8f1c...",
                                "name": "adverse-media-screener",
                                "version": "1.0.0",
                                "document_version": 1,
                                "moderation": {"verdict": "allow"},
                            },
                        }
                    }
                },
            },
            "409": {
                "description": "The name is already published, or is reserved by a built-in",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": {
                                "code": "name_taken",
                                "message": 'The name "Coder" is already published on the hub.',
                                "details": {
                                    "existing_agent_id": "8f1c...",
                                    "owned_by_caller": False,
                                },
                            }
                        }
                    }
                },
            },
            "422": {
                "description": "The document is invalid, or the review refused it",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": {
                                "code": "invalid_instruction_set",
                                "message": "The agent document is not valid: kind must be "
                                '"role" or "specialist".',
                                "details": {
                                    "field": "kind",
                                    "rule": 'must be "role" or "specialist"',
                                },
                            }
                        }
                    }
                },
            },
            "503": {
                "description": "Publication review is temporarily unavailable",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": {
                                "code": "moderation_unavailable",
                                "message": "Publication review is temporarily unavailable. "
                                "Try again shortly.",
                                "details": {"attempts": 2},
                            }
                        }
                    }
                },
            },
        }
    },
)
async def publish_agent_to_radient(
    agent_id: str = Path(
        ..., description="ID of the local agent to publish", examples=["agent123"]
    ),
    publication: AgentPublicationRequest = Body(default_factory=AgentPublicationRequest),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
    credential_manager: CredentialManager = Depends(get_credential_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """
    Publish the agent with the given ID to the Radient Agent Hub.

    The document is built from the local row (see :func:`_instruction_set_fields`)
    with any caller-supplied overrides applied, then posted to the hub's publish
    endpoint. Nothing else about the row travels: no conversation, no execution
    history, no learnings, no schedules, no plan, no pickled context, no working
    directory, no model, no hosting, no security prompt.
    """
    try:
        # Get config and credentials
        from local_operator.providers.radient_credentials import (
            resolve_radient_credential,
        )

        api_key = await resolve_radient_credential(
            credential_manager, env_config.radient_api_base_url, store=provider_auth_store
        )
        if not api_key:
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")

        try:
            agent = agent_registry.get_agent(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")

        try:
            overrides = validate_document_overrides(publication.document)
            document = build_instruction_set_document(
                **_instruction_set_fields(agent_registry, agent, overrides)
            )
        except InstructionSetError as exc:
            raise _invalid_document_error(exc)

        radient_client = RadientClient(api_key=api_key, base_url=env_config.radient_api_base_url)
        try:
            # On a worker thread: a publication is reviewed by a model on the hub,
            # which takes seconds to tens of seconds, and the legacy zip upload's
            # precedent of calling the client inline would park this server's
            # event loop -- and therefore every other session -- for that whole
            # time.
            result = await asyncio.to_thread(radient_client.publish_agent_instruction_set, document)
        except APIError as exc:
            logger.info(
                "Radient Agent Hub refused a publication (code=%s, HTTP %s)",
                exc.code,
                exc.status_code,
            )
            raise _publication_http_error(exc)

        return CRUDResponse(
            status=200,
            message="Agent published to Radient successfully",
            result=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error publishing agent to Radient")
        # If the error is about missing API key, return a clear message
        if "RADIENT_API_KEY" in str(e) or "credential" in str(e):
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")
        raise HTTPException(
            status_code=500,
            detail=_publication_detail(
                LOCAL_FAILURE_CODE, "This agent could not be published from this machine."
            ),
        )


@router.put(
    "/v1/agents/{agent_id}/publish",
    response_model=CRUDResponse,
    summary="Republish an agent's instruction set to the Radient Agent Hub",
    description=(
        "Update a hub listing with the agent's current instruction set. Requires "
        "RADIENT_API_KEY and the id of the listing to update; only the account that "
        "published it may."
    ),
)
async def republish_agent_to_radient(
    agent_id: str = Path(
        ..., description="ID of the local agent to republish", examples=["agent123"]
    ),
    publication: AgentPublicationRequest = Body(default_factory=AgentPublicationRequest),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    env_config: EnvConfig = Depends(get_env_config),
    credential_manager: CredentialManager = Depends(get_credential_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
):
    """
    Update a hub listing with the local agent's current instruction set.

    The listing is named by ``hub_agent_id``: the local registry holds no link to
    the listing an agent was published as, so the caller -- which is looking at
    the listing -- is the only side that knows it.
    """
    try:
        from local_operator.providers.radient_credentials import (
            resolve_radient_credential,
        )

        api_key = await resolve_radient_credential(
            credential_manager, env_config.radient_api_base_url, store=provider_auth_store
        )
        if not api_key:
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")

        if not publication.hub_agent_id:
            raise HTTPException(
                status_code=422,
                detail=_publication_detail(
                    "invalid_instruction_set",
                    "An update needs the id of the listing to update: "
                    "the agent document is not valid: hub_agent_id must not be empty.",
                    {"field": "hub_agent_id", "rule": "must not be empty"},
                ),
            )

        try:
            agent = agent_registry.get_agent(agent_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")

        try:
            overrides = validate_document_overrides(publication.document)
            document = build_instruction_set_document(
                **_instruction_set_fields(agent_registry, agent, overrides)
            )
        except InstructionSetError as exc:
            raise _invalid_document_error(exc)

        radient_client = RadientClient(api_key=api_key, base_url=env_config.radient_api_base_url)
        try:
            result = await asyncio.to_thread(
                radient_client.republish_agent_instruction_set,
                publication.hub_agent_id,
                document,
            )
        except APIError as exc:
            logger.info(
                "Radient Agent Hub refused a republish (code=%s, HTTP %s)",
                exc.code,
                exc.status_code,
            )
            raise _publication_http_error(exc)

        return CRUDResponse(
            status=200,
            message="Agent republished to Radient successfully",
            result=result,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error republishing agent to Radient")
        if "RADIENT_API_KEY" in str(e) or "credential" in str(e):
            raise HTTPException(status_code=401, detail="RADIENT_API_KEY is required")
        raise HTTPException(
            status_code=500,
            detail=_publication_detail(
                LOCAL_FAILURE_CODE, "This agent could not be published from this machine."
            ),
        )


@router.get(
    "/v1/agent-name-availability",
    response_model=CRUDResponse,
    summary="Check whether an agent name is publishable",
    description=(
        "Ask the Radient Agent Hub whether a name can be published. Public, and "
        "advisory: a name reported available can still be taken by a concurrent "
        "publication."
    ),
    openapi_extra={
        "responses": {
            "200": {
                "description": "Name availability checked",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Name availability checked",
                            "result": {
                                "name": "adverse-media-screener",
                                "name_key": "adverse media screener",
                                "available": False,
                                "code": "name_reserved_builtin",
                                "details": {"builtin_name": "reviewer"},
                            },
                        }
                    }
                },
            },
            "422": {
                "description": "The name is not a valid agent name",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": {
                                "code": "invalid_instruction_set",
                                "message": "The agent document is not valid: name must not "
                                "contain whitespace.",
                                "details": {
                                    "field": "name",
                                    "rule": "must not contain whitespace",
                                },
                            }
                        }
                    }
                },
            },
        }
    },
)
async def check_agent_name_availability(
    name: str = Query(
        ..., description="The agent name to check", examples=["adverse-media-screener"]
    ),
    env_config: EnvConfig = Depends(get_env_config),
):
    """
    Ask the hub whether a name is publishable.

    Public on the hub, so no credential is resolved: a courtesy check that needed
    a signed-in account would be unavailable in exactly the state where a user is
    deciding whether to sign in.
    """
    try:
        # API key not required for this endpoint
        radient_client = RadientClient(api_key=None, base_url=env_config.radient_api_base_url)
        try:
            result = await asyncio.to_thread(radient_client.check_agent_name_availability, name)
        except APIError as exc:
            logger.info(
                "Radient Agent Hub refused a name availability check (code=%s, HTTP %s)",
                exc.code,
                exc.status_code,
            )
            raise _publication_http_error(exc)

        return CRUDResponse(
            status=200,
            message="Name availability checked",
            result=result,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error checking agent name availability on Radient")
        raise HTTPException(
            status_code=500,
            detail=_publication_detail(
                LOCAL_FAILURE_CODE, "The agent name could not be checked from this machine."
            ),
        )


@router.delete(
    "/v1/agents/{agent_id}",
    response_model=CRUDResponse,
    summary="Delete an agent",
    description="Delete an existing agent by its ID.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent deleted successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent deleted successfully",
                            "result": {},
                        }
                    }
                },
            }
        },
    },
)
async def delete_agent(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to delete", examples=["agent123"]),
):
    """
    Delete an existing agent.
    """
    try:
        agent_registry.delete_agent(agent_id)
    except KeyError as e:
        logger.exception("Agent not found")
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")
    except Exception as e:
        logger.exception("Error deleting agent")
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {e}")

    return CRUDResponse(
        status=200,
        message="Agent deleted successfully",
        result={},
    )


@router.get(
    "/v1/agents/{agent_id}/conversation",
    response_model=CRUDResponse[AgentGetConversationResult],
    summary="Get agent conversation history",
    description="Retrieve the conversation history for a specific agent.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent conversation retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent conversation retrieved successfully",
                            "result": {
                                "agent_id": "agent123",
                                "last_message_datetime": "2023-01-01T12:00:00",
                                "first_message_datetime": "2023-01-01T11:00:00",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant",
                                        "should_summarize": False,
                                        "summarized": False,
                                        "timestamp": "2023-01-01T11:00:00",
                                    },
                                    {
                                        "role": "user",
                                        "content": "Hello, how are you?",
                                        "should_summarize": True,
                                        "summarized": False,
                                        "timestamp": "2023-01-01T11:00:00",
                                    },
                                ],
                                "page": 1,
                                "per_page": 10,
                                "total": 2,
                                "count": 2,
                            },
                        }
                    }
                },
            }
        }
    },
)
async def get_agent_conversation(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(
        ..., description="ID of the agent to get conversation for", examples=["agent123"]
    ),
    page: int = Query(1, ge=1, description="Page number to retrieve"),
    per_page: int = Query(10, ge=1, le=100, description="Number of messages per page"),
):
    """
    Retrieve the conversation history for a specific agent.

    Args:
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent
        page: The page number to retrieve (starts at 1)
        per_page: The number of messages per page (between 1 and 100)

    Returns:
        AgentGetConversationResult: The conversation history for the agent

    Raises:
        HTTPException: If the agent registry is not initialized or the agent is not found
    """
    try:
        conversation_history = agent_registry.get_agent_conversation_history(agent_id)
        total_messages = len(conversation_history)

        # Set default datetime values in case the conversation is empty
        first_message_datetime = datetime.now()
        last_message_datetime = datetime.now()

        if conversation_history:
            # Find the first and last message timestamps. ``timestamp`` is
            # optional on ConversationRecord, so records predating it are
            # skipped and an all-untimestamped history keeps the defaults set
            # above (min/max raise ValueError on an empty sequence).
            try:
                first_message_datetime = min(
                    msg.timestamp for msg in conversation_history if msg.timestamp is not None
                )

                last_message_datetime = max(
                    msg.timestamp for msg in conversation_history if msg.timestamp is not None
                )
            except (AttributeError, ValueError):
                # If timestamps aren't available, use current time
                pass

        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_messages)

        # Check if page is out of bounds
        if start_idx >= total_messages and total_messages > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} is out of bounds. "
                f"Total pages: {(total_messages + per_page - 1) // per_page}",
            )

        # Pages move backward in history, so we start from the end of the array and
        # move backward while maintaining the same order of messages
        paginated_messages = (
            conversation_history[-end_idx : -start_idx or None] if conversation_history else []
        )

        result = AgentGetConversationResult(
            agent_id=agent_id,
            first_message_datetime=first_message_datetime,
            last_message_datetime=last_message_datetime,
            messages=paginated_messages,
            page=page,
            per_page=per_page,
            total=total_messages,
            count=len(paginated_messages),
        )

        return CRUDResponse(
            status=200,
            message="Agent conversation retrieved successfully",
            result=result.model_dump(),
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving agent conversation")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving agent conversation: {str(e)}"
        )


@router.delete(
    "/v1/agents/{agent_id}/conversation",
    response_model=CRUDResponse,
    summary="Clear agent conversation",
    description="Clear the conversation history for a specific agent.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Agent conversation cleared successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": 200,
                            "message": "Agent conversation cleared successfully",
                            "result": {},
                        }
                    }
                },
            },
            "404": {
                "description": "Agent not found",
                "content": {
                    "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
                },
            },
            "500": {
                "description": "Internal server error",
                "content": {
                    "application/json": {"example": {"detail": "Error clearing agent conversation"}}
                },
            },
        },
    },
)
async def clear_agent_conversation(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(
        ..., description="ID of the agent to clear conversation for", examples=["agent123"]
    ),
):
    """
    Clear the conversation history for a specific agent.

    Args:
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent

    Returns:
        CRUDResponse: A response indicating success or failure

    Raises:
        HTTPException: If the agent registry is not initialized or the agent is not found
    """
    try:
        # Get the agent to verify it exists
        agent = agent_registry.get_agent(agent_id)

        # Get the current agent state
        agent_state = agent_registry.load_agent_state(agent_id)

        # Clear the conversation by saving an empty list
        agent_registry.save_agent_state(
            agent_id=agent_id,
            agent_state=AgentState(
                version=agent.version,
                conversation=[],
                execution_history=[],
                learnings=agent_state.learnings,
                schedules=agent_state.schedules,
                current_plan="",
                instruction_details=agent_state.instruction_details,
                agent_system_prompt=agent_state.agent_system_prompt,
            ),
        )
        agent_registry.save_agent_context(agent_id=agent_id, context={})

        return CRUDResponse(
            status=200,
            message="Agent conversation cleared successfully",
            result={},
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except Exception as e:
        logger.exception("Error clearing agent conversation")
        raise HTTPException(status_code=500, detail=f"Error clearing agent conversation: {str(e)}")


@router.post(
    "/v1/agents/import",
    # ``ImportedAgent``: same reason as the download route above. This handler
    # returns a JSONResponse directly (it needs 201), which FastAPI passes
    # through unvalidated — so the declaration is documentation here, and it has
    # to describe what the handler actually sends.
    response_model=CRUDResponse[ImportedAgent],
    summary="Import an agent",
    description=(
        "Import an agent from a ZIP file containing agent state files with an agent.yml file."
    ),
    responses={
        201: {
            "description": "Agent imported successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 201,
                        "message": "Agent imported successfully",
                        "result": {
                            "id": "imported-agent-123",
                            "name": "Imported Agent",
                            "renamed_from": None,
                            "created_date": "2024-01-01T00:00:00",
                            "version": "0.2.16",
                            "security_prompt": "Example security prompt",
                            "hosting": "openrouter",
                            "model": "openai/gpt-4o-mini",
                            "description": "An imported agent",
                            "last_message": "",
                            "last_message_datetime": "2024-01-01T00:00:00",
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {"example": {"detail": "Invalid ZIP file or missing agent.yml"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Error importing agent"}}},
        },
    },
)
async def import_agent(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    file: UploadFile = File(..., description="ZIP file containing agent state files"),
) -> JSONResponse:
    """
    Import an agent from a ZIP file.

    The ZIP file should contain agent state files with an agent.yml file.
    A new ID will be assigned to the imported agent, and the current working directory
    will be reset to local-operator-home.

    Args:
        agent_registry: The agent registry dependency
        file: The uploaded ZIP file containing agent state files

    Returns:
        CRUDResponse: A response containing the imported agent details

    Raises:
        HTTPException: If there is an error importing the agent
    """
    # Create a temporary directory to save the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = FilePath(temp_dir)
        zip_path = temp_dir_path / "agent.zip"

        # Save the uploaded file to the temporary directory
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # Use the AgentRegistry's import_agent method
        try:
            agent_obj, renamed_from = agent_registry.import_agent(zip_path)
            agent_serialized = agent_obj.model_dump()
            # Named on the response so the caller can say "imported as X — you
            # already have an agent called Y" rather than silently handing back
            # a name under which nothing the user asked for can be found. The
            # key is always present (null when nothing was renamed) so a client
            # never has to guess whether the backend was just too old to send it.
            agent_serialized["renamed_from"] = renamed_from

            response = CRUDResponse(
                status=201,
                message="Agent imported successfully",
                result=agent_serialized,
            )
            return JSONResponse(status_code=201, content=jsonable_encoder(response))
        except ValueError as e:
            # Handle ValueError directly as 400 Bad Request
            error_msg = str(e)
            logger.exception(f"Invalid agent import data: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except zipfile.BadZipFile:
            # Handle BadZipFile directly as 400 Bad Request
            logger.exception("Invalid ZIP file")
            raise HTTPException(status_code=400, detail="Invalid ZIP file")
        except Exception as e:
            # For other exceptions, check if they contain known error messages
            error_msg = str(e)
            logger.exception(f"Error importing agent: {error_msg}")

            # Check for specific error messages that should be 400 errors
            if "Missing agent.yml" in error_msg or "Invalid ZIP file" in error_msg:
                raise HTTPException(status_code=400, detail=error_msg)

            # Otherwise, return 500 Internal Server Error
            raise HTTPException(status_code=500, detail=f"Error importing agent: {error_msg}")


@router.get(
    "/v1/agents/{agent_id}/export",
    summary="Export an agent",
    description="Export an agent's state files as a ZIP file.",
    responses={
        200: {
            "description": "Agent exported successfully",
            "content": {"application/octet-stream": {}},
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Error exporting agent"}}},
        },
    },
)
async def export_agent(
    background_tasks: BackgroundTasks,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent to export", examples=["agent123"]),
) -> FileResponse:
    """
    Export an agent's state files as a ZIP file.

    Args:
        background_tasks: FastAPI background tasks for cleanup
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent to export

    Returns:
        StreamingResponse: A streaming response containing the ZIP file

    Raises:
        HTTPException: If the agent is not found or there is an error exporting the agent
    """
    try:
        # export_agent_archive rather than export_agent: this route cannot use
        # the exported_agent_archive context manager (FileResponse streams the
        # file after the handler returns), so it must reclaim the directory
        # itself — and it needs the directory the export actually created.
        temp_dir, zip_path, filename = agent_registry.export_agent_archive(agent_id)

        # Ensure the file exists before returning it
        if not zip_path.exists():
            raise FileNotFoundError(f"Failed to create ZIP file at {zip_path}")

        # Clean up after the response is sent. Removing the captured temp dir
        # rather than zip_path.parent keeps this correct regardless of the
        # filename: the agent name is attacker-controllable via import_agent,
        # and a traversal-shaped name once made .parent an unrelated directory
        # that this task then deleted.
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

        # Return the ZIP file as a streaming response
        return FileResponse(
            path=zip_path,
            filename=filename,
            media_type="application/octet-stream",
        )

    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except Exception as e:
        logger.exception("Error exporting agent")
        raise HTTPException(status_code=500, detail=f"Error exporting agent: {str(e)}")


@router.get(
    "/v1/agents/{agent_id}/history",
    response_model=CRUDResponse[AgentExecutionHistoryResult],
    summary="Get agent execution history",
    description="Retrieve the execution history for a specific agent.",
    responses={
        200: {
            "description": "Agent execution history retrieved successfully",
            "model": CRUDResponse[AgentExecutionHistoryResult],
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Agent execution history retrieved successfully",
                        "result": {
                            "agent_id": "agent123",
                            "history": [
                                {
                                    "code": "print('Hello, world!')",
                                    "stdout": "Hello, world!",
                                    "stderr": "",
                                    "logging": "",
                                    "message": "Code executed successfully",
                                    "formatted_print": "Hello, world!",
                                    "role": "system",
                                    "status": "success",
                                    "timestamp": "2024-01-01T12:00:00Z",
                                    "execution_type": "action",
                                    "action": "CODE",
                                    "task_classification": "data_science",
                                }
                            ],
                            "first_execution_datetime": "2024-01-01T12:00:00Z",
                            "last_execution_datetime": "2024-01-01T12:00:00Z",
                            "page": 1,
                            "per_page": 10,
                            "total": 1,
                            "count": 1,
                        },
                    }
                }
            },
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {"detail": "Page 2 is out of bounds. Total pages: 1"}
                }
            },
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error retrieving agent execution history"}
                }
            },
        },
    },
)
async def get_agent_execution_history(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(
        ..., description="ID of the agent to get execution history for", examples=["agent123"]
    ),
    page: int = Query(1, ge=1, description="Page number to retrieve"),
    per_page: int = Query(10, ge=1, le=100, description="Number of executions per page"),
):
    """
    Get the execution history for a specific agent.
    """
    try:
        execution_history = agent_registry.get_agent_execution_history(agent_id)
        total_executions = len(execution_history)

        # Default timestamps if no executions
        first_execution_datetime = datetime.now(timezone.utc)
        last_execution_datetime = datetime.now(timezone.utc)

        # Get actual timestamps if executions exist
        if execution_history:
            try:
                timestamps = [
                    execution.timestamp
                    for execution in execution_history
                    if execution.timestamp is not None
                ]
                if timestamps:
                    try:
                        first_execution_datetime = min(timestamps)
                        last_execution_datetime = max(timestamps)
                    except TypeError:
                        # Handle offset-naive and offset-aware datetime comparison
                        def to_aware(dt: datetime) -> datetime:
                            if dt.tzinfo is None:
                                return dt.replace(tzinfo=timezone.utc)
                            return dt

                        try:
                            aware_timestamps = [to_aware(dt) for dt in timestamps]
                            first_execution_datetime = min(aware_timestamps)
                            last_execution_datetime = max(aware_timestamps)
                        except Exception:
                            logger.exception(
                                "Failed to normalize datetimes in agent execution history"
                            )
                            # Fallback to current time if normalization fails
                            first_execution_datetime = datetime.now(timezone.utc)
                            last_execution_datetime = datetime.now(timezone.utc)
                else:
                    # No valid timestamps, use current time
                    first_execution_datetime = datetime.now(timezone.utc)
                    last_execution_datetime = datetime.now(timezone.utc)
            except (AttributeError, ValueError, TypeError):
                logger.exception("Error processing execution timestamps")
                # If timestamps aren't available or error occurs, use current time
                first_execution_datetime = datetime.now(timezone.utc)
                last_execution_datetime = datetime.now(timezone.utc)

        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_executions)

        # Check if page is out of bounds
        if start_idx >= total_executions and total_executions > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} is out of bounds. "
                f"Total pages: {(total_executions + per_page - 1) // per_page}",
            )

        # Pages move backward in history, so we start from the end of the array and
        # move backward while maintaining the same order of executions
        paginated_history = (
            execution_history[-end_idx : -start_idx or None] if execution_history else []
        )

        result = AgentExecutionHistoryResult(
            agent_id=agent_id,
            first_execution_datetime=first_execution_datetime,
            last_execution_datetime=last_execution_datetime,
            history=paginated_history,
            page=page,
            per_page=per_page,
            total=total_executions,
            count=len(paginated_history),
        )

        return CRUDResponse(
            status=200,
            message="Agent execution history retrieved successfully",
            result=result.model_dump(),
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving agent execution history")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving agent execution history: {str(e)}"
        )


@router.get(
    "/v1/agents/{agent_id}/system-prompt",
    response_model=CRUDResponse,
    summary="Get agent system prompt",
    description="Retrieve the system prompt for a specific agent.",
    responses={
        200: {
            "description": "Agent system prompt retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Agent system prompt retrieved successfully",
                        "result": {"system_prompt": "You are a helpful assistant..."},
                    }
                }
            },
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Error retrieving agent system prompt"}}
            },
        },
    },
)
async def get_agent_system_prompt(
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent", examples=["agent123"]),
):
    """
    Retrieve the system prompt for a specific agent.

    Args:
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent

    Returns:
        CRUDResponse: A response containing the agent's system prompt

    Raises:
        HTTPException: If the agent is not found or there is an error retrieving the system prompt
    """
    try:
        system_prompt = agent_registry.get_agent_system_prompt(agent_id)
        return CRUDResponse(
            status=200,
            message="Agent system prompt retrieved successfully",
            result={"system_prompt": system_prompt},
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except FileNotFoundError as e:
        logger.exception(f"System prompt file not found for agent {agent_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except IOError as e:
        logger.exception(f"Error reading system prompt for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Error retrieving agent system prompt")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving agent system prompt: {str(e)}"
        )


@router.put(
    "/v1/agents/{agent_id}/system-prompt",
    response_model=CRUDResponse,
    summary="Update agent system prompt",
    description="Update the system prompt for a specific agent.",
    responses={
        200: {
            "description": "Agent system prompt updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": 200,
                        "message": "Agent system prompt updated successfully",
                        "result": {},
                    }
                }
            },
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {"example": {"detail": "Agent with ID agent123 not found"}}
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {"example": {"detail": "Error updating agent system prompt"}}
            },
        },
    },
)
async def update_agent_system_prompt(
    system_prompt: Dict[str, str],
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    agent_id: str = Path(..., description="ID of the agent", examples=["agent123"]),
):
    """
    Update the system prompt for a specific agent.

    Args:
        system_prompt: A dictionary containing the system prompt text
        agent_registry: The agent registry dependency
        agent_id: The unique identifier of the agent

    Returns:
        CRUDResponse: A response indicating success or failure

    Raises:
        HTTPException: If the agent is not found or there is an error updating the system prompt
    """
    try:
        if "system_prompt" not in system_prompt:
            raise HTTPException(
                status_code=422, detail="Request body must contain 'system_prompt' field"
            )

        agent_registry.set_agent_system_prompt(agent_id, system_prompt["system_prompt"])
        return CRUDResponse(
            status=200,
            message="Agent system prompt updated successfully",
            result={},
        )
    except KeyError:
        logger.exception(f"Agent with ID {agent_id} not found")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except IOError as e:
        logger.exception(f"Error writing system prompt for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        if isinstance(e, HTTPException):
            raise

        logger.exception("Error updating agent system prompt")
        raise HTTPException(status_code=500, detail=f"Error updating agent system prompt: {str(e)}")


# Agent Execution Variables CRUD Endpoints
@router.get(
    "/v1/agents/{agent_id}/execution-variables",
    response_model=CRUDResponse[ExecutionVariablesResponse],
    summary="List agent execution variables",
    description="Retrieve all execution variables for a specific agent.",
)
async def list_agent_execution_variables(
    agent_id: str = Path(..., description="ID of the agent"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> CRUDResponse[ExecutionVariablesResponse]:
    try:
        variables = agent_registry.load_agent_context(agent_id)

        if variables is None:
            return CRUDResponse(
                status=200,
                message="No execution variables found",
                result=ExecutionVariablesResponse(execution_variables=[]),
            )

        string_variables = [
            ExecutionVariable(key=k, value=str(v), type=type(v).__name__)
            for k, v in variables.items()
        ]

        return CRUDResponse(
            status=200,
            message="Execution variables retrieved successfully",
            result=ExecutionVariablesResponse(execution_variables=string_variables),
        )
    except KeyError:
        logger.warning(f"Agent not found when listing execution variables: {agent_id}")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except Exception as e:
        logger.exception(f"Error listing execution variables for agent {agent_id}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving execution variables: {str(e)}"
        )


@router.post(
    "/v1/agents/{agent_id}/execution-variables",
    response_model=CRUDResponse[ExecutionVariable],
    summary="Create an agent execution variable",
    description="Create a new execution variable for a specific agent.",
)
async def create_agent_execution_variable(
    variable_data: ExecutionVariable,
    agent_id: str = Path(..., description="ID of the agent"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> JSONResponse:
    try:
        # Coerce the value to the correct type based on the type field
        coerced_value = variable_data.value
        if variable_data.type == "int":
            coerced_value = int(variable_data.value)
        elif variable_data.type == "float":
            coerced_value = float(variable_data.value)
        elif variable_data.type == "bool":
            coerced_value = variable_data.value.lower() in ("true", "1", "yes", "on")
        elif variable_data.type == "list":
            # Try to parse as JSON list, fallback to string split
            coerced_value = json.loads(variable_data.value)
            if not isinstance(coerced_value, list):
                raise ValueError("Value is not a valid list")
        elif variable_data.type == "dict":
            # Try to parse as JSON dict
            coerced_value = json.loads(variable_data.value)
            if not isinstance(coerced_value, dict):
                raise ValueError("Value is not a valid dict")
        # For 'str' type or any other type, keep as string

        agent_registry.create_context_variable(agent_id, variable_data.key, coerced_value)
        response_content = CRUDResponse(
            status=201,
            message="Execution variable created successfully",
            result=ExecutionVariable(
                key=variable_data.key,
                value=str(coerced_value),  # Ensure value is string for response
                type=type(coerced_value).__name__,
            ),
        )
        return JSONResponse(status_code=201, content=jsonable_encoder(response_content))
    except KeyError:
        logger.warning(f"Agent not found when creating execution variable: {agent_id}")
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    except (
        ValueError
    ) as e:  # Handles case where variable key already exists or type conversion errors
        logger.warning(
            f"Attempt to create existing execution variable '{variable_data.key}' "
            f"for agent {agent_id} or type conversion error: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Error creating execution variable '{variable_data.key}' for agent {agent_id}"
        )
        raise HTTPException(status_code=500, detail=f"Error creating execution variable: {str(e)}")


@router.get(
    "/v1/agents/{agent_id}/execution-variables/{variable_key}",
    response_model=CRUDResponse[ExecutionVariable],
    summary="Get an agent execution variable",
    description="Retrieve a specific execution variable for an agent by its key.",
)
async def get_agent_execution_variable(
    agent_id: str = Path(..., description="ID of the agent"),
    variable_key: str = Path(..., description="Key of the execution variable"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> CRUDResponse[ExecutionVariable]:
    try:
        value = agent_registry.get_context_variable(agent_id, variable_key)
        return CRUDResponse(
            status=200,
            message="Execution variable retrieved successfully",
            result=ExecutionVariable(key=variable_key, value=value, type=type(value).__name__),
        )
    except KeyError as e:
        logger.warning(
            f"Agent '{agent_id}' or variable '{variable_key}' not found "
            "when getting execution variable"
        )
        # Distinguish between agent not found and variable not found for clarity
        if f"Agent with id {agent_id} not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution variable '{variable_key}' not found for agent {agent_id}",
            )
    except Exception as e:
        logger.exception(
            f"Error retrieving execution variable '{variable_key}' for agent {agent_id}"
        )
        raise HTTPException(
            status_code=500, detail=f"Error retrieving execution variable: {str(e)}"
        )


@router.patch(
    "/v1/agents/{agent_id}/execution-variables/{variable_key}",
    response_model=CRUDResponse[ExecutionVariable],
    summary="Update an agent execution variable",
    description="Update an existing execution variable for a specific agent.",
)
async def update_agent_execution_variable(
    variable_data: ExecutionVariable,
    agent_id: str = Path(..., description="ID of the agent"),
    variable_key: str = Path(..., description="Key of the execution variable to update"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
) -> CRUDResponse[ExecutionVariable]:
    try:
        # Coerce the value to the correct type if type is provided
        coerced_value = variable_data.value
        if variable_data.type:
            try:
                if variable_data.type == "int":
                    coerced_value = int(variable_data.value)
                elif variable_data.type == "float":
                    coerced_value = float(variable_data.value)
                elif variable_data.type == "bool":
                    coerced_value = variable_data.value.lower() in ("true", "1", "yes", "on")
                elif variable_data.type == "str":
                    coerced_value = str(variable_data.value)
                elif variable_data.type == "list":
                    coerced_value = json.loads(variable_data.value)
                    if not isinstance(coerced_value, list):
                        raise ValueError("Value is not a valid list")
                elif variable_data.type == "dict":
                    coerced_value = json.loads(variable_data.value)
                    if not isinstance(coerced_value, dict):
                        raise ValueError("Value is not a valid dict")
                # Add more type coercions as needed
            except (ValueError, AttributeError) as type_error:
                raise ValueError(
                    f"Cannot convert '{variable_data.value}' to type "
                    f"'{variable_data.type}': {str(type_error)}"
                )

        updated_context = agent_registry.update_context_variable(
            agent_id, variable_key, coerced_value
        )
        updated_value = updated_context.get(variable_key)

        return CRUDResponse(
            status=200,
            message="Execution variable updated successfully",
            result=ExecutionVariable(
                key=variable_key,
                value=str(updated_value),  # Ensure value is string for the response model
                type=type(updated_value).__name__,
            ),
        )
    except KeyError as e:
        logger.warning(
            f"Agent '{agent_id}' or variable '{variable_key}' not found "
            "when updating execution variable"
        )
        if f"Agent with id {agent_id} not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution variable '{variable_key}' not found for agent {agent_id}",
            )
    except ValueError as e:
        logger.warning(
            f"Type conversion error when updating execution variable '{variable_key}' "
            f"for agent {agent_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error updating execution variable '{variable_key}' for agent {agent_id}")
        raise HTTPException(status_code=500, detail=f"Error updating execution variable: {str(e)}")


@router.delete(
    "/v1/agents/{agent_id}/execution-variables/{variable_key}",
    summary="Delete an agent execution variable",
    description="Delete an execution variable for a specific agent by its key.",
)
async def delete_agent_execution_variable(
    agent_id: str = Path(..., description="ID of the agent"),
    variable_key: str = Path(..., description="Key of the execution variable to delete"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    try:
        agent_registry.delete_context_variable(agent_id, variable_key)
        return CRUDResponse(
            status=200,
            message="Execution variable deleted successfully",
        )
    except KeyError as e:
        logger.warning(
            f"Agent '{agent_id}' or variable '{variable_key}' not found "
            "when deleting execution variable"
        )
        if f"Agent with id {agent_id} not found" in str(e):
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution variable '{variable_key}' not found for agent {agent_id}",
            )
    except Exception as e:
        logger.exception(f"Error deleting execution variable '{variable_key}' for agent {agent_id}")
        raise HTTPException(status_code=500, detail=f"Error deleting execution variable: {str(e)}")
