"""
Schedule management endpoints for the Local Operator API.

This module contains the FastAPI route handlers for schedule-related endpoints.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from local_operator.agents import AgentRegistry
from local_operator.scheduler_service import SchedulerService
from local_operator.server.dependencies import get_agent_registry, get_scheduler_service
from local_operator.server.models.schemas import (
    CRUDResponse,
    ScheduleCreateRequest,
    ScheduleListResponse,
    ScheduleResponse,
    ScheduleUpdateRequest,
)
from local_operator.types import Schedule as ScheduleModel  # Renaming to avoid conflict

logger = logging.getLogger("local_operator.server.routes.schedules")
router = APIRouter(tags=["Schedules"])


@router.post(
    "/v1/agents/{agent_id}/schedules",
    response_model=CRUDResponse[ScheduleResponse],
    summary="Create a new schedule for an agent",
    status_code=201,
)
async def create_schedule_for_agent(
    agent_id: UUID,
    schedule_data: ScheduleCreateRequest,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    scheduler_service: SchedulerService = Depends(get_scheduler_service),
):
    """
    Create a new schedule for a specific agent.
    """
    try:
        agent_registry.get_agent(str(agent_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    try:
        new_schedule_model = ScheduleModel(agent_id=agent_id, **schedule_data.model_dump())
        agent_state = agent_registry.load_agent_state(str(agent_id))
        agent_state.schedules.append(new_schedule_model)
        agent_registry.save_agent_state(str(agent_id), agent_state)
        if new_schedule_model.is_active:
            scheduler_service.add_or_update_job(new_schedule_model)
        response = CRUDResponse(
            status=201,
            message="Schedule created successfully",
            result=ScheduleResponse.model_validate(new_schedule_model.model_dump()).model_dump(),
        )
        return JSONResponse(status_code=201, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating schedule for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/schedules",
    response_model=CRUDResponse[ScheduleListResponse],
    summary="List all schedules",
)
async def list_all_schedules(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Number of schedules per page"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    """
    Retrieve a paginated list of all schedules across all agents.
    """
    all_schedules: List[ScheduleModel] = []
    try:
        agents = agent_registry.list_agents()
        for agent_data in agents:
            agent_state = agent_registry.load_agent_state(agent_data.id)
            all_schedules.extend(agent_state.schedules)
        all_schedules.sort(key=lambda s: s.created_at, reverse=True)
        total = len(all_schedules)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_schedules = all_schedules[start_idx:end_idx]
        response = CRUDResponse(
            status=200,
            message="Schedules retrieved successfully",
            result=ScheduleListResponse(
                total=total,
                page=page,
                per_page=per_page,
                schedules=[
                    ScheduleResponse.model_validate(s.model_dump()) for s in paginated_schedules
                ],
            ).model_dump(),
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error listing all schedules")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/agents/{agent_id}/schedules",
    response_model=CRUDResponse[ScheduleListResponse],
    summary="List schedules for a specific agent",
)
async def list_schedules_for_agent(
    agent_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Number of schedules per page"),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    """
    Retrieve a paginated list of schedules for a specific agent.
    """
    try:
        agent_registry.get_agent(str(agent_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    try:
        agent_state = agent_registry.load_agent_state(str(agent_id))
        agent_schedules = sorted(agent_state.schedules, key=lambda s: s.created_at, reverse=True)
        total = len(agent_schedules)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_schedules = agent_schedules[start_idx:end_idx]
        response = CRUDResponse(
            status=200,
            message=f"Schedules for agent {agent_id} retrieved successfully",
            result=ScheduleListResponse(
                total=total,
                page=page,
                per_page=per_page,
                schedules=[
                    ScheduleResponse.model_validate(s.model_dump()) for s in paginated_schedules
                ],
            ).model_dump(),
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error listing schedules for agent {agent_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/schedules/{schedule_id}",
    response_model=CRUDResponse[ScheduleResponse],
    summary="Get a single schedule by ID",
)
async def get_schedule_by_id(
    schedule_id: UUID,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
):
    """
    Retrieve a single schedule by its ID.
    """
    try:
        agents = agent_registry.list_agents()
        for agent_data in agents:
            agent_state = agent_registry.load_agent_state(agent_data.id)
            for schedule in agent_state.schedules:
                if schedule.id == schedule_id:
                    response = CRUDResponse(
                        status=200,
                        message=f"Schedule {schedule_id} retrieved successfully",
                        result=ScheduleResponse.model_validate(schedule.model_dump()).model_dump(),
                    )
                    return JSONResponse(status_code=200, content=jsonable_encoder(response))
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving schedule {schedule_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/v1/schedules/{schedule_id}",
    response_model=CRUDResponse[ScheduleResponse],
    summary="Edit an existing schedule",
)
async def edit_schedule(
    schedule_id: UUID,
    schedule_data: ScheduleUpdateRequest,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    scheduler_service: SchedulerService = Depends(get_scheduler_service),
):
    """
    Edit an existing schedule by its ID.
    """
    try:
        agents = agent_registry.list_agents()
        schedule_found = False
        updated_schedule_model: Optional[ScheduleModel] = None

        for agent_data in agents:
            agent_state = agent_registry.load_agent_state(agent_data.id)
            for i, existing_schedule in enumerate(agent_state.schedules):
                if existing_schedule.id == schedule_id:
                    update_data = schedule_data.model_dump(exclude_unset=True)
                    updated_schedule_model = existing_schedule.model_copy(update=update_data)
                    agent_state.schedules[i] = updated_schedule_model
                    agent_registry.save_agent_state(agent_data.id, agent_state)
                    schedule_found = True
                    break
            if schedule_found:
                break

        if not schedule_found or not updated_schedule_model:
            raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")

        if updated_schedule_model.is_active:
            scheduler_service.add_or_update_job(updated_schedule_model)
        else:
            scheduler_service.remove_job(updated_schedule_model.id)

        response = CRUDResponse(
            status=200,
            message=f"Schedule {schedule_id} updated successfully",
            result=ScheduleResponse.model_validate(
                updated_schedule_model.model_dump()
            ).model_dump(),
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating schedule {schedule_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/v1/schedules/{schedule_id}",
    response_model=CRUDResponse,
    summary="Remove a schedule by ID",
)
async def remove_schedule(
    schedule_id: UUID,
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    scheduler_service: SchedulerService = Depends(get_scheduler_service),
):
    """
    Remove a schedule by its ID.
    """
    try:
        agents = agent_registry.list_agents()
        schedule_found_and_removed = False

        for agent_data in agents:
            agent_state = agent_registry.load_agent_state(agent_data.id)
            original_len = len(agent_state.schedules)
            agent_state.schedules = [s for s in agent_state.schedules if s.id != schedule_id]

            if len(agent_state.schedules) < original_len:
                agent_registry.save_agent_state(agent_data.id, agent_state)
                scheduler_service.remove_job(schedule_id)
                schedule_found_and_removed = True
                break

        if not schedule_found_and_removed:
            raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")

        response = CRUDResponse(
            status=200,
            message=f"Schedule {schedule_id} removed successfully",
            result={},
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error removing schedule {schedule_id}")
        raise HTTPException(status_code=500, detail=str(e))
