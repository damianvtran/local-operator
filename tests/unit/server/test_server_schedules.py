"""
Tests for the schedule endpoints of the FastAPI server.

This module contains tests for the schedule management functionality,
including creating, listing, updating, and deleting schedules for agents.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from local_operator.agents import AgentRegistry, AgentState
from local_operator.scheduler_service import SchedulerService
from local_operator.server.app import app  # Import the FastAPI app instance
from local_operator.server.models.schemas import (
    Agent as AgentModel,  # Corrected import for AgentModel
)
from local_operator.server.models.schemas import (
    ScheduleCreateRequest,
    ScheduleUpdateRequest,
)
from local_operator.types import Schedule as ScheduleModel
from local_operator.types import ScheduleUnit


# Fixture for a dummy agent ID
@pytest.fixture
def dummy_agent_id() -> uuid.UUID:
    """Provides a dummy agent ID."""
    return uuid.uuid4()


# Fixture for a dummy schedule ID
@pytest.fixture
def dummy_schedule_id() -> uuid.UUID:
    """Provides a dummy schedule ID."""
    return uuid.uuid4()


# Fixture for a dummy agent model
@pytest.fixture
def dummy_agent_model(dummy_agent_id: uuid.UUID) -> AgentModel:
    """Provides a dummy AgentModel instance."""
    return AgentModel(
        id=str(dummy_agent_id),
        name="Test Agent for Schedules",
        created_date=datetime.now(timezone.utc),
        version="0.1.0",
        security_prompt="Test security prompt",
        hosting="test_hosting",
        model="test_model",
        description="A test agent for schedules",
        last_message="",
        last_message_datetime=datetime.now(timezone.utc),
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        current_working_directory=".",
        tags=[],
        categories=[],
        top_k=10,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


# Fixture for a dummy schedule model
@pytest.fixture
def dummy_schedule_model(dummy_agent_id: uuid.UUID, dummy_schedule_id: uuid.UUID) -> ScheduleModel:
    """Provides a dummy ScheduleModel instance."""
    return ScheduleModel(
        id=dummy_schedule_id,
        agent_id=dummy_agent_id,
        prompt="Test Schedule - Daily Checkin",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=True,
        one_time=False,
        created_at=datetime.now(timezone.utc),
        start_time_utc=None,
        end_time_utc=None,
        last_run_at=None,
        next_run_at=None,
    )


# Fixture for a mock AgentRegistry
@pytest.fixture
def mock_agent_registry(dummy_agent_model: AgentModel, dummy_schedule_model: ScheduleModel):
    """Provides a mock AgentRegistry."""
    mock = MagicMock(spec=AgentRegistry)
    mock.get_agent.return_value = dummy_agent_model
    mock.load_agent_state.return_value = AgentState(
        version="0.1.0",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[dummy_schedule_model],  # Add dummy schedule to agent state
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )
    mock.save_agent_state.return_value = None
    mock.list_agents.return_value = [dummy_agent_model]  # For list_all_schedules
    return mock


# Fixture for a mock SchedulerService
@pytest.fixture
def mock_scheduler_service():
    """Provides a mock SchedulerService."""
    mock = MagicMock(spec=SchedulerService)
    mock.add_or_update_job.return_value = None
    mock.remove_job.return_value = None
    return mock


# Override dependencies for testing
@pytest.fixture(autouse=True)
def override_dependencies_in_app(
    mock_agent_registry: AgentRegistry, mock_scheduler_service: SchedulerService
):  # Removed test_app parameter
    """Overrides dependencies for the test application."""
    from local_operator.server.dependencies import (
        get_agent_registry,
        get_scheduler_service,
    )

    app.dependency_overrides[get_agent_registry] = lambda: mock_agent_registry  # Use imported app
    app.dependency_overrides[get_scheduler_service] = (
        lambda: mock_scheduler_service
    )  # Use imported app
    yield
    app.dependency_overrides = {}  # Use imported app


# --- Test Cases ---


# Test for POST /v1/agents/{agent_id}/schedules
@pytest.mark.asyncio
async def test_create_schedule_for_agent_success(
    test_app_client,
    dummy_agent_id: uuid.UUID,
    mock_agent_registry: MagicMock,
    mock_scheduler_service: MagicMock,
):
    """Test successful creation of a schedule for an agent."""
    schedule_data = ScheduleCreateRequest(
        prompt="Execute a daily morning script.",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=True,
        one_time=False,
        start_time_utc=None,
        end_time_utc=None,
    )

    response = await test_app_client.post(
        f"/v1/agents/{dummy_agent_id}/schedules", json=schedule_data.model_dump()
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == 201
    assert data["message"] == "Schedule created successfully"
    result = data["result"]  # This is a ScheduleResponse
    assert result["prompt"] == schedule_data.prompt
    assert result["interval"] == schedule_data.interval
    assert result["unit"] == schedule_data.unit.value  # Enum value
    assert result["agent_id"] == str(dummy_agent_id)
    assert result["is_active"] == schedule_data.is_active
    assert "id" in result
    assert "created_at" in result

    mock_agent_registry.get_agent.assert_called_once_with(str(dummy_agent_id))
    mock_agent_registry.load_agent_state.assert_called_once_with(str(dummy_agent_id))
    mock_agent_registry.save_agent_state.assert_called_once()
    mock_scheduler_service.add_or_update_job.assert_called_once()


@pytest.mark.asyncio
async def test_create_schedule_for_agent_agent_not_found(
    test_app_client, mock_agent_registry: MagicMock
):
    """Test creating a schedule for a non-existent agent."""
    non_existent_agent_id = uuid.uuid4()
    mock_agent_registry.get_agent.side_effect = KeyError(f"Agent {non_existent_agent_id} not found")

    schedule_data = ScheduleCreateRequest(
        prompt="Task for Non-existent Agent",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=True,
        one_time=False,
        start_time_utc=None,
        end_time_utc=None,
    )

    response = await test_app_client.post(
        f"/v1/agents/{non_existent_agent_id}/schedules", json=schedule_data.model_dump()
    )

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_agent_id} not found" in data["detail"]
    mock_agent_registry.load_agent_state.assert_not_called()
    mock_agent_registry.save_agent_state.assert_not_called()


@pytest.mark.asyncio
async def test_create_schedule_internal_error(
    test_app_client, dummy_agent_id: uuid.UUID, mock_agent_registry: MagicMock
):
    """Test internal server error during schedule creation."""
    mock_agent_registry.save_agent_state.side_effect = Exception("DB write error")

    schedule_data = ScheduleCreateRequest(
        prompt="This will fail",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=True,
        one_time=False,
        start_time_utc=None,
        end_time_utc=None,
    )

    response = await test_app_client.post(
        f"/v1/agents/{dummy_agent_id}/schedules", json=schedule_data.model_dump()
    )

    assert response.status_code == 500
    data = response.json()
    assert "DB write error" in data["detail"]


# Test for GET /v1/schedules
@pytest.mark.asyncio
async def test_list_all_schedules_success(
    test_app_client, mock_agent_registry: MagicMock, dummy_schedule_model: ScheduleModel
):
    """Test successful listing of all schedules."""
    # Ensure list_agents returns an agent that has the dummy_schedule_model
    agent_with_schedule = dummy_schedule_model.agent_id
    mock_agent_registry.list_agents.return_value = [
        AgentModel(
            id=str(agent_with_schedule),
            name="AgentWithSchedule",
            created_date=datetime.now(timezone.utc),
            version="1",
            security_prompt="",
            hosting="",
            model="",
            temperature=0.7,
            top_p=1.0,
            max_tokens=100,
            current_working_directory=".",
            tags=[],
            categories=[],
            top_k=10,
            description="",
            stop=None,
            seed=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
        )
    ]
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[dummy_schedule_model],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.get("/v1/schedules?page=1&per_page=10")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Schedules retrieved successfully"
    result = data["result"]
    assert result["total"] == 1
    assert result["page"] == 1
    assert result["per_page"] == 10
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["id"] == str(dummy_schedule_model.id)

    mock_agent_registry.list_agents.assert_called_once()
    mock_agent_registry.load_agent_state.assert_called_once_with(str(agent_with_schedule))


@pytest.mark.asyncio
async def test_list_all_schedules_pagination(
    test_app_client, mock_agent_registry: MagicMock, dummy_agent_id: uuid.UUID
):
    """Test pagination for listing all schedules."""
    schedules = []
    num_schedules = 15
    for i in range(num_schedules):
        schedules.append(
            ScheduleModel(
                id=uuid.uuid4(),
                agent_id=dummy_agent_id,
                created_at=datetime.now(timezone.utc),
                prompt="Test Schedule",
                interval=1,
                unit=ScheduleUnit.DAYS,
            )
        )

    # Mock agent registry to return these schedules
    # For simplicity, assume one agent has all these schedules
    mock_agent_registry.list_agents.return_value = [
        AgentModel(
            id=str(dummy_agent_id),
            name="AgentWithManySchedules",
            created_date=datetime.now(timezone.utc),
            version="1",
            security_prompt="",
            hosting="",
            model="",
            description="",
            stop=None,
            seed=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            top_k=10,
            top_p=1.0,
            temperature=0.7,
            max_tokens=100,
            current_working_directory=".",
            tags=[],
            categories=[],
        )
    ]
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=schedules,
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    # Page 1
    response_page1 = await test_app_client.get("/v1/schedules?page=1&per_page=10")
    assert response_page1.status_code == 200
    data_page1 = response_page1.json()["result"]
    assert data_page1["total"] == num_schedules
    assert data_page1["page"] == 1
    assert data_page1["per_page"] == 10
    assert len(data_page1["schedules"]) == 10

    # Page 2
    response_page2 = await test_app_client.get("/v1/schedules?page=2&per_page=10")
    assert response_page2.status_code == 200
    data_page2 = response_page2.json()["result"]
    assert data_page2["total"] == num_schedules
    assert data_page2["page"] == 2
    assert data_page2["per_page"] == 10
    assert len(data_page2["schedules"]) == 5  # Remaining 5 schedules


@pytest.mark.asyncio
async def test_list_all_schedules_internal_error(test_app_client, mock_agent_registry: MagicMock):
    """Test internal server error when listing all schedules."""
    mock_agent_registry.list_agents.side_effect = Exception("Registry unavailable")

    response = await test_app_client.get("/v1/schedules")

    assert response.status_code == 500
    data = response.json()
    assert "Registry unavailable" in data["detail"]


# Test for GET /v1/agents/{agent_id}/schedules
@pytest.mark.asyncio
async def test_list_schedules_for_agent_success(
    test_app_client,
    mock_agent_registry: MagicMock,
    dummy_agent_id: uuid.UUID,
    dummy_schedule_model: ScheduleModel,
):
    """Test successful listing of schedules for a specific agent."""
    mock_agent_registry.load_agent_state.return_value.schedules = [dummy_schedule_model]

    response = await test_app_client.get(f"/v1/agents/{dummy_agent_id}/schedules?page=1&per_page=5")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert f"Schedules for agent {dummy_agent_id} retrieved successfully" in data["message"]
    result = data["result"]
    assert result["total"] == 1
    assert result["page"] == 1
    assert result["per_page"] == 5
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["id"] == str(dummy_schedule_model.id)

    mock_agent_registry.get_agent.assert_called_once_with(str(dummy_agent_id))
    mock_agent_registry.load_agent_state.assert_called_once_with(str(dummy_agent_id))


@pytest.mark.asyncio
async def test_list_schedules_for_agent_agent_not_found(
    test_app_client, mock_agent_registry: MagicMock
):
    """Test listing schedules for a non-existent agent."""
    non_existent_agent_id = uuid.uuid4()
    mock_agent_registry.get_agent.side_effect = KeyError(f"Agent {non_existent_agent_id} not found")

    response = await test_app_client.get(f"/v1/agents/{non_existent_agent_id}/schedules")

    assert response.status_code == 404
    data = response.json()
    assert f"Agent with ID {non_existent_agent_id} not found" in data["detail"]


# Test for GET /v1/schedules/{schedule_id}
@pytest.mark.asyncio
async def test_get_schedule_by_id_success(
    test_app_client, mock_agent_registry: MagicMock, dummy_schedule_model: ScheduleModel
):
    """Test successful retrieval of a schedule by its ID."""
    # Ensure the dummy_schedule_model is part of the agent's state returned by load_agent_state
    agent_id_of_schedule = dummy_schedule_model.agent_id
    mock_agent_registry.list_agents.return_value = [
        AgentModel(
            id=str(agent_id_of_schedule),
            name="AgentWithSchedule",
            created_date=datetime.now(timezone.utc),
            version="1",
            security_prompt="",
            hosting="",
            model="",
            description="",
            stop=None,
            seed=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            top_k=10,
            top_p=1.0,
            temperature=0.7,
            max_tokens=100,
            current_working_directory=".",
            tags=[],
            categories=[],
        )
    ]
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[dummy_schedule_model],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.get(f"/v1/schedules/{dummy_schedule_model.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert f"Schedule {dummy_schedule_model.id} retrieved successfully" in data["message"]
    result = data["result"]
    assert result["id"] == str(dummy_schedule_model.id)

    mock_agent_registry.list_agents.assert_called_once()
    # load_agent_state will be called for each agent in list_agents
    mock_agent_registry.load_agent_state.assert_called_with(str(agent_id_of_schedule))


@pytest.mark.asyncio
async def test_get_schedule_by_id_not_found(test_app_client, mock_agent_registry: MagicMock):
    """Test retrieving a non-existent schedule by ID."""
    non_existent_schedule_id = uuid.uuid4()
    # Simulate no schedules found
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.get(f"/v1/schedules/{non_existent_schedule_id}")

    assert response.status_code == 404
    data = response.json()
    assert f"Schedule with ID {non_existent_schedule_id} not found" in data["detail"]


# Test for PATCH /v1/schedules/{schedule_id}
@pytest.mark.asyncio
async def test_edit_schedule_success(
    test_app_client,
    mock_agent_registry: MagicMock,
    mock_scheduler_service: MagicMock,
    dummy_schedule_model: ScheduleModel,
):
    """Test successful editing of an existing schedule."""
    update_data = ScheduleUpdateRequest(
        prompt="Updated Test Schedule Prompt",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=False,
        one_time=False,
        start_time_utc=None,
        end_time_utc=None,
    )

    # Ensure the schedule to be updated exists in the mock agent state
    agent_id_of_schedule = dummy_schedule_model.agent_id
    mock_agent_registry.list_agents.return_value = [
        AgentModel(
            id=str(agent_id_of_schedule),
            name="AgentForUpdate",
            created_date=datetime.now(timezone.utc),
            version="1",
            security_prompt="",
            hosting="",
            model="",
            description="",
            stop=None,
            seed=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            top_k=10,
            top_p=1.0,
            temperature=0.7,
            max_tokens=100,
            current_working_directory=".",
            tags=[],
            categories=[],
        )
    ]
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[dummy_schedule_model],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.patch(
        f"/v1/schedules/{dummy_schedule_model.id}", json=update_data.model_dump(exclude_unset=True)
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert f"Schedule {dummy_schedule_model.id} updated successfully" in data["message"]
    result = data["result"]
    assert result["id"] == str(dummy_schedule_model.id)
    assert result["is_active"] == update_data.is_active
    assert result["prompt"] == update_data.prompt
    assert result["interval"] == update_data.interval
    assert result["unit"] == update_data.unit

    mock_agent_registry.save_agent_state.assert_called_once()
    mock_scheduler_service.remove_job.assert_called_once_with(
        dummy_schedule_model.id
    )  # Since is_active is False
    mock_scheduler_service.add_or_update_job.assert_not_called()


@pytest.mark.asyncio
async def test_edit_schedule_activate(
    test_app_client,
    mock_agent_registry: MagicMock,
    mock_scheduler_service: MagicMock,
    dummy_schedule_model: ScheduleModel,
):
    """Test activating a schedule during edit."""
    # Ensure the schedule is initially inactive
    dummy_schedule_model.is_active = False
    agent_id_of_schedule = dummy_schedule_model.agent_id
    mock_agent_registry.list_agents.return_value = [
        AgentModel(
            id=str(agent_id_of_schedule),
            name="AgentForActivate",
            created_date=datetime.now(timezone.utc),
            version="1",
            security_prompt="",
            hosting="",
            model="",
            description="",
            stop=None,
            seed=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            top_k=10,
            top_p=1.0,
            temperature=0.7,
            max_tokens=100,
            current_working_directory=".",
            tags=[],
            categories=[],
        )
    ]
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[dummy_schedule_model],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    update_data = ScheduleUpdateRequest(
        prompt="Updated Test Schedule Prompt",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=True,
        one_time=False,
        start_time_utc=None,
        end_time_utc=None,
    )
    response = await test_app_client.patch(
        f"/v1/schedules/{dummy_schedule_model.id}", json=update_data.model_dump(exclude_unset=True)
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["is_active"] is True
    mock_scheduler_service.add_or_update_job.assert_called_once()
    mock_scheduler_service.remove_job.assert_not_called()


@pytest.mark.asyncio
async def test_edit_schedule_not_found(test_app_client, mock_agent_registry: MagicMock):
    """Test editing a non-existent schedule."""
    non_existent_schedule_id = uuid.uuid4()
    update_data = ScheduleUpdateRequest(
        prompt="Non-existent Update",
        interval=1,
        unit=ScheduleUnit.DAYS,
        is_active=True,
        one_time=False,
        start_time_utc=None,
        end_time_utc=None,
    )
    # Simulate schedule not found
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.patch(
        f"/v1/schedules/{non_existent_schedule_id}", json=update_data.model_dump(exclude_unset=True)
    )

    assert response.status_code == 404
    data = response.json()
    assert f"Schedule with ID {non_existent_schedule_id} not found" in data["detail"]


# Test for DELETE /v1/schedules/{schedule_id}
@pytest.mark.asyncio
async def test_remove_schedule_success(
    test_app_client,
    mock_agent_registry: MagicMock,
    mock_scheduler_service: MagicMock,
    dummy_schedule_model: ScheduleModel,
):
    """Test successful removal of a schedule."""
    # Ensure the schedule to be removed exists
    agent_id_of_schedule = dummy_schedule_model.agent_id
    mock_agent_registry.list_agents.return_value = [
        AgentModel(
            id=str(agent_id_of_schedule),
            name="AgentForDelete",
            created_date=datetime.now(timezone.utc),
            version="1",
            security_prompt="",
            hosting="",
            model="",
            description="",
            stop=None,
            seed=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            top_k=10,
            top_p=1.0,
            temperature=0.7,
            max_tokens=100,
            current_working_directory=".",
            tags=[],
            categories=[],
        )
    ]
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[dummy_schedule_model],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.delete(f"/v1/schedules/{dummy_schedule_model.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert f"Schedule {dummy_schedule_model.id} removed successfully" in data["message"]
    assert data["result"] == {}  # Empty result for successful deletion

    mock_agent_registry.save_agent_state.assert_called_once()
    mock_scheduler_service.remove_job.assert_called_once_with(dummy_schedule_model.id)


@pytest.mark.asyncio
async def test_remove_schedule_not_found(test_app_client, mock_agent_registry: MagicMock):
    """Test removing a non-existent schedule."""
    non_existent_schedule_id = uuid.uuid4()
    # Simulate schedule not found
    mock_agent_registry.load_agent_state.return_value = AgentState(
        version="1",
        conversation=[],
        execution_history=[],
        learnings=[],
        schedules=[],
        current_plan="",
        instruction_details="",
        agent_system_prompt="",
    )

    response = await test_app_client.delete(f"/v1/schedules/{non_existent_schedule_id}")

    assert response.status_code == 404
    data = response.json()
    assert f"Schedule with ID {non_existent_schedule_id} not found" in data["detail"]
