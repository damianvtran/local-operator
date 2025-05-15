import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from pydantic import ValidationError

from local_operator.agents import AgentData, AgentRegistry, AgentState
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import Job, JobManager, JobStatus
from local_operator.operator import OperatorType
from local_operator.scheduler_service import (
    SchedulerService,
    _execute_scheduled_task_logic,
)
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import Schedule, ScheduleUnit


@pytest.fixture
def mock_agent_registry():
    mock = MagicMock(spec=AgentRegistry)
    mock.config_dir = "/mock/agent/registry_config_dir"
    return mock


@pytest.fixture
def mock_config_manager():
    return MagicMock(spec=ConfigManager)


@pytest.fixture
def mock_credential_manager():
    return MagicMock(spec=CredentialManager)


@pytest.fixture
def mock_env_config():
    return MagicMock(spec=EnvConfig)


@pytest.fixture
def mock_job_manager():
    mock = AsyncMock(spec=JobManager)
    return mock


@pytest.fixture
def mock_websocket_manager():
    return MagicMock(spec=WebSocketManager)


@pytest.fixture
def mock_scheduler():
    mock = MagicMock(spec=AsyncIOScheduler)
    mock.running = False  # Default to not running
    return mock


@pytest.fixture
@patch("local_operator.scheduler_service.AsyncIOScheduler")
def scheduler_service(
    mock_async_io_scheduler,
    mock_agent_registry,
    mock_config_manager,
    mock_credential_manager,
    mock_env_config,
    mock_job_manager,
    mock_websocket_manager,
    mock_scheduler,  # Add mock_scheduler as a parameter
):
    # Patch the AsyncIOScheduler constructor to return our mock_scheduler
    mock_async_io_scheduler.return_value = mock_scheduler  # Use the injected fixture

    service = SchedulerService(
        agent_registry=mock_agent_registry,
        config_manager=mock_config_manager,
        credential_manager=mock_credential_manager,
        env_config=mock_env_config,
        operator_type=OperatorType.CLI,  # Changed PYTHON to CLI
        verbosity_level=VerbosityLevel.DEBUG,
        job_manager=mock_job_manager,
        websocket_manager=mock_websocket_manager,
    )
    # The service's scheduler attribute should now be our mock_scheduler instance
    service.scheduler = mock_async_io_scheduler.return_value
    return service


# Helper to create a basic schedule
def create_schedule(
    agent_id: UUID,
    schedule_id: UUID,
    prompt: str = "Test prompt",
    interval: int = 1,
    unit: ScheduleUnit = ScheduleUnit.MINUTES,
    one_time: bool = False,
    is_active: bool = True,
    start_time_utc: Optional[datetime] = None,
    end_time_utc: Optional[datetime] = None,
    last_run_at: Optional[datetime] = None,
) -> Schedule:
    return Schedule(
        id=schedule_id,
        agent_id=agent_id,
        prompt=prompt,
        interval=interval,
        unit=unit,
        one_time=one_time,
        is_active=is_active,
        start_time_utc=start_time_utc,
        end_time_utc=end_time_utc,
        last_run_at=last_run_at,
        created_at=datetime.now(timezone.utc),
        # updated_at removed
    )


class TestSchedulerService:
    @pytest.mark.asyncio
    async def test_scheduler_service_initialization(self, scheduler_service, mock_agent_registry):
        assert scheduler_service.agent_registry == mock_agent_registry
        assert scheduler_service.operator_type == OperatorType.CLI  # Changed PYTHON to CLI
        assert scheduler_service.verbosity_level == VerbosityLevel.DEBUG
        assert scheduler_service.scheduler is not None
        assert isinstance(scheduler_service.scheduler, MagicMock)  # Check it's our patched mock

    @pytest.mark.asyncio
    async def test_start_scheduler(self, scheduler_service):
        scheduler_service.scheduler.running = False  # Ensure it starts as not running

        # Mock load_all_agent_schedules to prevent it from running its complex logic here
        scheduler_service.load_all_agent_schedules = AsyncMock()

        await scheduler_service.start()

        scheduler_service.scheduler.start.assert_called_once()
        scheduler_service.load_all_agent_schedules.assert_called_once()
        # After start, the mock_scheduler's running attribute should be True if start() sets it
        # However, the MagicMock itself doesn't change its attributes based on method calls
        # unless we explicitly configure it. For now, just check start() was called.

    @pytest.mark.asyncio
    async def test_start_scheduler_already_running(self, scheduler_service):
        scheduler_service.scheduler.running = True  # Start as already running
        scheduler_service.load_all_agent_schedules = AsyncMock()

        await scheduler_service.start()

        scheduler_service.scheduler.start.assert_not_called()
        scheduler_service.load_all_agent_schedules.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_scheduler(self, scheduler_service):
        scheduler_service.scheduler.running = True  # Assume it's running

        await scheduler_service.shutdown()

        scheduler_service.scheduler.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_scheduler_not_running(self, scheduler_service):
        scheduler_service.scheduler.running = False  # Assume it's not running

        await scheduler_service.shutdown()

        scheduler_service.scheduler.shutdown.assert_not_called()

    def test_remove_job_exists(self, scheduler_service):
        schedule_id = uuid4()
        job_id_str = str(schedule_id)
        scheduler_service.scheduler.get_job.return_value = MagicMock()  # Job exists

        scheduler_service.remove_job(schedule_id)

        scheduler_service.scheduler.get_job.assert_called_once_with(job_id_str)
        scheduler_service.scheduler.remove_job.assert_called_once_with(job_id_str)

    def test_remove_job_not_exists(self, scheduler_service):
        schedule_id = uuid4()
        job_id_str = str(schedule_id)
        scheduler_service.scheduler.get_job.return_value = None  # Job does not exist

        scheduler_service.remove_job(schedule_id)

        scheduler_service.scheduler.get_job.assert_called_once_with(job_id_str)
        scheduler_service.scheduler.remove_job.assert_not_called()

    def test_remove_job_remove_fails(self, scheduler_service, caplog):
        schedule_id = uuid4()
        job_id_str = str(schedule_id)
        scheduler_service.scheduler.get_job.return_value = MagicMock()
        scheduler_service.scheduler.remove_job.side_effect = Exception("Removal failed")

        with caplog.at_level(logging.ERROR):
            scheduler_service.remove_job(schedule_id)

        assert f"Failed to remove job {job_id_str} from scheduler: Removal failed" in caplog.text

    # --- Tests for add_or_update_job ---

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_scheduler_not_running(
        self, mock_datetime, scheduler_service, caplog
    ):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scheduler_service.scheduler.running = False
        agent_id = uuid4()
        schedule_id = uuid4()
        schedule = create_schedule(agent_id, schedule_id)

        with caplog.at_level(logging.ERROR):
            scheduler_service.add_or_update_job(schedule)

        assert "Scheduler is not running. Cannot add/update job." in caplog.text
        scheduler_service.scheduler.add_job.assert_not_called()

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_inactive_schedule(self, mock_datetime, scheduler_service, caplog):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scheduler_service.scheduler.running = True
        agent_id = uuid4()
        schedule_id = uuid4()
        schedule = create_schedule(agent_id, schedule_id, is_active=False)

        with caplog.at_level(logging.DEBUG):  # Check for debug log
            scheduler_service.add_or_update_job(schedule)

        assert f"Schedule {str(schedule_id)} is not active. Not adding to scheduler." in caplog.text
        scheduler_service.scheduler.add_job.assert_not_called()
        # Check if existing job was removed if it existed
        scheduler_service.scheduler.get_job.assert_called_once_with(str(schedule_id))
        # If get_job returned something, remove_job should be called
        if scheduler_service.scheduler.get_job.return_value:
            scheduler_service.scheduler.remove_job.assert_called_once_with(str(schedule_id))

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_past_end_time(self, mock_datetime, scheduler_service, caplog):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        scheduler_service.scheduler.running = True
        agent_id = uuid4()
        schedule_id = uuid4()
        schedule = create_schedule(agent_id, schedule_id, end_time_utc=now - timedelta(hours=1))

        with caplog.at_level(logging.DEBUG):
            scheduler_service.add_or_update_job(schedule)

        expected_log = (
            f"Schedule {str(schedule_id)} end time {schedule.end_time_utc} "
            "has already passed. Not adding to scheduler."
        )
        assert expected_log in caplog.text
        scheduler_service.scheduler.add_job.assert_not_called()
        scheduler_service.scheduler.get_job.assert_called_once_with(str(schedule_id))

    @patch("local_operator.scheduler_service.datetime")
    @patch("local_operator.scheduler_service.DateTrigger", spec=DateTrigger)
    def test_add_or_update_job_one_time_with_start_time(
        self, MockDateTrigger, mock_datetime, scheduler_service
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        start_time = now + timedelta(hours=1)
        mock_datetime.now.return_value = now
        scheduler_service.scheduler.running = True
        scheduler_service.scheduler.get_job.return_value = None  # No existing job

        agent_id = uuid4()
        schedule_id = uuid4()
        schedule = create_schedule(
            agent_id,
            schedule_id,
            one_time=True,
            start_time_utc=start_time,
            interval=0,  # interval/unit not used if start_time_utc is present for one-time
        )

        scheduler_service.add_or_update_job(schedule)

        MockDateTrigger.assert_called_once_with(run_date=start_time, timezone="UTC")
        scheduler_service.scheduler.add_job.assert_called_once()
        args, kwargs = scheduler_service.scheduler.add_job.call_args
        assert kwargs["trigger"] == MockDateTrigger.return_value
        assert kwargs["id"] == str(schedule_id)
        assert kwargs["args"] == [str(agent_id), str(schedule_id), schedule.prompt]
        assert kwargs["misfire_grace_time"] == 60  # Fixed for DateTrigger

    @patch("local_operator.scheduler_service.datetime")
    @patch(
        "local_operator.scheduler_service.DateTrigger", spec=DateTrigger
    )  # Changed to DateTrigger
    def test_add_or_update_job_one_time_with_interval_minutes(
        self, MockDateTrigger, mock_datetime, scheduler_service  # Changed to MockDateTrigger
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        start_time = now + timedelta(minutes=5)  # Start in 5 mins
        mock_datetime.now.return_value = now
        scheduler_service.scheduler.running = True
        scheduler_service.scheduler.get_job.return_value = None

        agent_id = uuid4()
        schedule_id = uuid4()
        # schedule_interval is not directly used by DateTrigger if start_time_utc is present
        schedule_interval = 10
        schedule = create_schedule(
            agent_id,
            schedule_id,
            one_time=True,
            start_time_utc=start_time,  # This will be used as the run_date
            interval=schedule_interval,
            unit=ScheduleUnit.MINUTES,
        )

        scheduler_service.add_or_update_job(schedule)

        # For one_time=True with start_time_utc, DateTrigger is used with run_date=start_time_utc
        MockDateTrigger.assert_called_once_with(run_date=start_time, timezone="UTC")
        scheduler_service.scheduler.add_job.assert_called_once()
        args, kwargs = scheduler_service.scheduler.add_job.call_args
        assert kwargs["trigger"] == MockDateTrigger.return_value
        assert kwargs["id"] == str(schedule_id)
        assert kwargs["args"] == [str(agent_id), str(schedule_id), schedule.prompt]
        assert kwargs["misfire_grace_time"] == 60  # Default for DateTrigger in this path

    @patch("local_operator.scheduler_service.datetime")
    @patch("local_operator.scheduler_service.CronTrigger", spec=CronTrigger)
    def test_add_or_update_job_recurring_days(
        self, MockCronTrigger, mock_datetime, scheduler_service
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        start_time = datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc)  # Specific start time
        end_time = now + timedelta(days=30)
        mock_datetime.now.return_value = now
        scheduler_service.scheduler.running = True
        scheduler_service.scheduler.get_job.return_value = None

        agent_id = uuid4()
        schedule_id = uuid4()
        schedule_interval = 2  # Every 2 days
        schedule = create_schedule(
            agent_id,
            schedule_id,
            one_time=False,  # Recurring
            start_time_utc=start_time,
            end_time_utc=end_time,
            interval=schedule_interval,
            unit=ScheduleUnit.DAYS,
        )

        scheduler_service.add_or_update_job(schedule)

        expected_cron_params = {
            "year": "*",
            "month": "*",
            "day": "*/2",
            "day_of_week": "*",
            "hour": "9",
            "minute": "30",  # From start_time_utc
        }
        MockCronTrigger.assert_called_once_with(
            timezone="UTC", start_date=start_time, end_date=end_time, **expected_cron_params
        )
        scheduler_service.scheduler.add_job.assert_called_once()
        args, kwargs = scheduler_service.scheduler.add_job.call_args
        assert kwargs["trigger"] == MockCronTrigger.return_value
        assert (
            kwargs["misfire_grace_time"] == schedule_interval * 24 * 60 * 60 / 2
        )  # 2 * 12 * 3600 = 86400

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_one_time_missing_start_and_interval(
        self, mock_datetime, scheduler_service, caplog
    ):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scheduler_service.scheduler.running = True
        agent_id = uuid4()
        schedule_id = uuid4()
        schedule = create_schedule(
            agent_id,
            schedule_id,
            one_time=True,
            start_time_utc=None,  # Missing
            interval=0,  # Effectively missing
        )
        with caplog.at_level(logging.ERROR):
            scheduler_service.add_or_update_job(schedule)

        assert (
            f"One-time schedule {str(schedule_id)} for agent {str(agent_id)} requires either "
            "interval/unit or start_time_utc. Skipping job creation."
        ) in caplog.text
        scheduler_service.scheduler.add_job.assert_not_called()

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_unsupported_unit_one_time(
        self, mock_datetime, scheduler_service, caplog
    ):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scheduler_service.scheduler.running = True
        agent_id = uuid4()
        schedule_id = uuid4()
        with pytest.raises(ValidationError) as excinfo:
            create_schedule(
                agent_id,
                schedule_id,
                one_time=True,
                interval=1,
                unit="INVALID_UNIT",  # type: ignore
            )
        assert "unit" in str(excinfo.value).lower()
        assert "invalid_unit" in str(excinfo.value).lower()
        # scheduler_service.add_or_update_job would not be called if create_schedule fails
        scheduler_service.scheduler.add_job.assert_not_called()

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_unsupported_unit_recurring(
        self, mock_datetime, scheduler_service, caplog
    ):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scheduler_service.scheduler.running = True
        agent_id = uuid4()
        schedule_id = uuid4()
        with pytest.raises(ValidationError) as excinfo:
            create_schedule(
                agent_id,
                schedule_id,
                one_time=False,  # Recurring
                interval=1,
                unit="INVALID_UNIT",  # type: ignore
            )
        assert "unit" in str(excinfo.value).lower()
        assert "invalid_unit" in str(excinfo.value).lower()
        # scheduler_service.add_or_update_job would not be called if create_schedule fails
        scheduler_service.scheduler.add_job.assert_not_called()

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_updates_existing_job(self, mock_datetime, scheduler_service):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        start_time = now + timedelta(hours=1)
        mock_datetime.now.return_value = now
        scheduler_service.scheduler.running = True

        agent_id = uuid4()
        schedule_id = uuid4()
        job_id_str = str(schedule_id)

        # Simulate existing job
        scheduler_service.scheduler.get_job.return_value = MagicMock()

        schedule = create_schedule(agent_id, schedule_id, one_time=True, start_time_utc=start_time)

        scheduler_service.add_or_update_job(schedule)

        scheduler_service.scheduler.get_job.assert_called_once_with(job_id_str)
        scheduler_service.scheduler.remove_job.assert_called_once_with(job_id_str)
        scheduler_service.scheduler.add_job.assert_called_once()  # New job added

    @patch("local_operator.scheduler_service.datetime")
    def test_add_or_update_job_add_fails(self, mock_datetime, scheduler_service, caplog):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        start_time = now + timedelta(hours=1)
        mock_datetime.now.return_value = now
        scheduler_service.scheduler.running = True
        scheduler_service.scheduler.get_job.return_value = None
        scheduler_service.scheduler.add_job.side_effect = Exception("Add failed")

        agent_id = uuid4()
        schedule_id = uuid4()
        schedule = create_schedule(agent_id, schedule_id, one_time=True, start_time_utc=start_time)
        with caplog.at_level(logging.ERROR):
            scheduler_service.add_or_update_job(schedule)

        assert (
            f"Failed to add/update job {str(schedule_id)} to scheduler: Add failed" in caplog.text
        )

    # --- Tests for _trigger_agent_task ---
    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.create_and_start_job_process_with_queue")
    @patch("local_operator.scheduler_service.datetime")
    async def test_trigger_agent_task_success(
        self,
        mock_datetime,
        mock_create_process,
        scheduler_service,
        mock_agent_registry,
        mock_job_manager,
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now

        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str = str(agent_id)
        schedule_id_str = str(schedule_id)
        prompt = "Test prompt"

        mock_schedule = create_schedule(
            agent_id,
            schedule_id,
            prompt=prompt,
            is_active=True,
            start_time_utc=now - timedelta(hours=1),  # Started
            end_time_utc=now + timedelta(hours=1),  # Not ended
        )
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[mock_schedule],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TestAgent",
            model="test_model",
            hosting="test_hosting",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.get_agent.return_value = mock_agent_data

        mock_job_entry = Job(
            id=schedule_id_str,
            prompt=prompt,
            model="test_model",
            hosting="test_hosting",
            agent_id=agent_id_str,
            status=JobStatus.PENDING,
            created_at=now.timestamp(),  # Corrected to float timestamp
        )
        mock_job_manager.create_job.return_value = mock_job_entry

        await scheduler_service._trigger_agent_task(agent_id_str, schedule_id_str, prompt)

        mock_agent_registry.load_agent_state.assert_called_once_with(agent_id_str)
        mock_agent_registry.get_agent.assert_called_once_with(agent_id_str)
        mock_job_manager.create_job.assert_called_once_with(
            prompt=prompt,
            model="test_model",
            hosting="test_hosting",
            agent_id=agent_id_str,
            job_id=schedule_id_str,
        )

        expected_process_args = (
            schedule_id_str,
            agent_id_str,
            schedule_id_str,
            prompt,
            mock_agent_registry.config_dir,  # Path object expected by _execute_scheduled_task_logic
            scheduler_service.env_config,
            scheduler_service.operator_type.name,
            scheduler_service.verbosity_level.name,
            "test_hosting",
            "test_model",
        )

        mock_create_process.assert_called_once()
        call_args, call_kwargs = mock_create_process.call_args
        assert call_kwargs["job_id"] == schedule_id_str
        assert call_kwargs["process_func"] == _execute_scheduled_task_logic
        assert call_kwargs["args"] == expected_process_args
        assert call_kwargs["job_manager"] == mock_job_manager
        assert call_kwargs["websocket_manager"] == scheduler_service.websocket_manager
        assert call_kwargs["scheduler_service"] == scheduler_service

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_trigger_agent_task_schedule_not_found(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id_str = str(uuid4())
        schedule_id_str = str(uuid4())

        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        scheduler_service.remove_job = MagicMock()  # Mock remove_job

        with caplog.at_level(logging.ERROR):
            await scheduler_service._trigger_agent_task(agent_id_str, schedule_id_str, "prompt")

        assert (
            f"Schedule {schedule_id_str} not found for agent "
            f"{agent_id_str}. Removing APScheduler job." in caplog.text
        )
        scheduler_service.remove_job.assert_called_once_with(UUID(schedule_id_str))

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_trigger_agent_task_schedule_inactive(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str = str(agent_id)
        schedule_id_str = str(schedule_id)

        mock_schedule = create_schedule(agent_id, schedule_id, is_active=False)
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[mock_schedule],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        scheduler_service.remove_job = MagicMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service._trigger_agent_task(agent_id_str, schedule_id_str, "prompt")

        assert (
            f"Schedule {schedule_id_str} is no longer active. Removing APScheduler job."
            in caplog.text
        )
        scheduler_service.remove_job.assert_called_once_with(schedule_id)

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_trigger_agent_task_past_end_time(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str = str(agent_id)
        schedule_id_str = str(schedule_id)

        mock_schedule = create_schedule(
            agent_id,
            schedule_id,
            is_active=True,
            end_time_utc=now - timedelta(minutes=1),  # Ended 1 min ago
        )
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[mock_schedule],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        scheduler_service.remove_job = MagicMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service._trigger_agent_task(agent_id_str, schedule_id_str, "prompt")

        assert (
            f"Schedule {schedule_id_str} passed end time ({mock_schedule.end_time_utc}). "
            "Removing APScheduler job, marking inactive in agent state."
        ) in caplog.text
        scheduler_service.remove_job.assert_called_once_with(schedule_id)
        # Check agent state was saved with schedule marked inactive
        saved_state = mock_agent_registry.save_agent_state.call_args[0][1]
        assert not saved_state.schedules[0].is_active

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.create_and_start_job_process_with_queue")
    @patch("local_operator.scheduler_service.datetime")
    async def test_trigger_agent_task_job_creation_fails(
        self,
        mock_datetime,
        mock_create_process,
        scheduler_service,
        mock_agent_registry,
        mock_job_manager,
        caplog,
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str = str(agent_id)
        schedule_id_str = str(schedule_id)
        prompt = "Test prompt"

        mock_schedule = create_schedule(agent_id, schedule_id, prompt=prompt, is_active=True)
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[mock_schedule],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TestAgent",
            model="test_model",
            hosting="test_hosting",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.get_agent.return_value = mock_agent_data

        mock_job_manager.create_job.side_effect = Exception("Job creation failed")

        with caplog.at_level(logging.ERROR):
            await scheduler_service._trigger_agent_task(agent_id_str, schedule_id_str, prompt)

        assert (
            f"Error creating or starting job for scheduled task agent {agent_id_str}, "
            f"schedule {schedule_id_str}: Job creation failed"
        ) in caplog.text
        mock_create_process.assert_not_called()

    # --- Tests for load_all_agent_schedules ---
    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_no_agents(
        self, mock_datetime, scheduler_service, mock_agent_registry
    ):
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_agent_registry.list_agents.return_value = []
        scheduler_service.add_or_update_job = MagicMock()

        await scheduler_service.load_all_agent_schedules()
        mock_agent_registry.list_agents.assert_called_once()
        mock_agent_registry.load_agent_state.assert_not_called()
        scheduler_service.add_or_update_job.assert_not_called()

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_agent_no_schedules(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        agent_id_str = str(agent_id)
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TestAgent",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]

        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        scheduler_service.add_or_update_job = MagicMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        assert f"No schedules found for agent {agent_id_str}" in caplog.text
        scheduler_service.add_or_update_job.assert_not_called()
        mock_agent_registry.save_agent_state.assert_not_called()

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_ended_schedule(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        ended_schedule = create_schedule(
            agent_id, schedule_id, is_active=True, end_time_utc=now - timedelta(minutes=1)  # Ended
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[ended_schedule],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        scheduler_service.remove_job = MagicMock()
        scheduler_service.add_or_update_job = MagicMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        log_msg = (
            f"Schedule {schedule_id_str} for agent {agent_id_str} has passed its "
            f"end time ({ended_schedule.end_time_utc}). Ensuring inactive and removed."
        )
        assert log_msg in caplog.text
        scheduler_service.remove_job.assert_called_once_with(schedule_id)
        scheduler_service.add_or_update_job.assert_not_called()

        # Check agent state was saved with schedule marked inactive and then removed
        mock_agent_registry.save_agent_state.assert_called_once()
        saved_state = mock_agent_registry.save_agent_state.call_args[0][1]
        assert len(saved_state.schedules) == 0  # Schedule should be removed as it's inactive

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_explicitly_inactive(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        inactive_schedule = create_schedule(agent_id, schedule_id, is_active=False)
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[inactive_schedule],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        scheduler_service.remove_job = MagicMock()
        scheduler_service.add_or_update_job = MagicMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        log_msg = (
            f"Schedule {schedule_id_str} for agent {agent_id_str} is marked inactive. "
            "Ensuring removal from scheduler."
        )
        assert log_msg in caplog.text
        scheduler_service.remove_job.assert_called_once_with(schedule_id)
        scheduler_service.add_or_update_job.assert_not_called()
        mock_agent_registry.save_agent_state.assert_called_once()
        saved_state = mock_agent_registry.save_agent_state.call_args[0][1]
        assert len(saved_state.schedules) == 0

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.asyncio.create_task")  # Mock create_task
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_one_time_past_due(
        self, mock_datetime, mock_create_async_task, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        past_due_one_time = create_schedule(
            agent_id,
            schedule_id,
            one_time=True,
            is_active=True,
            start_time_utc=now - timedelta(minutes=10),  # Past due
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[past_due_one_time],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state
        scheduler_service.add_or_update_job = MagicMock()
        # _trigger_agent_task is an async method of the instance
        scheduler_service._trigger_agent_task = AsyncMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        log_msg = (
            f"Past-due active one-time schedule {schedule_id_str} for agent {agent_id_str} "
            f"(start: {past_due_one_time.start_time_utc}). Triggering now (non-blocking)."
        )
        assert log_msg in caplog.text

        # Check that asyncio.create_task was called with _trigger_agent_task
        mock_create_async_task.assert_called_once()
        # The argument to create_task should be the coroutine object from _trigger_agent_task
        # We can check if _trigger_agent_task itself was called with correct args
        scheduler_service._trigger_agent_task.assert_called_once_with(
            agent_id_str=agent_id_str,
            schedule_id_str=schedule_id_str,
            prompt=past_due_one_time.prompt,
        )
        scheduler_service.add_or_update_job.assert_not_called()  # Not added if triggered
        mock_agent_registry.save_agent_state.assert_not_called()  # State saved by task logic

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_one_time_future(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        future_one_time = create_schedule(
            agent_id,
            schedule_id,
            one_time=True,
            is_active=True,
            start_time_utc=now + timedelta(minutes=10),  # Future
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[future_one_time],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        scheduler_service.add_or_update_job = MagicMock()
        scheduler_service._trigger_agent_task = AsyncMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        log_msg = (
            f"Future active one-time schedule {schedule_id_str} for agent {agent_id_str}. "
            "Adding to scheduler."
        )
        assert log_msg in caplog.text
        scheduler_service._trigger_agent_task.assert_not_called()
        scheduler_service.add_or_update_job.assert_called_once_with(future_one_time)
        mock_agent_registry.save_agent_state.assert_not_called()

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_one_time_no_start_time(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        no_start_one_time = create_schedule(
            agent_id,
            schedule_id,
            one_time=True,
            is_active=True,
            start_time_utc=None,  # Missing start time
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[no_start_one_time],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        scheduler_service.remove_job = MagicMock()
        scheduler_service.add_or_update_job = MagicMock()

        with caplog.at_level(logging.ERROR):  # Error log for this case
            await scheduler_service.load_all_agent_schedules()

        log_msg = (
            f"Active one-time schedule {schedule_id_str} for agent {agent_id_str} "
            "lacks start_time_utc. Marking inactive."
        )
        assert log_msg in caplog.text
        scheduler_service.remove_job.assert_called_once_with(schedule_id)
        scheduler_service.add_or_update_job.assert_not_called()
        mock_agent_registry.save_agent_state.assert_called_once()
        saved_state = mock_agent_registry.save_agent_state.call_args[0][1]
        assert len(saved_state.schedules) == 0  # Schedule removed as inactive

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.asyncio.create_task")
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_recurring_missed_run_with_last_run(
        self, mock_datetime, mock_create_async_task, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)  # Current time
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        # Schedule runs every 10 minutes, last run was 12 minutes ago
        # Grace period is 10 * 60 / 2 = 300s = 5 minutes
        # Next expected run: 11:53 + 10m = 12:03.
        # Current time 12:05 is within 12:03 + 5m grace (12:08)
        missed_recurring = create_schedule(
            agent_id,
            schedule_id,
            one_time=False,
            is_active=True,
            interval=10,
            unit=ScheduleUnit.MINUTES,
            last_run_at=datetime(2024, 1, 1, 11, 53, 0, tzinfo=timezone.utc),
            start_time_utc=datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),  # Started long ago
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[missed_recurring],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        scheduler_service.add_or_update_job = MagicMock()
        scheduler_service._trigger_agent_task = AsyncMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

            if missed_recurring.last_run_at is not None:
                next_expected_run_time = missed_recurring.last_run_at + timedelta(
                    minutes=missed_recurring.interval
                )
            else:
                next_expected_run_time = missed_recurring.start_time_utc

            grace_delta = timedelta(seconds=max(60, int(missed_recurring.interval * 60 / 2)))
            log_msg = (
                f"Missed recurring schedule {schedule_id_str} for agent {agent_id_str} "
                f"(last run: {missed_recurring.last_run_at}, "
                f"next expected: {next_expected_run_time}, "
                f"grace: {grace_delta}). Triggering now (non-blocking)."
            )
        assert log_msg in caplog.text
        mock_create_async_task.assert_called_once()
        scheduler_service._trigger_agent_task.assert_called_once_with(
            agent_id_str=agent_id_str,
            schedule_id_str=schedule_id_str,
            prompt=missed_recurring.prompt,
        )
        # Job should still be added for future runs
        scheduler_service.add_or_update_job.assert_called_once_with(missed_recurring)

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.asyncio.create_task")
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_recurring_missed_first_run(
        self, mock_datetime, mock_create_async_task, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 2, 0, tzinfo=timezone.utc)  # Current time
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        # Schedule runs every 10 minutes, started at 12:00. No last_run_at.
        # Grace period is 5 minutes.
        # First expected run: 12:00.
        # Current time 12:02 is within 12:00 + 5m grace (12:05)
        missed_first_recurring = create_schedule(
            agent_id,
            schedule_id,
            one_time=False,
            is_active=True,
            interval=10,
            unit=ScheduleUnit.MINUTES,
            last_run_at=None,  # No last run
            start_time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[missed_first_recurring],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        scheduler_service.add_or_update_job = MagicMock()
        scheduler_service._trigger_agent_task = AsyncMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        first_expected_run_time = missed_first_recurring.start_time_utc
        grace_delta = timedelta(seconds=max(60, int(missed_first_recurring.interval * 60 / 2)))
        log_msg = (
            f"Missed first recurring schedule {schedule_id_str} for agent {agent_id_str} "
            f"(start: {first_expected_run_time}, "
            f"grace: {grace_delta}). Triggering now (non-blocking)."
        )
        assert log_msg in caplog.text
        mock_create_async_task.assert_called_once()
        scheduler_service._trigger_agent_task.assert_called_once_with(
            agent_id_str=agent_id_str,
            schedule_id_str=schedule_id_str,
            prompt=missed_first_recurring.prompt,
        )
        scheduler_service.add_or_update_job.assert_called_once_with(missed_first_recurring)

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_recurring_normal_add(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 11, 50, 0, tzinfo=timezone.utc)  # Current time
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        schedule_id = uuid4()
        agent_id_str, schedule_id_str = str(agent_id), str(schedule_id)

        # Schedule runs every 10 minutes, started at 12:00. Not missed.
        normal_recurring = create_schedule(
            agent_id,
            schedule_id,
            one_time=False,
            is_active=True,
            interval=10,
            unit=ScheduleUnit.MINUTES,
            last_run_at=None,
            start_time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),  # Future start
        )
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TA",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]
        mock_agent_state = AgentState(
            version="1.0",
            conversation=[],
            current_plan="",
            instruction_details="",
            agent_system_prompt="",
            schedules=[normal_recurring],
        )
        mock_agent_registry.load_agent_state.return_value = mock_agent_state

        scheduler_service.add_or_update_job = MagicMock()
        scheduler_service._trigger_agent_task = AsyncMock()

        with caplog.at_level(logging.DEBUG):
            await scheduler_service.load_all_agent_schedules()

        log_msg = (
            f"Active recurring schedule {schedule_id_str} for agent {agent_id_str}. "
            "Adding/updating in scheduler."
        )
        assert log_msg in caplog.text
        scheduler_service._trigger_agent_task.assert_not_called()
        scheduler_service.add_or_update_job.assert_called_once_with(normal_recurring)

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_error_loading_agent_state(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        agent_id = uuid4()
        agent_id_str = str(agent_id)
        mock_agent_data = AgentData(
            id=agent_id_str,
            name="TestAgent",
            model="m",
            hosting="h",
            created_date=datetime.now(timezone.utc),  # Required
            version="1.0",  # Required
            security_prompt="",
            description="",
            tags=[],
            categories=[],
            last_message="",
            last_message_datetime=datetime.now(timezone.utc),
            current_working_directory=".",
            temperature=0.5,
            top_p=0.9,
            max_tokens=1000,
            top_k=50,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            seed=None,
        )
        mock_agent_registry.list_agents.return_value = [mock_agent_data]

        mock_agent_registry.load_agent_state.side_effect = Exception("Load state failed")

        with caplog.at_level(logging.ERROR):
            await scheduler_service.load_all_agent_schedules()

        assert (
            f"Error loading or processing schedules for agent {agent_id_str}: Load state failed"
            in caplog.text
        )

    @pytest.mark.asyncio
    @patch("local_operator.scheduler_service.datetime")
    async def test_load_all_schedules_general_error(
        self, mock_datetime, scheduler_service, mock_agent_registry, caplog
    ):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = now
        mock_agent_registry.list_agents.side_effect = Exception("List agents failed")

        with caplog.at_level(logging.ERROR):
            await scheduler_service.load_all_agent_schedules()

        assert (
            "An unexpected error occurred while loading all agent schedules: List agents failed"
            in caplog.text
        )
