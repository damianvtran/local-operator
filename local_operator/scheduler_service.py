import logging
from datetime import datetime, timezone
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from local_operator.agents import AgentRegistry
from local_operator.bootstrap import initialize_operator  # Added
from local_operator.config import ConfigManager  # Added
from local_operator.credentials import CredentialManager  # Added
from local_operator.env import EnvConfig  # Added
from local_operator.operator import OperatorType  # Added
from local_operator.types import Schedule, ScheduleUnit

logger = logging.getLogger(__name__)


class SchedulerService:
    """
    Service for managing and executing scheduled tasks for agents.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config_manager: ConfigManager,
        credential_manager: CredentialManager,
        env_config: EnvConfig,
    ):
        self.agent_registry = agent_registry
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self.env_config = env_config
        self.scheduler = AsyncIOScheduler(timezone="UTC")

    async def _trigger_agent_task(
        self, agent_id_str: str, schedule_id_str: str, prompt: str
    ) -> None:
        """
        The actual function called by APScheduler to trigger an agent's task.
        Checks end_time_utc before execution.
        """
        logger.info(
            f"Attempting to trigger task for agent {agent_id_str}, schedule {schedule_id_str}"
        )
        try:
            # agent_id = UUID(agent_id_str) # agent_id_str is used directly, agent_id was unused
            schedule_id = UUID(schedule_id_str)

            # Load schedule details to check end_time_utc
            agent_state = self.agent_registry.load_agent_state(agent_id_str)
            current_schedule: Schedule | None = None
            for sched in agent_state.schedules:
                if sched.id == schedule_id:
                    current_schedule = sched
                    break

            if not current_schedule:
                logger.error(
                    f"Schedule {schedule_id_str} not found for agent {agent_id_str}. Removing job."
                )
                self.remove_job(schedule_id)
                return

            if not current_schedule.is_active:
                logger.info(f"Schedule {schedule_id_str} is no longer active. Removing job.")
                self.remove_job(schedule_id)
                # Also update agent state to reflect this if necessary, or rely on external update
                return

            now_utc = datetime.now(timezone.utc)
            if current_schedule.end_time_utc and now_utc >= current_schedule.end_time_utc:
                logger.info(
                    f"Schedule {schedule_id_str} passed end time "
                    f"({current_schedule.end_time_utc}). Removing job, marking inactive."
                )
                current_schedule.is_active = False
                self.agent_registry.save_agent_state(
                    agent_id_str, agent_state
                )  # agent_state is already loaded
                self.remove_job(schedule_id)
                return

            logger.info(f"Triggering task for agent {agent_id_str}, schedule {schedule_id_str}")

            # Create an operator instance for this specific task execution
            try:
                target_agent_data = self.agent_registry.get_agent(agent_id_str)
                # The scheduler_service itself is passed to initialize_operator,
                # which then passes it to build_tool_registry.
                # This ensures tools have access to the *single* scheduler instance.
                task_operator = initialize_operator(  # Returns a single Operator
                    operator_type=OperatorType.SERVER,  # Using SERVER as a generic background type
                    config_manager=self.config_manager,
                    credential_manager=self.credential_manager,
                    agent_registry=self.agent_registry,
                    env_config=self.env_config,
                    current_agent=target_agent_data,
                    scheduler_service=self,
                )
                await task_operator.handle_user_input(prompt)
            except Exception as op_error:
                logger.error(
                    f"Failed to initialize operator or handle task for agent {agent_id_str}, "
                    f"schedule {schedule_id_str}: {op_error}",
                    exc_info=True,
                )
                # Optionally, decide if the job should be removed or retried based on op_error
                return  # Exit if operator init or task handling fails

            # Update last_run_at for the schedule
            # Re-load state in case it was modified by the agent task itself
            agent_state = self.agent_registry.load_agent_state(agent_id_str)
            schedule_updated = False
            for sched in agent_state.schedules:
                if sched.id == schedule_id:
                    sched.last_run_at = now_utc
                    schedule_updated = True
                    break
            if schedule_updated:
                self.agent_registry.save_agent_state(agent_id_str, agent_state)
            logger.info(
                f"Successfully triggered and updated last_run_at for schedule {schedule_id_str}"
            )

        except Exception as e:
            logger.error(
                f"Error triggering scheduled task for agent {agent_id_str}, "
                f"schedule {schedule_id_str}: {str(e)}"
            )

    def add_or_update_job(self, schedule: Schedule) -> None:
        """
        Adds a new job or updates an existing one in APScheduler based on the Schedule object.
        Considers start_time_utc and end_time_utc.
        """
        job_id = str(schedule.id)
        agent_id_str = str(schedule.agent_id)
        now_utc = datetime.now(timezone.utc)

        # Remove existing job if it exists, to ensure it's updated
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed existing job {job_id} to update schedule.")

        if not schedule.is_active:
            logger.info(f"Schedule {job_id} is not active. Not adding to scheduler.")
            return

        if schedule.end_time_utc and now_utc >= schedule.end_time_utc:
            logger.info(
                f"Schedule {job_id} end time {schedule.end_time_utc} has already passed. "
                "Not adding to scheduler."
            )
            return

        trigger_args = [agent_id_str, job_id, schedule.prompt]
        trigger_kwargs = {
            "timezone": "UTC",
            "start_date": schedule.start_time_utc,  # noqa: E501 APScheduler handles None as "immediate" for start_date
            "end_date": schedule.end_time_utc,
        }

        if schedule.unit == ScheduleUnit.DAYS and schedule.start_time_utc:
            # For daily cron, use start_time_utc to set the hour and minute
            trigger = CronTrigger(
                hour=schedule.start_time_utc.hour,
                minute=schedule.start_time_utc.minute,
                day=f"*/{schedule.interval}" if schedule.interval > 0 else "*",
                **trigger_kwargs,
            )
            start_str = schedule.start_time_utc.strftime("%H:%M")
            log_msg = (
                f"Adding/updating CRON job {job_id} for agent {agent_id_str}. "
                f"Start: {start_str} UTC, End: {schedule.end_time_utc}, "
                f"Every: {schedule.interval} {schedule.unit}. Next run: APScheduler."
            )
            logger.info(log_msg)
        else:
            # For interval-based schedules (minutes, hours, or days without specific time)
            interval_unit_value = schedule.unit.value
            # APScheduler's IntervalTrigger uses 'days', 'hours', 'minutes', 'seconds'
            # Our ScheduleUnit enum matches these directly for days, hours, minutes.
            interval_trigger_specific_kwargs = {interval_unit_value: schedule.interval}

            # If start_time_utc is in the past for an interval trigger, APScheduler might
            # fire immediately if the first scheduled run based on start_time_utc and
            # interval has passed. This is generally desired behavior.
            trigger = IntervalTrigger(**interval_trigger_specific_kwargs, **trigger_kwargs)
            logger.info(
                f"Adding/updating INTERVAL job {job_id} for agent {agent_id_str}. "
                f"Start: {schedule.start_time_utc}, End: {schedule.end_time_utc}, "
                f"Every: {schedule.interval} {schedule.unit}."
            )

        try:
            self.scheduler.add_job(
                self._trigger_agent_task,
                trigger=trigger,
                args=trigger_args,
                id=job_id,
                name=f"Agent {agent_id_str} - Schedule {job_id}",
                replace_existing=True,
                misfire_grace_time=600,  # 10 minutes
            )
            logger.info(f"Successfully added/updated job {job_id} to scheduler.")
        except Exception as e:
            logger.error(f"Failed to add/update job {job_id} to scheduler: {str(e)}")

    def remove_job(self, schedule_id: UUID) -> None:
        """
        Removes a job from APScheduler.
        """
        job_id = str(schedule_id)
        if self.scheduler.get_job(job_id):
            try:
                self.scheduler.remove_job(job_id)
                logger.info(f"Successfully removed job {job_id} from scheduler.")
            except Exception as e:
                logger.error(f"Failed to remove job {job_id} from scheduler: {str(e)}")
        else:
            logger.info(f"Job {job_id} not found in scheduler, no action taken.")

    async def start(self) -> None:
        """
        Starts the APScheduler and loads all existing active schedules.
        """
        logger.info("Starting SchedulerService...")
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("APScheduler started.")
        else:
            logger.info("APScheduler already running.")

        await self.load_all_agent_schedules()

    async def load_all_agent_schedules(self) -> None:
        """
        Loads all active schedules for all agents and adds them to the scheduler.
        """
        logger.info("Loading all agent schedules into APScheduler...")
        try:
            all_agents = self.agent_registry.list_agents()
            for agent_data in all_agents:
                try:
                    agent_state = self.agent_registry.load_agent_state(agent_data.id)
                    if agent_state.schedules:
                        logger.info(
                            f"Found {len(agent_state.schedules)} schedules "
                            f"for agent {agent_data.id}"
                        )
                        for schedule_item in agent_state.schedules:
                            if schedule_item.is_active:
                                self.add_or_update_job(schedule_item)
                            else:
                                # Ensure inactive schedules are not in the scheduler
                                self.remove_job(schedule_item.id)
                    else:
                        logger.info(f"No schedules found for agent {agent_data.id}")
                except Exception as e:
                    logger.error(f"Error loading schedules for agent {agent_data.id}: {str(e)}")
            logger.info("Finished loading agent schedules.")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading all agent schedules: {str(e)}"
            )

    async def shutdown(self) -> None:
        """
        Shuts down the APScheduler.
        """
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("APScheduler shut down.")
