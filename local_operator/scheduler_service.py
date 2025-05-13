import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from local_operator.agents import AgentRegistry
from local_operator.bootstrap import initialize_operator  # Added
from local_operator.config import ConfigManager  # Added
from local_operator.console import VerbosityLevel
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
        operator_type: OperatorType,
        verbosity_level: VerbosityLevel,
    ):
        self.agent_registry = agent_registry
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self.env_config = env_config
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.operator_type = operator_type
        self.verbosity_level = verbosity_level

    async def _trigger_agent_task(
        self, agent_id_str: str, schedule_id_str: str, prompt: str
    ) -> None:
        """
        The actual function called by APScheduler to trigger an agent's task.
        Checks end_time_utc before execution.
        """
        logger.debug(
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
                logger.debug(f"Schedule {schedule_id_str} is no longer active. Removing job.")
                self.remove_job(schedule_id)

                # Remove from agent state
                agent_state = self.agent_registry.load_agent_state(agent_id_str)
                agent_state.schedules = [
                    sched for sched in agent_state.schedules if sched.id != schedule_id
                ]
                self.agent_registry.save_agent_state(agent_id_str, agent_state)

                return

            now_utc = datetime.now(timezone.utc)
            if current_schedule.end_time_utc and now_utc >= current_schedule.end_time_utc:
                logger.debug(
                    f"Schedule {schedule_id_str} passed end time "
                    f"({current_schedule.end_time_utc}). Removing job, marking inactive."
                )
                current_schedule.is_active = False

                # Remove from agent state
                agent_state = self.agent_registry.load_agent_state(agent_id_str)
                agent_state.schedules = [
                    sched for sched in agent_state.schedules if sched.id != schedule_id
                ]
                self.agent_registry.save_agent_state(agent_id_str, agent_state)

                self.remove_job(schedule_id)
                return

            logger.debug(f"Triggering task for agent {agent_id_str}, schedule {schedule_id_str}")

            # Create an operator instance for this specific task execution
            try:
                target_agent_data = self.agent_registry.get_agent(agent_id_str)
                # The scheduler_service itself is passed to initialize_operator,
                # which then passes it to build_tool_registry.
                # This ensures tools have access to the *single* scheduler instance.
                task_operator = initialize_operator(  # Returns a single Operator
                    operator_type=self.operator_type,
                    config_manager=self.config_manager,
                    credential_manager=self.credential_manager,
                    agent_registry=self.agent_registry,
                    env_config=self.env_config,
                    request_hosting=target_agent_data.hosting,
                    request_model=target_agent_data.model,
                    current_agent=target_agent_data,
                    scheduler_service=self,
                    persist_conversation=True,
                    auto_save_conversation=False,
                    verbosity_level=self.verbosity_level,
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
            # Re-load state in case it was modified by the agent task itself or needs deactivation
            agent_state_after_task = self.agent_registry.load_agent_state(agent_id_str)
            schedule_to_update: Schedule | None = None
            for sched_idx, sched_item in enumerate(agent_state_after_task.schedules):
                if sched_item.id == schedule_id:
                    sched_item.last_run_at = now_utc  # now_utc from the beginning of this method
                    schedule_to_update = sched_item

                    # If it's a one-time job, remove it from schedules and from scheduler
                    if sched_item.one_time:
                        logger.debug(
                            f"One-time schedule {schedule_id_str} executed. "
                            "Removing from agent state and scheduler."
                        )
                        # Remove from schedules list
                        self.remove_job(schedule_id)
                        del agent_state_after_task.schedules[sched_idx]
                        self.agent_registry.save_agent_state(agent_id_str, agent_state_after_task)
                        logger.debug(
                            f"One-time schedule {schedule_id_str} removed "
                            "from agent state and scheduler after execution."
                        )
                        return  # Exit after removal to avoid further processing

                    # If end_time_utc is now passed (could be due to one_time logic setting it)
                    # or was already passed, ensure it's inactive.
                    # This check is also present at the start, but good to re-verify.
                    if sched_item.end_time_utc and now_utc >= sched_item.end_time_utc:
                        if sched_item.is_active:  # Only log/change if it wasn't already inactive
                            logger.debug(
                                f"Schedule {schedule_id_str} passed end time "
                                f"({sched_item.end_time_utc}) after task. Marking inactive."
                            )
                            sched_item.is_active = False
                        self.remove_job(schedule_id)  # Ensure removal if end time passed

                    break  # Found the schedule

            if schedule_to_update:  # If any changes were made to the schedule object
                self.agent_registry.save_agent_state(agent_id_str, agent_state_after_task)
                logger.debug(
                    f"Successfully triggered and updated state for schedule {schedule_id_str}"
                )
            else:
                logger.warning(
                    f"Schedule {schedule_id_str} not found in agent state after task "
                    "execution for update."
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

        # Ensure the scheduler is running before adding jobs
        if not self.scheduler.running:
            logger.error("Scheduler is not running. Cannot add/update job.")
            return

        # Remove existing job if it exists, to ensure it's updated
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            logger.debug(f"Removed existing job {job_id} to update schedule.")

        if not schedule.is_active:
            logger.debug(f"Schedule {job_id} is not active. Not adding to scheduler.")
            return

        if schedule.end_time_utc and now_utc >= schedule.end_time_utc:
            logger.debug(
                f"Schedule {job_id} end time {schedule.end_time_utc} has already passed. "
                "Not adding to scheduler."
            )
            return

        trigger_args = [agent_id_str, job_id, schedule.prompt]

        def get_cron_interval_field(interval_value: int) -> str:
            if interval_value <= 0:  # Treat 0 or negative as "every"
                return "*"
            # For interval_value == 1, "*/1" is equivalent to "*", but explicit is fine.
            return f"*/{interval_value}"

        # Base cron parameters, default to "every" for all fields
        cron_expression_params = {
            "year": "*",
            "month": "*",
            "day": "*",
            "day_of_week": "*",
            "hour": "*",
            "minute": "*",
        }

        effective_end_date = schedule.end_time_utc
        log_details = ""
        misfire_grace_time_seconds = 600  # Default 10 minutes

        if schedule.one_time:
            # Prefer interval/unit if provided, otherwise use start_time_utc
            if schedule.interval and schedule.unit:
                # This is a one-time job that behaves like a recurring job until it runs once.
                # We use CronTrigger for this.
                if schedule.unit == ScheduleUnit.MINUTES:
                    cron_expression_params["minute"] = get_cron_interval_field(schedule.interval)
                    misfire_grace_time_seconds = max(60, int(schedule.interval * 60 / 2))
                elif schedule.unit == ScheduleUnit.HOURS:
                    cron_expression_params["minute"] = str(
                        schedule.start_time_utc.minute if schedule.start_time_utc else 0
                    )
                    cron_expression_params["hour"] = get_cron_interval_field(schedule.interval)
                    misfire_grace_time_seconds = max(60, int(schedule.interval * 60 * 60 / 2))
                elif schedule.unit == ScheduleUnit.DAYS:
                    cron_expression_params["minute"] = str(
                        schedule.start_time_utc.minute if schedule.start_time_utc else 0
                    )
                    cron_expression_params["hour"] = str(
                        schedule.start_time_utc.hour if schedule.start_time_utc else 0
                    )
                    cron_expression_params["day"] = get_cron_interval_field(schedule.interval)
                    misfire_grace_time_seconds = max(60, int(schedule.interval * 24 * 60 * 60 / 2))
                else:
                    logger.error(
                        f"Unsupported schedule unit: {schedule.unit} for one-time "
                        f"schedule {job_id} with interval. "
                        "Skipping job creation."
                    )
                    return
                log_details = (
                    f"One-time schedule with interval: every {schedule.interval} "
                    f"{schedule.unit.value}. "
                    f"Cron: (M='{cron_expression_params['minute']}', "
                    f"H='{cron_expression_params['hour']}', "
                    f"DoM='{cron_expression_params['day']}', "
                    f"Mon='{cron_expression_params['month']}', "
                    f"DoW='{cron_expression_params['day_of_week']}'). "
                    f"Grace: {misfire_grace_time_seconds}s."
                )
                effective_end_date = schedule.end_time_utc
                trigger = CronTrigger(
                    timezone="UTC",
                    start_date=schedule.start_time_utc,
                    end_date=effective_end_date,
                    **cron_expression_params,
                )
            elif schedule.start_time_utc:
                # Use DateTrigger for one-time jobs with only start_time_utc
                misfire_grace_time_seconds = 60  # Fixed 60s for specific date triggers
                log_details = (
                    f"One-time at {schedule.start_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
                    f"Grace: {misfire_grace_time_seconds}s."
                )
                trigger = DateTrigger(
                    run_date=schedule.start_time_utc,
                    timezone="UTC",
                )  # misfire_grace_time is an add_job param
            else:
                logger.error(
                    f"One-time schedule {job_id} for agent {agent_id_str} requires either "
                    "interval/unit or start_time_utc. Skipping job creation."
                )
                return
        else:  # Recurring job
            if schedule.unit == ScheduleUnit.MINUTES:
                cron_expression_params["minute"] = get_cron_interval_field(schedule.interval)
                misfire_grace_time_seconds = max(60, int(schedule.interval * 60 / 2))
            elif schedule.unit == ScheduleUnit.HOURS:
                cron_expression_params["minute"] = str(
                    schedule.start_time_utc.minute if schedule.start_time_utc else 0
                )
                cron_expression_params["hour"] = get_cron_interval_field(schedule.interval)
                misfire_grace_time_seconds = max(60, int(schedule.interval * 60 * 60 / 2))
            elif schedule.unit == ScheduleUnit.DAYS:
                cron_expression_params["minute"] = str(
                    schedule.start_time_utc.minute if schedule.start_time_utc else 0
                )
                cron_expression_params["hour"] = str(
                    schedule.start_time_utc.hour if schedule.start_time_utc else 0
                )
                cron_expression_params["day"] = get_cron_interval_field(schedule.interval)
                misfire_grace_time_seconds = max(60, int(schedule.interval * 24 * 60 * 60 / 2))
            else:
                logger.error(
                    f"Unsupported schedule unit: {schedule.unit} for recurring schedule {job_id}. "
                    "Skipping job creation."
                )
                return
            log_details = (
                f"Recurring every {schedule.interval} {schedule.unit.value}. "
                f"Cron: (M='{cron_expression_params['minute']}', "
                f"H='{cron_expression_params['hour']}', "
                f"DoM='{cron_expression_params['day']}', "
                f"Mon='{cron_expression_params['month']}', "
                f"DoW='{cron_expression_params['day_of_week']}'). "
                f"Grace: {misfire_grace_time_seconds}s."
            )
            effective_end_date = schedule.end_time_utc
            trigger = CronTrigger(
                timezone="UTC",
                start_date=schedule.start_time_utc,
                end_date=effective_end_date,
                **cron_expression_params,
            )

        start_log = schedule.start_time_utc or "Immediate (if cron matches)"
        end_log = effective_end_date or "Never"
        log_msg = (
            f"Adding/updating job {job_id} for agent {agent_id_str}. {log_details} "
            f"Effective Start: {start_log}, Effective End: {end_log}."
        )
        logger.debug(log_msg)
        logger.debug(f"[SchedulerService] Trigger details: {trigger}")

        try:
            self.scheduler.add_job(
                self._trigger_agent_task,
                trigger=trigger,
                args=trigger_args,
                id=job_id,
                name=f"Agent {agent_id_str} - Schedule {job_id}",
                replace_existing=True,
                misfire_grace_time=misfire_grace_time_seconds,
                coalesce=True,
            )
            logger.debug(
                f"Successfully added/updated job {job_id} to scheduler with "
                f"misfire_grace_time {misfire_grace_time_seconds}s."
            )
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
                logger.debug(f"Successfully removed job {job_id} from scheduler.")
            except Exception as e:
                logger.error(f"Failed to remove job {job_id} from scheduler: {str(e)}")
        else:
            logger.debug(f"Job {job_id} not found in scheduler, no action taken.")

    async def start(self) -> None:
        """
        Starts the APScheduler and loads all existing active schedules.
        """
        logger.debug("Starting SchedulerService...")
        if not self.scheduler.running:
            self.scheduler.start()
            logger.debug("APScheduler started. (is running: %s)", self.scheduler.running)
        else:
            logger.debug("APScheduler already running. (is running: %s)", self.scheduler.running)

        # DEBUG: Log all jobs currently scheduled
        jobs = self.scheduler.get_jobs()
        logger.debug("Current scheduled jobs at start: %s", jobs)

        await self.load_all_agent_schedules()

    async def load_all_agent_schedules(self) -> None:
        """
        Loads all active schedules for all agents and adds them to the scheduler.
        Handles past-due one-time jobs by triggering them immediately.
        """
        logger.debug("Loading all agent schedules into APScheduler...")
        now_utc = datetime.now(timezone.utc)
        try:
            all_agents = self.agent_registry.list_agents()
            for agent_data in all_agents:
                agent_state_needs_saving = False
                try:
                    agent_state = self.agent_registry.load_agent_state(agent_data.id)
                    if not agent_state.schedules:
                        logger.debug(f"No schedules found for agent {agent_data.id}")
                        continue

                    logger.debug(
                        f"Processing {len(agent_state.schedules)} schedules "
                        f"for agent {agent_data.id}"
                    )

                    schedules_to_process = list(agent_state.schedules)

                    for schedule_item in schedules_to_process:
                        job_id_str = str(schedule_item.id)
                        agent_id_str = str(schedule_item.agent_id)

                        # A. Handle schedules that have ended
                        if schedule_item.end_time_utc and now_utc >= schedule_item.end_time_utc:
                            log_msg_ended = (
                                f"Schedule {job_id_str} for agent {agent_id_str} has passed its "
                                f"end time ({schedule_item.end_time_utc}). Ensuring "
                                "inactive and removed."
                            )
                            logger.debug(log_msg_ended)
                            if schedule_item.is_active:
                                schedule_item.is_active = False
                                agent_state_needs_saving = True
                            self.remove_job(schedule_item.id)
                            continue  # Move to the next schedule

                        # B. Handle explicitly inactive schedules
                        if not schedule_item.is_active:
                            log_msg_inactive = (
                                f"Schedule {job_id_str} for agent {agent_id_str} is "
                                "marked inactive. Ensuring removal from scheduler."
                            )
                            logger.debug(log_msg_inactive)
                            self.remove_job(schedule_item.id)
                            agent_state_needs_saving = True

                            continue  # Move to the next schedule

                        # At this point, the schedule is active and not past its end_time_utc

                        # C. Handle active one-time schedules
                        if schedule_item.one_time:
                            if not schedule_item.start_time_utc:
                                log_msg_no_start = (
                                    f"Active one-time schedule {job_id_str} "
                                    f"for agent {agent_id_str} lacks start_time_utc. "
                                    "Marking inactive."
                                )
                                logger.error(log_msg_no_start)
                                schedule_item.is_active = False
                                agent_state_needs_saving = True
                                self.remove_job(schedule_item.id)
                                continue

                            if now_utc > schedule_item.start_time_utc:
                                log_msg_past_due = (
                                    f"Past-due active one-time schedule {job_id_str} for "
                                    f"agent {agent_id_str} "
                                    f"(start: {schedule_item.start_time_utc}). "
                                    "Triggering now (non-blocking)."
                                )
                                logger.debug(log_msg_past_due)
                                asyncio.create_task(
                                    self._trigger_agent_task(
                                        agent_id_str=agent_id_str,
                                        schedule_id_str=job_id_str,
                                        prompt=schedule_item.prompt,
                                    )
                                )
                                # The task itself should mark it inactive and it will be removed.
                                # No need to call add_or_update_job for this one.
                            else:
                                # Future one-time job, add it to scheduler
                                log_msg_future_one_time = (
                                    f"Future active one-time schedule {job_id_str} for "
                                    f"agent {agent_id_str}. Adding to scheduler."
                                )
                                logger.debug(log_msg_future_one_time)
                                self.add_or_update_job(schedule_item)
                            continue

                        # D. Handle active recurring schedules
                        # Check if a recurring job was missed and needs to be run
                        triggered_missed_recurring = False
                        if (
                            not schedule_item.one_time
                            and schedule_item.last_run_at
                            and schedule_item.interval > 0
                            and schedule_item.unit
                        ):

                            delta = timedelta()
                            if schedule_item.unit == ScheduleUnit.MINUTES:
                                delta = timedelta(minutes=schedule_item.interval)
                            elif schedule_item.unit == ScheduleUnit.HOURS:
                                delta = timedelta(hours=schedule_item.interval)
                            elif schedule_item.unit == ScheduleUnit.DAYS:
                                delta = timedelta(days=schedule_item.interval)

                            if delta > timedelta(0):  # Ensure valid delta
                                next_expected_run_time = schedule_item.last_run_at + delta
                                grace_period_seconds = 0
                                if schedule_item.unit == ScheduleUnit.MINUTES:
                                    grace_period_seconds = int(schedule_item.interval * 60 / 2)
                                elif schedule_item.unit == ScheduleUnit.HOURS:
                                    grace_period_seconds = int(schedule_item.interval * 60 * 60 / 2)
                                elif schedule_item.unit == ScheduleUnit.DAYS:
                                    grace_period_seconds = int(
                                        schedule_item.interval * 24 * 60 * 60 / 2
                                    )
                                grace_period_seconds = max(
                                    60, grace_period_seconds
                                )  # Min 60s grace

                                grace_delta = timedelta(seconds=grace_period_seconds)

                                if now_utc > next_expected_run_time and now_utc <= (
                                    next_expected_run_time + grace_delta
                                ):
                                    logger.debug(
                                        f"Missed recurring schedule {job_id_str} for "
                                        f"agent {agent_id_str} "
                                        f"(last run: {schedule_item.last_run_at}, "
                                        f"next expected: {next_expected_run_time}, "
                                        f"grace: {grace_delta}). Triggering now (non-blocking)."
                                    )
                                    asyncio.create_task(
                                        self._trigger_agent_task(
                                            agent_id_str=agent_id_str,
                                            schedule_id_str=job_id_str,
                                            prompt=schedule_item.prompt,
                                        )
                                    )
                                    triggered_missed_recurring = True
                                    # The task itself will update last_run_at.
                                    # We still add the job to the scheduler for future runs.

                        if (
                            not triggered_missed_recurring
                        ):  # If not triggered as missed, or if it's first run
                            log_msg_recurring = (
                                f"Active recurring schedule {job_id_str} for agent {agent_id_str}. "
                                "Adding/updating in scheduler."
                            )
                            logger.debug(log_msg_recurring)

                        # Always ensure the job is (re)added to the scheduler for future runs
                        self.add_or_update_job(schedule_item)

                    if agent_state_needs_saving:
                        # Remove inactive schedules from agent state
                        agent_state.schedules = [
                            sched for sched in agent_state.schedules if sched.is_active
                        ]

                        self.agent_registry.save_agent_state(agent_data.id, agent_state)

                except Exception as e:
                    logger.error(
                        f"Error loading or processing schedules for "
                        f"agent {agent_data.id}: {str(e)}",
                        exc_info=True,
                    )
            logger.debug("Finished loading and processing agent schedules.")
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
            logger.debug("APScheduler shut down.")
