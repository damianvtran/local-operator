import asyncio
import logging
from datetime import datetime, timedelta, timezone
from multiprocessing import Queue
from pathlib import Path
from typing import Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from local_operator.agents import AgentData, AgentRegistry
from local_operator.bootstrap import initialize_operator
from local_operator.clients.radient import RadientClient, RadientTokenResponse
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import JobContext, JobContextRecord, JobManager, JobStatus
from local_operator.operator import OperatorType
from local_operator.prompts import ScheduleInstructionsPrompt
from local_operator.server.utils.job_processor_queue import (
    create_and_start_job_process_with_queue,
)
from local_operator.server.utils.websocket_manager import WebSocketManager
from local_operator.types import Schedule, ScheduleUnit

logger = logging.getLogger(__name__)

# Constants for Google Credentials (some retained for storing token)
GOOGLE_ACCESS_TOKEN_KEY = "GOOGLE_ACCESS_TOKEN"
GOOGLE_REFRESH_TOKEN_KEY = "GOOGLE_REFRESH_TOKEN"  # This is the token we use to refresh
GOOGLE_TOKEN_EXPIRY_TIMESTAMP_KEY = "GOOGLE_TOKEN_EXPIRY_TIMESTAMP"
RADIENT_TOKEN_REFRESH_JOB_ID = "radient_google_token_refresh_job"
# Refresh every 15 minutes, Google access tokens typically last 1 hour (3600s)
TOKEN_REFRESH_CRON_MINUTES = "*/15"  # Generic name, can be used for other providers too


def _execute_scheduled_task_logic(
    job_id: str,  # This will be the schedule_id_str
    agent_id_str: str,
    schedule_id_str: str,  # schedule_id is the same as job_id for scheduled tasks
    prompt: str,
    agent_registry_config_dir: str,
    env_config: EnvConfig,
    operator_type_str: str,
    verbosity_level_str: str,
    target_agent_hosting: str,  # From target_agent_data
    target_agent_model: str,  # From target_agent_data
    status_queue: Optional[Queue] = None,  # type: ignore
):
    """
    The core logic for executing a scheduled agent task in a separate process.
    This function is designed to be picklable and run by multiprocessing.Process.
    """
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Reconstruct managers
    agent_registry = AgentRegistry(config_dir=Path(agent_registry_config_dir))
    config_manager = ConfigManager(config_dir=Path(agent_registry_config_dir))
    credential_manager = CredentialManager(config_dir=Path(agent_registry_config_dir))
    operator_type = OperatorType[operator_type_str]
    verbosity_level = VerbosityLevel[verbosity_level_str]

    # The scheduler_service instance for tools within a scheduled task.
    # Tools should use the status_queue to request scheduling changes from the main process.
    # So, we pass None or a proxy that uses the queue. For now, None.
    scheduler_service_for_tools = None

    async def task_execution_coroutine():
        job_context = JobContext()
        with job_context:
            if status_queue:
                status_queue.put(("status_update", job_id, JobStatus.PROCESSING, None))

            try:
                logger.debug(
                    f"Process {job_id}: Starting task for agent {agent_id_str}, "
                    f"schedule {schedule_id_str}"
                )

                # AgentData needs to be reconstructed or key parts passed.
                # We passed hosting and model, assuming that's sufficient for initialize_operator
                # along with agent_id_str for loading state.
                # If current_agent object is complex, more data might be needed.
                # For now, initialize_operator will load the agent using agent_id_str.
                # We need to ensure get_agent can be called with agent_id_str.
                # The original code did:
                # target_agent_data = self.agent_registry.get_agent(agent_id_str)
                # So, agent_registry.get_agent(agent_id_str) should work.
                current_agent_obj: AgentData = agent_registry.get_agent(agent_id_str)

                task_operator = initialize_operator(
                    operator_type=operator_type,
                    config_manager=config_manager,
                    credential_manager=credential_manager,
                    agent_registry=agent_registry,
                    env_config=env_config,
                    request_hosting=target_agent_hosting,  # Use passed value
                    request_model=target_agent_model,  # Use passed value
                    current_agent=current_agent_obj,  # Loaded agent object
                    scheduler_service=scheduler_service_for_tools,
                    persist_conversation=True,
                    auto_save_conversation=False,
                    verbosity_level=verbosity_level,
                    job_id=job_id,
                    status_queue=status_queue,
                )

                if status_queue and hasattr(task_operator, "executor"):
                    task_operator.executor.status_queue = status_queue

                additional_instructions = ScheduleInstructionsPrompt

                _, final_response = await task_operator.handle_user_input(
                    prompt, additional_instructions=additional_instructions
                )
                log_msg_response = final_response[:100] if final_response else ""
                logger.debug(
                    f"Process {job_id}: Task completed for agent {agent_id_str}. "
                    f"Response: {log_msg_response}"
                )

                now_utc_after_task = datetime.now(timezone.utc)
                agent_state_after_task = agent_registry.load_agent_state(agent_id_str)
                schedule_id_uuid = UUID(schedule_id_str)
                schedule_modified_in_state = False

                schedules_copy = list(agent_state_after_task.schedules)
                for sched_idx, sched_item in enumerate(schedules_copy):
                    if sched_item.id == schedule_id_uuid:
                        sched_item.last_run_at = now_utc_after_task
                        schedule_modified_in_state = True

                        if sched_item.one_time:
                            logger.debug(
                                f"Process {job_id}: One-time schedule {schedule_id_str} executed. "
                                "Removing from agent state."
                            )
                            agent_state_after_task.schedules.pop(sched_idx)
                            # The job in JobManager will be marked COMPLETED.
                            # APScheduler job removal is handled by the main SchedulerService
                            # based on this completion or if it was a DateTrigger.
                            break

                        if (
                            sched_item.end_time_utc
                            and now_utc_after_task >= sched_item.end_time_utc
                        ):
                            if sched_item.is_active:
                                logger.debug(
                                    f"Process {job_id}: Schedule {schedule_id_str} passed end time "
                                    f"({sched_item.end_time_utc}) after task. Marking inactive."
                                )
                                sched_item.is_active = False
                            # schedule_modified_in_state is already true
                            # APScheduler job removal for end_time_utc is handled by
                            # main SchedulerService.
                        break

                if schedule_modified_in_state:
                    agent_registry.save_agent_state(agent_id_str, agent_state_after_task)
                    logger.debug(
                        f"Process {job_id}: Updated agent state for schedule {schedule_id_str}"
                    )

                result_payload = {
                    "response": final_response or "",
                    "context": [
                        JobContextRecord(role=msg.role, content=msg.content, files=msg.files)
                        for msg in task_operator.executor.agent_state.conversation
                    ],
                    "schedule_id": schedule_id_str,
                    "agent_id": agent_id_str,
                }
                if status_queue:
                    status_queue.put(("status_update", job_id, JobStatus.COMPLETED, result_payload))

            except Exception as op_error:
                logger.error(
                    f"Process {job_id}: Failed to execute task for agent {agent_id_str}, "
                    f"schedule {schedule_id_str}: {op_error}",
                    exc_info=True,
                )
                if status_queue:
                    status_queue.put(
                        (
                            "status_update",
                            job_id,
                            JobStatus.FAILED,
                            {"error": str(op_error), "schedule_id": schedule_id_str},
                        )
                    )
            finally:
                logger.debug(f"Process {job_id}: Exiting task execution coroutine.")

    loop.run_until_complete(task_execution_coroutine())
    loop.close()


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
        job_manager: JobManager,  # Added
        websocket_manager: WebSocketManager,  # Added
    ):
        self.agent_registry = agent_registry
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self.env_config = env_config
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.operator_type = operator_type
        self.verbosity_level = verbosity_level
        self.job_manager = job_manager  # Added
        self.websocket_manager = websocket_manager  # Added

    async def _execute_radient_token_refresh_task(self) -> None:
        """
        Task executed by the scheduler to refresh Google OAuth tokens via Radient.
        """
        logger.info("Attempting to refresh Google OAuth access token via Radient...")
        try:
            radient_client_id = self.env_config.radient_client_id
            google_refresh_token_secret = self.credential_manager.get_credential(
                GOOGLE_REFRESH_TOKEN_KEY
            )

            if not radient_client_id:
                logger.warning(
                    "RADIENT_CLIENT_ID not found in environment configuration. "
                    "Skipping token refresh."
                )
                return

            if not (google_refresh_token_secret and google_refresh_token_secret.get_secret_value()):
                logger.warning(
                    "Google refresh token not found or empty in credentials. "
                    "Skipping token refresh."
                )
                return

            logger.debug("Retrieved Radient Client ID and Google refresh token for token refresh.")

            radient_api_key = self.credential_manager.get_credential("RADIENT_API_KEY")
            radient_client = RadientClient(
                api_key=radient_api_key, base_url=self.env_config.radient_api_base_url
            )

            refresh_response: RadientTokenResponse = await asyncio.to_thread(
                radient_client.refresh_token,
                client_id=radient_client_id,
                refresh_token=google_refresh_token_secret,
                provider="google",  # Specify Google provider for Radient endpoint
            )

            # The refresh_response is now directly the RadientTokenResponse model
            new_access_token_secret = refresh_response.access_token
            expires_in = refresh_response.expires_in  # Seconds

            if not new_access_token_secret or expires_in is None:
                logger.error(
                    "Failed to get new access token or expiry from Radient refresh response: %s",
                    refresh_response.dict(),  # .dict() is available on RadientTokenResponse
                )
                return

            new_access_token = new_access_token_secret.get_secret_value()
            # Calculate new expiry timestamp
            # Add a small buffer (e.g., 60 seconds) to be safe
            expiry_timestamp = int(datetime.now(timezone.utc).timestamp()) + expires_in - 60

            self.credential_manager.set_credential(
                GOOGLE_ACCESS_TOKEN_KEY, new_access_token, write=False
            )
            self.credential_manager.set_credential(
                GOOGLE_TOKEN_EXPIRY_TIMESTAMP_KEY, str(expiry_timestamp), write=False
            )
            # If Radient issues a new refresh token, store it (optional, depends on Radient's flow)
            if refresh_response.refresh_token:
                self.credential_manager.set_credential(
                    GOOGLE_REFRESH_TOKEN_KEY,
                    refresh_response.refresh_token.get_secret_value(),
                    write=False,
                )
            self.credential_manager.write_to_file()  # Persist all changes

            logger.info(
                (
                    "Successfully refreshed Google OAuth access token via Radient. "
                    "New expiry (UTC): %s"
                ),
                datetime.fromtimestamp(expiry_timestamp, tz=timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.error(
                f"Unexpected error during Radient Google token refresh: {e}", exc_info=True
            )

    def _schedule_radient_token_refresh(self) -> None:
        """
        Schedules the Radient Google token refresh job if the necessary credentials exist.
        """
        if not self.scheduler.running:
            logger.error("Scheduler is not running. Cannot schedule Radient token refresh.")
            return

        # Check if GOOGLE_REFRESH_TOKEN is set, as it's essential
        # The job will now always be scheduled.
        # The execution task (_execute_radient_token_refresh_task)
        # will check for credentials at runtime.
        if self.scheduler.get_job(RADIENT_TOKEN_REFRESH_JOB_ID):
            logger.debug("Radient Google token refresh job already scheduled. Skipping re-adding.")
            return

        try:
            self.scheduler.add_job(
                self._execute_radient_token_refresh_task,
                trigger=CronTrigger(minute=TOKEN_REFRESH_CRON_MINUTES, timezone="UTC"),
                id=RADIENT_TOKEN_REFRESH_JOB_ID,
                name="Radient Google OAuth Token Refresh",
                replace_existing=True,
                misfire_grace_time=300,  # 5 minutes
                coalesce=True,
            )
            logger.info(
                "Scheduled Radient Google token refresh job with CRON expression: "
                f"minute='{TOKEN_REFRESH_CRON_MINUTES}'."
            )
        except Exception as e:
            logger.error(f"Failed to schedule Radient Google token refresh job: {e}", exc_info=True)

    async def _trigger_agent_task(
        self, agent_id_str: str, schedule_id_str: str, prompt: str
    ) -> None:
        """
        The actual function called by APScheduler to trigger an agent's task.
        It now creates a job and starts a new process for execution.
        """
        logger.debug(
            f"Attempting to trigger task for agent {agent_id_str}, schedule {schedule_id_str}"
        )
        try:
            schedule_id_uuid = UUID(schedule_id_str)

            # Load schedule details to check end_time_utc and active status
            agent_state = self.agent_registry.load_agent_state(agent_id_str)
            current_schedule: Schedule | None = None
            for sched in agent_state.schedules:
                if sched.id == schedule_id_uuid:
                    current_schedule = sched
                    break

            if not current_schedule:
                logger.error(
                    f"Schedule {schedule_id_str} not found for agent {agent_id_str}. "
                    "Removing APScheduler job."
                )
                self.remove_job(schedule_id_uuid)  # Removes from APScheduler
                return

            if not current_schedule.is_active:
                logger.debug(
                    f"Schedule {schedule_id_str} is no longer active. Removing APScheduler job."
                )
                self.remove_job(schedule_id_uuid)
                # Agent state for inactive schedules is handled during load_all_agent_schedules
                # or when a schedule is explicitly deactivated.
                return

            now_utc = datetime.now(timezone.utc)
            if current_schedule.end_time_utc and now_utc >= current_schedule.end_time_utc:
                logger.debug(
                    f"Schedule {schedule_id_str} passed end time "
                    f"({current_schedule.end_time_utc}). Removing APScheduler job, "
                    "marking inactive in agent state."
                )
                # Mark inactive in current state and save
                current_schedule.is_active = False
                # This modification to agent_state needs to be saved.
                # The _execute_scheduled_task_logic also handles this, but good to do
                # it here too for clarity.
                # However, to avoid race conditions, let the job process handle final state.
                # For now, just remove from APScheduler.
                self.remove_job(schedule_id_uuid)

                # Update agent state to reflect inactive
                updated_schedules = []
                schedule_found_for_deactivation = False
                for sched_in_state in agent_state.schedules:
                    if sched_in_state.id == schedule_id_uuid:
                        sched_in_state.is_active = False
                        schedule_found_for_deactivation = True
                    updated_schedules.append(sched_in_state)
                if schedule_found_for_deactivation:
                    agent_state.schedules = updated_schedules
                    self.agent_registry.save_agent_state(agent_id_str, agent_state)
                return

            logger.info(
                f"Creating job for scheduled task: agent {agent_id_str}, schedule {schedule_id_str}"
            )

            target_agent_data: AgentData = self.agent_registry.get_agent(agent_id_str)

            # Create a job in JobManager. The job_id will be the schedule_id_str.
            # The JobManager's create_job is async.
            job_entry = await self.job_manager.create_job(
                prompt=prompt,
                model=target_agent_data.model,  # Model from agent data
                hosting=target_agent_data.hosting,  # Hosting from agent data
                agent_id=agent_id_str,
                job_id=schedule_id_str,  # Pass schedule_id_str as job_id
            )
            # job_id from job_entry should be schedule_id_str

            # Prepare arguments for _execute_scheduled_task_logic
            # Ensure all managers' configurations are serializable (e.g., file paths)
            process_args = (
                job_entry.id,  # This is schedule_id_str
                agent_id_str,
                schedule_id_str,
                prompt,
                self.agent_registry.config_dir,
                self.env_config,
                self.operator_type.name,
                self.verbosity_level.name,
                target_agent_data.hosting,
                target_agent_data.model,
                # status_queue is added by create_and_start_job_process_with_queue
            )

            create_and_start_job_process_with_queue(
                job_id=job_entry.id,
                process_func=_execute_scheduled_task_logic,
                args=process_args,
                job_manager=self.job_manager,
                websocket_manager=self.websocket_manager,
                scheduler_service=self,  # Pass self for queue monitor to call add/remove schedule
            )
            logger.info(
                f"Scheduled task job {job_entry.id} for agent {agent_id_str} "
                "started in a new process."
            )

            # The responsibility of updating agent_state (last_run_at, one_time removal, etc.)
            # is now within _execute_scheduled_task_logic, which communicates status.
            # The main scheduler service might react to JobStatus.COMPLETED for one-time jobs
            # to remove them from APScheduler if they were cron-based one-time.
            # DateTrigger one-time jobs are automatically removed by APScheduler.

        except Exception as e:
            logger.error(
                f"Error creating or starting job for scheduled task agent {agent_id_str}, "
                f"schedule {schedule_id_str}: {str(e)}",
                exc_info=True,
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
            # Prefer start_time_utc if provided, otherwise use interval/unit
            if schedule.start_time_utc:
                # Use DateTrigger for one-time jobs with only start_time_utc
                misfire_grace_time_seconds = 60  # Fixed 60s for specific date triggers
                start_time_str = schedule.start_time_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
                log_details = f"One-time at {start_time_str}. Grace: {misfire_grace_time_seconds}s."
                trigger = DateTrigger(
                    run_date=schedule.start_time_utc,
                    timezone="UTC",
                )  # misfire_grace_time is an add_job param
            elif schedule.interval and schedule.unit:
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

        start_log_val = schedule.start_time_utc or "Immediate (if cron matches)"
        end_log_val = effective_end_date or "Never"
        log_msg = (
            f"Adding/updating job {job_id} for agent {agent_id_str}. {log_details} "
            f"Effective Start: {start_log_val}, Effective End: {end_log_val}."
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
        Starts the APScheduler, loads all existing active schedules,
        and schedules the Google token refresh job.
        """
        logger.debug("Starting SchedulerService...")
        if not self.scheduler.running:
            try:
                self.scheduler.start(paused=False)  # Ensure it's not started in paused state
                logger.debug("APScheduler started. (is running: %s)", self.scheduler.running)
            except Exception as e:
                logger.error(f"Failed to start APScheduler: {e}", exc_info=True)
                return  # Cannot proceed if scheduler doesn't start
        else:
            logger.debug("APScheduler already running. (is running: %s)", self.scheduler.running)

        # Schedule Radient Google token refresh
        self._schedule_radient_token_refresh()

        # DEBUG: Log all jobs currently scheduled
        jobs = self.scheduler.get_jobs()
        logger.debug(
            "Current scheduled jobs at start (after Radient Google refresh schedule): %s", jobs
        )

        await self.load_all_agent_schedules()

    def add_radient_token_refresh_job_if_needed(self) -> None:
        """
        Public method to explicitly try to add the Radient Google token refresh job.
        Useful if credentials are added after initial startup.
        """
        logger.info(
            "Explicitly checking and scheduling Radient Google token refresh job if needed."
        )
        self._schedule_radient_token_refresh()

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
                                # _trigger_agent_task is now async due to job_manager.create_job
                                # and handles process creation.
                                # We need to run it and not block load_all_agent_schedules.
                                asyncio.create_task(
                                    self._trigger_agent_task(
                                        agent_id_str=agent_id_str,
                                        schedule_id_str=job_id_str,
                                        prompt=schedule_item.prompt,
                                    )
                                )
                                # The new _trigger_agent_task will create a job.
                                # The job process (_execute_scheduled_task_logic)
                                # handles agent state.
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
                                grace_period_seconds = 0
                                if schedule_item.unit == ScheduleUnit.MINUTES:
                                    grace_period_seconds = int(schedule_item.interval * 60 / 2)
                                elif schedule_item.unit == ScheduleUnit.HOURS:
                                    grace_period_seconds = int(schedule_item.interval * 60 * 60 / 2)
                                elif schedule_item.unit == ScheduleUnit.DAYS:
                                    grace_period_seconds = int(
                                        schedule_item.interval * 24 * 60 * 60 / 2
                                    )
                                grace_period_seconds = max(60, grace_period_seconds)
                                grace_delta = timedelta(seconds=grace_period_seconds)

                                # If last_run_at exists, use it to determine missed run
                                if schedule_item.last_run_at:
                                    next_expected_run_time = schedule_item.last_run_at + delta
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
                                # If no last_run_at, check if the first run
                                # was missed based on start_time_utc
                                elif schedule_item.start_time_utc:
                                    first_expected_run_time = schedule_item.start_time_utc
                                    if now_utc > first_expected_run_time and now_utc <= (
                                        first_expected_run_time + grace_delta
                                    ):
                                        logger.debug(
                                            f"Missed first recurring schedule {job_id_str} for "
                                            f"agent {agent_id_str} "
                                            f"(start: {first_expected_run_time}, "
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
                                    # If now_utc is past the grace window, do
                                    # not trigger, just add to scheduler
                                    # If now_utc is before start_time_utc, do
                                    # nothing (future run)
                                    # If no start_time_utc, do nothing
                                    # (If both last_run_at and start_time_utc are None, do nothing)

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
