"""Scheduled task service — rewritten engine port (CL-07).

The scheduler loads each agent's ``schedules.jsonl`` (owned by
:class:`~local_operator.agents.AgentRegistry`), arms APScheduler triggers for
active :class:`~local_operator.types.Schedule` entries, and runs due prompts
**in-process** through the rewritten engine:

- one asyncio task per run, built via ``session_factory.create_session`` +
  ``Session.prompt`` — NO multiprocessing fork (the fork existed only to
  isolate the old ``exec()``-based engine; the new engine is async-native);
- ``train=True`` semantics: the run's transcript persists into the agent's
  own directory (``agents/<id>/transcript.jsonl``), mirroring the legacy
  ``persist_conversation=True`` behavior;
- ``yolo=True``: scheduled runs are unattended, tool approvals auto-grant;
- per-run timeout (default 30 minutes, configurable via the
  ``LOCAL_OPERATOR_SCHEDULED_TASK_TIMEOUT_SECONDS`` environment variable);
- failures are logged and recorded in the job ledger
  (:class:`~local_operator.jobs.JobStatus.FAILED`) — a failing run never
  kills the scheduler loop.

Deliberate removals from the legacy module (disposition notes):

- **Radient/Google OAuth refresh cron** (``RADIENT_TOKEN_REFRESH_JOB_ID``,
  ``_execute_radient_token_refresh_task``,
  ``add_radient_token_refresh_job_if_needed``) is REMOVED. Token refresh is a
  provider concern: the rewrite's providers auth store
  (``local_operator/providers/auth_store.py``) auto-refreshes on use. The
  legacy module mixed that unrelated cron into the task scheduler.
- **``_execute_scheduled_task_logic``** (the picklable child-process
  entrypoint) is gone with the multiprocessing fork. The legacy test file
  targeting it (``tests/unit/test_scheduler_service.py``) was removed with
  the old engine.
- **``ScheduleInstructionsPrompt``** is INLINED as ``SCHEDULE_INSTRUCTIONS``
  (verbatim): its home, ``local_operator.prompts``, was deleted with the
  legacy engine.
- ``operator_type`` / ``verbosity_level`` / ``env_config`` are accepted for
  interface compatibility with the legacy constructor (``server/app.py`` and
  ``cli.py`` pass them) but are informational only: the rewritten engine has
  no CLI/SERVER operator distinction and does its own env-config wiring.

Legacy semantics preserved:

- interval+unit (MINUTES/HOURS/DAYS) cron mapping with
  ``start_time_utc``/``end_time_utc`` bounds and identical
  misfire-grace/coalesce settings;
- one-time schedules pop from the agent state after a successful run;
  past-due one-time schedules replay immediately on load;
  ``last_run_at`` stamps only on success; end-time runs deactivate;
- ``load_all_agent_schedules`` purges inactive/ended schedules from
  ``schedules.jsonl`` when saving, and replays missed recurring runs inside
  the legacy grace window;
- in-flight runs are cancelled on shutdown (was: orphaned child processes).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from local_operator import session_factory
from local_operator.agents import AgentData, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import JobContextRecord, JobManager, JobStatus
from local_operator.types import ConversationRole, OperatorType, Schedule, ScheduleUnit

if TYPE_CHECKING:
    from local_operator.server.utils.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

#: Default per-run timeout (30 minutes). Override with the
#: ``LOCAL_OPERATOR_SCHEDULED_TASK_TIMEOUT_SECONDS`` environment variable.
DEFAULT_RUN_TIMEOUT_SECONDS = 30 * 60
RUN_TIMEOUT_ENV_VAR = "LOCAL_OPERATOR_SCHEDULED_TASK_TIMEOUT_SECONDS"

#: Verbatim copy of the legacy ``prompts.ScheduleInstructionsPrompt`` (see the
#: module docstring for why it is inlined).
SCHEDULE_INSTRUCTIONS: str = """
This is the next scheduled task that you must run again

Make sure to complete the task completely and with all required steps and details.  Compare any results with the previous task and update as needed.  Don't assume any steps have already been completed.

Don't make assumptions about variables or data that are already in your context and conversation history.  Even if you see completed text and statuses in your context variables, these are likely stale now and need to be re-done.  Fetch new information as needed, and if you are running a recurring task that depends on new information, make sure to consider the new information in your response if there is any.  Write summaries, emails, reports, and any other text based on the new information and make sure that you are appropriately communicating the new information to the user.
"""  # noqa: E501

#: Harness message roles that map onto the legacy job-context ledger shape.
_MESSAGE_ROLE_TO_CONVERSATION_ROLE = {
    "user": ConversationRole.USER,
    "assistant": ConversationRole.ASSISTANT,
    "tool": ConversationRole.TOOL,
}


class SchedulerService:
    """Service for managing and executing scheduled tasks for agents.

    Public surface (byte-compatible with the legacy service):
    ``start()``, ``shutdown()``, ``load_all_agent_schedules()``,
    ``add_or_update_job(schedule)``, ``remove_job(schedule_id)``.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config_manager: ConfigManager,
        credential_manager: CredentialManager,
        env_config: EnvConfig,
        operator_type: "OperatorType",
        verbosity_level: VerbosityLevel,
        job_manager: JobManager,
        websocket_manager: "WebSocketManager",
    ):
        self.agent_registry = agent_registry
        self.config_manager = config_manager
        self.credential_manager = credential_manager
        self.env_config = env_config
        self.operator_type = operator_type
        self.verbosity_level = verbosity_level
        self.job_manager = job_manager
        self.websocket_manager = websocket_manager

        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._run_tasks: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the APScheduler and load all existing agent schedules."""
        logger.debug("Starting SchedulerService...")
        if not self.scheduler.running:
            try:
                self.scheduler.start(paused=False)
                logger.debug("APScheduler started. (is running: %s)", self.scheduler.running)
            except Exception as e:
                logger.error(f"Failed to start APScheduler: {e}", exc_info=True)
                return  # Cannot proceed if scheduler doesn't start
        else:
            logger.debug("APScheduler already running.")

        await self.load_all_agent_schedules()

    async def shutdown(self) -> None:
        """Stop the APScheduler and cancel in-flight scheduled runs."""
        if self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=False)
                logger.debug("APScheduler shut down.")
            except Exception as e:
                logger.error(f"Failed to shut down APScheduler: {e}", exc_info=True)

        tasks = [t for t in self._run_tasks if not t.done()]
        if not tasks:
            return
        logger.debug("Cancelling %d in-flight scheduled run(s) for shutdown.", len(tasks))
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for %d scheduled run(s) to stop.", len(tasks))

    # ------------------------------------------------------------------
    # Job (trigger) management — public surface used by the server routes
    # ------------------------------------------------------------------

    def add_or_update_job(self, schedule: Schedule) -> None:
        """Add a new job or update an existing one in APScheduler.

        Trigger semantics ported from the legacy service: one-time schedules
        with a ``start_time_utc`` use a ``DateTrigger``; one-time schedules
        with only interval/unit and all recurring schedules use a
        ``CronTrigger`` with the interval mapped onto the matching cron
        field (``*/N``), anchored to the ``start_time_utc`` hour/minute for
        the HOURS and DAYS units. Misfire grace and coalescing match the
        legacy values exactly. The job function is the async coroutine
        ``_trigger_agent_task`` (no multiprocessing).
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
                )
            elif schedule.interval and schedule.unit:
                # One-time job that behaves like a recurring job until it runs once.
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
                        f"schedule {job_id} with interval. Skipping job creation."
                    )
                    return
                log_details = (
                    f"One-time schedule with interval: every {schedule.interval} "
                    f"{schedule.unit.value}. Grace: {misfire_grace_time_seconds}s."
                )
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
                    f"Unsupported schedule unit: {schedule.unit} for recurring schedule "
                    f"{job_id}. Skipping job creation."
                )
                return
            log_details = (
                f"Recurring every {schedule.interval} {schedule.unit.value}. "
                f"Grace: {misfire_grace_time_seconds}s."
            )
            trigger = CronTrigger(
                timezone="UTC",
                start_date=schedule.start_time_utc,
                end_date=effective_end_date,
                **cron_expression_params,
            )

        start_log_val = schedule.start_time_utc or "Immediate (if cron matches)"
        end_log_val = effective_end_date or "Never"
        logger.debug(
            f"Adding/updating job {job_id} for agent {agent_id_str}. {log_details} "
            f"Effective Start: {start_log_val}, Effective End: {end_log_val}."
        )

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
        """Remove a job from APScheduler."""
        job_id = str(schedule_id)
        try:
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
                logger.debug(f"Successfully removed job {job_id} from scheduler.")
            else:
                logger.debug(f"Job {job_id} not found in scheduler, no action taken.")
        except Exception as e:
            logger.error(f"Failed to remove job {job_id} from scheduler: {str(e)}")

    # ------------------------------------------------------------------
    # Loading schedules from disk
    # ------------------------------------------------------------------

    async def load_all_agent_schedules(self) -> None:
        """Load all active schedules for all agents into the scheduler.

        Handles past-due one-time jobs by triggering them immediately, and
        replays missed recurring runs inside the legacy grace window. Ended
        and inactive schedules are removed from persistence on save
        (legacy semantics).
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
                        continue

                    for schedule_item in list(agent_state.schedules):
                        job_id_str = str(schedule_item.id)
                        agent_id_str = str(schedule_item.agent_id)

                        # A. Schedules that have ended
                        if schedule_item.end_time_utc and now_utc >= schedule_item.end_time_utc:
                            logger.debug(
                                f"Schedule {job_id_str} for agent {agent_id_str} has passed "
                                f"its end time ({schedule_item.end_time_utc}). "
                                "Ensuring inactive and removed."
                            )
                            if schedule_item.is_active:
                                schedule_item.is_active = False
                                agent_state_needs_saving = True
                            self.remove_job(schedule_item.id)
                            continue

                        # B. Explicitly inactive schedules
                        if not schedule_item.is_active:
                            logger.debug(
                                f"Schedule {job_id_str} for agent {agent_id_str} is marked "
                                "inactive. Ensuring removal from scheduler."
                            )
                            self.remove_job(schedule_item.id)
                            agent_state_needs_saving = True
                            continue

                        # C. Active one-time schedules
                        if schedule_item.one_time:
                            if not schedule_item.start_time_utc:
                                logger.error(
                                    f"Active one-time schedule {job_id_str} for agent "
                                    f"{agent_id_str} lacks start_time_utc. Marking inactive."
                                )
                                schedule_item.is_active = False
                                agent_state_needs_saving = True
                                self.remove_job(schedule_item.id)
                                continue

                            if now_utc > schedule_item.start_time_utc:
                                logger.debug(
                                    f"Past-due active one-time schedule {job_id_str} for "
                                    f"agent {agent_id_str} "
                                    f"(start: {schedule_item.start_time_utc}). "
                                    "Triggering now (non-blocking)."
                                )
                                self._spawn_trigger(
                                    agent_id_str=agent_id_str,
                                    schedule_id_str=job_id_str,
                                    prompt=schedule_item.prompt,
                                )
                            else:
                                logger.debug(
                                    f"Future active one-time schedule {job_id_str} for "
                                    f"agent {agent_id_str}. Adding to scheduler."
                                )
                                self.add_or_update_job(schedule_item)
                            continue

                        # D. Active recurring schedules: replay a missed run that is still
                        # inside the legacy grace window (half the interval, min 60s).
                        if schedule_item.interval > 0 and schedule_item.unit:
                            delta: Optional[timedelta] = None
                            if schedule_item.unit == ScheduleUnit.MINUTES:
                                delta = timedelta(minutes=schedule_item.interval)
                            elif schedule_item.unit == ScheduleUnit.HOURS:
                                delta = timedelta(hours=schedule_item.interval)
                            elif schedule_item.unit == ScheduleUnit.DAYS:
                                delta = timedelta(days=schedule_item.interval)

                            if delta is not None and delta > timedelta(0):
                                grace_delta = timedelta(
                                    seconds=max(60, int(delta.total_seconds() / 2))
                                )
                                expected_run_time: Optional[datetime] = None
                                if schedule_item.last_run_at:
                                    expected_run_time = schedule_item.last_run_at + delta
                                elif schedule_item.start_time_utc:
                                    expected_run_time = schedule_item.start_time_utc

                                if (
                                    expected_run_time is not None
                                    and now_utc > expected_run_time
                                    and now_utc <= (expected_run_time + grace_delta)  # noqa: E501
                                ):
                                    logger.debug(
                                        f"Missed recurring schedule {job_id_str} for "
                                        f"agent {agent_id_str} (expected: "
                                        f"{expected_run_time}, grace: {grace_delta}). "
                                        "Triggering now (non-blocking)."
                                    )
                                    self._spawn_trigger(
                                        agent_id_str=agent_id_str,
                                        schedule_id_str=job_id_str,
                                        prompt=schedule_item.prompt,
                                    )

                        # Always ensure the job is (re)added for future runs
                        self.add_or_update_job(schedule_item)

                    if agent_state_needs_saving:
                        # Remove inactive schedules from agent state (legacy behavior)
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

    # ------------------------------------------------------------------
    # Run execution (new engine, in-process)
    # ------------------------------------------------------------------

    def _spawn_trigger(self, agent_id_str: str, schedule_id_str: str, prompt: str) -> None:
        """Trigger a run now, non-blocking (used by load-time replays)."""
        task = asyncio.create_task(
            self._trigger_agent_task(
                agent_id_str=agent_id_str,
                schedule_id_str=schedule_id_str,
                prompt=prompt,
            )
        )
        self._run_tasks.add(task)
        task.add_done_callback(self._run_tasks.discard)

    async def _trigger_agent_task(
        self, agent_id_str: str, schedule_id_str: str, prompt: str
    ) -> None:
        """APScheduler job function (async coroutine — no multiprocessing).

        Validates the schedule against current agent state, creates the job
        ledger entry, runs the prompt in-process, stamps schedule bookkeeping
        on success, and records outcomes in the job manager. NEVER raises:
        a failing run must not kill the scheduler loop.
        """
        try:
            await self._trigger_agent_task_inner(agent_id_str, schedule_id_str, prompt)
        except asyncio.CancelledError:
            logger.info(
                f"Scheduled run for agent {agent_id_str}, schedule {schedule_id_str} "
                "was cancelled."
            )
            raise
        except Exception as e:
            logger.error(
                f"Error handling scheduled task trigger for agent {agent_id_str}, "
                f"schedule {schedule_id_str}: {str(e)}",
                exc_info=True,
            )

    async def _trigger_agent_task_inner(
        self, agent_id_str: str, schedule_id_str: str, prompt: str
    ) -> None:
        schedule_id_uuid = UUID(schedule_id_str)

        agent_state = self.agent_registry.load_agent_state(agent_id_str)
        current_schedule: Optional[Schedule] = None
        for sched in agent_state.schedules:
            if sched.id == schedule_id_uuid:
                current_schedule = sched
                break

        if not current_schedule:
            logger.error(
                f"Schedule {schedule_id_str} not found for agent {agent_id_str}. "
                "Removing APScheduler job."
            )
            self.remove_job(schedule_id_uuid)
            return

        if not current_schedule.is_active:
            logger.debug(
                f"Schedule {schedule_id_str} is no longer active. Removing APScheduler job."
            )
            self.remove_job(schedule_id_uuid)
            return

        now_utc = datetime.now(timezone.utc)
        if current_schedule.end_time_utc and now_utc >= current_schedule.end_time_utc:
            logger.debug(
                f"Schedule {schedule_id_str} passed end time "
                f"({current_schedule.end_time_utc}). Marking inactive and removing."
            )
            current_schedule.is_active = False
            try:
                self.agent_registry.save_agent_state(agent_id_str, agent_state)
            except Exception:
                logger.exception(f"Failed to persist deactivated schedule {schedule_id_str}")
            self.remove_job(schedule_id_uuid)
            return

        try:
            target_agent_data: AgentData = self.agent_registry.get_agent(agent_id_str)
        except KeyError:
            logger.error(
                f"Agent {agent_id_str} not found for schedule {schedule_id_str}. "
                "Removing APScheduler job."
            )
            self.remove_job(schedule_id_uuid)
            return

        logger.info(f"Running scheduled task: agent {agent_id_str}, schedule {schedule_id_str}")

        # Job ledger entry; job_id is the schedule id (legacy contract).
        try:
            job_entry = await self.job_manager.create_job(
                prompt=prompt,
                model=target_agent_data.model,
                hosting=target_agent_data.hosting,
                agent_id=agent_id_str,
                job_id=schedule_id_str,
            )
        except Exception as e:
            logger.error(
                f"Failed to create job ledger entry for schedule {schedule_id_str}: " f"{str(e)}",
                exc_info=True,
            )
            return
        job_id = job_entry.id

        self._register_run_task(job_id)
        await self._push_job_status(job_id, JobStatus.PROCESSING)

        timeout_seconds = self._run_timeout_seconds()
        try:
            response_text, context_records = await asyncio.wait_for(
                self._run_agent_session(agent_id_str, target_agent_data, prompt),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Scheduled task {schedule_id_str} timed out after "
                f"{timeout_seconds:.0f} seconds."
            )
            await self._push_job_status(
                job_id,
                JobStatus.FAILED,
                {
                    "error": f"Scheduled task timed out after {timeout_seconds:.0f} seconds",
                    "schedule_id": schedule_id_str,
                    "agent_id": agent_id_str,
                },
            )
            return
        except Exception as e:
            logger.error(
                f"Scheduled task {schedule_id_str} for agent {agent_id_str} failed: " f"{str(e)}",
                exc_info=True,
            )
            await self._push_job_status(
                job_id,
                JobStatus.FAILED,
                {
                    "error": str(e),
                    "schedule_id": schedule_id_str,
                    "agent_id": agent_id_str,
                },
            )
            return

        # Success: legacy bookkeeping (last_run_at stamp, one_time pop,
        # end-time deactivation) happens on success only.
        try:
            self._record_successful_run(agent_id_str, schedule_id_uuid)
        except Exception:
            logger.exception(f"Failed to record successful run for schedule {schedule_id_str}")

        await self._push_job_status(
            job_id,
            JobStatus.COMPLETED,
            {
                "response": response_text,
                "context": context_records,
                "schedule_id": schedule_id_str,
                "agent_id": agent_id_str,
            },
        )

    def _register_run_task(self, job_id: str) -> None:
        """Register the running coroutine with the job manager so cancel_job works."""
        task = asyncio.current_task()
        if task is None:
            return

        async def _register() -> None:
            try:
                await self.job_manager.register_task(job_id, task)
            except KeyError:
                logger.debug(f"Job {job_id} missing when registering run task.")
            except Exception:
                logger.exception(f"Failed to register run task for job {job_id}.")

        asyncio.ensure_future(_register())

    def _run_timeout_seconds(self) -> float:
        """Per-run timeout: env override, else the 30-minute default."""
        raw = os.environ.get(RUN_TIMEOUT_ENV_VAR)
        if raw:
            try:
                value = float(raw)
                if value > 0:
                    return value
            except ValueError:
                pass
            logger.warning(
                f"Invalid {RUN_TIMEOUT_ENV_VAR}={raw!r}; using the default "
                f"{DEFAULT_RUN_TIMEOUT_SECONDS}s."
            )
        return float(DEFAULT_RUN_TIMEOUT_SECONDS)

    def _resolve_agent_cwd(self, agent_data: AgentData) -> Optional[str]:
        """The agent's working directory for session construction, if usable."""
        raw = getattr(agent_data, "current_working_directory", None)
        if not raw or raw == ".":
            return None
        expanded = os.path.expanduser(raw)
        if not os.path.isdir(expanded):
            logger.warning(
                f"Agent working directory {raw!r} does not exist; "
                "running in the scheduler's cwd instead."
            )
            return None
        return expanded

    async def _run_agent_session(
        self, agent_id_str: str, agent_data: AgentData, prompt: str
    ) -> tuple[str, list[JobContextRecord]]:
        """Build a fresh session for the scheduled run and drive one turn.

        Returns ``(response_text, context_records)``. Isolation: a fresh
        session per run with ``train=True`` so the turn persists into the
        agent's own transcript, and ``yolo=True`` (unattended approval
        auto-grant). The engine reads ``os.getcwd()`` at construction time,
        so the agent's working directory is applied around construction only.
        """
        session_args = argparse.Namespace(
            hosting=agent_data.hosting or None,
            model=agent_data.model or None,
            agent_name=None,
            agent_id=agent_id_str,
            yolo=True,
            train=True,
        )

        captured_events: list[Any] = []

        def _capture(event: Any) -> None:
            if getattr(event, "type", None) == "agent_end":
                captured_events.append(event)

        # The agent's working directory is passed, not chdir'd: os.chdir is
        # process-global and create_session awaits inside the window (skills,
        # MCP discovery, transport spawns), so every other session builder in
        # the same process (server routes, TUI turns) would read the wrong
        # directory while a scheduled run held the lock.
        cwd_override = self._resolve_agent_cwd(agent_data)
        session = await session_factory.create_session(
            session_args,
            self.config_manager,
            self.credential_manager,
            self.agent_registry,
            has_ui=False,
            cwd=cwd_override,
        )

        unsubscribe = None
        subscribe = getattr(session, "subscribe", None)
        if callable(subscribe):
            try:
                unsubscribe = subscribe(_capture)
            except Exception:
                logger.debug("Session subscribe failed for scheduled run.", exc_info=True)
                unsubscribe = None

        full_prompt = f"{prompt}\n\n## Additional Instructions\n\n{SCHEDULE_INSTRUCTIONS}"

        try:
            await session.prompt(full_prompt)
        finally:
            if callable(unsubscribe):
                try:
                    unsubscribe()
                except Exception:
                    pass
            try:
                await session.dispose()
            except Exception:
                logger.exception(
                    f"Failed to dispose scheduled-run session for agent {agent_id_str}."
                )

        end_event = captured_events[-1] if captured_events else None
        messages = list(getattr(end_event, "messages", []) or []) if end_event is not None else []

        if end_event is not None and getattr(end_event, "error", None):
            raise RuntimeError(f"Scheduled run ended with error: {end_event.error}")

        response_text = ""
        for message in reversed(messages):
            if getattr(message, "role", None) == "assistant":
                text = getattr(message, "text", "") or ""
                if text:
                    response_text = text
                    break

        context_records: list[JobContextRecord] = []
        for message in messages:
            conversation_role = _MESSAGE_ROLE_TO_CONVERSATION_ROLE.get(
                getattr(message, "role", None)
            )
            if conversation_role is None:
                continue  # CustomMessage and other plumbing entries
            context_records.append(
                JobContextRecord(
                    role=conversation_role,
                    content=getattr(message, "text", "") or "",
                    files=None,
                )
            )

        return response_text, context_records

    def _record_successful_run(self, agent_id_str: str, schedule_id_uuid: UUID) -> None:
        """Stamp schedule bookkeeping after a successful run (legacy semantics).

        ``last_run_at`` is set; one-time schedules are popped from the agent
        state and removed from APScheduler; recurring schedules past their
        ``end_time_utc`` are deactivated and removed from APScheduler.
        """
        now_utc = datetime.now(timezone.utc)
        agent_state = self.agent_registry.load_agent_state(agent_id_str)

        state_modified = False
        remove_from_scheduler = False
        for idx, sched in enumerate(list(agent_state.schedules)):
            if sched.id != schedule_id_uuid:
                continue
            sched.last_run_at = now_utc
            state_modified = True
            if sched.one_time:
                logger.debug(
                    f"One-time schedule {schedule_id_uuid} executed. " "Removing from agent state."
                )
                agent_state.schedules.pop(idx)
                remove_from_scheduler = True
            elif sched.end_time_utc and now_utc >= sched.end_time_utc and sched.is_active:
                logger.debug(
                    f"Schedule {schedule_id_uuid} passed end time "
                    f"({sched.end_time_utc}) after run. Marking inactive."
                )
                sched.is_active = False
                remove_from_scheduler = True
            break

        if state_modified:
            self.agent_registry.save_agent_state(agent_id_str, agent_state)
        if remove_from_scheduler:
            self.remove_job(schedule_id_uuid)

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    async def _push_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record the status in the job ledger and best-effort websocket broadcast."""
        try:
            await self.job_manager.update_job_status(job_id, status, result)
        except KeyError:
            logger.debug(f"Job {job_id} not found when recording status {status}.")
        except Exception:
            logger.exception(f"Failed to record job status {status} for job {job_id}.")
        await self._broadcast_status(job_id, status, result)

    async def _broadcast_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        """Best-effort broadcast; degrades gracefully if the manager shape differs."""
        manager = self.websocket_manager
        broadcast = getattr(manager, "broadcast", None)
        if not callable(broadcast):
            return
        payload: dict[str, Any] = {
            "type": "scheduled_job_status",
            "job_id": job_id,
            "status": getattr(status, "value", str(status)),
        }
        if isinstance(result, dict):
            for key in ("error", "schedule_id", "agent_id"):
                if result.get(key):
                    payload[key] = result[key]
        try:
            outcome = broadcast(job_id, payload)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:
            logger.debug(f"WebSocket broadcast for job {job_id} failed (degraded).", exc_info=True)
