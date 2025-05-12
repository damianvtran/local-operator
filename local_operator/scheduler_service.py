import logging
from datetime import datetime, timezone
from typing import Any  # Import Any

# from typing import List # No longer needed with Python 3.9+ for List type hint
from uuid import UUID

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from local_operator.agents import AgentRegistry

# from local_operator.operator import Operator # Removed circular import
from local_operator.types import Schedule, ScheduleUnit

logger = logging.getLogger(__name__)


class SchedulerService:
    """
    Service for managing and executing scheduled tasks for agents.
    """

    # Use type hinting with a forward reference or Any to avoid circular import
    def __init__(self, agent_registry: AgentRegistry, operator: Any):
        self.agent_registry = agent_registry
        self.operator = operator
        self.scheduler = AsyncIOScheduler(timezone="UTC")

    async def _trigger_agent_task(
        self, agent_id_str: str, schedule_id_str: str, prompt: str
    ) -> None:
        """
        The actual function called by APScheduler to trigger an agent's task.
        """
        logger.info(
            f"Triggering scheduled task for agent {agent_id_str}, schedule {schedule_id_str}"
        )
        try:
            agent_id = UUID(agent_id_str)
            schedule_id = UUID(schedule_id_str)

            # This is a simplified call; Operator needs a method to handle this.
            # The Operator would typically manage loading the agent, its context,
            # and then processing the message.
            # For now, we assume operator.process_scheduled_task exists or will be created.
            await self.operator.process_message_for_agent(
                agent_id=agent_id,
                message_content=prompt,
                schedule_id=schedule_id,
            )

            # Update last_run_at for the schedule
            agent_state = self.agent_registry.load_agent_state(agent_id_str)
            schedule_updated = False
            for sched in agent_state.schedules:
                if sched.id == schedule_id:
                    sched.last_run_at = datetime.now(timezone.utc)
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
        """
        job_id = str(schedule.id)
        agent_id_str = str(schedule.agent_id)

        # Remove existing job if it exists, to ensure it's updated
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed existing job {job_id} to update schedule.")

        if not schedule.is_active:
            logger.info(f"Schedule {job_id} is not active. Not adding to scheduler.")
            return

        trigger_args = [agent_id_str, job_id, schedule.prompt]

        if schedule.anchor_time_utc and schedule.unit == ScheduleUnit.DAYS:
            hour_str, minute_str = schedule.anchor_time_utc.split(":")
            trigger = CronTrigger(
                hour=int(hour_str),
                minute=int(minute_str),
                day=f"*/{schedule.interval}" if schedule.interval > 0 else "*",
                timezone="UTC",
            )
            logger.info(
                f"Adding/updating CRON job {job_id} for agent {agent_id_str} "
                f"at {schedule.anchor_time_utc} UTC every {schedule.interval} {schedule.unit}. "
                "Next run time will be calculated by APScheduler."
            )
        else:
            interval_kwargs = {schedule.unit.value: schedule.interval}
            trigger = IntervalTrigger(**interval_kwargs, timezone="UTC")
            logger.info(
                f"Adding/updating INTERVAL job {job_id} for agent {agent_id_str} "
                f"every {schedule.interval} {schedule.unit}."
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
