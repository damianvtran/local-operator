"""Composition root for the non-CLI entry points.

The legacy version of this module built a langchain chat model, a
``LocalCodeExecutor``, a ``ToolRegistry`` and an ``Operator``, then stitched
them together. All four are gone. What survives is the *adaptation* those
callers still need: the server and the scheduler speak in legacy terms
(``operator_type``, ``current_agent``, ``request_hosting``/``request_model``,
``persist_conversation``, ``job_id``) while the rewritten engine is built by
:func:`local_operator.session_factory.create_session` from an argparse
namespace.

Two functions carry that translation:

- :func:`resolve_model_configuration` — hosting/model precedence
  (agent override > request override > config file) plus the agent's sampling
  knobs, returning the ``ModelConfiguration`` the HTTP layer still reports.
  Cheap and synchronous: no network, no session.
- :func:`initialize_operator` — the async composition root. Builds a fully
  wired harness session (tools, skills, MCP, compaction, providers) for the
  requested agent.

Providers come from ``local_operator/providers/`` via
``model/configure.py``; nothing here knows about langchain.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Optional

from local_operator.agents import AgentData, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.logger import get_logger
from local_operator.model.configure import ModelConfiguration, configure_model
from local_operator.types import OperatorType

if TYPE_CHECKING:
    # Type-only: the server facade and scheduler both import bootstrap, so
    # naming their types here must not create an import cycle at runtime.
    from local_operator.scheduler_service import SchedulerService
    from local_operator.server.utils.operator import StatusQueue
    from local_operator.session.protocol import SessionProtocol

logger = get_logger()

#: Agent record fields copied onto ``configure_model``. Names match both
#: ``AgentData`` and the ``configure_model`` keyword arguments.
AGENT_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


def resolve_hosting_model(
    config_manager: ConfigManager,
    request_hosting: Optional[str],
    request_model: Optional[str],
    current_agent: Optional[AgentData],
) -> tuple[str, str]:
    """Apply the precedence agent > request override > config file.

    Raises:
        ValueError: with the legacy message shapes when either value is
            missing, so callers keep reporting the same errors.
    """
    hosting = request_hosting or config_manager.get_config_value("hosting")
    model_name = request_model or config_manager.get_config_value("model_name")

    if current_agent is not None:
        hosting = current_agent.hosting or hosting
        model_name = current_agent.model or model_name

    if not hosting:
        raise ValueError("Hosting platform is not configured.")
    if not model_name:
        # Fall back to the provider's default model rather than erroring on a
        # single empty field — see session_factory.resolve_hosting_model, which
        # this mirrors. A provider with no known default still raises.
        from local_operator.model.defaults import default_model_for

        model_name = default_model_for(hosting)
        if not model_name:
            # Reuse the resolver's own message rather than inlining a copy: the
            # two are the SAME error (no default model for this hosting) and a
            # second literal here drifts from it the first time either is
            # reworded. Imported lazily to keep this off the session_factory
            # stack until the rare no-default path is actually hit.
            from local_operator.session_factory import (
                ModelNotConfiguredError,
                _no_model_message,
            )

            raise ModelNotConfiguredError(_no_model_message(hosting), hosting)
    return hosting, model_name


def resolve_model_configuration(
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    env_config: EnvConfig,
    request_hosting: Optional[str] = None,
    request_model: Optional[str] = None,
    current_agent: Optional[AgentData] = None,
) -> tuple[ModelConfiguration, str, str]:
    """Resolve hosting/model and build the ``ModelConfiguration``.

    Returns ``(model_configuration, hosting, model_name)``. API keys resolve
    lazily at stream time through the auth store; the configuration carries a
    best-effort static key purely for legacy reporting.

    Raises:
        ValueError: when hosting/model config is missing or the provider is
            unknown.
    """
    hosting, model_name = resolve_hosting_model(
        config_manager, request_hosting, request_model, current_agent
    )

    chat_args: dict[str, Any] = {}
    if current_agent is not None:
        for field in AGENT_SAMPLING_FIELDS:
            value = getattr(current_agent, field, None)
            if value is not None:
                chat_args[field] = value

    try:
        model_configuration = configure_model(
            hosting=hosting,
            model_name=model_name,
            credential_manager=credential_manager,
            env_config=env_config,
            **chat_args,
        )
    except Exception as exc:
        logger.error(f"Failed to configure model {model_name} on {hosting}: {exc}", exc_info=True)
        raise ValueError(f"Failed to configure model {model_name} on {hosting}: {exc}") from exc

    return model_configuration, hosting, model_name


def build_session_args(
    hosting: str,
    model_name: str,
    current_agent: Optional[AgentData] = None,
    persist_conversation: bool = False,
    yolo: bool = True,
) -> argparse.Namespace:
    """Map the legacy kwargs onto the namespace ``create_session`` reads.

    ``train`` carries ``persist_conversation``: with it the session's
    transcript IS the agent's directory (history replays at startup and the
    turn appends to it); without it the turn runs in a throwaway session
    directory, which is exactly the legacy non-persisting behaviour.

    ``yolo`` defaults to true because no server request or background job has
    anyone available to answer a tool-approval prompt.
    """
    return argparse.Namespace(
        hosting=hosting or None,
        model=model_name or None,
        agent_name=None,
        agent_id=str(current_agent.id) if current_agent is not None else None,
        yolo=yolo,
        train=bool(persist_conversation),
    )


async def initialize_operator(
    operator_type: OperatorType,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
    env_config: EnvConfig,
    # Values are Optional because a model whose family policy is to OMIT
    # carries ``None`` on its spec; the filter below drops those so an absent
    # knob is never turned into an explicit send.
    sampling_overrides: Optional[dict[str, Optional[float]]] = None,
    scheduler_service: Optional[SchedulerService] = None,
    status_queue: Optional[StatusQueue] = None,
    request_hosting: Optional[str] = None,
    request_model: Optional[str] = None,
    current_agent: Optional[AgentData] = None,
    persist_conversation: bool = False,
    auto_save_conversation: bool = False,
    job_id: Optional[str] = None,
    verbosity_level: VerbosityLevel = VerbosityLevel.VERBOSE,
) -> "SessionProtocol":
    """Build a fully wired harness session for a server or scheduled run.

    Keeps the legacy keyword surface so existing call sites need no rewrite.
    ``scheduler_service``, ``status_queue``, ``auto_save_conversation``,
    ``job_id``, ``verbosity_level`` and ``operator_type`` no longer steer the
    engine — streaming reaches the UI through the session's ``AgentEvent``
    subscription (see ``server/utils/operator.AgentEventBridge``) rather than
    through executor-held handles — and are accepted for compatibility.

    ``sampling_overrides`` (``temperature`` / ``top_p``) is the seam for
    per-request knobs that are not part of the agent record: the HTTP
    ``options`` object. Sampling rides on the ``ModelSpec``, so the override
    is applied to the constructed session via ``Session.set_model``.

    Raises:
        ValueError: when hosting/model configuration is missing or invalid.
    """
    from local_operator import session_factory

    _, hosting, model_name = resolve_model_configuration(
        config_manager,
        credential_manager,
        env_config,
        request_hosting=request_hosting,
        request_model=request_model,
        current_agent=current_agent,
    )
    logger.debug(
        f"Initializing session (Type: {operator_type.name}) with Hosting: {hosting}, "
        f"Model: {model_name}, Agent: {current_agent.name if current_agent else 'None'}"
    )

    session_args = build_session_args(
        hosting=hosting,
        model_name=model_name,
        current_agent=current_agent,
        persist_conversation=persist_conversation,
    )
    session = await session_factory.create_session(
        session_args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=False,
    )

    if sampling_overrides:
        updates = {
            key: value
            for key, value in sampling_overrides.items()
            if key in ("temperature", "top_p") and value is not None
        }
        if updates:
            # ``set_model``/``model`` are declared on SessionProtocol, so the
            # override applies straight to the live spec; sampling rides on
            # the ModelSpec and the loop re-reads it every turn.
            session.set_model(session.model.model_copy(update=updates))
    return session
