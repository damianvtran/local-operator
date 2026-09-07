"""Session composition root for the CLI-facing entry points.

Why this module exists: ``cli.py`` (interactive + exec), ``exec_worker.py``
(detached background runs) and later the server facade all need the SAME
wiring from parsed args plus the three legacy managers to a harness
:class:`~local_operator.session.protocol.SessionProtocol`. Centralizing this
wiring keeps precedence rules, transcript-directory policy, lazy knowledge
integration, and the lazy-import discipline in exactly one place.

Constraints honored here (docs/REWRITE.md):

- No module-level imports of providers / session internals / semantic indexes /
  TUI. Every engine import happens inside functions, so importing this module
  is cheap and stays valid while parallel rewrite streams are mid-flight.
- Hosting/model resolution precedence: **agent > CLI flag > config file**
  (the legacy bootstrap order, minus the server-only request overrides).
- User skills and packaged guides are wired end-to-end: discovery + index build
  at session creation, first-task semantic selection, and chained
  ``skill://``/``guide://`` resolution. Any knowledge failure degrades to an
  empty listing with a warning — never a crashed startup.

``create_session`` is async: the TUI's committed factory contract is
``Callable[[], Awaitable[SessionProtocol]]`` and the eager skill-index build
needs an await. Headless callers wrap it in ``asyncio.run``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from local_operator.ansi import sanitize_prompt_line

# Stdlib-only and tiny, like ``paths`` below: safe at module level on the
# startup path that ``test_import_graph`` guards.
from local_operator.ecosystem_instructions import (
    content_digest,
    load_ecosystem_instructions,
)
from local_operator.harness.types import AgentMessage, Message

# Imported as an alias: ``config_dir`` is a parameter/local name in other
# functions here, and a module-level import of the same spelling would read
# like one of them.
from local_operator.paths import config_dir as app_config_dir

# Pure path policy, no engine — see local_operator/resume.py for why it is
# its own module rather than living here.
from local_operator.resume import resume_dir

if TYPE_CHECKING:
    # Type-only imports: this module's whole discipline is that the heavy
    # engine, registry and provider modules load lazily inside the functions
    # that need them. Annotations are strings under ``from __future__ import
    # annotations``, so naming the real types here costs nothing at runtime.
    from local_operator.agents import AgentData, AgentRegistry
    from local_operator.compaction.api import CompactionSettings
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.harness.types import AgentTool
    from local_operator.mcp.manager import McpManager
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers.auth_store import AuthStore
    from local_operator.session.goal import GoalState
    from local_operator.session.protocol import SessionProtocol
    from local_operator.session.session import Session
    from local_operator.skills.discovery import Skill
    from local_operator.skills.index import SkillIndex
    from local_operator.variables import VariableStore

logger = logging.getLogger("local_operator.session_factory")

#: Hard cap on the operator's custom instructions, in characters (~16k tokens
#: at 4 chars/token). Generous for hand-written standing rules, small enough
#: that a file pasted over by accident cannot silently consume the context
#: window of every request — the content rides the cached prompt prefix.
MAX_USER_INSTRUCTIONS_CHARS = 64_000

#: Floor held for the selected agent's own profile prompt, so a large global
#: file cannot crowd the chosen profile out entirely. A floor and not a fixed
#: slice: whichever source is smaller than its share leaves the remainder to
#: the other, so neither is taxed for room the other never uses.
_AGENT_INSTRUCTIONS_RESERVE = 16_000

#: Floor held for imported user-scope instructions (``~/.agents/AGENTS.md``;
#: see :mod:`local_operator.ecosystem_instructions`). Its OWN share, because
#: adding a third source to a two-way split silently converts one of the two
#: existing guarantees into a shared one — the operator's file and the selected
#: profile would start evicting each other because of a file neither of them
#: knows about. Sized to match the profile's reserve: an imported file is
#: standing preference of the same kind and the same order of magnitude.
_ECOSYSTEM_INSTRUCTIONS_RESERVE = 16_000


#: Modules whose import dominates :func:`create_session`, measured rather than
#: guessed: on this machine ``mcp`` costs 443 ms and ``httpx`` 234 ms to import,
#: and the engine entries below another 195 ms between them. Everything here is
#: imported lazily by the factory or by something it calls, which is what makes
#: session construction a ~700 ms burst of import machinery.
#:
#: Third-party names sit alongside our own deliberately: the cost is theirs, and
#: naming only our modules would warm the cheap half of the problem.
_WARM_IMPORTS: tuple[str, ...] = (
    "mcp",
    "httpx",
    "httpcore",
    "truststore",
    "local_operator.compaction.api",
    "local_operator.mcp.manager",
    "local_operator.model.configure",
    "local_operator.model.discovery",
    "local_operator.providers.auth_store",
    "local_operator.session.session",
    "local_operator.skills.discovery",
)


def warm_session_imports() -> None:
    """Pay :func:`create_session`'s import cost, off whatever loop is running.

    ``create_session`` is a coroutine, but its body is one long SYNCHRONOUS
    stretch — the awaits are few and none of them yield until the imports are
    done — so a caller with a live event loop is frozen for the whole of it.
    Under the TUI that is a ~700 ms window in which no frame is painted and no
    keypress is handled: the user types the first words of their prompt into a
    screen that does not move, and the characters all appear at once when it
    unfreezes.

    Importing is CPU and file I/O, both of which drop the GIL, so running this
    in a worker thread (``await asyncio.to_thread(warm_session_imports)``)
    turns that one long stall into interleaved sub-frame ones — measured at
    16 ms worst case, against 699 ms for the unwarmed factory. The factory
    itself is unchanged: it still imports what it needs, and finds it cached.

    Never raises. An optional extra that is not installed (``mcp``) or a module
    that fails to import is the factory's problem to report, in the factory's
    own words, at the point where it actually needs it.
    """
    import importlib

    for name in _WARM_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — a warm-up must never be the failure
            logger.debug("prewarm skipped %s", name, exc_info=True)


def coerce_compaction_settings(raw: object) -> CompactionSettings | None:
    """Coerce ``values.compaction`` into a :class:`CompactionSettings` (CL-01).

    ``ConfigManager`` returns the YAML shape verbatim — a plain ``dict`` — but
    the session consumes attribute-style settings. ``None`` and already-typed
    settings pass through; a dict is validated; an invalid dict degrades to
    defaults with a warning (a bad compaction block must never block startup).

    Anything else (``compaction: some-string`` in the YAML) is out of
    contract and reads as "no block": handing the session a junk object would
    only defer the failure to the first compaction check.
    """
    if raw is None:
        return None
    from pydantic import ValidationError

    from local_operator.compaction.api import CompactionSettings

    if isinstance(raw, CompactionSettings):
        return raw
    if not isinstance(raw, dict):
        return None

    try:
        return CompactionSettings.model_validate(raw)
    except ValidationError as exc:
        print(
            f"\033[1;33mWarning: invalid 'compaction' config, using defaults: {exc}\033[0m",
            file=sys.stderr,
        )
        return CompactionSettings()


#: Sampling knobs copied from an agent record onto ``configure_model`` when
#: the agent sets them. Names match both ``AgentData`` and the committed
#: ``configure_model`` keyword arguments (stream B).
_AGENT_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
)


def resolve_agent(args: argparse.Namespace, agent_registry: AgentRegistry) -> AgentData | None:
    """Resolve the session's agent record, creating it when named.

    Mirrors the legacy ``main()`` behavior: ``--agent-id`` (exec) selects by
    id and fails loudly on a miss; ``--agent``/``--agent-name`` selects by
    name and CREATES the agent when it does not exist yet. Returns ``None``
    for the default ephemeral session.

    Lazy-imports ``AgentEditFields`` so module import never pulls in the
    agent registry's heavy dependencies.
    """
    agent_id = getattr(args, "agent_id", None)
    if agent_id:
        try:
            return agent_registry.get_agent(agent_id)
        except KeyError as exc:
            raise ValueError(f"No agent found with ID: {agent_id}") from exc

    name = getattr(args, "agent_name", None)
    if not name:
        return None
    agent = agent_registry.get_agent_by_name(name)
    if agent is not None:
        return agent

    from local_operator.agents import AgentEditFields  # lazy: heavy module

    return agent_registry.create_agent(
        AgentEditFields(
            name=name,
            security_prompt=None,
            hosting=None,
            model=None,
            description=None,
            last_message=None,
            temperature=None,
            tags=[],
            categories=[],
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
    )


class HostingNotConfiguredError(ValueError):
    """Raised when no hosting provider is resolved at all.

    A dedicated subclass (rather than a bare ``ValueError`` matched by message)
    so two callers can treat this ONE condition as "first-run setup", not
    "error": the CLI preflight lets the interactive TUI open in a setup state
    instead of dying, and the TUI's boot-failure handler shows the guided
    ``/login`` affordance rather than a red "session failed to start". It stays
    a ``ValueError`` subclass so every existing ``except ValueError`` that
    reported the legacy message shape keeps working unchanged.
    """


class HostingUnknownError(HostingNotConfiguredError):
    """Raised when hosting names a provider the registry does not own.

    A SUBCLASS of :class:`HostingNotConfiguredError`, and that is the whole
    point of the fix it belongs to. The two conditions had been treated
    asymmetrically: "nothing configured" was a guided first-run state, while
    "configured to garbage" (a typo, a hand-edited config, a provider id
    removed by an upgrade) was a fatal crash. The user's remedy is IDENTICAL in
    both cases -- ``/login`` / ``/provider`` / ``/model`` from inside the app --
    so the recoverable classification has to cover both, or a one-character typo
    in ``config.yml`` locks the user out of the only surface that can repair it.
    Reported as ``Unsupported hosting platform: anthropicxyq`` from deep inside
    ``configure_model``, it left every session dead AND every provider-switch
    command answering "session is still starting...", because ``_session``
    stayed ``None``.

    Subclassing rather than adding a sibling means the existing
    recoverable-setup handling (the CLI preflight's
    ``except HostingNotConfiguredError``, the TUI's ``isinstance`` check in
    ``_on_boot_failed``) picks this up with no change, and cannot be updated for
    one condition while forgetting the other. The two stay DISTINGUISHABLE by
    type, which is what lets each surface say "nothing configured" or
    "configured to an unknown provider 'X'" rather than one vague message
    covering both.

    Do NOT "simplify" this back into a bare ``ValueError`` at the resolver, and
    do not relax ``configure_model``'s own guard: that guard is a correct
    programming-error backstop for callers that bypass this resolver, and the
    bug was that bad CONFIG could reach it, not that it existed.

    The offending value is carried as :attr:`hosting` rather than left to be
    re-parsed out of the message text: the TUI needs to name it in phrasing of
    its own (action-first, because its splash truncates from the right), and
    scraping it back out of a sentence is how the two surfaces drift apart the
    first time either is reworded.
    """

    def __init__(self, message: str, hosting: str = "", source: str = "config") -> None:
        super().__init__(message)
        self.hosting = hosting
        #: WHERE the bad value came from: ``"config"``, ``"flag"`` (``--hosting``)
        #: or ``"agent"`` (an agent record). Carried because the in-app repair
        #: writes the CONFIG FILE, so it can only fix the config case: precedence
        #: is agent > flag > config, and a login that rewrites config while the
        #: bad value comes from argv or an agent record changes nothing the next
        #: boot will read. The UI uses this to avoid promising a repair it cannot
        #: perform — telling the user to run `/login` against a `--hosting` typo
        #: is a loop, and a wrong instruction is worse than none.
        self.source = source


class ModelNotConfiguredError(HostingNotConfiguredError):
    """Raised when hosting is a real provider but no model can be resolved for it.

    A SIBLING of :class:`HostingUnknownError` under the recoverable base, for the
    same reason that class exists: the user's remedy is reachable only from
    inside the app, so the condition has to reach the surface that offers it.
    Raised as a bare ``ValueError`` it missed the ``isinstance`` gate in the
    TUI's ``_on_boot_failed``, landed in the red "session failed to start" branch
    with ``_session`` None and ``_setup_state`` False, and left every provider
    command answering "session is still starting..." -- the exact terminal state
    this error family was introduced to remove.

    That is reachable through the app's OWN repair: logging in to a provider with
    no known default model (``alibaba-token-plan``) writes a registry-VALID
    hosting with an empty model, so the next boot arrives here. Before this
    class the user was stuck HARDER after the repair than before it -- ``/login``
    wrote nothing because hosting was now valid, and ``/model`` had no session to
    talk to.

    DISTINCT from ``HostingUnknownError`` rather than reusing it, because the
    diagnosis differs and the surfaces say so: hosting is fine here, the MODEL is
    missing, and telling a user whose provider is correct that it "is not a known
    provider" sends them to fix the one thing that is not broken. The hosting is
    carried for the same reason its sibling carries it -- so the UI can name it
    without re-parsing a sentence.

    Recoverable does NOT mean permissive: the non-interactive paths (headless
    REPL, ``exec``, non-tty) still fail fast on this in ``_preflight_hosting_model``,
    because a scripted run has no one to answer the prompt and must not limp
    along picking a model nobody chose.
    """

    def __init__(self, message: str, hosting: str = "") -> None:
        super().__init__(message)
        #: The provider that resolved fine but has no model. Unlike its sibling
        #: this needs no ``source``: hosting came from somewhere valid, and the
        #: remedy (`/model`, which writes config) is the same wherever the empty
        #: model came from.
        self.hosting = hosting


#: How the user changes a bad hosting value, per source. Keyed by
#: :attr:`HostingUnknownError.source`, because the remedy genuinely differs: only
#: the config case is fixed by `login`/`config edit`, and naming the wrong one
#: sends the user round a loop that cannot terminate.
_HOSTING_SOURCE_REMEDY = {
    "config": (
        "Set a supported one with `local-operator config edit hosting <provider>` "
        "or `local-operator login <provider>` (e.g. openai, anthropic, google); "
        "`local-operator provider` lists them all."
    ),
    "flag": (
        "It came from the --hosting flag, so correct that flag (e.g. "
        "--hosting openai); `local-operator provider` lists the supported ids."
    ),
    "agent": (
        "It came from the agent's own record, which overrides config, so update "
        "the agent's hosting; `local-operator provider` lists the supported ids."
    ),
}


def _unknown_hosting_message(hosting: str, source: str = "config") -> str:
    """Error text for a hosting id the provider registry does not know.

    Names the offending value AND the remedy, because the message this replaces
    ("Unsupported hosting platform: anthropicxyq") named only the value and left
    the user to guess what a supported one looks like. Concrete provider ids are
    inlined rather than generated from the registry, matching
    :func:`_no_model_message` directly below -- a short, stable example list
    reads better than a dump of every id, and spelling the examples out is
    already this module's convention.

    Used by the non-interactive fail-fast paths (headless REPL, ``exec``,
    non-tty). The TUI writes its own action-first phrasing for the same
    condition, because its splash line truncates from the right.
    """
    where = {
        "config": "in your configuration",
        "flag": "passed with --hosting",
        "agent": "on the agent record",
    }.get(source, "in your configuration")
    remedy = _HOSTING_SOURCE_REMEDY.get(source, _HOSTING_SOURCE_REMEDY["config"])
    return f"Hosting '{hosting}' {where} is not a known provider. {remedy}"


def _no_model_message(hosting: str) -> str:
    """Error text for a provider with no known default model.

    Names two or three concrete, current model ids so the user has something to
    type rather than a bare "model is not configured" that leaves them to guess
    the vocabulary. Kept beside the resolver, stdlib-only, so the preflight path
    stays off the model-configuration stack.
    """
    return (
        f"Model name is not configured for hosting '{hosting}', and no default "
        "is known for it. Set one with `local-operator config edit model_name "
        "<model>` or the --model flag (e.g. gpt-4o, claude-3-5-sonnet-latest, "
        "deepseek-chat)."
    )


def resolve_hosting_model(
    agent: AgentData | None, args: argparse.Namespace, config_manager: ConfigManager
) -> tuple[str, str]:
    """Apply the precedence agent > CLI flag > config file.

    Raises ``ValueError`` with the legacy message shapes when either value is
    missing, so the CLI's red-banner handler reports it exactly like before.
    The pair-only shape every existing caller expects; the composition root
    uses :func:`resolve_hosting_model_with_source` because the SOURCE decides
    whether the session later follows a ``hosting``/``model_name`` edit.
    """
    hosting, model_name, _source = resolve_hosting_model_with_source(agent, args, config_manager)
    return hosting, model_name


def resolve_hosting_model_with_source(
    agent: AgentData | None, args: argparse.Namespace, config_manager: ConfigManager
) -> tuple[str, str, str]:
    """``resolve_hosting_model`` plus WHERE the model SELECTION came from.

    The third element is ``"agent"``, ``"flag"`` or ``"config"``. The session
    stores it as ``model_source``: only a ``"config"``-sourced session switches
    when the file's default changes, because for the other two the file never
    chose the model (see ``Session._apply_config_change``).

    Keyed on EITHER field of the pair, not on the hosting alone (review round
    1, R3). Both directions have to classify as chosen: a model name from
    config under a flagged hosting is a flag-chosen run, and — the harmful
    direction — ``--model X`` with no ``--hosting`` is equally chosen, because
    ``cli.py`` registers the two flags independently and ``lop exec --model
    <pinned>`` is the spelling a user reaches for to pin a run. Keyed on
    hosting only, that run classified ``"config"`` and another pane's ``/model
    default`` moved it off the model the flag named — a file default silently
    overriding an explicit flag, which is the converse of the rule this source
    exists to enforce.

    The HOSTING's own origin stays separate and narrower, because
    :class:`HostingUnknownError` uses it to name the place the bad value came
    from ("passed with --hosting", "on the agent record"). Widening that one
    would have ``--model gpt-5`` under a config hosting report the config's
    typo as having been passed on a flag the user never typed — a repair
    instruction pointing at the wrong file.
    """
    # The SOURCE is tracked alongside the value, not just the value: the repair
    # offered when this turns out to be unusable writes the config file, which
    # is only the last of these three. See HostingUnknownError.source.
    agent_hosting: str | None = getattr(agent, "hosting", None) if agent is not None else None
    flag_hosting: str | None = getattr(args, "hosting", None)
    hosting = agent_hosting or flag_hosting or config_manager.get_config_value("hosting")
    agent_model: str | None = getattr(agent, "model", None) if agent is not None else None
    flag_model: str | None = getattr(args, "model", None)
    # WHERE THE HOSTING VALUE CAME FROM — the repair prompt's subject, so it
    # stays keyed on the hosting fields alone.
    hosting_source = "agent" if agent_hosting else "flag" if flag_hosting else "config"
    # WHAT CHOSE THE RUN — the returned source, and the one the live-config
    # rule reads. An agent profile outranks a flag outranks the file, exactly
    # as for the values: naming EITHER field is choosing.
    model_source = (
        "agent"
        if (agent_hosting or agent_model)
        else "flag" if (flag_hosting or flag_model) else "config"
    )
    model_name: str | None = (
        agent_model or flag_model or config_manager.get_config_value("model_name")
    )
    if not hosting:
        raise HostingNotConfiguredError("Hosting platform is not configured.")
    # Validate the RESOLVED hosting here, in the same preflight that already
    # catches the not-configured case, rather than letting a garbage value sail
    # through and detonate in `configure_model` deep inside boot. WHERE this is
    # detected is what makes it recoverable: this is the one point both the CLI
    # preflight and the TUI boot handler classify, so the same condition raised
    # here reaches the guided setup state while raised later it reaches the red
    # "session failed to start" this fix exists to remove.
    #
    # Checked through `get_provider_definition`, NOT a membership test against
    # provider ids: that function resolves legacy aliases (`noop` -> `test`), so
    # an id test would newly reject an alias the engine still accepts and turn a
    # working config into a setup prompt. It is also the exact lookup
    # `configure_model` performs, so this accepts precisely what the engine
    # accepts -- a preflight stricter than the engine is its own outage.
    from local_operator.providers.registry import get_provider_definition

    if get_provider_definition(hosting) is None:
        # Before the default-model lookup below: an unknown provider has no
        # default model either, so checking the model first reported the missing
        # model (a symptom) and buried the unknown provider (the cause).
        raise HostingUnknownError(
            _unknown_hosting_message(hosting, hosting_source), hosting, hosting_source
        )
    if not model_name:
        # A hosting with no model is not a dead end: every mainstream provider
        # has a reasonable default, so resolve to it rather than raising. Only
        # a provider with no known default (a custom/unregistered hosting) still
        # errors, and its message now names current models to choose from.
        from local_operator.model.defaults import default_model_for

        model_name = default_model_for(hosting)
        if not model_name:
            # RECOVERABLE, not fatal: a provider with no known default is a
            # config the user can still fix from inside the app (`/model`), and
            # this is a config the app itself writes -- `/login` into a provider
            # with no default clears the model deliberately. Raised as a plain
            # ValueError it bypassed the TUI's recoverable-error gate and became
            # the dead "session failed to start" state, which is what made the
            # sanctioned repair leave the user worse off than the corruption it
            # repaired. The message is unchanged -- it names concrete model ids,
            # and the fail-fast paths still print exactly it.
            raise ModelNotConfiguredError(_no_model_message(hosting), hosting)
    return hosting, model_name, model_source


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Render transcript entries into the LLM-visible message list.

    Thin alias over the engine's single converter
    (:func:`local_operator.session.session._default_convert_to_llm`). Two
    renderings of the same entry type is exactly what let the snapcompact
    path diverge — the host converter replayed the archive's full text while
    dropping the frames, so a compaction pass reduced nothing. One renderer,
    imported, keeps the frame replay and the entry-id passthrough in the
    request path.
    """
    from local_operator.session.session import _default_convert_to_llm

    return _default_convert_to_llm(list(messages))


def _fullscreen_app_owns_terminal() -> bool:
    """True when a Textual app currently holds the terminal.

    Reading input from stdin then is not "interactive", it is a DEADLOCK: the
    app has the terminal in raw mode and consumes every keystroke, so a thread
    parked on ``input()`` waits for a line nobody can type and the turn awaiting
    approval never resumes. Probed through Textual's own active-app context var
    (import-guarded, because the TUI is an optional extra and this module sits on
    the headless path too).
    """
    try:
        from textual.app import active_app
    except Exception:  # textual absent: nothing can own the terminal
        return False
    return active_app.get(None) is not None


def _make_request_approval(yolo: bool) -> Callable[[str, str], Awaitable[bool]]:
    """Build the tool-approval gate.

    ``--yolo`` auto-approves every tier (read/write/exec). Otherwise approval
    is an interactive y/N prompt — which can only happen on a tty; headless
    runs deny, so a background job never hangs waiting for input it will
    never get. A non-tty denial is NEVER silent (CL-04): the user must see
    why the tool was rejected and how to change it (``--yolo``).

    A full-screen front end must REPLACE this gate with its own surface
    (``SessionProtocol.set_approval_handler``); the check below is the safety
    net for the window before it does, and for a UI that forgets to. Denying is
    the only safe answer there — the alternative is the hang described in
    :func:`_fullscreen_app_owns_terminal`, which looks to the user like the
    agent froze mid-task.
    """
    if yolo:

        async def auto_approve(tool_name: str, description: str) -> bool:
            return True

        return auto_approve

    async def prompt_approval(tool_name: str, description: str) -> bool:
        if not sys.stdin.isatty():
            print(
                f"approval required but no tty; run with --yolo to auto-approve "
                f"(tool '{tool_name}')",
                file=sys.stderr,
            )
            return False
        if _fullscreen_app_owns_terminal():
            # error, not warning: reaching this branch means a front end that owns
            # the terminal did not install an approval handler, which is a wiring
            # BUG, and the user pays for it with a tool that refuses for no
            # visible reason. Named remedies so whoever reads the log can act.
            #
            # Deliberately not stderr, unlike the non-tty branch above: a stray
            # stderr write under a full-screen app paints over the frame and stays
            # there (see tests/unit/tui/test_logger_silence.py), so the CL-04
            # spelling is unavailable here. The TUI routes this file's records to
            # a rotating log, which is where this lands.
            logger.error(
                "approval for %r denied: a full-screen UI owns the terminal and "
                "installed no approval handler — install one via "
                "SessionProtocol.set_approval_handler, or run with --yolo to "
                "auto-approve every tier",
                tool_name,
            )
            return False
        try:
            # Sanitised HERE as well as at the source. This is a second
            # human-facing approval surface, it renders onto a real terminal
            # with no widget between it and the escape codes, and the cost of
            # the belt-and-braces is one function call on a path that is about
            # to block on human input anyway.
            answer = await asyncio.to_thread(
                input,
                "Allow tool '{}' ({})? [y/N] ".format(
                    sanitize_prompt_line(tool_name, limit=120),
                    sanitize_prompt_line(description),
                ),
            )
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")

    return prompt_approval


def _latest_user_query(transcript: Any) -> str:
    """Extract the skill-selection query from the transcript.

    Per-turn selection embeds the last user message plus the latest
    compaction summary (docs/REWRITE.md section C). Reads the committed
    ``Transcript.entries()`` shape (``.type``, ``.payload``); any deviation
    degrades to an empty query, which skips selection — skill selection must
    never break a session.
    """
    try:
        # Production transcripts index these at durable append time. Keep the
        # historical fallback for embedders/test stores exposing entries only.
        if hasattr(transcript, "latest_user_entry"):
            entries = [
                entry
                for entry in (transcript.latest_entry("compaction"), transcript.latest_user_entry())
                if entry is not None
            ]
        else:
            entries = transcript.entries()
    except Exception:  # noqa: BLE001 — degradation is the contract
        return ""
    user_text = ""
    summary = ""
    for entry in reversed(entries):
        entry_type = getattr(entry, "type", None)
        payload = getattr(entry, "payload", None) or {}
        if not summary and entry_type == "compaction":
            # A snapcompact entry's summary is reading instructions for the
            # archive frames, not conversation content — as a selection query
            # it is constant boilerplate that would drown the user's actual
            # words. The archive's text_tail is the newest slice of the real
            # transcript, so prefer it (bounded: selection wants a signal, not
            # the whole edge).
            preserve = payload.get("preserve_data") or {}
            snap = preserve.get("snapcompact") if isinstance(preserve, dict) else None
            # Prefer text_tail, then text_head: a small archive stores ALL its
            # text in text_head with an empty tail, and falling straight
            # through to the summary there re-created the boilerplate-noise
            # defect for exactly the sessions with the least other signal.
            edge = ""
            if isinstance(snap, dict):
                for key in ("text_tail", "text_head"):
                    candidate = snap.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        edge = candidate.strip()[-2000:]
                        break
            summary = edge or str(payload.get("summary", "")).strip()
        if not user_text and entry_type == "message" and payload.get("role") == "user":
            content = payload.get("content") or []
            user_text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()
        if user_text and summary:
            break
    return "\n".join(part for part in (user_text, summary) if part)


def _latest_compaction_id(transcript: Any) -> str | None:
    """Entry id of the newest compaction marker, or ``None`` without one.

    This is the freeze key for the knowledge block: selection normally
    freezes after the first query so the prompt-cache prefix stays warm, but
    a compaction rewrites the transcript head anyway — the cache is already
    invalidated — so a NEW id here licenses one re-selection (see
    :func:`_select_knowledge_block`). Same contract as
    :func:`_latest_user_query`: any transcript deviation degrades to
    ``None`` (treated as "no compaction yet"), never breaks the turn.
    """
    try:
        if hasattr(transcript, "latest_entry"):
            entry = transcript.latest_entry("compaction")
            return entry.id if entry is not None else None
        entries = transcript.entries()
    except Exception:  # noqa: BLE001 — degradation is the contract
        return None
    for entry in reversed(entries):
        if getattr(entry, "type", None) == "compaction":
            entry_id = getattr(entry, "id", None)
            return str(entry_id) if entry_id else None
    return None


def _env_details(cwd: str | None = None) -> str:
    """Volatile environment facts for the env block (date rides there too,
    added by ``build_system_blocks``). Kept tiny and byte-stable within a
    run: no timestamps, no process ids. ``cwd`` comes from the session's
    working directory, never the process-global value at call time."""
    import platform

    return (
        f"Platform: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"Python: {platform.python_version()}\n"
        f"Working directory: {cwd if cwd is not None else os.getcwd()}"
    )


def load_user_instructions(agent_prompt: str = "") -> str:
    """Read the operator's standing custom instructions for the system prompt.

    Source of truth is ``<config_dir>/system_prompt.md`` — the same file the
    desktop UI's Settings "Instructions" box and the
    ``/v1/config/system-prompt`` endpoint write, so the three surfaces cannot
    drift into separate notions of "custom instructions".

    ``agent_prompt`` is the selected agent profile's own ``system_prompt.md``.
    It is appended rather than allowed to replace the global file: an agent is
    a specialization ("you review Python"), not a reason to forget the
    operator's machine-wide preferences, and a profile that genuinely must
    override one can say so in its own text.

    Instructions shared with other agent tools (``~/.agents/AGENTS.md``, see
    :mod:`local_operator.ecosystem_instructions`) are read too, PREPENDED so
    the operator's own file is read last and wins on conflict, and skipped
    entirely when their content is identical to ``system_prompt.md`` — the
    common case for anyone who currently generates the native file from the
    shared one with a sync script. Those files are never written by lop, so
    ``system_prompt.md`` remains the single write target of Settings →
    Instructions and ``GET``/``PATCH /v1/config/system-prompt``.

    Failures degrade instead of breaking startup: an unreadable file is
    skipped, and undecodable bytes are REPLACED rather than dropping the whole
    file, because a stray bad byte in a long instructions file should cost the
    operator one glyph and not every preference they wrote. Either way a bad
    edit never costs a session.

    The result is bounded at :data:`MAX_USER_INSTRUCTIONS_CHARS`. This rides
    the CACHED head block, so it is re-sent as the prefix of every request in
    every session and every subagent: an accidentally huge file (a log pasted
    over the wrong path) would otherwise cost context and money on every call,
    and on a small-context model would fail the session at startup with
    nothing pointing at the cause. Truncation is explicit — the marker tells
    the model its instructions were cut rather than letting it act on half a
    rule — and a warning names the source and the limit.

    ``utf-8-sig`` strips a BOM that a Windows editor writes; without it the
    ``\ufeff`` survives into the prompt ahead of the first rule.
    """
    parts: list[str] = []
    # ``is_file()`` follows symlinks deliberately: pointing the file at a
    # dotfiles checkout is a normal way to version instructions.
    path = app_config_dir() / "system_prompt.md"
    try:
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        pass

    # Each source is bounded on its OWN budget before joining. Capping only
    # the joined string let a full-size global file consume the whole budget
    # and silently discard the selected agent's profile prompt entirely,
    # inverting the documented layering: the machine-wide file would discard
    # the profile the operator explicitly chose.
    #
    # The split is a FLOOR each way, never a flat tax. Subtracting the
    # reserve unconditionally cut a 64k global file to 48k even with no agent
    # selected, handing the 16k to nobody; capping the profile at the reserve
    # unconditionally did the mirror image to a large profile when the global
    # file was small. So each source may spend whatever the other leaves,
    # down to its own guaranteed share.
    global_raw = "\n\n".join(part.strip() for part in parts if part.strip())
    agent_raw = agent_prompt.strip()
    # The digest is handed down so a shared file byte-identical to the native
    # one is dropped rather than duplicated into every cached request.
    ecosystem_raw = load_ecosystem_instructions(
        skip_digests=frozenset({content_digest(global_raw)} if global_raw else ())
    ).strip()

    # The "\n\n" joins are only emitted between sources that SURVIVE, so the
    # characters are only withheld then. Keyed off the agent text alone, a
    # profile that fits the documented cap exactly was truncated by two
    # characters while a global file of the same size passed whole.
    present = sum(1 for raw in (ecosystem_raw, global_raw, agent_raw) if raw)
    separator = 2 * max(0, present - 1)
    # Bounded in ascending order of ownership: the imported file first, then
    # the profile, and the operator's own file takes the remainder. Each may
    # spend what the others leave, down to its own floor — so a lone 64k source
    # is still whole, and no source is taxed for room the others never use.
    ecosystem_text = _bound_instructions(
        ecosystem_raw,
        "imported user-scope instructions",
        max(
            _ECOSYSTEM_INSTRUCTIONS_RESERVE,
            MAX_USER_INSTRUCTIONS_CHARS - len(global_raw) - len(agent_raw) - separator,
        ),
    )
    agent_text = _bound_instructions(
        agent_raw,
        "the selected agent's profile",
        max(
            _AGENT_INSTRUCTIONS_RESERVE,
            MAX_USER_INSTRUCTIONS_CHARS - len(ecosystem_text) - len(global_raw) - separator,
        ),
    )
    global_text = _bound_instructions(
        global_raw,
        str(path),
        MAX_USER_INSTRUCTIONS_CHARS - len(ecosystem_text) - len(agent_text) - separator,
    )
    # Imported first, native second, profile last: later text is read as the
    # more specific instruction, so lop's own file outranks the shared one and
    # the chosen profile outranks both.
    return "\n\n".join(part for part in (ecosystem_text, global_text, agent_text) if part)


def _bound_instructions(text: str, source: str, limit: int) -> str:
    """Cap one instruction source so it cannot silently eat the context window.

    ``source`` names the origin in the warning: passing the global path for
    text that came from an agent profile would send the operator looking for
    a file that may not even exist. The marker is counted INSIDE ``limit``, so
    the return value never exceeds it — including when ``limit`` is too small
    to hold the marker at all, where the marker is dropped rather than
    appended past the budget.
    """
    if len(text) <= limit:
        return text
    marker = f"\n\n[... custom instructions truncated at {limit} characters ...]"
    if limit < len(marker):
        marker = ""
    logger.warning(
        "custom instructions from %s are %d chars; truncating to %d "
        "(they are re-sent with every request)",
        source,
        len(text),
        limit,
    )
    return text[: max(0, limit - len(marker))].rstrip() + marker


def _build_variable_store(cwd: str, config_manager: ConfigManager) -> VariableStore:
    """Construct the session's VariableStore for the list/read variable
    tools. Config ``variables`` ride above the project file and environment;
    no values are ever written into the system prompt (that is the whole
    point — the model lists names and reads single values on demand)."""
    from local_operator.variables import VariableStore

    config_values: dict[str, str] | None = None
    try:
        raw = config_manager.get_config_value("variables", None)
        if isinstance(raw, dict):
            config_values = {str(k): str(v) for k, v in raw.items() if v is not None}
    except Exception:  # noqa: BLE001 — a config read failure must not block tools
        config_values = None
    return VariableStore(cwd=cwd, config_values=config_values)


@dataclass
class _KnowledgeHooks:
    """Session-owned semantic knowledge and progressive-disclosure resolvers.

    User skills and packaged guides share one index. Registered agent metadata
    gets a separate local-only index: it can select the generic agents guide,
    but names and descriptions never enter the prompt or a remote embedding
    request. Selection is reused within each task and refreshed on the next user row.
    """

    index: SkillIndex | None = None
    agent_hint_index: SkillIndex | None = None
    skills_by_name: dict[str, Skill] = field(default_factory=dict)
    guides_by_name: dict[str, Skill] = field(default_factory=dict)
    frozen_block: str | None = None
    #: Compaction entry id observed when ``frozen_block`` was computed. A
    #: change here (a new compaction marker) re-opens selection once — the
    #: transcript head is being rewritten anyway, so the prompt cache the
    #: freeze protects is already gone.
    frozen_compaction_id: str | None = None
    # A new admitted user row is a task boundary, unlike tool continuations.
    # Selection updates enter history as host state, so refreshing here no
    # longer rewrites the historical system prefix.
    frozen_task_id: str | None = None
    mcp_resolver: Callable[[str], str | None] | None = None
    # Takes the frozen selection query. Configured names are populated before
    # deferred connection work begins, closing the first-turn race without
    # making connection completion part of the prompt-cache key.
    mcp_catalogue: Callable[[str], str] | None = None


def _registered_agent_hints(agent_registry: AgentRegistry) -> list[Skill]:
    """Build bounded, local-only routing rows from meaningful agent metadata.

    A registry can grow indefinitely, and descriptions are user content. Each
    row is capped before hashing/embedding and the first 512 deterministic rows
    are used. Empty autosave-style profiles provide no routing signal and are
    skipped. The rows are never rendered by ``render_block``.
    """
    from local_operator.skills.discovery import Skill

    try:
        agents = sorted(
            agent_registry.list_agents(),
            key=lambda agent: (str(agent.name).lower(), str(agent.name), str(agent.id)),
        )[:512]
    except Exception:  # noqa: BLE001 — hints are optional enrichment
        return []

    hints: list[Skill] = []
    agents_dir = Path(agent_registry.config_dir) / "agents"
    for agent in agents:
        semantic_parts = [
            sanitize_prompt_line(str(agent.description or "")),
            " ".join(sanitize_prompt_line(str(tag)) for tag in (agent.tags or [])),
            " ".join(sanitize_prompt_line(str(category)) for category in (agent.categories or [])),
        ]
        semantic = " ".join(part for part in semantic_parts if part).strip()
        if not semantic:
            continue
        agent_dir = agents_dir / str(agent.id)
        hints.append(
            Skill(
                name=f"registered-agent-{agent.id}",
                description=f"{sanitize_prompt_line(str(agent.name))}: {semantic}"[:512],
                file_path=agent_dir / "agent.yml",
                base_dir=agent_dir,
                source=str(agents_dir),
                resource_type="agent_hint",
            )
        )
    return hints


def _seed_mcp_routing(hooks: _KnowledgeHooks, cwd: str) -> None:
    """Expose configured names before deferred live connections can race turn one."""
    try:
        from local_operator.mcp.config import load_all_mcp_configs
        from local_operator.mcp.resources import render_mcp_suggestions

        names = tuple(load_all_mcp_configs(cwd)[0])
        hooks.mcp_catalogue = lambda query: render_mcp_suggestions(names, query)
    except Exception:  # noqa: BLE001 — MCP hints remain optional enrichment
        logger.debug("early MCP name discovery failed", exc_info=True)


async def _setup_knowledge(
    credential_manager: CredentialManager,
    config_dir: Path,
    agent_registry: AgentRegistry,
    warnings_out: list[str],
) -> _KnowledgeHooks:
    """Discover and index user skills, packaged guides, and private agent hints.

    Guide bodies are release resources and therefore exist in every install;
    only their short descriptions join the ordinary skill descriptions sent to
    the configured semantic backend. Agent hints always use ``LocalEmbedder``.
    Any layer can fail independently without making session startup fail.
    """
    hooks = _KnowledgeHooks()
    try:
        from local_operator.guides import discover_guides
        from local_operator.skills.api import (
            SkillIndex,
            default_backend_from_env,
            default_skill_roots,
            discover_skills,
        )
        from local_operator.skills.embeddings import LocalEmbedder

        skills, discovery_warnings = discover_skills(default_skill_roots(Path.cwd()))
        warnings_out.extend(discovery_warnings)
        guides = discover_guides()
        hooks.skills_by_name = {skill.name: skill for skill in skills}
        hooks.guides_by_name = {guide.name: guide for guide in guides}
        resources = sorted(
            [*skills, *guides],
            key=lambda item: (
                item.resource_type,
                item.name.lower(),
                item.name,
                str(item.file_path),
            ),
        )

        def get_credential(key: str) -> str | None:
            secret = credential_manager.get_credential(key)
            value = secret.get_secret_value() if secret else ""
            return value or None

        if resources:
            backend = default_backend_from_env(get_credential)
            try:
                hooks.index = SkillIndex(resources, backend, cache_dir=config_dir / "cache")
                await hooks.index.build()
                warnings_out.extend(hooks.index.warnings)
            except Exception as exc:  # noqa: BLE001 — direct reads still work
                hooks.index = None
                if not isinstance(backend, LocalEmbedder):
                    warnings_out.append(
                        f"Knowledge embedding backend failed; using local routing: {exc}"
                    )
                    try:
                        hooks.index = SkillIndex(
                            resources,
                            LocalEmbedder(),
                            cache_dir=config_dir / "cache",
                        )
                        await hooks.index.build()
                    except Exception as fallback_exc:  # noqa: BLE001
                        warnings_out.append(
                            "Knowledge selection unavailable, continuing without routing: "
                            f"{fallback_exc}"
                        )
                        hooks.index = None
                else:
                    warnings_out.append(
                        f"Knowledge selection unavailable, continuing without routing: {exc}"
                    )

        agent_hints = _registered_agent_hints(agent_registry)
        if agent_hints and "agents" in hooks.guides_by_name:
            try:
                hooks.agent_hint_index = SkillIndex(
                    agent_hints,
                    LocalEmbedder(),
                    cache_dir=config_dir / "cache",
                )
                await hooks.agent_hint_index.build()
            except Exception as exc:  # noqa: BLE001 — generic guide routing remains
                warnings_out.append(f"Registered-agent semantic hints unavailable: {exc}")
                hooks.agent_hint_index = None
    except Exception as exc:  # noqa: BLE001 — knowledge is optional enrichment
        warnings_out.append(f"Knowledge guides unavailable, continuing without them: {exc}")
        hooks = _KnowledgeHooks()
    return hooks


async def _select_knowledge_block(
    hooks: _KnowledgeHooks,
    query: str,
    *,
    compaction_id: str | None = None,
    cwd: str | None = None,
    task_id: str | None = None,
) -> str:
    """Reuse routing within a task, refresh for a new user or compaction row.

    ``task_id`` is an admitted USER message identity, never a tool-step ID, so
    long tool loops pay one selection while a new task can discover different
    guidance. Production Session appends changes after history instead of
    changing the system prefix. Legacy callers omitting task_id retain their
    compaction-only selection contract. ``cwd`` supports skill globs.
    """
    if (
        hooks.frozen_block is not None
        and hooks.frozen_compaction_id == compaction_id
        and hooks.frozen_task_id == task_id
    ):
        return hooks.frozen_block

    picked: list[Skill] = []
    query = query.strip()
    if query:
        # cwd rides as a keyword ONLY when set: test doubles and alternate
        # index shapes in the wild implement ``select(query, ...)`` with the
        # historical signature, and there is no globs matching to do without
        # a cwd anyway.
        select_kwargs: dict[str, Any] = {"cwd": Path(cwd)} if cwd else {}
        picked = await hooks.index.select(query, **select_kwargs) if hooks.index is not None else []
        if hooks.agent_hint_index is not None:
            matching_agents = await hooks.agent_hint_index.select(query, k=1)
            agents_guide = hooks.guides_by_name.get("agents")
            if matching_agents and agents_guide is not None and agents_guide not in picked:
                picked.append(agents_guide)
                picked.sort(
                    key=lambda item: (
                        item.resource_type,
                        item.name.lower(),
                        item.name,
                        str(item.file_path),
                    )
                )

    from local_operator.skills.api import render_block

    sections = [section for section in [render_block(picked)] if section]
    if hooks.mcp_catalogue is not None:
        catalogue = hooks.mcp_catalogue(query)
        if catalogue:
            sections.append(catalogue)
    hooks.frozen_block = "\n\n".join(sections)
    hooks.frozen_compaction_id = compaction_id
    hooks.frozen_task_id = task_id
    return hooks.frozen_block


def _make_knowledge_resolver(hooks: _KnowledgeHooks) -> Callable[[str], str | None]:
    """Chain lazy knowledge protocols without mixing their namespaces."""
    from local_operator.guides import make_guide_resolver
    from local_operator.skills.api import make_skill_resolver

    guide_resolver = make_guide_resolver(hooks.guides_by_name)
    skill_resolver = make_skill_resolver(hooks.skills_by_name)

    def resolver(url: str) -> str | None:
        guide_result = guide_resolver(url)
        if guide_result is not None:
            return guide_result
        skill_result = skill_resolver(url)
        if skill_result is not None:
            return skill_result
        if hooks.mcp_resolver is not None:
            return hooks.mcp_resolver(url)
        return None

    return resolver


@dataclass
class _SessionPlan:
    """Everything needed to construct the session, split out so
    ``build_initial_blocks`` can render the startup system prompt without
    instantiating the facade (benchmark hook, orchestrator duty).

    ``auth_store`` rides along (CL-08): callers own its lifetime — folded
    into ``session.dispose`` by :func:`create_session`, closed directly by
    :func:`build_initial_blocks` (which never constructs a session).
    """

    session_kwargs: dict[str, Any]
    system_blocks_provider: Callable[..., Awaitable[list[str]]]
    knowledge_hooks: _KnowledgeHooks
    auth_store: AuthStore | None = None
    # Acquired before transcript construction and transferred to Session.dispose;
    # benchmark-only preparation releases it directly because no Session exists.
    session_lease: Any | None = None


def _make_system_blocks_provider(
    tools: list[AgentTool],
    transcript: Any,
    hooks: _KnowledgeHooks,
    cwd: str | None = None,
    goal_state: "GoalState | None" = None,
    user_instructions: str = "",
    repo_guidance: str = "",
    variable_store: "VariableStore | None" = None,
) -> Callable[..., Awaitable[list[str]]]:
    """Build the per-turn system-prompt closure.

    Semantic routing refreshes only at admitted user/compaction boundaries.
    Unchanged desired blocks reuse a small immutable snapshot. Session persists
    its initial prefix and journals later state changes at the conversation
    tail; inspecting this closure directly remains read-only.

    ``goal_state`` is the SAME holder the session facade exposes through
    ``set_goal``, which is how a ``/goal`` edit reaches the next model step's
    prompt without rebuilding the session. ``variable_store`` is the same
    store the session injects into every tool context, so a ``/credential``
    store reaches the next turn's ``<session-credentials>`` block the same
    way.

    ``user_instructions`` is captured once by the caller and closed over
    rather than re-read here: it lands in the byte-stable head block, so
    re-reading the file per turn would let a mid-session edit silently
    invalidate the whole cached prefix. Editing the file takes effect on
    the next session, which is also what makes a session's prompt reproducible.
    """

    environment = _env_details(cwd)
    cached_key: tuple[Any, ...] | None = None
    cached_blocks: list[str] = []

    async def provider(model_label: str = "") -> list[str]:
        nonlocal cached_key, cached_blocks
        # ``model_label`` is passed live by the Session on each provider step, so
        # a deliberate ``set_model`` or a failover fallback is reflected in the
        # env block at the next safe call boundary without rebuilding this
        # closure. The benchmark/preflight caller passes the spec label directly.
        from local_operator.prompts_api import build_system_blocks

        task = transcript.latest_user_entry() if hasattr(transcript, "latest_user_entry") else None
        task_id = task.id if task is not None else None
        compaction_id = _latest_compaction_id(transcript)
        unchanged = (
            hooks.frozen_block is not None
            and hooks.frozen_task_id == task_id
            and hooks.frozen_compaction_id == compaction_id
        )
        query = "" if unchanged else _latest_user_query(transcript)
        try:
            knowledge_block = await _select_knowledge_block(
                hooks,
                query,
                compaction_id=compaction_id,
                cwd=cwd,
                task_id=task_id,
            )
        except Exception:  # noqa: BLE001 — never break the turn
            knowledge_block = ""
        date_str = datetime.now().strftime("%Y-%m-%d")
        goal = goal_state.text if goal_state is not None else ""
        team_brief = goal_state.team_brief if goal_state is not None else ""
        agent_brief = goal_state.agent_brief if goal_state is not None else ""
        names = (
            variable_store.credential_names()
            if variable_store is not None and hasattr(variable_store, "credential_names")
            else []
        )
        interactive = goal_state.is_interactive() if goal_state is not None else True
        key = (
            knowledge_block,
            date_str,
            goal,
            team_brief,
            agent_brief,
            tuple(names),
            model_label,
            interactive,
            tuple((tool.name, tool.description) for tool in tools),
        )
        if key == cached_key:
            return list(cached_blocks)
        cached_blocks = build_system_blocks(
            tools,
            knowledge_block,
            environment,
            date_str,
            goal=goal,
            user_instructions=user_instructions,
            repo_guidance=repo_guidance,
            credentials=names,
            team_brief=team_brief,
            agent_brief=agent_brief,
            model_label=model_label,
            # Read LIVE, at turn start: the answer changes whenever a viewer
            # attaches or detaches, and reading it here is what keeps the
            # cost O(1) in the number of those events (round 2, operator
            # requirement 4). Absent a probe this is True, so every host that
            # is not a detached runtime is unaffected.
            interactive=interactive,
        )
        cached_key = key
        return list(cached_blocks)

    # A host-supplied arbitrary block provider retains its historical dynamic
    # semantics. Production providers opt into Session's persisted-prefix
    # protocol explicitly; benchmarks can still call this builder read-only.
    setattr(provider, "append_only_state", True)
    setattr(provider, "repo_guidance", repo_guidance)
    setattr(provider, "knowledge_hooks", hooks)
    return provider


def _transcript_dir_and_agent_id(
    agent: AgentData | None, args: argparse.Namespace, agent_registry: AgentRegistry
) -> tuple[Path, str]:
    """Pick where this session's JSONL transcript lives (CL-02).

    ``--resume <id>`` wins over every rule below: it names an existing session
    directory, and reusing it is what makes the transcript replay (the same
    mechanism ``--train`` uses for an agent directory).

    Legacy ``--train`` semantics:

    - named agent + ``--train`` -> the agent's own directory, so history is
      replayed at startup and appended after each turn;
    - named agent WITHOUT ``--train`` -> an ephemeral per-session directory:
      history is neither replayed from nor appended to the agent dir;
    - no agent but ``--train`` -> the registry's autosave agent (legacy
      ``create_autosave_agent`` semantics);
    - otherwise an ephemeral per-session directory under ``sessions/``: the
      default agent must not persist its session.
    """
    config_dir = Path(agent_registry.config_dir)
    resume = getattr(args, "resume", None)
    # `is not None`, not truthiness: `--resume ""` is a user error and must be
    # refused, where silently starting a NEW session would look like a resume
    # that lost the history.
    if resume is not None:
        # ADOPT vs RESUME. Under the viewer model the session id is minted in
        # the TUI before anything exists on disk: `lop` opens a viewer bound
        # to nothing, and the runtime the first message engages is what
        # materialises the directory. So a runtime asked for an id with no
        # directory is not a failed resume — it is the FIRST engage of a
        # session that has only ever been a name, and refusing it (as
        # `resume_dir` must, for a human typing `--resume`) would make every
        # new detached session fail to start.
        #
        # Gated on the runtime's own env flag rather than applied generally,
        # because the strictness is load-bearing everywhere else: a human's
        # `--resume typo` must still say "no session to resume" rather than
        # silently opening an empty conversation under that name.
        if os.environ.get("LOP_RUNTIME_ADOPT_SESSION") == "1":
            requested = str(resume)
            # The adopt branch relaxes the "must already exist" rule, NOT the
            # "must be one path component" rule. `resume_dir` enforces both
            # together, and dropping the second along with the first let
            # `../../escape` resolve outside `sessions/` (round 1, R3). Not
            # user-reachable today — `cli.py` runs `resolve_resume_id` first
            # and viewer ids are `uuid4().hex[:12]` — but the remaining
            # feeders (the wake index, the supervisor's cwd) are derived from
            # filenames, and this is the one branch that opted out of a
            # strictness the comment above calls load-bearing.
            if requested in ("", ".", "..") or Path(requested).name != requested:
                raise ValueError(f"not a session id: {requested!r}")
            adopted = config_dir / "sessions" / requested
            # Under `LOP_RUNTIME_DEFER_MATERIALISE` the directory is NOT
            # created here: a speculative warm engage (a viewer's first
            # keystroke, before the user has committed to a message) must
            # leave nothing on disk when the draft is abandoned. The first
            # real write materialises it — see `Transcript.__init__`.
            defer = os.environ.get("LOP_RUNTIME_DEFER_MATERIALISE") == "1"
            if not adopted.exists() and not defer:
                # `parents` because a fresh config dir has no sessions/ yet;
                # `exist_ok` because two contenders may race here and the
                # lease, not this mkdir, is what arbitrates between them.
                adopted.mkdir(parents=True, exist_ok=True)
            return adopted, str(agent.id) if agent is not None else "main"
        resumed = resume_dir(config_dir, str(resume))
        return resumed, str(agent.id) if agent is not None else "main"
    train = bool(getattr(args, "train", False))
    if agent is not None:
        agent_id = str(agent.id)
        if train:
            return config_dir / "agents" / agent_id, agent_id
        session_dir = uuid.uuid4().hex[:12]
        return config_dir / "sessions" / session_dir, agent_id
    if train:
        try:
            autosave = agent_registry.create_autosave_agent()
            agent_id = str(autosave.id)
            return config_dir / "agents" / agent_id, agent_id
        except Exception:  # noqa: BLE001 — fall through to ephemeral
            pass
    session_dir = uuid.uuid4().hex[:12]
    return config_dir / "sessions" / session_dir, "main"


#: The one store-maintenance pass this process will run, or ``None`` before the
#: first session is constructed. Store maintenance is a property of the STORE,
#: not of a session, so it is scoped to the process rather than to the call:
#: ``/new`` and ``/resume`` go through ``create_session`` exactly as boot does,
#: and re-sweeping a store this same process swept seconds earlier is pure
#: latency. Holding the task (not just a bool) also keeps a reference to it, so
#: the loop cannot garbage-collect a task nobody awaits.
_STORE_MAINTENANCE_TASK: "asyncio.Task[None] | None" = None

#: Give session construction and the front end a bounded uncontended window
#: before four whole-store walks enter the worker pool. A single ``sleep(0)``
#: only yields to ``_prepare``'s next ``to_thread`` and lets both paths race on a
#: cold filesystem cache; the elapsed delay is the contention barrier.
_STORE_MAINTENANCE_IDLE_DELAY_SECONDS = 0.75


async def _wait_for_store_maintenance_idle_window() -> None:
    """Wait until first paint can win the disk/thread-pool contention race."""
    await asyncio.sleep(_STORE_MAINTENANCE_IDLE_DELAY_SECONDS)


def reset_store_maintenance_for_tests() -> None:
    """Forget that maintenance ran, so every test starts un-swept.

    The once-per-process guard is deliberate production behaviour, but in a
    test interpreter it means the first test to call ``_prepare`` consumes the
    process's single pass and every later one silently exercises a no-op.
    Resetting around EACH test — the autouse fixture in ``tests/conftest.py``
    calls this before and after — is what lets any test assert on the sweeps'
    effects regardless of the order it happens to run in.
    """
    global _STORE_MAINTENANCE_TASK
    task = _STORE_MAINTENANCE_TASK
    if task is not None and not task.done():
        # Most session-factory tests return before the production idle window;
        # do not let their delayed task enter a later test's temporary store.
        task.cancel()
    _STORE_MAINTENANCE_TASK = None


async def await_store_maintenance_for_tests() -> None:
    """Wait for this process's maintenance pass, if one was dispatched.

    Production never waits — that is the entire point of the change — so this
    exists for tests that assert on what the passes DID (a sidecar stamped, a
    dead process group reaped). Without it such a test races the background task and
    fails intermittently, which is a worse outcome than the latency it is
    guarding. Swallows the task's failure because every pass is best-effort:
    the caller is asserting on effects, not on the task's success.
    """
    task = _STORE_MAINTENANCE_TASK
    if task is None:
        return
    try:
        await task
    except Exception:  # noqa: BLE001 — best-effort, exactly as in production
        pass


async def _run_store_maintenance(
    config_manager: ConfigManager, config_dir: Path, live_dir: Path | None
) -> None:
    """Run every whole-store maintenance pass, in a worker thread, in order.

    The initial idle window is load-bearing rather than cosmetic. Merely putting
    this coroutine in the background still lets it begin at ``_prepare``'s next
    ``to_thread`` and contend with model/session construction on a cold
    filesystem cache. Maintenance is unrelated to the current session, so it
    yields a short, explicit window for ``create_session`` to return and its
    caller to paint before the first disk walk reaches the worker pool.

    Each pass is a disk walk over OTHER sessions' directories and none of them
    has anything to do with the session being constructed; they are triggered by
    a session starting only because that is when the store is known to be quiet.
    They run sequentially rather than gathered because they share one disk and
    the origin/title backfills walk the same directories — the win here came
    from taking them OFF the critical path, not from overlapping them, and
    serial keeps the I/O pattern (and the failure attribution) simple.

    Every pass is best-effort in the strongest sense: this coroutine can fail in
    any way at all and a session must neither fail nor be delayed by it, which
    is why the caller never awaits it and why each pass carries its own guard.
    """
    # Delay before imports and callback construction too: on a cold cache even
    # loading maintenance-only modules can steal I/O from session construction.
    # Exit during this best-effort window is harmless; the next process retries.
    await _wait_for_store_maintenance_idle_window()

    from local_operator.analytics.backfill import backfill_analytics_session_names
    from local_operator.resume import backfill_session_origins, backfill_session_titles
    from local_operator.session.cleanup import cleanup_from_config
    from local_operator.tools.group_reaper import sweep_orphan_groups

    # NO pass here deletes a session directory on its own judgement. The
    # "retention sweep" that used to lead this list — and the unused-session
    # reaper it grew in #576 — removed 225 of an operator's 244 named sessions
    # in one night, behind an opt-out toggle that wrote a key nothing read.
    # The only thing that can remove a session directory now is the cleanup
    # policy in ``session/cleanup.py``, which is OFF unless the user turned
    # ``session.cleanup.enabled`` on in /settings; ``cleanup_from_config``
    # returns without touching the disk otherwise. ``live_dir`` is passed so
    # that even an enabled policy never considers the session being built.
    passes: list[tuple[str, Callable[[], Any]]] = [
        (
            "session cleanup policy",
            lambda: cleanup_from_config(config_manager, config_dir, live_dir=live_dir),
        ),
        # Hard-death process-group reaper (tools/group_reaper.py): reaps a bash
        # process group only when the lop process that spawned it is provably
        # dead — the one leak _kill() cannot cover, because a SIGKILLed owner
        # runs no in-process cleanup and start_new_session already stripped the
        # group's SIGHUP. Owner liveness is the ONLY signal, so a live session's
        # long command (e.g. a 10h trainer) is never touched.
        ("orphan process-group sweep", lambda: sweep_orphan_groups(config_dir)),
        # Stamp session directories that predate the origin marker, so the
        # ``/resume`` picker stops offering delegated runs on the FIRST launch
        # after an upgrade rather than once natural churn has cleared the store.
        ("session origin backfill", lambda: backfill_session_origins(config_dir)),
        # Stamp the title sidecar alongside the origin marker, so a pre-existing
        # session is findable by every name it has borne on the first launch
        # after upgrade rather than only after its next rename.
        ("session title backfill", lambda: backfill_session_titles(config_dir)),
        # Name the analytics ledger's unnamed sessions from their transcripts,
        # so ``/analytics`` stops rendering months of history as bare 12-hex
        # ids. Ordered AFTER the title backfill on purpose: that pass writes the
        # title sidecar this one reads through ``resume.session_name``, so a
        # session whose title sits in the untouched middle of a large transcript
        # is recovered on the same launch rather than the next one.
        (
            "analytics session-name backfill",
            lambda: backfill_analytics_session_names(config_dir),
        ),
    ]

    for label, work in passes:
        try:
            await asyncio.to_thread(work)
        except asyncio.CancelledError:
            # The process is shutting down mid-pass. Every pass is idempotent
            # and re-runs on the next launch, so stopping here loses nothing.
            raise
        except Exception:  # noqa: BLE001 — best-effort; never disturb a session
            # Debug, not warning: this is unattended housekeeping the user did
            # not ask for, and a store that cannot be swept is not a problem the
            # user can act on mid-session. Same level these carried when they
            # ran inline.
            logger.debug("%s failed", label, exc_info=True)


def _start_store_maintenance(
    config_manager: ConfigManager, config_dir: Path, live_dir: Path | None
) -> None:
    """Dispatch store maintenance ONCE per process, without blocking the caller.

    The four passes were previously awaited inline in ``_prepare``. They were
    already ``to_thread``'d, so the event loop was never blocked — but awaiting
    them kept them on the session-construction CRITICAL PATH, where they cost
    boot ~545 ms and, because ``/new`` and ``/resume`` re-enter the same
    ``create_session``, cost EVERY ``/resume`` the same again on a store the
    process had already swept. Measured on a 3574-session store, the sweeps were
    77% of a boot's ``create_session`` and the dominant term of a ``/resume``.

    Two changes, together:

    - **Dispatched after construction, not awaited, and delayed.** Every
      ``create_session`` path dispatches at its last synchronous point before
      return, after deferred/eager MCP setup has reached its intended state. The
      task therefore cannot execute until the completed coroutine gives control
      back to its caller. It then waits through a short idle window before the
      store walks, giving the TUI time to adopt the session and paint. Nothing in
      maintenance is read by session construction, so there is nothing to wait
      for. This is the same fire-and-track shape the deferred MCP wiring uses.
    - **Once per process.** Maintenance answers a question about the STORE, and
      the store does not become dirty again because the user pressed
      ``/resume``. The first session in the process runs it; later ones find the
      task already dispatched and return immediately.

    ``live_dir`` is the FIRST session's directory, and that is correct rather
    than incidental: it is the only ``live_dir`` the sweep will ever see in this
    process, and every LATER session protects itself with its claim marker
    (written synchronously before its directory exists — see ``_prepare``),
    which is the belt that protects concurrent sessions in OTHER processes too.
    The ``live_dir`` skip is a redundant second belt for the local case, not the
    load-bearing one.

    One window this opens, named rather than inherited by accident: the origin
    backfill used to COMPLETE before ``_prepare`` returned, so the in-TUI
    ``/resume`` picker could never offer a delegated run. It now races first
    paint, and on the FIRST launch after the upgrade that introduced origin
    markers the picker can briefly list subagent/reviewer sessions until the
    background pass stamps them — 67 ms for the origin pass on a 3574-session
    store, so the window is tens of milliseconds, once per upgrade, and it
    self-heals within the same launch. Resuming such a run appends to its
    transcript; nothing is destroyed. The CLI ``--resume`` path is unaffected:
    ``cli.py`` still runs both backfills eagerly and synchronously before
    resolving ``--resume``. If the picker ever grows a correctness dependency
    on origin — filtering delegated runs out by default, say — stamp origins
    eagerly here (it is the cheap pass) and background only the other three.

    A crash before first paint means the passes do not run this launch. That is
    acceptable by design: the cost is an unstamped sidecar, which is
    bytes rather than correctness, and the next launch stamps it. Every pass is
    idempotent for exactly this reason.

    Silently does nothing when called with no running loop (a synchronous test
    harness or a benchmark entry point): there is nowhere to schedule the work,
    and maintenance must never be the reason such a caller fails.
    """
    global _STORE_MAINTENANCE_TASK
    if _STORE_MAINTENANCE_TASK is not None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _STORE_MAINTENANCE_TASK = loop.create_task(
        _run_store_maintenance(config_manager, config_dir, live_dir)
    )


async def _prepare(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
    *,
    has_ui: bool,
    cwd: str | None = None,
) -> _SessionPlan:
    """Shared wiring core used by :func:`create_session` and
    :func:`build_initial_blocks`. Returns the Session kwargs plus the blocks
    provider; raises ``ValueError`` when hosting/model config is missing.
    ``cwd`` (default: process cwd) is the single working-directory source for
    the tool context, the session and MCP discovery."""
    agent = resolve_agent(args, agent_registry)
    hosting, model_name, model_source = resolve_hosting_model_with_source(
        agent, args, config_manager
    )
    yolo = bool(getattr(args, "yolo", False))

    transcript_dir, agent_id = _transcript_dir_and_agent_id(agent, args, agent_registry)

    from local_operator.session.retention import claim_session
    from local_operator.session_lease import acquire_session_lease

    # Sole-writer ownership is acquired at the shared construction boundary,
    # before transcript creation. Edge checks remain useful UX, but only O_EXCL
    # can make two simultaneous cold resumes safe. Agent training directories
    # retain their established non-session semantics and are not leased here.
    session_lease = (
        acquire_session_lease(transcript_dir) if transcript_dir.parent.name == "sessions" else None
    )

    # CLAIM BEFORE creating the directory, and in that order. The claim marker
    # is what tells anything scanning the store (the user-enabled cleanup
    # policy, the picker, the mobile daemon) that this directory belongs to a
    # live run; ``claim_session`` creates the directory itself and writes the
    # marker in one step, so there is no instant at which this directory
    # exists empty-and-unclaimed. Nothing deletes on that signal by default,
    # but the ordering is what makes the claim guard airtight when cleanup IS
    # enabled, and it costs nothing to keep.
    #
    # ``claim_session`` refuses agent directories itself (the gate lives with
    # the marker, not here), so the explicit ``mkdir`` below is what creates
    # the directory in the ``--train``/named-agent case, which is deliberately
    # never claimed and never scanned.
    claim_session(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    if transcript_dir.parent.name == "sessions":
        # Stamp the store as ours. The cleanup policy refuses to remove
        # anything from an unmarked ``sessions/`` directory, and this is the
        # one place the harness knows it is writing into its own store —
        # cleanup itself never marks, so it can never authorise its own
        # target. Idempotent and best-effort.
        from local_operator.session.cleanup import mark_store

        mark_store(transcript_dir.parent)

    # The lease/claim above stay synchronous and on the loop — sole-writer
    # ordering (lease before transcript creation) is an invariant, and putting a
    # yield inside that window is how two cold resumes lose the race the lease
    # exists to arbitrate. Whole-store maintenance is dispatched only after
    # ``create_session`` finishes ALL construction; starting it here lets the
    # runner contend as soon as the model configuration below yields to a worker.

    # --- model + stream fn (stream B contracts) ---------------------------
    from local_operator.env import get_env_config
    from local_operator.model.configure import configure_model, create_stream_fn

    chat_kwargs: dict[str, Any] = {}
    if agent is not None:
        for field_name in _AGENT_SAMPLING_FIELDS:
            value = getattr(agent, field_name, None)
            if value is not None:
                chat_kwargs[field_name] = value
    # OFF THE EVENT LOOP. `configure_model` is synchronous, and for a model the
    # shipped registry does not fully describe it fetches the provider's live
    # listing over a BLOCKING httpx client (see
    # `model.configure._info_from_discovery`). On the TUI's loop that is a
    # frozen screen and a swallowed keystroke buffer for as long as the
    # provider takes to answer — up to `discovery.DEFAULT_TIMEOUT_S`, 10 s, on
    # a bad network. A thread costs nothing here: every caller is already
    # awaiting this line, and the work is a network wait plus a memoised
    # lookup.
    model_configuration = await asyncio.to_thread(
        functools.partial(
            configure_model,
            hosting=hosting,
            model_name=model_name,
            credential_manager=credential_manager,
            env_config=get_env_config(),
            **chat_kwargs,
        )
    )
    spec = model_configuration.spec

    from local_operator.providers.auth_store import AuthStore

    auth_store = AuthStore(credential_manager=credential_manager)
    # A fork inherits its PARENT's provider cache key. The fork's transcript is
    # a byte-identical copy, so its first request reproduces the parent's cached
    # prefix exactly and should be routed to it rather than opening a fresh one.
    # Credential stickiness deliberately stays on this session's own id — see
    # ``create_stream_fn``. Empty for any session that is not a fork, which is
    # the ordinary case and costs one sidecar read at construction.
    from local_operator.fork import fork_parent

    stream_fn = create_stream_fn(
        auth_store,
        settings=config_manager.get_config().values,
        session_id=transcript_dir.name,
        cache_lineage_id=fork_parent(transcript_dir) or None,
    )

    # --- tools + lazy knowledge (streams A and C) --------------------------
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    config_dir = Path(agent_registry.config_dir)
    effective_cwd = cwd if cwd is not None else os.getcwd()
    knowledge_warnings: list[str] = []
    hooks = await _setup_knowledge(
        credential_manager, config_dir, agent_registry, knowledge_warnings
    )
    # Configuration discovery is local filesystem work and must precede the
    # first prompt. The TUI deliberately defers live MCP connections; deriving
    # names only from the eventual manager let the knowledge block freeze empty
    # before that background task won the race.
    _seed_mcp_routing(hooks, effective_cwd)
    for warning in knowledge_warnings:
        print(f"\033[1;33mWarning: {warning}\033[0m", file=sys.stderr)

    request_approval = _make_request_approval(yolo)
    # The variables surface behind list_variables/read_variable: config
    # overrides ride above the project file and process environment, and
    # values stay out of the system prompt (read on demand, not baked).
    #
    # Built once and handed to BOTH contexts. The factory context below is what
    # `create_tools` inspects to decide which tools exist; the context a tool
    # actually executes against is rebuilt by `Session._build_tool_context` on
    # every turn, so a store installed only here reached the createIf check and
    # nothing else — `list_variables` advertised itself and then read a bare
    # process-env store, in every session.
    variable_store = _build_variable_store(effective_cwd, config_manager)
    from local_operator.teams import TeamRegistry

    # R7-2: the session must start even when `teams/` cannot be read.
    #
    # `TeamRegistry.__init__` performs crash recovery, and that can fail for
    # reasons that have nothing to do with the session being built — a stranded
    # `.<id>.backup.*` under a directory whose permissions changed, a full
    # filesystem. Constructing it unguarded made a subdirectory the user may
    # never have touched abort the whole boot: no model, no tools, no
    # transcript, and an error naming only the teams registry as the remedy.
    #
    # So the failure degrades ONE feature instead of the session. The context
    # gets no registry, which is exactly the state `build_team_tool`'s createIf
    # and the TUI's `_team_registry()` already handle (the `team` tool is not
    # offered, `/team` says teams are unavailable). The reason is surfaced in
    # the same warning channel as the knowledge-discovery failures above rather
    # than swallowed, so the user is told what to fix.
    #
    # The registry ITSELF still refuses to answer with a half-truth: a
    # construction-time recovery failure is remembered and re-raised by the
    # first real read (see `TeamRegistry._raise_if_recovery_failed`), so the
    # CLI and tool guards keep reporting it rather than showing an empty list.
    team_registry: TeamRegistry | None
    try:
        team_registry = TeamRegistry(config_dir)
    except Exception as exc:  # noqa: BLE001 — one feature must not fail boot
        team_registry = None
        print(
            f"\033[1;33mWarning: teams are unavailable this session: {exc}\033[0m",
            file=sys.stderr,
        )
    tool_context = ToolContext(
        cwd=effective_cwd,
        session_id=transcript_dir.name,
        agent_id=agent_id,
        has_ui=has_ui,
        request_approval=request_approval,
        variables=variable_store,
        # Role profiles and the ``agent`` tool are backed by this registry; a
        # host without one keeps working off the packaged starters.
        agent_registry=agent_registry,
        team_registry=team_registry,
        web_search_settings=config_manager.get_config_value("web_search", None),
        web_fetch_settings=config_manager.get_config_value("web_fetch", None),
    )
    tools = create_tools(tool_context)

    from local_operator.session.goal import GoalState
    from local_operator.session.transcript import Transcript

    # See `_transcript_dir_and_agent_id`: a speculatively warmed runtime must
    # not materialise a session directory the user may never commit to. The
    # flag is read here rather than threaded through the signature because it
    # is set by `_spawn_runtime` on the child's environment and consumed only
    # on this path.
    transcript = Transcript(
        transcript_dir,
        defer_materialise=os.environ.get("LOP_RUNTIME_DEFER_MATERIALISE") == "1",
    )
    # One holder shared by the prompt provider and the session facade, so a
    # ``/goal`` change lands in the next model step without a session rebuild.
    goal_state = GoalState()
    # Read once, at session construction: see the provider's docstring for why
    # this must not be re-read per turn. A profile's own prompt is layered on
    # top of the global file rather than replacing it.
    agent_prompt = ""
    if agent is not None:
        try:
            agent_prompt = agent_registry.get_agent_system_prompt(str(agent.id))
        # ``ValueError`` covers ``UnicodeDecodeError``, which is NOT an
        # ``OSError``: a mis-encoded profile prompt used to raise straight
        # through here and kill session startup. The registry now reads with
        # ``errors="replace"`` so that specific route can no longer raise, but
        # the guard stays for any other decode path a registry might take —
        # an unreadable profile must never cost the operator their session.
        # Logged rather than swallowed in silence. The guard is deliberately
        # broader than its motivating decode error, because a profile prompt is
        # not worth a failed session whatever the registry raises reading it --
        # but a session that quietly drops the agent the operator selected
        # looks like the profile was empty, so the reason has to be findable.
        except (KeyError, OSError, ValueError) as exc:
            logger.warning(
                "could not read the system prompt for agent %s (%s: %s); "
                "continuing without the profile's own instructions",
                agent.id,
                type(exc).__name__,
                exc,
            )
            agent_prompt = ""
    user_instructions = load_user_instructions(agent_prompt)
    # Repo guidance (AGENTS.md/CLAUDE.md ancestors) joins the same read-once
    # contract: the head block must stay byte-stable for the session, so the
    # filesystem is consulted here and never again.
    from local_operator.context_files import load_repo_guidance

    try:
        repo_guidance = load_repo_guidance(effective_cwd)
    except Exception:  # noqa: BLE001 — never block session construction
        repo_guidance = ""

    system_blocks_provider = _make_system_blocks_provider(
        tools,
        transcript,
        hooks,
        cwd=effective_cwd,
        goal_state=goal_state,
        user_instructions=user_instructions,
        repo_guidance=repo_guidance,
        variable_store=variable_store,
    )

    session_kwargs: dict[str, Any] = dict(
        model=spec,
        stream_fn=stream_fn,
        tools=tools,
        transcript=transcript,
        agent_id=agent_id,
        system_blocks_provider=system_blocks_provider,
        convert_to_llm=default_convert_to_llm,
        compaction_settings=coerce_compaction_settings(
            config_manager.get_config_value("compaction", None)
        ),
        yolo=yolo,
        has_ui=has_ui,
        cwd=effective_cwd,
        # Session keeps the historical parameter name, but the chained
        # resolver handles both guide:// and skill:// without namespace leaks.
        skill_resolver=_make_knowledge_resolver(hooks),
        request_approval=request_approval,
        goal_state=goal_state,
        variables=variable_store,
        agent_registry=agent_registry,
        team_registry=team_registry,
        # Whether a later ``hosting``/``model_name`` edit switches this session
        # (config-sourced) or only prints a keep notice (agent/flag).
        model_source=model_source,
    )
    return _SessionPlan(
        session_kwargs=session_kwargs,
        system_blocks_provider=system_blocks_provider,
        knowledge_hooks=hooks,
        auth_store=auth_store,
        session_lease=session_lease,
    )


def _collapse_sdk_missing_failures(
    failures: dict[str, str], discovery_key: str, sdk_missing_error: str
) -> dict[str, str]:
    """Collapse an all-servers-failed-for-a-missing-SDK map to one entry.

    When the MCP SDK is not installed the manager fails every configured server
    with the SAME install instruction. Reported ONCE, as the setup problem it
    is: N identical 90-character notices (one toast line plus one transcript
    error per server, every launch) is noise proportional to server count for a
    single cause, and it accuses the servers of a fault that is not theirs.
    Compared by identity against the manager's own constant rather than by
    substring, so re-wording it cannot silently disable this. Anything else is
    returned unchanged. Shared by the boot snapshot and the settled re-report so
    both surfaces collapse the same way.
    """
    if failures and set(failures.values()) == {sdk_missing_error}:
        return {discovery_key: sdk_missing_error}
    return failures


def _fire_mcp_sink(session: Session) -> None:
    """Hand the just-recorded ``mcp_startup`` outcome to the front-end sink.

    Shared by the wiring's three completion points — the two degradation
    arms (no MCP layer; discovery raised) and the gate snapshot — because a
    deferred-boot TUI learns about ALL of them the same way: it installed
    its sink while the manager was still absent and needs exactly one
    nudge per outcome to re-run its wiring and report. Guarded like the
    settle callback's own lookup: a session without a sink (headless, an
    unadopted session) is the normal case, and a sink that raises must
    never take the wiring down with it.
    """
    sink = getattr(session, "_on_mcp_startup_settled", None)
    if sink is None:
        return
    try:
        sink(getattr(session, "mcp_startup", None))
    except Exception:  # noqa: BLE001 — a UI hook must never break the wiring
        logger.debug("session _on_mcp_startup_settled raised", exc_info=True)


async def wire_mcp_into_session(
    session: Session,
    builtin_tools: list[AgentTool],
    cwd: str,
    knowledge_hooks: _KnowledgeHooks | None = None,
    auth_store: AuthStore | None = None,
    *,
    has_ui: bool = False,
    _deferred_boot: bool = False,
) -> McpManager | None:
    """Discover MCPs but expose their schemas only after explicit reads.

    Startup connects and caches servers exactly as before, but the session
    begins with its non-MCP tools only. A bounded ``<mcps>`` catalogue tells
    the model which servers exist. ``read mcp://<server>`` lists tools without
    loading schemas; ``read mcp://<server>/<tool>`` activates exactly one tool.
    Live list-changed events refresh only schemas the model already selected.
    This keeps an unused MCP server at O(server names), not O(all tool schemas),
    on every provider request.

    ``has_ui`` selects how failures are announced, never whether they are
    recorded. Full-screen clients read ``session.mcp_startup``; headless callers
    get the warning on stderr. Any failure degrades to an empty catalogue, so
    MCP enrichment never becomes a session startup requirement. Returns the
    manager for the caller to dispose, or ``None``.
    """
    from local_operator.session.mcp_status import MCP_DISCOVERY_KEY, McpStartupOutcome

    if knowledge_hooks is None:
        knowledge_hooks = _KnowledgeHooks()

    try:
        from local_operator.mcp import discover_and_load_mcp_tools
        from local_operator.mcp.manager import MCP_SDK_MISSING_ERROR
    except ImportError:
        if not has_ui:
            print(
                "\033[1;33mWarning: MCP support unavailable, continuing without MCP tools\033[0m",
                file=sys.stderr,
            )
        # This does NOT catch a missing MCP SDK. Every SDK import in the package
        # is either ``TYPE_CHECKING`` or function-local, so ``local_operator.mcp``
        # imports cleanly with the SDK absent and that case lands in the error
        # loop below instead. What reaches here is our OWN package failing to
        # import — a partial or broken install. An EMPTY outcome is still the
        # right record for it: without the config layer we cannot read the config
        # files, so we do not know whether this machine wanted MCP at all, and
        # "MCP is broken" on a host that never used it is noise.
        session.mcp_startup = McpStartupOutcome()
        if _deferred_boot:
            _fire_mcp_sink(session)
        return None

    try:
        manager, mcp_tools, errors = await discover_and_load_mcp_tools(cwd, auth_store=auth_store)
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        # Discovery raising IS reportable, unlike the import gap above: reaching
        # this line means the config layer was present and still could not be
        # read, so the user has an MCP setup that is not working.
        if not has_ui:
            print(
                f"\033[1;33mWarning: MCP discovery failed, continuing without MCP tools: "
                f"{exc}\033[0m",
                file=sys.stderr,
            )
        session.mcp_startup = McpStartupOutcome(failures={MCP_DISCOVERY_KEY: str(exc)})
        if _deferred_boot:
            _fire_mcp_sink(session)
        return None

    # One pass over the error entries: the record keys on the BARE server name
    # (the discovery wrapper reports paths as ``mcp:<server>``) because that is
    # what the user typed in ``.mcp.json`` and what ``/mcp`` lists back. Entries
    # WITHOUT that prefix are the layer failing rather than a server — the
    # wrapper's synthetic hard-failure entry says ``.mcp.json`` — so they take
    # the same key the raising arm above uses. One synthetic key, not three
    # spellings of "not a server".
    failures: dict[str, str] = {}
    for entry in errors:
        path = str(entry.get("path", "?"))
        message = str(entry.get("error", "unknown error"))
        failures[path.partition("mcp:")[2] or MCP_DISCOVERY_KEY] = message

    failures = _collapse_sdk_missing_failures(failures, MCP_DISCOVERY_KEY, MCP_SDK_MISSING_ERROR)

    settling = False
    try:
        settling = manager.startup_settling()
    except Exception:  # noqa: BLE001 — a missing accessor must not break wiring
        logger.debug("MCP startup_settling() unavailable", exc_info=True)

    # Headless callers are one-shot and do not stay alive for the settle
    # re-report, so they print what the gate knows. But a PROVISIONAL failure
    # (a server still connecting past the gate) must not be announced as a hard
    # failure on stderr either — the same false alarm the toast used to raise.
    # While settling, print only the failures already terminal at the gate; a
    # server still deferred is neither connected nor failed yet, and its entry
    # (if any) is a not-yet-final one the settled re-report owns.
    if not has_ui and not settling:
        for name, message in failures.items():
            subject = "MCP discovery" if name == MCP_DISCOVERY_KEY else f"MCP server {name}"
            print(f"\033[1;33mWarning: {subject}: {message}\033[0m", file=sys.stderr)

    session.mcp_startup = McpStartupOutcome(
        configured=tuple(manager.get_all_server_names()),
        connected=tuple(manager.get_connected_servers()),
        failures=failures,
        tool_count=len(mcp_tools),
        settling=settling,
    )
    # The gate snapshot above is also the moment the wiring's MANAGER first
    # exists. On the deferred boot path the TUI adopted the session before
    # this line ran, found ``mcp_manager`` None, and installed its settle
    # sink in that state — so tell it now. Without this hop the sink waits
    # for SETTLE, which a manager with nothing deferred never fires: the
    # band's live subscriptions and the boot toast would depend on a
    # callback that a fast, fully-connected round never triggers.
    # FIRED ONLY on the deferred boot path (``_deferred_boot``): there the
    # front end adopted the session before this wiring ran, and the sink it
    # installed in that state is the one route the wiring's completion has
    # back into the app. The synchronous path keeps its existing contract
    # — the sink fires on SETTLE only, exactly as the factory's settle test
    # pins — because an already-adopted session gets its live wiring from
    # the caller's own return path, not from a mid-function nudge.
    if _deferred_boot:
        _fire_mcp_sink(session)

    # Re-report once the round settles: the boot snapshot above was taken at the
    # 250 ms gate while OAuth HTTP servers were still connecting. When the last
    # deferred server reaches a terminal state, rebuild ``session.mcp_startup``
    # from the manager's COMBINED tally (every failure, the final connected set)
    # and hand it to whatever front-end sink the session installed. Wired even
    # when ``settling`` is False right now: a fast machine can still defer a
    # server between this read and the callback install, and an unused callback
    # is free.
    def _on_startup_settled() -> None:
        try:
            settled_failures = _collapse_sdk_missing_failures(
                manager.startup_failures(), MCP_DISCOVERY_KEY, MCP_SDK_MISSING_ERROR
            )
            outcome = McpStartupOutcome(
                configured=tuple(manager.get_all_server_names()),
                connected=tuple(manager.get_connected_servers()),
                failures=settled_failures,
                tool_count=len(manager.get_tools()),
                settling=False,
            )
        except Exception:  # noqa: BLE001 — a settle rebuild must never break the manager
            logger.debug("MCP settled outcome rebuild failed", exc_info=True)
            return
        session.mcp_startup = outcome
        if hasattr(session, "_frontend_state_store"):
            session.refresh_frontend_state()
        if not has_ui:
            # A late failure that the gate never printed still deserves the
            # stderr line the settling guard above withheld.
            for name, message in settled_failures.items():
                subject = "MCP discovery" if name == MCP_DISCOVERY_KEY else f"MCP server {name}"
                print(f"\033[1;33mWarning: {subject}: {message}\033[0m", file=sys.stderr)
        sink = getattr(session, "_on_mcp_startup_settled", None)
        if sink is not None:
            try:
                sink(outcome)
            except Exception:  # noqa: BLE001 — a UI hook must never break the manager
                logger.debug("session _on_mcp_startup_settled raised", exc_info=True)

    manager.on_startup_settled = _on_startup_settled

    # The NON-MCP base is READ BACK from the session on every refresh, not
    # snapshotted once. Session capability tools live in that inventory even
    # though ``builtin_tools`` predates them, and some of them are merged in
    # AFTER this wiring runs: the TUI installs its ask handler in
    # ``_adopt_session``, long after the factory returned, and a frozen base
    # would silently un-advertise ``ask`` again the first time the model
    # activated any MCP tool. What is subtracted is the set this function last
    # installed itself, so classification never depends on ``get_tool_meta``
    # still answering for a server that has since dropped away. The snapshot
    # below survives only as the fallback for a host that exposes no inventory
    # to read back.
    base_inventory = list(
        getattr(session, "_tools", None) or getattr(session, "tools", None) or builtin_tools
    )
    installed_mcp: set[str] = set()
    enabled_origins: set[tuple[str, str]] = set()
    deferred_origins: set[tuple[str, str]] = set()
    setattr(session, "_mcp_deferred_origins", deferred_origins)

    def selected_tools(source: list[AgentTool]) -> list[AgentTool]:
        selected: list[AgentTool] = []
        for tool in source:
            meta = manager.get_tool_meta(tool.name) or {}
            origin = (str(meta.get("server_name", "")), str(meta.get("mcp_tool_name", "")))
            if origin in enabled_origins:
                selected.append(tool)
        return selected

    def refresh_selected(source: list[AgentTool]) -> None:
        live = list(getattr(session, "_tools", None) or getattr(session, "tools", None) or ())
        base = [tool for tool in live if tool.name not in installed_mcp] or base_inventory
        selected = selected_tools(source)
        installed_mcp.clear()
        installed_mcp.update(tool.name for tool in selected)
        session.refresh_tools(base + selected)

    def activate(server_name: str, raw_tool_name: str) -> None:
        enabled_origins.add((server_name, raw_tool_name))
        refresh_selected(manager.get_tools())

    from local_operator.mcp.resources import make_mcp_resolver, render_mcp_catalogue

    def defer(server_name: str, raw_tool_name: str) -> None:
        deferred_origins.add((server_name, raw_tool_name))

    prior = getattr(session, "_fallback_tool_resolver", None)

    def resolve_deferred(name: str) -> AgentTool | None:
        for tool in manager.get_tools():
            meta = manager.get_tool_meta(tool.name) or {}
            origin = (str(meta.get("server_name", "")), str(meta.get("mcp_tool_name", "")))
            if tool.name == name and origin in deferred_origins:
                return tool
        return prior(name) if prior is not None else None

    if hasattr(session, "set_fallback_tool_resolver"):
        session.set_fallback_tool_resolver(resolve_deferred)
    knowledge_hooks.mcp_resolver = make_mcp_resolver(manager, activate, defer=defer)
    # Once the manager exists, compaction-time reselection sees reloads. The
    # already frozen first-task block remains byte-stable until compaction.
    knowledge_hooks.mcp_catalogue = lambda query: render_mcp_catalogue(manager, query)

    def on_tools_changed(new_mcp_tools: list[AgentTool]) -> None:
        # Reconnects and tools/list_changed can replace AgentTool objects. Keep
        # the selected origins and swap in only their fresh schemas.
        refresh_selected(new_mcp_tools)
        if hasattr(session, "_frontend_state_store"):
            session.refresh_frontend_state()

    manager.set_on_tools_changed(on_tools_changed)
    return manager


def attach_mcp_dispose(session: Session, manager: McpManager) -> None:
    """Fold ``manager.disconnect_all()`` into the session's dispose path.

    The CLI/TUI/exec all call ``session.dispose()`` exactly once, so hanging
    MCP teardown off it tears the servers down everywhere without teaching
    each caller about the manager. The manager is also exposed as
    ``mcp_manager`` for diagnostics.
    """
    session.add_dispose_hook(manager.disconnect_all)
    session.mcp_manager = manager
    if hasattr(session, "_frontend_state_store"):
        session.refresh_frontend_state()
    # Breaker incidents become session incidents: the model learns a server's
    # tools are gone instead of hammering them (MCP-07's observable half).
    manager.on_incident = session._on_mcp_incident


def _cancel_task(task: "asyncio.Task[Any]") -> Callable[[], Awaitable[None] | None]:
    """A dispose hook that cancels one task and awaits its quietus.

    Used for the TUI boot path's background MCP wiring: a session disposed
    while wiring is still in flight must not leave the task running against
    a torn-down session (the wiring writes ``session.mcp_startup`` and merges
    tools into it). Returning an awaitable is part of the dispose-hook
    contract — hooks may be coroutines — so the cancellation is AWAITED
    before the rest of teardown proceeds, and a wiring coroutine already
    inside ``wire_mcp_into_session`` gets to run its own finally blocks.
    """

    async def _hook() -> None:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown proceeds
            pass

    return _hook


def attach_auth_dispose(session: Session, auth_store: AuthStore | None) -> None:
    """Fold ``auth_store.close()`` into the session's dispose path (CL-08).

    The ``AuthStore`` opens a SQLite connection per session; every front end
    calls ``session.dispose()`` exactly once, so registering here guarantees
    the connection (and its file lock) is released everywhere without
    teaching each caller.
    """
    if auth_store is None:
        return
    session.add_dispose_hook(auth_store.close)


def attach_stream_dispose(session: Session, stream_fn: SessionStreamFn) -> None:
    """Fold the session's shared ``httpx.AsyncClient`` close into dispose.

    ``create_stream_fn`` builds one client per session and hangs its close on
    the returned object; without this seam the pool leaks for the process
    lifetime (one per turn on the server facade).
    """
    session.add_dispose_hook(stream_fn.close)


def attach_config_watch(session: Session, config_dir: Path) -> None:
    """Subscribe ``session`` to live ``config.yml`` changes; unsubscribe on dispose.

    The config-watch seam (see :mod:`local_operator.config_watch`). Starts the
    PROCESS's watcher if this is the first session to ask — ``start`` is
    idempotent, so a ``/new`` in the same process finds it running — and hangs
    the session's listener on it. The watcher itself is process-scoped and is
    NOT stopped on dispose: the next session in this process needs it, and the
    loop closing reaps the task. Only the subscription is per-session, which
    is why the unsubscriber and not a ``stop`` is the dispose hook.

    Every front end (TUI, headless, exec worker, owned phone session) reaches
    this through ``create_session``, so they all follow config for free.
    ``RemoteSession`` followers never get here: the owner applies the change
    and the follower renders what the owner projects.

    Degrades to "this session does not follow config" on any failure rather
    than failing the boot: a watcher that cannot start (no loop in an unusual
    embedding, a config directory that cannot be opened) leaves the session
    exactly as it was before this seam existed.
    """
    try:
        from local_operator.config_watch import process_watcher

        watcher = process_watcher(config_dir)
        watcher.start(asyncio.get_running_loop())
        session.add_dispose_hook(watcher.subscribe(session._apply_config_change))
    except Exception:  # noqa: BLE001 — boot must not depend on the watcher
        logger.warning("config watcher could not be attached to the session", exc_info=True)


async def create_session(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
    *,
    has_ui: bool = False,
    cwd: str | None = None,
    _force_local_takeover: bool = False,
    defer_mcp_wiring: bool = False,
) -> "SessionProtocol":
    """Build a fully-wired harness session from parsed CLI args.

    This is THE factory shared by ``cli.py`` (interactive TUI / headless
    REPL), ``exec_mode.run_exec`` (foreground exec) and ``exec_worker``
    (background exec). All engine modules are imported lazily inside; the
    caller only needs the three legacy managers plus an argparse namespace
    carrying ``hosting``, ``model``, ``agent_name``/``agent_id``, ``yolo``
    and ``train``.

    ``cwd`` is the session's working directory; ``None`` means the process
    cwd (legacy behaviour). Hosts that must relocate a session (the
    scheduler's per-agent directory) pass it explicitly instead of mutating
    the process-global cwd across awaits — every other session builder in
    the same process would otherwise read the wrong directory.

    ``defer_mcp_wiring`` is the TUI boot path's OPT-IN to having MCP servers
    wired in the background after the session is returned, so the first
    frame does not wait for the 250 ms discovery gate. Every other caller
    keeps the old contract — a returned session has MCP wiring completed
    (or degraded and recorded) — because headless/exec runs have no front
    end to re-read ``mcp_startup`` when the background round settles; they
    would silently miss both the tool merge and the failure report. The
    deferral is safe for the TUI only because MCP tools were already lazy
    (see :func:`wire_mcp_into_session`): a turn started before wiring
    settles sees the same non-MCP tool surface as a session whose servers
    missed the gate today, and the ``refresh_selected`` merge lands
    mid-session exactly as a late ``list_changed`` event already does.

    Raises ``ValueError`` (caught by the CLI's red-banner handler) when the
    hosting/model configuration is missing.
    """

    # Create the agent's working-directory home HERE, lazily, rather than
    # unconditionally in main() before dispatch (where it hardcoded the path
    # and ignored the override). A session is a path that actually runs a task,
    # so an agent whose cwd is the default ``~/local-operator-home`` has a real
    # directory to land in. Best-effort: a session must not fail to build just
    # because the workspace root could not be created (a read-only home), so a
    # creation error degrades to the process cwd the same way an unset cwd does.
    from local_operator.paths import ensure_agent_home_dir
    from local_operator.session.session import Session

    try:
        ensure_agent_home_dir()
    except OSError:
        pass

    effective_cwd = cwd if cwd is not None else os.getcwd()

    # A full-screen TUI resuming a session already owned elsewhere consumes
    # that owner's v4 event relay through RemoteSession. This lives at the
    # shared session-factory seam (not in cli.py) so cold ``--resume`` and any
    # future TUI launcher cannot accidentally construct a second writer or
    # invent another attach UI. Headless/exec callers still take the lease and
    # get the existing refusal: they have no full front end to host the facade.
    resume_id = getattr(args, "resume", None)
    if has_ui and resume_id is not None and not _force_local_takeover:
        from local_operator.mobile.attach_client import find_owner_record
        from local_operator.session.remote import RemoteSession

        root = Path(agent_registry.config_dir)
        record, owner = await asyncio.to_thread(find_owner_record, root, str(resume_id))
        if owner is not None and owner != os.getpid():
            if record is None or record.protocol < 4:
                raise ValueError(
                    f"session {resume_id} is open in an older Local Operator process "
                    f"(pid {owner}); update or close it, then resume again"
                )

            async def takeover_factory() -> "SessionProtocol":
                # Owner death is the one time this process may try the writer
                # path. The lease is still the arbiter: racing followers call
                # this concurrently, one wins, losers get SessionLeaseHeldError
                # and RemoteSession rediscovers the winner.
                return await create_session(
                    args,
                    config_manager,
                    credential_manager,
                    agent_registry,
                    has_ui=True,
                    cwd=effective_cwd,
                    _force_local_takeover=True,
                )

            return await RemoteSession.connect(
                record,
                str(resume_id),
                config_dir=root,
                takeover_factory=takeover_factory,
            )

    plan = await _prepare(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=has_ui,
        cwd=effective_cwd,
    )
    try:
        session = Session(**plan.session_kwargs)
    except BaseException:
        # Construction never transferred ownership to Session.dispose, so the
        # factory must relinquish its generation without touching a successor.
        if plan.session_lease is not None:
            plan.session_lease.release()
        raise
    if plan.session_lease is not None:
        session.add_dispose_hook(plan.session_lease.release)

    # Auth seam (CL-08): the AuthStore's SQLite connection is owned by this
    # session; fold its close into dispose so every front end releases the
    # file lock on the single ``session.dispose()`` call.
    attach_auth_dispose(session, plan.auth_store)
    # Stream seam: release the session's shared httpx connection pool on
    # dispose (one leaked pool per turn on the server facade otherwise).
    attach_stream_dispose(session, plan.session_kwargs["stream_fn"])
    # Config seam: follow ``config.yml`` while the session lives, so an edit in
    # another pane (or on the page in this one) reaches compaction, retry and
    # the job cap without a ``/new``. The manager's directory, not
    # ``paths.config_dir()``: they agree in production, and where a caller
    # passed a manager on another directory that is the file to follow.
    attach_config_watch(session, Path(getattr(config_manager, "config_dir", app_config_dir())))

    # MCP seam (MCP-20): merge discovered MCP tools in, subscribe to live
    # changes, and fold server teardown into session.dispose. Degrades to
    # zero MCP tools on any failure. ``has_ui`` routes the announcement: a
    # front end with a full-screen terminal reads session.mcp_startup instead
    # of being written over by a stderr warning.
    #
    # DEFERRED wiring is the TUI boot path's opt-in (``defer_mcp_wiring``):
    # the session returns immediately and the same wiring runs as a background
    # task. The task is tracked on the session's dispose hooks so a quit
    # mid-wiring cancels it (a ``disconnect_all`` on a half-wired manager is
    # exactly the teardown the manager already handles); nothing else differs —
    # the outcome lands in ``session.mcp_startup`` and the settle sink fires
    # when the TUI has installed it, which is the same late-attach the 250 ms
    # gate already produces for slow OAuth servers.
    if defer_mcp_wiring:

        async def _wire_mcp_background() -> None:
            # Runs on the session's loop but OFF the boot critical path. An
            # exception here is the wiring's own degradation contract
            # (``wire_mcp_into_session`` never raises for provider reasons);
            # a genuine coding fault is logged rather than killing the task
            # silently, and the session keeps its non-MCP surface — the same
            # state a machine with no ``.mcp.json`` boots into.
            try:
                manager = await wire_mcp_into_session(
                    session,
                    list(plan.session_kwargs["tools"]),
                    effective_cwd,
                    knowledge_hooks=plan.knowledge_hooks,
                    auth_store=plan.auth_store,
                    has_ui=has_ui,
                    _deferred_boot=True,
                )
                if manager is not None:
                    attach_mcp_dispose(session, manager)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — boot must survive a wiring fault
                logger.warning("background MCP wiring failed", exc_info=True)

        wiring_task = asyncio.get_running_loop().create_task(_wire_mcp_background())
        # Dispose-during-wiring cancels the task. Folded as a hook rather than
        # tracking the task on the Session: every front end already calls
        # ``session.dispose()`` once, so this is the one place teardown can
        # live without teaching each caller about the boot path.
        session.add_dispose_hook(_cancel_task(wiring_task))
        # Dispatch at the last synchronous point before returning. A task cannot
        # execute until this coroutine gives the loop back to its caller, so the
        # runner's idle window begins only after construction has completed.
        _start_store_maintenance(
            config_manager,
            Path(agent_registry.config_dir),
            Path(session._transcript.directory),
        )
        return session

    mcp_manager = await wire_mcp_into_session(
        session,
        list(plan.session_kwargs["tools"]),
        effective_cwd,
        knowledge_hooks=plan.knowledge_hooks,
        auth_store=plan.auth_store,
        has_ui=has_ui,
    )
    if mcp_manager is not None:
        attach_mcp_dispose(session, mcp_manager)
    # Headless and exec callers wire MCP eagerly, so dispatch after that await as
    # well: every successful create_session return gets the same uncontended
    # construction boundary regardless of front end.
    _start_store_maintenance(
        config_manager,
        Path(agent_registry.config_dir),
        Path(session._transcript.directory),
    )
    return session


async def build_initial_blocks(
    args: argparse.Namespace,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    agent_registry: AgentRegistry,
) -> list[str]:
    """Render the session's initial system blocks WITHOUT running a turn.

    Benchmark hook (orchestrator duty): lets
    ``scripts/bench_context_budget.py`` measure the startup prompt size
    (instructions + tools inventory + skills + env) against the <=30k start
    budget without instantiating the session facade.
    """
    plan = await _prepare(args, config_manager, credential_manager, agent_registry, has_ui=False)
    # No session facade is built on this path, so the lease and store lifetimes
    # end here rather than waiting for a Session.dispose that cannot happen.
    if plan.session_lease is not None:
        plan.session_lease.release()
    # No session facade is built on this path, so the store's lifetime ends
    # here: close it directly (CL-08) to release the SQLite lock. Pass the
    # spec's label so the measured startup prompt includes the model line the
    # real session will carry (the benchmark budget must not under-count it).
    spec = plan.session_kwargs.get("model")
    model_label = f"{spec.provider}/{spec.model_id}" if spec is not None else ""
    try:
        return await plan.system_blocks_provider(model_label)
    finally:
        if plan.auth_store is not None:
            try:
                plan.auth_store.close()
            except Exception:  # noqa: BLE001
                pass
