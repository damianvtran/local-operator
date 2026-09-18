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
import inspect
import logging
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from local_operator.ansi import sanitize_prompt_line

# Stdlib-only and tiny, like ``paths`` below: safe at module level on the
# startup path that ``test_import_graph`` guards.
from local_operator.ecosystem_instructions import (
    content_digest,
    log_ecosystem_provenance,
    read_ecosystem_instructions,
)
from local_operator.harness.rows import is_harness_notice_row
from local_operator.harness.types import AgentMessage, Message

# Imported as an alias: ``config_dir`` is a parameter/local name in other
# functions here, and a module-level import of the same spelling would read
# like one of them.
from local_operator.paths import config_dir as app_config_dir

# Pure path policy, no engine — see local_operator/resume.py for why it is
# its own module rather than living here.
from local_operator.resume import ResumeNotFound, resume_dir

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
    from local_operator.session.runtime.publication import PublicationGate
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

#: Floor on a shared span before ``Overlaps:`` claims it is worth removing.
#: The containment test has no natural lower bound — a single ``-`` present in
#: both files is literal containment — and a WARNING row advertising a one-
#: character cost is exactly the warning operators learn to ignore, which is
#: the failure this row was added to avoid. 200 characters is roughly a
#: paragraph of standing rules: below it the row would cost more attention than
#: the duplication costs context (200 chars ≈ 50 tokens on the cached prefix,
#: and 0.3% of the 64,000-character budget), and no realistic shared rule set
#: is smaller. Deliberately not scaled to file size: the operator is being told
#: about an absolute cost re-paid on every request, not a ratio.
_OVERLAP_MIN_CHARS = 200


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

    The TOKENIZER rides along, and not for the same reason as the modules
    above: it is not an import, so nothing else warms it, but its first use
    sits inside the first turn's critical path all the same (122 ms to build
    cl100k_base's rank table — see
    :func:`local_operator.compaction.tokens.warm_tokenizer`). A TUI pays that
    cost once and keeps it; paying it at BOOT instead means the user's first
    prompt does not.

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

    # Both of these are guarded INCLUDING their imports, so this function keeps
    # the contract its docstring states. Neither callee raises on its own; the
    # import statement is the half a caller cannot see, and this one runs from
    # a boot thread where a raise would surface as a silent missing warm.
    try:
        from local_operator.compaction.tokens import warm_tokenizer

        warm_tokenizer()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("tokenizer prewarm skipped", exc_info=True)

    # And the bytecode cache, for the environments where a later process
    # cannot write its own: see ``local_operator.bytecode``. It is a no-op
    # unless this interpreter is under ``PYTHONDONTWRITEBYTECODE`` with a
    # ``PYTHONPYCACHEPREFIX``, and it is backgrounded because the work belongs
    # to the NEXT process, not to this one.
    try:
        from local_operator.bytecode import warm_bytecode_cache_in_background

        warm_bytecode_cache_in_background()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("bytecode prewarm skipped", exc_info=True)


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


class HostingNotChatError(HostingNotConfiguredError):
    """Raised when hosting names a provider that serves no chat completions.

    The fourth member of the recoverable family, and a SIBLING of
    :class:`HostingUnknownError` rather than a reuse of it: ``typesafe`` IS a
    known provider with a shipped paste-a-key login, so "not a known provider"
    would be false, and the ``/login`` that message prescribes has already
    happened — the remedy is `/model`, which is ALSO only reachable from inside
    the app, which is why this belongs to the family at all. Reported as a bare
    ``ValueError`` it would land in the red "session failed to start" branch
    with ``_session`` None, and every provider command would answer "session is
    still starting...": the terminal state this family exists to remove.

    The condition it names: a provider whose wire rejects ``chat/completions``
    on every host we reach it through (TypeSafe's Jev —
    ``ProviderDefinition.decision_only``). Booting a session on one produced a
    turn that could never be answered, with the failure arriving as a provider
    error nobody can read as "that hosting was never chattable".

    ``source`` is carried for the same reason its sibling carries it: the
    in-app repair writes the CONFIG FILE, so it can fix only the config case,
    and telling a user to run `/model` against a ``--hosting`` value is a loop
    that cannot terminate.
    """

    def __init__(self, message: str, hosting: str = "", source: str = "config") -> None:
        super().__init__(message)
        #: The decision-only provider id that was selected as a hosting.
        self.hosting = hosting
        #: Where the value came from: ``"config"``, ``"flag"`` (``--hosting``)
        #: or ``"agent"`` (an agent record) — see :attr:`HostingUnknownError.source`.
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

#: How a user leaves a DECISION-ONLY hosting, per source. The mirror of
#: :data:`_HOSTING_SOURCE_REMEDY` above, keyed the same way for the same reason:
#: `/model` writes the config file, so it repairs neither a `--hosting` argument
#: nor an agent record and promising it there would be a loop.
_HOSTING_NOT_CHAT_REMEDY = {
    "config": (
        "Point it at a chat provider with `/model` (or `local-operator config "
        "edit hosting <provider>`); the key you stored is still used, by the "
        "resource-classification layer."
    ),
    "flag": (
        "It came from the --hosting flag, so correct that flag (e.g. --hosting "
        "openai); `/model` cannot repair an argument."
    ),
    "agent": (
        "It came from the agent's own record, which overrides config, so update "
        "the agent's hosting instead."
    ),
}


def _not_chat_hosting_message(hosting: str, source: str = "config") -> str:
    """Error text for a hosting that can serve no chat completion at all.

    Names what the provider IS (a decision model reached through the
    classification layer) as well as what it cannot do, because the value is a
    real, working provider with a shipped login: a bare "unsupported hosting"
    would read as a typo the user should go and re-check in the console they just
    pasted a key from.
    """
    from local_operator.providers.registry import decision_only_message

    remedy = _HOSTING_NOT_CHAT_REMEDY.get(source, _HOSTING_NOT_CHAT_REMEDY["config"])
    # The FACT comes from the registry's one sentence (``decision_only_message``),
    # which ``build_model_spec`` refuses a live switch with too: the two surfaces
    # explain the same provider, so they must not drift into two spellings of it.
    # The REMEDY stays local because it is per-surface — this one knows whether the
    # value came from the config file, a flag, an agent record or a stored row.
    return f"{decision_only_message(hosting)} {remedy}"


def _refuse_decision_only(provider: str, source: str) -> None:
    """Raise ``HostingNotChatError`` when ``provider`` can serve no chat turn.

    One spelling for a check that now sits at four doors — the resolved config /
    agent / flag hosting, a resume's ``--hosting``/``--model`` pair, and the
    desktop pick boundary's own gate — because the failure it reports is the same
    fact every time and its message is the only place that fact is explained (see
    :func:`_not_chat_hosting_message`).

    Deliberately a raise rather than a predicate: every caller that needs the
    answer needs the SAME outcome from it, and a caller that wants to *report*
    instead of raise (the pick boundary, which answers a 422) does its own check
    where its own error shape lives.
    """
    from local_operator.providers.registry import is_decision_only

    if is_decision_only(provider):
        raise HostingNotChatError(_not_chat_hosting_message(provider, source), provider, source)


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
    uses :func:`resolve_hosting_model_with_source` to distinguish deliberate
    resume overrides from synthesized bootstrap arguments.
    """
    hosting, model_name, _source = resolve_hosting_model_with_source(agent, args, config_manager)
    return hosting, model_name


def resolve_hosting_model_with_source(
    agent: AgentData | None, args: argparse.Namespace, config_manager: ConfigManager
) -> tuple[str, str, str]:
    """Resolve conversation identity, retaining the provenance of real overrides.

    New: agent > CLI > defaults. Resume: deliberate CLI override > durable
    selection > birth precedence for legacy histories with no usable evidence.
    Agent/profile edits and synthesized bootstrap arguments are not overrides.
    """
    # Resolve durable identity BEFORE validating global defaults or building a
    # provider client. A removed/invalid default cannot make a valid saved
    # conversation impossible to resume. Bootstrap callers pass resolved pairs
    # too, so their values are not automatically deliberate resume overrides.
    from local_operator.providers.registry import (
        get_provider_definition,
        is_decision_only,
    )
    from local_operator.session.model_selection import (
        read_model_selection,
        refused_decision_only_selection,
    )

    directory = None
    resume = getattr(args, "resume", None)
    if resume:
        try:
            directory = resume_dir(config_manager.config_dir, str(resume))
        except (ResumeNotFound, ValueError, FileNotFoundError):
            # A viewer may own only a freshly minted id, with no directory yet.
            pass
    elif getattr(args, "train", False):
        # The legacy unnamed --train/server persistence path uses the registry's
        # stable autosave id, not a new conversation on each request.
        agent_id = str(agent.id) if agent is not None else "autosave"
        directory = Path(config_manager.config_dir) / "agents" / agent_id
    saved = read_model_selection(directory) if directory is not None else None
    explicit = getattr(args, "model_selection_override", True)
    flag_hosting = getattr(args, "hosting", None) if explicit else None
    flag_model = getattr(args, "model", None) if explicit else None
    if saved is not None:
        if flag_hosting or flag_model:
            from local_operator.model.defaults import default_model_for

            provider = flag_hosting or saved.provider
            model = flag_model or default_model_for(provider)
            if get_provider_definition(provider) is None:
                raise HostingUnknownError(
                    _unknown_hosting_message(provider, "flag"), provider, "flag"
                )
            # A deliberate override naming a decision model gets the same refusal
            # as the config path, and it has to be HERE rather than after the
            # default-model lookup: a decision-only provider has no default model,
            # so the lookup would report "no model configured" — a message about a
            # symptom, for a pair that can never run whatever model it names.
            _refuse_decision_only(provider, "flag")
            if not model:
                raise ModelNotConfiguredError(_no_model_message(provider), provider)
            return provider, model, "flag"
        return saved.provider, saved.model_id, "resume"

    # NO usable stored selection. Fall back to the birth precedence — but first ask
    # whether the journal HELD one this build refuses to run as a chat model. The
    # reader refuses such a row (``model_selection._selection``), so the row never
    # becomes ``saved`` and this is the only place the refusal can still be
    # EXPLAINED: silently resuming the conversation on the configured hosting would
    # hide that its own stored identity was the thing that cannot chat, and the
    # message here is the one the config path already produces, with the same
    # recoverable outcome (the app's setup state, where ``/model`` answers it).
    #
    # Skipped when the caller named a hosting/model deliberately: a flag is a
    # statement about this run, and refusing it because the journal is poisoned
    # would block the very exit the message prescribes.
    if not (flag_hosting or flag_model):
        refused = refused_decision_only_selection(directory) if directory is not None else None
        if refused is not None:
            raise HostingNotChatError(
                _not_chat_hosting_message(refused, "resume"), refused, "resume"
            )

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
        else "flag" if explicit and (flag_hosting or flag_model) else "config"
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
    if get_provider_definition(hosting) is None:
        # Before the default-model lookup below: an unknown provider has no
        # default model either, so checking the model first reported the missing
        # model (a symptom) and buried the unknown provider (the cause).
        raise HostingUnknownError(
            _unknown_hosting_message(hosting, hosting_source), hosting, hosting_source
        )
    if is_decision_only(hosting):
        # A KNOWN provider that can never serve a chat completion (TypeSafe's
        # Jev: every host we reach it through rejects ``chat/completions``).
        # Refused HERE, at the same preflight as an unknown id, for the same
        # reason: this is the one point every front end classifies, so the
        # condition reaches the guided setup state where ``/model`` supplies a
        # chat provider, instead of booting a session that dies on its first
        # turn with a provider error nobody can read as "that hosting was never
        # chattable". NOT ``HostingUnknownError``: saying "not a known provider"
        # about a provider this build ships a login for would be false, and the
        # repair it prescribes (``/login``) is already done.
        _refuse_decision_only(hosting, hosting_source)
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
    (:func:`local_operator.harness.render._default_convert_to_llm`, reached here
    through ``local_operator.session.session``'s re-export). Two
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
        # ``sys.stdin`` is checked for ``None`` BEFORE ``isatty()`` is called
        # because a launcher or daemoniser may start the process with fd 0
        # CLOSED, and Python leaves ``sys.stdin`` as ``None`` for that shape
        # rather than raising on it. ``lop exec`` without ``--tools`` reaches
        # this gate on the ordinary path, so the unguarded call was not a
        # corner: it raised ``'NoneType' object has no attribute 'isatty'`` out
        # of the gate, and the loop answers a raising gate as an approval-gate
        # FAULT — the call still did not run (fail closed), but the operator was
        # told "This is a harness fault, not a refusal by the user" instead of
        # the actionable CL-04 notice below, which is the same notice a pipe
        # emits. An absent stdin has to read the way a pipe does: nobody can be
        # asked. Same guard, and the same reason, as ``exec_startup``'s
        # declaration gate at its ``sys.stdin`` test.
        stdin_is_tty = sys.stdin is not None and sys.stdin.isatty()
        if not stdin_is_tty:
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

    Picks the newest row the OPERATOR wrote. A harness notice (the
    ``harness_injected`` stamp, or one of the notice heads a compaction block
    carried forward — see ``harness/rows.py``) is stored as a
    ``role="user"`` row and is not a query: handing selection the harness's
    prose would search the skills index for "[model switch] You are now
    running as …" and freeze the result against it as the block's task id.
    """
    try:
        # Production transcripts index these at durable append time. Keep the
        # historical fallback for embedders/test stores exposing entries only.
        if hasattr(transcript, "latest_user_entry"):
            newest_user = transcript.latest_user_entry()
            candidates: Any = [transcript.latest_entry("compaction"), newest_user]
            if newest_user is not None and is_harness_notice_row(
                getattr(newest_user, "payload", None) or {}
            ):
                # The indexed path names ONE user row, so a notice there would
                # end the scan with nothing to select on. Fall back to the
                # journal and walk back to the newest row the operator wrote.
                candidates = transcript.entries()
            entries = [entry for entry in candidates if entry is not None]
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
            if is_harness_notice_row(payload):
                continue
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


@dataclass(frozen=True)
class InstructionSource:
    """One contributor to the assembled custom instructions, as assembled.

    The accounting half of :func:`resolve_user_instructions`, and the reason
    that function exists at all. ``lop config instructions`` answers "what
    instructions am I actually running" from these records rather than from its
    own reimplementation of the budget arithmetic below — a report derived from
    a second copy of that arithmetic would drift from the prompt on the first
    change to either, which is the same class of divergence (documentation
    disagreeing with the code in the same install) that issue #822 reported.

    ``chars`` is what the source held; ``included`` is what survived the
    collapse and the cap, so a row can state the difference rather than
    reporting the smaller number as the whole truth. ``path`` is ``None`` for
    the agent profile, whose prompt comes from the registry database and has no
    file an operator could open.
    """

    label: str
    path: Path | None
    chars: int
    included: int
    collapsed: bool
    truncated: bool
    #: The file exists but could not be read (permissions, a fifo swapped in).
    #: Distinct from a zero-length file: one is a mistake to fix, the other is
    #: the ordinary state of an install that simply has no such file, and
    #: reporting both as "empty" sends the operator to the wrong one.
    unreadable: bool = False
    #: 1-based ASSEMBLY INDEX of an earlier source this one shares text with,
    #: and how many characters that is. The superset arrangement — shared rules
    #: plus a lop-only overlay in ``system_prompt.md`` — is the case the digest
    #: collapse cannot catch, so both copies ride the cached prefix of every
    #: request. Without this the frame is identical to two genuinely distinct
    #: files, and "two rows with non-zero Included" is true of every healthy
    #: multi-source install, so it cannot be the diagnosis.
    #:
    #: An index rather than a label because labels are not unique: several
    #: override paths all render as ``imported``, so a label named a row the
    #: operator could not pick out of the box, and the remedy (edit one of these
    #: two files) needs exactly that.
    #:
    #: Containment rather than equality, and it does not double-report the
    #: collapse: an imported file byte-identical to ``system_prompt.md`` is
    #: dropped upstream and arrives with empty ``text``, so it is excluded from
    #: the test. An agent PROFILE equal to an earlier source is a different
    #: matter — profiles are never collapsed — and is flagged, correctly: both
    #: copies really are in the prompt.
    overlaps_index: int | None = None
    overlap_chars: int = 0
    #: Which way round the containment runs: ``True`` when THIS source holds all
    #: of the earlier one, ``False`` when this source sits wholly inside it. The
    #: two arrangements have different remedies — trim the superset, or delete
    #: the subset — so the row has to say which one the operator is looking at,
    #: and a single "these overlap" would send half of them to the wrong file.
    overlap_contains: bool = True


def resolve_user_instructions(
    agent_prompt: str = "",
    *,
    log_provenance: bool = True,
) -> tuple[str, list[InstructionSource]]:
    """Assemble the custom instructions AND account for where they came from.

    Split out of :func:`load_user_instructions` — whose docstring carries the
    behavioural contract — so the provenance surface reads the same assembly
    the prompt does. See :class:`InstructionSource` for why a second
    implementation of this arithmetic was not acceptable.

    The returned records are in ASSEMBLY order, which is the fact operators get
    wrong and the reason the report exists at all.

    ``log_provenance=False`` is for the ONE caller whose entire output already
    IS the provenance (``lop config instructions``): there the INFO record
    would print the same file and size to stderr immediately above the box
    reporting it, so the command would contradict nothing and repeat
    everything. Every session path leaves it on — the log is the only trace a
    running session leaves of an import it did not name.
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
    ecosystem_records = read_ecosystem_instructions(
        skip_digests=frozenset({content_digest(global_raw)} if global_raw else ())
    )
    if log_provenance:
        log_ecosystem_provenance(ecosystem_records)
    ecosystem_raw = "\n\n".join(record.text for record in ecosystem_records if record.text).strip()

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

    # The whole-source cap is reported per imported FILE rather than against
    # the joined block: with several override paths the operator needs to know
    # which file lost text, and the join is what the budget acts on. The budget
    # truncates that joined block from the TAIL, so a cut larger than the last
    # file also eats the tail of the file before it. Charging the whole cut to
    # the last contributor therefore credited earlier files with text that never
    # reached the prompt — rows summing to 107,998 "included" characters inside a
    # 64,000-character total, with the file that actually lost 44k carrying no
    # truncation flag. Walking the SURVIVING length forward instead reproduces
    # how the block was actually cut, so each file is credited only what its own
    # span contributed.
    #
    # ``remaining`` is measured against the ASSEMBLED block rather than the raw
    # one, which means the truncation marker rides with the file whose tail it
    # replaced. That is deliberate: it keeps ``sum(included) + separators ==
    # len(assembled)`` exactly true, and a report whose own rows do not add up to
    # its own total is the defect class this whole surface exists to close.
    remaining = len(ecosystem_text)
    emitted = False
    sources: list[InstructionSource] = []
    # What each source actually held, positionally parallel to ``sources``, for
    # the containment test below. Empty for a source that contributed nothing.
    raw_texts: list[str] = []
    for record in ecosystem_records:
        if record.text:
            if emitted:
                # The "\n\n" join before this file, charged to neither side.
                remaining = max(0, remaining - 2)
            included = min(record.chars, remaining)
            remaining -= included
            emitted = emitted or included > 0
        else:
            # Collapsed, empty or unreadable: contributed nothing, and consumed
            # no separator either.
            included = 0
        raw_texts.append(record.text)
        sources.append(
            InstructionSource(
                label="imported",
                path=record.path,
                chars=record.chars,
                included=included,
                collapsed=record.collapsed,
                # Truncated by the per-FILE 64 KiB read cap, or by the shared
                # instructions budget landing on this file. Guarded on
                # ``record.text`` so a collapsed file — which also has
                # ``included`` 0 against a non-zero ``chars`` — is not reported
                # as truncated on top of being reported as collapsed.
                truncated=record.truncated or (bool(record.text) and included < record.chars),
                unreadable=record.unreadable,
            )
        )
    raw_texts.append(global_raw)
    sources.append(
        InstructionSource(
            label="system_prompt.md",
            path=path,
            chars=len(global_raw),
            included=len(global_text),
            collapsed=False,
            truncated=len(global_text) < len(global_raw),
        )
    )
    if agent_raw:
        raw_texts.append(agent_raw)
        sources.append(
            InstructionSource(
                label="agent profile",
                path=None,
                chars=len(agent_raw),
                included=len(agent_text),
                collapsed=False,
                truncated=len(agent_text) < len(agent_raw),
            )
        )

    # Duplicate content the collapse cannot catch. The digest collapse is keyed
    # on the WHOLE file, so a native file that is a superset of the shared one
    # ships both copies in every cached request — the expensive arrangement, and
    # the one the frame could not previously distinguish from two healthy
    # distinct files. A plain containment test on text already in hand: no new
    # read, no new arithmetic, and CPython's substring search over a handful of
    # sources bounded at 64,000 characters is not work worth avoiding.
    #
    # SYMMETRIC, because the cost is. An earlier version asked only whether a
    # LATER source contained an EARLIER one, which is silent on the natural
    # migration: rules move into ``~/.agents/AGENTS.md`` and grow there while the
    # old ``system_prompt.md`` is left behind as a subset. Both copies ship in
    # every request and no row fired — and the guide read that silence as an
    # all-clear, which is issue #822's own shape (a claim the code does not
    # make). Measured at 41 µs for 64 KiB-in-64 KiB, so the second direction is
    # free.
    #
    # Restricted to sources that survived WHOLE (``included == chars``) so the
    # row can say both copies are sent without qualification: a source the
    # budget already cut carries a ``Truncated:`` row, and claiming a verbatim
    # duplicate of text that was itself partly dropped would be the report
    # asserting something the prompt does not do.
    whole = [
        (index, raw)
        for index, (source, raw) in enumerate(zip(sources, raw_texts))
        if raw and source.included == source.chars
    ]
    for position, (index, raw) in enumerate(whole):
        for earlier_index, earlier_raw in whole[:position]:
            # Equal-length texts satisfy both directions; "contains" is tried
            # first so an agent profile identical to an earlier source keeps
            # reading as the superset case rather than flipping on tie order.
            if len(earlier_raw) >= _OVERLAP_MIN_CHARS and earlier_raw in raw:
                contains, shared_chars = True, len(earlier_raw)
            elif len(raw) >= _OVERLAP_MIN_CHARS and raw in earlier_raw:
                contains, shared_chars = False, len(raw)
            else:
                continue
            sources[index] = replace(
                sources[index],
                # 1-based: the box numbers its rows from 1, and an index the
                # operator cannot match to a printed row is not an answer.
                overlaps_index=earlier_index + 1,
                overlap_chars=shared_chars,
                overlap_contains=contains,
            )
            break

    # Imported first, native second, profile last: later text is read as the
    # more specific instruction, so lop's own file outranks the shared one and
    # the chosen profile outranks both.
    assembled = "\n\n".join(part for part in (ecosystem_text, global_text, agent_text) if part)
    return assembled, sources


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
    return resolve_user_instructions(agent_prompt)[0]


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


#: ``values.classification.maxRecommendations`` as the WIRING needs it when no
#: seam was built from the package (a test double, or a host that supplied its
#: own classifier). The package's ``DEFAULT_MAX_RECOMMENDATIONS`` is the
#: consumer default the settings registry is pinned to; this copy exists so the
#: request path never has to import the package, and
#: ``tests/unit/test_session_factory_classification.py`` pins the two together.
DEFAULT_CLASSIFICATION_MAX_RECOMMENDATIONS = 3

#: ``values.classification.timeoutMs`` — the CALL's deadline, as the wiring needs it.
#:
#: The shipped service enforces the same number itself (``timeout_s``, read from the
#: same key), and that is the deadline the breaker counts against. The wiring keeps
#: its own copy for one purpose only — capping the turn's WAIT by it
#: (:func:`_classification_wait_s`), so a wait budget configured larger than the
#: call could ever take does not spend itself on nothing. It is NOT a deadline
#: around the call: killing a call the turn has stopped waiting for would throw away
#: an answer the next message could have used. Same pinning as above.
DEFAULT_CLASSIFICATION_TIMEOUT_MS = 1500

#: ``values.classification.waitMs`` — how long a TURN waits for a recommendation
#: before it stops waiting and lets the call finish in the background.
#:
#: This is the operator's latency budget in one number ("our own overhead under
#: 100 ms, ideally under 50 ms, per user message"), and it is deliberately NOT the
#: call's deadline: ``timeoutMs`` says how long the VENDOR may take, and on a real
#: roster that is ~250 ms median — waiting for it would put the vendor's model time
#: on the turn's critical path, which the budget explicitly excludes.
#:
#: WHAT IT ACTUALLY COSTS, measured on the real path (27-candidate roster, real
#: vendor, default settings, ``/tmp/classify_wait_probe.py``), and reported as the
#: DIFFERENCE against the layer being off so the pre-existing cost of the turn path
#: is not claimed as ours:
#:
#: - warm message: +2 to +4 ms median, +33 to +56 ms worst (the spread is the shared
#:   machine, not the code — the layer-OFF arm shows +24 to +28 ms worst in the same runs);
#: - cache hit: ~+2 to +4 ms;
#: - THE FIRST MESSAGE OF A SESSION is the expensive one, and it is honest to say so:
#:   +22 to +32 ms of our own time across runs
#:   (``scripts/classification_latency_probe.py``, 27-candidate roster, real vendor),
#:   because the first call pays one-off setup before its first await, where no wait
#:   budget can reach it. That cost is NOT explained by the client construction: the
#:   paired arms — with and without ``ClassificationService.warm_up``, which moves a
#:   measured 19-36 ms first ``httpx.AsyncClient`` to session build — came out level
#:   (30.7 vs 32.0 ms median), and ``build_state`` measures 0.05 ms. Open in the
#:   contract, not explained here.
#:
#: So: the ceiling holds on the steady state, and a session's first message costs a
#: few tens of milliseconds once. Anything slower than the wait is delivered by the
#: next message instead (see ``_harvest_classification``).
DEFAULT_CLASSIFICATION_WAIT_MS = 50

#: How many background classification calls may be outstanding before the oldest is
#: abandoned. A session that keeps sending messages into a vendor that never answers
#: would otherwise accumulate one live task per message: the service's own deadline
#: normally retires each of them, and the breaker stops new ones after three
#: failures, so this is the belt for a seam that does neither. Dropping a task is a
#: cancelled ADVISORY call, never a lost turn.
_MAX_OUTSTANDING_CLASSIFICATION_CALLS = 4


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
    #: The roots ``skills_by_name`` was discovered from, kept so the skill
    #: resolver can rescan THE SAME set on a miss. Recomputing them at resolve
    #: time would be subtly different: ``default_skill_roots`` filters
    #: ecosystem roots by existence, so a root created mid-session would change
    #: the list and quietly widen what the session scans.
    skill_roots: list[Path] = field(default_factory=list)
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
    #: Names of the configured MCP servers, published by ``_seed_mcp_routing``
    #: before any connection work. Read by the classification roster, which needs
    #: the server id for a candidate and never its tools.
    mcp_server_names: tuple[str, ...] = ()
    #: The classification seam (docs/design/classification-layer.md §7): an
    #: object exposing ``async recommend_resources(request) -> Recommendation``
    #: and ``notice(recommendation) -> str | None``. ``None`` means the layer is
    #: off, unavailable, or its package failed to import — and that the prompt is
    #: exactly what it was before this seam existed.
    #:
    #: Built ONCE per session by ``_attach_classification`` when
    #: ``values.classification.auto`` is on, rather than at the first user
    #: message: the package's cold import is ~1.8 s of cumulative import time
    #: (``python -X importtime``), and a turn's prompt build must not pay for an
    #: import. Tests inject a double here and never touch the package.
    classifier: Any | None = None
    #: Where the seam's one-line notice goes: the session's own notice event,
    #: bound by ``create_session`` once the facade exists (see
    #: :func:`attach_classification_notices`). ``None`` — a provider rendered
    #: without a session, as the benchmark preflight does — drops the notice
    #: rather than inventing a second notification channel.
    notice_sink: Callable[[str, str], Any] | None = None
    #: The classification roster (one row per candidate resource) and the inputs
    #: it was derived from. Built ONCE per roster, never per user message: the
    #: walk and the row allocation are session-shaped work, and re-deriving them
    #: on every turn is what would push the wiring's added latency toward the
    #: per-message budget (see :func:`_classification_roster`).
    classification_roster: tuple[Any, ...] | None = None
    #: ``(index, row count, mcp server names)`` — the identity of the inputs the
    #: cached roster was built from. The index OBJECT, not just its size: a
    #: rebuild replaces it, and its ``skills`` list is never mutated in place.
    classification_roster_key: tuple[Any, ...] | None = None
    #: ``values.classification.maxRecommendations`` as read at session build.
    #: Carried because the REQUEST's own cap field is an upper bound over the
    #: package's reader (``min(request, settings)``), so leaving it at the
    #: dataclass default would silently cap a configured 5 at 3.
    classification_max_recommendations: int = DEFAULT_CLASSIFICATION_MAX_RECOMMENDATIONS
    #: ``values.classification.waitMs`` in SECONDS, as read at session build (the
    #: same NEW_SESSIONS snapshot the service got). The turn waits at most this
    #: long; see :data:`DEFAULT_CLASSIFICATION_WAIT_MS` for why it is not the
    #: call's deadline.
    classification_wait_s: float = DEFAULT_CLASSIFICATION_WAIT_MS / 1000.0
    #: Calls that are STILL RUNNING after their own turn stopped waiting. Harvested
    #: on a later user message (never awaited mid-turn): each is a task the turn has
    #: already given up on, whose answer is worth keeping — a late recommendation is
    #: still a recommendation for the conversation it was computed against.
    classification_outstanding: list[Any] = field(default_factory=list)
    #: The resource urls announced on the PREVIOUS message, so an unchanged set can
    #: stay quiet rather than printing the same sentence again (design round 1, D7).
    classification_last_announced: tuple[str, ...] | None = None
    #: Recommendations that arrived too late for their own turn and have not reached
    #: a prompt yet, oldest first. Rendered by the NEXT ``_select_knowledge_block``
    #: (a new user message re-renders block 3, which the harness journals as a
    #: ``[session-state]`` update) and consumed exactly once there.
    classification_pending: list[Any] = field(default_factory=list)


#: The capability line a configured MCP server contributes when no release-owned
#: hint covers it. It is the SAME text ``mcp/resources.py``'s
#: ``render_mcp_suggestions`` uses for a custom server, restated because that
#: module spells it inline and there is no exported constant to import — and
#: because the alternative, parsing it back out of the rendered catalogue, would
#: make the classifier's option text depend on a template.
_MCP_DEFAULT_CAPABILITY = "Configured MCP server."

#: The advisory block's fixed furniture (§7). The wording is the contract's:
#: "may help", never imperative, never exclusive, and never a claim that a
#: recommended resource is authoritative for the turn. A wrong recommendation
#: must cost a line of context, not a wrong action.
_RECOMMENDATION_BLOCK_OPEN = "<resource_recommendations>"
_RECOMMENDATION_BLOCK_PREAMBLE = (
    "These may help with this request — read the ones that actually fit, ignore the rest:"
)
_RECOMMENDATION_BLOCK_CLOSE = "</resource_recommendations>"


@dataclass(frozen=True)
class _ClassificationCandidate:
    """One row of the roster the classifier is offered.

    Field-for-field the contract's ``Candidate`` (§4), and read by ATTRIBUTE on
    both sides, so the package's own dataclass is interchangeable with this one.
    The wiring carries its own row for two reasons that both outlast the
    implementation detail: the layer is optional, so the turn path must not
    import the package (the offline path is byte-identical down to its import
    graph), and the unit tests inject a classifier double that never sees the
    real types.

    ``description`` is HARNESS-OWNED text only (§6): a skill's or guide's own
    description as discovered from the local filesystem, or an MCP server's name
    plus a release-owned capability hint. Config-authored or remote-authored
    prose here would re-open the prompt-injection surface ``mcp/resources.py``
    deliberately excludes — the option text is the one part of the request the
    model reads as a rubric.
    """

    kind: str
    name: str
    description: str
    resource_url: str


@dataclass(frozen=True)
class _RecommendationRequest:
    """One classification pass: the user's message, optional context, the roster.

    Structurally the contract's ``RecommendationRequest`` (§4), including the
    field NAMES the package's service reads (``user_message``, ``context``,
    ``candidates``, ``max_recommendations``), for the reason
    :class:`_ClassificationCandidate` records.

    ``context`` is ``None`` here and that is the documented optional case (§5):
    a short already-redacted representative line would have to come from the
    transcript, and the hooks do not hold one — the provider does. With nothing
    to add, the state is the user message plus the roster, which is exactly what
    the layer says it does when a caller has no context to give.
    """

    user_message: str
    context: str | None
    candidates: tuple[_ClassificationCandidate, ...]
    max_recommendations: int


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
        # Published for the classification roster, which needs a server's NAME
        # (plus the harness-owned capability hint) as a candidate. Set beside the
        # catalogue closure so both read the same discovery result, and before
        # any connection work, so a first-turn classification is complete on a
        # cold cache.
        hooks.mcp_server_names = names
    except Exception:  # noqa: BLE001 — MCP hints remain optional enrichment
        logger.debug("early MCP name discovery failed", exc_info=True)


async def _setup_knowledge(
    credential_manager: CredentialManager,
    config_dir: Path,
    agent_registry: AgentRegistry,
    warnings_out: list[str],
    cwd: str | Path | None = None,
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

        # The SESSION's cwd, not the process's. A session created with an
        # explicit cwd (bootstrap, the scheduler, owned runtimes) otherwise
        # discovered project-local skills for whatever directory the process
        # happened to start in -- every other consumer here already takes the
        # session's cwd, and this one silently did not.
        hooks.skill_roots = default_skill_roots(Path(cwd) if cwd is not None else None)
        skills, discovery_warnings = discover_skills(hooks.skill_roots)
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


def _classification_section(config_manager: ConfigManager) -> Mapping[str, Any]:
    """``values.classification`` as it stands at session build.

    A SNAPSHOT, deliberately, and not the live mapping the stream fn holds: the
    section is scoped NEW_SESSIONS in ``settings_io`` (the service is built per
    session), so an edit lands on the next session and the page's tag is true.
    """
    values = _classification_values(config_manager)
    section = values.get("classification") if isinstance(values, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def _classification_values(config_manager: ConfigManager) -> dict[str, Any]:
    """A shallow snapshot of ``config.yml``'s ``values`` for the layer.

    The package reads ``settings.get("classification", {})`` — the same shape
    ``values.effort.auto`` established (``model/effort_classifier.py``) — so the
    whole values mapping is what it is handed.
    """
    values = getattr(config_manager.get_config(), "values", None)
    return dict(values) if isinstance(values, Mapping) else {}


def _classification_enabled(section: Mapping[str, Any]) -> bool:
    """Whether ``values.classification.auto`` turns the layer on.

    THE ONE decision the wiring makes for itself, and it is read WITHOUT
    importing the package: this runs at session build, and ``auto: false`` (the
    default) must leave the process's import graph exactly as it was before the
    layer existed. The two values it can return are pinned by
    ``tests/unit/test_session_factory_classification.py`` to the package's
    ``DEFAULT_AUTO`` and the registry row's default.

    Read through ``settings_io.strict_bool``, the same reading the service's own
    ``enabled`` property applies, so a hand-edited ``auto: "false"`` is off here
    and there — two readings of one toggle is how a switch ends up honoured by
    one path and ignored by another.
    """
    raw = section.get("auto")
    if raw is None:
        # The absent case is answered without the settings_io import, which is
        # the whole point of this function existing separately.
        return False
    from local_operator.settings_io import strict_bool

    return strict_bool(raw, False)


def _attach_classification(
    hooks: _KnowledgeHooks,
    config_manager: ConfigManager,
    credential_manager: CredentialManager,
    warnings_out: list[str],
) -> None:
    """Build the classification seam when the layer is switched on (§7, §9).

    Built HERE, at session construction, rather than lazily on the first user
    message — and that ordering is a latency decision, not a style one: the
    package's cold import measured ~1.8 s of cumulative import time
    (``python -X importtime`` over ``local_operator.classification``), which must
    not land inside a turn's prompt build. Session construction already spends
    seconds on skill discovery and embeddings, so the one-off cost sits where the
    operator is already waiting. A default install never imports the package at
    all, because ``auto`` is read first and without it.

    The seam's keep-alive CLIENT is warmed here too, for the same reason and with a
    number: see ``ClassificationService.warm_up`` (19-36 ms of SSL-context setup, once
    per process, otherwise paid by a session's first call; the same docstring records
    that the first message did not measurably get faster in paired runs).

    Degrades rather than failing the boot, exactly as the sibling knowledge
    wiring does: a layer that cannot be built is a line in ``warnings_out`` and a
    ``classifier`` of ``None``, which is byte-for-byte today's prompt.
    """
    section = _classification_section(config_manager)
    if not _classification_enabled(section):
        return
    values = _classification_values(config_manager)
    try:
        from local_operator.classification import (
            ClassificationService,
            max_recommendations,
            setting_int,
        )

        hooks.classifier = ClassificationService(manager=credential_manager, settings=values)
        # …and its keep-alive client is built HERE, not on the first message: a
        # measured 19-36 ms of SSL-context setup (first construction in a process),
        # paid before the call's first await, so no wait budget can bound it. No
        # connection is opened — the object only — and the paired-run caveat on what
        # this buys is in ``ClassificationService.warm_up``.
        #
        # ``getattr`` and a try of its own: warming is an OPTIMISATION, so a seam that
        # does not publish it (the seam contract is ``recommend_resources`` +
        # ``notice``, and the tests inject exactly that) keeps working, and a warm-up
        # that fails must not cost the layer — the shared handler below would turn
        # both into ``classifier = None``, i.e. a session with no classification
        # because a prewarm went wrong.
        warm_up = getattr(hooks.classifier, "warm_up", None)
        if callable(warm_up):
            try:
                warm_up()
            except Exception:  # noqa: BLE001 — the layer still works, just colder
                logger.debug("classification: prewarm failed", exc_info=True)
        # The REQUEST's own cap field is an upper bound over the service's reader
        # (``min(request, settings)``), so it has to be the configured value:
        # left at the dataclass default it would silently cap a configured 5 at
        # 3. Read through the package's own reader, in the one branch that has
        # already imported the package.
        hooks.classification_max_recommendations = max_recommendations(values)
        # ``waitMs`` through the SAME reader the package uses for its integers
        # (``setting_int``): a hand-edited ``waitMs: "80"`` is a typo we can read,
        # ``waitMs: true`` is refused (``True`` is an ``int`` in Python, and a
        # boolean there would mean a 1 ms wait), and ``0`` means "use the default"
        # exactly as it does for ``timeoutMs``. This key is the WIRING's own — the
        # package never reads it — but it is a §8 number and is parsed like one.
        hooks.classification_wait_s = (
            setting_int(values, "waitMs", DEFAULT_CLASSIFICATION_WAIT_MS) / 1000.0
        )
    except Exception as exc:  # noqa: BLE001 — the layer is optional enrichment
        warnings_out.append(f"Resource classification unavailable: {exc}")
        hooks.classifier = None


def _classification_roster(hooks: _KnowledgeHooks) -> tuple[_ClassificationCandidate, ...]:
    """The candidate roster, cached against the inputs it was derived from.

    ONCE PER ROSTER, never once per user message. The walk below and the row it
    allocates per resource are session-shaped work; re-deriving them on every
    turn is exactly the per-message overhead the latency budget forbids, and it
    buys nothing — the inputs cannot change without the index object or the
    configured server names changing, which is what the cache key records. A warm
    turn therefore serializes the user message and nothing else.
    """
    index = hooks.index
    key: tuple[Any, ...] = (
        index,
        len(getattr(index, "skills", ()) or ()),
        hooks.mcp_server_names,
    )
    if hooks.classification_roster is not None and hooks.classification_roster_key == key:
        return hooks.classification_roster
    roster = _build_classification_roster(hooks)
    hooks.classification_roster = roster
    hooks.classification_roster_key = key
    return roster


def _build_classification_roster(hooks: _KnowledgeHooks) -> tuple[_ClassificationCandidate, ...]:
    """One row per resource the router itself can see (§7 step 1).

    Hidden skills are skipped because the router skips them
    (``SkillIndex.select`` filters ``hide``) — offering the model a resource the
    harness will not select would suggest a capability that does not exist. A row
    without a description is dropped for the same reason the index drops one: the
    description IS the routing signal, and an unnamed option is a coin flip.
    """
    from local_operator.skills.protocol import resource_url

    rows: list[_ClassificationCandidate] = []
    for resource in getattr(hooks.index, "skills", ()) or ():
        if getattr(resource, "hide", False):
            continue
        kind = str(getattr(resource, "resource_type", "") or "")
        if kind not in ("skill", "guide"):
            continue
        name = str(getattr(resource, "name", "") or "")
        description = str(getattr(resource, "description", "") or "")
        if not name or not description:
            continue
        rows.append(_ClassificationCandidate(kind, name, description, resource_url(kind, name)))
    for server in hooks.mcp_server_names:
        name = str(server or "")
        if not name:
            continue
        rows.append(
            _ClassificationCandidate(
                "mcp", name, _mcp_capability_hint(name), resource_url("mcp", name)
            )
        )
    return tuple(rows)


def _mcp_capability_hint(server: str) -> str:
    """The harness-owned capability line for one configured MCP server (§6).

    ``_CAPABILITY_HINTS`` is imported rather than restated: it is release-owned
    routing authority (``mcp/resources.py``'s module docstring is explicit that
    neither config nor remote servers may supply this text), and a second copy
    here would be free to drift from the one the catalogue renders. The private
    name is the only access there is; adding a public accessor would mean editing
    a module outside this slice, and the import is what keeps the two readings of
    "what this server is for" identical.
    """
    from local_operator.mcp.resources import _CAPABILITY_HINTS

    return _CAPABILITY_HINTS.get(server.casefold(), _MCP_DEFAULT_CAPABILITY)


def _classification_request(hooks: _KnowledgeHooks, query: str) -> _RecommendationRequest:
    """The request for one user message, over the CACHED roster."""
    return _RecommendationRequest(
        user_message=query,
        context=None,
        candidates=_classification_roster(hooks),
        max_recommendations=hooks.classification_max_recommendations,
    )


def _classification_deadline_s(service: Any) -> float:
    """The harness-side ceiling on one classification CALL, in seconds.

    The shipped service publishes ``timeout_s`` (``values.classification.timeoutMs``)
    and enforces it internally. The wiring knows the same number for a different
    reason: it caps the turn's WAIT by it (see :func:`_classification_wait_s`), and
    it is what a seam that publishes no deadline of its own is taken to honour. A
    seam that ignores it cannot hold a turn open, because the turn stops waiting at
    ``waitMs`` — which is the guarantee this layer actually owes a user message.
    """
    value = getattr(service, "timeout_s", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return DEFAULT_CLASSIFICATION_TIMEOUT_MS / 1000.0


def _classification_wait_s(hooks: _KnowledgeHooks, service: Any) -> float:
    """How long THIS turn waits for an answer, in seconds.

    ``values.classification.waitMs``, capped by the call's own deadline: waiting
    longer than the call could possibly take would spend budget on nothing.
    """
    return min(hooks.classification_wait_s, _classification_deadline_s(service))


async def _classification_call(service: Any, request: Any) -> Any:
    """One service call, with its cost logged WHERE IT LANDS.

    A task body rather than an inline await, because on a real roster the vendor's
    ~250 ms outlives the turn's 50 ms wait: the log line has to be written by
    whoever finishes the call, not by a turn that has already moved on. There is no
    error handling here on purpose — ``recommend_resources`` never raises (except
    cancellation, which propagates), and a foreign seam that raises leaves a task
    whose exception :func:`_harvest_classification` drops.
    """
    recommendation = await service.recommend_resources(request)
    _log_classification_cost(recommendation)
    return recommendation


def _prune_outstanding(hooks: _KnowledgeHooks) -> None:
    """Bound the background calls one session may leave running.

    See :data:`_MAX_OUTSTANDING_CLASSIFICATION_CALLS`: the oldest is abandoned
    (cancelled, never awaited) rather than kept, because it is an advisory call
    whose turn is long gone and the alternatives are worse — piling one live task
    per message against a vendor that never answers, or blocking the turn on the
    answer the budget just said to skip.
    """
    while len(hooks.classification_outstanding) > _MAX_OUTSTANDING_CLASSIFICATION_CALLS:
        abandoned = hooks.classification_outstanding.pop(0)
        if not abandoned.done():
            abandoned.cancel()


def _task_outcome(task: Any) -> Any | None:
    """A finished task's result, or ``None`` for anything else it could be.

    ``CancelledError`` included: an abandoned or cancelled call has no outcome to
    deliver, and reading it must not raise into the turn doing the harvesting.
    """
    try:
        return task.result()
    except BaseException:  # noqa: BLE001 — cancellation, a seam fault, anything
        return None


def _harvest_classification(hooks: _KnowledgeHooks) -> None:
    """Move every FINISHED background call into the pending slot. Never blocks.

    Called once per user message, before the block is rendered. ``done()`` is the
    only test that matters: a turn must never wait for a call it already gave up
    on, so a call still running stays outstanding and is looked at again next
    message. A recommendation with no resources is dropped here rather than
    queued — it has nothing to deliver, and a session whose answers are always
    empty must append nothing at all.
    """
    if not hooks.classification_outstanding:
        return
    running: list[Any] = []
    for task in hooks.classification_outstanding:
        if not task.done():
            running.append(task)
            continue
        recommendation = _task_outcome(task)
        if recommendation is not None and getattr(recommendation, "resources", ()):
            hooks.classification_pending.append(recommendation)
    hooks.classification_outstanding = running


async def _classification_recommendation(hooks: _KnowledgeHooks, query: str) -> Any | None:
    """One classification pass for one user message, waited on for ``waitMs``. NEVER raises.

    Runs INSIDE the gather that also runs the embedder selection (see
    :func:`_select_knowledge_block`), so everything synchronous in it — the
    roster lookup, the service's state build and its serialization — is paid
    CONCURRENTLY with the selection rather than before or after it. That is what
    keeps the added wall-clock to the difference instead of the sum, and it is
    why the request is built here rather than by the caller.

    THE TURN'S PATIENCE IS NOT THE CALL'S DEADLINE. The turn waits
    ``values.classification.waitMs`` (50 ms by default); the call is bounded by
    ``values.classification.timeoutMs``, which the service enforces itself. When
    the answer misses the wait, the turn does NOT pay the difference: the call is
    left running (``shield``, so the wait's own cancellation cannot reach it),
    kept in ``classification_outstanding``, and delivered by a later message. Two
    properties follow:

    - the added wall-clock per user message is bounded by the wait, whatever the
      vendor does — measured against a real roster the vendor takes ~250 ms, and
      the budget explicitly excludes that model time;
    - the breaker keeps counting the VENDOR's deadline, once per call, inside the
      service. The outer deadline this replaced started microseconds before the
      service's own, so when it won, the service's ``_record_failure`` landed
      after the turn had already been handed its empty block, and the warm-up
      reported three failures across four messages (QA round 1, Q1).

    A cancelled WAIT is deliberately not a cancelled call: "this turn has waited
    enough" and "throw that work away" are different statements, and only the
    second is a user's cancel.
    """
    service = hooks.classifier
    if service is None:
        return None
    try:
        request = _classification_request(hooks, query)
    except Exception:  # noqa: BLE001 — a roster fault must not fail a turn
        logger.warning("classification: could not build the request", exc_info=True)
        return None
    wait_s = _classification_wait_s(hooks, service)
    task = asyncio.create_task(_classification_call(service, request))
    hooks.classification_outstanding.append(task)
    _prune_outstanding(hooks)
    try:
        recommendation = await asyncio.wait_for(asyncio.shield(task), timeout=wait_s)
    except asyncio.TimeoutError:
        # NOT a failure and NOT an empty answer: the call is still in flight and
        # its result is collected by a later user message. The prompt is unchanged
        # either way (§7), so the only thing worth saying is where the answer went.
        logger.info(
            "classification: no recommendation within %.0f ms; the turn continues "
            "without one and the answer rides the next user message",
            wait_s * 1000,
        )
        return None
    except asyncio.CancelledError:
        # A cancelled TURN is a user's cancel, and they mean the WORK is over, not
        # merely that this turn stopped waiting: the answer would otherwise be
        # delivered onto the next message, which is how a "stop" quietly produces a
        # recommendation for the question the user walked away from. (The WAIT's own
        # timeout is the branch above, and it deliberately does NOT do this — "this
        # turn has waited enough" and "throw that work away" are different
        # statements. Review round 2, NIT 1: the comment here used to claim that
        # distinction while the code left the task running either way.)
        task.cancel()
        hooks.classification_outstanding = [
            outstanding
            for outstanding in hooks.classification_outstanding
            if outstanding is not task
        ]
        raise
    except Exception:  # noqa: BLE001 — the layer may never fail a turn (§4)
        logger.warning("classification: recommendation failed", exc_info=True)
        return None
    # Delivered to THIS turn, so the harvest must not deliver it a second time.
    hooks.classification_outstanding = [
        outstanding for outstanding in hooks.classification_outstanding if outstanding is not task
    ]
    return recommendation


def _classification_block(
    hooks: _KnowledgeHooks,
    recommendation: Any,
    *,
    picked: Sequence[Skill],
    catalogue: str,
    already: set[str] | None = None,
    limit: int | None = None,
    rendered_urls: list[str] | None = None,
) -> str:
    """Render §7's advisory block for ONE recommendation, or ``""``.

    THE WIRING RENDERS THIS, not the package's ``render_block``, and dedupe is
    the whole reason: dropping what the prompt already contains needs to know
    what the embedder selected and what the MCP catalogue already advertises, and
    the wiring is the only layer that knows either. The package keeps its own
    renderer for callers with nothing to dedupe against.

    Anything already selected is dropped rather than demoted: a line telling the
    model to read what the skills block has just told it to read immediately is
    noise that costs context and teaches nothing.

    ``already`` is the SAME test one step further out: resources an earlier
    section of THIS prompt already carries. One turn can deliver two answers (the
    one a previous message harvested, then this message's own), and a shared
    ``limit`` — the per-message cap minus what has already been spent — is what
    keeps the pair inside ``maxRecommendations`` instead of doubling it.

    ``rendered_urls`` comes back through an OUT-LIST rather than a tuple return, and
    the reason is outside this module: ``tests/unit/classification/test_block_parity.py``
    compares this function's return value against the package's ``render_block`` line
    for line, and that comparison is worth more than a tidier signature. The caller
    owes the NOTICE an honest account of what was appended — the line it builds from
    the recommendation names every resource the recommendation carried, which the
    dedupe or the cap may have dropped — and this is how it learns the difference.
    """
    from local_operator.skills.protocol import resource_url

    rendered: set[str] = set(already or ())
    for resource in picked:
        kind = str(getattr(resource, "resource_type", "") or "")
        name = str(getattr(resource, "name", "") or "")
        if name and kind in ("skill", "guide"):
            rendered.add(resource_url(kind, name))
    budget = hooks.classification_max_recommendations if limit is None else limit
    if budget <= 0:
        return ""
    urls: list[str] = []
    for resource in getattr(recommendation, "resources", ()) or ():
        url = str(getattr(resource, "resource_url", "") or "")
        if not url or url in rendered or url in urls:
            continue
        # The catalogue advertises at most one server, by exactly this URL, so a
        # recommendation repeating it would spend a line on something the same
        # prompt already says.
        if catalogue and url in catalogue:
            continue
        urls.append(url)
        if len(urls) >= budget:
            break
    if not urls:
        return ""
    if rendered_urls is not None:
        rendered_urls.extend(urls)
    lines = [
        _RECOMMENDATION_BLOCK_OPEN,
        _RECOMMENDATION_BLOCK_PREAMBLE,
        *(f"- {url}" for url in urls),
        _RECOMMENDATION_BLOCK_CLOSE,
    ]
    return "\n".join(lines)


def _delivered_view(recommendation: Any, urls: Sequence[str], *, late: bool) -> Any:
    """The recommendation AS DELIVERED: what reached the prompt, and when it was asked for.

    Handed to the seam's ``notice()`` instead of the original, for two reasons and
    both are honesty rather than polish:

    - the line it builds lists ``recommendation.resources``, and a line naming a
      resource the dedupe or the cap dropped would be the over-claim the
      delivery-time notice exists to avoid;
    - ``late`` says the answer missed its own turn's wait and is being delivered by
      THIS message, which the line has to say out loud or it reads as advice about
      the message it now sits under (design round 1, D2). On a real vendor that is
      the ordinary case, not the edge one: the wait is 50 ms against a ~250 ms
      answer.

    Returned as the original object when neither fact changes anything (nothing
    dropped, nothing late), so a host's own classifier keeps its own type unless it
    really was trimmed. A seam whose recommendation is not a dataclass keeps the
    untrimmed one: the line may then be optimistic by a resource, which is a
    smaller lie than failing the notice.
    """
    resources = tuple(getattr(recommendation, "resources", ()) or ())
    kept = tuple(
        resource
        for resource in resources
        if str(getattr(resource, "resource_url", "") or "") in set(urls)
    )
    if len(kept) == len(resources) and not late:
        return recommendation
    try:
        return replace(recommendation, resources=kept, late=late)
    except Exception:  # noqa: BLE001 — a foreign seam's result is not a dataclass
        return recommendation


def _log_classification_cost(recommendation: Any) -> None:
    """Record what one pass cost, at INFO, on the vendor's own figures.

    NOT accrued into ``Session.accrue_spend``, and that omission is deliberate
    rather than an oversight. That path is the frontend store's per-CALL
    accounting: it moves ``last_identity`` (which the status band reads as the
    model that priced the session), it feeds the turn-end remainder the store
    reconciles, and ``accrue_spend`` bumps ``_spend_live_calls`` — the flag that
    CANCELS the one-time ledger rebuild for a pre-ledger session. Writing a
    decision call through it would suppress that rebuild, quietly dropping the
    restored history's dollars from a resumed session's total. The cost is
    therefore logged here in the vendor's own units, and the accounting path is
    left to the slice that owns it; the contract's ``Recommendation`` carries no
    model id or token counts, so those fields print ``-`` unless the seam adds
    them.
    """
    vendor = getattr(recommendation, "vendor", None)
    if not vendor:
        # No leg answered, so nothing was spent; the skip reason is the useful
        # line and it is logged at debug because it is the ordinary state of a
        # machine with no decision credential.
        logger.debug(
            "classification: no recommendation (skipped=%s)",
            getattr(recommendation, "skipped", None),
        )
        return
    cost = getattr(recommendation, "cost_usd", None)
    latency = getattr(recommendation, "latency_s", 0.0)
    logger.info(
        "classification: vendor=%s model=%s tokens=%s/%s cost=%s latency=%.3fs resources=%d",
        vendor,
        getattr(recommendation, "model", "-"),
        getattr(recommendation, "input_tokens", "-"),
        getattr(recommendation, "output_tokens", "-"),
        f"${cost:.6f}" if isinstance(cost, (int, float)) else "-",
        float(latency) if isinstance(latency, (int, float)) else 0.0,
        len(getattr(recommendation, "resources", ()) or ()),
    )


async def _emit_classification_notice(hooks: _KnowledgeHooks, recommendation: Any) -> bool:
    """Emit the seam's one-line notice through the session's own notice event.

    Called once per admitted user message — the same cadence the frozen
    knowledge block already has, so no separate "once per message" bookkeeping
    is needed. The GATE is the seam's own: the shipped service applies
    ``values.classification.notice`` inside ``notice()`` (it returns ``None``
    when the key is off, or when there is nothing to announce), so the harness
    does not read that key a second time and cannot disagree with it.

    RETURNS whether a line was actually handed to the sink, and the caller uses
    that to decide what to remember: the D7 repeat-suppression key may only be
    updated for a line the user really saw, or a notice suppressed by its own
    gate (``notice`` off, or a seam that renders nothing) would silence the next
    message's identical set too — a small version of the failure the harness has
    already paid for once, where a suppressed paint was recorded as delivered.

    A MISSING SINK IS NOT A FALLBACK POINT: a provider rendered without a
    session (the benchmark preflight) or a host that never bound one simply gets
    no notice, which is better than inventing a second notification channel that
    only some front ends paint.
    """
    seam = hooks.classifier
    sink = hooks.notice_sink
    notice = getattr(seam, "notice", None)
    if sink is None or not callable(notice):
        return False
    try:
        line = notice(recommendation)
        if inspect.isawaitable(line):
            line = await line
    except Exception:  # noqa: BLE001 — a notice is never worth a turn
        logger.debug("classification: notice rendering failed", exc_info=True)
        return False
    if not line:
        return False
    try:
        delivered = sink(str(line), "info")
        if inspect.isawaitable(delivered):
            await delivered
    except Exception:  # noqa: BLE001 — a notice is never worth a turn
        logger.debug("classification: notice delivery failed", exc_info=True)
        return False
    return True


async def _empty_selection() -> list[Skill]:
    """The selection leg of the gather when the session has no index at all."""
    return []


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

    The classification layer rides the SAME cadence (docs/design/
    classification-layer.md §7): this function already runs once per admitted
    user message per ``task_id`` + ``compaction_id``, which is exactly the
    once-per-message cadence the layer wants, so it needs no freeze machinery of
    its own. Its call is gathered with the selection below, its block is appended
    after the catalogue, and when it is off, unavailable, unanswered or empty the
    joined block is byte-identical to what this function returned before the
    layer existed (asserted in ``tests/unit/test_session_factory_classification.py``).

    The turn only WAITS ``values.classification.waitMs`` for the answer. An answer
    that arrives later is not lost: the next admitted user message re-renders this
    block (that is what the per-message cadence buys here) and the late answer is
    appended then, which the harness journals as a ``[session-state]`` update —
    the existing channel for late host state, so nothing here has to rewrite the
    cached prefix or invent a second notification path.
    """
    if (
        hooks.frozen_block is not None
        and hooks.frozen_compaction_id == compaction_id
        and hooks.frozen_task_id == task_id
    ):
        return hooks.frozen_block

    # BEFORE the gather, and before anything can return early: a call that finished
    # while this session was idle belongs to the prompt this turn is about to
    # build. A ``done()`` check per outstanding call — never a wait.
    _harvest_classification(hooks)

    picked: list[Skill] = []
    recommendation: Any | None = None
    query = query.strip()
    if query:
        # cwd rides as a keyword ONLY when set: test doubles and alternate
        # index shapes in the wild implement ``select(query, ...)`` with the
        # historical signature, and there is no globs matching to do without
        # a cwd anyway.
        select_kwargs: dict[str, Any] = {"cwd": Path(cwd)} if cwd else {}
        if hooks.classifier is not None:
            # ONE gather, so the classification's latency is the DIFFERENCE
            # against the embedder selection rather than the sum (§7 step 2) —
            # and the request build, which is where the package serializes the
            # state, happens inside the gathered coroutine for the same reason.
            #
            # The classification coroutine bounds ITS own wait and never raises,
            # so a vendor outage costs the recommendation and nothing else; the
            # selection leg keeps its existing contract exactly, so its exception
            # still propagates to the provider's guard. Nothing here wraps the
            # gather in a deadline of its own: a deadline over the gather would
            # either wait for the classification (the budget) or abandon the
            # selection (the prompt).
            selection = (
                hooks.index.select(query, **select_kwargs)
                if hooks.index is not None
                else _empty_selection()
            )
            selected, recommendation = await asyncio.gather(
                selection, _classification_recommendation(hooks, query)
            )
            picked = selected
        elif hooks.index is not None:
            picked = await hooks.index.select(query, **select_kwargs)
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
    catalogue = ""
    if hooks.mcp_catalogue is not None:
        catalogue = hooks.mcp_catalogue(query)
        if catalogue:
            sections.append(catalogue)
    # DELIVERY. Appended LAST, after the skills block and the catalogue, because
    # this is the weakest claim in the prompt: advisory, deduped against both, and
    # the one thing a model may ignore in full.
    #
    # Late answers go first — they are older, they were computed for an EARLIER
    # message, and letting this turn's fresh answer take the per-message cap first
    # would starve them for as long as the vendor keeps missing the wait. The
    # pending list is consumed here, so an answer reaches a prompt exactly once.
    #
    # The BLOCKS are per answer (each is that answer's own text, capped and deduped
    # in order), but the NOTICE is per MESSAGE: all the resources this prompt gained
    # are announced on one line, which is what §7's "once per user message" says and
    # what the previous shape broke — a message that delivered a late answer AND its
    # own printed two lines (review round 2, MINOR 1). Late first, so the line reads
    # in the order the sections appear.
    pending, hooks.classification_pending = hooks.classification_pending, []
    answers: list[tuple[Any, bool]] = [(answer, True) for answer in pending]
    if recommendation is not None:
        answers.append((recommendation, False))
    carried: set[str] = set()
    announced: list[Any] = []
    announced_urls: list[str] = []
    announced_late = False
    for answer, late in answers:
        urls: list[str] = []
        block = _classification_block(
            hooks,
            answer,
            picked=picked,
            catalogue=catalogue,
            already=carried,
            limit=hooks.classification_max_recommendations - len(carried),
            rendered_urls=urls,
        )
        if not block:
            # Nothing survived the dedupe or the cap: the prompt already carries
            # every resource this answer named, so there is nothing to append —
            # and, one line down, nothing to announce either.
            continue
        sections.append(block)
        carried.update(urls)
        announced.append(answer)
        announced_urls.extend(urls)
        # LATENESS is a property of the LINE, not of any one answer: if any of the
        # resources on it came from an earlier message, the attribution has to say
        # so, or the sentence reads as advice about the message it sits under.
        announced_late = announced_late or late
    if announced and not _already_announced(hooks, announced_urls):
        # THE NOTICE RIDES DELIVERY, not the call: it is emitted here, once, for the
        # resources this prompt actually gained. A call that missed the wait is not
        # announced when it is abandoned (nothing was delivered then) or when it
        # lands, but when a prompt carries it; and the seam's own gate
        # (``values.classification.notice``) stays inside ``notice()``. The view is
        # what keeps the line from naming a resource the dedupe or the cap dropped,
        # and ``late`` is what stops it reading as an answer to the wrong question.
        #
        # Remembered only once the paint really happened: the D7 key means "the set
        # the user last saw", and a line its own gate suppressed was not seen.
        if await _emit_classification_notice(
            hooks,
            _delivered_view(announced[-1], announced_urls, late=announced_late),
        ):
            hooks.classification_last_announced = tuple(announced_urls)
    elif announced:
        logger.debug("classification: the same resources were just announced; staying quiet")
    hooks.frozen_block = "\n\n".join(sections)
    hooks.frozen_compaction_id = compaction_id
    hooks.frozen_task_id = task_id
    return hooks.frozen_block


def _already_announced(hooks: _KnowledgeHooks, urls: Sequence[str]) -> bool:
    """Whether the SAME resource set was announced on the previous message.

    Design round 1, D7: with the money tail gone, four consecutive messages that
    recommend the same skill printed four byte-identical rows. A repeat of the
    previous announcement says nothing new — the resource is in this prompt either
    way, and the block above is what the model reads — so the line is suppressed,
    not the delivery. Only the IMMEDIATELY preceding set counts: a resource that
    comes back later in the session is worth its line again, because by then the
    user has read other things.

    A PREDICATE, and deliberately not the recorder: it answers the question and
    leaves ``classification_last_announced`` alone, because what that field means is
    "the set the user last SAW". The caller updates it after the paint succeeds
    (review round 3, NIT 1) — recording here marked a line as shown even when the
    seam's own gate or a failed delivery meant nothing was painted, which silenced
    the next message's identical set as though it had been announced.
    """
    signature = tuple(urls)
    return bool(signature) and signature == hooks.classification_last_announced


def _make_knowledge_resolver(hooks: _KnowledgeHooks) -> Callable[[str], str | None]:
    """Chain lazy knowledge protocols without mixing their namespaces."""
    from local_operator.guides import make_guide_resolver
    from local_operator.skills.api import make_skill_resolver

    guide_resolver = make_guide_resolver(hooks.guides_by_name)
    # Passing the roots turns on the miss-path rescan, which is what makes a
    # skill authored mid-session readable -- here and in subagents already
    # running, because they inherit this closure and it mutates
    # ``hooks.skills_by_name`` IN PLACE. Guides are packaged release resources
    # and cannot change at runtime, so the guide resolver takes no roots.
    skill_resolver = make_skill_resolver(hooks.skills_by_name, hooks.skill_roots)

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

    from local_operator.analytics.backfill import (
        backfill_analytics_session_daily,
        backfill_analytics_session_names,
    )
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
        # Re-derive the per-session day rollup ``aggregate()`` reads. It is
        # created EMPTY on the release that ships it while the ledger already
        # holds up to 90 days of calls, so without this pass the panel's first
        # read after upgrading is still the ledger's 5-13 s scan. Bounded
        # chunked transactions, newest-first, resumable, and off the event loop
        # like every other pass here — see the pass's own docstring for what a
        # user sees while it is incomplete (nothing: refused windows are
        # answered by the ledger, never by a partial total).
        (
            "analytics session-daily rollup backfill",
            lambda: backfill_analytics_session_daily(config_dir),
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
    # Resolving a saved model now reads its journal, which can be large. Keep
    # agent/profile I/O and identity resolution off the loop together, before
    # the contiguous lease/claim critical section below. Snapshot the command
    # first so /new or /resume cannot retarget an in-flight factory at the await.
    args = argparse.Namespace(**vars(args))

    # The configured birth-default effort (``model_effort``). Imported here, in
    # the same local style as ``configure_model`` below, because this module is
    # on the startup import path that ``test_import_graph`` guards and the
    # reader is only needed once per session build.
    from local_operator.model.effort import configured_effort

    def resolve_birth():
        agent = resolve_agent(args, agent_registry)
        hosting, model_name, model_source = resolve_hosting_model_with_source(
            agent, args, config_manager
        )
        # Read in the same OFF-LOOP thread as the model resolution (a config read
        # is cheap, but there is no reason to hop back to the loop for it). This
        # is a BIRTH default only: the journal restore in ``Session.__init__``
        # runs after the spec is built and re-derives it (clamping), so a
        # conversation resumed on a stored selection outranks this value by
        # design — ``model_effort`` must not fight the per-conversation journal
        # (design §0.1).
        #
        # ``birth_effort`` is the deliberate per-launch choice (the desktop
        # draft's chip) and outranks the configured default for THIS
        # construction only; it is a separate name from the CLI's ``effort``
        # because that one is applied after construction by ``exec_session`` and
        # raises where this one must clamp (see ``spawn_owned_session``). Only
        # an explicit selection carries it, so every other caller — and every
        # session that omits a model — reads the configured default unchanged.
        return (
            agent,
            (hosting, model_name, model_source),
            getattr(args, "birth_effort", None) or configured_effort(config_manager),
        )

    agent, (hosting, model_name, model_source), effort_default = await asyncio.to_thread(
        resolve_birth
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
            # The standing config effort, CLAMPED inside ``configure_model``
            # against this spec's own ladder. Carried unconditionally, even when
            # ``model_source`` is ``"flag"``: a ``--model`` flag chooses a
            # MODEL, not an effort, and the clamp makes it safe to keep the
            # configured level across that choice (design D3).
            reasoning_effort=effort_default,
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
        credential_manager, config_dir, agent_registry, knowledge_warnings, effective_cwd
    )
    # Configuration discovery is local filesystem work and must precede the
    # first prompt. The TUI deliberately defers live MCP connections; deriving
    # names only from the eventual manager let the knowledge block freeze empty
    # before that background task won the race.
    _seed_mcp_routing(hooks, effective_cwd)
    # The classification seam, built here because the package's cold import must
    # not land inside a turn (see ``_attach_classification``). A no-op — not even
    # an import — unless values.classification.auto is on.
    _attach_classification(hooks, config_manager, credential_manager, knowledge_warnings)
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
    # Construction publishes immutable birth metadata before this runtime is
    # discoverable. Its fsync (and existing full-history replay) must not park
    # sibling sessions on the event loop; publication cannot be delayed until
    # the first turn without changing an empty live row's creation sort key.
    transcript = await asyncio.to_thread(
        Transcript,
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
        # Provenance distinguishes deliberate resume flags from persisted
        # identity; no provenance subscribes a session to mutable defaults.
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
    """Tell the front end that ``mcp_startup`` moved.

    Shared by the wiring's three completion points — the two degradation
    arms (no MCP layer; discovery raised) and the gate snapshot — because a
    deferred-boot TUI learns about ALL of them the same way: it installed
    its sink while the manager was still absent and needs exactly one
    nudge per outcome to re-run its wiring and report.

    A VIEWER is a front end too, and it is told through the frontend-state
    push rather than through a sink: a runtime child has no in-process app to
    call (``_on_mcp_startup_settled`` is None there), so this is the only hop
    the outcome has toward the screen. It has to happen in the arms WITHOUT a
    manager as well — when discovery raises or the MCP layer cannot import, the
    function returns before ``attach_mcp_dispose`` (which is what normally
    refreshes the store for the manager arm), so a viewer bound before the
    wiring keeps the empty outcome it was seeded with and never learns a round
    ran at all. Measured on the deferred path with discovery raising: 0 pushes
    carrying ``mcp_startup`` in the 3 s after the wiring, against a viewer told
    correctly by the same code on the eager path.

    Guarded like the settle path's own lookup: a session without a sink
    (headless, an unadopted session) is the normal case, and a sink that raises
    must never take the wiring down with it.
    """
    sink = getattr(session, "_on_mcp_startup_settled", None)
    if sink is not None:
        try:
            sink(getattr(session, "mcp_startup", None))
        except Exception:  # noqa: BLE001 — a UI hook must never break the wiring
            logger.debug("session _on_mcp_startup_settled raised", exc_info=True)
    # The store is the other front end, and the one a bound viewer reads. Same
    # call the settle path makes, for the same reason.
    #
    # GUARDED, and deliberately not like the eager path's unguarded call: this
    # runs inside the deferred wiring task, whose caller swallows the exception
    # with a warning, so a raising refresh here would skip the `attach_mcp_dispose`
    # that follows on the manager arm — no `disconnect_all` hook, no incident or
    # recovery callbacks — and leave that as one line in a log. A front end hook
    # must not be able to take the wiring down with it, which is the same rule
    # the sink above states.
    refresh = getattr(session, "refresh_frontend_state", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:  # noqa: BLE001 — see above: a UI hook must not break the wiring
            logger.warning("MCP outcome refresh of the frontend store failed", exc_info=True)


def _accepted_kwargs(callee: Callable[..., Any], **candidates: Any) -> dict[str, Any]:
    """``candidates`` narrowed to the keyword arguments ``callee`` accepts.

    **Why narrow rather than pass.** ``discover_and_load_mcp_tools`` grew the
    additive ``secret_base``/``register_secret`` seam, and this repo patches
    discovery with the pre-seam signature (``(cwd, auth_store=None)``) in about
    twenty places: passing the keywords unconditionally turned every one of those
    doubles into a ``TypeError``, which ``wire_mcp_into_session``'s degradation
    handler then reported as "no MCP tools" — silent, and 22 tests red on the PR
    head (QA Q3). The same shape reaches any embedder's own discovery wrapper, so
    the seam tolerates a callee that predates it instead of demanding it grow two
    parameters: a callee declaring both names (the real function) gets both, a
    callee taking ``**kwargs`` gets both, and a pre-seam callee gets exactly the
    call it was written for.

    What that costs is stated rather than implied: for such a callee the seam is
    NOT applied, so its manager resolves against the process config dir and
    registers no redaction sink. That is correct for a double (which returns its
    own canned result and builds no manager) and is why the fallback is a
    signature probe rather than a ``try``/``except TypeError`` retry — a retry
    would also swallow a ``TypeError`` raised from inside the real discovery.
    """
    try:
        parameters = inspect.signature(callee).parameters
    except (TypeError, ValueError):  # a C callable: assume it takes the seam
        return dict(candidates)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(candidates)
    return {name: value for name, value in candidates.items() if name in parameters}


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
        # The owner's config root (which store the references resolve against) and
        # its redaction sink, taken from the session rather than defaulted: a
        # manager built without them would read the wrong store and register
        # nothing for the MCP sinks to scrub. Both are getattr-probed because a
        # reduced host may implement neither, and the manager treats `None` as
        # "no registration" rather than requiring a stub.
        variables = getattr(session, "variables", None)
        manager, mcp_tools, errors = await discover_and_load_mcp_tools(
            cwd,
            auth_store=auth_store,
            **_accepted_kwargs(
                discover_and_load_mcp_tools,
                secret_base=getattr(session, "config_dir", None),
                register_secret=getattr(variables, "register_redaction", None),
            ),
        )
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

    # Which of ``failures`` were the NETWORK, read beside the settling flag and
    # NOT through the manager's public map: this wiring is exercised with
    # reduced manager doubles, and an accessor a double does not implement must
    # degrade to "nothing was recorded as connectivity" — which renders exactly
    # the pre-change copy — rather than take the whole MCP startup wiring down
    # with an AttributeError. Taken as a frozenset here so the outcome's field
    # type is the same on both the gate snapshot and the settle re-report.
    network_failures: frozenset[str] = frozenset()
    try:
        network_failures = frozenset(
            name for name in manager.startup_network_failures() if name in failures
        )
    except Exception:  # noqa: BLE001 — a missing accessor must not break wiring
        logger.debug("MCP startup_network_failures() unavailable", exc_info=True)

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
        network_failures=network_failures,
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
        # A deferred server's tools arrive WITH the settle, so an opted-in server
        # that missed the gate is picked up here rather than never — the cold
        # cache case, where the connect outlasts the 250 ms gate. Guarded
        # separately from the report below: a preload fault must not cost the
        # front end its settle report, which is the failure this callback exists
        # to deliver.
        try:
            apply_preload(manager.get_tools())
        except Exception:  # noqa: BLE001 — a preload fault must not disarm the report
            logger.debug("MCP preload on settle failed", exc_info=True)
        try:
            settled_failures = _collapse_sdk_missing_failures(
                manager.startup_failures(), MCP_DISCOVERY_KEY, MCP_SDK_MISSING_ERROR
            )
            # Guarded exactly like the gate snapshot's read: a reduced manager
            # double (or a host whose manager predates the accessor) must lose
            # the grouping, not the settled re-report that is the ONLY surface
            # a network failure reaches when it misses the startup gate.
            settled_network: frozenset[str] = frozenset()
            try:
                settled_network = frozenset(manager.startup_network_failures())
            except Exception:  # noqa: BLE001 — a missing accessor must not break wiring
                logger.debug("MCP startup_network_failures() unavailable", exc_info=True)
            # ``_collapse_sdk_missing_failures`` may fold the failure map down to
            # the single ``discovery`` key, and a name it dropped must not
            # survive in the network set as a phantom: the toast asks "are ALL
            # the reported failures network ones", so a stale name beside an
            # SDK-less single entry is still not a network story.
            settled_network = frozenset(
                name for name in settled_network if name in settled_failures
            )
            outcome = McpStartupOutcome(
                configured=tuple(manager.get_all_server_names()),
                connected=tuple(manager.get_connected_servers()),
                failures=settled_failures,
                tool_count=len(manager.get_tools()),
                settling=False,
                network_failures=settled_network,
            )
        except Exception:  # noqa: BLE001 — a settle rebuild must never break the manager
            logger.debug("MCP settled outcome rebuild failed", exc_info=True)
            return
        # A declaration made while servers were still connecting had nothing to
        # grant (see ``Session.materialize_declared_tools``). Re-run it now that
        # the round has settled, so a bounded runtime never ends up with its
        # declaration enforced and its declared MCP tools still unreachable.
        materialize = getattr(session, "materialize_declared_tools", None)
        if callable(materialize):
            try:
                materialize()
            except Exception:  # noqa: BLE001 — a grant must not break the settle path
                logger.debug("declared-tool materialization failed", exc_info=True)
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

    def preload_opted_in_tools(source: list[AgentTool]) -> bool:
        """Activate the whole inventory of every server that opted into it.

        WHY THIS EXISTS. MCP tools are lazy by design: a server's schemas are a
        permanent per-request context tax, so a tool enters ``session.tools``
        only once a ``read mcp://`` enables it. That default is right for a
        situational server and wrong for one whose workflow NAMES its tools — a
        tool the model cannot see is a tool the model does not use, so on such a
        workflow laziness degrades the work instead of saving context. A server
        whose config sets ``preload_tools`` says it is the second kind.

        THE ALLOWLIST STILL WINS, and not by a re-check here: the manager applies
        ``disabled_tools``/``enabled_tools`` when it BUILDS a tool, before this
        ever sees it, so an excluded tool is not in ``source`` at all (see
        ``McpManager._tool_is_enabled``, which filters cached/deferred and live
        tools alike). Re-filtering here would be a second, weaker copy of a rule
        the manager already enforces at the one place it cannot be bypassed.

        Returns True when the selected set grew, so a caller can skip a needless
        tool-list rebind when nothing changed.
        """
        added = False
        for tool in source:
            meta = manager.get_tool_meta(tool.name) or {}
            server_name = str(meta.get("server_name", ""))
            raw_name = str(meta.get("mcp_tool_name", ""))
            if not server_name or not raw_name:
                continue
            cfg = manager.get_server_config(server_name)
            if cfg is None or not bool(getattr(cfg, "preload_tools", False)):
                continue
            if (server_name, raw_name) not in enabled_origins:
                enabled_origins.add((server_name, raw_name))
                added = True
        return added

    def apply_preload(source: list[AgentTool]) -> None:
        """Grow the selection for opted-in servers, rebinding only if it grew."""
        if preload_opted_in_tools(source):
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
    # Same freshness, for the classification roster. ``_seed_mcp_routing`` fills
    # the names from the config before any connection work; the manager's own
    # list is the configured set as it stands NOW, so a reload between the two
    # cannot leave the catalogue naming a server the roster does not offer. The
    # roster's cache key includes these names, so the refresh is what rebuilds
    # it.
    knowledge_hooks.mcp_server_names = tuple(manager.get_all_server_names())

    def on_tools_changed(new_mcp_tools: list[AgentTool]) -> None:
        # Reconnects and tools/list_changed can replace AgentTool objects. Keep
        # the selected origins and swap in only their fresh schemas.
        # Preload FIRST, so a reconnect that brings a previously-deferred opted-in
        # server online surfaces its tools in this same rebind rather than one
        # event later — the server's tools becoming reachable is exactly the
        # moment an opted-in server expects to see them.
        preload_opted_in_tools(new_mcp_tools)
        refresh_selected(new_mcp_tools)
        if hasattr(session, "_frontend_state_store"):
            session.refresh_frontend_state()

    manager.set_on_tools_changed(on_tools_changed)
    # The initial pass, AFTER the callback is installed so a server that settles
    # during it cannot slip between the two. Servers still past the gate
    # contribute nothing yet; ``_on_startup_settled`` picks those up below, which
    # is what makes preload work on a cold cache where the connect outlasts the
    # 250 ms gate.
    apply_preload(manager.get_tools())
    return manager


def attach_mcp_dispose(session: Session, manager: McpManager) -> None:
    """Fold ``manager.disconnect_all()`` into the session's dispose path.

    The CLI/TUI/exec all call ``session.dispose()`` exactly once, so hanging
    MCP teardown off it tears the servers down everywhere without teaching
    each caller about the manager. The manager is also exposed as
    ``mcp_manager`` for diagnostics.
    """
    # BEFORE the disconnect hook, deliberately: dispose runs hooks in
    # REGISTRATION order, and ``disconnect_all`` bumps the manager's epoch, so
    # the revalidation poller has to be cancelled first or a tick could register
    # a connection into a manager that is tearing down.
    _attach_mcp_auth_revalidation(session, manager)
    session.add_dispose_hook(manager.disconnect_all)
    session.mcp_manager = manager
    if hasattr(session, "_frontend_state_store"):
        # GUARDED like its sibling in ``_fire_mcp_sink``, and for a sharper
        # reason than symmetry: this runs from the same task that swallows
        # exceptions (``_wire_mcp_background`` logs "background MCP wiring
        # failed" and carries on), and it sits BETWEEN ``disconnect_all``'
        # registration above and the two sink installs below. An unguarded raise
        # here therefore skips the incident and recovery sinks ENTIRELY — no
        # death notice and no healing notice, on every host, including the ones
        # this composition root exists to serve — while the only trace is one
        # warning about wiring (review round 1, MINOR-1 / QA Q1). The push is a
        # UI/store concern and must not be able to disarm the manager's sinks.
        try:
            session.refresh_frontend_state()
        except Exception:  # noqa: BLE001 — a store push must not disarm the sinks
            logger.warning("MCP outcome refresh of the frontend store failed", exc_info=True)
    # Breaker incidents become session incidents: the model learns a server's
    # tools are gone instead of hammering them (MCP-07's observable half).
    manager.on_incident = session._on_mcp_incident
    # ...and the RECOVERY half, installed here rather than in the TUI for the
    # same reason the failure is: this is the composition root every host goes
    # through, so a CLI, headless, exec or server session gets the notice too.
    # A recovery bolted onto the TUI's ``/mcp login`` worker would cover one of
    # the six routes back to a usable server and leave every other host holding
    # a death notice for a server that came back \u2014 the asymmetry itself was
    # the bug. Subagents deliberately do NOT reach this function (they BORROW
    # the parent's manager), so a child never overwrites the parent's sink.
    manager.on_recovery = session._on_mcp_recovery


def _attach_mcp_auth_revalidation(session: Session, manager: McpManager) -> None:
    """Poll the SHARED credential store for servers this session gave up on.

    The credential store is shared by every running process; propagation of a
    fresh grant was not. A session that hit an auth failure held the server
    blocked for its entire lifetime, so completing ``/mcp reauth`` in one
    session left every other running session dead — measured on this machine as
    sessions booted at 08:52 still reporting ``notion [disconnected]`` at 13:30
    against a grant re-authed at 12:31 with eight hours of life left.

    It lives HERE, beside ``on_incident``/``on_recovery``, for the reason those
    do: this is the composition root every host goes through, so the CLI,
    headless, exec and server hosts heal too. Bolting it onto the TUI would
    cover one of six routes. Subagents deliberately never reach this function
    (they BORROW the parent's manager), so a child starts no second poller
    against its parent's servers — adding the task anywhere else loses that.

    Degrades to "this host does not revalidate" rather than failing the boot
    when there is no running loop, exactly as ``attach_config_watch`` does: a
    synchronous embedding is left as it was before this seam existed.
    """
    # Imported here rather than at module scope: ``McpManager`` itself is a
    # TYPE_CHECKING-only import in this module, and the MCP package is an
    # optional extra, so an install without it must still import the factory.
    from local_operator.mcp.manager import AUTH_REVALIDATE_INTERVAL_S

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("no running loop: MCP auth revalidation not attached")
        return

    async def _revalidate_forever() -> None:
        while True:
            await asyncio.sleep(AUTH_REVALIDATE_INTERVAL_S)
            try:
                healed = await manager.revalidate_auth_blocked()
            except Exception:  # noqa: BLE001 — a poll must never kill the session
                logger.debug("MCP auth revalidation tick failed", exc_info=True)
                continue
            if healed:
                logger.info("MCP servers recovered after a peer re-auth: %s", ", ".join(healed))

    task = loop.create_task(_revalidate_forever())
    # Cancelled through the SAME dispose helper the boot wiring uses, which
    # AWAITS the task's quietus so a tick already inside a connect finishes its
    # own cleanup. ``revalidate_auth_blocked`` also re-checks ``_disposed`` and
    # the epoch after its await, as every other reconnect path does, so the
    # ordering above is defence in depth rather than the only guard.
    session.add_dispose_hook(_cancel_task(task))


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


def attach_classification_notices(session: Session, hooks: "_KnowledgeHooks | None") -> None:
    """Give the classification seam the session's own notice event.

    The seam may speak only through a channel the front end already paints.
    ``Session._stream_notice`` emits a ``NoticeEvent`` — the same event the
    ``auto effort:`` line rides (``model/configure.py``, via the stream fn's
    notice handler) — so the CLI, TUI, headless and phone sessions all render it
    with no new surface and no per-host wiring. It is bound HERE, at the
    composition root, because the facade owns the event stream and is built after
    the hooks are.

    Deliberately NOT the stream fn's ``set_notice_handler``, which is the same
    event by a shorter path: that handler is last-writer-wins on a stream fn
    SHARED with subagents (see the prompt-cache TTL note in ``Session.__init__``),
    so a child session would take over its parent's notices. The hooks object is
    per-session, and a child's ``system_blocks_provider`` never runs the
    classification pass at all, so nothing rebinds a parent's sink.

    ``_stream_notice`` is private, and that is the honest description of the
    situation: this is the session's own notice path, there is no public
    equivalent, and the factory already reaches into the facade for exactly this
    kind of binding (``_transcript``, ``_frontend_state_store``). A missing sink
    is not fatal — the layer simply gets no notice, which is the pre-layer
    behaviour — so the lookup is guarded rather than typed onto the protocol.
    """
    if hooks is None:
        return
    # ``queue_notice`` FIRST, and this ordering is design round 1's D1: the notice
    # describes the prompt this turn is BUILDING, so emitting it here painted it
    # between the user's question and the reply — in the answer's slot, in the
    # dimmest ink on screen, 2-4 rows depending on the width. ``queue_notice`` holds
    # it until the turn's answer has landed, and the session flushes it there.
    # ``_stream_notice`` stays the fallback for a facade-shaped double that has no
    # queue (the TUI pilot's ``FakeSession``, a benchmark preflight, any host that
    # supplies its own), so a missing queue costs placement rather than the line.
    sink = getattr(session, "queue_notice", None) or getattr(session, "_stream_notice", None)
    if callable(sink):
        hooks.notice_sink = sink


def attach_classification_dispose(session: Session, hooks: "_KnowledgeHooks | None") -> None:
    """Fold the classification seam's ``aclose()`` into the session's dispose path.

    The shipped service opens ONE keep-alive ``httpx.AsyncClient`` per session
    (that client is §5a rule 3 — a fresh TCP + TLS handshake per call would spend
    the whole latency budget before the request left the process) and memoizes the
    resolved credential and the roster lines for that session's life. Its
    ``aclose`` is documented as "the session owner calls this on dispose", and
    this is that owner: the composition root every front end goes through, beside
    ``attach_auth_dispose`` and ``attach_stream_dispose``, which exist for the
    same reason. Without it the pool and the memos are pinned once per SESSION,
    and the server and phone planes keep sessions alive for hours.

    Registered as ``getattr`` rather than typed onto the seam, matching
    ``attach_classification_notices``: a host's own classifier need not publish an
    ``aclose``, and an injected test double typically does not. A seam without one
    is not an error — it simply owns no resource this harness has to release.

    ONE hook, in ONE order, and the order is the point. The calls this session left
    running are cancelled FIRST, before anything closes the client they are using:
    with ``waitMs`` at its 50 ms default against a ~250 ms answer, a session disposed
    right after a message always has one in flight, and closing the keep-alive client
    under it turned that call into a transport error with a traceback — an answer
    nobody could use any more, reported as a failure (review round 2, MINOR 2). The
    service's own ``aclose`` cancels what it still has in flight for the same
    reason; this half cancels the wiring's wrappers so the session lets go of them
    too.

    The seam is resolved AT DISPOSE TIME rather than captured here, because it is an
    injectable attribute (``hooks.classifier``) and a hook that closed the object
    that happened to be there at REGISTRATION would leave a host's injected seam
    open. That is not hypothetical: the test that covers this path swaps the seam
    after ``create_session`` returns, which is the documented way to inject one, and
    the captured-bound-method version sailed straight past it.
    """
    if hooks is None:
        return

    def _release_the_seam() -> Any:
        """Cancel what is running, then close the seam. Returns the awaitable."""
        for task in hooks.classification_outstanding:
            if not task.done():
                task.cancel()
        hooks.classification_outstanding.clear()
        hooks.classification_pending.clear()
        # ``getattr`` rather than a typed call, matching
        # ``attach_classification_notices``: a host's own classifier need not publish
        # an ``aclose``, and a seam without one simply owns no resource this harness
        # has to release. The returned value is handed straight back to the dispose
        # runner, which awaits what it gets (``Session.dispose``).
        close = getattr(hooks.classifier, "aclose", None)
        return close() if callable(close) else None

    session.add_dispose_hook(cast("Callable[[], Awaitable[None] | None]", _release_the_seam))


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
    ``AttachedSession`` followers never get here: the owner applies the change
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


#: The module chain ``wire_mcp_into_session`` imports before its first await,
#: plus the chain its discovery path imports later (``local_operator.mcp.manager``
#: reaches ``local_operator.mcp.auth`` at module scope, and the connect path
#: imports ``mcp.types`` from INSIDE a function, so it is paid on the first
#: connect rather than at package import).
#:
#: The figures that made this a tuple with a comment: measured on a loaded host
#: with a single HTTP server declared on a CLOSED port (nothing spawned, the
#: refusal immediate), ``import local_operator.mcp.manager`` was 2.0-2.7 s and
#: ``import mcp`` 8.3 s, and a loop-gap probe over the same wiring measured the
#: event loop BLOCKED for 10.38 s of a 10.41 s span. The wiring is import time,
#: not I/O, which is why ordering alone cannot make it cheap.
_MCP_WIRING_IMPORTS: tuple[str, ...] = (
    (
        # ``wire_mcp_into_session``'s OWN function-local imports. They are not in the
        # factory's warm list because nothing else on the boot path wants them.
        "local_operator.session.mcp_status",
        "local_operator.mcp",
    )
    + tuple(
        # ... DERIVED from the factory's own warm list rather than restated. The two
        # lists describe the same import chain, and a second hand-maintained copy is
        # how they would drift: add a module to the wiring and the loop stall comes
        # back silently, with every gate still green because the correspondence is
        # what proves the warm covers the wiring's synchronous prefix.
        name
        for name in _WARM_IMPORTS
        if name == "mcp" or name.startswith("local_operator.mcp")
    )
    + (
        # The SDK submodules the discovery path imports from INSIDE functions, so
        # they are paid on the first connect rather than at package import and the
        # factory's list cannot see them.
        "mcp.types",
        "mcp.client.stdio",
        "mcp.client.streamable_http",
    )
)


def _warm_mcp_wiring_imports() -> None:
    """Import the MCP wiring's module chain, for a caller that is NOT the loop.

    WHY THIS EXISTS, given the deferred wiring already exists. The deferral was
    written so MCP cannot sit between the user and a bound session, and it did
    not achieve that: the work is synchronous module import (see
    ``_MCP_WIRING_IMPORTS``), so wherever it runs on a single-threaded loop it
    takes the loop for that whole duration. The engaging client's own round
    trips — the attach welcome, the prompt admission — queue behind it, so
    moving it after publication moves the cost into the dial instead of removing
    it (measured: ``dial`` absorbed 10.4 s while publication got 1.5 s earlier).

    A worker thread is what keeps the loop serving. The import costs the same
    wall time and the same CPU; what changes is that the process keeps answering
    while it happens.

    Failures are swallowed PER MODULE on purpose: a machine without the MCP SDK
    is a supported configuration, and ``wire_mcp_into_session`` already handles
    its absence and records an outcome. A warm that raised would replace that
    recorded degradation with a boot fault.
    """
    import importlib

    for name in _MCP_WIRING_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — an absent SDK is the wiring's own case
            logger.debug("MCP import warm skipped %s", name, exc_info=True)


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
    mcp_publication_gate: "PublicationGate | None" = None,
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
    deferral is safe for the TUI only because MCP tools are lazy by default
    (see :func:`wire_mcp_into_session`): a turn started before wiring
    settles sees the same non-MCP tool surface as a session whose servers
    missed the gate today, and the ``refresh_selected`` merge lands
    mid-session exactly as a late ``list_changed`` event already does. A server
    that opted into ``preload_tools`` is no exception — its tools land through
    that same mid-session merge, one wiring pass later, because its schemas
    cannot exist before its connection does.

    ``mcp_publication_gate`` is the RUNTIME CHILD's half of that deferral,
    and it exists because deferring the dispatch did not defer the work. The
    background task's first instruction is a function-local import of the MCP
    SDK plus the config parse, and a task cannot run until the loop is free —
    in ``process.amain`` the first free instant is ``await
    _drain_inbox_into(handle)``, which sits BEFORE ``RecordPublisher``. So a
    declared server's SDK import ran to completion inside the pre-publication
    window anyway: measured 2.3 s on a machine with two servers declared, in
    14 of 14 runs, all of it before the record was written and therefore in
    front of the user. ``spawn_owned_session`` passes an event here and
    ``RuntimeServer._serve`` sets it the moment the record exists, so the
    wiring starts where ``serving.spawn_owned_session`` says it should —
    after the record, riding it.

    ``None`` means NO GATE, and that default is the whole safety of this change:
    a latch that applied to a caller which never publishes a record would park
    its MCP wiring for the session's life — MCP silently never wired, which is
    the failure the latch exists to prevent, arrived at from the other side. So
    the gate is only ever created by ``spawn_owned_session``, the one spawn site
    whose runtime publishes.

    Who exercises the ungated path in this tree: ``tests/`` (it is the default),
    and two operator scripts that build an in-process Session for a screenshot
    or a cleanup sweep (``scripts/evidence_session_cleanup.py``,
    ``scripts/cleanup_notice_shot.py``). The TUI process does NOT — on this
    release its owner path is gone from ``lop`` (see ``cli.py``'s "THE OWNER
    PATH IS GONE" note, which says the TUI process never builds a ``Session``)
    and no TUI code passes ``defer_mcp_wiring=True``, so nothing about the
    TUI's own import behaviour changes here. The ungated branch is kept as the
    default because it is the honest answer for a caller with no publisher,
    not because a TUI depends on it.

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
    # that owner's v4 event relay through AttachedSession. This lives at the
    # shared session-factory seam (not in cli.py) so cold ``--resume`` and any
    # future TUI launcher cannot accidentally construct a second writer or
    # invent another attach UI. Headless/exec callers still take the lease and
    # get the existing refusal: they have no full front end to host the facade.
    resume_id = getattr(args, "resume", None)
    if has_ui and resume_id is not None and not _force_local_takeover:
        from local_operator.mobile.attach_client import find_runtime_record
        from local_operator.session.attached import AttachedSession

        root = Path(agent_registry.config_dir)
        record, owner = await asyncio.to_thread(find_runtime_record, root, str(resume_id))
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
                # and AttachedSession rediscovers the winner.
                return await create_session(
                    args,
                    config_manager,
                    credential_manager,
                    agent_registry,
                    has_ui=True,
                    cwd=effective_cwd,
                    _force_local_takeover=True,
                )

            return await AttachedSession.connect(
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
    # Classification seam: the layer's one-line notice rides the session's own
    # notice event (see ``attach_classification_notices``). Bound before the
    # first turn can run, and a no-op when the layer is off.
    attach_classification_notices(session, plan.knowledge_hooks)
    # Stream seam: release the session's shared httpx connection pool on
    # dispose (one leaked pool per turn on the server facade otherwise).
    # The classification seam's client and memos are per SESSION, so the
    # keep-alive client has to be released with everything else the root owns.
    attach_classification_dispose(session, plan.knowledge_hooks)
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
    # DEFERRED wiring is the runtime child's opt-in (``defer_mcp_wiring``): it
    # was written for the TUI's in-process Session, which this release no longer
    # builds in that process at all (``cli.py``: the owner path is gone from
    # ``lop``). The caller that matters today is the runtime child, and the gate
    # it passes below is what keeps the wiring off the record's own publication
    # path.
    #
    # The session returns immediately and the same wiring runs as a background
    # task. The task is tracked on the session's dispose hooks so a quit
    # mid-wiring cancels it (a ``disconnect_all`` on a half-wired manager is
    # exactly the teardown the manager already handles); nothing else differs —
    # the outcome lands in ``session.mcp_startup`` and the settle sink fires
    # when a front end has installed it, which is the same late-attach the 250 ms
    # gate already produces for slow OAuth servers.
    if defer_mcp_wiring:

        async def _wire_mcp_background() -> None:
            # Runs on the session's loop but OFF the boot critical path. An
            # exception here is the wiring's own degradation contract
            # (``wire_mcp_into_session`` never raises for provider reasons);
            # a genuine coding fault is logged rather than killing the task
            # silently, and the session keeps its non-MCP surface — the same
            # state a machine with no ``.mcp.json`` boots into.
            #
            # A GATED task parks HERE, before its first instruction, which is
            # the whole point: the SDK import below is synchronous, so on a
            # single-threaded loop it does not merely take time — it takes the
            # loop. Waiting on the publication latch is what keeps that import
            # out of the runtime's pre-publication window; without the wait the
            # task's first step lands on the drain's first await and the record
            # is held back behind an integration we deliberately do not gate
            # the session on. ``None`` means no publisher, so there is nothing
            # to wait for (see the parameter's docstring).
            if mcp_publication_gate is not None:
                await mcp_publication_gate.wait()
            # KEEP THE LOOP. The latch above fixes the ORDERING — the wiring no
            # longer holds the record back — and ordering alone does not deliver
            # it, because the wiring's first instruction is a synchronous import
            # of the MCP SDK and a task cannot run until the loop is free. Landing
            # that import on the loop right after publication simply moves the
            # stall to the client's own dial and welcome (measured: dial absorbed
            # 10.4 s while publication gained 1.5 s). Warming the same chain in a
            # worker thread is what makes the saving reach the user.
            await asyncio.to_thread(_warm_mcp_wiring_imports)
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
