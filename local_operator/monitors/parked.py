"""Check a PARKED session's monitors without reopening it.

WHY THIS EXISTS. A parked session (``control.park_session``: runtime ended to
free memory, ``parked_at`` stamped, nothing held) keeps its monitors armed, and
the wake supervisor's monitors pass would cold-engage it for EVERY due tick —
a 60 s watch would bring a parked session back every minute, which hands the
memory straight back (field gap 2026-11-12: reopening sessions to keep their
watches alive "defeated the memory saving"). Most ticks are quiet, so the
supervisor runs the check here first and reopens the session only when there
is something for it to do.

THE RULE IS "QUIET OR ENGAGE", NEVER "DELIVER". This module never writes a
snapshot and never delivers a delta. It commits exactly one outcome itself —
the quiet tick (same normalized hash as the stored baseline): the counters
advance as the in-session scheduler's quiet path would (``checks``,
``last_check_at``, ``next_due_at`` via :func:`scheduler.advance_due`). Anything
else answers ``engage``: a change, a first check with no baseline, an error, an
empty/blind result, a monitor in a failing/blind episode, or a tool that
cannot run outside a session (``web_search``; credentials only a session can
use — see :func:`probeable`). The ONE further outcome it commits is the
session's UNAVAILABLE tick, for an MCP server that is unreachable or needs
re-authentication (see :func:`_commit_unavailable`). The
supervisor then cold-engages the session exactly as for a due wake; the
session's own ``MonitorScheduler.load`` sees the row still due, re-runs the
check, and diffs it against the UNCHANGED baseline — so the delta is delivered
by the one owner of delivery, and nothing a parked session watched is missed.

SAME CODE, NOT A SECOND ALLOW-LIST. The tool objects are the builtin factories
the session itself uses (``build_bash_tool``/``build_read_tool``/
``build_grep_tool``/``build_glob_tool``; ``web_fetch``/``code_requests``
through the registry factory with a session-shaped context; MCP tools from a
throwaway ``McpManager``'s live ``tools/list`` — see :class:`_ProbeEnv`);
every check is re-gated by ``readonly.monitor_call_verdict`` at run time, as
``Session._run_monitor_check`` does; the comparable text is
``session._monitor_comparable_text`` and the hash is
``scheduler.normalize_result`` + ``diff.content_hash``. Any drift between this
and the session can only produce a spurious ``engage`` (the stored hash will
not match), never a missed change.

RUN AS A SHORT-LIVED SUBPROCESS (``python -m local_operator.monitors.parked``)
by the supervisor: importing the tool layer costs ~100 MB of RSS, which must not
become resident in the one long-lived process that runs during memory pressure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("local_operator.monitors.parked")

#: Builtin tools that run outside a session with no session-only state: their
#: factories need no ``ToolContext`` capability beyond ``cwd``. ``glob`` joined
#: with the MCP slice: it was unprobed for no reason (2 live watches on
#: 2026-11-12, design §0) and its factory takes no context at all.
HEADLESS_TOOLS = ("bash", "read", "grep", "glob")

#: Builtin tools whose factories need a session-shaped ``ToolContext``
#: (``web_fetch``: the configured ``web_fetch`` settings; ``code_requests``: the
#: session directory its ledger lives in). Built through the SAME registry
#: factory the session uses (``tools.registry.TOOL_BUILDERS``) with a context
#: carrying exactly those fields, so the SSRF policy, the fetch cache and the
#: ledger read are the session's own code.
CONTEXT_TOOLS = ("web_fetch", "code_requests")

#: Tools that stay session-only by decision, not by omission. ``web_search``:
#: its provider keys may come from the launching terminal's ``os.environ``
#: (``web_search/providers.py``), which the launchd supervisor does not have,
#: and the default round-robin strategy varies results by provider, so no
#: hash parity is provable (design §2; 0 live watches on 2026-11-12).
SESSION_ONLY_TOOLS = ("web_search",)

#: Counter keys whose presence means the monitor is mid-episode: a success there
#: owes a "restored"/"seeing again" notice that only the session can send.
#: ``unavailable_since`` is deliberately NOT here: an unavailable-only episode
#: is probeable (the probe commits further unavailable ticks itself and engages
#: the moment the tool answers again, which is when the "restored" notice is
#: owed — see :func:`_check_one`). Without that, a parked session whose MCP
#: server was down would be reopened on every tick of the outage.
_EPISODE_KEYS = ("failing", "blind", "disabled", "empty_checks")

QUIET = "quiet"
ENGAGE = "engage"
#: A row-level outcome only: the tool was out of reach, the session's own
#: unavailable tick was committed and nothing is owed yet. The SESSION-level
#: verdict stays ``quiet`` (nothing for a session to do).
UNAVAILABLE = "unavailable"

#: The supervisor's hard ceiling on one probe (``wakes.supervisor.
#: PARKED_PROBE_TIMEOUT_S``) minus the interpreter start + import the probe
#: spends before its first check. The probe stops starting new work once this
#: budget is spent and answers ``engage`` for the rest, so it finishes on its
#: own terms (and disconnects its MCP servers) instead of being SIGKILLed with
#: a half-open transport. The 15 s margin is the design's (§1 "Bounds"); the
#: measured start+import cost it covers is recorded beside
#: ``PARKED_PROBE_TIMEOUT_S`` in the supervisor.
DEFAULT_BUDGET_S = 165.0


def is_parked(entry: Mapping[str, Any] | None) -> bool:
    """A monitor/wake index entry stamped by a PARK (and not also held)."""
    return (
        isinstance(entry, Mapping) and bool(entry.get("parked_at")) and not entry.get("stopped_at")
    )


def probeable(
    row: Mapping[str, Any],
    tier: str,
    *,
    session_only_servers: frozenset[str] | set[str] | None = None,
) -> bool:
    """Whether the supervisor's probe can run ``row``'s check outside the session.

    The ONE predicate both callers share: pressure park
    (``session.runtime.park_pressure``) skips a session holding a row this
    answers False for (parking it would only reopen it on the next due tick),
    and the probe engages for such a row.

    ``tier`` is the secret store's ``secrets.keys.key_mode`` answer. On the
    ``passphrase`` tier the supervisor is not in any session's broker lineage,
    so a ``${REF}`` cannot resolve outside a session (``secrets/peer.py``); the
    probe would fail every tick. MCP rows are therefore session-only there
    unless the caller proved which servers need the store
    (``session_only_servers`` from :func:`session_only_servers`), and
    ``code_requests`` stays session-only there because a credential failure is
    written into the SHARED fetch cache (``code_requests/service.py``
    ``_record_failure``), where it would poison the session's own reads.

    ``session_only_servers``: configured MCP server names whose credentials
    cannot be resolved by the probe (passphrase-tier refs, brokered grants).
    ``None`` means "not computed": MCP rows are then probeable only on the
    keyfile tier.
    """
    tool = str(row.get("tool") or "")
    if not tool or tool in SESSION_ONLY_TOOLS:
        return False
    if tool in HEADLESS_TOOLS or tool == "web_fetch":
        return True
    if tool == "code_requests":
        return tier != "passphrase"
    if tool.startswith("mcp__"):
        if session_only_servers is None:
            return tier != "passphrase"
        return _server_for_tool(tool, session_only_servers) is None
    return False


def _server_for_tool(tool: str, servers: Any) -> str | None:
    """The configured server whose minted prefix owns ``tool``, longest first.

    ``create_mcp_tool_name`` is ``mcp__<server>_<tool>`` with both halves
    sanitized, so the mapping is lossy (``a_b`` + ``c`` and ``a`` + ``b_c`` mint
    the same name). Longest-prefix-first picks the most specific server; the
    probe then connects ONLY that one, and if it does not serve the tool the
    probe engages (the session, which holds the live registry meta, decides).
    Same prefix rule as ``Session._monitor_server_for``.
    """
    from local_operator.mcp.tool_bridge import create_mcp_tool_name

    names = sorted((str(name) for name in servers), key=len, reverse=True)
    for name in names:
        if tool.startswith(create_mcp_tool_name(name, "x")[:-1]):
            return name
    return None


def session_only_servers(config_dir: Path, cwd: str, tier: str) -> set[str]:
    """MCP servers visible from ``cwd`` whose credentials only a session can use.

    - ``passphrase`` tier + any ``${REF}`` in ``env``/``headers``: the supervisor
      is outside the broker's session lineage, so the ref cannot resolve.
    - a BROKERED grant (this device borrows the server's credential from its
      owner device, ``network/credentials/mcp_bearer.py``): whether a grant can
      be served non-interactively to a non-session process is unproven
      (design §1), so it is not attempted. Conservative on purpose: this asks
      the manager's ``should_borrow`` half only, not its local-grant-wins half,
      so a server with BOTH a local grant and a borrow route is also kept
      session-only — a wasted reopen at worst, never a broker request the
      owner did not expect.
    """
    from local_operator.mcp.config import load_all_mcp_configs
    from local_operator.mcp.secret_refs import has_references

    configs, _sources = load_all_mcp_configs(cwd or ".")
    out: set[str] = set()
    if tier == "passphrase":
        out.update(name for name, cfg in configs.items() if has_references(cfg))
    client: Any = None
    try:
        from local_operator.network.credentials.client import MeshCredentialClient

        client = MeshCredentialClient.for_this_device(config_dir)
    except Exception:  # noqa: BLE001 — no readable mesh state: nothing is brokered
        client = None
    if client is not None:
        from local_operator.network.credentials.types import credential_key_for_mcp

        for name, cfg in configs.items():
            url = getattr(cfg, "url", None)
            if isinstance(url, str) and url:
                try:
                    if client.should_borrow(credential_key_for_mcp(url)):
                        out.add(name)
                except Exception:  # noqa: BLE001 — unanswerable: keep it session-only
                    out.add(name)
    return out


def _build_tool(name: str) -> Any:
    from local_operator.tools import builtin

    if name == "bash":
        return builtin.build_bash_tool(None)
    if name == "read":
        return builtin.build_read_tool()
    if name == "grep":
        return builtin.build_grep_tool()
    if name == "glob":
        return builtin.build_glob_tool()
    return None


class _Resolved:
    """What :meth:`_ProbeEnv.resolve` found for one monitor's tool."""

    __slots__ = ("tool", "unavailable", "why")

    def __init__(self, tool: Any = None, *, unavailable: str = "", why: str = "") -> None:
        self.tool = tool
        #: Non-empty: the session's no-strike wait sentence (server down,
        #: re-auth needed); the probe commits an unavailable tick.
        self.unavailable = unavailable
        #: Non-empty with ``tool`` None: why the session must run this one.
        self.why = why


class _ProbeEnv:
    """Per-run state: the session's coordinates, the deadline, and ONE MCP manager.

    ONE THROWAWAY ``McpManager`` PER PROBE RUN, built lazily on the first MCP
    row and torn down in :meth:`close` in the SAME task that connected it
    (``disconnect_all(in_task=True)``: the transport's anyio cancel scope was
    entered here; closing it from a ``gather`` child cancels the entering task —
    the reason the desktop's sessionless Test op uses the same form,
    ``server/mcp_host.py``).

    NO NEW CREDENTIAL SURFACE. The manager is the session's own class with the
    session's own inputs: ``secret_base`` = the config dir (``${REF}`` values
    resolve per connect, in memory, through ``secrets.access``), the shared
    ``auth.db`` for OAuth (refreshes are lock-serialised across processes), and
    ``mcp.redaction.register`` so a resolved value is scrubbed from anything
    the probe could echo. ``interactive=False`` on every connect: a grant that
    cannot refresh raises ``McpAuthRequiredError`` — no browser, no prompt, no
    loopback callback port. ``tool_cache=None``: every tool comes from THIS
    connection's live ``tools/list``, so the read-only hint the gate reads is
    the server's current answer, never a cached one (the honesty rule of
    ``Session._monitor_mcp_liveness``). ``detach_stdio=False`` keeps stdio
    servers in the probe's process group, which the supervisor reaps.
    """

    def __init__(
        self,
        config_dir: Path,
        session_id: str,
        cwd: str,
        *,
        budget_s: float = DEFAULT_BUDGET_S,
        tier: str | None = None,
    ) -> None:
        self.config_dir = config_dir
        self.session_id = session_id
        self.cwd = cwd or "."
        self.deadline = time.monotonic() + budget_s
        self._tier = tier
        self._manager: Any = None
        self._configs: dict[str, Any] | None = None
        self._session_only: set[str] | None = None
        #: server name -> wait sentence for a server this run already failed to
        #: reach (a second row on it must not pay a second connect timeout).
        self._unreachable: dict[str, str] = {}
        self._connected: set[str] = set()
        self._web_fetch_settings: Any = None
        self._settings_read = False

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def tier(self) -> str:
        if self._tier is None:
            try:
                from local_operator.secrets.keys import key_mode

                self._tier = key_mode(self.config_dir)
            except Exception:  # noqa: BLE001 — unknown tier: assume the strict one
                self._tier = "passphrase"
        return self._tier

    def context(self, spec: Any) -> Any:
        """The check's ``ToolContext``: the session-shaped fields the factories read."""
        from local_operator.harness.types import ToolContext

        if not self._settings_read:
            self._settings_read = True
            try:
                from local_operator.config import ConfigManager

                self._web_fetch_settings = ConfigManager(
                    config_dir=self.config_dir
                ).get_config_value("web_fetch", None)
            except Exception:  # noqa: BLE001 — unreadable config: the tool's defaults
                self._web_fetch_settings = None
        return ToolContext(
            cwd=spec.cwd or self.cwd,
            monitor_check=True,
            session_id=self.session_id,
            session_dir=str(self.config_dir / "sessions" / self.session_id),
            web_fetch_settings=self._web_fetch_settings,
        )

    async def resolve(self, spec: Any) -> _Resolved:
        name = spec.tool
        if name in HEADLESS_TOOLS:
            return _Resolved(_build_tool(name))
        if name in CONTEXT_TOOLS:
            if not probeable({"tool": name}, self.tier()):
                return _Resolved(why=f"{name} is session-only on the {self.tier()} tier")
            from local_operator.tools.registry import TOOL_BUILDERS

            tool = TOOL_BUILDERS[name](self.context(spec))
            if tool is None:
                return _Resolved(why=f"{name} is not available outside a session")
            return _Resolved(tool)
        if name.startswith("mcp__"):
            return await self._resolve_mcp(name)
        return _Resolved(why=f"{name} cannot run outside a session")

    def _load_configs(self) -> dict[str, Any]:
        if self._configs is None:
            from local_operator.mcp.config import load_all_mcp_configs

            self._configs, _sources = load_all_mcp_configs(self.cwd)
            self._session_only = session_only_servers(self.config_dir, self.cwd, self.tier())
        return self._configs

    async def _resolve_mcp(self, name: str) -> _Resolved:
        from local_operator.mcp.auth import McpAuthChallengeError, McpAuthRequiredError
        from local_operator.mcp.manager import (
            McpConnectionError,
            McpManager,
            resolve_mcp_timeout_s,
        )
        from local_operator.mcp.redaction import register

        configs = self._load_configs()
        server = _server_for_tool(name, configs)
        if server is None:
            return _Resolved(why="no configured MCP server mints this tool")
        if server in (self._session_only or set()):
            return _Resolved(why=f'MCP server "{server}" credentials are session-only')
        if server in self._unreachable:
            return _Resolved(unavailable=self._unreachable[server])
        if self._manager is None:
            self._manager = McpManager(
                self.cwd,
                None,
                secret_base=self.config_dir,
                register_secret=register,
                detach_stdio=False,
            )
        if server not in self._connected:
            budget = self.remaining()
            if budget <= 0:
                return _Resolved(why="probe budget spent")
            per_request = resolve_mcp_timeout_s(configs.get(server))
            timeout = budget if per_request is None else min(per_request, budget)
            try:
                await asyncio.wait_for(
                    self._manager.connect_configured_server(server, interactive=False),
                    timeout=timeout,
                )
            except (McpAuthRequiredError, McpAuthChallengeError):
                # The session's own sentence (``_mcp_server_wait_detail``), so a
                # stall notice raised later reads the same either way.
                detail = f'MCP server "{server}" needs re-authentication — run /mcp reauth {server}'
                self._unreachable[server] = detail
                return _Resolved(unavailable=detail)
            except (McpConnectionError, OSError, asyncio.TimeoutError):
                # Unreachable/refused/timed out: what a session would see as a
                # disconnected server — the no-strike class. A secret-ref
                # failure (``McpSecretRefError``) is NOT this class and falls
                # through to ``engage`` below: it is the session's to report.
                detail = f'MCP server "{server}" is disconnected'
                self._unreachable[server] = detail
                return _Resolved(unavailable=detail)
            except Exception as exc:  # noqa: BLE001 — anything else: the session decides
                return _Resolved(why=f"connect raised {type(exc).__name__}")
            self._connected.add(server)
        tools = {tool.name: tool for tool in self._manager.get_tools()}
        tool = tools.get(name)
        if tool is None:
            return _Resolved(why=f'MCP server "{server}" does not list this tool')
        return _Resolved(tool)

    def lost_server(self, name: str) -> bool:
        """``Session._mcp_call_lost_its_server``: an MCP error met a dead server."""
        manager = self._manager
        if manager is None or not name.startswith("mcp__"):
            return False
        server = _server_for_tool(name, self._configs or {})
        if server is None:
            return False
        try:
            return manager.get_connection_status(server) != "connected"
        except Exception:  # noqa: BLE001 — an unanswerable manager proves nothing
            return False

    async def close(self) -> None:
        if self._manager is not None:
            manager, self._manager = self._manager, None
            await manager.disconnect_all(in_task=True)


async def _commit_unavailable(
    config_dir: Path,
    session_id: str,
    spec: Any,
    counters: dict[str, Any],
    snapshot: Mapping[str, Any],
    detail: str,
    now: int,
) -> tuple[str, str]:
    """The session's unavailable tick, committed here — unless it owes a notice.

    WHY A PARKED PROBE COMMITS THIS TICK instead of engaging: an engaged session
    whose server is equally unreachable records the same no-strike tick, idles,
    is re-parked, and is reopened on the next tick — reopen/re-park churn for
    the whole outage (design §4). Nothing is lost by not engaging: no session
    could observe the server either. The tick is the scheduler's own arithmetic
    (:func:`scheduler.apply_unavailable_tick`), so cadence and episode
    timestamps are identical to an in-session outage.

    ENGAGE ONLY WHEN A NOTICE IS OWED: the 30-minute stall notice or the 24 h
    disable. Those are session actions (a notice into the transcript, the
    disable bookkeeping), so the counters are NOT written and the engaged
    session's own scheduler re-runs the check and owns that edge.
    """
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors.scheduler import apply_unavailable_tick

    owed = apply_unavailable_tick(counters, spec, detail, now)
    if owed is not None:
        return ENGAGE, f"tool unavailable; {owed} notice owed"
    if not _unchanged_since(config_dir, session_id, spec.id, snapshot):
        return QUIET, "superseded by a concurrent check; nothing written"
    monitor_state.write_counters(config_dir, session_id, spec.id, counters)
    return UNAVAILABLE, detail


async def _check_one(
    config_dir: Path,
    session_id: str,
    row: Mapping[str, Any],
    settings: Any,
    now: int,
    env: _ProbeEnv | None = None,
) -> tuple[str, str]:
    """``(verdict, why)`` for one due monitor row; commits only a quiet or unavailable tick."""
    from local_operator.harness.types import AbortSignal
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors.diff import content_hash
    from local_operator.monitors.readonly import (
        monitor_call_arguments,
        monitor_call_verdict,
    )
    from local_operator.monitors.scheduler import advance_due, normalize_result
    from local_operator.monitors.spec import MonitorSpec

    if env is None:
        env = _ProbeEnv(config_dir, session_id, str(row.get("cwd") or "."))
    try:
        spec = MonitorSpec.model_validate(
            {key: row[key] for key in MonitorSpec.model_fields if key in row}
        )
    except Exception as exc:  # noqa: BLE001 — a bad row is the session's to report
        return ENGAGE, f"row unreadable ({type(exc).__name__})"
    counters = monitor_state.read_counters(config_dir, session_id, spec.id)
    if not counters or not counters.get("content_hash"):
        return ENGAGE, "no baseline yet"
    snapshot = dict(counters)  # the pre-check view the quiet commit compares against
    if any(counters.get(key) for key in _EPISODE_KEYS) or counters.get("consecutive_failures"):
        return ENGAGE, "monitor is in a failure/blind episode"
    in_outage = bool(counters.get("unavailable_since"))
    resolved = await env.resolve(spec)
    if resolved.unavailable:
        return await _commit_unavailable(
            config_dir, session_id, spec, counters, snapshot, resolved.unavailable, now
        )
    tool = resolved.tool
    if tool is None:
        return ENGAGE, resolved.why or f"{spec.tool} cannot run outside a session"
    # Re-gated on THIS run's tool object. For MCP that object was built from
    # the live ``tools/list`` of the connection this run opened, so a server
    # that dropped ``readOnlyHint`` is refused here and its tool never called.
    reason = monitor_call_verdict(
        tool, spec.arguments, mcp_annotations=getattr(tool, "mcp_annotations", None)
    )
    if reason is not None:
        # Never run a call the gate refuses now; the session will count it.
        return ENGAGE, f"no longer read-only: {reason}"
    from local_operator.session.session import _monitor_comparable_text

    arguments = monitor_call_arguments(tool, spec.arguments)
    context = env.context(spec)
    timeout_s = min(settings.run_timeout_ms / 1000.0, env.remaining())
    if timeout_s <= 0:
        return ENGAGE, "probe budget spent"
    signal = AbortSignal()
    try:
        result = await asyncio.wait_for(
            tool.execute(f"parked-monitor-{spec.id}", arguments, signal, None, context),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        signal.abort("parked monitor check timed out")
        return ENGAGE, "check timed out"
    except Exception as exc:  # noqa: BLE001
        return ENGAGE, f"check raised {type(exc).__name__}"
    if getattr(result, "is_error", False):
        if env.lost_server(spec.tool):
            # The session's ``_mcp_call_lost_its_server`` class: the call never
            # reached a live server, so it is a wait, not the tool's answer.
            text = str(getattr(result, "text", "") or "").strip()
            return await _commit_unavailable(
                config_dir,
                session_id,
                spec,
                counters,
                snapshot,
                text or "the MCP call did not reach its server",
                now,
            )
        return ENGAGE, "check returned an error"
    if in_outage:
        # The tool answers again: the episode's "restored" notice is owed, and
        # only the session sends it (its scheduler re-runs this check).
        return ENGAGE, "tool reachable again; restored notice owed"
    details = getattr(result, "details", None)
    if isinstance(details, Mapping):
        exit_code = details.get("exit_code")
        if (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code != 0
            and str(details.get("stderr_head") or "").strip()
        ):
            return ENGAGE, "check failed"
        if details.get("backgrounded") or details.get("no_output") is True:
            return ENGAGE, "no comparable output"
    comparable, blind, handle = _monitor_comparable_text(getattr(result, "text", "") or "", details)
    if blind is not None or not str(comparable or "").strip():
        return ENGAGE, "no comparable output"
    new_hash = content_hash(normalize_result(str(comparable), spec, settings))
    if new_hash != str(counters.get("content_hash") or ""):
        return ENGAGE, "changed"
    # THE QUIET TICK, as ``MonitorScheduler._apply_success`` commits it.
    counters["checks"] = int(counters.get("checks") or 0) + 1
    counters["last_check_at"] = now
    counters["consecutive_failures"] = 0
    counters["last_error"] = ""
    if handle:
        counters["last_handle"] = handle
    counters["next_due_at"] = advance_due(spec, now)
    # COMPARE BEFORE COMMIT (review round 1, R1-2). ``counters`` was read before
    # ``tool.execute``, which can take up to ``run_timeout_ms``. If the session
    # was opened meanwhile (an operator, a ``send``, a wake from another path),
    # its own scheduler may have run this check, delivered a change and stored
    # a new ``content_hash``; writing our stale dict back would restore the OLD
    # hash and the next tick would deliver the same delta twice. So re-read the
    # entry and the counters and commit only if nothing moved; otherwise drop
    # the write — whoever moved them owns this tick. Not an atomic CAS (the
    # counters file has no lock), but it shrinks the race from the whole check
    # to the microseconds between this read and the write.
    if not _unchanged_since(config_dir, session_id, spec.id, snapshot):
        return QUIET, "superseded by a concurrent check; nothing written"
    monitor_state.write_counters(config_dir, session_id, spec.id, counters)
    return QUIET, "unchanged"


#: Counter fields that move whenever anyone commits a check: the comparison set
#: for :func:`_unchanged_since`.
_CAS_KEYS = ("content_hash", "last_check_at", "checks", "next_due_at")


def _unchanged_since(
    config_dir: Path, session_id: str, monitor_id: str, snapshot: Mapping[str, Any]
) -> bool:
    """Whether the entry is still parked and the counters still match ``snapshot``."""
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors import store as monitor_store

    if not is_parked(monitor_store.read_entry(config_dir, session_id)):
        return False
    fresh = monitor_state.read_counters(config_dir, session_id, monitor_id) or {}
    return all(fresh.get(key) == snapshot.get(key) for key in _CAS_KEYS)


async def probe_session(
    config_dir: Path,
    session_id: str,
    *,
    now_ms: int | None = None,
    cwd: str = "",
    budget_s: float = DEFAULT_BUDGET_S,
    tier: str | None = None,
) -> dict[str, Any]:
    """Check every DUE monitor of a parked session; ``verdict`` quiet/engage.

    ``engage`` as soon as one row needs the session (the rest stay due and the
    engaged session checks them itself). Checks run sequentially: the
    in-session ceiling is 2 concurrent checks per session, and the supervisor
    already bounds how many sessions it probes at once. A row whose tool was
    out of reach commits the session's unavailable tick and keeps the verdict
    ``quiet`` (``unavailable`` counts them).

    ``cwd`` is the SESSION's working directory (the supervisor passes the
    index entry's): MCP configs are discovered per cwd exactly as the session's
    own ``McpManager(cwd, …)`` discovers them.
    """
    from local_operator.monitors import state as monitor_state
    from local_operator.monitors import store as monitor_store
    from local_operator.monitors.settings import read_monitor_settings

    now = int(time.time() * 1000) if now_ms is None else now_ms
    entry = monitor_store.read_entry(config_dir, session_id)
    if not is_parked(entry):
        return {"verdict": ENGAGE, "why": "not parked", "checked": 0}
    assert entry is not None
    settings = read_monitor_settings()
    entry_cwd = entry.get("cwd")
    env = _ProbeEnv(
        config_dir,
        session_id,
        cwd or (entry_cwd if isinstance(entry_cwd, str) else ""),
        budget_s=budget_s,
        tier=tier,
    )
    checked = 0
    unavailable = 0
    try:
        for raw in entry.get("monitors") or ():
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            monitor_id = str(row.get("id") or "")
            counters = monitor_state.read_counters(config_dir, session_id, monitor_id) or {}
            due = counters.get("next_due_at", row.get("next_due_at"))
            if counters.get("disabled") or row.get("disabled"):
                continue
            until = row.get("until_at")
            if isinstance(until, int) and now >= until:
                continue
            if not isinstance(due, int) or isinstance(due, bool) or due > now:
                continue
            verdict, why = await _check_one(config_dir, session_id, row, settings, now, env)
            checked += 1
            if verdict == ENGAGE:
                return {"verdict": ENGAGE, "why": f"{monitor_id}: {why}", "checked": checked}
            if verdict == UNAVAILABLE:
                unavailable += 1
    finally:
        # Every stdio server this run spawned is closed HERE, in the task that
        # opened it, before the verdict is printed; the supervisor's group
        # SIGKILL after exit is only the backstop.
        await env.close()
    outcome: dict[str, Any] = {"verdict": QUIET, "why": "unchanged", "checked": checked}
    if unavailable:
        outcome["why"] = "unchanged or unavailable (tick recorded)"
        outcome["unavailable"] = unavailable
    return outcome


def _maxrss_kib() -> dict[str, int]:
    """Peak RSS of this probe and of its reaped children (stdio MCP servers), KiB.

    Published in the verdict so the supervisor can log the distribution a
    memory cap must be derived from (delivery rule: measure before you cap).
    ``ru_maxrss`` is bytes on macOS and KiB on Linux.
    """
    import resource
    import sys

    scale = 1024 if sys.platform == "darwin" else 1
    return {
        "self": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // scale),
        "children": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // scale),
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m local_operator.monitors.parked --session <id>``: one JSON line."""
    parser = argparse.ArgumentParser(prog="parked-monitor-probe")
    parser.add_argument("--session", required=True)
    parser.add_argument("--config-dir", default="")
    parser.add_argument("--cwd", default="")
    parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    args = parser.parse_args(argv)
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.paths import config_dir as ambient_config_dir

    if args.config_dir:
        # ONE ROOT FOR EVERYTHING. Monitor settings, the spill store the MCP
        # result is spilled into and read back from, ``auth.db``, the fetch
        # caches and the secret base all resolve from the AMBIENT config dir,
        # so an explicit ``--config-dir`` that disagreed with the environment
        # would hash through a different pipeline than the session did.
        os.environ[CONFIG_DIR_ENV] = args.config_dir
    root = Path(args.config_dir) if args.config_dir else ambient_config_dir()
    try:
        outcome = asyncio.run(
            probe_session(root, args.session, cwd=args.cwd, budget_s=args.budget_s)
        )
    except Exception as exc:  # noqa: BLE001 — any failure means "let the session do it"
        outcome = {"verdict": ENGAGE, "why": f"probe raised {type(exc).__name__}", "checked": 0}
    outcome["maxrss_kib"] = _maxrss_kib()
    print(json.dumps(outcome))
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised as a subprocess
    raise SystemExit(main())
