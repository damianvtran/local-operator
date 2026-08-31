"""cmux: publish a TRUSTED auto-resume binding for this surface.

WHY THE SOCKET RPC AND NOT ``cmux surface resume set``
------------------------------------------------------
There is an obvious-looking CLI for this — ``cmux surface resume set`` — and
it CANNOT do what this module needs. That CLI hardcodes ``source = "cli"``,
and cmux resolves a ``cli``-sourced binding to ``approvalPolicy = .manual``,
``autoResume = false``. The binding it produces can only ever be replayed by
hand, which is exactly the dead end this feature exists to remove: after a
crash the user would still be reopening fifteen sessions manually.

The socket RPC ``surface.resume.set`` accepts a ``source`` and gates
auto-resume behind one value:

    autoResume: source == "agent-hook" ? (bool(params,"auto_resume") ?? false) : false

(cmux ``ControlCommandCoordinator+Surface3.swift``). So ``agent-hook`` is the
only source that can reach ``approval_policy: auto`` — and it does so with no
patch to cmux and no entry in cmux's built-in agent catalog, because
``RestorableAgentKind`` has a ``case custom(String)`` and ``local-operator``
collides with no built-in id. Measured on cmux 0.64.22: this call returns
``approval_policy: "auto", auto_resume: true, source: "agent-hook"``.

**Do not "simplify" this to the CLI.** It will look like it still works — a
binding is still created and ``surface resume show`` still prints one — and
auto-resume will be silently dead.

THE STALE-BINDING RETIREMENT RULE (why publishing once is not enough)
--------------------------------------------------------------------
cmux prunes agent-hook bindings whose agent is not a KNOWN LIVE PROCESS:
``Workspace.isStaleAgentHookBinding`` asks ``AgentResumeLiveness.hasLiveProcess``
whether the binding's kind+sessionId matches a process in its scanner index,
and ``retireAgentHookResumeBinding`` flips ``autoResume`` to false when it does
not. The rule is there to stop a dead session replaying, and it applies to us
because ``lop`` is not one of cmux's built-in agents.

Two consequences, both load-bearing, both measured on this machine:

1. **A vault registration is required** for the binding to survive at all.
   Without one, a freshly published binding retires to ``manual`` within
   ~10 seconds. With one, it holds indefinitely. The registration is the
   user's to install (it lives in THEIR ``cmux.json``); see
   ``docs/multiplexer-resume.md``. We publish regardless — an unretired
   binding is strictly better than none, and the user may install the
   registration later without restarting their sessions.

2. **Publishing once is not enough, because that index is CACHED for 60s**
   (``SharedLiveAgentIndex.cacheTTL``). A session that publishes at startup
   can be judged against an index snapshot taken BEFORE this process existed,
   in which case it is retired despite a correct registration. Retirement is
   **one-way**: ``retireAgentHookResumeBinding`` latches ``autoResume = false``
   and nothing in cmux ever sets it back, so a single missed window is
   permanent for that binding — the failure would be invisible until the next
   crash, which is the worst possible time to discover it. Hence
   :data:`REASSERT_INTERVAL_S`, which re-publishes on a timer longer than that
   TTL. Reproduced both ways: publishing against a stale index retired the
   binding; re-asserting afterwards restored ``auto_resume: true``.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess

from local_operator.multiplexer.types import EnvMap, SessionBinding

logger = logging.getLogger(__name__)

#: cmux's own name for what kind of agent this is. Free to choose (a custom
#: vault id is rejected only if it collides with a BUILT-IN kind), and it must
#: match the ``id`` of the vault registration the user installs or cmux will
#: look for liveness under a kind nothing is registered as.
AGENT_KIND = "local-operator"

#: The only ``source`` that reaches ``approval_policy: auto``. See the module
#: docstring; this string is the whole reason this module uses the RPC.
AGENT_HOOK_SOURCE = "agent-hook"

#: Seconds between re-assertions of the binding. MUST stay above cmux's 60s
#: ``SharedLiveAgentIndex.cacheTTL``: the point of re-asserting is to publish
#: again against an index refreshed since this process started, so an interval
#: shorter than the TTL would re-publish against the SAME stale snapshot and
#: fix nothing while costing a subprocess every time. Comfortably above it
#: rather than exactly at it, because the TTL is measured from cmux's last
#: load and not from our clock.
REASSERT_INTERVAL_S = 90.0

#: How long any one cmux call may take before it is abandoned. The TUI never
#: waits on these (they run on a worker thread), but a hung socket must not
#: leak a process per interval either.
CALL_TIMEOUT_S = 5.0

#: cmux injects these per surface. BOTH are needed: the workspace names a
#: workspace of many surfaces, and the surface is the pane actually holding
#: this session, so targeting on workspace alone would rebind a sibling pane.
SURFACE_ENV = "CMUX_SURFACE_ID"
WORKSPACE_ENV = "CMUX_WORKSPACE_ID"

#: Placement vocabulary shared with ``spawn.cmux`` without importing that
#: package back into the multiplexer layer.
_FORK_SURFACE_PLACEMENT = "surface"

#: cmux mints these as UUIDs. Validated before use because they are
#: interpolated into a JSON payload handed to a socket that acts on them, and
#: an inherited-but-stale value (a container, an ssh hop) should read as "no
#: cmux here" rather than as a malformed request against a real surface.
_UUID_RE = re.compile(r"\A[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def _surface_target(env: EnvMap) -> dict[str, str] | None:
    """``{workspace_id, surface_id}`` for this pane, or None when not in cmux."""
    workspace = (env.get(WORKSPACE_ENV) or "").strip()
    surface = (env.get(SURFACE_ENV) or "").strip()
    if not _UUID_RE.match(workspace) or not _UUID_RE.match(surface):
        return None
    return {"workspace_id": workspace, "surface_id": surface}


class CmuxBackend:
    """Publishes this surface's resume binding over the cmux control socket."""

    name = "cmux"

    def detect(self, env: EnvMap) -> bool:
        """True when this process is a cmux surface AND a cmux CLI exists.

        The BINARY is the gate and not the environment markers alone, which is
        the rule ``tools.builtin._cmux_binary`` already encodes and the reason
        this defers to it rather than testing ``CMUX_*`` here: every CMUX_*
        variable is inherited by descendants that crossed into a container or
        an ssh host where no cmux CLI exists, and publishing there would spawn
        a subprocess per session that can only fail.
        """
        if _surface_target(env) is None:
            return False
        return _cmux_binary() is not None

    def publish(self, binding: SessionBinding, env: EnvMap) -> bool:
        """Set a trusted auto-resume binding for this surface.

        ``checkpoint_id`` is the session id, which is what cmux hands back to
        the vault registration's ``sessionIdSource`` and what its liveness
        check matches on. ``launch_command`` carries the structured argv so
        cmux restores the exact launcher rather than re-deriving one from the
        shell string.
        """
        target = _surface_target(env)
        binary = _cmux_binary()
        if target is None or binary is None:
            return False
        params = {
            **target,
            "kind": AGENT_KIND,
            "name": binding.name,
            "source": AGENT_HOOK_SOURCE,
            "checkpoint_id": binding.session_id,
            "auto_resume": True,
            # The shell form cmux falls back to. Restore-and-idle by
            # construction: `binding.argv` is built by `broadcast` and can
            # never carry a prompt or a continue flag.
            #
            # shlex.join and not " ".join, matching the marker backends: cmux
            # re-tokenises this string, so a launcher path containing a space
            # (`/Applications/My Tools/lop`) would otherwise come back as two
            # arguments and restore nothing.
            "command": shlex.join(binding.argv),
            "cwd": binding.cwd,
            "launch_command": {
                "launcher": AGENT_KIND,
                "executable_path": binding.executable,
                "arguments": list(binding.argv),
                "working_directory": binding.cwd,
                "source": AGENT_HOOK_SOURCE,
            },
        }
        return _rpc(binary, "surface.resume.set", params) is not None

    def retire(self, binding: SessionBinding, env: EnvMap) -> bool:
        """Clear this surface's binding on a clean exit.

        ``checkpoint_id``/``source`` are sent as EXPECTATIONS, not just
        identifiers: cmux compares them against the stored binding and clears
        nothing if they disagree. That is what stops a quitting session from
        wiping a binding some other agent published into this surface after
        us — the clear is scoped to the binding we ourselves set.
        """
        target = _surface_target(env)
        binary = _cmux_binary()
        if target is None or binary is None:
            return False
        params = {
            **target,
            "checkpoint_id": binding.session_id,
            "source": AGENT_HOOK_SOURCE,
            # Tells cmux this is a session that ENDED rather than a binding
            # being replaced, which is what makes the clear final instead of
            # something its own reconciliation may put back.
            "agent_session_ended": True,
        }
        return _rpc(binary, "surface.resume.clear", params) is not None


def _cmux_binary() -> str | None:
    """The cmux CLI path, resolved by the ONE function that owns that rule.

    Imported lazily and from ``tools.builtin`` on purpose: that module already
    documents why PATH is tried before ``CMUX_BUNDLED_CLI_PATH`` and why the
    binary rather than the env markers is the gate, and a second copy of the
    rule here would be a second thing to keep in sync. Lazy because
    ``tools.builtin`` is a large module that the startup path must not import
    for a detection test.
    """
    from local_operator.tools.builtin import _cmux_binary as resolve

    return resolve()


def _rpc(binary: str, method: str, params: dict[str, object]) -> dict[str, object] | None:
    """One cmux RPC call. Returns the parsed result, or None on ANY failure.

    Never raises, by contract: every caller is on a best-effort path where the
    only correct response to a failure is to carry on (see the package
    docstring). A missing binary, a socket mid-restart, a surface that has
    since closed, malformed JSON and a timeout are all the same answer here —
    "not published" — and are logged at debug because a user running no cmux
    must not see a warning per session.
    """
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [binary, "rpc", method, json.dumps(params)],
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("cmux %s failed to spawn", method, exc_info=True)
        return None
    if completed.returncode != 0:
        logger.debug("cmux %s exited %s: %s", method, completed.returncode, completed.stderr[:200])
        return None
    try:
        parsed = json.loads(completed.stdout)
    except ValueError:
        logger.debug("cmux %s returned non-JSON", method)
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Shared surface for other packages
# ---------------------------------------------------------------------------
#
# ``local_operator.spawn.cmux`` opens a WORKSPACE or SURFACE for a forked
# session and needs exactly the client this module already owns: the binary
# resolution rule (PATH before ``CMUX_BUNDLED_CLI_PATH``, defined once in
# ``tools.builtin``), the RPC helper and the surface target. These wrappers
# exist so that reuse is an import of a public name rather than a second cmux
# client — which is the drift that would let the two disagree about where the
# binary lives.
#
# They DELEGATE to the underscore functions rather than replacing them: those
# are the definitions the existing tests monkeypatch, and a promotion that moved
# the body would silently make every one of those patches a no-op.


def rename_fork_target(env: EnvMap, title: str, *, placement: str) -> bool:
    """Best-effort rename of the cmux target owned by this fork process.

    cmux 0.64.22+ exposes ``workspace rename <id> --title`` and
    ``rename-tab --surface <id> <title>``. Target explicit ids from THIS
    process's environment: defaults could rename whichever workspace the user
    selected meanwhile. Callers additionally gate on fork provenance, which is
    what prevents an ordinary/parent session from ever reaching this mutation.
    """
    target = _surface_target(env)
    binary = _cmux_binary()
    clean = " ".join(str(title).split()).strip()
    if target is None or binary is None or not clean:
        return False
    if placement == _FORK_SURFACE_PLACEMENT:
        argv = [binary, "rename-tab", "--surface", target["surface_id"], clean]
    else:
        argv = [
            binary,
            "workspace",
            "rename",
            target["workspace_id"],
            "--title",
            clean,
        ]
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("cmux fork rename failed to spawn", exc_info=True)
        return False
    if completed.returncode != 0:
        logger.debug("cmux fork rename exited %s: %s", completed.returncode, completed.stderr[:200])
        return False
    return True


def cmux_binary() -> str | None:
    """The cmux CLI path, or None when there is no usable cmux here."""
    return _cmux_binary()


def cmux_rpc(binary: str, method: str, params: dict[str, object]) -> dict[str, object] | None:
    """One cmux RPC call. Returns the parsed result, or None on ANY failure."""
    return _rpc(binary, method, params)


def surface_target(env: EnvMap) -> dict[str, str] | None:
    """``{workspace_id, surface_id}`` for this pane, or None when not in cmux."""
    return _surface_target(env)
