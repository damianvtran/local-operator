"""The plain-text export of an :class:`InfoSnapshot`, for pasting into an issue.

**One export mode, and it is redacted. There is no unredacted mode.** A user
told "press ctrl+r, or ctrl+shift+r for the full one" will press the wrong one
under stress, and the mode that leaks would be the one with the better name. The
boundary is drawn where the risk actually changes instead: the SCREEN shows
verbatim paths (local debugging, private, and every high-value bug in this
codebase's history is a path bug — the editable-venv resolution trap,
``LOCAL_OPERATOR_CONFIG_DIR`` not redirecting the cache, ``lop-update``
installing from the wrong ref), while the EXPORT is home-relativised, because
that is the artifact that leaves the machine.

What can never appear here, at any width and through any future edit:

* ``SessionRecord.control_key`` — 64 hex characters that ARE the control
  socket's whole authorization story. It is not a field on
  :class:`~local_operator.info.model.SessionLine` at all, which is what makes
  the guarantee structural rather than a property of this file.
* ``BridgeState.session_key`` — same reasoning; only ``paired``,
  ``extension_connected``, ``browser_name`` and ``port`` are collected.
* Credential VALUES. Names only, and not a prefix, a length or a hash: a
  "first four characters" habit is how key prefixes end up in issues.
* An absolute ``/Users/<name>`` or ``/home/<name>`` prefix. A macOS username in
  an issue is low-sensitivity, but a ``cwd`` like
  ``/Users/x/clients/acme-merger-diligence`` is real information, and the export
  is the thing that gets pasted somewhere public.

``tests/unit/info/test_redaction.py`` asserts each of those against a snapshot
deliberately seeded with a known key, and is the highest-value file in this
package.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from local_operator.info.model import (
    UNKNOWN,
    InfoSnapshot,
    SubagentLine,
    format_bytes,
    format_duration,
    is_shadowed_install,
)

#: Fields the export never prints, restated here as a NAME LIST purely so a
#: reader of this module sees the rule without chasing three dataclasses. The
#: enforcement is structural (the fields do not exist on the collected shapes);
#: this is documentation, not a filter.
NEVER_EXPORTED = ("control_key", "session_key", "credential values")


def _home() -> str:
    """The home directory, or ``""`` when it cannot be resolved. NEVER raises.

    ``Path.home()`` raises ``RuntimeError`` when the home cannot be resolved,
    which is the same failure class ``collect.py`` is wrapped against — and the
    export is reached from ``action_copy_report``, a SYNCHRONOUS key action on
    the message pump. No worker sits under it, so ``exit_on_error=False``
    cannot help: an escape here takes the application down.

    That path only became reachable once the collect-side guards landed. Before
    them the app died at probe time and the user never got a screen to press
    ``ctrl+r`` on; after them the screen opens and re-probes cleanly, and the
    copy gesture — the one gesture this feature is advertised for — was the
    remaining way to kill the session (review round 2, F1).

    An empty string means "there is no home to relativise against", and both
    callers then return their input unchanged. That is the honest outcome
    rather than a redaction failure: what would be stripped is *this user's
    home prefix*, and on this host that concept is exactly what is unavailable.
    """
    try:
        return str(Path.home())
    except (RuntimeError, OSError):
        return ""


def relativise_home(path: str) -> str:
    """``/Users/x/repos/y`` → ``~/repos/y``. Everything below ``~`` is kept.

    The username is the identifying part; the path BELOW the home directory is
    the diagnostic part (``~/local-operator``, ``~/workspace/repos/lo-x`` all
    say something a maintainer needs). So exactly one segment is removed.
    """
    if not path:
        return path
    home = _home()
    if not home:
        return path
    if path == home:
        return "~"
    prefix = home + os.sep
    if path.startswith(prefix):
        return "~" + os.sep + path[len(prefix) :]
    return path


def relativise_home_everywhere(text: str) -> str:
    """Replace every occurrence of the home directory anywhere in ``text``.

    Unlike :func:`relativise_home`, which anchors at the START of a path, this
    rewrites a home path embedded mid-sentence — which is the shape a failure
    message has. That is why FREE TEXT is routed through here: MCP failure
    messages are ``str(exc)`` from a failed connect (typically ``command not
    found: <absolute path>``) and a ``degraded`` reason is
    ``f"{type(exc).__name__}: {exc}"``, where an ``OSError`` message contains
    the filename it failed on. Both routinely carry ``$HOME`` — exactly the
    disclosure this module's docstring says must never happen, in the artifact
    built to be pasted publicly (review round 1, B2).

    Applied per free-text channel AND again over the whole document by
    :func:`build_export`. That backstop covers the HOME DIRECTORY only — it is
    a literal replacement of one string. It does NOT cover the other
    identifying shapes: an absolute socket path outside ``$HOME``
    (``CMUX_SOCKET_PATH``) or a ``/var/folders/...`` temp path pass through it
    untouched. A new free-text field therefore still needs its own scrub at
    source; the document pass is a safety net for the one shape, not a general
    redaction pass (review round 2, F2).
    """
    if not text:
        return text
    home = _home()
    if not home:
        return text
    return text.replace(home + os.sep, "~" + os.sep).replace(home, "~")


def _value(text: str) -> str:
    """A scalar for the export, with the SAME unknown spelling the screen uses.

    The screen renders an unreadable value as ``UNKNOWN``; the export used to
    interpolate the empty string, so one snapshot produced seven blank fields
    and the two surfaces disagreed about what had been read. A blank in a
    pasted report is indistinguishable from a rendering bug, which is the second
    round trip this screen exists to remove (review round 1, M4).
    """
    return text or UNKNOWN


def _latest_line(snapshot: InfoSnapshot) -> str:
    """The three PyPI states, never collapsed into two.

    ``None`` is "we have never asked", which is a different fact from "you are
    up to date" — and on the broken network that ``/info`` is usually opened
    over, printing the latter from the former is an outright lie.
    """
    install = snapshot.install
    if not install.version:
        # The unknown is on the INSTALLED side here rather than the latest side,
        # but the lie is the same one: ``is_behind("", latest)`` is False by
        # design, so a failed version probe beside a cached NEWER release
        # reported currency. Both halves fail together on a broken install,
        # which is the state /info is opened in (review round 1, M3).
        return "unknown (installed version unreadable)"
    if install.latest_known is None:
        return "unknown (never checked)"
    from local_operator.update import TTL_S

    age = install.latest_age_s
    if age is None:
        return install.latest_known
    stale = " (stale)" if age > TTL_S else ""
    return f"{install.latest_known} · checked {format_duration(age)} ago{stale}"


def _tree_lines(tree: tuple[SubagentLine, ...], deeper: int) -> list[str]:
    out: list[str] = []
    for node in tree:
        indent = "  " * node.depth
        role = f" [{node.agent_role}]" if node.agent_role else ""
        out.append(f"  {indent}- {node.label or node.job_id}{role}: {node.status}")
    if deeper:
        out.append(f"  + {deeper} deeper")
    return out


def build_export(snapshot: InfoSnapshot) -> str:
    """The whole snapshot as plain text, home-relativised, ready to paste.

    No markdown code fence is added. The user is pasting into a GitHub issue and
    may want a fence, a details block or neither; one we add is one they have to
    remove first.

    The first line is a single-line triage header so a maintainer can classify
    the report without reading the body.
    """
    install = snapshot.install
    process = snapshot.process
    sessions = snapshot.sessions
    agents = snapshot.agents
    env = snapshot.env

    head = " · ".join(
        part
        for part in (
            f"local-operator {install.version or 'unknown'}",
            install.kind or "unknown install",
            install.platform or "",
            f"py{install.python_version}" if install.python_version else "",
        )
        if part
    )
    lines: list[str] = ["## Install"]
    lines += [
        f"  version           {_value(install.version)}",
        f"  latest known      {_latest_line(snapshot)}",
        f"  install kind      {_value(install.kind)}",
        f"  install path      {_value(relativise_home(install.prefix))}",
        f"  interpreter       {_value(relativise_home(install.executable))}",
        # Included unconditionally, and marked when it diverges: a maintainer
        # reading a bug report needs to know the reporter's runtime was running
        # a checkout rather than the version the first line claims.
        f"  running code      {_value(relativise_home(install.import_path))}"
        + (
            "  (NOT under the install path — a checkout is shadowing it)"
            if is_shadowed_install(install)
            else "  (an editable checkout, as expected)" if install.import_path_foreign else ""
        ),
        f"  source            {'git snapshot' if install.is_git_snapshot else 'PyPI wheel'}"
        + (f" @ {install.source_ref[:12]}" if install.source_ref else ""),
        f"  build age         {format_duration(install.build_age_s)}",
        "  python            "
        + (
            f"{install.python_version} ({install.python_implementation})"
            if install.python_version and install.python_implementation
            else _value(install.python_version or install.python_implementation)
        ),
        f"  platform          {_value(install.platform)} · {_value(install.machine)}",
        "",
        "## This session",
        f"  session id        {_value(process.session_id)}",
        f"  kind              {_value(process.kind)}",
        f"  pid               {process.pid}",
        f"  model             {_value(process.model_label)}",
        f"  effective model   {_value(process.effective_model or process.model_label)}",
        f"  uptime            {format_duration(process.uptime_s)}",
        f"  working dir       {_value(relativise_home(process.cwd))}",
        f"  config dir        {_value(relativise_home(process.config_dir))}"
        + ("  (redirected)" if process.config_dir_redirected else ""),
        f"  cache dir         {_value(relativise_home(process.cache_dir))}",
        f"  agent home        {_value(relativise_home(process.agent_home))}"
        + ("  (redirected)" if process.agent_home_redirected else ""),
        f"  log dir           {_value(relativise_home(process.log_dir))}",
        f"  control port      {process.control_port if process.control_port else '—'}",
        f"  protocol          {process.protocol if process.protocol else '—'}",
        "",
        "## Sessions on this machine",
    ]

    if not sessions.available:
        lines.append("  unavailable — the session registry could not be scanned")
    elif not sessions.lines:
        lines.append("  none")
    else:
        lines.append(
            f"  {sessions.live} live · {sessions.wedged} wedged · {sessions.total} total"
            + (" · BUILD SKEW" if sessions.build_skew else "")
        )
        for line in sessions.lines:
            mark = "*" if line.is_self else "-"
            name = line.conversation_name or line.session_id or str(line.pid)
            memory = format_bytes(line.footprint_bytes or line.rss_bytes)
            extra = " · busy" if line.busy else ""
            extra += f" · needs {line.pending}" if line.pending else ""
            lines.append(
                f"  {mark} [{line.state}] {name} · {line.kind} · pid {line.pid} · "
                f"{format_duration(line.uptime_s)} · {memory}{extra}"
            )
            lines.append(f"      {relativise_home(line.cwd)} · {line.model_label or '—'}")

    lines += ["", "## Agents and subagents"]
    lines.append(f"  profiles          {agents.profiles}")
    lines.append(f"  teams             {agents.teams}")
    lines.append(
        f"  subagents         {agents.running} running · {agents.queued} queued · "
        f"{agents.settled} settled (retained)"
    )
    if agents.max_running is not None:
        lines.append(
            f"  capacity          {agents.max_running} concurrent"
            + ("  (AT CAPACITY)" if agents.at_capacity else "")
        )
    if agents.tree:
        lines.append(f"  tree (this session only, depth {agents.max_depth}):")
        lines += _tree_lines(agents.tree, agents.deeper)
    else:
        lines.append("  no subagents launched in this session")
    # Stated, not implied: only this session's tree is observable, so a report
    # that showed one tree could otherwise be read as a fleet-wide total.
    lines.append("  (other sessions report busy/pending and memory only)")

    lines += [
        "",
        "## Environment",
        f"  terminal          {env.term or '—'}"
        + (f" · {env.terminal_size[0]}x{env.terminal_size[1]}" if env.terminal_size else "")
        + (f" · {env.colorterm}" if env.colorterm else ""),
        f"  multiplexer       {env.multiplexer or 'none'}",
        f"  tty               {'yes' if env.is_tty else 'no'}",
        f"  theme             {env.theme or '—'}",
        f"  approvals         {env.approval_mode or '—'}",
        f"  browser           {env.browser_backend or 'none'}"
        + (f" · {env.browser_name}" if env.browser_name else ""),
        "  mobile            "
        + (
            ("installed" if env.mobile_installed else "not installed")
            + (" · healthy" if env.mobile_healthy else "")
            + (f" · port {env.mobile_port}" if env.mobile_installed and env.mobile_port else "")
        ),
        f"  mcp               {env.mcp_connected}/{env.mcp_configured} connected"
        + (" · still connecting" if env.mcp_settling else "")
        + (f" · {env.mcp_failed} failed" if env.mcp_failed and not env.mcp_settling else ""),
    ]
    # Failures are suppressed while SETTLING for the reason McpStartupOutcome
    # states: naming a server as failed mid-handshake produces bug reports about
    # servers that came up a second later.
    if env.mcp_failures and not env.mcp_settling:
        for name, message in env.mcp_failures:
            lines.append(f"      {name}: {relativise_home_everywhere(message)}")
    lines.append(f"  guides            {env.guides}")
    lines.append(f"  skills            {env.skills}")
    # NAMES ONLY, and this is the line the redaction test reads.
    lines.append(
        f"  credentials       {len(env.credential_keys)} keys"
        + (f": {', '.join(env.credential_keys)}" if env.credential_keys else "")
    )

    if snapshot.degraded:
        lines += ["", "## Could not read"]
        for name, reason in snapshot.degraded:
            lines.append(f"  {name}: {relativise_home_everywhere(reason)}")

    body = "\n".join(lines)
    # A FINAL whole-document pass, on top of the per-channel `_scrub` above.
    # Belt and braces deliberately: the per-site calls document intent at the
    # two channels known to carry free text, and this one covers any field a
    # later change adds without remembering to scrub it. Both are cheap string
    # replacements over a ~50-line document (review round 1, B2).
    body = relativise_home_everywhere(body)

    # FENCED, reversing this module's original "no fence" position. The
    # reasoning against a fence is sound for prose and does not survive contact
    # with this payload: 43 of ~55 lines are `  label<spaces>value`, and GFM
    # collapses space runs and folds them into one `<p>...<br>` paragraph — the
    # two-space indent even reads as a list continuation, so a session block
    # came out as an actual `<ul><li>`. Verified against GitHub's own /markdown
    # API rather than assumed (UX round 1, U1). A user cannot "want it without a
    # fence" because without one the alignment that makes it readable is gone,
    # and the failure is invisible to the person pasting it.
    #
    # The triage header stays OUTSIDE the fence so it remains greppable and
    # readable in a notification email, and the `<details>` wrapper keeps a
    # 55-line block from burying the reporter's own words.
    stamp = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.captured_at))
        if snapshot.captured_at
        else "unknown"
    )
    return (
        f"{head}\n\n"
        f"<details><summary>/info report — captured {stamp}</summary>\n\n"
        f"```text\n{body}\n```\n\n"
        "</details>\n"
    )
