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

from local_operator.info.model import InfoSnapshot, SubagentLine

#: Fields the export never prints, restated here as a NAME LIST purely so a
#: reader of this module sees the rule without chasing three dataclasses. The
#: enforcement is structural (the fields do not exist on the collected shapes);
#: this is documentation, not a filter.
NEVER_EXPORTED = ("control_key", "session_key", "credential values")


def relativise_home(path: str) -> str:
    """``/Users/x/repos/y`` → ``~/repos/y``. Everything below ``~`` is kept.

    The username is the identifying part; the path BELOW the home directory is
    the diagnostic part (``~/local-operator``, ``~/workspace/repos/lo-x`` all
    say something a maintainer needs). So exactly one segment is removed.
    """
    if not path:
        return path
    home = str(Path.home())
    if path == home:
        return "~"
    prefix = home + os.sep
    if path.startswith(prefix):
        return "~" + os.sep + path[len(prefix) :]
    return path


def _duration(seconds: float | None) -> str:
    """``4m`` / ``3h 12m`` / ``2d 4h`` — the export's one duration spelling."""
    if seconds is None:
        return "unknown"
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h {(total % 3600) // 60}m"
    return f"{total // 86400}d {(total % 86400) // 3600}h"


def _bytes(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1 << 30:
        return f"{value / (1 << 30):.1f} GB"
    return f"{value / (1 << 20):.0f} MB"


def _latest_line(snapshot: InfoSnapshot) -> str:
    """The three PyPI states, never collapsed into two.

    ``None`` is "we have never asked", which is a different fact from "you are
    up to date" — and on the broken network that ``/info`` is usually opened
    over, printing the latter from the former is an outright lie.
    """
    install = snapshot.install
    if install.latest_known is None:
        return "unknown (never checked)"
    from local_operator.update import TTL_S

    age = install.latest_age_s
    if age is None:
        return install.latest_known
    stale = " (stale)" if age > TTL_S else ""
    return f"{install.latest_known} · checked {_duration(age)} ago{stale}"


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
    stamp = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.captured_at))
        if snapshot.captured_at
        else "unknown"
    )

    lines: list[str] = [head, f"captured {stamp}", "", "## Install"]
    lines += [
        f"  version           {install.version or '—'}",
        f"  latest known      {_latest_line(snapshot)}",
        f"  install kind      {install.kind or '—'}",
        f"  install path      {relativise_home(install.prefix)}",
        f"  interpreter       {relativise_home(install.executable)}",
        # Included unconditionally, and marked when it diverges: a maintainer
        # reading a bug report needs to know the reporter's runtime was running
        # a checkout rather than the version the first line claims.
        f"  running code      {relativise_home(install.import_path) or '—'}"
        + (
            "  (NOT under the install path — a checkout is shadowing it)"
            if install.import_path_foreign
            else ""
        ),
        f"  source            {'git snapshot' if install.is_git_snapshot else 'PyPI wheel'}"
        + (f" @ {install.source_ref[:12]}" if install.source_ref else ""),
        f"  build age         {_duration(install.build_age_s)}",
        f"  python            {install.python_version} ({install.python_implementation})",
        f"  platform          {install.platform or '—'} · {install.machine or '—'}",
        "",
        "## This session",
        f"  session id        {process.session_id or '—'}",
        f"  kind              {process.kind or '—'}",
        f"  pid               {process.pid}",
        f"  model             {process.model_label or '—'}",
        f"  effective model   {process.effective_model or process.model_label or '—'}",
        f"  uptime            {_duration(process.uptime_s)}",
        f"  working dir       {relativise_home(process.cwd)}",
        f"  config dir        {relativise_home(process.config_dir)}"
        + ("  (redirected)" if process.config_dir_redirected else ""),
        f"  cache dir         {relativise_home(process.cache_dir)}",
        f"  agent home        {relativise_home(process.agent_home)}"
        + ("  (redirected)" if process.agent_home_redirected else ""),
        f"  log dir           {relativise_home(process.log_dir)}",
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
            memory = _bytes(line.footprint_bytes or line.rss_bytes)
            extra = " · busy" if line.busy else ""
            extra += f" · needs {line.pending}" if line.pending else ""
            lines.append(
                f"  {mark} [{line.state}] {name} · {line.kind} · pid {line.pid} · "
                f"{_duration(line.uptime_s)} · {memory}{extra}"
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
            lines.append(f"      {name}: {message}")
    lines.append(f"  guides            {env.guides}")
    lines.append(f"  skills            {env.skills}")
    lines.append(f"  tools             {env.tools}")
    # NAMES ONLY, and this is the line the redaction test reads.
    lines.append(
        f"  credentials       {len(env.credential_keys)} keys"
        + (f": {', '.join(env.credential_keys)}" if env.credential_keys else "")
    )

    if snapshot.degraded:
        lines += ["", "## Could not read"]
        for name, reason in snapshot.degraded:
            lines.append(f"  {name}: {reason}")

    return "\n".join(lines) + "\n"
