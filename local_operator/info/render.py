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

from local_operator.i18n.keys import tui_info as _ik
from local_operator.i18n.keys import tui_small as _x
from local_operator.info.copy import info_duration, info_state
from local_operator.info.copy import info_text as _i
from local_operator.info.model import (
    UNKNOWN,
    InfoSnapshot,
    SessionsInfo,
    SubagentLine,
    format_bytes,
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

    DELIBERATELY NOT CACHED, and the round-2 commit message describing it as
    "memoised" was wrong about the code (review round 3, M1). Recomputing per
    call is the safer behaviour and the one that was verified: a cache would
    pin a TRANSIENT failure for the life of the process, so a home that became
    resolvable again would still export unrelativised paths. Both relativisers
    read this same function, so they cannot disagree within one document even
    though each call is independent. The cost is a `Path.home()` per row, which
    is an environment lookup on a screen the user opened by hand.
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
        return _i(_ik.latest_line_installed_unreadable())
    if install.latest_known is None:
        return _i(_ik.latest_line_never_checked())
    from local_operator.update import TTL_S

    age = install.latest_age_s
    if age is None:
        return install.latest_known
    stale = _i(_ik.latest_stale()) if age > TTL_S else ""
    return _i(
        _ik.latest_line_checked(
            latest_known=str(install.latest_known),
            age=info_duration(age),
            stale=str(stale),
        ),
    )


#: Nouns whose plural is not ``+ "s"``. A map rather than a second helper so
#: the next irregular noun is a data change.
_IRREGULAR_PLURALS = {"trajectory": "trajectories", "category": "categories"}


def plural(count: int, noun: str) -> str:
    """``"1 runtime"`` / ``"5 runtimes"`` / ``"5 trajectories"``.

    Lives HERE, one layer below the panel, because the panel imports this
    module and not the other way round — and because the export and the screen
    stating the same fact in different grammar is precisely the drift that put
    ``1 sessions + 1 subagents`` into pasted bug reports while the screen read
    correctly. One implementation, used by both.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {_IRREGULAR_PLURALS.get(noun, noun + 's')}"


#: How many pids the not-answering sentence names before it stops listing
#: them. A host with ten such sessions has one problem, so the sentence stays a
#: sentence: the count leads and the remainder is summarised.
_NOT_ANSWERING_PIDS = 3


def not_answering_clause(sessions: SessionsInfo | None) -> str:
    """The sentence for a runtime that has stopped reporting. ``""`` if none.

    ONE sentence, shared by the export and the panel, for the reason
    :func:`plural` is shared: this text lands in bug reports pasted off the
    screen, and two copies would drift into admitting different amounts.

    It says what was MEASURED and never what it means. ``registry.classify`` is
    explicit that a stale heartbeat is degraded-responsiveness evidence — the
    beat is authored by the runtime's own event loop, so a long turn produces
    the same reading as a frozen process — and the sentence therefore names the
    age, keeps the pid (which is the evidence that the process is still there),
    and stops short of any diagnosis.

    The remedy names the ladder's FIRST rung, and that is a correction rather
    than a preference (design round 2, D3). Which rung reaches which shape is
    readable from the ladder itself: ``control._identity_by_record`` re-reads
    the record's own fields only while the beat is INSIDE
    ``HEARTBEAT_TIMEOUT_S``, so on a record :func:`registry.classify` has
    already called ``wedged`` the forced rung cannot convert a refusal into a
    stop — the rung that acts here is the plain one, whose start-time proof
    ``_identity_by_start_time`` admits precisely BECAUSE the beat has lapsed.
    Naming only the forced stop handed the reader a heavier command that cannot
    do more on this state, and it said nothing about what running it costs.
    """
    if sessions is None or not sessions.available or not sessions.wedged:
        return ""
    # The COUNT is the header's, not the row list's: the two can disagree in a
    # hand-built ``SessionsInfo``, and the sentence sits under a header that
    # states this number — a lead that contradicts the line above it reads as
    # two different problems (review round 2, N2). The ROWS still supply the
    # age and the pids, because those are the parts no header carries.
    count = sessions.wedged
    one = count == 1
    rows = [line for line in sessions.lines if line.state == "wedged"]
    if not rows:
        # A hand-built ``SessionsInfo`` (a test, a partial snapshot) whose count
        # and rows disagree. The DISCLOSURE still holds — a count that includes
        # these runtimes is as of a heartbeat that stopped — but the age, the
        # pid and the remedy are the parts that need a row, and inventing a pid
        # would be worse than the vaguer sentence.
        return _i(_ik.not_answering_clause_no_rows(count=count))
    worst = max(rows, key=lambda line: line.heartbeat_age_s)
    listed = min(len(rows), _NOT_ANSWERING_PIDS)
    pids = ", ".join(str(line.pid) for line in rows[:listed])
    if count > listed:
        pids += _i(_ik.not_answering_clause_more(remaining=str(count - listed)))
    # The remedy names ONE pid on the single-row sentence and a PATTERN when the
    # row list is longer: the old form listed up to three pids and then handed
    # the reader a command for an arbitrary one of them (review round 2, N2).
    remedy = (
        _i(_ik.not_answering_clause_remedy_one(pid=str(worst.pid)))
        if one
        else _i(_ik.not_answering_clause_remedy_many())
    )
    return _i(
        _ik.not_answering_clause_full(
            count=count,
            age=info_duration(worst.heartbeat_age_s),
            pids=str(pids),
            remedy=remedy,
        ),
    )


def _tree_lines(tree: tuple[SubagentLine, ...], deeper: int) -> list[str]:
    out: list[str] = []
    for node in tree:
        indent = "  " * node.depth
        role = f" [{node.agent_role}]" if node.agent_role else ""
        out.append(f"  {indent}- {node.label or node.job_id}{role}: {info_state(node.status)}")
    if deeper:
        out.append(_i(_ik.tree_lines_deeper(deeper=deeper)))
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
    lines: list[str] = [_i(_ik.export_heading_install())]
    lines += [
        _i(_ik.export_version(version=str(_value(install.version)))),
        _i(_ik.export_latest_known(latest_line=str(_latest_line(snapshot)))),
        _i(_ik.export_install_kind(kind=str(_value(install.kind)))),
        _i(_ik.export_install_path(prefix=str(_value(relativise_home(install.prefix))))),
        _i(_ik.export_interpreter(executable=str(_value(relativise_home(install.executable))))),
        # Included unconditionally, and marked when it diverges: a maintainer
        # reading a bug report needs to know the reporter's runtime was running
        # a checkout rather than the version the first line claims.
        _i(
            _ik.export_running_code(
                import_path=str(_value(relativise_home(install.import_path))),
                has_is_shadowed_install=str("yes" if is_shadowed_install(install) else "no"),
                has_import_path_foreign=str("yes" if install.import_path_foreign else "no"),
            ),
        ),
        _i(
            _ik.export_source(
                has_is_git_snapshot=str("yes" if install.is_git_snapshot else "no"),
                has_source_ref=str("yes" if install.source_ref else "no"),
                source_ref=str(install.source_ref[:12]),
            ),
        ),
        _i(
            _ik.export_build_age(
                build_age_s=(
                    info_duration(install.build_age_s)
                    if install.build_age_s is not None
                    else "unmeasured"
                )
            )
        ),
        _i(
            _ik.export_python(
                has_install_python_version_and_install_python_implementation=str(
                    ("yes" if (install.python_version and install.python_implementation) else "no"),
                ),
                python_version=str(install.python_version),
                python_implementation=str(install.python_implementation),
                install_python_version_or_install_python_implementation=str(
                    _value(
                        (install.python_version or install.python_implementation),
                    ),
                ),
            ),
        ),
        _i(
            _ik.export_platform(
                platform=str(_value(install.platform)),
                machine=str(_value(install.machine)),
            ),
        ),
        "",
        _i(_ik.export_heading_session()),
        _i(_ik.export_session_id(session_id=str(_value(process.session_id)))),
        _i(_ik.export_kind(kind=str(_value(process.kind)))),
        _i(_ik.export_pid(pid=str(process.pid))),
        _i(_ik.export_model(model_label=str(_value(process.model_label)))),
        _i(
            _ik.export_effective_model(
                process_effective_model_or_process_model_label=str(
                    _value(
                        (process.effective_model or process.model_label),
                    ),
                ),
            ),
        ),
        _i(
            _ik.export_uptime(
                uptime_s=(
                    info_duration(process.uptime_s)
                    if process.uptime_s is not None
                    else "unmeasured"
                )
            )
        ),
        _i(_ik.export_working_dir(cwd=str(_value(relativise_home(process.cwd))))),
        _i(
            _ik.export_config_dir(
                config_dir=str(_value(relativise_home(process.config_dir))),
                has_config_dir_redirected=str("yes" if process.config_dir_redirected else "no"),
            ),
        ),
        _i(_ik.export_cache_dir(cache_dir=str(_value(relativise_home(process.cache_dir))))),
        _i(
            _ik.export_agent_home(
                agent_home=str(_value(relativise_home(process.agent_home))),
                has_agent_home_redirected=str("yes" if process.agent_home_redirected else "no"),
            ),
        ),
        _i(_ik.export_log_dir(log_dir=str(_value(relativise_home(process.log_dir))))),
        _i(
            _ik.export_control_port(
                has_control_port=str("yes" if process.control_port else "no"),
                control_port=str(process.control_port),
            ),
        ),
        _i(
            _ik.export_protocol(
                has_protocol=str("yes" if process.protocol else "no"),
                protocol=str(process.protocol),
            ),
        ),
        "",
        _i(_ik.export_heading_sessions()),
    ]

    if not sessions.available:
        lines.append(_i(_ik.export_sessions_unavailable()))
    elif not sessions.lines:
        lines.append(_i(_ik.export_sessions_empty()))
    else:
        # The prose says what the evidence supports; the bracketed token per row
        # below stays the STATE WORD, printed verbatim so a reader can grep it
        # against ``--json`` and against the record's own field. That split is
        # deliberate: the vocabulary is a wire contract with ~15 call sites and
        # the desktop catalogue's ``status.code``, and renaming it would change a
        # value to restate a sentence. Every SENTENCE is honest.
        lines.append(
            _i(
                _ik.export_sessions_summary(
                    live=sessions.live,
                    wedged=str(sessions.wedged),
                    total=sessions.total,
                    has_build_skew=str("yes" if sessions.build_skew else "no"),
                ),
            )
        )
        for line in sessions.lines:
            mark = "*" if line.is_self else "-"
            name = line.conversation_name or line.session_id or str(line.pid)
            # A wedged row sheds two fields rather than merely ordering them
            # last (design round 2, D4): usage is sampled for LIVE pids only, so
            # the "—" this field would print is a measurement that was never
            # taken for this one, and "busy" is the record's PRE-silence flag —
            # the row exists to remove unqualified progress, so it must not end
            # by asserting activity. It ends on the facts it exists for.
            fields = [
                f"{mark} [{line.state}] {name}",
                line.kind,
                _i(_ik.export_pid_fallback(pid=str(line.pid))),
            ]
            fields.append(info_duration(line.uptime_s))
            if line.state == "wedged":
                # The measured half of "not answering", beside the state token
                # rather than only in the caveat below: a pasted export is read
                # on its own, and `[wedged]` with no age beside it is exactly
                # the bare verdict the wording on every other surface avoids.
                fields.append(
                    _i(
                        _x.info_panel_last_heartbeat_ago(
                            age=info_duration(line.heartbeat_age_s),
                        ),
                    ),
                )
            else:
                fields.append(format_bytes(line.footprint_bytes or line.rss_bytes))
                if line.busy:
                    fields.append(_i(_x.info_panel_busy_word()))
            if line.pending:
                fields.append(_i(_x.info_panel_needs_pending(pending=str(line.pending))))
            lines.append("  " + " · ".join(fields))
            lines.append(f"      {relativise_home(line.cwd)} · {line.model_label or '—'}")

    lines += ["", _i(_ik.export_heading_agents())]
    if sessions.available:
        # The fleet answer FIRST: it is the question the section is opened
        # with, and the per-session detail below it is the breakdown.
        # ``live + wedged``: both name a running pid and both contribute to the
        # sum below, so the denominator printed here has to match it.
        lines.append(
            _i(
                _ik.export_runtimes(
                    sessions_live_sessions_wedged=sessions.live + sessions.wedged,
                    live=sessions.live,
                    wedged=str(sessions.wedged),
                ),
            )
        )
        # Same rule as the screen: the sum runs over runtimes that REPORTED, so
        # when some did not the total is a floor, and when none did there is no
        # total at all. This text is what lands in a bug report, where a
        # fabricated zero is hardest to challenge after the fact.
        # The SAME predicate as the panel's, deliberately: these two guards
        # drifted apart once (the export asserted ``>=0 total — 0 sessions + 0
        # subagents`` in a cell where the screen correctly refused to name a
        # number), and the export is the copy that outlives the screen inside
        # a bug report. Refuse the total whenever nothing was measured — not
        # merely when nobody reported.
        if sessions.subagents_unreported and not sessions.fleet_trajectories:
            lines.append(
                _i(
                    _ik.export_trajectories_unreported(
                        subagents_unreported=str(sessions.subagents_unreported),
                        sessions_live_sessions_wedged=sessions.live + sessions.wedged,
                    ),
                )
            )
        else:
            bound = ">=" if sessions.subagents_unreported else ""
            # Inflected through the shared helper, like the panel: this line is
            # pasted verbatim into issues, and ``1 sessions + 1 subagents`` —
            # the state every fresh install passes through — reads as a
            # template nobody finished rather than as a measurement.
            addends = _i(
                _ik.agents_addends(
                    sessions=sessions.fleet_session_trajectories,
                    subagents=sessions.fleet_subagents_running,
                ),
            )
            # Named beside the total, never added to it — see the panel's copy
            # of this rule. A queued child spends nothing, so it is not a
            # trajectory; but ``0 total`` alone, printed above a tree of
            # ``queued`` rows, is the export repeating a contradiction the
            # screen no longer makes.
            if sessions.fleet_subagents_queued:
                addends += _i(
                    _ik.export_queued_suffix(
                        fleet_subagents_queued=sessions.fleet_subagents_queued,
                    ),
                )
            lines.append(
                _i(
                    _ik.export_trajectories(
                        bound=str(bound),
                        fleet_trajectories=sessions.fleet_trajectories,
                        addends=str(addends),
                    ),
                )
            )
    lines.append(_i(_ik.export_profiles(profiles=str(agents.profiles))))
    lines.append(_i(_ik.export_teams(teams=str(agents.teams))))
    if agents.roster_unread:
        # An unreadable roster is not a roster of zero, in the export exactly
        # as on screen — this text is what lands in a bug report, where a
        # fabricated zero is hardest to challenge.
        lines.append(_i(_ik.export_roster_unreadable()))
    else:
        lines.append(
            _i(
                _ik.export_session_counts(
                    running=agents.running,
                    queued=agents.queued,
                    settled=agents.settled,
                ),
            )
        )
    if agents.max_running is not None:
        lines.append(
            _i(
                _ik.export_capacity(
                    max_running=agents.max_running,
                    has_at_capacity=str("yes" if agents.at_capacity else "no"),
                ),
            )
        )
    if agents.tree:
        lines.append(_i(_ik.export_tree_heading(max_depth=str(agents.max_depth))))
        lines += _tree_lines(agents.tree, agents.deeper)
    elif agents.roster_unread:
        lines.append(_i(_ik.export_tree_unreadable()))
    else:
        lines.append(_i(_ik.export_tree_empty()))
    # Stated, not implied: the COUNTS above are fleet-wide and the TREE is not,
    # so a report showing one tree could otherwise be read as the whole story.
    if agents.cross_session_known:
        lines.append(_i(_ik.export_tree_counts()))
    else:
        lines.append(_i(_ik.export_tree_legacy()))
    # Both caveats qualify the SAME total, so they are collected and emitted as
    # one parenthetical rather than as two stacked ones: a host that is both
    # mid-upgrade and holding a wedged pid printed two bracketed apologies in a
    # row, which reads as boilerplate in a pasted report and invites skipping
    # the second. The clause wording is unchanged — only the packaging.
    caveats: list[str] = []
    if sessions.available and sessions.subagents_unreported:
        # "lower bound", not a total: the sum above excluded every session that
        # did not report, and a bug report must not present it as complete.
        #
        # Inflected, like the panel's copy: this text is pasted into an issue
        # verbatim, and "1 sessions run ... and do not report" reads as a
        # template nobody finished rather than as a measurement.
        caveats.append(_i(_ik.export_lower_bound(count=sessions.subagents_unreported)))
    if sessions.available and sessions.wedged:
        # PARITY with the panel, which has said this since the counts landed,
        # now through ONE shared sentence: the totals above deliberately include
        # these runtimes — a quiet pid can still have children working — so their
        # contribution is as of their last heartbeat, and an export that omitted
        # the disclosure would present stale counts as current in the one
        # artifact that outlives the screen. The sentence also carries the
        # REMEDY, which is the part a reader of a pasted export has no other way
        # to get: the row above names the state and the measured age, and the
        # forced stop is the rung that acts on it.
        caveats.append(not_answering_clause(sessions))
    if caveats:
        # ``·`` and not ``;``: the wedged clause already contains a semicolon
        # of its own, so a semicolon joiner produced two different grammatical
        # levels sharing one punctuation mark. The interpunct is this screen's
        # established separator, and it cannot collide with clause punctuation.
        lines.append(f"  ({' · '.join(caveats)})")

    lines += [
        "",
        _i(_ik.export_heading_environment()),
        _i(
            _ik.export_terminal(
                env_term_or=str(env.term or "—"),
                has_terminal_size=str("yes" if env.terminal_size else "no"),
                terminal_size=str(env.terminal_size[0]) if env.terminal_size else "",
                env_terminal_size=str(env.terminal_size[1]) if env.terminal_size else "",
                has_colorterm=str("yes" if env.colorterm else "no"),
                colorterm=str(env.colorterm),
            ),
        ),
        _i(
            _ik.export_multiplexer(
                env_multiplexer_or_none=str(env.multiplexer or _i(_ik.env_none())),
            ),
        ),
        _i(_ik.export_tty(has_is_tty=str(("yes" if env.is_tty else "no")))),
        _i(_ik.export_theme(env_theme_or=str(env.theme or "—"))),
        _i(_ik.export_approvals(env_approval_mode_or=str(env.approval_mode or "—"))),
        _i(
            _ik.export_browser(
                env_browser_backend_or_none=str(env.browser_backend or _i(_ik.env_none())),
                has_browser_name=str("yes" if env.browser_name else "no"),
                browser_name=str(env.browser_name),
            ),
        ),
        _i(
            _ik.export_mobile(
                has_mobile_installed=str("yes" if env.mobile_installed else "no"),
                has_mobile_healthy=str("yes" if env.mobile_healthy else "no"),
                has_env_mobile_installed_and_env_mobile_port=str(
                    ("yes" if (env.mobile_installed and env.mobile_port) else "no"),
                ),
                mobile_port=str(env.mobile_port),
            ),
        ),
        _i(
            _ik.export_mcp(
                mcp_connected=str(env.mcp_connected),
                mcp_configured=str(env.mcp_configured),
                has_mcp_settling=str("yes" if env.mcp_settling else "no"),
                has_env_mcp_failed_and_not_env_mcp_settling=str(
                    ("yes" if (env.mcp_failed and not env.mcp_settling) else "no"),
                ),
                mcp_failed=env.mcp_failed,
            ),
        ),
    ]
    # Failures are suppressed while SETTLING for the reason McpStartupOutcome
    # states: naming a server as failed mid-handshake produces bug reports about
    # servers that came up a second later.
    if env.mcp_failures and not env.mcp_settling:
        for name, message in env.mcp_failures:
            lines.append(f"      {name}: {relativise_home_everywhere(message)}")
    lines.append(_i(_ik.export_guides(guides=str(env.guides))))
    lines.append(_i(_ik.export_skills(skills=str(env.skills))))
    # NAMES ONLY, and this is the line the redaction test reads.
    lines.append(
        _i(
            _ik.export_credentials(
                count=len(env.credential_keys),
                has_credential_keys=str("yes" if env.credential_keys else "no"),
                join=str(", ".join(env.credential_keys)),
            ),
        )
    )

    if snapshot.degraded:
        lines += ["", _i(_ik.export_heading_degraded())]
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
        else _i(_ik.latest_unknown())
    )
    summary = _i(_ik.export_summary(stamp=str(stamp)))
    return (
        f"{head}\n\n"
        f"<details><summary>{summary}</summary>\n\n"
        f"```text\n{body}\n```\n\n"
        "</details>\n"
    )
