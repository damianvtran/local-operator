"""Every session member the TUI reaches must be DECLARED on a protocol.

The bug this closes has a shape, and the shape has repeated. The front end
talks to its session through ``getattr(session, "name", None)`` duck-probes
against attributes that no protocol declares. A probe string is data, so:

* a rename on the facade leaves every probe reading ``None`` — pyright sees
  nothing, and the capability silently disappears rather than failing;
* a typo in the string is undetectable for the same reason;
* the graceful fallback the probe already has ("older owner, no such member")
  absorbs the failure and reports a plausible wrong answer instead of raising.

That is not hypothetical. ``/info`` reported zero subagents for a session that
had several because ``subagent_comms`` was private on the facade and the probe
returned ``None`` (see ``RemoteSession.subagent_comms``), and the
``is_remote``-conflation half of the same problem has been fixed site-locally
four times (#576, #609, #624, #625) with a fifth guarded in
``tests/unit/tui/test_noop_consumers.py``.

Site-local fixes do not close a class of bug. This does: it derives the member
set from the SOURCE — the session-valued attribute accesses and literal-string
duck-probes in the files listed in ``_SCANNED`` — and fails when one of them is
not declared on ``SessionProtocol`` or ``ViewerSessionProtocol``. Adding a new
undeclared duck-typed member therefore fails here rather than in a user's
terminal.

The derivation is deliberately syntactic rather than type-inferred: pyright
cannot follow ``getattr`` with a literal string, which is precisely why these
members escaped typing in the first place.

**What the derivation reaches, stated precisely — it is not "every access".**
It sees ``<expr>.member`` and a two-or-more-argument call to any name in
``_PROBE_CALLS`` (``getattr``, ``hasattr``, and ``info/collect.py``'s ``_attr``
wrapper) with a literal member name, where ``<expr>`` is one of the bindings
registered for that file in ``_SCANNED``. It is blind to:

* local aliases — ``s = self._session; s.member`` (see ``_SESSION_EXPRS``);
* any other attribute or helper return holding a session
  (``self._current_session().member``);
* computed probe names — ``getattr(session, probe, None)`` where ``probe`` is a
  variable. ``_session_is_busy`` (``app.py:9313``) is a live example: it loops
  ``for probe in ("is_busy", "busy")``, neither name exists on either class, so
  it always returns ``False`` and its caller takes a dead branch. The guard
  runs over that exact line and structurally cannot see it. That is the
  syntactic approach's ceiling, recorded here so nobody reads a green run as
  "no duck-probe is broken";
* sessions arriving as differently-named parameters;
* files outside ``_SCANNED``.

A green run means "no *reachable-by-this-derivation* probe is undeclared", not
"the front end is fully typed". Widening any of the above is a matter of adding
an expression or a path — the machinery does not change.

**Do not narrow pyright's path to ``local_operator/``.** Half of the
conformance claim is not in ``session/`` at all: it is carried by
``_static_conformance_is_checked_by_pyright`` at the bottom of THIS file, whose
body is the only place the two classes are assigned to the protocol types. A
viewer member broken with no internal caller gives ``pyright
local_operator/session/`` zero errors, and only checking this file reports it
(QA round 1). Excluding ``tests/`` from pyright, or pointing it at the package
alone, therefore disarms the static half silently and leaves the suite green.
"""

from __future__ import annotations

import ast
from pathlib import Path

import local_operator
from local_operator.session.protocol import SessionProtocol, ViewerSessionProtocol
from local_operator.session.remote import RemoteSession
from local_operator.session.session import Session

_ROOT = Path(local_operator.__file__).resolve().parent

#: Expressions in ``app.py`` that are unambiguously a session.
#:
#: Deliberately NOT ``target``/``current``/``sess``: those names are reused in
#: the file for widgets, rows and strings, and including them made an earlier
#: version of this probe report ``partition``, ``focus`` and ``label`` as
#: session members. A guard that cries wolf gets deleted, so it reads only the
#: bindings that always hold a session.
#:
#: ``source.session`` is the sidebar-source binding, and it is here because
#: omitting it let two real viewer-only members escape:
#: ``has_pending_gate_reply`` (``app.py:4715``) and
#: ``preserve_viewer_gate_reply`` (``app.py:15279``) — both approval-gate
#: members sitting directly beside ones this protocol already declared.
#:
#: ``self.app.session`` is NOT here, and its removal is a fix rather than a
#: narrowing: it was registered while occurring nowhere in the package, so it
#: implied coverage of a spelling that does not exist. A binding that matches
#: nothing is worse than an absent one — it reads as watched. Every entry here
#: is now asserted to derive at least one member by
#: ``test_every_registered_session_binding_still_matches_the_source`` (QA round
#: 2, Q5), which is what makes that claim checkable instead of aspirational.
#:
#: These are BINDINGS, matched literally. ``s = self._session; s.member`` is
#: not covered: a local alias is a different expression, and the guard does no
#: dataflow. Following aliases means resolving assignments per scope, which is
#: where the false positives that nearly killed this probe come from — so the
#: boundary is deliberate, and this list is the thing to extend when a new
#: session binding appears.
_SESSION_EXPRS = frozenset(
    {
        "self._session",
        "session",
        "source.session",
    }
)

#: The binding that holds a session in the TUI's interaction helper.
#:
#: ``self.session`` was registered against ``app.py``, where it derives nothing
#: — the app spells it ``self._session``. Rather than drop the spelling (which
#: stops watching a name that IS live elsewhere), it is pointed at the file that
#: actually uses it. Scanning that file adds no undeclared members today, so the
#: entry costs a parse and buys a real binding instead of a fictional one.
_INTERACTION_SESSION_EXPRS = frozenset({"self.session"})

#: The bindings that hold a viewer facade in the desktop host.
#:
#: Separate from ``_SESSION_EXPRS`` because the name differs by host: the
#: bridge stores its facade as ``self.remote`` and copies it into a local
#: ``remote`` before use, which is a session binding by the same reasoning
#: ``self._session`` is one in the TUI.
_HOST_SESSION_EXPRS = frozenset({"remote", "self.remote"})

#: The bindings that hold a viewer facade in a desktop ROUTE module.
#:
#: A third spelling, and it had to be added rather than folded into the set
#: above: the routes reach the facade through the bridge they are handed by the
#: ``host(request).session(...)`` context manager, so the binding is
#: ``bridge.remote`` (and ``child.remote`` for the fork route's child), while
#: ``desktop_catalogues.py`` also copies it into a local ``remote`` exactly as
#: the utils host does.
#:
#: This is the binding QA round 2 (Q4) found the guard could not read, and the
#: three members behind it — ``bind_runtime``, ``admit_prompt``, ``answer_gate``
#: — are the more severe shape of the escape: HARD accesses in live HTTP
#: routes, so a rename is a 500 on the phone portal rather than a silent
#: ``None``. Adding this set needed no new machinery, only another literal
#: binding, which is why it is fixed here rather than deferred.
#:
#: Split PER ROUTE HOST rather than shared, for the reason
#: ``_INTERACTION_SESSION_EXPRS`` is separate from ``_SESSION_EXPRS``: a binding
#: registered against a file that does not use it is a coverage claim the file
#: cannot honour. All three spellings against all three route hosts produced
#: four (host, binding) pairs deriving nothing, and a shared set hid every one
#: of them behind a sibling host that did use the name — which is exactly the
#: hole ``test_every_registered_session_binding_still_matches_the_source``
#: closes below (review round 3, MAJOR-3 / QA round 3, Q7).
_LIFECYCLE_SESSION_EXPRS = frozenset({"bridge.remote", "child.remote"})
_ROUTE_SESSION_EXPRS = frozenset({"bridge.remote"})
_CATALOGUE_SESSION_EXPRS = frozenset({"bridge.remote", "remote"})

#: The bindings that hold a session in ``info/collect.py``.
#:
#: The ``/info`` collector takes its session as a plain ``session`` parameter,
#: so one name covers it. Deliberately NOT ``_SESSION_EXPRS``: that set carries
#: TUI-specific spellings (``self._session``, ``source.session``) which do not
#: occur here, and re-using it would imply a coverage claim this file cannot
#: make. Verified equivalent on today's source — scanning ``collect.py`` with
#: either set yields the identical seven members — so the narrow set is honest
#: rather than merely cheaper.
_INFO_SESSION_EXPRS = frozenset({"session"})

#: Files scanned for session duck-probes, with the bindings to read in each.
#:
#: Not just the TUI. ``ViewerSessionProtocol`` lives in ``session/`` rather
#: than in ``tui/`` precisely because the viewer facade has more than one
#: consumer, and the desktop host is the other one: it duck-probed
#: ``supports_completion_ack`` (``desktop_sessions.py:223``/``262``) — on
#: ``RemoteSession``, absent from ``Session``, declared on neither protocol —
#: which is exactly the escape this guard exists to close, one file outside
#: its original scope. A rename there made the phone portal report
#: completion-attention as unsupported, silently and with no error.
#:
#: ``info/collect.py`` is here because it is the host of the MOTIVATING bug —
#: the ``/info`` screen that reported zero subagents — and round 2 found the
#: guard did not observe it. The reviewer renamed ``RemoteSession.
#: subagent_comms``, i.e. re-shipped that exact regression, and got a green
#: guard and zero pyright errors; only a site-local test in
#: ``tests/unit/info/`` objected, which is precisely the kind of coverage this
#: file's docstring argues does not close a class of bug (review round 2,
#: MAJOR-1). A guard that misses the defect it was built from is not a guard.
#:
#: The desktop ROUTE modules are here for the same reason one file below them
#: is: they hold the same facade under a different binding. Only these three of
#: the five ``desktop_*`` route modules touch a session at all
#: (``desktop_profiles.py`` and ``desktop_radient.py`` derive zero members), so
#: listing those two would buy scan cost and no coverage.
#:
#: Adding a host is one entry here plus whatever it turns out to be probing.
#: Other ``tui/`` modules also probe sessions (``session_interaction.py``,
#: ``widgets/session_panel.py``, ``widgets/todo_panel.py``,
#: ``widgets/subagent_panel.py``, ``widgets/wake_panel.py``), but every member
#: they read is already declared or excluded below, so listing them today buys
#: scan cost and no coverage; add one when it starts probing something new.
#:
#: The third element is that host's MINIMUM member count, and it is per host for
#: a structural reason: ``app.py`` derives 105 of the 113 DISTINCT PUBLIC
#: MEMBERS (and 363 of 414 ``file:line`` SITES, deduped per member and line), so
#: any single global floor loose enough to survive ordinary churn there cannot
#: notice a smaller host going dark at all. Measured, not guessed: dropping the
#: desktop utils host costs 3 VIEWER-ONLY MEMBERS out of 49 and dropping
#: ``info/collect.py`` costs 0, so both slid under a global ``>= 40`` — the exact
#: slack review round 2 (MINOR-1) raised, reproduced one floor higher. A count
#: stated beside each path fires on the host that actually decayed and names it.
#:
#: Every figure above names the quantity it counts, because two of them were
#: wrong when this argument was first made — "104 of the 139 sites" crossed a
#: member count with a site count, and the viewer-only total was off by one
#: (review round 3, MINOR-6 / QA round 3, Q9). The dominance claim is now
#: ASSERTED, as a RATIO, in
#: ``test_app_py_dominates_the_derivation_so_a_global_floor_cannot_work``; the
#: absolute counts here are a snapshot for the reader and will drift with
#: ordinary work, which is exactly why the test does not pin them.
#:
#: Set a few members below the current value: enough headroom that deleting a
#: call site is not a test failure, tight enough that losing a BINDING or a PATH
#: is. Removing a probe legitimately means lowering the number in the same
#: commit, which is where the argument for it belongs.
_SCANNED = (
    ("tui/app.py", _SESSION_EXPRS, 95),
    ("server/utils/desktop_sessions.py", _HOST_SESSION_EXPRS, 6),
    ("info/collect.py", _INFO_SESSION_EXPRS, 5),
    ("server/routes/desktop_lifecycle.py", _LIFECYCLE_SESSION_EXPRS, 7),
    ("server/routes/desktop_sessions.py", _ROUTE_SESSION_EXPRS, 4),
    ("server/routes/desktop_catalogues.py", _CATALOGUE_SESSION_EXPRS, 3),
    ("tui/session_interaction.py", _INTERACTION_SESSION_EXPRS, 2),
)

#: Call shapes that read one named attribute off a session, by function name.
#:
#: ``getattr``/``hasattr`` are the builtins. ``_attr`` is ``info/collect.py``'s
#: own wrapper (``_attr(session, "name", default)``): it exists because the
#: members it reads are PROPERTIES on the real ``Session`` and a property on an
#: unhealthy session can raise, which a bare ``getattr`` would let escape a
#: function whose contract is "safe on the paint path".
#:
#: It has to be listed because a wrapper is indistinguishable from any other
#: call to the AST, so adding ``collect.py`` to ``_SCANNED`` alone would have
#: derived nothing from the three lines that matter — the guard would have been
#: widened to the motivating host and still not observed it (review round 2,
#: MAJOR-1). The name is matched by ARITY and a literal second argument like
#: the builtins, so a same-named helper elsewhere cannot smuggle a probe past
#: this; and a NEW wrapper of this shape is one entry here.
_PROBE_CALLS = frozenset({"getattr", "hasattr", "_attr"})

#: Members the TUI probes on a session that belong to an OWNER, not a viewer.
#:
#: These are OPTIONAL-capability probes: every one is
#: ``getattr(session, name, None)`` followed by a ``callable()``/``None`` test
#: with a working fallback, because the TUI's session may be either kind. They
#: are excluded rather than declared because forcing a viewer to grow a no-op
#: stub for each would be worse than their absence — a stub returning a
#: plausible empty value cannot be distinguished by the caller from "this
#: session genuinely has nothing", which is the exact confusion that produced
#: the fabricated ``/info`` zero above.
#:
#: The soundness condition is machine-checked, not asserted in prose:
#: ``test_owner_only_probes_are_all_optional_capability_probes`` requires every
#: name here to appear ONLY as a 3-argument ``getattr`` (i.e. with a default)
#: and never as a hard attribute access. An earlier version of this comment
#: cited five ``app.py`` line numbers instead; they were accurate when written
#: and are worthless the moment anything above them moves, in a file of 33k
#: lines. The property is what makes the exclusion sound, so the property is
#: what gets asserted.
_OWNER_ONLY_CAPABILITY_PROBES = frozenset(
    {
        "active_team",
        "agent_brief",
        "attach_agent_profile",
        "attach_team",
        "attachment_restore_notice",
        "clear_agent_profile",
        "has_pending_fork",
        "journal_credential_change",
        "measure_preloaded_context",
        "preflight_usage",
        "refresh_frontend_usage",
        "routing_settings",
        "variables",
        "wears_inherited_title",
    }
)

#: Undeclared members that live on BOTH classes.
#:
#: Not viewer-surface, so out of this PR's scope, but real: the TUI duck-probes
#: them exactly like the viewer members and they are equally invisible to
#: pyright. Listed rather than silently skipped so the debt is visible and the
#: guard shrinks as they are declared. Declaring them belongs with the call-site
#: migration (Stage 3), not with this additive change.
#:
#: Every name here is asserted to be present on both classes by
#: ``test_the_exclusion_sets_state_true_facts`` — the set is a claim about the
#: code, not a mute list, so an entry that stops being true fails rather than
#: silently widening the guard's blind spot.
#:
#: ``subagent_comms`` is the MOTIVATING member of this whole file and it sits
#: here rather than on a protocol, which needs saying plainly. It is read by
#: ``/info`` and lives on both classes, so declaring it is Stage 3's call-site
#: work like every other name in this set. What matters for round 2 is that it
#: is now DERIVED at all: until ``info/collect.py`` joined ``_SCANNED`` the
#: reviewer could re-ship the original outage — rename it on the facade — with
#: a green guard and zero pyright errors (review round 2, MAJOR-1). Being an
#: entry in a checked set means a rename to a name that exists on NEITHER class
#: now fails here, which is the shape the original bug had.
_UNDECLARED_ON_BOTH_CLASSES = frozenset(
    {
        "acknowledge_attention",
        "active_agent",
        "active_team_name",
        "agent_registry",
        "context_breakdown",
        "epoch",
        "fork_snapshot",
        "frontend_state",
        "jobs",
        "mcp_manager",
        "mcp_startup",
        "pending_gate",
        "record_shell",
        "refresh_attention",
        "restored_usage",
        "subagent_comms",
        "subscribe_frontend",
        "team_registry",
        "wake_scheduler",
    }
)

#: Probed by a host but present on NEITHER class — the probe is already dead.
#:
#: Kept apart from ``_UNDECLARED_ON_BOTH_CLASSES`` because that set's whole
#: point is "this member exists, it is merely undeclared". Parking a name here
#: records the opposite and worse fact: the call site reads a member that does
#: not exist, so its ``getattr`` default is the only value it will ever see.
#: Filing these as ordinary debt would let the guard built to surface
#: silent-wrong-answers permanently silence one.
#:
#: * ``cwd`` — ``app.py`` passes ``getattr(self._session, "cwd", "")`` to
#:   ``saved_preview(...)``, so the saved preview's working directory is
#:   ALWAYS ``""``. Pre-existing (present at ``a8f98be3b``), behavioural, and
#:   out of scope for this additive change: deferred to Stage 3, where that
#:   call site is rewritten against a declared member. Cited by call SHAPE
#:   rather than by line number on purpose — a line number in a moving file is
#:   the rot this file eliminated elsewhere (review round 2, MINOR-3), and the
#:   name itself is machine-checked below, so the prose carries no claim the
#:   suite cannot verify.
#:
#: ``test_the_exclusion_sets_state_true_facts`` asserts these are absent from
#: both classes, so a name here that someone later implements fails the suite
#: and gets promoted rather than lingering as a false claim.
_KNOWN_MISSING_ON_BOTH_CLASSES = frozenset({"cwd"})

#: Removed by Stage 3; declaring it would entrench the flag this work retires.
_RETIRED = frozenset({"is_remote"})


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — an unparseable node is simply not a session
        return ""


def _session_members_touched(source: str, exprs: frozenset[str]) -> dict[str, list[int]]:
    """Session members one file reads, by name, with line numbers.

    ``exprs`` is per-file because the binding that holds a session differs by
    host: the TUI has ``self._session``, the desktop bridge has ``self.remote``.
    """
    tree = ast.parse(source)
    touched: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        # session.member / self._session.member
        if isinstance(node, ast.Attribute) and _unparse(node.value) in exprs:
            touched.setdefault(node.attr, []).append(node.lineno)
        # getattr(session, "member", ...) / hasattr(session, "member") /
        # _attr(session, "member", default) — see ``_PROBE_CALLS``.
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _PROBE_CALLS
                and len(node.args) >= 2
                and _unparse(node.args[0]) in exprs
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                touched.setdefault(node.args[1].value, []).append(node.lineno)
    return touched


def _site_label(relpath: str) -> str:
    """A scanned host's ``parent/file.py`` label for an assertion message.

    Keeps the parent directory, not just the basename: two scanned hosts are
    both called ``desktop_sessions.py`` (one under ``server/utils``, one under
    ``server/routes``), so a bare filename would name an ambiguous file in the
    very message whose job is to send the reader to the offending line.

    A shared helper rather than the rule inlined at each site, because it was
    inlined twice and the two copies had already drifted — one fixed, one still
    printing the ambiguous basename (review round 3, MINOR-5).
    """
    return "/".join(relpath.rsplit("/", 2)[-2:])


def _all_touched() -> dict[str, list[str]]:
    """Every scanned file's session members, mapped name -> ``file:line`` sites."""
    sites: dict[str, list[str]] = {}
    for relpath, exprs, _floor in _SCANNED:
        filename = _site_label(relpath)
        found = _session_members_touched((_ROOT / relpath).read_text(), exprs)
        for member, lines in found.items():
            sites.setdefault(member, []).extend(f"{filename}:{line}" for line in sorted(set(lines)))
    return sites


def _declared() -> set[str]:
    """Every name the two protocols declare.

    ``dir()`` alone is not enough: a bare annotation (``owner_version: str``)
    declares a member for both pyright and ``isinstance``, but creates no class
    attribute, so it does not appear in ``dir()``. Reading
    ``__annotations__`` as well is what makes the four instance attributes on
    ``ViewerSessionProtocol`` count as declared — the first run of this guard
    reported them as violations, which was the test being wrong rather than
    the protocol.
    """
    names: set[str] = set()
    for proto in (SessionProtocol, ViewerSessionProtocol):
        names.update(n for n in dir(proto) if not n.startswith("_"))
        # ``__mro__`` via getattr: pyright models it as a descriptor on the
        # metaclass and rejects the direct access on a Protocol class object.
        for klass in getattr(proto, "__mro__", ()):
            names.update(n for n in getattr(klass, "__annotations__", {}) if not n.startswith("_"))
    return names


def _members(klass: type, module: str, name: str) -> set[str]:
    """Every member of a session class, including instance attributes.

    ``dir(klass)`` sees only class-level members, so attributes assigned as
    ``self.x = ...`` in ``__init__`` are missing from it. That is not a corner
    case here: ``RemoteSession`` sets ``owner_version``, ``degraded_reason``,
    ``jobs`` and others that way, and ``Session`` sets ``agent_registry`` and
    ``team_registry`` that way while ``RemoteSession`` exposes them as
    properties. Comparing ``dir()`` to ``dir()`` therefore reported
    ``agent_registry`` as viewer-ONLY, which is false — both classes have it.

    So membership is ``dir()`` plus every ``self.<name> =`` store found by AST
    in the class body.
    """
    members = {n for n in dir(klass) if not n.startswith("_")}
    tree = ast.parse((_ROOT / "session" / module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and isinstance(sub.ctx, ast.Store)
                    and not sub.attr.startswith("_")
                ):
                    members.add(sub.attr)
            break
    return members


def test_every_session_member_a_host_touches_is_declared() -> None:
    """The guard itself.

    Fails with the offending names and their ``file:line`` sites, so the fix is
    "declare it on the protocol" rather than "go find what changed".
    """
    touched = _all_touched()
    declared = _declared()
    known = (
        declared
        | _OWNER_ONLY_CAPABILITY_PROBES
        | _UNDECLARED_ON_BOTH_CLASSES
        | _KNOWN_MISSING_ON_BOTH_CLASSES
        | _RETIRED
    )

    offenders = {
        name: sites
        for name, sites in touched.items()
        if not name.startswith("_") and name not in known
    }

    assert not offenders, (
        "a session host reads members that no protocol declares: "
        + ", ".join(f"{name} ({', '.join(sites)})" for name, sites in sorted(offenders.items()))
        + ". A duck-typed member is invisible to pyright, so a rename or a typo "
        "in the probe string degrades it to a silent None instead of an error. "
        "Declare it on SessionProtocol (both kinds of session have it) or on "
        "ViewerSessionProtocol (only an attached facade has it)."
    )


def test_remote_session_satisfies_the_viewer_protocol_at_runtime() -> None:
    """The static half is pyright's; this is the runtime half.

    ``isinstance`` against a ``runtime_checkable`` Protocol checks member
    PRESENCE only, not signatures — so it catches a rename or a deletion of a
    CLASS-level member (a method or property removed from ``RemoteSession``),
    which is this guard's job, and pyright catches the signatures.

    What it does NOT catch, contrary to what this docstring claimed for two
    rounds, is an ``__init__``-assigned attribute disappearing. The four below
    are hand-assigned to satisfy ``__new__``, so the ``isinstance`` passes
    whether or not ``__init__`` still sets them — the reviewer deleted
    ``owner_version`` from the class outright and got 7 tests passing against 2
    pyright errors (review round 2, MINOR-2). For those four the division of
    labour runs the other way: **pyright is the guard and this test is
    structurally blind.** Stated here because a test believed to cover a case it
    cannot is worse than an uncovered case.
    """
    session = RemoteSession.__new__(RemoteSession)
    # The instance attributes are assigned in ``__init__``, which dials a
    # socket. Set them directly: presence is what the protocol requires, and
    # constructing a real facade would make this a network test. See the
    # docstring: this assignment is exactly why the four are pyright's to guard.
    session.owner_version = ""
    session.owner_source_ref = ""
    session.degraded_reason = ""
    session.saved_preview_partial = False

    assert isinstance(session, ViewerSessionProtocol)
    assert isinstance(session, SessionProtocol)


def test_an_owner_session_is_not_a_viewer() -> None:
    """The negative case, which is what makes the positive one mean anything.

    If ``Session`` also satisfied ``ViewerSessionProtocol`` the split would be
    decorative and the TUI could not use the type to tell the two apart.
    """
    assert not isinstance(Session.__new__(Session), ViewerSessionProtocol)


def test_the_runtime_role_predicates_disagree_between_the_two_classes() -> None:
    """The predicates must actually discriminate.

    Asserted as a PAIR rather than per class: three predicates that returned
    the same value on both sides would type-check, pass a conformance test, and
    still be useless — which is the failure mode of the flag they replace
    (``is_remote`` is constant-True for every `lop` TUI session).
    """
    owner = Session.__new__(Session)
    viewer = RemoteSession.__new__(RemoteSession)

    assert owner.owns_runtime is True
    assert viewer.owns_runtime is False

    assert owner.outcome_is_synchronous is True
    assert viewer.outcome_is_synchronous is False

    assert owner.runtime_locality == "this-process"
    assert viewer.runtime_locality == "this-machine"


def test_the_viewer_protocol_covers_what_only_the_facade_has() -> None:
    """The viewer surface is DERIVED, not curated.

    Recomputes "members ``app.py`` touches that exist on ``RemoteSession`` and
    not on ``Session``" and asserts every one is declared. Without this, the
    guard above could be satisfied forever by appending names to the exclusion
    sets instead of declaring them.
    """
    touched = _all_touched()
    declared = _declared()

    owner_members = _members(Session, "session.py", "Session")
    viewer_members = _members(RemoteSession, "remote.py", "RemoteSession")

    viewer_only = {
        name
        for name in touched
        if not name.startswith("_")
        and name not in _RETIRED
        and name in viewer_members
        and name not in owner_members
    }
    # A FLOOR, not a non-empty check. Emptiness only catches TOTAL collapse of
    # the derivation; the realistic decay is partial — a renamed binding in
    # ``_SESSION_EXPRS``, a moved path in ``_SCANNED`` — which drops a slice of
    # the surface while leaving the set plausibly populated, and a bare
    # ``assert viewer_only`` would still pass (review m3).
    #
    # PER HOST, and that is the whole point rather than a refinement. A GLOBAL
    # floor cannot do this job at any value: ``app.py`` contributes 105 of the
    # 113 distinct public MEMBERS, so a number that survives ordinary churn
    # there is necessarily far above every other host's entire contribution.
    # Measured on this head — dropping the desktop utils host costs 3
    # viewer-only members of 49, dropping ``info/collect.py`` costs 0 — so both
    # single-point decays slid under a global ``>= 40`` exactly as they slid
    # under the ``>= 20`` that review round 2 (MINOR-1) rejected. Raising one
    # number would only move the blind spot. Each host is now asserted against
    # its own count, so the failure names the host that decayed.
    #
    # That dominance is itself asserted, as a ratio, by
    # ``test_app_py_dominates_the_derivation_so_a_global_floor_cannot_work``.
    thin = {
        relpath: (len(members), floor)
        for relpath, exprs, floor in _SCANNED
        for members in [
            {
                name
                for name in _session_members_touched(
                    (_ROOT / relpath).read_text(encoding="utf-8"), exprs
                )
                if not name.startswith("_")
            }
        ]
        if len(members) < floor
    }
    assert not thin, (
        f"these scanned hosts derive fewer members than they should: {thin} "
        "(actual, floor). The bindings registered for that host no longer match "
        "its source, or the path moved — either way the members it used to "
        "guard are silently unguarded now. If probes were genuinely removed, "
        "lower that host's floor in _SCANNED in the same commit and say which."
    )
    # The aggregate floor is kept BELOW the per-host ones as a backstop for a
    # decay that is spread too thinly to trip any single host.
    assert len(viewer_only) >= 40, (
        f"derivation found only {len(viewer_only)} viewer-only members "
        f"({sorted(viewer_only)}) — expected at least 40, so the probe has "
        "partially broken across several hosts at once: check that the "
        "bindings and paths in _SCANNED still match the source."
    )

    undeclared = sorted(viewer_only - declared)
    assert not undeclared, (
        f"viewer-only members the TUI uses but no protocol declares: {undeclared}. "
        "These exist on RemoteSession and not on Session, so they belong on "
        "ViewerSessionProtocol."
    )


def test_app_py_dominates_the_derivation_so_a_global_floor_cannot_work() -> None:
    """Assert the dominance the per-host floor's justification rests on.

    ``_SCANNED`` argues for a floor PER HOST rather than one global number, and
    the argument is entirely quantitative: ``app.py`` derives so much of the
    total that any global floor loose enough to survive churn there sits above
    every other host's whole contribution. If that ratio ever stops holding, the
    per-host design is over-engineering and the comment defending it is wrong.

    Asserted rather than left in prose because two of these numbers WERE wrong
    (review round 3, MINOR-6 / QA round 3, Q9): the text said "104 of the 139
    sites", conflating distinct members with ``file:line`` site strings, and
    said 49 viewer-only members where there are 48. Two independent reviewers
    measured two different pairs of numbers from the same tree, which is what a
    figure nothing executes looks like from outside. Everything this file
    asserts about the source is derived; the argument for its own shape should
    not be the one exception.

    Each number states exactly WHICH QUANTITY it counts, because that ambiguity
    is what produced the wrong figure and then hid it. Two reviewers measuring
    this tree independently reported 104/112 members with 362/413 sites and
    104/124 members with 364/415 sites, and BOTH were arithmetically right — the
    three axes they silently differed on are:

    * PUBLIC vs ALL members. Underscore-prefixed names are excluded here (112),
      included there (124). ``_session_members_touched`` collects both; every
      assertion in this file that consumes it filters, so public is the number
      that matches what is guarded.
    * DISTINCT MEMBERS vs SITE STRINGS. A member touched in twelve places is one
      member and twelve sites. The original prose said "104 of the 139 sites"
      while 104 is a MEMBER count — the two axes crossed in a single sentence.
    * RAW occurrences (364/415) vs sites DEDUPED by ``(member, line)``
      (362/413). ``_all_touched`` dedupes, so two probes of the same member on
      one line collapse; ``session.history`` on ``app.py`` lines 19279 and 19356
      is the only such pair today, and it is the whole 2-site gap.

    The RATIO is asserted rather than the absolute counts, and that is the point
    rather than a weakening. The counts churn on ordinary work — over the last
    30 commits touching ``app.py`` the site total moved eight times and the
    member total four, none of them a decay — so pinning them exactly would fire
    on unrelated PRs and train the next author to bump a number without reading
    what it claims. AGENTS.md ("Prefer a structural invariant to a numeric one")
    is the standing guidance. Dominance is the fact the per-host design rests
    on, it is what a global floor cannot accommodate, and it is stable: measured
    at 0.925-0.929 across that same history, against a bound of 0.80.
    """
    sites = {name: places for name, places in _all_touched().items() if not name.startswith("_")}
    app_sites = {
        name: [place for place in places if place.startswith("tui/app.py")]
        for name, places in sites.items()
    }
    app_sites = {name: places for name, places in app_sites.items() if places}

    # Guard the ratio in both currencies: a member-only bound would miss app.py
    # shedding sites while keeping the names, which is the shape a refactor into
    # helper modules actually takes.
    member_share = len(app_sites) / len(sites)
    site_share = sum(len(p) for p in app_sites.values()) / sum(len(p) for p in sites.values())
    assert member_share >= 0.80 and site_share >= 0.80, (
        f"app.py now derives {len(app_sites)} of {len(sites)} distinct public "
        f"members ({member_share:.3f}) and "
        f"{sum(len(p) for p in app_sites.values())} of "
        f"{sum(len(p) for p in sites.values())} deduped file:line sites "
        f"({site_share:.3f}). _SCANNED's comment justifies a PER-HOST floor by "
        "app.py dominating the derivation; below ~0.80 the other hosts are "
        "comparable enough that one global floor could do the job, so either "
        "restore the ratio or rewrite that comment — do not just lower this "
        "bound to make the failure go away."
    )

    owner = _members(Session, "session.py", "Session")
    viewer = _members(RemoteSession, "remote.py", "RemoteSession")
    viewer_only = {
        name for name in sites if name not in _RETIRED and name in viewer and name not in owner
    }
    # An exact pin, unlike the ratio above: this is the population the aggregate
    # floor of 40 is set against, so a drop here is the decay that floor exists
    # to catch rather than ordinary churn.
    assert len(viewer_only) == 50, (
        f"there are {len(viewer_only)} viewer-only members; _SCANNED's comment "
        "says 50, and the aggregate floor is set at 40 against that number. A "
        "drop here is the decay that floor exists to catch, so check it is "
        "genuinely a removal before editing this figure."
    )


def test_owner_only_probes_are_all_optional_capability_probes() -> None:
    """``_OWNER_ONLY_CAPABILITY_PROBES`` is sound only if every name has a default.

    The set is excused from declaration on the grounds that each entry is an
    OPTIONAL capability: probed with a fallback, so a session lacking it is a
    supported state rather than a bug. That justification collapses the moment
    one is read as a hard ``session.member`` — then absence is an
    ``AttributeError`` in a user's terminal and the name belonged on a protocol
    all along.

    Asserted rather than commented because the previous form was five hard-coded
    ``app.py`` line numbers (review n2), which rot into a false claim without
    failing anything.
    """
    hard_accesses: dict[str, list[str]] = {}
    for relpath, exprs, _floor in _SCANNED:
        filename = _site_label(relpath)
        tree = ast.parse((_ROOT / relpath).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _OWNER_ONLY_CAPABILITY_PROBES
                and _unparse(node.value) in exprs
            ):
                hard_accesses.setdefault(node.attr, []).append(f"{filename}:{node.lineno}")
            # A 2-arg getattr raises exactly like an attribute access does.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 2
                and _unparse(node.args[0]) in exprs
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in _OWNER_ONLY_CAPABILITY_PROBES
            ):
                hard_accesses.setdefault(str(node.args[1].value), []).append(
                    f"{filename}:{node.lineno}"
                )

    assert not hard_accesses, (
        "these are excluded as OPTIONAL capability probes, but are read without "
        f"a default: {hard_accesses}. Absence is an AttributeError at that site, "
        "not a supported state, so the name must be declared on a protocol "
        "rather than excluded here."
    )


def test_the_exclusion_sets_state_true_facts() -> None:
    """Each exclusion set asserts something about the classes; check it holds.

    An exclusion set is a claim, and a false claim inside the guard is worse
    than no guard: it silences a member while telling the reader the silence is
    justified. ``cwd`` was listed as living on BOTH classes when it lives on
    neither, which converted an always-empty-value defect into permanently
    silenced debt (review M2). Each set now has to be true.
    """
    owner_members = _members(Session, "session.py", "Session")
    viewer_members = _members(RemoteSession, "remote.py", "RemoteSession")

    missing = sorted(
        n for n in _UNDECLARED_ON_BOTH_CLASSES if n not in owner_members or n not in viewer_members
    )
    assert not missing, (
        f"_UNDECLARED_ON_BOTH_CLASSES claims these live on both classes: {missing}. "
        "They do not. A name absent from both is a DEAD probe reading its own "
        "default forever — move it to _KNOWN_MISSING_ON_BOTH_CLASSES, recording "
        "the value it actually returns, or declare it."
    )

    resurrected = sorted(
        n for n in _KNOWN_MISSING_ON_BOTH_CLASSES if n in owner_members or n in viewer_members
    )
    assert not resurrected, (
        "_KNOWN_MISSING_ON_BOTH_CLASSES claims these exist on neither class: "
        f"{resurrected}. They now exist, so the probe is live: declare them on a "
        "protocol and drop them from this set."
    )

    not_owner_only = sorted(n for n in _OWNER_ONLY_CAPABILITY_PROBES if n in viewer_members)
    assert not not_owner_only, (
        "_OWNER_ONLY_CAPABILITY_PROBES claims these are absent from the viewer "
        f"facade: {not_owner_only}. They exist on RemoteSession, so they are "
        "viewer surface and belong on ViewerSessionProtocol."
    )

    # The OTHER half of that set's claim, and the one that was missing. The
    # exclusion reads "an owner capability the viewer legitimately lacks", so
    # absence-from-the-viewer alone does not justify it: a name on NEITHER class
    # is a dead probe, and without this assertion an agent facing a red guard
    # could silence one by appending a single line here and stay green on every
    # test. Proved in review round 2 (MAJOR-2) with a fabricated name. Every
    # sibling set above is two-sided; this one was not.
    not_on_owner = sorted(n for n in _OWNER_ONLY_CAPABILITY_PROBES if n not in owner_members)
    assert not not_on_owner, (
        "_OWNER_ONLY_CAPABILITY_PROBES justifies each name as an OWNER "
        f"capability, but these are absent from Session too: {not_on_owner}. A "
        "name on neither class is a DEAD probe reading its own default forever "
        "— move it to _KNOWN_MISSING_ON_BOTH_CLASSES, recording the value it "
        "actually returns, rather than laundering it as an owner capability."
    )


def _static_conformance_is_checked_by_pyright() -> None:
    """The STATIC half of the conformance claim, checked by pyright, not pytest.

    Assigning each class to a variable of the protocol type is what makes
    pyright verify signatures — return types, parameter names, defaults,
    async-ness — which ``isinstance`` cannot see: a ``runtime_checkable``
    Protocol checks member PRESENCE only, so a facade whose
    ``load_older_display_page`` stopped being a coroutine would still pass the
    runtime assertion above.

    Never called. Its body is type-checked where it is written; running it
    would construct sessions and dial sockets, which is the opposite of what a
    conformance assertion should cost.

    **This function is load-bearing, and pytest cannot tell you when it stops
    being.** It is the ONLY place either class is assigned to a protocol type,
    so it is the only thing that makes pyright verify signatures — return
    types, parameter names, async-ness — rather than mere presence. Proven, not
    assumed: breaking a viewer member that has no internal caller gives
    ``pyright local_operator/session/`` zero errors, and only checking THIS
    FILE reports it (QA round 1).

    Two consequences for anyone editing configuration rather than code:

    * narrowing pyright's path to the package, or excluding ``tests/``, silently
      disarms the static half of the conformance claim while every test still
      passes — the repo's pyright invocation is whole-tree for this reason;
    * deleting this function because "nothing calls it" removes the check
      entirely, with no failing test to object — except that
      ``test_the_static_conformance_anchor_still_exists`` below now does object,
      which is what turned that disclosure into detection (QA round 2, Q6).
    """
    viewer: ViewerSessionProtocol = RemoteSession.__new__(RemoteSession)
    owner: SessionProtocol = Session.__new__(Session)
    attached: SessionProtocol = RemoteSession.__new__(RemoteSession)
    _ = (viewer, owner, attached)


def test_the_static_conformance_anchor_still_exists() -> None:
    """Deleting the function above must not be silent in BOTH checkers.

    It is never called, so pytest is indifferent to its existence and pyright
    only reports what it can still see: delete it and the signature half of the
    conformance claim vanishes with 7 tests passing and 0 pyright errors — QA
    round 2 (Q6) verified exactly that. The disclosure in its docstring was
    honest but disclosure is not detection.

    Asserted on the ANNOTATIONS, not merely on the name: a body reduced to
    ``pass`` keeps the symbol while removing every assignment that makes pyright
    check anything, which is the same loss by a quieter route. Read out of the
    module's own source because the values are type annotations on locals, which
    do not survive into the compiled function object.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    anchor = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_static_conformance_is_checked_by_pyright"
        ),
        None,
    )
    assert anchor is not None, (
        "_static_conformance_is_checked_by_pyright has been deleted. It is the "
        "ONLY place either class is assigned to a protocol type, so removing it "
        "silently drops signature checking (return types, parameter names, "
        "async-ness) from the conformance claim: isinstance sees member "
        "presence only. Restore it rather than deleting this test."
    )

    annotated = {
        _unparse(node.annotation)
        for node in ast.walk(anchor)
        if isinstance(node, ast.AnnAssign) and node.annotation is not None
    }
    assert {"ViewerSessionProtocol", "SessionProtocol"} <= annotated, (
        "_static_conformance_is_checked_by_pyright no longer assigns both "
        f"classes to protocol-typed variables (found annotations: {annotated}). "
        "Those annotations ARE the static check; a body without them keeps the "
        "symbol and loses the coverage."
    )


def test_every_registered_session_binding_still_matches_the_source() -> None:
    """Each (host, binding) pair in ``_SCANNED`` must derive at least one member.

    The floor in ``test_the_viewer_protocol_covers_what_only_the_facade_has``
    catches decay in AGGREGATE, and QA round 2 (Q5) showed that is not enough:
    renaming ``self._session`` in ``_SESSION_EXPRS`` drops five members and the
    total stays above any floor loose enough to be maintainable. A per-pair
    assertion catches the same decay at its source and names the host and the
    binding, which a total never can.

    Zero sites means one of two things and both need the reader's attention: the
    binding was renamed in the source (fix the set), or it never matched and the
    coverage it implies was always fictional. ``self.session`` and
    ``self.app.session`` were the second case — registered in ``_SESSION_EXPRS``
    and matching nothing in ``app.py`` — so they are asserted against the files
    that DO use them rather than being quietly dropped, since dropping a
    binding is how a real spelling stops being watched.

    The count is pinned as well as the contribution, because the two decays are
    different and only one of them is a rename. DELETING ``source.session``
    outright costs 2 of 49 viewer-only members — under any floor, per-host or
    aggregate, and invisible to the zero-sites check because a removed entry is
    not an entry that derives nothing. It was the last planted violation this
    guard did not catch. A registered binding is a coverage claim, so removing
    one has to be a deliberate edit to a stated number rather than a quiet
    deletion.

    Counted per (HOST, BINDING) rather than over the union across hosts, which
    round 3 found was hiding two distinct failures at once (review MAJOR-3, QA
    Q7). A union credits a binding as live as long as ANY scanned host uses it,
    so a spelling registered against a host that never had it read as covered:
    four such pairs existed on the round-2 head, and ``desktop_lifecycle.py``
    was passing the check solely on ``child.remote``, a single site in the whole
    package. Per-pair counting also makes the DELETION of a whole ``_SCANNED``
    entry visible where the union could not see it — dropping the
    ``info/collect.py`` line removed its pair rather than emptying it, which is
    how the reviewer re-shipped the motivating outage (rename
    ``subagent_comms``, drop the exclusion entry it no longer backs) against a
    fully green guard.

    The scanned PATHS are pinned for the residue that per-pair counting still
    cannot reach: a deleted entry contributes no pair to check, so the pin is
    what converts "host silently removed" into a failure. The two assertions are
    complementary rather than redundant — the pin catches removing a host, the
    per-pair check catches a host that is still listed but no longer derives
    what it claims to.
    """
    per_pair: dict[tuple[str, str], int] = {}
    for relpath, exprs, _floor in _SCANNED:
        source = (_ROOT / relpath).read_text(encoding="utf-8")
        # Counted per binding rather than per member: a member reached through
        # two bindings must credit both, or dropping either looks harmless.
        tree = ast.parse(source)
        for expr in exprs:
            per_pair.setdefault((relpath, expr), 0)
        for node in ast.walk(tree):
            expr = None
            if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
                expr = _unparse(node.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _PROBE_CALLS
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                expr = _unparse(node.args[0])
            if expr is not None and expr in exprs:
                per_pair[(relpath, expr)] = per_pair[(relpath, expr)] + 1

    dead = sorted(f"{relpath}:{expr}" for (relpath, expr), n in per_pair.items() if n == 0)
    assert not dead, (
        f"these registered (host, binding) pairs derive ZERO members: {dead}. "
        "Either the binding was renamed in that host — in which case every "
        "member it reached there has silently stopped being guarded — or it "
        "never matched in that file and the coverage it implies is fictional. "
        "Fix the spelling or drop the binding from that host's set with a note "
        "saying which."
    )

    # The scanned PATHS, pinned by name. A ``_SCANNED`` entry is the claim that
    # this host is watched at all, and deleting one is invisible to every other
    # assertion here: the per-pair check loses the pairs it would have failed
    # on, and both floors only ever see the hosts still listed.
    assert {relpath for relpath, _exprs, _floor in _SCANNED} == {
        "tui/app.py",
        "server/utils/desktop_sessions.py",
        "info/collect.py",
        "server/routes/desktop_lifecycle.py",
        "server/routes/desktop_sessions.py",
        "server/routes/desktop_catalogues.py",
        "tui/session_interaction.py",
    }, (
        "the set of scanned hosts changed: "
        f"{sorted(relpath for relpath, _e, _f in _SCANNED)}. Removing one "
        "unguards every member it derived, and three of these hosts cost ZERO "
        "viewer-only members to delete — no floor, per-host or aggregate, can "
        "notice their absence. Deleting the 'info/collect.py' line is how the "
        "motivating /info outage was re-shipped against a green guard. Adding a "
        "host is good news and needs this list updated too; either way, say "
        "which in the commit."
    )

    # The bindings themselves, pinned by name. Deliberately the whole set rather
    # than a count: a count would let a deletion be paid for with an unrelated
    # addition, and the message that matters names the spelling that stopped
    # being watched.
    registered = {expr for _relpath, exprs, _floor in _SCANNED for expr in exprs}
    assert registered == {
        "self._session",
        "session",
        "source.session",
        "remote",
        "self.remote",
        "bridge.remote",
        "child.remote",
        "self.session",
    }, (
        f"the set of registered session bindings changed: {sorted(registered)}. "
        "Each one is a claim that this spelling of a session is watched, so "
        "REMOVING one silently unguards every member it reached (deleting "
        "'source.session' costs 2 viewer-only members — too few for any floor to "
        "notice). Adding one is good news and needs this list updated too; "
        "either way, say which in the commit."
    )
