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
It sees ``<expr>.member`` and ``getattr``/``hasattr(<expr>, "literal", ...)``
where ``<expr>`` is one of the bindings registered for that file in
``_SCANNED``. It is blind to:

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
#: These are BINDINGS, matched literally. ``s = self._session; s.member`` is
#: not covered: a local alias is a different expression, and the guard does no
#: dataflow. Following aliases means resolving assignments per scope, which is
#: where the false positives that nearly killed this probe come from — so the
#: boundary is deliberate, and this list is the thing to extend when a new
#: session binding appears.
_SESSION_EXPRS = frozenset(
    {
        "self.session",
        "self._session",
        "session",
        "self.app.session",
        "source.session",
    }
)

#: The bindings that hold a viewer facade in the desktop host.
#:
#: Separate from ``_SESSION_EXPRS`` because the name differs by host: the
#: bridge stores its facade as ``self.remote`` and copies it into a local
#: ``remote`` before use, which is a session binding by the same reasoning
#: ``self._session`` is one in the TUI.
_HOST_SESSION_EXPRS = frozenset({"remote", "self.remote"})

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
#: Adding a host is one entry here plus whatever it turns out to be probing.
#: Other ``tui/`` modules also probe sessions (``session_interaction.py``,
#: ``widgets/session_panel.py``, ``widgets/todo_panel.py``,
#: ``widgets/subagent_panel.py``, ``widgets/wake_panel.py``), but every member
#: they read is already declared or excluded below, so listing them today buys
#: scan cost and no coverage; add one when it starts probing something new.
_SCANNED = (
    ("tui/app.py", _SESSION_EXPRS),
    ("server/utils/desktop_sessions.py", _HOST_SESSION_EXPRS),
)

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
#: * ``cwd`` — ``app.py:4049`` passes ``getattr(self._session, "cwd", "")`` to
#:   ``saved_preview(...)``, so the saved preview's working directory is
#:   ALWAYS ``""``. Pre-existing (present at ``a8f98be3b``), behavioural, and
#:   out of scope for this additive change: deferred to Stage 3, where that
#:   call site is rewritten against a declared member.
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
        # getattr(session, "member", ...) / hasattr(session, "member")
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) >= 2
                and _unparse(node.args[0]) in exprs
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                touched.setdefault(node.args[1].value, []).append(node.lineno)
    return touched


def _all_touched() -> dict[str, list[str]]:
    """Every scanned file's session members, mapped name -> ``file:line`` sites."""
    sites: dict[str, list[str]] = {}
    for relpath, exprs in _SCANNED:
        filename = relpath.rsplit("/", 1)[-1]
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
    PRESENCE only, not signatures — so it catches a rename or a deletion on the
    facade, which is this guard's job, and pyright catches the signatures.
    Both are needed: pyright alone would not see a member deleted at runtime by
    a refactor of ``__init__``.
    """
    session = RemoteSession.__new__(RemoteSession)
    # The instance attributes are assigned in ``__init__``, which dials a
    # socket. Set them directly: presence is what the protocol requires, and
    # constructing a real facade would make this a network test.
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
    assert len(viewer_only) >= 20, (
        f"derivation found only {len(viewer_only)} viewer-only members "
        f"({sorted(viewer_only)}) — expected at least 20, so the probe has "
        "partially broken: check that _SESSION_EXPRS' bindings and _SCANNED's "
        "paths still match the source."
    )

    undeclared = sorted(viewer_only - declared)
    assert not undeclared, (
        f"viewer-only members the TUI uses but no protocol declares: {undeclared}. "
        "These exist on RemoteSession and not on Session, so they belong on "
        "ViewerSessionProtocol."
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
    for relpath, exprs in _SCANNED:
        filename = relpath.rsplit("/", 1)[-1]
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
      entirely, with no failing test to object.
    """
    viewer: ViewerSessionProtocol = RemoteSession.__new__(RemoteSession)
    owner: SessionProtocol = Session.__new__(Session)
    attached: SessionProtocol = RemoteSession.__new__(RemoteSession)
    _ = (viewer, owner, attached)
