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
set from the SOURCE — every session-valued attribute access and duck-probe in
``tui/app.py`` — and fails when one of them is not declared on
``SessionProtocol`` or ``ViewerSessionProtocol``. Adding a new undeclared
duck-typed member to the TUI therefore fails here rather than in a user's
terminal.

The derivation is deliberately syntactic rather than type-inferred: pyright
cannot follow ``getattr`` with a literal string, which is precisely why these
members escaped typing in the first place.
"""

from __future__ import annotations

import ast
from pathlib import Path

import local_operator
from local_operator.session.protocol import SessionProtocol, ViewerSessionProtocol
from local_operator.session.remote import RemoteSession
from local_operator.session.session import Session

_ROOT = Path(local_operator.__file__).resolve().parent
_APP = _ROOT / "tui" / "app.py"

#: Expressions in ``app.py`` that are unambiguously a session.
#:
#: Deliberately NOT ``target``/``current``/``sess``: those names are reused in
#: the file for widgets, rows and strings, and including them made an earlier
#: version of this probe report ``partition``, ``focus`` and ``label`` as
#: session members. A guard that cries wolf gets deleted, so it reads only the
#: bindings that always hold a session.
_SESSION_EXPRS = frozenset(
    {
        "self.session",
        "self._session",
        "session",
        "self.app.session",
    }
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
#: Verified as capability probes, not hard requirements, at:
#: ``app.py:10345`` (attach_team), ``9761`` (has_pending_fork),
#: ``6717`` (preflight_usage), ``24969`` (routing_settings), ``13626``
#: (variables).
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
_UNDECLARED_ON_BOTH_CLASSES = frozenset(
    {
        "acknowledge_attention",
        "active_agent",
        "active_team_name",
        "agent_registry",
        "context_breakdown",
        "cwd",
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

#: Removed by Stage 3; declaring it would entrench the flag this work retires.
_RETIRED = frozenset({"is_remote"})


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — an unparseable node is simply not a session
        return ""


def _session_members_touched(source: str) -> dict[str, list[int]]:
    """Every session member ``app.py`` reads, by name, with line numbers."""
    tree = ast.parse(source)
    touched: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        # session.member / self._session.member
        if isinstance(node, ast.Attribute) and _unparse(node.value) in _SESSION_EXPRS:
            touched.setdefault(node.attr, []).append(node.lineno)
        # getattr(session, "member", ...) / hasattr(session, "member")
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) >= 2
                and _unparse(node.args[0]) in _SESSION_EXPRS
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                touched.setdefault(node.args[1].value, []).append(node.lineno)
    return touched


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


def test_every_session_member_the_tui_touches_is_declared() -> None:
    """The guard itself.

    Fails with the offending names and their ``app.py`` lines, so the fix is
    "declare it on the protocol" rather than "go find what changed".
    """
    touched = _session_members_touched(_APP.read_text())
    declared = _declared()
    known = declared | _OWNER_ONLY_CAPABILITY_PROBES | _UNDECLARED_ON_BOTH_CLASSES | _RETIRED

    offenders = {
        name: sorted(set(lines))
        for name, lines in touched.items()
        if not name.startswith("_") and name not in known
    }

    assert not offenders, (
        "the TUI reads session members that no protocol declares: "
        + ", ".join(f"{name} (app.py:{lines})" for name, lines in sorted(offenders.items()))
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
    touched = _session_members_touched(_APP.read_text())
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
    assert viewer_only, "derivation found no viewer-only members — the probe has broken"

    undeclared = sorted(viewer_only - declared)
    assert not undeclared, (
        f"viewer-only members the TUI uses but no protocol declares: {undeclared}. "
        "These exist on RemoteSession and not on Session, so they belong on "
        "ViewerSessionProtocol."
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
    """
    viewer: ViewerSessionProtocol = RemoteSession.__new__(RemoteSession)
    owner: SessionProtocol = Session.__new__(Session)
    attached: SessionProtocol = RemoteSession.__new__(RemoteSession)
    _ = (viewer, owner, attached)
