"""The relay's attach allowlist is CLOSED — and this is the guard that it stays COMPLETE.

``network/dial.py``'s ``AUTH_FIELDS`` is a CLOSED allowlist on the owner-dial path:
``dial_owner`` copies those names and nothing else out of a viewer's ``auth`` dict.
A name that is missing is not refused — it is silently dropped, which is how the
entry-times declaration shipped inert across the mesh (QA round 1, Q1: the owner
built the ``{entry id: ts}`` join, read no declaration, and stripped it, so every
wire row was served ``unstated``/``served`` and never ``entry``).

The allowlist has to stay CLOSED as well as complete, and that half is a security
property rather than tidiness: ``locality`` and the ``kind`` the dial authenticates
as are the DIALER's own fields, and server-side ``locality`` is what separates the
narrow "watch a session here" lane from the widest one that types into the owner's
terminal — an open allowlist would let a viewer's ``auth`` dict smuggle them
(``session/runtime/server.py`` near ``locality``, and ``network/authorizer.py``
refuses the frame that tries). So the fix for a dropped declaration is always to
name it here, never to widen the copy.

WHAT THIS FILE EXISTS FOR (agent review round 2, R2-2): before it, nothing tied the
fields the attach clients declare to the tuple the relay forwards, so the next
declaration could ship inert exactly as this one had. The scan reads the CLIENTS'
own source rather than a hand-kept list, so adding a declaration there reds this
test until it is either forwarded or explicitly excused below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from local_operator.network import dial

REPO = Path(__file__).resolve().parents[3]

#: The modules that build a viewer's per-connection ``auth`` dict. Both are
#: clients of the same ``dial_owner`` path: the mesh viewer and the mobile
#: attach client.
_DECLARING_MODULES = (
    "local_operator/network/projection.py",
    "local_operator/mobile/attach_client.py",
)

#: Declared by a client and deliberately NOT forwarded, each with the reason it is
#: safe to drop. An entry here is a claim about the mesh, so it has to be
#: falsifiable: the test below refuses an excuse for a name no client declares.
_NOT_FORWARDED: dict[str, str] = {
    "operator_nonce": (
        "Operator authority is LOCAL-ONLY, and the runtime says so at the point it "
        "matters: a connection with no nonce is one of 'a follower, or a RELAY', and "
        "the conservative refusal is the right answer for all three "
        "(session/runtime/server.py, the operator-handshake branch). The mesh viewer "
        "never declares it either — projection.py sets no such key — so forwarding it "
        "would hand a peer the operator lane this device's own desktop has."
    ),
    # The three below are the mobile attach client's own handshake literal. They
    # became VISIBLE when the scan learned the dict-literal shape (agent review
    # round 3, R3-1); they were always declared, and always intentionally absent
    # from the allowlist, because ``dial_owner`` writes its OWN values for them.
    "key": (
        "The owner's control key is read from the runtime record on THIS device by "
        "dial_owner itself (network/dial.py, the auth_frame literal). A viewer never "
        "holds it and must never choose it, so copying it out of the viewer's dict "
        "would let a peer name the credential the dial authenticates with."
    ),
    "client": (
        "dial_owner authenticates every relayed dial as client 'attach' (network/"
        "dial.py, the auth_frame literal). The client kind selects the owner's lane, "
        "so it is the dialler's own fact and is never taken from a viewer's dict."
    ),
    "locality": (
        "Server-side locality is what separates the narrow watch-a-session lane from "
        "the widest one that types into the owner's terminal, and dial_owner pins it "
        "to 'remote' because it dialled on a peer's behalf (network/dial.py, §2.2: "
        "locality is never taken from the frame that arrived). Forwarding it would "
        "let a viewer claim to be local."
    ),
}

#: The floor for the scan's own health, PER CLIENT (agent review round 3, R3-2).
#: If a refactor renames the local dict the scan reads (or splits the declarations
#: across a helper), these names stop being found — and a scan that finds nothing
#: would otherwise pass vacuously. It is per module because a union floor only
#: catches BOTH clients going blind: one client could rename its dict and lose every
#: declaration while the other still satisfied the floor. ``attach_client`` also
#: carries the three dict-LITERAL keys of its handshake, so its floor is what proves
#: the literal shape is read on the real file, not only on the synthetic controls.
_SCAN_FLOOR: dict[str, frozenset[str]] = {
    "local_operator/network/projection.py": frozenset(
        {"events", "frontend_state", "display_window", "surface"}
    ),
    "local_operator/mobile/attach_client.py": frozenset(
        {
            "key",
            "client",
            "locality",
            "events",
            "frontend_state",
            "display_window",
            "surface",
            "operator_nonce",
        }
    ),
}

#: The one name the scan reads. Both clients call their dict ``auth``; a client that
#: builds it under another name is exactly the blindness the floor exists to catch.
_AUTH_NAME = "auth"


def _is_auth(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == _AUTH_NAME


def _str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dict_keys(node: ast.AST, line: int, declared: set[str], opaque: list[str]) -> None:
    """The constant keys of a dict-shaped expression; anything unreadable is OPAQUE."""
    if isinstance(node, ast.Dict):
        for key in node.keys:
            name = _str_const(key)
            if name is None:
                # ``**other`` (key is None) or a computed key: a declaration the scan
                # cannot name. Reported rather than skipped, because skipping is how
                # a declaration ships inert.
                opaque.append(f"line {line}: a dict entry whose key is not a string literal")
            else:
                declared.add(name)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
    ):
        for keyword in node.keywords:
            if keyword.arg is None:
                opaque.append(f"line {line}: dict(**...) hides its keys")
            else:
                declared.add(keyword.arg)
    else:
        opaque.append(f"line {line}: a value assigned into auth that is not a dict display")


def _scan_source(source: str) -> tuple[set[str], list[str]]:
    """``(declared names, opaque sites)`` for one client's source text.

    THE SHAPES READ, each one a way a declaration can be written and each pinned by
    a positive control below (agent review round 3, R3-1 — the scan used to see only
    ``auth["x"] = ...`` and was blind to the rest):

    * ``auth["x"] = ...`` and ``auth["x"] |= ...``
    * ``auth = {"x": ...}`` and ``auth: T = {"x": ...}`` (plus ``dict(x=...)``)
    * ``auth.update({"x": ...})``, ``auth.update(x=...)``, ``auth |= {"x": ...}``
    * ``auth.setdefault("x", ...)``

    Anything with the right target but a shape the scan cannot read (a computed key,
    ``**splat``, ``auth.update(other)``) is returned as OPAQUE, and the invariant
    refuses it: a scan that quietly skips what it cannot read goes blind in exactly
    the way it exists to prevent.
    """
    declared: set[str] = set()
    opaque: list[str] = []
    for node in ast.walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and _is_auth(target.value):
                    name = _str_const(target.slice)
                    if name is None:
                        opaque.append(f"line {line}: auth[...] with a computed key")
                    else:
                        declared.add(name)
                elif _is_auth(target) and node.value is not None:
                    _dict_keys(node.value, line, declared, opaque)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if not _is_auth(node.func.value):
                continue
            if node.func.attr == "update":
                for arg in node.args:
                    _dict_keys(arg, line, declared, opaque)
                for keyword in node.keywords:
                    if keyword.arg is None:
                        opaque.append(f"line {line}: auth.update(**...) hides its keys")
                    else:
                        declared.add(keyword.arg)
            elif node.func.attr == "setdefault":
                name = _str_const(node.args[0]) if node.args else None
                if name is None:
                    opaque.append(f"line {line}: auth.setdefault with a computed key")
                else:
                    declared.add(name)
    return declared, opaque


def _declared_by_module() -> dict[str, tuple[set[str], list[str]]]:
    """Each client's scan, kept SEPARATE so the floor can be checked per client."""
    return {
        relative: _scan_source((REPO / relative).read_text(encoding="utf-8"))
        for relative in _DECLARING_MODULES
    }


def _declared_auth_fields() -> set[str]:
    """Every string key either client declares into its ``auth`` dict.

    Read from the SOURCE (an AST walk), not from a list kept here: the invariant
    is about what the clients do, and a hand-kept list beside a hand-kept
    allowlist is one more copy of the same fact to forget to update.
    """
    declared: set[str] = set()
    for names, _opaque in _declared_by_module().values():
        declared |= names
    return declared


@pytest.mark.parametrize("relative", _DECLARING_MODULES)
def test_the_scan_finds_the_declarations_it_claims_to_guard(relative: str) -> None:
    """Positive control, PER CLIENT: a scan that finds nothing in one file guards nothing there."""
    declared, _opaque = _declared_by_module()[relative]
    missing = sorted(_SCAN_FLOOR[relative] - declared)
    assert not missing, (
        f"the auth scan no longer finds {missing} in {relative} — the declarations "
        "moved and this guard went blind for that client"
    )


def test_the_floor_names_every_declaring_module() -> None:
    """A client added to the scan without a floor would be unguarded by it."""
    assert set(_SCAN_FLOOR) == set(_DECLARING_MODULES)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('auth["a"] = 1', {"a"}),
        ('auth["a"] |= 1', {"a"}),
        ('auth = {"a": 1, "b": 2}', {"a", "b"}),
        ('auth: dict[str, int] = {"a": 1}', {"a"}),
        ("auth = dict(a=1, b=2)", {"a", "b"}),
        ('auth.update({"a": 1})', {"a"}),
        ("auth.update(a=1)", {"a"}),
        ('auth |= {"a": 1}', {"a"}),
        ('auth.setdefault("a", 1)', {"a"}),
        ('x = 1\nif x:\n    auth["a"] = 1\n    auth.update(b=2)', {"a", "b"}),
        ('other["a"] = 1\nother.update({"b": 2})', set()),
    ],
    ids=[
        "subscript",
        "subscript-augassign",
        "dict-literal",
        "annotated-dict-literal",
        "dict-call",
        "update-literal",
        "update-keywords",
        "ior-literal",
        "setdefault",
        "nested-in-a-branch",
        "a-different-name-is-not-auth",
    ],
)
def test_the_scan_reads_every_shape_a_declaration_can_take(source: str, expected: set[str]) -> None:
    """R3-1 positive controls: the shapes the subscript-only scan was blind to."""
    declared, opaque = _scan_source(source)
    assert declared == expected
    assert opaque == []


@pytest.mark.parametrize(
    "source",
    [
        "auth[name] = 1",
        'auth = {**base, "a": 1}',
        "auth = {name: 1}",
        "auth.update(other)",
        "auth.update(**other)",
        "auth.setdefault(name, 1)",
        "auth = build()",
    ],
    ids=[
        "computed-subscript",
        "dict-splat",
        "computed-dict-key",
        "update-a-variable",
        "update-splat",
        "computed-setdefault",
        "assigned-a-call",
    ],
)
def test_a_declaration_the_scan_cannot_read_is_refused_not_skipped(source: str) -> None:
    """A shape the scan cannot name is reported OPAQUE, which the invariant refuses."""
    _declared, opaque = _scan_source(source)
    assert opaque, f"{source!r} hid a declaration from the scan without saying so"


def test_no_client_declares_through_a_shape_the_scan_cannot_read() -> None:
    """The invariant's other half: the real clients stay inside the readable shapes."""
    unreadable = {
        relative: opaque for relative, (_d, opaque) in _declared_by_module().items() if opaque
    }
    assert not unreadable, (
        f"{unreadable}: a declaration written this way is invisible to the allowlist "
        "guard, so it could ship inert. Write it as auth['name'] = ... (or a literal "
        "dict display) so the scan can name it."
    )


def test_every_declared_attach_field_is_forwarded_or_explicitly_excused() -> None:
    """The invariant R2-2 asks for: no declaration is dropped in silence."""
    declared = _declared_auth_fields()
    forwarded = set(dial.AUTH_FIELDS)
    silent = sorted(declared - forwarded - set(_NOT_FORWARDED))
    assert not silent, (
        f"{silent} are declared by an attach client and forwarded to the owner by "
        "neither AUTH_FIELDS nor _NOT_FORWARDED. A name missing from the allowlist is "
        "NOT refused — dial_owner drops it silently, and the owner then reads no "
        "declaration at all (this is exactly how the entry-times join shipped inert "
        "across the mesh). Add it to AUTH_FIELDS, or excuse it here with a reason."
    )


def test_an_excuse_for_a_field_nobody_declares_is_refused() -> None:
    """Every excuse is a claim about live code, and a stale one hides the next bug.

    ``_NOT_FORWARDED`` says "this declaration is safe to drop"; if no client
    declares it any more, the sentence is describing code that no longer exists —
    and, worse, it is a pre-written excuse that a future declaration of the same
    name would silently inherit.
    """
    declared = _declared_auth_fields()
    stale = sorted(set(_NOT_FORWARDED) - declared)
    assert not stale, f"{stale} are excused here but declared by no client"


def test_an_excuse_never_shadows_a_forwarded_field() -> None:
    """An exception that is also on the allowlist is not an exception, it is noise."""
    both = sorted(set(_NOT_FORWARDED) & set(dial.AUTH_FIELDS))
    assert not both, f"{both} are excused AND forwarded — the excuse is misleading"


@pytest.mark.parametrize("name", sorted(_NOT_FORWARDED))
def test_the_excuses_carry_their_reason(name: str) -> None:
    reason = _NOT_FORWARDED[name]
    assert len(reason) > 80, f"the excuse for {name} does not say why it is safe to drop"
