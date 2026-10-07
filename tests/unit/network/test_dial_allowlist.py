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
}

#: The floor for the scan's own health. If a refactor renames the local dict the
#: scan reads (or splits the declarations across a helper), these names stop being
#: found — and a scan that finds nothing would otherwise pass vacuously.
_SCAN_FLOOR = frozenset({"events", "frontend_state", "display_window", "surface"})


def _declared_auth_fields() -> set[str]:
    """Every string key either client assigns into its ``auth`` dict.

    Read from the SOURCE (an AST walk), not from a list kept here: the invariant
    is about what the clients do, and a hand-kept list beside a hand-kept
    allowlist is one more copy of the same fact to forget to update.
    """
    declared: set[str] = set()
    for relative in _DECLARING_MODULES:
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                base = target.value
                if not (isinstance(base, ast.Name) and base.id == "auth"):
                    continue
                key = target.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    declared.add(key.value)
    return declared


def test_the_scan_finds_the_declarations_it_claims_to_guard() -> None:
    """Positive control: a scan that finds nothing guards nothing."""
    declared = _declared_auth_fields()
    missing = sorted(_SCAN_FLOOR - declared)
    assert not missing, (
        f"the auth scan no longer finds {missing} in {list(_DECLARING_MODULES)} — the "
        "declarations moved and this guard went blind"
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
