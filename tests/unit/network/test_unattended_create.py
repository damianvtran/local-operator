"""The granted-``unattended`` create (remote-onboarding §2 OQ4, defect 2).

WHAT THIS FILE EXISTS TO PIN. A create that asks for full-auto (``yolo``) used to
be refused structurally on both ends, with no capability that could unlock it. The
boundary is now a GRANT, checked RECEIVER-side against the sender's member row:

* without the grant the create is still refused, with a sentence that names the
  grant rather than pretending the wish is absurd — and it is refused BEFORE
  anything is written, which the first cell asserts against the disk;
* with the grant the create proceeds and the accepted authority travels ON THE
  SESSION (``mesh.json``'s ``unattended``), because the runtime that will run
  unattended is constructed by a LATER process that reads the stamp, not this
  frame. The stamp is a carry, never the authority: the construction re-checks
  the grant at every engage (``serving._carried_auto_authority``, pinned in
  ``tests/unit/session/runtime/test_carried_auto_authority.py``), so a revocation
  takes effect at the next engage and a hand-edited stamp alone cannot loosen
  anything.

Two stubs make the split honest: the LINK is a namespace carrying exactly what the
gate reads (``context.capabilities``, ``device_id``, ``network_id``), and the
warm is monkeypatched because a real spawn belongs to the e2e cell, not here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import identity, relay, types
from local_operator.session.placement import read_stamp


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _link(*, capabilities: tuple[str, ...]) -> Any:
    """A stub link carrying exactly what ``_op_session_create`` reads.

    ``context.capabilities`` is the live grant set in production
    (``PeerLink.context`` -> ``role_capabilities()`` -> the member row), so a
    namespace is the honest shape here: the store read is the server's own code
    and is covered by the membership cells elsewhere.
    """
    return SimpleNamespace(
        device_id="d_" + "a" * 32,
        network_id="n_0123456789abcdef0123456789abcdef",
        epoch=1,
        context=SimpleNamespace(
            capabilities=frozenset(capabilities),
            device_id="d_" + "a" * 32,
            network_id="n_0123456789abcdef0123456789abcdef",
            epoch=1,
        ),
    )


def _frame(*, yolo: Any) -> dict[str, Any]:
    return {"op": "net_session_create", "req": 41, "cwd": "", "yolo": yolo}


def test_a_yolo_create_without_the_grant_is_refused_and_names_it(root: Path) -> None:
    """The structural refusal became a granted one; WITHOUT the grant it stands.

    The refusal must arrive before anything exists: the old structural guard sat
    above the mint for a reason, and the granted guard keeps the position — a
    refused create leaves no directory, no stamp and no claim behind (the rule the
    desktop route states for its own admissions).
    """
    # A NAMED node makes the remedy's naming deterministic and mirrors the drill
    # (`cloud-node-1`); unnamed, the identity would carry this host's name.
    identity.mint(root, name="cloud-node-1")
    server = _server(root)
    link = _link(capabilities=("prompt",))
    with pytest.raises(types.MeshRefusal) as refused:
        server._op_session_create(link, _frame(yolo=True))  # noqa: SLF001 — the seam under test
    assert refused.value.code == "not_permitted", refused.value
    message = str(refused.value)
    assert "unattended" in message, message
    # The remedy NAMES the deciding device (F7 design round 1, D1: this sentence
    # is relayed to the requesting device, where "this device" would mean the
    # READER's machine) and names a PRODUCT action — approving its setup in the
    # Mesh tab, whose cards are titled "Onboard <device>". It must NOT be the
    # "ask an admin" dead end (no wire op writes another device's copy) and must
    # not name a terminal command (§2.9).
    assert "approve setup for cloud-node-1 in the Mesh tab" in message, message
    assert "this device" not in message, message
    assert "ask an admin" not in message, message
    assert not (root / "sessions").exists(), "a refused create left something on disk"


def test_a_nameless_device_is_described_not_called_this_device(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F7 design round 1 (D1), fallback arm: with no name, describe the device.

    The named arm is pinned by the refusal cell above and by the member-caps flip
    cell; this keeps the no-name arm honest — the sentence still never says "this
    device", which is a fact about the READER's machine on the requesting side.
    """
    server = _server(root)
    monkeypatch.setattr(server.identity, "name", "")
    link = _link(capabilities=("prompt",))
    with pytest.raises(types.MeshRefusal) as refused:
        server._op_session_create(link, _frame(yolo=True))  # noqa: SLF001 — the seam under test
    message = str(refused.value)
    assert "approve that device's setup in the Mesh tab" in message, message
    assert "the device that would run the session grants" in message, message
    assert "this device" not in message, message


@pytest.mark.parametrize("value", ["false", "0", "no", None, False])
def test_a_non_request_never_hits_the_gate(
    root: Path, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``yolo``'s spellings that are NOT a request pass the gate untouched.

    The gate is a request detector (``wire.yolo_requested``), and a create that
    never asked for unattended must not meet the grant check at all — the same
    discipline the truthiness cell in ``test_refusals`` pins, exercised here
    through the create so the wiring cannot regress separately.
    """
    server = _server(root)
    link = _link(capabilities=("prompt",))

    def _no_warm(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(relay.RelayServer, "_warm_after_create", _no_warm)
    reply = server._op_session_create(link, _frame(yolo=value))  # noqa: SLF001
    assert reply["session_id"], reply
    stamp = read_stamp(root, str(reply["session_id"]))
    assert stamp is not None
    assert stamp.unattended is False, stamp


def test_a_granted_yolo_create_is_accepted_and_the_stamp_carries_it(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the grant, the create proceeds and ``unattended`` rides the stamp.

    The stamp is the channel the LATER runtime actually reads, so this cell is the
    one that proves the accepted authority travels — and it asserts the negative
    half too: the ungranted stamp reads False, so the field cannot drift into
    "always true" without going red here.
    """
    server = _server(root)
    link = _link(capabilities=("prompt", "unattended"))

    def _no_warm(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(relay.RelayServer, "_warm_after_create", _no_warm)
    reply = server._op_session_create(link, _frame(yolo=True))  # noqa: SLF001
    session_id = str(reply["session_id"])
    assert session_id, reply
    stamp = read_stamp(root, session_id)
    assert stamp is not None
    assert stamp.unattended is True, stamp
    # The origin names the requestING device, which is the row the construction
    # re-checks the grant against (``serving._carried_auto_authority``).
    assert stamp.origin.get("source_device") == link.device_id, stamp.origin
