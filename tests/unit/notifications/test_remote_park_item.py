"""A park on ANOTHER device raises a DURABLE row on the owning origin.

THE GAP THESE PIN (operator report, 2026-09-28 and the inventory item after it).
A session offloaded to a peer parks on a tool call there; the peer's runtime
holds the gate and nothing is pushed to the machine that started the work. The
TUI's notice is a toast plus one OS banner (``test_remote_park_notify.py``),
which is the right edge for a reader who is looking and nothing at all for one
who is not — the offload test's session sat parked for hours. The inbox is the
surface that exists for that (``docs/design/notifications-inbox.md`` §1), and
``Session.raise_gate_item`` already gives a LOCAL park a row; a remote park had
no writer, because there is no local ``Session`` behind it.

WHAT IS PINNED, and each one is a behaviour rather than an implementation:

* one row per live park, keyed by ``(device, kind)``, carrying the peer's name
  and the conversation's — a park is told apart from another device's by the
  device, and from another conversation's by the name;
* a second poll inside the same episode UPDATES that row instead of adding a
  second one (the keyed collapse every inbox writer relies on);
* the row carries the command-exact remedy, from the same constants the toast
  and the banner use — the copy is the deliverable, so it is asserted;
* a park that clears DISMISSES its row: a row still saying "waiting" after the
  answer landed is the same lie the toast used to tell;
* ``LOP_NOTIFY=0`` and a broken store both leave no row AND no exception — this
  write rides a five-second poll and a sidebar refresh, and a raise must never
  cost the notice it sits beside;
* THE NEGATIVE CONTROL, in this file because it is the promise the copy makes:
  denying needs no authority, and allowing does. The four cells at the end drive
  the production predicates (``frame_authority``, ``signature_verdict``,
  ``admit_increasing``) rather than restating them, so "a peer without operator
  authority cannot be allowed from the origin" stays a checked fact about the
  approval seam and not a sentence in a docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.notifications import remote_park as items
from local_operator.notifications.store import NotificationStore
from local_operator.tui.notify import remote_park_item_body, remote_park_item_title


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "config"


def _row(
    root: Path, *, session_id: str = "s_parked", device_id: str = "d_bb", kind: str = "approval"
) -> dict[str, Any] | None:
    return NotificationStore(root).get_by_key(
        session_id, items.item_key(device_id, session_id, kind)
    )


def _raise(root: Path, **overrides: Any) -> str | None:
    kwargs: dict[str, Any] = {
        "session_id": "s_parked",
        "device_id": "d_bb",
        "kind": "approval",
        "device_name": "demo-laptop",
        "name": "Backfill the audit log",
    }
    kwargs.update(overrides)
    return items.raise_park_item(root, **kwargs)


def test_a_park_on_a_peer_raises_one_row_naming_the_device_and_the_conversation(
    root: Path,
) -> None:
    """The row a reader away from the screen finds when they come back."""
    assert _raise(root) == "created"

    row = _row(root)
    assert row is not None, "a park on a peer must leave a row the operator can return to"
    assert row["title"] == remote_park_item_title("demo-laptop", "approval")
    assert "demo-laptop" in row["title"], "the row must say WHICH device holds the gesture"
    assert row["session_title"] == "Backfill the audit log"
    assert row["priority"] == "high", "a blocked agent outranks an informational update"
    assert row["session_id"] == "s_parked"
    assert row["dismissed"] is False


def test_the_row_carries_the_command_exact_remedy(root: Path) -> None:
    """Behaviour 3: the one privileged step, what it grants, and that deny works.

    The copy is the deliverable of this half as much as the row is — an operator
    who is told to "set up operator authority" has been given a topic, not a
    step. Asserted against the shipped constants so a later wording change
    cannot quietly drop one of the three facts.
    """
    _raise(root)
    row = _row(root)
    assert row is not None
    body = str(row["body_md"])

    assert body == remote_park_item_body("demo-laptop", "approval")
    # The commands, spelled as ``operator/cli.py`` ships them.
    assert "lop operator anchor export --file anchor.json" in body
    assert "lop operator install --from anchor.json" in body
    # WHAT IT GRANTS, said narrowly: verification of THIS machine's signature.
    assert "check a signature made on this machine" in body
    assert "grants nothing else" in body
    # AND THE ONE GESTURE THAT NEEDS NO AUTHORITY AT ALL.
    assert "deny" in remote_park_item_body("demo-laptop", "approval").lower()
    assert "Denying it works from here" not in body, (
        "the item body's lead is the card's own lead, which is the plain form: the "
        "attached reader's variant is a property of the card, not of the row"
    )
    assert "Open the session here to deny it." in body


def test_a_second_poll_inside_one_episode_updates_the_same_row(root: Path) -> None:
    """One park is one row, however many polls notice it."""
    assert _raise(root) == "created"
    first = _row(root)
    assert first is not None

    assert _raise(root) == "unchanged"
    again = _row(root)
    assert again is not None
    assert again["id"] == first["id"], "a repeat poll created a second row"
    assert again["revision"] == first["revision"]

    # A different conversation on the SAME device is its OWN row: the key
    # carries the session because a row is about one conversation.
    assert _raise(root, session_id="s_other") == "created"
    rows = _unread(root)
    assert len(rows) == 2, "two parked conversations on one device are two rows"


def test_two_devices_park_two_rows(root: Path) -> None:
    """The key carries the DEVICE, so one device's park cannot hide another's."""
    _raise(root)
    _raise(root, device_id="d_cc", device_name="cloud-node-1")

    rows = {str(row["title"]) for row in _unread(root)}
    assert len(rows) == 2
    assert any("demo-laptop" in title for title in rows)
    assert any("cloud-node-1" in title for title in rows)


def test_a_park_that_clears_dismisses_its_row(root: Path) -> None:
    """The row leaves with the park, the way the toast does."""
    _raise(root)
    assert _row(root) is not None

    assert items.clear_park_item(root, session_id="s_parked", device_id="d_bb", kind="approval")
    row = _row(root)
    assert row is not None
    assert row["dismissed"] is True
    assert row not in NotificationStore(root).list_items(state="unread")[0]

    # The second clear is a non-write, not an error: no row, nothing to dismiss.
    assert not items.clear_park_item(root, session_id="s_parked", device_id="d_bb", kind="approval")


def test_a_park_that_ends_before_its_row_was_written_is_not_an_error(root: Path) -> None:
    """A clear for a key this machine never raised answers ``False``."""
    assert not items.clear_park_item(
        root, session_id="s_never", device_id="d_none", kind="approval"
    )


def test_the_kill_switch_and_a_broken_store_both_leave_no_row_and_no_raise(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LOP_NOTIFY=0`` is the operator's own switch; a failing store is chrome's."""
    from local_operator.tools import builtin

    monkeypatch.setattr(builtin, "notify_enabled", lambda: False)
    assert _raise(root) is None
    assert _row(root) is None

    monkeypatch.setattr(builtin, "notify_enabled", lambda: True)

    class Exploding:
        def __init__(self, _root: Path) -> None:
            raise RuntimeError("no store today")

    monkeypatch.setattr(
        "local_operator.notifications.remote_park.NotificationStore",
        Exploding,
        raising=False,
    )
    # The store is imported inside the function, so patch the module's source.
    monkeypatch.setattr("local_operator.notifications.store.NotificationStore", Exploding)
    assert _raise(root) is None


def _park(
    device_id: str = "d_bb", session_id: str = "s_parked", kind: str = "approval"
) -> dict[str, str]:
    return {"device_id": device_id, "session_id": session_id, "kind": kind}


def _fresh(
    device_id: str = "d_bb", session_id: str = "s_parked", kind: str = "approval"
) -> dict[str, str]:
    return {
        **_park(device_id, session_id, kind),
        "device_name": "demo-laptop",
        "name": "Backfill the audit log",
    }


def test_a_clear_alone_ends_the_row(root: Path) -> None:
    """ORDERING PIN 1/3: a clear and nothing else leaves no live row."""
    _raise(root)
    items.write_transitions(root, [_park()], [])
    row = _row(root)
    assert row is not None and row["dismissed"] is True


def test_a_clear_and_a_raise_in_one_poll_end_present_and_unread(root: Path) -> None:
    """ORDERING PIN 2/3: the same key ending and starting is a LIVE park.

    This is the re-park inside one poll — ``park_edges`` reports the old key as
    ended and the new one as an edge — and the row must finish present, because
    the call really is waiting. ``write_transitions`` orders the clears before
    the raises for exactly this cell; swapping the two loops fails it.
    """
    _raise(root)
    items.write_transitions(root, [_park()], [_fresh()])
    row = _row(root)
    assert row is not None
    assert row["dismissed"] is False, "a live park's row was left dismissed"
    assert row["read"] is False, "a live park's row was left read"


def test_a_raise_then_a_clear_in_one_poll_ends_absent(root: Path) -> None:
    """ORDERING PIN 3/3: the OTHER order ends the row — the caller decides.

    Written as two calls because that is the only way to express it: the point
    is that the pair is not commutative, so the module's fixed order (clear,
    then raise) is what makes pin 2 true rather than an accident of list order.
    """
    _raise(root)
    items.write_transitions(root, [], [_fresh()])
    items.write_transitions(root, [_park()], [])
    row = _row(root)
    assert row is not None and row["dismissed"] is True


def test_the_banner_ladder_never_sees_a_park_row(tmp_path: Path) -> None:
    """R1-2: one park, one card — the ladder declines both park families.

    The exclusion is part of ``undelivered``'s SQL, so a burst of parks cannot
    fill the window and starve a plain item behind them (the failure a post-read
    filter would have caused), and an UNKEYED row is untouched.
    """
    from local_operator.notifications.delivery import (
        PARK_ITEM_PREFIXES,
        announced_by_its_park,
    )

    remote_key = items.item_key("d_bb", "s_parked", "approval")
    gate_key = ":".join(("gate", "approval", "req-1"))

    store = NotificationStore(tmp_path)
    remote = store.notify(
        "s_parked", title="Waiting for approval on demo-laptop", priority="high", key=remote_key
    )
    local = store.notify("s_local", title="Approve the command", priority="high", key=gate_key)
    plain = store.notify("s_plain", title="A plain item", priority="high")

    owed = store.undelivered(priority="high", exclude_prefixes=PARK_ITEM_PREFIXES)
    assert [row["id"] for row in owed] == [plain.item_id]
    # The parks were DECLINED, not consumed: their own announce paths still own
    # them, so both are still claimable by the leg that speaks for a park.
    everything = {row["id"] for row in store.undelivered(priority="high")}
    assert {remote.item_id, local.item_id} <= everything
    assert announced_by_its_park({"key": remote_key})
    assert announced_by_its_park({"key": gate_key})
    assert not announced_by_its_park({"key": ""})


def test_a_new_park_reopens_a_row_that_was_read_or_dismissed(root: Path) -> None:
    """The collapse must not swallow a SECOND wait on the same key.

    ``store.notify`` leaves an unchanged row alone when it is unread, and its
    ``notify`` on a dismissed row only un-dismisses it on a PRIORITY escalation
    — a park is already ``high``, so nothing would ever bring it back. Without
    the reopen, a session that was parked, read (or answered and cleared), and
    then parked again the same way would show nothing, which is the gap this
    slice exists to close, re-created by its own collapse rule.
    """
    _raise(root)
    row = _row(root)
    assert row is not None

    # The operator read it, then the park was answered and the row dismissed.
    store = NotificationStore(root)
    store.apply("read", ids=[(str(row["id"]), int(row["revision"]))], via="explicit")
    assert items.clear_park_item(root, session_id="s_parked", device_id="d_bb", kind="approval")
    settled = _row(root)
    assert settled is not None and settled["dismissed"] is True

    # The conversation parks again, the same way.
    _raise(root)
    again = _row(root)
    assert again is not None
    assert again["id"] == settled["id"], "a settled park's row was replaced instead of reopened"
    assert again["dismissed"] is False, "a second park stayed hidden behind the first row"
    assert again["read"] is False, "a second park left the row already-read"


def _unread(root: Path) -> list[dict[str, Any]]:
    return NotificationStore(root).list_items(state="unread")[0]


# --- the negative control: authority is not widened by this slice -------------
#
# The remedy claims a peer can be given the ability to CHECK a signature made
# here, and that denying needs nothing. Both are claims about the approval
# seam, so they are asserted there. If any of these ever fail, the copy in
# ``tui/notify.py`` has become a promise the code does not keep.


def test_denying_a_parked_call_needs_no_authority_and_allowing_does() -> None:
    """The asymmetry the whole copy rests on, from the production predicates."""
    from local_operator.harness.approval import frame_authority, signature_target

    deny = {"op": "approval_answer", "approved": False, "request_id": "req-1"}
    allow = {"op": "approval_answer", "approved": True, "request_id": "req-1"}

    # DENY: ordinary, so it travels on the connection's own key — this is why
    # "denying it works from here" is true from any attached surface.
    assert frame_authority(deny) == "ordinary"
    assert signature_target(deny) is None

    # ALLOW: authority-increasing, so no actor without the operator key can do
    # it, and the request id it settles is baked into what is signed.
    assert frame_authority(allow) == "authority-increasing"
    assert signature_target(allow) == ("approve", "req-1")


def test_a_peer_with_no_anchor_refuses_a_presented_signature() -> None:
    """A signature this machine makes is worthless where nothing can place it.

    ``operator_spki=None`` is a host with no installed anchor — the state the
    remedy exists to end — and a well-formed signature there is a REFUSAL, not
    an "absent" that some other source may paper over.
    """
    from local_operator.harness.approval import admit_increasing
    from local_operator.operator.verify import signature_verdict

    verdict = signature_verdict(
        action="approve",
        session_id="s_parked",
        request_id="req-1",
        challenge="c0ffee",
        signature_hex="00" * 64,
        operator_spki=None,
        operator_key_id="a" * 32,
        operator_cert=None,
        device_spki=None,
        now=0,
    )
    assert verdict is False, "a signature with no anchor to check it against was not refused"
    assert admit_increasing(capability=False, signature=verdict) is False


def test_the_relayed_challenge_cannot_be_asked_for_without_the_approve_capability() -> None:
    """The mesh's own gate, so this slice cannot have widened who may attempt it.

    ``operator_challenge`` is the material a remote allow is signed FROM; asking
    for one needs the ``approve`` grant, and a member without it is refused at
    the relay rather than merely failing later at the runtime.
    """
    from local_operator.network.types import GRANTABLE_CAPABILITIES, INNER_OP_CAPABILITY

    assert INNER_OP_CAPABILITY["operator_challenge"] == "approve"
    assert INNER_OP_CAPABILITY["approval_answer"] == "prompt"
    assert "approve" in GRANTABLE_CAPABILITIES
    # And nothing in this slice's copy asks the reader to do it from here.
    body = remote_park_item_body("demo-laptop", "approval").lower()
    assert "allow it here" not in body
    assert "allow from here" not in body
