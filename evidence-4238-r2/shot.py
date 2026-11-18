"""PR #4238 round-1 remediation frames: TUI Team Bridge card states.

usage: shot.py <repo_root> <out_dir> <tag>
Runs the REAL OperatorApp (stylesheet loaded) under an isolated HOME/config.
The resolver is real (local display store, local holds/reservations, journal
rows); only the hub wire is a double, and only the refused card reaches it.
"""

import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO, OUT, TAG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(REPO))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

import yaml  # noqa: E402

from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.providers.auth_store import AuthStore  # noqa: E402
from local_operator.team_bridge import read_context  # noqa: E402
from local_operator.team_bridge.display_bodies import DisplayBodyStore  # noqa: E402
from local_operator.team_bridge.store import (  # noqa: E402
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    _owner_key,
)
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.bridge_card_body import card_body_loader  # noqa: E402
from local_operator.tui.widgets.transcript import TeamBridgeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CID = "bcv_abcdefgh"
SESSION = "abcdef012345"
OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_1",
    tenant_id="3f2a1111-2222-4333-8444-555566667777",
    endpoint_id="ep_1",
    canonical_base_url="https://hub.example",
    enrollment_generation=1,
)
SHORT = "**Hello** from Ola — can you check the `release` notes before Friday?"
LONG = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog again" for i in range(30))

# (entry, message id, sender name or "", state seed)
CARDS = [
    ("e_admit", "bcmsg_admit001", "Ola", "local"),
    ("e_noname", "bcmsg_noname01", "", "local"),
    ("e_held", "bcmsg_held0001", "Priya", "hold_review"),
    ("e_heldt", "bcmsg_heldt001", "Priya", "hold_transient"),
    ("e_pend", "bcmsg_pend0001", "Sam", "pending"),
    ("e_old", "bcmsg_old00001", "Ola", "none"),
    ("e_refused", "bcmsg_refus001", "Kim", "hub_refused"),
    ("e_long", "bcmsg_long0001", "Ola", "long"),
]


def seed(root: Path) -> None:
    store = Store(root)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION)
    (root / "config.yml").write_text(yaml.safe_dump({"values": {"team_bridge": {"enabled": True}}}))
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    row = auth.upsert_credential(
        "radient", {"refresh": "r", "access": "a", "expires": int(time.time() * 1000) + 3_600_000}
    )
    auth.close()
    with sqlite3.connect(root / "auth.db") as conn:
        conn.execute("UPDATE auth_credentials SET id = 7 WHERE id = ?", (row.id,))
    epoch = store.publish_screening_policy(
        OWNER, version="v1", template_sha256="ab" * 32, default_model="m", revision="r"
    )
    display = DisplayBodyStore(root)
    directory = root / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    for seq, (entry, mid, name, kind) in enumerate(CARDS, start=1):
        body = LONG if kind == "long" else SHORT
        if kind in ("local", "long"):
            display.record(
                _owner_key(OWNER),
                message_id=mid,
                conversation_id=CID,
                body=body,
                body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                sender_kind="human",
                source="turn",
            )
        if kind == "pending":
            store.record_body_reservation(
                OWNER,
                conversation_id=CID,
                message_id=mid,
                conversation_seq=seq,
                mention=True,
                body_sha256="cd" * 32,
                screening_fingerprint="aa11" * 16,
                policy_epoch=epoch,
                consent_version="bridge_consent_v1",
                designated_session=SESSION,
                now=time.time(),
            )
        from tests.unit.team_bridge.test_card_body import _details

        details = _details([mid], seq=seq)
        details["bridge"] = {"sender_name": name, "organization": "Acme"} if name else {}
        line = {
            "id": entry,
            "ts": time.time(),
            "type": "custom",
            "payload": {
                "kind": "custom",
                "custom_type": TEAM_BRIDGE_MESSAGE_TYPE,
                "attribution": "system",
                "details": details,
            },
        }
        with (directory / "transcript.jsonl").open("a") as handle:
            handle.write(json.dumps(line) + "\n")


HOLDS = {
    "bcmsg_held0001": {"hold_class": "model_hold", "created_at": time.time() - 40},
    "bcmsg_heldt001": {"hold_class": "screening_unavailable", "created_at": time.time() - 40},
}
Store.hold_states_readonly = lambda self, owner, ids: {m: HOLDS[m] for m in ids if m in HOLDS}  # type: ignore[method-assign]


class Wire:
    base_url = "https://hub.example"

    async def current_access(self):
        return SimpleNamespace(account_id="acc_1", kind="oauth", credential_id=7)

    async def conversation_messages(self, conversation_id, *, before_seq, limit, **kwargs):
        # A realistic page: the newest row with seq < before_seq, as the hub
        # serves it (admitted rows carry their body; refused a null body).
        rows = []
        for seq, (entry, mid, name, kind) in enumerate(CARDS, start=1):
            if seq >= before_seq:
                continue
            body = LONG if kind == "long" else SHORT
            row = {"id": mid, "conversation_id": CID, "seq": seq, "body": None}
            if kind in ("local", "long"):
                row.update(body=body, body_hash=hashlib.sha256(body.encode()).hexdigest())
            elif kind == "hub_refused":
                row["body_state"] = {"state": "rejected"}
            elif kind.startswith("hold"):
                row["body_state"] = {"state": "held"}
            else:
                row["body_state"] = {"state": "not_admitted"}
            rows.append(row)
        return {"messages": rows[::-1][:limit]}

    async def conversation_body(self, *a, **k):  # pragma: no cover
        raise AssertionError("classification fetch")


read_context.wire_for = lambda _c: Wire()


async def main() -> None:
    root = config_dir()
    seed(root)
    session_view = SimpleNamespace(_transcript=SimpleNamespace(directory=root / "sessions" / SESSION))
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("dark", "light"):
        for cols in (110, 60):
            app = OperatorApp(lambda: _factory(FakeSession()))
            async with app.run_test(size=(cols, 50)) as pilot:
                for _ in range(4):
                    await pilot.pause()
                if theme == "light":
                    app._apply_theme("light")
                cards = {}
                for entry, mid, name, kind in CARDS:
                    sender = {"name": name, "organization": "Acme"} if name else {}
                    text = "an organization member — (no preview available)" if not name else "(no preview available)"
                    block = TeamBridgeBlock(text, sender, body_loader=card_body_loader(session_view, entry))
                    app._append_block(block)
                    cards[kind if kind != "local" or name else "noname"] = block
                for _ in range(6):
                    await pilot.pause(0.05)
                if cols == 110:
                    print(theme, {k: (b._card_body or {}).get("items") and [(i["state"], i.get("reason")) for i in b._card_body["items"]] for k, b in cards.items()}, flush=True)
                    save_capture(app, OUT / f"{TAG}-collapsed-{theme}.svg")
                    # States opened: review hold, transient hold, refused (hub).
                    for key in ("local", "hold_review", "hold_transient", "hub_refused"):
                        cards[key].toggle_expanded()
                    for _ in range(8):
                        await pilot.pause(0.05)
                    save_capture(app, OUT / f"{TAG}-states-open-{theme}.svg")
                    for key in ("local", "hold_review", "hold_transient", "hub_refused"):
                        cards[key].toggle_expanded()
                    for _ in range(4):
                        await pilot.pause(0.05)
                    save_capture(app, OUT / f"{TAG}-states-reclosed-{theme}.svg")
                long_card = cards["long"]
                long_card.toggle_expanded()
                for _ in range(8):
                    await pilot.pause(0.05)
                long_card.scroll_visible(animate=False)
                await pilot.pause(0.1)
                save_capture(app, OUT / f"{TAG}-long-{cols}-{theme}.svg")
                if theme == "dark":
                    long_card.focus()
                    for _ in range(4):
                        await pilot.pause(0.05)
                    save_capture(app, OUT / f"{TAG}-long-{cols}-{theme}-focused.svg")
                    geo = long_card.region
                    print(f"{TAG} {cols}: long card height={geo.height} show_more_row={long_card._show_more_row}")


asyncio.run(main())
