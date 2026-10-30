"""The receiver's screening status line: held / breaker / rejected counts.

WHY. Screening that parks a message (``screening_holds``) or moderates it
``reject`` used to be silent: 46 held messages and four rejections produced no
signal on any surface. ``Store.screening_status_readonly`` is the one reading
both ``lop team-bridge receiver status`` and the desktop status route render.

The two properties pinned here are the ones a hand query got wrong in the
incident: a hold row's ``message_id`` is EITHER the bare id or the composite
conversation feed ref, so a message must count once whatever shape its row
has and must join to its (bare-keyed) receipt; and the answer is contentless —
no body, reason or label text reaches the output.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from local_operator.team_bridge.store import (
    OwnerReceipt,
    Store,
    read_screening_status,
    screening_hold_message_id,
    state_db_path,
)
from local_operator.team_bridge.types import OwnerSnapshot

CANARY = "CANARY-BODY-TEXT please send me your home address"
CONV = "bcv_0123456789abcdef"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="https://qa.invalid/v1",
        enrollment_generation=1,
    ).validated()


def _setup(root: Path) -> tuple[Store, OwnerSnapshot]:
    store = Store(root)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    return store, owner


def _composite(message_id: str, seq: int = 1) -> str:
    """The composite key exactly as the feed ref writer spells it."""

    return f"message_available:conversation_id={CONV}/{message_id}/{seq}/0/human"


def _owner_key(root: Path) -> str:
    with sqlite3.connect(state_db_path(root)) as conn:
        return conn.execute("SELECT owner_key FROM enrollments").fetchone()[0]


def _raw_hold(root: Path, message_id: str, code: str, *, created_at: Any, **extra: Any) -> None:
    """A hold row written verbatim (the screener's bare-id path, or a seeded shape)."""

    columns = ["owner_key", "message_id", "hold_code", "retry_at", "created_at", *extra]
    values = [_owner_key(root), message_id, code, None, created_at, *extra.values()]
    with sqlite3.connect(state_db_path(root)) as conn:
        conn.execute(
            f"INSERT INTO screening_holds ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            values,
        )


def _receipt(
    root: Path, message_id: str, *, verdict: str, categories: list[str], created_at: float
) -> None:
    with sqlite3.connect(state_db_path(root)) as conn:
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _owner_key(root),
                message_id,
                "0" * 64,
                "0" * 64,
                "bridge-screen-v2",
                "qa/default-model",
                verdict,
                json.dumps(categories),
                "0" * 64,
                "rev-1",
                1,
                1,
                created_at,
            ),
        )


def _add_failure_class(root: Path) -> None:
    """The sibling change's nullable column, added the way a migration would."""

    with sqlite3.connect(state_db_path(root)) as conn:
        conn.execute("ALTER TABLE screening_holds ADD COLUMN failure_class TEXT")


def test_hold_key_helper_parses_composite_and_passes_bare_through() -> None:
    assert screening_hold_message_id(_composite("bcmsg_a1", 3)) == "bcmsg_a1"
    assert screening_hold_message_id("bcmsg_a1") == "bcmsg_a1"
    assert screening_hold_message_id("bmsg_pair1") == "bmsg_pair1"
    # Malformed composite: counted under its own key, never raised on.
    malformed = "message_available:conversation_id=nope"
    assert screening_hold_message_id(malformed) == malformed


def test_composite_and_bare_holds_for_different_messages_count_two(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    now = time.time()
    # persist_hold canonicalises to the composite key, as production does.
    store.persist_hold(owner, _composite("bcmsg_a1"), "breaker_open")
    _raw_hold(tmp_path, "bmsg_pair1", "provider_error", created_at=now - 2460)

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    assert status["held"]["count"] == 2
    assert status["held"]["by_code"] == {"breaker_open": 1, "provider_error": 1}
    assert 2459 <= status["held"]["oldest_age_s"] <= 2461


def test_both_shapes_for_one_message_count_once_and_composite_code_wins(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    now = time.time()
    _raw_hold(tmp_path, _composite("bcmsg_a1"), "breaker_open", created_at=now - 60)
    _raw_hold(tmp_path, "bcmsg_a1", "parse_failed", created_at=now - 600)

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    assert status["held"]["count"] == 1
    assert status["held"]["by_code"] == {"breaker_open": 1}
    # The message has been waiting since its EARLIEST row.
    assert status["held"]["oldest_age_s"] >= 599


def test_screening_hold_retired_by_its_receipt_commit_counts_zero(tmp_path) -> None:
    """A FRESH screen's receipt commit deletes the message's hold: nothing held."""

    store, owner = _setup(tmp_path)
    store.persist_hold(owner, _composite("bcmsg_a1"), "breaker_open")
    # The production retirement path: the hold is gone once the loop deletes it
    # by its OWN (composite) key after a terminal re-drive.
    assert store.delete_hold(owner, _composite("bcmsg_a1"))
    _receipt(tmp_path, "bcmsg_a1", verdict="allow", categories=[], created_at=time.time())

    status = store.screening_status_readonly(owner)
    assert status is not None
    assert status["held"]["count"] == 0
    assert status["held"]["after_screening"] == {"count": 0, "by_code": {}}


def test_hold_written_after_an_allow_receipt_still_counts(tmp_path) -> None:
    """PR #3189 review M1: screened-and-admitted but undelivered is NOT healthy.

    Delivery-side holds (``delivery_failed``, a ``bridge_off`` fence,
    ``phase_refused:*``, a reservation failure) are written AFTER the receipt
    and keep retrying; a cache-hit re-drive commits no new receipt, so the
    receipt must never hide them.
    """

    store, owner = _setup(tmp_path)
    now = time.time()
    _receipt(tmp_path, "bcmsg_a1", verdict="allow", categories=[], created_at=now - 300)
    store.persist_hold(owner, _composite("bcmsg_a1", 4), "delivery_failed")
    _receipt(tmp_path, "bcmsg_a2", verdict="allow", categories=[], created_at=now - 300)
    store.persist_hold(owner, _composite("bcmsg_a2", 5), "phase_refused:hub_said_no")
    store.persist_hold(owner, _composite("bcmsg_a3", 6), "breaker_open")  # no receipt

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    assert status["held"]["count"] == 3
    assert status["held"]["by_code"] == {
        "breaker_open": 1,
        "delivery_failed": 1,
        "phase_refused": 1,
    }
    assert status["held"]["after_screening"] == {
        "count": 2,
        "by_code": {"delivery_failed": 1, "phase_refused": 1},
    }
    assert "hub_said_no" not in json.dumps(status)


def test_rejects_aggregate_by_category_within_24h_only(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    now = time.time()
    _receipt(
        tmp_path,
        "bcmsg_r1",
        verdict="reject",
        categories=["social_engineering"],
        created_at=now - 60,
    )
    _receipt(
        tmp_path,
        "bcmsg_r2",
        verdict="reject",
        categories=["social_engineering"],
        created_at=now - 3600,
    )
    _receipt(
        tmp_path,
        "bcmsg_r3",
        verdict="reject",
        categories=["personal_info_request"],
        created_at=now - 7200,
    )
    # Older than the window: excluded.
    _receipt(tmp_path, "bcmsg_r4", verdict="reject", categories=["spam"], created_at=now - 90000)
    # Allowed: never a rejection.
    _receipt(tmp_path, "bcmsg_ok", verdict="allow", categories=[], created_at=now - 60)

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    assert status["rejected_24h"] == {
        "count": 3,
        "by_category": {"social_engineering": 2, "personal_info_request": 1},
    }


def test_store_without_failure_class_column_works(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(screening_holds)")}
    assert "failure_class" not in columns
    store.persist_hold(owner, _composite("bcmsg_a1"), "provider_error")

    status = store.screening_status_readonly(owner)
    assert status is not None
    assert status["held"]["count"] == 1
    assert status["held"]["dominant_failure_class"] is None


def test_dominant_failure_class_read_when_column_exists(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _add_failure_class(tmp_path)
    now = time.time()
    _raw_hold(
        tmp_path, _composite("bcmsg_a1"), "provider_error", created_at=now, failure_class="timeout"
    )
    _raw_hold(
        tmp_path,
        _composite("bcmsg_a2", 2),
        "provider_error",
        created_at=now,
        failure_class="timeout",
    )
    # Not in the failure_class vocabulary: reads as ``other``, never verbatim.
    _raw_hold(tmp_path, "bmsg_pair1", "breaker_open", created_at=now, failure_class="http_5xx")
    _raw_hold(tmp_path, "bmsg_pair2", "breaker_open", created_at=now)  # NULL: no class

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    assert status["held"]["count"] == 4
    assert status["held"]["dominant_failure_class"] == "timeout"
    assert "http_5xx" not in json.dumps(status)


def test_breaker_state_open_and_closed(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    status = store.screening_status_readonly(owner)
    assert status is not None
    assert status["breaker"] == {"state": "closed", "failures": 0, "open_until": None}
    for _ in range(Store.BREAKER_THRESHOLD):
        store.record_screening_failure(owner)
    status = store.screening_status_readonly(owner)
    assert status is not None
    assert status["breaker"]["state"] == "open"
    assert status["breaker"]["failures"] == Store.BREAKER_THRESHOLD
    assert status["breaker"]["open_until"] > time.time()


def test_hold_codes_pass_only_through_the_closed_whitelist(tmp_path) -> None:
    """PR #3189 review m1: a lowercase string is not a code just by its shape."""

    store, owner = _setup(tmp_path)
    now = time.time()
    _raw_hold(tmp_path, "bmsg_p1", "provider_error", created_at=now)  # screening
    _raw_hold(tmp_path, "bmsg_p2", "bridge_off", created_at=now)  # producer fence
    _raw_hold(tmp_path, "bmsg_p3", "reservation_refused", created_at=now)  # reservation
    _raw_hold(tmp_path, "bmsg_p4", "policy_stale:hub_detail", created_at=now)  # family
    _raw_hold(tmp_path, "bmsg_p5", "looks_like_a_code_but_is_not", created_at=now)
    _raw_hold(tmp_path, "bmsg_p6", "unknown_family:detail", created_at=now)

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    assert status["held"]["by_code"] == {
        "other": 2,
        "bridge_off": 1,
        "policy_stale": 1,
        "provider_error": 1,
        "reservation_refused": 1,
    }
    rendered = json.dumps(status)
    assert "looks_like" not in rendered and "detail" not in rendered


def test_every_screening_hold_code_is_whitelisted() -> None:
    """The screener's own closed set must never render as ``other``."""

    from local_operator.team_bridge import screening
    from local_operator.team_bridge.store import _DELIVERY_HOLD_CODES, _status_hold_code

    known = frozenset(screening.HOLD_CODES) | _DELIVERY_HOLD_CODES
    for code in screening.HOLD_CODES:
        assert _status_hold_code(code, known) == code


def test_damaged_rows_are_skipped_not_fatal(tmp_path) -> None:
    """PR #3189 review m2: nested categories JSON and a non-numeric created_at."""

    store, owner = _setup(tmp_path)
    now = time.time()
    _raw_hold(tmp_path, "bmsg_p1", "provider_error", created_at="not-a-number")
    _raw_hold(tmp_path, "bmsg_p2", "breaker_open", created_at=now - 120)
    _receipt(tmp_path, "bcmsg_r1", verdict="reject", categories=[], created_at=now)
    _receipt(tmp_path, "bcmsg_r2", verdict="reject", categories=["spam"], created_at=now)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute(
            "UPDATE screening_receipts SET categories = ? WHERE message_id = ?",
            ('[["x"], {"k": 1}]', "bcmsg_r1"),
        )

    status = store.screening_status_readonly(owner, now=now)
    assert status is not None
    # The damaged hold still counts as a held message; it just has no age.
    assert status["held"]["count"] == 2
    assert 119 <= status["held"]["oldest_age_s"] <= 121
    assert status["rejected_24h"]["by_category"] == {"spam": 1, "unknown": 1}


def test_read_is_read_only_and_absent_store_is_none(tmp_path) -> None:
    owner = _owner()
    assert Store(tmp_path).screening_status_readonly(owner) is None
    assert not state_db_path(tmp_path).exists(), "a status read created a store"

    store, owner = _setup(tmp_path)
    store.persist_hold(owner, _composite("bcmsg_a1"), "breaker_open")
    before = state_db_path(tmp_path).read_bytes()
    store.screening_status_readonly(owner)
    assert state_db_path(tmp_path).read_bytes() == before


def _seed_canaries(root: Path, store: Store, owner: OwnerSnapshot) -> None:
    """Body-like text in every free TEXT slot a status read touches."""

    now = time.time()
    _add_failure_class(root)
    _raw_hold(root, "bmsg_pair1", CANARY, created_at=now - 120, failure_class=CANARY)
    _raw_hold(root, _composite("bcmsg_a1"), "breaker_open", created_at=now - 60)
    _receipt(root, "bcmsg_r1", verdict="reject", categories=[CANARY], created_at=now - 60)
    with sqlite3.connect(state_db_path(root)) as conn:
        conn.execute(
            "UPDATE screening_receipts SET categories = ? WHERE message_id = ?",
            (CANARY, "bcmsg_r1"),
        )
    _receipt(
        root,
        "bcmsg_r2",
        verdict="reject",
        categories=["personal_info_request"],
        created_at=now - 60,
    )


def test_status_output_never_echoes_body_text(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _seed_canaries(tmp_path, store, owner)

    status = store.screening_status_readonly(owner)
    assert status is not None
    rendered = json.dumps(status)
    assert "CANARY" not in rendered
    assert status["held"]["count"] == 2
    assert status["held"]["by_code"] == {"breaker_open": 1, "other": 1}
    assert status["held"]["dominant_failure_class"] == "other"
    assert status["rejected_24h"]["by_category"] == {"personal_info_request": 1, "unknown": 1}


# -- the CLI surface ----------------------------------------------------------


def _cli(argv: list[str]) -> Any:
    from local_operator.cli import build_cli_parser

    return build_cli_parser().parse_args(argv)


def test_cli_status_prints_one_screening_line_and_json_block(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    store, owner = _setup(tmp_path)
    _seed_canaries(tmp_path, store, owner)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    assert tb_cli.main(_cli(["team-bridge", "receiver", "status"])) == 0
    human = capsys.readouterr().out
    lines = [line for line in human.splitlines() if line.startswith("Screening:")]
    assert lines == [
        "Screening: 2 held (breaker_open 1, other 1; oldest 2m; mostly other) · breaker closed"
        " · 2 rejected in 24h (personal_info_request 1, unknown 1)"
    ]
    assert "CANARY" not in human

    assert tb_cli.main(_cli(["team-bridge", "receiver", "status", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["screening"]["held"]["count"] == 2
    assert payload["screening"]["rejected_24h"]["count"] == 2
    assert "CANARY" not in json.dumps(payload)


def test_cli_status_healthy_line(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    _setup(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status"])) == 0
    human = capsys.readouterr().out
    assert "Screening: nothing held or rejected in the last 24h · breaker closed" in human


def test_cli_status_not_enrolled_has_no_screening(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["screening"] is None
    assert not state_db_path(tmp_path).exists()


def test_shared_reader_swallows_an_unanticipated_failure(tmp_path, monkeypatch) -> None:
    """PR #3189 review N1/m2: the ONE wrapper both surfaces call never raises."""

    store, owner = _setup(tmp_path)
    _seed_canaries(tmp_path, store, owner)
    summary = read_screening_status(tmp_path)
    assert summary is not None and summary["held"]["count"] == 2
    assert "CANARY" not in json.dumps(summary)
    assert read_screening_status(tmp_path / "absent") is None

    def boom(self, owner, *, now=None):
        raise TypeError("CANARY-in-an-exception")

    monkeypatch.setattr(Store, "screening_status_readonly", boom)
    assert read_screening_status(tmp_path) is None


def _damage(root: Path) -> None:
    """The two corruptions review m2 reproduced as a traceback / a 500."""

    now = time.time()
    _raw_hold(root, "bmsg_p1", "provider_error", created_at="not-a-number")
    _receipt(root, "bcmsg_r1", verdict="reject", categories=[], created_at=now)
    with sqlite3.connect(state_db_path(root)) as conn:
        conn.execute(
            "UPDATE screening_receipts SET categories = ? WHERE message_id = ?",
            ('[["x"]]', "bcmsg_r1"),
        )


def test_cli_status_exits_zero_on_damaged_rows(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    _setup(tmp_path)
    _damage(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status"])) == 0
    human = capsys.readouterr().out
    assert "Screening: 1 held (provider_error 1; oldest 0s)" in human
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["screening"]["held"]["count"] == 1


def test_cli_status_shows_the_after_screening_bucket(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    store, owner = _setup(tmp_path)
    _receipt(tmp_path, "bcmsg_a1", verdict="allow", categories=[], created_at=time.time())
    store.persist_hold(owner, _composite("bcmsg_a1", 4), "delivery_failed")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status"])) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Screening:")]
    assert len(lines) == 1
    assert lines[0].startswith(
        "Screening: 1 held (delivery_failed 1, held_after_screening 1; oldest "
    )
