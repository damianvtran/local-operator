"""QwenCloud console ticket storage: upsert identity, mode refusal, and the
rule that the VALUE never reaches stdout, stderr, or an error message.

Every test uses a `tmp_path` store. Nothing here may touch
``~/.local-operator/auth.db``, and the only ticket-shaped literal is the
obvious fake below.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.cli import _qwencloud_ticket_action, qwencloud_ticket_command
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.qwencloud_console import (
    QWENCLOUD_CONSOLE_PROJECT_ID,
    QWENCLOUD_CONSOLE_PROVIDER,
    QWENCLOUD_TICKET_STALE_MS,
    TicketStoreError,
    TicketStoreUnreadable,
    delete_ticket,
    read_ticket_record,
    store_ticket,
)

#: Never a real cookie. A test that needs a "different" value uses the -2 form.
FAKE_TICKET = "fake-console-ticket"
FAKE_TICKET_2 = "fake-console-ticket-2"


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AuthStore]:
    # Hermeticity: the legacy env tier reads ~/.local-operator/credentials.env;
    # point the whole config dir at tmp_path so no real credential is in reach.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    opened = AuthStore(db_path=tmp_path / "auth.db")
    try:
        yield opened
    finally:
        opened.close()


def _persisted_data(store: AuthStore) -> dict[str, Any]:
    rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER)
    assert len(rows) == 1
    return rows[0].data


def test_set_then_reset_upserts_in_place(store: AuthStore) -> None:
    """Re-entry UPDATES one row. Two rows here means the identity key broke."""
    store_ticket(store, FAKE_TICKET)
    store_ticket(store, FAKE_TICKET_2)

    rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER)
    assert len(rows) == 1, f"expected one row, got identity keys {[r.identity_key for r in rows]}"
    assert rows[0].identity_key == QWENCLOUD_CONSOLE_PROJECT_ID
    assert rows[0].data["ticket"] == FAKE_TICKET_2


def test_the_row_carries_no_type_or_key_field(store: AuthStore) -> None:
    """`key` would route a browser cookie into the API-key cascade as a bearer."""
    store_ticket(store, FAKE_TICKET)
    data = _persisted_data(store)

    assert data["ticket"] == FAKE_TICKET
    assert data["project_id"] == QWENCLOUD_CONSOLE_PROJECT_ID
    assert isinstance(data["captured_at"], int)
    # The store stamps `type` on the way in; that is expected and coerced to
    # api_key because there is no refresh+access pair.
    assert data["type"] == "api_key"
    assert "key" not in data
    assert "source" not in data


def test_it_refuses_a_world_readable_store(tmp_path: Path) -> None:
    db_path = tmp_path / "auth.db"
    opened = AuthStore(db_path=db_path)
    try:
        os.chmod(db_path, 0o644)
        with pytest.raises(TicketStoreError) as excinfo:
            store_ticket(opened, FAKE_TICKET)
        message = str(excinfo.value)
        assert "0644" in message
        assert "chmod 600" in message
        assert FAKE_TICKET not in message
    finally:
        os.chmod(db_path, 0o600)
        opened.close()


def test_it_refuses_a_world_readable_directory(tmp_path: Path) -> None:
    directory = tmp_path / "wide"
    directory.mkdir(mode=0o755)
    opened = AuthStore(db_path=directory / "auth.db")
    try:
        with pytest.raises(TicketStoreError) as excinfo:
            store_ticket(opened, FAKE_TICKET)
        message = str(excinfo.value)
        assert "0755" in message
        assert "chmod 700" in message
        assert FAKE_TICKET not in message
    finally:
        opened.close()


def test_an_unresolvable_store_path_raises_rather_than_skipping() -> None:
    """A precondition that cannot be EVALUATED must not be treated as SATISFIED.

    The regression is the fail-open shape: a falsy check would skip the mode
    test and write the cookie while reporting success.
    """

    class PathlessStore:
        def __init__(self) -> None:
            self.written: list[tuple[str, dict[str, Any]]] = []

        def upsert_credential(self, provider: str, credential: dict[str, Any]) -> None:
            self.written.append((provider, credential))

    pathless = PathlessStore()
    with pytest.raises(TicketStoreError) as excinfo:
        store_ticket(pathless, FAKE_TICKET)
    assert FAKE_TICKET not in str(excinfo.value)
    assert pathless.written == [], "the cookie must not be written when the path is unknown"


class UnreadableStore:
    """A store whose rows cannot be read: SQLITE_BUSY, locked, or corrupt.

    `busy_timeout` is 5s (auth_store.py:407), so on a machine running many
    concurrent `lop` processes this is an ordinary outcome, not a contrivance.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or sqlite3.OperationalError("database is locked")
        self.deleted: list[int] = []

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[Any]:
        raise self.error

    def delete_credential(self, credential_id: int) -> None:
        self.deleted.append(credential_id)


def test_an_unreadable_store_is_not_reported_as_an_empty_one() -> None:
    """The false negative that made `rm` claim success over a live cookie.

    "cannot read" and "nothing stored" must not collapse into one answer.
    """
    with pytest.raises(TicketStoreUnreadable):
        read_ticket_record(UnreadableStore())

    with pytest.raises(TicketStoreUnreadable):
        delete_ticket(UnreadableStore())


def test_a_caller_bug_still_propagates_rather_than_degrading() -> None:
    """ProgrammingError is a BUG (connection across threads), not an environment fact.

    The repo's rule at controller.py:268-278: a bug dressed as a plausible
    degraded state is one nobody finds.
    """
    unreadable = UnreadableStore(sqlite3.ProgrammingError("closed database"))
    with pytest.raises(sqlite3.ProgrammingError):
        read_ticket_record(unreadable)


def test_rm_exits_non_zero_when_it_cannot_prove_the_ticket_is_gone(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A revoke it cannot confirm must fail loudly and say the ticket may remain."""
    assert _qwencloud_ticket_action("rm", UnreadableStore()) == 1
    captured = capsys.readouterr()
    assert "MAY STILL BE STORED" in captured.err
    assert "No QwenCloud console ticket stored." not in captured.out


def test_status_reports_unknown_rather_than_absent_on_an_unreadable_store(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _qwencloud_ticket_action("status", UnreadableStore()) == 1
    captured = capsys.readouterr()
    assert "UNKNOWN" in captured.err
    assert "No QwenCloud console ticket stored." not in captured.out


def test_delete_ticket_confirms_the_row_is_actually_gone(store: AuthStore) -> None:
    """`delete_credential` returns None either way, so deletion must be re-read."""
    store_ticket(store, FAKE_TICKET)

    # A store that accepts the delete but keeps the row: the exact shape of a
    # silent failed revoke.
    class PretendingStore:
        def __init__(self, real: AuthStore) -> None:
            self.real = real

        def list_credentials(
            self, provider: str | None = None, include_disabled: bool = False
        ) -> list[Any]:
            return self.real.list_credentials(provider, include_disabled)

        def delete_credential(self, credential_id: int) -> None:
            return None  # accepted, deleted nothing

    with pytest.raises(TicketStoreError, match="still present"):
        delete_ticket(PretendingStore(store))
    # And the real row is genuinely still there, which is why it must raise.
    assert len(store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER)) == 1


def test_set_never_reports_zero_characters_after_a_successful_write(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "(0 characters)" reads as "nothing stored" on the secret-handling command."""

    class PipedStdin:
        def isatty(self) -> bool:
            return False

        def read(self) -> str:
            return FAKE_TICKET

    monkeypatch.setattr("sys.stdin", PipedStdin())
    # Write succeeds, then the confirming read-back fails.
    real_list = store.list_credentials
    calls = {"n": 0}

    def flaky_list(*args: Any, **kwargs: Any) -> list[Any]:
        # `>= 1`, not `> 1`: the `set` path makes exactly ONE
        # `list_credentials` call (the confirming read-back), so a fault armed
        # on the second call never fires and the test passes whether or not
        # the production fix is present.
        calls["n"] += 1
        if calls["n"] >= 1:
            raise sqlite3.OperationalError("database is locked")
        return real_list(*args, **kwargs)

    monkeypatch.setattr(store, "list_credentials", flaky_list)
    assert _qwencloud_ticket_action("set", store) == 0
    out = capsys.readouterr().out
    assert f"({len(FAKE_TICKET)} characters)" in out
    assert "(0 characters)" not in out
    assert FAKE_TICKET not in out


def _disable_the_row(store: AuthStore) -> None:
    """Soft-delete the ticket row the way `disable_credential` would."""
    rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)
    assert len(rows) == 1
    store.disable_credential(rows[0].id, "invalidated-token")


def test_a_soft_deleted_row_is_still_visible_and_removable(store: AuthStore) -> None:
    """A disabled row still holds the plaintext cookie, so it must not hide.

    `list_credentials` filters `disabled_cause is None` by default
    (auth_store.py:578), which made `rm` answer "No ... ticket stored." with
    exit 0 while the credential was on disk.
    """
    store_ticket(store, FAKE_TICKET)
    _disable_the_row(store)

    record = read_ticket_record(store)
    assert record is not None, "a disabled row still contains the cookie"
    assert record["length"] == len(FAKE_TICKET)

    assert delete_ticket(store) is True
    assert store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True) == []


def test_rm_removes_a_soft_deleted_row_rather_than_reporting_none(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store_ticket(store, FAKE_TICKET)
    _disable_the_row(store)

    assert _qwencloud_ticket_action("rm", store) == 0
    out = capsys.readouterr().out
    assert "Removed the stored QwenCloud console ticket." in out
    assert "No QwenCloud console ticket stored." not in out
    assert store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True) == []


def test_rm_points_at_the_console_on_the_SUCCESS_path(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Success is when a user worried about exposure stops looking.

    Deleting the row ends local use; the browser session stays valid until it
    is signed out server-side.
    """
    store_ticket(store, FAKE_TICKET)
    assert _qwencloud_ticket_action("rm", store) == 0
    out = capsys.readouterr().out
    assert "QwenCloud console" in out
    assert "still valid" in out
    assert FAKE_TICKET not in out


def test_a_store_reporting_a_none_path_is_refused(store: AuthStore) -> None:
    """A path that cannot be checked must not be treated as checked."""

    class PathlessStore:
        db_path = None

        def upsert_credential(self, provider: str, credential: dict[str, Any]) -> None:
            raise AssertionError("must not write when the path is unknown")

    with pytest.raises(TicketStoreError):
        store_ticket(PathlessStore(), FAKE_TICKET)


def test_an_empty_ticket_is_refused(store: AuthStore) -> None:
    with pytest.raises(TicketStoreError):
        store_ticket(store, "   ")
    assert store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER) == []


def test_read_ticket_record_never_returns_the_value(store: AuthStore) -> None:
    store_ticket(store, FAKE_TICKET)
    record = read_ticket_record(store)
    assert record is not None
    assert record["length"] == len(FAKE_TICKET)
    assert FAKE_TICKET not in json.dumps(record)


def _run(monkeypatch: pytest.MonkeyPatch, store: AuthStore, command: str | None) -> int:
    """Drive one CLI verb against the tmp_path store, never the real one.

    Calls the action seam rather than `qwencloud_ticket_command`, because the
    command owns the store's lifetime and closes it on the way out — which
    would leave the assertions after the call reading a closed database. The
    closing behaviour itself is asserted in
    `test_the_command_closes_the_store_it_opened`.
    """
    del monkeypatch  # kept for signature symmetry with the patched-stdin tests
    return _qwencloud_ticket_action(command, store)


def test_the_command_closes_the_store_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CLI verb must not leak the SQLite connection (or the usage cache's).

    Every sibling command in cli.py wraps the store in try/finally; this is the
    regression for that discipline.
    """
    opened = AuthStore(db_path=tmp_path / "auth.db")
    closed: list[bool] = []
    real_close = opened.close

    def recording_close() -> None:
        closed.append(True)
        real_close()

    monkeypatch.setattr(opened, "close", recording_close)
    monkeypatch.setattr("local_operator.providers.auth_store.AuthStore", lambda *a, **k: opened)

    assert qwencloud_ticket_command(argparse.Namespace(qwencloud_command="status")) == 0
    capsys.readouterr()
    assert closed == [True], "the command must close the store it opened"


def test_the_store_is_closed_even_when_the_verb_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`finally`, not a trailing call: an unexpected error must still release it."""
    opened = AuthStore(db_path=tmp_path / "auth.db")
    closed: list[bool] = []
    real_close = opened.close

    def recording_close() -> None:
        closed.append(True)
        real_close()

    monkeypatch.setattr(opened, "close", recording_close)
    monkeypatch.setattr("local_operator.providers.auth_store.AuthStore", lambda *a, **k: opened)
    monkeypatch.setattr(
        "local_operator.cli._qwencloud_ticket_action",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        qwencloud_ticket_command(argparse.Namespace(qwencloud_command="status"))
    assert closed == [True]


def test_status_never_prints_the_value(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store_ticket(store, FAKE_TICKET)
    assert _run(monkeypatch, store, "status") == 0
    captured = capsys.readouterr()
    assert FAKE_TICKET not in captured.out
    assert FAKE_TICKET not in captured.err
    assert str(len(FAKE_TICKET)) in captured.out


def test_status_reports_an_empty_store(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(monkeypatch, store, "status") == 0
    assert "No QwenCloud console ticket stored." in capsys.readouterr().out


def test_status_warns_when_the_ticket_is_stale(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = int(time.time() * 1000) - QWENCLOUD_TICKET_STALE_MS - 86_400_000
    store_ticket(store, FAKE_TICKET, now_ms=stale)
    assert _run(monkeypatch, store, "status") == 0
    out = capsys.readouterr().out
    assert "older than a console session usually lasts" in out
    assert FAKE_TICKET not in out


def test_status_does_not_warn_on_a_fresh_ticket(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store_ticket(store, FAKE_TICKET)
    assert _run(monkeypatch, store, "status") == 0
    assert "older than a console session usually lasts" not in capsys.readouterr().out


def test_set_refuses_a_tty(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class TtyStdin:
        def isatty(self) -> bool:
            return True

        def read(self) -> str:  # pragma: no cover - must never be reached
            raise AssertionError("set must not read a tty")

    monkeypatch.setattr("sys.stdin", TtyStdin())
    assert _run(monkeypatch, store, "set") == 2
    assert store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER) == []
    assert "never from the command line" in capsys.readouterr().err


def test_set_reads_stdin_and_strips_one_trailing_newline(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class PipedStdin:
        def isatty(self) -> bool:
            return False

        def read(self) -> str:
            return FAKE_TICKET + "\n"

    monkeypatch.setattr("sys.stdin", PipedStdin())
    assert _run(monkeypatch, store, "set") == 0
    assert _persisted_data(store)["ticket"] == FAKE_TICKET
    out = capsys.readouterr().out
    assert f"({len(FAKE_TICKET)} characters)" in out
    assert FAKE_TICKET not in out


def test_rm_removes_the_row(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store_ticket(store, FAKE_TICKET)
    assert _run(monkeypatch, store, "rm") == 0
    assert "Removed the stored QwenCloud console ticket." in capsys.readouterr().out
    assert store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER) == []


def test_rm_is_a_clear_no_op_when_nothing_is_stored(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(monkeypatch, store, "rm") == 0
    assert "No QwenCloud console ticket stored." in capsys.readouterr().out
    assert delete_ticket(store) is False


def test_no_subcommand_prints_usage_and_exits_two(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(monkeypatch, store, None) == 2
    assert "usage: lop qwencloud-ticket {set,status,rm}" in capsys.readouterr().err
