"""The memo behind ``_unusable_credential``, and the one way it could do harm.

Diagnosing WHY nothing was usable costs a token-endpoint POST per stored OAuth row,
so the verdict is remembered for the store's own block window (see
``DIAGNOSIS_TTL_S``). A memo of a "your grant is dead" answer is only safe if it
cannot outlive the ROW it was made about: a user who signs in again, or a refresh a
peer lands, changes that row, and replaying the old verdict over the new credential
would be exactly the failure the diagnosis exists to prevent. These tests pin the
key that guarantees it — a real ``AuthStore`` per test, because the key is composed
from the store's own file and its rows' ``updated_at``, and a stand-in for either
would be pinning the stand-in.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.auth_store import AuthStore, StoredCredential
from local_operator.server.routes import desktop_radient


def row(credential_id: int, updated_at: int) -> StoredCredential:
    """A stored OAuth row, with only the fields the key reads made meaningful."""
    return StoredCredential(
        id=credential_id,
        provider="radient",
        credential_type="oauth",
        data={"access": "a", "refresh": "r"},
        updated_at=updated_at,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[AuthStore]:
    made = AuthStore(tmp_path / "auth.db")
    try:
        yield made
    finally:
        made.close()


@pytest.fixture(autouse=True)
def empty_memo() -> Iterator[None]:
    """Each test starts from no remembered verdict, and leaves none behind."""
    desktop_radient._DIAGNOSIS.clear()
    yield
    desktop_radient._DIAGNOSIS.clear()


def test_a_verdict_is_reused_while_the_rows_it_was_made_about_are_unchanged(
    store: AuthStore,
) -> None:
    rows = [row(1, 100)]
    desktop_radient._remember_diagnosis(store, rows, desktop_radient.REASON_GRANT_INVALID)
    assert (
        desktop_radient._remembered_diagnosis(store, rows) == desktop_radient.REASON_GRANT_INVALID
    )


def test_a_row_that_changed_is_diagnosed_again(store: AuthStore) -> None:
    """The re-login case: same id, new ``updated_at`` ⇒ a different key.

    ``updated_at`` is what the store writes on every change to a row, so this is the
    guarantee that a fresh credential's sign-in is served rather than refused off the
    previous grant's verdict.
    """
    desktop_radient._remember_diagnosis(store, [row(1, 100)], desktop_radient.REASON_GRANT_INVALID)
    assert desktop_radient._remembered_diagnosis(store, [row(1, 101)]) is None


def test_a_different_stores_rows_are_not_the_same_rows(tmp_path: Path) -> None:
    """Two daemons share this process, and every config dir restarts ids at 1."""
    one = AuthStore(tmp_path / "one" / "auth.db")
    two = AuthStore(tmp_path / "two" / "auth.db")
    try:
        desktop_radient._remember_diagnosis(
            one, [row(1, 100)], desktop_radient.REASON_GRANT_INVALID
        )
        assert desktop_radient._remembered_diagnosis(two, [row(1, 100)]) is None
    finally:
        one.close()
        two.close()


def test_a_verdict_expires_with_the_store_block_window(
    store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded in time as well as in scope: a transient verdict must not persist.

    A ``credential_unavailable`` answer is a statement about RIGHT NOW, so it is held
    no longer than the window the failed refresh has the row out of rotation — after
    that the next request asks the store again.
    """
    monkeypatch.setattr(desktop_radient, "DIAGNOSIS_TTL_S", 0.0)
    rows = [row(1, 100)]
    desktop_radient._remember_diagnosis(store, rows, desktop_radient.REASON_UNAVAILABLE)
    assert desktop_radient._remembered_diagnosis(store, rows) is None


def test_a_remembered_verdict_maps_onto_the_class_a_client_reads(store: AuthStore) -> None:
    """The memo stores a ``reason``, so a reason has to still name its own class."""
    rows = [row(1, 100)]
    desktop_radient._remember_diagnosis(store, rows, desktop_radient.REASON_UNAVAILABLE)
    reason = desktop_radient._remembered_diagnosis(store, rows)
    assert reason is not None
    failure = desktop_radient._credential_refusal(reason)
    # `HTTPException.detail` is typed as the sentence it carries for the plane's OWN
    # refusals; this route puts the structured payload there instead, because that is
    # the shape the renderer classifies on. So read it as the payload it is.
    payload: Any = failure.detail
    assert failure.status_code == 502
    assert payload["code"] == "radient_upstream_failed"
    assert payload["details"]["reason"] == desktop_radient.REASON_UNAVAILABLE
