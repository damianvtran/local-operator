"""Shared CLI/TUI semantic output; rendered-frame approval remains separate."""

import pytest

from local_operator.providers.auth_cli import format_logins, list_logins
from local_operator.providers.auth_store import AuthStore
from tests.unit.providers import test_radient_account_retention as _retention
from tests.unit.providers.test_radient_account_retention import _login, _stored

store = _retention.store


def _retained(store: AuthStore) -> int:
    row = store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "do-not-print-access",
            "refresh": "do-not-print-refresh",
            "email": "unverified-label@example.test",
            "expires": 9999999999999,
        },
    )
    store.disable_credential(row.id, "identity_unattributed")
    return row.id


@pytest.mark.parametrize("known_accounts", [(), ("a",), ("a", "b")])
def test_retained_status_is_distinct_from_usable_selection(
    store: AuthStore, known_accounts
) -> None:
    retained_id = _retained(store)
    for account in known_accounts:
        store.upsert_credential("radient", _login(account))
    before = store.list_credentials("radient", include_disabled=True)
    text = "\n".join(format_logins(store))
    assert "Retained Radient sign-ins (unavailable):" in text
    assert f"[{retained_id}] Account could not be verified" in text
    assert "local-operator login radient" in text
    assert "/login radient" in text
    assert "not been deleted or assigned" in text
    assert "do-not-print" not in text
    assert "unverified-label" not in text
    assert "you disabled" not in text.lower()
    if not known_accounts:
        assert "No usable stored credentials." in text
        assert "No stored credentials." not in text
    assert all(row.id != retained_id for row in store.list_credentials("radient"))
    assert store.list_credentials("radient", include_disabled=True) == before


def test_real_cli_status_prints_retention_and_recovery_without_writes(
    store: AuthStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    retained_id = _retained(store)
    assert list_logins(store) == 0
    output = capsys.readouterr().out
    assert f"[{retained_id}]" in output
    assert "Account could not be verified" in output
    assert "local-operator login radient" in output
    assert _stored(store, retained_id).disabled_cause == "identity_unattributed"


def test_explicit_disable_is_not_mislabeled_as_automatic_quarantine(store: AuthStore) -> None:
    row = store.upsert_credential("radient", _login("known"))
    store.disable_credential(row.id, "operator_disabled")
    assert store.retained_radient_login_ids() == ()
    assert "Account could not be verified" not in "\n".join(format_logins(store))
    assert _stored(store, row.id).disabled_cause == "operator_disabled"


def test_retained_status_reopens_read_only_and_never_revives(store: AuthStore) -> None:
    retained_id = _retained(store)
    path = store.db_path
    store.close()
    reopened = AuthStore.open_read_only(path)
    try:
        assert reopened.retained_radient_login_ids() == (retained_id,)
        assert "Account could not be verified" in "\n".join(format_logins(reopened))
        assert reopened.list_credentials("radient") == []
    finally:
        reopened.close()
