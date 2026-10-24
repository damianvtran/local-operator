"""A catalogue-only fallback cannot imply a completed login or readable history."""

import pytest

from local_operator.mobile.daemon import _UnreadableAuthStore
from local_operator.providers.controller import ControllerAuthStore


@pytest.mark.parametrize("operation", ["complete_radient_login", "retained_radient_login_ids"])
def test_unreadable_store_refuses_radient_operations(operation: str) -> None:
    # Keep the structural assignment: new protocol members must be implemented,
    # not hidden by a cast or a success-shaped empty history result.
    store: ControllerAuthStore = _UnreadableAuthStore()
    assert store.list_credentials("radient") == []
    with pytest.raises(RuntimeError, match=f"store is unreadable; {operation} is not available"):
        if operation == "complete_radient_login":
            store.complete_radient_login({"type": "oauth"})
        else:
            store.retained_radient_login_ids()
