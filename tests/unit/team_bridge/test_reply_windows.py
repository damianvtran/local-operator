"""Reply paging negotiation must not synthesize an incompatible after=0."""

import pytest

from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.types import reply_window_params


@pytest.mark.parametrize("query", [
    {"before_seq": 0}, {"after_seq": -1}, {"before_seq": True},
    {"order": "DESC"}, {"order": ""}, {"before_seq": "2.0"},
    {"before_seq": 3, "after_seq": 0}, {"before_seq": 3, "order": "asc"},
    {"after_seq": 0, "order": "desc"}, {"limit": 101}, {"limit": 0},
    {"before_seq": "9" * 501},
])
def test_reply_window_rejects_invalid_or_conflicting_modes(query):
    with pytest.raises(ValueError):
        reply_window_params(**query)


def test_reply_window_preserves_legacy_and_explicit_empty_mode():
    assert reply_window_params() == {}
    assert reply_window_params(after_seq="0") == {"after_seq": 0}
    assert reply_window_params(order="desc") == {"order": "desc"}
    assert reply_window_params(before_seq="51", order="desc", limit="50") == {
        "before_seq": 51, "order": "desc", "limit": 50,
    }


@pytest.mark.parametrize("method", ["conversation_replies", "owner_audit_replies"])
@pytest.mark.asyncio
async def test_reply_client_serializes_only_explicit_cursors(method, monkeypatch):
    # Serialization-only: no config/credential lookup or transport is involved.
    client = object.__new__(TeamBridgeClient)
    captured = []

    async def send(verb, path, **kwargs):
        captured.append((verb, path, kwargs))
        return {"root": {}, "replies": [], "has_more": False, "order": "desc"}

    monkeypatch.setattr(client, "_send", send)
    call = getattr(client, method)
    owner = {"expected_account_id": "acc_1", "tenant_id": "tnt_1"}
    result = await call("bcv_abc12345", "bcmsg_root0001", order="desc", **owner)
    assert result["order"] == "desc"  # empty/terminal explicit echo survives
    assert captured[-1][2]["params"] == {"tenant_id": "tnt_1", "order": "desc"}
    await call("bcv_abc12345", "bcmsg_root0001", before_seq=51, order="desc", **owner)
    assert captured[-1][2]["params"] == {"tenant_id": "tnt_1", "before_seq": 51, "order": "desc"}
    await call("bcv_abc12345", "bcmsg_root0001", **owner)
    assert captured[-1][2]["params"] == {"tenant_id": "tnt_1"}
    count = len(captured)
    with pytest.raises(ValueError):
        await call("bcv_abc12345", "bcmsg_root0001", after_seq=0, order="desc", **owner)
    assert len(captured) == count


def test_reply_window_keyword_types_are_per_key_and_optional():
    # A homogeneous dictionary passed runtime tests but broke both typed HTTP
    # dispatches. Keep the public helper contract precise without casts/ignores.
    from typing import Literal, get_type_hints, is_typeddict

    result_type = get_type_hints(reply_window_params)["return"]
    assert is_typeddict(result_type)
    assert get_type_hints(result_type) == {
        "limit": int, "after_seq": int, "before_seq": int,
        "order": Literal["asc", "desc"],
    }
    assert result_type.__required_keys__ == frozenset()
    assert result_type.__optional_keys__ == frozenset({"limit", "after_seq", "before_seq", "order"})
