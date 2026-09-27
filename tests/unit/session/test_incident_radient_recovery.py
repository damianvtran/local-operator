"""The Radient usage-limit remedy must reach the DESKTOP's incident text.

WHY THIS FILE. The desktop renders a failed turn as the ``[session incident …]``
row ``journal_incident`` writes, so guidance the user must see has to land in
THAT string — not only in the TUI's error line. These tests drive the real
``journal_incident`` with an injected store and probe seam (no network, no
operator credentials) and pin the gates: a non-Radient provider, a non-quota
failure and a caller-supplied ``rendered`` text all stay exactly as they were.

The append itself (branches, cache, tolerance) is covered by
``tests/unit/providers/test_radient_recovery.py``; this file is about the
WIRING — that the incident the desktop reads is the string that gains the line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.message_types import SESSION_INCIDENT_MESSAGE_TYPE
from local_operator.providers import radient_recovery as rr
from local_operator.providers.auth_store import AuthStore

from .test_session import MODEL, ScriptedStream, make_session

RENDERED_QUOTA = "rate limit or quota exceeded (HTTP 402): insufficient credits"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No test may read the operator's store or environment for a key."""
    import local_operator.providers.registry as registry

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    monkeypatch.setattr(registry, "provider_secret_value", lambda *args, **kwargs: None)
    rr.reset_recovery_cache()
    yield
    rr.reset_recovery_cache()


def _arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A signed-in store plus a probe seam answering the pending-grant payload."""
    import local_operator.providers.auth_store as auth_store_module

    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("radient", {"type": "oauth", "access": "tok-1", "refresh": "r"})
    monkeypatch.setattr(auth_store_module, "shared_auth_store", lambda *args, **kwargs: store)
    verification = rr.parse_verification({"signup_grant": "pending", "grant_amount": 5})

    async def probe(token: str):
        return verification

    monkeypatch.setattr(rr, "_probe_verification_async", probe)


def _last_incident_text(session_dir: Path) -> str:
    lines = (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])["payload"]
    assert payload.get("custom_type") == SESSION_INCIDENT_MESSAGE_TYPE
    details: dict[str, Any] = payload["details"]
    return str(details["text"])


async def _journal(tmp_path: Path, raw: str, *, provider: str = "radient", rendered: str = ""):
    model = MODEL.model_copy(update={"provider": provider})
    session = make_session(tmp_path, ScriptedStream([]), model=model)
    try:
        await session.journal_incident(raw, rendered=rendered)
    finally:
        await session.dispose()
    return _last_incident_text(tmp_path / "sess")


@pytest.mark.asyncio
async def test_radient_quota_incident_carries_the_recovery_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm(monkeypatch, tmp_path)

    text = await _journal(tmp_path, RENDERED_QUOTA)

    # The desktop's row keeps the incident's own shape AND gains the remedy.
    assert text.startswith("[session incident (radient/")
    assert "rate limit or quota exceeded (HTTP 402)" in text
    assert "check your email" in text and "$5.00" in text


@pytest.mark.asyncio
async def test_a_non_quota_failure_stays_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm(monkeypatch, tmp_path)

    text = await _journal(tmp_path, "authentication failed (HTTP 401): bad key")

    assert "check your email" not in text
    assert "console.radienthq.com" not in text


@pytest.mark.asyncio
async def test_a_non_radient_provider_stays_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _arm(monkeypatch, tmp_path)

    text = await _journal(tmp_path, RENDERED_QUOTA, provider="openai")

    assert "check your email" not in text
    assert "console.radienthq.com" not in text


@pytest.mark.asyncio
async def test_a_harness_authored_rendered_text_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rendered`` callers (cut-offs, lost deliveries) are not provider errors."""
    _arm(monkeypatch, tmp_path)

    text = await _journal(tmp_path, RENDERED_QUOTA, rendered="[session incident] cut off")

    assert text == "[session incident] cut off"


@pytest.mark.asyncio
async def test_the_line_is_not_duplicated_when_the_text_already_carries_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The append is idempotent — a replayed journal cannot stack a second remedy."""
    _arm(monkeypatch, tmp_path)
    first = await _journal(tmp_path, RENDERED_QUOTA)
    delivered = first + "\n" + rr.recovery_line(rr.RecoveryFacts(signed_in=True))

    assert rr.append_recovery_line_once(delivered, rr._GENERIC_LINE) == delivered
