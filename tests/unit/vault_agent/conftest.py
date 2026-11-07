"""Shared stubs for the Vault agent arm: a recording transport, an open gate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_operator.vault_agent import eligibility, keys

#: The authority vector's public synthetic seed (00..1f); never a credential.
SEED = bytes(range(32))
TENANT = "tnt_a"
#: Mixed case on purpose (H8): the endpoint is signed and sent byte for byte.
ENDPOINT = "bep_Raw-Mixed_123"
VAULT = "vlt_a"
ITEM = "vit_a"
LOCAL_CHALLENGE = "vch_" + "1" * 64
AUTHORITY_CHALLENGE = "vch_" + "2" * 32


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = self.content.decode()
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


@dataclass
class Transport:
    """``requests.request``-shaped; answers from a queue and records each call."""

    answers: list[tuple[int, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.answers:
            raise AssertionError(f"unexpected call {method} {url}")
        status, body = self.answers.pop(0)
        return FakeResponse(status, body)


def make_client(transport: Transport):
    from local_operator.vault_agent.client import VaultClient

    return VaultClient(
        base_url="https://api.example.test/v1",
        bearer_headers={"Authorization": "Bearer person-token"},
        secrets=["person-token"],
        request=transport,
    )


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


@pytest.fixture
def open_gate(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> eligibility.Gate:
    """An open gate, a stored seed and a live cached grant for it."""

    gate = eligibility.Gate(tenant_id=TENANT, endpoint_id=ENDPOINT, tenant_name="Acme")
    monkeypatch.setattr(eligibility, "gate", lambda _root: gate)
    monkeypatch.setattr(
        keys, "load_seed", lambda endpoint, base=None: SEED if endpoint == ENDPOINT else None
    )
    eligibility.write_grant(
        config_dir,
        eligibility.Grant(
            delegation_id="vdl_a",
            expires_at="2999-01-01T00:00:00Z",
            tenant_id=TENANT,
            endpoint_id=ENDPOINT,
            key_fingerprint=keys.fingerprint_for_seed(SEED),
        ),
    )
    return gate


def mint_answer(**overrides: Any) -> dict[str, Any]:
    answer = {
        "challenge_id": LOCAL_CHALLENGE,
        "authority_challenge_id": AUTHORITY_CHALLENGE,
        "op": "resolve",
        "tenant_id": TENANT,
        "ledger_version": str((1 << 53) + 1),
        "server_time": "2026-11-07T00:00:00Z",
        "expires_at": "2026-11-07T00:02:00Z",
    }
    answer.update(overrides)
    return answer
