"""The session arm's pre-launch resolution SHAPE: one budgeted store call.

WHY THIS FILE EXISTS. The resolution budget lives on
``CredentialStoreResolver.resolve``, but the session arm reaches the store
through ``scripts.run_episode._LayeredResolver`` -- which used to fan out one
single-name ``resolve`` per ref, so every deadline check saw ~0 elapsed and
the bound could never fire on the very shape it was written for. Reviewer
repro, 2026-10-17: three store refs x 30 ms each against a 50 ms budget
produced NO timeout through the layered resolver while the direct call raised
at 2/3; in the real shape a broker that accepts and never answers costs
``REQUEST_TIMEOUT_S`` (10 s) per ref, i.e. ~180 s of boot silence on an 18-ref
episode with the budget asleep. These tests drive the LAYERED resolver itself
(the direct-call tests in ``test_secrets.py`` could not catch this), plus the
two Script seams the fix depends on: the owner env must carry the knob, and
the diagnostic classification must treat the timeout as a pre-allocation
secret refusal (exit 2), like a missing ref.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.runner.secrets import MissingSecret
from scripts import run_episode


def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point HOME and the config dir inside ``tmp_path``, and open the store."""
    from local_operator.secrets.access import open_store

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "home").mkdir()
    (tmp_path / "config").mkdir()
    return open_store(create=True)


def test_the_layered_resolver_makes_one_budgeted_store_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K store refs cost ONE resolve call, so the deadline's unit is the boot.

    The stalled-store shape: with the batched call the budget fires (named,
    bounded); had the resolver kept fanned-out single-name calls, each check
    would have seen ~0 elapsed and this test's ``pytest.raises`` would never
    see the timeout -- the exact regression the reviewer drove.
    """
    import local_operator.secrets.access as access
    from local_operator.evaluation.runner.host_secrets import (
        CredentialStoreResolver,
        SecretResolutionTimeout,
    )

    store = _isolated_store(tmp_path, monkeypatch)
    slow_names = ("SLOW_ONE", "SLOW_TWO", "SLOW_THREE")
    for name in slow_names:
        store.set(name, b"canary-value")

    real_retrieve = access.retrieve_secret

    def slow(name: str, base: Path | None = None, *, role: str = "agent") -> bytes:
        time.sleep(0.03)
        return real_retrieve(name, base, role=role)

    monkeypatch.setattr(access, "retrieve_secret", slow)
    monkeypatch.setenv("LOCAL_OPERATOR_SECRET_RESOLUTION_TIMEOUT_MS", "50")

    layered = run_episode._LayeredResolver((), CredentialStoreResolver(tmp_path / "config"))
    started = time.monotonic()
    with pytest.raises(SecretResolutionTimeout) as raised:
        layered.resolve(list(slow_names))
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, "the budget must fail far sooner than a hang would"
    error = raised.value
    assert not isinstance(error, MissingSecret)
    assert error.name in slow_names
    assert error.position in (2, 3) and error.total == 3


def test_the_layered_resolver_batches_and_keeps_request_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One store fetch per distinct store ref, values emitted per occurrence.

    Order and duplication are observable to the caller (the arm's
    ``spec.secret_refs`` order), so batching must not reorder or drop: the
    env-side refs still resolve at their own positions and a repeated ref
    still yields one ResolvedSecret per appearance -- while the STORE sees
    each distinct name exactly once, in first-appearance order.
    """
    import local_operator.secrets.access as access
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    store = _isolated_store(tmp_path, monkeypatch)
    store.set("STORE_ONE", b"store-value-1")
    store.set("STORE_TWO", b"store-value-2")

    fetched: list[str] = []
    real_retrieve = access.retrieve_secret

    def watching(name: str, base: Path | None = None, *, role: str = "agent") -> bytes:
        fetched.append(name)
        return real_retrieve(name, base, role=role)

    monkeypatch.setattr(access, "retrieve_secret", watching)
    monkeypatch.setenv("FROM_ENV", "env-value")

    layered = run_episode._LayeredResolver(
        ("FROM_ENV",), CredentialStoreResolver(tmp_path / "config")
    )
    resolved: tuple[Any, ...] = layered.resolve(
        ["FROM_ENV", "STORE_ONE", "FROM_ENV", "STORE_TWO", "STORE_ONE"]
    )
    assert [(secret.name, secret.value) for secret in resolved] == [
        ("FROM_ENV", "env-value"),
        ("STORE_ONE", "store-value-1"),
        ("FROM_ENV", "env-value"),
        ("STORE_TWO", "store-value-2"),
        ("STORE_ONE", "store-value-1"),
    ]
    assert fetched == ["STORE_ONE", "STORE_TWO"]


def test_the_resolution_knob_rides_the_owner_env_allowlist() -> None:
    """The knob must reach the OWNER, because the owner is its only reader.

    The resolver executes inside the managed owner, which is spawned with
    ``_owner_env_from_self()``'s closed list; the refusal sentence names the
    knob as the lever, so exporting it must not be a silent no-op there
    (review MAJOR-2, 2026-10-17: it was, before this list carried it).
    """
    from local_operator.evaluation.runner import host_secrets

    assert host_secrets.RESOLUTION_TIMEOUT_ENV in run_episode._OWNER_ENV_NAMES


def test_the_script_classifies_the_timeout_as_a_pre_allocation_secret_refusal() -> None:
    """``SecretResolutionTimeout`` exits 2, exactly like a missing ref.

    ``_diagnostic`` renders the class name as the prefix, so the tuple below
    is what routes the failure to ``EXIT_PREFLIGHT`` instead of falling
    through to ``EXIT_EPISODE`` on the reply arm (review MINOR-1, 2026-10-17).
    """
    assert "SecretResolutionTimeout:" in run_episode._SECRET_DIAGNOSTICS
