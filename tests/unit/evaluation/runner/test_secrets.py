"""Secret resolution in the runner: order, redaction, delivery, and the inbox.

The properties here are what make a paid episode safe to run: a secret is
resolved AFTER the handshake and BEFORE the writer opens (so every byte the
bundle ever holds is canaried against it), it reaches the worker only on
``reset_start`` and ``begin_rescue``, a missing one fails before anything is
allocated and names only the ref, and a clean episode retires its own
rescue descriptor so the sweep's inbox is exact.

Every test drives the REAL ``VerifiedAdapterSession``, the REAL lifecycle
authorities and a REAL ``EvidenceWriter`` on ``tmp_path`` (see
``conftest.FakeAdapter``); only the subprocess boundary is faked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import ResolvedSecret, SecretRef
from local_operator.evaluation.adapters.supervisor import (
    SupervisionError,
    load_pending_rescue,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import EpisodeRunner
from local_operator.evaluation.runner.secrets import (
    EnvSecretResolver,
    MissingSecret,
    SecretResolver,
    StaticSecretResolver,
)
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    ScriptedModel,
    build_config,
    build_spec,
    selector,
)

# Long enough that every encoded variant (base64, hex, percent) is generated
# by RedactionSet; a value under 8 bytes only gets the plaintext canary.
CANARY = "canary-secret-value-9f8e7d6c5b4a"
KEY_ID = "AKIACANARY0000000001"
# A StrictIdentifier-shaped value so it can ride a ``stop_reason`` field.
SHORT_CANARY = "canary-stop-reason-value-1234"


class _Aggregate:
    def __init__(self, descriptor: Any, complete: bool) -> None:
        self.complete = complete
        self.descriptor_id = descriptor.descriptor_id
        self.receipts = ()


def _rescue_returning(complete: bool) -> Any:
    async def rescue(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        return _Aggregate(descriptor, complete)

    return rescue


def _spec_with_refs(episode_id: str, *names: str) -> Any:
    spec = build_spec(episode_id)
    object.__setattr__(spec, "secret_refs", tuple(SecretRef(name=name) for name in names))
    return spec


def _all_bytes_under(root: Path) -> bytes:
    """Every file under ``root``, concatenated: what a leak would have to hide in."""

    chunks: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            chunks.append(path.read_bytes())
    return b"\n".join(chunks)


# ---------------------------------------------------------------------------
# resolvers
# ---------------------------------------------------------------------------


def test_static_and_env_resolvers_return_in_request_order() -> None:
    static = StaticSecretResolver({"B": "2", "A": "1"})
    assert [s.name for s in static.resolve(["A", "B"])] == ["A", "B"]
    env = EnvSecretResolver({"A": "1", "B": "2"})
    assert [(s.name, s.value) for s in env.resolve(["B", "A"])] == [("B", "2"), ("A", "1")]
    assert isinstance(static, SecretResolver) and isinstance(env, SecretResolver)


def test_env_resolver_reads_only_the_mapping_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never ``os.environ`` implicitly: an ambient variable is invisible."""

    monkeypatch.setenv("AMBIENT_SECRET", "should-not-be-seen")
    resolver = EnvSecretResolver({})
    with pytest.raises(MissingSecret) as raised:
        resolver.resolve(["AMBIENT_SECRET"])
    assert raised.value.name == "AMBIENT_SECRET"
    assert "should-not-be-seen" not in str(raised.value)
    # And an explicit snapshot that includes it does see it.
    assert EnvSecretResolver(dict(os.environ)).resolve(["AMBIENT_SECRET"])[0].value == (
        "should-not-be-seen"
    )


@pytest.mark.parametrize("value", ["", None])
def test_empty_value_is_a_missing_secret(value: str | None) -> None:
    resolver = StaticSecretResolver({"K": value} if value is not None else {})
    with pytest.raises(MissingSecret, match="missing secret K"):
        resolver.resolve(["K"])


def test_credential_store_resolver_wraps_the_manager_and_names_only_the_ref() -> None:
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    class _Secret:
        def __init__(self, value: str) -> None:
            self._value = value

        def get_secret_value(self) -> str:
            return self._value

    class _Manager:
        def get_credential(self, key: str) -> _Secret:
            if key == "BOOM":
                raise RuntimeError("store exploded reading /path/credentials.env")
            return _Secret({"PRESENT": CANARY}.get(key, ""))

    resolver = CredentialStoreResolver(_Manager())
    assert resolver.resolve(["PRESENT"]) == (ResolvedSecret(name="PRESENT", value=CANARY),)
    with pytest.raises(MissingSecret) as missing:
        resolver.resolve(["ABSENT"])
    assert missing.value.name == "ABSENT"
    with pytest.raises(MissingSecret) as errored:
        resolver.resolve(["BOOM"])
    assert errored.value.name == "BOOM"
    assert "credentials.env" not in str(errored.value)


def test_redaction_set_with_values_widens_without_dropping_existing() -> None:
    base = RedactionSet.from_resolved_values(("first-canary-value",))
    widened = base.with_values([CANARY])
    for value in ("first-canary-value", CANARY):
        with pytest.raises(ValueError):
            widened.assert_clear({"x": f"...{value}..."})
    widened.assert_clear({"x": "clean"})


# ---------------------------------------------------------------------------
# runner: order and delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secrets_resolve_after_handshake_before_writer_and_ride_reset_start(
    tmp_path: Path, episode_id: str
) -> None:
    """resolve -> redactions -> writer -> reset_start(secrets=...), in that order."""

    events: list[str] = []
    adapter = FakeAdapter(tmp_path, episode_id)
    original = adapter._call_raw
    original_handshake = adapter.handshake

    async def handshake(**kwargs: Any) -> Any:
        events.append("handshake")
        return await original_handshake(**kwargs)

    adapter.handshake = handshake  # type: ignore[method-assign]
    delivered: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    async def watching(method: Any, params: Any, result_type: Any, **kwargs: Any) -> Any:
        events.append(method)
        if method == "reset_start":
            delivered.append(("reset_start", tuple((s.name, s.value) for s in params.secrets)))
        if method == "prepare":
            assert not hasattr(params, "secrets"), "prepare must stay secret-free"
        return await original(method, params, result_type, **kwargs)

    adapter._call_raw = watching  # type: ignore[method-assign]

    class _Recording:
        def resolve(self, names: Any) -> Any:
            events.append("resolve")
            return StaticSecretResolver({"AWS_SECRET_ACCESS_KEY": CANARY}).resolve(names)

    config = build_config(tmp_path)
    runner = EpisodeRunner(
        _spec_with_refs(episode_id, "AWS_SECRET_ACCESS_KEY"),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        secrets=_Recording(),
        launch=lambda _: adapter,
        rescue=_rescue_returning(True),
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert events.index("handshake") < events.index("resolve") < events.index("prepare")
    assert delivered == [("reset_start", (("AWS_SECRET_ACCESS_KEY", CANARY),))]
    # The writer opened AFTER resolution, with the value in its canary set:
    # the bundle carries the ref name (in the descriptor-free manifest paths)
    # but never the value in any encoding.
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    assert CANARY.encode() not in _all_bytes_under(tmp_path)
    assert runner._redactions is not None
    with pytest.raises(ValueError):
        runner._redactions.assert_clear({"leak": CANARY})


@pytest.mark.asyncio
async def test_value_reaching_the_bundle_is_refused_by_the_canaried_writer(
    tmp_path: Path, episode_id: str
) -> None:
    """The writer built after resolution refuses an event carrying the value."""

    class _Leaky(ScriptedModel):
        """Smuggles the value into a model_response through ``stop_reason``."""

        async def decide(self, observation: Any, history: Any) -> Any:
            decision = await super().decide(observation, history)
            return decision.model_copy(update={"stop_reason": SHORT_CANARY})

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        _spec_with_refs(episode_id, "AWS_SECRET_ACCESS_KEY"),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=_Leaky(["finish"]),
        secrets=StaticSecretResolver({"AWS_SECRET_ACCESS_KEY": SHORT_CANARY}),
        launch=lambda _: adapter,
        rescue=_rescue_returning(True),
    )

    outcome = await runner.run()

    # The writer raised on the poisoned event; the runner abandoned rather
    # than sealing a bundle with the value in it.
    assert outcome.status in ("abandoned", "abandonment_failed"), outcome.status
    assert SHORT_CANARY.encode() not in _all_bytes_under(tmp_path / "evidence")


@pytest.mark.asyncio
async def test_missing_secret_fails_before_any_allocation_and_names_only_the_ref(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    config = build_config(tmp_path)
    runner = EpisodeRunner(
        _spec_with_refs(episode_id, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        secrets=StaticSecretResolver({"AWS_ACCESS_KEY_ID": KEY_ID}),
        launch=lambda _: adapter,
    )

    outcome = await runner.run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert outcome.diagnostic == "MissingSecret: missing secret AWS_SECRET_ACCESS_KEY"
    assert KEY_ID not in (outcome.diagnostic or "")
    # Nothing was allocated: the side-effect boundary was never crossed, no
    # descriptor was persisted (so the sweep has nothing to reclaim), and the
    # worker was reaped.
    assert "prepare" not in adapter.calls and "reset_start" not in adapter.calls
    assert load_pending_rescue(config.rescue_root) is None
    assert adapter.terminated
    assert KEY_ID.encode() not in _all_bytes_under(tmp_path)


@pytest.mark.asyncio
async def test_spec_with_refs_and_no_resolver_fails_closed(tmp_path: Path, episode_id: str) -> None:
    """``secrets=None`` is only right for a spec with no refs."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        _spec_with_refs(episode_id, "AWS_SECRET_ACCESS_KEY"),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
    )
    outcome = await runner.run()
    assert outcome.status == "failed_pre_bundle"
    assert "AWS_SECRET_ACCESS_KEY" in (outcome.diagnostic or "")


# ---------------------------------------------------------------------------
# runner: rescue and the inbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rescue_receives_the_resolved_secrets(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"execute": SupervisionError("worker died")}
    )
    seen: list[tuple[str, ...]] = []

    async def rescue(descriptor: Any, **kwargs: Any) -> Any:
        seen.append(tuple(f"{s.name}={s.value}" for s in kwargs["secrets"]))
        return _Aggregate(descriptor, True)

    runner = EpisodeRunner(
        _spec_with_refs(episode_id, "AWS_SECRET_ACCESS_KEY"),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "finish"]),
        secrets=StaticSecretResolver({"AWS_SECRET_ACCESS_KEY": CANARY}),
        launch=lambda _: adapter,
        rescue=rescue,
    )

    outcome = await runner.run()

    assert outcome.rescue_required is True and outcome.rescue_complete is True
    assert seen == [(f"AWS_SECRET_ACCESS_KEY={CANARY}",)]
    assert CANARY.encode() not in _all_bytes_under(tmp_path / "evidence")
    assert CANARY.encode() not in _all_bytes_under(tmp_path / "rescue")


@pytest.mark.asyncio
async def test_clean_episode_retires_its_descriptor(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    config = build_config(tmp_path)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_returning(True),
    )
    outcome = await runner.run()
    assert outcome.status == "completed"
    assert outcome.rescue_required is False
    assert not (config.rescue_root / "rescue.json").exists()


@pytest.mark.asyncio
async def test_incomplete_cleanup_keeps_the_descriptor_when_rescue_cannot_confirm(
    tmp_path: Path, episode_id: str
) -> None:
    """Only a confirmation retires the inbox entry; ``attempted`` is not one."""

    adapter = FakeAdapter(tmp_path, episode_id, cleanup_status="attempted")
    config = build_config(tmp_path)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_returning(False),
    )
    outcome = await runner.run()
    assert outcome.rescue_required is True
    assert outcome.rescue_complete is False
    assert (config.rescue_root / "rescue.json").exists()


@pytest.mark.asyncio
async def test_complete_rescue_after_incomplete_cleanup_retires_the_descriptor(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id, cleanup_status="attempted")
    config = build_config(tmp_path)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_returning(True),
    )
    outcome = await runner.run()
    assert outcome.rescue_required is True
    assert outcome.rescue_complete is True
    assert not (config.rescue_root / "rescue.json").exists()
