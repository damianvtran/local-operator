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


def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point HOME and the config dir inside ``tmp_path``, and open the store."""
    from local_operator.secrets.access import open_store

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "home").mkdir()
    (tmp_path / "config").mkdir()
    return open_store(create=True)


def _store_resolver_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str):
    """A ``CredentialStoreResolver`` over an isolated store holding ``name``."""
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    store = _isolated_store(tmp_path, monkeypatch)
    store.set(name, value.encode())
    return CredentialStoreResolver(tmp_path / "config")


def test_credential_store_resolver_reads_the_store_and_names_only_the_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolver serves names from the AGENT store and names only the ref.

    The legacy ``CredentialManager`` mapping it used to fall back to is GONE
    (PR2a): a name the store does not hold is honestly missing, and the message
    carries the REF name, never a value or a path a store error might quote.
    """
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    store = _isolated_store(tmp_path, monkeypatch)
    store.set("PRESENT", CANARY.encode())

    resolver = CredentialStoreResolver(tmp_path / "config")
    assert resolver.resolve(["PRESENT"]) == (ResolvedSecret(name="PRESENT", value=CANARY),)
    for absent in ("ABSENT", "EMPTY"):
        with pytest.raises(MissingSecret) as missing:
            resolver.resolve([absent])
        assert missing.value.name == absent


def test_credential_store_resolver_never_falls_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name not on ``--secret-env`` is documented as store-only.

    Serving an ambient variable — or an old ``credentials.env`` value — for it
    would make that claim false in the operator's own proof.
    """
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    store = _isolated_store(tmp_path, monkeypatch)
    store.set("IN_STORE", b"in-store-value")
    # A decoy plaintext file must not resurface a value either (PR2a).
    (tmp_path / "config" / "credentials.env").write_text("ONLY_IN_FILE=file-value")
    monkeypatch.setenv("ONLY_IN_ENV", "ambient-value")
    resolver = CredentialStoreResolver(tmp_path / "config")
    assert resolver.resolve(["IN_STORE"])[0].value == "in-store-value"
    for absent in ("ONLY_IN_ENV", "ONLY_IN_FILE"):
        with pytest.raises(MissingSecret) as raised:
            resolver.resolve([absent])
        assert raised.value.name == absent
    assert "ambient-value" not in str(raised.value)


# ---------------------------------------------------------------------------
# resolution scope and budget (2026-10-17: the silent-boot storm)
# ---------------------------------------------------------------------------


def test_store_resolution_reads_only_the_requested_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K refs cost K reads, not a full-store scan -- the boot-storm property.

    MEASURED WHY. On 2026-10-17 the resolver read every non-provider row (130
    on the campaign's scratch stores) through the broker before serving ~18
    refs, and an episode boot resolves twice (infra refs, then secret refs);
    four concurrent session-arm runs spend ~11 minutes of boot SILENCE with
    stacks in ``retrieve_secret -> fetch_master_key -> recv_frame``. This pins
    the scope: the fetch count equals the asked-for names, a repeated name is
    fetched once, and a big store is never walked.
    """
    import local_operator.secrets.access as access
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    store = _isolated_store(tmp_path, monkeypatch)
    for index in range(8):
        store.set(f"BULK_{index}", f"bulk-value-{index}".encode())
    store.set("WANTED", b"wanted-value")

    fetched: list[str] = []
    real_retrieve = access.retrieve_secret

    def watching(name: str, base: Path | None = None, *, role: str = "agent") -> bytes:
        fetched.append(name)
        return real_retrieve(name, base, role=role)

    monkeypatch.setattr(access, "retrieve_secret", watching)
    resolver = CredentialStoreResolver(tmp_path / "config")
    resolved = resolver.resolve(["WANTED", "WANTED"])
    assert [(secret.name, secret.value) for secret in resolved] == [
        ("WANTED", "wanted-value"),
        ("WANTED", "wanted-value"),
    ]
    # One fetch for the repeated ref; none for the eight bulk rows.
    assert fetched == ["WANTED"]


def test_a_stalled_store_fails_the_resolution_budget_by_name_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget converts a silent storm into a named preflight failure.

    A per-call bound is not enough: each row's broker call is bounded
    (``REQUEST_TIMEOUT_S``), but a storm of slow answers costs that bound PER
    ROW, and the old code swallowed each per-row failure -- so the aggregate
    was unbounded and the boot emitted nothing. The failure must name the ref
    and the budget, must NOT be ``MissingSecret`` (a different problem with a
    different fix), and must never carry a value.
    """
    import time

    import local_operator.secrets.access as access
    from local_operator.evaluation.runner.host_secrets import (
        CredentialStoreResolver,
        SecretResolutionTimeout,
    )

    store = _isolated_store(tmp_path, monkeypatch)
    slow_names = ("SLOW_ONE", "SLOW_TWO", "SLOW_THREE")
    for name in slow_names:
        store.set(name, CANARY.encode())

    real_retrieve = access.retrieve_secret

    def slow(name: str, base: Path | None = None, *, role: str = "agent") -> bytes:
        time.sleep(0.03)
        return real_retrieve(name, base, role=role)

    monkeypatch.setattr(access, "retrieve_secret", slow)
    monkeypatch.setenv("LOCAL_OPERATOR_SECRET_RESOLUTION_TIMEOUT_MS", "50")

    resolver = CredentialStoreResolver(tmp_path / "config")
    started = time.monotonic()
    with pytest.raises(SecretResolutionTimeout) as raised:
        resolver.resolve(list(slow_names))
    elapsed = time.monotonic() - started
    # The property is "bounded": far below any hang, above the budget itself.
    assert 0.05 <= elapsed < 5.0
    error = raised.value
    assert not isinstance(error, MissingSecret)
    message = str(error)
    assert error.name in slow_names
    # The check runs BETWEEN fetches: under normal scheduling the third fetch's
    # re-check fires; a stalled host fires at the second. Both are correct.
    assert error.position in (2, 3) and error.total == 3
    assert f"{error.budget_s:g}s budget" in message
    assert "LOCAL_OPERATOR_SECRET_RESOLUTION_TIMEOUT_MS" in message
    assert CANARY not in message


def test_a_zero_budget_disables_the_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``0`` restores the unbudgeted behaviour (the documented off-switch)."""
    import time

    import local_operator.secrets.access as access
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver

    store = _isolated_store(tmp_path, monkeypatch)
    store.set("SLOW_OK", CANARY.encode())

    real_retrieve = access.retrieve_secret

    def slow(name: str, base: Path | None = None, *, role: str = "agent") -> bytes:
        time.sleep(0.02)
        return real_retrieve(name, base, role=role)

    monkeypatch.setattr(access, "retrieve_secret", slow)
    monkeypatch.setenv("LOCAL_OPERATOR_SECRET_RESOLUTION_TIMEOUT_MS", "0")

    resolver = CredentialStoreResolver(tmp_path / "config")
    assert resolver.resolve(["SLOW_OK"])[0].value == CANARY


# ---------------------------------------------------------------------------
# MAJOR-1: an over-long value never reaches a message
# ---------------------------------------------------------------------------

LONG_CANARY = "SECRETVALUE-" + "z" * 9000


@pytest.mark.parametrize("kind", ["static", "env", "store"])
def test_over_long_value_is_unusable_by_name_only(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ResolvedSecret.max_length`` is 8192; a cert chain exceeds it.

    pydantic's ValidationError embeds ``input_value=`` -- the value -- in its
    message, so every resolver has to translate it before it can escape. The
    store resolver is driven through the REAL store (PR2a): it reads agent-class
    rows, so a real over-long row is what its path is exercised with.
    """

    from local_operator.evaluation.runner.secrets import UnusableSecret

    resolver: SecretResolver = {
        "static": StaticSecretResolver({"K": LONG_CANARY}),
        "env": EnvSecretResolver({"K": LONG_CANARY}),
        "store": _store_resolver_with(tmp_path, monkeypatch, "K", LONG_CANARY),
    }[kind]
    with pytest.raises(UnusableSecret) as raised:
        resolver.resolve(["K"])
    assert isinstance(raised.value, MissingSecret)
    assert raised.value.name == "K"
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert "SECRETVALUE" not in str(raised.value) and "zzz" not in repr(raised.value)


def test_diagnostic_strips_inputs_from_any_validation_error() -> None:
    """Second line of defence: even a raw ValidationError renders without its input."""

    from pydantic import ValidationError

    from local_operator.evaluation.runner.episode import _diagnostic

    with pytest.raises(ValidationError) as raised:
        ResolvedSecret(name="K", value=LONG_CANARY)
    # ``None``: this asserts the ValidationError stripping specifically, which
    # holds with no redaction set at all -- that is what makes it a SECOND line
    # of defence rather than a restatement of the canary scan.
    rendered = _diagnostic(raised.value, None)
    assert rendered == "ValidationError: ResolvedSecret (value: string_too_long)"
    assert "SECRETVALUE" not in rendered and "zzz" not in rendered


@pytest.mark.asyncio
async def test_over_long_secret_fails_pre_bundle_with_the_name_only(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    config = build_config(tmp_path)
    runner = EpisodeRunner(
        _spec_with_refs(episode_id, "AWS_SECRET_ACCESS_KEY"),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        secrets=StaticSecretResolver({"AWS_SECRET_ACCESS_KEY": LONG_CANARY}),
        launch=lambda _: adapter,
    )
    outcome = await runner.run()
    assert outcome.status == "failed_pre_bundle"
    assert outcome.diagnostic == "UnusableSecret: unusable secret AWS_SECRET_ACCESS_KEY"
    assert load_pending_rescue(config.rescue_root) is None
    assert b"SECRETVALUE" not in _all_bytes_under(tmp_path)
    assert b"zzzz" not in _all_bytes_under(tmp_path)


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

        async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
            decision = await super().decide(observation, history, **kwargs)
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


# ---------------------------------------------------------------------------
# MAJOR-3: a synthetic model client never seals reportable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthetic_model_seals_a_valid_but_non_reportable_bundle(
    tmp_path: Path, episode_id: str
) -> None:
    """The label is the runner's own claim; metadata alone can be overlooked."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        synthetic_model=True,
        launch=lambda _: adapter,
        rescue=_rescue_returning(True),
    )
    outcome = await runner.run()
    assert outcome.status == "completed"
    assert outcome.reportability_label == "synthetic_model"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.outcome is not None
    assert report.outcome.reportable is False
    assert report.outcome.reportability_label == "synthetic_model"


@pytest.mark.asyncio
async def test_synthetic_model_yields_to_more_severe_labels(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id, cleanup_status="attempted")
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        synthetic_model=True,
        launch=lambda _: adapter,
        rescue=_rescue_returning(False),
    )
    outcome = await runner.run()
    assert outcome.reportability_label == "cleanup_incomplete"
