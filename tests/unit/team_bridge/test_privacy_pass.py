"""A8 privacy pass, slice 1: the default-off production seam, offline.

WHAT THIS FILE PROVES (design memo §7 acceptance, first milestone). Every row
drives the PRODUCTION decision step (``privacy_pass.decide_admission``) through
the public screening entries (``screen_message_async`` / ``screen_batch_async``
and, for the physical-send rows, the shipped ``run_screening_cycle`` with the
real wire guard). Only two things are injected: the first-stage chain runner (a
``ScreenChainRunner`` whose answering hop is a trusted ``DefaultSelection``, the
same stand-in the rest of the screening suite uses) and the privacy runner
FACTORY at the composition boundary, which supplies raw replies and typed
failures. The parser, identity checks, branch selection, state machine and the
native allow fence are production code. No network.

For each row the assertions cover what the memo asks for: privacy-call count,
reservation/physical-send rows, the receipt and hold rows, contentless
persisted state, and that no body/reason canary reaches repr, logs or the store.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import privacy_pass, screening
from local_operator.team_bridge.model_boundary import DefaultSelection
from local_operator.team_bridge.store import (
    Store,
    StoreRefused,
    _owner_key,
    state_db_path,
)
from tests.unit.team_bridge.test_screening import (  # noqa: F401 - autouse fixture
    CANARY_BODY,
    Policy,
    Runner,
    _fixture_models_are_evaluated,
    _item,
    _setup,
)

PRIVACY_CANARY = "PRIVACY-REASON-CANARY"
P_DELIVER = f"VERDICT: deliver\nREASON: {PRIVACY_CANARY} nothing asked."
P_HOLD = f"VERDICT: hold\nREASON: {PRIVACY_CANARY} cannot tell."
P_BOUNCE = f"VERDICT: bounce\nREASON: {PRIVACY_CANARY} asks for the calendar."
FIRST_HOLD = "VERDICT: hold\nREASON: needs a human."
EVAL_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "evals/team_bridge_moderation/screen/results/2026-11-04/a8-fallback-e2e"
    / "bridge-privacy-pass.txt"
)


# -- rig ----------------------------------------------------------------------


class ChainRunner(Runner):
    """The suite's trusted chain stand-in, with an explicit preferred bit."""

    def __init__(
        self, *, preferred: bool, reply: str = "VERDICT: deliver\nREASON: ordinary.", **kw: Any
    ) -> None:
        super().__init__(reply=reply, **kw)
        self.answered_hop = DefaultSelection(
            provider="qa",
            model_id=self.model or "",
            spec=None,
            preferred=preferred,
        )


@dataclasses.dataclass
class A8Policy(Policy):
    privacy_generation: int = 0


class PrivacyFactory:
    """Composition-boundary injection: scripted replies / typed failures."""

    def __init__(self, reply: Any = P_DELIVER, *, served: str | None = None) -> None:
        self.reply = reply
        self.served = served
        self.calls: list[str] = []  # prompts, ephemeral test memory only
        self.built = 0

    def __call__(self, hop: Any) -> Any:
        self.built += 1
        if (
            isinstance(self.reply, type)
            and issubclass(self.reply, BaseException)
            and (self.reply is privacy_pass.PrivacyNoCredential)
        ):
            raise self.reply("no credential")
        factory = self

        async def run(prompt: str, *, default_model: str) -> tuple[str, str | None]:
            factory.calls.append(prompt)
            reply = factory.reply
            if callable(reply) and not isinstance(reply, str):
                reply = reply(prompt)
                if asyncio.iscoroutine(reply):
                    reply = await reply
            if isinstance(reply, BaseException):
                raise reply
            return str(reply), factory.served or hop.model_id

        return run


def screening_deliver() -> str:
    return "VERDICT: deliver\nREASON: ordinary."


def _enable(tmp_path: Path, value: bool = True) -> privacy_pass.PrivacyAuthority:
    return privacy_pass.write_privacy_pass_enabled(tmp_path, value)


def _policy(tmp_path: Path, store: Store, **kw: Any) -> A8Policy:
    authority = privacy_pass.read_authority(tmp_path, store)
    suffix = privacy_pass.revision_suffix(authority)
    return A8Policy(
        admission_revision="rev-1" + suffix,
        privacy_generation=authority.policy_generation,
        **kw,
    )


def _publish(store: Store, owner: Any, policy: A8Policy) -> None:
    """Stand in for the scheduler's trusted publisher after an authority flip."""

    current = store.current_screening_policy(owner)
    if current is None or current["revision"] == policy.admission_revision:
        return
    store.publish_screening_policy(
        owner,
        version=policy.screening_version,
        template_sha256=policy.template_sha256,
        default_model=policy.default_model,
        revision=policy.admission_revision,
        expected_epoch=int(current["epoch"]),
        privacy_generation=policy.privacy_generation,
    )


def _screen(tmp_path, owner, runner, *, factory=None, item=None, policy=None):
    store = Store(tmp_path)
    if policy is None:
        policy = _policy(tmp_path, store)
        _publish(store, owner, policy)
    return asyncio.run(
        screening.screen_message_async(
            owner,
            item or _item(),
            config_dir=tmp_path,
            policy=policy,
            runner=runner,
            privacy_runner_factory=factory,
        )
    )


def _rows(tmp_path: Path, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        try:
            return db.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []


def _decisions(tmp_path: Path) -> list[tuple[Any, ...]]:
    return _rows(
        tmp_path,
        "SELECT message_id, trigger, first_preferred, stage_state, privacy_result, final, "
        "hold_code FROM privacy_decisions ORDER BY created_at",
    )


def _receipt(tmp_path: Path, owner: Any, message_id: str = "bmsg_1") -> list[tuple[Any, ...]]:
    return _rows(
        tmp_path,
        "SELECT verdict, model FROM screening_receipts WHERE owner_key = ? AND message_id = ?",
        _owner_key(owner),
        message_id,
    )


def _hold(tmp_path: Path, owner: Any, message_id: str = "bmsg_1") -> list[tuple[Any, ...]]:
    return _rows(
        tmp_path,
        "SELECT hold_code FROM screening_holds WHERE owner_key = ? AND message_id = ?",
        _owner_key(owner),
        message_id,
    )


def _privacy_sends(tmp_path: Path) -> int:
    return len(
        _rows(
            tmp_path,
            "SELECT 1 FROM screening_attempts WHERE kind = 'http' AND fingerprint LIKE 'privacy:%'",
        )
    )


def _logical(tmp_path: Path) -> int:
    return len(_rows(tmp_path, "SELECT 1 FROM screening_attempts WHERE kind = 'logical'"))


def _assert_contentless(tmp_path: Path, caplog: Any, *objects: Any) -> None:
    """No body or privacy-reason canary in the store, the logs or any repr."""

    raw = state_db_path(tmp_path).read_bytes()
    for suffix in ("-wal",):
        wal = Path(str(state_db_path(tmp_path)) + suffix)
        if wal.exists():
            raw += wal.read_bytes()
    for canary in (CANARY_BODY, PRIVACY_CANARY):
        assert canary.encode() not in raw, f"{canary} persisted"
        assert canary not in caplog.text
        for obj in objects:
            assert canary not in repr(obj)


# -- prompt, flag and authority --------------------------------------------


def test_privacy_template_is_byte_identical_to_the_preregistered_prompt() -> None:
    data = privacy_pass.PRIVACY_TEMPLATE.encode("utf-8")
    assert data == EVAL_TEMPLATE.read_bytes()
    assert len(data) == 1432
    assert hashlib.sha256(data).hexdigest() == privacy_pass.PRIVACY_TEMPLATE_SHA256
    assert (
        privacy_pass.PRIVACY_TEMPLATE_SHA256
        == "5208c727ff308167e58707a7e15ee67b59fbf982668c1884b9199fba00bc8c3b"
    )


def test_render_carries_only_the_body_in_a_fresh_nonce_envelope() -> None:
    first = privacy_pass.render_privacy_prompt("hello BODY")
    second = privacy_pass.render_privacy_prompt("hello BODY")
    assert first.nonce != second.nonce
    text = first.private_text()
    assert "hello BODY" in text and f"<submission-{first.nonce}>" in text
    assert "hello BODY" not in repr(first)


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({}, False),
        ({"team_bridge": {}}, False),
        ({"team_bridge": {"privacy_pass": {}}}, False),
        ({"team_bridge": {"privacy_pass": {"enabled": False}}}, False),
        ({"team_bridge": {"privacy_pass": {"enabled": True}}}, True),
        ({"team_bridge": {"privacy_pass": {"enabled": "true"}}}, None),
        ({"team_bridge": {"privacy_pass": {"enabled": 1}}}, None),
        ({"team_bridge": {"privacy_pass": {"enabled": None}}}, None),
        ({"team_bridge": {"privacy_pass": True}}, None),
        ({"team_bridge": "on"}, None),
    ],
)
def test_flag_is_strict_absent_is_false(document: Any, expected: Any) -> None:
    assert privacy_pass._strict_flag(document) is expected


def test_no_registry_row_can_write_the_key() -> None:
    """Only the trusted local writer reaches it: no /settings row, no tool path."""

    from local_operator.settings_io import SETTINGS

    assert not [
        s
        for s in SETTINGS
        if "privacy_pass" in s.key or s.path[:2] == ("team_bridge", "privacy_pass")
    ]


def test_writer_advances_a_monotonic_generation_with_no_aba(tmp_path: Path) -> None:
    _setup(tmp_path)
    store = Store(tmp_path)
    assert privacy_pass.read_authority(tmp_path, store) == privacy_pass.PrivacyAuthority(False, 0)
    gens = [_enable(tmp_path, v).generation for v in (True, False, True, False)]
    assert gens == [1, 2, 3, 4]
    assert privacy_pass.read_authority(tmp_path, store) == privacy_pass.PrivacyAuthority(False, 4)
    assert privacy_pass.revision_suffix(privacy_pass.PrivacyAuthority(False, 4)) == ""
    assert privacy_pass.revision_suffix(privacy_pass.PrivacyAuthority(True, 3)).endswith(":3")


def _write_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "config.yml").write_text(text)


@pytest.mark.parametrize(
    "case", ["malformed", "mismatch_off", "missing_row", "wrong_type", "writer_crash"]
)
def test_unreadable_or_inconsistent_authority_holds_before_any_spend(
    tmp_path: Path, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    policy = _policy(tmp_path, store)
    if case == "malformed":
        _write_config(tmp_path, "values: [unclosed\n")
    elif case == "mismatch_off":
        # A direct hand edit is not a supported live flip: it disagrees with
        # the control row until the writer reconciles, and that HOLDS.
        _write_config(
            tmp_path, "values:\n  team_bridge:\n    privacy_pass:\n      enabled: false\n"
        )
    elif case == "missing_row":
        with sqlite3.connect(state_db_path(tmp_path)) as db:
            db.execute("DELETE FROM privacy_control")
    elif case == "wrong_type":
        _write_config(
            tmp_path, "values:\n  team_bridge:\n    privacy_pass:\n      enabled: 'yes'\n"
        )
    elif case == "writer_crash":
        # Config written, control row write crashes: the two disagree.
        monkeypatch.setattr(
            Store, "advance_privacy_control", lambda self, **kw: (_ for _ in ()).throw(OSError())
        )
        with pytest.raises(OSError):
            _enable(tmp_path, False)
    runner = ChainRunner(preferred=False)
    factory = PrivacyFactory()
    result = _screen(tmp_path, owner, runner, factory=factory, policy=policy)
    assert result.disposition == "hold"
    assert result.hold_code in ("privacy_control_unavailable",)
    assert runner.calls == [] and factory.calls == []
    assert _logical(tmp_path) == 0 and _receipt(tmp_path, owner) == []


def test_lock_held_by_a_writer_is_unavailable_not_off(tmp_path: Path) -> None:
    from local_operator.config_writer import config_write_lock

    store, _owner = _setup(tmp_path)
    _enable(tmp_path)
    held, release = threading.Event(), threading.Event()

    def writer() -> None:
        with config_write_lock(tmp_path):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=writer)
    thread.start()
    held.wait(5)
    try:
        with pytest.raises(privacy_pass.PrivacyAuthorityUnavailable):
            with privacy_pass.fresh_authority(tmp_path, store, wait=0.2):
                pass
    finally:
        release.set()
        thread.join(5)


# -- §7 five-row matrix (single path) ---------------------------------------


@pytest.mark.parametrize(
    ("first_reply", "preferred", "privacy_reply", "final", "hold_code", "calls", "trigger"),
    [
        (screening_deliver(), False, P_DELIVER, "deliver", None, 1, "nonpreferred_deliver"),
        (
            screening_deliver(),
            False,
            P_HOLD,
            "hold",
            "privacy_verdict_hold",
            1,
            "nonpreferred_deliver",
        ),
        (
            screening_deliver(),
            False,
            P_BOUNCE,
            "hold",
            "privacy_verdict_bounce",
            1,
            "nonpreferred_deliver",
        ),
        (
            screening_deliver(),
            False,
            '{"verdict": "deliver"}',
            "hold",
            "privacy_invalid",
            1,
            "nonpreferred_deliver",
        ),
        (screening_deliver(), False, "", "hold", "privacy_invalid", 1, "nonpreferred_deliver"),
        (
            screening_deliver(),
            False,
            "VERDICT: deliver",
            "hold",
            "privacy_invalid",
            1,
            "nonpreferred_deliver",
        ),
        (FIRST_HOLD, False, "TRIPWIRE", "hold", "model_hold", 0, "first_not_deliver"),
        (screening_deliver(), True, "TRIPWIRE", "deliver", None, 0, "preferred_bypass"),
    ],
    ids=[
        "deliver+privacy-deliver",
        "deliver+privacy-hold",
        "deliver+privacy-bounce",
        "deliver+privacy-json",
        "deliver+privacy-empty",
        "deliver+privacy-missing-reason",
        "first-hold",
        "preferred-bypass",
    ],
)
def test_five_row_matrix(
    tmp_path, caplog, first_reply, preferred, privacy_reply, final, hold_code, calls, trigger
) -> None:
    caplog.set_level(logging.DEBUG)
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    runner = ChainRunner(preferred=preferred, reply=first_reply)
    factory = PrivacyFactory(privacy_reply)

    result = _screen(tmp_path, owner, runner, factory=factory)

    assert len(runner.calls) == 1
    assert len(factory.calls) == calls, "privacy-call count"
    assert _privacy_sends(tmp_path) == calls, "one claimed physical send per privacy call"
    assert result.disposition == final
    assert result.hold_code == hold_code
    decisions = _decisions(tmp_path)
    assert len(decisions) == 1 and decisions[0][1] == trigger
    if final == "deliver":
        assert _receipt(tmp_path, owner) == [("allow", "qa/default-model")]
        # The hub tuple is untouched: a two-stage allow stays non-preferred.
        assert result.receipt is not None and result.receipt.screener_preferred is preferred
        assert decisions[0][5] == ("admit" if calls else "bypass")
    else:
        assert [r for r in _receipt(tmp_path, owner) if r[0] == "allow"] == []
        assert _hold(tmp_path, owner) == [(hold_code,)]
    if calls:
        # Only the message body, in the privacy envelope; nothing else.
        assert CANARY_BODY in factory.calls[0]
        assert "needs a human" not in factory.calls[0] and "ordinary." not in factory.calls[0]
    _assert_contentless(tmp_path, caplog, result, decisions)


def test_disabled_is_the_legacy_path(tmp_path, caplog) -> None:
    """Flag absent: no decision row, no privacy call, the same receipt and revision."""

    store, owner = _setup(tmp_path)
    runner = ChainRunner(preferred=False)
    factory = PrivacyFactory("TRIPWIRE")
    policy = _policy(tmp_path, store)
    assert policy.admission_revision == "rev-1" and policy.privacy_generation == 0

    result = _screen(tmp_path, owner, runner, factory=factory, policy=policy)

    assert result.disposition == "deliver" and factory.built == 0 and factory.calls == []
    assert _decisions(tmp_path) == []
    assert _receipt(tmp_path, owner) == [("allow", "qa/default-model")]
    assert _rows(tmp_path, "SELECT privacy_generation FROM screening_policy") == [(0,)]
    assert _rows(tmp_path, "SELECT * FROM privacy_control") == []


def test_disabled_after_enable_is_legacy_for_new_decisions(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    _enable(tmp_path, False)
    factory = PrivacyFactory("TRIPWIRE")
    result = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert result.disposition == "deliver" and factory.built == 0
    assert _decisions(tmp_path) == []


@pytest.mark.parametrize(
    ("failure", "code", "sends"),
    [
        (privacy_pass.PrivacyNoCredential, "privacy_no_credential", 0),
        (ConnectionError("VENDOR-CANARY"), "privacy_unreachable", 1),
        ("timeout", "privacy_timeout", 1),
        ("foreign", "privacy_identity_mismatch", 1),
        ("none", "privacy_identity_unknown", 1),
    ],
)
def test_failure_rows_hold_without_resend(tmp_path, monkeypatch, failure, code, sends) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    if failure == "timeout":
        monkeypatch.setattr(privacy_pass, "PRIVACY_DEADLINE_S", 0.05)

        async def slow(_prompt: str) -> str:
            await asyncio.sleep(5)
            return P_DELIVER

        factory = PrivacyFactory(slow)
    elif failure == "foreign":
        factory = PrivacyFactory(P_DELIVER, served="other/model")
    elif failure == "none":
        factory = PrivacyFactory(P_DELIVER)
        factory.served = None
        original = factory.__call__

        def no_identity(hop: Any) -> Any:
            run = original(hop)

            async def wrapped(prompt: str, *, default_model: str) -> Any:
                reply, _ = await run(prompt, default_model=default_model)
                return reply, None

            return wrapped

        factory = no_identity  # type: ignore[assignment]
    else:
        factory = PrivacyFactory(failure)
    result = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert (result.disposition, result.hold_code) == ("hold", code)
    assert _privacy_sends(tmp_path) == sends
    assert _receipt(tmp_path, owner) == []
    # Retry: the settled hold is re-served; no stage is re-run, nothing re-sent.
    again = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=PrivacyFactory())
    assert again.hold_code == code
    assert _privacy_sends(tmp_path) == sends and _logical(tmp_path) == 1


def test_unknown_provenance_holds(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    runner = ChainRunner(preferred=True)
    runner.answered_hop = None  # a runner that cannot evidence who answered
    factory = PrivacyFactory("TRIPWIRE")
    result = _screen(tmp_path, owner, runner, factory=factory)
    assert result.hold_code == "privacy_identity_unknown" and factory.calls == []


def test_no_factory_in_production_holds_before_any_send(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    result = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=None)
    assert result.hold_code == "privacy_runner_unavailable"
    assert _privacy_sends(tmp_path) == 0 and _receipt(tmp_path, owner) == []


def test_attachments_hold_unsupported_input() -> None:
    """The decision step itself: a non-preferred deliver with images/files holds."""

    for images, files in ((object(), None), (None, object())):
        trigger, prewire = privacy_pass.reduce_admission(
            enabled=True,
            first_disposition="deliver",
            first_preferred=False,
            has_attachments=images is not None or files is not None,
        )
        assert (trigger, prewire) == ("nonpreferred_deliver", "privacy_unsupported_input")
    # Preferred bypass is unchanged for attachments.
    assert privacy_pass.reduce_admission(
        enabled=True, first_disposition="deliver", first_preferred=True, has_attachments=True
    ) == ("preferred_bypass", None)


def test_attachments_hold_through_the_seam(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    authority = _enable(tmp_path)

    class Env:
        message_id = "bmsg_1"
        envelope_digest = "e" * 64
        body_sha256 = "b" * 64
        sender_account = "acc_sender1"
        sender_kind = ""

        def private_images(self) -> Any:
            return object()

        def private_files(self) -> Any:
            return None

        def private_body(self) -> str:
            return CANARY_BODY

    class Prepared:
        envelope = Env()
        snapshot = type("S", (), {"admission_revision": "rev-x"})()
        epoch = 1

    factory = PrivacyFactory("TRIPWIRE")
    decision = asyncio.run(
        privacy_pass.decide_admission(
            store=store,
            owner=owner,
            config_dir=tmp_path,
            prepared=Prepared(),
            authority=authority,
            attempt_id="a1",
            first_disposition="deliver",
            actual_model="qa/default-model",
            runner=ChainRunner(preferred=False),
            factory=factory,
            turn="t",
        )
    )
    assert (decision.proceed, decision.hold_code) == (False, "privacy_unsupported_input")
    assert factory.built == 0 and _privacy_sends(tmp_path) == 0


# -- native proof refusal -----------------------------------------------------


def test_native_fence_refuses_an_allow_without_privacy_evidence(tmp_path, caplog) -> None:
    """A forgotten finalizer: ``_conclude_parsed`` called with no decision proof.

    RED->GREEN: with ``Store._require_privacy_proof`` neutralized this commits
    an allow receipt (recorded in the PR evidence); with the fence it holds.
    """

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    policy = _policy(tmp_path, store)
    runner = ChainRunner(preferred=False)
    captured: dict[str, Any] = {}
    real_decide = screening._decide

    async def bypass(*args: Any, **kw: Any) -> Any:
        # Simulates a future caller that skipped the seam entirely.
        captured["called"] = True
        return privacy_pass.AdmissionDecision("nonpreferred_deliver", proceed=True)

    screening._decide = bypass  # type: ignore[assignment]
    try:
        result = _screen(tmp_path, owner, runner, factory=PrivacyFactory(), policy=policy)
    finally:
        screening._decide = real_decide  # type: ignore[assignment]
    assert captured["called"]
    assert (result.disposition, result.hold_code) == ("hold", "privacy_proof_missing")
    assert _receipt(tmp_path, owner) == []


def test_native_fence_refuses_a_proof_for_another_message(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    assert (
        _screen(tmp_path, owner, ChainRunner(preferred=False), factory=PrivacyFactory()).disposition
        == "deliver"
    )
    (decision_id,) = _rows(tmp_path, "SELECT decision_id FROM privacy_decisions")[0]
    real_decide = screening._decide

    async def stolen(*args: Any, **kw: Any) -> Any:
        return privacy_pass.AdmissionDecision(
            "nonpreferred_deliver", proceed=True, decision_id=decision_id
        )

    screening._decide = stolen  # type: ignore[assignment]
    try:
        result = _screen(
            tmp_path,
            owner,
            ChainRunner(preferred=False),
            factory=PrivacyFactory(),
            item=_item(message_id="bmsg_2"),
        )
    finally:
        screening._decide = real_decide  # type: ignore[assignment]
    assert result.hold_code == "privacy_proof_missing"
    assert _receipt(tmp_path, owner, "bmsg_2") == []


# -- restart / no double send -------------------------------------------------


class _Crash(BaseException):
    """A process death at a chosen instant (not an Exception: nothing catches it)."""


def _crash_screen(tmp_path, owner, factory) -> None:
    with pytest.raises(_Crash):
        _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)


@pytest.mark.parametrize(
    ("point", "sends", "code"),
    [
        # Round 1 M-2: a crash INSIDE the claim transaction rolled it back, so
        # nothing was sent and the open/pending row RESUMES (one send, total).
        ("before_claim", 1, "resumed"),
        ("after_claim", 1, "privacy_uncertain"),
        ("after_response", 1, "privacy_uncertain"),
        ("after_completion", 1, None),
    ],
)
def test_restart_never_sends_again(tmp_path, monkeypatch, point, sends, code) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    factory = PrivacyFactory()
    if point == "before_claim":
        monkeypatch.setattr(
            Store, "claim_privacy_send", lambda *a, **k: (_ for _ in ()).throw(_Crash())
        )
    elif point == "after_claim":
        factory = PrivacyFactory(_Crash())
    elif point == "after_response":
        monkeypatch.setattr(
            Store, "complete_privacy_stage", lambda *a, **k: (_ for _ in ()).throw(_Crash())
        )
    else:
        real = Store.finish_screening

        def crash_on_allow(self, *a: Any, **k: Any) -> Any:
            if k.get("receipt") is not None:
                raise _Crash()
            return real(self, *a, **k)

        monkeypatch.setattr(Store, "finish_screening", crash_on_allow)
    _crash_screen(tmp_path, owner, factory)
    monkeypatch.undo()
    # Re-admit the fixture models (undo removed them) and retry.
    from local_operator.team_bridge import model_boundary

    monkeypatch.setattr(
        model_boundary,
        "SCREENING_EVALUATED_MODELS",
        (("qa", "qa/default-model"),) + model_boundary.SCREENING_EVALUATED_MODELS,
    )
    monkeypatch.setattr(model_boundary, "_credentialed_providers", lambda _c: {"qa"})
    first_runner = ChainRunner(preferred=False)
    retry_factory = PrivacyFactory()
    result = _screen(tmp_path, owner, first_runner, factory=retry_factory)
    if code == "resumed":
        assert (result.disposition, result.hold_code) == ("deliver", None)
        assert len(retry_factory.calls) == 1 and _privacy_sends(tmp_path) == sends
        assert _decisions(tmp_path)[0][3:6] == ("completed", "deliver", "admit")
        return
    assert retry_factory.calls == [], "no new privacy draw"
    assert _privacy_sends(tmp_path) == sends
    if code is None:
        # Crash after a completed privacy DELIVER, before the receipt: the
        # retry is the existing non-preferred re-screen (one new first-stage
        # draw, as without A8) and REUSES the completed privacy stage.
        assert result.disposition == "deliver"
        assert len(first_runner.calls) == 1 and _logical(tmp_path) == 2
        assert _receipt(tmp_path, owner) == [("allow", "qa/default-model")]
        return
    assert (result.disposition, result.hold_code) == ("hold", code)
    assert first_runner.calls == [], "no new first-stage draw"
    assert _logical(tmp_path) == 1
    assert [r for r in _receipt(tmp_path, owner) if r[0] == "allow"] == []


def test_receipt_resend_and_concurrent_duplicate_do_not_redraw(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    factory = PrivacyFactory()
    policy = _policy(tmp_path, store)

    async def both() -> Any:
        return await asyncio.gather(
            *(
                screening.screen_message_async(
                    owner,
                    _item(),
                    config_dir=tmp_path,
                    policy=policy,
                    runner=ChainRunner(preferred=False),
                    privacy_runner_factory=factory,
                )
                for _ in range(2)
            )
        )

    results = asyncio.run(both())
    assert [(r.disposition, r.hold_code) for r in results] == [("deliver", None)] * 2
    again = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory, policy=policy)
    assert again.disposition == "deliver"
    assert len(factory.calls) == 1 and _privacy_sends(tmp_path) == 1
    assert _receipt(tmp_path, owner) == [("allow", "qa/default-model")]


# -- batch: per-message conjunction, no neighbour mixing --------------------


def _triples(*verdicts: tuple[str, str]) -> str:
    lines: list[str] = []
    for label, verdict in verdicts:
        lines += [f"ID: {label}", f"VERDICT: {verdict}", "REASON: ordinary."]
    return "\n".join(lines)


def test_batch_runs_one_isolated_privacy_call_per_message(tmp_path, caplog) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    bodies = ["BODY-ALPHA canary one", "BODY-BRAVO canary two", "BODY-CHARLIE canary three"]
    items = [_item(body=b, message_id=f"bmsg_{i}") for i, b in enumerate(bodies, 1)]

    def batch_reply() -> str:
        # The batch renderer labels messages m1..mN in call order.
        return _triples(*[(f"m{n}", "deliver") for n in range(1, len(items) + 1)])

    runner = ChainRunner(preferred=False)
    runner.fn = None
    original = runner.__class__.__call__

    def call(self, prompt: str, *, default_model: str, images=None):  # type: ignore[no-untyped-def]
        self.calls.append(prompt)
        return batch_reply(), self.model

    runner.__class__ = type("BatchChain", (ChainRunner,), {"__call__": call})

    def privacy_reply(prompt: str) -> str:
        return P_HOLD if "BODY-BRAVO" in prompt else P_DELIVER

    factory = PrivacyFactory(privacy_reply)
    results = asyncio.run(
        screening.screen_batch_async(
            owner,
            items,
            config_dir=tmp_path,
            policy=_policy(tmp_path, store),
            runner=runner,
            privacy_runner_factory=factory,
        )
    )
    del original
    assert len(runner.calls) == 1, "one first-stage batch call"
    assert len(factory.calls) == 3, "one privacy call per message"
    for prompt, body in zip(factory.calls, bodies):
        assert body in prompt
        assert [b for b in bodies if b != body and b in prompt] == [], "no neighbour content"
    assert [(r.disposition, r.hold_code) for r in results] == [
        ("deliver", None),
        ("hold", "privacy_verdict_hold"),
        ("deliver", None),
    ]
    assert _hold(tmp_path, owner, "bmsg_2") == [("privacy_verdict_hold",)]
    assert _privacy_sends(tmp_path) == 3
    assert all(d[1] == "nonpreferred_deliver" for d in _decisions(tmp_path))


# -- S-1 physical sender allowance -------------------------------------------


def _budget(tmp_path: Path, sender: int) -> None:
    from local_operator.team_bridge.settings import ScreeningBudget
    from tests.unit.team_bridge.test_batch_screening import _write_budget

    _write_budget(tmp_path, ScreeningBudget(account=40, ambient=30, unaddressed=35, sender=sender))


def _seed_http(tmp_path: Path, owner: Any, *, sender: str | None, at: float, n: int = 1) -> None:
    store = Store(tmp_path)
    with store._connect() as conn:
        store._init(conn)
        store._ensure_attempt_allocation_columns(conn)
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        for i in range(n):
            db.execute(
                "INSERT INTO screening_attempts (attempt_id, kind, account_scope, owner_key, "
                "message_id, fingerprint, at, sender_account_id) VALUES (?,?,?,?,?,?,?,?)",
                (
                    f"seed-{sender}-{at}-{i}",
                    "http",
                    store._account_scope(owner),
                    _owner_key(owner),
                    "seed",
                    "seed",
                    at,
                    sender,
                ),
            )


@pytest.mark.parametrize("variant", ["same_body_fresh_ids", "variant_bodies"])
def test_fresh_ids_and_variant_bodies_share_one_sender_allowance(tmp_path, variant) -> None:
    """sender ceiling 3, two privacy-stage sends seeded per message pattern."""

    store, owner = _setup(tmp_path)
    _budget(tmp_path, sender=3)
    _enable(tmp_path)
    # One prior physical send by this sender (a first-stage POST: the test
    # chain stand-in does not POST, so it is seeded), so the PHYSICAL share
    # binds before the LOGICAL one, which uses the same configured value.
    _seed_http(tmp_path, owner, sender="acc_sender1", at=time.time())
    factory = PrivacyFactory()
    codes = []
    for i in range(5):
        body = CANARY_BODY if variant == "same_body_fresh_ids" else f"{CANARY_BODY} v{i}"
        result = _screen(
            tmp_path,
            owner,
            ChainRunner(preferred=False),
            factory=factory,
            item=_item(body=body, message_id=f"bmsg_{i}"),
        )
        codes.append(result.hold_code)
    # Physical share 3 = 1 seeded + 2 privacy sends; the third fresh id holds
    # BEFORE the provider (factory call count stops at 2); from the fourth on,
    # the logical sender ceiling (same configured value) refuses first.
    assert codes == [
        None,
        None,
        "privacy_budget_exhausted",
        "budget_exhausted",
        "budget_exhausted",
    ]
    assert len(factory.calls) == 2 and _privacy_sends(tmp_path) == 2
    attributed = _rows(
        tmp_path,
        "SELECT DISTINCT sender_account_id FROM screening_attempts WHERE kind = 'http'",
    )
    assert attributed == [("acc_sender1",)]


def test_another_verified_sender_keeps_its_own_share(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _budget(tmp_path, sender=2)
    _enable(tmp_path)
    _seed_http(tmp_path, owner, sender="acc_sender1", at=time.time(), n=2)
    factory = PrivacyFactory()
    item = _item(message_id="bmsg_9")
    item["sender"] = {"account_id": "acc_other"}
    assert (
        _screen(
            tmp_path, owner, ChainRunner(preferred=False), factory=factory, item=item
        ).disposition
        == "deliver"
    )
    held = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert held.hold_code == "privacy_budget_exhausted"
    assert len(factory.calls) == 1


def test_unattributed_history_counts_conservatively_and_expires(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _budget(tmp_path, sender=2)
    _enable(tmp_path)
    now = time.time()
    _seed_http(tmp_path, owner, sender=None, at=now - 10, n=2)  # pre-upgrade, in window
    factory = PrivacyFactory()
    held = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert held.hold_code == "privacy_budget_exhausted" and factory.calls == []
    # Exactly-at-window rows (at <= now - 3600) no longer count.
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        db.execute(
            "UPDATE screening_attempts SET at = ? WHERE attempt_id LIKE 'seed-%'", (now - 3601,)
        )
    ok = _screen(
        tmp_path,
        owner,
        ChainRunner(preferred=False),
        factory=factory,
        item=_item(message_id="bmsg_2"),
    )
    assert ok.disposition == "deliver" and len(factory.calls) == 1


def test_allowance_survives_flag_round_trip_and_restart(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    _budget(tmp_path, sender=2)
    _enable(tmp_path)
    _seed_http(tmp_path, owner, sender="acc_sender1", at=time.time())
    factory = PrivacyFactory()
    assert (
        _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory).disposition
        == "deliver"
    )
    _enable(tmp_path, False)
    _enable(tmp_path, True)  # a new generation must NOT reset the physical ledger
    held = _screen(
        tmp_path,
        owner,
        ChainRunner(preferred=False),
        factory=factory,
        item=_item(message_id="bmsg_2"),
    )
    assert held.hold_code == "privacy_budget_exhausted" and len(factory.calls) == 1


def test_last_unit_race_admits_exactly_one(tmp_path) -> None:
    """Two Stores (two processes' connections) race the last physical unit."""

    store, owner = _setup(tmp_path)
    _budget(tmp_path, sender=1)
    authority = _enable(tmp_path)
    policy = _policy(tmp_path, store)
    turn = store.claim_screen_turn(owner, holder_prefix="t", now=time.time(), ttl=120)
    assert turn is not None
    ids = []
    for mid in ("m1", "m2"):
        row = store.open_privacy_decision(
            owner,
            message_id=mid,
            envelope_digest="e",
            body_sha256="b",
            admission_revision=policy.admission_revision,
            policy_epoch=1,
            control_generation=authority.generation,
            rule_version=privacy_pass.PRIVACY_RULE_VERSION,
            trigger="nonpreferred_deliver",
            first_preferred=False,
            first_model="m",
            first_provider="p",
            first_disposition="deliver",
            first_attempt_id="a",
            channel="single",
        )
        ids.append(row["decision_id"])
    store.publish_screening_policy(
        owner,
        version=policy.screening_version,
        template_sha256=policy.template_sha256,
        default_model=policy.default_model,
        revision=policy.admission_revision,
        privacy_generation=authority.generation,
    )
    outcomes = []
    for decision_id in ids:
        try:
            Store(tmp_path).claim_privacy_send(
                owner,
                decision_id,
                turn=turn,
                control_generation=authority.generation,
                expected_epoch=1,
                admission_revision=policy.admission_revision,
                sender_account_id="acc_sender1",
                sender_kind="",
                template_sha256=privacy_pass.PRIVACY_TEMPLATE_SHA256,
                route="qa/m",
                deadline_at=time.time() + 45,
            )
            outcomes.append("claimed")
        except StoreRefused as refusal:
            outcomes.append(refusal.code)
    assert outcomes == ["claimed", "sender_exhausted"]
    # The one-send token: a claimed stage is never claimed twice.
    with pytest.raises(StoreRefused) as again:
        store.claim_privacy_send(
            owner,
            ids[0],
            turn=turn,
            control_generation=authority.generation,
            expected_epoch=1,
            admission_revision=policy.admission_revision,
            sender_account_id="acc_sender1",
            sender_kind="",
            template_sha256=privacy_pass.PRIVACY_TEMPLATE_SHA256,
            route="qa/m",
            deadline_at=time.time() + 45,
        )
    assert again.value.code == "stage_taken"


# -- S-2 two-instance stale manager / barriers -------------------------------


def test_stale_enabled_policy_at_entry_holds(tmp_path) -> None:
    """Process A keeps an enabled policy object; B disables through the writer."""

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    stale = _policy(tmp_path, store)  # A's captured, enabled
    _enable(tmp_path, False)  # B, an independently opened operator writer
    runner, factory = ChainRunner(preferred=False), PrivacyFactory("TRIPWIRE")
    result = _screen(tmp_path, owner, runner, factory=factory, policy=stale)
    assert result.hold_code == "policy_stale"
    assert runner.calls == [] and factory.calls == [] and _logical(tmp_path) == 0


def test_stale_enabled_publisher_cannot_republish(tmp_path) -> None:
    store, owner = _setup(tmp_path)
    authority = _enable(tmp_path)
    _enable(tmp_path, False)
    with pytest.raises(StoreRefused) as refused:
        store.publish_screening_policy(
            owner,
            version="v",
            template_sha256="t" * 64,
            default_model="m",
            revision="rev" + privacy_pass.revision_suffix(authority),
            privacy_generation=authority.generation,
        )
    assert refused.value.code == "policy_stale"
    # An off-generation baseline publication is still permitted.
    assert (
        store.publish_screening_policy(
            owner, version="v", template_sha256="t" * 64, default_model="m", revision="rev"
        )
        == 1
    )


def _barrier_run(tmp_path, owner, store, *, barrier: str, aba: bool = False):
    """Pause A at ``barrier``; B flips the authority; resume A."""

    policy = _policy(tmp_path, store)
    gate = {"done": False}

    def flip() -> None:
        if gate["done"]:
            return
        gate["done"] = True
        _enable(tmp_path, False)
        if aba:
            _enable(tmp_path, True)  # false->true: a NEW generation, still stale

    runner = ChainRunner(preferred=False)
    factory = PrivacyFactory()
    if barrier == "prewire_first":
        real = privacy_pass.read_authority
        calls = {"n": 0}

        def hooked(config_dir: Path, s: Store) -> Any:
            calls["n"] += 1
            if calls["n"] == 2:  # after entry, before the first-stage wire
                flip()
            return real(config_dir, s)

        screening_read = hooked
    else:
        screening_read = None
    if barrier == "prewire_privacy":
        runner.fn = flip  # B disables while A's first stage is in flight
    if barrier == "precommit":

        def late(prompt: str) -> str:
            flip()  # a pre-disable authorized call finishing late
            return P_DELIVER

        factory = PrivacyFactory(late)
    return policy, runner, factory, screening_read


@pytest.mark.parametrize("aba", [False, True], ids=["disable", "disable-reenable-aba"])
@pytest.mark.parametrize(
    ("barrier", "privacy_calls", "first_calls"),
    [("prewire_first", 0, 0), ("prewire_privacy", 0, 1), ("precommit", 1, 1)],
)
def test_two_instance_barriers(
    tmp_path, monkeypatch, barrier, privacy_calls, first_calls, aba
) -> None:
    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    policy, runner, factory, hooked = _barrier_run(tmp_path, owner, store, barrier=barrier, aba=aba)
    if hooked is not None:
        monkeypatch.setattr(privacy_pass, "read_authority", hooked)
    result = _screen(tmp_path, owner, runner, factory=factory, policy=policy)
    assert (result.disposition, result.hold_code) == ("hold", "policy_stale")
    assert len(runner.calls) == first_calls
    assert len(factory.calls) == privacy_calls, "provider tripwire"
    assert _privacy_sends(tmp_path) == privacy_calls
    assert _receipt(tmp_path, owner) == [], "no old-generation allow after disable"


def test_stale_first_stage_reservation_refuses_natively(tmp_path) -> None:
    """The wire guard's reservation sees the control row, not a cached policy."""

    store, owner = _setup(tmp_path)
    authority = _enable(tmp_path)
    store.publish_screening_policy(
        owner,
        version="v",
        template_sha256="t" * 64,
        default_model="m",
        revision="r",
        privacy_generation=authority.generation,
    )
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        before = db.execute("SELECT COUNT(*) FROM screening_attempts").fetchone()[0]
    _enable(tmp_path, False)
    with pytest.raises(StoreRefused) as refused:
        with store._connect() as conn:
            store._init(conn)
            store._require_privacy_generation(conn, authority.generation)
    assert refused.value.code == "policy_stale"
    with sqlite3.connect(state_db_path(tmp_path)) as db:
        assert db.execute("SELECT COUNT(*) FROM screening_attempts").fetchone()[0] == before


# -- the SHIPPED composition: run_screening_cycle + real wire guard ----------


def _real_cycle(
    tmp_path: Path, monkeypatch: Any, *, preferred: bool, factory: Any, items: Any = None
) -> Any:
    """``run_screening_cycle`` with the real chain resolver, factory and wire guard.

    Only the vendor client and the physical transport are faked (the
    ``test_screening_chain`` rig). ``preferred`` decides whether the default
    pair sits in the evaluated table (a PREFERRED hop) or comes from the
    receiver's own cascade (``preferred=False``): provenance is the resolver's,
    never a fixture flag.
    """

    from local_operator.team_bridge import model_boundary
    from local_operator.team_bridge import scheduler as _scheduler
    from tests.unit.team_bridge.test_screening import _consent_status
    from tests.unit.team_bridge.test_screening_chain import _install_wire, _Wire

    monkeypatch.setattr(
        model_boundary,
        "SCREENING_EVALUATED_MODELS",
        (("openai", "gpt-5.4"),) if preferred else (),
    )
    monkeypatch.setattr(model_boundary, "_credentialed_providers", lambda _c: {"openai"})
    # The model config is seeded by ``_seed_model_config`` BEFORE any flag
    # write, so the operator writer merges into it like a real config.
    wire = _Wire({"gpt-5.4": ("reply", "VERDICT: deliver\nREASON: ordinary.")})
    _install_wire(monkeypatch, wire)
    from tests.unit.team_bridge.test_screening import _owner

    owner = _owner()
    raw = _consent_status(owner, {"version": "bridge_consent_v1", "text": "v1 consent"})
    verified = _scheduler.VerifiedStatus(_scheduler._VERIFIED_TOKEN, raw)

    async def fetch() -> Any:
        return verified

    async def drive() -> Any:
        return await _scheduler.run_screening_cycle(
            config_dir=tmp_path,
            owner=owner,
            items=items or [_item()],
            auth_store=None,
            status_fetcher=fetch,
            privacy_runner_factory=factory,
        )

    return asyncio.run(drive()), wire


def _revision(store: Store, owner: Any) -> str:
    published = store.current_screening_policy(owner)
    assert published is not None
    return str(published["revision"])


def _seed_model_config(tmp_path: Path) -> None:
    (tmp_path / "config.yml").write_text("values:\n  hosting: openai\n  model_name: gpt-5.4\n")


def _http_rows(tmp_path: Path) -> list[tuple[Any, ...]]:
    return _rows(
        tmp_path,
        "SELECT CASE WHEN fingerprint LIKE 'privacy:%' THEN 'privacy' ELSE 'first' END, "
        "sender_account_id FROM screening_attempts WHERE kind = 'http' ORDER BY at",
    )


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled-legacy", "enabled"])
def test_shipped_cycle_nonpreferred(tmp_path, monkeypatch, caplog, enabled) -> None:
    store, owner = _setup(tmp_path)
    _seed_model_config(tmp_path)
    if enabled:
        _enable(tmp_path)
    factory = PrivacyFactory(P_DELIVER)
    result, wire = _real_cycle(tmp_path, monkeypatch, preferred=False, factory=factory)
    assert (result.disposition, result.hold_code) == ("deliver", None), _decisions(tmp_path)
    assert result.receipt.screener_preferred is False
    assert len(wire.sends) == 1, "first stage: one guarded POST"
    published = store.current_screening_policy(owner)
    assert published is not None
    revision = published["revision"]
    if not enabled:
        # Byte-identical legacy: no privacy factory built, no decision row,
        # one attributed first-stage row, policy minted at generation 0.
        assert factory.built == 0 and _decisions(tmp_path) == []
        assert _http_rows(tmp_path) == [("first", "acc_sender1")]
        assert _rows(tmp_path, "SELECT privacy_generation FROM screening_policy") == [(0,)]
        return
    assert len(factory.calls) == 1
    # The first stage's guarded send and the privacy send are BOTH attributed
    # to the verified sender (S-1): one physical ledger for both stages.
    assert _http_rows(tmp_path) == [("first", "acc_sender1"), ("privacy", "acc_sender1")]
    assert _decisions(tmp_path)[0][1:6] == (
        "nonpreferred_deliver",
        0,
        "completed",
        "deliver",
        "admit",
    )
    gen = privacy_pass.read_authority(tmp_path, store).generation
    assert _rows(tmp_path, "SELECT privacy_generation FROM screening_policy") == [(gen,)]
    assert revision.startswith("accepted:")
    _assert_contentless(tmp_path, caplog, result)


def test_shipped_cycle_preferred_bypass(tmp_path, monkeypatch) -> None:
    store, owner = _setup(tmp_path)
    _seed_model_config(tmp_path)
    _enable(tmp_path)
    factory = PrivacyFactory("TRIPWIRE")
    result, wire = _real_cycle(tmp_path, monkeypatch, preferred=True, factory=factory)
    assert result.disposition == "deliver" and result.receipt.screener_preferred is True
    assert factory.built == 0 and len(wire.sends) == 1
    assert _decisions(tmp_path)[0][1:3] == ("preferred_bypass", 1)


def test_shipped_cycle_disable_moves_the_revision_back(tmp_path, monkeypatch) -> None:
    """Off publishes the legacy revision; enabled appends the A8 binding."""

    store, owner = _setup(tmp_path)
    _seed_model_config(tmp_path)
    _real_cycle(tmp_path, monkeypatch, preferred=False, factory=None)
    legacy = _revision(store, owner)
    _enable(tmp_path)
    monkeypatch.undo()
    _real_cycle(tmp_path, monkeypatch, preferred=False, factory=None)
    enabled = _revision(store, owner)
    _enable(tmp_path, False)
    monkeypatch.undo()
    _real_cycle(tmp_path, monkeypatch, preferred=False, factory=None)
    assert enabled != legacy
    assert _revision(store, owner) == legacy


# -- round 1 remediation regressions (PR #3778) -------------------------------


def _fail_once(monkeypatch: Any, name: str, *, when: Any = None) -> dict[str, int]:
    """Make ``Store.<name>`` refuse ``busy`` once (optionally only when ``when``)."""

    real = getattr(Store, name)
    state = {"n": 0}

    def flaky(self: Store, *a: Any, **k: Any) -> Any:
        if state["n"] == 0 and (when is None or when(*a, **k)):
            state["n"] += 1
            raise StoreRefused("busy", "injected")
        return real(self, *a, **k)

    monkeypatch.setattr(Store, name, flaky)
    return state


def test_m1_retry_after_uncommitted_bypass_runs_privacy_once_and_admits(
    tmp_path, monkeypatch
) -> None:
    """M-1: a bypass whose commit never landed must not freeze the row.

    At 654f9658 every non-preferred retry held ``privacy_pending`` (stage
    claim refused ``stage_taken`` on the settled ``bypass`` row) and paid the
    first stage again, forever.
    """

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    _fail_once(monkeypatch, "finish_screening")
    first = _screen(
        tmp_path, owner, ChainRunner(preferred=True), factory=PrivacyFactory("TRIPWIRE")
    )
    assert first.disposition == "hold"
    assert _decisions(tmp_path)[0][1] == "preferred_bypass"
    factory = PrivacyFactory(P_DELIVER)
    retry = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert (retry.disposition, retry.hold_code) == ("deliver", None), _decisions(tmp_path)
    assert len(factory.calls) == 1 and _privacy_sends(tmp_path) == 1
    # Re-stamped, not duplicated: one row for the binding, now the privacy admit.
    assert _decisions(tmp_path) == [
        ("bmsg_1", "nonpreferred_deliver", 0, "completed", "deliver", "admit", None)
    ]
    assert _receipt(tmp_path, owner) == [("allow", "qa/default-model")]


def test_m1_preferred_retry_on_a_completed_admit_reuses_it(tmp_path, monkeypatch) -> None:
    """M-1 reverse: an ``admit`` row then a PREFERRED retry.

    At 654f9658 the retry settled ``bypass`` on a ``first_preferred = 0`` row,
    the fence refused ``privacy_proof_missing`` and every later replay held
    ``privacy_commit_interrupted``. Now the completed stage is reused: no new
    privacy call, and the allow carries the admit proof.
    """

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    _fail_once(monkeypatch, "finish_screening", when=lambda *a, **k: k.get("receipt") is not None)
    factory = PrivacyFactory(P_DELIVER)
    first = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert first.disposition == "hold"
    assert _decisions(tmp_path)[0][5] == "admit"
    retry_factory = PrivacyFactory("TRIPWIRE")
    retry = _screen(tmp_path, owner, ChainRunner(preferred=True), factory=retry_factory)
    assert (retry.disposition, retry.hold_code) == ("deliver", None), _decisions(tmp_path)
    assert retry_factory.built == 0 and _privacy_sends(tmp_path) == 1
    assert _decisions(tmp_path)[0][1:6] == (
        "nonpreferred_deliver",
        0,
        "completed",
        "deliver",
        "admit",
    )


def test_m1_preferred_retry_cannot_skip_an_owed_privacy_verdict(tmp_path, monkeypatch) -> None:
    """S-1 kept: a stored non-preferred requirement survives a preferred re-draw.

    The first attempt's privacy claim is refused transiently (row stays open);
    the retry is answered by a PREFERRED hop. It must still run the privacy
    stage (and so hold on a privacy hold), never settle a bypass.
    """

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    _fail_once(monkeypatch, "claim_privacy_send")
    first = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=PrivacyFactory())
    assert first.hold_code == "retry_later"
    factory = PrivacyFactory(P_HOLD)
    retry = _screen(tmp_path, owner, ChainRunner(preferred=True), factory=factory)
    assert (retry.disposition, retry.hold_code) == ("hold", "privacy_verdict_hold")
    assert len(factory.calls) == 1
    assert _receipt(tmp_path, owner) == []


@pytest.mark.parametrize("where", ["renew", "claim"])
def test_m2_transient_refusal_is_not_a_final_hold(tmp_path, monkeypatch, where) -> None:
    """M-2: a lost turn / transient claim refusal leaves the decision open.

    At 654f9658 both settled ``final = 'hold'`` and replay re-served
    ``retry_later`` forever with no further attempt.
    """

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    if where == "renew":
        real = Store.renew_screen_turn
        state = {"armed": False}

        def renew(self: Store, *a: Any, **k: Any) -> Any:
            if state["armed"]:
                state["armed"] = False
                return False
            return real(self, *a, **k)

        monkeypatch.setattr(Store, "renew_screen_turn", renew)
        real_render = privacy_pass.render_privacy_prompt

        def render(*a: Any, **k: Any) -> Any:
            # Arms the refusal for the renew made right after the render,
            # i.e. the one inside ``decide_admission``, not the first stage's.
            state["armed"] = not state.get("used")
            state["used"] = True
            return real_render(*a, **k)

        monkeypatch.setattr(privacy_pass, "render_privacy_prompt", render)
    else:
        _fail_once(monkeypatch, "claim_privacy_send")
    first = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=PrivacyFactory())
    assert first.hold_code == "retry_later"
    assert _decisions(tmp_path)[0][3:] == ("pending", None, "open", None)
    factory = PrivacyFactory(P_DELIVER)
    retry = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert (retry.disposition, retry.hold_code) == ("deliver", None), _decisions(tmp_path)
    assert len(factory.calls) == 1 and _privacy_sends(tmp_path) == 1


def test_m2_budget_refusal_still_settles(tmp_path) -> None:
    """M-2 boundary: the S-1 budget hold is one of the codes meant to stick."""

    store, owner = _setup(tmp_path)
    _budget(tmp_path, sender=1)
    _enable(tmp_path)
    _seed_http(tmp_path, owner, sender="acc_sender1", at=time.time())
    held = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=PrivacyFactory())
    assert held.hold_code == "privacy_budget_exhausted"
    assert _decisions(tmp_path)[0][5:] == ("hold", "privacy_budget_exhausted")


def test_m2_sync_privacy_runner_is_refused_before_any_send(tmp_path) -> None:
    """m-2: the deadline bounds only a coroutine, so a sync runner never runs."""

    store, owner = _setup(tmp_path)
    _enable(tmp_path)
    calls: list[str] = []

    def factory(hop: Any) -> Any:
        def run(prompt: str, *, default_model: str) -> tuple[str, str]:
            calls.append(prompt)
            return P_DELIVER, hop.model_id

        return run

    held = _screen(tmp_path, owner, ChainRunner(preferred=False), factory=factory)
    assert held.hold_code == "privacy_unreachable"
    assert calls == [] and _privacy_sends(tmp_path) == 0


def test_q1_one_senders_spent_physical_share_never_opens_the_account_breaker(
    tmp_path, monkeypatch
) -> None:
    """Q-1 (QA Q6c shape, real wire): sender 1 spends its physical share.

    The logical share counts 1 per message, the physical share 2 (first stage
    + privacy), so messages 5-7 pass the logical check and are refused by the
    guard's physical check. At 654f9658 each refusal advanced the ACCOUNT
    breaker (``provider_error``); after three, sender 2 held ``breaker_open``
    with no send. Now each is a typed ``budget_exhausted`` and sender 2 is
    screened.
    """

    from local_operator.team_bridge.settings import ScreeningBudget
    from tests.unit.team_bridge.test_batch_screening import _write_budget

    store, owner = _setup(tmp_path)
    _seed_model_config(tmp_path)
    _write_budget(tmp_path, ScreeningBudget(account=40, ambient=30, unaddressed=35, sender=8))
    _enable(tmp_path)
    seen = []
    for i in range(1, 9):
        monkeypatch.undo()
        result, wire = _real_cycle(
            tmp_path,
            monkeypatch,
            preferred=False,
            factory=PrivacyFactory(P_DELIVER),
            items=[_item(body=f"{CANARY_BODY} {i}", message_id=f"bmsg_x{i}")],
        )
        seen.append((result.disposition, result.hold_code, len(wire.sends)))
    assert seen[:4] == [("deliver", None, 1)] * 4
    assert all(code == "budget_exhausted" and sends == 0 for _, code, sends in seen[4:]), seen
    breaker = store.screening_breaker_state(owner) or {}
    assert int(breaker.get("failures") or 0) == 0, breaker
    other = _item(body="hello from someone else", message_id="bmsg_y1")
    other["sender"] = {"account_id": "acc_sender2"}
    monkeypatch.undo()
    result, wire = _real_cycle(
        tmp_path, monkeypatch, preferred=False, factory=PrivacyFactory(P_DELIVER), items=[other]
    )
    assert (result.disposition, result.hold_code) == ("deliver", None)
    assert len(wire.sends) == 1
