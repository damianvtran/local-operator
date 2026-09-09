"""PR 3's agent surfaces: the tool, the eval library, and the §6 stream fix.

The redaction tests here are the load-bearing ones. Each asserts the SAFE
outcome and is paired with a check that the guard is actually what produces it
— a redaction test that passes because the value never reached the filter is
worthless, and this series has produced false-passing harnesses repeatedly.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.secrets.promote import promote_session_credential
from local_operator.secrets.runtime import SecretsMapping, SecretValue
from local_operator.tools import builtin
from local_operator.tools.secret_tool import build_secret_tool, execute_secret
from local_operator.variables import VariableStore, redact_secret_values


class BrokerAwareStore(VariableStore):
    """A VariableStore carrying the §6 redaction sink.

    The sink (``register_redaction`` / ``redaction_values``) ships on the
    BROKER branch, not on main, and this PR consumes it defensively so it
    composes whichever order the two land in. That is exactly why the tests
    supply it as a subclass: asserting against a locally-defined seam proves
    the consumer works when the seam is present, while
    ``test_stream_values_without_the_seam_keep_the_injected_set`` covers the
    tree as it stands today. A test that called the real method would simply
    fail to run here and would prove nothing about either arrangement.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._registered: set[str] = set()

    def register_redaction(self, value: str) -> bool:
        trimmed = value.strip()
        if not trimmed or trimmed in self._registered:
            return False
        self._registered.add(trimmed)
        return True

    def redaction_values(self) -> list[str]:
        return [*self.credential_env().values(), *self._registered]

    def redact(self, text: str) -> str:
        return redact_secret_values(text, self.redaction_values())


# --------------------------------------------------------------------------
# The agent tool
# --------------------------------------------------------------------------


@pytest.fixture
def isolated(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every default-base store call inside tmp_path.

    ``config_root`` already redirects HOME and the config-dir override; the
    surfaces under test call ``open_store()`` with no base, so this asserts the
    redirect actually reaches them rather than trusting it.
    """
    from local_operator.secrets.keys import secrets_dir

    assert secrets_dir().is_relative_to(config_root), "store escaped the sandbox"
    return config_root


async def _call(op: str, **kwargs: object):
    return await execute_secret("call-1", {"op": op, **kwargs}, AbortSignal(), None, ToolContext())


@pytest.mark.asyncio
async def test_store_then_retrieve_never_returns_the_value(isolated: Path) -> None:
    """The whole inversion: retrieve confirms and instructs, and does not tell."""
    secret = "ghp_thisisthevalue_9f3a"
    stored = await _call("store", name="GH_TOKEN", value=secret, description="github")
    assert not stored.is_error, stored.text
    assert secret not in stored.text

    got = await _call("retrieve", name="GH_TOKEN")
    assert not got.is_error, got.text
    assert secret not in got.text
    # The receipt has to be actionable, or the model retries looking for bytes.
    assert "lop secret get GH_TOKEN" in got.text
    assert 'secrets["GH_TOKEN"]' in got.text

    listed = await _call("list")
    assert "GH_TOKEN" in listed.text and secret not in listed.text
    described = await _call("describe", name="GH_TOKEN")
    assert "github" in described.text and secret not in described.text


@pytest.mark.asyncio
async def test_retrieve_of_a_missing_secret_is_an_actionable_error(isolated: Path) -> None:
    await _call("store", name="PRESENT", value="v")
    missing = await _call("retrieve", name="ABSENT")
    assert missing.is_error
    assert "ABSENT" in missing.text


@pytest.mark.asyncio
async def test_retrieve_against_no_store_at_all_says_so(isolated: Path) -> None:
    """`retrieve` must not create an empty store and then report 'not found'."""
    result = await _call("retrieve", name="ANY")
    assert result.is_error
    assert "No secret store" in result.text
    from local_operator.secrets.keys import store_path

    assert not store_path().exists(), "retrieve created a store"


@pytest.mark.asyncio
async def test_store_refuses_to_overwrite_and_update_replaces(isolated: Path) -> None:
    await _call("store", name="K", value="first")
    clash = await _call("store", name="K", value="second")
    assert clash.is_error, "set must refuse an existing name"

    updated = await _call("update", name="K", value="second")
    assert not updated.is_error, updated.text
    from local_operator.secrets.access import open_store

    assert open_store().get("K") == b"second"


@pytest.mark.asyncio
async def test_delete_removes_and_missing_name_is_rejected(isolated: Path) -> None:
    await _call("store", name="K", value="v")
    assert not (await _call("delete", name="K")).is_error
    assert (await _call("retrieve", name="K")).is_error
    assert (await _call("retrieve")).is_error, "a name is required"


def test_the_tool_is_one_verb_tool_not_six() -> None:
    """The footprint ladder's whole point; asserted so a split shows up red."""
    tool = build_secret_tool(ToolContext())
    assert tool is not None
    ops = tool.parameters["properties"]["op"]["enum"]
    assert set(ops) == {"store", "retrieve", "list", "describe", "update", "delete"}


def test_the_tool_is_gated_and_the_registry_carries_it() -> None:
    from local_operator.tools.registry import DEFAULT_TOOL_NAMES, TOOL_BUILDERS

    assert TOOL_BUILDERS["secret"] is not None
    assert "secret" in DEFAULT_TOOL_NAMES


def test_the_builder_returns_none_when_the_store_cannot_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """createIf: a tree where the crypto stack will not import advertises nothing."""
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any):
        if name.startswith("local_operator.secrets"):
            raise ImportError("no crypto here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert build_secret_tool(ToolContext()) is None


# --------------------------------------------------------------------------
# The eval runtime library
# --------------------------------------------------------------------------


def test_secret_value_repr_redacts_but_str_does_not(isolated: Path) -> None:
    """`f"Bearer {token}"` must send the REAL token; only repr may lie."""
    value = SecretValue("abc123", "K")
    assert "abc123" not in repr(value)
    assert "[redacted]" in repr(value)
    assert str(value) == "abc123"
    assert f"Bearer {value}" == "Bearer abc123"
    assert value.encode() == b"abc123"


def test_mapping_retrieves_registers_and_reports_existence(isolated: Path) -> None:
    from local_operator.secrets.access import open_store
    from local_operator.secrets.runtime import registered_values, scrub

    store = open_store(create=True)
    store.set("API", b"sk-live-4417", description="d")

    mapping = SecretsMapping()
    value = mapping["API"]
    assert value == "sk-live-4417"
    # Registered BEFORE it was handed out, so the worker filter already knows.
    assert "sk-live-4417" in registered_values()
    assert scrub("token=sk-live-4417 end") == "token=[redacted] end"

    assert "API" in mapping
    assert "NOPE" not in mapping
    assert list(mapping) == ["API"]
    with pytest.raises(KeyError):
        mapping["NOPE"]
    assert mapping.get("NOPE") is None


def test_existence_check_does_not_write_a_retrieval_audit_row(isolated: Path) -> None:
    """`in` is not a read; recording it as one would corrupt the audit trail."""
    from local_operator.secrets.access import open_store

    store = open_store(create=True)
    store.set("API", b"v")
    before = sum(1 for e in store.audit_entries() if e.event == "get")
    assert "API" in SecretsMapping()
    after = sum(1 for e in store.audit_entries() if e.event == "get")
    assert after == before


def test_mapping_repr_does_not_touch_the_store(isolated: Path) -> None:
    assert "retrieves one value" in repr(SecretsMapping())


# --------------------------------------------------------------------------
# §6 redaction — the correctness-critical piece
# --------------------------------------------------------------------------


def test_pipe_redactor_accepts_bare_values_not_only_a_credential_map() -> None:
    """§6 values have no name in this process, so there is no map to pass."""
    redactor = builtin._PipeRedactor(["broker-value"])
    assert redactor.feed(b"a broker-value b", final=True) == b"a [redacted] b"


def test_stream_values_union_injected_and_registered() -> None:
    """The exact hole: a registered value must reach the STREAM filter."""
    store = BrokerAwareStore()
    store.store_credential("SESSION_KEY", "injected-secret")
    store.register_redaction("broker-fetched-secret")
    values = builtin._stream_redaction_values(store, store.credential_env())
    assert "injected-secret" in values
    assert "broker-fetched-secret" in values


def test_stream_values_without_the_seam_keep_the_injected_set() -> None:
    """A pre-broker / third-party store has no announcements to miss."""

    class Minimal:
        def credential_env(self) -> dict[str, str]:
            return {"A": "a-value"}

    values = builtin._stream_redaction_values(Minimal(), {"A": "a-value"})
    assert values == ["a-value"]
    assert values is not builtin._REDACTION_SEAM_BROKEN


def test_an_unreadable_redaction_sink_fails_closed() -> None:
    """Present-but-broken must NOT degrade to 'stream everything'."""

    class Broken:
        def redaction_values(self):
            raise RuntimeError("sink down")

    assert builtin._stream_redaction_values(Broken(), {}) is builtin._REDACTION_SEAM_BROKEN

    class WrongType:
        def redaction_values(self):
            return "not a sequence"

    assert builtin._stream_redaction_values(WrongType(), {}) is builtin._REDACTION_SEAM_BROKEN


def test_redactor_refresh_scrubs_a_value_registered_mid_stream() -> None:
    """A `$(lop secret get X)` value arrives DURING the command."""
    redactor = builtin._PipeRedactor([])
    early = redactor.feed(b"starting\n")
    assert b"starting" in early
    redactor.refresh(["late-secret"])
    assert redactor.feed(b"got late-secret ok", final=True) == b"got [redacted] ok"


@pytest.mark.asyncio
async def test_live_bash_stream_scrubs_a_registered_value(tmp_path: Path) -> None:
    """END TO END: a value registered on the store never paints in the stream.

    Registered BEFORE the command runs (the ordering §6's ack-before-reply
    invariant guarantees), and the command prints it repeatedly so a
    single-chunk fluke cannot pass this.
    """
    store = BrokerAwareStore()
    store.register_redaction("brokered-abcdef123456")
    context = ToolContext(cwd=str(tmp_path), variables=store)

    updates: list[str] = []
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "import sys,time\n"
        "for _ in range(40):\n"
        '    sys.stdout.write("line brokered-abcdef123456\\n"); sys.stdout.flush()\n'
    )
    result = await builtin.execute_bash(
        "bash-live",
        {"command": command},
        AbortSignal(),
        lambda update: updates.append(getattr(update.content[0], "text", "")),
        context,
    )
    assert not result.is_error, result.text
    # The value appeared 40 times; if the filter is off, it is in the result.
    assert "brokered-abcdef123456" not in result.text
    assert "[redacted]" in result.text
    for painted in updates:
        assert "brokered-abcdef123456" not in painted


@pytest.mark.asyncio
async def test_the_stream_guard_is_load_bearing(tmp_path: Path, monkeypatch) -> None:
    """THE RED. With the union reverted, the same value paints in the peek buffer.

    R2. The previous form was provably inert: it monkeypatched
    ``_stream_redaction_values``, collected the live updates but never asserted
    on them, and its only real assertion — an empty ``_PipeRedactor`` passing
    bytes through — is tautological. The pipe filter's guarantee is genuinely
    carried by the two peek-buffer tests below, so this RED is pointed at the
    same surface they guard: a background job's peek buffer, which the model
    reads with NO finished ToolResult and which the result path's
    ``redact_tool_result`` never touches. The union is reverted the way the
    reviewer demonstrated (``_stream_redaction_values`` returns the injected
    map alone), the command is run as a background job, and the secret is
    asserted to be PRESENT in the peek buffer — the leak must be shown to
    exist, not assumed. Mutation-verified: this test reds under the revert.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    store = BrokerAwareStore()
    store.register_redaction("loadbearing-9f3e1a")
    context = ToolContext(cwd=str(tmp_path), variables=store, jobs=manager, session_id="s")

    # The pre-fix behaviour exactly: the stream filter sees the INJECTED
    # credential map alone, ignoring broker registrations.
    monkeypatch.setattr(
        builtin,
        "_stream_redaction_values",
        lambda store, injected: [str(v) for v in injected.values() if v],
    )
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "import sys,time\n"
        'for ch in "loadbearing-9f3e1a":\n'
        "    sys.stdout.write(ch); sys.stdout.flush(); time.sleep(0.005)\n"
        'sys.stdout.write("\\n"); sys.stdout.flush()\n'
        "time.sleep(0.4)\n"
    )
    started = await builtin.execute_bash(
        "bash-red", {"command": command, "background": True}, AbortSignal(), None, context
    )
    job_id = (started.details or {})["job_id"]
    try:
        for _ in range(60):
            await asyncio.sleep(0.05)
            probe = manager.read_output(job_id, 0)
            if probe is not None and "loadbearing-9f3e1a" in probe[0]:
                break
        window = manager.read_output(job_id, 0)
        assert window is not None
        # The union is dead, so the pipe filter does not know the registered
        # value and the peek buffer carries it raw — the leak, demonstrated on
        # the channel the filter alone guards.
        assert "loadbearing-9f3e1a" in window[0], (
            "with the union reverted the secret must paint in the peek buffer; "
            "if it does not, the RED harness is broken"
        )
    finally:
        with contextlib.suppress(Exception):
            await manager.cancel(job_id)


@pytest.mark.asyncio
async def test_a_broken_sink_withholds_live_output_instead_of_leaking(
    tmp_path: Path,
) -> None:
    """Fail-closed, end to end, on the channel the stream guard owns.

    R3. The previous form asserted on ``result.text`` — a channel the finished
    result's ``redact_tool_result`` keeps clean regardless of what the pipe
    filter published, so it proved nothing about the fail-closed path it names.
    The live snapshot and the peek buffer are doubly guarded (both also pass
    ``_redact_tool_text``), so neither can isolate the pipe filter's branch
    either. The one observable no downstream guard re-cleans is the filter's
    own sink — exactly what it appended. This captures the sink and asserts
    the withheld notice — not the secret — is what landed there. Mutation-
    verified: reds under M3 (the sink failing OPEN, the reviewer-demonstrated
    revert). The command still completes: failing closed must cost the live
    view, not the run.
    """

    class Broken(BrokerAwareStore):
        def redaction_values(self):
            raise RuntimeError("sink down")

    store = Broken()
    context = ToolContext(cwd=str(tmp_path), variables=store, session_id="s")
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        'import sys\nfor _ in range(5): sys.stdout.write("would-be-secret\\n")\n'
    )
    sinks: list[Any] = []
    real_sink = builtin._BashOutput
    builtin._BashOutput = lambda: sinks.append(real_sink()) or sinks[-1]  # type: ignore[assignment]
    try:
        result = await builtin.execute_bash(
            "bash-closed", {"command": command}, AbortSignal(), None, context
        )
    finally:
        builtin._BashOutput = real_sink  # type: ignore[assignment]
    sink_bytes = b"".join(b"".join(s.chunks()) for s in sinks) if sinks else b""
    assert (
        b"would-be-secret" not in sink_bytes
    ), "the broken sink's bytes reached the pipe filter unfiltered (fail-open)"
    assert b"withheld" in sink_bytes, "the fail-closed notice must be what the filter emitted"
    # Failing closed costs the live view only — the command still completed.
    assert result.is_error is False


# --------------------------------------------------------------------------
# §5.4 — the persist route
# --------------------------------------------------------------------------


def test_promote_copies_the_session_credential_and_keeps_it(isolated: Path) -> None:
    store = VariableStore()
    store.store_credential("DEPLOY_KEY", "s3cr3t-value")
    result = promote_session_credential(store, "DEPLOY_KEY")
    assert result.ok, result.message
    assert "s3cr3t-value" not in result.message

    from local_operator.secrets.access import open_store

    assert open_store().get("DEPLOY_KEY") == b"s3cr3t-value"
    # Additive: the session copy still feeds bash children in flight.
    assert store.credential_env()["DEPLOY_KEY"] == "s3cr3t-value"


def test_promote_refuses_to_clobber_an_existing_long_term_secret(isolated: Path) -> None:
    from local_operator.secrets.access import open_store

    open_store(create=True).set("SHARED", b"the-original")
    store = VariableStore()
    store.store_credential("SHARED", "the-session-one")
    result = promote_session_credential(store, "SHARED")
    assert not result.ok and result.already_present
    assert open_store().get("SHARED") == b"the-original", "promotion overwrote a secret"


def test_promote_of_an_unknown_key_says_what_to_do(isolated: Path) -> None:
    result = promote_session_credential(VariableStore(), "NOPE")
    assert not result.ok
    assert "/credential NOPE" in result.message


def test_credential_command_parses_the_persist_verb() -> None:
    from local_operator.variables import parse_credential_command

    parsed = parse_credential_command("--persist github token")
    assert parsed.action == "persist" and parsed.key == "GITHUB_TOKEN"
    assert parse_credential_command("--persist").action == "error"


def test_ask_question_persist_requires_a_secret_question() -> None:
    from pydantic import ValidationError

    from local_operator.harness.types import AskQuestion

    ok = AskQuestion(id="API_KEY", question="key?", secret=True, persist=True)
    assert ok.persist
    with pytest.raises(ValidationError):
        AskQuestion(
            id="q",
            question="pick",
            options=[{"label": "a"}, {"label": "b"}],  # type: ignore[list-item]
            persist=True,
        )


def test_ask_secret_answer_persists_when_asked(isolated: Path) -> None:
    """The operator's explicit ask: `ask secret=true` reaches the same route."""
    from local_operator.harness.types import AskQuestion

    store = VariableStore()
    context = ToolContext(variables=store)
    question = AskQuestion(id="NPM_TOKEN", question="token?", secret=True, persist=True)
    reported = builtin._report_secret_answers([question], {"NPM_TOKEN": ["npm-abc-999"]}, context)
    answer = reported["NPM_TOKEN"]
    assert answer[0] == "NPM_TOKEN"
    assert "npm-abc-999" not in " ".join(answer), "the value rode the ask result"

    from local_operator.secrets.access import open_store

    assert open_store().get("NPM_TOKEN") == b"npm-abc-999"


def test_ask_secret_without_persist_stays_session_only(isolated: Path) -> None:
    from local_operator.harness.types import AskQuestion

    store = VariableStore()
    context = ToolContext(variables=store)
    question = AskQuestion(id="TMP_TOKEN", question="token?", secret=True)
    builtin._report_secret_answers([question], {"TMP_TOKEN": ["only-here"]}, context)
    assert store.credential_env()["TMP_TOKEN"] == "only-here"
    from local_operator.secrets.keys import store_path

    assert not store_path().exists(), "a non-persist ask created a long-term store"


def test_promotion_failure_does_not_lose_the_session_credential(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The secret the user just pasted must survive a broken long-term store."""
    from local_operator.harness.types import AskQuestion

    def explode(*args: object, **kwargs: object):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("local_operator.secrets.promote.promote_session_credential", explode)
    store = VariableStore()
    context = ToolContext(variables=store)
    question = AskQuestion(id="K", question="k?", secret=True, persist=True)
    reported = builtin._report_secret_answers([question], {"K": ["kept"]}, context)
    assert store.credential_env()["K"] == "kept"
    assert "session only" in " ".join(reported["K"])


@pytest.mark.asyncio
async def test_a_value_registered_MID_command_is_still_scrubbed(tmp_path: Path) -> None:
    """The real §6 shape: the child fetches the secret while it is running.

    `$(lop secret get X)` resolves inside the child, so the session learns the
    value from the broker DURING the command — after the pipe filter was
    built. A filter constructed once at spawn misses exactly this, which is why
    the pump re-reads the value set on every chunk.
    """
    store = BrokerAwareStore()
    context = ToolContext(cwd=str(tmp_path), variables=store)

    # Print a marker, wait for the registration, then print the secret.
    flag = tmp_path / "registered"
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "import sys,time\n"
        'sys.stdout.write("started\\n"); sys.stdout.flush()\n'
        "for _ in range(400):\n"
        f"    if __import__('os').path.exists({str(flag)!r}): break\n"
        "    time.sleep(0.01)\n"
        'sys.stdout.write("fetched midstream-9d2f1a\\n"); sys.stdout.flush()\n'
    )

    async def register_once_running() -> None:
        await asyncio.sleep(0.05)
        store.register_redaction("midstream-9d2f1a")
        flag.write_text("go")

    task = asyncio.create_task(register_once_running())
    result = await builtin.execute_bash(
        "bash-mid", {"command": command}, AbortSignal(), None, context
    )
    await task
    assert not result.is_error, result.text
    assert "started" in result.text, "the harness never ran the command"
    assert "midstream-9d2f1a" not in result.text
    assert "[redacted]" in result.text


@pytest.mark.asyncio
async def test_background_job_peek_scrubs_a_registered_value(tmp_path: Path) -> None:
    """The peek buffer is fed by the PIPE filter, not by the result path.

    `jobs(op='peek')` reads what `_mirror` appended while the command ran, so
    this is the surface where `_PipeRedactor` is the only guard — the model
    sees these bytes without a finished ToolResult ever existing. The secret is
    written one byte at a time so it necessarily straddles pipe reads, which is
    the exact case the redactor's held-back suffix exists for.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    store = BrokerAwareStore()
    store.register_redaction("peeked-7c1e5b")
    context = ToolContext(cwd=str(tmp_path), variables=store, jobs=manager, session_id="s")

    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "import sys,time\n"
        'for ch in "peeked-7c1e5b":\n'
        "    sys.stdout.write(ch); sys.stdout.flush(); time.sleep(0.005)\n"
        'sys.stdout.write("\\n"); sys.stdout.flush()\n'
        "time.sleep(0.4)\n"
    )
    started = await builtin.execute_bash(
        "bash-bg", {"command": command, "background": True}, AbortSignal(), None, context
    )
    job_id = (started.details or {})["job_id"]
    for _ in range(60):
        await asyncio.sleep(0.05)
        probe = manager.read_output(job_id, 0)
        if probe is not None and probe[0].strip():
            break

    window = manager.read_output(job_id, 0)
    assert window is not None
    mirrored = window[0]
    assert "peeked-7c1e5b" not in mirrored, "the secret painted in the peek buffer"
    assert "[redacted]" in mirrored
    with contextlib.suppress(Exception):
        await manager.cancel(job_id)


@pytest.mark.asyncio
async def test_peek_scrubs_a_value_registered_AFTER_the_command_started(
    tmp_path: Path,
) -> None:
    """The full §6 shape, on the surface where the pipe filter is the only guard.

    `$(lop secret get X)` resolves INSIDE the child, so the session is told the
    value while the command is already streaming — after the pipe filter was
    constructed. Combined with a background job (whose peek buffer never passes
    through the result path) this is what makes the per-chunk `refresh` load
    bearing rather than decorative.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    store = BrokerAwareStore()
    context = ToolContext(cwd=str(tmp_path), variables=store, jobs=manager, session_id="s")

    flag = tmp_path / "registered"
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "import os,sys,time\n"
        'sys.stdout.write("started\\n"); sys.stdout.flush()\n'
        "for _ in range(400):\n"
        f"    if os.path.exists({str(flag)!r}): break\n"
        "    time.sleep(0.01)\n"
        'for ch in "afterreg-2b8d4f":\n'
        "    sys.stdout.write(ch); sys.stdout.flush(); time.sleep(0.005)\n"
        'sys.stdout.write("\\n"); sys.stdout.flush()\n'
        "time.sleep(0.5)\n"
    )
    started = await builtin.execute_bash(
        "bash-bg2", {"command": command, "background": True}, AbortSignal(), None, context
    )
    job_id = (started.details or {})["job_id"]

    # Wait for the command to be streaming, THEN register — the ordering under
    # test. Waits on the observed output, never on the clock.
    for _ in range(60):
        await asyncio.sleep(0.05)
        probe = manager.read_output(job_id, 0)
        if probe is not None and "started" in probe[0]:
            break
    store.register_redaction("afterreg-2b8d4f")
    flag.write_text("go")

    for _ in range(80):
        await asyncio.sleep(0.05)
        probe = manager.read_output(job_id, 0)
        if probe is not None and probe[0].count("\n") >= 2:
            break

    window = manager.read_output(job_id, 0)
    assert window is not None
    assert "started" in window[0], "the harness never ran the command"
    assert "afterreg-2b8d4f" not in window[0], "a mid-command registration painted in peek"
    with contextlib.suppress(Exception):
        await manager.cancel(job_id)


# --------------------------------------------------------------------------
# The eval worker's scrubbing of library-retrieved values
# --------------------------------------------------------------------------


def test_worker_scrub_is_free_until_a_secret_is_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No import, no cost — and provably nothing to scrub in that state."""
    from local_operator.tools import eval_worker

    monkeypatch.delitem(sys.modules, "local_operator.secrets.runtime", raising=False)
    assert eval_worker._scrub_secrets("anything at all") == "anything at all"


def test_worker_scrub_removes_a_retrieved_value(isolated: Path) -> None:
    from local_operator.secrets.access import open_store
    from local_operator.secrets.runtime import SecretsMapping
    from local_operator.tools import eval_worker

    open_store(create=True).set("K", b"worker-value-8812")
    assert SecretsMapping()["K"] == "worker-value-8812"
    assert eval_worker._scrub_secrets("out: worker-value-8812") == "out: [redacted]"


def test_worker_scrub_fails_closed_when_the_filter_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken scrubber must withhold, never publish."""
    from local_operator.tools import eval_worker

    class Exploding:
        @staticmethod
        def scrub(text: str) -> str:
            raise RuntimeError("filter down")

    monkeypatch.setitem(sys.modules, "local_operator.secrets.runtime", Exploding)
    out = eval_worker._scrub_secrets("would-be-secret")
    assert "would-be-secret" not in out
    assert "withheld" in out


def test_worker_response_scrubs_every_model_visible_channel(isolated: Path) -> None:
    """stdout, stderr, result and display all pass the filter."""
    from local_operator.secrets.access import open_store
    from local_operator.secrets.runtime import SecretsMapping
    from local_operator.tools import eval_worker

    open_store(create=True).set("K", b"chan-value-4417")
    assert SecretsMapping()["K"] == "chan-value-4417"

    namespace: dict[str, object] = {"__name__": "__eval__"}
    response = eval_worker._handle(
        namespace,
        {
            "id": "r1",
            "code": (
                "import sys\n"
                "v = 'chan-value-4417'\n"
                "print(v)\n"
                "sys.stderr.write(v)\n"
                "display(v)\n"
                "v\n"
            ),
        },
    )
    blob = "".join(
        [
            str(response["stdout"]),
            str(response["stderr"]),
            str(response["result"]),
            "".join(response["display"]),  # type: ignore[arg-type]
        ]
    )
    assert "chan-value-4417" not in blob
    assert "[redacted]" in blob


def test_secrets_is_prebound_in_the_worker_namespace_without_importing_crypto() -> None:
    """The documented `secrets["NAME"]` form works with no import line."""
    from local_operator.tools import eval_worker

    proxy = eval_worker._LazySecrets()
    assert "retrieves one value" in repr(proxy)


@pytest.mark.asyncio
async def test_a_crash_with_a_secret_on_real_fd2_is_scrubbed(isolated: Path) -> None:
    """R1 regression: the worker's REAL fd-2 tail is scrubbed with the ledger.

    ``redirect_stderr`` swaps the ``sys.stderr`` OBJECT, not the fd, so
    ``os.write(2, ...)`` (the deliberate form) and any inherited-stderr
    subprocess (``curl -v`` with a token in the Authorization header — the
    accidental form) write straight past the in-worker ledger. The value set
    lives in the worker, so it is published to the parent on a side channel at
    the moment of retrieval; this drives a REAL worker crash with a secret on
    fd 2 and asserts the returned tool result carries the marker, not the
    bytes.
    """
    from local_operator.secrets.access import open_store
    from local_operator.tools import eval as eval_tool

    open_store(create=True).set("CRASH_DEMO", b"CRASHLEAK-f00dcafe-9517")
    tool = eval_tool.build_eval_tool()
    context = ToolContext(cwd="/tmp", session_id="crash-leak")
    code = (
        "import os\n"
        "token = secrets['CRASH_DEMO']\n"
        "os.write(2, f'BOOM got {token}'.encode())\n"
        "os._exit(3)\n"
    )
    result = await tool.execute("c1", {"code": code}, AbortSignal(), None, context)
    assert result.is_error, "a hard worker crash must surface as an error"
    assert "crashed" in result.text
    assert "CRASHLEAK-f00dcafe-9517" not in result.text, "the fd-2 tail leaked raw"
    assert "BOOM got [redacted]" in result.text


@pytest.mark.asyncio
async def test_list_on_a_machine_with_no_store_is_not_an_error(isolated: Path) -> None:
    """`list` is the orienting verb; an error there reads as 'broken'."""
    result = await _call("list")
    assert not result.is_error, result.text
    assert "No secrets are stored" in result.text
