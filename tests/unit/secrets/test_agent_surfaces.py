"""PR 3's agent surfaces: the tool, the eval library, and the §6 stream fix.

The redaction tests here are the load-bearing ones. Each asserts the SAFE
outcome and is paired with a check that the guard is actually what produces it
— a redaction test that passes because the value never reached the filter is
worthless, and this series has produced false-passing harnesses repeatedly.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import re
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import local_operator
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.secrets.promote import promote_session_credential
from local_operator.secrets.protocol import PROTOCOL_VERSION
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


@pytest.mark.asyncio
async def test_the_agent_tool_cannot_describe_or_delete_a_provider_row(isolated: Path) -> None:
    """The agent surface must not reach the reserved namespace (Q-1/R2).

    ``SecretStore.describe``/``delete`` now carry a ``role`` that defaults to the
    restrictive ``"agent"``, and the tool passes nothing — so an agent (or a
    prompt-injected call) cannot ENUMERATE a provider key's metadata nor DESTROY
    it through its ordinary ``secret`` tool. The delete half is the destructive
    one: the value is retained nowhere, so an unguarded delete removed the
    credential a provider authenticates with. Verified through the real tool
    dispatch, and paired with an existence check so a refusal that still removed
    the row cannot pass.
    """
    from local_operator.providers.registry import (
        provider_secret_value,
        store_provider_key,
    )

    store_provider_key("OPENROUTER_API_KEY", "sk-secret", base=isolated)

    described = await _call("describe", name="LOP_PROVIDER_OPENROUTER_API_KEY")
    assert described.is_error, described.text
    assert "reserved" in described.text

    deleted = await _call("delete", name="LOP_PROVIDER_OPENROUTER_API_KEY")
    assert deleted.is_error, deleted.text
    # Still stored and still resolving — the guard must not be a delete that
    # reports failure after removing the record.
    assert provider_secret_value("OPENROUTER_API_KEY", base=isolated) == "sk-secret"


def test_the_agent_tool_list_omits_provider_rows_the_operator_list_keeps(
    isolated: Path,
) -> None:
    """Enumeration is public to the OPERATOR, not to the agent tool (PATCH C, R2).

    The agent surface is the one a model drives, so provider rows are filtered out
    of ITS list: a name the model never learns is not one it can name back into the
    destructive verbs, and the map to a machine credential is exactly what the prefix
    exists to withhold. The operator's own ``lop secret list`` still shows them — it
    reads the store list directly, which is what the second assertion pins, so the
    filter cannot quietly become a store-level change that hides rows from the person
    who owns them.
    """
    import asyncio

    from local_operator.providers.registry import store_provider_key
    from local_operator.secrets.access import open_store

    store_provider_key("OPENROUTER_API_KEY", "sk-secret", base=isolated)
    asyncio.run(_call("store", name="GH_TOKEN", value="ghp-secret", description="github"))

    listed = asyncio.run(_call("list"))
    assert not listed.is_error, listed.text
    assert "GH_TOKEN" in listed.text
    assert "LOP_PROVIDER" not in listed.text

    operator_view = [record.name for record in open_store().list()]
    assert "LOP_PROVIDER_OPENROUTER_API_KEY" in operator_view


def test_the_agent_tool_list_says_so_when_only_provider_rows_exist(
    isolated: Path,
) -> None:
    """The filtered-empty case must not read as "you have stored nothing".

    The guidance tells the agent to check what exists before asking for a credential;
    if the filter collapsed to the empty-store message, a host holding only provider
    rows would invite the model to ask the user to store one it is deliberately not
    shown.
    """
    import asyncio

    from local_operator.providers.registry import store_provider_key

    store_provider_key("OPENROUTER_API_KEY", "sk-secret", base=isolated)

    listed = asyncio.run(_call("list"))
    assert not listed.is_error, listed.text
    assert "No secrets are stored yet" not in listed.text
    # The message NAMES the prefix to explain itself, so the assertion is against
    # the row's actual name — the thing the filter exists to withhold.
    assert "LOP_PROVIDER_OPENROUTER_API_KEY" not in listed.text


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

    # The refusal is installed and REMOVED around the one call under test
    # rather than left to monkeypatch's unwind. The broker (#815) added an
    # autouse teardown in tests/conftest.py that imports
    # `local_operator.secrets.client` to sweep stray broker sockets, and
    # fixture teardown runs BEFORE monkeypatch undoes a setattr made in the
    # test body — so a still-installed refusal turns this passing test into a
    # teardown ERROR. Scoping it here keeps the assertion identical while
    # leaving no global hook alive past the line that needs it.
    monkeypatch.setattr(builtins, "__import__", refuse)
    try:
        assert build_secret_tool(ToolContext()) is None
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)


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


def test_mapping_contract_holds_over_the_broker_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SAME `Mapping` contract, served by a REAL broker (round-4 R11/Q5).

    **Why this exists beside the test directly above it, which asserts the same
    three things.** That one runs under `isolated`, which never starts a
    broker, so `retrieve_secret` falls through to the local decrypt — it pins
    the contract on the branch this PR did NOT change. Routing the mapping
    through the broker broke it on the branch that has no test: the broker
    reported every store failure as one flat `"store"` code, the class was
    erased, `__getitem__`'s `except SecretNotFound` stopped matching, and
    `secrets.get("ABSENT", "dflt")` RAISED instead of returning the default.
    The suite stayed green throughout.

    So the assertions are duplicated ON PURPOSE. The contract is one contract
    and the two paths must be indistinguishable to a caller; a test that only
    ever exercises one of them cannot notice when they diverge.
    """
    pytest.importorskip("local_operator.secrets.client")

    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.secrets import client
    from local_operator.secrets.broker import SecretBroker
    from local_operator.secrets.crypto import generate_master_key
    from local_operator.secrets.keys import key_path, write_private_file
    from local_operator.secrets.store import SecretStore

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))

    key = generate_master_key()
    store_on_disk = SecretStore(key, base=root)
    store_on_disk.initialize()
    store_on_disk.set("API", b"sk-live-4417", description="d")
    write_private_file(key_path(root), key)

    broker = SecretBroker(root, key_provider=lambda: key, idle_shutdown_s=0)
    broker.start()
    channel = None
    stop = threading.Event()
    try:
        # **A running broker is NOT enough to reach the broker path, and
        # discovering that is part of this finding.** Without a registered
        # session the broker denies the retrieval on ancestry, and §8's keyfile
        # fallback quietly serves it from the local decrypt — so a version of
        # this test that merely started a broker PASSED against the unfixed
        # head, reproducing the very "green because it exercised the other
        # branch" defect it was written to catch. The registration is what puts
        # the retrieval on the broker path, exactly as the TUI's startup
        # registration does in production.
        channel = client.register_session(root, session_id="mapping-contract")
        assert channel is not None, "registration failed; the broker path is unreachable"

        def drain() -> None:
            while not stop.is_set():
                notice = client.read_notification(channel)
                if notice is None:
                    return
                client.acknowledge(channel)

        # A real session answers §6 notifications; an unanswered notice would
        # make the broker refuse as unredactable and mask the contract.
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()

        assert client.is_running(root), "no broker; this test would re-cover the local path"
        mapping = SecretsMapping(root)
        assert mapping["API"] == "sk-live-4417"

        assert mapping.get("ABSENT", "dflt") == "dflt"
        assert mapping.get("ABSENT") is None
        with pytest.raises(KeyError):
            mapping["MISSING"]
        assert "MISSING" not in mapping
        assert "API" in mapping
    finally:
        stop.set()
        if channel is not None:
            channel.close()
        broker.stop(timeout=5)


def test_a_stale_broker_refuses_rather_than_serving_unnotified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LIVE broker this runtime cannot speak to must not degrade (round-4 Q4).

    **The case is a runtime update under a live daemon**, which `AGENTS.md`
    calls routine: the new client speaks a protocol the running broker does
    not. `retrieve_secret`'s docstring named this as the accepted cost and said
    it raises — but `is_running` caught the version refusal and answered
    `False`, laundering "live but stale" into "unreachable", so the fallback
    fired and the value came back through the UNNOTIFIED local decrypt. That is
    a full reproduction of Q3, and it arms itself at exactly the moment a
    protocol bump ships.

    `PROTOCOL_VERSION` has only ever been 1, so the skew is produced here by
    running a real broker whose module constant is bumped — a real daemon, a
    real socket and a real refusal, rather than a mock that would only restate
    the assumption being tested.
    """
    pytest.importorskip("local_operator.secrets.client")

    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.secrets import broker as broker_module
    from local_operator.secrets import client
    from local_operator.secrets.access import retrieve_secret
    from local_operator.secrets.crypto import generate_master_key
    from local_operator.secrets.errors import BrokerIncompatible
    from local_operator.secrets.keys import key_path, write_private_file
    from local_operator.secrets.store import SecretStore

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))

    secret = b"stale-broker-payload-71c2"
    key = generate_master_key()
    store_on_disk = SecretStore(key, base=root)
    store_on_disk.initialize()
    store_on_disk.set("STALE_TOKEN", secret, description="q4")
    write_private_file(key_path(root), key)

    # The daemon speaks a version this client does not. Patched on the broker
    # module only, so the CLIENT keeps the real constant and the two disagree
    # exactly as they would across an update.
    monkeypatch.setattr(broker_module, "PROTOCOL_VERSION", PROTOCOL_VERSION + 1)
    stale = broker_module.SecretBroker(root, key_provider=lambda: key, idle_shutdown_s=0)
    stale.start()
    try:
        with pytest.raises(BrokerIncompatible) as refusal:
            retrieve_secret("STALE_TOKEN", root)
        # Actionable, not merely fatal: the operator is told what to run, and
        # the pid is present because `status` cannot answer through the same
        # version gate.
        assert "lop secret broker restart" in str(refusal.value)
        assert isinstance(refusal.value.pid, int)

        # The distinction that carries the safety property: an unreachable
        # broker still degrades, a live one does not.
        with pytest.raises(BrokerIncompatible):
            client.is_running(root)
        status = client.broker_status(root)
        assert status is not None and status.get("incompatible") is True

        # Metadata is the deliberate exception: it hands out no bytes and owes
        # no §6 notice, so the read-only surfaces stay up under skew.
        assert "STALE_TOKEN" in SecretsMapping(root)
    finally:
        stale.stop(timeout=5)


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
    bytes through — is tautological. The union is reverted the way the reviewer
    demonstrated (``_stream_redaction_values`` returns the injected map alone),
    the command is run as a background job, and the secret is asserted to be
    PRESENT in the bytes the pipe filter emitted — the leak must be shown to
    exist, not assumed. Mutation-verified: this test reds under the revert.

    **Why the assertion reads the pipe's OWN sink rather than the peek buffer.**
    It used to assert on the peek window, and that stopped being the channel the
    union exclusively guards when the filter learned to release whole lines:
    the peek is fed by a second pass (``_redact_tool_text``) that only ever saw
    fragments while the filter published per-read, so a split value painted it —
    and now that the filter hands it one complete line, that pass matches the
    value and masks it. The union's own guarantee is about what the FILTER
    publishes, so this reads the sink the filter publishes into. The peek
    surface keeps its coverage in the tests below, which assert the masked
    value and the withheld notice.
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
    # Spy on the CAPTURE SINK rather than replacing the class: the background
    # path does ``isinstance(chunks, _BashOutput)`` on the module global, so a
    # stand-in class makes the tool fail for a reason that has nothing to do
    # with redaction. ``append`` is where the filter's bytes land.
    published: list[bytes] = []
    real_append = builtin._BashOutput.append

    def spy_append(self: Any, chunk: bytes) -> None:
        published.append(bytes(chunk))
        real_append(self, chunk)

    monkeypatch.setattr(builtin._BashOutput, "append", spy_append)
    started = await builtin.execute_bash(
        "bash-red", {"command": command, "background": True}, AbortSignal(), None, context
    )
    job_id = (started.details or {})["job_id"]
    try:
        for _ in range(60):
            await asyncio.sleep(0.05)
            if "loadbearing-9f3e1a" in b"".join(published).decode("utf-8", "replace"):
                break
        # The union is dead, so the pipe filter does not know the registered
        # value and it publishes the line raw — the leak, demonstrated on the
        # channel the filter alone guards.
        assert "loadbearing-9f3e1a" in b"".join(published).decode("utf-8", "replace"), (
            "with the union reverted the secret must reach the pipe filter's output; "
            "if it does not, the RED harness is broken"
        )
        # ...and the model-facing window does NOT carry it anyway, because the
        # peek re-runs the store's composed scrub on whatever the filter
        # published. Asserted here rather than assumed: it is the property that
        # makes the peek safe even when the live filter's own value set is not,
        # and it is the reason this RED no longer lives on the peek buffer.
        window = manager.read_output(job_id, 0)
        assert window is not None
        assert "loadbearing-9f3e1a" not in window[0]
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


def test_promote_of_an_unknown_key_advises_a_gesture_that_still_works(isolated: Path) -> None:
    """The advice must name something the operator can actually DO.

    It used to say ``/credential NOPE``, which the typed capture retired: the
    space after the token opens a masked span, so typing a key name mints it as
    a short secret instead of reaching the ``<KEY>`` prompt (QA round 1, Q1).
    Advice that cannot be followed is worse than none — the operator would have
    typed it and watched their key name become a credential. So this pins the
    property (the message routes them to a working gesture) rather than the old
    literal, and explicitly refuses the retired one.
    """
    result = promote_session_credential(VariableStore(), "NOPE")
    assert not result.ok
    assert "NOPE" in result.message, "it still names the key they asked for"
    assert "/credential" in result.message, "and the command that hands one over"
    assert "/credential NOPE" not in result.message, "but never the untypable form"


#: Advice that tells the operator to TYPE the retired ``/credential <KEY>``
#: form. The verb is what makes it advice rather than a description, and the
#: argument is what makes it the RETIRED form rather than the live gesture.
#:
#: THE KEY HALF IS DELIBERATELY CASE-SENSITIVE. Compiling the whole pattern
#: with ``re.IGNORECASE`` makes ``[A-Z]`` match lowercase too, and the rule then
#: fires on ordinary prose like "run /credential from the owner session" — which
#: it did on the first draft of this guard. Only the VERBS are spelled both
#: ways.
_TYPE_THE_RETIRED_FORM = re.compile(
    r"(?:[Uu]se|[Tt]ype|[Rr]un|[Ee]nter|[Tt]ry|[Ss]ay)\s+(?:the\s+)?[`'\"]*"
    r"/credential\s+(?!--)(?:<[A-Za-z_]+>|[A-Z][A-Z0-9_]{2,})"
)


def test_no_shipped_string_tells_the_operator_to_type_the_retired_form() -> None:
    """The RETIREMENT SWEEP itself, as a mechanism rather than a grep somebody ran.

    Q1 retired the typed ``/credential <KEY>`` form and corrected the usage
    text, the ``--persist`` advice and the design doc. A live viewer notice was
    missed, and the miss was found INDEPENDENTLY by the reviewer and by QA on
    the same line (review round 2, R5; QA round 2, Q5) — two agents finding one
    site means the sweep was the defect, not just the site. So the sweep is
    pinned here and runs on every commit.

    Why it matters more than a stale string usually would: following that advice
    mints the operator's KEY NAME as a short secret, and on the viewer path it
    fires on the submit seam, so the chip the retyping mints re-enters the same
    branch and reprints the same advice — the operator LOOPS.

    Scoped to shipped strings under ``local_operator/`` — tests and design docs
    describe the retired form deliberately, and a guard that forbade naming it
    would forbid explaining why it was retired.
    """
    package = Path(local_operator.__file__).parent
    offenders: list[str] = []
    scanned = 0
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):  # pragma: no cover - unreadable file
            continue
        scanned += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            found = _TYPE_THE_RETIRED_FORM.search(node.value)
            if found is not None:
                offenders.append(
                    f"{path.relative_to(package.parent)}:{node.lineno}: {found.group(0)!r}"
                )

    # A ZERO FROM A DEAD INSTRUMENT READS EXACTLY LIKE A PASS, and this rule is
    # narrow enough to break silently, so the detector is proven live in the
    # same run against the exact string that was missed.
    missed = (
        "inline /credential needs the session that runs the tools; "
        "use /credential <KEY> here and the secret is stored on the owner"
    )
    assert _TYPE_THE_RETIRED_FORM.search(missed), "the detector is dead; its zero means nothing"
    assert scanned > 100, f"the sweep only reached {scanned} modules"

    assert not offenders, (
        "these shipped strings tell the operator to TYPE the retired "
        "`/credential <KEY>` form, which mints their key name as a secret:\n  "
        + "\n  ".join(offenders)
    )


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


def test_streamed_frames_mask_a_value_split_across_two_writes(isolated: Path) -> None:
    """The CHUNK BOUNDARY, on the one streamed surface that had no window.

    The sink emitted one frame per ``write``, masked on its own, and a value
    split across two of them is present in NEITHER half — so no ``replace``
    fired in either and both halves were published. That is a leak rather than a
    technicality because of what reads the frames: the parent appends each one to
    a background job's tail and ``jobs(op='peek')`` JOINS them back into a single
    string the model reads, so two clean halves make one dirty whole.

    Asserted on the JOINED frames, which is the reading that has to be clean.
    """
    from local_operator.secrets.access import open_store
    from local_operator.secrets.runtime import SecretsMapping
    from local_operator.tools import eval_worker

    open_store(create=True).set("K", b"frame-value-7712")
    assert SecretsMapping()["K"] == "frame-value-7712"

    frames: list[str] = []
    sink = eval_worker._StreamingTextIO(eval_worker.STREAM_CHAR_LIMIT, frames.append)
    sink.write("out: frame-va")
    sink.write("lue-7712\n")

    # Read the JOINED frames and no further: this is the reading the model gets
    # from `jobs(op='peek')` while the cell is still running, and it is the one
    # that has to be clean. Deliberately no end-of-cell flush here — a value that
    # only survived because a later call tidied up is still a value the live view
    # published.
    joined = "".join(frames)
    assert "frame-value-7712" not in joined
    assert "frame-va" not in joined and "lue-7712" not in joined, "not even in halves"


def test_streamed_frames_release_the_held_tail_when_the_cell_ends(isolated: Path) -> None:
    """Delayed, never dropped: the window's hold is released at the end.

    The window is the price of the guarantee above, and the failure mode it can
    introduce is its own defect: a cell whose LAST characters sit inside the hold
    would lose them from the live view. ``release_held`` is what keeps the tail —
    and the final response carries the whole text either way, so this is about
    liveliness rather than about coverage.
    """
    from local_operator.secrets.access import open_store
    from local_operator.secrets.runtime import SecretsMapping
    from local_operator.tools import eval_worker

    open_store(create=True).set("K", b"tail-value-9031")
    assert SecretsMapping()["K"] == "tail-value-9031"

    frames: list[str] = []
    sink = eval_worker._StreamingTextIO(eval_worker.STREAM_CHAR_LIMIT, frames.append)
    sink.write("first line\nlast line")
    assert "last line" not in "".join(frames), "the tail should still be held open"
    sink.release_held()
    joined = "".join(frames)
    assert "last line" in joined, "the held tail was dropped, not delayed"
    assert REDACTION_MARKER not in joined, "nothing here needed masking"

    # And the value case in the same shape: the tail is released MASKED rather
    # than dropped, which is the whole reason the window is safe to use.
    frames = []
    sink = eval_worker._StreamingTextIO(eval_worker.STREAM_CHAR_LIMIT, frames.append)
    sink.write("before tail-value-9031")
    sink.release_held()
    joined = "".join(frames)
    assert "tail-value-9031" not in joined
    assert REDACTION_MARKER in joined


def test_streamed_frames_are_not_delayed_without_a_registered_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that never touched a secret streams exactly as it always did.

    The hold window is sized from the values the ledger knows, so with none it is
    zero — a per-write mask with no latency, which is the property that keeps
    this off the path of every ordinary cell.
    """
    from local_operator.tools import eval_worker

    monkeypatch.delitem(sys.modules, "local_operator.secrets.runtime", raising=False)
    frames: list[str] = []
    sink = eval_worker._StreamingTextIO(eval_worker.STREAM_CHAR_LIMIT, frames.append)
    sink.write("no newline, mid-line, short")
    assert "".join(frames) == "no newline, mid-line, short"


def test_streamed_frames_fail_closed_when_the_ledger_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable ledger withholds the frame rather than publishing it.

    An empty value set means "nothing to mask", which is the one reading that
    would publish a secret the session is holding, so a fault has to be
    distinguishable from emptiness — the same shape as the settled response's
    fail-closed path, and sticky for the same reason the bash pipe filter's is.
    """
    from local_operator.tools import eval_worker

    class Exploding:
        @staticmethod
        def registered_values() -> list[str]:
            raise RuntimeError("ledger down")

    monkeypatch.setitem(sys.modules, "local_operator.secrets.runtime", Exploding)
    frames: list[str] = []
    sink = eval_worker._StreamingTextIO(eval_worker.STREAM_CHAR_LIMIT, frames.append)
    sink.write("would-be-secret")
    sink.write("a later write too")
    sink.release_held()
    joined = "".join(frames)
    assert "would-be-secret" not in joined
    assert "a later write too" not in joined, "the withhold must be sticky"
    assert "withheld" in joined, "and it must say so rather than look empty"


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
async def test_a_retry_loop_of_one_secret_does_not_wedge_the_kernel(isolated: Path) -> None:
    """R9 regression: the ORDINARY retry shape, not a pathological big value.

    The scrub channel is a 64 KiB pipe the worker writes from inside
    ``register`` — before the value reaches the cell — so anything that parks
    the write parks the retrieval, and the kernel dies at the timeout taking
    all session state with it. Nothing caches a retrieval, so a loop that
    re-fetches ONE token per attempt (a retry, a paginated API walk) publishes
    once per iteration while the ledger stays size 1: at 200 bytes a record
    that filled the pipe at ~319 iterations.

    This asserts the CELL COMPLETES. It is the availability half of the
    invariant the fix is built on: a dropped scrub record degrades to an
    unscrubbed crash tail (the documented pre-R1 fallback), never to a blocked
    retrieval. 400 iterations is comfortably past the old wedge point, and the
    generous timeout means a FAIL here is a park, not a slow machine.
    """
    from local_operator.secrets.access import open_store
    from local_operator.tools import eval as eval_tool

    open_store(create=True).set("RETRY_TOKEN", b"R" * 200)
    tool = eval_tool.build_eval_tool()
    context = ToolContext(cwd="/tmp", session_id="retry-loop")
    code = "n = 0\nfor _ in range(400):\n    t = secrets['RETRY_TOKEN']\n    n += 1\nn\n"
    result = await tool.execute("c1", {"code": code, "timeout": 60}, AbortSignal(), None, context)
    assert not result.is_error, f"the retrieval loop wedged: {result.text[:400]}"
    assert "400" in result.text
    await eval_tool.close_session_kernel("retry-loop")


def test_publication_dedups_with_the_ledger_not_beside_it() -> None:
    """R9 regression: re-registering one value publishes exactly once.

    ``add()`` on a set dedups for free, which hid that publication sat OUTSIDE
    it: five registrations of one identical value produced five publishes
    against a ledger of size 1. That asymmetry is what turned an ordinary
    retry loop into hundreds of records for a scrub set with one entry in it,
    and it is the half of the fix that removes the pressure rather than
    surviving it.
    """
    from local_operator.secrets import runtime

    published: list[str] = []
    ledger = runtime._RedactionLedger()
    original = runtime._PUBLISH_HOOK
    runtime.set_publish_hook(published.append)
    try:
        for _ in range(5):
            ledger.register("one-identical-value")
        ledger.register("a-second-value")
    finally:
        runtime.set_publish_hook(original)

    assert published == ["one-identical-value", "a-second-value"]
    assert sorted(ledger.values()) == ["a-second-value", "one-identical-value"]


def test_a_full_scrub_pipe_drops_the_record_instead_of_parking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9: past the channel's capacity, publication must return, not block.

    The acceptance criterion stated as an assertion, and it takes the REAL
    setup path: the fd is handed over the way the parent hands it over, so the
    non-blocking mode under test is the one ``_install_scrub_channel`` sets
    rather than one the test quietly supplied for it. That ordering matters —
    a test that sets the mode itself passes against a worker that never sets
    it at all.

    The mode assertion is deliberately first. It is the property that makes
    the park impossible, and checking it fails FAST where the behaviour it
    protects can only fail by hanging forever (a park is not an exception and
    no ``except`` or assertion can catch it). With that established, filling
    the pipe and publishing exercises the drop: the record is lost, which
    downgrades the crash tail for that one value to the documented pre-R1
    unscrubbed behaviour, and the retrieval stays free to complete.
    """
    import os as _os

    from local_operator.tools import eval_worker

    read_fd, write_fd = _os.pipe()
    monkeypatch.setattr(eval_worker, "_SCRUB_FD", None)
    try:
        monkeypatch.setenv(eval_worker._SCRUB_FD_ENV, str(write_fd))
        eval_worker._install_scrub_channel()
        assert eval_worker._SCRUB_FD == write_fd, "the worker ignored the handed-over fd"
        assert not _os.get_blocking(write_fd), (
            "the scrub fd is BLOCKING: a full pipe now parks the retrieval that "
            "published onto it, killing the kernel and all session state"
        )

        # Fill it. A non-blocking write to a full pipe raises rather than
        # parking, which is how this loop terminates at all.
        with contextlib.suppress(BlockingIOError):
            while True:
                _os.write(write_fd, b"x" * 4096)

        eval_worker._publish_secret_value("value-that-does-not-fit")
        # Dropped, not buffered, and the channel stays usable for the next
        # value once the parent drains: a full pipe is a transient condition.
        assert eval_worker._SCRUB_FD == write_fd
    finally:
        eval_worker._SCRUB_FD = None
        _os.close(read_fd)
        _os.close(write_fd)


@pytest.mark.asyncio
async def test_secrets_retrieved_across_cells_still_scrub_a_later_crash(
    isolated: Path,
) -> None:
    """R9 must not cost R1: draining early keeps values, it does not spend them.

    Draining the pipe on the healthy path means the crash path is no longer
    its only reader, so the values a crash needs are the ones EARLIER
    exchanges already consumed. This retrieves in one cell and crashes in a
    later one — if the drain dropped what it read, the fd-2 tail would come
    back raw.
    """
    from local_operator.secrets.access import open_store
    from local_operator.tools import eval as eval_tool

    open_store(create=True).set("EARLY", b"EARLYLEAK-2b7c-4410")
    tool = eval_tool.build_eval_tool()
    context = ToolContext(cwd="/tmp", session_id="cross-cell-crash")
    first = await tool.execute(
        "c1", {"code": "token = secrets['EARLY']\nlen(token)\n"}, AbortSignal(), None, context
    )
    assert not first.is_error, first.text
    result = await tool.execute(
        "c2",
        {"code": "import os\nos.write(2, f'TAIL {token}'.encode())\nos._exit(3)\n"},
        AbortSignal(),
        None,
        context,
    )
    assert result.is_error
    assert "EARLYLEAK-2b7c-4410" not in result.text, "a drained value stopped scrubbing"
    assert "TAIL [redacted]" in result.text


@pytest.mark.asyncio
async def test_list_on_a_machine_with_no_store_is_not_an_error(isolated: Path) -> None:
    """`list` is the orienting verb; an error there reads as 'broken'."""
    result = await _call("list")
    assert not result.is_error, result.text
    assert "No secrets are stored" in result.text


# --------------------------------------------------------------------------
# The production read path, end to end (QA Q3)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform not in ("darwin",) and not sys.platform.startswith("linux"),
    reason="the broker is implemented for macOS and Linux",
)
@pytest.mark.asyncio
async def test_a_real_bash_child_running_lop_secret_get_is_redacted_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole §6 chain with NOTHING mocked — the seam Q3 found open.

    Every other redaction test here supplies one side of the chain: the store
    subclass above stands in for the broker's notice, and the broker PR's own
    tests stand in for this consumer. Each passed while the two were never
    connected, because each mocked the other's half — so this asserts across
    the join instead. A real broker, a real registered session answering
    notices, and a real `bash` child running the documented
    `$(lop secret get NAME)` from the credentials guide.

    Then it checks the three channels a model can actually read: the live
    stream the TUI paints, the `jobs(op='peek')` buffer, and the finished tool
    result. Before the fix the child was served by a LOCAL decrypt that never
    told the session anything, so `redaction_values()` stayed empty and the
    credential was painted verbatim in all three.
    """
    pytest.importorskip("local_operator.secrets.client")

    from local_operator.secrets.broker import SecretBroker
    from local_operator.secrets.crypto import generate_master_key
    from local_operator.secrets.keys import key_path, write_private_file
    from local_operator.secrets.session import register_session
    from local_operator.secrets.store import SecretStore

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    from local_operator.paths import CONFIG_DIR_ENV

    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))

    secret = "Q3-REAL-BROKER-9f4c1a2e"
    key = generate_master_key()
    store_on_disk = SecretStore(key, base=root)
    store_on_disk.initialize()
    store_on_disk.set("Q3_TOKEN", secret.encode(), description="q3")
    write_private_file(key_path(root), key)

    broker = SecretBroker(root, key_provider=lambda: key, idle_shutdown_s=0)
    broker.start()

    variables = BrokerAwareStore()

    def on_secret(_name: str, value: bytes) -> None:
        # Returns None: `register_session` types the callback as `-> None`, and
        # `register_redaction` answers bool, so the result is deliberately
        # dropped rather than returned.
        variables.register_redaction(value.decode())

    registration = register_session(on_secret, session_id="q3-e2e", base=root)
    assert registration is not None, "no session registered; the seam is untestable"

    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), variables=variables, jobs=manager, session_id="q3-e2e")
    # The child must import the tree UNDER TEST, not whatever `lop` happens to
    # be installed on this machine — without this the subprocess silently
    # exercises the released runtime and the assertion below reports on the
    # wrong code. Derived from the package location rather than hardcoded so it
    # follows the worktree.
    import local_operator

    tree = str(Path(local_operator.__file__).resolve().parent.parent)
    cli = f"{shlex.quote(sys.executable)} -m local_operator.cli secret get Q3_TOKEN"
    # The documented path, verbatim from guide://credentials, in a real shell.
    command = (
        f"export PYTHONPATH={shlex.quote(tree)}; "
        f"{cli} > /dev/null; sleep 0.2; "
        f'printf %s "$({cli})"'
    )
    try:
        result = await builtin.execute_bash(
            "q3", {"command": command}, AbortSignal(), None, context
        )
    finally:
        registration.close()
        broker.stop(timeout=5)

    # The session was actually told — this is the notifier/consumer join.
    assert secret in variables.redaction_values(), (
        "the session was never notified of a value its own child retrieved; "
        "the redaction notifier and its consumer are not wired together"
    )
    assert secret not in result.text, "the credential reached the model-visible tool result"
    assert "[redacted]" in result.text
