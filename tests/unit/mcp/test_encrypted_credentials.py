"""Real encrypted-store and child-stderr regressions, synthetic values only."""

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest

from local_operator.mcp.config import MCPHttpServerConfig, MCPStdioServerConfig
from local_operator.mcp.credentials import MCPCredentials, store_credentials
from local_operator.mcp.manager import McpManager
from local_operator.mcp.secret_refs import (
    McpSecretRefError,
    public_secret_refs,
    resolve_config_secrets,
)
from local_operator.secrets import access
from local_operator.secrets.errors import SecretCorrupt
from local_operator.variables import VariableStore


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    base = tmp_path / "config"
    base.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base))
    # Exercise the real encrypted store without launching a persistent broker
    # for every test. Broker denial is independently tested at the access seam.
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **kw: False)
    return base


def test_metadata_is_pristine_deduplicated_and_cold():
    cfg = MCPHttpServerConfig(
        url="https://example.invalid/mcp",
        headers={
            "Authorization": "Bearer ${HUBSPOT_TOKEN}",
            "Other": "${HUBSPOT_TOKEN}",
            "Literal": "$${NOT_A_KEY}",
            "Region": "us-east-1",
        },
    )
    assert public_secret_refs(cfg) == [
        {
            "id": "HUBSPOT_TOKEN",
            "bindings": [
                {"field": "headers", "key": "Authorization"},
                {"field": "headers", "key": "Other"},
            ],
        }
    ]


def test_the_encrypted_store_is_the_only_leg_read(isolated):
    """The plaintext fallback is GONE: only the encrypted store resolves a ref.

    The legacy ``credentials.env`` has no writers left, so a value that lives
    only there must NOT be served — it reads as missing until ``lop secret
    migrate-env`` moves it, which is the honest state of an install that has not
    migrated rather than a silent downgrade to a store nothing maintains.
    """
    # The writer that used to seed the in-memory mapping is deleted (PR2b), so
    # the decoy is written as the FILE a host mid-migration would actually have:
    # the point of the test is that a value living only in it must not resolve.
    (isolated / "credentials.env").write_text("TOKEN=legacy-synthetic\n", encoding="utf-8")
    cfg = MCPStdioServerConfig(command="unused", env={"API_KEY": "${TOKEN}"})
    with pytest.raises(McpSecretRefError):
        resolve_config_secrets("test", cfg)
    access.open_store(isolated, create=True).set("TOKEN", b"encrypted-synthetic")
    assert resolve_config_secrets("test", cfg).env["API_KEY"] == "encrypted-synthetic"
    # The config keeps the reference it was written with.
    assert cfg.env["API_KEY"] == "${TOKEN}"


@pytest.mark.parametrize("failure", [PermissionError("denied"), SecretCorrupt("corrupt"), b""])
def test_encrypted_failure_never_downgrades(isolated, monkeypatch, failure):
    access.open_store(isolated, create=True).set("TOKEN", b"encrypted-synthetic")

    def retrieve(*args):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(access, "retrieve_secret", retrieve)
    with pytest.raises(McpSecretRefError, match="unavailable"):
        resolve_config_secrets(
            "test", MCPStdioServerConfig(command="unused", env={"X": "${TOKEN}"})
        )


@pytest.mark.asyncio
async def test_real_split_child_stderr_is_scrubbed(isolated, caplog):
    sentinel = "synthetic-split-canary-91730"
    access.open_store(isolated, create=True).set("TOKEN", sentinel.encode())
    # The split has to land INSIDE ``STDERR_LINE_LIMIT`` (2000): everything a
    # child writes past it is truncated before any sink sees it, so a probe that
    # splits at 8190 passes with the whole scrub removed — it was measuring the
    # truncation, not the redaction (agent review R-2). 37 characters of banner
    # leave the credential's own first fragment inside the retained line.
    cfg = MCPStdioServerConfig(
        command=sys.executable,
        timeout=3000,
        env={"X": "${TOKEN}"},
        args=[
            "-c",
            'import os,sys,time; x=os.environ["X"]; sys.stderr.write("a"*37+x[:7]); '
            'sys.stderr.flush(); time.sleep(.02); sys.stderr.write(x[7:]+"\\n"); sys.exit(1)',
        ],
    )
    variables = VariableStore(env={})
    manager = McpManager(
        isolated, secret_base=isolated, register_secret=variables.register_redaction
    )
    caplog.set_level(logging.DEBUG, logger="local_operator.mcp")
    with pytest.raises(Exception) as caught:
        await asyncio.wait_for(manager._connect_server("canary", cfg), 15)
    await manager.disconnect_all()
    # The child's own reason reached the raised error, so the assertion below is
    # about the QUOTED TAIL and not about a transport error that happens to be
    # secret-free: the banner is the only place those 37 characters exist.
    assert "a" * 37 in str(caught.value), str(caught.value)
    assert sentinel not in str(caught.value)
    assert sentinel not in _chain_text(caught.value)
    assert sentinel not in caplog.text
    assert sentinel[:7] not in caplog.text
    assert not variables.credential_env()
    assert variables.redact(sentinel) == "[redacted]"


def test_the_stderr_redactor_masks_a_word_like_value_on_word_boundaries() -> None:
    """A letters-only credential in a child's stderr masks as a TOKEN, never inside a word.

    The stderr redactor takes the shared value pass now (agent review round 1, R1), so a
    word-like value behaves as it does on every other surface: the standalone token is
    masked, and an ordinary word that merely CONTAINS it stays readable — the over-mask
    this pass exists to remove, on the one sink that runs before a tool result exists.
    """
    from local_operator.mcp.redaction import StderrRedactor

    word = "syn" + "thetic"
    redactor = StderrRedactor([word])
    out = redactor.feed(("un" + word + "ally and a " + word + " store\n").encode())
    assert "un" + word + "ally" in out, "the embedded word must stay readable"
    assert f"a {word} store" not in out, "the standalone token must stay masked"
    assert "[redacted]" in out
    # ...and a cut inside the line must not publish the standalone token either.
    second = StderrRedactor([word])
    payload = f"pre {word} post\n".encode()
    joined = second.feed(payload[:8]) + second.feed(payload[8:]) + second.feed(b"", final=True)
    assert word not in joined.replace("[redacted]", ""), joined


def test_every_split_of_an_escape_before_a_word_keeps_it_masked() -> None:
    """Every cut of a payload whose word follows a JSON escape must keep it masked.

    Deferred QA Q-3 (#2140): a release cut that landed between the escape's ``\\``
    and its letter published the letter+word into the NEXT release, where the
    boundary rule read the escape's tail (``n``, ``t``, or up to four hex digits of
    ``\\uXXXX``) as an ordinary letter run and left the standalone token — on the
    one sink that runs before a tool result exists. Swept here over EVERY two-way
    split, every three-way split and a one-byte feed for the escape shapes and the
    two backslash-RUN shapes whose pairing decides the boundary (review round 1,
    R-3 re-checked the run classes on the parity hold), plus a value-length sweep:
    no arrangement may publish the word unmasked.
    """
    from local_operator.mcp.redaction import StderrRedactor

    word = "qw" * 5

    def joined(payload: str, cuts: tuple[int, ...]) -> str:
        redactor = StderrRedactor([word])
        out = ""
        previous = 0
        for cut in (*cuts, len(payload)):
            out += redactor.feed(payload[previous:cut].encode())
            previous = cut
        return out + redactor.feed(b"", final=True)

    for escape in (
        "\\" + "n",
        "\\" + "t",
        "\\" + "u000a",
        "\\" + "\\",
        "\\" + "\\" + "\\",
    ):
        payload = '{"msg": "x' + escape + word + ' z"}\n'
        for i in range(1, len(payload)):
            out = joined(payload, (i,))
            assert word not in out.replace("[redacted]", ""), (escape, "2-way", i, out)
            for j in range(i, len(payload)):
                out = joined(payload, (i, j))
                assert word not in out.replace("[redacted]", ""), (escape, "3-way", i, j, out)
        out = joined(payload, tuple(range(1, len(payload))))
        assert word not in out.replace("[redacted]", ""), (escape, "1-byte", out)
        assert "[redacted]" in out, (escape, "premise: the word was there to mask")
    # The word's own length moves where the historical cut landed; every length
    # 8..17 on the ``\\uXXXX`` shape must hold.
    for length in range(8, 18):
        word = ("qw" * 10)[:length]
        payload = '{"msg": "x' + "\\" + "u000a" + word + ' z"}\n'
        for i in range(1, len(payload)):
            out = joined(payload, (i,))
            assert word not in out.replace("[redacted]", ""), (length, i, out)


def test_a_pure_backslash_stream_stays_bounded() -> None:
    """A backslash run is counted in one step, never walked one hold at a time.

    Review round 1, R-3: the hold re-entered its helper once per backslash, so a
    child streaming backslashes with no newline pinned every byte in ``pending``
    (three 4 KB feeds: released 0/0/0, pending 12288) instead of the class's own
    bounded holdback (base: released 4088/4096/4096, pending 8). With the parity
    hold an unpaired backslash still crosses feeds safely, at the documented cost
    of ONE byte of delay per feed; the run's own length changes nothing else.
    """
    from local_operator.mcp.redaction import StderrRedactor

    word = "syn" + "thetic"
    redactor = StderrRedactor([word])
    # One byte of delay per feed at most: every release is the chunk minus the
    # class's holdback (longest value - 1) and the one withheld backslash.
    assert [len(redactor.feed(b"\\" * 4096)) for _ in range(3)] == [4088, 4096, 4096]
    assert len(redactor._pending) == 8, len(redactor._pending)
    # ...and the final flush releases the held tail: the stream is byte-identical.
    assert redactor.feed(b"", final=True) == "\\" * 8


def test_an_oversized_unterminated_line_never_releases_an_escape_fragment(caplog) -> None:
    """The real-pump shape: an unterminated line past ``STDERR_LINE_LIMIT``.

    The stderr pump (``manager._stderr_pump``) flushes its buffer as a line once it
    passes the limit, so two releases of ONE unterminated line land as SEPARATE
    lines and no later whole-line scrub can re-join them: before the fix, a cut
    landing right after the escape's ``\\`` released it in the first half and the
    word arrived in the second half as ``n<word>`` — retained unmasked in the tail
    and the log (the QA Q-3 real-child repro; word length 9 lands that cut exactly
    there). Drives the real ``StderrRedactor`` and the real ``McpServerStderr``
    through the pump's own six-line buffer/flush rule.
    """
    from local_operator.mcp.manager import STDERR_LINE_LIMIT, McpServerStderr
    from local_operator.mcp.redaction import StderrRedactor

    caplog.set_level(logging.INFO, logger="local_operator.mcp.server.canary")
    word = "split" + "word"  # 9 letters: the historical cut lands on the backslash
    first = ("A" * 2100) + "\\" + "n" + word[:7]
    second = word[7:] + ' z"}\n'

    sink = McpServerStderr("canary")
    redactor = StderrRedactor([word])
    emitted: list[str] = []
    buffer = ""
    for chunk in (first, second):
        buffer += redactor.feed(chunk.encode())
        lines = buffer.split("\n")
        buffer = lines.pop()
        if len(buffer) > STDERR_LINE_LIMIT:
            lines.append(buffer)
            buffer = ""
        for line in lines:
            emitted.append(line)
            sink.feed(line)
    buffer += redactor.feed(b"", final=True)
    if buffer:
        emitted.append(buffer)
        sink.feed(buffer)

    out = "\n".join(emitted)
    assert len(out) > STDERR_LINE_LIMIT, "premise: the pump's flush path ran"
    for surface in (out, sink.tail_text(), sink.quoted_tail(), caplog.text):
        assert word not in surface, surface
    assert "[redacted]" in out, "premise: the word was there to mask"


def test_a_complete_line_still_releases_through_the_escape_hold() -> None:
    """No regression on the normal line behaviour the holdback exists beside.

    The hold only retimes a cut INSIDE an unterminated line; a complete line
    releases as soon as it arrives (that is what lets a child print its startup
    line and go quiet), a line ending right before its newline is not delayed, and
    the value-prefix holdback still refuses to split a registered value.
    """
    from local_operator.mcp.redaction import StderrRedactor

    word = "syn" + "thetic"
    redactor = StderrRedactor([word])
    assert redactor.feed((f"report {word} done\n").encode()) == "report [redacted] done\n"
    assert redactor.feed(b"path C:\\\n") == "path C:\\\n"
    split = StderrRedactor([word])
    out = split.feed(("see " + word[:6]).encode())
    out += split.feed((word[6:] + " here\n").encode())
    assert word not in out.replace("[redacted]", ""), out


def _chain_text(exc: BaseException) -> str:
    """``str()`` of an exception and everything it is chained to.

    ``str()`` deliberately, and not ``repr()``/``args``: an ``MCPError`` keeps
    its text in ``error.message`` and builds ``__str__`` from that field, so a
    walk over ``repr`` — or over ``args``, which the scrub DOES rewrite — reads a
    clean tuple while ``str()``, the string every sink renders, still carries the
    credential. That gap is why the earlier guard could not see this regression
    (agent review R-2).
    """
    parts: list[str] = []
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        parts.append(f"{type(node).__name__}: {node}")
        node = node.__cause__ or node.__context__
    return " | ".join(parts)


#: A stdio child that answers ``initialize`` with a JSON-RPC ERROR whose message
#: echoes the credential it was handed. That is the shape a badly behaved
#: third-party server has, and the one an SDK ``MCPError`` keeps in
#: ``error.message`` — so it is the shape a sink publishes verbatim unless the
#: value was registered for scrubbing. ``tail`` also writes one line to stderr,
#: which is what makes ``explain()`` quote a tail; without it ``explain()``
#: hands the exception itself back, which is the case for every HTTP transport
#: (they spawn nothing) and for a stdio child that stays quiet.
_ECHO_CHILD = (
    "import json,os,sys;"
    "tail = sys.argv[1] == 'tail';"
    "sys.stderr.write('booting canary\\n') if tail else None;"
    "sys.stderr.flush();"
    "req = json.loads(sys.stdin.readline());"
    "sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'error':"
    " {'code': -32000, 'message': 'rejected credential ' + os.environ['X']}}) + '\\n');"
    "sys.stdout.flush();"
    "sys.exit(1)"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [True, False], ids=["with-tail", "no-tail"])
async def test_a_server_echoed_credential_never_reaches_a_sink(isolated, tail):
    """A credential echoed in a JSON-RPC error is scrubbed on all three paths.

    The three paths are the finding (agent review R-1): the raised exception,
    the CAUSE it deliberately keeps as evidence, and the startup-failure text
    that becomes ``McpStartupOutcome.failures`` — which the toast, the transcript
    notice, ``/mcp`` and the desktop projection all render. Both arms are
    exercised, because they fail differently: with a stderr tail the raised error
    is rebuilt from scrubbed text and only the CHAIN leaked; without one the raw
    exception went out, and that is every HTTP connect.

    Fails with redaction removed, in every arm — that is what makes it a guard.
    """
    sentinel = "synthetic-echo-canary-40711"
    access.open_store(isolated, create=True).set("TOKEN", sentinel.encode())
    cfg = MCPStdioServerConfig(
        command=sys.executable,
        timeout=3000,
        env={"X": "${TOKEN}"},
        args=["-c", _ECHO_CHILD, "tail" if tail else "quiet"],
    )
    variables = VariableStore(env={})
    manager = McpManager(
        isolated, secret_base=isolated, register_secret=variables.register_redaction
    )
    manager._configs = {"canary": cfg}
    manager._sources = {"canary": "test"}
    with pytest.raises(Exception) as caught:
        await asyncio.wait_for(manager._connect_server("canary", cfg), 15)
    assert sentinel not in str(caught.value)
    assert sentinel not in _chain_text(caught.value)
    # The published surface: one round is what fills `startup_failures()`, which
    # `session_factory` turns into the outcome the front ends read. Wait for it to
    # SETTLE rather than assuming which arm the gate took: a spawn that takes
    # longer than the 250 ms gate is deferred, and its failure is then recorded by
    # the continuation — the same single write path, just later.
    await manager._connect_round({"canary": cfg}, {"canary": "test"})
    deadline = asyncio.get_running_loop().time() + 10
    while manager.startup_settling() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert manager.startup_settling() is False, "the startup round never settled"
    failures = manager.startup_failures()
    await manager.disconnect_all()
    # Not vacuous: the SERVER's own sentence is in there, with only the value
    # replaced — so the round recorded this failure and not some other one.
    assert failures["canary"].startswith("rejected credential [redacted]"), failures
    assert sentinel not in str(failures)
    assert variables.redact(sentinel) == "[redacted]"
    assert not variables.credential_env()


@pytest.mark.asyncio
async def test_metadata_probe_never_creates_the_legacy_file(isolated):
    """A cold metadata read must not write the store it is describing.

    A probe that constructs a store (the retired ``CredentialManager.__init__`` wrote
    ``credentials.env``; opening the encrypted store initialises it) turns a read
    into a write of state the caller never asked for. Reproduced by the assembled
    desktop probe.
    """
    from local_operator.mcp.credentials import credential_source

    assert credential_source("TOKEN", isolated) == "missing"
    assert not (isolated / "credentials.env").exists()
    assert not (isolated / "secrets" / "store.db").exists()


@pytest.mark.asyncio
async def test_store_validates_all_ids_and_confirms_replacement(isolated, monkeypatch):
    cfg = MCPHttpServerConfig(
        url="https://example.invalid", headers={"Authorization": "Bearer ${TOKEN}"}
    )
    monkeypatch.setattr(
        "local_operator.mcp.credentials.load_all_mcp_configs", lambda cwd: ({"api": cfg}, {})
    )
    owner = SimpleNamespace(
        session_id="synthetic-owner",
        config_dir=isolated,
        variables=VariableStore(env={}),
        mcp_manager=McpManager(isolated),
    )

    async def save(values, confirmed=()):
        return await store_credentials(
            owner, MCPCredentials(name="api", values=values, confirmed_replace=list(confirmed))
        )

    assert (await save({"TOKEN": "one", "Authorization": "wrong"}))["code"] == "invalid_target"
    assert not (isolated / "secrets" / "store.db").exists()
    assert (await save({"TOKEN": "one"}))["code"] == "saved"
    assert (await save({"TOKEN": "two"}))["code"] == "replace_confirmation_required"
    assert access.open_store(isolated).get("TOKEN") == b"one"
    assert (await save({"TOKEN": "two"}, ["TOKEN"]))["code"] == "saved"
    assert access.open_store(isolated).get("TOKEN") == b"two"
    assert not owner.variables.credential_env()
    assert not (isolated / "credentials.env").exists()
