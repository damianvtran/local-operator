"""The ``@``-reference resolver's failure semantics, one test per table row.

The governing rule these tests exist to pin: a token that does not resolve to
an existing path is NOT a reference — it is prose, and it is left
byte-identical. That rule is the whole safety argument for expanding any text
without a provenance flag, so the tests asserting it
(``test_glab_assignee_me_is_not_a_reference`` above all) are load-bearing
rather than illustrative.

Identity assertions use ``is``, not ``==``. The guarantee the TUI codes against
is "we did not touch your string", and an equal-but-rebuilt copy cannot
distinguish a no-op from a round trip through the expander.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.references import REFERENCE_BLOCK_OPEN, at_token, expand_references

REPO = Path(__file__).resolve().parent.parent.parent


class SpyGate:
    """An approval gate that records what it was asked, and answers ``reply``.

    Records the ASK rather than just counting it: a test asserting the gate was
    consulted before a read has to prove the question named the right path, or
    it would pass on a prompt about something else entirely.
    """

    def __init__(self, reply: bool = True) -> None:
        self.reply = reply
        self.asks: list[tuple[str, str]] = []

    async def __call__(self, tool_name: str, description: str) -> bool:
        self.asks.append((tool_name, description))
        return self.reply


@pytest.mark.asyncio
async def test_a_token_that_names_no_path_is_left_byte_identical(tmp_path):
    text = "look at @nonexistent please"

    result = await expand_references(text, str(tmp_path))

    assert result.expanded is False
    assert result.sent is text
    assert REFERENCE_BLOCK_OPEN not in result.sent
    assert result.notices == ["@nonexistent — no such path; sent as written"]


@pytest.mark.asyncio
async def test_glab_assignee_me_is_not_a_reference(tmp_path):
    """D7's entire safety argument, named after the case that motivated it.

    ``@me`` is a boundary ``@`` and therefore a candidate token. Nothing under
    the cwd is called ``me``, so nothing is captured and the command reaches
    the model exactly as the operator typed it.
    """
    text = "glab mr create --assignee @me"

    result = await expand_references(text, str(tmp_path))

    assert result.expanded is False
    assert result.sent is text
    assert REFERENCE_BLOCK_OPEN not in result.sent


def test_an_email_address_never_opens_a_token():
    """No boundary before the ``@``, so the parse stops before any filesystem work."""
    text = "mail ben@host.com now"

    assert [at_token(text, cursor) for cursor in range(len(text) + 1)] == [None] * (len(text) + 1)


@pytest.mark.asyncio
async def test_an_existing_file_is_expanded_with_its_content(tmp_path):
    (tmp_path / "notes.md").write_text("the body of the file\n", encoding="utf-8")

    result = await expand_references("read @notes.md", str(tmp_path))

    assert result.expanded is True
    assert "the body of the file" in result.sent
    assert result.sent.startswith("read @notes.md")


@pytest.mark.asyncio
async def test_the_typed_token_survives_verbatim_inside_the_block(tmp_path):
    """``typed=`` mirrors ``render_invocation``'s ``invocation=``: the payload is
    what gets persisted, so a resumed session recovers the short row from it."""
    (tmp_path / "notes.md").write_text("body\n", encoding="utf-8")

    result = await expand_references("read @notes.md", str(tmp_path))

    assert 'typed="@notes.md"' in result.sent


@pytest.mark.asyncio
async def test_a_directory_expands_to_a_flat_one_level_listing(tmp_path):
    """Flat, not recursive: the grandchild must be absent."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "child.md").write_text("x", encoding="utf-8")
    (tmp_path / "docs" / "nested").mkdir()
    (tmp_path / "docs" / "nested" / "grandchild.md").write_text("x", encoding="utf-8")

    result = await expand_references("see @docs", str(tmp_path))

    assert result.expanded is True
    assert "child.md" in result.sent
    assert "nested/" in result.sent
    assert "grandchild.md" not in result.sent


@pytest.mark.asyncio
async def test_a_binary_file_emits_metadata_and_never_bytes(tmp_path):
    payload = b"\x00\x01RECOGNISABLE_BINARY_MARKER\x00"
    (tmp_path / "blob.bin").write_bytes(payload)

    result = await expand_references("check @blob.bin", str(tmp_path))

    assert result.expanded is True
    assert str(len(payload)) in result.sent
    assert "RECOGNISABLE_BINARY_MARKER" not in result.sent


@pytest.mark.asyncio
async def test_an_image_emits_the_v1_metadata_stub(tmp_path):
    """v1 ships the stub, not ``ImageContent`` — carrying an image would change
    ``ExpansionResult``'s frozen shape."""
    # A real PNG header, so the content sniffer classifies it as an image
    # rather than as generic binary. Type is decided by CONTENT here, never by
    # extension, which is why the bytes have to be right.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 40
    (tmp_path / "shot.png").write_bytes(png)

    result = await expand_references("look at @shot.png", str(tmp_path))

    assert result.expanded is True
    assert 'kind="image"' in result.sent
    assert "read(path=" in result.sent


@pytest.mark.asyncio
async def test_a_large_file_emits_head_outline_and_a_real_path_pointer(tmp_path):
    big = tmp_path / "huge.md"
    body = "\n".join(
        ["# Title", "x" * 80, "## Middle Section", *["filler line" for _ in range(2000)]]
        + ["## Trailing Section", "tail"]
    )
    big.write_text(body, encoding="utf-8")

    result = await expand_references("read @huge.md", str(tmp_path))

    assert result.expanded is True
    assert str(big) in result.sent
    assert "Trailing Section" in result.sent
    # A real line range, the `context_files._render_index_rows` shape.
    assert "- L" in result.sent and "-" in result.sent
    assert "read(path=" in result.sent


@pytest.mark.asyncio
async def test_no_payload_ever_contains_a_spill_handle(tmp_path):
    """Spill is LRU-evicted under a byte ceiling with a 30-minute grace, so a
    handle in a PERSISTED user message is a dead link after lunch."""
    big = tmp_path / "huge.md"
    big.write_text("# H\n" + "\n".join("line" for _ in range(20000)), encoding="utf-8")
    wide = tmp_path / "wide"
    wide.mkdir()
    for index in range(500):
        (wide / f"entry_{index:04d}.txt").write_text("x", encoding="utf-8")

    result = await expand_references("read @huge.md and @wide", str(tmp_path))

    assert result.expanded is True
    assert "spill://" not in result.sent


@pytest.mark.asyncio
async def test_the_same_path_twice_is_one_reference_block(tmp_path):
    (tmp_path / "notes.md").write_text("body\n", encoding="utf-8")

    result = await expand_references("@notes.md versus @./notes.md", str(tmp_path))

    assert result.sent.count("<reference ") == 1
    # Both tokens stay in the prose: the user wrote them, and the model reads
    # the sentence rather than the block.
    assert "@notes.md versus @./notes.md" in result.sent


@pytest.mark.asyncio
async def test_a_path_outside_the_workspace_asks_for_approval(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_file = outside / "elsewhere.txt"
    secret_file.write_text("CONTENT_BEHIND_THE_GATE", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gate = SpyGate(reply=False)

    result = await expand_references(f"read @{secret_file}", str(workspace), request_approval=gate)

    # Consulted, and consulted BEFORE any read: the gate declined, so content
    # absent proves the read did not happen first and get discarded.
    assert len(gate.asks) == 1
    assert str(secret_file) in gate.asks[0][1]
    assert "CONTENT_BEHIND_THE_GATE" not in result.sent


@pytest.mark.asyncio
async def test_a_declined_approval_leaves_the_token_verbatim(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "elsewhere.txt").write_text("nope", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    text = f"read @{outside / 'elsewhere.txt'}"

    result = await expand_references(text, str(workspace), request_approval=SpyGate(reply=False))

    assert result.expanded is False
    assert result.sent is text
    assert result.notices == [f"@{outside / 'elsewhere.txt'} — not included; approval declined"]


@pytest.mark.asyncio
async def test_a_dotenv_inside_the_workspace_still_asks_for_approval(tmp_path):
    """The one guard ``read`` does not have. ``read`` is a deliberate act by an
    agent under instructions; ``@`` expansion is automatic and silent."""
    # PLACEHOLDER, never a real credential: a fixture that holds one puts it in
    # the repo, the test log and every transcript that quotes the failure.
    (tmp_path / ".env").write_text("PLACEHOLDER=x\n", encoding="utf-8")
    gate = SpyGate(reply=True)

    result = await expand_references("see @.env", str(tmp_path), request_approval=gate)

    assert len(gate.asks) == 1
    assert "may hold secrets" in gate.asks[0][1]
    # Approval escalates; it does not refuse. Saying yes still works.
    assert result.expanded is True
    assert "PLACEHOLDER=x" in result.sent


@pytest.mark.asyncio
async def test_a_quoted_path_with_spaces_resolves(tmp_path):
    (tmp_path / "my file.txt").write_text("quoted body", encoding="utf-8")

    result = await expand_references('read @"my file.txt" now', str(tmp_path))

    assert result.expanded is True
    assert "quoted body" in result.sent


@pytest.mark.asyncio
async def test_an_unquoted_space_terminates_the_token(tmp_path):
    """The shell's answer: ``@my file.txt`` references ``my``, which does not
    resolve, so the whole line passes through."""
    (tmp_path / "my file.txt").write_text("quoted body", encoding="utf-8")
    text = "read @my file.txt now"

    result = await expand_references(text, str(tmp_path))

    assert result.expanded is False
    assert result.sent is text
    assert "quoted body" not in result.sent


@pytest.mark.asyncio
async def test_expansion_is_idempotent(tmp_path):
    """MANDATORY. The TUI expands at submit and ``Session.prompt`` expands
    again; if this is false, every operator message carries a doubled block."""
    (tmp_path / "notes.md").write_text("body\n", encoding="utf-8")
    text = "what does @notes.md do?"

    once = await expand_references(text, str(tmp_path))
    twice = await expand_references(once.sent, str(tmp_path))

    assert once.expanded is True
    assert twice.expanded is False
    assert twice.sent is once.sent
    assert twice.sent.count(REFERENCE_BLOCK_OPEN) == 1


@pytest.mark.asyncio
async def test_expansion_never_raises(tmp_path, monkeypatch):
    """A raised exception here does not lose a reference, it loses the user's
    message — so every failure degrades to the text and a notice."""
    (tmp_path / "notes.md").write_text("body\n", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("local_operator.references._file_payload", boom)
    text = "read @notes.md"

    result = await expand_references(text, str(tmp_path))

    assert result.expanded is False
    assert result.sent is text
    assert result.notices == ["references could not be expanded: disk on fire"]


@pytest.mark.asyncio
async def test_the_kill_switch_disables_expansion(tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("body\n", encoding="utf-8")
    text = "read @notes.md"

    monkeypatch.setenv("LOCAL_OPERATOR_AT_REFERENCES", "0")
    disabled = await expand_references(text, str(tmp_path))
    assert disabled.expanded is False
    assert disabled.sent is text

    # Unset INSIDE the same test: this is what pins "read per call", not
    # "cached at import". A module-level read would leave expansion dead for
    # the rest of the process and this second half would fail.
    monkeypatch.delenv("LOCAL_OPERATOR_AT_REFERENCES")
    enabled = await expand_references(text, str(tmp_path))
    assert enabled.expanded is True
    assert "body" in enabled.sent


def test_references_imports_no_tui_module_at_import_time():
    """A FRESH SUBPROCESS, for ``test_import_graph.py:13-16``'s reason: pytest
    has already imported half the tree, so an in-process ``sys.modules``
    assertion would pass on a real regression."""
    probe = (
        "import sys, json; import local_operator.references; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    modules = json.loads(completed.stdout)

    assert [name for name in modules if name.startswith("local_operator.tui")] == []
    assert [name for name in modules if name == "textual" or name.startswith("textual.")] == []
