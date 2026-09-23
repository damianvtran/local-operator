"""The ``lop credential update`` prompt, and the provider keys it saves.

This module used to be ``tests/unit/test_credentials.py``, and it carried three
unrelated subjects: the file format of the retired plaintext ``credentials.env``,
the ``/info`` name reader, and the interactive prompt. The file format and the
name reader went with ``local_operator.credentials`` (the reader moved to
:mod:`local_operator.secrets.legacy_env` and its tests to
``tests/unit/secrets/test_legacy_env*.py``). What remains here is the prompt,
which moved to :mod:`local_operator.providers.key_prompt` — the surface that
would otherwise have been deleted along with the module it happened to live in.

The behavioural pins that survive are the ones that cost review rounds: the
piped-stdin automation contract, the empty-input contracts in both branches, and
the ASCII banner fallback.
"""

import io
import os
import tempfile
from pathlib import Path

import pytest

from local_operator.providers.key_prompt import prompt_for_provider_key

#: The value the prompt tests feed and assert against. Named rather than
#: repeated, so a mock and its assertion cannot silently diverge.
SECRET = "new-secret-value"


@pytest.fixture
def temp_config(monkeypatch):
    initial_env = os.environ.copy()
    """A temp config ROOT, pointed at by the store the prompt writes to."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # The prompt saves a provider-class STORE row, which is rooted at the
        # HOME-derived config dir unless told otherwise. Point the process's
        # config dir at this test's temp root so the store lands there and never
        # in the operator's live ~/.local-operator.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", temp_dir)
        yield Path(temp_dir)
        # Clear any used environment variables after each test
        os.environ.clear()
        os.environ.update(initial_env)


def _stored(key: str) -> str:
    from local_operator.secrets.access import open_store
    from local_operator.secrets.store import provider_secret_name

    return open_store().get(provider_secret_name(key), role="provider").decode()


@pytest.mark.skipif(os.name != "posix", reason="File permission tests only run on Unix")
def test_prompt_saves_a_provider_store_row(temp_config, monkeypatch):
    """The prompt's whole job: an interactive paste lands as a store row."""
    # Pin the interactive branch: the prompt reads a line from stdin when stdin
    # is NOT a tty (the scripted-automation contract). Under pytest in CI stdin
    # is not a tty, so without forcing isatty True this test would silently
    # exercise the pipe path and ignore the getpass mock below.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda _: SECRET)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    credential = prompt_for_provider_key("NEW_API_KEY")
    assert credential.get_secret_value() == SECRET
    assert _stored("NEW_API_KEY") == SECRET


def test_empty_interactive_input_raises_value_error(temp_config, monkeypatch):
    """Pinning the interactive branch (see above): an empty value is a ValueError."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda _: "")
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError) as exc_info:
        prompt_for_provider_key("NON_EXISTENT_KEY")
    assert "is required for this step" in str(exc_info.value)


def test_prompt_reads_from_piped_stdin_without_getpass(temp_config, monkeypatch):
    """Non-tty stdin: the value is read as one line (the automation contract),
    NOT through getpass's echo fallback (item 13)."""
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-key-value\n"))
    # getpass must NOT be called on the non-tty path.
    monkeypatch.setattr(
        "getpass.getpass",
        lambda *_a, **_k: pytest.fail("getpass used on non-tty stdin"),
    )
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    cred = prompt_for_provider_key("PIPED_KEY")
    assert cred.get_secret_value() == "piped-key-value"
    assert _stored("PIPED_KEY") == "piped-key-value"


def test_prompt_empty_piped_stdin_raises_eof(temp_config, monkeypatch):
    """A closed/empty pipe raises EOFError, which the command handler turns into
    one plain line + exit 1 (item 4/13)."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    with pytest.raises(EOFError):
        prompt_for_provider_key("PIPED_KEY")


class _TtyStub:
    """A stdin stand-in that claims to be a tty (so getpass is used)."""

    def isatty(self) -> bool:
        return True


def test_prompt_ascii_banner_when_stdout_cannot_encode(temp_config, monkeypatch, capsys):
    """PYTHONIOENCODING=ascii (or a legacy code page) cannot encode the box
    drawing, so the banner falls back to ASCII rather than crashing (item 14)."""
    # Patched in the prompt module's namespace (it imports the name directly),
    # not on cli_style, or the already-bound reference would win.
    monkeypatch.setattr("local_operator.providers.key_prompt.can_encode", lambda *a, **k: False)
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "k")
    monkeypatch.setattr("sys.stdin", _TtyStub())
    prompt_for_provider_key("ASCII_KEY")
    out = capsys.readouterr().out
    assert "─" not in out and "╭" not in out
    assert "+" in out  # ASCII border corner
    # The success glyph also falls back — a stdout that cannot encode the box
    # cannot encode ✓ either, and crashing after the key is saved is the worst
    # place to fail.
    assert "✓" not in out and "[ok]" in out


def test_a_store_write_failure_is_one_plain_line_not_a_traceback(temp_config, monkeypatch, capsys):
    """A store that refuses the row must not draw a stack-trace panel: the
    prompt reports one line on stderr and raises a ValueError the handler turns
    into an exit code."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "k")

    def _boom(*_a, **_k):
        raise RuntimeError("store is locked")

    monkeypatch.setattr("local_operator.providers.registry.store_provider_key", _boom)

    with pytest.raises(ValueError, match="Could not save"):
        prompt_for_provider_key("LOCKED_KEY")
    assert "store is locked" in capsys.readouterr().err
