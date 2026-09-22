"""``secrets.legacy_env.read_credentials`` — the file format, the errno policy.

This module replaces ``tests/unit/test_credentials.py``'s file-format half and
``tests/unit/test_credentials_encoding.py``'s encoding half, both of which pinned
``CredentialManager``'s reader. The class is deleted; the reader survives in
:mod:`local_operator.secrets.legacy_env`, for one consumer (`lop secret
migrate-env`), and these are the properties it must keep.

The errno policy is the load-bearing part and it is not re-derivable from the
code: "no file recorded" (``ENOENT``) is ``{}``; every other errno — ``EACCES``,
``ELOOP``, ``ENOTDIR``, ``EISDIR`` — is RAISED, because "nothing there" and
"could not look" are different answers and a migration must be able to say the
second.
"""

from __future__ import annotations

import codecs
import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.secrets.legacy_env import CREDENTIALS_FILE_NAME, read_credentials

REPO_ROOT = Path(__file__).resolve().parents[3]


def _reveal(values: dict) -> dict[str, str]:
    return {key: value.get_secret_value() for key, value in values.items()}


def test_reads_key_value_pairs_and_skips_comments_and_blanks(tmp_path: Path) -> None:
    (tmp_path / CREDENTIALS_FILE_NAME).write_text(
        "# a comment\n"
        "\n"
        "DEEPSEEK_API_KEY=sk-value\n"
        "TOKEN=fixture=value\n"  # only the FIRST '=' splits
        "EMPTY_KEY=\n"
    )
    assert _reveal(read_credentials(tmp_path)) == {
        "DEEPSEEK_API_KEY": "sk-value",
        "TOKEN": "fixture=value",
    }


def test_non_empty_false_keeps_blank_valued_keys(tmp_path: Path) -> None:
    (tmp_path / CREDENTIALS_FILE_NAME).write_text("EMPTY_KEY=\nREAL_KEY=v\n")
    assert _reveal(read_credentials(tmp_path, non_empty=False)) == {
        "EMPTY_KEY": "",
        "REAL_KEY": "v",
    }


def test_an_absent_file_is_an_empty_mapping_and_creates_nothing(tmp_path: Path) -> None:
    """``ENOENT`` is the migration's expected END state, not an error — and the
    read must not bring the file back (the defect PR2a closed)."""
    assert read_credentials(tmp_path) == {}
    assert list(tmp_path.iterdir()) == []


def test_a_symlinked_regular_file_is_followed(tmp_path: Path) -> None:
    """A regular-file symlink is a supported shape; only special files are refused."""
    target = tmp_path / "fixture-store"
    target.write_text("# comment\nSYNTHETIC_KEY=fixture=value\nEMPTY_KEY=\n")
    (tmp_path / CREDENTIALS_FILE_NAME).symlink_to(target.name)
    assert _reveal(read_credentials(tmp_path)) == {"SYNTHETIC_KEY": "fixture=value"}
    assert list(_reveal(read_credentials(tmp_path, non_empty=False))) == [
        "SYNTHETIC_KEY",
        "EMPTY_KEY",
    ]


@pytest.mark.skipif(os.name != "posix", reason="mode bits and symlink loops are POSIX-only")
def test_only_enoent_is_an_empty_answer(tmp_path: Path) -> None:
    """Each non-ENOENT errno is pinned to the raise.

    ``ENOTDIR``: the config root is a regular file, so the store cannot exist.
    ``ELOOP``: the store is a symlink to itself.
    ``EACCES``: a directory on the way to the store cannot be searched.
    """
    root_is_a_file = tmp_path / "file-root"
    root_is_a_file.write_text("")
    with pytest.raises(NotADirectoryError):
        read_credentials(root_is_a_file)

    looped = tmp_path / "loop-root"
    looped.mkdir()
    (looped / CREDENTIALS_FILE_NAME).symlink_to(CREDENTIALS_FILE_NAME)
    with pytest.raises(OSError) as loop_error:
        read_credentials(looped)
    assert loop_error.value.errno == errno.ELOOP

    locked = tmp_path / "locked-root"
    locked.mkdir()
    (locked / CREDENTIALS_FILE_NAME).write_text("DEEPSEEK_API_KEY=sk-value\n")
    locked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            read_credentials(locked)
    finally:
        # Restore traversal so the tmp_path tree can still be cleaned up.
        locked.chmod(0o700)


_READ_PROBE = """
import errno
import sys
from pathlib import Path
from unittest.mock import patch
from local_operator.secrets.legacy_env import read_credentials

root, expected, fault = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
def read():
    try:
        result = read_credentials(root)
    except OSError as exc:
        assert expected != errno.ENOENT and exc.errno == expected, repr(exc)
    else:
        assert expected == errno.ENOENT and result == {}, (expected, result)

if fault:
    with patch('local_operator.secrets.legacy_env.os.' + fault,
               side_effect=OSError(expected, 'injected diagnostic failure')):
        read()
else:
    read()
print('PASS')
"""


def _probe(root: Path, expected: int, fault: str = "") -> None:
    """Run the reader in a subprocess, with a deadline.

    A timeout in this pytest process cannot reliably interrupt a blocked syscall.
    A subprocess deadline kills and reaps just our probe, so restoring the FIFO
    regression cannot hang the whole suite.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))}
    env.update(HOME=str(root.parent), LOCAL_OPERATOR_CONFIG_DIR=str(root))
    result = subprocess.run(
        [sys.executable, "-c", _READ_PROBE, str(root), str(expected), fault],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"


@pytest.mark.skipif(os.name != "posix", reason="FIFOs and device nodes are POSIX-only")
@pytest.mark.parametrize("kind", ["fifo", "device", "directory"])
def test_special_files_are_rejected_with_deadline(tmp_path: Path, kind: str) -> None:
    """A FIFO would block the read forever and /dev/null would read unbounded;
    both are refused by the regular-file check on the OPENED descriptor."""
    store = tmp_path / CREDENTIALS_FILE_NAME
    if kind == "fifo":
        os.mkfifo(store)
    elif kind == "device":
        # The same character-device rejection as /dev/zero, but a regressed
        # reader reaches EOF instead of allocating unbounded host memory.
        store.symlink_to("/dev/null")
    else:
        store.mkdir()
    _probe(tmp_path, errno.EISDIR if kind == "directory" else errno.EINVAL)


@pytest.mark.parametrize("fault", ["open", "fstat"])
@pytest.mark.parametrize(
    "code", [errno.ENOENT, errno.EACCES, errno.ELOOP, errno.ENOTDIR, errno.EIO]
)
def test_diagnostic_errnos_are_preserved_with_deadline(
    tmp_path: Path, fault: str, code: int
) -> None:
    (tmp_path / CREDENTIALS_FILE_NAME).write_text("SYNTHETIC_KEY=fixture\n")
    _probe(tmp_path, code, fault)


def test_the_descriptor_is_closed_on_a_validation_failure(tmp_path: Path, monkeypatch) -> None:
    """The opener owns the descriptor until it hands it to ``open`` — including
    every validation failure, so a refused special file leaks nothing."""
    from local_operator.secrets.legacy_env import _open_regular_store

    store = tmp_path / CREDENTIALS_FILE_NAME
    store.write_text("SYNTHETIC_KEY=fixture\n")
    real_open = os.open
    opened = []

    def track_open(path, flags):
        fd = real_open(path, flags)
        opened.append(fd)
        return fd

    def fail_stat(fd):
        raise OSError(errno.EIO, "injected fstat failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "open", track_open)
        patcher.setattr(os, "fstat", fail_stat)
        with pytest.raises(OSError, match="injected fstat failure"):
            _open_regular_store(str(store), os.O_RDONLY)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.read(opened[0], 1)
    assert closed.value.errno == errno.EBADF


# ---------------------------------------------------------------------------
# The encoding pins (audit D10), moved from tests/unit/test_credentials_encoding.py.
#
# Text mode without an ``encoding`` uses the platform default — UTF-8 on POSIX,
# the ANSI code page on Windows (``cp1252`` on an en-US install). The read side
# of that pair is the silent one: a UTF-8 file written by a POSIX build read back
# as mojibake with NO error, in a credential store. The read default is fixed when
# the interpreter starts and cannot be monkeypatched, so it is reproduced the only
# way it can be: a subprocess whose default IS ``US-ASCII``.
# ---------------------------------------------------------------------------

VALUE_WITH_UNENCODABLE_CHARACTER = "s\u00e9cr\u00e8t-\u2713"


def _run_script(
    tmp_path: Path, statement: str, *, ascii_default: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``statement``, by default with ``US-ASCII`` as the text-encoding default.

    THE SCRIPT GOES IN A FILE, NOT IN ``-c``. An argument is bytes on POSIX,
    encoded by the PARENT with its filesystem encoding and ``surrogateescape``,
    so a value this test exists to make unrepresentable (``\u2713``) reaches the
    child as a lone surrogate and dies in ``-c``'s decoder — the test failing for
    a reason that has nothing to do with the credential reader. Writing the
    script UTF-8 to a file keeps every byte of the value inside the file and the
    argv pure ASCII.
    """
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))
    }
    environment.update(
        HOME=str(tmp_path / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        PYTHONPATH=str(REPO_ROOT),
    )
    if ascii_default:
        environment.update(
            LC_ALL="C",
            LANG="C",
            PYTHONCOERCECLOCALE="0",
            PYTHONUTF8="0",
        )
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    script = tmp_path / "statement.py"
    script.write_text(statement, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )


def test_the_forced_default_encoding_is_ascii(tmp_path: Path) -> None:
    """The premise of the tests below, checked rather than assumed.

    Asserted through the CODEC, not through the locale name: glibc answers
    ``ANSI_X3.4-1968`` for the C locale and macOS answers ``US-ASCII``, and a
    test that pins one spelling fails on the other for no reason that concerns
    this file.
    """
    result = _run_script(tmp_path, "import locale; print(locale.getencoding())")
    assert result.returncode == 0, result.stderr
    encoding = result.stdout.strip()
    assert codecs.lookup(encoding).name == "ascii", encoding
    with pytest.raises(UnicodeEncodeError):
        VALUE_WITH_UNENCODABLE_CHARACTER.encode(encoding)


def test_a_utf8_file_reads_back_byte_for_byte_under_that_default(tmp_path: Path) -> None:
    """What a POSIX build wrote must not come back as mojibake.

    The file is written as raw UTF-8 bytes (no writer survives to produce it), and
    read under the ASCII default — the cross-platform read the explicit encoding
    argument exists to make lossless.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / CREDENTIALS_FILE_NAME).write_bytes(
        f"TOKEN={VALUE_WITH_UNENCODABLE_CHARACTER}\n".encode("utf-8")
    )

    read_back = _run_script(
        tmp_path,
        "from pathlib import Path\n"
        "from local_operator.secrets.legacy_env import read_credentials\n"
        f"values = read_credentials(Path({str(config)!r}))\n"
        # Hex rather than the value itself: stdout under this default is ASCII
        # too, so printing the value would fail on the assertion's OWN encoding
        # and hide whether the read was correct.
        "value = values['TOKEN'].get_secret_value()\n" "print(value.encode('utf-8').hex())\n",
    )
    assert read_back.returncode == 0, read_back.stderr
    assert read_back.stdout.strip() == VALUE_WITH_UNENCODABLE_CHARACTER.encode("utf-8").hex()


def test_invalid_utf8_bytes_round_trip_losslessly(tmp_path: Path) -> None:
    """``surrogateescape`` on the read: a byte no encoding claims (a file written
    by an older build, or hand-edited) comes back as the SAME byte rather than
    U+FFFD, so a migrated secret is not corrupted in flight."""
    config = tmp_path / "config"
    raw = b"TOKEN=caf\xe9-latin1\n"
    config.mkdir(exist_ok=True)
    (config / CREDENTIALS_FILE_NAME).write_bytes(raw)

    result = _run_script(
        tmp_path,
        "from pathlib import Path\n"
        "from local_operator.secrets.legacy_env import read_credentials\n"
        f"values = read_credentials(Path({str(config)!r}))\n"
        "print(values['TOKEN'].get_secret_value().encode('utf-8', 'surrogateescape').hex())\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"caf\xe9-latin1".hex()
