"""The credential store under a non-UTF-8 default encoding (audit D10).

``credentials.env`` is read and written in text mode, and text mode without an
``encoding`` uses the platform default — UTF-8 on POSIX, but the ANSI code page
on Windows (``cp1252`` on an en-US install), where PEP 538/540's C-locale
coercion does not apply. Two failures follow, and they are not symmetric: a
value outside that code page cannot be SAVED (``UnicodeEncodeError``), and a
UTF-8 file written by a POSIX build is read back as mojibake with no error at
all — the silent one, in a credential store.

**Why this test forces the encoding rather than reasoning about it.** On this
host the default is already UTF-8, so a round trip here would pass on the
unfixed code and prove nothing. The read default is fixed when the interpreter
starts and cannot be monkeypatched afterwards, so the unset-encoding case is
reproduced the only way it can be: a subprocess whose default IS ``US-ASCII``
(``LC_ALL=C`` with PEP 538 coercion switched off and UTF-8 mode off — the same
class of default Windows has), asserted rather than assumed.
"""

from __future__ import annotations

import codecs
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

VALUE_WITH_UNENCODABLE_CHARACTER = "s\u00e9cr\u00e8t-\u2713"


def _run(
    tmp_path: Path, statement: str, *, ascii_default: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``statement``, by default with ``US-ASCII`` as the text-encoding default.

    THE SCRIPT GOES IN A FILE, NOT IN ``-c``. An argument is bytes on POSIX,
    encoded by the PARENT with its filesystem encoding and ``surrogateescape``,
    so a value this test exists to make unrepresentable (``\u2713``) reaches the
    child as a lone surrogate and dies in ``-c``'s decoder with
    ``'utf-8' codec can't encode characters ... surrogates not allowed`` — the
    test failing for a reason that has nothing to do with the credential store.
    Writing the script UTF-8 to a file keeps every byte of the value inside the
    file and the argv pure ASCII, which is also the shape a real caller has.
    Python reads a source file as UTF-8 regardless of the locale, so the child's
    ASCII default still governs everything the test is about.
    """
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
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
    """The premise of every test below, checked rather than assumed.

    Asserted through the CODEC, not through the locale name: glibc answers
    ``ANSI_X3.4-1968`` for the C locale and macOS answers ``US-ASCII``, and a
    test that pins one spelling fails on the other for no reason that concerns
    this file. ``codecs.lookup`` names both ``ascii``.
    """
    result = _run(tmp_path, "import locale; print(locale.getencoding())")
    assert result.returncode == 0, result.stderr
    encoding = result.stdout.strip()
    assert codecs.lookup(encoding).name == "ascii", encoding
    with pytest.raises(UnicodeEncodeError):
        VALUE_WITH_UNENCODABLE_CHARACTER.encode(encoding)


def test_a_credential_outside_that_code_page_can_still_be_saved(tmp_path: Path) -> None:
    """The crash half: ``\u2713`` exists in no ANSI code page, so the write used to fail."""
    config = tmp_path / "config"
    result = _run(
        tmp_path,
        "from pathlib import Path\n"
        "from local_operator.credentials import CredentialManager\n"
        f"manager = CredentialManager(Path({str(config)!r}))\n"
        f"manager.set_credential('TOKEN', {VALUE_WITH_UNENCODABLE_CHARACTER!r})\n"
        "print('SAVED')\n",
    )
    assert result.returncode == 0, result.stderr
    assert "SAVED" in result.stdout


def test_a_utf8_file_reads_back_byte_for_byte_under_that_default(tmp_path: Path) -> None:
    """The silent half: what a POSIX build wrote must not come back as mojibake.

    The file is written by the SAME code under a UTF-8 default (a POSIX
    install's file) and read under the ASCII one, which is the cross-platform
    read the encoding argument exists to make lossless.
    """
    config = tmp_path / "config"
    store = config / "credentials.env"

    store_repr = repr(str(store))
    written = _run(
        tmp_path,
        "from pathlib import Path\n"
        "import locale\n"
        "from local_operator.credentials import CredentialManager\n"
        f"manager = CredentialManager(Path({str(config)!r}))\n"
        f"manager.set_credential('TOKEN', {VALUE_WITH_UNENCODABLE_CHARACTER!r})\n"
        f"raw = Path({store_repr}).read_bytes()\n"
        "assert raw.count(b'\\xe2\\x9c\\x93') == 1, 'not UTF-8 on disk'\n"
        "print('wrote with', locale.getencoding())\n",
        ascii_default=False,
    )
    assert written.returncode == 0, written.stderr
    assert "UTF-8" in written.stdout, written.stdout

    read_back = _run(
        tmp_path,
        "from pathlib import Path\n"
        "from local_operator.credentials import CredentialManager\n"
        f"manager = CredentialManager(Path({str(config)!r}))\n"
        # Hex rather than the value itself: stdout under this default is ASCII
        # too, so printing the value would fail on the assertion's OWN encoding
        # and hide whether the read was correct.
        "value = manager.get_credential('TOKEN').get_secret_value()\n"
        "print(value.encode('utf-8').hex())\n",
    )
    assert read_back.returncode == 0, read_back.stderr
    assert read_back.stdout.strip() == VALUE_WITH_UNENCODABLE_CHARACTER.encode("utf-8").hex()
