"""``lop secret`` end to end, asserting the EXACT bytes on stdout.

The contract this file exists to defend: ``$(lop secret get NAME)`` must yield
the value and nothing else. A banner, a colour code or a stray newline does not
fail loudly — it corrupts a credential, which surfaces as a puzzling 401 from a
remote service that nobody traces back to this command. So the assertions here
are byte-equality on the real subprocess's stdout, not substring checks on a
captured string.

Real subprocesses rather than calling ``main()`` in-process, because that is
the only way to catch the failure modes that matter: a library that prints a
warning at import, a logging handler attached to stdout, the locale's codec
mangling a non-ASCII value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def cli(tmp_path: Path):
    """Run ``lop secret ...`` in a fully isolated config dir.

    Both ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` are redirected per
    `AGENTS.md`: the config override alone does not redirect the cache root,
    and a test that writes into the operator's real home while believing it is
    sandboxed is the documented hazard here.

    ``CMUX_*`` is scrubbed for the reason the team's QA rules give: an
    inherited ``CMUX_WORKSPACE_ID`` has previously let a headless test rename
    the operator's real workspaces.
    """
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()

    def run(*arguments: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("CMUX_")
        }
        environment.update(
            HOME=str(home),
            LOCAL_OPERATOR_CONFIG_DIR=str(config),
            PYTHONPATH=str(REPO_ROOT),
            # Force a colour-capable terminal so a careless `print` that paints
            # would actually emit escapes here. Asserting purity under NO_COLOR
            # would prove nothing.
            TERM="xterm-256color",
        )
        environment.pop("NO_COLOR", None)
        return subprocess.run(
            [sys.executable, "-m", "local_operator.cli", "secret", *arguments],
            input=stdin,
            capture_output=True,
            env=environment,
            timeout=120,
        )

    run.config = config  # type: ignore[attr-defined]
    return run


# --- the stdout purity contract ---------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"ghp_exampletoken1234567890", id="token"),
        pytest.param(b"line one\nline two", id="newlines"),
        pytest.param("s\u00e9cr\u00e8t-\u4f60\u597d-\U0001f510".encode(), id="unicode"),
        pytest.param(b"trailing spaces   ", id="trailing-spaces"),
        pytest.param(b"-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----", id="pem"),
    ],
)
def test_get_writes_exactly_the_value_and_nothing_else(cli, value: bytes) -> None:
    stored = cli("set", "TOKEN", stdin=value + b"\n")
    assert stored.returncode == 0, stored.stderr

    got = cli("get", "TOKEN")
    assert got.returncode == 0, got.stderr
    assert got.stdout == value, f"stdout was {got.stdout!r}, expected {value!r}"


def test_get_adds_no_trailing_newline(cli) -> None:
    """The single most likely corruption, called out explicitly.

    ``$( )`` happens to strip a trailing newline, which is exactly why this is
    easy to ship broken — every other consumer (``subprocess.check_output``, a
    file redirect, a here-string) keeps it.
    """
    cli("set", "TOKEN", stdin=b"value\n")
    got = cli("get", "TOKEN")
    assert got.stdout == b"value"
    assert not got.stdout.endswith(b"\n")


def test_get_emits_no_ansi_escapes_on_a_colour_terminal(cli) -> None:
    cli("set", "TOKEN", stdin=b"value\n")
    got = cli("get", "TOKEN")
    assert b"\x1b" not in got.stdout


def test_a_100kb_value_survives_the_pipe(cli) -> None:
    """Large values cross pipe buffer boundaries; a partial write would truncate."""
    value = bytes(range(256)) * 400
    # Only bytes that survive being written to stdin as-is; the CLI reads the
    # raw buffer, so this is a genuine binary round trip through two pipes.
    cli("set", "BIG", stdin=value + b"\n")
    got = cli("get", "BIG")
    assert len(got.stdout) == 102_400
    assert got.stdout == value


def test_command_substitution_yields_the_value(cli, tmp_path: Path) -> None:
    """The documented usage, exercised through a real shell.

    This is the assertion that matches how an agent actually calls it — the
    whole feature is `$(lop secret get NAME)` interpolated into a command.
    """
    cli("set", "GITHUB_TOKEN", stdin=b"ghp_realtoken\n")
    script = (
        f'value="$({sys.executable} -m local_operator.cli secret get GITHUB_TOKEN)"; '
        'printf "[%s]" "$value"'
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(tmp_path / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(cli.config),
        PYTHONPATH=str(REPO_ROOT),
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, env=environment, timeout=120
    )
    assert result.stdout == b"[ghp_realtoken]", result.stderr


# --- list never prints values ------------------------------------------------


def test_list_never_prints_a_value(cli) -> None:
    cli("set", "ALPHA", "--description", "first", stdin=b"alpha-secret-value\n")
    cli("set", "BETA", stdin=b"beta-secret-value\n")

    listed = cli("list")
    assert listed.returncode == 0, listed.stderr
    assert b"ALPHA" in listed.stdout and b"BETA" in listed.stdout
    assert b"alpha-secret-value" not in listed.stdout
    assert b"beta-secret-value" not in listed.stdout
    assert b"alpha-secret-value" not in listed.stderr
    assert b"beta-secret-value" not in listed.stderr


def test_list_json_never_carries_a_value_field(cli) -> None:
    cli("set", "ALPHA", "--description", "first", stdin=b"alpha-secret-value\n")
    listed = cli("list", "--json")
    records = json.loads(listed.stdout)
    assert [record["name"] for record in records] == ["ALPHA"]
    assert "value" not in records[0]
    assert b"alpha-secret-value" not in listed.stdout


def test_describe_shows_metadata_but_no_value(cli) -> None:
    cli("set", "TOKEN", "--description", "a note", stdin=b"the-secret-value\n")
    described = cli("describe", "TOKEN")
    assert b"a note" in described.stdout
    assert b"the-secret-value" not in described.stdout


def test_set_confirmation_does_not_echo_the_value(cli) -> None:
    """The confirmation goes to stderr and names the size, never the bytes."""
    stored = cli("set", "TOKEN", stdin=b"the-secret-value\n")
    assert stored.stdout == b""
    assert b"the-secret-value" not in stored.stderr
    assert b"stored TOKEN" in stored.stderr


# --- failure cases -----------------------------------------------------------


def test_a_missing_secret_fails_with_empty_stdout(cli) -> None:
    """Critical: a failed `get` must not put ANYTHING on stdout.

    ``$(lop secret get ABSENT)`` assigning an error message to a variable that
    is then sent to an API is precisely the corruption this guards.
    """
    cli("set", "PRESENT", stdin=b"value\n")
    got = cli("get", "ABSENT")
    assert got.returncode == 2
    assert got.stdout == b""
    assert b"No secret named" in got.stderr


def test_a_missing_store_fails_with_empty_stdout(cli) -> None:
    got = cli("get", "ANYTHING")
    assert got.returncode != 0
    assert got.stdout == b""
    assert b"No secret store found" in got.stderr


def test_a_corrupted_record_fails_with_empty_stdout(cli) -> None:
    """Fail closed all the way out to the process boundary."""
    import sqlite3

    cli("set", "TOKEN", stdin=b"value\n")
    database = Path(cli.config) / "secrets" / "store.db"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE secrets SET kind = 'file'")

    got = cli("get", "TOKEN")
    assert got.returncode == 2
    assert got.stdout == b""
    assert b"failed authentication" in got.stderr


def test_loosened_permissions_are_reported_and_stdout_stays_empty(cli) -> None:
    if os.name == "nt":
        pytest.skip("POSIX modes only")
    cli("set", "TOKEN", stdin=b"value\n")
    os.chmod(Path(cli.config) / "secrets" / "master.key", 0o644)

    got = cli("get", "TOKEN")
    assert got.returncode == 2
    assert got.stdout == b""
    assert b"must not be readable by group or others" in got.stderr


def test_set_refuses_an_existing_name(cli) -> None:
    cli("set", "TOKEN", stdin=b"first\n")
    again = cli("set", "TOKEN", stdin=b"second\n")
    assert again.returncode == 2
    assert b"already exists" in again.stderr
    assert cli("get", "TOKEN").stdout == b"first"


def test_there_is_no_value_flag(cli) -> None:
    """argv is readable by any same-uid process, so a --value flag must not exist."""
    result = cli("set", "TOKEN", "--value", "leaked-on-the-command-line")
    assert result.returncode != 0
    assert b"unrecognized arguments" in result.stderr or b"--value" in result.stderr


def test_rm_without_yes_refuses_when_stdin_is_not_a_terminal(cli) -> None:
    cli("set", "TOKEN", stdin=b"value\n")
    removed = cli("rm", "TOKEN", stdin=b"")
    assert removed.returncode == 2
    assert b"--yes" in removed.stderr
    assert cli("get", "TOKEN").stdout == b"value"


def test_rm_with_yes_deletes(cli) -> None:
    cli("set", "TOKEN", stdin=b"value\n")
    removed = cli("rm", "TOKEN", "--yes", stdin=b"")
    assert removed.returncode == 0
    assert cli("get", "TOKEN").returncode == 2


# --- other verbs -------------------------------------------------------------


def test_update_replaces_the_value(cli) -> None:
    cli("set", "TOKEN", stdin=b"first\n")
    updated = cli("update", "TOKEN", stdin=b"second\n")
    assert updated.returncode == 0
    assert cli("get", "TOKEN").stdout == b"second"


def test_status_reports_keyfile_mode_without_overclaiming(cli) -> None:
    """The honesty requirement, asserted as a test.

    Design §9 is explicit that this must not be described as a vault, and
    status is where an operator forms their mental model. The note must state
    the residual risk.
    """
    cli("set", "TOKEN", stdin=b"value\n")
    status = cli("status")
    assert status.returncode == 0
    assert b"keyfile" in status.stdout
    assert b"does not " in status.stdout and b"key file" in status.stdout
    for overclaim in (b"vault", b"unbreakable", b"military", b"impossible"):
        assert overclaim not in status.stdout.lower()


def test_status_json_is_parseable(cli) -> None:
    cli("set", "TOKEN", stdin=b"value\n")
    status = cli("status", "--json")
    payload = json.loads(status.stdout)
    assert payload["key_mode"] == "keyfile"
    assert payload["secrets"] == 1
    assert payload["audit_ok"] is True


def test_audit_verify_reports_an_intact_chain(cli) -> None:
    cli("set", "TOKEN", stdin=b"value\n")
    cli("get", "TOKEN")
    verified = cli("audit", "--verify")
    assert verified.returncode == 0
    assert b"intact" in verified.stdout


def test_audit_verify_detects_an_edit(cli) -> None:
    import sqlite3

    cli("set", "TOKEN", stdin=b"value\n")
    cli("get", "TOKEN")
    cli("get", "TOKEN")
    database = Path(cli.config) / "secrets" / "store.db"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE audit SET event = 'list' WHERE rowid = 2")

    verified = cli("audit", "--verify")
    assert verified.returncode == 1
    assert b"broken at entry 2" in verified.stdout


def test_rotate_preserves_values(cli) -> None:
    cli("set", "ALPHA", stdin=b"alpha-value\n")
    cli("set", "BETA", stdin=b"beta-value\n")
    rotated = cli("rotate")
    assert rotated.returncode == 0, rotated.stderr
    assert cli("get", "ALPHA").stdout == b"alpha-value"
    assert cli("get", "BETA").stdout == b"beta-value"


def test_run_exports_the_secret_to_the_child(cli) -> None:
    cli("set", "API_KEY", stdin=b"key-value\n")
    result = cli("run", "--secret", "API_KEY", "--", "bash", "-c", 'printf "[%s]" "$API_KEY"')
    assert result.stdout == b"[key-value]", result.stderr


def test_run_can_rename_the_variable(cli) -> None:
    cli("set", "API_KEY", stdin=b"key-value\n")
    result = cli(
        "run", "--secret", "API_KEY=OTHER_NAME", "--", "bash", "-c", 'printf "[%s]" "$OTHER_NAME"'
    )
    assert result.stdout == b"[key-value]"


def test_file_materialises_a_readable_path_and_removes_it(cli) -> None:
    """§7's mechanism: a real, re-openable file that is gone afterwards."""
    cli("set", "SA_JSON", "--kind", "file", stdin=b'{"type":"service_account"}\n')
    script = (
        'printf "[%s]" "$(cat "$GOOGLE_APPLICATION_CREDENTIALS")"; '
        # Re-open it: the design rejected FIFOs and /dev/fd because a second
        # open reads zero bytes there, and google-auth opens more than once.
        'printf "[%s]" "$(cat "$GOOGLE_APPLICATION_CREDENTIALS")"; '
        'printf "%s" "$GOOGLE_APPLICATION_CREDENTIALS" > /tmp/lop-secret-path-probe'
    )
    result = cli("file", "SA_JSON", "--", "bash", "-c", script)
    assert result.stdout == b'[{"type":"service_account"}][{"type":"service_account"}]'

    leaked = Path("/tmp/lop-secret-path-probe")
    try:
        materialised = Path(leaked.read_text())
        assert not materialised.exists(), "the plaintext file outlived the command"
        assert not materialised.parent.exists(), "the private directory was not removed"
    finally:
        leaked.unlink(missing_ok=True)


def test_file_uses_a_custom_env_var(cli) -> None:
    cli("set", "CERT", "--kind", "file", stdin=b"cert-body\n")
    result = cli(
        "file",
        "CERT",
        "--env-var",
        "SSL_CERT_FILE",
        "--",
        "bash",
        "-c",
        'printf "[%s]" "$(cat "$SSL_CERT_FILE")"',
    )
    assert result.stdout == b"[cert-body]"


def test_our_own_flags_survive_before_the_command_separator(cli) -> None:
    """Regression: ``argparse.REMAINDER`` swallowed the flags before ``--``.

    With REMAINDER, ``file NAME --env-var VAR -- cmd`` parsed ``--env-var`` as
    part of the child command and silently fell back to the default variable
    name, so the child saw an unset variable. The failure was silent — the
    command ran, and only the wrong variable was set — which is why this is
    pinned for both verbs.
    """
    cli("set", "CERT", "--kind", "file", stdin=b"cert-body\n")
    cli("set", "API_KEY", stdin=b"key-value\n")

    filed = cli(
        "file",
        "CERT",
        "--env-var",
        "SSL_CERT_FILE",
        "--",
        "bash",
        "-c",
        'printf "[%s][%s]" "$SSL_CERT_FILE" "$GOOGLE_APPLICATION_CREDENTIALS"',
    )
    assert filed.stdout.endswith(b"[]"), "the default variable was set instead"
    assert filed.stdout != b"[][]"

    ran = cli(
        "run",
        "--secret",
        "API_KEY=RENAMED",
        "--",
        "bash",
        "-c",
        'printf "[%s][%s]" "$RENAMED" "$API_KEY"',
    )
    assert ran.stdout == b"[key-value][]"


def test_child_command_flags_after_the_separator_are_preserved(cli) -> None:
    """The other half: the child's own flags must reach it untouched."""
    cli("set", "API_KEY", stdin=b"key-value\n")
    result = cli("run", "--secret", "API_KEY", "--", "bash", "-c", 'printf -- "-n [%s]" "$API_KEY"')
    assert result.stdout == b"-n [key-value]", result.stderr


def test_unlock_refuses_a_store_that_is_not_hardened(cli) -> None:
    """``unlock`` on a keyfile store has nothing to unlock, and says so.

    Replaces PR 1's "the broker does not exist yet" assertion: the broker ships
    here, so the honest failure is now about the store's TIER rather than about
    a missing capability.
    """
    cli("set", "API_KEY", stdin=b"key-value\n")
    result = cli("unlock")
    assert result.returncode == 2
    assert b"keyfile mode" in result.stderr
    assert b"harden" in result.stderr
    assert result.stdout == b""


def test_harden_refuses_without_a_terminal_rather_than_reading_a_flag(cli) -> None:
    """A passphrase never comes from argv; argv is readable by any same-uid process.

    With stdin not a terminal ``getpass`` cannot prompt, and the CLI must fail
    with that explanation instead of inventing a ``--passphrase`` flag.
    """
    cli("set", "API_KEY", stdin=b"key-value\n")
    result = cli("harden", stdin=b"")
    assert result.returncode == 2
    assert result.stdout == b""
    assert b"terminal" in result.stderr or b"passphrase" in result.stderr


def test_secret_help_does_not_promise_a_vault(cli) -> None:
    """CLI help is documentation; §9 forbids overclaiming there too."""
    helped = cli("--help")
    text = helped.stdout.lower()
    for overclaim in (b"vault", b"unbreakable", b"military-grade", b"hacker-proof"):
        assert overclaim not in text
