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
import pty
import subprocess
import sys
import time
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
    first = cli("set", "TOKEN", stdin=b"first\n")
    # Asserted explicitly so a failure of the FIRST set is attributable. QA saw
    # this test fail once as `assert 0 == 2` — the second set succeeding, which
    # means the first set's row was absent — and could not reproduce it in ~250
    # attempts. Without this line the symptom is laundered into a confusing
    # assertion about the second command; with it, the next occurrence names
    # itself.
    assert first.returncode == 0, f"the FIRST set failed: rc={first.returncode} {first.stderr!r}"
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


def test_file_materialises_a_readable_path_and_removes_it(cli, tmp_path: Path) -> None:
    """§7's mechanism: a real, re-openable file that is gone afterwards.

    The probe path is under ``tmp_path`` and NOT a fixed ``/tmp`` name. A
    shared path made this test delete a concurrent run's probe on a machine
    that is explicitly worked through many worktrees at once: four simultaneous
    invocations reproducibly gave two passes and two ``FileNotFoundError``
    failures, which the next agent has to diagnose from scratch.
    """
    cli("set", "SA_JSON", "--kind", "file", stdin=b'{"type":"service_account"}\n')
    probe = tmp_path / "materialised-path"
    script = (
        'printf "[%s]" "$(cat "$GOOGLE_APPLICATION_CREDENTIALS")"; '
        # Re-open it: the design rejected FIFOs and /dev/fd because a second
        # open reads zero bytes there, and google-auth opens more than once.
        'printf "[%s]" "$(cat "$GOOGLE_APPLICATION_CREDENTIALS")"; '
        f'printf "%s" "$GOOGLE_APPLICATION_CREDENTIALS" > "{probe}"'
    )
    result = cli("file", "SA_JSON", "--", "bash", "-c", script)
    assert result.stdout == b'[{"type":"service_account"}][{"type":"service_account"}]'

    materialised = Path(probe.read_text())
    assert not materialised.exists(), "the plaintext file outlived the command"
    assert not materialised.parent.exists(), "the private directory was not removed"


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


def _typed_cli(config: Path, arguments: list[str], answers: list[str]) -> tuple[int, str]:
    """Run ``lop secret ...`` on a real pty, typing ``answers`` at each prompt.

    ``harden`` and ``unlock`` read through ``getpass``, which opens
    ``/dev/tty``: without a pty they take the "no terminal" refusal branch and
    the success path is never exercised. That is exactly why CI missed QA's Q2
    — the only ``unlock`` test asserted the keyfile rejection, which returns
    before the broker is contacted.
    """
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(config.parent / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(config),
        PYTHONPATH=str(REPO_ROOT),
        TERM="xterm-256color",
    )
    environment.pop("NO_COLOR", None)

    pid, handle = pty.fork()
    if pid == 0:  # pragma: no cover - the child execs immediately
        os.environ.clear()
        os.environ.update(environment)
        os.execv(sys.executable, [sys.executable, "-m", "local_operator.cli", "secret", *arguments])
    pending = list(answers)
    captured = b""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            chunk = os.read(handle, 4096)
        except OSError:  # the child closed the pty
            break
        if not chunk:
            break
        captured += chunk
        # Answer as each prompt appears, and keep draining meanwhile: a pty
        # buffer that fills while nobody reads blocks the child on write.
        if pending and b"assphrase" in chunk:
            os.write(handle, (pending.pop(0) + "\n").encode())
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), captured.decode(errors="replace")


def test_harden_restart_unlock_get_round_trip(cli) -> None:
    """The passphrase tier must be enterable AND usable (QA Q2).

    The tier the design nominates as load-bearing had never worked end to end
    in either direction: `unlock` was dispatched behind the ancestry gate, so
    it required descending from a registered session — while nothing in
    shipping code registered one. After `harden` the correct passphrase was
    refused and even `status` failed, with the plaintext key already deleted.

    This drives the whole operator journey through the real CLI, including the
    broker restart that stands in for a reboot, because every step in
    isolation passed while the sequence did not.
    """
    config: Path = cli.config
    passphrase = "journey-passphrase"

    assert cli("set", "JOURNEY", stdin=b"journey-value\n").returncode == 0
    assert cli("get", "JOURNEY").stdout == b"journey-value"

    code, output = _typed_cli(config, ["harden"], [passphrase, passphrase])
    assert code == 0, output
    assert "hardened" in output

    try:
        # A reboot, in effect: the unlocked key lives only in broker memory.
        assert cli("broker", "restart").returncode == 0

        # Before unlocking there is no key anywhere, so this must fail cleanly
        # and print nothing on stdout.
        locked = cli("get", "JOURNEY")
        assert locked.returncode == 2
        assert locked.stdout == b""

        code, output = _typed_cli(config, ["unlock"], [passphrase])
        assert code == 0, output
        assert "Unlocked" in output

        # And the store is reachable again from the operator's own terminal.
        served = cli("get", "JOURNEY")
        assert served.returncode == 0, served.stderr
        assert served.stdout == b"journey-value"

        status = cli("status")
        assert status.returncode == 0, status.stderr
        assert b"passphrase" in status.stdout

        # **The whole lifecycle, not just `get` (QA Q9).** This test stopped at
        # `get`, and that is precisely why the tier shipped unable to accept a
        # NEW secret: `set` was the one verb passing `create=True`, which
        # short-circuited the broker and hit the "no key on disk" refusal, so a
        # hardened store was frozen at whatever it held when it was hardened —
        # with no workaround, since `update` refuses unknown names. That is the
        # harden-then-migrate sequence this tier exists for. One `set` after the
        # unlock would have caught it, so every verb the operator needs after
        # unlocking is exercised here rather than one.
        added = cli("set", "JOURNEY_TWO", stdin=b"second-value\n")
        assert added.returncode == 0, added.stderr
        assert cli("get", "JOURNEY_TWO").stdout == b"second-value"

        assert cli("update", "JOURNEY_TWO", stdin=b"updated-value\n").returncode == 0
        assert cli("get", "JOURNEY_TWO").stdout == b"updated-value"

        listed = cli("list")
        assert listed.returncode == 0, listed.stderr
        assert b"JOURNEY_TWO" in listed.stdout and b"JOURNEY" in listed.stdout

        assert cli("rm", "JOURNEY_TWO", "--yes").returncode == 0
        assert cli("get", "JOURNEY_TWO").returncode == 2
        assert b"JOURNEY_TWO" not in cli("list").stdout

        # **`rotate`, and then the whole store again after it (QA Q10).** The
        # journey stopped short of the one verb that REPLACES the key, and that
        # is exactly the verb that was not tier-aware: it installed a plaintext
        # `master.key` while `unlock` kept unwrapping the old one, so a single
        # rotation made every secret undecryptable AND silently undid the
        # hardening, with `status` still reporting `passphrase`. Any verb that
        # can run on a hardened store belongs in this sequence.
        code, output = _typed_cli(config, ["rotate"], [passphrase, passphrase])
        assert code == 0, output
        assert "rotated 1 secret(s)" in output

        # The tier's defining property (design §2.3) survives the rotation.
        assert not (config / "secrets" / "master.key").exists(), "plaintext key after rotate"
        assert (config / "secrets" / "master.key.wrapped").exists()

        # A reboot after the rotation: the NEW key must be the one the wrapped
        # file yields, which is what the pre-fix code got wrong.
        assert cli("broker", "restart").returncode == 0
        code, output = _typed_cli(config, ["unlock"], [passphrase])
        assert code == 0, output

        survived = cli("get", "JOURNEY")
        assert survived.returncode == 0, survived.stderr
        assert survived.stdout == b"journey-value"

        status = cli("status")
        assert b"passphrase" in status.stdout
        assert b"WARNING" not in status.stdout
    finally:
        cli("broker", "stop")


def test_rotate_on_a_hardened_store_keeps_every_secret(cli) -> None:
    """QA Q10, as reported: one rotation, exit 0, and the store was destroyed.

    The repro verbatim — set, harden, stop the broker (a reboot), unlock,
    rotate — after which `get` returned "No secret named 'API_KEY'" and
    `status` reported `secrets 0 / damaged 1`. Two assertions, because the bug
    had two independent consequences and either one alone would have let it
    ship: the value must survive, AND no plaintext master key may exist
    afterwards. The second is the tier's whole claim (design §2.3) and it
    failed silently — `key_mode` answered `passphrase` from the stale wrapped
    file while the live key sat unwrapped beside the database.
    """
    config: Path = cli.config
    passphrase = "rotate-passphrase"

    assert cli("set", "API_KEY", stdin=b"hunter2").returncode == 0
    code, output = _typed_cli(config, ["harden"], [passphrase, passphrase])
    assert code == 0, output

    try:
        assert cli("broker", "stop").returncode == 0
        code, output = _typed_cli(config, ["unlock"], [passphrase])
        assert code == 0, output
        assert cli("get", "API_KEY").stdout == b"hunter2"

        code, output = _typed_cli(config, ["rotate"], [passphrase, passphrase])
        assert code == 0, output

        # Assertion 1: the data survives a rotation on a hardened store. The
        # broker restart stands in for the reboot the operator would next do,
        # and proves the WRAPPED file now yields the post-rotation key.
        assert cli("broker", "restart").returncode == 0
        code, output = _typed_cli(config, ["unlock"], [passphrase])
        assert code == 0, output
        served = cli("get", "API_KEY")
        assert served.returncode == 0, served.stderr
        assert served.stdout == b"hunter2"

        # Assertion 2: no plaintext master key on disk, ever, in this tier.
        assert not (config / "secrets" / "master.key").exists()
        assert cli("status").stdout.count(b"damaged") == 0
    finally:
        cli("broker", "stop")


def test_harden_repairs_a_store_a_pre_fix_rotate_damaged(cli) -> None:
    """An operator who already hit Q10 has a CLI way out (QA Q10, recovery).

    The damaged state is a plaintext `master.key` beside a STALE
    `master.key.wrapped`. Before this fix there was no exit: `unlock` still
    returned 0 while decrypting nothing, and `harden` refused with "already
    hardened" because it saw the stale wrapped file. `key_mode` now answers on
    the file that decides the tier — a plaintext key means `keyfile` — so
    `harden` reaches the store and re-wraps the LIVE key.

    The damage is recreated through the real key primitives rather than by
    hand-writing files, so this stays a test of the recovery path and not of
    the fixture's idea of what the bug looked like.
    """
    config: Path = cli.config
    passphrase = "repair-passphrase"

    assert cli("set", "API_KEY", stdin=b"hunter2").returncode == 0
    code, output = _typed_cli(config, ["harden"], [passphrase, passphrase])
    assert code == 0, output

    try:
        assert cli("broker", "stop").returncode == 0
        code, output = _typed_cli(config, ["unlock"], [passphrase])
        assert code == 0, output

        damage = (
            "import sys; sys.path.insert(0, %r)\n"
            "from local_operator.secrets.access import open_store\n"
            "from local_operator.secrets.crypto import generate_master_key\n"
            "from local_operator.secrets.keys import stage_master_key\n"
            "from local_operator.secrets.store import install_master_key_if_current\n"
            "store = open_store(); key = generate_master_key()\n"
            "stage_master_key(None, key)\n"
            "store.rotate(key); install_master_key_if_current(key)\n" % str(REPO_ROOT)
        )
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("CMUX_")
        }
        environment.update(
            HOME=str(config.parent / "home"),
            LOCAL_OPERATOR_CONFIG_DIR=str(config),
            PYTHONPATH=str(REPO_ROOT),
        )
        broken = subprocess.run(
            [sys.executable, "-c", damage], capture_output=True, env=environment, timeout=120
        )
        assert broken.returncode == 0, broken.stderr
        assert (config / "secrets" / "master.key").exists(), "failed to recreate the damage"

        # `status` must NAME the inconsistency rather than reporting a tier it
        # is no longer providing.
        status = cli("status")
        assert status.returncode == 0, status.stderr
        assert b"WARNING" in status.stdout
        assert b"plaintext master key" in status.stdout

        # And `harden` is the way out: it re-wraps the live key.
        code, output = _typed_cli(config, ["harden"], [passphrase, passphrase])
        assert code == 0, output
        assert not (config / "secrets" / "master.key").exists()

        assert cli("broker", "restart").returncode == 0
        code, output = _typed_cli(config, ["unlock"], [passphrase])
        assert code == 0, output
        recovered = cli("get", "API_KEY")
        assert recovered.returncode == 0, recovered.stderr
        assert recovered.stdout == b"hunter2"
    finally:
        cli("broker", "stop")
