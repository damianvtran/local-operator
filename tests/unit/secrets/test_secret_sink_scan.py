"""The pre-execution secret-sink scan: the rule table, the non-findings, refusal.

**What this file is for.** ``local_operator/harness/secret_sinks.py`` refuses a
bash call or eval cell in which a stored secret reaches a printing sink, before
any child process exists. Everything else in ``tests/unit/secrets`` guards
redaction — a value that has ALREADY been printed, masked on the way out. This
file guards the half that decides not to print, so it is written against the
same three things the module is: the rule table, the four required
non-findings, and the two tools that call it.

Two kinds of test live here deliberately:

* **Table-driven** — every entry in ``RULES.examples`` must fire its own rule and
  every entry in ``RULES.counterexamples`` must not. A rule whose examples stop
  firing fails here rather than drifting into dead policy, and a reviewer can
  read one rule's block and check it without holding the whole table.
* **Behavioural** — the refusal through ``execute_bash``/``execute_eval``, the
  sanctioned form working end to end against a real consumer with a real store,
  and the fail-closed boundary. A unit test that the helper returns its own
  verdict is not evidence that the leak path is closed.

The end-to-end cases run ``lop`` from a **PATH shim that points at this
worktree's own console script**, never the machine's global install: the global
one reads the operator's live store, and a test must never do that.
"""

from __future__ import annotations

import base64
import http.server
import json
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.harness.secret_sinks import (
    RULE_LABELS,
    RULES,
    refusal_text,
    scan_command,
    scan_python,
)
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools import builtin
from local_operator.tools.eval import execute_eval
from local_operator.variables import VariableStore

#: A synthetic value. Nothing here is a real credential, and the shape is
#: deliberately unlike a token so a failure message cannot be mistaken for one.
_SYNTHETIC = "synthetic-sink-scan-value-9f2c41"
_SECRET_NAME = "SINK_SCAN_TEST_TOKEN"


def _scan(rule_label: str, text: str):  # type: ignore[no-untyped-def]
    spec = next(item for item in RULES if item.label == rule_label)
    return scan_python(text) if spec.lang == "python" else scan_command(text)


@pytest.mark.parametrize("rule_label", [item.label for item in RULES])
def test_every_rule_fires_its_examples_and_not_its_counterexamples(rule_label: str) -> None:
    """The table is the test corpus: one rule, read and checked on its own.

    ``consumer``-verdict rules are the allow half of the table and never appear
    in a finding (the allowance IS the absence of a refusal), so their examples
    assert "allowed and not this rule" instead of "this label fired".
    """
    spec = next(item for item in RULES if item.label == rule_label)
    for example in spec.examples:
        result = _scan(rule_label, example)
        labels = {finding.rule for finding in result.findings}
        if spec.verdict in ("printing", "unresolved"):
            assert spec.label in labels, f"{spec.label} did not fire on {example!r}: {result}"
        else:
            assert not result.refused, f"{spec.label} refused its own example {example!r}"
            assert spec.label not in labels
    for counterexample in spec.counterexamples:
        result = _scan(rule_label, counterexample)
        labels = {finding.rule for finding in result.findings}
        assert spec.label not in labels, f"{spec.label} fired on counterexample {counterexample!r}"


def test_the_table_has_unique_labels_and_the_code_only_uses_table_labels() -> None:
    """Bidirectional: no duplicate rule, and no label the code invents.

    ``_add`` resolves its label through ``rule(label)``, which raises on an
    unknown one — so a typo'd label would be a crash inside a tool call rather
    than a red test. Reading the literals out of the module source catches that
    at the source, and the reachability half (no rule is dead) is covered by
    :func:`test_every_rule_fires_its_examples_and_not_its_counterexamples`.
    """
    labels = [item.label for item in RULES]
    assert len(labels) == len(set(labels))
    source = Path("local_operator/harness/secret_sinks.py").read_text()
    used = set(re.findall(r'_add\(\s*"([a-z0-9_.-]+)"', source))
    assert used, "no rule labels found in the module source — the guard is inert"
    assert (
        used <= RULE_LABELS
    ), f"labels used by the code but absent from RULES: {used - RULE_LABELS}"


# ---------------------------------------------------------------------------
# The four required non-findings
# ---------------------------------------------------------------------------


def test_non_finding_i_the_sanctioned_inline_use_is_allowed() -> None:
    """(i) `v=$(lop secret get X)` then a curl header / client argv.

    This is the form the guide recommends, and Case B's intent: the value
    crosses a pipe into the child and never enters the transcript. A guard that
    refused it would be worked around within a day.
    """
    allowed = [
        'v=$(lop secret get GITHUB_TOKEN); curl -H "Authorization: Bearer $v" https://x',
        'curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://x',
        'v=$(lop secret get GITHUB_TOKEN); docker login --username u --password-stdin <<< "$v"',
        'v=$(lop secret get GITHUB_TOKEN); gcloud auth activate-service-account --key-file "$v"',
    ]
    for command in allowed:
        result = scan_command(command)
        assert result.verdict == "consumer", (command, result)
        assert not result.refused


def test_a_copy_of_a_written_value_carries_the_read_rule_with_it() -> None:
    """`cp f g; cat g` is the same leak one path later.

    The read rule tracks the *value*, not one filename: a copy does not print
    anything, but it is what would make a targeted read rule trivial to walk
    around.
    """
    refused = scan_command(
        "lop secret get [redacted] > /tmp/tok; cp /tmp/tok /tmp/copy; cat /tmp/copy"
    )
    assert refused.refused
    assert refused.findings[0].rule == "shell.read-of-secret-file-path"
    # ... and the copy alone is still the guide-sanctioned contained form.
    assert not scan_command("lop secret get [redacted] > /tmp/tok; cp /tmp/tok /tmp/copy").refused


def test_non_finding_ii_length_only_is_allowed() -> None:
    """(ii) `${#VAR}` and `lop secret get X | wc -c` — a length is not a value.

    ``handlers._set`` prints ``len(value)`` on purpose ("reveals nothing"), and
    this is the same claim: the pipeline ends in a sink that cannot return the
    bytes.
    """
    allowed = [
        "lop secret get GITHUB_TOKEN | wc -c",
        "lop secret get GITHUB_TOKEN | wc -m",
        "lop secret get GITHUB_TOKEN | shasum -a 256",
        'v=$(lop secret get GITHUB_TOKEN); echo "${#v}"',
        "v=$(lop secret get GITHUB_TOKEN); printf '%s' \"$v\" | wc -c",
    ]
    for command in allowed:
        assert not scan_command(command).refused, command


def test_non_finding_iii_the_value_free_verbs_are_not_sources() -> None:
    """(iii) `lop secret list`, `get --help`, `describe` — no value exists.

    `list` never decrypts a value (there is no flag that prints one), `--help`
    short-circuits before the handler, and `describe` answers with metadata. So
    there is nothing to refuse, and refusing would train the model away from the
    surfaces that exist to replace reading a value.
    """
    for command in (
        "lop secret list",
        "lop secret list --json",
        "lop secret get --help",
        "lop secret describe GITHUB_TOKEN",
        "lop secret status",
    ):
        result = scan_command(command)
        assert result.verdict == "none", (command, result)
        assert not result.sources


def test_non_finding_iv_a_heredoc_that_only_writes_the_pattern_is_not_flagged() -> None:
    """(iv) Case B's literal shape: a heredoc WRITING a script that contains it.

    The discriminating test, and the reason the module tokenizes instead of
    matching: the pattern is in a REGION, and a quoted here-doc body is literal
    — nothing there is executed, nothing is printed. The unquoted spelling is
    the same write with a real expansion, whose bytes go to the file rather than
    to this result; both must stay unflagged, and both are asserted because a
    pattern match gets the quoted one right by luck and the unquoted one wrong.
    """
    quoted = (
        "cat > /tmp/probe.sh <<'EOF'\n"
        'ADMIN_API_KEY="$(lop secret get MINERVA_API_KEY_DEV)"\n'
        "EOF"
    )
    unquoted = (
        "cat > /tmp/probe.sh <<EOF\n"
        'ADMIN_API_KEY="$(lop secret get MINERVA_API_KEY_DEV)"\n'
        "EOF"
    )
    for command in (quoted, unquoted):
        result = scan_command(command)
        assert not result.refused, (command, result)
    assert scan_command(quoted).verdict == "none", "a literal region is not a source"
    assert scan_command(quoted).sources == ()
    # The same pattern WITHOUT the file target is a different command: `cat`
    # prints the expanded body, so the rule must still fire there.
    assert scan_command("cat <<EOF\n$(lop secret get X)\nEOF").verdict == "printing"


def test_a_refused_printing_form_stays_refused_however_it_is_respelled() -> None:
    """The incident's aggravating act was a re-spelling to defeat the mask.

    ``rev``, ``base64``, a substring, a different quoting, a here-string: the
    rule keys on the flow, so none of them launders it. This is the property
    that makes the guard a guard rather than a filter with a longer word list,
    and it is asserted on the re-spellings the incident used.
    """
    respellings = [
        'v=$(lop secret get GITHUB_TOKEN); echo "$v" | rev',
        'v=$(lop secret get GITHUB_TOKEN); base64 <<< "$v"',
        'v=$(lop secret get GITHUB_TOKEN); echo "$v" | tr -d "-"',
        'v=$(lop secret get GITHUB_TOKEN); echo "${v:0:8}"',
        'v=$(lop secret get GITHUB_TOKEN); printf %s "$v"',
        "lop secret get GITHUB_TOKEN | rev",
        'echo "$(lop secret get GITHUB_TOKEN | rev)"',
        'v=$(lop secret get GITHUB_TOKEN); od -c <<< "$v"',
    ]
    for command in respellings:
        result = scan_command(command)
        assert result.refused, f"a re-spelling evaded the scan: {command!r} -> {result.verdict}"


def test_an_ordinary_command_is_never_touched() -> None:
    """The blast radius is exactly the set of commands that fetch a secret.

    A scan that can refuse a build, a test run or a process listing for a lexing
    fault would be turned off; a guard nobody leaves on protects nothing. The
    `ps`, `set -x` and `grep` cases here are the ones a wider rule WOULD have
    caught.
    """
    ordinary = [
        "ls -la /tmp",
        "python -m pytest -q",
        "terraform apply -auto-approve",
        "ps -ef | grep python",
        "set -x; make build",
        "sh -c 'echo hello'",
        "grep -rn 'lop secret get' docs/",
        "awk '{print $1}' /var/log/system.log",
        'v=$(date); echo "$v"',
        'for f in *.py; do wc -c "$f"; done',
    ]
    for command in ordinary:
        result = scan_command(command)
        assert result.verdict == "none", (command, result)


def test_the_prefilter_is_sound_about_which_text_can_hold_a_source() -> None:
    """The prefilter is what keeps the scan off ordinary commands, so it is
    asserted rather than assumed: text it rejects cannot produce a value, and the
    quoting-piece spelling (`lop sec"ret" get X`) is exactly why it cannot key on
    the literal phrase alone.
    """
    from local_operator.harness.secret_sinks import may_carry_a_shell_source

    assert not may_carry_a_shell_source("ls -la /tmp")
    assert not may_carry_a_shell_source("make build")
    assert not may_carry_a_shell_source("terraform apply -auto-approve")
    assert may_carry_a_shell_source("lop secret get NAME")
    assert may_carry_a_shell_source('echo "$(lop secret get NAME)"')
    assert may_carry_a_shell_source('lop sec"ret" get NAME')


def test_fail_closed_at_the_source_boundary_only() -> None:
    """Unparseable text refuses ONLY when a source could really be in it.

    A false refusal costs one re-spelling (the message hands over the accepted
    form); a false allow is a credential in the transcript, which nothing
    undoes. That asymmetry justifies the fail-closed half — and the condition
    that bounds it: no source, no refusal, so an ordinary broken command is
    nobody's business but the shell's.
    """
    assert scan_command('echo "$(lop secret get GITHUB_TOKEN)').verdict == "unresolved"
    assert scan_command("cat <<EOF\n$(lop secret get GITHUB_TOKEN)\n").verdict == "unresolved"
    assert scan_command("echo 'unterminated and no secret here").verdict == "none"
    assert scan_python('token = secrets["GITHUB_TOKEN"]\nprint(token').verdict == "unresolved"
    assert scan_python("x = 1\nprint(x").verdict == "none"


def test_the_scanner_never_raises_on_hostile_text() -> None:
    """Every prefix of a real command, plus junk: a refusal, never a traceback.

    A scan on the tool path that can raise turns one malformed command into a
    broken tool call, and the model cannot tell that from a refusal it should
    act on.
    """
    sample = (
        "v=$(lop secret get 'X'); cat <<'EOF' > /tmp/f\n$(lop secret get X)\nEOF\n"
        'if [ -n "$v" ]; then echo "$v" | rev; fi; ps -ef & set -x'
    )
    junk = ["", "\\", "$(", "`", "<<EOF", "'", '"', "\x00\x01", "a" * 5000, "$((" * 40]
    for command in [sample[:index] for index in range(len(sample) + 1)] + junk:
        result = scan_command(command)  # must not raise
        assert result.verdict in ("consumer", "printing", "none", "unresolved")
        if result.refused:
            assert result.findings
            assert result.findings[0].rewrite


def test_the_module_imports_nothing_from_the_package() -> None:
    """It sits on the tool path, so it must be stdlib-only and cheap to import.

    Same reason ``secrets/cli.py`` keeps the crypto stack off the CLI's path,
    and the same thing ``tests/unit/test_import_graph.py`` pins for the startup
    graph: one module-level import here costs every bash call in every session.
    """
    probe = (
        "import json, importlib, sys\n"
        "importlib.import_module(sys.argv[1])\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, "local_operator.harness.secret_sinks"],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert not any(name.startswith("local_operator.secrets") for name in loaded), [
        name for name in loaded if name.startswith("local_operator.secrets")
    ]
    for heavy in ("sqlite3", "cryptography", "pydantic", "asyncio", "httpx", "requests"):
        assert heavy not in loaded, (heavy, [name for name in loaded if heavy in name])


# ---------------------------------------------------------------------------
# The refusal through the tools
# ---------------------------------------------------------------------------


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), variables=VariableStore(cwd=str(tmp_path)))


@pytest.mark.asyncio
async def test_execute_bash_refuses_a_printing_command_before_anything_runs(
    tmp_path: Path,
) -> None:
    """The gate: an error result naming rule, span and rewrite — and no side effect.

    The marker file is the point. A refusal that returned the message after the
    child had already run would leave the value in the transcript, so the test
    asserts the command did NOT execute rather than that the message is present.
    """
    marker = tmp_path / "ran"
    command = f'v=$(lop secret get GITHUB_TOKEN); echo "$v"; touch {marker}'
    result = await builtin.execute_bash(
        "bash-sink", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    assert result.is_error
    text = "".join(str(getattr(part, "text", "")) for part in result.content)
    assert "shell.print-of-source" in text
    assert "secret: GITHUB_TOKEN" in text
    assert "curl -H" in text, "the rewrite must be in the message"
    assert "guide://credentials" in text
    assert not marker.exists(), "the command ran anyway — the refusal was not pre-execution"


@pytest.mark.asyncio
async def test_a_session_credential_is_out_of_the_scan_s_blast_radius(tmp_path: Path) -> None:
    """Documented BOUNDARY, not an oversight: this scan's source is the store.

    ``echo $NAME`` for a credential the harness injected into the child really
    does put a value in this result, and the design doc's source list includes
    that namespace — but the delegation ruled the gate condition to be
    "text that fetches from ``lop secret``", and the wider rule is not free: it
    refuses the command two existing defences measure
    (``test_bash_injects_session_credentials_and_redacts_them_from_output``,
    whose subject is the mask, and the R1 fd-2 crash tail). This test pins the
    boundary so a later change that widens it has to come here and say so — and
    asserts the existing mask still does its job in the meantime.
    """

    class _CredentialStore(VariableStore):
        """A real store holding one injected credential, as a proxied session has."""

        def credential_env(self) -> dict[str, str]:
            return {"SESSION_TOKEN": _SYNTHETIC}

    assert scan_command('printf %s "$SESSION_TOKEN"').verdict == "none"
    context = ToolContext(cwd=str(tmp_path), variables=_CredentialStore(cwd=str(tmp_path)))
    result = await builtin.execute_bash(
        "bash-inline", {"command": 'printf %s "$SESSION_TOKEN"'}, AbortSignal(), None, context
    )
    assert not result.is_error, result.text
    assert _SYNTHETIC not in result.text, "the mask must still scrub an echoed credential"
    assert "[redacted]" in result.text


@pytest.mark.asyncio
async def test_execute_eval_refuses_a_printing_cell_before_the_kernel_sees_it(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"
    code = f'open("{marker}", "w").write("ran")\nprint(secrets["{_SECRET_NAME}"])'
    result = await execute_eval(
        "eval-sink", {"code": code}, AbortSignal(), None, _context(tmp_path)
    )
    assert result.is_error
    text = "".join(str(getattr(part, "text", "")) for part in result.content)
    assert "python.print-of-source" in text
    assert not marker.exists(), "the cell ran anyway — the refusal was not pre-execution"


@pytest.mark.asyncio
async def test_execute_eval_still_runs_a_consuming_cell(tmp_path: Path) -> None:
    """The refusal must not have become a ban on retrieving a value at all."""
    code = 'blob = "not-a-secret"\nprint(len(blob))'
    result = await execute_eval("eval-ok", {"code": code}, AbortSignal(), None, _context(tmp_path))
    assert not result.is_error


# ---------------------------------------------------------------------------
# End to end: a real store, a real consumer
# ---------------------------------------------------------------------------


def _worktree_lop() -> Path:
    """This worktree's own console script — never the machine's global install."""
    candidate = Path(sys.executable).parent / "lop"
    if not candidate.exists():  # pragma: no cover - a venv without console scripts
        pytest.skip("no `lop` console script beside the test interpreter")
    return candidate


class _AuthEchoHandler(http.server.BaseHTTPRequestHandler):
    """Answers `authorized` only for the exact bearer this test stored."""

    expected: str = ""

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        header = self.headers.get("Authorization", "")
        ok = header == f"Bearer {self.expected}"
        body = b"authorized\n" if ok else b"unauthorized\n"
        self.send_response(200 if ok else 403)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the request log; http.server's own signature is positional."""
        return


@pytest.fixture
def local_authorizer() -> Iterator[str]:
    """A localhost endpoint that verifies the header and never echoes it."""
    handler = type("_Handler", (_AuthEchoHandler,), {"expected": _SYNTHETIC})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def stored_secret(config_root: Path) -> str:
    """Store the synthetic value through the REAL CLI, in the isolated store."""
    proc = subprocess.run(
        [str(_worktree_lop()), "secret", "set", _SECRET_NAME],
        input=_SYNTHETIC.encode(),
        capture_output=True,
        cwd=str(config_root),
    )
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]
    listing = subprocess.run(
        [str(_worktree_lop()), "secret", "list"], capture_output=True, cwd=str(config_root)
    )
    assert _SECRET_NAME in listing.stdout.decode(), listing.stderr.decode()[-2000:]
    return _SECRET_NAME


@pytest.fixture
def shimmed_path(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put this worktree's `lop` first on the child's PATH."""
    shim = config_root / "bin"
    shim.mkdir(exist_ok=True)
    link = shim / "lop"
    if not link.exists():
        link.symlink_to(_worktree_lop())
    import os

    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.asyncio
async def test_the_sanctioned_form_still_works_end_to_end(
    tmp_path: Path, config_root: Path, local_authorizer: str, stored_secret: str, shimmed_path: None
) -> None:
    """Real store, real CLI, real consumer, real header — and no value in the result.

    The endpoint answers from the header it received, so a 200 proves the value
    arrived intact; it answers with a WORD rather than echoing the header, so a
    pass cannot be an artefact of redaction hiding the very leak under test.
    """
    commands = [
        f"v=$(lop secret get {_SECRET_NAME}); "
        f'curl -s -H "Authorization: Bearer $v" {local_authorizer}',
        f'curl -s -H "Authorization: Bearer $(lop secret get {_SECRET_NAME})" '
        f"{local_authorizer}",
    ]
    for command in commands:
        result = await builtin.execute_bash(
            "bash-e2e-ok", {"command": command}, AbortSignal(), None, _context(tmp_path)
        )
        text = "".join(str(getattr(part, "text", "")) for part in result.content)
        assert not result.is_error, text
        assert "authorized" in text and "unauthorized" not in text, text
        assert _SYNTHETIC not in text, "the value reached the tool result"


@pytest.mark.asyncio
async def test_the_length_only_form_works_end_to_end(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """`lop secret get NAME | wc -c` prints the length and nothing else."""
    result = await builtin.execute_bash(
        "bash-e2e-len",
        {"command": f"lop secret get {_SECRET_NAME} | wc -c"},
        AbortSignal(),
        None,
        _context(tmp_path),
    )
    text = "".join(str(getattr(part, "text", "")) for part in result.content)
    assert not result.is_error, text
    assert _SYNTHETIC not in text
    assert str(len(_SYNTHETIC)) in text, text


@pytest.mark.asyncio
async def test_the_incident_shape_is_refused_through_the_real_tool(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """Case A, verbatim: the loop that leaked, refused with the rewrite.

    Reproduced as the incident wrote it — a loop over names, a substitution into
    a variable, and an `echo` per name — and asserted against the refusal text
    rather than a helper's verdict.
    """
    marker = tmp_path / "ran"
    command = (
        f'for k in {_SECRET_NAME} MISSING_NAME; do v=$(lop secret get "$k") '
        f'&& echo "$k = $v" || echo "$k = MISSING"; done; touch {marker}'
    )
    result = await builtin.execute_bash(
        "bash-e2e-incident", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    text = "".join(str(getattr(part, "text", "")) for part in result.content)
    assert result.is_error, text
    assert "shell.print-of-source" in text, text
    assert "do:" in text, "the rewrite must be present"
    assert not marker.exists(), "the loop ran anyway"
    assert _SYNTHETIC not in text


@pytest.mark.asyncio
async def test_case_b_heredoc_runs_and_writes_the_literal_pattern(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """The false positive the design must not reintroduce: it RUNS, unrefused.

    A quoted here-doc body is literal, so the file it writes contains the
    spelling and the tool result contains neither the value nor a refusal —
    which is what an agent writing a script expects.
    """
    target = tmp_path / "probe.sh"
    command = (
        f"cat > {target} <<'EOF'\n"
        f'ADMIN_API_KEY="$(lop secret get {_SECRET_NAME})"\n'
        "EOF\n"
        f"wc -c < {target}"
    )
    result = await builtin.execute_bash(
        "bash-e2e-heredoc", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    text = "".join(str(getattr(part, "text", "")) for part in result.content)
    assert not result.is_error, text
    written = target.read_text()
    assert f"$(lop secret get {_SECRET_NAME})" in written, "the literal was not written"
    assert _SYNTHETIC not in written
    assert _SYNTHETIC not in text


def test_refusal_text_names_rule_span_and_rewrite() -> None:
    """The message is the interface the model has to act on."""
    command = 'v=$(lop secret get GITHUB_TOKEN); echo "$v"'
    result = scan_command(command)
    message = refusal_text(result, text=command, tool_name="bash")
    assert "shell.print-of-source" in message
    assert "span " in message and "of the command" in message
    assert "GITHUB_TOKEN" in message
    assert "^" in message, "the offending span must be marked"
    assert "curl -H" in message


# ---------------------------------------------------------------------------
# Remediation round 1 — one test per finding, because each was a hypothesis the
# suite could not falsify before (the table carried no backtick, no `export`ed
# value, no `read`, and no derived-value print).
# ---------------------------------------------------------------------------


def _result_text(result: object) -> str:
    """The text of a tool result, spelled once because three tests read it."""
    parts = getattr(result, "content", ())
    return "".join(str(getattr(part, "text", "")) for part in parts)


def test_r1_1_a_backtick_substitution_is_a_substitution_not_a_word_break() -> None:
    """The POSIX spelling of `$( )` is the same flow, and was invisible.

    ``echo "`+backtick+`lop secret get X`+backtick+`"`` answered `none` while the
    child ran and the RAW value landed in the tool result — the incident's harm,
    in a spelling no rule example named, which is how it stayed green.
    """
    refused = [
        'echo "`lop secret get [redacted]`"',
        'v="`lop secret get [redacted]`"; echo "$v"',
        "echo `lop secret get [redacted]`",
        "sh -c 'echo `lop secret get [redacted]`'",
        'echo "`lop secret get [redacted] | rev`"',
        'cat "`lop secret get [redacted]`"',
    ]
    for command in refused:
        result = scan_command(command)
        assert result.refused, f"a backtick evaded the scan: {command!r} -> {result.verdict}"
        assert result.findings, command
        assert result.findings[0].rewrite
    # The same spelling with nothing to fetch is still nobody's business.
    assert scan_command("echo `date`").verdict == "none"
    assert scan_command("echo `hostname` | rev").verdict == "none"


@pytest.mark.asyncio
async def test_r1_1_the_backtick_leak_is_refused_before_the_child_runs(tmp_path: Path) -> None:
    """R1-1 through the real tool, with a marker proving nothing executed."""
    marker = tmp_path / "ran"
    command = f'echo "`lop secret get [redacted]`"; touch {marker}'
    result = await builtin.execute_bash(
        "bash-backtick", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    assert result.is_error
    text = _result_text(result)
    assert "shell.print-of-source" in text
    # The lex fault must not be "empty word" any more: the call lexes.
    assert "unresolved" not in text
    assert not marker.exists(), "the child ran anyway — the refusal was not pre-execution"


def test_r1_2_an_exported_value_is_refused_at_any_dump_of_the_environment() -> None:
    """`export V=$(…); printenv V` handed back the raw value (R1-2).

    None of these puts the value in a printer's argv, which is exactly why the
    emitter rule could not see them: the printer is `printenv`, `env` or `set`,
    and the operand is the environment.
    """
    refused = [
        "export V=$(lop secret get [redacted]); printenv V",
        "V=$(lop secret get [redacted]); export V; env | grep V=",
        "declare -x V=$(lop secret get [redacted]); printenv",
        "export V=$(lop secret get [redacted]); env",
        "V=$(lop secret get [redacted]); set | grep V=",
        "export V=$(lop secret get [redacted]); export",
        "export V=$(lop secret get [redacted]); declare -p V",
        "export V=$(lop secret get [redacted]); export -p",
    ]
    for command in refused:
        result = scan_command(command)
        assert result.refused, f"{command!r} -> {result.verdict}"
        assert "shell.environment-dump-of-source" in {item.rule for item in result.findings}
    allowed = [
        # A dumper with a real operand is a consumer handing the value on.
        'v=$(lop secret get [redacted]); env V="$v" some-client --flag',
        "printenv PATH",
        "printenv HOME",
        "set -e",
        "env",
        "export MY_FLAG=1",
        # `declare` without `-x` binds and does not export (R2-4): printenv
        # prints nothing, so this is not a dump of the value.
        "declare V=$(lop secret get [redacted]); printenv",
        # `declare -x V` marks a variable for export and prints nothing.
        "V=$(lop secret get [redacted]); declare -x V",
    ]
    for command in allowed:
        assert not scan_command(command).refused, command


def test_r1_3_a_value_read_by_the_read_builtin_is_tainted() -> None:
    """`read -r l < <(lop secret get X)` binds with no `=` anywhere (R1-3)."""
    result = scan_command('read -r l < <(lop secret get [redacted]); echo "$l"')
    assert result.refused
    assert result.findings[0].rule == "shell.print-of-source"
    # A `read` from an ordinary source is untouched.
    assert not scan_command('read -r line < /etc/hosts; echo "$line"').refused
    # ... and so is a `read` that never sees the value.
    assert scan_command('v=$(lop secret get [redacted]); read -r x <<< "$v"; echo "$x"').refused


def test_r1_4_an_apostrophe_in_an_unquoted_heredoc_body_is_not_a_lex_fault() -> None:
    """A body is data with expansion, not shell text (R1-4).

    Lexing it as a command list made `It's fine` an unterminated quote, refused
    a call that lexes, and diagnosed it with the wrong span. The refusal it
    produced was also not the one the fail-closed paragraph justifies.
    """
    sanctioned = (
        "cat <<EOF > /tmp/m\nIt's fine\nEOF\n"
        'v=$(lop secret get [redacted]); curl -H "Bearer $v" https://x'
    )
    result = scan_command(sanctioned)
    assert result.verdict == "consumer", result
    assert not result.refused
    assert result.fault == ""

    printing = "cat <<EOF > /tmp/m\nIt's fine\nEOF\n" 'v=$(lop secret get [redacted]); echo "$v"'
    refused = scan_command(printing)
    assert refused.refused
    # The RIGHT reason: the `echo`, not a lex fault about the apostrophe.
    assert {item.rule for item in refused.findings} == {"shell.print-of-source"}
    assert refused.fault == ""

    # The quoted spelling of the same body is still the discriminating
    # non-finding: nothing runs and nothing is refused.
    assert (
        scan_command("cat <<'EOF' > /tmp/m\nIt's fine\n$(lop secret get X)\nEOF").verdict == "none"
    )


def test_q1_a_print_of_something_derived_from_the_request_is_allowed() -> None:
    """The boundary QA measured: a response is not the value it was built with.

    The argument test was argument-inclusive, so `resp = get(url, headers=…)`
    poisoned every later print — of `resp.status`, of `str(resp.status)`, of a
    child's exit code — while bash allows printing a whole curl response body.
    """
    prefix = (
        'import requests\n\ntoken = secrets["[redacted]"]\n'
        'req = requests.Request(url, headers={"Authorization": f"Bearer {token}"})\n'
        "resp = requests.Session().send(req.prepare())\n"
    )
    allowed = [
        prefix + "print(resp.status)",
        prefix + "print(str(resp.status))",
        prefix + "print(int(resp.status))",
        prefix + "print(resp.url)",
        prefix + "print(len(resp.read()))",
        prefix + "print(resp.headers['Content-Type'])",
        'import subprocess\n\ntoken = secrets["[redacted]"]\n'
        'done = subprocess.run(["curl", "-H", "Authorization: Bearer " + token, url], '
        "capture_output=True)\n"
        'print("rc", done.returncode, "len", len(done.stdout))',
    ]
    for cell in allowed:
        result = scan_python(cell)
        assert not result.refused, f"{cell!r} -> {result.verdict} {result.labels}"
    # The value itself is still refused, through every spelling that carries it.
    refused = [
        'token = secrets["[redacted]"]\nprint(token)',
        'token = secrets["[redacted]"]\nprint(token[:8])',
        'token = secrets["[redacted]"]\nprint(f"Bearer {token}")',
        'token = secrets["[redacted]"]\nprint(str(token))',
        'token = secrets["[redacted]"]\nprint(repr(token))',
        'token = secrets["[redacted]"]\nblob = token.encode()\nprint(blob)',
    ]
    for cell in refused:
        assert scan_python(cell).refused, cell


def test_q2_a_multi_line_cell_points_the_caret_at_the_call() -> None:
    """A Python span is ``(line, column)``, so the snippet must be that line.

    Read as offsets it showed the model line 1 of its own cell with a caret in a
    meaningless column — the refusal naming the wrong line is worse than naming
    none.
    """
    cell = 'import urllib.request\n\ntoken = secrets["[redacted]"]\nprint(token)'
    message = refusal_text(scan_python(cell), text=cell, tool_name="eval")
    here = message.split("here:")[1].split("why:")[0]
    assert "print(token)" in here, message
    assert "import urllib.request" not in here, message
    assert "cell line 4" in message


# ---------------------------------------------------------------------------
# Remediation round 2 — the reviewer's measured rows, in both directions: each
# refusal restored AND each counterexample consumer still allowed, because R2-1
# and R2-4 are the same premise (bind vs export) read from opposite sides.
# ---------------------------------------------------------------------------

#: R2-1: every one of these handed back the RAW value through the real tool at
#: d99767f7 (bash 3.2.57 measured: bare `declare` lists a non-exported `v`, and
#: `env -0`/`env -u OTHER` still print the whole environment).
_R2_1_DUMPS = [
    "v=$(lop secret get NAME); declare",
    "v=$(lop secret get NAME); typeset",
    "v=$(lop secret get NAME); typeset -p v",
    "v=$(lop secret get NAME); declare -p",
    "export V=$(lop secret get NAME); env -0 | tr '\\0' '\\n' | grep '^V='",
    "export V=$(lop secret get NAME); env -u HOME | grep '^V='",
    "export V=$(lop secret get NAME); env --null",
    "V=$(lop secret get NAME) printenv V",
    "set -a; V=$(lop secret get NAME); printenv V",
    "readonly V=$(lop secret get NAME); readonly",
    'v=$(lop secret get NAME); env V="$v"',
]


def test_r2_1_a_dump_s_reach_is_what_the_shell_did_with_the_name() -> None:
    for command in _R2_1_DUMPS:
        result = scan_command(command)
        assert result.refused, f"{command!r} -> {result.verdict}"
        assert result.labels == ("shell.environment-dump-of-source",), (command, result.labels)
    allowed = [
        # `env` that RUNS a command is a consumer, flags or not.
        'v=$(lop secret get NAME); env V="$v" some-client --flag',
        "export V=$(lop secret get NAME); env -u HOME some-client",
        "export V=$(lop secret get NAME); env -i some-client",
        # Listings that cannot hold a bound string.
        "v=$(lop secret get NAME); declare -f",
        "v=$(lop secret get NAME); set -o",
        # No source at all: the bare dumpers are ordinary commands.
        "declare",
        "typeset",
        "env -0",
    ]
    for command in allowed:
        assert not scan_command(command).refused, command


def test_r2_4_binding_is_not_exporting() -> None:
    """`local`/`readonly`/`declare` without `-x` put nothing in a child's env.

    Measured on bash 3.2.57: `f() { local V=…; printenv V; }; f` and
    `readonly V=…; printenv V` print nothing and exit 1, so refusing them was a
    false refusal whose diagnosis described nothing the command did.
    """
    allowed = [
        "f() { local V=$(lop secret get NAME); printenv V; }; f",
        "readonly V=$(lop secret get NAME); printenv V",
        "declare V=$(lop secret get NAME); printenv",
        "declare V=$(lop secret get NAME); env",
        "export V=$(lop secret get NAME); export -n V; printenv V",
    ]
    for command in allowed:
        result = scan_command(command)
        assert not result.refused, f"{command!r} -> {result.verdict} {result.labels}"
    # ... while the spellings that DO export are refused.
    refused = [
        "declare -x V=$(lop secret get NAME); printenv V",
        "typeset -x V=$(lop secret get NAME); env",
        "f() { local -x V=$(lop secret get NAME); printenv V; }; f",
        "declare V=$(lop secret get NAME); export V; printenv V",
    ]
    for command in refused:
        assert scan_command(command).refused, command


def test_r2_3_the_dump_caret_points_at_the_dumper() -> None:
    """`(0, 0)` is truthy, so the old `span or …` fallback never fell back."""
    command = "export V=$(lop secret get NAME); printenv V"
    result = scan_command(command)
    start, end = result.findings[0].span
    assert command[start:end] == "printenv", result.findings[0].span
    # And the rendered caret sits under it: the `here:` and caret lines share a
    # 10-column prefix, so the columns compare directly.
    lines = refusal_text(result, text=command, tool_name="bash").splitlines()
    here = next(index for index, line in enumerate(lines) if line.startswith("  here:"))
    caret = lines[here + 1]
    assert caret.index("^") == lines[here].index("printenv"), "\n".join(lines[here : here + 2])
    assert caret.strip() == "^" * len("printenv")


#: R2-2: `print(...)` of a method call whose VALUE is built from its argument.
#: The base commit refused every row; d99767f7 allowed all of them.
_R2_2_PRINTS = [
    'token = secrets["NAME"]\nprint("".join([token]))',
    'token = secrets["NAME"]\nprint(",".join([token]))',
    'token = secrets["NAME"]\nprint("{}".format(token))',
    'token = secrets["NAME"]\nprint("".replace("", token))',
    'token = secrets["NAME"]\nblob = ",".join([token])\nprint(blob)',
    'token = secrets["NAME"]\nsep = ","\nblob = sep.join([token])\nprint(blob)',
    'import base64\ntoken = secrets["NAME"]\nenc = base64.b64encode(token.encode())\nprint(enc)',
    'import json\ntoken = secrets["NAME"]\nd = json.dumps({"t": token})\nprint(d)',
    'token = secrets["NAME"]\nprint(requests.get(url, headers={"A": token}).status_code)',
]


def test_r2_2_an_argument_carrying_method_is_still_the_value() -> None:
    for cell in _R2_2_PRINTS:
        result = scan_python(cell)
        assert result.refused, f"{cell!r} -> {result.verdict}"
        assert "python.print-of-source" in result.labels
    # The Q1 boundary, which the reviewer re-checked against this direction: a
    # RESPONSE built with the value is not the value.
    prefix = (
        'token = secrets["NAME"]\n'
        'resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})\n'
    )
    allowed = [
        prefix + "print(resp.status_code)",
        prefix + "print(resp.url)",
        prefix + "print(resp.headers['x'])",
        prefix + "resp",
        'token = secrets["NAME"]\n'
        'done = subprocess.run(["curl", "-H", "Authorization: Bearer " + token, url], '
        "capture_output=True)\n"
        'print("rc", done.returncode, "len", len(done.stdout))',
        'token = secrets["NAME"]\nprint(len(token))',
        'token = secrets["NAME"]\nprint(len(",".join([token])))',
    ]
    for cell in allowed:
        result = scan_python(cell)
        assert not result.refused, f"{cell!r} -> {result.verdict} {result.labels}"


def test_nit_1_one_rule_twice_is_not_also_refused_by_nothing() -> None:
    cell = 'token = secrets["NAME"]\nprint(repr(token))'
    result = scan_python(cell)
    message = refusal_text(result, text=cell, tool_name="eval")
    assert "also refused by" not in message, message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "v=$(lop secret get {name}); declare",
        "v=$(lop secret get {name}); typeset -p v",
        "export V=$(lop secret get {name}); env -u HOME | grep '^V='",
    ],
)
async def test_r2_1_the_dump_is_refused_before_the_child_runs(
    command: str,
    tmp_path: Path,
    config_root: Path,
    stored_secret: str,
    shimmed_path: None,
) -> None:
    """R2-1 through the real tool, the real CLI and a real isolated store."""
    marker = tmp_path / "ran"
    full = command.format(name=stored_secret) + f"; touch {marker}"
    result = await builtin.execute_bash(
        "bash-r2-1", {"command": full}, AbortSignal(), None, _context(tmp_path)
    )
    text = _result_text(result)
    assert result.is_error, text
    assert "shell.environment-dump-of-source" in text
    assert not marker.exists(), "the child ran anyway"
    assert _SYNTHETIC not in text


@pytest.mark.asyncio
async def test_r2_4_a_bound_but_unexported_value_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """The R2-4 false refusal, fixed: it runs, and prints nothing of the value."""
    marker = tmp_path / "ran"
    command = (
        f"f() {{ local V=$(lop secret get {stored_secret}); printenv V; }}; f; "
        f"echo rc=$?; touch {marker}"
    )
    result = await builtin.execute_bash(
        "bash-r2-4", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    text = _result_text(result)
    assert not result.is_error, text
    assert marker.exists()
    assert "rc=1" in text, text
    assert _SYNTHETIC not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "print_line",
    [
        'print(",".join([token]))',
        'print("{}".format(token))',
        'blob = "".join([token])\nprint(blob)',
    ],
)
async def test_r2_2_the_carrier_print_is_refused_before_the_kernel_sees_it(
    print_line: str, tmp_path: Path
) -> None:
    marker = tmp_path / "ran"
    code = f'open("{marker}", "w").write("ran")\ntoken = secrets["{_SECRET_NAME}"]\n{print_line}'
    result = await execute_eval(
        "eval-r2-2", {"code": code}, AbortSignal(), None, _context(tmp_path)
    )
    assert result.is_error
    assert "python.print-of-source" in _result_text(result)
    assert not marker.exists(), "the cell ran anyway"


# ---------------------------------------------------------------------------
# Round-3 remediation: R3-1…R3-4, and the `KEY=value` argv fold-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # R3-1: each of these prints the value the verb put in the child's
        # environment, and none of them was examined before this round — the
        # consumer scan stopped at the `--secret` word `run` always carries.
        "lop secret run --secret [redacted] -- printenv NAME",
        "lop secret run --secret [redacted] -- printenv NAME | rev",
        "lop secret run --secret NAME=TOKEN -- env",
        "lop secret run --secret NAME=TOKEN -- set",
        "lop secret run --secret NAME=TOKEN -- sh -c 'echo \"$TOKEN\" | rev'",
        "lop secret run --secret NAME=TOKEN -- sh -c 'printenv TOKEN | base64'",
        "lop secret run --secret NAME=TOKEN -- python3 -c 'import os;"
        ' print(os.environ["TOKEN"])\'',
        # The separator-less spelling is real: argparse consumes the `--` it
        # uses, so the child command follows the options directly.
        "lop secret run --secret [redacted] printenv NAME",
        # `--env-var` moves the FILE secret's path, not the leak.
        "lop secret file GCP_SA_JSON --env-var TOKFILE -- cat",
    ],
)
def test_r3_1_the_runs_consumer_is_checked_for_the_side_the_value_is_on(command: str) -> None:
    """The verb's consumer is refused when it prints the value it was handed.

    ``file`` hands over a PATH, ``run`` hands over an ENVIRONMENT, so the same
    consumer word can leak through one and not the other — which is why the two
    arms are asserted separately below. This one pins the leaking half.
    """
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)
    assert "shell.secret-verb-emitting-consumer" in result.labels


@pytest.mark.parametrize(
    "command",
    [
        # The consumers the verb exists for: a client, a filter, a request, an
        # attribute-setting builtin, and a length-only python check.
        "lop secret run --secret NAME=TOKEN -- python3 client.py",
        "lop secret run --secret [redacted] -- sed -n 1p /etc/hosts",
        "lop secret run --secret [redacted] -- cat",
        "lop secret run --secret [redacted] -- declare -x TOKEN",
        "lop secret run --secret NAME=TOKEN -- env V=1 client",
        "lop secret run --secret NAME=TOKEN -- sh -c 'curl -sS -H \"Authorization:"
        " Bearer $TOKEN\" http://127.0.0.1:9/ >/dev/null; echo done'",
        "lop secret run --secret NAME=TOKEN -- python3 -c 'import os,sys;"
        ' sys.exit(0 if os.environ["TOKEN"] else 1)\'',
        "lop secret file GCP_SA_JSON -- python3 -c 'import os;"
        ' print(os.path.getsize(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]))\'',
        "lop secret file GCP_SA_JSON -- /bin/true",
    ],
)
def test_r3_1_the_verbs_own_consumers_stay_allowed(command: str) -> None:
    """No over-refusal: an argument-only printer reaches no environment value."""
    result = scan_command(command)
    assert not result.refused, (command, result)


@pytest.mark.parametrize(
    "command",
    [
        # R3-2: the other console script for the same entry point,
        "local-operator secret get [redacted]",
        'v=$(local-operator secret get [redacted]); echo "$v" | rev',
        # the precommand wrappers, each with its own option grammar
        "command lop secret get [redacted]",
        "command lop secret get [redacted] | base64",
        "exec lop secret get [redacted]",
        "env lop secret get [redacted]",
        "nohup lop secret get [redacted] >/dev/stdout 2>/dev/null",
        "timeout 30 lop secret get [redacted] | rev",
        "timeout -s KILL 30 stdbuf -oL lop secret get [redacted]",
        'v=$(timeout 30 lop secret get [redacted]); echo "$v" | rev',
        'v=$(nice -n 5 lop secret get [redacted]); command echo "$v" | rev',
        'env V="$(lop secret get [redacted])" printenv V | rev',
        # a name bound to the program
        "l=lop; $l secret get [redacted]",
        'l=lop; v=$("$l" secret get [redacted]); echo "$v" | rev',
        # and the value in command position, which bash prints back
        "v=$(lop secret get [redacted]); $v",
    ],
)
def test_r3_2_the_reach_covers_the_program_names_and_the_wrappers(command: str) -> None:
    """R3-2: a source is a source however the program name is spelled."""
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)


@pytest.mark.parametrize(
    "command",
    [
        "command -v lop",
        "timeout 5 sleep 0.1",
        "env | grep -c lop",
        "lop secret get [redacted] > /tmp/contained 2>/dev/null; echo contained",
        'v=$(lop secret get [redacted]); curl -H "Authorization: Bearer $v" https://x',
    ],
)
def test_r3_2_the_wrapper_reach_does_not_refuse_ordinary_commands(command: str) -> None:
    result = scan_command(command)
    assert not result.refused, (command, result)


@pytest.mark.parametrize(
    "command",
    [
        # R3-3: the ways a value is re-bound that the `NAME=$(…)` rule never saw.
        "for x in $(lop secret get [redacted]); do echo $x; done",
        'v=$(lop secret get [redacted]); a=("$v"); echo "${a[@]}"',
        'v=$(lop secret get [redacted]); a[0]="$v"; echo "${a[0]}"',
        'v=$(lop secret get [redacted]); v+=$(lop secret get [redacted]); echo "$v"',
        'lop secret get [redacted] > /tmp/f; read -r l < /tmp/f; echo "$l"',
        'lop secret get [redacted] > /tmp/f; mapfile -t a < /tmp/f; echo "${a[0]}"',
        'lop secret get [redacted] > /tmp/f; l=$(< /tmp/f); echo "$l"',
        'lop secret get [redacted] > /tmp/f; l=$(cat /tmp/f); echo "$l"',
        "lop secret get [redacted] > /tmp/f; cat /tmp/f",
        # stderr is part of this result, and a `2>` beside it changes nothing
        "lop secret get [redacted] 2>/dev/null",
        "lop secret get [redacted] >&2",
    ],
)
def test_r3_3_a_rebinding_carries_the_value(command: str) -> None:
    """R3-3: bound by a loop, an array, a builtin, a file — still the value."""
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)


def test_r3_3_an_export_declared_before_the_assignment_still_exports() -> None:
    """`export V; V=$(…)` exports the later binding (bash: measured, rc 0)."""
    result = scan_command("export V; V=$(lop secret get [redacted]); printenv V")
    assert result.verdict == "printing"
    assert "shell.environment-dump-of-source" in result.labels
    # …and an `export` whose name is never bound reaches nothing of ours.
    assert not scan_command("export V; printenv V").refused


def test_r3_3_a_loop_over_a_literal_list_binds_nothing() -> None:
    """The loop binding is the iterable's taint, not the `for` keyword."""
    result = scan_command('for k in A B; do echo "$k"; done; echo done')
    assert not result.refused


@pytest.mark.parametrize(
    "cell",
    [
        # R3-4: bindings and sinks the eval walker missed.
        'token = secrets["NAME"]\nfor c in token: print(c)',
        'token = secrets["NAME"]\nfor i, c in enumerate(token): print(c)',
        'token = secrets["NAME"]\nfor c in token[::-1]: print(c, end="")',
        'alias = secrets\nprint(alias["NAME"][::-1])',
        'token = secrets["NAME"]\nraise ValueError(token[::-1])',
        'token = secrets["NAME"]\nassert False, token',
        'import warnings\ntoken = secrets["NAME"]\nwarnings.warn(token[::-1])',
        'import pprint\ntoken = secrets["NAME"]\npprint.pprint(token[::-1])',
    ],
)
def test_r3_4_the_eval_walker_sees_the_loop_the_alias_and_the_exception(cell: str) -> None:
    result = scan_python(cell)
    assert result.verdict == "printing", (cell, result)


@pytest.mark.parametrize(
    "cell",
    [
        'token = secrets["NAME"]\nprint(len(token))',
        'assert len(secrets["NAME"]) > 0',
        'token = secrets["NAME"]\nassert token',
        "raise ValueError('plain')",
        "for i in range(3):\n    print(i)",
        'import requests\ntoken = secrets["NAME"]\nresp = requests.get("https://x",'
        ' headers={"Authorization": f"Bearer {token}"})\nprint(resp.status_code)',
    ],
)
def test_r3_4_the_new_eval_rules_do_not_refuse_derived_observations(cell: str) -> None:
    result = scan_python(cell)
    assert not result.refused, (cell, result)


@pytest.mark.parametrize(
    "command",
    [
        # Fold-in (round 4): the `=` inside an ARGUMENT is not a shell
        # assignment, and skipping assignment-shaped words stage-wide let the
        # incident's own `KEY = value` print through as a consumer.
        "v=$(lop secret get [redacted]); echo TOKEN=$v",
        'v=$(lop secret get [redacted]); echo "TOKEN=$v" | rev',
        'v=$(lop secret get [redacted]); printf "%s\\n" TOKEN=$v',
    ],
)
def test_the_key_value_print_is_not_a_binding(command: str) -> None:
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)


@pytest.mark.parametrize(
    "command",
    [
        "V=1 echo TOKEN=literal",
        "v=$(lop secret get [redacted]); printf '%s\\n' X=$v > /tmp/contained",
        "export V=1; printenv V",
    ],
)
def test_a_real_prefix_assignment_is_still_a_binding(command: str) -> None:
    """The skip is positional: only words BEFORE the command word bind."""
    result = scan_command(command)
    assert not result.refused, (command, result)


@pytest.mark.asyncio
async def test_r3_1_the_run_leak_is_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """R3-1 through the REAL tool, CLI and store — the leak that started the round.

    `lop secret run --secret N -- printenv N` returned the value raw on the
    previous head, and `… | rev` returned it reversed past the mask. The marker
    makes `ran=False` the pre-execution property rather than a reading of the
    refusal text alone.
    """
    for command in (
        f"lop secret run --secret {stored_secret} -- printenv {stored_secret}",
        f"lop secret run --secret {stored_secret} -- printenv {stored_secret} | rev",
    ):
        marker = tmp_path / "ran"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r3-1",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        text = _result_text(result)
        assert result.is_error, text
        assert "shell.secret-verb-emitting-consumer" in text
        assert not marker.exists(), "the child ran anyway"
        assert _SYNTHETIC not in text
        assert _SYNTHETIC[::-1] not in text


@pytest.mark.asyncio
async def test_r3_2_the_wrapped_and_renamed_program_is_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """The console script's other name and a wrapper prefix, through the tool."""
    for command in (
        f"local-operator secret get {stored_secret}",
        f'v=$(timeout 30 lop secret get {stored_secret}); echo "$v" | rev',
        f"l=lop; $l secret get {stored_secret}",
    ):
        marker = tmp_path / "ran"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r3-2",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        text = _result_text(result)
        assert result.is_error, (command, text)
        assert not marker.exists(), "the child ran anyway"
        assert _SYNTHETIC not in text
        assert _SYNTHETIC[::-1] not in text


@pytest.mark.asyncio
async def test_the_key_value_print_is_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """The fold-in row, through the real tool: `KEY=value` in an ARGUMENT."""
    marker = tmp_path / "ran"
    command = f'v=$(lop secret get {stored_secret}); echo "TOKEN=$v" | rev; touch {marker}'
    result = await builtin.execute_bash(
        "bash-key-value", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    text = _result_text(result)
    assert result.is_error, text
    assert not marker.exists(), "the child ran anyway"
    assert _SYNTHETIC not in text
    assert _SYNTHETIC[::-1] not in text


@pytest.mark.asyncio
async def test_r3_3_the_file_read_back_rows_are_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """The `$(<f)` / `read` / `mapfile` spellings, through the real tool."""
    for command in (
        f'lop secret get {stored_secret} > {tmp_path}/f; l=$(< {tmp_path}/f); echo "$l" | rev',
        f"lop secret get {stored_secret} > {tmp_path}/f; read -r l < {tmp_path}/f;"
        ' echo "$l" | rev',
        f"for x in $(lop secret get {stored_secret}); do echo $x; done | rev",
    ):
        marker = tmp_path / "ran"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r3-3",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        text = _result_text(result)
        assert result.is_error, (command, text)
        assert not marker.exists(), "the child ran anyway"
        assert _SYNTHETIC not in text
        assert _SYNTHETIC[::-1] not in text


@pytest.mark.asyncio
async def test_r3_4_the_loop_and_the_exception_are_refused_before_the_kernel_sees_it(
    tmp_path: Path,
) -> None:
    """The eval half, through the real kernel, with a marker written first."""
    for print_line in (
        "for c in token: print(c)",
        "raise ValueError(token[::-1])",
        "import warnings\nwarnings.warn(token[::-1])",
    ):
        marker = tmp_path / "ran"
        if marker.exists():
            marker.unlink()
        code = (
            f'open("{marker}", "w").write("ran")\ntoken = secrets["{_SECRET_NAME}"]\n'
            f"{print_line}"
        )
        result = await execute_eval(
            "eval-r3-4", {"code": code}, AbortSignal(), None, _context(tmp_path)
        )
        assert result.is_error, (print_line, _result_text(result))
        assert not marker.exists(), "the cell ran anyway"


@pytest.mark.asyncio
async def test_r3_1_the_run_consumer_still_delivers_the_value(
    tmp_path: Path,
    config_root: Path,
    local_authorizer: str,
    stored_secret: str,
    shimmed_path: None,
) -> None:
    """The sanctioned `run` form still works end to end after the new arm.

    A pass here cannot be the mask hiding a leak: the endpoint answers
    `authorized` only for the exact bearer and answers with a word rather than
    echoing the header.
    """
    marker = tmp_path / "ran"
    command = (
        f"lop secret run --secret {stored_secret}=TOK -- sh -c "
        f"'curl -sS -H \"Authorization: Bearer $TOK\" {local_authorizer}'; touch {marker}"
    )
    result = await builtin.execute_bash(
        "bash-r3-1-ok", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    text = _result_text(result)
    assert not result.is_error, text
    assert marker.exists()
    assert "authorized" in text, text
    assert _SYNTHETIC not in text


# ---------------------------------------------------------------------------
# Round 4: the `run`/`file` consumer shares the wrapper grammar (R4-1), reads
# more inline languages and `xargs` (R4-2), and walks `file`'s shell program
# with the path variable bound (R4-3). Every row below ran unrefused on
# `bcdec7c49` and put a spelling of the value in the tool result.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # R4-1: the unwrapped spelling of each was already refused.
        "lop secret run --secret NAME=TOK -- command printenv TOK | rev",
        "lop secret run --secret NAME=TOK -- env printenv TOK | rev",
        "lop secret run --secret NAME=TOK -- timeout 5 printenv TOK | rev",
        "lop secret run --secret NAME=TOK -- nice printenv TOK | rev",
        "lop secret run --secret NAME=TOK -- stdbuf -oL printenv TOK | rev",
        "lop secret run --secret NAME=TOK -- env sh -c 'echo $TOK' | rev",
        "lop secret run --secret NAME=TOK -- timeout 5 python3 -c "
        "'import os;print(os.environ[\"TOK\"][::-1])'",
        "lop secret run --secret NAME=TOK -- timeout -s KILL 5 nice -n 3 printenv TOK",
        "lop secret run --secret NAME=TOK -- env -u HOME printenv TOK",
        # R4-2: other interpreters, and `xargs` in front of the consumer.
        "lop secret run --secret NAME=TOK -- perl -e 'print scalar reverse $ENV{TOK}'",
        "lop secret run --secret NAME=TOK -- ruby -e 'puts ENV[\"TOK\"].reverse'",
        "lop secret run --secret NAME=TOK -- node -e "
        '\'console.log(process.env.TOK.split("").reverse().join(""))\'',
        "lop secret run --secret NAME=TOK -- node -p 'process.env.TOK'",
        "lop secret run --secret NAME=TOK -- awk 'BEGIN{print ENVIRON[\"TOK\"]}' | rev",
        "lop secret run --secret NAME=TOK -- python3.12 -c 'import os;print(os.environ[\"TOK\"])'",
        "lop secret run --secret NAME=TOK -- bash -lc 'echo \"$TOK\" | rev'",
        "echo x | lop secret run --secret NAME=TOK -- xargs printenv TOK | rev",
        # R4-3: `file`'s inline shell program, with the default and a named
        # `--env-var`, and the same through a wrapper and in Python/Perl.
        "lop secret file NAME -- sh -c 'rev \"$GOOGLE_APPLICATION_CREDENTIALS\"'",
        "lop secret file NAME -- timeout 5 sh -c 'rev \"$GOOGLE_APPLICATION_CREDENTIALS\"'",
        "lop secret file NAME --env-var KF -- sh -c 'rev \"$KF\"'",
        "lop secret file NAME -- python3 -c "
        "'import os;print(open(os.environ[\"GOOGLE_APPLICATION_CREDENTIALS\"]).read()[::-1])'",
        "lop secret file NAME -- perl -e "
        "'open(F,$ENV{GOOGLE_APPLICATION_CREDENTIALS}); print reverse <F>'",
    ],
)
def test_r4_the_verbs_consumer_is_found_through_wrappers_and_interpreters(command: str) -> None:
    """R4-1/R4-2/R4-3: the consumer check sees the command that really runs."""
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)
    assert "shell.secret-verb-emitting-consumer" in result.labels


@pytest.mark.parametrize(
    "command",
    [
        "echo NAME | xargs lop secret get | rev",
        "xargs lop secret get <<< NAME | rev",
        "echo NAME | xargs lop secret get",
    ],
)
def test_r4_2_xargs_in_front_of_the_source_is_the_source(command: str) -> None:
    """The NAME is on `xargs`'s stdin, but the fetch — and its stdout — are the same."""
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)


@pytest.mark.parametrize(
    "command",
    [
        # The wrapped consumer that NEEDS the value is still the sanctioned form.
        "lop secret run --secret NAME=TOK -- timeout 5 curl -sS "
        '-H "Authorization: Bearer $TOK" http://127.0.0.1:9/',
        "lop secret run --secret NAME=TOK -- nice python3 client.py",
        # `env -i` starts the child empty: nothing `run` exported survives.
        "lop secret run --secret NAME=TOK -- env -i printenv",
        # A program in a FILE is the stated residual, not a refusal.
        "lop secret run --secret NAME=TOK -- perl script.pl",
        "lop secret run --secret NAME=TOK -- node app.js",
        # An interpreter that never reads the environment.
        "lop secret run --secret NAME=TOK -- awk '{print $1}' /etc/hosts",
        "lop secret run --secret NAME=TOK -- ruby -e 'puts 1'",
        # `file`'s program that reads a FACT about the path, or hands it on.
        "lop secret file NAME -- sh -c 'wc -c < \"$GOOGLE_APPLICATION_CREDENTIALS\"'",
        "lop secret file NAME -- sh -c 'gcloud auth activate-service-account "
        '--key-file "$GOOGLE_APPLICATION_CREDENTIALS"\'',
        "lop secret file NAME -- python3 -c "
        "'import os;print(os.path.getsize(os.environ[\"GOOGLE_APPLICATION_CREDENTIALS\"]))'",
        # `xargs` outside a verb keeps its own branch.
        "find . -name '*.py' | xargs grep -l TODO",
        "echo a b | xargs -n1 echo",
    ],
)
def test_r4_the_wider_consumer_reach_does_not_refuse_the_sanctioned_forms(command: str) -> None:
    result = scan_command(command)
    assert not result.refused, (command, result)


def test_r4_3_the_default_file_variable_matches_the_cli() -> None:
    """The scanner mirrors `lop secret file`'s default rather than importing the CLI."""
    from local_operator.harness import secret_sinks
    from local_operator.secrets.cli import DEFAULT_FILE_ENV_VAR

    assert secret_sinks._DEFAULT_FILE_ENV_VAR == DEFAULT_FILE_ENV_VAR


@pytest.mark.asyncio
async def test_r4_wrapped_and_interpreted_consumers_are_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """R4-1/R4-2/R4-3 through the REAL tool: each of these leaked on `bcdec7c49`."""
    commands = [
        f"lop secret run --secret {stored_secret}=TOK -- timeout 5 printenv TOK | rev",
        f"lop secret run --secret {stored_secret}=TOK -- env sh -c 'echo $TOK' | rev",
        f"lop secret run --secret {stored_secret}=TOK -- timeout 5 python3 -c "
        "'import os;print(os.environ[\"TOK\"][::-1])'",
        f"lop secret run --secret {stored_secret}=TOK -- "
        "awk 'BEGIN{print ENVIRON[\"TOK\"]}' | rev",
        f"echo {stored_secret} | xargs lop secret get | rev",
        f"lop secret file {stored_secret} -- sh -c 'rev \"$GOOGLE_APPLICATION_CREDENTIALS\"'",
    ]
    for command in commands:
        marker = tmp_path / "ran"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r4",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        text = _result_text(result)
        assert result.is_error, (command, text)
        assert not marker.exists(), f"the child ran anyway: {command}"
        assert _SYNTHETIC not in text
        assert _SYNTHETIC[::-1] not in text


@pytest.mark.asyncio
async def test_r4_1_a_wrapped_run_consumer_still_delivers_the_value(
    tmp_path: Path,
    config_root: Path,
    local_authorizer: str,
    stored_secret: str,
    shimmed_path: None,
) -> None:
    """The reviewer's named counterexample: `run -- timeout 5 curl …` still works."""
    marker = tmp_path / "ran"
    command = (
        f"lop secret run --secret {stored_secret}=TOK -- timeout 5 sh -c "
        f"'curl -sS -H \"Authorization: Bearer $TOK\" {local_authorizer}'; touch {marker}"
    )
    result = await builtin.execute_bash(
        "bash-r4-ok", {"command": command}, AbortSignal(), None, _context(tmp_path)
    )
    text = _result_text(result)
    assert not result.is_error, text
    assert marker.exists()
    assert "authorized" in text and "unauthorized" not in text, text
    assert _SYNTHETIC not in text


# -- round 5 ---------------------------------------------------------------
# Each row below leaked (or, for the controls, would regress) on `589318e78`.
# R5-1's class is "a non-command word read as the command": the redirect
# words, anywhere a word-walk decides which word runs.


@pytest.mark.parametrize(
    ("command", "label"),
    [
        # R5-1: a redirect on the `run` stage hid the `env` in front of it.
        (
            "lop secret run --secret NAME=TOK -- env 2>/dev/null",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- env 2>&1 | grep TOK",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- env < /dev/null | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- env >/dev/stdout | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- timeout 5 env 2>/dev/null | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- env {fd}>/dev/null | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- env <<< x | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret file NAME -- 2>/dev/null cat",
            "shell.secret-verb-emitting-consumer",
        ),
        # R5-1's class sweep: the same words in the OUTER walk.
        ("2>/dev/null lop secret get NAME | rev", "shell.pipe-of-source"),
        ("</dev/null lop secret get NAME | rev", "shell.pipe-of-source"),
        ("{fd}>/dev/null lop secret get NAME | rev", "shell.pipe-of-source"),
        (
            "export V=$(lop secret get NAME); env 2>/dev/null | rev",
            "shell.environment-dump-of-source",
        ),
        (
            "export V=$(lop secret get NAME); export 2>/dev/null",
            "shell.environment-dump-of-source",
        ),
        (
            "export V=$(lop secret get NAME); set 2>/dev/null",
            "shell.environment-dump-of-source",
        ),
        # External `time` and GNU long operands in the consumer's wrappers.
        (
            "lop secret run --secret NAME=TOK -- time printenv TOK | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- nice --adjustment 5 printenv TOK | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- stdbuf --output L printenv TOK | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- gstdbuf --output L printenv TOK | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        # R5-2: an `-i` that `env` does not own is no `env -i`.
        (
            "lop secret run --secret NAME=TOK -- stdbuf -i 0 env printenv TOK | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret run --secret NAME=TOK -- env -u -i printenv TOK | rev",
            "shell.secret-verb-emitting-consumer",
        ),
        # R5-3: node's `-p` clustered with `-e`.
        (
            "lop secret run --secret NAME=TOK -- node -pe "
            "'[...process.env.TOK].reverse().join(\"\")'",
            "shell.secret-verb-emitting-consumer",
        ),
        # R5-4: `--env-var` before the NAME.
        (
            "lop secret file --env-var KF NAME -- sh -c 'rev \"$KF\"'",
            "shell.secret-verb-emitting-consumer",
        ),
        (
            "lop secret file --env-var=KF NAME -- sh -c 'base64 < \"$KF\"'",
            "shell.secret-verb-emitting-consumer",
        ),
        # R5-5: a `run` whose consumer is the fetch.
        ("lop secret run --secret NAME=TOK -- lop secret get NAME | rev", "shell.pipe-of-source"),
        (
            "lop secret run --secret NAME=TOK -- timeout 5 lop secret get NAME | rev",
            "shell.pipe-of-source",
        ),
        ("lop secret run --secret NAME=TOK lop secret get NAME | rev", "shell.pipe-of-source"),
    ],
)
def test_r5_a_non_command_word_is_not_read_as_the_command(command: str, label: str) -> None:
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)
    assert label in result.labels, (command, result.labels)


@pytest.mark.parametrize(
    "command",
    [
        # The sanctioned consumer with the habitual stderr suffix.
        "lop secret run --secret NAME=TOK -- sh -c "
        "'curl -sS -H \"Authorization: Bearer $TOK\" http://127.0.0.1:9/' 2>/dev/null",
        "lop secret run --secret NAME=TOK -- python3 client.py 2>&1 | tail -1",
        "lop secret run --secret NAME=TOK -- echo 2 >/dev/null",
        # `env -i` that env owns, however it is spelled or wrapped.
        "lop secret run --secret NAME=TOK -- env -i printenv",
        "lop secret run --secret NAME=TOK -- timeout 5 env -iv printenv",
        "lop secret run --secret NAME=TOK -- stdbuf -i 0 env -i printenv",
        "lop secret run --secret NAME=TOK -- env -u X -i printenv",
        "lop secret run --secret NAME=TOK -- nice --adjustment 5 python3 client.py",
        "lop secret run --secret NAME=TOK -- time -p curl -sS http://127.0.0.1:9/",
        "lop secret run --secret NAME=TOK -- node -pe '1 + 1'",
        "lop secret file --env-var KF NAME -- sh -c 'wc -c < \"$KF\"'",
        'lop secret file --env-var KF NAME -- gcloud --key-file "$KF"',
        "lop secret run --secret NAME=TOK -- lop secret get NAME | wc -c",
        "lop secret run --secret NAME=TOK -- lop secret get NAME >/dev/null",
        "lop secret run --secret NAME=TOK -- lop secret list",
        "2>/dev/null lop secret get NAME | wc -c",
        "lop secret get NAME > /tmp/o 2>/dev/null",
        "export V=1; env 2>/dev/null | wc -l",
    ],
)
def test_r5_the_redirect_and_ownership_fixes_keep_the_sanctioned_forms(command: str) -> None:
    result = scan_command(command)
    assert not result.refused, (command, result)


@pytest.mark.asyncio
async def test_r5_1_a_redirected_run_dump_is_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """R5-1 through the REAL tool: on `589318e78` these returned the value raw or reversed."""
    commands = [
        f"lop secret run --secret {stored_secret}=TOK -- env 2>/dev/null",
        f"lop secret run --secret {stored_secret}=TOK -- env 2>&1 | grep TOK",
        f"lop secret run --secret {stored_secret}=TOK -- env < /dev/null | rev",
        f"lop secret run --secret {stored_secret}=TOK -- lop secret get {stored_secret} | rev",
    ]
    for command in commands:
        marker = tmp_path / "ran"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r5",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        text = _result_text(result)
        assert result.is_error, (command, text)
        assert not marker.exists(), f"the child ran anyway: {command}"
        assert _SYNTHETIC not in text
        assert _SYNTHETIC[::-1] not in text


# -- round 6 ---------------------------------------------------------------
# R6-1's class is "the stdout destination decision, and the devices that ARE a
# descriptor". Every row below leaked (or, for the controls, would regress) on
# `0e6bd482b`: `>/dev/null 1>&2` read as discarded because the redirects were
# decided with sticky flags instead of applied in order, and `/dev/stderr` read
# as an ordinary file because only `/dev/null` and the stdout devices were
# recognised. Both put the raw value in the tool result's `--- stderr ---`
# section through the real tool.


def _assert_no_value(text: str) -> None:
    """Fail for every spelling the value could come back in, not the literal.

    A refusal that lets the value out reversed, upper-cased, hexed or base64'd
    is the same leak, and a 12-character window is the smallest piece that is
    still worth calling the value.
    """
    forms = {
        "raw": _SYNTHETIC,
        "reversed": _SYNTHETIC[::-1],
        "upper": _SYNTHETIC.upper(),
        "hex": _SYNTHETIC.encode().hex(),
        "base64": base64.b64encode(_SYNTHETIC.encode()).decode(),
    }
    for name, form in forms.items():
        assert form not in text, f"the value came back {name}: {form}"
    for start in range(len(_SYNTHETIC) - 11):
        piece = _SYNTHETIC[start : start + 12]
        assert piece not in text, f"a 12-character piece of the value came back: {piece!r}"


def _result_sections(result: object) -> tuple[str, str]:
    """The tool result's `--- stdout ---` and `--- stderr ---` bodies.

    Read apart rather than joined: the R6-1 shapes put the value in the STDERR
    section while stdout was empty, so the section is part of the claim.
    """
    text = _result_text(result)
    _, _, rest = text.partition("--- stdout ---\n")
    stdout, _, stderr = rest.partition("--- stderr ---\n")
    return stdout, stderr


@pytest.mark.parametrize(
    ("command", "label"),
    [
        # R6-1: the redirects, applied in the order the shell applies them. The
        # first row is the incident: an earlier `>/dev/null` beat a later `1>&2`
        # and the value went to stderr — which IS this tool result.
        ("lop secret get NAME >/dev/null 1>&2", "shell.source-to-stderr"),
        ("lop secret get NAME > $LOG 1>&2", "shell.source-to-stderr"),
        ("lop secret get NAME 2>&1 1>/dev/stderr", "shell.bare-source-in-command-position"),
        # R6-2: a fd-2 DEVICE, not a file — however it is spelled.
        ("lop secret get NAME >/dev/stderr", "shell.source-to-stderr"),
        ("lop secret get NAME >/dev/fd/2", "shell.source-to-stderr"),
        ("lop secret get NAME 1>/dev/fd/2", "shell.source-to-stderr"),
        ("lop secret get NAME >/proc/self/fd/2", "shell.source-to-stderr"),
        ("lop secret get NAME >>/dev/stderr", "shell.source-to-stderr"),
        ("lop secret get NAME >|/dev/stderr", "shell.source-to-stderr"),
        ("lop secret get NAME 1>&2", "shell.source-to-stderr"),
        ("lop secret get NAME >&2", "shell.source-to-stderr"),
        # `>&WORD` is bash's older `&>WORD`: both streams to the word. Only two
        # words are absorbed (a fd-2 device is stderr; a plain literal file path
        # is a contained write) — every other word is refused, see R7-2 below.
        ("lop secret get NAME >& /dev/stderr", "shell.source-to-stderr"),
        ("lop secret get NAME &>/dev/stderr", "shell.source-to-stderr"),
        # A redirect BEFORE the command word, and after the consumer word.
        (">/dev/stderr lop secret get NAME", "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); echo "$v" >/dev/stderr', "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); echo "$v" >/dev/null 1>&2', "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); echo "$v" >/dev/fd/2', "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); echo "$v" 1>&2', "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); echo "$v" >&2', "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); printf "%s" "$v" > /dev/stderr', "shell.source-to-stderr"),
        # Inside a subshell and inside a brace group.
        ("(lop secret get NAME >/dev/stderr)", "shell.source-to-stderr"),
        ("{ lop secret get NAME >/dev/stderr; }", "shell.source-to-stderr"),
        # `tee` writes the value to EVERY operand, so a device operand is the
        # same leak even when the stage's own stdout is a file or /dev/null.
        ("lop secret get NAME | tee /dev/stderr >/dev/null", "shell.source-to-stderr"),
        ("lop secret get NAME | tee /dev/fd/2 >/dev/null", "shell.source-to-stderr"),
        ("lop secret get NAME | tee /tmp/r6-a /dev/stderr >/dev/null", "shell.source-to-stderr"),
        ("lop secret get NAME | tee -- /dev/stderr >/dev/null", "shell.source-to-stderr"),
        ("lop secret get NAME | tee /dev/stderr | rev", "shell.source-to-stderr"),
        # `|&` is the same pipe with stderr folded in.
        ("lop secret get NAME |& cat", "shell.pipe-of-source"),
        # A here-string hands the value to a consumer's stdin: the consumer's
        # own stdout — or its stderr — is where it goes.
        ('v=$(lop secret get NAME); cat <<< "$v"', "shell.print-of-source"),
        ('v=$(lop secret get NAME); cat <<< "$v" >/dev/stderr', "shell.source-to-stderr"),
        # A stdout DEVICE is this result, so a redirect to one prints.
        ("lop secret get NAME >/dev/fd/1", "shell.bare-source-in-command-position"),
    ],
)
def test_r6_the_stdout_destination_is_the_last_redirect_and_a_device_is_its_fd(
    command: str, label: str
) -> None:
    result = scan_command(command)
    assert result.verdict == "printing", (command, result)
    assert label in result.labels, (command, result.labels)


@pytest.mark.parametrize(
    "command",
    [
        # `>` on a GROUP: the redirect is the group's, and the source inside it
        # inherits the group's stdout, so the value goes to stderr.
        "(lop secret get NAME) >/dev/stderr",
        "{ lop secret get NAME; } >/dev/stderr",
        # `exec 1>…` rebinds the shell's stdout for every LATER stage. Redirects
        # inside a stage are not followed across `;`, so the source stage below
        # is judged as reaching this result — the safe side of a false refusal,
        # and the bound is stated rather than left to look tracked.
        ("exec 1>/dev/stderr; lop secret get NAME"),
        "exec 1>/tmp/r6-o; lop secret get NAME",
    ],
)
def test_r6_a_group_or_exec_redirect_refuses_rather_than_being_worked_out(command: str) -> None:
    """Refused, but not under a rule this round claims.

    These four are the conservative half: the test asserts the refusal, which is
    what keeps the value out of the result, and deliberately does not pin a rule
    label for a shape whose label the round did not choose.
    """
    assert scan_command(command).refused, command


@pytest.mark.parametrize(
    "command",
    [
        # The controls: a redirect that drops the value, in both orders, and a
        # `2>&1` whose value reaches no visible stream.
        "lop secret get NAME >/dev/null",
        "lop secret get NAME >/dev/null 2>&1",
        "lop secret get NAME 2>&1 >/dev/null",
        "lop secret get NAME 1>&2 >/dev/null",
        "lop secret get NAME >/dev/null 2>&1 1>&2",
        "lop secret get NAME 1>&2 2>&1 1>&2 >/dev/null",
        "lop secret get NAME >/dev/null 3>&2",
        # `/dev/stderr` is fd 2 AT THAT MOMENT, so a row that has already sent fd 2
        # to /dev/null makes this target name /dev/null: bash discards the value
        # (verified against bash itself) and so does the scan.
        "lop secret get NAME 2>/dev/null >/dev/stderr",
        # A `>`-spelled target that IS `/dev/null` after normalisation is
        # discarded, including the spellings that are not the exact string
        # (R7-3's normalisation reaches every device lookup, both ways round).
        "lop secret get NAME > /dev//null",
        "lop secret get NAME > /dev/./null",
        "lop secret get NAME >& /tmp/r6-both",
        "lop secret get NAME &> /tmp/r6-both2",
        "lop secret get NAME > /tmp/r6-c 2>&1",
        "lop secret get NAME > /dev/null 2>/dev/null",
        # A `2` that is an ARGUMENT, not a descriptor: bash reads a digit as one
        # only when it touches the operator.
        "echo 2 >/dev/null",
        "echo a 2 b",
        "lop secret run --secret NAME=TOK -- echo 2 >/dev/null",
        # Containment that is not `/dev/null`: a file, a length, or a `tee`
        # operand that is a file (`-` included — GNU tee writes a file named
        # `-`, measured, so it is a contained write and not stdout).
        "lop secret get NAME | wc -c",
        "lop secret get NAME | tee /tmp/r6-t >/dev/null",
        "lop secret get NAME | tee - >/dev/null",
        "lop secret get NAME | tee /tmp/r6-t /dev/null >/dev/null",
        ('v=$(lop secret get NAME); cat <<< "$v" >/dev/null'),
        # `/dev/tty` is NOT a captured stream and NOT a leak here: the bash tool
        # starts the child in its own session, so the device cannot be opened
        # ("Device not configured", measured on the real child) and the write
        # fails. The bound: a child WITH a controlling terminal would print, and
        # the scan does not model one.
        "lop secret get NAME >/dev/tty",
    ],
)
def test_r6_the_ordered_walk_keeps_the_sanctioned_forms(command: str) -> None:
    result = scan_command(command)
    assert not result.refused, (command, result)


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        # R7-1: a CLOSED descriptor is not a DISCARDED one. On the shell this
        # tool resolves for itself (`resolve_bash_shell(None)` → `/bin/bash`
        # 3.2.57) a BUILTIN keeps writing to the shell's stream while it reports
        # the close — `bash -c 'echo AAA >&-' | wc -c` is 4 and `printf BBB >&-`
        # is 3, while the external `/bin/echo CCC >&-` is 0 — so the five builtin
        # rows printed the raw value in `--- stdout ---` at `0e5c3c573` while the
        # scan read the fd as discarded. fd 1 goes back to base's refusal, and the
        # external rows go with it: the family gets ONE verdict rather than a
        # split that depends on which emitter the writer happens to be.
        ("lop secret get NAME >&-", "shell.bare-source-in-command-position"),
        ("lop secret get NAME >& /dev/null", "shell.bare-source-in-command-position"),
        ("lop secret get NAME >&- 2>&1", "shell.bare-source-in-command-position"),
        ('v=$(lop secret get NAME); echo "$v" >&-', "shell.print-of-source"),
        ('v=$(lop secret get NAME); printf "%s" "$v" >&-', "shell.print-of-source"),
        ('v=$(lop secret get NAME); echo "$v" 1>&- >&1', "shell.print-of-source"),
        ('v=$(lop secret get NAME); echo "$v" >& -', "shell.print-of-source"),
        ('v=$(lop secret get NAME); echo "$v" >&- 2>&2', "shell.print-of-source"),
        # The `2>&-` dups: a closed descriptor is not a discard for fd 2 either,
        # so these refuse rather than resting on the premise this round retired.
        # They cost nothing real — bash aborts them (`echo AAA 2>&- 1>&2` →
        # `bash: 2: Bad file descriptor`, rc=1, nothing on either stream).
        ("lop secret get NAME 2>&- 1>&2", "shell.source-to-stderr"),
        ("lop secret get NAME 2>&- >&2", "shell.source-to-stderr"),
        ('v=$(lop secret get NAME); echo "$v" 2>&- >&2', "shell.source-to-stderr"),
    ],
)
def test_r7_1_a_closed_descriptor_is_not_a_discarded_one(command: str, rule: str) -> None:
    """The five rows that put the raw value in stdout, and their family.

    A false refusal costs a user one workaround; a false allow publishes a
    secret into the tool result, which is the whole harm this scan exists to
    prevent — so an unrecognised spelling is REFUSED, not allowed.
    """
    result = scan_command(command)
    assert result.refused, (command, result)
    assert rule in result.labels, (command, result.labels)


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        # R7-2: a `>&WORD` / `&>WORD` target the guard cannot read — an
        # expansion, a glob, or a device word it has no rule for — is REFUSED
        # rather than resolved as an ordinary file. `>& $LOG` with
        # `LOG=/dev/stderr` was the leak: `_resolve_destination` answered
        # `("path", "$LOG")`, so an unresolved variable was registered as a
        # contained file and the raw value landed in `--- stderr ---`.
        ("lop secret get NAME >& $LOG", "shell.bare-source-in-command-position"),
        ("lop secret get NAME &> $LOG", "shell.bare-source-in-command-position"),
        ('lop secret get NAME >& "$LOG"', "shell.bare-source-in-command-position"),
        ("lop secret get NAME >& /tmp/*.f", "shell.bare-source-in-command-position"),
        # A device word earns no device resolution under the LEGACY spelling:
        # the modern `&>` keeps it (`&> /dev/null` still discards, below), but
        # this overload of the `>&` duplication operator only earned the fd-2
        # device and the plain literal file path, each with driven evidence.
        ("lop secret get NAME >& /dev/tty", "shell.bare-source-in-command-position"),
        ("lop secret get NAME >& /dev/null", "shell.bare-source-in-command-position"),
        ("lop secret get NAME &> /dev/tty", "shell.bare-source-in-command-position"),
        ("lop secret get NAME &> /dev/fd/3", "shell.bare-source-in-command-position"),
    ],
)
def test_r7_2_an_unreadable_both_streams_target_is_refused(command: str, rule: str) -> None:
    """`>&WORD` is not resolved when the word is not a word the guard can read."""
    result = scan_command(command)
    assert result.refused, (command, result)
    assert rule in result.labels, (command, result.labels)


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        # R7-3: device recognition is an exact-string lookup, so a non-canonical
        # spelling of the device was read as an ordinary contained file and the
        # raw value landed in the very section it named. `.` and empty segments
        # are collapsed before the lookup; `..` is NOT, because resolving it
        # lexically would be a guess about symlinks.
        ("lop secret get NAME > /dev//stderr", "shell.source-to-stderr"),
        ("lop secret get NAME > /dev/./stderr", "shell.source-to-stderr"),
        ("lop secret get NAME > //dev/stderr", "shell.source-to-stderr"),
        ("lop secret get NAME > /dev//fd/2", "shell.source-to-stderr"),
        ("lop secret get NAME > /dev//stdout", "shell.bare-source-in-command-position"),
        ("lop secret get NAME | tee /dev//stderr >/dev/null", "shell.source-to-stderr"),
        ("lop secret get NAME | tee /dev/./stderr >/dev/null", "shell.source-to-stderr"),
    ],
)
def test_r7_3_a_non_canonical_device_spelling_is_the_device_it_names(
    command: str, rule: str
) -> None:
    """`/dev//stderr` and `/dev/./stderr` are `/dev/stderr`, and now read so.

    Pre-existing at both heads and reachable only by writing the spelling
    deliberately — but the close is two lines and a leak's cost is not.
    """
    result = scan_command(command)
    assert result.refused, (command, result)
    assert rule in result.labels, (command, result.labels)


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        # R8-1: a LEADING redirect must not decide the verdict. The `&>`/`&>>`
        # arm left fd 1 alone when the word was unreadable, so the first
        # redirect on the line was the last word on whether the value reached
        # the result.
        ("lop secret get NAME > /tmp/pre-amp &> $LOG", "shell.bare-source-in-command-position"),
        ("lop secret get NAME > /tmp/pre-amp &>> $LOG", "shell.bare-source-in-command-position"),
        ("lop secret get NAME >/dev/null &> $LOG", "shell.bare-source-in-command-position"),
        (
            'v=$(lop secret get NAME); echo "$v" > /tmp/pre-amp &> $LOG',
            "shell.print-of-source",
        ),
    ],
)
def test_r8_1_a_leading_redirect_does_not_decide_an_unreadable_both_streams_word(
    command: str, rule: str
) -> None:
    """R8-1: `&>WORD` rebinds fd 1 instead of inheriting whatever bound it.

    `lop secret get NAME > /tmp/pre-amp &> $LOG` (`LOG=/dev/stderr`) read as a
    contained write to `/tmp/pre-amp` while the raw value went to the descriptor
    the `--- stderr ---` section IS. `_legacy_word_destination` refuses exactly
    that hazard for the `>&WORD` spelling by answering `("result", "")` rather
    than the current fd 1; both arms now answer the same way, which is REFUSED
    rather than allowed.
    """
    result = scan_command(command)
    assert result.refused, (command, result)
    assert rule in result.labels, (command, result.labels)


@pytest.mark.asyncio
async def test_r6_1_a_redirect_to_stderr_is_refused_before_the_child_runs(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """R6-1 through the REAL tool: on `0e6bd482b` these returned the value raw.

    The whole point of this round is the section the value came back in, so each
    result's stdout and stderr are scanned SEPARATELY, in every spelling, and a
    `touch` marker says whether a child ran at all.
    """
    commands = [
        f"lop secret get {stored_secret} >/dev/null 1>&2",
        f"lop secret get {stored_secret} >/dev/stderr",
        f"lop secret get {stored_secret} >/dev/fd/2",
        f"lop secret get {stored_secret} > $LOG 1>&2",
        f'v=$(lop secret get {stored_secret}); echo "$v" >/dev/stderr',
        f'v=$(lop secret get {stored_secret}); echo "$v" >/dev/null 1>&2',
        f"lop secret get {stored_secret} | tee /dev/stderr >/dev/null",
        f">/dev/stderr lop secret get {stored_secret}",
    ]
    for command in commands:
        marker = tmp_path / "ran-r6"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r6",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        stdout, stderr = _result_sections(result)
        assert result.is_error, (command, _result_text(result))
        assert not marker.exists(), f"the child ran anyway: {command}"
        _assert_no_value(stdout)
        _assert_no_value(stderr)


@pytest.mark.asyncio
async def test_r7_a_closed_fd_or_an_unreadable_both_streams_word_never_reaches_the_result(
    tmp_path: Path, config_root: Path, stored_secret: str, shimmed_path: None
) -> None:
    """R7-1, R7-2 and R7-3 through the REAL tool, with the positive half beside.

    The refusal half is the builtin `>&-` rows, the external `>&-` rows, the
    unresolved `>& $LOG` / `&> $LOG` rows and the non-canonical device
    spellings: at `0e5c3c573` these ran and returned the raw value in one of the
    two sections scanned below. The positive half is why the two absorbed shapes
    stay allowed — the literal file a both-streams word opens HOLDS the value,
    and `&> /dev/null` still discards it — so "no value in the result" is a
    measurement rather than a vacuous pass.
    """
    refused = [
        f"lop secret get {stored_secret} >&-",
        f"lop secret get {stored_secret} >& -",
        f"lop secret get {stored_secret} 1>&-",
        f"lop secret get {stored_secret} >& /dev/null",
        f"lop secret get {stored_secret} >&- 2>&1",
        f"lop secret get {stored_secret} 2>&- 1>&2",
        f"lop secret get {stored_secret} 2>&- >&2",
        f'v=$(lop secret get {stored_secret}); echo "$v" >&-',
        f'v=$(lop secret get {stored_secret}); printf "%s" "$v" >&-',
        f'v=$(lop secret get {stored_secret}); echo "$v" 1>&- >&1',
        f'v=$(lop secret get {stored_secret}); echo "$v" >&- 2>&2',
        f"lop secret get {stored_secret} >& $LOG",
        f"lop secret get {stored_secret} &> $LOG",
        f"lop secret get {stored_secret} >& /dev/tty",
        f"lop secret get {stored_secret} > /dev//stderr",
        f"lop secret get {stored_secret} > /dev/./stderr",
        f"lop secret get {stored_secret} > /dev//stdout",
        f"lop secret get {stored_secret} | tee /dev//stderr >/dev/null",
    ]
    for command in refused:
        marker = tmp_path / "ran-r7"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r7",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        stdout, stderr = _result_sections(result)
        assert result.is_error, (command, _result_text(result))
        assert not marker.exists(), f"the child ran anyway: {command}"
        _assert_no_value(stdout)
        _assert_no_value(stderr)

    positives = [
        (f"lop secret get {stored_secret} >& {tmp_path}/r7-both", tmp_path / "r7-both"),
        (f"lop secret get {stored_secret} &> {tmp_path}/r7-modern", tmp_path / "r7-modern"),
        (
            f'v=$(lop secret get {stored_secret}); echo "$v" > {tmp_path}/r7-echo',
            tmp_path / "r7-echo",
        ),
        (f"lop secret get {stored_secret} &> /dev/null", None),
    ]
    for command, written in positives:
        marker = tmp_path / "ran-r7-ok"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r7-ok",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        stdout, stderr = _result_sections(result)
        assert not result.is_error, (command, _result_text(result))
        assert marker.exists(), f"the child did not run: {command}"
        _assert_no_value(stdout)
        _assert_no_value(stderr)
        if written is not None:
            assert (
                _SYNTHETIC in written.read_text()
            ), f"the contained write is not where the verdict says it is: {command}"


@pytest.mark.asyncio
async def test_r8_1_the_leading_redirect_never_reaches_the_result(
    tmp_path: Path, config_root: Path, stored_secret: str
) -> None:
    """R8-1 through the REAL tool, with the leak it closes stated as a shape.

    Before this fix the first command below RAN, `is_error` was False and the raw
    value was sitting in the `--- stderr ---` section: the leading redirect bound
    fd 1 to a contained file, the unreadable `&> $LOG` (`LOG=/dev/stderr`) left
    that binding alone, and the value went to the very descriptor that section is.
    Both sections are scanned rather than only the one the verdict names, because
    "refused" has to mean nowhere, and the marker says whether the child ran.
    """
    refused = [
        f"LOG=/dev/stderr; lop secret get {stored_secret} > {tmp_path}/pre-amp &> $LOG",
        f"LOG=/dev/stderr; lop secret get {stored_secret} >/dev/null &> $LOG",
        f"LOG=/dev/stderr; lop secret get {stored_secret} > {tmp_path}/pre-amp &>> $LOG",
    ]
    for command in refused:
        marker = tmp_path / "ran-r8"
        if marker.exists():
            marker.unlink()
        result = await builtin.execute_bash(
            "bash-r8",
            {"command": f"touch {marker}; {command}"},
            AbortSignal(),
            None,
            _context(tmp_path),
        )
        stdout, stderr = _result_sections(result)
        assert result.is_error, (command, _result_text(result))
        assert not marker.exists(), f"the child ran anyway: {command}"
        _assert_no_value(stdout)
        _assert_no_value(stderr)
