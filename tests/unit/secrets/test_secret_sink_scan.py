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
    RULES,
    RULE_LABELS,
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
