"""Value-agnostic credential-echo scrubbing.

The leak this defends against is a property of a STRING, not of a cluster: a
parser rejected a credential-typed option and echoed it with its value. So the
bulk of the coverage here is pure-function and needs no BusyBox — a real
``busybox`` binary, when one is installed, only guards the fixture against
drifting from reality.

The synthetic secret is ``"f" * 64``: same length and character class as the
real leaked value, so length- and shape-dependent behaviour is preserved. This
repository is public; the real value must never appear in it, which would turn
an internal leak into a published one.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.redaction import (
    CREDENTIAL_ECHO_NOTICE,
    CredentialFlagFinding,
    format_credential_flag_warning,
    lint_credential_flags,
    redact_tool_output,
    scrub_credential_echo,
    scrub_details,
)

_FAKE = "f" * 64  # same shape as the real value; never the real value

_FIXTURE = Path(__file__).parent / "fixtures" / "busybox_wget_reject.txt"

_MARKER = "«REDACTED:flag=password»"


# --------------------------------------------------------------------------
# The three real leak shapes, sanitized.
# --------------------------------------------------------------------------


def test_the_incident_line_is_scrubbed_and_the_flag_is_reported() -> None:
    """The exact BusyBox output that leaked, with a synthetic value."""
    text, flags = scrub_credential_echo(f"wget: unrecognized option: password={_FAKE}")

    assert _FAKE not in text
    assert text == f"wget: unrecognized option: password={_MARKER}"
    assert flags == ["password"]


def test_the_quoted_busybox_form_keeps_its_closing_quote() -> None:
    """The opening quote is consumed by the lead, not by the value, and the
    value's character class excludes quotes — so the surrounding syntax
    survives the substitution and the line stays readable."""
    scrubbed, flags = scrub_credential_echo(f"wget: unrecognized option '--password={_FAKE}'")

    assert scrubbed == f"wget: unrecognized option '--password={_MARKER}'"
    assert flags == ["password"]


def test_a_quoted_placeholder_is_still_suppressed() -> None:
    """Widening the lead by a quote must not weaken the suppressor."""
    line = "wget: unrecognized option '--password=$OPENSEARCH_PASSWORD'"

    assert scrub_credential_echo(line) == (line, [])


def test_the_full_busybox_stdout_loses_only_the_value() -> None:
    """The usage block below the leak line must survive byte-for-byte.

    A scrubber that mangles the rest of the output trades one problem for
    another: the agent still needs to read why the command failed.
    """
    raw = _FIXTURE.read_text(encoding="utf-8")
    assert _FAKE in raw, "fixture must carry the synthetic value"

    scrubbed, flags = scrub_credential_echo(raw)

    assert _FAKE not in scrubbed
    assert flags == ["password"]
    assert scrubbed.splitlines()[1:] == raw.splitlines()[1:]
    assert scrubbed.splitlines()[0] == f"wget: unrecognized option: password={_MARKER}"


@pytest.mark.parametrize(
    "line",
    [
        # The three leaked sessions all carried this BusyBox line verbatim.
        f"wget: unrecognized option: password={_FAKE}",
        # Dashes retained, as a parser that echoes them would print it.
        f"wget: unrecognized option: --password={_FAKE}",
        # A second credential flag under the same rejection sentence.
        f"wget: unrecognized option: token={_FAKE}",
        # BusyBox's OTHER rejection path, quoted. A stock busybox:1.37.0
        # container prints this form; the pod in the incident printed the bare
        # one above. A pattern built from the transcript alone misses it.
        f"wget: unrecognized option '--password={_FAKE}'",
    ],
)
def test_each_observed_leak_shape_is_scrubbed(line: str) -> None:
    """The bare form matters most: BusyBox drops the leading dashes when it
    echoes, so a matcher requiring ``--password=`` misses the real incident.

    Note what is NOT here: GNU's ``invalid option -- 'x'`` separated form. It
    puts a space between the dashes and the name and never carries a value, so
    it is out of scope by construction rather than by omission.
    """
    scrubbed, flags = scrub_credential_echo(line)

    assert _FAKE not in scrubbed
    assert len(flags) == 1


@pytest.mark.parametrize(
    "verb",
    ["unrecognized", "unrecognised", "invalid", "illegal", "unknown", "bad"],
)
def test_every_rejection_verb_anchors_a_match(verb: str) -> None:
    scrubbed, flags = scrub_credential_echo(f"tool: {verb} option: password={_FAKE}")

    assert _FAKE not in scrubbed
    assert flags == ["password"]


@pytest.mark.parametrize(
    "flag",
    [
        "password",
        "passwd",
        "pass",
        "passphrase",
        "token",
        "secret",
        "credential",
        "credentials",
        "api-key",
        "api_key",
        "apikey",
        "access-key",
        "secret-key",
        "auth-token",
        "client-secret",
        "bearer",
    ],
)
def test_every_credential_flag_is_covered(flag: str) -> None:
    scrubbed, flags = scrub_credential_echo(f"tool: unrecognized option: {flag}={_FAKE}")

    assert _FAKE not in scrubbed
    assert flags == [flag]


def test_a_non_credential_flag_is_left_alone() -> None:
    """Without the credential-typed narrowing the rule would mangle ordinary
    output like ``unrecognized option: color=auto``."""
    line = "ls: unrecognized option: color=always"

    assert scrub_credential_echo(line) == (line, [])


# --------------------------------------------------------------------------
# The conjuncts that buy the zero false-positive rate.
# --------------------------------------------------------------------------


def test_space_separated_does_not_match() -> None:
    """Measured: every space-separated match in the corpus was English prose
    ("invalid al...", "unknown argument for ..."); every real leak used ``=``.
    Accepting a space here reintroduces the failure mode the ``=`` rules out.
    """
    line = f"wget: unrecognized option password {_FAKE}"

    assert scrub_credential_echo(line) == (line, [])


@pytest.mark.parametrize(
    "prose",
    [
        "The OpenSearch admin password was rotated last week.",
        "This was an invalid argument for the admin password policy.",
        "unknown argument for password rotation",
        "Set the admin password before running the invalid option check.",
    ],
)
def test_prose_mentioning_credentials_is_untouched(prose: str) -> None:
    """The "admin"-inside-"admin password" class.

    This design cannot produce that error by construction: it never matches on
    a credential WORD, only on flag + ``=`` + value inside a rejection
    sentence. No substring search for secret-ish words exists anywhere here.
    """
    assert scrub_credential_echo(prose) == (prose, [])


def test_a_bare_credential_flag_outside_a_rejection_sentence_is_untouched() -> None:
    """The rejection lead is load-bearing. Without it, 40% of corpus matches
    were user- and assistant-authored prose, i.e. the operator's own messages.
    """
    line = f"Running: wget --password={_FAKE} https://example.invalid"

    assert scrub_credential_echo(line) == (line, [])


@pytest.mark.parametrize(
    "placeholder",
    [
        "«REDACTED»",
        "«REDACTED:flag=password»",
        "$OPENSEARCH_PASSWORD",
        "${OPENSEARCH_PASSWORD}",
        "[redacted]",
        "<value>",
        "<password>",
        "{password}",
        "***",
        "xxxxxx",
        "PASSWORD",
        "SECRET",
        "API_KEY",
    ],
)
def test_placeholders_are_suppressed(placeholder: str) -> None:
    """Three sessions in the corpus discuss this very incident and quote
    ``password=«REDACTED»``. A detector without this suppressor "finds" and
    mangles them.
    """
    line = f"wget: unrecognized option: password={placeholder}"

    assert scrub_credential_echo(line) == (line, [])


@pytest.mark.parametrize("value", ["a", "ab", "abc"])
def test_values_below_the_length_floor_are_left_alone(value: str) -> None:
    """Every sub-4-char capture in the whole corpus was noise (punctuation, a
    quote fragment, an English word), and mangling those costs more than it
    saves."""
    line = f"wget: unrecognized option: password={value}"

    assert scrub_credential_echo(line) == (line, [])


def test_the_length_floor_admits_the_first_real_length() -> None:
    """The floor is a boundary, so pin both sides of it."""
    scrubbed, flags = scrub_credential_echo("wget: unrecognized option: password=abcd")

    assert flags == ["password"]
    assert scrubbed == f"wget: unrecognized option: password={_MARKER}"


def test_scrubbing_is_idempotent() -> None:
    """The same text passes both choke points; a second pass must be a no-op.

    Non-idempotence would also corrupt prose written about a leak, which is
    the failure the placeholder suppressor exists for.
    """
    once, first_flags = scrub_credential_echo(f"wget: unrecognized option: password={_FAKE}")
    twice, second_flags = scrub_credential_echo(once)

    assert twice == once
    assert first_flags == ["password"]
    assert second_flags == []


def test_multiple_flags_on_one_line_are_all_reported() -> None:
    scrubbed, flags = scrub_credential_echo(
        f"tool: unrecognized option: password={_FAKE}\ntool: unrecognized option: token={_FAKE}"
    )

    assert _FAKE not in scrubbed
    assert flags == ["password", "token"]


# --------------------------------------------------------------------------
# Composition with the value-keyed pass.
# --------------------------------------------------------------------------


def test_value_keyed_runs_first_and_wins() -> None:
    """The value-keyed marker is more precise, so it must not be displaced.

    Reversed, the shape pass would consume the value and the value-keyed pass
    would find nothing left to match.
    """
    from local_operator.variables import redact_secret_values

    line = f"wget: unrecognized option: password={_FAKE}"

    out = redact_tool_output(line, lambda text: redact_secret_values(text, {"OSP": _FAKE}))

    assert out == "wget: unrecognized option: password=[redacted]"
    assert _MARKER not in out


def test_the_shape_pass_runs_with_no_credential_store_at_all() -> None:
    """The central regression: the harness never held this value, so a session
    with no stored credentials is exactly the case that leaked."""
    out = redact_tool_output(f"wget: unrecognized option: password={_FAKE}", None)

    assert out == f"wget: unrecognized option: password={_MARKER}"


def test_a_non_string_from_a_hosts_redactor_is_ignored() -> None:
    """Hosts inject the hook; a misbehaving one must not corrupt the text."""
    bad_hook: Any = lambda _: None  # noqa: E731 - a deliberately misbehaving hook
    out = redact_tool_output(f"wget: unrecognized option: password={_FAKE}", bad_hook)

    assert out == f"wget: unrecognized option: password={_MARKER}"


def test_the_notice_is_appended_only_when_a_redaction_fired() -> None:
    """The agent's actionable signal: the command FAILED, which the pipe's
    ``exit code: 0`` had hidden."""
    fired = redact_tool_output(f"wget: unrecognized option: password={_FAKE}", None, notice=True)
    quiet = redact_tool_output("all fine", None, notice=True)

    assert CREDENTIAL_ECHO_NOTICE in fired
    assert CREDENTIAL_ECHO_NOTICE not in quiet


def test_the_notice_is_not_stacked_on_a_second_pass() -> None:
    once = redact_tool_output(f"wget: unrecognized option: password={_FAKE}", None, notice=True)
    twice = redact_tool_output(once, None, notice=True)

    assert twice == once
    assert twice.count(CREDENTIAL_ECHO_NOTICE) == 1


# --------------------------------------------------------------------------
# details (Hole A): never seen by a model, always written to disk.
# --------------------------------------------------------------------------


def _scrub(text: str) -> str:
    return redact_tool_output(text, None)


def test_details_string_leaves_are_scrubbed() -> None:
    out = scrub_details({"stdout": f"wget: unrecognized option: password={_FAKE}"}, _scrub)

    assert _FAKE not in out["stdout"]


def test_details_scrubs_a_nested_mcp_server_result() -> None:
    """``tool_bridge`` puts the ENTIRE raw server result here, and strips its
    text blocks only when a spill occurred. Under the spill threshold it
    reaches ``transcript.jsonl`` verbatim."""
    details = {
        "server_result": {
            "content": [
                {"type": "text", "text": f"wget: unrecognized option: password={_FAKE}"},
                {"type": "image", "mimeType": "image/png"},
            ],
            "isError": False,
        }
    }

    out = scrub_details(details, _scrub)

    assert _FAKE not in repr(out)
    assert _MARKER in out["server_result"]["content"][0]["text"]
    # Structure preserved: renderers and compaction index on these keys.
    assert out["server_result"]["content"][1] == {"type": "image", "mimeType": "image/png"}
    assert out["server_result"]["isError"] is False


def test_details_preserves_non_string_leaves_and_container_types() -> None:
    details = {"lines": 12, "complete": True, "span": (1, 2), "handle": None}

    assert scrub_details(details, _scrub) == details


def test_details_keys_are_not_rewritten() -> None:
    """Keys are structural. A credential echo is a value-shaped event."""
    out = scrub_details({"unrecognized option: password=abcd": "x"}, _scrub)

    assert list(out) == ["unrecognized option: password=abcd"]


def test_details_walk_is_depth_bounded() -> None:
    """A third-party MCP payload must not turn redaction into an unbounded
    walk on the event loop. Over budget the subtree is returned UNMODIFIED
    rather than dropped: ``provider_payload`` is load-bearing for compaction,
    and silently mutating a structure we declined to walk would trade a narrow
    leak for broad corruption.
    """
    leaf = f"wget: unrecognized option: password={_FAKE}"
    deep: object = leaf
    for _ in range(40):
        deep = {"next": deep}

    out = scrub_details({"root": deep}, _scrub)

    assert out["root"] is not None  # returned, not dropped
    node = out["root"]
    depth = 0
    while isinstance(node, dict) and "next" in node:
        node = node["next"]
        depth += 1
    assert depth == 40
    assert node == leaf  # beyond the bound, untouched


def test_details_walk_is_node_bounded() -> None:
    """A wide payload is bounded too, and still returns every element."""
    wide = {"items": [{"text": f"row {index}"} for index in range(4000)]}

    out = scrub_details(wide, _scrub)

    assert len(out["items"]) == 4000


def test_details_scrubbing_is_idempotent() -> None:
    details = {"stdout": f"wget: unrecognized option: password={_FAKE}"}

    once = scrub_details(details, _scrub)
    twice = scrub_details(once, _scrub)

    assert twice == once


# --------------------------------------------------------------------------
# The pre-flight lint: warn only, narrow by measurement.
# --------------------------------------------------------------------------


def test_the_lint_fires_on_the_incidents_real_command_shape() -> None:
    command = (
        'kubectl --context eks-prod-2 -n backend-services exec "$POD" -- sh -c '
        "'wget -q -O - --no-check-certificate "
        '--user="$OPENSEARCH_USERNAME" --password="$OPENSEARCH_PASSWORD" '
        '"$OPENSEARCH_URL/_cat/thread_pool/write?v"\' 2>&1 | head -15'
    )

    findings = lint_credential_flags(command)

    assert [finding.flag for finding in findings] == ["--password"]
    assert findings[0].fetcher == "wget"


@pytest.mark.parametrize(
    "command",
    [
        "export OPENSEARCH_PASSWORD=hunter2",
        "set -a; API_TOKEN=abcd; set +a",
        "export FOO_PASSWORD=x && echo done",
        # No fetcher: a credential flag to some other tool is not this risk.
        "psql --password=abcd -h localhost",
        # Fetcher present but no credential flag.
        "wget -q -O - https://example.invalid",
        # Fetcher AFTER the flag: an earlier assignment does not borrow risk.
        "PASSWORD=abcd; echo hi",
    ],
)
def test_the_lint_does_not_fire_on_benign_commands(command: str) -> None:
    """Measured: the unscoped rule is ~80% ``export``/``set`` assignments.
    Requiring leading dashes AND a fetcher is what excludes them without
    parsing shell syntax."""
    assert lint_credential_flags(command) == []


def test_the_lint_reports_each_flag_once() -> None:
    findings = lint_credential_flags("curl --token=a --token=b --api-key=c https://x.invalid")

    assert [finding.flag for finding in findings] == ["--token", "--api-key"]


def test_the_lint_never_rewrites_or_blocks() -> None:
    """The operator's hard constraint. The lint is a pure function returning
    findings: it has no way to alter or refuse the command, and the warning is
    delivered on the RESULT after the command has already run.
    """
    command = 'wget --password="$P" https://example.invalid'

    findings = lint_credential_flags(command)

    assert findings  # it fired
    assert isinstance(findings, list)
    warning = format_credential_flag_warning(findings)
    assert command not in warning  # the command is not echoed back
    assert "--password" in warning and "wget" in warning


def test_the_warning_names_the_flag_and_the_fetcher() -> None:
    warning = format_credential_flag_warning(
        [CredentialFlagFinding(flag="--token", fetcher="curl")]
    )

    assert "--token" in warning
    assert "curl" in warning
    assert "BusyBox curl" in warning


# --------------------------------------------------------------------------
# Layer 2: the real binary, when it happens to be installed.
# --------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("busybox") is None, reason="busybox is not installed")
def test_real_busybox_output_is_scrubbed() -> None:
    """Guards the fixtures against drifting from the real parser's output.

    Never the primary evidence — the detector is a pure function over a string
    — and deliberately not worth a Docker dependency in CI. A local BusyBox
    (or the container run recorded in the PR) is what caught that BusyBox has
    TWO rejection shapes: the bare ``password=VALUE`` the incident recorded and
    the quoted ``'--password=VALUE'`` a stock image prints.
    """
    proc = subprocess.run(
        ["busybox", "wget", f"--password={_FAKE}", "http://127.0.0.1:1/"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = proc.stdout + proc.stderr
    assert _FAKE in combined, f"busybox did not echo the value: {combined[:200]!r}"

    scrubbed, flags = scrub_credential_echo(combined)

    assert _FAKE not in scrubbed
    assert flags == ["password"]
