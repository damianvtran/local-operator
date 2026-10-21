"""Code is not a credential: the SHAPE pass must never rewrite it.

The incident (session 6ccea512c372, 2026-10-21) and the three controls that stop
it: a keyword or an identifier-shaped name is released at the VALUE level, it is
never promoted to a session redaction, and tool-call ARGUMENTS take a separate
high-confidence pass because they are the program the harness executes and the
model reads back from its own replayed history.

The literals are assembled from parts rather than written out: a file that spells
a live credential-shaped assignment is itself masked by the pass it tests (the
`never_matches_its_own_source` guard in the shape suite makes the same point for
the defining module).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.loop import _scrub_history_arguments
from local_operator.harness.types import (
    LoopConfig,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)
from local_operator.redaction_shapes import REDACTION_MARKER, scrub_shapes
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.variables import VariableStore

#: The Python keyword whose registration poisoned the live session. Built from
#: parts so this file does not itself carry the word the pass once masked.
KW = "conti" + "nue"

#: The passphrase-shaped corpus positives that must KEEP masking, assembled the way
#: the corpus assembles them.
BATTERY = "corr" + "ect-horse-bat" + "tery"
BATTERY_STAPLE = BATTERY + "-staple"
SWORDFISH = "sword" + "fish"

#: The exact lines from the live incident's replayed bash call and from the code
#: the model then wrote from that corrupted history.
CODE_LINES = (
    'if s.get("type")=="kubernetes.io/service-account-token": ' + KW,
    "except Exception: " + KW,
    "if k=='_from': " + KW,
    "if o['kind'] not in ('Pod', 'Node'): " + KW,
)

SECRET_PATH = "user-dashboard" + "/secrets-qa"
K8S_SECRET_NAME = "auth-svc" + "-qa"
K8S_LONG_NAME = "auth-svc-user-directory-dashboard-secret"


def _store() -> VariableStore:
    return VariableStore(cwd=".")


def _assistant(calls: list[ToolCall]) -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=calls,
    )


def _retained(message: Message, store: VariableStore) -> Message:
    """The copy history stores and replays, through BOTH real session hooks."""

    def result_pass(text: str) -> str:
        return store.redact_with_report(text)[0]

    scrubbed = _scrub_history_arguments(message, result_pass, store.redact_arguments)
    assert isinstance(scrubbed, Message)
    return scrubbed


# --- 1. the value-level releases ------------------------------------------------


@pytest.mark.parametrize("line", CODE_LINES)
def test_a_code_keyword_is_never_masked(line: str) -> None:
    """RED before the fix: the whole line came back with the keyword masked.

    The measurable first failure was the k8s line; the other three are the same
    clause on the spellings the model writes next.
    """
    store = _store()
    out, report = store.redact_with_report(line)
    assert out == line
    assert report.labels == ()
    assert store._shape_registrations == set()


@pytest.mark.parametrize("line", CODE_LINES)
def test_a_code_keyword_is_never_registered(line: str) -> None:
    """The promotion is the amplifier: one false hit, every later line rewritten."""
    store = _store()
    store.redact_with_report(line)
    assert store._shape_registrations == set()
    follow, _ = store.redact_with_report("otherwise " + KW + " with the work")
    assert follow == "otherwise " + KW + " with the work"


def test_the_session_does_not_poison_itself_after_a_near_miss() -> None:
    """The reported chain, end to end: near miss -> registration -> later rewrite."""
    store = _store()
    store.redact_with_report(CODE_LINES[0])
    assert store._shape_registrations == set()
    assert store.redact("print(1)\n" + KW + "\nprint(2)") == "print(1)\n" + KW + "\nprint(2)"


@pytest.mark.parametrize(
    "text",
    [
        "auth-svc-user-directory-dashboard-secret: " + SECRET_PATH,
        "AUTH_SVC_BASE_URL [secret:" + K8S_SECRET_NAME + "] -> " + K8S_SECRET_NAME,
        "secret: " + K8S_LONG_NAME,
        "envFrom blue ['blue-secret']",
    ],
)
def test_identifier_shaped_names_are_not_masked_or_registered(text: str) -> None:
    """Kubernetes Secret names and SM key paths are NAMES the agent needs to read."""
    store = _store()
    out, report = store.redact_with_report(text)
    assert out == text
    assert report.labels == ()
    assert store._shape_registrations == set()


def test_the_identifier_release_stops_at_the_passphrase_boundary() -> None:
    """The measured boundary of the qualifier arm, in both directions.

    ``DB_PASSWORD=<three lowercase words>`` and ``api_key=<the same with a fourth>``
    are corpus positives and stay masked; a single lowercase word stays masked. A
    hyphen count alone cannot separate these from a Kubernetes name — the qualifier
    segment is what does, and this pins both sides of it.
    """
    store = _store()
    for text in (
        "PASSWORD=" + SWORDFISH,
        "DB_PASSWORD=" + BATTERY,
        "api_key=" + BATTERY_STAPLE,
    ):
        out, _ = store.redact_with_report(text)
        assert out != text, text
        assert REDACTION_MARKER in out


# --- 2. the argument path -------------------------------------------------------


@pytest.mark.parametrize("line", CODE_LINES)
def test_retained_bash_arguments_keep_code_bytes(line: str) -> None:
    """The retained copy is replayed and copied: it must equal what ran."""
    store = _store()
    message = _assistant([ToolCall(id="c1", name="bash", arguments={"command": line})])
    retained = _retained(message, store)
    assert retained.tool_calls[0].arguments["command"] == line


def test_retained_eval_and_raw_arguments_keep_code_bytes() -> None:
    """Both carriers: ``arguments`` (decoded) and ``raw_arguments`` (the wire)."""
    store = _store()
    code = "\n".join(CODE_LINES)
    message = _assistant(
        [
            ToolCall(id="c1", name="eval", arguments={"code": code}),
            ToolCall(
                id="c2",
                name="bash",
                arguments={"command": CODE_LINES[0]},
                raw_arguments=json.dumps({"command": CODE_LINES[0]}),
            ),
        ]
    )
    retained = _retained(message, store)
    assert retained.tool_calls[0].arguments["code"] == code
    assert retained.tool_calls[1].arguments["command"] == CODE_LINES[0]
    assert retained.tool_calls[1].raw_arguments == json.dumps({"command": CODE_LINES[0]})


def test_retained_mcp_arguments_keep_the_word() -> None:
    """Prose in an MCP argument is not a credential either."""
    store = _store()
    text = "Run the audit, then " + KW + " to the next repository."
    message = _assistant(
        [
            ToolCall(
                id="c1",
                name="slack_send_message",
                arguments={"channel": "#eng", "message": text},
            )
        ]
    )
    retained = _retained(message, store)
    assert retained.tool_calls[0].arguments["message"] == text


@pytest.mark.parametrize(
    "command",
    [
        "mysql -u root -pS3cret!xyz -h db.internal",
        "app --password=Hunter2Secret99",
        "psql postgres://svc:Hunter2Secret99@db.invalid/app",
        'curl -H "Authorization: Bearer '
        + ("eyJ" + "hbGciOiJIUzI1NiJ9")
        + ".e30.sig"
        + " https://h/v1",
        "export GITHUB_TOKEN=ghp_" + "A" * 36,
    ],
)
def test_retained_arguments_still_mask_a_credential_typed_into_the_command(
    command: str,
) -> None:
    """The other direction: a real credential in CODE is still contained."""
    store = _store()
    message = _assistant([ToolCall(id="c1", name="bash", arguments={"command": command})])
    retained = _retained(message, store)
    scrubbed = retained.tool_calls[0].arguments["command"]
    assert scrubbed != command
    assert REDACTION_MARKER in scrubbed
    # The executor's own copy is untouched: only the retained copy changes.
    assert message.tool_calls[0].arguments["command"] == command


def test_a_known_secret_value_is_still_masked_in_arguments() -> None:
    """The exact-value pass runs first in the argument hook, so a known secret goes."""
    store = _store()
    secret = "s3cr3t-" + "value-42"
    store.store_credential("SYNTHETIC_ARGUMENT_SECRET", secret)
    message = _assistant([ToolCall(id="c1", name="bash", arguments={"command": "echo " + secret})])
    scrubbed = _retained(message, store).tool_calls[0].arguments["command"]
    assert secret not in scrubbed
    assert REDACTION_MARKER in scrubbed


def test_the_argument_hook_never_registers() -> None:
    store = _store()
    message = _assistant(
        [ToolCall(id="c1", name="bash", arguments={"command": "mysql -pS3cret!xyz"})]
    )
    _retained(message, store)
    assert store._shape_registrations == set()


def test_without_the_argument_hook_the_result_pass_is_the_fallback() -> None:
    """A caller that supplies only ``redact_tool_result`` keeps its old behaviour."""
    store = _store()

    def result_pass(text: str) -> str:
        return store.redact_with_report(text)[0]

    message = _assistant(
        [ToolCall(id="c1", name="bash", arguments={"command": "PASSWORD=" + SWORDFISH})]
    )
    fallback = _scrub_history_arguments(message, result_pass)
    assert isinstance(fallback, Message)
    assert REDACTION_MARKER in fallback.tool_calls[0].arguments["command"]
    # ...while the argument pass, which has the confidence floor, releases it. A
    # FRESH store: the result pass above registered what it masked, as it must, and
    # the exact-value pass in the argument hook would then mask it for the right
    # reason — the floor is what this half measures.
    preferred = _scrub_history_arguments(message, result_pass, _store().redact_arguments)
    assert preferred.tool_calls[0].arguments["command"] == "PASSWORD=" + SWORDFISH


def test_loop_config_defaults_to_no_argument_hook() -> None:
    field = LoopConfig.model_fields["redact_tool_arguments"]
    assert field.default is None


# --- 2b. the name-driven amplifier, closed over the whole DNS grammar -----------

#: Real-shaped Kubernetes Secret names whose segments all miss the qualifier
#: vocabulary. Review round 1 measured ten of nineteen realistic names masked in
#: place AND registered at b55c948f9a: the in-place mask is the accepted residual
#: (one line, the next read shows it once), the registration is the amplifier.
K8S_NAMES_OUTSIDE_THE_VOCABULARY = (
    "payments-api-db",
    "monitoring-grafana",
    "backstage-db-credentials",
    "ops-pagerduty-key",
    "redis-cache",
    "postgres-primary",
    "platform-db",
    "grafana-admin",
    "orders-db-credentials",
    "team-alpha-api",
)


@pytest.mark.parametrize("name", K8S_NAMES_OUTSIDE_THE_VOCABULARY)
def test_a_secret_name_outside_the_vocabulary_is_never_registered(name: str) -> None:
    """R1-1: the in-place release is a VOCABULARY, the containment is the GRAMMAR.

    ``secret: <name>`` still masks in place — the stated residual, and recoverable.
    The promotion is not: a registered name becomes the session's word for it, and
    every later result, replayed request and newly written cell loses it.
    """
    store = _store()
    line = "secret: " + name
    out, _ = store.redact_with_report(line)
    # The residual, in the direction that heals on the next read...
    assert out != line
    assert REDACTION_MARKER in out
    # ...and the containment that matters: never promoted, so readable again.
    assert store._shape_registrations == set()
    assert store.redact("the manifest uses " + name) == "the manifest uses " + name


@pytest.mark.parametrize("name", K8S_NAMES_OUTSIDE_THE_VOCABULARY)
def test_a_secret_name_outside_the_vocabulary_is_never_rewritten_in_arguments(
    name: str,
) -> None:
    """R1-1's argument half: a hyphen in a NAME is structure, not credential alphabet.

    Measured at b55c948f9a: ``kubectl get secret payments-api-db -n prod`` came back
    with the name replaced, because ``_carries_credential_evidence`` counted the
    hyphen as a credential symbol and the vocabulary release did not cover it.
    """
    store = _store()
    command = "kubectl get secret " + name + " -n prod -o yaml"
    assert store.redact_arguments(command) == command
    assert store._shape_registrations == set()


def test_an_identifier_shaped_dsn_password_is_registered() -> None:
    """R1-3: the identifier refusal is NAME-driven, not value-only.

    A DSN password spelled like a path (``prod/secret``) is masked by the DSN rule,
    whose spelling is its own evidence, and then refused registration at head: masked
    in place once and free again in the next result. The label decides which gate
    applies, so this registers while ``secret: payments-api-db`` above does not.
    """
    store = _store()
    password = "prod" + "/" + "secret"
    dsn = "MONGO_DSN=mongodb://user:" + password + "@host/db"
    out, _ = store.redact_with_report(dsn)
    assert out != dsn
    assert REDACTION_MARKER in out
    assert password in store._shape_registrations
    later = "the password is " + password + " exactly"
    assert store.redact(later) != later


#: 60 characters, thirty hyphens, ending out of grammar: the ``fullmatch`` FAILS,
#: which is where a separator that is also a legal label character backtracks. A
#: matching string took the first split every time, which is why the hazard hid.
ADVERSARIAL_HYPHENATED = "a-" * 29 + "aZ"

#: A bound, not a benchmark: the linear form measures 0.0002s here, the old one did
#: not return in 5s. Generous enough that a loaded machine cannot make it flaky.
TIME_BOUND_SECONDS = 0.5


def test_the_hyphenated_name_check_is_linear() -> None:
    """R1-2: ``-`` is both the separator and an interior label character.

    The old form (``_DNS_LABEL(?:-_DNS_LABEL)+``) left every hyphen of a failing
    ``fullmatch`` an ambiguous split point: measured 0.30s at 37 characters, 3.1s at
    45, no return at this 60-character payload — on a pass that scans arbitrary tool
    output, so a fetched page or a read file could wedge the agent loop with no
    timeout to rescue it. Both entry points that reach it are measured.
    """
    text = "secret: " + ADVERSARIAL_HYPHENATED
    store = _store()
    # Warm once: the first call pays for the one-time compile of the rule table,
    # which is not what this test measures.
    scrub_shapes(text)
    store.redact_with_report(text)

    start = time.perf_counter()
    store.redact_with_report(text)
    assert time.perf_counter() - start < TIME_BOUND_SECONDS

    start = time.perf_counter()
    scrub_shapes(text)
    assert time.perf_counter() - start < TIME_BOUND_SECONDS


# --- 3. the legitimate masking the corpus already pins --------------------------


@pytest.mark.parametrize(
    "text",
    [
        "MONGO_DSN=mongodb+srv://user:pw@host/db",
        '{"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl"}',
        "token ghp_" + "A" * 36,
        "AWS_SECRET_ACCESS_KEY=" + "dGhpcy1pcy1hLXNlY3JldA==",
        "PASSWORD=" + SWORDFISH,
        "DB_PASSWORD=" + BATTERY,
        '{"has_token": "' + "ghp_" + "B" * 36 + '"}',
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz\n"
        "-----END RSA PRIVATE KEY-----",
    ],
)
def test_legitimate_credentials_are_still_masked(text: str) -> None:
    out, _ = _store().redact_with_report(text)
    assert out != text
    assert REDACTION_MARKER in out


# --- 4. the wiring the loop and the session actually use ------------------------


async def _never_streams(_request: Any, _signal: Any = None) -> AsyncIterator[Any]:
    """A provider stream these tests never reach: they call the hooks directly."""
    if False:  # pragma: no cover - makes this an async generator
        yield None


def test_a_real_session_wires_both_hooks_and_they_disagree_on_code(tmp_path: Path) -> None:
    """The session's argument hook keeps code while its result hook would mask it.

    This is the wiring the incident was missing, measured on the REAL hooks the
    session hands to ``LoopConfig`` rather than on the module function alone.
    """
    session = Session(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        stream_fn=_never_streams,
        tools=[],
        transcript=Transcript(tmp_path / "session"),
        system_blocks_provider=lambda *_a: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
    )
    line = CODE_LINES[0]
    store = session._variables
    assert store is not None

    # The argument hook leaves the line alone and registers nothing...
    assert session._redact_tool_arguments_text(line) == line
    assert store._shape_registrations == set()

    # ...while the argument hook still masks a credential typed into a command.
    dsn = "psql postgres://svc:" + ("S3cret" + "Pass1") + "@db.invalid/app"
    masked = session._redact_tool_arguments_text(dsn)
    assert masked != dsn and REDACTION_MARKER in masked

    # And the retained copy of an assistant turn goes through the argument hook.
    message = _assistant(
        [
            ToolCall(id="c1", name="bash", arguments={"command": line}),
            ToolCall(id="c2", name="write", arguments={"path": "x.py", "content": line}),
        ]
    )
    retained = _scrub_history_arguments(
        message, session._redact_tool_result_text, session._redact_tool_arguments_text
    )
    assert retained.tool_calls[0].arguments["command"] == line
    assert retained.tool_calls[1].arguments["content"] == line
