"""Credential exposure through the shared error-body helpers, and its regression guard.

WHY THIS FILE EXISTS. ``response_body`` was fixed to report a 4xx/5xx body it used
to call absent, and because that helper is shared by every client in this package
the fix reached five call sites that redacted nothing: a body an upstream composes
by reflecting the request it received -- the ``Authorization`` header included --
went straight into a ``RuntimeError`` the desktop app renders and the log keeps.
Review round 1 reproduced it end-to-end through two of those clients.

The fix is central (``_http.scrub_secrets`` on the body's way out of the module),
so the tests here are the two halves of that property:

* ``test_every_consumer_*`` drives the REAL method of every client that surfaces a
  body, over two transports (a ``requests`` exception carrying a 401, and a 401
  returned to code that inspects the status itself) and three spellings of the
  same credential, and asserts the canary is absent from what the caller sees.
* ``test_no_client_module_interpolates_a_response_body`` is the guard: it parses
  the package and fails when a module reads ``response.content``/``response.text``
  into a message without going through the shared helpers, so the NEXT call site
  is caught by CI rather than by a reviewer.
"""

import ast
import json
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients import _http, fal, ollama, openrouter, serpapi, tavily
from local_operator.clients._http import (
    NO_RESPONSE_BODY,
    REDACTION_MARKER,
    response_body,
    scrub_secrets,
    scrubbed_response_body,
)

CLIENTS_PACKAGE = Path(_http.__file__).parent

#: The helpers through which a body may be read. A read that happens INSIDE one of
#: these calls is the sanctioned shape; anything else that puts a body into a
#: message is what the guard below reports.
SANCTIONED_HELPERS = frozenset(
    {"response_body", "scrubbed_response_body", "scrub_secrets", "redact_secrets"}
)

#: The constructors a surfaced message is built for -- the "message" half of the
#: guard, so an exception argument is scanned as well as an f-string.
MESSAGE_BUILDERS = frozenset(
    {"RuntimeError", "APIError", "GoogleAPIError", "ValueError", "Exception"}
)


def genuine_response(status: int, body: str) -> requests.Response:
    """A real ``requests.Response``, because ``__bool__`` is what the bug turned on.

    A ``MagicMock`` would answer ``True`` to ``__bool__`` and quietly test the other
    branch, and the whole defect was that a real 4xx/5xx is falsy.
    """
    response = requests.Response()
    response.status_code = status
    response._content = body.encode("utf-8")
    response.url = "https://upstream.invalid/v1/endpoint"
    return response


def raising_transport(status: int, body: str) -> Callable[..., Any]:
    """A transport that RAISES, carrying the response -- the shape most clients handle."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        raise requests.exceptions.HTTPError(
            f"{status} Client Error: upstream refused", response=genuine_response(status, body)
        )

    return _call


def returning_transport(status: int, body: str) -> Callable[..., Any]:
    """A transport that RETURNS the 4xx, for the clients that inspect the status."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        return genuine_response(status, body)

    return _call


# --- the credential shapes a real upstream reflects ---------------------------------


def reflected_bodies(canary: str) -> List[Tuple[str, str]]:
    """Three spellings of the same credential, as an upstream might echo it.

    The bare-token case is the one a caller cannot recognise from context: no
    header, no field name, just the value the vendor issued.
    """
    return [
        ("bearer-header", json.dumps({"error": f"invalid key: Authorization: Bearer {canary}"})),
        ("named-field", json.dumps({"error": "request rejected", "api_key": canary})),
        ("bare-token", json.dumps({"error": f"unknown token {canary}"})),
    ]


# --- the clients that surface a body ------------------------------------------------


def _openrouter(canary: str) -> Callable[[], Any]:
    client = openrouter.OpenRouterClient(api_key=SecretStr(canary))
    return client.list_models


def _tavily(canary: str) -> Callable[[], Any]:
    client = tavily.TavilyClient(api_key=SecretStr(canary))
    return lambda: client.search("query")


def _serpapi(canary: str) -> Callable[[], Any]:
    client = serpapi.SerpApiClient(api_key=SecretStr(canary))
    return lambda: client.search("query")


def _fal(canary: str) -> Callable[[], Any]:
    client = fal.FalClient(api_key=SecretStr(canary))
    return lambda: client._get_request_status("req-1")


def _ollama(canary: str) -> Callable[[], Any]:
    client = ollama.OllamaClient()
    return client.list_models


def _ollama_transport(transport: Callable[..., Any]) -> Callable[..., Any]:
    """Ollama gates ``list_models`` on a health probe, so let only that one answer.

    Without this the client returns before it ever builds the message under test,
    and the test passes for the wrong reason.
    """
    healthy = genuine_response(200, "Ollama is running")
    root = ollama.OllamaClient().base_url.rstrip("/")

    def _call(url: str, *args: Any, **kwargs: Any) -> Any:
        if url.rstrip("/") == root:
            return healthy
        return transport(url, *args, **kwargs)

    return _call


#: ``(id, module, http method the client calls, canary, call builder)``. The canary
#: carries the vendor's own prefix, because that is how a real key is spelled and
#: the bare-token case above is only recognised by it.
SUBJECTS: Tuple[Tuple[str, Any, str, str, Callable[[str], Callable[[], Any]]], ...] = (
    ("openrouter", openrouter, "get", "sk-or-v1-LEAKME-000000000000000000", _openrouter),
    ("tavily", tavily, "post", "tvly-LEAKME-00000000000000000000", _tavily),
    ("serpapi", serpapi, "get", "serp-LEAKME-00000000000000000000", _serpapi),
    ("fal", fal, "get", "fal-LEAKME-00000000000000000000", _fal),
    ("ollama", ollama, "get", "xai-LEAKME-00000000000000000000", _ollama),
)


@pytest.mark.parametrize("transport_name", ["raising", "returning"])
@pytest.mark.parametrize(
    "client_name,module,http_method,canary,build",
    SUBJECTS,
    ids=[subject[0] for subject in SUBJECTS],
)
def test_every_consumer_scrubs_a_credential_out_of_its_error_message(
    client_name: str,
    module: Any,
    http_method: str,
    canary: str,
    build: Callable[[str], Callable[[], Any]],
    transport_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No client that surfaces an upstream body may pass a credential through it.

    Both request shapes, because they are different code paths in the same client:
    a ``requests`` exception carries one and a client that checks the status itself
    reads the other, and only the first of those used to be redacted anywhere.
    """
    for label, body in reflected_bodies(canary):
        make_transport = raising_transport if transport_name == "raising" else returning_transport
        transport = make_transport(401, body)
        if client_name == "ollama":
            transport = _ollama_transport(transport)
        monkeypatch.setattr(module.requests, http_method, transport)

        with pytest.raises(Exception) as exc_info:
            build(canary)()

        message = str(exc_info.value)
        assert canary not in message, f"{client_name}/{label}/{transport_name} leaked the key"
        assert (
            REDACTION_MARKER in message
        ), f"{client_name}/{label}/{transport_name} dropped the body instead of redacting it"


def test_an_objection_that_is_not_a_credential_survives_the_scrubber() -> None:
    """The scrubber must not make an upstream's refusal unreadable.

    A body is often the only account of what happened, so a rule that masks an
    ordinary sentence, a model parameter or a field name would trade one defect for
    another. ``max_tokens`` is the specific near-miss the name rule must not take.
    """
    body = '{"error": "unknown model: max_tokens is 4096", "code": "invalid_request"}'

    assert scrub_secrets(body) == body


@pytest.mark.parametrize(
    "body,canary",
    [
        (
            '{"error": "invalid key: Authorization: Bearer sk-or-v1-ABC123XYZ"}',
            "sk-or-v1-ABC123XYZ",
        ),
        ('{"error": "invalid key: authorization=bearer tvly-ABC123XYZ"}', "tvly-ABC123XYZ"),
        ('{"detail": {"api_key": "abcdefgh12345678"}}', "abcdefgh12345678"),
        (
            '{"error": "https://upstream.invalid/v1?api_key=abcdefgh12345678 failed"}',
            "abcdefgh12345678",
        ),
        (
            '{"error": "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}',
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ),
        ('{"error": "AKIAIOSFODNN7EXAMPLE was rejected"}', "AKIAIOSFODNN7EXAMPLE"),
        ('{"error": "provider said no to xai-ABC123XYZ789"}', "xai-ABC123XYZ789"),
        (
            '{"error": "expired: AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345678"}',
            "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
        ),
    ],
    ids=[
        "bearer",
        "bearer-lowercase",
        "named-field",
        "query-parameter",
        "github-token",
        "aws-key-id",
        "vendor-prefix",
        "google-api-key",
    ],
)
def test_scrub_secrets_masks_credential_shapes_and_only_those(body: str, canary: str) -> None:
    """Each rule, with the credential it names gone and the surrounding text kept."""
    scrubbed = scrub_secrets(body)

    assert canary not in scrubbed
    assert REDACTION_MARKER in scrubbed
    # The body is otherwise intact: masking is not deleting.
    assert len(scrubbed) > len(REDACTION_MARKER)


def test_response_body_scrubs_the_body_it_reports() -> None:
    """The helper that every legacy call site shares is a redacting accessor."""
    exc = requests.exceptions.HTTPError(
        "refused",
        response=genuine_response(
            401, '{"error": "invalid key: Authorization: Bearer sk-or-v1-LEAKME-1234"}'
        ),
    )

    body = response_body(exc)

    assert "sk-or-v1-LEAKME-1234" not in body
    assert body == '{"error": "invalid key: Authorization: Bearer [redacted]"}'
    # The legacy falsy-bug behaviour this helper was fixed for is unchanged.
    assert response_body(requests.exceptions.ConnectionError("refused")) == NO_RESPONSE_BODY


def test_scrubbed_response_body_reads_a_response_and_reports_absence() -> None:
    """The direct-site accessor: a body in, a scrubbed body out, ``None`` still absent."""
    response = genuine_response(500, '{"error": "proxy said no to Bearer tvly-LEAKME-1"}')

    assert "tvly-LEAKME-1" not in scrubbed_response_body(response)
    assert scrubbed_response_body(None) == NO_RESPONSE_BODY
    assert scrubbed_response_body(genuine_response(204, "   ")) == NO_RESPONSE_BODY


# --- the guard: the next call site is caught by CI ----------------------------------


def _sanctioned_call_spans(tree: ast.AST) -> List[Tuple[int, int]]:
    """Line spans of every call to a sanctioned helper, by its subtree's extent."""
    spans: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in SANCTIONED_HELPERS:
                spans.append(
                    (node.lineno, max(getattr(node, "end_lineno", node.lineno), node.lineno))
                )
    return spans


def _body_reads_into_messages(tree: ast.AST) -> List[Tuple[int, str]]:
    """Line and attribute of every response-body read that reaches a message.

    Two contexts count, because both are how a body reaches a user: an f-string
    (``f"... {response.text}"``) and an argument to an exception constructor
    (``RuntimeError("...", response.content)``). A read anywhere else -- a health
    probe matching a sentinel, a JSON parse -- is not a leak and is left alone.
    """
    sanctioned = _sanctioned_call_spans(tree)
    hits: List[Tuple[int, str]] = []

    def _in_sanctioned(line: int) -> bool:
        return any(start <= line <= end for start, end in sanctioned)

    def _scan(subtree: ast.AST) -> None:
        for node in ast.walk(subtree):
            if isinstance(node, ast.Attribute) and node.attr in {"content", "text"}:
                if not _in_sanctioned(node.lineno):
                    hits.append((node.lineno, node.attr))

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            _scan(node)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in MESSAGE_BUILDERS:
                _scan(node)
    # An f-string inside an exception constructor is reached by both scans above,
    # and a site reported twice reads as two defects.
    return sorted(set(hits))


def test_no_client_module_interpolates_a_response_body() -> None:
    """A body may only reach a message through the shared, scrubbing helpers.

    This is the regression guard for the class of defect review round 1 found: the
    leak was not one call site but five sharing a helper whose contract had
    changed. A new client that reads ``response.content`` into its own message
    fails here, by name and line, instead of shipping.
    """
    modules = sorted(path for path in CLIENTS_PACKAGE.glob("*.py") if path.name != "_http.py")
    # Guards the walk itself: an empty or half-read package would pass vacuously.
    assert len(modules) >= 6, f"only {len(modules)} client modules found; the walk is wrong"

    offenders: List[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, attribute in _body_reads_into_messages(tree):
            offenders.append(f"{path.name}:{line} interpolates response.{attribute}")

    assert not offenders, (
        "these sites put an upstream response body into a message without the "
        "scrubbing helpers, so a credential an upstream reflects reaches the "
        "message and the log:\n  "
        + "\n  ".join(offenders)
        + "\nUse `response_body(exc)`, `scrubbed_response_body(response)` or "
        "`scrub_secrets(text)` instead."
    )


def test_the_body_guard_actually_catches_the_shape_it_names() -> None:
    """The guard's own falsification test: the pre-fix lines must fail it.

    A guard that cannot fail reads as coverage while proving nothing, so it is run
    against the two spellings the leak actually had.
    """
    leaky = (
        "def f(response, exc):\n"
        '    raise RuntimeError(f"failed: {response.content.decode()}")\n'
        '    raise RuntimeError("failed: " + response.text)\n'
    )
    tree = ast.parse(leaky)

    hits = _body_reads_into_messages(tree)

    assert [attribute for _line, attribute in hits] == ["content", "text"]


def test_the_body_guard_accepts_the_sanctioned_shapes() -> None:
    """And it must not fire on the shapes the fix actually uses."""
    clean = (
        "def f(response, exc):\n"
        '    raise RuntimeError(f"failed: {scrubbed_response_body(response)}")\n'
        '    raise RuntimeError(f"failed: {scrub_secrets(response.text)}")\n'
        '    return "Ollama is running" in response.text\n'
        "    # a read that never reaches a message\n"
        "    body = response.content.decode()\n"
        "    return body\n"
    )

    assert _body_reads_into_messages(ast.parse(clean)) == []
