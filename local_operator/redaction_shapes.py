"""Credential SHAPES: the scrubber that does not need to be told the secret.

**Why this module exists, and why it is the one place the patterns live.** The
harness already removed credentials it KNOWS about — session credentials and
values registered for scrubbing — by exact byte-for-byte replacement
(:func:`local_operator.variables.VariableStore.redact`). That covers a secret
the session was handed, and it covers nothing else. A remote host's environment
is precisely the set of secrets the harness has never seen: on 2026-09-18 a
subagent ran ``kubectl exec -n backend-services <pod> -- sh -c 'env | grep -iE
...'`` against a production pod and the tool result came back carrying
``MONGO_DSN=mongodb+srv://agent_runtime_model_worker:<pw>@mongodb-prod.…`` in
full. The masking in that pipeline was the AGENT's own ``sed``, its pattern had
no ``DSN``, and the harness had no fallback: nothing on the result path
recognised a credential that was never handed to the session. The value was
then plaintext in a transcript on disk, and a transcript is read by humans,
copied into bug reports and replayed into later model calls.

The response is a SHAPE pass: a pattern table that masks a value because of how
it is SPELLED — a DSN, a connection string, an ``AWS_SECRET_ACCESS_KEY=`` line,
a PEM block, an issuer-prefixed token — not because the session knew it. It is
deliberately the same policy the HTTP clients already shipped
(:mod:`local_operator.clients._http`), moved here so there is ONE table rather
than a second copy per surface, and widened to the shapes a real CLI prints.

**STDLIB ONLY, and no imports from the rest of the package.** This module is
imported from :mod:`local_operator.variables`, which is deliberately
stdlib-only and sits on the CLI startup path (see
``tests/unit/test_import_graph.py``). Anything heavier here would tax every
``lop --version`` and every scheduler tick, and a package import would also risk
a cycle through the session layer.

**What this does NOT guarantee, stated plainly rather than implied.** It
recognises a credential that is spelled the way something spells a credential.
It does NOT recognise an opaque value with none of these spellings around it: a
bare tenant id, a session cookie pasted without its header, a high-entropy
fragment quoted in isolation, a secret whose name the table does not know. That
residual is real and is the reason overlap with the exact-value pass matters
(a value the session once saw, or one this pass has already matched and the
store registered, is contained for the rest of the session). There is
deliberately no "looks random" / entropy rule: see the note on
:data:`CREDENTIAL_SHAPES`.

**Over-masking is a defect, not headroom.** Every rule below is
context-anchored — a credential has to be spelled the way its issuer spells one
— because the text this runs over is also the text the agent must be able to
read to do its job. A guard that masks every ``*_URL`` or every occurrence of
the word "token" blinds the agent to ordinary output and teaches it to
distrust tool results. ``tests/unit/secrets/test_credential_shapes.py`` pins
both directions: the corpus carries a large NEGATIVE set that must survive
byte-identical, and it is as much a part of the contract as the positive set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Match, Optional, Pattern, Union

REDACTION_MARKER = "[redacted]"
"""What a credential is replaced with in anything about to be surfaced.

Defined here rather than in the HTTP client because every surface shares it: a
second marker string in a second module is how two redaction paths drift into
disagreeing about what a scrubbed value looks like, and the marker is also what
:func:`scrub_shapes` uses to avoid re-masking what it has already masked.
"""


@dataclass(frozen=True)
class Shape:
    """One credential shape: how to find it, which part is the credential, and
    what the masked text looks like.

    ``replacement`` is a template string for the rules that keep a prefix and
    mask a suffix, and a CALLABLE for the two rules that have to keep text on
    both sides of the credential (a DSN password sits between the user and the
    host; an assignment's value sits after its name).

    ``secret_group`` names the capture group holding the CREDENTIAL rather than
    the whole match, so a caller can register exactly that value for later
    exact-value scrubbing (containment) instead of registering a whole sentence.
    ``None`` means the whole match is the credential — which is the right answer
    for a bare issuer-prefixed token and for a PEM block.
    """

    label: str
    pattern: Pattern[str]
    replacement: Union[str, Callable[[Match[str]], str]]
    secret_group: Optional[int] = None


# --- shared sub-patterns -----------------------------------------------------

#: The value characters an assignment may carry: a token, a URL, a base64 blob.
#: Stops at whitespace, quotes, commas and closing braces so the mask cannot run
#: away into the rest of a document when a value is unterminated.
_ASSIGNED_VALUE = r"[^\s\"',;}\]]{4,}"

#: The value of a named assignment, floored at 8 characters and required NOT to
#: END on a separator character.
#:
#: The last-character rule is what keeps the rule off prose. In
#: ``invalid key: Authorization: Bearer …`` a colon-tolerant value class makes
#: ``Authorization:`` look like the value of a variable named ``key``, and the
#: mask lands on a header name rather than on a credential; requiring the value
#: to end on a value character (plus the assertion at the call site) makes that
#: match impossible instead of merely unlikely.
_ASSIGNED_VALUE_GROUP = r"([^\s\"',;}\]{]{7,}[^\s\"',;}\]{:])"

#: A guard for the two rules that consume a WHOLE value: skip when that value
#: already carries the marker.
#:
#: Rules run in order, so one value can be reached twice — ``MONGO_DSN=`` is
#: caught by the DSN rule first (password masked, user and host kept), and the
#: assignment rule would then match the same value again and swallow the whole
#: thing INCLUDING the host it had just preserved — or split the marker on the
#: ``]`` the value charset stops at. Measured while building this table, the
#: unguarded pair produced ``MONGO_DSN=[redacted]]@host``.
_NOT_ALREADY_MASKED = r"(?!\S*\[redacted\])"

#: A credential NAME, in the forms real environments use.
#:
#: This is the part the first version got wrong. The original pattern was a bare
#: word match (``\b(token|secret|...)\b``), which misses every name an actual
#: environment carries: ```` is ``TOKEN`` preceded by ``_``, and ``_`` is a
#: word character, so there is no word boundary to match. That is exactly the
#: incident that motivated this module — ``MONGO_DSN`` — and the same hole
#: covered ``AWS_SECRET_ACCESS_KEY``, ``DB_PASSWORD``, ``GITHUB_TOKEN`` and
#: ``API_KEY``.
#:
#: The shape is therefore: an identifier-ish prefix made of ``word-`` segments,
#: an optional single leading separator (``_authToken``, ``.netrc``-style names),
#: an optional qualifier, and then the credential word as the LAST segment —
#: anchored at a boundary on BOTH sides so ``monkey`` is not ``key`` and
#: ``keyboard_layout`` is not ``key``.
_CREDENTIAL_NAME = (
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z0-9]+[_.\-])*"
    r"[_.\-]?"
    r"(?:"
    r"(?:api|auth|access|refresh|id|session|private|public|client|proxy|secret|"
    r"signing|signed|encryption|master|service|bearer|oauth)[_.\-]?"
    r")?"
    r"(?:key|keys|key[_.\-]?data|token|tokens|secret|secrets|password|passwords|"
    r"passwd|pwd|credential|credentials|dsn)"
    r"\b"
)

#: Names that END in a credential word but are ordinary counts or cache handles,
#: not credentials.
#:
#: ``max_tokens`` is the measured trap: it is a model parameter that appears in
#: tool output and in provider echoes constantly, and masking its value would
#: hide a number the agent needs while protecting nothing. The distinction is
#: real rather than stylistic — a token COUNT is plural and carries no secret, a
#: token is singular — but it cannot be inferred from the tail word alone, so
#: the count-shaped prefixes are named here explicitly. This is half the reason
#: the negative corpus is as large as the positive one.
#:
#: TWO guards, because a regex can reach the same name from two positions and
#: both have to be closed:
#:
#: * the lookbehinds close a match that starts AT the tail — ``context_tokens``
#:   offers ``tokens`` as a name start, and ``_`` is not a word character, so the
#:   boundary check alone lets it through;
#: * the lookahead closes a match that starts at the FRONT of the name, where
#:   ``context_`` is consumed as an ordinary prefix segment.
#:
#: Each lookbehind is a separate fixed-width assertion because Python's ``re``
#: refuses an alternation of different lengths in a lookbehind.
_COUNT_PREFIX_WORDS = (
    "max",
    "min",
    "num",
    "n",
    "total",
    "sum",
    "count",
    "avg",
    "average",
    "context",
    "ctx",
    "prompt",
    "completion",
    "input",
    "output",
    "usage",
    "used",
    "cache",
    "cached",
    "remaining",
    "left",
    "budget",
    "free",
    "limit",
)
_COUNT_TAIL_WORDS = (
    "tokens",
    "token",
    "keys",
    "key",
    "secrets",
    "secret",
    "passwords",
    "password",
    "dsn",
)
_COUNT_PREFIXES = (
    "".join(rf"(?<!{w}[_.\-])" for w in _COUNT_PREFIX_WORDS)
    + "(?!"
    + "|".join(rf"{w}[_.\-]?(?:{'|'.join(_COUNT_TAIL_WORDS)})\b" for w in _COUNT_PREFIX_WORDS)
    + ")"
)

#: Credential names that RUN the prefix into the word, with no separator for the
#: grammar above to split on: ``PGPASSWORD``, ``PGPASSFILE``, ``sshpass``. Listed
#: explicitly because no rule can derive them — the boundary check that keeps
#: ``monkey`` out of ``key`` is exactly what makes ``PGPASSWORD`` invisible, and
#: ``PGPASSWORD=`` is the conventional way to hand postgres a password to a
#: ``pg_dump`` in CI.
_RUN_TOGETHER_NAMES = r"(?:pgpassword|pgpassfile|pgpass|dbpassword|appsecret|sshpass)"

#: The schemes a connection string is spelled with. ``http(s)`` is in the list
#: only because the pattern below requires userinfo: a bare endpoint URL never
#: matches it (see :data:`CREDENTIAL_SHAPES` on over-masking).
_DSN_SCHEMES = (
    r"mongodb(?:\+srv)?|postgres(?:ql)?|mysql|mariadb|rediss?|amqps?|mssql|"
    r"clickhouse|jdbc:[a-z][a-z0-9]*|https?|ftp|sftp|ssh|ldaps?"
)

#: Triple-grouped so the password can be masked with the user and host kept:
#: ``(scheme://user:)(password)(@)``. ``[^\s:/@"']*`` allows an EMPTY user,
#: which is how ``redis://:password@host`` is spelled.
_DSN_PATTERN = re.compile(rf"(?i)\b((?:{_DSN_SCHEMES})://[^\s:/@\"']*:)([^\s/@\"']+)(@)")


#: The credential SHAPES, as ``(label, pattern, replacement, secret_group)``.
#:
#: ORDER IS LOAD-BEARING and runs most-specific first:
#:
#: * the PEM block precedes the name rules, because a ``"private_key"``
#:   assignment would otherwise mask the ``-----BEGIN`` header alone and leave
#:   the key material readable;
#: * the DSN password precedes the ``*_URL`` rule, so ``MONGO_DSN=`` keeps its
#:   user and host readable instead of being masked wholesale;
#: * the named rules precede the bare issuer prefixes, so a value keeps the
#:   label that explains WHY it was masked.
#:
#: Every rule needs a credential to be spelled in a way its issuer spells one.
#: A high-entropy fragment with none of these around it is left alone, and that
#: is a decision, not an oversight: an entropy heuristic on this surface would
#: rewrite build ids, content hashes, model names and base64 thumbnails — text
#: the agent must read — while still missing every secret that is a command's
#: own choice of words (see the module docstring's residual note).
CREDENTIAL_SHAPES: tuple[Shape, ...] = (
    # --- key material, before anything that could take it apart ---------------
    Shape(
        "pem-private-key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]*?)PRIVATE KEY-----"
            r"[\s\S]*?-----END (?:[A-Z0-9 ]*?)PRIVATE KEY-----"
        ),
        REDACTION_MARKER,
    ),
    # ``"private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIE…"`` — the GCP
    # service-account form. The escaped ``\n`` sequences sit inside the JSON
    # string, so the PEM rule above spans them too and this rule only has to
    # catch the header when the body was truncated before an END line arrived.
    Shape(
        "gcp-service-account-key",
        re.compile(r"(?i)(\"?private[_-]?key\"?\s*:\s*\"?)(-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----)"),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # --- connection strings / DSNs ------------------------------------------
    # ``scheme://user:pass@host``. The scheme IS the context, and the shape is
    # credential-carrying by construction: the pattern requires a ``:secret@``
    # before it fires at all. Only the PASSWORD is masked, so the user and host
    # stay readable — that is what lets an operator act on the line (which host,
    # which user) without the credential in hand.
    Shape(
        "dsn-password",
        _DSN_PATTERN,
        lambda m: m.group(1) + REDACTION_MARKER + m.group(3),
        2,
    ),
    # A named URL/URI/endpoint value that CARRIES a credential: a userinfo
    # password, or a credential-shaped query parameter.
    #
    # **Why the value has to prove itself here.** ``*_URI``/``*_URL``/
    # ``*_ENDPOINT`` names are the ones an environment is full of, and almost
    # all of them are ordinary endpoints (``SERVICE_URL=https://api.example.com``,
    # ``CALLBACK_REDIRECT_URI=http://localhost:8080/cb``). Masking every one of
    # them would blind the agent to the output it is reading. Requiring the value
    # to carry a credential keeps those readable and still catches the one that
    # matters — including the schemes the DSN rule above does not enumerate
    # (``SMTP_URL=smtp://user:pw@host``).
    #
    # The ``(?!\[redacted\])`` guard keeps this from masking a value the DSN rule
    # already masked: without it, a DSN that survived the first pass would be
    # swallowed whole by this one, taking the readable host with it.
    Shape(
        "credential-url-value",
        re.compile(
            r"(?i)(?<![A-Za-z0-9])(?:[A-Za-z0-9]+[_.\-])*[_.\-]?"
            r"(?:uri|url|endpoint|conn(?:ection)?[_.\-]?str(?:ing)?|dsn)(?![A-Za-z0-9])"
            r"(\"?\s*[:=]\s*\"?)"
            rf"{_NOT_ALREADY_MASKED}"
            r"(?=[^\s\"',;}\]]*(?:"
            r"://[^\s/@\"']*:(?!\[redacted\])[^\s/@\"']+@"
            r"|(?!\[redacted\])[?&](?:api[-_]?key|apikey|key|token|secret|password|"
            r"passwd|pwd|sig|signature)=))"
            rf"({_ASSIGNED_VALUE})"
        ),
        r"\1" + REDACTION_MARKER,
    ),
    # A credential in a query string — how several of these services accept one
    # and therefore how one comes back in a URL that a log or an upstream quotes.
    Shape(
        "credential-query-param",
        re.compile(
            r"(?i)([?&](?:api[-_]?key|apikey|key|token|access[-_]?token|secret|"
            r"password|passwd|pwd|sig|signature|auth)=)([^&\s\"'#]{4,})"
        ),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # --- named assignments ---------------------------------------------------
    # ``AWS_SECRET_ACCESS_KEY=…``, ``"api_key": "…"``, ``DB_PASSWORD=…``,
    # ``GITHUB_TOKEN=…``, ``MONGO_DSN=…`` — an assignment to a name that ends in
    # a credential word, with ``=`` or ``:`` and optional quotes.
    #
    # The 8-character floor is what keeps ordinary prose out of it: ``"secret":
    # "missing"`` and ``token=<none>`` stay readable while a key of real length is
    # masked. It is a floor on the VALUE only — a short value under a
    # credential-shaped name is a placeholder or a word, not a credential.
    Shape(
        "credential-assignment",
        re.compile(
            rf"(?i)(?:{_RUN_TOGETHER_NAMES}(?![A-Za-z0-9])"
            rf"|{_COUNT_PREFIXES}{_CREDENTIAL_NAME})"
            # ``[\"']?`` before the separator so a QUOTED name is one of these:
            # ``"api_key": "…"`` is how JSON spells every one of them, and a
            # pattern that only accepts the bare ``name: value`` form misses the
            # whole shape on the most common surface there is.
            r"([\"']?\s*[:=]\s*)([\"']?)"
            rf"{_NOT_ALREADY_MASKED}"
            rf"{_ASSIGNED_VALUE_GROUP}"
            # ...and the value must not be a NAME followed by its own
            # separator. Without this the rule fires on prose:
            # ``invalid key: Authorization: Bearer …`` matched ``key:`` as the
            # name and ``Authorization:`` as the value, masking a header name
            # and a scheme keyword instead of a credential. Measured against
            # tests/unit/clients/test_error_body_redaction.py, which pins that
            # exact body.
            r"(?![A-Za-z0-9_.\-:/])"
        ),
        # Keep the name, the separator and the opening quote; mask only the
        # value. A template cannot express that (the value is the LAST group and
        # the text before it must survive), so this rule carries a callable.
        lambda m: m.group(0)[: m.start(3) - m.start(0)] + REDACTION_MARKER,
        3,
    ),
    # ``.netrc``: ``machine api.example.com login robot password …``. A
    # whitespace-separated assignment, so the ``[:=]`` rules above never see it.
    # Anchored on the ``machine … login`` pair rather than on the bare word,
    # because "the password was rotated today" is ordinary prose and must not be
    # rewritten.
    Shape(
        "netrc-password",
        re.compile(r"(?i)(\bmachine\s+\S+\s+login\s+\S+\s+password\s+)(\S{4,})"),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # --- headers -------------------------------------------------------------
    # ``Authorization: Bearer <token>``, in every casing. The scheme keyword IS
    # the credential context; the 4-character floor keeps a dangling
    # ``Authorization: Bearer`` (no value) from being rewritten.
    Shape(
        "authorization-bearer",
        re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._~+/=-]{4,})"),
        r"\1 " + REDACTION_MARKER,
        2,
    ),
    Shape(
        "authorization-basic",
        re.compile(r"(?i)\b(basic)\s+([A-Za-z0-9+/=]{8,})"),
        r"\1 " + REDACTION_MARKER,
        2,
    ),
    # A session cookie is a credential and the value is opaque by design, so the
    # whole header value goes. ``Set-Cookie`` is the response half, ``Cookie``
    # the request half, and both reach a transcript through an echoing debug
    # line or a ``curl -v`` far more often than anyone expects.
    Shape(
        "cookie-header",
        re.compile(r"(?i)\b(set-cookie|cookie)\s*:\s*([^\r\n]{4,})"),
        r"\1: " + REDACTION_MARKER,
        2,
    ),
    # --- CLI / ecosystem shapes ---------------------------------------------
    # ``--password=hunter2``, ``--password hunter2``, ``--token …`` — how a CLI
    # that takes a credential as a flag spells it. The flag name is the context
    # and the ``-``/whitespace after it is required, so ``--token-ttl=3600``
    # (a duration) is not one of these.
    Shape(
        "cli-credential-flag",
        re.compile(
            r"(?i)(--(?:password|passwd|pwd|token|api[-_]?key|apikey|secret|"
            r"client[-_]?secret|auth[-_]?token|access[-_]?token)(?:=|\s+))([^\s\"']{3,})"
        ),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # ``mysql -u root -phunter2``, ``psql -p…``. Anchored on the client binary
    # because ``-p`` is a PORT on almost everything else (``docker run -p
    # 8080:80``, ``ssh -p 2222``) and masking a port would be both unhelpful and
    # wrong. The binary name is the only thing that distinguishes the two, which
    # is why this rule exists at all rather than a bare ``-p`` pattern.
    Shape(
        "client-inline-password",
        re.compile(
            r"(?i)(\b(?:mysql|mysqldump|psql|pg_dump|pg_restore|mongo|mongosh|"
            r"clickhouse-client|redis-cli|influx)\b[^\n]{0,200}?\s-(?:p|a)(?:\s+)?)(\S{2,})"
        ),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # ``.npmrc``: ``//registry.npmjs.org/:_authToken=npm_…``.
    Shape(
        "npmrc-auth-token",
        re.compile(r"(?i)(_auth(?:token)?=)([^\s\"']{6,})"),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # ``docker login -u robot -p <password>``. Anchored on ``docker login``
    # because ``-p`` is a PORT on ``docker run`` — the published-port case is in
    # the negative corpus, and one pattern cannot serve both.
    Shape(
        "docker-login-password",
        re.compile(r"(?i)(\bdocker\s+login\b[^\n]{0,200}?\s-p\s?)(\S{4,})"),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # ``curl -u user:password https://…`` — the credential is the userinfo of a
    # flag rather than of a URL, so the DSN rule cannot see it. Anchored on the
    # curl binary for the same reason as ``-p`` above: ``-u`` means other things
    # to other programs.
    Shape(
        "curl-user-credential",
        re.compile(r"(?i)(\bcurl\b[^\n]{0,200}?\s-u\s+['\"]?[^:'\"\s]+:)([^'\"\s]+)"),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # ``openssl … -passin pass:<password>`` / ``-passout``. The flag IS the
    # context; a bare ``pass:`` would be prose.
    Shape(
        "openssl-pass-phrase",
        re.compile(r"(?i)(-pass(?:in|out|phrase|wd|wdin|wdout)?\s+)(?:pass:)?(\S{4,})"),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # A Slack incoming-webhook URL: the HOST is the context and the path IS the
    # credential — anyone holding it can post to the channel.
    Shape(
        "slack-webhook-url",
        re.compile(r"https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]{10,}"),
        REDACTION_MARKER,
    ),
    # ``{"auths": {... "auth": "dXNlcjpwYXNz"}}`` — the docker config.json form.
    Shape(
        "docker-config-auth",
        re.compile(r"(?i)([\"']auth[\"']\s*:\s*[\"'])([A-Za-z0-9+/=]{16,})([\"'])"),
        r"\1" + REDACTION_MARKER + r"\3",
        2,
    ),
    # --- bare tokens, by the prefix their issuer gives them -------------------
    # The shape left over when a header value is quoted without its header, or a
    # CLI prints the token alone. These are issuer PREFIXES, not a general
    # "looks random" heuristic — each one is a string no other issuer uses.
    Shape(
        "stripe-key",
        re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "sendgrid-key",
        re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "google-oauth-token",
        # ``ya29.`` — the prefix Google's OAuth access tokens carry. It is
        # dotted rather than dashed, so the generic prefix rule below never saw
        # them: measured, ``ya29.a0AfH6…`` came back unmasked.
        re.compile(r"\bya29\.[A-Za-z0-9._\-]{20,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "vendor-prefixed-token",
        re.compile(
            r"\b(?:sk|pk|rk|hf|gsk|xai|tvly|fal|serp|glpat|ya29|npm|pypi|whsec|"
            r"dckr_pat|shpat|shpss|lin_api|syt_|doo_v1|pat_)[-_][A-Za-z0-9_.\-]{6,}"
        ),
        REDACTION_MARKER,
    ),
    Shape(
        "github-token",
        re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "github-pat",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "aws-access-key-id",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "google-api-key",
        re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "slack-token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        REDACTION_MARKER,
    ),
)

#: Shortest matched value worth registering as a session redaction. Below this a
#: value cannot be told apart from ordinary text, and registering it would
#: rewrite unrelated output for the rest of the session — the failure measured
#: on ``mcp.redaction``'s three-character value (see that module's
#: ``MIN_SCRUBBED_LENGTH``, which draws the same line for the same reason).
_MIN_REGISTERABLE_SECRET = 8


@dataclass(frozen=True)
class ShapeHit:
    """One shape that fired: its label, and the credential it matched.

    ``value`` is the credential itself — never logged, never rendered. It exists
    so the caller can register it for exact-value scrubbing for the rest of the
    session, which is what stops the same secret reappearing later in a form
    this table does not know (a bare paste, a concatenation, a piece of another
    command's output).
    """

    label: str
    value: str


def scrub_shapes_with_hits(text: str) -> tuple[str, list[ShapeHit]]:
    """Scrub ``text`` and report every shape that fired.

    The one place the table is run. :func:`scrub_shapes` and
    :func:`match_shape_names` are both views onto this, so a surface cannot
    inspect shapes without the same rules being applied to the text.
    """
    hits: list[ShapeHit] = []
    for shape in CREDENTIAL_SHAPES:
        matches = list(shape.pattern.finditer(text))
        if matches:
            for match in matches:
                value = _hit_value(shape, match)
                if len(value) >= _MIN_REGISTERABLE_SECRET:
                    hits.append(ShapeHit(shape.label, value))
        text = shape.pattern.sub(shape.replacement, text)
    return text, hits


def _hit_value(shape: Shape, match: "re.Match[str]") -> str:
    """The credential a match carries, for registration as a session redaction.

    ``secret_group`` where the table names one, the whole match otherwise — and
    the whole match is the credential exactly for the rules that exist to find a
    bare token or a PEM block. A group that the pattern does not have is a table
    typo, not a runtime state, and yields "" (nothing registered) rather than an
    exception: this pass runs on the result path, where raising turns a tool
    result into a tool crash.
    """
    if shape.secret_group is None:
        return match.group(0)
    try:
        return match.group(shape.secret_group) or ""
    except IndexError:  # pragma: no cover - a table typo, not a runtime state
        return ""


def scrub_shapes(text: str) -> str:
    """Mask every credential-shaped value in ``text``. Patterns only.

    The pattern pass alone, for callers that have no value list to offer (a
    command's own arguments, a notice line). See :func:`scrub_secrets` for the
    composed pass, which is what a tool result should go through.
    """
    scrubbed, _ = scrub_shapes_with_hits(text)
    return scrubbed


def scrub_secrets(text: str, values: Iterable[Optional[str]] = ()) -> str:
    """Remove known credential VALUES, then credential SHAPES, from ``text``.

    The composed pass, and the one every model-visible surface should use.

    Exact values first, longest first, so a value that is a prefix of another
    cannot leave a tail behind; empty and ``None`` entries are skipped, because
    replacing the empty string would insert the marker between every character.
    Then the shapes, which catch what the session was never told — the case this
    module exists for (see the module docstring).
    """
    return scrub_secrets_with_hits(text, values)[0]


def scrub_secrets_with_hits(
    text: str, values: Iterable[Optional[str]] = ()
) -> tuple[str, list[ShapeHit]]:
    """
    :func:`scrub_secrets`, plus the shapes that fired.

    The caller that needs the labels (a session reporting WHY a tool result was
    rewritten) and the caller that needs the matched VALUES (a store registering
    them for containment) both read this, so neither can observe a mask without
    the same rule having produced it.
    """
    return scrub_shapes_with_hits(_scrub_values(text, values))


def _scrub_values(text: str, values: Iterable[Optional[str]]) -> str:
    """The exact-value half on its own: longest first, empties skipped.

    ``str`` rather than the declared ``Optional[str]``: the list is built from a
    store, and a non-string entry there is a bug this pass should survive rather
    than raise on — a redaction failure that raises turns a tool result into a
    tool crash.
    """
    result = text
    ordered = sorted((value for value in values if value), key=len, reverse=True)
    for value in ordered:
        if isinstance(value, str) and value in result:
            result = result.replace(value, REDACTION_MARKER)
    return result


def match_shape_names(text: str) -> list[str]:
    """The LABELS of the shapes present in ``text`` — never the values.

    For the notice path, which has to say WHY a result was rewritten without
    putting the credential back on screen.
    """
    _, hits = scrub_shapes_with_hits(text)
    # De-duplicated, insertion-ordered: a rule that fires five times in one
    # result is one fact about the result, not five.
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.label, None)
    return list(seen)


# --- credential-printing CLI detection ---------------------------------------
#
# The second half of "detect more": the shape pass masks a value that is already
# in the output, and this table catches the COMMAND about to produce one. The
# two are complementary rather than redundant — a command that prints a secret
# the table has no shape for is still worth a warning, and the warning reaches
# the model BEFORE it repeats the mistake in a different form.
#
# The notice is ADVISORY and must never fire on ordinary work: ``npm list``,
# ``kubectl get pods``, ``docker ps``, ``env -i HOME=… cmd`` and ``printenv
# PATH`` are all ordinary, and a guard that nags on them is one the agent (and
# the operator) learns to ignore — which costs more than the guard is worth.
# Every rule below is therefore anchored on the verb/flag combination that
# PRINTS a value, not on the tool name.


@dataclass(frozen=True)
class DumpShape:
    """A command shape that prints credentials, the safer form to suggest, and
    any spelling of the same command that is already safe.

    ``excluded`` is what keeps a notice honest: the table recommends a safer
    form in its own text, so a rule that then fires on that form is warning
    about the thing it just told the agent to do.
    """

    label: str
    pattern: Pattern[str]
    safer: str
    excluded: Optional[Pattern[str]] = None


DUMP_SHAPES: tuple[DumpShape, ...] = (
    # A bare ``env`` / ``printenv`` / ``set`` prints every value in the
    # environment — this is the incident's own shape. The command has to be at a
    # command position and be IMMEDIATELY followed by a separator or the end of
    # the command: ``env -i HOME=… cmd`` runs a command (and is how a clean
    # environment is built), ``printenv PATH`` asks for one harmless name, and
    # neither is a dump.
    DumpShape(
        "environment-dump",
        re.compile(r"(?:^|[;&|(]\s*|\$\(\s*)(?:env|printenv|set)\s*(?=[|;&)]|$)"),
        "print names, not values: `env | cut -d= -f1`; or use the value inside the "
        "command that needs it, e.g. `$(lop secret get NAME)`",
    ),
    # ``printenv MONGO_DSN`` — a single VARIABLE whose name says it is a
    # credential. Narrow on purpose: ``printenv PATH`` must stay ordinary.
    DumpShape(
        "named-variable-dump",
        re.compile(rf"(?i)(?:^|[;&|(]\s*|\$\(\s*)printenv\s+{_COUNT_PREFIXES}{_CREDENTIAL_NAME}"),
        "print only the part you need, or read the value inside the command that "
        "needs it, e.g. `$(lop secret get NAME)`",
    ),
    DumpShape(
        "kubectl-exec-env",
        re.compile(
            # ``(?<![A-Za-z0-9_.\-])`` rather than a space: env is reached as
            # ``-- env``, as ``-c 'env'`` inside a quoted shell, and as the first
            # word of a sh -c string, and only the word boundary is common to
            # all three. It also keeps ``.env`` (a file) and ``printenv``
            # (matched by the ``env`` alternative) out of it.
            r"(?i)\bkubectl\s+(?:exec|run)\b[^\n]*?"
            r"(?<![A-Za-z0-9_.\-])(?:env|printenv)\s*(?=[|;&)\"'`]|$)"
        ),
        "print names only inside the pod: `kubectl exec … -- sh -c 'env | cut -d= -f1'`",
    ),
    DumpShape(
        "kubectl-secret-read",
        # Fires on the spellings that DUMP: every secret in the namespace, a
        # secret rendered as yaml/json, or a describe. The one-key jsonpath form
        # is in ``excluded`` because it is the safer form this table recommends.
        re.compile(
            r"(?i)\bkubectl\s+(?:"
            r"get\s+secrets?\b[^\n]*?\s-o\s*(?:yaml|json|wide)\b"
            r"|get\s+secrets\b"
            r"|describe\s+secrets?\b"
            r")"
        ),
        "select one key inside the command that needs it: "
        "`kubectl get secret NAME -o jsonpath='{.data.KEY}' | base64 -d`",
        excluded=re.compile(r"(?i)-o\s*(?:jsonpath|go-template|custom-columns)"),
    ),
    DumpShape(
        "docker-inspect",
        re.compile(r"(?i)\bdocker\s+inspect\b"),
        "select the field you need: `docker inspect --format '{{.State.Status}}' NAME`",
    ),
    DumpShape(
        "docker-exec-env",
        re.compile(
            r"(?i)\bdocker\s+(?:exec|run)\b[^\n]*?"
            r"(?<![A-Za-z0-9_.\-])(?:env|printenv)\s*(?=[|;&)\"'`]|$)"
        ),
        "print names only: `docker exec … env | cut -d= -f1`",
    ),
    DumpShape(
        "docker-compose-config",
        re.compile(r"(?i)\bdocker\s+compose\s+config\b"),
        "read the one service or key you need: `docker compose config --services`",
        # ``config --services`` (and the other SELECTOR flags) print names, not
        # resolved values, and this table recommends exactly that form.
        excluded=re.compile(r"(?i)--(?:services|images|hash|volumes|profiles|networks|list)\b"),
    ),
    DumpShape(
        "aws-credentials-read",
        # The token-ISSUING reads, plus reading the stored credential itself.
        # ``aws sts get-caller-identity`` is deliberately NOT here: it prints an
        # account id and an ARN, which is identity rather than a credential, and
        # the safer-form column for it would be noise.
        re.compile(
            r"(?i)\baws\s+(?:configure\s+(?:list|get)(?:\s|$)|"
            r"secretsmanager\s+get-secret-value|"
            r"sts\s+get-(?:session|federation|delegation)-token|"
            r"iam\s+list-access-keys|"
            r"configure\s+export-credentials)\b"
        ),
        "use the credential inside the command that needs it, or read one field: "
        "`aws secretsmanager get-secret-value --query SecretString --output text` "
        "still prints it — prefer a scoped temporary credential",
    ),
    DumpShape(
        "gcloud-token",
        re.compile(
            r"(?i)\bgcloud\s+(?:auth\s+print-[a-z\-]*token|secrets\s+versions\s+access|"
            r"auth\s+application-default\s+print-access-token)\b"
        ),
        "use the token inside the command that needs it, e.g. "
        '`curl -H "Authorization: Bearer $(gcloud auth print-access-token)"`',
    ),
    DumpShape(
        "github-auth-token",
        re.compile(r"(?i)\bgh\s+auth\s+token\b"),
        'use it inside the command that needs it: `gh api -H "'
        '"Authorization: token $(gh auth token)"`',
    ),
    DumpShape(
        "gitlab-auth-token",
        re.compile(r"(?i)\bglab\s+auth\s+(?:status|token)\b[^\n]*\s(?:-t|--show-token)\b"),
        "drop `-t`: `glab auth status` reports the login without printing the token",
    ),
    DumpShape(
        "vault-read",
        re.compile(r"(?i)\bvault\s+kv\s+(?:get|read)\b"),
        "select one field inside the command that needs it: "
        "`vault kv get -field=KEY secret/name`",
    ),
    DumpShape(
        "heroku-config",
        re.compile(r"(?i)\bheroku\s+config(?::get)?\b"),
        "read one key: `heroku config:get KEY -a APP`",
    ),
    DumpShape(
        "terraform-output",
        re.compile(r"(?i)\bterraform\s+(?:output|state\s+show)\b"),
        "read one output by name: `terraform output name`, and mark secrets "
        "`sensitive = true` so they are not printed at all",
    ),
    DumpShape(
        "npm-token-list",
        re.compile(r"(?i)\bnpm\s+token\s+(?:list|ls|create)\b"),
        "`npm token list` prints the tokens themselves; revoke the unused one "
        "and mint a fresh one when you need it",
    ),
    DumpShape(
        "credential-file-read",
        re.compile(
            r"(?i)\b(?:cat|less|more|head|tail|bat|strings|xxd|base64)\b[^\n]*?"
            r"(?:\.netrc\b|\.npmrc\b|\.docker/config\.json\b|\.kube/config\b|"
            r"mcp\.json\b|credentials\.env\b|\.env\b|[^\s]*\.pem\b|"
            r"[^\s]*service-account[^\s]*\.json\b)"
        ),
        "read the one field you need (e.g. `grep -c .`, `jq '.client_email'`), or "
        "use the value inside the command that needs it, e.g. `$(lop secret get NAME)`",
    ),
)

#: A pipeline that prints only NAMES is the form the notice itself recommends,
#: so it must not trigger the notice. Kept as one pattern rather than a flag on
#: every rule: what makes a pipeline name-only is the extractor at its end.
_NAME_ONLY_PIPELINE = re.compile(
    r"(?i)^\s*\|?\s*(?:(?:cut\s+-d\s*=)|(?:awk\s+-F\s*=)|(?:sed\s+-E?\s*\"?'?s/=))"
)


def credential_dump_notice(command: str) -> Optional[str]:
    """One advisory line when ``command`` is shaped like a credential dump.

    ``None`` when the command is ordinary — which is the common case and the
    one the table is written to protect (see :data:`DUMP_SHAPES`). The returned
    text never contains a value from the command: it names what was detected and
    the safer form, because a notice is not the place to repeat the mistake it
    is warning about.
    """
    if not command:
        return None
    for shape in DUMP_SHAPES:
        match = shape.pattern.search(command)
        if match is None:
            continue
        if shape.excluded is not None and shape.excluded.search(command):
            continue
        if _is_name_only_pipeline(command, match):
            continue
        return (
            f"[credential guard] this command prints credentials "
            f"({shape.label}). Safer: {shape.safer}."
        )
    return None


def _is_name_only_pipeline(command: str, match: Match[str]) -> bool:
    """Whether the dump is piped straight into a names-only extractor."""
    return bool(_NAME_ONLY_PIPELINE.match(command[match.end() :]))
