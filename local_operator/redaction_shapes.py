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

**Two model-visible surfaces this pass does not reach, named rather than
implied.** The composed scrubber is the tool/result seam and everything the
session hands to the loop, which is where the audit that produced this module
looked and where it found the leak. It is NOT on:

* the **system-prompt blocks** (``prompts_api.build_system_blocks``: repo
  guidance, the skills block, the base prompt). A credential sitting in a
  checked-in file that the session injects therefore reaches the model verbatim;
  the session is not the source of that value and cannot contain it;
* the **MCP diagnostics scrubber** (``mcp/redaction.py``), which stays values-only
  by design — the surfaces it protects (``report_failure`` logs, ``explain()``
  text, the TUI notice) are not model-visible, and every model-visible MCP path
  crosses the loop's result hook.

Claiming "every model-visible surface" without those two would be overclaiming,
so this module states the set it covers instead.

**Over-masking is a defect, not headroom.** Every rule below is
context-anchored — a credential has to be spelled the way its issuer spells one
— because the text this runs over is also the text the agent must be able to
read to do its job. A guard that masks every ``*_URL`` or every occurrence of
the word "token" blinds the agent to ordinary output and teaches it to
distrust tool results. ``tests/unit/secrets/test_credential_shapes.py`` pins
both directions: the corpus carries a large NEGATIVE set that must survive
byte-identical, and it is as much a part of the contract as the positive set.


INVARIANT: **a mask is all of the credential or none of it** — never a masked prefix
with the remainder readable. A notice that says a credential was masked has to be
true: a readable fragment left behind has entered the model's context window, which
is the one condition this harness treats as a compromise (see the next paragraph).
`_close_partial_masks` enforces it for every rule, and
`tests/unit/secrets/test_credential_shapes.py` sweeps it over the corpus with a
frozen, ratcheted residual (three rules, each with its reason recorded beside the
table).

**WHAT "COMPROMISED" MEANS HERE, because the severity of the notice hangs off it.**
A credential is compromised when a VALUE reaches the MODEL'S CONTEXT WINDOW — the
unmasked text of a request, the transcript it is journaled to, and so plausibly a
training corpus. A credential that reaches `bash` (its `argv`, a child's
environment), that lives in this process's memory, or that is written to disk in
plaintext is NOT compromised: each of those is a containment, and the only thing it
owes anyone is cleanup. That is why :attr:`ShapeHit.exposed` exists and why it is
the sole input to the escalated notice: a hit that was masked whole is reported as
an event with NO exposure, and only a fragment that survived into the text the model
reads escalates. This paragraph is about SEVERITY; it changes nothing about
coverage, where under-masking is still a leak and over-masking is still a defect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Match, Optional, Pattern, Sequence, Union

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
    #: ``None`` only for a GUARDED rule: the guard decides, and the guarded path
    #: renders the mask from the group the credential is in. Constructing a shape
    #: with neither a replacement nor a guard raises at import (see
    #: :meth:`__post_init__`) — a rule that matched and then published the
    #: credential unchanged is the one failure mode this table must not have.
    replacement: Optional[Union[str, Callable[[Match[str]], str]]] = None
    secret_group: Optional[int] = None
    #: A condition the PATTERN cannot express cheaply, checked only on a match.
    #:
    #: **Why it is not in the pattern.** The two name-driven rules have to ask
    #: whether an identifier ENDS in a credential word and whether it is a
    #: count-shaped false friend (``max_tokens``). Expressing that as lookarounds
    #: puts a fixed-width assertion per rejected word at EVERY character position
    #: of every input, and this pass runs over every tool result and every live
    #: pipe chunk: measured at 3.0 µs per input byte (1.4 s for 460 KB of
    #: ordinary log output) with the guards in the pattern, against 0.4 µs with
    #: them here, checked on the handful of positions where an assignment
    #: actually is. The semantics are the same and the corpus pins them.
    guard: Optional[Callable[[Match[str]], bool]] = None

    def __post_init__(self) -> None:
        """Refuse a shape that could DETECT a credential and let it through.

        The two optional halves of a rule are the replacement and the guard, and
        exactly one of them must be present: a template or callable renders the
        mask, or a guard decides whether to render one. A shape with neither
        would match, do nothing, and register nothing — the silent
        detect-but-publish state this whole module exists to prevent — so it is
        rejected where it is written, at import, rather than at the first tool
        result that happens to match it.
        """
        if self.replacement is None and self.guard is None:
            raise ValueError(
                f"shape {self.label!r} has neither a replacement nor a guard: "
                "it would match text and mask nothing"
            )
        if self.replacement is not None and self.guard is not None:
            raise ValueError(
                f"shape {self.label!r} has both a replacement and a guard: "
                "the guarded path renders the mask, so the replacement is dead"
            )
        if self.guard is not None and self.secret_group is None:
            raise ValueError(
                f"shape {self.label!r} is guarded but names no secret_group: "
                "the guarded path masks the group holding the credential"
            )


# --- shared sub-patterns -----------------------------------------------------

#: The value characters an assignment may carry: a token, a URL, a base64 blob.
#: Stops at whitespace, quotes, commas and closing braces so the mask cannot run
#: away into the rest of a document when a value is unterminated.
_ASSIGNED_VALUE = r"[^\s]{4,200}"

#: The value of a named assignment, floored at 8 characters and required NOT to
#: END on a separator character.
#:
#: The last-character rule is what keeps the rule off prose. In
#: ``invalid key: Authorization: Bearer …`` a colon-tolerant value class makes
#: ``Authorization:`` look like the value of a variable named ``key``, and the
#: mask lands on a header name rather than on a credential; requiring the value
#: to end on a value character (plus the assertion at the call site) makes that
#: match impossible instead of merely unlikely.
#: A value is any run of non-whitespace, bounded, TERMINATED by a delimiter.
#:
#: The old class EXCLUDED ``,;{}"'``, which meant a credential containing one of
#: them was published in part (``PASSWORD=hunter2hunter2,hunter2`` →
#: ``PASSWORD=[redacted],hunter2``) or in full (``{"password": "abcdef,ghij"}``,
#: ``DB_PASSWORD=abc,defghij`` — nothing masked at all, because the group could
#: not reach its own floor). Excluding characters from a credential's charset is
#: the wrong direction to bound a match in: the bound belongs in a LENGTH and a
#: terminating delimiter, so the value keeps every character a real credential
#: can contain and still cannot run away into the rest of a document.
_ASSIGNED_VALUE_GROUP = (
    # Greedy run of non-whitespace, whose LAST character must be a value
    # character (not a separator, a closer, a quote or a full stop) and which
    # must be followed by a delimiter or the end. Greedy is what makes
    # ``PASSWORD=hunter2,hunter2`` mask the COMMA TOO rather than stopping at it;
    # the last-character rule is what makes ``{"password": "abc,defghij"}`` stop
    # before the closing quote instead of swallowing it.
    #
    # There is deliberately NO UPPER BOUND on the value. A bound with a
    # "followed by a delimiter" requirement is a leak for a credential longer
    # than the bound in a whitespace-free run: the engine finds no delimiter
    # inside the window, gives up, and publishes the whole thing untouched.
    # ``[^\s]`` already cannot cross a line, so there is nothing to run away
    # into.
    r"([^\s]{7,}[^\s,;)\]}\"'.])(?=[\s,;)\]}\"']|$)"
)

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
#: ``scheme://user:password@host``.
#:
#: The password class admits ``@`` and ``/`` and is GREEDY, so it takes the run
#: that ends at the LAST ``@`` of the authority rather than the first: a password
#: containing either character (``p@ssw0rd``, ``p/ssw0rd`` — both ordinary in
#: base64 and generated secrets) was masked up to the first one and the REST WAS
#: PUBLISHED, with an incident row then telling the model the credential had been
#: masked. Measured before the fix: ``mongodb+srv://svc:p@ssw0rd@db/x`` came out
#: ``svc:[redacted]@ssw0rd@db/x`` and ``postgres://svc:p/ssw0rd@db/app`` — whose
#: name is not credential-shaped, so the assignment rule cannot rescue it — was
#: not masked at all, with ZERO incidents. The class still refuses whitespace,
#: quotes and ``:`` so an unterminated line cannot run away, and the authority
#: part (``user``) still cannot contain ``:``/``@``.
_DSN_PATTERN = re.compile(
    rf"(?i)\b((?:{_DSN_SCHEMES})://[^\s:/@\"']*:)(?!\[redacted\])"
    r"([^\s]*?)@(?=[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]+(?:[/:?#]|$))"
)


#: The words a NAME may end in, and the qualifiers that may precede one inside a
#: run-together name (``apiKey``, ``authToken``). Kept as data rather than as a
#: regex so the check runs on a MATCH instead of at every character position —
#: see :attr:`Shape.guard` for the measurement that forced the move.
_CRED_TAIL_WORDS = (
    "key",
    "keys",
    "keydata",
    "token",
    "tokens",
    "secret",
    "secrets",
    "password",
    "passwords",
    "passwd",
    "pwd",
    "credential",
    "credentials",
    "dsn",
)
_CRED_QUALIFIERS = (
    "api",
    "auth",
    "access",
    "refresh",
    "id",
    "session",
    "private",
    "public",
    "client",
    "proxy",
    "secret",
    "signing",
    "signed",
    "encryption",
    "master",
    "service",
    "bearer",
    "oauth",
)
#: Every spelling a name (or its last separator-delimited segment) may have.
_CRED_NAME_FORMS = frozenset(
    qualifier + tail for qualifier in _CRED_QUALIFIERS for tail in _CRED_TAIL_WORDS
) | frozenset(_CRED_TAIL_WORDS)

#: Names that RUN the qualifier into the word with no separator at all, which no
#: split can recover: ``PGPASSWORD`` is how CI hands postgres a password.
_RUN_TOGETHER_NAMES = frozenset(
    {"pgpassword", "pgpassfile", "pgpass", "dbpassword", "appsecret", "sshpass"}
)

#: Prefixes that mark a name as a COUNT or a cache/handle rather than a
#: credential (``max_tokens``, ``context_tokens``, ``cache_key``). Named
#: explicitly because the distinction cannot be inferred from the tail word.
_COUNT_WORDS = frozenset(
    {
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
        "page",
    }
)

_SPLIT_NAME = re.compile(r"[_.\-]")


def _name_segments(name: str) -> list[str]:
    return [part for part in _SPLIT_NAME.split(name.strip("_.-").lower()) if part]


def is_credential_name(name: str) -> bool:
    """Whether an identifier ENDS in a credential word, at a segment boundary.

    ``AWS_SECRET_ACCESS_KEY``, ``MONGO_DSN``, ``DB_PASSWORD``, ``_authToken``,
    ``apiKey`` and ``PGPASSWORD`` are; ``monkey`` is not (the word must start a
    segment), ``keyboard_layout`` is not (it does not END in one), and
    ``num_tokens`` is not a credential either — it is rejected as a count, which
    is the separate judgement :func:`is_count_shaped` makes.
    """
    lowered = name.strip("_.-").lower()
    if not lowered:
        return False
    if lowered in _RUN_TOGETHER_NAMES:
        return True
    segments = _name_segments(lowered)
    if not segments:
        return False
    tail = segments[-1]
    if tail in _CRED_NAME_FORMS:
        return True
    # A trailing pair split apart by the separator: ``client-key-data`` is
    # kubeconfig's private key, spelled in three segments.
    return len(segments) > 1 and segments[-2] + tail in _CRED_NAME_FORMS


def is_count_shaped(name: str) -> bool:
    """Whether a credential-shaped name is actually a count or a cache handle.

    Plural token COUNTS (``max_tokens``, ``context_tokens``) and cache handles
    (``cache_key``) end in a credential word and carry no secret. Masking them
    would hide numbers the agent needs while protecting nothing.
    """
    segments = _name_segments(name)
    return len(segments) > 1 and segments[0] in _COUNT_WORDS


#: Issuer prefixes whose separator is one of ``-``/``_`` (so the gate needs both
#: spellings) and the ones that carry a fixed literal separator already.
_VENDOR_SEPARATED_PREFIXES: tuple[str, ...] = (
    "sk",
    "pk",
    "rk",
    "hf",
    "gsk",
    "xai",
    "tvly",
    "fal",
    "serp",
    "glpat",
    "ya29",
    "npm",
    "pypi",
)
_VENDOR_FIXED_PREFIXES: tuple[str, ...] = (
    "whsec",
    "dckr_pat",
    "shpat",
    "shpss",
    "lin_api",
    "syt_",
    "doo_v1",
    "pat_",
)

#: The token-tail grammar, shared by every vendor prefix: at least 8 token
#: characters that must include a digit. The digit is the cheapest honest
#: discriminator between a real issuer token and an ordinary hyphenated name
#: (``pypi-local-operator.json``); a length floor alone was not enough.
#: The token-tail grammar, shared by every vendor prefix: NO dot, and at least
#: 12 characters. Two structural decisions, each measured:
#:
#: * **no dot** — a dot is what a filename has and an issuer token does not, and
#:   it is what separates ``pypi-local-operator.json`` and
#:   ``pypi-local-operator.json.<random>.tmp`` (ordinary cache filenames, both
#:   published while the dot counted toward a length floor) from a real tail;
#: * **≥12 characters** — long enough to keep round 1's own repro (``pk-`` plus
#:   16 letters) masked, which the digit-or-20-chars rule published.
#:
#: The third discriminator, an underscore-joined lowercase phrase, is checked by
#: :func:`_vendor_tail_guard` rather than by the charset: a look-behind cannot
#: express it (Python requires fixed width), and a guard runs only on a match.
#: The tail is what distinguishes a token from a name. Eight characters rather than
#: twelve: pre-existing callers of this pass treat `tvly-ABC123XYZ` (a 9-character
#: tail) as a credential, and a longer floor published it. The all-letter 8-19
#: escape window is closed by the GUARD instead — an underscore-joined lowercase
#: run is a name (`npm_config_update_notifier`) — and the lookahead keeps a
#: filename (`pypi-local-operator.json`) readable.
_VENDOR_TAIL = r"[A-Za-z0-9_+/=\-]{8,}(?![A-Za-z0-9.])"

_VENDOR_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(_VENDOR_SEPARATED_PREFIXES)
    + r")[-_]"
    + _VENDOR_TAIL
    + r"|\b(?:"
    + "|".join(_VENDOR_FIXED_PREFIXES)
    + r")"
    + _VENDOR_TAIL
)

#: Anchors DERIVED from that table — both separator spellings for every
#: separated prefix, the literal for the rest. Derivation is the point: an anchor
#: list maintained by hand beside a pattern is exactly how `pk-` went missing.
_VENDOR_ANCHORS: tuple[str, ...] = (
    tuple(
        f"{prefix}{separator}" for prefix in _VENDOR_SEPARATED_PREFIXES for separator in ("_", "-")
    )
    + _VENDOR_FIXED_PREFIXES
)


#: ``scheme://user:password@``, for the URL-named rule's value check.
_URL_USERINFO_PASSWORD = re.compile(
    r"://[^\s:/@\"']*:(?!\[redacted\])[^\s]*?@"
    r"(?=[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]+(?:[/:?#]|$))"
)

#: A credential-shaped query parameter, for the same check.
#: A userinfo with NO password but a key-shaped user part — Sentry's DSN
#: spelling (``https://<key>@o1.ingest.sentry.io/2``), where the public key IS
#: the credential. Hex/base64-ish and long, so an ordinary ``user@host`` URL is
#: not caught by it.
#: The same shape for a host with no dot (``mongodb://user:pw@host/db``,
#: ``postgresql://u:pw@h/db``, ``rediss://default:pass@h:6380``). It is a SECOND
#: rule rather than a relaxed lookahead on the first because the dotted form must
#: be tried FIRST: with a dotless lookahead the lazy class settles on the earliest
#: ``@``, which is right for a real host and wrong for a password containing one
#: (``svc:qA2S3n7x9@x/y,z"w@db.invalid/x`` masked to the first ``@`` with the
#: rest published). Ordered, the dotted rule takes every host that has a dot and
#: this one only sees what is left.
_DSN_PATTERN_PLAIN = re.compile(
    rf"(?i)\b((?:{_DSN_SCHEMES})://[^\s:/@\"']*:)(?!\[redacted\])"
    r"([^\s]*?)@(?=[A-Za-z0-9_.\-]+(?:[/:?#]|$))"
)

_URL_USERINFO_PASSWORD_PLAIN = re.compile(
    r"://[^\s:/@\"']*:(?!\[redacted\])[^\s]*?@(?=[A-Za-z0-9_.\-]+(?:[/:?#]|$))"
)

_URL_KEY_USERINFO = re.compile(r"://[A-Za-z0-9_.\-]{16,}@")

_URL_CRED_QUERY = re.compile(
    r"(?i)(?:^|[?&])(?:api[-_]?key|apikey|key|token|access[-_]?token|secret|password|"
    r"passwd|pwd|sig|signature)=(?!\[redacted\])"
)

#: Names whose VALUE decides whether the rule fires at all.
_URL_NAME_HINTS = ("uri", "url", "dsn", "endpoint", "conn")


def _HEADER_SCHEME_REPLACEMENT(match: Match[str]) -> str:
    """Keep the header/flag context and the scheme keyword; mask the value."""
    return match.group(0)[: match.start(3) - match.start(0)] + REDACTION_MARKER


#: Values that are CODE rather than credentials, whatever the name says.
#:
#: Reading and editing source is a first-class tool surface, so the cost of a
#: false positive is not cosmetic: a masked token inside a line the agent may
#: write back into a file is a correctness bug. Measured on this repository's own
#: ``local_operator/**`` before these rules: 1044 lines rewritten in 209 files,
#: 948 of them by this one rule (``key = key.strip()`` → ``key = [redacted]``,
#: ``COMPONENT_KEYS: tuple[str, ...] = (`` → ``COMPONENT_KEYS: [redacted]``).
_EXPRESSION_CHARS = "()[]{}"


#: Something a credential has and an ordinary word does not: a digit, or one of
#: the characters every token/blob/DSN is full of. ``-``/``_``/``:``/``.`` are
#: deliberately NOT in here — they are identifier and prose punctuation, and
#: admitting them put ``ctrl+pageup``, ``models-dev.listing`` and
#: ``PRESERVED_USER_TURN_KEY`` back on the masking side.
_CREDENTIALISH = re.compile(r"[0-9=@/,;]")

#: A bare URL, which is not a credential unless it carries one (see
#: :func:`_url_value_guard` for the same judgement on ``*_URL`` names).
_URL_LIKE = re.compile(r"(?i)^[a-z][a-z0-9+.\-]*://")

#: A dotted attribute path (``current.session_key``, ``models-dev.listing``).
_DOTTED_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+")


#: Qualifier words that make an otherwise ambiguous ``*_KEY``/``*_KEYS`` name a
#: CREDENTIAL name rather than a code one. ``PRIVATE_KEY`` is a secret;
#: ``env_keys``, ``class_key``, ``exclude_keys`` and ``TOKENS_OBTAINED_AT_KEY``
#: are variable names, and the value beside them is a reference to another
#: variable.
_KEY_QUALIFIERS = (
    "api",
    "private",
    "public",
    "signing",
    "encryption",
    "master",
    "access",
    "secret",
    "client",
    "auth",
    "session",
    "ssh",
    "rsa",
    "pgp",
    "gpg",
    "jwt",
    "webhook",
    "aws",
    "gcp",
    "azure",
    "openai",
    "anthropic",
    "github",
    "gitlab",
    "stripe",
    "slack",
    "npm",
    "pypi",
    "docker",
    "hugging",
)

#: Credential words that make a name strong on their own, whatever else it says.
_STRONG_CRED_WORDS = (
    "password",
    "passwd",
    "passphrase",
    "pwd",
    "secret",
    "credential",
    "credentials",
    "token",
    "tokens",
    "dsn",
)


def is_strong_credential_name(name: str) -> bool:
    """Whether a credential reading is the ONLY reading of this name.

    The split exists because the value proof that removed most of the
    ordinary-code rewrites (``key = key.strip()``, ``env_keys="OPENAI_API_KEY"``)
    also refused every credential VALUE with no digit in it — ``PASSWORD=
    swordfish``, ``token=abcdefghijklmnop``, ``AWS_SECRET_ACCESS_KEY=<26
    letters>`` — which is a leak, and the corpus is the specification for this
    control. Measured both ways: with the digit clause applied to every name the
    corpus's own positives escape; applied to none, the census returns to
    210 lines / 69 files from 28 / 13.

    So the proof is applied where the name is WEAK — ``key``, ``keys``, and
    composites whose qualifier is code-ish — and a strong name masks any opaque,
    non-expression, non-path value whatever it contains. The clauses that do not
    depend on digits (an expression, a quoted path, a bare filesystem path, a
    credential-less URL, a value that repeats its own name) apply to both.
    """
    segments = _name_segments(name)
    if not segments:
        return False
    lowered = name.lower()
    if any(word in lowered for word in _STRONG_CRED_WORDS):
        return True
    squashed = lowered.replace("_", "").replace("-", "")
    if squashed.endswith(("key", "keys", "keydata", "keystore")):
        # ``apikey``/``authToken``/``PRIVATE_KEY`` are credential names; ``envkeys``
        # and ``classkey`` are not, which is what the qualifier list decides.
        return any(qualifier in squashed for qualifier in _KEY_QUALIFIERS)
    return False


#: A PLACEHOLDER, not a credential. Masking ``$TOKEN`` hides the variable NAME the
#: model needs to fix the command, and registering it contains nothing. Detection is
#: still COUNTED for these (the notice is the signal for a bare value), so this is a
#: predicate over masking and registration alone.
_PLACEHOLDER_SHAPES = re.compile(
    r"^(?:"
    r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|"  # ${CI_JOB_TOKEN}, $GITLAB_TOKEN
    r"<[^<>]{1,40}>|"  # <password>
    r"%s|%\([a-z_]+\)s|"  # printf-style
    r"\{\{[^{}]{1,80}\}\}|"  # {{ … }}
    r"\[\[.*?\]\]|"
    r"(.)\1{2,}"  # ***, xxx, ...
    r")$"
)

_PLACEHOLDER_WORDS = frozenset(
    {
        "changeme",
        "change-me",
        "change_me",
        "example",
        "gitlab-ci-token",
        "nonemptystring",
        "notset",
        "password",
        "placeholder",
        "redacted",
        "replace-me",
        "secret",
        "token",
        "todo",
        "your-token-here",
    }
)


def is_placeholder_component(value: str) -> bool:
    """Whether a matched component is a PLACEHOLDER rather than a credential.

    ``${CI_JOB_TOKEN}``, ``$GITLAB_TOKEN``, ``<password>``, ``%s``, ``{{ … }}``,
    ``changeme``, ``gitlab-ci-token``, ``nonEmptyString``, a single repeated
    character. Two consequences, and they are different decisions:

    * it is never MASKED — the text stays readable, because the value IS the
      variable's name and the model needs it to fix the command;
    * it is never REGISTERED session-wide — there is nothing to contain.

    The DETECTION is still counted and still noticed: for a bare value the notice is
    the only signal there is, and a detector that went quiet on placeholders would
    also go quiet on a real credential spelled like one.
    """
    stripped = value.strip()
    if not stripped:
        return True
    if _PLACEHOLDER_SHAPES.match(stripped):
        return True
    return stripped.lower() in _PLACEHOLDER_WORDS


def _repeats_its_own_name(value: str, name: str) -> bool:
    """Whether the value is a REFERENCE to the name rather than a secret.

    ``"access_token": access_token``, ``exclude_keys=exclude_keys``,
    ``tokens=tokens_before``, ``TOKENS_OBTAINED_AT_KEY = "tokens_obtained_at"`` —
    the value is plumbing, and masking it mangles a line the agent may write back
    into a file. A credential never spells its own variable name.
    """
    value_norm = re.sub(r"[^a-z0-9]", "", value.lower())
    name_norm = re.sub(r"[^a-z0-9]", "", name.lower())
    if not value_norm or not name_norm:
        return False
    return value_norm in name_norm or name_norm in value_norm


def _value_is_not_a_credential(value: str, *, name: str, strong: bool) -> bool:
    """Whether a matched value must NOT be masked, and why.

    ``strong`` relaxes the DIGIT-shaped clauses only (see
    :func:`is_strong_credential_name`); everything that identifies an
    expression, a reference or a path applies to every name.

    These are the shapes ordinary code puts on the right of ``=``: a call
    (``_tokens(query)``), a subscript (``started["token"]``), a tuple/collection
    literal (``("a", "b")``), a private reference (``_node_order``), a dotted
    attribute path, a value that repeats the name it is assigned to. None of them
    is a credential, and masking one mangles the line the agent is reading.
    """
    if is_placeholder_component(value):
        return True
    if any(char in value for char in _EXPRESSION_CHARS):
        return True
    if value.startswith("_") and not strong:
        # A generated secret may START with ``_`` (base64url), so the leading
        # underscore is evidence of code only when the name is ambiguous.
        return True
    if value[0] in ".,;:'\"`" or value[-1] in ".,;:'\"`":
        return True
    # A bare filesystem path is not a credential. This is what keeps
    # ``PWD=/Users/example/project`` — and every ``env`` dump's cwd — readable:
    # ``pwd`` is a legitimate credential tail word (``pwd=…`` is how several
    # client CLIs spell one, and ``?pwd=`` how a URL does), so the exclusion
    # belongs on the VALUE rather than on the name.
    if value.startswith("/"):
        return True
    if _repeats_its_own_name(value, name):
        return True
    # A bare URL is not a credential. ``AUTH_CLAIM_KEY = "https://api.openai.com/
    # auth"`` is a claim NAME; the value proves itself the same way a ``*_URL``
    # name's does — userinfo password or credential-shaped query parameter.
    if _URL_LIKE.match(value):
        return not (
            _URL_USERINFO_PASSWORD.search(value)
            or _URL_USERINFO_PASSWORD_PLAIN.search(value)
            or _URL_CRED_QUERY.search(value)
            or _URL_KEY_USERINFO.search(value)
        )
    # A dotted attribute path is a REFERENCE to a credential, not one:
    # ``key="providers.anthropic.cache_ttl_1h_min_context_tokens"``. Digits in a
    # path are ordinary (``cache_ttl_1h``), so this clause does not depend on
    # them; a JWT — the one credential that looks dotted — is caught by its own
    # bare-token rule rather than by a name-driven one.
    if _DOTTED_PATH.fullmatch(value):
        return True
    # A lowercase hyphenated word-phrase under a CODE-ish name is a NAME, not a
    # secret (``_PUBLIC_LISTING_TOKEN = "public-catalogue-read"``). The
    # underscore-led spelling is the discriminator: ``DB_PASSWORD=correct-horse-battery``
    # is a credential and is in the original corpus, and so is
    # ``SECRET_KEY=django-insecure-…``.
    if name.startswith("_") and re.fullmatch(r"[a-z]+(?:-[a-z]+)+", value):
        return True
    # A CLASS or TYPE name is a reference, whatever the name beside it says:
    # ``refresh_token: RefreshFn | None = None``, ``_credentials:
    # CredentialManager``, ``reasoning_tokens: SafeCount``, ``refresh_token:
    # SecretStr``. CamelCase and single-capital identifiers are how a TYPE is
    # spelled; a credential value is lowercase or random, and the mixed-case
    # secrets that do exist carry a symbol (``wJalrXUtnFEMI/K7MDENG``).
    if re.fullmatch(r"[A-Z][A-Za-z0-9]*|[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*", value) and not any(
        char.isdigit() for char in value
    ):
        # ...and no digit: ``FwoGZXIvYXdzEBYaDExampleTokenValue1234567890`` is a
        # real AWS session token and is in the original corpus, while every type
        # name in this tree is digit-free.
        return True
    # A bare ``_``-led identifier with no digit is a reference, even under a strong
    # name: ``get_api_key=_oauth_api_key``.
    if (
        strong
        and value.startswith("_")
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
        and not any(char.isdigit() for char in value)
    ):
        return True
    if strong:
        return False
    # --- the digit-shaped clauses, for ambiguous names only ------------------
    # A WORD: ``never one shared token: revocation`` in a docstring, ``part of
    # the key: ``max_recommendations`` in a comment.
    if not _CREDENTIALISH.search(value) and len(value) < 20:
        return True
    # ``key: raise/replace`` — two words, and the only signal is the slash.
    if value.count("/") and not any(char.isdigit() for char in value) and len(value) < 16:
        return True
    # A bare IDENTIFIER with no digit is a NAME, not a token:
    # ``env_keys="OPENAI_API_KEY"``, ``key="model_name"``.
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) and not any(char.isdigit() for char in value):
        return True
    return False


def _is_keyword_argument(match: Match[str]) -> bool:
    """Whether ``NAME=`` here is a call's keyword argument rather than an env var.

    ``sorted(roots, key=_node_order)`` is not an assignment to a credential
    named ``key``; it is one argument of a call. The signal is the nearest
    non-space character before the name: ``(`` or ``,`` opens an argument list
    that this name is inside.
    """
    # ``match.string`` is the text the pattern was searched in — the whole tool
    # result or line — which is what the look-back has to walk.
    text = match.string
    index = match.start(1) - 1
    while index >= 0 and text[index] in " \t":
        index -= 1
    if index < 0 or text[index] not in "(,":
        return False
    # ...and the ``(``/``,`` has to be INSIDE an open call. ``host=db,password=pw``
    # and ``git commit,password=pw`` are comma-separated data: the comma is at
    # paren depth 0, so the name is an assignment and the value is masked. The
    # call case (``run(password=…)``) is at depth ≥ 1.
    before = text[: index + 1]
    return before.count("(") > before.count(")")


def _base64_value_guard(match: Match[str]) -> bool:
    """Reject a bare ``Basic <word>``: base64 has case, padding or a symbol."""
    value = match.group(2)
    if any(char in value for char in "+/="):
        return True
    return any(char.isupper() for char in value) and any(char.islower() for char in value)


def _vendor_tail_guard(match: Match[str]) -> bool:
    """Reject an issuer-looking prefix followed by an ordinary NAME.

    ``npm_config_update_notifier`` and ``docker_compose_build_args`` are
    environment variables: lowercase words joined by underscores. A real issuer
    tail is one unbroken run (``npm_<base64>``, ``docker_pat_…``). The rule's
    pattern cannot express that without a variable-width look-behind, and a guard
    costs nothing on text the gate has already skipped.
    """
    tail = match.group(0)
    for prefix in _VENDOR_SEPARATED_PREFIXES:
        if tail.lower().startswith(prefix.lower()):
            tail = tail[len(prefix) :].lstrip("_-")
            break
    else:
        for prefix in _VENDOR_FIXED_PREFIXES:
            if tail.lower().startswith(prefix.lower()):
                tail = tail[len(prefix) :]
                break
    return not re.fullmatch(r"[a-z]+(?:_[a-z]+)+", tail)


#: An ENVIRONMENT-VARIABLE (or secret-store NAME) spelling: capitals, digits and
#: underscores, no lower case. ``OS_PROD2_ADMIN_PASSWORD``, ``API_KEY``.
_ENV_NAME_SHAPED = re.compile(r"[A-Z][A-Z0-9_]*")


def _flag_value_guard(match: Match[str]) -> bool:
    """A ``--flag VALUE`` pair, unless the value is syntax or a NAME.

    The syntax half is the original rule: a value carrying brackets or parentheses
    is a usage-string placeholder, not a secret.

    **The NAME half is measured, not hypothetical.** ``lop secret run --secret
    OS_PROD2_ADMIN_PASSWORD -- <command>`` is the documented way to hand a stored
    secret to a child, and the token after that flag is the secret's NAME in the
    store — the one thing the operator needs to be able to read. On 2026-09-19 a
    watch-log entry that QUOTED that command set this rule off, which filed a
    rotation ticket in a production transcript for a credential that was not in
    the text at all (the second firing of this class). A value that is spelled as
    an environment variable AND ends in a credential word is a reference to a
    credential, never one: the same judgement the rule already makes for
    ``--secret NAME[=VAR]``.

    **The SEPARATOR is required, and that is the whole of the narrowing.** Capitals
    plus a credential-word tail is not enough on its own: it also describes exactly
    the values this rule exists to catch — ``--password PASSWORD``, ``--token
    TOKEN``, ``--api-key APIKEY``, ``--secret DBPASSWORD`` — and a first cut that
    omitted the underscore stopped masking all four (agent review R1, reproduced
    through the session hook: the values came back byte-identical with no mask and
    no notice at all). Multi-segment capitals is what a NAME in a store or an
    environment looks like; a single run of capitals is a credential someone chose,
    and it stays masked. The residual is a real value spelled ``PROD_SECRET``-style,
    and it is accepted deliberately — that spelling IS the shape of a stored
    secret's name, which is the same judgement the rule already made for
    ``--secret NAME[=VAR]``.

    A value that could BE a credential — mixed case, any lower case, a single
    unseparated token, or any token without a credential-word tail — is still masked.
    """
    value = match.group(2)
    if any(char in value for char in _EXPRESSION_CHARS):
        return False
    if "_" in value and _ENV_NAME_SHAPED.fullmatch(value) and is_credential_name(value):
        return False
    return True


def _BARE_SCHEME_REPLACEMENT(match: Match[str]) -> str:
    """Keep the ``bearer `` keyword; mask the value."""
    return match.group(0)[: match.start(2) - match.start(0)] + REDACTION_MARKER


def _assignment_guard(match: Match[str]) -> bool:
    """The name half of the named-assignment rule, checked on the match.

    The marker check is the "do not mask twice" half: rules run in order, so a
    DSN inside ``MONGO_DSN=`` is masked by the DSN rule first — password gone,
    user and host kept readable — and this rule would otherwise match the same
    value again and swallow the whole thing, taking back the host it had just
    preserved and splitting the marker on the ``]`` the value class stops at.
    Measured while building the table: the unguarded pair produced
    ``MONGO_DSN=[redacted]]@host``.
    """
    if REDACTION_MARKER in match.group(4):
        return False
    name = match.group(1)
    if not is_credential_name(name) or is_count_shaped(name):
        return False
    # The value has to look like a credential. Everything below this line is a
    # false positive that rewrites ordinary code (see
    # ``_looks_like_an_expression`` for the census that forced it).
    if _value_is_not_a_credential(
        match.group(4), name=name, strong=is_strong_credential_name(name)
    ):
        return False
    return not _is_keyword_argument(match)


def _url_value_guard(match: Match[str]) -> bool:
    """A ``*_URL``/``*_URI``/``*_DSN`` name whose VALUE carries a credential.

    See :data:`CREDENTIAL_SHAPES` for why the value has to prove itself: masking
    every endpoint URL would blind the agent to ordinary output.
    """
    name = match.group(1).lower()
    if not any(hint in name for hint in _URL_NAME_HINTS):
        return False
    value = match.group(4)
    return bool(
        _URL_USERINFO_PASSWORD.search(value)
        or _URL_USERINFO_PASSWORD_PLAIN.search(value)
        or _URL_CRED_QUERY.search(value)
        or _URL_KEY_USERINFO.search(value)
    )


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
#: Character classes for a credential VALUE. A value runs to its real end and a
#: quote INSIDE it is part of it: excluding the quote characters from a class is
#: how three separate rules came to mask a prefix and publish the tail under a
#: notice that claimed the whole credential was masked. Use these (with a floor
#: lookahead) for every opaque token; keep a wrapping quote out with a lookahead
#: where the spelling allows one.
_MIXED_CLASS = r"[A-Za-z0-9._~+/=-]"
_B64_CLASS = r"[A-Za-z0-9+/=]"


def _tolerant(token_class: str) -> str:
    """A token-shaped value that may contain quotes."""
    return token_class + r"+(?:\x27\x22" + token_class + r"+)*"


#: The PEM header phrase, as a compiled test rather than a substring: `"BEGIN"` alone
#: matches prose (`"private_key": "BEGINNER guide"`) and the file's own marker words.
_PEM_HEADER_PHRASE = re.compile(r"-{1,4}[\x27\x22]?-{1,4}BEGIN [A-Z0-9 ]*PRIVATE KEY")


#: The line-number prefix tools actually emit, as ONE definition shared by the shape
#: table and the pipe's own body classifier (``tools/builtin.py`` imports these). A second
#: hand-written allowance is what published the body for `cat -n` output after the shape
#: rule had been fixed (Q10-F1): the pipe masks BEFORE the table ever runs, so a divergence
#: between the two is a silent leak rather than a missed match.
#:
#: Covered: `12|`, `12:`, `12>`, `12->`, `12)`, `12.]`, `[12]`, `12<TAB>` (cat -n), bat's
#: `│ 12 │`, any of them repeated (`3| 4| …`), with spaces or a TAB around the separator.
LINE_PREFIX = (
    r"[ \t]*(?:(?:"
    r"\[?\d+\]?[ \t]*(?:[.)\]]|\.\]|->|[|:>-])?[ \t]*"
    r"|\u2502[ \t]*\d+[ \t]*\u2502[ \t]*"
    r")+)?"
)

#: A line separator in either spelling: escaped (inside a JSON value) or real, CRLF
#: included.
LINE_SEP = r"(?:\\r\\n|\\n|\r\n|\n|\r)"

#: One PEM body line with that prefix. Eight characters is the floor for a line that
#: stands on its own; a SHORTER line counts only when a full one follows it (a truncated
#: run) or when it is the block's last line before the closing quote or the end of the
#: text — inside an open block nothing may be published, which is the block's whole point
#: (Q10-F2: a sub-eight-character line in the MIDDLE published everything after it, and a
#: short FINAL line published where the previous head masked).
_PEM_FULL_LINE = r"[A-Za-z0-9+/=]{8,},?[ \t]*"
_PEM_SHORT_MID_LINE = (
    r"[A-Za-z0-9+/=]{1,7},?[ \t]*(?=" + LINE_SEP + LINE_PREFIX + r"[A-Za-z0-9+/=]{8,})"
)
#: A short line is a body line when a full one FOLLOWS it, and the run may end with one
#: short line. A lone short line — `12| done`, `12| 42` — is numbered PROSE and must
#: survive, which is why the end-of-run allowance is not a free-standing alternative.
_PEM_LINE_CONTENT = _PEM_FULL_LINE + r"|" + _PEM_SHORT_MID_LINE
_PEM_RUN = (
    r"(?:" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")"
    r"|" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")?)*"
    r"(?:" + LINE_PREFIX + r"[A-Za-z0-9+/=]{1,7},?)?"
)
PEM_BODY_LINE_RE = re.compile(r"^" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")$", re.MULTILINE)
PEM_HEADER_LINE_RE = re.compile(
    r"^" + LINE_PREFIX + r"-{1,4}[\x27\x22]?-{1,4}BEGIN [A-Z0-9 ]*PRIVATE KEY"
    r"-{1,4}[\x27\x22]?-{1,4}$",
    # MULTILINE, and that is not cosmetic: the pipe layer SEARCHES a multi-line read for
    # this header, so without the flag it matched only when the read was exactly one
    # header line — which the release point's hold makes impossible — and the entire
    # body-masking path was unreachable (R11-1: `head -n 6 key.pem` published 5 of 25
    # body lines through the real tool call while the unit suite stayed green).
    re.MULTILINE,
)
PEM_END_LINE_RE = re.compile(r"^" + LINE_PREFIX + r"-{1,4}[\x27\x22]?-{1,4}END ", re.MULTILINE)


def _anchored_key_value_guard(match: Match[str]) -> bool:
    """Only a value that CARRIES a PEM header phrase is a key, not every `private_key`."""
    return _PEM_HEADER_PHRASE.search(match.group(2).upper()) is not None


CREDENTIAL_SHAPES: tuple[Shape, ...] = (
    # --- key material, before anything that could take it apart ---------------
    Shape(
        "pem-private-key",
        re.compile(
            r"-{1,4}[\x27\x22]?-{1,4}BEGIN (?:[A-Z0-9 ]*?)PRIVATE KEY-{1,4}[\x27\x22]?-{1,4}"
            r"[\s\S]*?-----END (?:[A-Z0-9 ]*?)PRIVATE KEY-----"
        ),
        REDACTION_MARKER,
    ),
    # ``"private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIE…"`` — the GCP
    # service-account form. The escaped ``\n`` sequences sit inside the JSON
    # string, so the PEM rule above spans them too and this rule only has to
    # catch the header when the body was truncated before an END line arrived.
    Shape(
        "gcp-service-account-value",
        # The ANCHORED spelling — `"private_key": "-----BEGIN …` — is masked to the
        # VALUE's closing quote, which is a real, bounded delimiter, instead of to a
        # line boundary. The line-bounded iteration is what round 6 broke: a body line
        # whose padding is followed by more base64 (`…, note`) or by an ANSI reset
        # failed the first iteration, the alternation collapsed to the header, and the
        # ENTIRE body was published with nothing flagged complete (M6-1) — silently,
        # because a withheld claim means no notice either. The two obvious remedies are
        # wrong and measurably so: `[^\r\n]*` reopens B4-1's eaten anchor, and a
        # `{16,}` run floor leaks a short truncated final line. Masking to the closing
        # quote cannot run away (the quote bounds it) and cannot eat a neighbouring
        # line's anchor (it never crosses the value).
        #
        # The guard is what keeps it narrow: the value must actually carry a BEGIN
        # marker, so an ordinary `"private_key": "projects/x/keys/k1"` stays readable.
        # An unterminated string masks to the end of the input and its claim is
        # withheld by ``_is_truncated_pem`` (no END marker to find).
        # The CLOSING QUOTE is required here, and that is load-bearing: without it the
        # value group ran to the end of the input and swallowed whatever followed —
        # `…<BODY>\nNORMAL, more text\n` lost that line (M6-2, returned as an over-mask).
        # The unquoted/truncated spelling is the next rule's job, and it stops at the
        # first non-credential line.
        re.compile(r'(?i)("?private[_-]?key"?\s*:\s*")([^"]*)"'),
        None,
        2,
        guard=_anchored_key_value_guard,
    ),
    Shape(
        "gcp-service-account-value-open",
        # The anchored spelling with NO closing quote — a truncated JSON string, how a
        # tool prints a value it cut off. Masking "the remainder" was wrong and the
        # manager's round-7 direction said so: the remainder is not all credential, and
        # `…<BODY>\nNORMAL, more text\n` lost that whole line (M6-2 returned as an
        # over-mask). This rule masks the maximal run of CREDENTIAL-SHAPED lines and
        # stops before the first line that is not one. As the pattern writes it — not as
        # "stripped", which it does not do — a line is credential-shaped when it is an
        # optional `N| ` line-number prefix, then optional spaces/tabs, then either
        # nothing, a PEM header/footer, or solely `[A-Za-z0-9+/=]` with at most one
        # trailing comma, then optional trailing spaces/tabs. The prefix and the padding
        # are load-bearing: without them a body line that is indented, tab-prefixed, or
        # rendered by this product's own `read` tool as `2| MIIE…` ended the run and every
        # line after it was published silently (R8-1 / QA's N7-1). `NORMAL, more text`,
        # `", "other": "value"}` and prose are not credential-shaped and survive
        # byte-identical.
        #
        # KNOWN EDGE, recorded rather than patched: a line whose content is a single
        # bare word, or shorter than eight characters, is NOT masked — measured on
        # `done`, `INFO` (both stay readable) and `NORMAL,` (readable; the comma is
        # stripped by the class and the six remaining characters are under the floor).
        # A length floor is what keeps numbered PROSE alive (`12| done`, `12| 42`), and
        # the price is measured in both directions: a run that has not started cannot be
        # begun by a short line (prose survives), and TWO consecutive short lines publish
        # the second one. Both are stated here because the earlier wording claimed only
        # the over-mask direction, which is not what the measurements show.
        #
        # The prefix spelling above is general on purpose (see the block comment); the
        # trailing `[ \t]*` before it is part of that spelling.
        #
        # RECORDED, with the SCOPE QA measured — wider than "the final line": a
        # sub-floor line publishes on the plain bash pipe path whether it is LAST or in
        # the MIDDLE, up to seven base64 characters per event (<=5 bytes of a ~1.7 KB
        # key: real key bytes, not usable material), a `head -c` cut publishes a
        # 5-character residue, and the total is unbounded across lines (60 characters
        # over ten lines measured). A SHORT FINAL line
        # (`…<BODY>\nMIIEo`, and the second of two consecutive short lines) still
        # publishes. It cannot be distinguished from numbered PROSE — `…<BODY>\n12| done`
        # is the same shape, and the prose case is a requirement (a log line must survive),
        # so the two demands are mutually exclusive for a word-shaped final line. What IS
        # implemented is the shape QA measured: a short line in the MIDDLE, with a full
        # line after it, masks. Reported as `deferred — short final line, indistinguishable
        # from numbered prose, recorded on the unterminated-block follow-up`.
        # `INFO` and `NORMAL,` all stay READABLE — they are under the floor and start no
        # run. (An earlier comment said the opposite; the measurement is the source.)
        #
        # Not "stop at the first newline": that would publish the rest of a key whose
        # body is bare newline-separated base64, which is every real PEM.
        re.compile(
            r'(?i)("?private[_-]?key"?\s*:\s*")'
            r"(-{1,4}[\x27\x22]?-{1,4}BEGIN [A-Z0-9 ]*PRIVATE KEY-{1,4}[\x27\x22]?-{1,4}"
            r"(?:"
            r"(?:\\r\\n|\\n|\r\n|\n|\r)"
            # A body line, from the SHARED grammar (`LINE_PREFIX` / `_PEM_LINE_CONTENT`)
            # so the shape table and the pipe's classifier can never drift apart. The
            # prefix is optional and repeated because every way a tool numbers a line has
            # to be covered — `read` of a `cat -n` file writes `number<TAB>`, `grep -n`
            # writes `number:`, bat writes `│ 12 │`, and a file already numbered doubles
            # it. A one-spelling allowance published the whole body for the rest
            # (Q9-F1, then Q10-F1 in the pipe layer).
            # At least ONE full (or short-followed-by-full) line, then an optional
            # short line to close the block. Requiring the first line is what keeps a
            # lone short line — `12| done`, `12| 42` — readable: numbered PROSE has no
            # full body line before it, so the run never starts.
            # The BODY RUN, as one expression: one or more body lines, each behind its
            # separator, then an optional SHORT closing line — also behind a separator,
            # which is what the previous spelling omitted, leaving that allowance unable
            # to fire and a short FINAL line published (R11-3).
            # The outer group already supplies the separator for each repeated line; the
            # optional SHORT closing line carries its own, which is what the previous
            # spelling omitted, leaving that allowance unable to fire (R11-3).
            # The FIRST line must be a full one: that is what keeps numbered PROSE
            # (`12| done`, `12| 42`) readable — prose has no full body line before it, so
            # the run never starts — while a run that HAS started may end with a short
            # line (R11-3: the allowance had no separator and could never fire).
            # Each element may be a full line or a short one WITH A FULL LINE AFTER IT —
            # the SHORT-MID shape QA measured (`3| Qw9z` then `4| MIIE…`) — which also
            # keeps numbered prose safe: a prose line has no full body line after it, so
            # it can neither start nor continue the run.
            r"(?:" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")[ \t]*)+"
            r"(?=" + LINE_SEP + r"|[\x27\x22]|$)"
            r")*)"
        ),
        None,
        2,
        guard=_anchored_key_value_guard,
    ),
    Shape(
        "gcp-service-account-key",
        # The masked group is the WHOLE BLOCK, not the BEGIN marker: masking the
        # marker alone published the entire key body while the hit still filed as
        # complete, so a "credential was masked, rotate it" row was queued over a
        # live private key (the reviewer recovered it with `openssl pkey`). The body
        # is base64 lines, each preceded by an escaped or a real newline, optionally
        # closed by the END marker — bounded, because a `\s`-run without a bound
        # would walk to the end of the document.
        re.compile(
            r"(?i)(\"?private[_-]?key\"?\s*:\s*\"?)"
            r"(-{1,4}[\x27\x22]?-{1,4}BEGIN [A-Z0-9 ]*PRIVATE KEY"
            r"-{1,4}[\x27\x22]?-{1,4}"
            r"(?:"
            # A body LINE, and it must reach its own line end: the run is required
            # to be followed by a separator (escaped or real, CRLF included), a
            # closing quote or the end of the text. Without that the loop consumed
            # the leading run of the NEXT line and stopped mid-line, eating the
            # anchor a following rule needed — `password = hunter2hunter2` after a
            # block kept 6 of 7 surfaces readable, `ghp_…` on the next line lost its
            # prefix, and unrelated text came back truncated (`deployment` →
            # `-finished`). The match is bounded by the BLOCK's own extent rather
            # than by a fixed line count — a real key body is ~100 lines, and a
            # cap would release key material beyond it, which is the one thing this
            # rule exists to prevent. The pass is linear in the input: 7.9 s for a
            # 4.44 MB unterminated body, with the whole block masked.
            r"(?:\\r\\n|\\n|\r\n|\n|\r)"
            # The terminator may sit after NON-BASE64 padding (`,`, a trailing
            # space) — `MIIE…,\n` is a body line with a comma on it, and requiring
            # the base64 run to touch the terminator made the whole block unmasked
            # where the previous head masked it (R5-1). What must NOT happen is the
            # run continuing into the next line's base64, and `[^A-Za-z0-9+/=]*`
            # stops exactly there — which is what keeps B4-1 closed.
            r"(?:[A-Za-z0-9+/=]{2,}(?=[^A-Za-z0-9+/=]*(?:\\r\\n|\\n|\r\n|\n|\r|[\x27\x22]|$))"
            r"|-{1,4}[\x27\x22]?-{1,4}END [A-Z0-9 ]*PRIVATE KEY-{1,4}[\x27\x22]?-{1,4})"
            r")*)"
        ),
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
        lambda m: m.group(1) + REDACTION_MARKER + "@",
        2,
    ),
    Shape(
        "dsn-password-plain",
        _DSN_PATTERN_PLAIN,
        lambda m: m.group(1) + REDACTION_MARKER + "@",
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
            r"(?<![A-Za-z0-9_.\-])([A-Za-z0-9_.\-]{2,48})"
            r"([\"']?\s*[:=]\s*)([\"']?)"
            rf"({_ASSIGNED_VALUE})"
        ),
        None,  # rendered by the guarded path; the guard owns the name and value checks
        4,
        guard=lambda match: _url_value_guard(match),
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
            r"(?<![A-Za-z0-9_.\-])([A-Za-z0-9_.\-]{2,48})"
            # ``[\"']?`` before the separator so a QUOTED name is one of these:
            # ``"api_key": "…"`` is how JSON spells every one of them, and a
            # pattern that only accepts the bare ``name: value`` form misses the
            # whole shape on the most common surface there is.
            r"([\"']?\s*[:=]\s*)([\"']?)"
            rf"{_ASSIGNED_VALUE_GROUP}"
        ),
        # Keep the name, the separator and the opening quote; mask only the
        # value. Rendered by the guarded path, which knows the value group.
        None,
        4,
        guard=_assignment_guard,
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
        # The HEADER, not the word: a bare ``Bearer`` matched ordinary English
        # ("# Bearer serves both API keys and OAuth tokens on this wire"), and
        # because a match registers its value for the session that false hit then
        # masked the word in every later result and queued a rotation incident
        # for a credential that does not exist. The scheme keyword must be the
        # start of an ``Authorization``/``Proxy-Authorization`` header value, or
        # be preceded by a ``-H``/`` --header`` style argument.
        re.compile(
            # The header NAME, in every spelling a header reaches a transcript
            # in: the wire form (``Authorization:``), the quoted JSON/Python-repr
            # form (``{"Authorization": "Bearer …"}``), the dict form
            # (``'authorization': 'bearer …'``), the assignment form
            # (``authorization="Bearer …"``), the env-var form
            # (``HTTP_AUTHORIZATION="…"``) and a HAR dump, where the name and the
            # value are separate JSON keys (``{"name": "Authorization", "value":
            # "Bearer …"}``). Round 1's fix required the literal wire form and
            # published the opaque token in all of the others — a JSON log line or
            # a HAR file is ordinary tool output, and 71ebb68e handled it.
            r"(?i)((?:(?:proxy-|http_)?authorization)[\"']?\s*"
            r"(?:,\s*[\"']value[\"']\s*:|[:=]\s*)\s*[\"']?\s*"
            r"|(?:-H|--header)\s+\S{0,2})"
            r"(?i:(bearer)\s+)(?=" + _MIXED_CLASS + r"{8})(" + _tolerant(_MIXED_CLASS) + r")"
        ),
        _HEADER_SCHEME_REPLACEMENT,
        3,
    ),
    Shape(
        "authorization-basic",
        # Same treatment, and the value floor is a base64 blob's, not a word's:
        # ``Basic authentication is required`` was masked and the word
        # ``authentication`` registered as a session credential.
        re.compile(
            r"(?i)((?:(?:proxy-|http_)?authorization)[\"']?\s*"
            r"(?:,\s*[\"']value[\"']\s*:|[:=]\s*)\s*[\"']?\s*"
            r"|(?:-H|--header)\s+\S{0,2})"
            r"(?i:(basic)\s+)(?=" + _B64_CLASS + r"{8})(" + _tolerant(_B64_CLASS) + r")"
        ),
        _HEADER_SCHEME_REPLACEMENT,
        3,
    ),
    Shape(
        "authorization-bearer-bare",
        # The bare keyword with a TOKEN-SHAPED value and no header name — an
        # original corpus case. The floor (16) is what keeps it off prose:
        # ``# Bearer serves both API keys and OAuth access tokens on this wire``
        # has a six-letter English word after the keyword, and a real bearer
        # credential is a long opaque run.
        re.compile(
            r"(?i)\b(bearer\s+)(?=" + _MIXED_CLASS + r"{16})(" + _tolerant(_MIXED_CLASS) + r")"
        ),
        _BARE_SCHEME_REPLACEMENT,
        2,
    ),
    Shape(
        "authorization-basic-bare",
        # The bare keyword with a BASE64-shaped value and no header name, which is
        # how a scrubbed log or a `curl -v` blob often shows it
        # (``Basic dXNlcjpwYXNz`` is ``admin:pw``). The guard is what keeps the
        # prose safe: ``Basic authentication`` is a word, and base64 carries case
        # or an explicit ``+``/``/``/``=``.
        re.compile(
            r"(?i)\b(basic\s+)((?=[A-Za-z0-9+/=]{8})[A-Za-z0-9+/=]+(?:\x27\x22[A-Za-z0-9+/=]+)*)"
        ),
        None,
        2,
        guard=_base64_value_guard,
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
        # ``--secret NAME[=VAR]`` in a usage string is a PLACEHOLDER: a value
        # carrying brackets or parentheses is syntax, not a secret, and masking
        # it rewrites the help text the agent is reading. The pattern stays
        # simple and the guard does the judging, which keeps the gate's anchors
        # easy to keep honest.
        #
        # The guard's NAME clause extends that judgement to the spelling real
        # commands use — ``--secret OS_PROD2_ADMIN_PASSWORD``, where the token
        # after the flag is a reference to a stored secret. See
        # ``_flag_value_guard`` for the production incident that measured it.
        re.compile(
            r"(?i)(--(?:password|passwd|pwd|token|api[-_]?key|apikey|secret|"
            r"client[-_]?secret|auth[-_]?token|access[-_]?token)(?:=|\s+))([^\s\"']{3,})"
        ),
        None,
        2,
        guard=_flag_value_guard,
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
        re.compile(
            # A quote INSIDE the token does not end it (round 3's partial-mask class).
            r"(?i)(_auth(?:token)?=)(?=[^\s]{6})([^\s]+(?:[\x27\x22][^\s]+)*)"
        ),
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
        # The password runs to the END OF THE ARGUMENT (whitespace or end), so a
        # quote inside it is consumed rather than ending the match. The old
        # `[^'"\s]+` paused at the first quote and published the rest of the
        # password in the same line while the hit was STILL RECORDED — a masked
        # prefix beside a readable tail, under a notice saying it was masked. The
        # trailing `(?=['"]?(?:\s|$))` keeps a WRAPPING quote out of the match.
        re.compile(
            r"(?i)(\bcurl\b[^\n]{0,200}?\s-u\s+['\"]?[^:'\"\s]+:)" r"([^\s]*?)(?=['\"]?(?:\s|$))"
        ),
        r"\1" + REDACTION_MARKER,
        2,
    ),
    # ``openssl … -passin pass:<password>`` / ``-passout``. The flag IS the
    # context; a bare ``pass:`` would be prose.
    Shape(
        "openssl-pass-phrase",
        # The FLAG, with its argument: ``-passw``/``-passphrase``/``-passin`` and
        # friends are how openssl takes a passphrase, and a bare ``-pass``
        # followed by any word matched prose ("the post-pass occupancy").
        # Every spelling ``openssl`` takes, including the bare ``-pass val`` that
        # ``openssl enc --help`` documents. Prose protection is the LOOK-BEHIND,
        # not a narrower alternation: ``post-pass occupancy`` and ``pre-pass
        # usage`` are hyphenated nouns, where ``-pass`` follows a word character.
        re.compile(r"(?i)((?<![\w-])-pass(?:in|out|phrase|wd|wdin|wdout)?\s+)(?:pass:)?(\S{4,})"),
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
        # Prefixes and their anchors come from ONE table (`_VENDOR_TOKENS`), which
        # is a correctness requirement rather than tidiness: the anchors gate this
        # pass, so a prefix spelled with a separator the anchor list does not
        # carry is a token this rule can match and the gate will skip. That was a
        # real defect — `pk-`, `rk-`, `hf-` and `npm-` were published verbatim
        # while the rule itself masked them, and the corpus had no `-` variant to
        # notice. The suffix must also look like a token: at least 8 characters
        # AND at least one digit, which is what keeps `pypi-local-operator.json`
        # (a filename, 159 such lines in this repo) readable.
        _VENDOR_PATTERN,
        None,
        0,
        guard=_vendor_tail_guard,
    ),
    Shape(
        "github-token",
        re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}\b"),
        REDACTION_MARKER,
    ),
    Shape(
        "github-pat",
        re.compile(
            # A quote injected anywhere in the prefix must not orphan it: the whole
            # `github_pat_…` run is the credential, prefix included.
            r"\bgithub[_\x27\x22]?p[_\x27\x22]?at[_\x27\x22]?"
            r"[A-Za-z0-9_]{19,}(?:[\x27\x22][A-Za-z0-9_]+)*\b"
        ),
        REDACTION_MARKER,
    ),
    Shape(
        "aws-access-key-id",
        re.compile(r"\bA(?:[\x27\x22])?(?:KIA|SIA)[0-9A-Z]+(?:[\x27\x22][0-9A-Z]+)*\b"),
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


#: The shared floor for a component worth scrubbing or registering. Derived from the
#: masking floor above so ``mcp/redaction.MIN_SCRUBBED_LENGTH`` and this module
#: cannot drift apart (round 2, §4 of the handover: one floor, two users).
DETECTED_COMPONENT_FLOOR = _MIN_REGISTERABLE_SECRET


def is_registerable_component(value: str) -> bool:
    """Whether a matched value may become a session-wide redaction.

    Registration is a PROMOTION: the value is masked in every later result for
    the rest of the session, so a false positive here outlives the line that
    caused it (``Basic authentication`` was registered as a credential once, and
    the word was then masked in every subsequent result). The floor is well
    floor is the masking floor and the discriminator is the SHAPE: a value that
    is a plain word is never registered, whatever its length.
    """
    # Length only. The word-shape refusal was added to stop ``Basic
    # authentication`` poisoning a session, and the header rules now need their
    # context to fire at all; keeping it refused the REGISTRATION of a
    # word-shaped credential a real rule had masked (`--password swordfish`,
    # `-pswordfish`, a DSN whose password is a word) — masked in place, then free
    # to reappear in the next result.
    if is_placeholder_component(value):
        return False
    return len(value) >= DETECTED_COMPONENT_FLOOR


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
    #: The text the rule MATCHED. For a DSN that is the whole ``scheme://…``
    #: authority, which is what the completeness check has to prove gone — the
    #: value alone (the password group) is a fragment of the credential, and a
    #: partial mask leaves the rest of the authority behind it.
    window: str = ""
    #: Whether the ENTIRE credential is gone from the scrubbed text. A hit with
    #: ``complete=False`` still registers (containment of what can be contained)
    #: but may not be announced AS CONTAINED: the containment notice tells the
    #: operator the value was masked, and that claim has to be true.
    complete: bool = True
    #: Whether READABLE credential material survived the mask in this text.
    #:
    #: This is the whole severity classification, and it is deliberately a
    #: separate fact from ``complete``: ``complete`` says whether the mask may be
    #: CLAIMED (a truncated PEM is fully masked and still unclaimable, because
    #: nothing proves the rest of the key is not further down), while ``exposed``
    #: says whether anything readable is left in the text the model is about to
    #: read. Only ``exposed`` is a compromise: a value in the model's context may
    #: be in training data, which is the one exposure this harness cannot undo, so
    #: it is the one that asks the operator for a rotation. Everything else the
    #: table catches — masked whole in a command's `argv`, in a tool result, in a
    #: file on disk — is contained before the model sees it.
    exposed: bool = False


@dataclass(frozen=True)
class ShapeReport:
    """One run of the table over one piece of text: what it contained, and what escaped.

    The pair every consumer needs, and the reason it is one object rather than two
    return values: the two facts have to travel together, because the notice's
    SEVERITY (see :attr:`ShapeHit.exposed`) is decided by the second while its
    wording is decided by the first, and a caller that reads one without the other
    either loses a real compromise or announces a containment it cannot prove.

    ``labels`` are shape NAMES, never values — a report that carried the credential
    would be the leak it exists to describe. Only hits whose mask may be CLAIMED
    appear in it, so it is exactly the set of shapes the contained notice may name.

    ``reached_model`` is true when any hit left readable credential material in the
    text the model reads. A text can produce both — one rule masks a DSN whole while
    another leaves a fragment of a different credential behind — and the louder fact
    wins, which is why this is a single boolean on the pair rather than a per-label
    flag.
    """

    labels: tuple[str, ...] = ()
    reached_model: bool = False


def shape_report(hits: Sequence[ShapeHit]) -> ShapeReport:
    """Summarise one run's hits: the claimable labels, and whether anything escaped.

    The single place that decides which hits may be named in a notice, so the
    contained notice and the escalated one can never disagree about what the table
    found: containment takes every hit (values are registered elsewhere, by
    :meth:`local_operator.variables.VariableStore._register_shape_hits`), the
    ANNOUNCEMENT takes only the claimable ones, and the escalation takes any hit
    that left something readable.
    """
    ordered: dict[str, None] = {}
    for hit in hits:
        if hit.complete:
            ordered.setdefault(hit.label, None)
    return ShapeReport(
        labels=tuple(ordered),
        reached_model=any(hit.exposed for hit in hits),
    )


def scrub_shapes_with_hits(text: str) -> tuple[str, list[ShapeHit]]:
    """Scrub ``text`` and report every shape that fired.

    The one place the table is run. :func:`scrub_shapes` and
    :func:`match_shape_names` are both views onto this, so a surface cannot
    inspect shapes without the same rules being applied to the text.
    """
    hits: list[ShapeHit] = []
    if not text or not has_shape_anchor(text):
        # Nothing in here can match any rule — see :data:`_SHAPE_ANCHORS`.
        return text, hits
    # Key material first: a PEM body spans lines, so it is the one pass that has
    # to see the whole text, and it must run before anything takes it apart.
    text = _run_shapes(_MULTILINE_SHAPES, text, hits)
    if len(_LINE_SHAPES) == 0:  # pragma: no cover - the table would be empty
        return text, hits
    # Then line by line, gated per line: the rules that remain are line-anchored
    # by construction, and ordinary lines never reach them.
    scrubbed_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if not has_shape_anchor(line):
            scrubbed_lines.append(line)
            continue
        line_hits: list[ShapeHit] = []
        scrubbed_lines.append(_run_shapes(_LINE_SHAPES, line, line_hits))
        hits.extend(line_hits)
    text = "".join(scrubbed_lines)
    return text, _only_fully_masked(hits, text)


def _make_hit(shape: Shape, match: Match[str], value: str) -> ShapeHit:
    """One hit, carrying the region the completeness check has to prove gone.

    The region is the USERINFO of a connection string (everything between the
    scheme and the ``@`` that introduces the host) or the value itself. Not the
    whole match: a mask deliberately keeps the NAME and, for a DSN, the user and
    the host readable, so a whole-match check would call every correct mask
    incomplete. And not the
    value alone either: the password GROUP is a fragment of the credential, and a
    mask that stopped at the first ``@`` inside it left the rest of the userinfo
    in the transcript while the value itself disappeared.
    """
    region = value
    whole = match.group(0)
    if "://" in whole and shape.secret_group:
        # From the CREDENTIAL's own start to the ``@`` that introduces the host:
        # the user is deliberately kept readable, so including it would call every
        # correct mask incomplete (``svc_us`` survives in ``svc_user:…``), while
        # stopping at the value misses exactly the leak this check exists for (a
        # password masked to its first ``@``, with the rest of the userinfo left
        # in the transcript).
        offset = match.start(shape.secret_group) - match.start(0)
        region = whole[offset:]
        if "@" in region:
            region = region.rsplit("@", 1)[0]
    return ShapeHit(label=shape.label, value=value, window=region)


#: The shortest run of a credential worth calling a leak. Short enough that a
#: partial mask cannot hide behind it, long enough not to fire on ordinary text.
_FRAGMENT_WINDOW = 6


class _GramIndex:
    """The six-character runs of one model-visible text, built on FIRST use.

    **Why an index rather than a substring search per fragment.** The check that
    came before this one spells it ``fragment in text`` for every offset of the
    credential, so it scans the whole text once per offset and its price is linear
    in the text for EVERY credential in it. Building the text's six-character runs
    ONCE turns each fragment test into a set lookup instead.

    **It is a trade, not a win, and the numbers below are the whole of the claim.**
    Measured on this host (CPython 3.12.13, best of three runs, against a 1 MB
    high-entropy tool result built from 32-hex-character lines; the scan column is
    ``_credential_fragments_survive`` at ``91d70791``):

    * one hit, 64-character value — scan **8 ms**, this index **224 ms**;
    * one hit, 512-character value — scan **102 ms**, this index **221 ms**;
    * one hit, 4 KB PEM body — scan **823 ms**, this index **278 ms**;
    * 20 hits of 512 characters over one text — scan **2729 ms**, this index **241 ms**.

    So the index pays a fixed one-pass build — **224 ms** and a peak of **76 MB** of
    transient set for a 1 MB text (708,574 distinct runs) — and only earns it back
    where the scan it replaces is longer than that build: a value of a couple of KB
    or more, several hits sharing one text, or a text large enough that one scan is
    already the expensive half. For a single short credential in a large result it
    is genuinely SLOWER than the loop it replaces, and the memory is proportional to
    the text. Both facts are why it stays lazy and why most results never touch it.

    **How large that memory gets, measured, because 76 MB is the small end and
    nothing below bounds it (agent review R2, finding 2).** The peak is set by the
    TEXT and not by the credential — one six-character run per text position, each a
    fresh string rather than a slice, at roughly **100 bytes of transient set per
    byte of text** — so it grows linearly with no plateau. Measured through the
    shipped entry point on this host, one masked 40-character credential inside a
    high-entropy hex result, one process per reading (peak RSS delta against the
    same text built without the scan; the pre-index column is
    ``_credential_fragments_survive`` at ``91d70791``):

    * 1 MB text — pre-index **3.0 MB**, this index **101.7 MB** (~761,000 runs),
      and the index is slower here too: **0.39 s** against **0.25 s**, because the
      mask pass dominates and the build is pure addition. In the single-hit shape it
      is therefore strictly memory-negative — it buys no wall time and pays ~100 MB;
    * 4 MB text — this index **391.0 MB** (~2,719,000 runs), i.e. the same ~98 bytes
      per text byte, i.e. linear in the text.

    So a multi-megabyte tool result carrying ONE secret pays hundreds of megabytes
    transiently, on a check that runs per tool result. Read the ratio rather than
    the megabyte — the absolute figure follows the text's distinct-run count (1 MB
    of one repeated line is far cheaper than 1 MB of random hex) — and read it as a
    disclosure, not a bound: the build is bounded only by the length of the text
    handed in.

    Reproduce a row by loading this module by path in one process (it has no
    intra-package imports, so the pre-index revision loads beside it), building the
    text, and diffing ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` around a single
    ``scrub_shapes_with_hits`` call — one process per reading, since ``ru_maxrss`` is
    a high-water mark and never falls back.

    LAZY, because most results contain no hit at all: nothing is built unless a hit
    needs a fragment tested, so the ordinary-text path is untouched. Most results
    are also far too small for either side to matter — every text in the credential
    corpus is under 200 characters, where the whole question is moot.

    The redaction MARKER is stripped before the runs are taken: it is what a mask
    writes, never material a mask left behind, and a run straddling one would
    otherwise match a credential whose own value IS the marker — the two ``.npmrc``
    cases and the cookie-header case QA round 1 found escalating on nothing readable
    at all. The seam the strip creates is harmless (a credential's own characters
    are never joined by it, and a seam match would require the value's characters
    to be readable on both sides of a marker, which is a real survivor anyway); what
    it does cost is stated in :func:`_credential_fragments_survive`.
    """

    __slots__ = ("_text", "_grams")

    def __init__(self, text: str) -> None:
        self._text = text
        self._grams: set[str] | None = None

    def __contains__(self, gram: str) -> bool:
        if self._grams is None:
            readable = self._text.replace(REDACTION_MARKER, "")
            self._grams = {
                readable[start : start + _FRAGMENT_WINDOW]
                for start in range(len(readable) - _FRAGMENT_WINDOW + 1)
            }
        return gram in self._grams


def _credential_fragments_survive(
    hit: ShapeHit, text: str, grams: "_GramIndex | None" = None
) -> bool:
    """Whether a readable piece of the CREDENTIAL survived in the masked ``text``.

    **Anchored on the credential's own characters, never on the matched region**,
    and that is the whole of the judgement (QA round 1, Q1). The region is not all
    secret: a DSN rule keeps ``amqp://user:`` and ``@host`` readable BY DESIGN, so a
    region-wide window search graded a fully-masked ``amqp://guest:guest@host`` as
    exposed — the surviving window was the USERNAME — and filed a rotation demand
    for a value that never left the tool. Four corpus hits were affected, three of
    the four because the "survivor" was the redaction MARKER itself. So:

    * the marker is not material (``_GramIndex`` strips it, and a value that IS the
      marker is never a survivor);
    * the whole VALUE present in the text is a leak: the rule's group was narrower
      than the credential, or this copy was never masked;
    * a window OF THE VALUE present is a partial leak — the mask stopped inside the
      credential, which is the case the floor exists for and what a truncating rule
      (a PEM without its END line, a base64 body) produces.

    Reading the VALUE rather than the region means material a rule deliberately
    preserves can never be counted as a survivor.

    **A stated limit, not an oversight.** Because ``_GramIndex`` strips the marker
    before taking its runs, a credential whose own value literally contains
    ``[redacted]`` and survives only PARTIALLY is unrepresentable: the fragment
    still in the text is spelled exactly like the marker a mask would have written,
    and no reading of the text can tell them apart. That identity is what makes the
    ``.npmrc`` and cookie false positives above impossible to grade correctly by
    inspection, so it is not closable here — only the wholly-surviving copy of such
    a value is still caught, by the ``value in text`` test above. Reaching it needs
    an operator secret that itself contains the harness's marker string, which is
    why the limit is recorded rather than paid for.
    """
    value = hit.value
    if not value or value == REDACTION_MARKER:
        return False
    if value in text:
        return True
    if len(value) < _FRAGMENT_WINDOW:
        return False
    if grams is None:
        grams = _GramIndex(text)
    return any(
        value[start : start + _FRAGMENT_WINDOW] in grams
        for start in range(len(value) - _FRAGMENT_WINDOW + 1)
    )


def _is_truncated_pem(hit: ShapeHit) -> bool:
    """Whether a PEM-shaped match has no END marker for its BEGIN.

    The completeness check scores a hit against its credential region, and for a
    ``private_key`` the region is the whole block. A block that was cut short — the
    spelling a service-account file takes when a tool truncates it, and the shape
    the reviewer recovered a live key from — has no END, so the region's extent is
    unknown and no masking claim may be made about it.
    """
    upper = hit.value.upper()
    if "BEGIN" not in upper or "PRIVATE KEY" not in upper:
        return False
    # The END of the SAME KIND of key, matched as a phrase: ``END PRIVATE KEY`` is
    # the PKCS#8 spelling only, so keying on that literal string withheld the claim
    # for `RSA`/`OPENSSH`/`EC` blocks — the most common spellings — and the rotation
    # notice never fired for them (round 5, R5-2). ``END`` alone is the opposite
    # error: a base64 body spells those three letters often, and the substring test
    # then reported a completed mask over readable key material (M4-1).
    return re.search(r"END [A-Z0-9 ]*PRIVATE KEY", upper) is None


def _only_fully_masked(hits: list[ShapeHit], text: str) -> list[ShapeHit]:
    """Grade every hit: contained, contained-but-unclaimable, or EXPOSED.

    **A notice may never announce a masking that did not happen.** The row the
    operator is asked to act on says the value "was masked before you saw it", and
    a claim like that is worse than silence when it is false: it is the difference
    between rotating a credential and believing you already have. This was a real
    defect — a DSN password containing ``@`` was masked only to the first ``@``
    while the incident row still promised the whole thing was gone, and the tail
    was in the transcript.

    Two flags come out of here, and they are different facts:

    * ``exposed`` — readable credential material is still in ``text``. That text
      is what the model reads, so this hit has reached the context window and is
      the one case that asks for a rotation.
    * ``complete`` — the mask may be CLAIMED as whole. A truncated PEM is fully
      masked (nothing readable survives) and still unclaimable, because nothing
      proves the rest of the key is not further down a transcript we have not
      read. Withholding the claim is the honest half of that fix, and such a hit
      is neither announced as contained nor escalated: no claim of any kind is
      made about it.

    A hit that is neither exposed nor unclaimable is CONTAINED, and that is the
    ordinary outcome: the value was masked whole before the model could read it,
    whatever surface it arrived on.

    The check is a substring test per hit against the credential's own region,
    not a proof: it catches a value that survives whole (the group was narrower
    than the credential) and one that survives in six-character fragments.
    Fixing the patterns is the real work; this is the backstop that stops the
    false claim if one slips through again.
    """
    marked: list[ShapeHit] = []
    # One index for the text, shared by every hit and built on first need: the
    # fragment test is per-credential and the text is the same for all of them.
    grams = _GramIndex(text)
    for hit in hits:
        # Readable material is checked FIRST, so the truncated-PEM branch below
        # cannot swallow an exposure: a block that was masked is contained, and
        # one that left a fragment readable is not.
        exposed = _credential_fragments_survive(hit, text, grams)
        if _is_truncated_pem(hit) or exposed:
            # A BEGIN with no END is a key whose LENGTH we cannot see: everything
            # visible is masked, and the claim is still withheld, because nothing
            # proves the rest of the key is not further down a transcript we have
            # not read. The hit is kept for CONTAINMENT either way — the value is
            # registered for the rest of the session — and the honest thing to
            # withhold is the claim, not the protection.
            #
            # A hit graded this way therefore files NO notice at all, and that is a
            # stated limit rather than an oversight: both notice texts make a
            # containment claim ("nothing entered your context" / "it was contained
            # at the tool"), and the reason the claim is withheld here is that the
            # key's extent is unknown — so neither text would be true. Raising it
            # needs a third, claim-free wording, which is a product decision rather
            # than something to bolt onto this change (agent review R1, finding 3).
            marked.append(replace(hit, complete=False, exposed=exposed))
        else:
            marked.append(hit)
    return marked


#: A mask followed by a QUOTE and more credential-shaped characters is an
#: INCOMPLETE mask: the rule's value class stopped at the quote while the credential
#: continued, so the rest of the credential stayed readable under a notice that said
#: it had been masked. Three review rounds found this in a different rule each time,
#: which is why it is closed STRUCTURALLY rather than by widening one more class:
#:
#: **A credential does not end at a quote.** The tail is masked too, and the
#: lookahead requires a real delimiter after it (whitespace or one of
#: `,;:)]}&?=<>|`), so a quote that genuinely delimits a value
#: (`password: "[redacted]"`) is left where it is and nothing else is touched.
_INCOMPLETE_MASK_RE = re.compile(
    # The run is CREDENTIAL MATERIAL, not punctuation: `"}` after a mask is a JSON
    # closing quote and brace, and masking it damaged a body the client hands to a
    # human (measured on `test_error_body_redaction`). Only word characters and the
    # symbols a credential is spelled with may extend a mask.
    r"\[redacted\](['\"])(?:,[\w.~+/=@%$!:-]+)?([\w.~+/=@%$!:-]+(?:['\"][\w.~+/=@%$!:-]+)*)"
    r"(?=['\"]?(?:[\s,;:)\]}&?=<>|]|$))"
)


#: The mirror case: a mask whose LEFT side is a readable run followed by a quote
#: (``_authToken=npm_abcd'[redacted]``). The run is credential material the rule
#: could not see, and it is masked for the same reason as the tail.
_INCOMPLETE_MASK_LEFT_RE = re.compile(
    # `^` as well as a delimiter: a credential at the start of a line has nothing
    # before it, and that is where a prefix-orphaning split lands most often.
    # The run must not be an assignment NAME: `AWS_ACCESS_KEY_ID='AKIA…'` is a name
    # the operator (and this session's own notice) needs to see, and swallowing it
    # also left the quoting unbalanced. An env-var-style name — capitals, digits and
    # underscores — is excluded; a token prefix like `github` or `dckr_` is not.
    r"(?:(?<=[\s,;:(\[=])|^)(?![A-Z][A-Z0-9_]*\b)([\w.~+/=@%$!:-]+)(['\"])\[redacted\]"
)


def _close_partial_masks(text: str) -> str:
    """Mask the readable tail (or head) of a credential whose mask stopped at a quote."""
    for _ in range(3):
        closed = _INCOMPLETE_MASK_RE.sub(REDACTION_MARKER, text)
        closed = _INCOMPLETE_MASK_LEFT_RE.sub(REDACTION_MARKER, closed)
        if closed == text:
            break
        text = closed
    return text


def _run_shapes(shapes: tuple[Shape, ...], text: str, hits: list[ShapeHit]) -> str:
    """Apply one group of rules, in table order."""
    for shape in shapes:
        if shape.guard is not None:
            text = _apply_guarded(shape, text, hits)
            continue
        # ``__post_init__`` guarantees one of the two is present; the checker
        # cannot see through that, so the assertion states it here as well.
        assert shape.replacement is not None
        matches = list(shape.pattern.finditer(text))
        if matches:
            for match in matches:
                value = _hit_value(shape, match)
                # A hit is recorded for every mask, whatever the value's length:
                # the FLOOR decides what is worth registering, not what is worth
                # REPORTING. Gating the record on it silenced the notice for a
                # short-but-real credential (a 5-character DSN password, a
                # 7-character `-pass` value): masked in the text, no hit, no row,
                # no containment — the silent half of this whole PR.
                if value:
                    hits.append(_make_hit(shape, match, value))
        text = shape.pattern.sub(shape.replacement, text)
    return _close_partial_masks(text)


#: Cheap necessary conditions for the whole table, as lowercase substrings.
#:
#: **Why a gate at all.** Every rule is a full scan of the text, and CPython's
#: regex engine costs ~30 ns per character scanned: twenty-three ungated rules
#: measure 1.3 µs per input byte — 5 s of loop-thread CPU for a 4 MB tool result,
#: and more than the live pipe's whole CPU budget for a 64 KB chunk. The gate is
#: one compiled alternation, so ordinary text (a build log, a directory listing,
#: a JSON payload) pays ~0.03 µs per byte and none of the table runs.
#:
#: A SUBSET is not enough — a missing anchor would silently stop a rule from
#: firing — so ``test_every_positive_case_trips_the_gate`` asserts that every
#: positive corpus case carries at least one anchor, and adding a rule without
#: one fails loudly there rather than in production.
_SHAPE_ANCHORS: tuple[str, ...] = (
    "://",
    "-----begin",
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "pwd",
    "credential",
    "dsn",
    "bearer",
    "basic",
    "cookie",
    "machine",
    "sig",
    "auth",
    "curl",
    "docker",
    "-pass",
    "mysql",
    "psql",
    "pg_dump",
    "pg_restore",
    "mongo",
    "clickhouse-client",
    "redis-cli",
    "influx",
    "hooks.slack.com",
    # Bare issuer tokens, whose rules have no name to anchor on. Each group
    # names the rule it gates; a prefix added to one of those patterns without a
    # line here is the defect `_VENDOR_ANCHORS` was derived to make impossible
    # for the vendor rule (B-1), and the corpus test
    # `test_every_rule_that_fires_trips_the_gate` catches the rest.
    "ghp_",  # github-token
    "gho_",  # github-token
    "ghs_",  # github-token
    "ghu_",  # github-token
    "github_pat_",  # github fine-grained token
    "akia",  # aws-access-key-id
    "asia",  # aws temporary access key id
    "aiza",  # google-api-key
    "ya29",  # google-oauth-token
    "eyj",  # jwt
    "xox",  # slack-token
    "sg.",  # sendgrid-key
) + _VENDOR_ANCHORS


#: Answers "could anything in the table match?" — cheaply, and that is the whole
#: design constraint. The obvious form, one compiled alternation of the anchors,
#: is the WRONG shape here: CPython's ``re`` has no multi-literal fast path, so an
#: alternation of 61 literals costs one attempt per alternative AT EVERY
#: POSITION — measured at 1.7 s for 730 KB of ordinary log text, i.e. worse than
#: the table it was meant to skip. ``str.__contains__`` is the C-level search the
#: engine does not do for us: 61 of them cost ~25 ms for the same text, and the
#: first miss short-circuits nothing but nothing needs it to.
def has_shape_anchor(text: str) -> bool:
    lowered = text.lower()
    return any(anchor in lowered for anchor in _SHAPE_ANCHORS)


#: Rules whose shape can only be complete on one line, and the ones that can span
#: lines. Splitting them is what lets the gate run per line: a 40-line log with
#: one candidate line pays the table for that line only.
#: Blocks and anchored values are MULTILINE rules because their match spans a line by
#: construction (escaped newlines included). `gcp-service-account-value` must be here and
#: FIRST in the table: the marker rule matching the same value first would consume the
#: BEGIN marker and leave the value rule's guard looking at a value with no key material
#: in it, which is how the whole body stayed published after the rule was added (M6-1).
_MULTILINE_LABELS = frozenset(
    {
        "pem-private-key",
        "gcp-service-account-key",
        "gcp-service-account-value",
        "gcp-service-account-value-open",
    }
)
_MULTILINE_SHAPES = tuple(s for s in CREDENTIAL_SHAPES if s.label in _MULTILINE_LABELS)
_LINE_SHAPES = tuple(s for s in CREDENTIAL_SHAPES if s.label not in _MULTILINE_LABELS)


def _apply_guarded(shape: Shape, text: str, hits: list[ShapeHit]) -> str:
    """Run one guarded rule: the pattern proposes, the guard decides.

    The guard runs only where the pattern matched, so a rule whose condition
    cannot be expressed cheaply (does this identifier END in a credential word?
    does this URL actually carry a credential?) costs nothing on the ordinary
    text that makes up almost every result. See :attr:`Shape.guard`.

    A rejected span is re-scanned from ONE CHARACTER IN, not past its end, and
    that is not a detail: the cheap name pattern happily matches a name that is
    really somebody else's argument — ``--from-literal=password=hunter2`` matches
    with the name ``--from-literal`` and the value ``password=hunter2`` — and
    resuming past it would swallow the genuine assignment sitting inside the
    span. Measured: the corpus's ``--from-literal=password=…`` and
    ``vault kv get …: password=…`` cases both stopped masking.
    """
    guard = shape.guard
    group = shape.secret_group
    assert guard is not None and group is not None  # guarded rules only

    pieces: list[str] = []
    cursor = 0
    search_from = 0
    matched = False
    while True:
        match = shape.pattern.search(text, search_from)
        if match is None:
            break
        if not guard(match):
            search_from = match.start() + 1
            continue
        value = _hit_value(shape, match)
        if value:
            hits.append(_make_hit(shape, match, value))
        # Keep everything before the credential, mask the credential: an
        # assignment keeps its name and separator, a URL-valued name keeps the
        # name, and the credential inside the value is what goes.
        pieces.append(text[cursor : match.start()])
        # Keep the text AFTER the credential too: a rule may end its match on a
        # delimiter it must not eat (the closing quote of a JSON value). Dropping it
        # silently unquoted the output (`…[redacted], "other": "value"}`).
        matched = match.group(0)
        pieces.append(
            matched[: match.start(group) - match.start(0)]
            + REDACTION_MARKER
            + matched[match.end(group) - match.start(0) :]
        )
        cursor = match.end()
        search_from = match.end()
        matched = True
    if not matched:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


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
        "use it inside the command that needs it: "
        '`gh api -H "Authorization: token $(gh auth token)"`',
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
        # `terraform output NAME` IS the safer form; the rule is for the bare
        # `terraform output` (every output, secrets included) and for
        # `state show`. Flagging the command the notice recommends is how a
        # guard teaches the model to ignore it.
        excluded=re.compile(r"(?i)\bterraform\s+output\s+(?:-json\s+)?[A-Za-z_]\w*"),
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
        # An EXAMPLE file is explicitly not a secret: `cat .env.example` is how
        # a newcomer reads the shape of the config, and nagging about it is
        # noise the operator learns to skip.
        excluded=re.compile(r"(?i)\.env\.(?:example|sample|template|dist|example\.[a-z]+)\b"),
    ),
)

#: A pipeline that prints only NAMES is the form the notice itself recommends,
#: so it must not trigger the notice. Kept as one pattern rather than a flag on
#: every rule: what makes a pipeline name-only is the extractor at its end.
_NAME_ONLY_PIPELINE = re.compile(
    r"(?i)^\s*\|?\s*(?:(?:cut\s+-d\s*=)|(?:awk\s+-F\s*=)|(?:sed\s+-E?\s*\"?'?s/=))"
)


#: The BRIEF form of each rule's advice, which is what the notice emits.
#:
#: Why a second, shorter string rather than the table's ``safer`` text: the
#: notice lands at the END of a result the operator reads through the tool card,
#: which renders one row per line, ellipsised at the row width (92 cells at a
#: 100-column frame) and cropped to the first 40 lines — so the full advice
#: (120-282 cells, measured across all sixteen rules) was unreadable at every
#: width and absent entirely on a long result. A notice the operator cannot read
#: is not a notice. ``test_every_notice_fits_one_narrow_card_row`` pins the
#: budget; ``safer`` keeps the long form for the module documentation.
_BRIEF_ADVICE: dict[str, str] = {
    "environment-dump": "print names, not values: `env | cut -d= -f1`",
    "named-variable-dump": "use it in place: `$(lop secret get NAME)`",
    "kubectl-exec-env": "print names in the pod: `env | cut -d= -f1`",
    "kubectl-secret-read": "select one key: `-o jsonpath='{.data.KEY}'`",
    "docker-inspect": "select a field: `--format '{{.State.Status}}'`",
    "docker-exec-env": "print names only: `env | cut -d= -f1`",
    "docker-compose-config": "read one part: `compose config --services`",
    "aws-credentials-read": "prefer `aws sso login` and a scoped role",
    "gcloud-token": "use: `$(gcloud auth print-access-token)`",
    "github-auth-token": "use it in place: `$(gh auth token)`",
    "gitlab-auth-token": "drop `-t`: `glab auth status`",
    "vault-read": "select one field: `-field=KEY`",
    "heroku-config": "read one key: `heroku config:get KEY -a APP`",
    "terraform-output": "read one output: `terraform output <name>`",
    "npm-token-list": "revoke unused tokens: `npm token revoke <id>`",
    "credential-file-read": "read one field: `grep -c .`, `jq .field`",
}


#: What the notice calls each rule. Short because the card's inner measure at 80
#: columns is 74 cells and the ADVISORY is the point of the line: the rule's own
#: label (``docker-compose-config``) spends a third of the row on internal
#: vocabulary, and round 2 measured the advice clipped mid-command at the most
#: common terminal width.
_BRIEF_LABEL: dict[str, str] = {
    "environment-dump": "env",
    "named-variable-dump": "variable",
    "kubectl-exec-env": "k8s env",
    "kubectl-secret-read": "k8s secret",
    "docker-inspect": "docker",
    "docker-exec-env": "docker env",
    "docker-compose-config": "compose",
    "aws-credentials-read": "aws",
    "gcloud-token": "gcloud",
    "github-auth-token": "gh",
    "gitlab-auth-token": "glab",
    "vault-read": "vault",
    "heroku-config": "heroku",
    "terraform-output": "terraform",
    "npm-token-list": "npm",
    "credential-file-read": "file",
}


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
        brief = _BRIEF_ADVICE.get(shape.label, shape.safer)
        label = _BRIEF_LABEL.get(shape.label, shape.label)
        return f"[credential guard] {label}: {brief}"
    return None


def _is_name_only_pipeline(command: str, match: Match[str]) -> bool:
    """Whether the dump is piped straight into a names-only extractor."""
    return bool(_NAME_ONLY_PIPELINE.match(command[match.end() :]))
