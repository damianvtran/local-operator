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
the sole input to the escalated notice: only a fragment that survived into the text the
model reads escalates, and a hit that was masked whole is CONTAINED.

**CONTAINED IS SILENT, and the cleanup obligation went with it.** The operator's
instruction is that the contained case files nothing — "as long as something wasn't
actually leaked to the transcript we shouldn't get a session incident indicated
anywhere" — and one consequence belongs here rather than in a diff comment: the
contained wording carried the only cleanup obligation this system ever stated,
*delete any plaintext copy a tool call may have written*, and with contained hits
filed nowhere in-tree there is now no surface that tells an operator a plaintext
copy may be sitting on disk. That is a deliberate trade, not an oversight; the
wording is kept in :func:`~local_operator.incidents.format_shape_incident_message`
for a caller that deliberately has something to say about a contained hit, and
today nothing in-tree does.

This paragraph is about SEVERITY; it changes nothing about
coverage, where under-masking is still a leak and over-masking is still a defect.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Iterable,
    Mapping,
    Match,
    Optional,
    Pattern,
    Sequence,
    Union,
)

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

#: The shortest value the two grammars below accept: the run plus the character
#: that must END it. Named because the value JUDGEMENT asks the same question of a
#: rendered value — see :func:`_value_before_an_escape`, which refuses to treat an
#: escape as the value's end below this floor.
_ASSIGNED_VALUE_MIN_CHARS = 8

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
    # ``[^\s]`` cannot cross a REAL line, and a real line is the only kind this
    # class may stop at.
    #
    # IT CROSSES A RENDERED LINE, AND THAT IS THE SAFE DIRECTION. A JSON payload
    # spells every newline inside a string as the two characters ``\\`` and ``n``,
    # and neither of them is whitespace, so a value arriving in a rendering runs
    # past the line break and takes the next line's material with it. Masking that
    # material is an over-mask — it can hide the following line of ordinary code in
    # a notice that stays CONTAINED (``complete=True``, ``exposed=False``) — and the
    # alternative was measured and refused (agent review R1-2, 2026-09-21):
    #
    #   A VALUE THAT STOPS AT THE ESCAPE LOSES CREDENTIAL MATERIAL. Through both
    #   modules in one process: a credential-named assignment whose own bytes carry
    #   an escaped break — a JSON "client_secret" field holding a short run, then
    #   the two characters backslash and ``n``, then forty more characters of
    #   body — is masked WHOLE and contained at ``origin/main``, and with the
    #   exclusion in
    #   place came back with the tail readable while the hit was still graded
    #   ``complete=True``, so the notice claimed a containment that did not happen;
    #   and when the run before the escape was shorter than the seven-character
    #   floor the rule did not fire AT ALL — the value readable, and nothing
    #   registered for containment either, so no later pass could contain it.
    #
    # Under-masking a real credential is the one direction this table refuses to
    # buy with an over-mask, so the value keeps every byte the rendering gave it.
    # The escape is handled where it belongs — on the NAME, by
    # ``_name_after_an_escape``, which is what actually closed the false positive
    # this rule was being changed for.
    r"([^\s]{"
    + str(_ASSIGNED_VALUE_MIN_CHARS - 1)
    + r",}[^\s,;)\]}\"'.])(?=[\s,;)\]}\"']|$)"
)

#: The value of an assignment whose value is QUOTED, sharing the grammar above
#: with one addition: the run may not cross a quote that TERMINATES it.
#:
#: **Why a second value grammar rather than a smarter class in the one above.**
#: The greedy run above is bounded by a LENGTH and a terminating delimiter, and
#: a delimiter is exactly what a quote looks like in compact JSON — so on the
#: surface JSON actually travels on (no whitespace anywhere) the run walked
#: straight through the closing quote and into the NEXT FIELD. Measured on the
#: operator's own transcripts (2026-09-20):
#:
#:   \{"access_token":"…","refresh_token":"…"\}
#:
#: matched the value `access_token`s value PLUS `","refresh_token":"…`, so the
#: mask destroyed the neighbouring KEY (over-masking, the defect the negative
#: corpus exists to prevent) and the grader — correctly, given that match — found
#: a six-character window of the swallowed text inside the unmasked first key and
#: filed a rotation demand for a credential that had been masked whole.
#:
#: The rule added here is the bound the greedy class cannot express: the run stops
#: at the opening quote when that quote is followed by a delimiter or the end,
#: because that is where the value ENDS. It is the last-character rule of the
#: class above, applied recursively to the char that delimits the spelling, and it
#: keeps every case the greedy form handled correctly — `"abc,defghij"`, an
#: escaped quote (`"abc\"def"`) and an inner quote followed by a value
#: character (`"abc"def"`, which is today's behaviour: the value runs to the last
#: quote) all take the same span as before, while the field-crossing match does
#: not exist any more.
#:
#: The unquoted spelling deliberately keeps the grammar above: an unquoted value
#: has NO delimiter to stop at, and narrowing it would publish the tail of a
#: credential containing a quote (the case the negative corpus already carries as
#: `DB_PASSWORD=abc"defghij"`).
_QUOTED_ASSIGNED_VALUE_GROUP = (
    r"((?:(?!(?P=quote)(?=[\s,;)\]}\"']|$))[^\s]){"
    + str(_ASSIGNED_VALUE_MIN_CHARS - 1)
    + r",}[^\s,;)\]}\"'.])(?=[\s,;)\]}\"']|$)"
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
#: The tail nouns that make a count word in a NON-first segment decisive.
#:
#: Not "any credential word the name could end in" — that is the leak this set
#: exists to prevent — and not ``token``, which is a credential word that
#: happens to be spelled like one of these (``FACEBOOK_PAGE_ACCESS_TOKEN``).
#: A tail here is a QUANTITY by itself, so no qualifier can make it a secret.
#:
#: **What that trades away, stated rather than left to a differential** (QA round 2,
#: Q2-1). A name of the form ``PREFIX_<count word>_TOKENS`` — ``REDIS_CACHE_TOKENS``
#: and its relatives, 300 spellings in QA's sweep — was masked at ``origin/main``
#: (its first segment is not a count word, and the first-segment arm was all there
#: was) and is released here, because this arm reads the TAIL as a quantity. The
#: reading is deliberate: a qualified ``TOKENS``/``COUNT`` tail is a count whatever
#: precedes it, which is the same judgement that keeps the Anthropic counters
#: unmasked, and QA's round-2 pass read them the same way. It is pinned in the
#: NEGATIVE half of the corpus (``COUNT_TAIL_RELEASED_NAMES``) rather than left to
#: agree with a differential, and what would change it is a real credential
#: spelling in that family — a corpus case, not a hunch.
_COUNT_TAIL_NOUNS = frozenset({"tokens", "count", "counts"})

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

    **The first segment decides, and then the TAIL decides.** The first-segment
    form is right for the vocabulary above and wrong for the Anthropic usage
    counters the Bedrock cost-tracking work is full of:
    ``ephemeral_5m_input_tokens`` and ``ephemeral_1h_input_tokens`` are counts,
    but their first segment is a MODE, so the tail ``tokens`` won the judgement,
    the counter was masked, and the grader then found a fragment inside the
    swallowed next field and filed a rotation demand for a NUMBER (the operator's
    ``write`` of ``idv-bedrock-ca-pin/EVIDENCE.md``, 2026-09-20).

    The obvious widening — a count word anywhere in the name — is a LEAK, and it
    was measured on the shipped store path before it shipped.
    :func:`is_credential_name` asks only that the TAIL be a credential word, so a
    qualifier that merely CONTAINS a quantity word released real credential names
    that ``origin/main`` masks: ``REDIS_CACHE_PASSWORD`` (``cache``),
    ``KAFKA_OUTPUT_SECRET`` (``output``), ``OPENAI_PROMPT_KEY`` (``prompt``),
    ``MY_PAGE_ACCESS_TOKEN`` (``page``) and their relatives — left fully
    readable, with no mask, no label and nothing registered for containment
    (agent review R1-1, QA round 1 Q-1).

    So the sweep is scoped by the TAIL, which is the segment that says what a name
    IS. A tail that is itself a quantity (``tokens``, ``count``) makes the name a
    count whatever qualifies it; ``…_ACCESS_TOKEN`` and ``…_CACHE_PASSWORD`` are
    credentials whatever qualifies them. The first-segment arm is kept exactly as
    ``origin/main`` had it, because it carries the vocabulary the tail cannot: a
    cache HANDLE keys a count, and a model parameter is named for what it limits.

    One residual is knowingly left, and it is pre-existing rather than this rule's:
    a name whose FIRST segment is a count word and whose tail is a credential word
    — the shape where a quantity word opens the name and a credential word closes
    it — is released here, and was released at ``origin/main`` too, because
    ``segments[0]`` is what that arm reads. Closing it means re-deciding the first
    arm (which would re-mask ``cache_key``, the case that arm exists for), not
    widening this sweep, so it is recorded rather than fixed here.
    """
    segments = _name_segments(name)
    if len(segments) < 2:
        return False
    if segments[0] in _COUNT_WORDS:
        return True
    return segments[-1] in _COUNT_TAIL_NOUNS and any(
        segment in _COUNT_WORDS for segment in segments[1:-1]
    )


def _tail_is_a_quantity_noun(name: str) -> bool:
    """Whether a name's LAST segment is a quantity by itself.

    The narrow half of :func:`is_count_shaped`, and the scope a value judgement
    needs (agent review R1-1). ``is_count_shaped`` releases a name whose FIRST
    segment is a count word, which is the vocabulary a model parameter is named
    with, and its tail arm needs a count word in the middle as well — so a usage
    counter that names no quantity (``reasoning_tokens``, ``extra_native_tokens``)
    is a credential-shaped name to both of those and to everything else in this
    table.

    What the tail alone buys is the one judgement those counters need: a name whose
    tail is ``tokens``/``count``/``counts`` cannot hold a secret, because a plural
    quantity noun IS the quantity. That is deliberately NOT extended to the
    singular ``token``: a ``…_TOKEN`` name is how every issuer credential is
    spelled, and it stays under the full credential judgement.
    """
    segments = _name_segments(name)
    return bool(segments) and segments[-1] in _COUNT_TAIL_NOUNS


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

#: The token-tail grammar, shared by every vendor prefix: a run of at least 8
#: token characters carrying NO dot. Three structural decisions, each measured:
#:
#: * **8 characters, not 12** — pre-existing callers of this pass treat a
#:   9-character tail as a credential, and a longer floor published it. Reading
#:   all-letter runs of this length is deliberate: what keeps a NAME among them from
#:   being read as a token is the GUARD rather than the charset — round 1's own
#:   repro (``pk-`` plus sixteen letters) is masked precisely because its tail
#:   carries no separator, and that is :func:`_vendor_tail_guard`'s rule;
#: * **no dot** — a dot is what a filename has and an issuer token does not, and
#:   it is what separates ``pypi-local-operator.json`` and
#:   ``pypi-local-operator.json.<random>.tmp`` (ordinary cache filenames, both
#:   published while the dot counted toward a length floor) from a real tail; the
#:   lookahead keeps both readable;
#: * **no digit requirement** — the charset witnesses a token's ALPHABET and
#:   nothing more. "Must include a digit" was the first attempt at separating a
#:   token from an ordinary hyphenated name, and it is wrong in both directions: a
#:   tail of nothing but letters is a credential to the callers above, and a NAME
#:   may carry digits (``npm_config_manage_package_manager_versions=11.22.0``).
#:
#: The token/name discrimination is therefore the GUARD's, and the reason it is a
#: guard rather than part of this pattern is representational: a phrase cannot be
#: spelled in a look-behind (Python requires fixed width), and a guard runs only on
#: a match, so it costs nothing on text that never reaches a prefix.
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
    # ``refresh_token: RefreshFn | None = None``, ``_store:
    # AuthStore``, ``reasoning_tokens: SafeCount``, ``refresh_token:
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
    # name: ``get_api_key=_oauth_api_key``. Restored unchanged from ``origin/main``
    # when the arm below was narrowed: this clause never released a credential name
    # (agent review R1-1).
    if (
        strong
        and value.startswith("_")
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
        and not any(char.isdigit() for char in value)
    ):
        return True
    # A USAGE COUNTER's value is the name of another local, and this arm is scoped
    # to the names that makes true for. Measured 2026-09-21 on a ``read`` of
    # ``local_operator/providers/clients.py``: ``reasoning_tokens=<another local>``
    # and ``extra_native_tokens=<another local>`` are counters whose TAIL is a
    # credential word, so :func:`is_count_shaped` does not cover them — the tail is a
    # QUANTITY noun, which is a different test from that arm's first-segment
    # vocabulary — and the identifier on the right was read as a credential and
    # ESCALATED to a rotation demand for a variable name.
    #
    # **The arm used to be "any multi-segment identifier" and that was a leak**
    # (agent review R1-1). Through both modules in one process: a credential-named
    # assignment whose value is a digit-free underscore-joined phrase — the shape a
    # person writing a passphrase reaches for — went from MASKED to NO HIT AT ALL
    # under a database-password name, a bare password name, a Postgres password
    # name, a Mongo password name, an API token name and the AWS secret access
    # key name, in the ``export``-prefixed and
    # docker-compose spellings and inside JSON, while the HYPHENATED and
    # digit-carrying spellings of the identical value stayed masked. Nothing was
    # registered either, so the later exact-value pass could not contain it. The
    # class is pinned in the POSITIVE half of the corpus now, which is where it
    # should have been from the start.
    #
    # So the scope is the names the two reports actually share: a tail that is a
    # QUANTITY NOUN, which is what ``_COUNT_TAIL_NOUNS`` holds. The boundary that
    # leaves is real, and it is pinned in the NEGATIVE half rather than left to a
    # paragraph — a passphrase spelled with underscores under a ``…_TOKENS`` name is
    # read as a NAME. Every other credential name keeps masking it.
    #
    # The digit floor is not negotiable: every issuer-prefixed key, every AWS key id,
    # and every hex, base64 and UUID-shaped value carries one and stays masked — none
    # of them is spelled as a phrase. Hyphenated phrases are untouched here for the
    # same reason the corpus pins a hyphenated multi-word password: a hyphen is a
    # separator a person writing a password reaches for, an underscore is not.
    # The issuer clause keeps the vendor rules' own cases: a value that OPENS with an
    # issuer prefix is judged by ``vendor-prefixed-token``, which knows the alphabet
    # and the tail each one really carries. Without it the two rules contradict each
    # other on the same string — an npm token's own spelling is a lowercase
    # underscore-joined run — and the corpus's ``.npmrc`` rows measured exactly that:
    # the mask stayed, and the credential reading silently dropped to a duplicate hit
    # on the rule beside it.
    if (
        strong
        and _tail_is_a_quantity_noun(name)
        and value.count("_")
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
        and not any(char.isdigit() for char in value)
        and not _VENDOR_PATTERN.match(value)
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


#: The NAME FORM of a vendor tail — the NEGATIVE SPACE of the real one, and the
#: only half of that judgement a regex can spell: lowercase words joined by ``_``
#: or ``-``, optionally with an ``=value`` assignment appended. A real tail is ONE
#: unbroken base64-ish run (the npm and PyPI cases in the corpus carry seven
#: capitals each), while a string made only of lowercase letters and separators is
#: a NAME someone wrote down, and ``docker_compose_build_args``,
#: ``npm_config_update_notifier`` and
#: ``npm_config_manage_package_manager_versions=false`` are all the same thing.
#:
#: **What runs here is that NEGATIVE test, and the distinction is the whole of
#: agent review R1-1.** The positive reading — "a real tail carries mixed case or a
#: digit" — is true only in the direction that such a tail can never be a NAME; as
#: a statement of the predicate it is false, because an issuer tail is a random run
#: that USUALLY carries case or a digit rather than one that must. Only the
#: negative form is what the code enforces, so only the negative form is written
#: down here.
#:
#: **The ``=value`` arm is what the underscore-only rule missed, and the omission
#: was expensive.** An environment variable written in PROSE carries its
#: assignment, so the matched tail is the name AND the value: the old rule
#: fullmatched ``[a-z]+(?:_[a-z]+)+`` against the whole tail, failed on the ``=``,
#: and announced the pair as an npm token. On 2026-09-21 that produced an
#: ESCALATED rotation notice (session ``78e6409f2ba1``, the second firing of this
#: class — the first is the ``--secret NAME`` case on :func:`_flag_value_guard`).
#: The escalation, not the mask, is what made it an incident: the tail is 48
#: characters, ``_FRAGMENT_WINDOW`` is 6, and six-character fragments of an
#: ordinary name (``config``, ``manage``, ``_versi``) are all over the prose
#: around it, so ``_credential_fragments_survive`` graded the mask EXPOSED and the
#: operator was asked to rotate a package-manager toggle. A guard that admits a
#: name does not merely under-mask: it manufactures a false compromise.
#:
#: The two joins are ONE form, not two: ``-`` is the same name spelled the other
#: way (``npm-config-manage-package-manager-versions``), and both spellings are
#: already how every prefix in the table is written. The value half is deliberately
#: character-agnostic (``\S*``) because a value may hold digits, dots and slashes
#: (``npm_config_cache=/Users/…``) — the arm is anchored on the NAME half, which is
#: the half that decides.
#:
#: **The rule's cost is one class of token, and it is accepted deliberately.** A
#: tail that is lowercase words joined by separators is a NAME whatever separator
#: joins it, so an issuer tail spelled that way (``sk-lowercase-words-here``) is
#: left readable, where the underscore-only rule masked the dash-joined spelling of
#: it. Measured against ``origin/main``, 29 of the 42 lowercase-word spellings
#: across both prefix tables change verdict this way — 21 dash-joined, because the
#: old rule knew only underscores, and 8 underscore-joined under the FIXED table,
#: whose own ``_`` defeated its own NAME rule — while 0 tails carrying a digit or an
#: uppercase letter do. That second set is every real issuer token the corpus
#: knows, and no all-lowercase separator-carrying issuer tail has been observed
#: anywhere. The trade is taken because the alternative is the false-positive class
#: above, whose cost is not a missing mask but a manufactured compromise.
#: ``glpat-lowercase-token-value`` in the corpus pins this boundary, so a real
#: token of this shape would break a row rather than pass silently.
#:
#: A run of lowercase letters with NO separator is untouched by any of this: round
#: 1's own repro (``pk-`` plus sixteen letters) still masks.
#:
#: **One residual is recorded rather than closed, and it is narrower still.** The
#: ``\S*`` value arm takes anything after the ``=``, so a credential spelled
#: ``<prefix>-<lowercase words>=<secret>`` survives where the underscore-only rule
#: masked it. Measured: 0 occurrences across 2.6 GB of the fleet's transcripts, and
#: a tail carrying mixed case or a digit cannot reach the form. Closing it needs a
#: predicate on the VALUE half — a wider rule than this fix, with the same
#: false-positive risk on the other side (QA round 1, Q8).
#:
#: **``/`` is a separator here for the same reason ``-`` is, and it was measured.**
#: A path is a name spelled with slashes, and a vendor-looking prefix in front of one
#: is the commonest spelling of a repo slug. Measured 2026-09-21 against
#: ``origin/main``: the prose tail of a docstring in ``local_operator/providers/clients.py``
#: (line 2298) — an issuer prefix, then a lowercase org/repo path — was the one
#: string the pass masked in that whole file, and it is what a ``read`` of the file
#: reported as ``vendor-prefixed-token``, escalating a rotation demand for a
#: docstring mention of a model name on a call that read a source file. The tail
#: after the prefix is a lowercase name joined by a separator; ``/`` was simply
#: missing from the separator class, so the SAME slug survived when its separator
#: was written ``_`` or ``-`` — both already in the class — and masked only in the
#: slash spelling, which is the one a path is actually written with. The cost is the
#: class the two existing separator arms already accept: an issuer tail spelled as
#: all-lowercase words joined by ``/`` is left readable, and a real one would have
#: to be a run carrying no uppercase letter and no digit while containing a slash.
#: ``glpat-lowercase-token-value`` pins that boundary for the same reason.
#:
#: A tail ending in a DIGIT is still a token here (``xai``-prefixed org and repo
#: names that end in a version number included): the charset witnesses a token's
#: alphabet, and dropping the digit discriminator is the change this table already
#: measured and refused (see ``_VENDOR_TAIL``). That residual is recorded, not
#: closed.
_VENDOR_TAIL_IS_A_NAME = re.compile(r"[a-z]+(?:[-_/][a-z]+)+(?:=\S*)?")


def _vendor_tail_guard(match: Match[str]) -> bool:
    """Reject an issuer-looking prefix followed by an ordinary NAME.

    ``npm_config_update_notifier`` and ``docker_compose_build_args`` are
    environment variables: lowercase words joined by underscores. A real issuer
    tail is one unbroken run (``npm_<base64>``, ``docker_pat_…``), so what a NAME
    looks like is its negative space — ``_VENDOR_TAIL_IS_A_NAME`` is the predicate
    actually applied, and the token/name judgement is stated there rather than
    restated here. The rule's pattern cannot express a phrase without a
    variable-width look-behind, and a guard costs nothing on text the gate has
    already skipped.

    Both prefix tables strip their own separators before the test, because
    ``whsec``/``lin_api``/``pat_`` spell the separator INSIDE the prefix while
    ``npm``/``pypi`` spell it in the pattern: a name is a name under either, and
    the fixed table was the same defect left half-fixed.
    """
    tail = match.group(0)
    for prefix in _VENDOR_SEPARATED_PREFIXES:
        if tail.lower().startswith(prefix.lower()):
            tail = tail[len(prefix) :]
            break
    else:
        for prefix in _VENDOR_FIXED_PREFIXES:
            if tail.lower().startswith(prefix.lower()):
                tail = tail[len(prefix) :]
                break
    return not _VENDOR_TAIL_IS_A_NAME.fullmatch(tail.lstrip("_-"))


#: An ENVIRONMENT-VARIABLE (or secret-store NAME) spelling: capitals, digits and
#: underscores, no lower case. ``OS_PROD2_ADMIN_PASSWORD``, ``API_KEY``.
_ENV_NAME_SHAPED = re.compile(r"[A-Z][A-Z0-9_]*")


#: What ends an argument on a command line: whitespace, and the shell's separators.
#: Used to read the TOKEN an assignment sits inside, for the flag-argument check in
#: :func:`_is_a_credential_flags_argument`.
_ARGUMENT_BOUNDARIES = frozenset(" \t\r\n|;&")

#: The flag names that carry a credential, in ONE place. Three readers depend
#: on this vocabulary — the ``cli-credential-flag`` shape's own pattern, the
#: flag-before-an-argument check and the ``--flag=VALUE`` check — and three
#: copies of it drift, so a spelling added to one and not the others is a hole
#: nobody is looking at.
_CREDENTIAL_FLAG_WORDS = (
    r"(?:password|passwd|pwd|token|api[-_]?key|apikey|secret|"
    r"client[-_]?secret|auth[-_]?token|access[-_]?token)"
)

#: A credential FLAG immediately before the argument under test, built from the flag
#: rule's own vocabulary rather than retyped, in either separator spelling. The
#: joined spelling is deliberately absent: it puts the whole argument inside the flag
#: token, so an assignment can never begin inside it.
_CLI_CREDENTIAL_FLAG_BEFORE = re.compile(r"(?:^|[\s|;&])(?i:--" + _CREDENTIAL_FLAG_WORDS + r")\s+$")

#: The same flag read as an ASSIGNMENT's NAME. ``--secret=NAME`` binds the flag to
#: its argument with ``=``, so the assignment rules read ``--secret`` as the name
#: and the store's entry as the value; recognising the flag there is what gives the
#: ``=`` spelling the verdict the space spelling gets from
#: :func:`_flag_value_guard`.
_CLI_CREDENTIAL_FLAG_NAME = re.compile(r"(?i)^--" + _CREDENTIAL_FLAG_WORDS + r"$")


def _is_a_name_in_the_store_grammar(token: str) -> bool:
    """Whether ``token`` is spelled the way a stored secret's NAME is spelled.

    Caps, digits and underscores, with **at least one underscore**. That is the spelling
    :func:`local_operator.variables.normalize_credential_key` collapses a MULTI-WORD
    operator-typed key to — ``github token``, ``github-token`` and ``GITHUB_TOKEN`` are
    one entry named ``GITHUB_TOKEN`` — and the spelling ``lop secret run`` exports into a
    child's environment.

    **A ONE-WORD store name is left masked, and that residual is stated rather than
    implied** (agent review R1-4). ``normalize_credential_key("prod")`` is ``PROD``: one
    word collapses to a single run of capitals, which is exactly the spelling the next
    paragraph refuses, so ``lop secret run --secret prod`` is still masked and the
    operator who names an entry with one word does not get the release this change is
    for. ``PROD`` is pinned in the corpus as that residual, in the half that asserts the
    MASK, so a later round narrowing or widening it has a row to argue against. Case
    does not rescue it: a lower-case ``prod`` is refused for the separate reason below,
    and admitting a bare run of capitals released the five real credential values in
    this paragraph's next sentence.

    **The underscore is the measured floor, not a stylistic preference.** A single run
    of capitals is a credential someone chose, and agent review R1-1 measured that
    dropping the separator released ``--password PASSWORD``, ``--token TOKEN``,
    ``--api-key KEY``, ``--api-key APIKEY`` and ``--secret DBPASSWORD`` — silently,
    with no mask and no notice, because no hit means no labels and no exposure.

    Lower case is deliberately NOT admitted, and that is the arm this judgement refuses
    to widen: ``PASSWORD=correct_horse_battery`` is pinned in the corpus as a credential
    that must stay masked (the identifier arm's R1-1 class), and a lowercase identifier
    in a flag position is not distinguishable from it. The cost is a false positive on a
    store entry named in lower case — the store permits one, because
    :func:`local_operator.secrets.crypto.normalize_name` keeps case so ``token`` and
    ``TOKEN`` can coexist — and masking it is the direction this pass errs in.
    """
    return "_" in token and _ENV_NAME_SHAPED.fullmatch(token) is not None


def _value_is_a_reference_to_a_credential(value: str) -> bool:
    """Whether a flag's argument is the NAME of a stored credential, not one.

    Two spellings, both of them a NAME: the store's own
    (``--secret OS_PROD2_ADMIN_PASSWORD``) and the two-part one ``guide://credentials``
    teaches for handing that secret to a child
    under a different name (``--secret NPM_TOKEN=NODE_AUTH_TOKEN``).

    **Why a NAME no longer has to END in a credential word.** The previous predicate
    asked for that, and it is the defect this function no longer has. The judgement
    belongs to the argument's POSITION, not to the name's last word:
    :func:`local_operator.secrets.handlers._run` reads every token after ``--secret``
    as a key in the store (``retrieve_secret(name)`` is the lookup,
    ``environment[variable or name]`` the exported variable), and an operator names a
    store entry after the SYSTEM it belongs to rather than after the credential word —
    ``MINERVA_UI_NPROD_USERNAME`` names an account whose password lives elsewhere. Requiring
    ``PASSWORD``/``TOKEN``/``KEY`` at the tail masked that name in every tool result,
    and the masked text is what an agent copies: the operator authored a publish script
    from the displayed output and the script asked the store for a secret literally
    named ``[redacted]`` (2026-09-22, the reported failure; the guide's own command is
    pinned in the corpus's negative half).

    **The two-part form's right half is a NAME for the same reason.** The flag's grammar
    is ``NAME[=VAR]`` (``secrets/cli.py``'s ``metavar``), so both halves are references
    and neither has to end in a credential word. The halves are judged by the env-name
    SHAPE alone rather than by :func:`_is_a_name_in_the_store_grammar`, because a
    run-together name is the conventional spelling of the CHILD's variable — ``--secret
    OS_PROD2_ADMIN_PASSWORD=PGPASSWORD`` hands the store's entry to a ``pg_dump`` under the one name
    that tool reads.

    **One predicate, two rules** (agent review R1-1). It is factored out because the
    assignment rule sees the same text from inside: ``--secret NPM_TOKEN=NODE_AUTH_TOKEN``
    is also an assignment whose name is ``NPM_TOKEN`` and whose value is the other NAME, and a mask
    there files an ESCALATED rotation demand for the guide's own documentation.
    Whichever rule sees it must reach the same verdict, so they share the clause rather
    than each carrying a copy.

    **What this keeps masked, and the residual it accepts.** A real issuer token after
    the flag (``--secret ghp_…``) carries lower case and stays masked; a value with no
    separator (``--password hunter2xyz``) stays masked; a single run of capitals
    (``--secret DBPASSWORD``) stays masked; and the digit-free lowercase phrase R1-1
    pinned stays masked. The residual is a real credential spelled all-caps and
    underscore-separated (``--token ABC_123_XYZ``-shaped): that spelling IS what a store
    entry looks like and this arm cannot tell the two apart, so it is read as a NAME,
    released, and pinned as a corpus negative carrying that reason rather than left to a
    differential to discover.

    **The two-part residual is WIDER than that one, and it is pinned too** (agent review
    R1-3). Because the halves are read by the env-name shape ALONE, the two-part spelling
    does not need a separator in either half: ``--token ABCDEF=ABCDEF`` was masked before
    this change and is released by it, where the one-part ``--token ABCDEF`` is still
    masked for want of an underscore. That asymmetry is the grammar rather than an
    accident — the left half is a store entry's name, which need not carry a separator
    (``prod`` is a legal entry), and the right half is the child's variable, which
    conventionally does not (``PGPASSWORD``) — so it is stated and pinned as two corpus
    negatives, one with a separator in each half and one with none, instead of being
    narrowed into breaking ``--secret prod=PGPASSWORD``.
    """
    if _is_a_name_in_the_store_grammar(value):
        return True
    left, sep, right = value.partition("=")
    return bool(sep and _ENV_NAME_SHAPED.fullmatch(left) and _ENV_NAME_SHAPED.fullmatch(right))


def _is_a_credential_flags_argument(match: Match[str]) -> bool:
    """Whether this assignment is the NAME=VAR argument of a credential flag.

    ``lop secret run --secret NPM_TOKEN=NODE_AUTH_TOKEN -- <command>`` is the
    documented way to hand a stored secret to a child under a second name, and the
    assignment grammar sees the middle of it as an assignment. The flag rule already
    judges that argument a REFERENCE; this is how that verdict reaches the rule that
    would otherwise mask it, because a rejected span is re-scanned from one character
    in (see :func:`_apply_guarded`) and the second reading is an assignment.

    Measured 2026-09-21: without this, narrowing the identifier arm for R1-1 put the
    mask back on the guide's own example, and because ``_TOKEN`` is a six-character
    window of the neighbouring NAME the hit graded ``exposed`` — an ESCALATED
    rotation demand for a variable name, through a rule nobody had pointed at it.
    """
    text = match.string
    left = match.start(0)
    while left and text[left - 1] not in _ARGUMENT_BOUNDARIES:
        left -= 1
    right = match.end(0)
    while right < len(text) and text[right] not in _ARGUMENT_BOUNDARIES:
        right += 1
    if not _value_is_a_reference_to_a_credential(text[left:right]):
        return False
    return _CLI_CREDENTIAL_FLAG_BEFORE.search(text[:left]) is not None


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
    credential, never one — the same judgement this rule already made for the
    ``NAME[=VAR]`` form.

    **The bracket in that form was load-bearing, and it was not honoured.** The
    clause used to test the value as a SINGLE token, so the two-part spelling
    fell straight through it and was masked — the harness's own
    ``guide://credentials`` teaches that spelling as the way to rename a stored
    secret for a child, so following the documentation filed an incident.
    Measured 2026-09-21: a ``read`` of that guide masked the guide's own example
    and filed an ESCALATED rotation demand naming no shape at all.

    **The SEPARATOR is required, and that is the whole of the narrowing.** Capitals
    alone is not enough: it also describes exactly the values this rule exists to catch
    — ``--password PASSWORD``, ``--token TOKEN``, ``--api-key KEY``, ``--api-key
    APIKEY``, ``--secret DBPASSWORD`` — and a first cut that omitted the underscore
    stopped masking all five (agent review R1, reproduced through the session hook: the
    values came back byte-identical with no mask and no notice at all).

    **The credential-word TAIL is no longer required, and that requirement was the
    reported failure.** It masked ``--secret MINERVA_UI_NPROD_USERNAME`` in every tool result — a
    name that ends in the SYSTEM it belongs to rather than in a credential word — and
    the masked text is what an agent then copies: the operator authored a publish script
    from the displayed output and the script asked the store for a secret literally
    named ``[redacted]`` (2026-09-22). See
    :func:`_value_is_a_reference_to_a_credential` for the judgement that replaces it,
    for the value-side cases it keeps masked, and for the residual it accepts.
    """
    value = match.group(2)
    if any(char in value for char in _EXPRESSION_CHARS):
        return False
    if _value_is_a_reference_to_a_credential(value):
        return False
    return True


def _BARE_SCHEME_REPLACEMENT(match: Match[str]) -> str:
    """Keep the ``bearer `` keyword; mask the value."""
    return match.group(0)[: match.start(2) - match.start(0)] + REDACTION_MARKER


#: The letters an escape leaves glued to the front of a NAME (agent review R1-3/R1-4).
#: ``json.dumps`` is the live renderer (``local_operator/harness/redaction.py``): it
#: writes the two-character spellings for a newline, a carriage return, a tab, a
#: backspace and a form feed, and ``\uXXXX`` for everything outside ASCII — so a
#: payload carrying a raw U+2028, which ``str.splitlines`` treats as a break too,
#: arrives as its own six-character spelling rather than as itself. ``\xNN`` is the
#: same spelling at byte width, which a decoded-at-the-wrong-width payload carries.
#:
#: **A named assumption, not a closed set** (agent review R1-4). A renderer that
#: spelled a break some other way — percent-encoding, say — would reproduce the false
#: positive these helpers close, and it is recorded rather than guessed at because the
#: surfaces this pass runs on are both JSON: a tool call's arguments and a tool result.
_ESCAPE_LETTERS = re.compile(r"u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[nrtbf]")
#: The grouping is load-bearing: concatenation binds tighter than ``|``, so an
#: unparenthesised alternation would carry the backslash on its FIRST alternative
#: only and the rest would match a bare letter anywhere in the run.
_ESCAPED_BREAK = re.compile(r"\\(" + _ESCAPE_LETTERS.pattern + r")")


def _name_after_an_escape(match: Match[str]) -> str:
    """The name to judge, with the escape before it detached from its front.

    An assignment is scrubbed on more than one surface, and one of them is a
    RENDERING: a tool call is journaled as its JSON payload, where every newline
    inside a string arrives as the two characters ``\\`` and ``n``. The name group's
    class is ``[A-Za-z0-9_.\\-]``, so ``n`` is a name character to it, and a name
    that begins immediately after an escaped newline is therefore matched WITH the
    newline's own letter glued on. That is not a cosmetic difference: the count-trap
    exclusion (:func:`is_count_shaped`) keys on the FIRST segment of the name, so a
    count name arrives as itself with a stray ``n`` in front of it, the first
    segment stops being the count word, and the exclusion stops applying to exactly
    the construct it exists for.

    Measured 2026-09-21, and it is why this function exists: a ``write`` of ordinary
    Python source was flagged as carrying a credential, its value graded READABLE,
    and an escalated rotation notice filed for an assignment of a small integer
    constant, the file on disk holding no shape at all.

    **Only an ESCAPE's letters are detached, and only when the backslash is not
    itself escaped** (agent review R1-3, R2-F2). The first revision of this helper
    detached whatever followed a backslash, so a literal backslash before a
    single-segment credential name ate the name's own first letter, the name stopped
    being credential-shaped, and the mask was lost where ``origin/main`` had kept it.
    :data:`_ESCAPE_LETTERS` accepts only the escape spellings above, so a backslash
    that is not one changes nothing.

    The R1-3 answer left the other half of the same reading open, and R2-F2 measured
    it: a rendering writes a LITERAL backslash as TWO of them, so the character
    before the name is still a backslash and ``t`` is an escape letter — the strip
    ate it, ``oken`` is not a credential name, and the mask was gone with no hit and
    nothing registered, so no later exact-value pass could contain it either.
    Measured 2026-09-21: ``"\\\\" + "token=" + <value>`` was masked at
    ``origin/main`` and came back READABLE here. A doubled backslash is the one
    spelling where this reading is DECIDABLE — a backslash with a backslash before it
    is an escaped literal, and the letter after it is the name's own — so it is the
    one spelling where the answer may not be guessed.

    What stays undecidable is stated rather than smoothed over: an UNDOUBLED
    backslash before a name whose first letter is an escape letter IS that escape's
    spelling on the surfaces this pass reads (a tool call's arguments and a tool
    result are both JSON), so the name really does begin after it and the mask is
    correctly absent — the same reading that spares ``C:\\tokens``, where the
    pre-escape grammar masked a path segment. On a surface where such a backslash is
    a literal (a shell word) that is a loss, and it is the residual this helper
    cannot settle from the bytes at hand.

    The letters are read off the NAME rather than off the text before it, because the
    name group starts INSIDE the escape: its first character is the escape's own
    letter, and the backslash is the character before the match.

    An escape's letters belong to the escape, so they are removed before the name is
    judged. The mask does not move: only the VERDICT depends on the name, and for
    every name that is not count-shaped the stripped reading and the matched one
    agree.
    """
    name = match.group(1)
    start = match.start(1)
    # ``start < 2 or match.string[start - 2] != "\\"`` is the "the backslash is not
    # ITSELF escaped" half of the docstring's rule (agent review R2-F2): a doubled
    # backslash is how a rendering writes a literal one, so the name after it
    # follows a literal backslash and the letter at its front is its own.
    if start and match.string[start - 1] == "\\" and (start < 2 or match.string[start - 2] != "\\"):
        letters = _ESCAPE_LETTERS.match(name)
        if letters is not None:
            return name[len(letters.group(0)) :]
    return name


def _value_before_an_escape(value: str) -> str:
    """The value the JUDGEMENT reads: the run before the rendering's first break.

    The mirror of :func:`_name_after_an_escape`, and the division of labour between
    the judgement and the mask is the whole of it (agent review R1-2). The MASK
    keeps the entire run — the grammar runs across an escaped break, because a value
    whose own bytes carry one is a value the rendering only re-spelled, and
    stopping the mask there published its tail while the hit still graded
    ``complete=True``, and published it with no hit at all when the run before the
    break was shorter than the floor. The JUDGEMENT reads the run before the break,
    because that is the value as the operator wrote it: on a surface where newlines
    are newlines the same assignment's value stops at the same place, so the two
    surfaces agree on what the value IS, and the rendered one masks strictly more.

    Measured against ``origin/main``, in one process over the whole corpus: this
    releases nothing the truncated reading did not already release — the run before
    the break IS the value that reading judged — and it masks everything that
    reading masked, plus the tail it had left readable. 0 of the 346 pre-existing
    rows move.

    **Below the rule's own floor the escape is not a break this rule may trust.**
    A run shorter than :data:`_ASSIGNED_VALUE_MIN_CHARS` is not a value at all (it
    is EMPTY when the value opens with an escape), and the mask's floor is the
    evidence that what follows the escape is part of the value rather than the next
    line — the grammar could not have matched otherwise. So the whole run is judged,
    which is what ``origin/main`` did, and a value that opens with an escape keeps
    its mask.
    """
    before = _ESCAPED_BREAK.split(value, maxsplit=1)[0]
    if len(before) >= _ASSIGNED_VALUE_MIN_CHARS:
        return before
    return value


def _assignment_value_guard(match: Match[str]) -> bool:
    """The checks both spellings of the named-assignment rule share.

    The marker check is the "do not mask twice" half: rules run in order, so a
    DSN inside ``MONGO_DSN=`` is masked by the DSN rule first — password gone,
    user and host kept readable — and this rule would otherwise match the same
    value again and swallow the whole thing, taking back the host it had just
    preserved and splitting the marker on the ``]`` the value class stops at.
    Measured while building the table: the unguarded pair produced
    ``MONGO_DSN=[redacted]]@host``.

    The rest is the name/value judgement, and none of it depends on which
    spelling matched: a credential NAME that is not a count, a value that looks
    like a credential rather than an expression, and not a keyword argument.
    Everything below the name check is a false positive that rewrites ordinary
    code (see ``_looks_like_an_expression`` for the census that forced it).
    """
    if REDACTION_MARKER in match.group(4):
        return False
    name = _name_after_an_escape(match)
    if not is_credential_name(name) or is_count_shaped(name):
        return False
    # ...and a NAME sitting in a credential FLAG's argument is the FLAG rule's
    # judgement, not this one's: both rules look at the same bytes, and the second
    # reading of a span the flag rule rejected is an assignment (agent review R1-1,
    # see :func:`_is_a_credential_flags_argument`).
    if _is_a_credential_flags_argument(match):
        return False
    # ...and a credential FLAG's argument is a NAME this rule must not judge: on
    # ``--secret=NAME`` the text left of the ``=`` is the flag itself, so this rule
    # reads ``--secret`` as the assignment's NAME and the store's entry as its VALUE.
    # Refusing here — and only when the value is name-shaped — hands the ``=``
    # spelling the verdict :func:`_flag_value_guard` reaches for the space spelling
    # (one verdict per argument, whichever character binds it) and keeps a
    # VALUE-shaped argument on this rule's own path, which is why a token after
    # ``--token=`` still masks and still carries the flag rule's label. See
    # :func:`_value_is_a_reference_to_a_credential` for the failure that motivated it.
    if _CLI_CREDENTIAL_FLAG_NAME.match(name) and _value_is_a_reference_to_a_credential(
        match.group(4)
    ):
        return False
    # The VALUE is read the way the NAME is: the rendering's line breaks are line
    # breaks for the judgement too, so what is judged is the run before the first
    # of them while the mask covers the whole run (see
    # :func:`_value_before_an_escape`).
    if _value_is_not_a_credential(
        _value_before_an_escape(match.group(4)),
        name=name,
        strong=is_strong_credential_name(name),
    ):
        return False
    return not _is_keyword_argument(match)


def _assignment_guard(match: Match[str]) -> bool:
    """The guard of the UNQUOTED spelling: a quoted value is never this rule's.

    The delegation is the fix for the over-masking measured on the operator's own
    transcripts (2026-09-20). This grammar's value class is greedy to a delimiter
    and a quote IS a delimiter, so on compact JSON the run crossed the closing
    quote into the NEXT FIELD: an ``access_token``/``refresh_token`` pair in one
    object matched the first value PLUS the neighbouring key and value, which
    masked the neighbour's KEY (over-masking, the defect the negative half of the
    corpus exists to prevent) and then, quite correctly for that match, graded a
    fragment of the swallowed text as exposed and filed a rotation demand for a
    credential that had been covered whole.

    A quoted value therefore belongs to the QUOTED spelling of this same rule
    (the second ``credential-assignment`` entry in the table), which is the only
    one of the two that can find where a quoted value ENDS. The
    unquoted spelling keeps this grammar unchanged: a value with no delimiter to
    stop at must keep the length bound and the terminating-delimiter rule, and
    narrowing it would publish the tail of a credential containing a quote — the
    corpus case whose reason is "a quote inside an unquoted value".
    """
    if match.group(3):
        return False
    return _assignment_value_guard(match)


def _assignment_guard_quoted(match: Match[str]) -> bool:
    """The guard of the QUOTED spelling: group 3 is the delimiter, not a refusal.

    Written out rather than aliased to ``_assignment_guard`` because the
    delegation there is exactly what must not happen here — every match of this
    rule has a non-empty group 3 by construction, so the shared body is the whole
    of the judgement.
    """
    return _assignment_value_guard(match)


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


#: One NUMBERED unit of a tool's line numbering: an optional opening bracket, the digit
#: run, an optional closing bracket, whitespace, and an optional separator.
#:
#: It is a NAMED fragment rather than inline text because the `-open` rule needs a second
#: spelling of this unit (`_PEM_PREFIX_UNIT_MAX`, below), and a second hand-written copy
#: of a grammar fragment is exactly how the shape table and the pipe's classifier drift
#: apart — the defect Q9-F1/Q10-F1 recorded above, one level down.
_PEM_PREFIX_UNIT = r"\[?\d+\]?[ \t]*(?:(?:[.)\]]|\.\]|->|[|:>-])[ \t]*)?"

#: bat's boxed numbering (`│ 12 │`). Its digit run is delimited on BOTH sides, so it has
#: no partial form to spell: no unit and no content can extend into it.
_PEM_PREFIX_BOX_UNIT = r"\u2502[ \t]*\d+[ \t]*\u2502[ \t]*"

#: The line-number prefix tools actually emit, as ONE definition shared by the shape
#: table and the pipe's own body classifier (``tools/builtin.py`` imports these). A second
#: hand-written allowance is what published the body for `cat -n` output after the shape
#: rule had been fixed (Q10-F1): the pipe masks BEFORE the table ever runs, so a divergence
#: between the two is a silent leak rather than a missed match.
#:
#: Covered: `12|`, `12:`, `12>`, `12->`, `12)`, `12.]`, `[12]`, `12<TAB>` (cat -n), bat's
#: `│ 12 │`, any of them repeated (`3| 4| …`), with spaces or a TAB around the separator,
#: and it is built from the two units above so that a third spelling (this rule's maximal
#: one) cannot be written by hand beside it.
LINE_PREFIX = (
    r"[ \t]*(?:(?:"
    # THE SEPARATOR'S TRAILING WHITESPACE LIVES INSIDE THE OPTIONAL GROUP, and that
    # is the difference from `[ \t]*(?:SEP)?[ \t]*`. Both accept exactly the same
    # strings (`[ \t]*` | `[ \t]*SEP[ \t]*`), but this one consumes a run of
    # whitespace in ONE place per unit instead of two, and that removes a whole
    # family of partition paths: the old spelling could split a gap between two
    # digit runs across the trailing and the leading `[ \t]*`, so 2**k paths for
    # k digit runs on a line. Measured on the released fragment: one 65-character
    # numeric table row (`%8d`-padded columns, i.e. a numpy row or a padded column
    # dump) cost 1.34 s in `PEM_HEADER_LINE_RE.search` and 1.74 s in
    # `PEM_BODY_LINE_RE.match`; 48 characters of space-separated 2-digit runs did
    # not return in 190 s. This spelling: 12 ms and 14 ms for the same 65
    # characters.
    #
    # IT DOES NOT REMOVE THE EXPONENTIAL, and the comment says so because the next
    # reader will otherwise "simplify" one of two things back. The remaining paths
    # are the DIGIT runs: a unit may end in the middle of `\d+` (nothing forbids it
    # when no separator follows), and every split is live, so the engine still
    # walks 2**k for k digit runs. THREE OTHER language-preserving spellings were
    # measured alongside this one — whitespace made maximal with a `(?![ \t])`
    # assertion, possessive `[ \t]*+`, and both together — and all four still grow
    # by a factor per digit run. Two narrower spellings are NOT available: making `\d+`
    # possessive or forbidding a unit to end before a digit both DROP matches
    # (`12 34 MIIEowIBAAKCA` is a doubly-numbered body line that the second loses
    # outright), and a dropped body line is a published key, which is the one
    # failure this grammar must not have. The cost is therefore bounded by a
    # number of partitions INHERENT to the language, and the fix for a caller that
    # runs it over arbitrary text is a guard or a linear matcher, not a cheaper
    # fragment: `local_operator/tools/builtin.py` shields its two hot call sites
    # with necessary-condition gates and records the residual.
    + _PEM_PREFIX_UNIT
    + r"|"
    + _PEM_PREFIX_BOX_UNIT
    + r")+)?"
)

#: A line separator in either spelling: escaped (inside a JSON value) or real, CRLF
#: included.
LINE_SEP = r"(?:\\r\\n|\\n|\r\n|\n|\r)"

#: The body grammar's FLOOR: how many class characters a body line needs to stand on its
#: own. It is a NAME because three readers ask three different questions of it — the body
#: grammar accepts at exactly this width, `tools/builtin.py` HOLDS a cap-forced cut back
#: to the line boundary by at most `PEM_BODY_FLOOR - 1` bytes when a release can end in
#: the MIDDLE of a line, and this branch's linear deciders
#: (`pem_body_line` / `pem_end_line` / `pem_header_line_end`) read it for both arms of
#: the grammar they decide. `#1445` landed the constant and the hold (#1445's own comment
#: carried the note that this branch would be the third reader, spelling an `8` / `7`
#: pair by hand); the deciders read `PEM_BODY_FLOOR` and `PEM_BODY_FLOOR - 1` now, because
#: a decider a byte BELOW the hold's floor under-holds — the fragment is then read as
#: PROSE, which closes the block and publishes what follows, which is the leak direction.
PEM_BODY_FLOOR = 8

#: The floor's two spellings, as the regexes need them and built from the number above:
#: a MINIMUM run (a line that stands on its own) and a run bounded one below it (the
#: truncated-line allowance). Not restated as a literal anywhere below.
_PEM_FLOOR_MIN_RUN = "{" + str(PEM_BODY_FLOOR) + ",}"
_PEM_FLOOR_SUB_RUN = "{1," + str(PEM_BODY_FLOOR - 1) + "}"

#: One PEM body line with that prefix. `PEM_BODY_FLOOR` characters is the floor for a
#: line that stands on its own; a SHORTER line counts only when a full one follows it (a
#: truncated run) or when it is the block's last line before the closing quote or the end
#: of the text — inside an open block nothing may be published, which is the block's
#: whole point (Q10-F2: a sub-eight-character line in the MIDDLE published everything
#: after it, and a short FINAL line published where the previous head masked).
_PEM_FULL_LINE = r"[A-Za-z0-9+/=]" + _PEM_FLOOR_MIN_RUN + r",?[ \t]*"
_PEM_SHORT_MID_LINE = (
    r"[A-Za-z0-9+/=]"
    + _PEM_FLOOR_SUB_RUN
    + r",?[ \t]*(?="
    + LINE_SEP
    + LINE_PREFIX
    + r"[A-Za-z0-9+/=]"
    + _PEM_FLOOR_MIN_RUN
    + r")"
)
#: A short line is a body line when a full one FOLLOWS it, and the run may end with one
#: short line. A lone short line — `12| done`, `12| 42` — is numbered PROSE and must
#: survive, which is why the end-of-run allowance is not a free-standing alternative.
_PEM_LINE_CONTENT = _PEM_FULL_LINE + r"|" + _PEM_SHORT_MID_LINE

#: The `-open` rule's OWN body grammar: the two fragments above with every digit run
#: required to be MAXIMAL (`\d+(?!\d)`).
#:
#: WHY IT EXISTS. This is the one rule in the table that runs the prefix fragment
#: over text nobody shaped — a tool result, a grep hit, a README quoting a banner —
#: and `LINE_PREFIX`'s digit runs are ambiguous BY CONSTRUCTION (see the fragment's
#: own comment: `12` is one unit or two, and every split is a live path). For the
#: pipe's classifiers that ambiguity is bounded by the linear deciders; inside a
#: PATTERN it is a backtracking walk of 2**(digits-1) paths per digit run, and the
#: walk is EXHAUSTED rather than pruned whenever a body line does not parse — so the
#: cost is not a constant factor away, it is unbounded in the input. Measured
#: through the real `_PipeRedactor` (QA round 2, Q3 on #1427): an anchored
#: `"private_key": ` spelling plus a header phrase plus ONE `%8d` row of four-digit
#: columns costs 23 s of CPU at six columns and runs past a 60 s cap at eight, and
#: the 140-row payload that `_release_point` hands this rule once a read exceeds the
#: deferral cap is past 60 s at `bdf3b6cd` (past 45 s at the base revision, so the
#: worst of it is pre-existing rather than this branch's). Nothing is masked in that
#: shape, so the whole cost is pure.
#:
#: WHY IT SPELLS THE SAME LANGUAGE, which is the obligation a second spelling of a
#: language takes on. A split inside a CONTIGUOUS digit run always has an equivalent
#: one-unit parse: the characters between the two chunks are the unit's own `\]`,
#: whitespace and separator, and all three are epsilon exactly when the next
#: character is still a digit — so extending the run by one digit and re-parsing
#: leaves the rest of the line untouched, and by induction every split collapses to
#: the maximal run. What DOES drop strings is a restriction on the unit END (the
#: fragment's comment cites `12 34 MIIEowIBAAKCA`, which needs a unit to end
#: immediately before `34`), and that is a different change. Both directions are
#: enumerated rather than argued:
#: `test_the_digit_maximal_prefix_spells_the_released_fragment_language` sweeps the
#: two fragment languages against each other in both directions, and
#: `test_the_open_rule_matches_its_released_spelling_on_every_fixture` compares the
#: two RULES' matches (start, end, group 1, group 2) over the corpus, the dense
#: payloads and a generated sweep — a divergence either way is a released mask or a
#: published body line, so neither direction may be sampled.
#:
#: SCOPE, and it is deliberately narrow: THIS rule only. `LINE_PREFIX` itself and the
#: three `PEM_*_RE` patterns keep the released spelling, because the pipe's deciders
#: (`_pem_prefix_end_flags`, and the three classifiers built on it) mirror THAT
#: fragment and are pinned against those patterns, while those three patterns are
#: the DEFINITION of their languages rather than a hot path — the deciders decide
#: them. The other two block rules (`pem-private-key`, `gcp-service-account-key`) do
#: not embed the fragment at all, and the fragment alone decides nothing here: it is
#: the ambiguity's combination with this rule's unbounded body run that made the
#: engine walk 2**(digits-1) partitions per digit run.
#: The two units above with every digit run required to be MAXIMAL.
#:
#: The box unit is maximalised only for symmetry with the fragment it mirrors: its run is
#: delimited by `│` on both sides, so it can never be split and the predicate changes
#: nothing it matches.
_PEM_PREFIX_UNIT_MAX = _PEM_PREFIX_UNIT.replace(r"\d+", r"\d+(?!\d)")
_PEM_PREFIX_BOX_UNIT_MAX = _PEM_PREFIX_BOX_UNIT.replace(r"\d+", r"\d+(?!\d)")

#: The `-open` rule's prefix: maximal units, THEN AT MOST ONE RELEASED UNIT — and that
#: trailing unit is what makes this spelling the released LANGUAGE rather than a subset
#: of it (agent review R3-1 on #1427).
#:
#: WHY A UNIT MUST BE ALLOWED TO STOP MID-RUN. `\d+(?!\d)` on EVERY unit is not the same
#: language, because the released grammar lets a unit's tail be epsilon — `\]?`, the
#: whitespace and the separator are all optional — exactly when the next character is
#: still a digit, and the BODY CONTENT may then consume the rest of that run. The witness
#: is a body line of `[` followed by nine `1`s under an anchored header: `origin/main`
#: parses it as the unit `[1` plus the content `11111111` and masks the line (span
#: `(0, 54)`); a fully maximal spelling must eat all nine digits as the unit's run, leaves
#: no eight-character content behind it, and PUBLISHES the line — measured `(0, 43)`, i.e.
#: a released mask lost, which is the one direction this table may never move.
#:
#: WHY EXACTLY ONE, AND WHY LAST. Any split of a CONTIGUOUS run into several units is
#: redundant — collapsing it into one unit does not move where the prefix ENDS — so the
#: only split that can matter is the one that decides where the content begins, and that
#: split is by definition in the LAST unit's run: the released parse's prefix end is
#: either a maximal-run end (those are `*` above) or a position strictly inside the run
#: that the content immediately follows (this trailing unit). Allowing the released unit
#: ONCE, at the END, is therefore the whole difference — and it is what keeps the walk
#: linear: per run the engine has one maximal unit to try, and the trailing unit adds a
#: bounded walk of the FINAL run only, instead of a partition of every run.
#:
#: The full enumeration of both directions is pinned by
#: `test_the_open_rule_restores_the_released_rules_bracket_family` (the witness class and
#: its neighbours, against the independently rebuilt released rule) and by
#: `test_the_open_rule_matches_its_released_spelling_on_every_fixture` (the corpus and a
#: generated sweep, whose alphabet carries `[`).
_PEM_DIGIT_MAX_PREFIX = (
    r"[ \t]*(?:(?:"
    + _PEM_PREFIX_UNIT_MAX
    + r"|"
    + _PEM_PREFIX_BOX_UNIT_MAX
    + r")*(?:"
    + _PEM_PREFIX_UNIT
    + r")?)?"
)
_PEM_DIGIT_MAX_LINE_CONTENT = (
    _PEM_FULL_LINE + r"|" + _PEM_SHORT_MID_LINE.replace(LINE_PREFIX, _PEM_DIGIT_MAX_PREFIX)
)
#: NIT-2 (agent review round 3): UNREFERENCED AT RUNTIME — this is the pipe's pre-decider
#: block spelling, and the deciders (`pem_body_line` / `pem_end_line` / `pem_header_line_end`)
#: replaced every caller, so nothing reaches it. Kept rather than deleted, and recorded
#: rather than left implicit: #1445 rewrites the `{1,7}` in its last line, so a deletion
#: here is a merge surface for no gain. Deleting it is a follow-up once both land.
_PEM_RUN = (
    r"(?:" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")"
    r"|" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")?)*"
    r"(?:" + LINE_PREFIX + r"[A-Za-z0-9+/=]" + _PEM_FLOOR_SUB_RUN + r",?)?"
)
PEM_BODY_LINE_RE = re.compile(r"^" + LINE_PREFIX + r"(?:" + _PEM_LINE_CONTENT + r")$", re.MULTILINE)
#: The ARMOUR TAIL: what may follow the closing run of dashes on a BEGIN/END armour
#: line. Trailing space or TAB (an editor's, a wiki's, a CRLF file's carriage return),
#: then that line's own end — the carriage return of a CRLF or bare-CR terminator, the
#: line feed, or the end of the text.
#:
#: ONE DEFINITION, READ BY BOTH THE PATTERN AND THE DECIDER — and that is not tidiness.
#: `#1445` landed this tail on the pattern (`$` alone cannot match in front of a `\r`, so a
#: CRLF, bare-CR or trailing-whitespace armour line was not a header at all); this branch
#: owns the DECIDER, and while a decider models a tail by hand the two can answer the same
#: question differently — which is exactly what happened when the pattern widened and the
#: decider kept `$`: the pipe then PUBLISHED the block's body (measured on the previous
#: head through the real filter: 3 of 3 body lines out, no marker, for an unterminated CRLF
#: block and for a trailing-whitespace one). A documentary obligation ("keep the tail in
#: step with the pattern's") is what failed there, so the decider DECODES its two halves
#: from this text below rather than restating them, and
#: `test_the_header_deciders_tail_is_the_patterns_tail` holds the composition together.
_PEM_ARMOUR_TAIL_RUN_TEXT = r" \t"
_PEM_ARMOUR_TAIL_TERMINATOR_TEXT = (r"\r", r"\n")
_PEM_ARMOUR_TAIL = (
    "[" + _PEM_ARMOUR_TAIL_RUN_TEXT + "]*(?=" + "|".join(_PEM_ARMOUR_TAIL_TERMINATOR_TEXT) + "|$)"
)
#: The same two halves as the CHARACTERS the linear decider scans for, decoded from the
#: text above — so a decider that could disagree with the pattern is not something a reader
#: has to notice, it is something they would have to construct.
_PEM_ARMOUR_TAIL_RUN = codecs.decode(_PEM_ARMOUR_TAIL_RUN_TEXT, "unicode_escape")
_PEM_ARMOUR_TAIL_TERMINATORS = "".join(
    codecs.decode(spelling, "unicode_escape") for spelling in _PEM_ARMOUR_TAIL_TERMINATOR_TEXT
)

PEM_HEADER_LINE_RE = re.compile(
    r"^" + LINE_PREFIX + r"-{1,4}[\x27\x22]?-{1,4}BEGIN [A-Z0-9 ]*PRIVATE KEY"
    # The NAMED tail below is the CRLF / CR / trailing-whitespace spelling
    # of the armour line, and it is not cosmetic either: ``$`` alone cannot match in
    # front of a ``\r``, so ``...KEY-----\r\n`` — an ordinary key file written on
    # Windows, or quoted by a wiki, or left with a trailing space by an editor — was not
    # a header here at all. The shape table's ``pem-private-key`` needs a COMPLETE
    # BEGIN … END, so for an unterminated view the pipe's mask is the only layer that can
    # hide the body, and for that spelling it never engaged: measured through the real
    # tool, ``head -n 6`` on a complete CRLF key published 5 of its 25 body lines, and an
    # unterminated CRLF block published 137 body lines on the transcript, 200 in the raw
    # spill and 129 over ``read spill://`` (identical at the base — pre-existing, and
    # this fix closes it for the ordinary spelling rather than licensing it).
    #
    # The terminator is TOLERATED BY LOOKAHEAD rather than consumed, which is the
    # spelling that covers all three of them: a bare CR line ending (a key file that
    # came off an old Mac, or through a filter that normalised to CR) satisfies ``\r``
    # with nothing after it, where ``\r?$`` could not — the optional CR is backtracked
    # away and ``$`` then has no ``\n`` to sit in front of. Leaving the terminator
    # outside the match is also what the mask wants: its caller hands the bytes after
    # the match to ``_PEM_LINE_BREAK``, which emits the separator verbatim.
    #
    # Widening the TAIL is the whole of the change, and it cannot open a block on text
    # that is not an armour line: the phrase (``BEGIN `` … ``PRIVATE KEY``) and both
    # dash runs are untouched, a line still has to be that WHOLE line (trailing prose
    # after the dashes does not match), and the prefix grammar is unchanged. That is
    # why the over-mask cost of this decision is measured at ZERO rather than asserted:
    # across the 423-case shape corpus and all 648 of the repository's source and docs
    # files, NOT ONE line is newly classified as a header (the corpus's own armour lines
    # matched under the old tail too, and no repository file carries one at all), so no
    # text changes its pipe output. The decision was still made in the mask-more
    # direction — a real key file with CRLF endings masks now, and it published its body
    # before.
    r"-{1,4}[\x27\x22]?-{1,4}" + _PEM_ARMOUR_TAIL,
    # MULTILINE, and that is not cosmetic: the pipe layer SEARCHES a multi-line read for
    # this header, so without the flag it matched only when the read was exactly one
    # header line — which the release point's hold makes impossible — and the entire
    # body-masking path was unreachable (R11-1: `head -n 6 key.pem` published 5 of 25
    # body lines through the real tool call while the unit suite stayed green).
    re.MULTILINE,
)
PEM_END_LINE_RE = re.compile(r"^" + LINE_PREFIX + r"-{1,4}[\x27\x22]?-{1,4}END ", re.MULTILINE)


# --- the LINEAR decision procedures for the three patterns above -------------
#
# WHY THESE EXIST — and it is the one part of the pipe filter's cost that no
# necessary-condition guard can reach. `LINE_PREFIX` is AMBIGUOUS BY
# CONSTRUCTION: a unit's `\d+` may end in the middle of a digit run (`12` is one
# unit or two, and every split is a live path), so a line with k digit runs
# walks 2**k partitions, and `re` has no memoisation to cut them down. Measured
# on this tree at the classifier call sites in `tools/builtin.py`: ONE
# 65-character `%8d` table row (`%8d`-padded columns — a numpy row, a padded
# column dump) costs 6.7 s in `PEM_BODY_LINE_RE.match` and 3.9 s in
# `PEM_HEADER_LINE_RE.search`; a 3.3 KB read that merely QUOTES a banner costs
# 1.86 s, of which 1814 ms is the body classifier; a 6.5 KB read runs past
# 120 s. `tools/builtin.py` gates what it can, and the gates are what make
# ordinary output free, but they cannot reach this: a read that carries the
# literals OPENS the state, and the body classifier is then asked about lines it
# genuinely MATCHES — `10000000 10000001` is a numbered body line — so no
# cheaper test rejects the pathological line.
#
# WHAT THEY ARE. The same three languages, decided by an explicit mode-set
# simulation of the prefix grammar — the modes are positions in the grammar, one
# character moves the whole set on, and nothing is ever revisited — followed by
# the rigid literal tail each pattern requires, whose candidate offsets are
# ENUMERATED (the tail is at most nine characters, so there are at most a
# handful) instead of searched. Cost is O(len) per call with a small constant,
# for every input shape.
#
# THE EQUIVALENCE OBLIGATION, stated plainly because it is the whole risk of a
# second spelling of a language that was already written once. A divergence has
# two directions and only one of them survives review: a decider that ACCEPTS
# where the pattern does not masks text the agent needed to read (a defect, per
# the module docstring), while a decider that REJECTS where the pattern accepts
# DROPS A BODY LINE — a published key, silently, because the line loop also
# closes the state on it. So the deciders are pinned against the patterns
# THEMSELVES rather than against hand-written expectations:
# `tests/unit/secrets/test_credential_shapes.py` sweeps every character-KIND
# sequence up to a bounded length (these languages are functions of the
# character kind, which is what lets a bounded sweep stand for longer strings,
# and one arm there pins that digit predicate against `\d` over the whole code
# space) and then fuzzes the shapes a caller actually sees. `PEM_BODY_LINE_RE`,
# `PEM_HEADER_LINE_RE` and `PEM_END_LINE_RE` REMAIN THE DEFINITION of the
# language; the functions below are how the hot path decides it.

_PFX_BETWEEN = 1 << 0  # between units: a unit may start, whitespace may run, the prefix may END
_PFX_OPEN = 1 << 1  # after the `[` of a unit: a digit must follow
_PFX_DIGITS = 1 << 2  # inside a unit's digit run
_PFX_CLOSED = 1 << 3  # after digits + the closing `]`
_PFX_DIGITS_WS = 1 << 4  # after digits (+ `]`) + whitespace
_PFX_SEP = 1 << 5  # after a separator (and any whitespace that followed it)
_PFX_SEP_DOT = 1 << 6  # after a `.` separator: `]` may extend it (the `.\]` spelling)
_PFX_SEP_DASH = 1 << 7  # after a `-` separator: `>` may extend it (the `->` spelling)
_PFX_BOX_OPEN = 1 << 8  # after the opening `│` of `│ 12 │`
_PFX_BOX_WS = 1 << 9  # after `│` + whitespace
_PFX_BOX_DIGITS = 1 << 10  # inside the box form's digit run
_PFX_BOX_DIGITS_WS = 1 << 11  # after the box form's digits + whitespace
_PFX_BOX_CLOSED = 1 << 12  # after the box form's closing `│`

#: Modes in which a UNIT HAS JUST COMPLETED. From any of them the grammar allows
#: the prefix to end, and it also allows a new unit to start on the very next
#: character (`12` is two units as readily as one) — the epsilon edge that
#: `_pem_prefix_end_flags` applies after every step, which is why the machine
#: needs no separate "between units" transition per mode.
_PFX_COMPLETE = (
    _PFX_DIGITS
    | _PFX_CLOSED
    | _PFX_DIGITS_WS
    | _PFX_SEP
    | _PFX_SEP_DOT
    | _PFX_SEP_DASH
    | _PFX_BOX_CLOSED
)

#: THE MACHINE, as data: for each character kind, the `(mode, mode-after-it)`
#: pairs. Written from the fragment rather than derived from it, and that is the
#: point — a derivation would be the same expression rewritten, which is what
#: the algebraic rewrites in this file's history were, and three of them were
#: language-changing. `tools/builtin.py` does not use this machine; the patterns
#: above stay the definition, and the test file pins the two together.
_PFX_STEP_WS = (
    (_PFX_BETWEEN, _PFX_BETWEEN),
    (_PFX_DIGITS, _PFX_DIGITS_WS),
    (_PFX_CLOSED, _PFX_DIGITS_WS),
    (_PFX_DIGITS_WS, _PFX_DIGITS_WS),
    (_PFX_SEP, _PFX_SEP),
    (_PFX_SEP_DOT, _PFX_SEP),
    (_PFX_SEP_DASH, _PFX_SEP),
    (_PFX_BOX_OPEN, _PFX_BOX_WS),
    (_PFX_BOX_WS, _PFX_BOX_WS),
    (_PFX_BOX_DIGITS, _PFX_BOX_DIGITS_WS),
    (_PFX_BOX_DIGITS_WS, _PFX_BOX_DIGITS_WS),
    (_PFX_BOX_CLOSED, _PFX_BOX_CLOSED),
)
_PFX_STEP_DIGIT = (
    (_PFX_BETWEEN, _PFX_DIGITS),
    (_PFX_OPEN, _PFX_DIGITS),
    (_PFX_DIGITS, _PFX_DIGITS),
    (_PFX_BOX_OPEN, _PFX_BOX_DIGITS),
    (_PFX_BOX_WS, _PFX_BOX_DIGITS),
    (_PFX_BOX_DIGITS, _PFX_BOX_DIGITS),
)
_PFX_STEP_OPEN = ((_PFX_BETWEEN, _PFX_OPEN),)
#: `]` is both the unit's closing bracket and one of the separators, so it lands
#: in both (`1]2` is a bracketed unit then a new one, `1]` a unit whose separator
#: is `]`), and it is the second half of the `.\]` spelling.
_PFX_STEP_CLOSE = (
    (_PFX_DIGITS, _PFX_CLOSED | _PFX_SEP),
    (_PFX_CLOSED, _PFX_SEP),
    (_PFX_DIGITS_WS, _PFX_SEP),
    (_PFX_SEP_DOT, _PFX_SEP),
)
#: `.` and `-` carry their longer spelling as well as themselves: `1.]` and `1->2`
#: are real, and the spelling that dropped them measured 146/1140 lost strings.
_PFX_STEP_DOT = (
    (_PFX_DIGITS, _PFX_SEP | _PFX_SEP_DOT),
    (_PFX_CLOSED, _PFX_SEP | _PFX_SEP_DOT),
    (_PFX_DIGITS_WS, _PFX_SEP | _PFX_SEP_DOT),
)
_PFX_STEP_DASH = (
    (_PFX_DIGITS, _PFX_SEP | _PFX_SEP_DASH),
    (_PFX_CLOSED, _PFX_SEP | _PFX_SEP_DASH),
    (_PFX_DIGITS_WS, _PFX_SEP | _PFX_SEP_DASH),
)
#: `)`, `|` and `:` have no longer spelling; `>` has one, only after `-`.
_PFX_STEP_SEP = (
    (_PFX_DIGITS, _PFX_SEP),
    (_PFX_CLOSED, _PFX_SEP),
    (_PFX_DIGITS_WS, _PFX_SEP),
)
_PFX_STEP_GT = _PFX_STEP_SEP + ((_PFX_SEP_DASH, _PFX_SEP),)
#: The box form has no separator and its digits are optional only in the sense
#: that `│` may be followed by whitespace: `│ 12 │`, `│12│` and `│ 12│` all read.
_PFX_STEP_BOX = (
    (_PFX_BETWEEN, _PFX_BOX_OPEN),
    (_PFX_BOX_DIGITS, _PFX_BOX_CLOSED),
    (_PFX_BOX_DIGITS_WS, _PFX_BOX_CLOSED),
)
_PFX_WS = " \t"
_PFX_SINGLE_SEP = ")|:"

#: The two character classes the deciders check by name, spelled once here. They
#: are the patterns' own classes (`[A-Za-z0-9+/=]` for a body token, `[A-Z0-9 ]`
#: between a header's literals); a change to either pattern that this does not
#: follow shows up as a divergence in the differential arms rather than as a
#: quietly different answer on the hot path.
_PEM_TOKEN_CLASS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
_PEM_HEADER_CLASS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
_PEM_BEGIN = "BEGIN "
_PEM_KEY = "PRIVATE KEY"
_PEM_END = "END "


def _pem_prefix_end_flags(text: str) -> bytearray:
    """`flags[i]` is 1 exactly when `text[:i]` is in the `LINE_PREFIX` language.

    The whole decision in one left-to-right pass: `modes` is the set of grammar
    positions the prefix could be in after `i` characters, and the prefix is in
    the language exactly when `_PFX_BETWEEN` is in that set — nothing is pending,
    so what was consumed is `[ \\t]*` followed by a complete `(unit)+`.

    AN EMPTY MODE SET IS FINAL, which is what makes this cheap on ordinary text:
    nothing in the fragment consumes a newline, a letter or a `#`, so a prose
    line kills the set within a few characters and the loop stops — the flags
    beyond that point are already zero. That is also why the flags array is
    written by index instead of built positionally.
    """
    flags = bytearray(len(text) + 1)
    modes = _PFX_BETWEEN
    flags[0] = 1
    for index, char in enumerate(text):
        if char in _PFX_WS:
            steps = _PFX_STEP_WS
        elif char.isdecimal():
            # `\d`, not `[0-9]`: the fragment's class is Unicode-decimal, and a
            # hand-written ASCII class here would reject a body line the pattern
            # accepts. `isdecimal()` is the same predicate `\d` uses (the test
            # file checks the two against each other over the whole code space).
            steps = _PFX_STEP_DIGIT
        elif char == "]":
            steps = _PFX_STEP_CLOSE
        elif char == ".":
            steps = _PFX_STEP_DOT
        elif char == "-":
            steps = _PFX_STEP_DASH
        elif char == "\u2502":
            steps = _PFX_STEP_BOX
        elif char == ">":
            steps = _PFX_STEP_GT
        elif char == "[":
            steps = _PFX_STEP_OPEN
        elif char in _PFX_SINGLE_SEP:
            steps = _PFX_STEP_SEP
        else:
            break
        reached = 0
        for mode, after in steps:
            if modes & mode:
                reached |= after
        if reached & _PFX_COMPLETE:
            reached |= _PFX_BETWEEN
        modes = reached
        if not modes:
            break
        if modes & _PFX_BETWEEN:
            flags[index + 1] = 1
    return flags


def _pem_rigid_end_lengths(text: str, end: int) -> set[int]:
    """Lengths 2..9 with which `-{1,4}['\\"]?-{1,4}` can END at `end`.

    An enumeration rather than a scan, because the structure is a literal: four
    dashes at most, an optional quote, four dashes at most, so every spelling is
    nine characters or fewer and the candidate offsets of a header or END line
    are a handful. `end` is exclusive and the structure must begin inside
    `text`, so a caller passing a region that starts mid-string cannot be told
    the region matched something in front of it.
    """
    lengths: set[int] = set()
    dashes = 0
    index = end - 1
    while index >= 0 and dashes < 4 and text[index] == "-":
        dashes += 1
        index -= 1
    for head in range(1, dashes + 1):
        start = end - head
        if start >= 1 and text[start - 1] in "\x27\x22":
            quoted = 0
            index = start - 2
            while index >= 0 and quoted < 4 and text[index] == "-":
                quoted += 1
                index -= 1
            lengths.update(head + 1 + tail for tail in range(1, quoted + 1) if head + 1 + tail <= 9)
        closing = 0
        index = start - 1
        while index >= 0 and closing < 4 and text[index] == "-":
            closing += 1
            index -= 1
        lengths.update(head + tail for tail in range(1, closing + 1) if head + tail <= 9)
    return lengths


def _pem_body_arm_span(piece: str, floor: int, ceiling: int | None) -> tuple[int, int] | None:
    """Offsets a body line's base64 token may start at, as a low/high span.

    The token is the piece's trailing run of `[A-Za-z0-9+/=]`, optionally
    followed by one `,` and then whitespace to the piece end. Its width is a
    RANGE rather than one number in both arms — the full arm is floored at
    `PEM_BODY_FLOOR`, the short arm is a ceiling — and the PREFIX may enter the run as
    well as start before
    it, which is why this returns a span: `[112345678` is a prefix of `[1` with a
    token of `12345678`. No span means no offset can work, and the common case —
    `%8d` columns end in a short digit run, prose ends in a word — stops here
    without touching the machine.

    `floor` and `ceiling` are the width bounds of the arm being asked about;
    `ceiling=None` is the full arm, whose token may take the whole run.
    """
    trimmed = piece.rstrip(" \t")
    if trimmed.endswith(","):
        trimmed = trimmed[:-1]
    end = len(trimmed)
    run = end - len(trimmed.rstrip(_PEM_TOKEN_CLASS))
    if run < floor:
        return None
    low = max(0, end - (run if ceiling is None else min(run, ceiling)))
    high = end - floor
    if low > high:
        return None
    return low, high


def _pem_prefix_leads_a_token(piece: str) -> bool:
    """Is some prefix end in `piece` followed by eight body characters?

    The short arm's lookahead, read the other way round: the separator is
    consumed, a `LINE_PREFIX` follows it, and a full token must sit immediately
    behind that prefix. Read this way the lookahead is an ordinary linear
    question — every prefix end of the piece is already known from the flags, and
    the eight characters behind each one are a slice.
    """
    flags = _pem_prefix_end_flags(piece)
    for start in range(len(piece) - 7):
        if flags[start] and all(char in _PEM_TOKEN_CLASS for char in piece[start : start + 8]):
            return True
    return False


def pem_body_line(line: str) -> bool:
    """`PEM_BODY_LINE_RE.match(line)`, in linear time, for any string.

    `match` is ANCHORED AT POSITION 0, so the question is about the string's FIRST
    line: neither the prefix nor a token can contain a newline, so no match can
    start anywhere else, and the only thing the pattern reads past the first line
    is the short arm's lookahead. That is why this asks the first piece, and the
    second one only when the short arm is live — a run over every piece would
    answer `search`'s question instead, which is a different question and is how
    this decider first over-accepted a table row that follows a banner.

    `$` under `MULTILINE` means "the end, or just before a newline", so each arm
    ends at its piece's end: the full arm is a `PEM_BODY_FLOOR`-character token, and the
    short arm is a sub-floor token (one to `PEM_BODY_FLOOR - 1` characters) followed by
    a separator and then a full token behind a prefix. Both widths are read from
    `PEM_BODY_FLOOR` and not restated, because this decider's floor and the pipe's own
    hold are the same number on purpose — a decider a byte below the hold's floor
    under-holds, and under-holding publishes (see the constant). The caller is the
    pipe's line loop (`tools/builtin.py`),
    which passes one line with its terminator stripped — where the short arm can
    never fire — and this is exact for that input and for a whole multi-line read
    alike, because the arms are the pattern's own rather than an in-domain
    approximation of them.
    """
    pieces = line.split("\n", 1)
    piece = pieces[0]
    full = _pem_body_arm_span(piece, PEM_BODY_FLOOR, None)
    short = _pem_body_arm_span(piece, 1, PEM_BODY_FLOOR - 1) if len(pieces) > 1 else None
    if full is None and short is None:
        return False
    flags = _pem_prefix_end_flags(piece)
    if full is not None and any(flags[full[0] : full[1] + 1]):
        return True
    return bool(
        short is not None
        and any(flags[short[0] : short[1] + 1])
        and _pem_prefix_leads_a_token(pieces[1])
    )


def pem_end_line(line: str) -> bool:
    """`PEM_END_LINE_RE.match(line)`, in linear time.

    The pattern is not end-anchored: the line only has to START with the prefix
    grammar and then `-{1,4}['\\"]?-{1,4}END `, so the decision is "is any
    enumerated structure start also a prefix end". The enumeration is over the
    occurrences of the literal `END `, which is a necessary condition of the
    pattern itself, and this is exact for any string rather than only for a
    single line: a separator or a newline outside the fragment's character set
    empties the mode set, so a candidate after it can never be a prefix end.
    """
    if _PEM_END not in line:
        return False
    flags = _pem_prefix_end_flags(line)
    offset = 0
    while True:
        at = line.find(_PEM_END, offset)
        if at < 0:
            return False
        offset = at + 1
        for length in _pem_rigid_end_lengths(line, at):
            start = at - length
            if start >= 0 and flags[start]:
                return True


def pem_header_line_end(text: str) -> int | None:
    """Where `PEM_HEADER_LINE_RE.search(text)` ends, or None, in linear time.

    Returns the offset the caller needs (`match.end()`) rather than a match:
    the pipe uses it to split the read at the header's line end and re-enter the
    body loop on the remainder, and that offset is what releases the output
    ahead of the header.

    The search is modelled as the pattern's own `MULTILINE` anchors: `^` matches
    at the start of the text and after every `\\n`, `$` at the end and before
    every `\\n` — so the text is cut at its `\\n`s and each piece is asked the
    end-anchored question. A piece keeps any `\\r` of a CRLF terminator, and THAT is
    where the tail matters rather than where it can be ignored: the pattern's tail is
    `_PEM_ARMOUR_TAIL` (`[ \\t]*(?=\\r|\\n|$)`, read here as its two halves), so a
    CRLF, bare-CR or trailing-whitespace header line matches BOTH the pattern and this
    decider, and the offset returned is the position of that `\\r` — the same `end()`
    the pattern's own lookahead gives. An earlier revision modelled the tail as `$`
    alone and returned the piece's length, so it answered `None` for those spellings
    while the widened pattern matched them — which on the pipe means the block's whole
    body published.
    `test_the_header_deciders_tail_is_the_patterns_tail` and the armour-spelling
    differential beside it are what hold the two together now.

    Both literals the tail needs are required before any of the work below, and
    they are necessary conditions of the pattern (the fragment cannot consume a
    letter, so the tail's own `BEGIN ` and `PRIVATE KEY` cannot be assembled out
    of prefix characters).
    """
    offset = 0
    for piece in text.split("\n"):
        end = _pem_header_piece_end(piece)
        if end is not None:
            return offset + end
        offset += len(piece) + 1
    return None


def _pem_armour_tail_stops(piece: str) -> list[tuple[int, set[int]]]:
    """Every position the armour tail may end at, each with the rigid ends that reach it.

    `_PEM_ARMOUR_TAIL` is `[ \\t]*(?=\\r|\\n|$)`: after the closing dashes there may be
    trailing whitespace, and then that line's own end — a `\\r` (the CR of a CRLF file,
    or a bare CR used as the terminator) or the end of the piece (the `\\n` was the
    split). So a stop's rigid end is any position from which the tail's own run class,
    `_PEM_ARMOUR_TAIL_RUN`, leads to that stop, and the terminators are read from
    `_PEM_ARMOUR_TAIL_TERMINATORS` — both halves are the constants the PATTERN is built
    from, so the pattern and this decider cannot drift apart (which is what #1445's
    widening of the pattern did to the previous, `$`-only, model).

    Ordered, and in the pattern's own order: the terminator stops by position, then the
    piece's end. Bounded by the piece and allocation-free per candidate: a stop's run is
    walked back over the run class only, so this is O(len(piece)).
    """
    stops = [index for index, char in enumerate(piece) if char in _PEM_ARMOUR_TAIL_TERMINATORS]
    stops.append(len(piece))
    out: list[tuple[int, set[int]]] = []
    for stop in stops:
        start = stop
        while start > 0 and piece[start - 1] in _PEM_ARMOUR_TAIL_RUN:
            start -= 1
        out.append((stop, set(range(start, stop + 1))))
    return out


def _pem_header_piece_end(piece: str) -> int | None:
    """One `^…$` line of `pem_header_line_end`: `LINE_PREFIX` + the whole tail.

    Returns the offset the pattern's match would END at inside this piece, or None. The
    tail is
    `-{1,4}['\\"]?-{1,4}BEGIN [A-Z0-9 ]*PRIVATE KEY-{1,4}['\\"]?-{1,4}` + the
    armour tail, and every piece of it is local: a rigid structure, the `BEGIN `
    literal, a run of `[A-Z0-9 ]`, the `PRIVATE KEY` literal, a rigid structure whose
    end is pinned by the tail. So the candidates are enumerated from the literals and
    each one is a constant-time question against the prefix machine's flags — no
    search, and no partition to walk.

    The answer is the EARLIEST stop an accepted structure can reach — the pattern's own
    priority on the stops — and the scan below keeps that by taking the minimum over the
    accepted candidates rather than returning from a loop. The tail's `[ \\t]*` is greedy,
    so after the rigid structure it consumes the trailing whitespace and the lookahead
    must hold THERE: the earliest reachable terminator, and the piece's end only when no
    terminator precedes it. That is also why the returned offset is the stop and not the
    piece's length: the pattern's lookahead does not consume its `\\r`.

    The scan is hoisted out of the stop loop and the flags conjunct out of the per-key
    test (`PR1427 D1`), because as written the two mechanisms MULTIPLIED: measured on this
    box, a 40 KB line of `BEGIN ` literals cost 3.34 s of CPU and a 46 KB line of CR
    terminators 2.71 s, against 2.1 ms and 6.3 ms after the correction. No answer moves:
    the conjunct is independent of the key and a necessary condition of every acceptance
    through its `at`, so asking it once per BEGIN only skips candidates that could not
    have been accepted anyway.
    """
    if _PEM_BEGIN not in piece or _PEM_KEY not in piece:
        return None
    length = len(piece)
    # (1) The endings map, built from the STOPS side: ONE O(len(piece)) walk, instead of a
    # whole BEGIN/KEY scan per stop. A candidate's tail stop is the greedy run's own end,
    # and where two stops could offer the same `key_end` the SMALLEST stop wins — that is
    # what the ascending stop loop returned.
    stop_of: dict[int, int] = {}
    for stop, rigid_ends in _pem_armour_tail_stops(piece):
        for end in rigid_ends:
            for rigid in _pem_rigid_end_lengths(piece, end):
                key_end = end - rigid
                if key_end >= 0:
                    previous = stop_of.get(key_end)
                    if previous is None or stop < previous:
                        stop_of[key_end] = stop
    if not stop_of:
        return None
    flags = _pem_prefix_end_flags(piece)
    best: int | None = None
    at = piece.find(_PEM_BEGIN)
    while at >= 0:
        # (3) The flags conjunct is independent of the key and necessary for every
        # acceptance through this `at`, so asking it here — before the run walk and the KEY
        # window — cannot change an answer, and it keeps both off every BEGIN occurrence
        # that cannot accept at all.
        if any(
            at - rigid >= 0 and flags[at - rigid] for rigid in _pem_rigid_end_lengths(piece, at)
        ):
            body = at + len(_PEM_BEGIN)
            # The `[A-Z0-9 ]*` between the literals is a run, so its end bounds where
            # `PRIVATE KEY` may begin — and a `PRIVATE KEY` that starts inside it and
            # ends past it is still a match, which is why the search window is the
            # run's end plus the literal's own length.
            bound = body
            while bound < length and piece[bound] in _PEM_HEADER_CLASS:
                bound += 1
            key = piece.find(_PEM_KEY, body, bound + len(_PEM_KEY))
            while 0 <= key <= bound:
                stop = stop_of.get(key + len(_PEM_KEY))
                if stop is not None and (best is None or stop < best):
                    best = stop
                key = piece.find(_PEM_KEY, key + 1, bound + len(_PEM_KEY))
        at = piece.find(_PEM_BEGIN, at + 1)
    return best


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
            # THE DIGIT-MAXIMAL FRAGMENTS, not `LINE_PREFIX` / `_PEM_LINE_CONTENT`:
            # this rule is the table's one unbounded walk over unshaped text, and the
            # released fragment makes it exponential in the digits of a dense row (Q3
            # on #1427). Both are substituted — the body unit AND the short arm's
            # lookahead — because they carry the same ambiguity. Why that is the same
            # language, why a restriction on the unit END is not, and what pins the
            # equivalence: `_PEM_DIGIT_MAX_PREFIX` above.
            r"(?:" + _PEM_DIGIT_MAX_PREFIX + r"(?:" + _PEM_DIGIT_MAX_LINE_CONTENT + r")[ \t]*)+"
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
            # Group 3 is the opening quote when the value is quoted, and this rule
            # must not match such a value at all: the guard below refuses it and
            # hands it to the QUOTED spelling below, which is the only rule that
            # can find where a quoted value ENDS (see
            # :data:`_QUOTED_ASSIGNED_VALUE_GROUP` for the over-masking that cost
            # the operator a neighbouring KEY). The refusal is in the guard rather
            # than in this pattern because that is where every other rule in this
            # table states its refusals, and because a pattern-level lookahead
            # buys nothing measurable here: interleaved best-of-7 ``process_time``
            # on credential-dense text (4000 compact lines) put the guard form and
            # the lookahead form at 1.637x and 1.636x of the previous rule's cost,
            # so the second mechanism would be cost with no benefit.
            r"([\"']?\s*[:=]\s*)([\"']?)"
            rf"{_ASSIGNED_VALUE_GROUP}"
        ),
        # Keep the name, the separator and the opening quote; mask only the
        # value. Rendered by the guarded path, which knows the value group.
        None,
        4,
        guard=_assignment_guard,
    ),
    # The QUOTED spelling of the same assignment — `"api_key": "…"` — which is
    # how JSON, Python reprs and every provider's token response spell one. A
    # separate rule rather than a branch in the one above because the value's END
    # is knowable only when the opening quote is part of the match: see
    # :data:`_QUOTED_ASSIGNED_VALUE_GROUP` for the measured defect the greedy run
    # produced on compact JSON, and `_assignment_guard` for the delegation that
    # keeps the two from fighting (a quoted match is this rule's, never the
    # unquoted rule's).
    # ONE LABEL for both spellings, deliberately (agent review R1, finding 5): a
    # shape label is operator-facing text — it is rendered into the notice's
    # ``(shapes)`` list and into the journal row — and the two rules are one
    # shape to whoever reads that line. The rules stay separate because the
    # VALUE's bound differs, which is an implementation fact, not something an
    # operator triaging a notice can act on differently.
    Shape(
        "credential-assignment",
        re.compile(
            r"(?<![A-Za-z0-9_.\-])([A-Za-z0-9_.\-]{2,48})"
            r"([\"']?\s*[:=]\s*)"
            # NAMED, not positional. The value group below stops at the delimiter
            # this group captures, and a positional backreference would silently
            # start bounding against a different group the day a group is added or
            # reordered above it — an over-reaching mask again, with nothing
            # failing (agent review R1, finding 6). A group NAME cannot drift.
            r"(?P<quote>[\"'])"
            rf"{_QUOTED_ASSIGNED_VALUE_GROUP}"
        ),
        None,
        4,
        guard=_assignment_guard_quoted,
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
        # commands use — ``--secret MINERVA_UI_NPROD_USERNAME``, where the token after the
        # flag is the NAME of a stored secret rather than a value. See
        # ``_flag_value_guard`` and ``_value_is_a_reference_to_a_credential`` for the
        # production incident that measured it.
        re.compile(r"(?i)(--" + _CREDENTIAL_FLAG_WORDS + r"(?:=|\s+))([^\s\"']{3,})"),
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
        # notice. The suffix must also look like a token: at least 8 characters,
        # no dot, and no digit requirement — the dot is what keeps
        # `pypi-local-operator.json` (a filename, 159 such lines in this repo) readable.
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

#: The two costs every question below chooses between, in nanoseconds per byte of
#: the text: one C-level ``str`` search (``value in text``, ``window in text``),
#: and one gated pass over the text, which is one Python-level step per character.
#:
#: Measured on this host (CPython 3.12.13, best of three, on 1.3-1.8 MB of
#: credential-dense text — a large tool result or transcript body carrying a
#: credential row every few hundred bytes, which is the shape this pass is
#: expensive on):
#:
#: * the search is **~1 ns per byte PER KEY**, and the negative case — the common
#:   one, the value or window that was masked — has to walk the whole text;
#: * the pass is **~25-190 ns per byte whatever the key count**, set by how much
#:   of the text its gate admits (2 keys that start on rare characters: 25 ns;
#:   the 49 windows of two credentials in text that is full of them: 190 ns).
#:
#: So a question with a handful of keys is answered by one search each — 2 keys
#: cost 2 ms against 44 ms for the pass on the same text — and a question with
#: thousands of them is answered by the pass, because at that end the searches are
#: exactly the shape the freeze came in: 6000 keys cost 8.9 s of searches against
#: 44 ms. Both arms answer the same question byte for byte (a test pins them
#: against each other on the corpus and on the incident's own shape), so which one
#: runs is a cost decision and never a behaviour one.
#:
#: The pass's span is quoted at both ends because it is the arm with the flat
#: cost; the crossover uses the upper end, so a question near it errs toward the
#: arm whose cost does not depend on the answer.
_SEARCH_NS_PER_BYTE = 1.0
_PASS_NS_PER_BYTE = 190.0


def _searches_are_cheaper(keys: int) -> bool:
    """Whether ``keys`` C-level searches beat one gated pass over the text.

    The whole decision, in one place, from the two measured terms above: the
    searches are linear in the KEY count and the pass is not, so this is the same
    comparison for both questions the index asks.
    """
    return keys * _SEARCH_NS_PER_BYTE <= _PASS_NS_PER_BYTE


class _SurvivalIndex:
    """Whether readable material survived in one model-visible text, for every hit.

    **The defect this replaces.** The check was ``value in text`` per hit, and
    ``_only_fully_masked`` asks it once per hit, so the pass cost ``hits x bytes``.
    Five session runtimes on this machine were found frozen for 1.5 to 7.2 hours
    with 100% of their event-loop main thread sampled inside the C-level search of
    this pass (``_sre_SRE_Pattern_search`` -> ``sre_search`` -> ``sre_ucs1_match`` /
    ``sre_ucs2_match``) while heartbeats went stale for hours, and one stall dump
    landed exactly here: ``redaction_shapes.py:1963`` in the revision it ran (the
    pre-change line number of ``if value in text``), with ~5.4 G byte-scans for
    0.86 MB of text carrying 6,270 hits. The session
    runtime's heartbeat is an asyncio task on that same loop, so an occupation past
    its 45 s timeout is indistinguishable from a dead runtime and every control call
    refuses without ``--force``.

    **Why 6.4 s of scan is a freeze and not a slowdown.** Cost per byte RISES with
    size, because the hits rise with it: at a fixed hit density the grading half
    measured 1.19 / 1.45 / 1.82 / 2.63 microseconds per byte at 64 / 128 / 256 /
    512 KB. A text big enough to matter is therefore the text this pass cannot
    finish inside a heartbeat — which is the loop occupancy #1363 bounds and this
    change removes.

    **Two questions, both unchanged.** A hit is EXPOSED when the whole VALUE is in
    the text as delivered, or when the value is at least a window long and one of
    its six-character windows is in the text WITH THE MARKER STRIPPED. The
    judgement is the one :func:`_credential_fragments_survive` documents at
    length — anchored on the credential's own characters, the marker never
    material — and this index changes only how the two questions are asked.

    **Each question is asked once for the whole call, over its own keys.** The
    keys are per credential, not per hit, and there are only two of them: the
    distinct VALUES (the whole-value half) and the distinct six-character WINDOWS
    of those values (the partial-mask half). A text with 6270 hits over two
    credentials asks two questions of two keys and 49 windows, and reads the text
    by whichever of the two arms :func:`_searches_are_cheaper` picks — one C-level
    search per key while the keys are few, one gated pass when they are not. That
    is the whole of the change: the text is read a BOUNDED number of times per
    call, and the number of hits does not appear in the cost.

    **The two arms are not interchangeable, and the difference is where the
    marker is read.** The whole-value half reads the text AS DELIVERED, marker
    included, because a credential that is (or contains) the marker survives only
    if its own bytes are in there — the limit pinned by
    ``test_a_marker_inside_the_credentials_own_value_is_a_recorded_limit``, where
    a wholly surviving ``tok[redacted]tail`` must escalate. The window half reads
    it with the marker stripped, because the marker is what a mask WRITES and a run
    straddling one would otherwise match a credential whose own value IS the
    marker (the two ``.npmrc`` spellings and the cookie-header case QA round 1
    found escalating on nothing readable at all). Stripping it also makes the two
    characters either side of a removed marker adjacent, which is what lets a mask
    that stopped inside a credential be seen at all — the case the floor exists
    for.

    **Nothing is built unless a hit needs it, and the ordinary result pays
    nothing.** ``scrub_shapes_with_hits`` returns before this class is constructed
    when the text carries no anchor, and each half is built on its first question,
    so a text whose hits never reach the fragment half never pays for the windows.
    Most results are also far too small for either arm to matter: every text in the
    credential corpus is under 200 characters.

    **What this bounds, and what it does not.** The transient set the previous
    index built (one six-character run per text position, ~100 bytes per byte of
    text, 101.7 MB measured for a 1 MB text) is GONE: both arms are keyed on the
    credentials, so the memory follows the values (~100 bytes per value character)
    rather than the text: those values are substrings of the text, so the length of
    the text remains the bound in the pathological case where every hit is a
    distinct multi-kilobyte value, and the credentials are the bound in the
    ordinary one. The cost that remains is at most ``_PASS_NS_PER_BYTE``
    per byte plus ``_SEARCH_NS_PER_BYTE`` per key per byte, i.e. linear in the text
    with a bound that does not contain the hit count — measured at 0.03-0.21
    microseconds per byte on the shapes that froze a runtime, against 4.3-4.7
    microseconds per byte before, flat as the hit count grows. The memory follows
    the credentials too: on the 1 MB of high-entropy hex text the previous index's
    disclosure was measured against, one credential row now adds **1.0 MB** of peak
    RSS where it added **24.5 MB** (one process per reading, `ru_maxrss` delta
    around a single ``scrub_shapes_with_hits``).
    """

    __slots__ = (
        "_text",
        "_values",
        "_whole_done",
        "_whole_found",
        "_windows",
        "_readable",
        "_windows_done",
        "_windows_found",
    )

    def __init__(self, text: str, values: Iterable[str]) -> None:
        self._text = text
        # Distinct VALUES, and the marker is never one of them: the two questions
        # are per credential, so a text with 6270 hits over two credentials asks
        # two questions, and a value that IS the marker is not a survivor.
        self._values = tuple(
            value for value in dict.fromkeys(values) if value and value != REDACTION_MARKER
        )
        self._whole_done = False
        self._whole_found: set[str] = set()
        self._windows_done = False
        self._windows: dict[str, tuple[str, ...]] = {}
        self._readable: Optional[str] = None
        self._windows_found: set[str] = set()

    def whole_survives(self, value: str) -> bool:
        """Whether the whole value occurs in the text as the model reads it."""
        if not self._whole_done:
            self._scan_whole()
        return value in self._whole_found

    def window_survives(self, value: str) -> bool:
        """Whether any six-character window of the value survived in the text."""
        if not self._windows_done:
            self._scan_windows()
        return value in self._windows_found

    def _scan_whole(self) -> None:
        """Decide every distinct value's presence, by the cheaper of the two arms.

        Both arms are the same predicate — an occurrence of the value in the text
        — so which one runs cannot change an answer, only the cost.
        """
        self._whole_done = True
        if not self._values:
            return
        if _searches_are_cheaper(len(self._values)):
            self._whole_found.update(value for value in self._values if value in self._text)
            return
        self._whole_found.update(_present_heads(self._text, self._values))

    def _scan_windows(self) -> None:
        """Decide which values kept a window, by the cheaper of the two arms."""
        self._windows_done = True
        self._windows = _value_windows(self._values)
        if not self._windows:
            return
        readable = self._readable_text()
        if _searches_are_cheaper(len(self._windows)):
            for window, carriers in self._windows.items():
                if window in readable:
                    self._windows_found.update(carriers)
            return
        self._windows_found = _present_windows(readable, self._windows)

    def _readable_text(self) -> str:
        """The text with the marker stripped, built once, and only when asked for.

        The strip is skipped when there is no marker to strip: on that text the two
        readings ARE the same string, and a copy of a multi-megabyte result to
        change nothing in it is pure cost.
        """
        if self._readable is None:
            self._readable = (
                self._text.replace(REDACTION_MARKER, "")
                if REDACTION_MARKER in self._text
                else self._text
            )
        return self._readable


def _value_windows(values: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Every six-character window of every value, mapped back to the values holding it.

    Keyed by the WINDOW rather than by the value because the pass below reads the
    text once and asks "which values does this position speak for?" — the reverse
    direction would have to hold a position list per window, which is the whole
    text again. A value shorter than a window has none and so can never survive
    this half, which is the floor's own behaviour (the check this replaces
    returned there without reading the text at all).
    """
    grouped: dict[str, list[str]] = {}
    for value in values:
        for start in range(len(value) - _FRAGMENT_WINDOW + 1):
            grouped.setdefault(value[start : start + _FRAGMENT_WINDOW], []).append(value)
    return {window: tuple(carriers) for window, carriers in grouped.items()}


def _present_windows(readable: str, windows: Mapping[str, tuple[str, ...]]) -> set[str]:
    """Which values have a window in ``readable``, in ONE pass over it.

    One step per character, gated on the first character of the needed windows, so
    the per-character work is a set lookup and a window is only ever compared where
    it could start. The gate is what keeps this arm's cost near the floor rather
    than at one dict lookup per character of a multi-megabyte text.
    """
    gate = {window[0] for window in windows}
    found: set[str] = set()
    for position, char in enumerate(readable):
        if char in gate:
            carriers = windows.get(readable[position : position + _FRAGMENT_WINDOW])
            if carriers is not None:
                found.update(carriers)
    return found


def _present_heads(text: str, values: Sequence[str]) -> set[str]:
    """Which of ``values`` occur in ``text``, in ONE pass over it.

    The arm for a question with too many keys for one search each, where those
    searches are what the freeze was made of. Two structural facts make it exact
    rather than approximate, and both matter:

    * the head is the value's own first ``_FRAGMENT_WINDOW`` characters, so any
      occurrence of the value starts on one of them — the pass cannot miss one;
    * the head is only a CANDIDATE: the characters after it are checked against
      the whole value, because a text can carry the head without carrying the
      credential (a shorter value that prefixes a longer one, a name that starts
      like a token). Nothing enters the answer that was not verified in full.

    A value shorter than the window is looked up at its own length, which is why
    the keys carry widths: a five-character password is a real hit (``dsn-password``
    on the ``amqp`` case, which is the one escalating positive the corpus pins), and
    it must be found by its own five characters rather than by a window it does not
    have.
    """
    by_width: dict[int, dict[str, list[str]]] = {}
    for value in values:
        if not value:
            # An empty value has no character to key on and no occurrence to find,
            # and its own width-0 key would be indexed at 0 below. Both callers
            # filter it out today (``_SurvivalIndex`` drops it, and
            # ``_credential_fragments_survive`` returns before the index), so this
            # keeps the helper total over its declared input rather than relying on
            # both of them to stay that way.
            continue
        width = min(len(value), _FRAGMENT_WINDOW)
        by_width.setdefault(width, {}).setdefault(value[:width], []).append(value)
    # First character -> the widths whose keys can start there, so the common
    # position pays one comparison and one lookup rather than one per width.
    widths_of: dict[str, tuple[int, ...]] = {}
    for width, heads in by_width.items():
        for head in heads:
            known = widths_of.get(head[0], ())
            if width not in known:
                widths_of[head[0]] = known + (width,)

    found: set[str] = set()
    for position, char in enumerate(text):
        widths = widths_of.get(char)
        if widths is None:
            continue
        for width in widths:
            candidates = by_width[width].get(text[position : position + width])
            if candidates is None:
                continue
            for value in candidates:
                if value not in found and text[position : position + len(value)] == value:
                    found.add(value)
    return found


#: A bare English word: letters only, all lower case. What ordinary prose leaves in a
#: credential flag's argument position, and the one class whose letters cannot be told
#: from prose anywhere in the text (see :func:`_is_prose_after_a_flag`).
_BARE_WORD = re.compile(r"[a-z]+")

#: The LONGEST bare lowercase run in a flag's argument position that is still read as
#: prose. Four, because that is the width of the two specimens the refusal was measured
#: for (``when``, and the three-character word the suite pins) — and NO wider.
#:
#: It deliberately does not borrow :data:`_ASSIGNED_VALUE_MIN_CHARS` (eight), which it
#: used to: that floor answers a different question (is a short value under a
#: credential-shaped NAME a placeholder?), and borrowing it swallowed the escalation for
#: every five-, six- and seven-character value printed a second time in the clear — the
#: canonical short weak passwords, which is the case the survival question exists to
#: catch (agent review R1-1). Measured: the 439-row corpus grades identically for every
#: bound at or above four (four, five, six, seven, eight, eleven and fifteen were run),
#: and differently only at three — so the wider bound bought nothing.
_FLAG_PROSE_MAX_CHARS = 4


def _is_prose_after_a_flag(hit: ShapeHit) -> bool:
    """Whether this hit is the flag rule's over-mask of an English word.

    **Measured 2026-09-22, on a documentation read.** A ``read`` of a project's
    ``AGENTS.md`` — 33 KB of ordinary prose — escalated to a "rotate it" demand on ONE
    hit: the sentence ``It also accepts --api-key [redacted] you need to override the
    env-backed credential`` puts ``when`` in a credential flag's argument position, the
    flag rule masked that word (deliberately, see
    ``test_prose_after_a_flag_can_match_but_may_never_demand_a_rotation``), and the
    exposure question then answered YES — because a four-character English word occurs
    again somewhere in 33 KB of prose. The rotation demand named the word ``when``.

    **Why the MASK stays and the CLAIM goes.** Masking a word-shaped argument is a
    deliberate over-mask: the cost is one unreadable English word, and the alternative —
    letting a flag whose value really is a short word through — is unrecoverable. The
    ESCALATION was never deliberate, and the module's own test says so in its name: this
    shape "may never demand a rotation".

    **Why the question cannot be answered rather than answered differently.** A value
    that cannot be told from prose cannot be told from prose ANYWHERE in the text: the
    whole-value half of :func:`_credential_fragments_survive` asks whether those letters
    are present, and for a word the answer is yes for reasons that have nothing to do
    with this mask. No amount of reading the text distinguishes the second ``when`` from
    the first, so the claim is REFUSED rather than downgraded — the mask is complete,
    which is what ``complete`` means, and ``exposed`` is the half that cannot be read
    here. The ``vendor-prefixed-token`` arm took the same judgement in the other
    direction (:data:`_VENDOR_TAIL_IS_A_NAME`: there the match is refused); keeping the
    mask and refusing the claim is the protective half of that trade.

    **The class, and the length bound on it.** A bare run of lowercase letters, no
    longer than :data:`_FLAG_PROSE_MAX_CHARS` (four). The bound is the width of the two
    specimens this refusal was written for — the four-character ``when`` above, and the
    three-character word the suite pins — not the module's own masked-value floor it
    first borrowed. :data:`_ASSIGNED_VALUE_MIN_CHARS` (eight) answers a DIFFERENT
    question: whether a short value under a credential-shaped NAME is a placeholder or a
    word. Applied here it read a judgement nobody made, and cost the escalation for
    every five-, six- and seven-character value — the canonical short weak passwords,
    printed twice in the text the model reads, which is the case the survival question
    exists to catch (agent review R1-1). Measured on the 439-row corpus: the grading is
    identical for every bound at or above four, so the wider bound bought nothing.
    Nothing with a digit, a separator, a symbol or any upper-case letter is this class at
    all.

    **The stated limit, narrowed to four characters.** A credential-shaped value of
    three or four lowercase characters that really IS printed a second time in the clear
    is still graded contained, so it files no rotation demand: at that width a word
    cannot be told from a credential ANYWHERE in the text, which is the whole of the
    refusal, and it is the price of not manufacturing a rotation demand for every English
    word that recurs in ordinary prose. It is pinned on BOTH sides of the boundary — the
    four-character ``when`` row in the corpus has to stay contained, and the five-, six-
    and seven-character cases in
    ``test_the_flag_prose_refusal_stops_at_four_characters`` have to escalate again —
    rather than left for a differential to find.
    """
    return (
        hit.label == "cli-credential-flag"
        and len(hit.value) <= _FLAG_PROSE_MAX_CHARS
        and _BARE_WORD.fullmatch(hit.value) is not None
    )


def _credential_fragments_survive(hit: ShapeHit, index: _SurvivalIndex) -> bool:
    """Whether a readable piece of the CREDENTIAL survived in the masked text.

    **Anchored on the credential's own characters, never on the matched region**,
    and that is the whole of the judgement (QA round 1, Q1). The region is not all
    secret: a DSN rule keeps ``amqp://user:`` and ``@host`` readable BY DESIGN, so a
    region-wide window search graded a fully-masked ``amqp://guest:guest@host`` as
    exposed — the surviving window was the USERNAME — and filed a rotation demand
    for a value that never left the tool. Four corpus hits were affected, three of
    the four because the "survivor" was the redaction MARKER itself. So:

    * the marker is not material (the window half reads the text with it stripped,
      and a value that IS the marker is never a survivor);
    * the whole VALUE present in the text is a leak: the rule's group was narrower
      than the credential, or this copy was never masked;
    * a window OF THE VALUE present is a partial leak — the mask stopped inside the
      credential, which is the case the floor exists for and what a truncating rule
      (a PEM without its END line, a base64 body) produces.

    Reading the VALUE rather than the region means material a rule deliberately
    preserves can never be counted as a survivor.

    **The questions are asked of an index, not of the text.** The order is
    deliberate and load-bearing: readable material is checked FIRST, so a
    truncated-PEM hit cannot have an exposure swallowed by the caller's other
    branch. Neither question scans the text once per hit — that is what held a
    runtime's event loop for hours (:class:`_SurvivalIndex`).

    **A stated limit, not an oversight.** Because the window half reads the text
    with the marker stripped, a credential whose own value literally contains
    ``[redacted]`` and survives only PARTIALLY is unrepresentable: the fragment
    still in the text is spelled exactly like the marker a mask would have written,
    and no reading of the text can tell them apart. That identity is what makes the
    ``.npmrc`` and cookie false positives above impossible to grade correctly by
    inspection, so it is not closable here — only the wholly-surviving copy of such
    a value is still caught, by the whole-value half. Reaching it needs an operator
    secret that itself contains the harness's marker string, which is why the limit
    is recorded rather than paid for.
    """
    value = hit.value
    if not value or value == REDACTION_MARKER:
        return False
    if _is_prose_after_a_flag(hit):
        # Not a downgrade of the answer but a refusal to ask: for a value that cannot
        # be told from prose, the question has no discriminating power. See
        # :func:`_is_prose_after_a_flag` for the measured escalation this refuses and
        # for the limit it states.
        return False
    if index.whole_survives(value):
        return True
    return index.window_survives(value)


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

    The check asks two questions of the credential's own characters — is the
    whole VALUE still in the text, or is one of its six-character windows — and it
    is a backstop rather than a proof: it catches a value that survives whole (the
    group was narrower than the credential) and one that survives in fragments.
    Fixing the patterns is the real work; this is what stops the false claim if one
    slips through again. Neither question scans the text once per hit: the index
    reads it once for all of them, which is what a runtime wedged inside this pass
    paid for.
    """
    marked: list[ShapeHit] = []
    # One index for the text, shared by every hit, holding the values of all of
    # them: the two questions are per-credential and the text is the same for all
    # of them, so the reading is done once (see :class:`_SurvivalIndex`).
    index = _SurvivalIndex(text, (hit.value for hit in hits))
    for hit in hits:
        # Readable material is checked FIRST, so the truncated-PEM branch below
        # cannot swallow an exposure: a block that was masked is contained, and
        # one that left a fragment readable is not.
        #
        # A PLACEHOLDER/REFERENCE value is never graded EXPOSED, and this consult is
        # the whole of the false-positive fix rather than a wording change. The DSN
        # rule masks the copy INSIDE the URL and deliberately leaves a bare second
        # mention READABLE — the value is a ``$VAR`` reference, and
        # ``is_placeholder_component`` is exactly what keeps it unmasked (it is the
        # predicate ``_value_is_not_a_credential`` consults at the masking floor).
        # The fragment test then found that deliberately-readable survivor under the
        # hit's own value and read it as a partial mask, so a duplicated reference
        # filed a rotation demand for a value that was never credential material:
        # ``postgres://u:$VAR@host`` plus a later ``$VAR`` escalated, and the same
        # line without the second mention did not. The exposed path was the ONE site
        # that did not consult the predicate — the masking floor and the registration
        # floor (:func:`is_registerable_component`) both do — and no word-list change
        # can reach it, because the value is correctly on the list already.
        #
        # The conservative direction is unchanged for every real value: only a
        # placeholder is excused, and a genuinely half-masked SECRET (or a duplicate
        # of one) still escalates, because its own characters are genuinely readable.
        exposed = not is_placeholder_component(hit.value) and _credential_fragments_survive(
            hit, index
        )
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


#: How far after the reading verb the credential FILENAME may sit, and how long
#: the path-shaped token naming it may be. Bounds on COST, sized from the spellings
#: that have to keep working rather than picked for round numbers: over 39,111
#: harvested real commands, the largest gap any firing needed was 71 characters and
#: the longest path token 79, so both bounds clear the real work with headroom.
#: They are also what makes the rule's cost independent of the line's length — see
#: the rule's own comment for the measurement that forced them.
#:
#: THE CUT IS DELIBERATE, and these are the measured edges of it (agent review
#: R1/E3, reproduced here so the next reader does not have to re-derive them):
#: ``cat `` + N×``x`` + `` .env`` fires at N=94 — a 96-character gap, the two
#: spaces included, which is the whole of :data:`_FILE_GAP_CHARS` — and is silent
#: at N=95 (97). ``cat /`` + N×``d`` + ``.pem`` fires at N=222 — 224 characters
#: between the verb and the suffix, i.e. the two windows summed — and is silent
#: at N=223 (225). Both are a shade INSIDE the nominal windows (a real read whose
#: path is longer than that gets no advisory where the unbounded rule advised),
#: and that is the trade for turning a quadratic scan linear — a deep path is
#: rare, a 140 KB line cost 142 s. The residual is named here rather than left to
#: be discovered.
#: One more measurement, from QA round 1 (Q-3), sizes the population that sits
#: outside those edges: taking EVERY harvested line that carries both a read verb
#: and a credential filename — firing or not — the gap reaches 664 characters at
#: its extreme and 48 at its median. Those extremes are prose that happens to hold
#: a verb and a filename rather than read commands, which is why the bound still
#: clears the real work (largest gap a firing command needed: 71); they are named
#: here because the exemption they describe is exactly what the cut gives up, and
#: raising it is one constant if a real spelling beyond it turns up.
_FILE_GAP_CHARS = 96
_FILE_PATH_CHARS = 128


#: What counts as a COMMAND POSITION for a dump rule: the start of the command, or
#: the word that follows a shell separator.
#:
#: ``(`` is an arm of its own here, and it requires WHITESPACE after it, because a
#: bare ``(`` is not lexical evidence of anything. Measured 2026-09-21 on a peer
#: session's own count-only scan: ``where=collections.defaultdict(set)`` — a Python
#: expression — drew ``[credential guard] env: print names, not values``, because the
#: ``(`` opened a "command position" and ``set`` is the shell builtin that dumps
#: variables. The command held no ``env``, no ``printenv``, no ``cut``, and no dump of
#: any kind: the guard was matching the shape of the SEARCH QUERY the agent had just
#: written, which is the circular case — an agent cannot audit its own detector
#: without writing the pattern that trips it. A shell subshell is written ``( env`` or
#: ``(env``; only the spaced spelling survives, and that is the recorded cost. An
#: advisory is all this rule produces, so a missing one is cheap next to a false one
#: naming an idiom the command never used.
#:
#: ``$(`` keeps its zero-width arm deliberately: ``x=$(set)`` IS a dump, and a command
#: substitution has no other spelling.
_COMMAND_POSITION = r"[;&|]\s*|\(\s+|\$\(\s*"


DUMP_SHAPES: tuple[DumpShape, ...] = (
    # A bare ``env`` / ``printenv`` / ``set`` prints every value in the
    # environment — this is the incident's own shape. The command has to be at a
    # command position and be IMMEDIATELY followed by a separator or the end of
    # the command: ``env -i HOME=… cmd`` runs a command (and is how a clean
    # environment is built), ``printenv PATH`` asks for one harmless name, and
    # neither is a dump.
    DumpShape(
        "environment-dump",
        re.compile(r"(?:^|" + _COMMAND_POSITION + r")(?:env|printenv|set)\s*(?=[|;&)]|$)"),
        "print names, not values: `env | cut -d= -f1`; or use the value inside the "
        "command that needs it, e.g. `$(lop secret get NAME)`",
    ),
    # ``printenv MONGO_DSN`` — a single VARIABLE whose name says it is a
    # credential. Narrow on purpose: ``printenv PATH`` must stay ordinary.
    DumpShape(
        "named-variable-dump",
        re.compile(
            r"(?i)(?:^|" + _COMMAND_POSITION + rf")printenv\s+{_COUNT_PREFIXES}{_CREDENTIAL_NAME}"
        ),
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
        # THE GAP IS BOUNDED AND SEPARATOR-FREE, and both halves are load-bearing:
        #
        # * BOUNDED, because an unbounded lazy gap followed by an unbounded
        #   ``[^\s]*`` is quadratic in the line's length — the engine retries the
        #   path alternatives at every gap length, and a failing ``[^\s]*\.pem``
        #   walks the rest of the line at each one. Measured on a 140,005-character
        #   single line: 114.4 s of CPU for ONE ``search``, against 0.0009 s here;
        #   on a line of 28,000 verb words, 171.6 s against 0.24 s, so the cost went
        #   from quadratic to linear. Real commands reach 30,849 characters (p99
        #   4.9 KB over 39,111 harvested commands), so the blowup was latent rather
        #   than live — and latent is not safe: the same class of unbounded run had
        #   frozen six sessions on this fleet hours earlier.
        # * SEPARATOR-FREE, because the rule means "a command that READS a
        #   credential file", and the filename has to be an argument OF the verb
        #   for that to be true. ``[^\n]*`` let the two halves sit in different
        #   commands on one line, in a comment, or in a heredoc body — measured
        #   over the same corpus, 47 of the 74 firings were exactly that (``head -30;
        #   echo ---; ls -la .env`` and ``cat /tmp/x); cd ... && cp .env``), i.e.
        #   ordinary reads and unrelated commands nagged about a token they merely
        #   MENTION. ``#`` is in the class for the same reason: a comment is not a
        #   read, and the cost is one unreachable spelling (``cat file#1.env``).
        re.compile(
            r"(?i)\b(?:cat|less|more|head|tail|bat|strings|xxd|base64)\b"
            rf"[^;&|#\n]{{0,{_FILE_GAP_CHARS}}}?"
            r"(?:\.netrc\b|\.npmrc\b|\.docker/config\.json\b|\.kube/config\b|"
            r"mcp\.json\b|credentials\.env\b|\.env\b|"
            rf"[^\s]{{0,{_FILE_PATH_CHARS}}}\.pem\b|"
            rf"[^\s]{{0,{_FILE_PATH_CHARS}}}service-account"
            rf"[^\s]{{0,{_FILE_PATH_CHARS}}}\.json\b)"
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
