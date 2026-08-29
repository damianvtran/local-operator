"""Value-agnostic scrubbing of credentials a TOOL echoed into its own output.

``variables.redact_secret_values`` is keyed on **values the harness holds**. It
is correct and stays exactly as it is, but it structurally cannot catch the
class of leak this module exists for: a secret that lived only as an
environment variable inside a remote container, was never in the harness's
context, and reached the transcript because the *tool* printed it.

The originating incident: an agent ran, inside a pod,
``wget --user="$OPENSEARCH_USERNAME" --password="$OPENSEARCH_PASSWORD" ...``.
The image ships BusyBox wget, which does not implement those flags, and
BusyBox's ``getopt_long`` failure path prints the rejected option **together
with its value**::

    wget: unrecognized option: password=<the actual password>

Every secret-hygiene rule was followed — the value was never interpolated by
the calling shell, never read, never printed by the agent — and it leaked
anyway. Nothing value-keyed had anything to match on. The only thing
recoverable from that output is its SHAPE, which is what this module detects.

The detector is deliberately ONE anchored pattern rather than a family of
"looks like a credential" heuristics. It was measured against 2,807 stored
transcripts (453,632 JSONL lines): 14 matches, 11 suppressed as placeholders,
3 redacted, and those 3 are exactly the 3 known leaks. Zero false positives.
Widening it is not free — see the rationale on each component below.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

#: Verbs a getopt-family parser uses when it refuses an option. BusyBox says
#: "unrecognized option"; GNU getopt_long says "unrecognized option" or
#: "invalid option"; BSD and Go's flag package say "flag provided but not
#: defined"; Python argparse says "unrecognized arguments". The set is closed
#: deliberately: this is an allowlist of REJECTION SENTENCES, not a general
#: "looks like a flag" matcher, and every addition must be justified by a real
#: parser's output because each one widens the false-positive surface.
#:
#: The rejection lead is the load-bearing conjunct. It is the only context in
#: which a well-behaved program prints a credential flag *together with* its
#: value: ordinary usage text prints ``--password=PASSWORD`` (a metavariable),
#: ``--help`` prints the flag alone, and neither is this shape. Without the
#: lead, the same rule matched user- and assistant-authored prose in 40% of its
#: hits across the corpus — it would redact the operator's own messages.
#:
#: The trailing ``['\"`]?`` accepts one OPENING QUOTE between the sentence and
#: the option, because BusyBox has two rejection paths and the incident only
#: showed one of them. The pod that leaked printed the bare form
#: ``unrecognized option: password=VALUE``; a stock ``busybox:1.37.0`` image
#: prints the quoted form ``unrecognized option '--password=VALUE'``. Both are
#: real BusyBox output for the same mistake (verified by running the container,
#: see the PR's testing evidence), and a pattern built from the transcript
#: alone silently misses the second. The quote is consumed by the lead rather
#: than the value, and the value's character class excludes quotes, so the
#: closing quote survives the substitution untouched.
_REJECT_LEAD: Final = (
    r"(?:unrecognized|unrecognised|invalid|illegal|unknown|bad)"
    r"[ \t]+(?:option|options|argument|arguments|switch|flag)"
    r"(?:[ \t]+provided)?[ \t]*:?[ \t]*"
    r"['\"`]?"
)

#: Credential-typed option names. Matched WITHOUT requiring leading dashes,
#: because BusyBox strips them when it echoes — the real leak line reads
#: ``unrecognized option: password=X``, not ``--password=X``. A naive
#: ``--password=`` matcher misses the very incident it would be written for.
#:
#: The credential typing is what keeps ordinary output intact: without it the
#: rule would fire on ``unrecognized option: color=auto`` and mangle it.
_CRED_FLAG: Final = (
    r"(?:password|passwd|pass|passphrase|token|secret|credential|credentials"
    r"|api[-_]?key|apikey|access[-_]?key|secret[-_]?key|auth[-_]?token"
    r"|client[-_]?secret|bearer)"
)

#: The ``=`` glue is not cosmetic — it is the discriminator that does the
#: heaviest lifting, and the single most valuable measurement behind this
#: module. Across the whole corpus, every prose match ("invalid **al**...",
#: "unknown argument **for**...") was space-separated and every real
#: credential echo used ``=``. With ``=`` the token after the flag IS the
#: value; with a space it is the next English word. Accepting a space here
#: converts a rule with an observed clean margin back into a heuristic.
CREDENTIAL_ECHO_RE: Final = re.compile(
    rf"(?P<lead>{_REJECT_LEAD})"
    rf"(?P<flag>-{{0,2}}{_CRED_FLAG})"
    r"(?P<eq>=)"
    r"(?P<value>[^\s\"'`]+)",
    re.IGNORECASE,
)

#: Values that are ALREADY safe and must not be re-redacted. Re-redacting is
#: not merely cosmetic: it makes this module non-idempotent, so text that is
#: scrubbed at one choke point and scrubbed again at the next would keep
#: mutating. It would also corrupt prose written ABOUT a leak — the incident
#: report itself, and every agent message quoting it, contain the literal
#: ``password=«REDACTED»``. Three sessions in the corpus are exactly that:
#: discussions of this incident, which a detector without this suppressor
#: would "find" and mangle.
_PLACEHOLDER_RE: Final = re.compile(
    r"^(?:"
    r"\$\{?\w+\}?"  # $VAR / ${VAR} — a shell reference, never a value
    r"|«[^»]*»"  # our own marker, and the incident report's
    r"|\[redacted\]"  # the existing value-keyed marker
    r"|<[^>]*>"  # <value>, <password> — metavariable
    r"|\{[^}]*\}"  # {password} — template placeholder
    r"|[*x]{3,}"  # ***, xxxx — already masked upstream
    r"|PASSWORD|SECRET|TOKEN|APIKEY|API_KEY"  # getopt usage metavariables
    r")$",
    re.IGNORECASE,
)

#: Below this length a "value" is far more likely to be punctuation, a quote
#: fragment, or an English word than a credential, and redacting it costs more
#: in mangled output than it saves. Measured: every sub-4-char capture in the
#: whole corpus was noise.
_MIN_VALUE_LEN: Final = 4

#: Cheap gate for the hot path. ``spill_truncate`` runs on every oversized tool
#: output, so the scrub must not add a full regex pass over megabytes of
#: unrelated text. Every match of :data:`CREDENTIAL_ECHO_RE` necessarily
#: contains one of these nouns, so a miss here is a guaranteed miss there.
#:
#: Deliberately NOT a regex. Measured on a 20 MB buffer with no match, which is
#: the shape that matters (``cat`` of a large file):
#:
#:     re.compile(r"option|...", re.IGNORECASE).search   3.27 s
#:     re.compile(r"option|...").search                  0.72 s
#:     text.lower() + four ``in`` tests                   0.07 s
#:
#: The intuition that a single C-level regex beats lowercasing the string is
#: wrong by 45x: ``str.lower`` is one linear pass and ``str.__contains__`` is
#: memchr-backed, while an alternation forces the regex engine to try each
#: branch at every position. This gate ran on the event loop and cost 0.5 s on
#: the oversized-bash liveness test until it was measured.
_PREFILTER_NOUNS: Final = ("option", "argument", "switch", "flag")


def _may_contain_echo(text: str) -> bool:
    """True when ``text`` could possibly hold a rejection sentence.

    A false positive here only costs the full regex pass; a false negative
    would be a missed leak, so the nouns must stay in sync with
    :data:`_REJECT_LEAD`.
    """
    lowered = text.lower()
    return any(noun in lowered for noun in _PREFILTER_NOUNS)


def _marker(flag: str) -> str:
    """The replacement marker for one scrubbed value.

    Names the FLAG and nothing else, because the flag is all this path
    actually observed. The variable name the report wanted
    (``«REDACTED:OPENSEARCH_PASSWORD»``) was expanded by a shell inside a
    remote pod and appears nowhere in the output; only the value-keyed path
    could ever know it. Claiming it here would be dishonest output.

    The length of the removed value is deliberately omitted too: it is a small
    but real oracle on a short credential, and side-channels through
    "harmless" metadata are the lesson of the incident.

    Guillemets are chosen because they do not occur in ordinary tool output,
    so the marker is greppable and unambiguous, and because
    :data:`_PLACEHOLDER_RE` recognises them — which is what makes scrubbing
    idempotent across the two choke points.
    """
    return f"«REDACTED:flag={flag.lstrip('-').lower()}»"


#: Emitted once, next to a result whose text was scrubbed, to tell the agent
#: WHAT happened. This is the actionable half of the fix: in the incident the
#: command's failure was hidden behind ``exit code: 0`` (a ``2>&1 | head`` pipe
#: manufactured it), so the agent had no signal that anything went wrong at
#: all. Measured firing rate is 3 in 2,807 sessions, so it can afford to be a
#: full sentence.
CREDENTIAL_ECHO_NOTICE: Final = (
    "[harness] A credential-typed option was echoed with its value by a tool "
    "that rejected it; the value has been removed. This usually means the "
    "target binary does not implement that flag (BusyBox wget rejects "
    '--user/--password). Prefer --header="Authorization: Basic ..." built '
    "inside the target environment. The command's exit code may be 0 anyway "
    "if its output was piped."
)


def scrub_credential_echo(text: str) -> tuple[str, list[str]]:
    """Replace credential values echoed by a tool's option-rejection message.

    Value-AGNOSTIC: unlike ``redact_secret_values``, this does not need to know
    the secret. It recognises the SHAPE of a parser rejecting a
    credential-typed option with its value attached — the BusyBox
    ``wget --password`` family — which is the only class of leak where the
    harness never held the value and therefore had nothing to match on.

    Returns ``(scrubbed_text, flags)`` where ``flags`` names each credential
    flag whose value was removed, so a caller can surface WHAT was scrubbed
    without re-handling the value.
    """
    if not text or not _may_contain_echo(text):
        return text, []

    flags: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        value = match.group("value")
        # Order matters: the placeholder check is what makes this idempotent
        # and what keeps prose about the incident readable. The length floor
        # runs second because it is the cheaper "this is noise" test.
        if _PLACEHOLDER_RE.match(value) or len(value) < _MIN_VALUE_LEN:
            return match.group(0)
        flag = match.group("flag")
        flags.append(flag.lstrip("-").lower())
        return f"{match.group('lead')}{flag}{match.group('eq')}{_marker(flag)}"

    return CREDENTIAL_ECHO_RE.sub(_replace, text), flags


def redact_tool_output(
    text: str,
    value_redact: Callable[[str], str] | None,
    *,
    notice: bool = False,
) -> str:
    """The composed scrubber: value-keyed first, then shape-based.

    Order matters. Value-keyed runs FIRST so a secret the harness *does* know
    is replaced by ``[redacted]`` — which the shape pass then recognises as a
    placeholder and leaves alone. Reversed, the shape pass would consume the
    value and the value-keyed pass would find nothing, losing the more precise
    marker that names the variable.

    ``notice`` appends :data:`CREDENTIAL_ECHO_NOTICE` when the shape pass
    actually fired. It is opt-in rather than automatic because this function
    runs at more than one choke point on the same text: only the
    model-visible one (``Loop._append_results``) has an audience for the
    advice, and appending it at each pass would stack duplicates. The
    already-present guard keeps even that path idempotent.
    """
    if value_redact is not None:
        redacted = value_redact(text)
        if isinstance(redacted, str):
            text = redacted
    text, flags = scrub_credential_echo(text)
    if notice and flags and CREDENTIAL_ECHO_NOTICE not in text:
        return f"{text}\n\n{CREDENTIAL_ECHO_NOTICE}"
    return text


#: Bounds for :func:`scrub_details`. ``details`` carries arbitrary third-party
#: MCP payloads and the walk runs on the event loop, so it must have a hard
#: ceiling rather than trusting the shape of a remote server's response. Both
#: numbers sit far above any real payload observed in the stored corpus
#: (persisted keys are ``spill``, ``diff``, ``path``, ``url``, ``status`` and
#: ``server_result``); they exist to bound the pathological case, not to
#: trim a normal one.
_MAX_DETAILS_DEPTH: Final = 8
_MAX_DETAILS_NODES: Final = 5000


def scrub_details(details: Mapping[str, Any], redact: Callable[[str], str]) -> dict[str, Any]:
    """Recursively apply ``redact`` to every string leaf of a details mapping.

    ``details`` never reaches a provider but IS persisted to the transcript
    (``loop.py`` copies it into ``provider_payload``, which
    ``session/transcript.py`` serializes to ``transcript.jsonl``), so an
    unredacted leaf is a durable on-disk leak even though no model ever saw
    it. ``mcp/tool_bridge.py`` puts the ENTIRE raw MCP server result in
    ``details['server_result']``, and its text blocks are stripped only when a
    spill occurred — so a server that echoes a credential in a
    under-the-threshold text block writes it to disk past both text choke
    points. That is a live, value-keyed gap independent of the shape problem.

    Bounded by depth and node count. Over budget, the offending subtree is
    returned UNMODIFIED rather than dropped or marked: ``provider_payload``
    is load-bearing for compaction and transcript rendering, and silently
    mutating a structure we declined to walk would trade a narrow leak for
    broad data corruption. The caps are set where no real payload reaches
    them.

    Mapping KEYS are left alone. They are structural — renderers, compaction
    and the transcript writer index on them — and a credential echo is a
    value-shaped event.
    """
    budget = [_MAX_DETAILS_NODES]

    def _walk(node: Any, depth: int) -> Any:
        if depth > _MAX_DETAILS_DEPTH or budget[0] <= 0:
            return node
        budget[0] -= 1
        if isinstance(node, str):
            return redact(node)
        if isinstance(node, Mapping):
            return {key: _walk(value, depth + 1) for key, value in node.items()}
        # str is a Sequence, so it must be handled above this branch; bytes
        # are excluded because they are opaque payloads, not text the scrubber
        # can reason about.
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            walked = [_walk(item, depth + 1) for item in node]
            return tuple(walked) if isinstance(node, tuple) else walked
        return node

    return {key: _walk(value, 1) for key, value in details.items()}


@dataclass(frozen=True)
class CredentialFlagFinding:
    """One credential-typed flag passed to a known HTTP fetcher."""

    flag: str
    fetcher: str


#: Fetchers whose flag support varies by build in a way that bites. The list
#: is short on purpose: scoping the lint to ``wget``/``curl`` was measured at
#: ~4.3 firings per 1,000 sessions, rare enough to be signal. The unscoped
#: version fires ~36.7 per 1,000 and is ~80% ``export FOO_PASSWORD=...``,
#: which carries no flag-echo risk at all — warning fatigue there would train
#: the model to ignore harness notes, degrading the warning that does matter.
_FETCHER_RE: Final = re.compile(r"(?<![\w./-])(wget|curl)(?![\w.-])")

#: The OUTGOING-command form: leading dashes are REQUIRED here, which is the
#: single rule that excludes ``export FOO_PASSWORD=x`` and ``set -a; PASS=x``
#: without needing to parse shell syntax. (Contrast :data:`_CRED_FLAG`, which
#: must accept the bare form because BusyBox strips the dashes when echoing.)
_CRED_ARG_RE: Final = re.compile(rf"(?<![\w-])(--?{_CRED_FLAG})(?=[= ])", re.IGNORECASE)


def lint_credential_flags(command: str) -> list[CredentialFlagFinding]:
    """Pre-flight: credential-typed flags in an OUTGOING shell command.

    WARN ONLY. This never blocks and never rewrites, both deliberately.
    Rewriting ``--password=$P`` into a header requires knowing the tool, its
    auth scheme, whether the endpoint accepts Basic and whether the value
    needs base64 — while editing a string that may sit inside nested quoting,
    a heredoc, or a ``kubectl exec -- sh -c '...'`` (the incident's command is
    three quoting levels deep). A rewrite that silently changes semantics
    inside a production pod is worse than the leak it prevents.

    A finding requires the flag to appear AFTER a fetcher invocation. That
    ordering is how the real shape reads (``wget ... --password=...``) and it
    keeps an unrelated earlier assignment from borrowing a later ``wget``'s
    risk. Quoting depth is ignored on purpose: the incident's command is
    nested inside two layers of ``sh -c``, so any parser strict enough to
    respect quoting would have missed it.
    """
    fetcher = _FETCHER_RE.search(command)
    if fetcher is None:
        return []
    findings: list[CredentialFlagFinding] = []
    seen: set[str] = set()
    for match in _CRED_ARG_RE.finditer(command, fetcher.end()):
        flag = match.group(1)
        key = flag.lower()
        if key in seen:
            continue
        seen.add(key)
        findings.append(CredentialFlagFinding(flag=flag, fetcher=fetcher.group(1)))
    return findings


def format_credential_flag_warning(findings: Sequence[CredentialFlagFinding]) -> str:
    """The one-line note prepended to a bash result when the lint fires.

    Delivered on the RESULT rather than as a pre-execution gate or a UI
    notice: a gate would have to block (forbidden) and a notice reaches the
    user, not the model that must change the command. On the result it is read
    in the same breath as the output, costs nothing when it does not fire, and
    stays useful after the fact — it explains the ``exit code: 0`` that a pipe
    manufactured.
    """
    flags = ", ".join(finding.flag for finding in findings)
    fetcher = findings[0].fetcher
    return (
        f"[harness] Note: {flags} was passed to {fetcher}. If the target ships "
        f"BusyBox {fetcher} (common in slim images), it rejects that flag and "
        "ECHOES ITS VALUE to stdout. Build an Authorization header inside the "
        "target instead. Exit code may be 0 anyway if the output is piped."
    )
