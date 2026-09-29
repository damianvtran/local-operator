"""The final-response OUTPUT CONTRACT: a format (and optional schema) on a turn's end.

WHY THIS EXISTS. There is no provider-side structured output anywhere in this
tree — no ``response_format``/``json_schema`` reaches any wire client — so
"the final response must be JSON (and match this schema)" can only be a
CLIENT-side check plus a bounded retry: the harness validates the turn's
terminal assistant text at the one point a turn is finalized, and feeds a
failed check back as one more model step inside the same turn. This module
owns the vocabulary and the validation; the loop owns the seam
(``LoopConfig.final_response_gate``) and the session owns the state
(``Session.set_output_contract``).

THE SHAPE OF A CHECK. ``check`` is deliberately TOLERANT ABOUT LOCATING the
payload and STRICT ABOUT ITS CONTENT — a model that wraps its answer in a
fence, or in a sentence, is still answering; a model that emits ``NaN`` is
not emitting JSON. Candidates are tried in a fixed order (labelled fences in
document order; unlabelled/unknown-label fences in document order; the whole
text; for json only, bare ``{``/``[``-anchored spans), and the first
candidate that both decodes and satisfies the schema wins. A block labelled
with a *different* recognised format is skipped rather than crossed: the
label is an explicit claim, and crossing it is where silent acceptance
starts (the retry message is the remedy). Cross-format schema misuse is
refused at construction, not at the first prompt.

IMPORT WEIGHT. Stdlib + pydantic + pyyaml at module scope; ``jsonschema`` is
imported lazily inside the raw-JSON-Schema branch only, so the type-schema
path (and every session without a contract) never pays for it. A session
with no contract installs no block and runs byte-identically to before this
module existed.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from pydantic import TypeAdapter, ValidationError

from local_operator.harness.types import AgentMessage, FinalResponseCheck, Message

#: The formats a contract can enforce, in the order error messages name them.
OutputFormat = Literal["markdown", "json", "yaml", "toml"]
OUTPUT_FORMATS: tuple[str, ...] = ("markdown", "json", "yaml", "toml")

#: Fence info-string first tokens that map to a payload format. ``yml`` is
#: YAML's usual alias. ``markdown`` is here so a ```markdown block is treated
#: as an explicit claim by a structured target (skipped, like any other
#: recognised label) — one of the four formats IS markdown, so the label is
#: recognised even though markdown mode itself extracts nothing.
_FENCE_LABELS: dict[str, str] = {
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "markdown": "markdown",
}

#: One failure reason is carried into a retry prompt, an event and a stderr
#: line; a JSON-schema failure can render several paragraphs of context, so
#: everything downstream of a check gets ONE bound. 600 characters is the
#: design's number: long enough for a real schema message, short enough to
#: stay a sentence.
_ERROR_LIMIT = 600
_TRUNCATION_SUFFIX = "… (truncated)"

#: Minimum fence length (CommonMark's rule, and ours: the closer must repeat
#: the opener's marker at least this many times).
_MIN_FENCE = 3

#: How many ``{``/``[`` starts the json prose scan tries. Bounded work on
#: adversarial text; a payload behind fifty brace starts is not a model
#: answer, it is a corpus.
_BARE_SCAN_LIMIT = 50

#: How many schema violations render into one reason. Three names the class of
#: the mistake; the retry is the remedy for the rest.
_SCHEMA_ERROR_LIMIT = 3

#: The pinned reason for an empty candidate text (the design names it).
_EMPTY_REASON = "the final message was empty"
_UNCLOSED_FENCE_REASON = "the markdown contains an unterminated fenced code block"
_NO_PAYLOAD_REASON = "no payload could be extracted"

#: ATX headings only (``#``…``######`` followed by whitespace): markdown has
#: no other heading shape, and a setext underline is not what models emit for
#: a requested section.
_ATX_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


class OutputContractError(ValueError):
    """Construction-time problems: unknown format, bad retries, unusable schema."""


class OutputDecodeError(ValueError):
    """A payload could not be located or decoded.

    Public through :func:`decode_output` so callers share the contract's own
    extraction ladder instead of writing a second one that drifts.
    """


@dataclass(frozen=True, slots=True)
class MarkdownSchema:
    """The markdown half of the schema vocabulary: ordered required sections.

    A JSON Schema has nothing to say about a markdown document, so markdown
    schemas are this object — or its ``{"required_sections": [...]}`` mapping
    spelling, which is what a ``--output-schema`` file carries. Sections name
    ATX headings; comparison is case-insensitive with whitespace collapsed,
    and the order given is the order REQUIRED (subsequence, not adjacency).
    """

    required_sections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputContract:
    """One session's final-response contract: format, optional schema, retries.

    Installed on a session post-open (``Session.set_output_contract``) exactly
    like the tool inventory, and read by the loop through the
    ``FinalResponseGate`` protocol this class implements. It is a VALUE, not
    session state: everything expensive (the schema validator, the system
    block, the section list) is built once here, at construction — so a bad
    schema fails BEFORE the first prompt, and a check costs one decode plus
    one validation.

    ``schema`` is deliberately ``object``-typed: for json/yaml/toml it is a
    raw JSON Schema mapping, or anything ``pydantic.TypeAdapter`` accepts (a
    model class, a dataclass, a TypedDict, a generic alias); for markdown it
    is a :class:`MarkdownSchema` or its mapping spelling. ``None`` enforces
    the format alone.
    """

    format: OutputFormat
    schema: object | None = None
    retries: int = 2
    # Caches, built once in ``__post_init__``. Declared as fields because a
    # frozen slotted dataclass has no other place to put instance state; they
    # are ``init=False`` so the public constructor stays three arguments.
    _type_adapter: Any = field(default=None, init=False, repr=False, compare=False)
    _json_validator: Any = field(default=None, init=False, repr=False, compare=False)
    _required_sections: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _system_block: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.format not in OUTPUT_FORMATS:
            raise OutputContractError(
                "format must be one of " + ", ".join(OUTPUT_FORMATS) + f", not {self.format!r}"
            )
        if (
            not isinstance(self.retries, int)
            or isinstance(self.retries, bool)
            or not 0 <= self.retries <= 5
        ):
            raise OutputContractError("retries must be between 0 and 5")
        if self.format == "markdown":
            object.__setattr__(self, "_required_sections", _markdown_sections(self.schema))
        else:
            self._build_structured_validator()
        object.__setattr__(self, "_system_block", self._render_system_block())

    # -- the gate protocol ------------------------------------------------

    @property
    def max_attempts(self) -> int:
        """1 + retries: the total number of checked attempts in one run."""
        return 1 + self.retries

    @property
    def label(self) -> str:
        """The format name as every surface spells it ("json", "markdown", …)."""
        return self.format

    def check(self, text: str) -> FinalResponseCheck:
        """Validate one candidate final response.

        Never raises on a bad payload — a rejection IS the verdict here — and
        the failure reason prefers the FIRST schema failure seen across
        candidates, then the first decode failure in candidate order, so the
        retry message names the most actionable problem rather than the last
        one alphabetically.
        """
        if not text.strip():
            return FinalResponseCheck(ok=False, error=_EMPTY_REASON)
        if self.format == "markdown":
            return self._check_markdown(text)
        first_decode_error = ""
        first_schema_error = ""
        for payload in _iter_candidates(text, self.format):
            try:
                value = _decode_payload(payload, self.format)
            except OutputDecodeError as error:
                if not first_decode_error:
                    first_decode_error = str(error)
                continue
            schema_error = self._schema_error(value)
            if schema_error:
                if not first_schema_error:
                    first_schema_error = schema_error
                continue
            return FinalResponseCheck(ok=True, payload_text=payload)
        reason = first_schema_error or first_decode_error or _NO_PAYLOAD_REASON
        return FinalResponseCheck(ok=False, error=_bounded(reason))

    def retry_message(self, *, attempt: int, error: str) -> AgentMessage:
        """The harness-injected user message that asks for a corrected answer.

        An ordinary (visible, persisted) user message on purpose — the shape
        the repeated-error recovery notice established — so a resumed
        transcript explains why the conversation continued. The failing text
        is NOT echoed: the model already has it as its own last message, and
        echoing it would double its cost against the context window.
        """
        return Message.user(
            f"Harness output check: {_bounded(error)}\n"
            f"The required output format is {self.label}. {self.instructions()} "
            f"Reply with the corrected final answer only (attempt {attempt} of "
            f"{self.max_attempts})."
        )

    def exhausted_error(self, *, attempts: int, error: str) -> str:
        """The terminal error text for a run that never satisfied the contract.

        This exact sentence is what the CLI prints, what a background job's
        ledger failure names, and what lands on the SDK's ``agent_end`` — one
        spelling, so no surface has to re-derive why the turn failed.
        """
        return (
            f"final response did not satisfy the output contract ({self.label}) "
            f"after {attempts} attempts: {_bounded(error)}"
        )

    # -- prompt furniture --------------------------------------------------

    def instructions(self) -> str:
        """The per-format instruction sentence used by the retry message.

        Composed from ``_payload_shape()`` — the same clause the system block
        shows — and the shared sections clause: ONE spelling of each, so the
        retry message, the announcement and the decoder cannot drift apart
        (review R-5).
        """
        if self.format == "markdown":
            sentence = "Reply with the corrected markdown document only."
            clause = self._sections_clause()
            return f"{sentence} {clause}" if clause else sentence
        return f"Reply with {self._payload_shape()}."

    def system_block(self) -> str:
        """The one-shot announcement of this contract, built at construction.

        Shipped whole (a partial schema is worse than none, and the caller
        opted in): installed before the first turn it rides the session's
        frozen system prefix, and a rare mid-session set arrives as an
        ordinary ``[session-state]`` delta.
        """
        return self._system_block

    # -- internals ---------------------------------------------------------

    def _build_structured_validator(self) -> None:
        """Build the cached validator for a json/yaml/toml schema, or refuse."""
        schema = self.schema
        if schema is None:
            return
        if isinstance(schema, MarkdownSchema):
            raise OutputContractError("MarkdownSchema applies to format 'markdown' only")
        if isinstance(schema, Mapping) or isinstance(schema, bool):
            # Raw JSON Schema. The meta-schema check happens HERE (not at the
            # first prompt) so a bad schema is a startup failure. The lazy
            # import is the module docstring's contract: the type-schema path
            # and every contract-less run never load jsonschema.
            from jsonschema import Draft202012Validator

            try:
                Draft202012Validator.check_schema(_plain(schema))
            except Exception as error:  # noqa: BLE001 — jsonschema's own failure classes
                raise OutputContractError(
                    f"schema is not a valid JSON Schema: {_reason(error)}"
                ) from error
            if self.format == "toml":
                # A TOML document's root is always a table, so a schema that
                # requires any other root type can never be satisfied —
                # refused here rather than failing every attempt at runtime.
                declared = schema.get("type") if isinstance(schema, Mapping) else None
                if declared is not None and declared != "object":
                    raise OutputContractError(
                        "a TOML payload is always a table; the schema must be an object "
                        'schema ("type": "object", or no type at all)'
                    )
            from jsonschema.validators import validator_for

            validator_class = validator_for(_plain(schema))
            object.__setattr__(self, "_json_validator", validator_class(_plain(schema)))
            return
        try:
            adapter = TypeAdapter(schema)
        except Exception as error:  # noqa: BLE001 — pydantic's own failure classes
            raise OutputContractError(
                f"schema is not a type pydantic can validate: {error}"
            ) from error
        object.__setattr__(self, "_type_adapter", adapter)

    def _schema_error(self, value: Any) -> str:
        """``""`` when the payload satisfies the schema, else one bounded reason.

        Lax pydantic semantics by default (``"3"`` coerces to ``3``) — that is
        ``model_validate``'s familiar behaviour, and STRICTNESS is the model's
        own business (``StrictInt`` and friends). Raw JSON Schemas are checked
        with ``iter_errors``, format annotations deliberately not asserted.
        """
        adapter = self._type_adapter
        if adapter is not None:
            try:
                adapter.validate_python(value)
            except ValidationError as error:
                return "payload does not match the schema: " + _render_pydantic_error(error)
            return ""
        validator = self._json_validator
        if validator is not None:
            problems: list[str] = []
            for violation in validator.iter_errors(value):
                problems.append(f"{violation.json_path}: {violation.message}")
                if len(problems) >= _SCHEMA_ERROR_LIMIT:
                    break
            if problems:
                return "payload does not match the schema: " + "; ".join(problems)
        return ""

    def _check_markdown(self, text: str) -> FinalResponseCheck:
        """Markdown's whole contract: non-empty, balanced fences, sections."""
        for fence in _scan_fences(text):
            if not fence.closed:
                return FinalResponseCheck(ok=False, error=_UNCLOSED_FENCE_REASON)
        sections_error = self._sections_error(text)
        if sections_error:
            return FinalResponseCheck(ok=False, error=sections_error)
        return FinalResponseCheck(ok=True, payload_text=text.strip())

    def _sections_error(self, text: str) -> str:
        """Ordered-subsequence check over ATX headings, or ``""``.

        Greedy matching with a lookahead EXPLANATION: when a required section
        is not found after the previous match, it either appears earlier in
        the document (a better message than "missing") or it is genuinely
        absent.
        """
        required = self._required_sections
        if not required:
            return ""
        headings = [_normalize_section(body) for body in _atx_headings(text)]
        position = 0
        matched = ""
        for name, wanted in zip(required, (_normalize_section(item) for item in required)):
            found = -1
            for scan in range(position, len(headings)):
                if headings[scan] == wanted:
                    found = scan
                    break
            if found == -1:
                if wanted in headings:
                    return (
                        f"section '{name}' appears before '{matched}'; "
                        f"required order: {', '.join(required)}"
                    )
                return f"missing required section '{name}'"
            position = found + 1
            matched = name
        return ""

    def _sections_clause(self) -> str:
        """``Required sections, in order: …`` — one spelling for the retry
        message and the system block (review R-5)."""
        if not self._required_sections:
            return ""
        return f"Required sections, in order: {', '.join(self._required_sections)}."

    def _payload_shape(self) -> str:
        """The format's payload clause, shared by the system block and the
        retry instruction (review R-5)."""
        if self.format == "json":
            return (
                "exactly one JSON value and nothing else; a fenced ```json "
                "block is also accepted"
            )
        if self.format == "yaml":
            return (
                "exactly one YAML mapping or sequence and nothing else; a fenced "
                "```yaml block is also accepted"
            )
        if self.format == "toml":
            return (
                "exactly one TOML document and nothing else; a fenced ```toml "
                "block is also accepted"
            )
        return "a whole markdown document"

    def _render_system_block(self) -> str:
        """Build the announcement once, at construction."""
        if self.format == "markdown":
            lines = [
                "Output contract: the final response for this session is enforced.",
                "- Format: markdown. The final response must be a whole markdown document.",
            ]
            clause = self._sections_clause()
            if clause:
                lines.append(f"- {clause}")
            return "\n".join(lines)
        lines = [
            "Output contract: the final response for this session is enforced.",
            f"- Format: {self.format}. The final response must be {self._payload_shape()}.",
        ]
        schema_block = self._schema_announcement()
        if schema_block:
            lines.append(schema_block)
        return "\n".join(lines)

    def _schema_announcement(self) -> str:
        """The schema's line in the system block, or ``""`` without a schema.

        A type schema is announced as its GENERATED JSON Schema: the model is
        shown the same shape a raw mapping would give it, in the vocabulary
        ``json.dumps`` can carry.
        """
        if self._type_adapter is not None:
            return "- It must validate against this schema (JSON Schema): " + json.dumps(
                self._type_adapter.json_schema(), indent=2
            )
        if self._json_validator is not None:
            return "- It must validate against this schema (JSON Schema): " + json.dumps(
                _plain(self.schema), indent=2
            )
        return ""


def decode_output(text: str, format: str) -> Any:
    """The payload of ``text`` for ``format``, using the contract's extraction.

    The public counterpart of the contract's decode step: the SAME candidate
    ladder (labelled fences, unknown-label fences, the whole text, json's
    bare-delimiter scan) and the same strictness, so a caller that saved an
    ``output_validation`` event's ``payload_text`` and a caller that hands the
    raw final message here get the same answer. Raises
    :class:`OutputDecodeError` with the contract's own reason on failure.

    The format name is validated first: an unknown name is a caller bug, not
    a property of the text.
    """
    if format not in OUTPUT_FORMATS:
        raise OutputDecodeError(
            "format must be one of " + ", ".join(OUTPUT_FORMATS) + f", not {format!r}"
        )
    if not text.strip():
        raise OutputDecodeError(_EMPTY_REASON)
    if format == "markdown":
        # Markdown has no decode step; the structural half of the contract's
        # check is its decode arm, so the two entry points agree about what
        # "this text is a usable markdown payload" means.
        for fence in _scan_fences(text):
            if not fence.closed:
                raise OutputDecodeError(_UNCLOSED_FENCE_REASON)
        return text.strip()
    first_error = ""
    for payload in _iter_candidates(text, format):
        try:
            return _decode_payload(payload, format)
        except OutputDecodeError as error:
            if not first_error:
                first_error = str(error)
    raise OutputDecodeError(_bounded(first_error or _NO_PAYLOAD_REASON))


# ---------------------------------------------------------------------------
# Candidate extraction — the tolerant LOCATOR (see the module docstring)
# ---------------------------------------------------------------------------


def _iter_candidates(text: str, format: str) -> Iterator[str]:
    """Yield candidate payload spans for ``format``, in the pinned order.

    Labelled fences first, then unlabelled/unknown-label fences, then the
    whole text, then — for json only — bare ``{``/``[``-anchored spans. Each
    yielded value is stripped; each is a CANDIDATE, decoded and validated by
    the caller, which moves on when either step says no.
    """
    labelled: list[str] = []
    unlabelled: list[str] = []
    claimed_spans: list[tuple[int, int]] = []
    for fence in _scan_fences(text):
        label = fence.info.split()[0].lower() if fence.info else ""
        claimed = _FENCE_LABELS.get(label)
        if claimed is None:
            unlabelled.append(fence.content)
        elif claimed == format:
            labelled.append(fence.content)
        else:
            # A different recognised label — an explicit claim, SKIPPED rather
            # than crossed: the retry message is the remedy. The bare json
            # scan honours the same skip below; without that, a ```yaml block
            # that happens to hold valid JSON would be re-found by the scan
            # and the skip would be vacuous.
            claimed_spans.append((fence.content_start, fence.content_end))
    for payload in (*labelled, *unlabelled):
        stripped = payload.strip()
        if stripped:
            yield stripped
    whole = text.strip()
    if whole:
        yield whole
    if format == "json":
        yield from _scan_bare_json_spans(text, skip=claimed_spans)


def _scan_bare_json_spans(text: str, *, skip: Sequence[tuple[int, int]] = ()) -> Iterator[str]:
    """Bare ``{``/``[``-anchored json spans, document order, bounded.

    ``skip`` regions (a cross-labelled fence's content) are not scanned: a
    label is an explicit claim, and the skip rule would be vacuous if the
    same bytes could be re-found here. YAML and TOML deliberately have no
    equivalent tier at all: every text is a YAML scalar, and a TOML document
    cannot be meaningfully located inside prose — a hand-heuristic there
    would be tolerance in the wrong direction.
    """
    decoder = json.JSONDecoder()
    starts = 0
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        if any(start <= index < end for start, end in skip):
            continue
        starts += 1
        if starts > _BARE_SCAN_LIMIT:
            return
        try:
            _value, end = decoder.raw_decode(text, index)
        except ValueError:
            continue
        yield text[index:end].strip()


@dataclass(frozen=True, slots=True)
class _Fence:
    """One top-level fenced block: info string, content, and its text span.

    ``content_start``/``content_end`` are offsets into the SCANNED text, used
    by the json bare scan to honour the cross-label skip; ``content_end``
    excludes the line ending before the closing fence.
    """

    info: str
    content: str
    closed: bool
    content_start: int
    content_end: int


def _scan_fences(text: str) -> list[_Fence]:
    """Every top-level fenced block, document order.

    The rules are pinned by the design and are NOT CommonMark: a fence opens
    on a line whose STRIPPED form starts with three or more backticks or
    tildes; it closes on the next line whose stripped form is only the same
    marker character, at least as long as the opener, and nothing else.
    Content inside an open fence is never rescanned, so a same-length inner
    fence closes the outer (a documented limit, standard for same-marker
    fences); an unterminated fence runs to the end of the text with
    ``closed=False`` — structured targets may still find a payload in it,
    while markdown's contract rejects the imbalance.
    """
    blocks: list[_Fence] = []
    lines = text.splitlines(keepends=True)
    offset = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        opened = _fence_open(line)
        if opened is None:
            offset += len(line)
            index += 1
            continue
        marker, run, info = opened
        index += 1
        offset += len(line)
        content_start = offset
        content_end = offset
        content: list[str] = []
        closed = False
        while index < len(lines):
            candidate = lines[index]
            if _fence_closes(candidate, marker, run):
                closed = True
                break
            content.append(candidate.rstrip("\r\n"))
            offset += len(candidate)
            content_end = offset
            index += 1
        blocks.append(_Fence(info, "\n".join(content), closed, content_start, content_end))
        if closed:
            offset += len(lines[index])
        index += 1
    return blocks


def _fence_open(line: str) -> tuple[str, int, str] | None:
    """``(marker, run length, info string)`` for a fence opener, else ``None``."""
    stripped = line.strip()
    if not stripped or stripped[0] not in "`~":
        return None
    marker = stripped[0]
    run = len(stripped) - len(stripped.lstrip(marker))
    if run < _MIN_FENCE:
        return None
    return marker, run, stripped[run:].strip()


def _fence_closes(line: str, marker: str, run: int) -> bool:
    """Whether ``line`` closes a fence opened with ``marker`` × ``run``."""
    stripped = line.strip()
    if not stripped or stripped[0] != marker:
        return False
    length = len(stripped) - len(stripped.lstrip(marker))
    return length >= run and not stripped[length:].strip()


# ---------------------------------------------------------------------------
# Decoding — the STRICT half (tolerance is about locating, never accepting)
# ---------------------------------------------------------------------------


def _decode_payload(payload: str, format: str) -> Any:
    """Decode one candidate, or raise :class:`OutputDecodeError` with the reason.

    Strictness per format is the contract's stated table: json is full strict
    JSON (``NaN``/``Infinity``/``-Infinity`` rejected — they are not JSON);
    YAML is PyYAML's SAFE loader with the structured-value rule (a bare
    scalar is prose, not a payload; ``None``/empty is a decode failure); TOML
    is stdlib 1.0, whose root is always a table.
    """
    if format == "json":
        try:
            return json.loads(payload, parse_constant=_reject_non_finite)
        except ValueError as error:
            raise OutputDecodeError(f"not valid JSON: {error}") from error
    if format == "yaml":
        try:
            value = yaml.safe_load(payload)
        except yaml.YAMLError as error:
            raise OutputDecodeError(f"not valid YAML: {_one_line(str(error))}") from error
        if value is None:
            raise OutputDecodeError("the YAML payload is empty")
        if not isinstance(value, (dict, list)):
            raise OutputDecodeError(
                "the YAML payload is a bare scalar; a mapping or sequence is required"
            )
        return value
    if format == "toml":
        try:
            return tomllib.loads(payload)
        except tomllib.TOMLDecodeError as error:
            raise OutputDecodeError(f"not valid TOML: {error}") from error
    raise OutputDecodeError(f"unknown output format {format!r}")


def _reject_non_finite(token: str) -> Any:
    """``json.loads`` hook for NaN/Infinity/-Infinity: not JSON, so not a payload.

    Python's decoder accepts these three literals by default (they are valid
    *Python*, not valid *JSON*); a contract whose whole job is "the answer is
    JSON" cannot inherit that.
    """
    raise ValueError(f"{token} is not valid JSON")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _markdown_sections(schema: object | None) -> tuple[str, ...]:
    """Normalize a markdown schema (object or mapping spelling) to a tuple."""
    if schema is None:
        return ()
    if isinstance(schema, MarkdownSchema):
        return tuple(schema.required_sections)
    if isinstance(schema, Mapping) and set(schema) == {"required_sections"}:
        value = schema["required_sections"]
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return tuple(value)
    raise OutputContractError(
        "a markdown schema is MarkdownSchema(required_sections=[...]) or "
        '{"required_sections": ["<section>", ...]}, not ' + repr(schema)
    )


def _atx_headings(text: str) -> list[str]:
    """Every ATX heading body, document order (trailing closing hashes trimmed)."""
    headings: list[str] = []
    for line in text.splitlines():
        match = _ATX_HEADING_RE.match(line.strip())
        if match:
            headings.append(re.sub(r"\s*#+\s*$", "", match.group(1)).strip())
    return headings


def _normalize_section(text: str) -> str:
    """Case-insensitive, whitespace-collapsed section identity."""
    return " ".join(text.casefold().split())


def _render_pydantic_error(error: ValidationError) -> str:
    """The first few pydantic violations as ``<loc path>: <message>``."""
    parts: list[str] = []
    for item in error.errors(include_url=False)[:_SCHEMA_ERROR_LIMIT]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def _bounded(reason: str) -> str:
    """One collapsed, length-bounded reason (see ``_ERROR_LIMIT``)."""
    collapsed = _one_line(reason)
    if len(collapsed) <= _ERROR_LIMIT:
        return collapsed
    keep = _ERROR_LIMIT - len(_TRUNCATION_SUFFIX) - 1
    return collapsed[:keep].rstrip() + " " + _TRUNCATION_SUFFIX


def _one_line(text: str) -> str:
    """Whitespace-collapse (a YAML error is several lines; a stderr line is one)."""
    return " ".join(text.split())


def _plain(schema: object) -> Any:
    """A plain dict/bool for jsonschema, whatever mapping subclass arrived.

    A ``Mapping`` check admits the contract; this hands jsonschema the same
    value in the exact types it documents (it walks ``dict`` attributes, and
    a user-supplied mapping class is not its contract).
    """
    if isinstance(schema, Mapping):
        return dict(schema)
    return schema


def _reason(error: Exception) -> str:
    """jsonschema exceptions carry a ``message``; fall back to ``str``."""
    message = getattr(error, "message", None)
    return _one_line(str(message if message else error))
