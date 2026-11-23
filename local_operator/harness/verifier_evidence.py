"""Opt-in typed evidence and fail-closed assessments for completion checks.

Evidence IDs name content supplied by the embedder, never paths/URLs invented
by the checking model. This is deliberately independent of evaluation tasks:
any SDK caller can require a visual check without relying on prose disclaimers.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import ImageContent, TextContent, ToolResult
from local_operator.harness.verifier import MAX_FINDINGS_CHARS, VerdictStatus, _bounded

Modality = Literal["text", "image"]


def validate_required_image(image: ImageContent) -> None:
    """Required evidence cannot rely on the ordinary unknown-format tolerance.

    The normal render deliberately preserves opaque bytes for the provider.
    A completion check instead needs a positive decodable-image answer; use
    Pillow's existing decoder, without adding a second resizing/size policy.
    """
    from PIL import Image

    try:
        payload = base64.b64decode(image.data, validate=True)
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.verify()
        # Some containers' verify() checks only their header. A truncated JPEG
        # is not visual evidence merely because its dimensions can be read.
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.load()
    except Exception as error:
        raise ValueError("required image evidence is missing or corrupt") from error


@dataclass(frozen=True)
class VerifierEvidenceItem:
    evidence_id: str
    content: TextContent | ImageContent

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be nonempty")

    @property
    def modality(self) -> Modality:
        return "image" if isinstance(self.content, ImageContent) else "text"


@dataclass(frozen=True)
class VerifierEvidence:
    items: tuple[VerifierEvidenceItem, ...] = ()
    required_modalities: frozenset[Modality] = frozenset()
    # Only results actually delivered through this check's counted host tools
    # enter the registry. Mutable state is per verification, not global/session.
    _inspections: dict[str, VerifierEvidenceItem] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @property
    def registry(self) -> dict[str, VerifierEvidenceItem]:
        return {**{item.evidence_id: item for item in self.items}, **self._inspections}

    def record_result(self, result: ToolResult) -> ToolResult:
        if result.is_error:
            return result
        registered = []
        for block in result.content:
            if isinstance(block, (TextContent, ImageContent)):
                # A caller-selected ID must not shadow an inspection receipt.
                number = len(self._inspections) + 1
                ref = f"inspection-{number}"
                while ref in self.registry:
                    number += 1
                    ref = f"inspection-{number}"
                self._inspections[ref] = VerifierEvidenceItem(ref, block)
                registered += [TextContent(text=f"Evidence ID: {ref}"), block]
            else:
                registered.append(block)
        return result.model_copy(update={"content": registered})

    def __post_init__(self) -> None:
        if len({item.evidence_id for item in self.items}) != len(self.items):
            raise ValueError("evidence IDs must be unique")
        if self.required_modalities - {"text", "image"}:
            raise ValueError("unsupported evidence modality")

    @property
    def images(self) -> tuple[ImageContent, ...]:
        return tuple(item.content for item in self.items if isinstance(item.content, ImageContent))

    def prompt(self) -> str:
        # ONE schema in words and ONE example that is itself a valid answer.
        # THE EXAMPLE ROW IS ``UNVERIFIED`` ON PURPOSE: it cites a real
        # supplied ID, so as ``MET`` it decoded to PASS whenever that item
        # satisfied the required modality -- a capped child that echoed the
        # prompt back verbatim would have passed the check it never ran
        # (PR #4236 review m2). ``UNVERIFIED`` is still a well-formed answer
        # and fails closed (INCONCLUSIVE) if copied.
        # The earlier text carried a pseudo-JSON template and listed evidence
        # as ``- frame:screen: image``; twelve recorded checks show what a
        # model made of that: eight added a ``verdict``/``verdict_line`` key to
        # reconcile this prompt with the role's VERDICT line, and two cited
        # the supplied frame as ``frame:screen:image``. So IDs are quoted and
        # the keys are enumerated, with the redundant ones named.
        cite = self.items[0] if self.items else None
        example = {
            "requirements": [
                {
                    "requirement": "one requirement from the request",
                    "status": "UNVERIFIED",
                    "modality": cite.modality if cite else "text",
                    # Only an ID this check really holds: an invented one in
                    # the example would be copied into an unresolvable row.
                    "evidence_ids": [cite.evidence_id if cite else "inspection-1"],
                    "findings": "what you observed in the cited evidence",
                }
            ]
        }
        rows = [
            "TYPED EVIDENCE (IDs refer only to the supplied content, never controller paths; "
            "cite an ID exactly as quoted, the modality in brackets is not part of it):",
            *[
                f"- {json.dumps(item.evidence_id)} ({item.modality})"
                + (f"\n{item.content.text}" if isinstance(item.content, TextContent) else "")
                for item in self.items
            ],
            "Image blocks follow in the image-ID order listed above.",
            "Required modalities: " + ", ".join(sorted(self.required_modalities)),
            "Return ONE JSON object and nothing else: no markdown fence, no VERDICT line, "
            "no prose before or after it.",
            'Schema: the object has exactly one key, "requirements": a non-empty array of rows. '
            'Each row has exactly five keys: "requirement" (string, one requirement from the '
            'request), "status" ("MET", "UNMET" or "UNVERIFIED"), "modality" ("text" or '
            '"image", the kind of evidence the row rests on), "evidence_ids" (array of '
            'Evidence ID strings) and "findings" (string, what you observed).',
            'Do not add "verdict", "verdict_line", a summary or any other key at either level: '
            "the overall verdict is computed from the rows alone and any other key is ignored.",
            "Example: " + json.dumps(example, separators=(",", ":")),
            # The example's status is a placeholder (see above); say so, or
            # the one concrete row in the prompt anchors answers on it.
            "The example shows the shape only: give each row the status your own "
            "observation of the evidence supports.",
            "Assess each requirement you identify independently. UNMET means observed failure; "
            "UNVERIFIED means insufficient evidence. "
            # Two row rules that RAISE the bar, each from recorded checks. A
            # row went UNMET because its source could not be read (an
            # unreadable PDF) -- a failure of the check, not of the work -- and
            # rows went MET on a filename or a mean-colour statistic, which
            # say a file exists, not that its content is right; those were the
            # rows that were actually wrong.
            "UNMET requires an observed failure; a missing justification or an unreadable "
            "source is UNVERIFIED. MET requires observed content, not a filename, size or "
            "summary statistic. "
            "A MET row must cite at least one Evidence "
            "ID of its own modality (an image row cites an image ID, not only text ones). "
            "Required modalities must actually be used. Do not infer unseen pixels from text. "
            "Successful inspections return additional Evidence IDs you may cite. "
            "This structured response replaces the role's legacy VERDICT-line format.",
        ]
        return "\n".join(rows)


#: Budget cues for a typed check. The shared ones end "ending with the VERDICT
#: line", which is right for the legacy format and contradicts this one: all
#: twelve recorded typed checks spent their command budget, so that sentence
#: was the last thing each read before answering, and eight answered with a
#: verdict key bolted onto the JSON.
TYPED_COMMAND_BUDGET_REFUSAL = (
    "Command budget used ({limit} of {limit}): answer now from what you have already "
    "seen, with the JSON assessment object only."
)
TYPED_REQUEST_BUDGET_REFUSAL = (
    "Request budget used: answer now from what you have already seen, with the JSON "
    "assessment object only."
)

MALFORMED_ASSESSMENT = "Missing or malformed typed completion assessment"


# DECODE CONTRACT (both models). Every field the verdict reads is REQUIRED and
# strictly typed; a key the models do not declare is IGNORED, never read.
#
# Ignoring is safe only because of the first half: no field has a default, so
# a misspelt key (``evidence`` for ``evidence_ids``) still fails as a missing
# field instead of silently dropping what it meant to say, and nothing outside
# these fields can reach the verdict. ``extra="forbid"`` looked stricter and
# was not safer: models append a redundant ``verdict`` key to otherwise valid
# rows, and forbidding it threw away the rows -- an all-MET answer became
# INCONCLUSIVE and UNVERIFIED findings never reached the worker. A redundant
# key is not trusted in EITHER direction (it is free text; one was a whole
# sentence): ``"verdict": "PASS"`` beside an UNMET row is a FAIL, and a model
# that observed a failure must say so in a row.
class RequirementAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    requirement: str = Field(min_length=1, pattern=r"\S")
    status: Literal["MET", "UNMET", "UNVERIFIED"]
    modality: Modality
    evidence_ids: list[str]
    findings: str


class CompletionAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    requirements: list[RequirementAssessment] = Field(min_length=1)


_JSON_KINDS = {list: "array", str: "string", bool: "boolean", type(None): "null"}


def _quoted(value: object) -> str:
    """Model-authored text inside a diagnostic, on one line.

    JSON-escaping keeps a newline in it from starting a line of its own: the
    challenge builder reads ``UNMET: ...`` lines as repair cues, and a
    rejected answer must not be able to forge one.
    """

    return json.dumps(value, ensure_ascii=False)


def _decode_assessment(text: str | None) -> CompletionAssessment:
    """Parse one answer or raise ``ValueError`` saying what was wrong with it.

    The message goes back to the worker and into the record, so it names the
    field: one fixed "malformed" sentence left eight recorded checks with
    nothing to act on, and nothing in the record to show that each answer
    was valid apart from one redundant key.
    """

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        # ``json`` keeps the LAST of two equal keys, so a duplicated
        # ``status`` could show a reader UNMET and the decoder MET.
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {_quoted(key)}.")
            result[key] = value
        return result

    if not (text or "").strip():
        raise ValueError("the check returned no answer.")
    try:
        decoded = json.loads(text or "", object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        # Prose, a markdown fence or a trailing VERDICT line all land here.
        raise ValueError(
            f"the answer is not one JSON object ({error.msg} at line {error.lineno} "
            f"column {error.colno})."
        ) from error
    if not isinstance(decoded, dict):
        kind = _JSON_KINDS.get(type(decoded), "number")
        raise ValueError(f'the top level is a JSON {kind}, not an object with "requirements".')
    try:
        return CompletionAssessment.model_validate(decoded)
    except ValidationError as error:
        problems = []
        for item in error.errors(include_url=False):
            where = ""
            for part in item["loc"]:
                where += f"[{part}]" if isinstance(part, int) else f".{part}" if where else part
            got = item["input"]
            # A scalar is worth echoing ("PARTIAL"); a container would repeat
            # the answer, and a missing field's input is its whole parent.
            echo = item["type"] != "missing" and not isinstance(got, (dict, list))
            problems.append(
                f"- {where}: {item['msg']}" + (f" (got {_quoted(got)})" if echo else "")
            )
        raise ValueError("these fields are invalid:\n" + "\n".join(problems)) from error


def assess_completion(text: str | None, evidence: VerifierEvidence) -> tuple[VerdictStatus, str]:
    """One decoder for normal and capped answers; no prose phrase matching.

    The verdict is derived ONLY from the typed rows and the receipts this
    check admitted (see the decode contract above the models). This validates
    the requirements the model enumerated, not a claim that the model found
    every requirement in the request. UNMET remains a repair cue in the legacy
    outward findings shape used by completion challenges.
    """

    try:
        assessment = _decode_assessment(text)
    except ValueError as error:
        return "INCONCLUSIVE", _bounded(f"{MALFORMED_ASSESSMENT}: {error}", MAX_FINDINGS_CHARS)
    # Repair cues first so a long MET explanation cannot crowd them out of the
    # existing outward findings budget: an observed failure, then what could
    # not be checked, then what held.
    order = ("UNMET", "UNVERIFIED", "MET")
    rows = "\n".join(
        f"{row.status}: {row.requirement}\n{row.findings}"
        for row in sorted(assessment.requirements, key=lambda row: order.index(row.status))
    )
    if any(row.status == "UNMET" for row in assessment.requirements):
        return "FAIL", _bounded(rows, MAX_FINDINGS_CHARS)
    registry = {ref: item.modality for ref, item in evidence.registry.items()}
    used: set[str] = set()
    # Why each row fell short, by its position in the answer. "Required
    # evidence was not verified." alone gave a worker whose every row was MET
    # nothing to act on, and hid that the shortfall was the checker's citation
    # (an ID it mistyped, an image claim resting on text receipts).
    gaps: list[str] = []
    for number, row in enumerate(assessment.requirements, 1):
        unknown = [ref for ref in row.evidence_ids if ref not in registry]
        if row.status != "MET":
            gaps.append(f"requirement {number} is {row.status}")
        elif not row.evidence_ids:
            gaps.append(f"requirement {number} cites no evidence")
        elif unknown:
            gaps.append(
                f"requirement {number} cites evidence that was never supplied or returned "
                f"({', '.join(map(_quoted, unknown))})"
            )
        elif not any(registry[ref] == row.modality for ref in row.evidence_ids):
            gaps.append(
                f"requirement {number} is assessed as {row.modality} but cites no "
                f"{row.modality} evidence"
            )
        else:
            used.add(row.modality)
    # The reason leads and shares the one budget with the rows, so truncation
    # can never drop the sentence that says why this is not a PASS.
    if gaps:
        reason = "Required evidence was not verified: " + "; ".join(gaps) + "."
        return "INCONCLUSIVE", _bounded(f"{reason}\n{rows}", MAX_FINDINGS_CHARS)
    if missing := sorted(evidence.required_modalities - used):
        reason = (
            "Required media was not assessed: no MET requirement cites "
            f"{' or '.join(missing)} evidence."
        )
        return "INCONCLUSIVE", _bounded(f"{reason}\n{rows}", MAX_FINDINGS_CHARS)
    return "PASS", _bounded(rows, MAX_FINDINGS_CHARS)
