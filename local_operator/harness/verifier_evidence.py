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
        rows = [
            "TYPED EVIDENCE (IDs refer only to the supplied content, never controller paths):",
            *[
                f"- {item.evidence_id}: {item.modality}"
                + (f"\n{item.content.text}" if isinstance(item.content, TextContent) else "")
                for item in self.items
            ],
            "Image blocks follow in the image-ID order listed above.",
            "Required modalities: " + ", ".join(sorted(self.required_modalities)),
            "Return ONE JSON object, no markdown, verdict line or trailing prose. Shape:",
            '{"requirements":[{"requirement":"a requirement from the request",'
            '"status":"MET|UNMET|UNVERIFIED","modality":"text|image",'
            '"evidence_ids":["supplied ID"],"findings":"what you observed"}]}',
            "Assess each requirement you identify independently. UNMET means observed failure; "
            "UNVERIFIED means insufficient evidence. MET must cite evidence of its modality. "
            "Required modalities must actually be used. Do not infer unseen pixels from text. "
            "Successful inspections return additional Evidence IDs you may cite. "
            "This structured response replaces the role's legacy VERDICT-line format.",
        ]
        return "\n".join(rows)


class RequirementAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requirement: str = Field(min_length=1, pattern=r"\S")
    status: Literal["MET", "UNMET", "UNVERIFIED"]
    modality: Modality
    evidence_ids: list[str]
    findings: str


class CompletionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requirements: list[RequirementAssessment] = Field(min_length=1)


def assess_completion(text: str | None, evidence: VerifierEvidence) -> tuple[VerdictStatus, str]:
    """One decoder for normal and capped answers; no prose phrase matching.

    This validates the requirements the model enumerated, not a claim that the
    model found every requirement in the request. UNMET remains a repair cue in
    the legacy outward findings shape used by completion challenges.
    """

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate assessment key")
            result[key] = value
        return result

    try:
        assessment = CompletionAssessment.model_validate(
            json.loads(text or "", object_pairs_hook=unique_object)
        )
    except (ValidationError, ValueError):
        return "INCONCLUSIVE", "Missing or malformed typed completion assessment."
    # Preserve the existing outward findings budget, with repair cues first so
    # a long MET explanation cannot crowd an observed failure out of it.
    findings = _bounded(
        "\n".join(
            f"{row.status}: {row.requirement}\n{row.findings}"
            for row in sorted(assessment.requirements, key=lambda row: row.status != "UNMET")
        ),
        MAX_FINDINGS_CHARS,
    )
    if any(row.status == "UNMET" for row in assessment.requirements):
        return "FAIL", findings
    registry = {ref: item.modality for ref, item in evidence.registry.items()}
    used: set[str] = set()
    for row in assessment.requirements:
        if (
            row.status != "MET"
            or not row.requirement.strip()
            or not row.evidence_ids
            or any(ref not in registry for ref in row.evidence_ids)
            or not any(registry.get(ref) == row.modality for ref in row.evidence_ids)
        ):
            return "INCONCLUSIVE", findings + "\nRequired evidence was not verified."
        used.add(row.modality)
    if evidence.required_modalities - used:
        return "INCONCLUSIVE", findings + "\nRequired media was not assessed."
    return "PASS", findings
