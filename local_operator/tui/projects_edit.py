"""The projects page's EDIT model (LOH-19 P2): what one save would actually send.

Why this exists, and why it is separate from the form: the store's
``ProjectRegistry.update_project`` is per-field last-writer-wins — it touches
only the fields present in ``ProjectEdit.model_fields_set`` — so a form that
always sends every field, prefilled or not, would turn one reader's title
change into a whole-row clobber of a concurrent session's owner change. The
conversion from "what the form shows" to "what the reader changed" is the
contract this module owns, and it is pure so the rules can be asserted without
a message pump.

The state is TEXT because the form's controls are text: comparing typed values
would send ``estimate="5.0"`` as a change against a stored ``5``. Reading a
project and comparing two states both normalise the same way, so a trailing
newline or a number's spelling cannot invent a write.

The one clear the store cannot express is an estimate: ``ProjectEdit.estimate
= None`` means UNTOUCHED (the store reads ``model_fields_set``, not the value),
so there is deliberately no clear spelling. The builder refuses it here, with
a sentence the form paints under the field, rather than returning a no-op that
would silently keep the old value while the reader believed it was gone.

Refusals are NOT raised: a refusal is a value the form paints under a field,
and an exception would have to be re-narrowed at the call site into the same
shape. The field name is the form block's own key (``name`` is the form's
``key`` block; ``_blocks`` maps both), so the caller never translates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from local_operator.i18n.keys import tui_projects_edit as _x
from local_operator.i18n.text import tr
from local_operator.projects import ESTIMATE_MAX, ProjectDoneGateError, ProjectEdit

_t = tr("tui.projects_edit")

#: Pydantic wraps a validator's ``ValueError`` as ``"Value error, <sentence>"``.
#: Same unwrap the create form does, so an edit refusal reads like a create one.
_VALUE_ERROR_PREFIX = "Value error, "


@dataclass(frozen=True)
class ProjectEditState:
    """One project's editable fields, as the form's controls spell them.

    ``tags`` is a tuple so the state is hashable and comparable without the
    list-mutation aliasing a dataclass field of ``list`` would invite.
    """

    title: str = ""
    name: str = ""
    description: str = ""
    status: str = "active"
    tags: tuple[str, ...] = ()
    team: str = ""
    owner: str = ""
    start_date: str = ""
    target_date: str = ""
    estimate: str = ""
    estimate_unit: str = "points"

    @classmethod
    def from_project(cls, project: Mapping[str, Any]) -> ProjectEditState:
        """Read one composed project payload into the form's vocabulary.

        Nulls become empty text (a cleared field and an absent one are the same
        thing to a reader looking at the form), and the estimate is spelled the
        way the form's input shows it. The STATUS is kept verbatim on purpose:
        a row written by a newer build may carry a word this build does not
        know, and normalising it into this build's list would rewrite the row
        on an unrelated edit (the store's own forward-compat rule).
        """

        def text(key: str) -> str:
            value = project.get(key)
            return "" if value is None else str(value)

        tags = project.get("tags")
        estimate = project.get("estimate")
        unit = project.get("estimate_unit")
        return cls(
            title=text("title"),
            name=text("name"),
            description=text("description"),
            status=text("status") or "active",
            tags=tuple(str(tag) for tag in tags) if isinstance(tags, (list, tuple)) else (),
            team=text("team"),
            owner=text("owner"),
            start_date=text("start_date"),
            target_date=text("target_date"),
            estimate="" if estimate is None else f"{float(estimate):g}",
            estimate_unit=str(unit) if unit in ("points", "days") else "points",
        )


@dataclass(frozen=True)
class EditPlan:
    """What one form save does: one edit, a no-op, or a refusal under a field."""

    edit: ProjectEdit | None = None
    field: str = ""
    refusal: str = ""
    no_change: bool = False


def _refusal(field: str, sentence: str) -> EditPlan:
    return EditPlan(field=field, refusal=sentence)


def _sentence(exc: Exception) -> str:
    text = str(exc)
    return text[len(_VALUE_ERROR_PREFIX) :] if text.startswith(_VALUE_ERROR_PREFIX) else text


def _clean(value: str) -> str:
    """The store's own normalisation for a one-line text field (it strips)."""
    return value.strip()


def _parse_estimate(text: str) -> tuple[float | None, bool]:
    """``(number, ok)`` — ``None`` for empty, ``ok`` False for non-numeric text.

    Parsed rather than string-compared for the comparison's sake: the field
    shows ``5`` for a stored ``5.0``, and a reader retyping ``5.0`` has not
    changed anything the store would record.
    """
    stripped = text.strip()
    if not stripped:
        return None, True
    try:
        return float(stripped), True
    except ValueError:
        return None, False


def build_edit_plan(initial: ProjectEditState, final: ProjectEditState) -> EditPlan:
    """The changed-only edit, or the reason there is nothing to send.

    Comparison is NORMALISED, not byte-equal: the store strips text fields and
    keeps a list's order, so ``" body "`` is not a change but a reordered tag
    list is. A clear is spelled the way ``ProjectEdit``/``update_project`` read
    it: ``None`` for title/owner/team, ``""`` for a date or a description,
    ``[]`` for tags.
    """
    title, name = _clean(final.title), _clean(final.name)
    description = final.description.strip()
    team, owner = _clean(final.team), _clean(final.owner)
    start, target = _clean(final.start_date), _clean(final.target_date)

    changed: dict[str, Any] = {}
    if title != _clean(initial.title):
        changed["title"] = title or None
    if name != _clean(initial.name):
        if not name:
            # The store's ProjectEdit would answer with its grammar sentence;
            # the form's own sentence is the actionable one (the create form's
            # rule: name the value, not the field's regex).
            return _refusal("name", _t(_x.refusal_name_required()))
        changed["name"] = name
    if description != initial.description.strip():
        changed["description"] = description
    if final.status != initial.status:
        changed["status"] = final.status
    if final.tags != initial.tags:
        changed["tags"] = list(final.tags)
    if team != _clean(initial.team):
        changed["team"] = team or None
    if owner != _clean(initial.owner):
        changed["owner"] = owner or None
    if start != _clean(initial.start_date):
        changed["start_date"] = start
    if target != _clean(initial.target_date):
        changed["target_date"] = target
    initial_estimate, _ = _parse_estimate(initial.estimate)
    final_estimate, estimate_ok = _parse_estimate(final.estimate)
    if not estimate_ok:
        return _refusal("estimate", _t(_x.refusal_estimate_number()))
    if final_estimate != initial_estimate:
        if final_estimate is None:
            # Deliberate, and stated in the module docstring: the store has no
            # clear state for an estimate, so the save stops here rather than
            # silently keeping the old number.
            return _refusal("estimate", _t(_x.refusal_estimate_clear()))
        if not (0.0 < final_estimate <= ESTIMATE_MAX):
            return _refusal(
                "estimate", _t(_x.refusal_estimate_range(maximum=f"{ESTIMATE_MAX:g}"))
            )
        changed["estimate"] = final_estimate
    if final.estimate_unit != initial.estimate_unit:
        changed["estimate_unit"] = final.estimate_unit

    if not changed:
        return EditPlan(no_change=True)
    try:
        edit = ProjectEdit(**changed)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = str(first["loc"][0]) if first.get("loc") else "key"
        return _refusal(field, _sentence(ValueError(first["msg"])))
    # FORMAT before ORDER (agent review round 1, R2): the store's date
    # validators must speak first, or a malformed start date is reported as an
    # inverted range under the untouched target field — the reader is sent to
    # repair a value they never typed. Only a validated pair reaches the
    # ordering rule, which is the same test the store's Project validator runs
    # on the MERGED row, one step earlier so the refusal lands on a field.
    if start and target and target < start:
        return _refusal("target_date", _t(_x.refusal_date_order()))
    return EditPlan(edit=edit)


def refusal_field(exc: Exception) -> str:
    """Which form block an APP-side store refusal belongs under.

    The write happens in the app (the page never touches the store), so the
    field the refusal is about has to be recovered from the exception. The
    done gate is the one rule whose sentence spans the MILESTONES while the
    field the reader can act on is ``status`` — a store ``ValidationError``
    from the merged row names its own field; everything else (a name conflict,
    the schema guard, an i/o failure) belongs to the handle.
    """
    if isinstance(exc, ProjectDoneGateError):
        return "status"
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors and errors[0].get("loc"):
            return str(errors[0]["loc"][0])
        # A model-level validator (the date range) reports an empty loc; the
        # only such rule on Project that an edit can break is the range.
        return "target_date" if "target_date" in str(exc) else "key"
    return "key"
