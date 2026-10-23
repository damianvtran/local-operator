"""The ``/agent promote`` and ``/team promote`` grammar, shared by BOTH hosts (LOH-9 S4).

WHY THIS MODULE SITS AT THE PACKAGE ROOT
========================================

``/agent`` and ``/team`` are ``AUTHORITATIVE_SESSION`` commands
(``session/frontend_state.py::_slash_capabilities``), so in a live session the
terminal forwards the line to the runtime owner (``route_shared_slash``) and the
owner's dispatcher answers it. The app-hosted branch in ``tui/app.py`` runs only
where nothing routes — tests, and viewers from before 0.46. A grammar implemented
in one of those two is silently missing from the other: QA round 1 (Q1) measured
the real owner answering ``no agent named 'promote'`` for a verb the TUI offered.

So the grammar and the receipts live HERE, below both callers: ``serving.py`` can
import this without pulling the TUI (and textual) into an owner process, and the
app adapts the same result to its notice plumbing. That is the
``action_class.class_switch_receipt`` precedent — one implementation, two hosts,
so the two can never word the confirmation differently.

WHY A TYPED ``yes`` RATHER THAN A MODAL
=======================================

The authoring forms (``widgets/agent_form.py``, ``widgets/team_form.py``) pin a
discard question above their fields and take ``y``/``n``. They can: the user is
already inside a full-page editor with a focus owner. ``promote`` has no draft —
it names one existing definition and copies it — and design §7 explicitly rules
out a second picker or a modal mode for it.

So it uses the confirmation the command grammar already owns: ``/project delete
<name>`` rehearses and ``/project delete <name> yes`` commits, ``/delete yes``
does the same for a conversation, and ``/network member rm`` likewise. The
rehearsal is a REAL dry run of the same preflight, so the sentence the user reads
names exactly the definitions the confirmed call will write (and, under
``replace``, exactly the ones it will overwrite).

Grammar::

    /agent promote <name> [as <new name>] [replace] [yes]

``as`` and the flags are order-insensitive after the name; ``yes`` is the only
thing that turns a rehearsal into a write. Anything else is a usage refusal, not
a guess: a mistyped flag must never fall through to a copy under the wrong name.

NOTICE KINDS ARE PART OF THE DESIGN (round 1)
=============================================

The rehearsal's kind is chosen by what it would do: a plain copy is ``note`` (the
"answer to something the user just did" tier), while a plan that overwrites an
existing account definition is ``warning`` — the tier for a state the user must
act on — and its destructive sentence comes FIRST, above the list of copies. A
receipt is ``success`` ("a completed action worth confirming"). All three are
existing tiers with their own ink; nothing here invents a colour or a layout, and
none of them is ``info``/``dim``, whose 3.77:1 light-theme contrast failed the
design review (D3) for text this important.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromoteRequest:
    """One parsed promote command: what to copy, where, and whether to commit."""

    name: str
    as_name: str | None = None
    replace: bool = False
    confirmed: bool = False


def parse_promote(rest: str) -> PromoteRequest | None:
    """Parse the arguments after ``promote``; ``None`` means "show the usage".

    ``None`` rather than an exception: every caller answers a malformed command
    with the same one-line usage, and a refusal object per mistake would be four
    copies of that sentence. A name that STARTS with ``=`` is reached by the
    usual escape (one leading ``=`` is stripped), because ``=`` is not a reserved
    character in a role or team name.
    """

    tokens = rest.split()
    if not tokens:
        return None
    name = tokens[0][1:] if tokens[0].startswith("=") else tokens[0]
    if not name:
        return None
    as_name: str | None = None
    replace = False
    confirmed = False
    index = 1
    while index < len(tokens):
        token = tokens[index].casefold()
        if token == "yes":
            confirmed = True
            index += 1
        elif token == "replace":
            replace = True
            index += 1
        elif token == "as":
            if index + 1 >= len(tokens):
                return None
            as_name = tokens[index + 1]
            if as_name.startswith("="):
                as_name = as_name[1:]
            if not as_name:
                return None
            index += 2
        else:
            # An unrecognised word is a mistyped command, never a second name:
            # promoting under a name the user did not mean is the one mistake
            # this verb cannot take back.
            return None
    return PromoteRequest(name=name, as_name=as_name, replace=replace, confirmed=confirmed)


def _notice(SlashResult: Any, text: str, style: str) -> Any:
    """One notice result in the shape both hosts already render."""

    return SlashResult(kind="notice", text=text, style=style)


def notice_kind(style: str) -> str:
    """Map a ``SlashResult.style`` to a transcript notice tier.

    Both hosts need this (the app's own notice path and the viewer that paints a
    routed result), so it lives here rather than being spelled twice: a tier
    added for a promotion receipt must render the same on either side. Anything
    unrecognised stays ``info``, the pre-existing default.
    """

    return style if style in ("warning", "error", "note", "success") else "info"


def promote_receipt(
    session: Any, kind: str, rest: str, SlashResult: Any, *, agents: Any = None
) -> Any:
    """Answer one ``/agent promote …`` or ``/team promote …`` as a ``SlashResult``.

    ``agents`` is the conversation's agent composite for a TEAM promotion; the
    app and the owner dispatcher both pass it explicitly, and it is required
    there because a team's conversation-scoped members must be resolved and
    copied in the same call. For an agent promotion the registry IS the agent
    composite, so the parameter is ignored.

    Every failure the composite raises is a sentence meant to be read verbatim
    (``PromotionIncomplete``, the preflight ``ValueError``s); a registry I/O
    failure is reported the same way rather than crashing the caller mid-turn.
    """

    from local_operator.i18n.keys import tui_authoring as keys
    from local_operator.i18n.text import tr
    from local_operator.profile_scope import promote_missing

    translate = tr("tui.authoring")
    request = parse_promote(rest)
    if request is None:
        return _notice(
            SlashResult,
            translate(keys.promote_usage(command=f"/{kind} promote")),
            "warning",
        )
    if kind == "agent":
        registry = getattr(session, "agent_registry", None)
        target_agents = registry
    else:
        registry = getattr(session, "team_registry", None)
        target_agents = agents if agents is not None else getattr(session, "agent_registry", None)
    if registry is None or not hasattr(registry, "promote"):
        return _notice(
            SlashResult,
            translate(keys.promote_failed(reason=translate(keys.registry_missing()))),
            "warning",
        )
    arguments: dict[str, Any] = {
        "as_name": request.as_name,
        "replace": request.replace,
        "authorized_by": "surface:tui",
    }
    if kind == "team":
        arguments["agents"] = target_agents
    try:
        if not request.confirmed:
            plan = registry.promote(request.name, dry_run=True, **arguments)
            return _rehearsal(plan, translate, keys, SlashResult)
        result = registry.promote(request.name, **arguments)
    except KeyError:
        noun = "agent" if kind == "agent" else "team"
        return _notice(
            SlashResult,
            translate(keys.promote_failed(reason=promote_missing(noun, request.name))),
            "warning",
        )
    except Exception as error:  # noqa: BLE001 — every failure here is a refusal
        return _notice(
            SlashResult,
            translate(keys.promote_failed(reason=str(error))),
            "warning",
        )
    return _receipt(result, translate, keys, SlashResult)


def _rehearsal(plan: Any, translate: Any, keys: Any, SlashResult: Any) -> Any:
    """The dry run's sentence(s): the OVERWRITE first, then what is new.

    Order is the finding (design round 1, D1): a mixed plan that listed the new
    copies first read as an ordinary creation, with the destructive half as an
    afterthought. The overwrite leads, names the account rows it would replace,
    and raises the notice to ``warning`` so the frame's own ink says the same
    thing the sentence does.
    """

    created = [row["name"] for row in plan.created]
    replaced = [str(name) for name in plan.replaced]
    if replaced and created:
        text = translate(
            keys.promote_mixed_plan(replaced=", ".join(replaced), created=", ".join(created))
        )
    elif replaced:
        text = translate(keys.promote_replace_plan(names=", ".join(replaced)))
    else:
        text = translate(keys.promote_plan(names=", ".join(created)))
    return _notice(SlashResult, text, "warning" if replaced else "note")


def _receipt(result: Any, translate: Any, keys: Any, SlashResult: Any) -> Any:
    """The committed call's receipt: created and replaced named SEPARATELY.

    One ``copied a, b`` list made a replacement indistinguishable from a copy
    (design round 1, D2; UX round 1, U1) — a reader could not tell whether an
    existing account definition had been overwritten or a new one added. The two
    outcomes are now separate sentences, and the conversation-copy clause stays
    as the closing fact it always was.
    """

    created = [row["name"] for row in result.created]
    replaced = [str(name) for name in result.replaced]
    sentences: list[str] = []
    if created:
        sentences.append(translate(keys.promote_receipt_created(names=", ".join(created))))
    if replaced:
        sentences.append(translate(keys.promote_receipt_replaced(names=", ".join(replaced))))
    sentences.append(translate(keys.promote_receipt_keeps()))
    return _notice(SlashResult, " ".join(sentences), "success")
