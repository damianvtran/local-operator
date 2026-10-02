"""The shared display-LABEL rule for key/label pairs (teams and agents).

A NAME is a key: a lowercase slug with no spaces (``ux-reviewer``,
``data-quality``). The name is what every resolver, collision check and
addressing path reads -- it is typed as a slash-command argument
(``/agent ux-reviewer``), passed to ``task(agent=...)`` and ``--profile``, and
listed on team rosters. A LABEL is the human spelling a surface paints
(``UX Reviewer``); it is DISPLAY-ONLY -- nothing resolves by it, no collision
rule sees it, and it never rides the hub publish wire.

ONE place owns the rule so the TUI listing, the picker, the settings pane, the
status band, the org chart, the CLI, the agent tool and the routed runtime
cannot disagree about how a name reads. This module is that place for the
AGENT side, and it also hosts the primitives the TEAM side reuses
(:func:`default_label`, :func:`normalize_label`, :func:`validate_label`,
:func:`bounded_form`): a second copy of the derivation is how the two domains
would later spell the same slug differently.

Why teams keep their own :func:`~local_operator.teams.display_form`: teams
shipped an EXACT comparison against the derived default, frozen by its own
tests and remediation rounds. The agent side refines that ONE comparison to a
casefold match (see :func:`display_form`), so the two functions are separate
on purpose -- the shared half is everything that is genuinely identical
(deriving, normalizing, validating, bounding), and each domain's composition
lives in ONE function whose cases its tests freeze.
"""

from __future__ import annotations

import re
import unicodedata

#: Tokens whose display form is upper-case rather than Title Case (D3, shared
#: with teams): six common initialisms whose Title-Cased spelling reads as a
#: bug (``qa-tester`` -> ``QA Tester``, ``ux-reviewer`` -> ``UX Reviewer``).
#: Case-insensitive; deliberately small -- every entry is a bet on how a token
#: reads, and Title Case stays the rule.
DEFAULT_LABEL_INITIALISMS = frozenset({"qa", "ai", "api", "tui", "ux", "ui"})

#: Cap on a stored label, shared by teams and agents: it is free text shown in
#: every listing, so the bound keeps one runaway paste from silencing the rows
#: it rides in; both derived defaults are at most the name cap, so every shape
#: fits the same surfaces.
LABEL_MAX_CHARS = 80

#: Cap on a LISTING row's composed display form (see :func:`bounded_form`).
#: 48 cells leaves room for the row's indentation and border at 80x24, the
#: narrow terminal the wrap was measured in -- the same geometry the team
#: listing sizes against.
AGENT_LISTING_CAP = 48


def default_label(name: str) -> str:
    """The display label a row with no stored one derives from its name.

    Names are keys (``ux-reviewer``), so the derived default is the
    human-readable form of the same tokens: split on the separators the name
    rules allow, uppercase each token's first character and keep the rest
    (``data-quality`` -> ``Data Quality``; a token that starts with a digit or
    symbol passes through). Six common INITIALISMS stay upper-cased instead
    (:data:`DEFAULT_LABEL_INITIALISMS`) -- ``qa-tester`` reading ``Qa Tester``
    is the defect the allowlist closes; it stays deliberately small because
    every entry is a bet on how a token reads, and Title Case remains the rule.
    The value is persisted on the next write, so a legacy row's rendering is
    stable once it has been written.
    """
    tokens = [token for token in re.split(r"[._-]+", name) if token]
    return " ".join(render_label_token(token) for token in tokens)


def render_label_token(token: str) -> str:
    """One name token as its display form: initialism upper-case, else Title Case."""
    if token.casefold() in DEFAULT_LABEL_INITIALISMS:
        return token.upper()
    return token[0].upper() + token[1:]


def display_form(name: str, label: str) -> str:
    """The display form EVERY agent render site paints for ``(name, label)``.

    ONE shared rule, so the TUI listing, the picker, the settings pane, the
    status band, the org chart, the CLI, the agent tool and the routed runtime
    cannot disagree. The four arms, frozen:

    * no label -> the raw name (a legacy row before its first write);
    * the label casefolds to the NAME -> the raw name (``lopdev`` painted over
      a stored ``Lopdev``; ``ops`` over ``OPS``: a difference only of case is
      noise either way);
    * the label casefolds to the DERIVED default (:func:`default_label`) ->
      the label alone (``ux-reviewer`` / ``UX Reviewer``: the canonical
      spelling the seed ships, and the space-vs-separator is information the
      key cannot carry). This is the ONE refinement over the team rule, which
      compares EXACTLY and would treat a case variant like ``Ux Reviewer`` as
      a chosen label and parenthesise the key onto it;
    * any other (chosen) label -> ``label (name)``, because the key is the only
      string that ADDRESSES the agent.
    """
    if not label:
        return name
    if label.casefold() == name.casefold():
        return name
    if label.casefold() == default_label(name).casefold():
        return label
    return f"{label} ({name})"


def enrichment_label(name: str, label: str) -> str:
    """The label a DESCRIPTION column may prefix, or "" when it adds nothing.

    A description slot sits beside a name column that already paints the key,
    so a derived default restating it (``lopdev`` -> ``Lopdev · ``) or a label
    that casefolds to the name is noise: only a CHOSEN-and-different label
    earns the prefix, matching :func:`display_form`'s comparisons arm for arm.
    """
    if not label or label.casefold() == name.casefold():
        return ""
    if label.casefold() == default_label(name).casefold():
        return ""
    return label


def bounded_form(form: str, name: str, *, cap: int) -> str:
    """Bound a LISTING row's composed display form: the key never wraps away (N1).

    An over-cap custom label wraps the row and pushes ``(name)`` onto a second
    line, so the reader loses the string that addresses the row (measured for
    teams: an 80-character label at 80x24). The composed form is truncated on
    the LABEL side and the ``…`` is re-appended before the keyed tail, so the
    one string that ADDRESSES the row keeps its place on the line. A form with
    no keyed tail is returned as-is (the label family already bounds its own
    inputs, and ellipsizing a bare name is not this rule's business).

    When the key alone leaves no room under the cap (a name past ~44 cells),
    the WHOLE form is returned as-is and the row wraps: there is no label cell
    left to ellipsize, and slicing ``form[:room]`` with a negative room cut
    INTO the keyed tail before re-appending it, duplicating the key on the
    very rows that need it most (R2-1). A wrapped row is the honest bound
    here; a maimed or repeated key is not.
    """
    keyed = f" ({name})"
    if len(form) <= cap or not form.endswith(keyed):
        return form
    room = cap - len(keyed) - 1
    if room >= 1:
        truncated = form[:room].rstrip()
        if truncated:
            return f"{truncated}…{keyed}"
    return form


def bounded_display_form(name: str, label: str, *, cap: int = AGENT_LISTING_CAP) -> str:
    """``display_form`` bounded for a LISTING row (see :func:`bounded_form`).

    Deliberately a listing-only concern: the other sites bound themselves (the
    band truncates, chart boxes clamp) and both sides of the listing family --
    the local block and the wire's first slot -- must agree byte for byte.
    """
    return bounded_form(display_form(name, label), name, cap=cap)


def normalize_label(value: str) -> str:
    """The stored shape of a label: ends trimmed, runs of whitespace to one space.

    The same normalization agent-server is moving published names to
    (``dev-name-spaces``), so a pulled name round-trips as a label unchanged.
    """
    return " ".join((value or "").split())


def validate_label(
    value: str, *, max_chars: int = LABEL_MAX_CHARS, subject: str = "an agent label"
) -> str:
    """Normalize a label and enforce the storage rules, or raise ``ValueError``.

    Whitespace is collapsed FIRST: a control character that is also whitespace
    (a tab, a form feed) folds into the spacing, while the ones that would
    paint invisibly or not at all (a NUL, a bidi override, a zero-width
    joiner) survive normalization and are refused. The check is the whole
    ``C*`` family -- every category whose characters render as nothing or as a
    control -- not just Cc, because a label is a display string and a Cf
    character is exactly the invisible difference the listings must not
    carry. Empty is legal and means "no custom label".

    ``subject`` is the noun phrase the refusal speaks in ("a team label",
    "an agent label"), so each domain keeps its own wording while the rule
    itself exists once.
    """
    label = normalize_label(value)
    if len(label) > max_chars:
        raise ValueError(
            f"{subject} must be at most {max_chars} characters; "
            f"this one is {len(label)} after whitespace was collapsed"
        )
    for character in label:
        if unicodedata.category(character).startswith("C"):
            raise ValueError(
                f"{subject} cannot contain control characters "
                f"({character!r} is Unicode category {unicodedata.category(character)})"
            )
    return label
