"""The ``/agent promote`` and ``/team promote`` grammar (LOH-9 S4).

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
"""

from __future__ import annotations

from dataclasses import dataclass


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
