"""Keyed copy for the two outcome rows ``rows.completion_notice`` words itself.

WHY A SEPARATE MODULE. ``harness.rows`` decides which sentence and which ink an
outcome gets and must stay free of a catalogue dependency (it is imported below
every host; see its CONSTRAINT paragraph and ``harness.bridge_copy``, which
makes the same split for the bridge row). The words for the ``closed`` and
``retired`` arms therefore live here, and ``completion_notice`` imports this
module lazily.

WHY FUNCTIONS, NOT CONSTANTS. These sentences used to be module-level strings
(``CLOSED_NOTICE_TEXT`` / ``RETIRED_NOTICE_TEXT``) that ``tui/notify.py``
imported BY VALUE into the banner vocabulary, so a non-English language got a
localized subtitle over an English body. A function resolves the ACTIVE locale
on every call, so the transcript row, the phone projection frame and the
desktop bridge follow a live language change with no bookkeeping; only the
banner vocabulary still holds copies, and it re-materialises them through
``register_refresh`` (``tui.notify._reload_notice_copy``).

NOTHING PARSES OR STORES THIS TEXT. The persisted attention record carries only
``kind`` / ``reason`` / ``cause`` tokens; the sentence is derived from them each
time it is painted. Clients key off ``kind``, never the words, so no stable-code
migration is needed (the desktop UI keeps its own copy of the sentence).

THE COPY lives in ``wire.outcome_notices`` — a ``wire.*`` namespace because the
same words are painted by the terminal, the phone projection and the desktop
bridge, and the latter two receive them as already-rendered text.
"""

from __future__ import annotations

from local_operator.i18n.keys import wire_outcome_notices as _k
from local_operator.i18n.text import tr

_t = tr("wire.outcome_notices")


def closed_notice_text() -> str:
    """The neutral post-completion closure's sentence, in the active language.

    A RECEIPT, not a failure (v2 operator directive, 2026-09-29): a disposal
    caught a run that had spent no provider round-trip, so it renders as this
    sentence rather than the ``Stopped with an error`` the operator re-reported.
    """
    return _t(_k.closed())


def retired_notice_text() -> str:
    """The update-cut sentence, in the active language.

    TRUTHFUL but DISTINGUISHED FROM A FAILURE (retire-for-build arm, 2026-09-29):
    the turn really was cut, but the update was routine and the operator asked
    for these transitions to stop reading as errors.

    Carries the kept-output clause (design round 2, D1): the one fact a user who
    lost work needs is that what the turn wrote before the cut is still there.
    """
    return _t(_k.retired())
