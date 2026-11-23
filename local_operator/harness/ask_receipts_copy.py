"""Keyed copy for the ask receipts ``rows.ask_response_parts`` composes.

WHY A SEPARATE MODULE. ``harness.rows`` decides which state an ask settled in
and which ink it gets, and must stay free of a catalogue dependency (it is
imported below every host; see its CONSTRAINT paragraph). ``harness.notice_copy``
makes the same split for the two outcome rows, and ``harness.bridge_copy`` for
the bridge row. The words therefore live here, and the parts functions import
this module lazily.

WHY FUNCTIONS, NOT CONSTANTS. A function resolves the ACTIVE locale on every
call, so the terminal row, the phone projection frame and the desktop bridge
follow a live language change with no bookkeeping — the same reason
``harness.notice_copy`` holds functions.

WHY ``wire.*``. The same sentence is painted by the terminal, the phone
projection and the desktop bridge, and the last two receive it as
already-rendered text; that is the namespace this repository uses for copy that
travels as text rather than as a code a client re-renders.

NOTHING PARSES OR STORES THIS TEXT. Clients key off ``kind`` / ``status`` /
``ask_id`` from the row's details; the sentence is derived from those on every
paint, so rewording it needs no migration.

THE SENTENCE'S ID IS NOT HERE. ``rows._ask_receipt_line`` owns the
``<id> · "<name>" — <state>`` frame, because the id, the quote marks and the
separators are structural: the catalogue carries only the state phrases, and a
surface that lays the pieces out in separate columns still reads them from one
author.
"""

from __future__ import annotations

from local_operator.i18n.keys import wire_ask_receipts as _k
from local_operator.i18n.text import tr

_t = tr("wire.ask_receipts")


def answered_text() -> str:
    """The state phrase for an answer that landed in time."""
    return _t(_k.response_answered())


def declined_text() -> str:
    """The state phrase for a decline — the agent is told, and nothing is owed."""
    return _t(_k.response_declined())


def late_text() -> str:
    """The state phrase for an answer that arrived after the deadline."""
    return _t(_k.response_late())


def dismissed_text(*, surface: str, reason: str) -> str:
    """The state phrase for an operator dismissal.

    ``surface`` and ``reason`` are STABLE CODES, never sentence fragments: the
    catalogue selects the surface's word (``remote`` is the phone, ``tui`` the
    terminal) and the reason's clause (``answered_in_chat`` / ``moot``, or the
    ``other`` arm for an absent or unknown code). Passing either code through
    as text would put an English word inside a translated sentence.
    """
    return _t(_k.response_dismissed(surface=surface, reason=reason))


def timed_out_text(*, waited: str) -> str:
    """The state phrase for a lapsed ask, naming the wait that actually happened."""
    return _t(_k.timeout_timed_out(waited=waited))


def secret_ask_name() -> str:
    """What a receipt calls a SECRET ask — a label, never its question or key."""
    return _t(_k.name_secret_question())


def withdrawn_text() -> str:
    """The state phrase for an ask the ASKER retracted (``ask_withdraw``).

    "By the agent" is load-bearing: the same row family says who acted
    elsewhere (a dismissal says "by you"), and a reader who cannot tell the two
    apart cannot tell a question they ignored from one the agent took back.
    """
    return _t(_k.withdrawn_by_agent())
