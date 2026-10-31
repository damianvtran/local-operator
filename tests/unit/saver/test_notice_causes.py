"""One disclosure key, many causes: each cause must get the sentence that fits it.

Independent audit round 5, F-1: `notice.disabled` was given an unconditional
"turn it on in /settings" tail, but the chain emits that key for causes where the
global setting is NOT the blocker — a credential equal to a provider key, a
non-canonical base, a 403/404 route, an un-followed 3xx. In those states the
instruction asked for a change that was already the case, and it contradicted the
key's own brief ("No action is implied"). The settings pointer now lives on
`notice.gate_off`, emitted by the one cause it can fix.

The matrix below is the test: for every cause the chain routes to the `disabled`
class, the sentence carries no settings instruction; for the gate, it does.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.saver import chain as chain_mod
from local_operator.saver import command as saver_command
from local_operator.saver.chain import SaverChain
from local_operator.saver.client import SaverHTTPError, SaverRefused


def _sentences() -> dict[str, str]:
    """The two renderings, read through the real i18n path."""
    from local_operator.i18n.text import tr

    neutral = str(tr("wire.radient_saver")(chain_mod.notice_message(chain_mod.NOTICE_DISABLED)))
    gate = saver_command.refusal_text(saver_command.SaverDecision(source="gate", on=False))
    return {"neutral": neutral, "gate": gate}


def test_the_two_causes_do_not_share_one_sentence() -> None:
    words = _sentences()
    assert words["neutral"] != words["gate"]
    # The gate is the only one that can be fixed by the setting, so it is the
    # only one that names it.
    assert "/settings" not in words["neutral"], words["neutral"]
    assert "/settings" in words["gate"], words["gate"]
    # And the neutral sentence still tells the user what leaves the machine.
    assert "conversation" in words["neutral"].lower(), words["neutral"]


class _RecordingChain:
    """A chain whose only behaviour is recording which reason was noticed."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self._noticed: set[str] = set()
        self._paused_session = False
        self._paused_until = 0.0
        self._refreshed_after_401 = True
        self.needs_availability_refresh = False
        self.client = SimpleNamespace()

        async def notice(_msg: Any) -> None:  # pragma: no cover — not reached here
            return None

        self.notice = notice

    async def _fail_open(self, reason: str) -> None:
        self.seen.append(reason)


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(SaverRefused("credential_conflict"), id="credential-conflict"),
        pytest.param(SaverRefused("noncanonical_base"), id="non-canonical-base"),
        pytest.param(SaverHTTPError(403), id="http-403"),
        pytest.param(SaverHTTPError(404), id="http-404"),
        pytest.param(SaverHTTPError(302), id="http-302-unfollowed"),
    ],
)
def test_every_chain_cause_of_the_disabled_class_gets_the_neutral_sentence(
    exc: BaseException,
) -> None:
    rec = _RecordingChain()
    asyncio.run(SaverChain._on_failure(rec, exc))  # type: ignore[arg-type]
    assert rec.seen == [chain_mod.NOTICE_DISABLED], rec.seen
    from local_operator.i18n.text import tr

    text = str(tr("wire.radient_saver")(chain_mod.notice_message(rec.seen[0])))
    assert "/settings" not in text, text


def test_the_global_gate_is_the_one_cause_that_names_the_setting() -> None:
    decision = saver_command.SaverDecision(source="gate", on=False)
    text = saver_command.refusal_text(decision)
    assert "/settings" in text
    # `auto` is a mode an explicit session choice may be honoured past, so it is
    # NOT the gate: it gets the neutral sentence.
    auto = saver_command.refusal_text(
        saver_command.SaverDecision(source="auto_unavailable", on=False)
    )
    assert "/settings" not in auto, auto
