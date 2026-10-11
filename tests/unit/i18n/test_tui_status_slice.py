"""T2 (``tui.status``): the status line's en copy is byte-identical to extraction time.

The fixture next to this file is a SNAPSHOT of the status band's copy as it
was before the extraction (``origin/main`` @ the slice's base): the two band
activity captions, the model row's two state words (moved here from T1's
deferral — D1 requires the splash and the band answer with the same word), the
parked-tunnel alarm line, and the two segment counters as the ICU plurals that
replace their hand-rolled singular/plural branches.

The tests below re-derive the live values and require every rendered string to
match the golden EXACTLY — the committed form of the slice's byte-identical
proof, re-runnable by review and QA, and the trip-wire for any later edit that
changes band copy without regenerating the catalogue. ``PR``-time capture
before/after (constants, formatter outputs, 15 rendered geometries) is recorded
in the PR body; this file is its permanent, in-suite half.

The last tests pin what a string comparison cannot: the segment-disappears-at-
zero contract (the plurals are never rendered for a zero count), the band
painting the golden words through the REAL render path, and the degradation
contract of the shared resolver.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import tr
from local_operator.tui.widgets import status_line as sl
from local_operator.tui.widgets import welcome

FIXTURE = Path(__file__).parent / "fixtures" / "tui_status_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_catalogue_matches_the_pre_slice_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.status") == GOLDEN


def test_every_live_constant_is_the_golden_rendering() -> None:
    assert sl.TUNNEL_PARKED_TEXT == _en("tui.status.tunnel.parked")
    assert welcome.MODEL_PENDING == _en("tui.status.model.connecting")
    assert welcome.MODEL_SETUP == _en("tui.status.model.setup")


def test_the_counters_are_the_golden_plurals() -> None:
    # Plural selection, not a hand-rolled branch: en renders '1 agent' / '2
    # agents' byte for byte, and a wave changes the CATEGORIES where the
    # language's rules say so rather than an if/else in this module.
    for count in (1, 2, 5, 11):
        assert sl.format_agents(count) == _en("tui.status.agents", count=count)
        assert sl.format_jobs(count) == _en("tui.status.jobs", count=count)


def test_zero_counts_still_disappear_without_rendering_the_plural() -> None:
    # The segment-disappears-at-zero contract is the CALLER's (format_agents /
    # format_jobs), unchanged by the extraction: '0 agents' must not become
    # renderable just because the string moved.
    assert sl.format_agents(0) == ""
    assert sl.format_jobs(0) == ""
    # The other branch renders if it were ever asked (it is not): documenting
    # that zero is a REAL rendering, which is why the caller keeps the guard.
    assert _en("tui.status.agents", count=0) == "0 agents"


def test_the_tunnel_alarm_is_the_golden_line() -> None:
    assert sl.format_tunnel(True) == _en("tui.status.tunnel.parked")
    assert sl.format_tunnel(False) == ""


def test_the_band_paints_the_golden_words_through_the_real_render_path() -> None:
    """The extraction's whole point: what a user READS is the catalogue copy.

    Drives ``StatusLine`` through its public render entry with the same
    geometry harness the band's own suite uses, so a wiring mistake (a stale
    literal left in ``_left_text``) fails here even if the constants are right.
    """
    from tests.unit.tui.test_status_line import FakeClock, _dock

    starting = sl.StatusLine(_dock(200), clock=FakeClock())
    starting.update(model_label="radiant/gpt-5")
    starting._starting = True
    assert _en("tui.status.band.starting") in starting.render_text(200).plain

    import local_operator.tui.shimmer as shimmer_mod

    streaming = sl.StatusLine(_dock(200), clock=FakeClock())
    streaming.update(model_label="radiant/gpt-5")
    streaming._streaming = True
    with patch.object(shimmer_mod, "shimmer_enabled", lambda: False):
        plain = streaming.render_text(200).plain
    assert _en("tui.status.band.working") in plain

    # The setup word reaches the band as the MODEL LABEL the app passes (the
    # shared constant), not as band-internal copy: paint it and read it back.
    setup = sl.StatusLine(_dock(200), clock=FakeClock())
    setup.update(model_label=welcome.MODEL_SETUP)
    assert _en("tui.status.model.setup") in setup.render_text(200).plain


def test_unknown_messages_degrade_to_their_code() -> None:
    resolve = tr("tui.status")
    assert resolve(Msg("tui.status.nope", {})) == "tui.status.nope"
