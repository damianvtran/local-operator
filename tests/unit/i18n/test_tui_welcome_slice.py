"""T1 (`tui.welcome`): the splash's en copy is byte-identical to extraction time.

The fixture next to this file is a SNAPSHOT of the welcome/splash copy as it
was before the extraction (`origin/main` @ `8b6ed6b854`): the tip pool (the
twelve plain entries, ``TIP_PASTE``, ``TIP_MESH`` and the three keyed
templates), the setup opening tip, the hint descriptions, the three setup
checklist tiers, and the two status-row sentences. The tests below re-derive
the live values and require every rendered string to match the golden
EXACTLY — the committed form of the slice's byte-identical proof, re-runnable
by review and QA, and the trip-wire for any later edit that changes splash
copy without regenerating the catalogue.

The last two tests pin the parts a string comparison cannot: the width
budgets (a 59-cell painted tip row; keyed templates at 32 cells excluding
``{key}``, whose placeholder must survive for the per-render substitution)
and the degradation contract of the shared resolver.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.cells import cell_len

from local_operator import keymap
from local_operator.i18n import catalogues, runtime
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import tr
from local_operator.providers.login_catalog import RECOMMENDED_LOGIN
from local_operator.tui.widgets import welcome

FIXTURE = Path(__file__).parent / "fixtures" / "tui_welcome_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))

_STATIC_TIP_KEYS = (
    "tui.welcome.tip.resume",
    "tui.welcome.tip.team",
    "tui.welcome.tip.model",
    "tui.welcome.tip.paste",
    "tui.welcome.tip.approvals",
    "tui.welcome.tip.esc",
    "tui.welcome.tip.subagents",
    "tui.welcome.tip.shell",
    "tui.welcome.tip.fork",
    "tui.welcome.tip.fork_placement",
    "tui.welcome.tip.cmux",
    "tui.welcome.tip.classification_off",
    "tui.welcome.tip.classification_login",
    "tui.welcome.tip.mesh",
    "tui.welcome.tip.key_new_session",
    "tui.welcome.tip.key_resume",
    "tui.welcome.tip.hotkeys",
)


def _en(key: str, **params: str) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_catalogue_matches_the_pre_slice_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.welcome") == GOLDEN


def test_every_live_constant_is_the_golden_rendering() -> None:
    # TIPS is the ring: order is the contract (TIP_MIN_WIDTH and the opening
    # frame both index into it), so the whole tuple is compared in order.
    assert welcome.TIPS == tuple(_en(key) for key in _STATIC_TIP_KEYS)
    assert welcome.TIP_PASTE == _en("tui.welcome.tip.paste")
    assert welcome.TIP_MESH == _en("tui.welcome.tip.mesh")
    assert welcome.TIP_SETUP == _en("tui.welcome.tip.setup", login=RECOMMENDED_LOGIN.title())
    assert welcome.KEYED_TIPS == (
        (_en("tui.welcome.tip.key_new_session"), "keymap.new_session"),
        (_en("tui.welcome.tip.key_resume"), "keymap.resume"),
        (_en("tui.welcome.tip.hotkeys"), None),
    )
    assert welcome.HINTS == (
        ("/", _en("tui.welcome.hint.command_picker")),
        ("/help", _en("tui.welcome.hint.all_commands")),
        ("ctrl/cmd+d", _en("tui.welcome.hint.quit")),
    )
    assert welcome.HINTS_SETUP == (
        (f"/login {RECOMMENDED_LOGIN}", _en("tui.welcome.hint.sign_in")),
        ("/login", _en("tui.welcome.hint.other_provider")),
        ("/help", _en("tui.welcome.hint.all_commands")),
    )
    assert welcome.WARNING_SHORT == _en("tui.welcome.status.login_required.short")


def test_the_keymap_tips_are_the_golden_templates() -> None:
    assert keymap.BY_ID["keymap.new_session"].tip == _en("tui.welcome.tip.key_new_session")
    assert keymap.BY_ID["keymap.resume"].tip == _en("tui.welcome.tip.key_resume")
    # The placeholder must SURVIVE the catalogue round trip: `_resolve_tip`
    # substitutes the persisted key per render, which is why the catalogue
    # stores `{key}` ICU-quoted.
    assert keymap.BY_ID["keymap.new_session"].tip.format(key="ctrl+g") == (
        "ctrl+g starts a new conversation"
    )


def _step(width: int) -> str:
    line = welcome.setup_step_line(width, "Aida")
    assert line is not None
    return line.plain


def test_the_setup_steps_pick_the_golden_tier_for_their_width() -> None:
    # The tiers are tried longest-first: 80 selects the full form, 65 the
    # middle one, 55 the short one, and under 45 the line is withdrawn.
    assert _step(80) == _en("tui.welcome.step.full", aida="Aida")
    assert _step(65) == _en("tui.welcome.step.medium", aida="Aida")
    assert _step(55) == _en("tui.welcome.step.short", aida="Aida")
    assert welcome.setup_step_line(40, "Aida") is None


def test_status_rows_compose_the_golden_sentences() -> None:
    info = welcome.WelcomeInfo(
        version="0.69.0",
        model_label="radiant/gpt-5",
        model_name="gpt-5",
        update_available="0.70.0",
        missing_credential="anthropic",
    )
    painted = "\n".join(text.plain for _, text in welcome._status_rows(info, 80))
    assert _en("tui.welcome.status.update", version="0.70.0") in painted
    assert _en("tui.welcome.status.login_required", provider="anthropic") in painted


def test_the_width_budgets_and_the_sidecar_are_intact() -> None:
    painted = [
        cell_len(f"{welcome.TIP_GLYPH} {tip}")
        for tip in (*welcome.TIPS, welcome.TIP_SETUP, welcome.TIP_PASTE)
    ]
    assert max(painted) == welcome.TIP_MIN_WIDTH == 59
    for template, binding_id in welcome.KEYED_TIPS:
        if binding_id is not None:
            assert cell_len(template.replace("{key}", "")) <= 32, template

    # The en-only context sidecar names the budget for every tip-family key,
    # so a translation wave has the numbers rather than inferring them.
    sidecar = catalogues.context("en", "tui.welcome")
    tip_keys = {key for key in GOLDEN if key.startswith("tui.welcome.tip.")}
    assert tip_keys <= set(sidecar)
    keyed = {"tui.welcome.tip.key_new_session", "tui.welcome.tip.key_resume"}
    assert {key: sidecar[key].get("maxCells") for key in tip_keys} == {
        key: 32 if key in keyed else 57 for key in tip_keys
    }


def test_unknown_messages_degrade_to_their_code() -> None:
    resolve = tr("tui.welcome")
    assert resolve(Msg("tui.welcome.nope", {})) == "tui.welcome.nope"
