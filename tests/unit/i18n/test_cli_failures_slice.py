"""Core literals slice 1: argparse failure lines, group summaries, first-run cues.

WHY: ``--help`` was localised but argparse's own failure sentences (unknown
subcommand, bad choice, missing required argument) and ten top-level group
summaries were not, so a vi/hi user met English on the first mistake. The failure
seam is ``LocalizedArgumentParser.error``; these tests pin that (1) English output
is byte-identical to what stock argparse printed, (2) each recognised shape is
routed through its catalogue key (a marker renderer proves the route without
needing a shipped translation), (3) unrecognised shapes pass through untouched.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr

import pytest

from local_operator.i18n import cli_failures
from local_operator.i18n.cli_failures import LocalizedArgumentParser, localise_failure
from local_operator.i18n.messages import Msg


def _stock_and_localised(argv: list[str]) -> tuple[str, str]:
    outputs = []
    for cls in (argparse.ArgumentParser, LocalizedArgumentParser):
        parser = cls(prog="lop")
        parser.add_argument("--hosting", choices=["a", "b"])
        sub = parser.add_subparsers(dest="subcommand")
        child = sub.add_parser("send")
        child.add_argument("target")
        buffer = io.StringIO()
        with redirect_stderr(buffer), pytest.raises(SystemExit) as exit_info:
            parser.parse_args(argv)
        assert exit_info.value.code == 2
        outputs.append(buffer.getvalue())
    return outputs[0], outputs[1]


@pytest.mark.parametrize(
    "argv",
    [["bogus"], ["--hosting", "zzz"], ["send"], ["--nope"]],
    ids=["unknown-subcommand", "invalid-choice", "missing-required", "unrecognized"],
)
def test_english_is_byte_identical_to_stock_argparse(argv: list[str]) -> None:
    stock, localised = _stock_and_localised(argv)
    assert localised == stock


def test_recognised_shapes_route_through_the_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    # Msg parameters can carry strings, numbers and selectors, not only counts;
    # object keeps this recording honest without propagating an untyped dict.
    seen: list[tuple[str, dict[str, object]]] = []

    def marker(message: Msg, locale: str | None = None) -> str:
        seen.append((message.code, dict(message.params)))
        return f"<{message.code}>"

    monkeypatch.setattr(cli_failures, "help_text", marker)
    assert localise_failure("the following arguments are required: name, other") == (
        "<cli.failures.required>"
    )
    assert seen[-1] == ("cli.failures.required", {"names": "name, other"})
    choice = "argument subcommand: invalid choice: 'bogus' (choose from a, b)"
    assert localise_failure(choice) == "<cli.failures.invalid_choice>"
    assert seen[-1] == (
        "cli.failures.invalid_choice",
        {"argument": "subcommand", "value": "'bogus'", "choices": "a, b"},
    )


def test_unrecognised_shapes_pass_through() -> None:
    for text in ("plugin-specific refusal: --x", "expected one argument", "exec takes ONE prompt"):
        assert localise_failure(text) == text


def test_python_314_quoted_choices_are_captured_verbatim() -> None:
    # CPython 3.14 renders the choices as 'a', 'b'; the tail is passed through, not rebuilt.
    text = "argument --hosting: invalid choice: 'z' (choose from 'a', 'b')"
    rendered = localise_failure(text, locale="fr")
    # English byte identity cannot distinguish the keyed path from pass-through.
    # French copy must appear while the technical choices retain their quotes.
    assert "choix invalide" in rendered
    assert "invalid choice" not in rendered
    assert "'z'" in rendered
    assert "'a', 'b'" in rendered


def test_invalid_command_helper_is_english_stable() -> None:
    parser = LocalizedArgumentParser(prog="lop")
    buffer = io.StringIO()
    with redirect_stderr(buffer), pytest.raises(SystemExit):
        parser.fail_invalid_command("hub link", None)
    assert buffer.getvalue().endswith("lop: error: Invalid hub link command: None\n")


def test_first_run_cues_keep_their_english_bytes() -> None:
    from local_operator.tui.app import AIDA_NO_PROVIDER_CUE, SETUP_PLACEHOLDER

    assert SETUP_PLACEHOLDER == "Type /login radient to begin"
    assert AIDA_NO_PROVIDER_CUE == "Connect an AI account first: type /login radient"
    assert len(AIDA_NO_PROVIDER_CUE) <= 74  # context.json maxCells for the splash row
