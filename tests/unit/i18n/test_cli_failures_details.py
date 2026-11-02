"""Slice 2: preserve argparse semantics while localizing its remaining failures.

The same parser recipe runs through stock argparse and our subclass. This is a
negative control for matching: English equality alone cannot detect a message
that accidentally passed through, so the marker-renderer test separately proves
which key and exact technical tokens reached the localization boundary.
"""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr

import pytest

from local_operator.i18n import cli_failures
from local_operator.i18n.cli_failures import LocalizedArgumentParser, localise_failure
from local_operator.i18n.messages import Msg


def _parser(cls: type[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = cls(prog="lop", allow_abbrev=True)
    parser.add_argument("--integer", type=int)
    parser.add_argument("--float", type=float)
    parser.add_argument("--pair", nargs=2)
    parser.add_argument("--many", nargs="+")
    parser.add_argument("--title")
    parser.add_argument("--timeout")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name")
    group.add_argument("--id")
    return parser


CASES = [
    ["--name", "n", "--unknown"],
    ["--name"],
    ["--name", "n", "--integer", "oops"],
    ["--name", "n", "--float", "oops"],
    ["--ti", "oops"],
    [],
    ["--name", "n", "--id", "i"],
    ["--help=oops"],
    ["--pair", "one"],
    ["--many"],
]


@pytest.mark.parametrize("argv", CASES)
def test_failure_matches_stock_argparse_bytes_and_exit(argv: list[str]) -> None:
    outputs = []
    for cls in (argparse.ArgumentParser, LocalizedArgumentParser):
        buffer = io.StringIO()
        with redirect_stderr(buffer), pytest.raises(SystemExit) as exc:
            _parser(cls).parse_args(argv)
        assert exc.value.code == 2
        outputs.append(buffer.getvalue())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("text", "suffix", "params"),
    [
        ("unrecognized arguments: --x 'a b'", "unrecognized", {"arguments": "--x 'a b'"}),
        (
            "ambiguous option: --ti could match --title, --timeout",
            "ambiguous",
            {"option": "--ti", "matches": "--title, --timeout"},
        ),
        (
            "one of the arguments --name --id is required",
            "required_group",
            {"names": "--name --id"},
        ),
        ("argument --name: expected one argument", "expected_one", {}),
        ("argument --name: expected at most one argument", "expected_optional", {}),
        ("argument --name: expected at least one argument", "expected_some", {}),
        ("argument --pair: expected 2 arguments", "expected_count", {"count": 2}),
        (
            "argument --integer: invalid int value: 'oops'",
            "invalid_number",
            {"type": "int", "value": "'oops'"},
        ),
        (
            "argument --float: invalid float value: 'x: y'",
            "invalid_number",
            {"type": "float", "value": "'x: y'"},
        ),
        ("argument --id: not allowed with argument --name", "conflict", {"argument": "--name"}),
        (
            "argument -h/--help: ignored explicit argument 'oops'",
            "ignored_value",
            {"value": "'oops'"},
        ),
    ],
)
def test_every_stdlib_shape_uses_key_and_preserves_tokens(
    monkeypatch: pytest.MonkeyPatch, text: str, suffix: str, params: dict[str, object]
) -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def marker(message: Msg, locale: str | None = None) -> str:
        seen.append((message.code, dict(message.params)))
        return f"<{message.code}>"

    monkeypatch.setattr(cli_failures, "help_text", marker)
    output = localise_failure(text, "fr")
    assert seen[0] == (f"cli.failure_details.{suffix}", params)
    if text.startswith("argument "):
        assert output == "<cli.failure_details.argument>"
        assert seen[1][1]["message"] == f"<cli.failure_details.{suffix}>"
    else:
        assert output == f"<cli.failure_details.{suffix}>"


@pytest.mark.parametrize("count", [1, 2, 5, 21])
def test_fixed_nargs_keeps_exact_count_and_english_plural(count: int) -> None:
    noun = "argument" if count == 1 else "arguments"
    text = f"argument --values: expected {count} {noun}"
    assert localise_failure(text, "en") == text


def test_unknown_validator_body_is_not_rewritten() -> None:
    text = "argument --custom: plugin says: invalid int value is forbidden"
    assert localise_failure(text, "en") == text
    assert localise_failure("plugin says invalid int value: 'x'", "fr") == (
        "plugin says invalid int value: 'x'"
    )


@pytest.mark.parametrize("value", ["oops", "nan", "inf", "-1", "0"])
def test_seconds_validator_routes_to_key(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from local_operator import cli

    seen: list[str] = []

    def marker(message: Msg, locale: str | None = None) -> str:
        seen.append(message.code)
        return f"<{message.code}>"

    monkeypatch.setattr(cli, "_tr", marker)
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        cli._positive_seconds(value)
    expected = (
        "not_seconds"
        if value == "oops"
        else ("finite_seconds" if value in {"nan", "inf"} else "positive_seconds")
    )
    assert seen == [f"cli.failure_details.{expected}"]
    assert str(exc.value) == f"<cli.failure_details.{expected}>"
    assert cli._positive_seconds("0.5") == 0.5


def test_exec_refusal_preserves_shell_safe_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import shlex

    from local_operator import cli

    seen: list[Msg] = []

    def marker(message: Msg, locale: str | None = None) -> str:
        seen.append(message)
        return "<localized>"

    monkeypatch.setattr(cli, "_tr", marker)
    parser = LocalizedArgumentParser(prog="lop")
    for extras in (["do the work"], ["do the work", "never discard me"]):
        seen.clear()
        with redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as exc:
            cli._reject_extra_positionals(
                parser, argparse.Namespace(command="O'Brien's audit", extra=extras)
            )
        assert exc.value.code == 2
        assert seen[0].code == "cli.failure_details.positional_pair"
        message = seen[1]
        assert message.params["count"] == len(extras) + 1
        if len(extras) == 1:
            assert message.code == "cli.failure_details.exec_two"
            assert shlex.split(str(message.params["command"])) == [
                "lop",
                "exec",
                "--name",
                "O'Brien's audit",
                "do the work",
            ]
        else:
            assert message.code == "cli.failure_details.exec_many"
            assert "command" not in message.params
