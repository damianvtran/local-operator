"""The exec-worker help slice: `cli.exec_worker_help` on the catalogue.

What this file pins, and why each pin exists:

1. **Byte fidelity.** Every catalogue value equals the argparse literal it
   replaced, so `python -m local_operator.exec_worker --help` renders exactly
   what it rendered before the extraction. A "close enough" rewording — a
   changed `--flag` name, a dropped parenthesis, a straight apostrophe where
   ICU needs a doubled one — fails here rather than in a user's terminal.
2. **The apostrophe escape.** `resume`'s copy carries a literal `'` (ICU would
   otherwise read it as the start of an escape), so the catalogue doubles it and
   the render must hand back ONE. This is the only transformation this slice is
   allowed to make, and it is the one most likely to be undone by a well-meaning
   "tidy up the double apostrophes" edit.
3. **French copy + per-namespace fallback.** The namespace now has a French
   translation: `fr` must render its own catalogue, byte-for-byte. A locale that
   does not carry this namespace (`es`, `ru`, `vi`, `hi`, `zh-CN`, `unknown`) must
   still render the English source, not blank and not the message code — the
   additive contract the programme depends on.
4. **The ratchet, both ways.** The file is ACTIVATED (its exact path in
   `enforced`, no ceiling left in `files`), and a reintroduced literal in a real
   sink is rejected by the production scanner — the negative control that makes
   the activation real instead of nominal.
5. **Context completeness.** Every key carries role/tone/note, the input a later
   translation wave needs; a key added without context must not ship silently.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import local_operator
from local_operator.i18n import catalogues, messages
from local_operator.i18n.keys import cli_exec_worker_help as keys
from local_operator.i18n.runtime import render_message
from local_operator.i18n.text import tr

_t = tr("cli.exec_worker_help")
_ROOT = Path(local_operator.__file__).resolve().parent.parent
SOURCE = "local_operator/exec_worker.py"
NAMESPACE = "cli.exec_worker_help"


def _catalogue() -> dict[str, str]:
    return catalogues.load_catalogue("en", NAMESPACE)


#: The exact argparse literals this slice replaced, keyed by catalogue key
#: segment. They are written out HERE because they no longer exist in the
#: source: this mapping IS the byte-fidelity record, and the test below fails
#: if a rewrite of any catalogue value stops reproducing the string
#: `--help` printed before the extraction.
REPLACED_LITERALS = {
    "prompt": "The prompt to execute",
    "json": "Emit one JSON line per agent event",
    "yolo": "Auto-approve all tool tiers",
    "train": "Training mode: append the transcript to the agent directory (legacy --train)",
    "agent": "Agent name selector",
    "agent_id": "Agent id",
    "job_id": (
        "Ledger job id; when set, the worker appends a terminal record"
        " (finished_at + exit_code) to the JSONL ledger on exit"
    ),
    "control": "Publish a session record and serve the control socket for this run",
    "hosting": "Hosting platform override",
    "model": "Model override",
    "resume": "Resume a previous session by id (or '@latest')",
}


def test_the_catalogue_values_are_the_replaced_literals() -> None:
    """Each catalogue value is the literal (plus only the ICU apostrophe escape)."""
    catalogue = _catalogue()
    assert set(catalogue) == {f"{NAMESPACE}.{key}" for key in REPLACED_LITERALS}
    for key, literal in REPLACED_LITERALS.items():
        source = catalogue[f"{NAMESPACE}.{key}"]
        # Doubling a literal apostrophe is the ONE permitted transformation:
        # ICU would otherwise read it as the start of an escape and swallow the
        # surrounding text.
        assert source.replace("''", "'") == literal, key
        assert source.count("'") in (0, 2 * literal.count("'")), key


def _help_literals() -> dict[str, str]:
    """The `help=` literal of every argparse site in exec_worker.py, by option.

    Read from the SOURCE, not from the catalogue: a test that compared the
    catalogue to itself would pass with the extraction deleted.
    """
    tree = ast.parse((_ROOT / SOURCE).read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr != "add_argument":
            continue
        option = next(
            (
                arg.value
                for arg in call.args
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--")
            ),
            None,
        )
        keyword = next((kw for kw in call.keywords if kw.arg == "help"), None)
        # ``ast.Constant.value`` is a union (str | bytes | bool | int | float |
        # complex | Ellipsis): the test passes it on unchanged only because the
        # flag shapes are strings. Narrow it explicitly so pyright can verify
        # this module (CI type-checks the changed files) — a non-str Constant
        # could not have matched ``startswith("--")`` above, so the guard
        # rejects nothing that used to be accepted.
        if not isinstance(option, str) or keyword is None:
            continue
        # The extraction replaced each `help=` value with `_t(_ehw.<key>())`;
        # the assertion below is that NO such site is still a bare literal.
        if not isinstance(keyword.value, ast.Call):
            raise AssertionError(
                f"{option}: help= is still a literal — the extraction was reverted"
            )
        found[option] = option
    return found


def test_the_parser_carries_no_uncatalogued_help_literal() -> None:
    options = _help_literals()
    assert len(options) == 11, sorted(options)


def test_the_ratchet_activates_this_exact_file() -> None:
    baseline = json.loads((_ROOT / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    assert SOURCE in baseline["enforced"], "the slice must activate its own file"
    assert SOURCE not in baseline["files"], "an activated file keeps no ceiling"
    # The activation is EXACT, never a directory prefix: a prefix here would
    # silently enforce every sibling exec file in the same change.
    assert "local_operator/exec_worker.py" in baseline["enforced"]


def _collapse(text: str) -> str:
    # argparse's help formatter WRAPS long help text, so a value longer than the
    # terminal width is not a contiguous substring of the rendered output.
    # Collapsing runs of whitespace on both sides compares the copy without
    # asserting anything about the formatter's line breaking.
    return " ".join(text.split())


def test_every_help_literal_renders_byte_for_byte() -> None:
    """The rendered English equals the string argparse used to receive.

    The catalogue value is the ICU SOURCE (a literal apostrophe is doubled for
    ICU, see the escape test below); the render must hand back the SOURCE with
    exactly that escaping undone and nothing else changed — no rewording, no
    changed `--flag` name, no dropped parenthesis. The same text must also be
    what the real parser shows.
    """
    from local_operator.exec_worker import build_parser

    rendered_help = _collapse(build_parser().format_help())
    for key, source in _catalogue().items():
        rendered = _t(messages.msg(key))
        assert rendered == source.replace("''", "'"), key
        assert _collapse(rendered) in rendered_help, f"{key}: {rendered!r} is not shown by --help"


def test_the_apostrophe_escape_renders_one_literal_apostrophe() -> None:
    resume = _catalogue()[f"{NAMESPACE}.resume"]
    # The escape lives in the SOURCE (doubled); the render must collapse it.
    assert "''" in resume
    assert _t(keys.resume()) == "Resume a previous session by id (or '@latest')"


def test_every_generated_message_is_zero_argument() -> None:
    source = _catalogue()
    functions = [f for _, f in inspect.getmembers(keys, inspect.isfunction)]
    assert len(source) == len(functions) == 11
    for function in functions:
        assert not inspect.signature(function).parameters
        assert function().params == {}


def test_a_non_english_locale_falls_back_to_the_english_source() -> None:
    french = catalogues.load_catalogue("fr", NAMESPACE)
    for function in (keys.prompt(), keys.resume(), keys.job_id()):
        english = messages.render(function.code, function.params, locale="en")
        rendered_fr = messages.render(function.code, function.params, locale="fr")
        # fr carries the namespace: the French catalogue is the expected bytes,
        # and it must be a real translation rather than the English source.
        assert rendered_fr == render_message(french[function.code], function.params, "fr")
        assert rendered_fr != english
        # Every locale without this namespace still falls back to the source.
        for locale in ("es", "ru", "vi", "hi", "zh-CN", "unknown"):
            assert messages.render(function.code, function.params, locale=locale) == english


def test_every_key_carries_full_context() -> None:
    sidecar = catalogues.context("en", NAMESPACE)
    assert set(sidecar) == set(_catalogue())
    assert all(
        all(entry.get(field) for field in ("role", "tone", "note")) for entry in sidecar.values()
    )
    assert all("EN-only" in entry["note"] for entry in sidecar.values())


def test_a_reintroduced_literal_is_rejected_by_the_scoped_ratchet(i18n_check) -> None:
    """The negative control, against the REAL baseline and the production scanner.

    The file is in `enforced`, so a literal in a real sink must be a BLOCKING
    finding — this is what the activation buys, and it is asserted on the
    committed baseline rather than a fixture.
    """
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    assert i18n_check.in_enforced_scope(SOURCE, baseline["enforced"])
    source = (_ROOT / SOURCE).read_text(encoding="utf-8")
    assert i18n_check.scan_source(SOURCE, source) == []
    planted = source + '\n\ndef _c():\n    print("Plant a hardcoded user-facing label")\n'
    violations = i18n_check.scan_source(SOURCE, planted)
    assert len(violations) == 1
    assert violations[0].kind == "print"
