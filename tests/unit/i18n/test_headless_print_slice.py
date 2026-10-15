"""Exec runtime extraction: baseline bytes, data boundaries and live locale moves.

The golden sources are transcribed from 46c1710acf; the capture test drives the
real one-shot subscriber, not a parallel implementation of its renderer.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable, cast

import pytest
from rich.console import Console

from local_operator import headless_print as hp
from local_operator.harness.types import (
    FAULT_KEY,
    AgentEndEvent,
    AgentEvent,
    CompactionStartEvent,
    ModelChangeEvent,
    OutputValidationEvent,
    ReasoningDeltaEvent,
    RetryStartEvent,
    ToolExecutionEndEvent,
    ToolResult,
)
from local_operator.i18n import catalogues, resolve, text
from local_operator.session.protocol import SessionProtocol

GOLDEN = json.loads((Path(__file__).parent / "fixtures/headless_print_en_golden.json").read_text())


def cases() -> list[tuple[AgentEvent, str, dict[str, object]]]:
    return [
        (ReasoningDeltaEvent(message_id="m", delta="private"), "thinking", {}),
        (
            ToolExecutionEndEvent(
                tool_call_id="c",
                tool_name="bash",
                is_error=True,
                result=ToolResult(
                    tool_call_id="c", tool_name="bash", details={FAULT_KEY: "aborted"}
                ),
            ),
            "tool_interrupted",
            {"name": "bash"},
        ),
        (
            ToolExecutionEndEvent(
                tool_call_id="c",
                tool_name="bash",
                is_error=True,
                result=ToolResult(tool_call_id="c", tool_name="bash"),
            ),
            "tool_failed",
            {"name": "bash"},
        ),
        (
            OutputValidationEvent(
                format="json", attempt=2, max_attempts=3, ok=False, error="bad\n  data\x1b[2J"
            ),
            "output_check_failed",
            {"format": "json", "attempt": 2, "max_attempts": 3, "reason": "bad data"},
        ),
        (
            RetryStartEvent(attempt=2, error="temporary"),
            "retry",
            {"attempt": 2, "error": "temporary"},
        ),
        (
            ModelChangeEvent(provider="p", model_id="m", is_fallback=True),
            "model_fallback",
            {"selector": "p/m"},
        ),
        (
            ModelChangeEvent(provider="p", model_id="m", reason="model switched"),
            "model_switched",
            {"selector": "p/m"},
        ),
        (ModelChangeEvent(provider="p", model_id="m"), "model_returned", {"selector": "p/m"}),
        (CompactionStartEvent(reason="manual"), "compacting", {}),
        (AgentEndEvent(error="synthetic failure"), "error", {"sentence": "synthetic failure"}),
        (AgentEndEvent(aborted=True), "aborted", {}),
    ]


def render(event: AgentEvent) -> str:
    buffer = io.StringIO()
    hp.PrintRenderer(Console(file=buffer, width=240, no_color=True)).handle(event)
    return buffer.getvalue()


def test_inventory_and_context() -> None:
    assert catalogues.load_catalogue("en", "cli.exec_runtime") == GOLDEN
    assert len(GOLDEN) == 11
    context = json.loads(
        (catalogues.catalogue_root() / "en/cli.exec_runtime.context.json").read_text()
    )
    assert set(context) == set(GOLDEN)
    assert all(all(row.get(k) for k in ("role", "tone", "note")) for row in context.values())


@pytest.mark.parametrize("event,key,params", cases())
def test_english_bytes_and_print_options(event, key, params, monkeypatch) -> None:
    expected = GOLDEN["cli.exec_runtime." + key].format(**params)
    console = Console(file=io.StringIO(), width=240, no_color=True)
    calls = []
    monkeypatch.setattr(console, "print", lambda *args, **kwargs: calls.append((args, kwargs)))
    hp.PrintRenderer(console).handle(event)
    options: dict[str, object] = {"highlight": False}
    if key in ("thinking", "tool_interrupted", "output_check_failed", "tool_failed", "error"):
        options.update(style="red" if key in ("tool_failed", "error") else "dim", markup=False)
    if key == "output_check_failed":
        options["soft_wrap"] = True
    if key == "error":
        options["soft_wrap"] = False
    assert calls == [((expected,), options)]
    assert (
        render(event)
        == expected.replace("[dim]", "")
        .replace("[/dim]", "")
        .replace("[red]", "")
        .replace("[/red]", "")
        + "\n"
    )


def test_silent_metadata_refresh_and_success() -> None:
    assert render(ModelChangeEvent(provider="p", model_id="m", context_metadata=True)) == ""
    assert render(OutputValidationEvent(format="json", attempt=1, max_attempts=3, ok=True)) == ""


@pytest.mark.parametrize(
    "event,expected",
    [
        (
            OutputValidationEvent(
                format="json",
                attempt=1,
                max_attempts=2,
                ok=False,
                error="[bold]bad[/bold]\x1b[2J\n next",
            ),
            "! output check failed (json, attempt 1/2): [bold]bad[/bold] next\n",
        ),
        (ModelChangeEvent(provider="p", model_id="[bold]m[/bold]"), "back to p/[bold]m[/bold]\n"),
        (RetryStartEvent(attempt=1, error="[bold]bad[/bold]"), "retry 1: [bold]bad[/bold]\n"),
    ],
)
def test_placeholders_never_gain_markup_privileges(event, expected) -> None:
    assert render(event) == expected


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    for locale, values in (("en", GOLDEN), ("fr", {k: "FR " + v for k, v in GOLDEN.items()})):
        folder = tmp_path / locale
        folder.mkdir()
        (folder / "cli.exec_runtime.json").write_text(json.dumps(values))
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [hp._reload_copy])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    hp._reload_copy()
    text._sources.cache_clear()


def test_backend_locale_reresolves_every_branch(corpus) -> None:
    text.set_active_locale("fr")
    for event, _, _ in cases():
        assert render(event).startswith("FR ")
    text.set_active_locale("en")
    for event, _, _ in cases():
        assert not render(event).startswith("FR ")


def test_json_wire_is_locale_independent(corpus, capsys) -> None:
    event = ModelChangeEvent(provider="p", model_id="m", reason="model switched")
    hp.PrintRenderer(json_mode=True).handle(event)
    english = capsys.readouterr().out
    text.set_active_locale("fr")
    hp.PrintRenderer(json_mode=True).handle(event)
    assert capsys.readouterr().out == english
    assert json.loads(english) == hp.printable_event(event)


def test_planted_literal_fails_activated_checker(tmp_path, monkeypatch, i18n_check) -> None:
    # Run the actual CLI entrypoint on an isolated copy, never mutate the live
    # checkout while another role may be reading it.
    rel = "local_operator/headless_print.py"
    source = Path(hp.__file__).read_text()
    target = tmp_path / rel
    target.parent.mkdir()
    target.write_text(source)
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(json.dumps({"note": "slice fixture", "enforced": [rel], "files": {}}))
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted runtime prose")\n')
    assert i18n_check.main([]) == 1
    target.write_text(source)
    assert i18n_check.main([]) == 0


class FakeSession:
    session_id = "synthetic-headless"
    model_label = "synthetic/model"

    def __init__(self) -> None:
        self.handler: Callable[[AgentEvent], None] | None = None
        self.disposed = False

    def subscribe(self, handler: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self.handler = handler
        return lambda: None

    async def prompt(self, message: str) -> None:
        assert self.handler is not None
        for event, _, _ in cases():
            self.handler(event)

    async def dispose(self) -> None:
        self.disposed = True


@pytest.mark.asyncio
async def test_real_one_shot_path(capsys) -> None:
    session = FakeSession()
    code = await hp.run_print_mode(cast(SessionProtocol, session), ["synthetic prompt"])
    captured = capsys.readouterr()
    assert code == 1 and session.disposed
    assert captured.out == ""
    assert "fell back to p/m\nswitched to p/m\nback to p/m\n" in captured.err
    assert "! output check failed (json, attempt 2/3): bad data" in captured.err
    assert "Error: synthetic failure\naborted\n" in captured.err
