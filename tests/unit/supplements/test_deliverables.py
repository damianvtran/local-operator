"""Highlights deliverables: the scratchpad carve-out, detection, pins, withheld and file audio.

Contract: the lopdev highlights contract (session 7c5c172b0e89's picks turn showed no
Highlights because every scratchpad file was denied as ``config-dir``). Each test pins one
clause, and the carve-out tests carry NEGATIVE CONTROLS: another session's scratchpad, a
credential name inside our own scratchpad, the config dir's own files, and a symlink out of
the scratchpad all stay denied.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.supplements import denylist
from local_operator.supplements.candidates import TIER_PIN, prefilter
from local_operator.supplements.contract import (
    SUPPLEMENT_CUSTOM_TYPE,
    reader_disposition,
)
from local_operator.supplements.decision import decide
from local_operator.supplements.evidence import Evidence
from local_operator.supplements.persistence import build_details
from local_operator.supplements.trigger import ToolCallItem, TurnItem
from tests.unit.supplements.support import call, result


def _write(path: Path, text: str = "x" * 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A relocated config dir with two sessions; the cwd and home are separate trees.

    The config dir lives under ``tmp_path`` (a temp root), which is exactly what a relocated
    ``LOCAL_OPERATOR_CONFIG_DIR`` sandbox looks like: the scratch rule must yield to the
    session's own scratchpad there too.
    """
    config = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    own = config / "sessions" / "aaaaaaaaaaaa" / "scratchpad"
    other = config / "sessions" / "bbbbbbbbbbbb" / "scratchpad"
    own.mkdir(parents=True)
    other.mkdir(parents=True)
    # The live runtime exports its own scratchpad; the carve-out must not depend on it.
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(own))
    work = tmp_path / "work"
    work.mkdir()
    return {"config": config, "own": own, "other": other, "work": work, "home": tmp_path}


def _run(items, answer: str, s: dict[str, Path], **kw):
    kw.setdefault("scratchpad", str(s["own"]))
    return prefilter(items, answer, cwd=str(s["work"]), home=str(s["home"]), **kw)


def _pin(*paths: str, name: str = "bash", call_id: str = "p1", command: str = "true") -> TurnItem:
    return TurnItem(
        id=f"a-{call_id}",
        role="assistant",
        tool_calls=(ToolCallItem(id=call_id, name=name, args={"command": command}, deliver=paths),),
    )


# -- the carve-out --------------------------------------------------------------------------


def test_own_scratchpad_is_allowed_only_by_the_supplements_entry_point(store) -> None:
    deliverable = _write(store["own"] / "notif" / "picks" / "01.mp3")
    # The shared predicate's meaning is unchanged for its other callers...
    assert denylist.is_sensitive(str(deliverable)) == denylist.RULE_CONFIG_DIR
    # ...and only the supplements variant, given THIS session's pad, lifts the root rule.
    assert (
        denylist.is_sensitive_for_supplements(deliverable, session_scratchpad=str(store["own"]))
        == ""
    )
    assert denylist.is_sensitive_for_supplements(deliverable) == denylist.RULE_CONFIG_DIR


def test_negative_controls_stay_denied(store) -> None:
    pad = str(store["own"])
    other = _write(store["other"] / "report.md")
    secret = _write(store["own"] / ".env")
    keyfile = _write(store["own"] / "deploy" / "id_rsa")
    tokenish = _write(store["own"] / "api_token.json")
    config_file = _write(store["config"] / "config.yml")
    master = _write(store["config"] / "master.key")
    session_file = _write(store["own"].parent / "transcript.jsonl")
    check = denylist.is_sensitive_for_supplements
    assert check(other, session_scratchpad=pad) == denylist.RULE_CONFIG_DIR
    assert check(secret, session_scratchpad=pad) != ""
    assert check(keyfile, session_scratchpad=pad) != ""
    assert check(tokenish, session_scratchpad=pad) == denylist.RULE_TOKEN_SECRET
    assert check(config_file, session_scratchpad=pad) == denylist.RULE_CONFIG_DIR
    assert check(master, session_scratchpad=pad) != ""
    # The session dir itself (transcript, attachments) is not the scratchpad.
    assert check(session_file, session_scratchpad=pad) == denylist.RULE_CONFIG_DIR


def test_a_symlink_from_the_scratchpad_out_to_the_config_dir_is_judged_by_its_target(
    store,
) -> None:
    target = _write(store["config"] / "audit.md")
    link = store["own"] / "looks-fine.md"
    link.symlink_to(target)
    assert (
        denylist.is_sensitive_for_supplements(link, session_scratchpad=str(store["own"]))
        == denylist.RULE_CONFIG_DIR
    )


def test_prefilter_admits_own_scratchpad_and_refuses_the_rest(store) -> None:
    mine = _write(store["own"] / "report.md")
    theirs = _write(store["other"] / "report2.md")
    secret = _write(store["own"] / "prod.env")
    items = [
        call("write", "c1", path=str(mine)),
        call("write", "c2", path=str(theirs)),
        call("write", "c3", path=str(secret)),
    ]
    out = _run(items, "Done.", store)
    assert [c.name for c in out.candidates] == ["report.md"]
    # The other session's file is refused as a LOCATION: here by the temp-root rule (the
    # fixture's config dir sits under tmp), at ~/.local-operator by ``config-dir`` -- the
    # denylist assertion in ``test_negative_controls_stay_denied`` pins that path.
    assert sum(out.rejected.values()) == 2 and out.rejected.get("sensitive", 0) >= 1
    # Without the carve-out (an agent directory, an older caller), nothing changes.
    assert _run(items, "Done.", store, scratchpad="").candidates == ()


def test_scratchpad_scheme_resolves_to_the_own_root(store) -> None:
    _write(store["own"] / "notes" / "plan.md")
    out = _run([call("write", path="scratchpad://notes/plan.md")], "Done.", store)
    assert [c.name for c in out.candidates] == ["plan.md"]


# -- detection ------------------------------------------------------------------------------


def test_cd_chain_through_the_scratchpad_variable_and_a_directory_destination(store) -> None:
    """Session 7c5c's shape: ``cd $PAD/x/shortlist && cp a.mp3 b.mp3 ../picks/``."""
    shortlist = store["own"] / "notif" / "shortlist"
    picks = store["own"] / "notif" / "picks"
    for name in ("a.mp3", "b.mp3"):
        _write(shortlist / name)
        _write(picks / name)
    command = "cd $LOCAL_OPERATOR_SCRATCHPAD/notif && cd shortlist; cp a.mp3 b.mp3 ../picks/"
    out = _run([call("bash", command=command)], "Done.", store)
    assert sorted(c.absolute for c in out.candidates) == sorted(
        str(picks / n) for n in ("a.mp3", "b.mp3")
    )
    assert {c.tier for c in out.candidates} == {2}


@pytest.mark.parametrize("tool", ["mv", "rsync -a", "ln -s"])
def test_more_copy_commands_name_their_destination(tool: str, store) -> None:
    _write(store["work"] / "out" / "deck.pdf")
    out = _run([call("bash", command=f"{tool} src.pdf out/deck.pdf")], "Done.", store)
    assert [c.path for c in out.candidates] == ["out/deck.pdf"]


def test_rsync_t_is_preserve_times_not_a_target_directory(store) -> None:
    _write(store["work"] / "out" / "a.csv")
    out = _run([call("bash", command="rsync -avt a.csv out/")], "Done.", store)
    assert [c.path for c in out.candidates] == ["out/a.csv"]


def test_heredoc_variable_path_write_is_found_by_the_scratchpad_sweep(store) -> None:
    """``out=os.path.join(os.environ[...], ...)`` names no literal: the bounded sweep finds it."""
    since = time.time() - 30
    _write(store["own"] / "notif-sounds" / "synth" / "s01.wav")
    stale = _write(store["own"] / "old" / "s00.wav")
    os.utime(stale, (1.0, 1.0))
    deep = _write(store["own"] / "a" / "b" / "c" / "d" / "too-deep.wav")
    command = (
        "python3 - <<'PY'\nimport os\nout=os.path.join(os.environ['LOCAL_OPERATOR_SCRATCHPAD'],"
        "'notif-sounds','synth')\nopen(os.path.join(out,'s01.wav'),'wb').write(b'')\nPY"
    )
    out = _run([call("bash", command=command)], "Done.", store, since=since)
    names = [c.name for c in out.candidates]
    assert names == ["s01.wav"], names
    assert deep.name not in names  # depth bound (<= 3) holds
    # No turn start, no sweep: an unbounded "everything" would over-match.
    assert _run([call("bash", command=command)], "Done.", store).candidates == ()


def test_answer_named_paths_touched_since_session_start_are_write_evidence(store) -> None:
    made = _write(store["own"] / "picks" / "01-two-note-chime.mp3")
    old = _write(store["work"] / "ancient.pdf")
    os.utime(old, (1.0, 1.0))
    session_start = time.time() - 3600
    answer = "Here: `picks/01-two-note-chime.mp3` and ~/work/ancient.pdf"
    # The bare-ish relative name resolves only against a written dir or the scratchpad.
    out = _run([], answer.replace("picks/", ""), store, session_since=session_start)
    assert out.candidates == ()  # a bare name with no landed dir resolves nowhere
    named = f"Here: `{made}` and ~/work/ancient.pdf"
    out = _run([], named, store, session_since=session_start)
    by_name = {c.name: c for c in out.candidates}
    assert by_name["01-two-note-chime.mp3"].tier == 2
    assert by_name["01-two-note-chime.mp3"].tool == "answer"
    assert "ancient.pdf" not in by_name or by_name["ancient.pdf"].tier == 3


def test_a_bare_name_in_a_code_span_resolves_under_the_scratchpad(store) -> None:
    _write(store["own"] / "summary.html")
    out = _run([], "Open `summary.html`.", store, session_since=time.time() - 60)
    assert [c.name for c in out.candidates] == ["summary.html"]


# -- pins -----------------------------------------------------------------------------------


def test_a_pin_is_tier_zero_and_always_featured_without_a_vendor_question(store) -> None:
    _write(store["work"] / "out" / "chart.png")
    written = [_write(store["work"] / f"r{i}.md") for i in range(5)]
    items = [_pin("out/chart.png")] + [
        call("write", f"w{i}", path=str(p)) for i, p in enumerate(written)
    ]
    out = _run(items, "Done.", store)
    assert out.candidates[0].tier == TIER_PIN and out.candidates[0].pinned

    class NeverPicksThePin:
        vendor_name = "fake"
        offered: list[str] = []

        async def decide(self, *, state, question):
            from types import SimpleNamespace

            NeverPicksThePin.offered = [str(v) for v in dict(question.criteria).values()]
            return SimpleNamespace(value="none", probabilities={"none": 1.0})

    decision = asyncio.run(
        decide(
            NeverPicksThePin(),
            user_text="",
            answer_text="Done.",
            candidates=out.candidates,
            evidence=Evidence(structured=False),
            want_files=True,
            want_graphics=False,
            max_featured=4,
        )
    )
    assert [c.name for c in decision.featured][:1] == ["chart.png"]
    assert all("chart.png" not in label for label in NeverPicksThePin.offered)


def test_pins_overflow_into_more(store) -> None:
    paths = [str(_write(store["work"] / f"p{i}.pdf")) for i in range(6)]
    out = _run([_pin(*paths)], "Done.", store)
    decision = asyncio.run(
        decide(
            None,
            user_text="",
            answer_text="Done.",
            candidates=out.candidates,
            evidence=Evidence(structured=False),
            want_files=True,
            want_graphics=False,
            max_featured=4,
        )
    )
    assert len(decision.featured) == 4 and len(decision.more) == 2
    assert decision.vendor in {"pin", "heuristic"}


def test_a_refused_pin_is_withheld_by_name_and_reason(store) -> None:
    _write(store["work"] / "x.key")
    _write(store["other"] / "theirs.pdf")
    items = [_pin("x.key", str(store["other"] / "theirs.pdf"), "gone.mp3")]
    out = _run(items, "Done.", store)
    assert out.candidates == ()
    assert dict(out.withheld) == {
        "x.key": "sensitive",
        "theirs.pdf": "outside",
        "gone.mp3": "missing",
    }
    assert out.skipped is None  # a withheld-only turn still reaches the writer


def test_a_pin_on_a_failed_call_is_still_disclosed(store) -> None:
    out = _run([_pin("never.pdf"), result("boom", "p1", error=True)], "Done.", store)
    assert dict(out.withheld) == {"never.pdf": "missing"}


def test_incidental_refusals_are_not_withheld(store) -> None:
    _write(store["work"] / ".env")
    out = _run([call("bash", command="echo x > .env")], "Done.", store)
    assert out.withheld == () and out.rejected == {"sensitive": 1}


def test_the_call_time_notice_is_one_line_only_when_refused(store) -> None:
    from local_operator.supplements.pins import refusal_line

    _write(store["work"] / "ok.pdf")
    _write(store["work"] / "x.key")
    cwd, pad = str(store["work"]), str(store["own"])
    assert refusal_line(["ok.pdf"], cwd=cwd, scratchpad=pad) == ""
    line = refusal_line(["ok.pdf", "x.key"], cwd=cwd, scratchpad=pad)
    assert line == "deliver: 1 not shown (sensitive file: x.key)"
    assert "\n" not in line


# -- the row --------------------------------------------------------------------------------


def test_a_withheld_only_row_is_written_and_renders(store) -> None:
    from local_operator.supplements.decision import Decision

    details = build_details(
        anchor="a" * 32,
        job="0" * 12,
        version=1,
        state="done",
        decision=Decision(),
        withheld=[(f"f{i}.pdf", "outside") for i in range(12)],
    )
    assert len(details.get("withheld", [])) == 10 and details.get("withheld_more") == 2
    assert details.get("withheld", [])[0] == {"name": "f0.pdf", "reason": "outside"}
    assert reader_disposition(details, job_live=False) == "block"
    # A row without the key is byte-identical to the old shape.
    plain = build_details(
        anchor="a" * 32, job="0" * 12, version=1, state="done", decision=Decision()
    )
    assert "withheld" not in plain and "withheld_more" not in plain


def test_pinned_files_carry_the_flag(store) -> None:
    from local_operator.supplements.persistence import file_entry

    _write(store["work"] / "deck.pdf")
    out = _run([_pin("deck.pdf")], "Done.", store)
    entry = file_entry(out.candidates[0])
    assert entry.get("pinned") is True and entry["why"] == "pinned by bash"


# -- written audio through the real runtime ------------------------------------------------


@pytest.mark.asyncio
async def test_written_audio_reaches_sounds_with_file_origin_through_a_live_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bash call writes a wav into THIS session's scratchpad and pins it; the live runtime
    stores the bytes and the row carries it in ``sounds[]`` (``origin: "file"``), not files[].
    """
    import io
    import wave

    from local_operator.harness.types import (
        AgentTool,
        StreamEndEvent,
        StreamTextDelta,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )
    from local_operator.session.attachments import AttachmentStore
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.test_session import ScriptedStream
    from tests.unit.supplements.test_runner import _await_row, _make_session

    config = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    session_dir = config / "sessions" / "cccccccccccc"
    pad = session_dir / "scratchpad"
    clip = pad / "picks" / "01-chime.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle_wav:
        handle_wav.setnchannels(1)
        handle_wav.setsampwidth(2)
        handle_wav.setframerate(8000)
        handle_wav.writeframes(b"\0\0" * 4000)  # 0.5 s

    async def execute(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(buffer.getvalue())
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="bash", content=[TextContent(text="ok")]
        )

    tool = AgentTool(
        name="bash",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "deliver": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
        },
        interruptible=True,
        execute=execute,
    )
    args = '{"command": "synth", "deliver": ["scratchpad://picks/01-chime.wav"]}'
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="bash", argument_delta=args),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Here is the chime."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    work = tmp_path / "work"
    work.mkdir()
    session = _make_session(session_dir, stream, [tool], work)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(work)
    )
    handle.subscribe(lambda: None)
    try:
        await session.prompt("make me a chime")
        row = await _await_row(session.transcript)
        assert row["files"] == [] and "withheld" not in row
        [sound] = row["sounds"]
        assert sound["origin"] == "file" and sound["pinned"] is True
        assert sound["mime"] == "audio/wav" and sound["duration_s"] == 0.5
        assert sound["path"].endswith("scratchpad/picks/01-chime.wav")
        assert not os.path.isabs(sound["path"])
        stored = AttachmentStore().get_bytes(sound["attachment"])
        assert stored is not None and stored[0] == buffer.getvalue()
    finally:
        await handle.dispose()
        await session.dispose()


# -- round-1 remediation --------------------------------------------------------------------


async def _live_turn(
    tmp_path: Path,
    monkeypatch,
    *,
    write,
    command: str,
    answer: str,
    vendor=None,
    deliver=None,
    expect_row=True,
):
    """One real Session turn whose single bash call runs ``write(pad)``; returns the row."""
    import json as _json

    from local_operator.harness.types import (
        AgentTool,
        StreamEndEvent,
        StreamTextDelta,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.test_session import ScriptedStream
    from tests.unit.supplements.test_runner import _await_row, _make_session

    config = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    session_dir = config / "sessions" / "dddddddddddd"
    pad = session_dir / "scratchpad"

    async def execute(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        write(pad)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="bash", content=[TextContent(text="ok")]
        )

    tool = AgentTool(
        name="bash",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "deliver": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
        },
        interruptible=True,
        execute=execute,
    )
    args = {"command": command, **({"deliver": deliver} if deliver else {})}
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(
                    index=0, id="c1", name="bash", argument_delta=_json.dumps(args)
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta=answer), StreamEndEvent(stop_reason="stop")],
        ]
    )
    work = tmp_path / "work"
    work.mkdir()
    session = _make_session(session_dir, stream, [tool], work)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(work)
    )
    if vendor is not None:
        handle._supplements._seam = lambda: vendor
    events: list[Any] = []
    session.subscribe(events.append)
    handle.subscribe(lambda: None)
    try:
        await session.prompt("go")
        if expect_row:
            row = await _await_row(session.transcript)
        else:
            # A turn with nothing to show writes NO row: wait for the job, then read.
            from tests.unit.session.test_session import wait_for

            await wait_for(
                lambda: handle._supplements._task is not None and handle._supplements._task.done(),
                timeout=10,
            )
            row = session.transcript.latest_custom(SUPPLEMENT_CUSTOM_TYPE) or {}
        beats = [e for e in events if e.type == "supplement_progress"]
        return row, beats
    finally:
        await handle.dispose()
        await session.dispose()


# A variable-path heredoc: no literal names any output, so only the sweeps can find them.
_HEREDOC = (
    "cd \"$LOCAL_OPERATOR_SCRATCHPAD\" && python3 - <<'PY'\nfor p in out(): open(p, 'wb')\nPY"
)


@pytest.mark.asyncio
async def test_r1_1_unnamed_swept_chunks_never_become_players(tmp_path, monkeypatch) -> None:
    """Review R1-1 repro: 12 staged chunk_NN.wav, answer names none -> 0 players."""

    def write(pad: Path) -> None:
        for n in range(12):
            _write(pad / "work" / f"chunk_{n:02d}.wav", "RIFF" + "x" * 60)

    row, _ = await _live_turn(
        tmp_path,
        monkeypatch,
        write=write,
        command=_HEREDOC,
        answer="Done, final mix is ready.",
        expect_row=False,
    )
    assert not row.get("sounds")
    # They were offered as ordinary files at most; the heuristic never auto-features
    # unnamed swept audio, and nothing beyond the sweep cap vanished silently.
    assert all(f["kind"] != "audio" for f in row.get("files", []))


@pytest.mark.asyncio
async def test_q1_ten_heredoc_mp3s_named_in_the_answer_all_become_sounds(
    tmp_path, monkeypatch
) -> None:
    """QA round 1 Q1: 10 heredoc-written MP3s + an HTML, all named by bare name in the
    answer, live two levels down in the scratchpad -> 10 sounds + the html, no cut."""
    names = [f"{n:02d}-clip.mp3" for n in range(1, 11)]

    def write(pad: Path) -> None:
        for name in names:
            _write(pad / "notif" / "shortlist" / name, "ID3" + "x" * 60)
        _write(pad / "notif" / "shortlist" / "shortlist.html", "<html></html>")

    answer = "Here they are:\n" + "\n".join(f"- `{n}`" for n in names) + "\n- `shortlist.html`"
    row, _ = await _live_turn(tmp_path, monkeypatch, write=write, command=_HEREDOC, answer=answer)
    assert sorted(s["path"].rsplit("/", 1)[-1] for s in row.get("sounds", [])) == names
    assert all(s["origin"] == "file" for s in row["sounds"])
    assert [f["name"] for f in row["files"]] == ["shortlist.html"]
    assert "files_more" not in row


def test_unnamed_sweep_hits_past_the_cap_are_counted_not_dropped(store) -> None:
    from local_operator.supplements.candidates import SWEEP_MAX_FILES

    for n in range(SWEEP_MAX_FILES + 3):
        _write(store["own"] / "out" / f"r{n:02d}.md")
    out = _run([call("bash", command=_HEREDOC)], "Done.", store, since=time.time() - 60)
    assert len([c for c in out.candidates if c.swept]) == SWEEP_MAX_FILES
    assert out.sweep_more == 3
    decision = asyncio.run(
        decide(
            None,
            user_text="",
            answer_text="Done.",
            candidates=out.candidates,
            evidence=Evidence(structured=False),
            want_files=True,
            want_graphics=False,
            max_featured=4,
        )
    )
    from dataclasses import replace as _replace

    details = build_details(
        anchor="a" * 32,
        job="0" * 12,
        version=1,
        state="done",
        decision=_replace(decision, more_unlisted=out.sweep_more),
    )
    assert details.get("files_more") == len(decision.more) + 3


def test_q2_the_sweep_floor_drops_the_previous_turns_files(store) -> None:
    now = time.time()
    old = _write(store["own"] / "a" / "prev.md")
    os.utime(old, (now - 1.0, now - 1.0))  # inside the 2 s slack of this turn's start
    _write(store["own"] / "a" / "this.md")
    items = [call("bash", command=_HEREDOC)]
    without = _run(items, "Done.", store, since=now)
    with_floor = _run(items, "Done.", store, since=now, sweep_floor=now - 0.5)
    assert {c.name for c in without.candidates} == {"prev.md", "this.md"}
    assert {c.name for c in with_floor.candidates} == {"this.md"}


def test_r1_2_the_call_time_notice_applies_deny_prefixes(store) -> None:
    from local_operator.supplements.pins import refusal_line

    _write(store["own"] / "a.mp3")
    pad = str(store["own"])
    common = {"cwd": str(store["work"]), "scratchpad": pad}
    assert refusal_line(["scratchpad://a.mp3"], deny_prefixes=(), **common) == ""
    line = refusal_line(["scratchpad://a.mp3"], deny_prefixes=(pad,), **common)
    assert line == "deliver: 1 not shown (outside allowed locations: a.mp3)"
    turn_end = _run([_pin("scratchpad://a.mp3")], "Done.", store, deny_prefixes=(pad,))
    assert dict(turn_end.withheld) == {"a.mp3": "outside"}


def test_ui_r2_open_frame_keeps_withheld() -> None:
    from local_operator.session.open_frame import _supplement_payload

    clean = _supplement_payload(
        {
            "details": {
                "anchor": "a" * 32,
                "job": "0" * 12,
                "version": 1,
                "state": "done",
                "files": [],
                "components": [],
                "at": 1.0,
                "withheld": [{"name": "x.key", "reason": "outside", "extra": "dropped"}],
                "withheld_more": 2,
            }
        }
    )
    assert clean is not None
    assert clean["details"]["withheld"] == [{"name": "x.key", "reason": "outside"}]
    assert clean["details"]["withheld_more"] == 2


@pytest.mark.asyncio
async def test_ui_r1_the_live_done_beat_carries_withheld(tmp_path, monkeypatch) -> None:
    """A refused pin yields a withheld-only row; the live done beat must carry it too."""
    row, beats = await _live_turn(
        tmp_path,
        monkeypatch,
        write=lambda pad: pad.mkdir(parents=True, exist_ok=True),
        command="true",
        answer="Done.",
        deliver=["scratchpad://never-written.pdf"],
    )
    assert row.get("withheld") == [{"name": "never-written.pdf", "reason": "missing"}]
    done = [b for b in beats if b.state == "done"]
    assert done and done[-1].withheld == row["withheld"]
    # Absence rule: a beat with nothing withheld stays byte-identical to the old shape.
    from local_operator.harness.types import SupplementProgressEvent

    plain = SupplementProgressEvent(anchor="a", job="j", version=1, state="done").model_dump()
    assert "withheld" not in plain and "withheld_more" not in plain


@pytest.mark.parametrize(
    ("locale", "separator"), [("en", ", "), ("fr", ", "), ("zh-CN", "、"), ("ur", "، ")]
)
def test_q3_tui_withheld_reasons_join_with_the_locale_separator(
    locale: str, separator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.tui import supplements as tui_supplements

    monkeypatch.setattr(tui_supplements, "active_locale", lambda: locale)
    line = tui_supplements.withheld_line(
        [{"name": "a", "reason": "outside"}, {"name": "b", "reason": "sensitive"}]
    )
    assert line.count(separator.strip() or separator) >= 1
