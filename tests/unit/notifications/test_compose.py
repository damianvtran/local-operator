"""Notification CONTENT: what a banner says, and what it must never say.

The composer is the single place that answers "may this banner carry the
conversation's name, or a line of its content?", for every surface at once —
the TUI's own notifier, the detached runtime's OS fallback, and the desktop
app's `notification` frame. That makes two classes of defect possible here and
nowhere else, and both are pinned below:

- **A privacy promise that only half holds.** `display.notification_session_
  name` is one flag over two facts (the name AND the snippet), because a
  snippet is strictly more session-derived than a name: a name is a topic, a
  snippet is content. A composer that honoured the flag for the title and
  leaked the last assistant line would make the settings copy false.
- **A banner that asserts something untrue about the session.** An errored or
  interrupted turn's last assistant line routinely reads as a success ("All
  412 tests pass."), so the snippet is `complete`-only — a rule decidable from
  the kind alone rather than from a judgement about prose.

Everything here is a pure function over a transcript directory, so the whole
file is fast and needs no app, no runtime and no store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.notifications import (
    NOTIFICATION_CONTRACT_VERSION,
    ComposedNotification,
    compose,
    gate_body,
)
from local_operator.tui.notify import (
    APP_NAME,
    BACKGROUND_FALLBACK_TITLE,
    BACKGROUND_SNIPPET_MAX_CHARS,
    BODIES,
    CONTEXTS,
    MAX_TITLE_CHARS,
)


def _session(tmp_path: Path, *, assistant: str = "", title: str = "") -> Path:
    """A session directory with an optional last assistant line and title.

    Writes the sidecar and the transcript in the shapes `resume.py` reads, so
    this exercises the real `stored_session_title`/`session_preview` rather
    than a double: the composer's whole job is to be correct against the files
    a live session actually leaves behind.
    """
    session = tmp_path / "sessions" / "abcdef123456"
    session.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [
        {
            "id": "user-1",
            "ts": 1.0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "do the thing"}]},
        }
    ]
    if assistant:
        rows.append(
            {
                "id": "assistant-1",
                "ts": 2.0,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": assistant}],
                },
            }
        )
    (session / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    if title:
        (session / "title.json").write_text(
            json.dumps({"text": title, "user_set": True, "names": [title]}), encoding="utf-8"
        )
    return session


@pytest.fixture
def names_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """`display.notification_session_name` on, whatever the developer's config says."""
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)


@pytest.fixture
def names_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: False)


def test_a_completion_carries_the_name_the_status_and_the_last_line(
    tmp_path: Path, names_on: None
) -> None:
    """T-C1. The whole point of the feature, in one assertion.

    Before this, every surface said "Turn complete / The agent finished its
    turn." — the session's name, its outcome and its last line were all
    reachable and none of them were shown.
    """
    session = _session(tmp_path, assistant="Migrated 14 tables.", title="Quota reporting fix")
    composed = compose("complete", session_dir=session)

    assert isinstance(composed, ComposedNotification)
    assert composed.title == "Quota reporting fix"
    assert composed.status == CONTEXTS["complete"] == "Complete"
    assert composed.body == "Migrated 14 tables."
    assert composed.body_is_snippet is True
    assert composed.title_is_session_name is True


def test_the_privacy_flag_removes_the_name_AND_the_snippet(tmp_path: Path, names_off: None) -> None:
    """T-C2. One flag, both facts — a half-honoured promise is the defect.

    A user who opted out of the name has necessarily opted out of the snippet:
    the flag exists to keep model-written session text off a screen other
    people can see, and a snippet is content where a name is only a topic. The
    banner that remains asserts nothing about the conversation at all.
    """
    session = _session(tmp_path, assistant="Migrated 14 tables.", title="Quota reporting fix")
    composed = compose("complete", session_dir=session)

    assert composed.title == APP_NAME
    assert composed.body == BODIES["complete"] == "Task complete"
    assert composed.body_is_snippet is False
    assert composed.title_is_session_name is False
    # The strongest form of the assertion: neither fact survives anywhere in
    # the frame, including the structured fields a renderer might read.
    assert "Quota reporting" not in repr(composed)
    assert "Migrated" not in repr(composed)


def test_an_error_never_borrows_the_last_assistant_line(tmp_path: Path, names_on: None) -> None:
    """T-C3. Pins review round 1 M1: a failure must not read as a success.

    The last thing a failing turn SAID is whatever it was saying before it
    stopped, and on a lock screen that sentence sits beside "Needs attention"
    with nothing to check it against.
    """
    session = _session(tmp_path, assistant="All 412 tests pass.", title="Release prep")
    composed = compose("error", session_dir=session)

    assert composed.body == BODIES["error"] == "Stopped with an error"
    assert composed.body_is_snippet is False
    assert composed.status == CONTEXTS["error"] == "Needs attention"
    # The NAME is still shown: it identifies which session needs attention,
    # which is the one thing the user has to know to act.
    assert composed.title == "Release prep"
    assert composed.title_is_session_name is True


def test_an_interruption_likewise_keeps_the_house_sentence(tmp_path: Path, names_on: None) -> None:
    """T-C4. Same rule, and the reasoning is the same shape.

    An interrupted session whose last assistant turn happened to close a
    sub-task lands in exactly M1's frame, and nothing in the kind tells the two
    apart — so the rule is "snippet iff complete", decidable from the kind
    alone rather than from a judgement about whether a particular line reads
    honestly.
    """
    session = _session(tmp_path, assistant="The migration is complete.", title="Schema work")
    composed = compose("interrupted", session_dir=session)

    assert composed.body == BODIES["interrupted"] == "Stopped before finishing"
    assert composed.body_is_snippet is False
    assert composed.status == CONTEXTS["interrupted"] == "Interrupted"


def test_a_long_line_is_cut_to_the_banner_budget_on_a_word_boundary(
    tmp_path: Path, names_on: None
) -> None:
    """T-C5. Proves `max_chars` was PASSED, not applied afterwards.

    `session_preview` defaults to a 200-char list-row budget and computes its
    word-boundary ellipsis against whatever budget it is given. Trimming its
    200-char answer down to 120 afterwards would cut mid-word, after the
    ellipsis had already been placed somewhere else — so the observable
    difference between the two implementations is exactly the boundary this
    asserts.
    """
    line = " ".join(f"word{n:03d}" for n in range(80))
    assert len(line) > 400
    session = _session(tmp_path, assistant=line, title="Long answer")
    composed = compose("complete", session_dir=session)

    assert composed.body_is_snippet is True
    assert len(composed.body) <= BACKGROUND_SNIPPET_MAX_CHARS == 120
    assert composed.body.endswith("…")
    # The cut landed between words: the text before the ellipsis is a run of
    # whole tokens, so no partial "wor" tail survives.
    kept = composed.body.rstrip("…").split()
    assert all(word in line.split() for word in kept)


def test_control_characters_never_reach_the_wire(tmp_path: Path, names_on: None) -> None:
    """T-C6. Both halves are model-written and both reach argv or an escape.

    ESC and BEL terminate an OSC string, so a session name or a snippet
    carrying `\\x1b]0;pwned\\x07` would close the sequence early and leave the
    remainder executing as terminal commands.
    """
    attack = "before \x1b]0;pwned\x07 after"
    session = _session(tmp_path, assistant=attack, title=attack)
    composed = compose("complete", session_dir=session)

    for value in (composed.title, composed.body):
        assert "\x1b" not in value
        assert "\x07" not in value
        assert "pwned" in value, "the scrub must strip the control bytes, not the text"
    assert len(composed.title) <= MAX_TITLE_CHARS


def test_a_nameless_session_says_something_true_for_its_kind(
    tmp_path: Path, names_on: None
) -> None:
    """T-C7. The fallback splits on kind because one sentence is not true of both.

    "A session finished" is a true sentence for a completion and a false one
    for a parked question, so a nameless gate falls back to the brand instead —
    which is what the detached runtime's gate fallback already does.
    """
    session = _session(tmp_path, assistant="done")
    assert compose("complete", session_dir=session).title == BACKGROUND_FALLBACK_TITLE
    assert compose("error", session_dir=session).title == BACKGROUND_FALLBACK_TITLE
    assert compose("interrupted", session_dir=session).title == BACKGROUND_FALLBACK_TITLE
    assert compose("ask", session_dir=session).title == APP_NAME
    assert compose("approval", session_dir=session).title == APP_NAME
    for kind in ("complete", "error", "interrupted", "ask", "approval"):
        assert compose(kind, session_dir=session).title_is_session_name is False


def test_a_gate_body_does_not_repeat_the_tool_name(tmp_path: Path, names_on: None) -> None:
    """T-C8. Pins round 4 Q3: "write: write: /path" on the headline surface.

    A tool's `describe_approval` already leads with its own action word and the
    title IS the tool name, so the naive `f"{title}: {detail}"` doubled it on
    every approval toast, every time.
    """
    session = _session(tmp_path, title="Deploy")
    already = compose(
        "approval", session_dir=session, gate_title="write", gate_detail="write: /etc/hosts"
    )
    assert already.body == "write: /etc/hosts"
    assert already.body.count("write:") == 1

    # The prefix IS applied when the detail does not already carry it, which is
    # the case the rule must not over-correct.
    plain = compose("approval", session_dir=session, gate_title="bash", gate_detail="rm -rf /tmp/x")
    assert plain.body == "bash: rm -rf /tmp/x"

    # No detail at all falls back to the vocabulary: the bare tool name says
    # less than "Waiting for approval", and an `ask` with no text rendered as a
    # single word with no hint that it was a question.
    assert compose("ask", session_dir=session, gate_title="ask").body == BODIES["ask"]
    assert gate_body("approval", "", "") == BODIES["approval"]
    # A gate never takes the snippet path even with assistant text on disk.
    assert compose("ask", session_dir=session).body_is_snippet is False


def test_an_unreadable_session_degrades_to_the_house_vocabulary(
    tmp_path: Path, names_on: None
) -> None:
    """T-C9. A notification is chrome; it must never raise into its caller.

    This runs inside the desktop bridge's 1 s attention poll and inside the
    TUI's turn-end handler. Both treat a failure here as costing the banner,
    never the poll or the turn — so every route out of `compose` returns a
    frame that asserts nothing rather than an exception.
    """
    missing = tmp_path / "sessions" / "nothinghere12"
    for kind in ("complete", "error", "interrupted", "ask", "approval"):
        composed = compose(kind, session_dir=missing)
        assert composed.body == BODIES[kind]
        assert composed.status == CONTEXTS[kind]
        assert composed.body_is_snippet is False

    # `session_dir=None` is the cold-bridge case: no directory to read at all.
    assert compose("complete", session_dir=None).body == BODIES["complete"]

    # A transcript that is not JSON at all, and a directory where a file is
    # expected — the two shapes a half-written or hand-mangled session takes.
    broken = tmp_path / "sessions" / "broken123456"
    broken.mkdir(parents=True)
    (broken / "transcript.jsonl").write_bytes(b"\xff\xfe not json at all\n")
    (broken / "title.json").mkdir()
    assert compose("complete", session_dir=broken).body == BODIES["complete"]

    # A settings store that cannot be read fails CLOSED: an unreadable config
    # must not be the thing that puts a conversation's name on a lock screen.
    session = _session(tmp_path, assistant="secret work", title="Secret client")

    def explode(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("settings unavailable")

    import local_operator.tui.notify as notify_module

    original = notify_module.settings_get
    notify_module.settings_get = explode  # type: ignore[assignment]
    try:
        degraded = compose("complete", session_dir=session)
    finally:
        notify_module.settings_get = original  # type: ignore[assignment]
    assert degraded.title == APP_NAME
    assert degraded.body == BODIES["complete"]


def test_a_live_rename_wins_over_the_stored_title(tmp_path: Path, names_on: None) -> None:
    """The caller's resolved name beats the sidecar, because a rename lands there first.

    A toast naming a session by the name it had ten seconds ago is worse than
    one naming it by none: the user goes looking for a conversation that no
    longer exists under that title.
    """
    session = _session(tmp_path, assistant="done", title="Old name")
    assert compose("complete", session_dir=session, session_name="New name").title == "New name"
    # An EMPTY live name is "I do not know", not "it has none", so the sidecar
    # still answers — this is the cold-bridge path.
    assert compose("complete", session_dir=session, session_name="").title == "Old name"


def test_the_contract_version_is_the_one_the_wire_advertises() -> None:
    """The frame's `contract` field and `/v1/capabilities` must agree.

    They are read by two different consumers (the payload validator and the
    capability gate that decides whether the renderer owns completion toasts),
    and a skew between them would silently disable the new path while the
    frames kept arriving.
    """
    import asyncio

    from local_operator.server.routes.capabilities import capabilities

    result = asyncio.run(capabilities()).result
    # `CRUDResponse.result` is a permissive union, so the shape is asserted
    # before it is indexed — which is also what makes a failure here say
    # "capabilities stopped returning a mapping" rather than a TypeError.
    assert isinstance(result, dict)
    features = result["features"]
    assert isinstance(features, dict)
    assert features["notification_contract"] == NOTIFICATION_CONTRACT_VERSION == 1


def _incident(session: Path, raw: str) -> None:
    """Append the `session_incident` record a classified failure journals.

    The exact shape `incidents.py` writes — a custom message whose `details`
    carry both the rendered model-facing `text` and the unrendered `raw`. Both
    are written here because the composer must be seen to pick the right one.
    """
    from local_operator.incidents import format_incident_message

    row = {
        "id": "incident-1",
        "ts": 3.0,
        "type": "message",
        "payload": {
            "kind": "custom",
            "custom_type": "session_incident",
            "details": {"text": format_incident_message(raw, "test", "model"), "raw": raw},
        },
    }
    with (session / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_an_error_body_names_what_actually_failed(tmp_path: Path, names_on: None) -> None:
    """D4. "Stopped with an error" is unactionable; the provider's text is not.

    The banner is the surface where the user decides whether to get up. A
    state with no cause tells them they must act while withholding the only
    fact that says WHICH action — top up a quota, fix a credential, or simply
    retry — so they have to open the session to learn anything at all.

    The text is the incident's `raw`, NOT its rendered `text`: the rendered
    form is a multi-line model-facing block tailed with "This is why the
    previous turn ended. Take it into account before repeating the same
    request.", which is an instruction addressed to the model and reads as
    nonsense on a lock screen.
    """
    session = _session(tmp_path, assistant="All 412 tests pass.", title="Release prep")
    _incident(session, "rate limit: quota exhausted for anthropic/claude-opus-5")
    composed = compose("error", session_dir=session)

    assert composed.body == "rate limit: quota exhausted for anthropic/claude-opus-5"
    assert composed.body_is_failure is True
    assert composed.status == CONTEXTS["error"] == "Needs attention"
    # Still never the last assistant line: that is the M1 rule, and the reason
    # this body is safe is precisely that it describes the FAILURE rather than
    # the work.
    assert composed.body_is_snippet is False
    assert "412 tests pass" not in composed.body
    # And the two flags are never both true: a body is one kind of text.
    assert not (composed.body_is_failure and composed.body_is_snippet)


def test_a_failure_with_no_recorded_cause_keeps_the_house_sentence(
    tmp_path: Path, names_on: None
) -> None:
    """The fallback, which is what makes the new body safe to add at all.

    Not every error path journals an incident (a failure classified nowhere, a
    transcript truncated before the record). The absence must read as the old
    banner rather than as an empty line under a title.
    """
    session = _session(tmp_path, assistant="All 412 tests pass.", title="Release prep")
    composed = compose("error", session_dir=session)

    assert composed.body == BODIES["error"] == "Stopped with an error"
    assert composed.body_is_failure is False

    # An incident record whose `raw` is missing or blank is the same case.
    with (session / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "incident-blank",
                    "ts": 4.0,
                    "type": "message",
                    "payload": {
                        "kind": "custom",
                        "custom_type": "session_incident",
                        "details": {"text": "rendered only", "raw": "   "},
                    },
                }
            )
            + "\n"
        )
    assert compose("error", session_dir=session).body == BODIES["error"]


def test_the_failure_text_is_bounded_and_scrubbed_like_every_other_body(
    tmp_path: Path, names_on: None
) -> None:
    """A provider's error envelope is untrusted text on the same wires.

    It reaches argv (`cmux notify`, `notify-send`) and an AppleScript literal,
    and it is not length-bounded at the source: a provider may return a wall of
    JSON. Same `sanitize_text(..., BACKGROUND_SNIPPET_MAX_CHARS)` discipline as
    the snippet, for the same reasons.
    """
    session = _session(tmp_path, title="Release prep")
    _incident(session, "boom \x1b]0;pwned\x07 " + " ".join(f"detail{n:03d}" for n in range(60)))
    composed = compose("error", session_dir=session)

    assert composed.body_is_failure is True
    assert len(composed.body) <= BACKGROUND_SNIPPET_MAX_CHARS == 120
    assert "\x1b" not in composed.body and "\x07" not in composed.body
    assert "pwned" in composed.body


def test_the_privacy_flag_removes_the_failure_text_too(tmp_path: Path, names_off: None) -> None:
    """One flag over every session-derived fact, failure text included.

    A provider's error envelope can quote a prompt fragment, a file path or an
    account identifier, so it is at least as sensitive as the name the flag was
    written for — a gate that covered the snippet but not this would leak the
    more identifying of the two.
    """
    session = _session(tmp_path, title="Project Atlas")
    _incident(session, "auth: credential ATLAS_PROD_KEY rejected for acct-99182")
    composed = compose("error", session_dir=session)

    assert composed.body == BODIES["error"]
    assert composed.body_is_failure is False
    assert "ATLAS" not in repr(composed) and "acct-99182" not in repr(composed)


def test_the_incident_type_this_scan_keys_on_is_the_one_incidents_writes() -> None:
    """`resume.py` spells the custom type literally; this is why that is safe.

    The module is import-guarded — nothing may drag the engine onto
    `local-operator --help` — so it cannot import `incidents` for one string.
    The duplication is fine only while something fails when the two diverge: a
    rename would otherwise turn the scan into one that matches nothing, and the
    banner would quietly degrade to "Stopped with an error" for every failure,
    looking exactly like a session that recorded no incident.
    """
    from local_operator.incidents import SESSION_INCIDENT_MESSAGE_TYPE
    from local_operator.resume import _SESSION_INCIDENT_TYPE

    assert _SESSION_INCIDENT_TYPE == SESSION_INCIDENT_MESSAGE_TYPE
