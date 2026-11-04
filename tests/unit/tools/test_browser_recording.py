"""Contract, approval and disk-verification tests for browser tab recording.

The wire is frozen by the design record's D8: two methods. ``record_start``
carries an absolute ``.mp4`` path plus the duration and byte caps;
``record_stop`` carries the tab and nothing else. Everything asserted here is a
property the HARNESS must enforce without trusting the host — the path is
composed and policed in Python (D3/D4), the approval names the real destination
(D4), and a reported recording is judged from disk rather than believed (D5).
"""

from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import local_operator.tools.builtin as builtin
from local_operator import browser_files, scratchpad
from local_operator.browser_bridge import backend as bridge_backend
from local_operator.browser_bridge.backend import (
    HOST_EXTENSION,
    HOST_UI,
    BridgeError,
    HostCapabilities,
    capability_refusal,
    format_error,
)
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    DESKTOP_ONLY_METHODS,
    METHODS,
    ErrorCode,
)
from local_operator.harness.types import BrowserSurface, ToolContext

RECORD_ACTIONS = ("record_start", "record_stop")

#: A minimal well-formed MP4 head: a box whose type is ``ftyp`` sits at bytes
#: 4..8, the only part of the container the harness inspects. A real clip would
#: prove nothing more about the CHECK.
MP4_HEAD = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40

NAME_RE = re.compile(r"^rec-\d{8}T\d{6}-[0-9a-f]{4}\.mp4$")


def _params(action: str = "record_start", **kwargs: Any) -> builtin.BrowserParams:
    return builtin.BrowserParams(action=action, **kwargs)


def _ctx(tmp_path: Path, **kwargs: Any) -> ToolContext:
    ctx = ToolContext(cwd=str(tmp_path), **kwargs)
    ctx.browser = BrowserSurface()
    ctx.browser.surface_id = "ui:7:nonce"
    return ctx


def _surface(surface: str = "ui:7:nonce") -> BrowserSurface:
    holder = BrowserSurface()
    holder.surface_id = surface
    return holder


# The approved-target memory is process-global (see `_APPROVED_RECORDINGS`), so
# every test starts and ends with it empty rather than inheriting a previous
# test's approval.
@pytest.fixture(autouse=True)
def _clear_approved_recordings():
    builtin._APPROVED_RECORDINGS.clear()
    yield
    builtin._APPROVED_RECORDINGS.clear()


class _AdvertisedHost:
    """A desktop host whose record advertises the recording methods.

    ``_capability_problem`` reads the discovery RECORD, so a test that passed no
    client would be asserting on the pre-feature refusal instead of on the call
    under test — and the real tool path always has a client.
    """

    host = HOST_UI

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(methods=RECORD_ACTIONS, version="0.93.0", capabilities_known=True)


def _call(
    action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: dict[str, Any] | None = None,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    context: ToolContext | None = None,
    client: Any = None,
    **kwargs: Any,
) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    """One `_browser_record` call over a fake wire, recording what it sent."""
    seen: list[tuple[str, dict[str, Any]]] = calls if calls is not None else []

    async def fake_bridge(tool_call_id, action_name, wire, *, surface="", client=None):
        seen.append((action_name, dict(wire)))
        if reply is not None:
            return reply, None
        return {"recording": True, "path": wire.get("path", "")}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    result = asyncio.run(
        builtin._browser_record(
            "call",
            _surface(),
            action,
            _params(action, **kwargs),
            context or _ctx(tmp_path),
            client=client or _AdvertisedHost(),
        )
    )
    return result, seen


def _run_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    start_path: str = "",
    stop_reply: dict[str, Any] | None = None,
    host_writes: Path | None = None,
) -> tuple[Any, Any, BrowserSurface, list[tuple[str, dict[str, Any]]]]:
    """A `record_start` then a `record_stop` on ONE surface, over a fake wire.

    `record_stop` can only prove it is saving the approved file if the approval
    survived the intervening tool call, so the interesting cases need both halves
    in one process — which is exactly the shape a real session has.

    ``host_writes`` is the file the fake HOST publishes at stop: the start must
    be given an ABSENT target (a recording never overwrites), and the bytes only
    appear when the host stops, which is the real order.
    """
    seen: list[tuple[str, dict[str, Any]]] = []
    state = _surface()

    async def fake_bridge(tool_call_id, action_name, wire, *, surface="", client=None):
        seen.append((action_name, dict(wire)))
        if action_name == "record_start":
            return {"recording": True, "path": wire["path"]}, None
        if host_writes is not None:
            host_writes.write_bytes(MP4_HEAD)
        return stop_reply, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    ctx = _ctx(tmp_path)
    start = asyncio.run(
        builtin._browser_record(
            "call",
            state,
            "record_start",
            _params("record_start", path=start_path),
            ctx,
            client=_AdvertisedHost(),
        )
    )
    stop = asyncio.run(
        builtin._browser_record(
            "call", state, "record_stop", _params("record_stop"), ctx, client=_AdvertisedHost()
        )
    )
    return start, stop, state, seen


# ---------------------------------------------------------------------------
# The protocol contract (D8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", RECORD_ACTIONS)
def test_record_actions_have_the_additive_protocol_contract(action: str) -> None:
    assert action in builtin.BROWSER_ACTIONS
    assert action in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS
    assert action in METHODS
    assert action in CAPABILITY_GATED_METHODS
    assert action in DESKTOP_ONLY_METHODS


def test_record_timeouts_are_the_ones_the_host_budget_nests_under() -> None:
    # start: attach + encoder launch; stop: host budget 45 s < this 60 s.
    assert COMMAND_TIMEOUTS["record_start"] == 20.0
    assert COMMAND_TIMEOUTS["record_stop"] == 60.0


# ---------------------------------------------------------------------------
# The approval describer (D4): the prompt names the file that will be written
# ---------------------------------------------------------------------------


def test_describer_names_the_resolved_path_inside_the_workspace(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    row = builtin._describe_browser_approval(
        {"action": "record_start", "path": str(target)}, str(tmp_path)
    )
    assert row == f"record: {target}"


def test_describer_marks_a_path_outside_the_workspace(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = tmp_path / "elsewhere.mp4"
    row = builtin._describe_browser_approval(
        {"action": "record_start", "path": str(target)}, str(workdir)
    )
    assert row.startswith(builtin.OUTSIDE_WORKSPACE_MARKER)
    assert row.endswith(str(target))


def test_describer_shows_a_scheme_shaped_target_without_resolving_it() -> None:
    row = builtin._describe_browser_approval(
        {"action": "record_start", "path": "scratchpad://clip.mp4"}, "/tmp"
    )
    assert row.startswith("record: ")
    assert "scratchpad" in row


def test_describer_names_the_directory_when_no_path_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    row = builtin._describe_browser_approval({"action": "record_start"}, str(tmp_path))
    assert row.startswith("record \u2192 ")
    assert row.endswith("/")
    assert str(Path(tmp_path / "cfg" / "browser" / "recordings")) in row


def test_describer_folds_the_action_case(tmp_path: Path) -> None:
    target = tmp_path / "clip.mp4"
    row = builtin._describe_browser_approval(
        {"action": "RECORD_START", "path": str(target)}, str(tmp_path)
    )
    assert row == f"record: {target}"


def test_describer_leaves_other_actions_alone() -> None:
    row = builtin._describe_browser_approval(
        {"action": "read", "path": "/tmp/clip.mp4", "url": "https://example.test/x"}, "/tmp"
    )
    assert row == "read: example.test/x"


# ---------------------------------------------------------------------------
# Argument validation (D4/D6/D13): `path` and `timeout_s` are reused, not new
# ---------------------------------------------------------------------------


def test_validation_accepts_a_plain_mp4_start_with_a_duration() -> None:
    assert builtin._validate_browser_args("record_start", _params(path="/tmp/a.mp4")) == ""
    assert builtin._validate_browser_args("record_start", _params(timeout_s=30)) == ""


def test_validation_accepts_an_omitted_path() -> None:
    assert builtin._validate_browser_args("record_start", _params()) == ""


def test_validation_refuses_a_path_on_record_stop() -> None:
    problem = builtin._validate_browser_args(
        "record_stop", _params("record_stop", path="/tmp/a.mp4")
    )
    assert "record_start" in problem


def test_validation_refuses_a_duration_on_record_stop() -> None:
    problem = builtin._validate_browser_args("record_stop", _params("record_stop", timeout_s=10))
    assert "record_start" in problem


@pytest.mark.parametrize("value", [0, -1, 601, 5000])
def test_validation_bounds_the_recording_length(value: float) -> None:
    problem = builtin._validate_browser_args("record_start", _params(timeout_s=value))
    assert problem
    assert "600" in problem


@pytest.mark.parametrize("value", [1.5, 90.5, 599.5])
def test_validation_refuses_a_fractional_length(value: float) -> None:
    """The host takes WHOLE seconds (`Number.isInteger`); refuse before capture.

    A fraction used to reach the wire, and the host refused it AFTER the
    approval prompt with `record_failed` and no capture (review round 1, R1-2).
    """
    problem = builtin._validate_browser_args("record_start", _params(timeout_s=value))
    assert "whole number" in problem
    assert f"{value:g}" in problem


def test_validation_accepts_an_integral_float_length() -> None:
    # `timeout_s=30.0` is a JSON number a caller can reasonably send; it is not a
    # fraction, so it must not be refused (the wire carries 30, an integer).
    assert builtin._validate_browser_args("record_start", _params(timeout_s=30.0)) == ""


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validation_refuses_a_non_finite_length(value: float) -> None:
    """NaN/inf pass every range test and blow up `int()`; refuse them first (F2)."""
    problem = builtin._validate_browser_args("record_start", _params(timeout_s=value))
    assert "finite" in problem


def test_a_non_finite_length_arrives_through_json() -> None:
    """The wire accepts bare `NaN`/`Infinity`, so the shape is reachable (F2)."""
    params = builtin.BrowserParams.model_validate_json(
        '{"action": "record_start", "timeout_s": NaN}'
    )
    value = params.timeout_s
    assert value is not None
    assert math.isnan(value)
    assert "finite" in builtin._validate_browser_args("record_start", params)


@pytest.mark.parametrize("name", ["clip.mkv", "clip", "clip.mp4.part", "clip.MPEG4"])
def test_validation_requires_an_mp4_destination(name: str) -> None:
    problem = builtin._validate_browser_args("record_start", _params(path=name))
    assert ".mp4" in problem


def test_validation_accepts_an_uppercase_suffix() -> None:
    assert builtin._validate_browser_args("record_start", _params(path="/tmp/CLIP.MP4")) == ""


# ---------------------------------------------------------------------------
# Approval tiers (D4): the write is approved at start, the stop is a read
# ---------------------------------------------------------------------------


def test_start_is_write_and_stop_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "ui_browser_advertisable", lambda: True)
    tool = builtin.build_browser_tool(_ctx(Path("/tmp")))
    assert tool is not None
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"action": "record_start"}) == "write"
    assert tool.call_approval_tier({"action": "RECORD_START", "path": "/tmp/a.mp4"}) == "write"
    assert tool.call_approval_tier({"action": "record_stop"}) == "read"


# ---------------------------------------------------------------------------
# The wire (D8): frozen fields, an absolute path, and the caps as numbers
# ---------------------------------------------------------------------------


def test_start_sends_the_frozen_wire_with_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    result, seen = _call("record_start", tmp_path, monkeypatch, path=str(target))
    assert not result.is_error
    assert len(seen) == 1
    action, wire = seen[0]
    assert action == "record_start"
    assert wire["tab"] == "ui:7:nonce"
    assert wire["path"] == str(target)
    assert wire["max_duration_s"] == 120.0
    assert wire["max_bytes"] == browser_files.RECORD_MAX_BYTES
    # Nothing the design did not name: the identity fields and the three
    # recording fields, and no second spelling of the destination.
    assert set(wire) == {
        "tab",
        "requester",
        "session_label",
        "path",
        "max_duration_s",
        "max_bytes",
    }


def test_start_composes_the_default_directory_when_no_path_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, seen = _call("record_start", tmp_path, monkeypatch)
    assert not result.is_error
    path = Path(seen[0][1]["path"])
    assert path.is_absolute()
    assert path.parent == Path(tmp_path / "cfg" / "browser" / "recordings" / "nosession")
    assert NAME_RE.match(path.name)
    assert path.parent.is_dir()
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_start_carries_the_requested_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, seen = _call("record_start", tmp_path, monkeypatch, timeout_s=600)
    assert not result.is_error
    assert seen[0][1]["max_duration_s"] == 600.0
    assert "600" in result.text


def test_start_under_a_scratchpad_sends_the_pad_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = tmp_path / "pad"
    pad.mkdir()
    result, seen = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        context=_ctx(tmp_path, scratchpad_dir=str(pad)),
        path=str(pad / "clip.mp4"),
    )
    assert not result.is_error
    assert seen[0][1]["max_bytes"] == scratchpad.SCRATCHPAD_MAX_WRITE_BYTES


# ---------------------------------------------------------------------------
# Refusals that must be LOCAL (D3/D4): a denied call writes and starts nothing
# ---------------------------------------------------------------------------


def test_start_refuses_a_path_outside_the_confinement_before_any_wire_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jail = tmp_path / "jail"
    jail.mkdir()
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        context=_ctx(tmp_path, confinement_root=str(jail)),
        path=str(tmp_path / "outside.mp4"),
        calls=seen,
    )
    assert result.is_error
    assert seen == []
    assert "confined" in result.text


def test_start_refuses_a_scratchpad_build_directory_before_any_wire_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = tmp_path / "pad"
    (pad / "node_modules").mkdir(parents=True)
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        context=_ctx(tmp_path, scratchpad_dir=str(pad)),
        path=str(pad / "node_modules" / "clip.mp4"),
        calls=seen,
    )
    assert result.is_error
    assert seen == []
    assert "node_modules" in result.text


def test_start_refuses_a_scheme_shaped_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call(
        "record_start", tmp_path, monkeypatch, path="scratchpad://clip.mp4", calls=seen
    )
    assert result.is_error
    assert seen == []


def test_start_refuses_an_existing_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "taken.mp4"
    target.write_bytes(MP4_HEAD)
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call("record_start", tmp_path, monkeypatch, path=str(target), calls=seen)
    assert result.is_error
    assert seen == []
    assert "exists" in result.text


def test_start_refuses_an_absent_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_parent = tmp_path / "gone" / "clip.mp4"
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call("record_start", tmp_path, monkeypatch, path=str(missing_parent), calls=seen)
    assert result.is_error
    assert seen == []
    assert "gone" in result.text


def test_start_refuses_a_host_that_says_it_did_not_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        path=str(target),
        reply={"recording": False, "path": str(target)},
    )
    assert result.is_error
    assert "nothing was captured" in result.text.lower()


def test_start_refuses_a_traversal_path_outside_the_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`..` is resolved BEFORE the confinement answer, so it cannot escape (R1-4)."""
    jail = tmp_path / "jail"
    jail.mkdir()
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        context=_ctx(tmp_path, confinement_root=str(jail)),
        path=str(jail / ".." / "escape.mp4"),
        calls=seen,
    )
    assert result.is_error
    assert seen == []
    assert "confined" in result.text


def test_start_refuses_a_symlink_leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leaf that is a link is refused even when it points at a real file.

    The assertion names the sentence, not the word "symlink": pytest builds the
    temp directory from the test's own name, so a bare `"symlink" in text`
    matched THIS test's path and passed while the resolver was quietly following
    the leaf (QA round 1, Q1). The resolved target must not be named either —
    naming it is what "wrote through the link" looked like.
    """
    real = tmp_path / "real.mp4"
    real.write_bytes(MP4_HEAD)
    link = tmp_path / "link.mp4"
    link.symlink_to(real)
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call("record_start", tmp_path, monkeypatch, path=str(link), calls=seen)
    assert result.is_error
    assert seen == []
    assert "a recording is never written through one" in result.text
    assert str(link) in result.text
    assert str(real) not in result.text


def test_start_refuses_a_dangling_symlink_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link to a MISSING file is still a link: resolving it found a free name.

    This is the case that actually wrote through the link — the resolved target
    did not exist, so `exists()` and `is_symlink()` on the RESOLVED path both
    said the destination was fine (QA round 1, Q1).
    """
    link = tmp_path / "dangling.mp4"
    link.symlink_to(tmp_path / "never-written.mp4")
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call("record_start", tmp_path, monkeypatch, path=str(link), calls=seen)
    assert result.is_error
    assert seen == []
    assert "a recording is never written through one" in result.text
    assert str(link) in result.text


def test_start_refuses_a_symlink_parent_that_leaves_the_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link DIRECTORY that leaves the jail is caught by resolution (R1-4)."""
    jail = tmp_path / "jail"
    outside = tmp_path / "outside"
    jail.mkdir()
    outside.mkdir()
    (jail / "alias").symlink_to(outside)
    seen: list[tuple[str, dict[str, Any]]] = []
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        context=_ctx(tmp_path, confinement_root=str(jail)),
        path=str(jail / "alias" / "clip.mp4"),
        calls=seen,
    )
    assert result.is_error
    assert seen == []
    assert "confined" in result.text


def test_start_resolves_a_symlink_parent_that_stays_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-workspace link directory is normalised: the wire carries the real path.

    The approval prompt and the host must both be promised the same file, so the
    resolved spelling is what travels — never the alias the model typed.
    """
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real)
    result, seen = _call(
        "record_start", tmp_path, monkeypatch, path=str(tmp_path / "alias" / "clip.mp4")
    )
    assert not result.is_error
    assert seen[0][1]["path"] == str(real / "clip.mp4")


# ---------------------------------------------------------------------------
# The stop is judged from disk (D5), never from the host's account alone
# ---------------------------------------------------------------------------


def test_stop_reports_what_landed_and_the_av1_caveat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, seen = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 4200,
            "auto_stopped": False,
            "fps": 10,
            "container": "mp4",
            "codec": "av1",
        },
    )
    assert not result.is_error
    # `record_stop` carries the tab and the identity fields, and NOT a path: the
    # destination was approved at start (D4/D13).
    assert set(seen[0][1]) == {"tab", "requester", "session_label"}
    assert result.details == {"path": str(target), "bytes": len(MP4_HEAD)}
    assert "AV1" in result.text
    assert "will not play" not in result.text  # one browser, one platform: no hard claim
    assert "QuickTime" in result.text


def test_stop_rejects_a_byte_count_that_does_not_match_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={"path": str(target), "bytes": len(MP4_HEAD) + 1, "duration_ms": 1},
    )
    assert result.is_error
    assert str(target) in result.text


def test_stop_rejects_a_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "gone.mp4"
    result, _ = _call(
        "record_stop", tmp_path, monkeypatch, reply={"path": str(target), "bytes": 10}
    )
    assert result.is_error
    assert str(target) in result.text


def test_stop_rejects_a_file_that_is_not_an_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    junk = b"not-an-mp4-at-all"
    target.write_bytes(junk)
    result, _ = _call(
        "record_stop", tmp_path, monkeypatch, reply={"path": str(target), "bytes": len(junk)}
    )
    assert result.is_error
    assert "mp4" in result.text.lower()


def test_stop_rejects_a_result_with_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _call("record_stop", tmp_path, monkeypatch, reply={"bytes": 10})
    assert result.is_error


def test_stop_tightens_a_recording_it_created_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file under the recordings root is ours, so it lands 0600 (D5)."""
    root = Path(tmp_path / "cfg" / "browser" / "recordings" / "nosession")
    root.mkdir(parents=True)
    target = root / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    target.chmod(0o644)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={"path": str(target), "bytes": len(MP4_HEAD), "duration_ms": 1},
    )
    assert not result.is_error
    assert target.stat().st_mode & 0o777 == 0o600


def test_stop_leaves_an_explicit_destination_mode_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    target.chmod(0o644)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={"path": str(target), "bytes": len(MP4_HEAD), "duration_ms": 1},
    )
    assert not result.is_error
    assert target.stat().st_mode & 0o777 == 0o644


def test_a_closed_tab_drops_the_pinned_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same rule as every other verb: a `tab_closed` refusal forgets the handle.

    The recording calls return BEFORE the shared tail that applies this, so the
    cleanup has to be spelled in the handler; without it a stale handle keeps
    being targeted by the next command, including the recovery verb.
    """

    async def fake_bridge(tool_call_id, action, wire, *, surface="", client=None):
        problem = builtin._error(tool_call_id, "browser", "the tab is gone")
        problem.details = {"error_code": "tab_closed"}
        return None, problem

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    state = _surface()
    builtin._APPROVED_RECORDINGS[state.surface_id] = "/tmp/approved.mp4"
    result = asyncio.run(
        builtin._browser_record(
            "call",
            state,
            "record_stop",
            _params("record_stop"),
            _ctx(tmp_path),
            client=_AdvertisedHost(),
        )
    )
    assert result.is_error
    assert state.surface_id == ""
    # F3: the approval names a file on a tab that no longer exists, so it goes too.
    assert not builtin._APPROVED_RECORDINGS


# ---------------------------------------------------------------------------
# Error rendering (D9): Python owns the copy, keyed by `data.reason`
# ---------------------------------------------------------------------------


def test_stop_must_report_the_path_that_was_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different file is refused, naming BOTH paths (review round 1, R1-3).

    Without the comparison Python certified whatever absolute `.mp4` the host
    named — the approval covered one file and the receipt vouched for another.
    """
    approved = tmp_path / "approved.mp4"
    other = tmp_path / "other.mp4"
    start, stop, _state, _seen = _run_pair(
        tmp_path,
        monkeypatch,
        start_path=str(approved),
        stop_reply={"path": str(other), "bytes": len(MP4_HEAD), "duration_ms": 1},
        host_writes=other,
    )
    assert not start.is_error
    assert stop.is_error
    assert str(other) in stop.text
    assert str(approved) in stop.text
    assert "nothing is reported as saved" in stop.text.lower()


def test_stop_accepts_the_approved_path_and_forgets_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    start, stop, state, _seen = _run_pair(
        tmp_path,
        monkeypatch,
        start_path=str(target),
        stop_reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 1000,
            "auto_stopped": False,
        },
        host_writes=target,
    )
    assert not start.is_error
    assert not stop.is_error
    assert stop.details == {"path": str(target), "bytes": len(MP4_HEAD)}
    # The approval is consumed by the stop that used it: a second stop must not
    # compare against a recording that no longer exists.
    assert state.surface_id not in builtin._APPROVED_RECORDINGS


def test_stop_without_a_remembered_approval_still_checks_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed process has no approval memory; the structural checks remain.

    The receipt is honest about what was verified: the file is there, its size
    matches the host's report and it is an MP4 — not that it is the approved one.
    """
    target = tmp_path / "clip.mp4"
    junk = b"not-an-mp4-at-all"
    target.write_bytes(junk)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={"path": str(target), "bytes": len(junk), "duration_ms": 1},
    )
    assert result.is_error
    assert "mp4" in result.text.lower()


def test_a_receipt_without_an_approval_says_it_was_not_re_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a restart the shape is verified and the approval is not (R2-F4)."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={"path": str(target), "bytes": len(MP4_HEAD), "duration_ms": 1000},
    )
    assert not result.is_error
    assert "(approval not re-verified after restart)" in result.text


def test_a_refused_second_start_keeps_the_live_recordings_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`record_active` proves a capture is LIVE, so its approval must survive.

    Dropping it there left the later stop with nothing to compare, so it
    certified whatever absolute `.mp4` the host named (review round 2, F1).
    """
    approved = tmp_path / "approved.mp4"
    other = tmp_path / "other.mp4"
    state = _surface()
    starts = 0

    async def fake_bridge(tool_call_id, action_name, wire, *, surface="", client=None):
        nonlocal starts
        if action_name == "record_start":
            starts += 1
            if starts == 1:
                return {"recording": True, "path": wire["path"]}, None
            problem = builtin._error(tool_call_id, "browser", "already recording")
            problem.details = {"error_code": "busy", "reason": "record_active"}
            return None, problem
        other.write_bytes(MP4_HEAD)
        return {"path": str(other), "bytes": len(MP4_HEAD), "duration_ms": 1}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    ctx = _ctx(tmp_path)
    first = asyncio.run(
        builtin._browser_record(
            "call",
            state,
            "record_start",
            _params("record_start", path=str(approved)),
            ctx,
            client=_AdvertisedHost(),
        )
    )
    second = asyncio.run(
        builtin._browser_record(
            "call",
            state,
            "record_start",
            _params("record_start", path=str(approved)),
            ctx,
            client=_AdvertisedHost(),
        )
    )
    assert not first.is_error
    assert second.is_error
    assert builtin._APPROVED_RECORDINGS[state.surface_id] == str(approved)
    stop = asyncio.run(
        builtin._browser_record(
            "call", state, "record_stop", _params("record_stop"), ctx, client=_AdvertisedHost()
        )
    )
    assert stop.is_error
    assert str(other) in stop.text
    assert str(approved) in stop.text


def test_a_transient_stop_error_keeps_the_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout or busy lane does not prove the encoder stopped (R2-F1)."""
    target = tmp_path / "clip.mp4"
    state = _surface()
    stops = 0

    async def fake_bridge(tool_call_id, action_name, wire, *, surface="", client=None):
        nonlocal stops
        if action_name == "record_start":
            return {"recording": True, "path": wire["path"]}, None
        stops += 1
        if stops == 1:
            problem = builtin._error(tool_call_id, "browser", "the bridge timed out")
            problem.details = {"error_code": "internal"}
            return None, problem
        target.write_bytes(MP4_HEAD)
        return {"path": str(target), "bytes": len(MP4_HEAD), "duration_ms": 1}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    ctx = _ctx(tmp_path)
    asyncio.run(
        builtin._browser_record(
            "call",
            state,
            "record_start",
            _params("record_start", path=str(target)),
            ctx,
            client=_AdvertisedHost(),
        )
    )
    failed = asyncio.run(
        builtin._browser_record(
            "call", state, "record_stop", _params("record_stop"), ctx, client=_AdvertisedHost()
        )
    )
    assert failed.is_error
    assert builtin._APPROVED_RECORDINGS[state.surface_id] == str(target)
    retried = asyncio.run(
        builtin._browser_record(
            "call", state, "record_stop", _params("record_stop"), ctx, client=_AdvertisedHost()
        )
    )
    assert not retried.is_error


def _wire_error(code: ErrorCode, reason: str, **data: Any) -> BridgeError:
    return BridgeError(code, "host text", {"reason": reason, **data})


@pytest.mark.parametrize(
    "reason, needle, extra",
    [
        # The payloads are the HOST's shapes (`recording.ts`): `chromium` is a
        # NUMBER major version, `age_ms` is integer milliseconds, `limit` and
        # `max_bytes` are numbers. A test with invented string fields cannot
        # catch a unit mismatch — which is how R1-1 shipped.
        ("record_unsupported", "Chromium 152", {"chromium": 152}),
        ("record_av1_unavailable", "AV1", {}),
        ("record_path", "path", {}),
        ("record_active", "already", {"path": "/tmp/clip.mp4", "age_ms": 12000}),
        ("record_not_active", "nothing to stop", {}),
        ("record_limit", "2", {"limit": 2}),
        ("record_too_large", "shorter", {"max_bytes": browser_files.RECORD_MAX_BYTES}),
        ("record_aborted", "discarded", {}),
    ],
)
def test_every_record_reason_renders_its_own_copy(
    reason: str, needle: str, extra: dict[str, Any]
) -> None:
    message = format_error(
        _wire_error(ErrorCode.INTERNAL, reason, **extra),
        action="record_stop",
        host=HOST_UI,
    )
    assert needle.lower() in message.lower()


def test_record_active_renders_the_host_age_in_seconds() -> None:
    """`age_ms` from the host becomes seconds in the copy (review round 1, R1-1)."""
    message = format_error(
        _wire_error(ErrorCode.BUSY, "record_active", path="/tmp/clip.mp4", age_ms=12500),
        action="record_start",
        host=HOST_UI,
    )
    assert "12.5 s ago" in message
    assert "/tmp/clip.mp4" in message


def test_record_active_without_an_age_says_so_without_inventing_one() -> None:
    message = format_error(
        _wire_error(ErrorCode.BUSY, "record_active"), action="record_start", host=HOST_UI
    )
    assert "started" not in message
    assert "already running" in message


def test_unsupported_names_the_chromium_version_the_host_reported() -> None:
    message = format_error(
        _wire_error(ErrorCode.INTERNAL, "record_unsupported", chromium=152),
        action="record_start",
        host=HOST_UI,
    )
    assert "152" in message
    assert "Nothing was written" in message


def test_an_unknown_record_reason_falls_back_to_the_host_message() -> None:
    message = format_error(
        _wire_error(ErrorCode.INTERNAL, "record_teleported"), action="record_start", host=HOST_UI
    )
    assert "host text" in message


def test_the_global_cap_keeps_the_count_copy() -> None:
    message = format_error(
        _wire_error(ErrorCode.BUSY, "record_limit", limit=2),
        action="record_start",
        host=HOST_UI,
    )
    assert "maximum of 2 recordings" in message


def test_a_non_count_record_limit_keeps_the_hosts_remedy() -> None:
    """M2: the size rule reuses `record_limit` and has no count."""
    error = BridgeError(
        ErrorCode.BUSY,
        "stop the other 4K clip, or record at 1920x1080 or below",
        {"reason": "record_limit"},
    )
    message = format_error(error, action="record_start", host=HOST_UI)
    assert "record at 1920x1080 or below" in message
    assert "maximum of" not in message


def test_a_visible_refusal_keeps_the_hosts_remedy() -> None:
    error = BridgeError(
        ErrorCode.INTERNAL,
        "activate another tab, or omit width/height",
        {"reason": "record_size", "param": "size", "why": "visible"},
    )
    sentence = bridge_backend._record_reason_message(error, "record_size")
    assert "activate another tab, or omit width/height" in sentence
    assert sentence.endswith("Nothing was started.")


def test_an_old_app_is_refused_as_a_build_that_predates_the_action() -> None:
    capabilities = HostCapabilities(methods=(), version="0.92.0")
    error = capability_refusal("record_start", host=HOST_UI, capabilities=capabilities)
    message = format_error(error, action="record_start", host=HOST_UI)
    assert "built before this action existed" in message


# ---------------------------------------------------------------------------
# A desktop-only method on the extension host names the right remedy (D12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["record_start", "record_stop", "network_start"])
def test_desktop_only_method_on_the_extension_says_desktop_app_only(method: str) -> None:
    error = capability_refusal(method, host=HOST_EXTENSION, capabilities=HostCapabilities())
    message = format_error(error, action=method, host=HOST_EXTENSION)
    assert "desktop app" in message
    assert "toggle" not in message.lower()
    assert "backend='ui'" in message


def test_a_capability_the_extension_can_serve_keeps_its_own_remedy() -> None:
    error = capability_refusal("download", host=HOST_EXTENSION, capabilities=HostCapabilities())
    message = format_error(error, action="download", host=HOST_EXTENSION)
    assert "desktop app" not in message


# ---------------------------------------------------------------------------
# Agent-chosen size up to 4K (RS1-RS9)
# ---------------------------------------------------------------------------


class _SizedHost(_AdvertisedHost):
    """The same desktop host, advertising sized recording as well (RS3)."""

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            methods=(*RECORD_ACTIONS, "record_size"), version="0.99.0", capabilities_known=True
        )


def test_an_unsized_start_is_the_old_wire_and_the_old_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backward compatibility: no size ⇒ byte-identical wire, no size in the copy."""
    result, seen = _call("record_start", tmp_path, monkeypatch)
    assert not result.is_error
    assert "width" not in seen[0][1]
    assert "height" not in seen[0][1]
    assert " at " not in result.text
    assert "width" not in (result.details or {})


def test_a_sized_start_sends_the_viewport_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, seen = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_SizedHost(),
        width=1920,
        height=1080,
        reply={"recording": True, "path": "", "width": 1920, "height": 1080},
    )
    assert not result.is_error
    assert (seen[0][1]["width"], seen[0][1]["height"]) == (1920, 1080)
    assert "1920x1080" in result.text
    assert (result.details or {})["width"] == 1920
    assert (result.details or {})["height"] == 1080


@pytest.mark.parametrize(
    "size, needle",
    [
        ({"width": 1920}, "given together"),
        ({"width": 100, "height": 100}, "240 to 3840"),
        ({"width": 1920, "height": 4000}, "240 to 3840"),
        ({"width": 1921, "height": 1080}, "must be even"),
        # Area, not the longer edge: portrait 2160x3840 is legal, a 1:1 4K square
        # is not, because the encoder's price tracks pixels.
        ({"width": 3840, "height": 3840}, "8,294,400"),
    ],
)
def test_an_unusable_size_is_refused_before_dispatch(size: dict[str, int], needle: str) -> None:
    refusal = builtin._browser_args_problem("call", "record_start", _params("record_start", **size))
    assert refusal is not None
    assert needle in refusal.text
    assert refusal.text.endswith("Nothing was started.")
    assert (refusal.details or {})["reason"] == "record_size"


def test_a_portrait_size_at_the_area_limit_is_allowed() -> None:
    assert (
        builtin._browser_args_problem("call", "record_start", _params(width=2160, height=3840))
        is None
    )


@pytest.mark.parametrize(
    "action, size",
    [
        ("record_stop", {"width": 1920, "height": 1080}),
        ("screenshot", {"width": 1920, "height": 1080}),
        ("snapshot", {"height": 720}),
    ],
)
def test_a_size_on_any_other_action_is_refused(action: str, size: dict[str, int]) -> None:
    problem = builtin._validate_browser_args(action, _params(action, **size))
    assert "only valid for 'record_start'" in problem


def test_a_bad_size_is_typed_through_the_real_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: the tool's OWN `execute` (i.e. `execute_browser`) types the refusal.

    The old text-only check in `_validate_browser_args` shadowed this result on
    the real path; `_browser_record` is a private lane, so a test there proved
    nothing about what a caller sees. No host is faked beyond the tool's
    availability gate: the argument refusal must not depend on one being
    reachable.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx(tmp_path))
    assert tool is not None

    async def _run():
        return await tool.execute(
            "call",
            {"action": "record_start", "width": 1921, "height": 1080},
            None,
            None,
            _ctx(tmp_path),
        )

    result = asyncio.run(_run())
    assert result.is_error
    details = result.details or {}
    assert details["reason"] == "record_size"
    assert (details["param"], details["why"]) == ("width", "odd")
    assert result.text.endswith("Nothing was started.")


@pytest.mark.parametrize("width", [True, 3840.5, "abc"])
def test_a_malformed_size_is_typed_through_the_real_entry(
    width: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same fault, typed instead of answered by the generic refusal.

    `_validation_error` names the field but carries no `record_size` reason, so a
    caller could not tell "nothing started" from any other bad argument.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx(tmp_path))
    assert tool is not None

    async def _run():
        return await tool.execute(
            "call",
            {"action": "record_start", "width": width, "height": 1080},
            None,
            None,
            _ctx(tmp_path),
        )

    result = asyncio.run(_run())
    assert result.is_error
    details = result.details or {}
    assert details["reason"] == "record_size"
    assert (details["param"], details["why"]) == ("width", "type")
    assert "must be a whole number of pixels" in result.text
    assert result.text.endswith("Nothing was started.")


def test_a_malformed_argument_beside_a_bad_size_keeps_the_plain_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a fault CONFINED to width/height is the size's to answer.

    A bad `path` is the caller's other mistake, and the generic validation error
    names it; claiming a size refusal would hide which field to fix.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx(tmp_path))
    assert tool is not None

    async def _run():
        return await tool.execute(
            "call",
            {"action": "record_start", "width": "abc", "height": 1080, "path": 5},
            None,
            None,
            _ctx(tmp_path),
        )

    result = asyncio.run(_run())
    assert result.is_error
    assert "path" in result.text
    assert (result.details or {}).get("reason") != "record_size"


@pytest.mark.parametrize(
    "size, needle",
    [
        ({"width": 1921, "height": 1080}, "must be even"),
        # A malformed size is type-checked by the hook, not by the loop's schema
        # pass: `anyOf: [integer, null]` carries no top-level type, so that pass
        # leaves width/height alone and these used to reach the gate.
        ({"width": True, "height": 1080}, "must be a whole number of pixels"),
        ({"width": 3840.5, "height": 1080}, "must be a whole number of pixels"),
        ({"width": "abc", "height": 1080}, "must be a whole number of pixels"),
    ],
)
def test_a_bad_size_is_refused_before_any_approval(
    size: dict[str, Any], needle: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan-time hook refuses without the gate being asked.

    Driven through the real `AgentLoop` planning seam with the real tool object;
    `request_approval` is the same callback the gate would call, so zero calls
    there is the evidence the operator never saw a consent row for a clip that
    could not start.
    """
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import LoopConfig, Message, ModelSpec, ToolCall

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx(tmp_path))
    assert tool is not None and tool.refuse_args is not None

    approvals: list[str] = []

    async def request_approval(tool_name: str, summary: str, **kwargs: object) -> bool:
        approvals.append(summary)
        return True

    context = _ctx(tmp_path, session_id="rec-refuse")
    context.request_approval = request_approval  # type: ignore[assignment]
    planned = asyncio.run(
        AgentLoop()._plan_call(  # noqa: SLF001 — the planning seam is the unit
            ToolCall(
                id="c",
                name="browser",
                arguments={"action": "record_start", **size},
            ),
            LoopContext(system_blocks=[], tools=[tool], tool_context=context),
            LoopConfig(
                model=ModelSpec(provider="test", model_id="m", context_window=100_000),
                convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
                stream_fn=lambda request, signal: None,
            ),
        )
    )
    assert planned.failure is not None
    assert needle in planned.failure.text
    assert "'width'" in planned.failure.text
    assert planned.failure.text.endswith("Nothing was started.")
    assert approvals == []


def test_an_app_without_sized_recording_is_refused_with_its_own_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OLD host would record 1280x720 and report success, so the gate is local."""
    result, seen = _call("record_start", tmp_path, monkeypatch, width=1920, height=1080)
    assert result.is_error
    assert seen == []
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert "1280x720" in result.text
    assert "built before this action existed" not in result.text
    assert "update" in result.text.lower()


def test_the_record_size_capability_copy_names_the_working_fallback() -> None:
    error = capability_refusal(
        "record_size",
        host=HOST_UI,
        capabilities=HostCapabilities(methods=RECORD_ACTIONS, version="0.93.0"),
    )
    message = format_error(error, action="record_size", host=HOST_UI)
    assert "only at the default 1280x720" in message
    assert "Update Local Operator" in message
    assert "built before this action existed" not in message


def test_the_extension_host_says_sized_recording_is_desktop_only() -> None:
    error = capability_refusal("record_size", host=HOST_EXTENSION, capabilities=HostCapabilities())
    message = format_error(error, action="record_size", host=HOST_EXTENSION)
    assert "desktop app" in message
    assert "backend='ui'" in message
    # The real gate on that host is the ACTION, because `record_start` is
    # DESKTOP_ONLY: this is the sentence a sized call actually meets there.
    action_error = capability_refusal(
        "record_start", host=HOST_EXTENSION, capabilities=HostCapabilities()
    )
    action_message = format_error(action_error, action="record_start", host=HOST_EXTENSION)
    assert "record_start" in action_message
    assert "backend='ui'" in action_message


def test_a_clip_that_comes_back_at_another_size_keeps_the_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_SizedHost(),
        width=1920,
        height=1080,
        reply={"recording": True, "path": "", "width": 1280, "height": 720},
    )
    assert result.is_error
    assert "NOT at the 1920x1080" in result.text
    assert "1280x720" in result.text
    details = result.details or {}
    assert details["why"] == "not_applied"
    assert details["reason"] == "record_size_mismatch"
    # A clip IS recording at the wrong size: the absent-reason whitelist must not
    # read this as proof that nothing runs, or a `record_stop` would be refused.
    assert "record_size_mismatch" not in builtin._RECORD_ABSENT_REASONS
    target = (result.details or {})["target"]
    # The encoder is running: the approved destination survives so the stop can
    # save what is actually being recorded.
    assert builtin._APPROVED_RECORDINGS["ui:7:nonce"] == target


def test_a_sized_start_without_an_echo_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _call(
        "record_start", tmp_path, monkeypatch, client=_SizedHost(), width=1920, height=1080
    )
    assert result.is_error
    assert "unreported size" in result.text


def test_record_stop_still_saves_a_clip_that_came_back_at_the_wrong_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_SizedHost(),
        width=1920,
        height=1080,
        reply={"recording": True, "path": "", "width": 1280, "height": 720},
    )
    target = Path((start.details or {})["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(MP4_HEAD)
    stop, seen = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        client=_SizedHost(),
        reply={"path": str(target), "bytes": len(MP4_HEAD), "duration_ms": 10},
    )
    assert not stop.is_error
    assert str(target) in stop.text
    assert seen[0][0] == "record_stop"


def test_record_size_proves_no_recording_is_running() -> None:
    assert "record_size" in builtin._RECORD_ABSENT_REASONS


def test_the_approval_row_and_the_executor_agree_on_string_sizes(tmp_path: Path) -> None:
    """M3: `"3840"` and `3840.0` ARE 3840 to the executor, so the row names it."""
    row = builtin._describe_browser_approval(
        {
            "action": "record_start",
            "path": str(tmp_path / "clip.mp4"),
            "width": "3840",
            "height": "2160.0",
        },
        str(tmp_path),
    )
    assert row.endswith(" at 3840x2160")
    assert builtin._record_size_from_args(
        {"action": "record_start", "width": "3840", "height": "2160"}
    ) == (3840, 2160)


@pytest.mark.parametrize("width", [True, False, "3840.5", 3840.5, "abc"])
def test_a_size_that_is_not_whole_pixels_is_refused(width: Any) -> None:
    with pytest.raises(ValidationError):
        builtin.BrowserParams(action="record_start", width=width, height=1080)


def test_a_string_size_reaches_the_wire_at_the_size_the_row_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, seen = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_SizedHost(),
        width="1920",
        height=1080.0,
        reply={"recording": True, "path": "", "width": 1920, "height": 1080},
    )
    assert not result.is_error
    assert seen[0][1]["width"] == 1920
    assert seen[0][1]["height"] == 1080


def test_the_approval_row_names_the_size(tmp_path: Path) -> None:
    row = builtin._describe_browser_approval(
        {
            "action": "record_start",
            "path": str(tmp_path / "clip.mp4"),
            "width": 3840,
            "height": 2160,
        },
        str(tmp_path),
    )
    assert row.endswith(" at 3840x2160")
    default_dir = builtin._describe_browser_approval(
        {"action": "record_start", "width": 1920, "height": 1080}, str(tmp_path)
    )
    assert default_dir.endswith(" at 1920x1080")


def test_the_hosts_own_size_refusal_ends_nothing_was_started() -> None:
    from local_operator.browser_bridge import backend as bridge_backend

    error = _wire_error(ErrorCode.INTERNAL, "record_size", param="width", why="odd")
    sentence = bridge_backend._record_reason_message(error, "record_size")
    assert "both sides must be even" in sentence
    assert sentence.endswith("Nothing was started.")
    message = format_error(error, action="record_start", host=HOST_UI)
    assert "Nothing was started." in message


# ---------------------------------------------------------------------------
# Opt-in high-quality recording: quality="high" (H.264)
# ---------------------------------------------------------------------------


class _QualityHost(_AdvertisedHost):
    """The same desktop host, advertising `record_quality` as well."""

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            methods=(*RECORD_ACTIONS, "record_size", "record_quality"),
            version="0.99.0",
            capabilities_known=True,
        )


#: A high-quality start result as the desktop host shapes it (`recording.ts`).
HIGH_START = {
    "recording": True,
    "path": "",
    "quality": "high",
    "codec": "h264",
    "fps": 10,
    "encoder": "hardware",
    "bitrate_bps": 8_000_000,
    "width": 1920,
    "height": 1080,
}


def test_a_default_start_never_mentions_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backward compatibility: no `quality` ⇒ the wire and the copy are as before."""
    result, seen = _call("record_start", tmp_path, monkeypatch, client=_QualityHost())
    assert not result.is_error
    assert "quality" not in seen[0][1]
    assert "AV1" in result.text
    assert "H.264" not in result.text
    assert "quality" not in (result.details or {})


def test_standard_is_the_same_request_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`standard` is not sent and needs no capability: it is the default clip."""
    result, seen = _call("record_start", tmp_path, monkeypatch, quality="standard")
    assert not result.is_error
    assert "quality" not in seen[0][1]
    assert "AV1" in result.text


def test_quality_is_folded_and_an_empty_string_means_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _params("record_start", quality="").quality is None
    assert _params("record_start", quality="  High ").quality == "high"
    result, seen = _call("record_start", tmp_path, monkeypatch, quality="")
    assert not result.is_error
    assert "quality" not in seen[0][1]


def test_a_high_start_sends_quality_and_reports_the_h264_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, seen = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_QualityHost(),
        quality="high",
        reply=dict(HIGH_START),
    )
    assert not result.is_error
    assert seen[0][1]["quality"] == "high"
    details = result.details or {}
    assert details["quality"] == "high"
    assert details["codec"] == "h264"
    assert details["encoder"] == "hardware"
    assert (details["width"], details["height"]) == (1920, 1080)
    assert details["bitrate_bps"] == 8_000_000
    assert "H.264" in result.text
    assert "Chromium" in result.text
    assert "most players" not in result.text  # only Chromium and AVFoundation were tested
    # The AV1 sentence would be false for this clip.
    assert "AV1" not in result.text
    assert "hardware encoder" in result.text
    assert "1920x1080" in result.text
    assert builtin._APPROVED_RECORDINGS["ui:7:nonce"] == details["target"]


def test_a_sized_high_start_sends_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, seen = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_QualityHost(),
        quality="high",
        width=1920,
        height=1080,
        reply=dict(HIGH_START),
    )
    assert not result.is_error
    assert (seen[0][1]["width"], seen[0][1]["height"], seen[0][1]["quality"]) == (
        1920,
        1080,
        "high",
    )


def test_a_host_without_record_quality_refuses_before_any_path_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old app would ignore the field and record AV1 while reporting success."""
    result, seen = _call("record_start", tmp_path, monkeypatch, quality="high")
    assert result.is_error
    assert seen == []
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert "cannot record in high quality" in result.text
    assert "without quality" in result.text
    assert "built before this action existed" not in result.text
    # No destination was composed, so nothing can be approved or stopped later.
    assert builtin._APPROVED_RECORDINGS == {}
    assert not (tmp_path / "cfg" / "browser" / "recordings").exists()
    # QA Q2: the remedy already says "then retry"; the sentence must not say it twice.
    assert result.text.lower().count("retry") == 1


def test_the_extension_host_says_high_quality_is_desktop_only() -> None:
    error = capability_refusal(
        "record_quality", host=HOST_EXTENSION, capabilities=HostCapabilities()
    )
    message = format_error(error, action="record_quality", host=HOST_EXTENSION)
    assert "desktop app" in message
    assert "backend='ui'" in message


def test_a_stale_host_that_recorded_av1_is_a_failed_start_that_keeps_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record said `record_quality`, the host ignored it: the echo catches it."""
    result, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_QualityHost(),
        quality="high",
        reply={"recording": True, "path": "", "codec": "av1", "fps": 10},
    )
    assert result.is_error
    assert "NOT high quality" in result.text
    assert "av1" in result.text
    details = result.details or {}
    assert details["reason"] == "record_quality_mismatch"
    assert details["why"] == "not_applied"
    # A clip IS recording: the reason must not read as proof that nothing runs,
    # or the stop that would save it is refused.
    assert "record_quality_mismatch" not in builtin._RECORD_ABSENT_REASONS
    assert builtin._APPROVED_RECORDINGS["ui:7:nonce"] == details["target"]


def test_a_high_start_with_no_codec_echo_is_the_same_failed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _call("record_start", tmp_path, monkeypatch, client=_QualityHost(), quality="high")
    assert result.is_error
    assert "unreported" in result.text
    assert (result.details or {})["reason"] == "record_quality_mismatch"


def test_the_clip_a_mismatched_start_left_running_can_still_be_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start, _ = _call(
        "record_start",
        tmp_path,
        monkeypatch,
        client=_QualityHost(),
        quality="high",
        reply={"recording": True, "path": "", "codec": "av1"},
    )
    target = Path((start.details or {})["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(MP4_HEAD)
    stop, seen = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        client=_QualityHost(),
        reply={"path": str(target), "bytes": len(MP4_HEAD), "duration_ms": 10, "codec": "av1"},
    )
    assert not stop.is_error
    assert seen[0][0] == "record_stop"
    assert "AV1" in stop.text


@pytest.mark.parametrize("value", ["ultra", "HIGHEST", "h264", "1", "low"])
def test_an_unknown_quality_is_refused_before_dispatch(value: str) -> None:
    refusal = builtin._browser_args_problem(
        "call", "record_start", _params("record_start", quality=value)
    )
    assert refusal is not None
    details = refusal.details or {}
    assert (details["reason"], details["param"], details["why"]) == (
        "record_quality",
        "quality",
        "value",
    )
    assert "'standard' or 'high'" in refusal.text
    assert refusal.text.endswith("Nothing was started.")


@pytest.mark.parametrize("value", ["high", "standard", None])
def test_a_known_quality_passes_the_argument_check(value: str | None) -> None:
    assert (
        builtin._browser_args_problem(
            "call", "record_start", _params("record_start", quality=value)
        )
        is None
    )


@pytest.mark.parametrize("action", ["record_stop", "screenshot", "snapshot"])
def test_quality_on_any_other_action_is_refused(action: str) -> None:
    problem = builtin._validate_browser_args(action, _params(action, quality="high"))
    assert "only valid for 'record_start'" in problem


@pytest.mark.parametrize("value", [True, 3, ["high"]])
def test_a_non_text_quality_is_typed_through_the_real_entry(
    value: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx(tmp_path))
    assert tool is not None

    async def _run() -> Any:
        return await tool.execute(
            "call", {"action": "record_start", "quality": value}, None, None, _ctx(tmp_path)
        )

    result = asyncio.run(_run())
    assert result.is_error
    details = result.details or {}
    assert (details["reason"], details["why"]) == ("record_quality", "value")
    assert result.text.endswith("Nothing was started.")
    # Plan time answers the same fault without asking for approval.
    assert tool.refuse_args is not None
    hint = tool.refuse_args({"action": "record_start", "quality": value}, frozenset())
    assert hint is not None and hint.endswith("Nothing was started.")


def test_an_unknown_quality_is_refused_at_plan_time_and_for_the_real_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(_ctx(tmp_path))
    assert tool is not None and tool.refuse_args is not None
    args = {"action": "record_start", "quality": "ultra"}
    assert "'standard' or 'high'" in (tool.refuse_args(args, frozenset()) or "")
    # `high` and a size error: the size sentence wins, the order is stable.
    assert "even" in (
        tool.refuse_args(
            {"action": "record_start", "quality": "high", "width": 1921, "height": 1080},
            frozenset(),
        )
        or ""
    )
    assert tool.refuse_args({"action": "record_start", "quality": "high"}, frozenset()) is None

    async def _run() -> Any:
        return await tool.execute("call", args, None, None, _ctx(tmp_path))

    result = asyncio.run(_run())
    assert (result.details or {})["reason"] == "record_quality"


def test_the_approval_row_names_high_quality(tmp_path: Path) -> None:
    row = builtin._describe_browser_approval(
        {
            "action": "record_start",
            "path": str(tmp_path / "clip.mp4"),
            "quality": "high",
            "width": 1920,
            "height": 1080,
        },
        str(tmp_path),
    )
    assert row.endswith(" at 1920x1080 (high quality)")
    plain = builtin._describe_browser_approval(
        {"action": "record_start", "path": str(tmp_path / "clip.mp4")}, str(tmp_path)
    )
    assert "high quality" not in plain
    assert "high quality" not in builtin._describe_browser_approval(
        {"action": "record_start", "quality": {"bad": 1}}, str(tmp_path)
    )


def test_the_two_new_host_refusals_prove_no_recording_is_running() -> None:
    assert "record_quality" in builtin._RECORD_ABSENT_REASONS
    assert "record_quality_unavailable" in builtin._RECORD_ABSENT_REASONS


def test_a_host_value_refusal_renders_the_harness_copy() -> None:
    error = _wire_error(ErrorCode.INTERNAL, "record_quality", param="quality", why="value")
    message = format_error(error, action="record_start", host=HOST_UI)
    assert "'standard' or 'high'" in message
    assert message.endswith("Nothing was started.")


@pytest.mark.parametrize(
    "why, needle, offers_smaller",
    [
        ("no_h264_encoder", "no H.264 encoder", False),
        ("software_over_1080p", "limited to 1920x1080", True),
        ("size_unsupported", "does not support that clip size", True),
        ("encoder_start", "failed to start", False),
    ],
)
def test_every_unavailable_reason_renders_a_way_forward(
    why: str, needle: str, offers_smaller: bool
) -> None:
    error = _wire_error(ErrorCode.INTERNAL, "record_quality_unavailable", why=why)
    message = format_error(error, action="record_start", host=HOST_UI)
    assert needle in message
    # The app never falls back by itself, so the copy must say how to proceed.
    assert "without quality" in message
    assert message.endswith("Nothing was started.")
    assert ("smaller" in message or "1920x1080 or smaller" in message) is offers_smaller


def test_an_unknown_unavailable_why_falls_back_to_the_host_words() -> None:
    error = BridgeError(
        ErrorCode.INTERNAL,
        "the GPU is on fire",
        {"reason": "record_quality_unavailable", "why": "from_the_future"},
    )
    message = format_error(error, action="record_start", host=HOST_UI)
    assert "the GPU is on fire" in message
    assert "without quality" in message


def test_a_refused_high_start_forgets_the_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`record_quality_unavailable` from the host: nothing runs, nothing is kept."""
    seen: list[str] = []

    async def refusing(tool_call_id, action_name, wire, *, surface="", client=None):
        seen.append(action_name)
        error = _wire_error(
            ErrorCode.INTERNAL, "record_quality_unavailable", why="software_over_1080p"
        )
        problem = builtin._error(
            tool_call_id, "browser", format_error(error, action=action_name, host=HOST_UI)
        )
        problem.details = {"reason": "record_quality_unavailable", "error_code": "internal"}
        return None, problem

    monkeypatch.setattr(builtin, "_bridge_call", refusing)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    builtin._APPROVED_RECORDINGS["ui:7:nonce"] = "stale"
    result = asyncio.run(
        builtin._browser_record(
            "call",
            _surface(),
            "record_start",
            _params("record_start", quality="high", width=3840, height=2160),
            _ctx(tmp_path),
            client=_QualityHost(),
        )
    )
    assert result.is_error
    assert seen == ["record_start"]
    assert "ui:7:nonce" not in builtin._APPROVED_RECORDINGS


def test_a_high_stop_passes_the_new_fields_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        client=_QualityHost(),
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 11_000,
            "auto_stopped": False,
            "codec": "h264",
            "quality": "high",
            "encoder": "software",
            "frames": 32,
            "measured_fps": 2.9,
            "avg_bitrate_kbps": 1234.4,
            "stop_reason": "stop",
        },
    )
    assert not result.is_error
    assert result.details == {
        "path": str(target),
        "bytes": len(MP4_HEAD),
        "quality": "high",
        "encoder": "software",
        "frames": 32,
        "measured_fps": 2.9,
        "avg_bitrate_kbps": 1234.4,
        "stop_reason": "stop",
    }
    assert "H.264" in result.text
    assert "32 frames" in result.text
    assert "2.9 fps measured" in result.text
    assert "1234 kbps" in result.text
    assert "AV1" not in result.text


def test_a_clip_that_hit_the_byte_cap_is_kept_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 40_000,
            "auto_stopped": False,
            "codec": "h264",
            "quality": "high",
            "stop_reason": "max_bytes",
        },
    )
    assert not result.is_error
    assert (result.details or {})["stop_reason"] == "max_bytes"
    assert "size limit" in result.text
    assert "complete" in result.text
    assert "time limit" not in result.text


def test_a_max_duration_stop_still_says_the_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 120_000,
            "auto_stopped": True,
            "codec": "h264",
            "quality": "high",
            "stop_reason": "max_duration",
        },
    )
    assert "time limit" in result.text


def test_malformed_stop_fields_are_dropped_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 1,
            "quality": "high",
            "encoder": "quantum",
            "frames": True,
            "measured_fps": float("nan"),
            "avg_bitrate_kbps": "lots",
            "stop_reason": "exploded",
        },
    )
    assert not result.is_error
    assert result.details == {"path": str(target), "bytes": len(MP4_HEAD), "quality": "high"}


_HOSTILE_NUMBERS = [
    float("nan"),
    float("inf"),
    float("-inf"),
    1e999,
    10**400,
    -(10**400),
    -1,
    1.5,
    True,
    False,
    "7",
    None,
]


@pytest.mark.parametrize("field", ["frames"])
@pytest.mark.parametrize("bad", _HOSTILE_NUMBERS, ids=repr)
def test_a_hostile_stop_count_never_escapes_after_the_file_is_written(
    field: str, bad: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is on disk and the approval popped by now: nothing here may raise."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 1,
            "quality": "high",
            field: bad,
        },
    )
    assert not result.is_error
    assert field not in (result.details or {})
    assert result.details["path"] == str(target)  # type: ignore[index]


@pytest.mark.parametrize("field", ["measured_fps", "avg_bitrate_kbps"])
@pytest.mark.parametrize("bad", _HOSTILE_NUMBERS, ids=repr)
def test_a_hostile_stop_rate_never_escapes_after_the_file_is_written(
    field: str, bad: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Includes bool: `True` must not render as "1.0 fps measured"."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 1,
            "quality": "high",
            field: bad,
        },
    )
    assert not result.is_error
    expected_kept = isinstance(bad, (int, float)) and not isinstance(bad, bool)
    kept = field in (result.details or {})
    # Only a finite, non-negative number may be kept (1.5 is a fine rate).
    assert kept == (expected_kept and bad in (1.5,))


@pytest.mark.parametrize("field", ["width", "height", "fps", "bitrate_bps"])
@pytest.mark.parametrize("bad", _HOSTILE_NUMBERS, ids=repr)
def test_a_hostile_start_field_never_escapes_and_is_dropped(
    field: str, bad: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = {**HIGH_START, field: bad}
    result, _ = _call(
        "record_start", tmp_path, monkeypatch, client=_QualityHost(), quality="high", reply=reply
    )
    # A hostile width/height only drops the NUMBERS; the codec echo is what
    # decides success, and it is intact.
    assert not result.is_error
    assert field not in (result.details or {})
    assert "H.264" in result.text


def test_a_default_stop_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The AV1 receipt keeps its sentence and its details, even from a newer host."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(MP4_HEAD)
    result, _ = _call(
        "record_stop",
        tmp_path,
        monkeypatch,
        reply={
            "path": str(target),
            "bytes": len(MP4_HEAD),
            "duration_ms": 1000,
            "codec": "av1",
            "fps": 10,
            "container": "mp4",
        },
    )
    assert result.details == {"path": str(target), "bytes": len(MP4_HEAD)}
    assert "AV1" in result.text
    assert "QuickTime" in result.text
