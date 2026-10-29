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
import re
from pathlib import Path
from typing import Any

import pytest

import local_operator.tools.builtin as builtin
from local_operator import browser_files, scratchpad
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
            client=_AdvertisedHost(),
        )
    )
    return result, seen


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


@pytest.mark.parametrize("value", [0, -1, 0.5, 601, 5000])
def test_validation_bounds_the_recording_length(value: float) -> None:
    problem = builtin._validate_browser_args("record_start", _params(timeout_s=value))
    assert problem
    assert "600" in problem


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
            "width": 1280,
            "height": 720,
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


# ---------------------------------------------------------------------------
# Error rendering (D9): Python owns the copy, keyed by `data.reason`
# ---------------------------------------------------------------------------


def _wire_error(code: ErrorCode, reason: str, **data: Any) -> BridgeError:
    return BridgeError(code, "host text", {"reason": reason, **data})


@pytest.mark.parametrize(
    "reason, needle, extra",
    [
        ("record_unsupported", "Chromium 153", {}),
        ("record_av1_unavailable", "AV1", {}),
        ("record_path", "path", {}),
        ("record_active", "already", {}),
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


def test_unsupported_names_the_chromium_version_the_host_reported() -> None:
    message = format_error(
        _wire_error(ErrorCode.INTERNAL, "record_unsupported", chromium="152.0.7977.78"),
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
