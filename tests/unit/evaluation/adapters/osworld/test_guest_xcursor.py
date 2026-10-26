"""The guest X-display leak fix: upstream's exact bytes, applied fail-open, always recorded.

WHAT THESE TESTS PROTECT. The released 2.1 guest leaks one X client per
``/screenshot`` (``pyxcursor.Xcursor`` opens a display and never closes it) and
its X session dies at ~480-500 leaked clients -- the measured cause of the
``bridge-wedged`` terminal that ended 42% of sealed 2.1 episodes. The fix is
upstream OSWorld PR #500's ``pyxcursor.py``, written onto the guest before the
episode. ``guest_xcursor`` carries the live measurements.

Three things are pinned here:

1. the BYTES -- the vendored file is upstream's at the merge commit, and the
   stock digest is the one read back from a bare guest;
2. the BEHAVIOUR -- both files are executed against a fake libX11 and the
   stock handler's per-request pattern leaks one display per call while the
   patched one leaks none (this is the test that fails without the fix);
3. the DISCLOSURE -- every outcome, including every failure, produces a
   record with ``applied`` true or false, and nothing raises.
"""

from __future__ import annotations

import base64
import ctypes
import gc
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest
from lop_osworld_v2_adapter import guest_xcursor
from lop_osworld_v2_adapter.guest_disk import CommandResult

_PKG = Path(guest_xcursor.__file__).parent

#: The stock file is upstream's pre-fix file (parent ef43297b of the fix) plus
#: one trailing newline, as read back from the 2.1 AMI. Rebuilt here from the
#: vendored patched file by REMOVING exactly upstream's PR #500 additions, so
#: the test does not need a second vendored copy and proves the diff is only
#: that PR's lines.
_PR500_ADDED_BLOCKS = (
    "        # Set up XCloseDisplay function signature for proper cleanup\n"
    "        self.xlib.XCloseDisplay.argtypes = [ctypes.POINTER(Display)]\n"
    "        self.xlib.XCloseDisplay.restype = ctypes.c_int\n"
    "\n",
    "    def close(self):\n"
    '        """Close the X11 display connection to prevent resource leaks."""\n'
    "        if self.display:\n"
    "            self.xlib.XCloseDisplay(self.display)\n"
    "            self.display = None\n"
    "\n"
    "    def __del__(self):\n"
    '        """Safety net to ensure display connection is closed on garbage collection."""\n'
    "        self.close()\n"
    "\n"
    "    def __enter__(self):\n"
    '        """Context manager entry."""\n'
    "        return self\n"
    "\n"
    "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
    '        """Context manager exit - ensures proper cleanup."""\n'
    "        self.close()\n"
    "        return False\n"
    "\n",
)


def _stock_bytes() -> bytes:
    text = guest_xcursor.patched_source().decode("utf-8")
    for block in _PR500_ADDED_BLOCKS:
        assert text.count(block) == 1, "vendored file no longer carries PR #500's block"
        text = text.replace(block, "")
    return (text + "\n").encode("utf-8")


# ----------------------------------------------------------------------
# 1. The bytes
# ----------------------------------------------------------------------


def test_the_vendored_file_is_upstreams_bytes_at_the_merge_commit() -> None:
    data = (_PKG / guest_xcursor.VENDORED_RESOURCE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == guest_xcursor.PATCHED_SHA256
    assert guest_xcursor.patched_source() == data
    assert guest_xcursor.UPSTREAM_MERGE_SHA == "8253399dfa8b3c1d5783a73c29a91f11f7c6d7c3"
    assert guest_xcursor.UPSTREAM_PR == 500


def test_the_stock_digest_is_the_pre_fix_file_and_only_pr_500_differs() -> None:
    """The measured stock digest equals the vendored file minus PR #500's additions.

    If this fails, either the vendored file is not upstream's fix or the
    pinned stock digest is not the file the image ships -- and the gate that
    only patches a recognised stock file would be gating on the wrong thing.
    """

    assert hashlib.sha256(_stock_bytes()).hexdigest() == guest_xcursor.STOCK_SHA256


def test_a_tampered_vendored_file_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Res:
        def joinpath(self, _name: str) -> "_Res":
            return self

        def read_bytes(self) -> bytes:
            return b"tampered"

    monkeypatch.setattr(guest_xcursor.resources, "files", lambda _pkg: _Res())
    with pytest.raises(ValueError, match="digest"):
        guest_xcursor.patched_source()


# ----------------------------------------------------------------------
# 2. The behaviour: execute both files against a fake libX11
# ----------------------------------------------------------------------


class _FakeX:
    """Counts open displays the way the X server counts clients."""

    def __init__(self) -> None:
        self.open = 0
        self._handles: list[Any] = []

    def XOpenDisplay(self, _name: Any) -> Any:  # noqa: N802 - libX11's name
        self.open += 1
        handle = ctypes.pointer(ctypes.c_int(len(self._handles) + 1))
        self._handles.append(handle)
        return handle

    def XCloseDisplay(self, _display: Any) -> int:  # noqa: N802 - libX11's name
        self.open -= 1
        return 0

    def XFixesGetCursorImage(self, _display: Any) -> Any:  # noqa: N802 - libXfixes' name
        return None


class _Fn:
    """A ctypes-function stand-in: callable, and accepts argtypes/restype."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        return self._fn(*args)


class _Lib:
    def __init__(self, x: _FakeX) -> None:
        for name in ("XOpenDisplay", "XCloseDisplay", "XFixesGetCursorImage"):
            setattr(self, name, _Fn(getattr(x, name)))


def _load(source: bytes, x: _FakeX, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Execute a guest ``pyxcursor.py`` with libX11/libXfixes replaced by ``x``."""

    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    monkeypatch.setenv("DISPLAY", ":0")
    module = types.ModuleType("guest_pyxcursor")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: name)
    monkeypatch.setattr(ctypes.cdll, "LoadLibrary", lambda _name: _Lib(x))
    exec(compile(source, "pyxcursor.py", "exec"), module.__dict__)  # noqa: S102
    return module


def _stock_handler(module: types.ModuleType) -> None:
    """The stock ``/screenshot`` handler's use of Xcursor, verbatim in shape.

    main.py on the AMI (line 321): ``cursor_obj = Xcursor()`` then
    ``getCursorImageArrayFast()``; the local goes out of scope on return. The
    cursor read is elided (it needs real pixels); the display lifetime is the
    thing under test.
    """

    cursor_obj = module.Xcursor()  # noqa: F841 - the local's lifetime IS the test


@pytest.mark.parametrize("calls", [1, 50])
def test_the_stock_file_leaks_one_display_per_screenshot(
    calls: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reproduction: the file the image ships never closes a display."""

    x = _FakeX()
    module = _load(_stock_bytes(), x, monkeypatch)
    for _ in range(calls):
        _stock_handler(module)
    gc.collect()
    assert x.open == calls


@pytest.mark.parametrize("calls", [1, 50, 600])
def test_the_upstream_fix_closes_every_display_with_the_stock_handler(
    calls: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix: the SAME stock handler, unchanged, leaks nothing with the new file.

    This is why main.py is not patched: its local ``cursor_obj`` is collected
    on return and the new ``__del__`` closes the display. 600 is past the
    ~480-500 point where the live guest's X session died.
    """

    x = _FakeX()
    module = _load(guest_xcursor.patched_source(), x, monkeypatch)
    for _ in range(calls):
        _stock_handler(module)
    gc.collect()
    assert x.open == 0


# ----------------------------------------------------------------------
# 3. The application: a fake guest with a reloading server
# ----------------------------------------------------------------------


class _Guest:
    """The guest's control server, with werkzeug's reloader simulated.

    ``file`` is ``/home/user/server/pyxcursor.py``; a write that changes it
    replaces the serving child's PID after ``reload_after`` inspections, the way
    the stat reloader re-execs the child.
    """

    def __init__(
        self,
        *,
        file: bytes | None,
        pids: tuple[int, ...] = (1580, 1733),
        reload_after: int | None = 1,
        fail: dict[str, Exception] | None = None,
        write_rc: int = 0,
    ) -> None:
        self.file = file
        self.pids = pids
        self.reload_after = reload_after
        self.fail = fail or {}
        self.write_rc = write_rc
        self.commands: list[list[str]] = []
        self._pending: int | None = None

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        argv = list(argv)
        self.commands.append(argv)
        source = argv[2]
        if source == guest_xcursor._WRITE_SOURCE:
            if "write" in self.fail:
                raise self.fail["write"]
            if self.write_rc:
                return CommandResult(returncode=self.write_rc, stdout="", stderr="denied")
            path, expected, payload = argv[3], argv[4], base64.b64decode(argv[5])
            assert path == guest_xcursor.GUEST_PATH
            current = hashlib.sha256(self.file or b"").hexdigest()
            if current != expected:
                return CommandResult(0, json.dumps({"written": False, "sha256": current}), "")
            self.file = payload
            self._pending = self.reload_after
            return CommandResult(
                0, json.dumps({"written": True, "sha256": hashlib.sha256(payload).hexdigest()}), ""
            )
        assert source == guest_xcursor._INSPECT_SOURCE
        if "inspect" in self.fail:
            raise self.fail["inspect"]
        if self._pending is not None:
            if self._pending <= 0:
                self.pids = (self.pids[0], self.pids[-1] + 100)
                self._pending = None
            else:
                self._pending -= 1
        digest = hashlib.sha256(self.file).hexdigest() if self.file is not None else None
        return CommandResult(
            0,
            "escalation noise\n" + json.dumps({"sha256": digest, "server_pids": list(self.pids)}),
            "",
        )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _apply(guest: _Guest) -> guest_xcursor.PatchReport:
    clock = _Clock()
    return guest_xcursor.apply_xcursor_patch(guest.run, clock=clock, sleep=clock.sleep)


def test_a_stock_guest_is_patched_and_the_reload_is_proved() -> None:
    guest = _Guest(file=_stock_bytes())
    report = _apply(guest)

    assert report.status == "applied" and report.applied is True
    assert guest.file == guest_xcursor.patched_source()
    assert report.sha256_before == guest_xcursor.STOCK_SHA256
    assert report.sha256_after == guest_xcursor.PATCHED_SHA256
    # A NEW serving pid is the proof the server imported the new module.
    assert set(report.server_pids_after) - set(report.server_pids_before)
    record = report.to_json()
    assert record["applied"] is True
    assert record["upstream"] == {
        "repository": "xlang-ai/OSWorld",
        "pr": 500,
        "merge_sha": "8253399dfa8b3c1d5783a73c29a91f11f7c6d7c3",
        "path": "desktop_env/server/pyxcursor.py",
    }
    assert record["patched_sha256"] == guest_xcursor.PATCHED_SHA256


def test_every_guest_command_is_an_unprivileged_argv_with_no_shell() -> None:
    """The file is owned by the server's user, so no sudo and no password."""

    guest = _Guest(file=_stock_bytes())
    _apply(guest)
    for argv in guest.commands:
        assert argv[:2] == ["python3", "-c"]
        assert not any("sudo" in part for part in argv)


def test_an_already_patched_guest_is_left_alone_and_reported_applied() -> None:
    guest = _Guest(file=guest_xcursor.patched_source())
    report = _apply(guest)
    assert report.status == "already-patched" and report.applied is True
    assert all(argv[2] == guest_xcursor._INSPECT_SOURCE for argv in guest.commands)


def test_an_unrecognised_file_is_never_overwritten() -> None:
    guest = _Guest(file=b"some other image's cursor helper\n")
    report = _apply(guest)
    assert report.status == "unrecognised" and report.applied is False
    assert guest.file == b"some other image's cursor helper\n"
    assert all(argv[2] == guest_xcursor._INSPECT_SOURCE for argv in guest.commands)


def test_a_missing_file_is_unrecognised_not_created() -> None:
    guest = _Guest(file=None)
    report = _apply(guest)
    assert report.status == "unrecognised" and report.applied is False
    assert guest.file is None


@pytest.mark.parametrize(
    ("guest", "status", "reason_fragment"),
    [
        (
            _Guest(file=b"", fail={"inspect": ConnectionError("http://10.0.0.1:5000")}),
            "unreachable",
            "ConnectionError",
        ),
        (_Guest(file=None, pids=()), "unrecognised", "stock"),
        (_Guest(file=_stock_bytes(), pids=()), "failed", "no running action server"),
        (_Guest(file=_stock_bytes(), fail={"write": TimeoutError()}), "failed", "not replaced"),
        (_Guest(file=_stock_bytes(), write_rc=1), "failed", "not replaced"),
        (_Guest(file=_stock_bytes(), reload_after=None), "failed", "did not reload"),
    ],
)
def test_every_failure_is_recorded_as_not_applied_and_never_raises(
    guest: _Guest, status: str, reason_fragment: str
) -> None:
    report = _apply(guest)
    assert report.status == status
    assert report.applied is False
    assert reason_fragment in report.reason
    record = report.to_json()
    assert record["applied"] is False
    # Transport exception TEXT can carry the guest address; only its type is kept.
    assert "10.0.0.1" not in json.dumps(record)


def test_a_reload_that_never_happens_is_bounded_by_the_wait() -> None:
    guest = _Guest(file=_stock_bytes(), reload_after=None)
    clock = _Clock()
    guest_xcursor.apply_xcursor_patch(guest.run, clock=clock, sleep=clock.sleep)
    assert clock.now <= guest_xcursor.RELOAD_WAIT_S + guest_xcursor.RELOAD_POLL_S
