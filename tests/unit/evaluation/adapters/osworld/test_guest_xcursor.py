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

#: The stock file is upstream's pre-fix file (the fix merge's parent 36a6adf4) plus
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


def test_a_tampered_vendored_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / guest_xcursor.VENDORED_RESOURCE).write_bytes(b"tampered")
    monkeypatch.setattr(guest_xcursor, "__file__", str(tmp_path / "guest_xcursor.py"))
    with pytest.raises(ValueError, match="digest"):
        guest_xcursor.patched_source()


def test_the_vendored_file_loads_under_the_workers_verified_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker imports adapter modules through discovery's verified loader.

    That loader has no resource reader, so ``importlib.resources`` raised
    ``Can't open orphan path`` there and the first validation episode recorded
    ``applied: false``. Execute this module through the SAME loader class and
    read the vendored file from it.
    """

    import importlib.util

    from local_operator.evaluation.adapters import discovery

    path = Path(guest_xcursor.__file__)
    # The PACKAGE is what the worker's finder serves through the verified
    # loader, and ``importlib.resources`` resolves resources from the package's
    # spec -- so the package must be the verified one too, or this test would
    # pass against the very code that failed in the worker.
    init = path.parent / "__init__.py"
    package = types.ModuleType("lop_osworld_v2_adapter")
    package.__spec__ = importlib.util.spec_from_file_location(
        "lop_osworld_v2_adapter",
        init,
        loader=discovery._VerifiedSourceLoader(init, init.read_bytes()),
        submodule_search_locations=[str(path.parent)],
    )
    package.__path__ = [str(path.parent)]
    package.__file__ = str(init)
    monkeypatch.setitem(sys.modules, "lop_osworld_v2_adapter", package)
    loader = discovery._VerifiedSourceLoader(path, path.read_bytes())
    spec = importlib.util.spec_from_file_location(
        "lop_osworld_v2_adapter.guest_xcursor_verified", path, loader=loader
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "lop_osworld_v2_adapter"
    # A dataclass resolves its module through sys.modules while executing.
    monkeypatch.setitem(sys.modules, spec.name, module)
    loader.exec_module(module)
    assert hashlib.sha256(module.patched_source()).hexdigest() == guest_xcursor.PATCHED_SHA256


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
        # The write never reached the guest (the transport raised before
        # delivery here), and a later inspection OBSERVED the stock digest:
        # only that observation licenses "stock".
        (_Guest(file=_stock_bytes(), fail={"write": TimeoutError()}), "failed", "stock digest"),
        (_Guest(file=_stock_bytes(), write_rc=1), "failed", "stock digest"),
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


class _LostResponseGuest(_Guest):
    """The reviewer's M1 case: the write LANDS, then nothing answers.

    ``/execute`` is served by the very werkzeug child the write makes reload,
    so the write's response and every later inspect can be lost while the
    patched file sits on disk. ``lose_write_response`` makes the write raise
    AFTER it has replaced the file, the way a dropped response looks.
    """

    def __init__(self, *, lose_write_response: bool) -> None:
        super().__init__(file=_stock_bytes(), reload_after=None)
        self.lose_write_response = lose_write_response
        self.writes = 0

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        if argv[2] == guest_xcursor._INSPECT_SOURCE and self.writes:
            raise ConnectionError("server reloading")
        result = super().run(argv, timeout)
        if argv[2] == guest_xcursor._WRITE_SOURCE:
            self.writes += 1
            if self.lose_write_response:
                raise ConnectionError("response lost")
        return result


@pytest.mark.parametrize("lose_write_response", [False, True])
def test_a_write_whose_effect_is_never_observed_is_unconfirmed_never_stock(
    lose_write_response: bool,
) -> None:
    guest = _LostResponseGuest(lose_write_response=lose_write_response)
    report = _apply(guest)

    # The guest really does carry the patched file now.
    assert guest.file == guest_xcursor.patched_source()
    assert report.status == "unconfirmed"
    assert report.applied is None
    record = report.to_json()
    assert record["applied"] is None
    assert record["write_attempted"] is True
    # Nothing was observed after the write, and the record says exactly that.
    assert record["sha256_after"] is None and record["server_pids_after"] is None
    # The record must not read as the stock guest: the reason names both.
    assert "stock or the patched" in report.reason
    assert "not replaced" not in report.reason


def test_a_patched_file_with_no_observed_reload_is_unconfirmed() -> None:
    """Patched on disk, no new PID within the wait: the reloader (or any later
    restart) can still import it, so this is neither applied nor stock."""

    guest = _Guest(file=_stock_bytes(), reload_after=None)
    report = _apply(guest)
    assert report.status == "unconfirmed" and report.applied is None
    assert report.sha256_after == guest_xcursor.PATCHED_SHA256
    assert "no reload was observed" in report.reason


def test_a_guest_side_refusal_is_stock_because_nothing_was_replaced() -> None:
    """The guest's own pre-write check saw a changed file and wrote nothing."""

    class _Raced(_Guest):
        def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
            if argv[2] == guest_xcursor._WRITE_SOURCE:
                self.file = b"changed underneath us\n"
            return super().run(argv, timeout)

    guest = _Raced(file=_stock_bytes())
    report = _apply(guest)
    assert report.status == "failed" and report.applied is False
    assert guest.file == b"changed underneath us\n"
    assert "left unchanged" in report.reason


def test_records_that_never_inspected_after_say_so_rather_than_server_gone() -> None:
    """QA note B: no post-write inspection -> ``None``, never an empty list."""

    for guest in (_Guest(file=b"other image\n"), _Guest(file=guest_xcursor.patched_source())):
        record = _apply(guest).to_json()
        assert record["write_attempted"] is False
        assert record["server_pids_after"] is None
        assert record["sha256_after"] is None
        assert record["server_pids_before"] == [1580, 1733]


def test_the_guest_write_script_removes_its_temp_file_when_the_write_fails(
    tmp_path: Path,
) -> None:
    """n3: a failed write must not leave ``pyxcursor.py.lop-tmp`` behind."""

    import subprocess

    target = tmp_path / "pyxcursor.py"
    target.write_bytes(_stock_bytes())
    target.chmod(0o664)
    # A payload that is not base64 fails inside the guest script before the
    # temp file exists; a read-only directory fails at the temp-file create.
    # Force a failure AFTER the temp file is created instead: chmod to a mode
    # the replace cannot keep by making the target vanish mid-way is not
    # portable, so make ``os.replace`` fail by turning the target into a
    # directory-shaped conflict.
    conflict = tmp_path / "conflict"
    conflict.mkdir()
    (conflict / "child").write_text("x")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            guest_xcursor._WRITE_SOURCE.replace(
                "os.replace(tmp, path)", "os.replace(tmp, sys.argv[4])"
            ),
            str(target),
            guest_xcursor.STOCK_SHA256,
            base64.b64encode(guest_xcursor.patched_source()).decode(),
            str(conflict),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "pyxcursor.py.lop-tmp").exists()
    assert target.read_bytes() == _stock_bytes()


def test_the_guest_write_script_replaces_and_leaves_no_temp_file(tmp_path: Path) -> None:
    import subprocess

    target = tmp_path / "pyxcursor.py"
    target.write_bytes(_stock_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            guest_xcursor._WRITE_SOURCE,
            str(target),
            guest_xcursor.STOCK_SHA256,
            base64.b64encode(guest_xcursor.patched_source()).decode(),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["written"] is True
    assert target.read_bytes() == guest_xcursor.patched_source()
    assert not (tmp_path / "pyxcursor.py.lop-tmp").exists()


def test_a_reload_that_never_happens_is_bounded_by_the_wait() -> None:
    guest = _Guest(file=_stock_bytes(), reload_after=None)
    clock = _Clock()
    guest_xcursor.apply_xcursor_patch(guest.run, clock=clock, sleep=clock.sleep)
    assert clock.now <= guest_xcursor.RELOAD_WAIT_S + guest_xcursor.RELOAD_POLL_S
