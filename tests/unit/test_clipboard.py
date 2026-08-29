"""Reading an image off the system clipboard, on every platform.

The bug these guard (issue #372) was a PLATFORM ASSUMPTION, not a logic error:
the composer only ever saw a clipboard image because cmux wrote it to a temp
file and pasted the path, so a design that worked in exactly one terminal
looked complete. So the shape of this file matters as much as its coverage —
every backend is driven through the same dispatch with an injected
``platform``/``env``, on any host, and none of them is only exercised on the
machine that happens to be running the suite.

Real execution of the backends against real clipboard tooling is separate
evidence and lives in the PR: macOS was driven against this Mac's pasteboard,
and the X11 and Wayland backends against ``xclip``/``wl-paste`` in Docker. What
is HERE is the branch coverage those runs cannot give cheaply — a missing
binary, a timeout, an oversized payload, the SSH refusal.

Windows is unit-tested only. There is no Windows host in this project's
environment, so the PowerShell backend's *invocation* is pinned here and its
command has been reviewed, but it has never been executed on real hardware.
That is stated plainly rather than implied away.
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import time
import zlib
from pathlib import Path

import pytest

from local_operator import clipboard as clipboard_module
from local_operator.clipboard import (
    CLIPBOARD_TIMEOUT_S,
    MAX_CLIPBOARD_TEXT_BYTES,
    ClipboardImage,
    clipboard_reads_are_local,
    read_clipboard,
)


def _png(width: int = 4, height: int = 3) -> bytes:
    """A real, sniffable PNG.

    Built rather than faked with a bare magic number, because the module now
    SNIFFS every payload before returning it — see
    ``test_x11_hands_back_text_as_an_image``, which is a bug real ``xclip``
    exhibited and a header-only fixture would have hidden.
    """
    rows = b"".join(b"\x00" + bytes([10, 20, 30] * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(rows))
        + _chunk(b"IEND", b"")
    )


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _jpeg(width: int = 4, height: int = 3) -> bytes:
    """A real JPEG, encoded by Pillow.

    Hand-rolling one is not worth it: the sniffer walks to the SOF marker for
    the dimensions, so a fixture of magic bytes plus padding is not an image
    and (correctly) gets refused.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, "JPEG")
    return buffer.getvalue()


PNG = _png()
JPEG = _jpeg()
BIG = 4 * 1024 * 1024


class FakeProcess:
    """The subset of ``Popen`` that :func:`clipboard._run` actually drives.

    A ``Popen`` stand-in rather than a ``subprocess.run`` one because the read
    is bounded by ``stdout.read(limit)`` — the module stopped using ``run``
    precisely so a hostile clipboard owner cannot make it buffer 300 MB before
    the ceiling is consulted (review round 1, F3). Faking ``run`` would test a
    code path that no longer exists.

    ``stdout.read(n)`` honours ``n``, so a test can assert the bound is applied
    to the READ and not merely to the result.
    """

    def __init__(self, payload: bytes | None, error: BaseException | None = None) -> None:
        # `_run` remembers the pgid at spawn (it equals the pid, since the child
        # is its own group leader) so the group stays killable after the leader
        # exits — see `_kill_tree` and the F4 regression test.
        self.pid = 424242
        self._payload = b"" if payload is None else payload
        self._error = error
        self.returncode = 1 if payload is None else 0
        self.stdout = io.BytesIO(self._payload)
        self.stdin = io.BytesIO()
        self.killed = False

    def wait(self, timeout=None):  # noqa: ANN001, ANN201
        if self._error is not None:
            raise self._error
        return self.returncode

    def poll(self):  # noqa: ANN201
        # Reports "already exited" unless a test staged a hang, so `_run` only
        # reaches its kill path where a real one would.
        return None if self._error is not None and not self.killed else self.returncode

    def kill(self) -> None:
        self.killed = True

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        return None


class FakeRun:
    """Stands in for ``subprocess.Popen`` and records the argv it was given.

    Keyed on the FIRST argument (the binary), so a test says what ``xclip``
    returns without also having to model ``wl-paste``. A binary with no answer
    staged exits non-zero, which is how every one of these tools reports "there
    is nothing of that type on the clipboard".
    """

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN204
        self.calls.append(list(argv))
        answer = self.answers.get(Path(argv[0]).name)
        if callable(answer):
            answer = answer(list(argv), kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return FakeProcess(answer if answer is None or isinstance(answer, bytes) else None)


@pytest.fixture
def which_all(monkeypatch):
    """Every binary this module looks for is present."""
    monkeypatch.setattr(clipboard_module.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def which_none(monkeypatch):
    """No clipboard tooling installed at all."""
    monkeypatch.setattr(clipboard_module.shutil, "which", lambda name: None)


def _install(monkeypatch, answers: dict[str, object]) -> FakeRun:
    fake = FakeRun(answers)
    monkeypatch.setattr(clipboard_module.subprocess, "Popen", fake)
    return fake


# -- the SSH refusal ----------------------------------------------------------
@pytest.mark.parametrize("var", ["SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"])
def test_a_remote_session_never_reads_a_clipboard(var: str) -> None:
    """The clipboard on the far end is the SERVER's, and attaching it to a
    prompt the user typed on their laptop would send something they never
    chose to send. That is a confidentiality failure, not a wrong result, so
    each variable disqualifies the read on its own — different sshd
    configurations set different subsets.
    """
    assert clipboard_reads_are_local({var: "value"}) is False
    assert clipboard_reads_are_local({}) is True


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_the_ssh_skip_applies_to_every_backend(platform: str, monkeypatch, which_all) -> None:
    """Uniform across platforms. A skip implemented per-backend is one someone
    adding the fourth backend forgets, so it is enforced at the dispatch."""
    fake = _install(monkeypatch, {"osascript": PNG, "xclip": PNG, "pwsh": PNG})
    env = {"SSH_TTY": "/dev/ttys001", "DISPLAY": ":0"}
    assert read_clipboard(BIG, platform=platform, env=env).image is None
    assert list(read_clipboard(platform=platform, env=env).paths) == []
    assert fake.calls == [], "a remote session must not even spawn the reader"


# -- macOS --------------------------------------------------------------------
def test_macos_reads_the_png_flavor(monkeypatch, which_all, tmp_path) -> None:
    """The reported case: a native screenshot leaves PNG bytes and no text."""

    def osascript(argv, kwargs):
        # The script writes to the destination path passed as its argv and
        # prints the verdict, which is what the backend then reads back.
        Path(argv[2]).write_bytes(PNG)
        return b"image"

    _install(monkeypatch, {"osascript": osascript})
    image = read_clipboard(BIG, platform="darwin", env={}).image
    assert image == ClipboardImage(PNG, "image/png")


def test_macos_reads_the_script_from_stdin_not_the_argv(monkeypatch, which_all) -> None:
    """The AppleScript goes in over stdin, so it never appears in ``ps`` and
    the destination path is never spliced into source text."""

    def osascript(argv, kwargs):
        assert argv[1] == "-", "the script must come from stdin"
        assert kwargs["stdin"] is subprocess.PIPE, "the script must not ride in argv"
        Path(argv[2]).write_bytes(PNG)
        return b"image"

    _install(monkeypatch, {"osascript": osascript})
    assert read_clipboard(BIG, platform="darwin", env={}).image is not None
    # The script itself carries both flavours; asserted on the source rather
    # than on the pipe write, which the fake swallows.
    assert "NSPasteboard" in clipboard_module._MACOS_CLIPBOARD_SCRIPT
    assert "public.tiff" in clipboard_module._MACOS_CLIPBOARD_SCRIPT


def test_macos_with_an_empty_pasteboard_is_no_image(monkeypatch, which_all) -> None:
    """The script writes no file and reports "none"; the backend must not then
    hand back a zero-byte payload that sniffs as nothing."""
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b"none"})
    assert read_clipboard(BIG, platform="darwin", env={}).image is None


def test_the_sniffed_type_is_what_comes_back_not_the_requested_one(monkeypatch, which_all) -> None:
    """The result names what the bytes ARE. A backend that reported its own
    request would have laundered the ``xclip`` text case above into a
    confident, wrong ``image/png``."""

    def wl_paste(argv, kwargs):
        # Offers PNG, delivers JPEG. Nothing stops a clipboard owner doing this.
        return b"image/png\n" if "--list-types" in argv else JPEG

    _install(monkeypatch, {"wl-paste": wl_paste})
    image = read_clipboard(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"}).image
    assert image is not None and image.mime_type == "image/jpeg"


def test_macos_file_urls_come_back_as_paths(monkeypatch, which_all) -> None:
    """Finder's Cmd+C puts only ``public.file-url`` on the pasteboard — no
    text, no image bytes — so this is the only route to what was copied."""
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/a.png\x00/tmp/b c.png\x00"})
    assert list(read_clipboard(platform="darwin", env={}).paths) == ["/tmp/a.png", "/tmp/b c.png"]


def test_a_copied_filename_containing_a_newline_survives(monkeypatch, which_all) -> None:
    """A newline is legal in a macOS filename, and splitting the script's
    output on newlines turned one real path into two nonexistent ones — which
    the composer's all-or-nothing rule then refused, so a Finder copy of such a
    file silently attached nothing (review round 1, F5). NUL cannot occur in a
    path, so it is the one separator that cannot be ambiguous.
    """
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/we\nird.png\x00"})
    assert list(read_clipboard(platform="darwin", env={}).paths) == ["/tmp/we\nird.png"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_file_urls_are_a_macos_only_route(platform: str, monkeypatch, which_all) -> None:
    """Linux file managers put ``text/uri-list`` on the clipboard next to plain
    text of the same paths, so the terminal already bracket-pastes them and the
    composer's path branch already handles it. A second route there would be a
    way for one copy to be attached twice.
    """
    fake = _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/a.png\n"})
    assert list(read_clipboard(platform=platform, env={}).paths) == []
    assert all(call[0] != "osascript" for call in fake.calls)


# -- X11 ----------------------------------------------------------------------
def test_x11_reads_the_png_target(monkeypatch, which_all) -> None:
    _install(monkeypatch, {"xclip": PNG})
    image = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image
    assert image == ClipboardImage(PNG, "image/png")


def test_x11_asks_for_the_clipboard_selection_not_the_primary(monkeypatch, which_all) -> None:
    """``PRIMARY`` is the middle-click selection and holds whatever text was
    last highlighted; ``Cmd+V``/``Ctrl+V`` pastes ``CLIPBOARD``."""
    fake = _install(monkeypatch, {"xclip": PNG})
    read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image
    assert fake.calls[0] == ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]


def test_x11_falls_through_to_the_next_type(monkeypatch, which_all) -> None:
    """A browser copy is often JPEG or WebP and never PNG."""
    _install(
        monkeypatch,
        {"xclip": lambda argv, kwargs: JPEG if "image/jpeg" in argv else None},
    )
    image = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image
    assert image is not None and image.mime_type == "image/jpeg"


def test_x11_never_hands_back_clipboard_text_as_an_image(monkeypatch, which_all) -> None:
    """A REAL bug, found by running the backend against real ``xclip`` under
    Xvfb and invisible to any mock that answers what it was asked for.

    ``xclip -selection clipboard -t image/png -o`` on a clipboard holding plain
    text exits ZERO and prints the text: the target is advisory, and the
    selection owner answers with whatever it has. The first implementation
    trusted the request and returned
    ``ClipboardImage(b'just text, no image', 'image/png')``, which would have
    reached a provider as a corrupt image block.

    Fixed by sniffing the bytes rather than the request, which is why the MIME
    type on the result is the sniffed one.
    """
    _install(monkeypatch, {"xclip": b"just text, no image"})
    assert read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image is None


def test_an_unsendable_format_on_the_clipboard_is_refused(monkeypatch, which_all) -> None:
    """HEIC sniffs fine and no provider accepts it. "No image" is a better
    answer than an attachment that earns a 400 mid-turn."""
    heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
    _install(monkeypatch, {"xclip": heic})
    assert read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image is None


def test_x11_without_xclip_installed_is_silent(monkeypatch, which_none) -> None:
    """A missing ``xclip`` is not an error the user should see on every stray
    empty paste — it is simply a machine with no clipboard images."""
    fake = _install(monkeypatch, {})
    assert read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image is None
    assert fake.calls == []


# -- Wayland ------------------------------------------------------------------
def test_wayland_lists_types_before_reading_one(monkeypatch, which_all) -> None:
    """``wl-paste --type image/png`` against a clipboard with no PNG both fails
    and, on some compositors, blocks while the offer is negotiated. One cheap
    listing replaces four speculative reads."""

    def wl_paste(argv, kwargs):
        if "--list-types" in argv:
            return b"text/plain\nimage/png\n"
        if "image/png" in argv:
            return PNG
        return None

    fake = _install(monkeypatch, {"wl-paste": wl_paste})
    image = read_clipboard(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"}).image
    assert image == ClipboardImage(PNG, "image/png")
    assert fake.calls[0][1] == "--list-types"


def test_wayland_with_no_image_type_on_offer_reads_no_image(monkeypatch, which_all) -> None:
    """No image type offered means no image, and no image read attempted.

    The listing offers ``text/plain``, so exactly one FURTHER call is expected
    and it is the text read that ``Ctrl+V`` needs \u2014 never a speculative
    ``--type image/png`` for a type the compositor did not list. That
    anti-speculation rule is the point of querying the listing at all, and it
    is pinned on its own below for a clipboard offering nothing readable.
    """
    fake = _install(monkeypatch, {"wl-paste": lambda argv, kwargs: b"text/plain\ntext/html\n"})
    contents = read_clipboard(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"})
    assert contents.image is None
    assert not any(
        "image/" in " ".join(call) for call in fake.calls
    ), "no image type was offered, so none may be requested"
    assert [call for call in fake.calls[1:]] == [
        ["wl-paste", "--no-newline", "--type", "text/plain"]
    ], "the only follow-up read is for a type the compositor actually listed"


def test_wayland_offering_nothing_readable_makes_exactly_one_call(monkeypatch, which_all) -> None:
    """The whole reason the listing is queried first: a type that is not on
    offer is never asked for, because ``wl-paste --type X`` on a clipboard with
    no X both fails and, on some compositors, blocks while the offer is
    negotiated. With neither an image nor a text type listed, the listing is
    the only spawn.
    """
    fake = _install(monkeypatch, {"wl-paste": lambda argv, kwargs: b"application/x-thing\n"})
    contents = read_clipboard(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"})
    assert contents.image is None and contents.text == ""
    assert len(fake.calls) == 1, "an unlisted type must never be read speculatively"


def test_wayland_is_preferred_when_xwayland_also_sets_display(monkeypatch, which_all) -> None:
    """A Wayland session usually runs XWayland too, so ``DISPLAY`` is set in
    both. Testing ``DISPLAY`` first would route a Wayland session to ``xclip``
    and read XWayland's separate, usually empty selection."""

    def wl_paste(argv, kwargs):
        return b"image/png\n" if "--list-types" in argv else PNG

    fake = _install(monkeypatch, {"wl-paste": wl_paste, "xclip": b"WRONG-SELECTION"})
    image = read_clipboard(
        BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    ).image
    assert image is not None and image.data == PNG
    assert all(call[0] != "xclip" for call in fake.calls)


def test_a_headless_linux_box_spawns_nothing(monkeypatch, which_all) -> None:
    """No display server means no clipboard, and discovering that by shelling
    out would cost a subprocess on every empty paste."""
    fake = _install(monkeypatch, {"xclip": PNG, "wl-paste": PNG})
    assert read_clipboard(BIG, platform="linux", env={}).image is None
    assert fake.calls == []


# -- Windows ------------------------------------------------------------------
# NOTE: mocked invocations only. No Windows host exists in this project's
# development or CI environment, so these pin the command's SHAPE — the pieces
# that are wrong in the obvious implementation — and not its behaviour on
# Windows.
def test_windows_reads_the_bitmap_through_a_file(monkeypatch, which_all) -> None:
    def pwsh(argv, kwargs):
        Path(argv[-1]).write_bytes(PNG)
        return b""

    _install(monkeypatch, {"pwsh": pwsh})
    assert read_clipboard(BIG, platform="win32", env={}).image == ClipboardImage(PNG, "image/png")


def test_windows_never_pipes_image_bytes_through_stdout(monkeypatch, which_all) -> None:
    """PowerShell's stdout is a TEXT stream and applies an output encoding to
    whatever crosses it, which corrupts binary silently — the payload still
    starts with a valid PNG magic and then fails to decode. A file is the only
    lossless channel, so the script must never write the bytes to the host.
    """
    seen: dict[str, str] = {}

    def pwsh(argv, kwargs):
        seen["script"] = Path(argv[-2]).read_text(encoding="utf-8")
        Path(argv[-1]).write_bytes(PNG)
        return b""

    _install(monkeypatch, {"pwsh": pwsh})
    read_clipboard(BIG, platform="win32", env={}).image
    script = seen["script"]
    assert "WriteAllBytes" in script, "bytes must go to a file, not to stdout"
    assert "Write-Output" not in script and "Write-Host" not in script


def test_windows_avoids_the_encoding_flag_the_two_generations_disagree_on(
    monkeypatch, which_all
) -> None:
    """Windows PowerShell 5.1 spells it ``-Encoding Byte``; PowerShell 7+
    removed that value for ``-AsByteStream``. ``[IO.File]::WriteAllBytes`` is
    identical on both, which is what lets one script serve both."""
    seen: dict[str, str] = {}

    def pwsh(argv, kwargs):
        seen["script"] = Path(argv[-2]).read_text(encoding="utf-8")
        Path(argv[-1]).write_bytes(PNG)
        return b""

    _install(monkeypatch, {"pwsh": pwsh})
    read_clipboard(BIG, platform="win32", env={}).image
    assert "-Encoding Byte" not in seen["script"]
    assert "-AsByteStream" not in seen["script"]


def test_windows_runs_in_a_single_threaded_apartment(monkeypatch, which_all) -> None:
    """The Windows clipboard API is STA-only and ``Clipboard::GetImage`` throws
    outright from the MTA that ``pwsh`` uses by default."""
    fake = _install(
        monkeypatch, {"pwsh": lambda argv, kwargs: (Path(argv[-1]).write_bytes(PNG), b"")[1]}
    )
    read_clipboard(BIG, platform="win32", env={}).image
    argv = fake.calls[0]
    assert "-STA" in argv
    assert "-NoProfile" in argv
    # `-File` and not `-Command`: only `-File` binds trailing tokens to `$args`,
    # so a `-Command` form would run with an empty `$args[0]` and write nowhere.
    assert "-File" in argv and "-Command" not in argv


def test_windows_prefers_pwsh_then_falls_back_to_powershell(monkeypatch) -> None:
    """PowerShell 7 starts faster, which matters inside a 2 s cap on a
    keystroke; ``powershell.exe`` is what is always present."""
    monkeypatch.setattr(
        clipboard_module.shutil,
        "which",
        lambda name: "/c/powershell.exe" if name == "powershell" else None,
    )
    fake = _install(
        monkeypatch,
        {"powershell.exe": lambda argv, kwargs: (Path(argv[-1]).write_bytes(PNG), b"")[1]},
    )
    assert read_clipboard(BIG, platform="win32", env={}).image is not None
    assert fake.calls[0][0] == "/c/powershell.exe"


def test_windows_without_any_powershell_is_silent(monkeypatch, which_none) -> None:
    fake = _install(monkeypatch, {})
    assert read_clipboard(BIG, platform="win32", env={}).image is None
    assert fake.calls == []


# -- shared guarantees --------------------------------------------------------
@pytest.mark.parametrize(
    ("platform", "env", "binary"),
    [
        ("darwin", {}, "osascript"),
        ("linux", {"DISPLAY": ":0"}, "xclip"),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "wl-paste"),
        ("win32", {}, "pwsh"),
    ],
)
def test_every_backend_is_bounded_by_the_timeout(
    platform: str, env: dict[str, str], binary: str, monkeypatch, which_all
) -> None:
    """A clipboard daemon that never answers must cost one pause, not a frozen
    composer. The cap is applied at the shared subprocess seam so a fifth
    backend cannot be added without it."""

    def hang(argv, kwargs):
        return subprocess.TimeoutExpired(argv, CLIPBOARD_TIMEOUT_S)

    _install(monkeypatch, {binary: hang})
    assert read_clipboard(BIG, platform=platform, env=env).image is None


@pytest.mark.parametrize(
    ("platform", "env", "binary"),
    [
        ("darwin", {}, "osascript"),
        ("linux", {"DISPLAY": ":0"}, "xclip"),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "wl-paste"),
        ("win32", {}, "pwsh"),
    ],
)
def test_every_backend_refuses_a_payload_over_the_cap(
    platform: str, env: dict[str, str], binary: str, monkeypatch, which_all
) -> None:
    """The clipboard is an untrusted-size source in exactly the way a pasted
    path is, so the same ceiling applies before the bytes are handed on."""
    oversized = PNG + b"\x00" * 4096

    def answer(argv, kwargs):
        if binary == "wl-paste" and "--list-types" in argv:
            return b"image/png\n"
        if binary in ("osascript", "pwsh"):
            Path(argv[-1]).write_bytes(oversized)
            return b"ok"
        return oversized

    _install(monkeypatch, {binary: answer})
    assert read_clipboard(1024, platform=platform, env=env).image is None


@pytest.mark.parametrize(
    ("platform", "env", "binary"),
    [
        ("darwin", {}, "osascript"),
        ("linux", {"DISPLAY": ":0"}, "xclip"),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "wl-paste"),
        ("win32", {}, "pwsh"),
    ],
)
def test_every_backend_survives_the_binary_disappearing(
    platform: str, env: dict[str, str], binary: str, monkeypatch, which_all
) -> None:
    """``which`` says yes, then the exec fails — an uninstall mid-session, a
    broken symlink. Every one of these runs on a keystroke, so none of them may
    raise."""

    def boom(argv, kwargs):
        raise OSError(2, "No such file or directory")

    _install(monkeypatch, {binary: boom})
    assert read_clipboard(BIG, platform=platform, env=env).image is None


def test_an_unknown_platform_reads_nothing(monkeypatch, which_all) -> None:
    """No guessing. An unrecognised platform has no known clipboard tool, and
    spawning something speculative on a keystroke is worse than doing nothing.
    """
    fake = _install(monkeypatch, {"xclip": PNG, "osascript": PNG})
    assert read_clipboard(BIG, platform="sunos5", env={"DISPLAY": ":0"}).image is None
    assert fake.calls == []


# -- the bounds that review round 1 corrected ---------------------------------
def test_the_whole_operation_shares_one_deadline_not_one_per_subprocess(
    monkeypatch, which_all
) -> None:
    """The X11 loop tries four MIME types and the composer adds a file-URL
    read, so a per-``_run`` cap let a wedged selection owner cost **8.0 s**
    against a constant that says 2 (review round 1, F2).

    Asserted on the budget each call is handed: the second spawn must see less
    time than the first, and once the deadline is spent no further process is
    spawned at all. A wall-clock assertion would either sleep for the real
    timeout or be flaky under load.
    """
    budgets: list[float] = []

    def slow(argv, kwargs):
        budgets.append(kwargs["timeout"] if "timeout" in kwargs else 0.0)
        return None

    fake = FakeRun({"xclip": slow})

    class TimedProcess(FakeProcess):
        def wait(self, timeout=None):  # noqa: ANN001, ANN201
            budgets.append(float(timeout if timeout is not None else 0.0))
            return self.returncode

    def popen(argv, **kwargs):  # noqa: ANN001, ANN003
        fake.calls.append(list(argv))
        return TimedProcess(None)

    monkeypatch.setattr(clipboard_module.subprocess, "Popen", popen)
    assert read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"}).image is None
    assert len(budgets) > 1, "the X11 loop should have tried several types"
    assert budgets == sorted(budgets, reverse=True), (
        "each call must be handed the REMAINING budget, so the sequence is "
        "non-increasing; equal budgets mean each subprocess got a fresh 2 s"
    )
    assert budgets[0] <= CLIPBOARD_TIMEOUT_S


def test_an_exhausted_deadline_spawns_nothing_further(monkeypatch, which_all) -> None:
    """Once the budget is gone, further spawns can only end in a kill, and each
    one still costs an interpreter start on the keystroke handler."""
    fake = _install(monkeypatch, {"xclip": PNG})
    spent = clipboard_module._Deadline(0.0)
    assert spent.expired is True
    assert clipboard_module._run(["xclip", "-o"], spent) is None
    assert fake.calls == [], "an expired deadline must not reach the process table"


def test_the_read_ceiling_stops_the_read_rather_than_judging_it_after(
    monkeypatch, which_all
) -> None:
    """``subprocess.run`` reads the pipe to EOF, so a ceiling applied to its
    result is a verdict on memory already spent — round 1 (F3) measured 300 MB
    buffered and 750 MB of peak RSS against a 4 MB cap.

    Pinned by asserting the READ was bounded: the module must ask for at most
    ``max_bytes + 1`` bytes, which is the smallest amount that still proves the
    stream was longer than allowed.
    """
    asked: list[int | None] = []

    class RecordingProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__(b"x" * 5000)

        @property
        def stdout(self):  # noqa: ANN201
            outer = self

            class Reader:
                def read(self, size=None):  # noqa: ANN001, ANN201
                    asked.append(size)
                    return outer._payload[: size or len(outer._payload)]

            return Reader()

        @stdout.setter
        def stdout(self, value) -> None:  # noqa: ANN001
            pass

    monkeypatch.setattr(
        clipboard_module.subprocess, "Popen", lambda argv, **kwargs: RecordingProcess()
    )
    deadline = clipboard_module._Deadline(CLIPBOARD_TIMEOUT_S)
    assert clipboard_module._run(["xclip", "-o"], deadline, max_bytes=1024) is None
    assert asked == [1025], "the read itself must carry the bound, not the check after it"


def test_a_payload_over_the_ingest_ceiling_is_refused_without_truncation(
    monkeypatch, which_all
) -> None:
    """Dropped rather than cut short: a truncated PNG still sniffs as a PNG and
    would be attached as a corrupt image block."""
    _install(monkeypatch, {"xclip": PNG + b"\x00" * 4096})
    assert read_clipboard(64, platform="linux", env={"DISPLAY": ":0"}).image is None


def test_the_ingest_ceiling_is_far_above_the_attachment_budget(monkeypatch) -> None:
    """The blocker from round 1 (U1), stated as the invariant that prevents it.

    A native ``Cmd+Shift+Ctrl+4`` on a Retina display puts 8.4-8.5 MB on the
    pasteboard and bounds down to 0.28 MB, so an ingest ceiling at the 4 MB
    attachment budget discarded the reported gesture's screenshot before the
    resize that makes it attachable could run.
    """
    from local_operator.tui.widgets.editor import MAX_ATTACHMENT_BYTES

    assert clipboard_module.MAX_CLIPBOARD_READ_BYTES > MAX_ATTACHMENT_BYTES * 4
    # A real full-screen Retina PNG is ~8.5 MB; the ceiling must clear that
    # with room for a multi-display capture rather than sitting just above it.
    assert clipboard_module.MAX_CLIPBOARD_READ_BYTES >= 32 * 1024 * 1024


def test_read_clipboard_reports_a_remote_refusal_as_its_own_outcome() -> None:
    """ "No image on the clipboard" is false over SSH: the clipboard was never
    read. The caller needs that distinction to avoid sending a remote user to
    re-copy something, which cannot help (round 1, D2/U2)."""
    contents = clipboard_module.read_clipboard(env={"SSH_TTY": "/dev/ttys001"})
    assert contents.refused_remote is True
    assert contents.image is None
    assert contents.paths == (), "a refusal reads nothing at all"


def test_read_clipboard_answers_both_shapes_under_one_deadline(monkeypatch, which_all) -> None:
    """One gesture, one budget. Two entry points meant two deadlines and a 4 s
    worst case on macOS (F2)."""

    fake = _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/a.png\x00"})
    contents = clipboard_module.read_clipboard(platform="darwin", env={})
    assert contents.paths == ("/tmp/a.png",)
    assert contents.image is None
    assert len(fake.calls) == 1, (
        "one osascript spawn must answer both shapes: the spawn is 2-4s of wall "
        "time and asking separately doubled the cost of every clipboard miss"
    )


# -- the bound that only a real process can prove -----------------------------
#
# These spawn REAL children. Every other test in this file fakes `Popen`, and
# round 2 showed exactly what that costs: `FakeProcess.stdout` is a `BytesIO`
# that never blocks, and the hang was simulated by raising `TimeoutExpired`
# from `wait()` — injecting the failure at the one line a real hang can never
# reach. Both timeout tests passed against a `_run` that blocked forever on
# `stdout.read()` before it ever got to `wait()`, freezing the composer
# indefinitely and orphaning the child. A mock cannot fail on that bug.
def test_a_hung_tool_returns_within_the_deadline_and_leaves_no_child() -> None:
    """A real child that holds stdout open and never writes.

    This is the wedged-daemon shape the budget exists for: `sleep` inherits the
    pipe, so the parent's read has neither bytes nor EOF to end on. Wall-clock
    assertion, because the property under test IS elapsed time — round 2
    measured 15 s on X11 and 12 s on macOS against this same 2 s budget.
    """
    deadline = clipboard_module._Deadline(1.0)
    started = time.monotonic()
    # `sh -c` so the child is a real process tree, matching how a wedged
    # `xclip`/`osascript` actually presents.
    process_marker = f"lo-clip-test-{os.getpid()}"
    result = clipboard_module._run(
        ["/bin/sh", "-c", f": {process_marker}; sleep 30"],
        deadline,
        max_bytes=4096,
    )
    elapsed = time.monotonic() - started

    assert result is None, "a tool that never answers has no clipboard image"
    assert elapsed < 5.0, (
        f"the read took {elapsed:.1f}s against a 1.0s budget; the deadline is "
        "not bounding the READ, only the wait after it"
    )
    # No orphan: the child must be killed AND reaped, or a wedged clipboard
    # leaks a process per paste.
    time.sleep(0.3)
    leftover = subprocess.run(
        ["pgrep", "-f", process_marker], capture_output=True, text=True, check=False
    )
    assert leftover.stdout.strip() == "", f"child survived _run: {leftover.stdout!r}"


def test_a_hung_tool_is_bounded_on_the_unbounded_read_shape_too() -> None:
    """The ``max_bytes=None`` call shape, which is how the type listing runs.

    ``read()`` and ``read(n)`` are different calls and only one of them was
    covered when this regressed, so both are pinned.
    """
    deadline = clipboard_module._Deadline(1.0)
    started = time.monotonic()
    result = clipboard_module._run(["/bin/sh", "-c", "sleep 30"], deadline)
    elapsed = time.monotonic() - started

    assert result is None
    assert elapsed < 5.0, f"unbounded-read shape took {elapsed:.1f}s against a 1.0s budget"


def test_a_tool_that_answers_then_hangs_is_still_bounded() -> None:
    """Partial output followed by silence — the case the original comment
    claimed to cover and the ordering defeated."""
    deadline = clipboard_module._Deadline(1.0)
    started = time.monotonic()
    result = clipboard_module._run(
        ["/bin/sh", "-c", "printf xx; sleep 30"], deadline, max_bytes=4096
    )
    elapsed = time.monotonic() - started

    assert result is None
    assert elapsed < 5.0, f"partial-then-hang took {elapsed:.1f}s against a 1.0s budget"


def test_a_real_healthy_tool_still_returns_its_output() -> None:
    """The control: the bound must not break the ordinary path. Without this,
    a `_run` that returned `None` unconditionally would pass every test above.
    """
    deadline = clipboard_module._Deadline(CLIPBOARD_TIMEOUT_S)
    assert clipboard_module._run(["/bin/echo", "-n", "hello"], deadline) == b"hello"


def test_a_real_oversized_stream_is_refused_without_buffering_it_all() -> None:
    """A real producer of more bytes than the ceiling allows."""
    deadline = clipboard_module._Deadline(CLIPBOARD_TIMEOUT_S)
    result = clipboard_module._run(
        ["/bin/sh", "-c", "head -c 200000 /dev/zero"], deadline, max_bytes=1024
    )
    assert result is None


def test_a_descendant_holding_the_pipe_is_bounded_after_its_parent_exits() -> None:
    """The fourth hang shape: the DIRECT CHILD exits, a descendant lives on.

    Round 3 (F4). The other hang tests all keep the direct child in the
    foreground, so `process.poll()` is always `None` and a kill gated on that
    always ran. Here the shell exits immediately and the backgrounded process
    inherits stdout, so `poll()` returns 0, the gated kill was skipped, the
    reader stayed blocked on a pipe nothing would close, and `Popen.__exit__`
    then deadlocked the MAIN thread on `stdout.close()` waiting for the
    `BufferedReader` lock the reader held. Measured: never returned after 20 s
    on a 2 s budget.

    Two things this pins that no mock could: that the kill is gated on the
    READER rather than on the child, and that the pgid is remembered from
    spawn — `os.getpgid()` on a reaped leader raises `ProcessLookupError`, so
    the lookup form degrades to no kill in exactly this case.
    """
    marker = f"lo-clip-orphan-{os.getpid()}"
    deadline = clipboard_module._Deadline(1.0)
    started = time.monotonic()
    # The shell exits (`exit 0`) while `sleep` keeps the inherited stdout open.
    result = clipboard_module._run(
        ["/bin/sh", "-c", f"sleep 60 & : {marker}; exit 0"],
        deadline,
        max_bytes=4096,
    )
    elapsed = time.monotonic() - started

    assert result is None
    assert elapsed < 5.0, (
        f"took {elapsed:.1f}s against a 1.0s budget: the kill is gated on the "
        "child rather than the reader, so a descendant holding the pipe "
        "deadlocks Popen.__exit__"
    )
    time.sleep(0.3)
    leftover = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True, check=False)
    assert leftover.stdout.strip() == "", f"descendant survived _run: {leftover.stdout!r}"


# -- clipboard TEXT, on every platform ----------------------------------------
#
# Text is read because `Ctrl+V` is a SYSTEM paste: it is the reachable route
# for issue #372 (the terminal sends nothing at all on Cmd+V with an image
# pasteboard, so the composer's empty-paste branch never runs), and a system
# paste that dropped the ordinary case would be a worse key than the one it
# replaced. Every backend is driven through the same dispatch with an injected
# platform, for the reason this file's docstring gives: the original bug was a
# single-environment assumption, and a text shape that worked on macOS alone
# would put one straight back.
def test_macos_reads_clipboard_text_through_the_same_single_spawn(monkeypatch, which_all) -> None:
    """One `osascript` spawn answers all three shapes. The spawn IS the
    latency (2-4 s of wall time on a loaded machine), so asking text as a
    second spawn would undo round 2's U3/U7 fix on the shape ctrl+v meets most
    often."""

    def osascript(argv, kwargs):
        Path(argv[2]).write_bytes("hello clipboard".encode("utf-8"))
        return b"text"

    fake = _install(monkeypatch, {"osascript": osascript})
    contents = read_clipboard(BIG, platform="darwin", env={})
    assert contents.text == "hello clipboard"
    assert contents.image is None and contents.paths == ()
    assert len(fake.calls) == 1, "text must not cost a second spawn"


def test_macos_clipboard_text_survives_unicode_and_newlines(monkeypatch, which_all) -> None:
    """Text goes through a FILE rather than osascript's stdout because the
    result coercion normalises line endings and can re-encode. A multi-line
    clipboard would otherwise come back altered — and would be
    indistinguishable from the one-word verdicts the same channel carries."""
    payload = "héllo — ünicode ✅\nline2\n\tindented"

    def osascript(argv, kwargs):
        Path(argv[2]).write_bytes(payload.encode("utf-8"))
        return b"text"

    _install(monkeypatch, {"osascript": osascript})
    assert read_clipboard(BIG, platform="darwin", env={}).text == payload


def test_an_image_wins_over_text_on_the_same_pasteboard(monkeypatch, which_all) -> None:
    """The shapes are mutually exclusive, in the order image, paths, text.
    A screenshot tool can leave both an image and a caption; the image is what
    the user copied in the reported gesture."""

    def osascript(argv, kwargs):
        Path(argv[2]).write_bytes(PNG)
        return b"image"

    _install(monkeypatch, {"osascript": osascript})
    contents = read_clipboard(BIG, platform="darwin", env={})
    assert contents.image == ClipboardImage(PNG, "image/png")
    assert contents.text == "", "an image and text at once would attach AND type"


def test_a_finder_copy_attaches_the_file_rather_than_typing_its_name(
    monkeypatch, which_all
) -> None:
    """Finder's Cmd+C puts `public.file-url` AND the file's display name on the
    pasteboard together, so a backend that asked for text before file URLs
    would type a filename instead of attaching the file the user copied. The
    script's own ordering is what prevents it."""
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/a.png\x00"})
    contents = read_clipboard(BIG, platform="darwin", env={})
    assert list(contents.paths) == ["/tmp/a.png"]
    assert contents.text == ""


def test_x11_reads_clipboard_text(monkeypatch, which_all) -> None:
    """`-t UTF8_STRING` rather than a bare `-o`: with no target xclip picks one
    itself and can hand back a non-text flavor from a clipboard offering
    several, which reaches the composer as mojibake."""
    fake = _install(monkeypatch, {"xclip": lambda argv, kwargs: b"x11 text"})
    contents = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert contents.text == "x11 text"
    assert ["-t", "UTF8_STRING"] == fake.calls[-1][3:5]


def test_x11_prefers_an_image_and_never_reads_text_after_finding_one(
    monkeypatch, which_all
) -> None:
    """Skipping the text read when an image was found keeps the common
    screenshot gesture at one subprocess and keeps the two fields exclusive."""
    fake = _install(monkeypatch, {"xclip": lambda argv, kwargs: PNG})
    contents = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert contents.image is not None and contents.text == ""
    assert not any("UTF8_STRING" in call for call in fake.calls)


def test_wayland_reads_a_listed_text_type(monkeypatch, which_all) -> None:
    """Only a type the compositor LISTED is read, text included."""

    def wl_paste(argv, kwargs):
        if "--list-types" in argv:
            return b"text/plain;charset=utf-8\n"
        return b"wayland text"

    fake = _install(monkeypatch, {"wl-paste": wl_paste})
    contents = read_clipboard(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"})
    assert contents.text == "wayland text"
    assert fake.calls[-1][-1] == "text/plain;charset=utf-8"


def test_windows_reads_clipboard_text(monkeypatch, which_all) -> None:
    """Stdout is safe for TEXT where it is not for image bytes: the output
    encoding that corrupts binary is the right treatment for a string.

    Windows is unit-tested only — there is no Windows host in this project's
    environment — so this pins the invocation, not a real read."""
    fake = _install(monkeypatch, {"pwsh": lambda argv, kwargs: b"windows text"})
    contents = read_clipboard(BIG, platform="win32", env={})
    assert contents.text == "windows text"
    script = Path(fake.calls[-1][-1])
    assert "-Raw" in clipboard_module._WINDOWS_TEXT_SCRIPT, (
        "without -Raw, Get-Clipboard returns an ARRAY of lines that PowerShell "
        "then re-joins with the host separator, altering a multi-line clipboard"
    )
    assert "-STA" in fake.calls[-1], "the Windows clipboard API is STA-only"
    assert script.name.endswith(".ps1")


def test_clipboard_text_that_is_not_valid_utf8_degrades_rather_than_raising(
    monkeypatch, which_all
) -> None:
    """This runs on a keystroke. A clipboard holding bytes that are not valid
    UTF-8 is a paste that should degrade, not an exception on the key that
    pasted it."""
    _install(monkeypatch, {"xclip": lambda argv, kwargs: b"caf\xff\xfe"})
    contents = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert contents.text.startswith("caf")


def test_clipboard_text_over_the_ingest_ceiling_is_refused(monkeypatch, which_all) -> None:
    """The clipboard is an untrusted-size source and this text is about to be
    inserted into the composer's document. Over the bound reads as an empty
    clipboard — the same collapse every other unusable case takes."""
    _install(monkeypatch, {"xclip": lambda argv, kwargs: b"x" * 64})
    assert read_clipboard(16, platform="linux", env={"DISPLAY": ":0"}).text == ""


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_a_remote_session_reads_no_text_either(platform: str, monkeypatch, which_all) -> None:
    """The SSH refusal is about confidentiality, and clipboard TEXT on the
    server is exactly as much the user's private data as an image is."""
    fake = _install(monkeypatch, {name: b"secret" for name in ("osascript", "xclip", "pwsh")})
    contents = read_clipboard(
        BIG, platform=platform, env={"SSH_CONNECTION": "1.2.3.4 22 5.6.7.8 22", "DISPLAY": ":0"}
    )
    assert contents.text == "" and contents.refused_remote is True
    assert fake.calls == [], "a remote session must not spawn a clipboard tool at all"


# -- a timeout is not an empty clipboard -------------------------------------
def test_a_wedged_tool_reports_a_timeout_rather_than_an_empty_clipboard(
    monkeypatch, which_all
) -> None:
    """The distinction the composer needs to word its notice honestly.

    A backend returning `None` is ambiguous by design - "no image of that type"
    and "the tool never answered" are the same value - and collapsing them told
    a user holding a valid screenshot that their clipboard was empty. Measured
    under CPU load at 8 failures in 10 reads (ux round 1, U3).
    """

    def wedged(argv, kwargs):
        time.sleep(CLIPBOARD_TIMEOUT_S + 0.4)
        return b""

    _install(monkeypatch, {"osascript": wedged})
    contents = read_clipboard(BIG, platform="darwin", env={})
    assert contents.image is None and contents.text == ""
    assert contents.timed_out is True


def test_a_clipboard_that_simply_has_nothing_is_not_reported_as_a_timeout(
    monkeypatch, which_all
) -> None:
    """The other half: an empty clipboard answers promptly and truthfully, and
    must not borrow the timeout's remedy ("try again") for a state a retry
    cannot change."""
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b""})
    contents = read_clipboard(BIG, platform="darwin", env={})
    assert contents.image is None and contents.text == ""
    assert contents.timed_out is False


def test_a_successful_read_is_never_flagged_as_timed_out(monkeypatch, which_all) -> None:
    """A payload that arrived is a success even if a later cheap call expired.
    A notice claiming a timeout beside an attached image would describe a
    failure that did not happen."""

    def osascript(argv, kwargs):
        Path(argv[2]).write_bytes(PNG)
        return b"image"

    _install(monkeypatch, {"osascript": osascript})
    contents = read_clipboard(BIG, platform="darwin", env={})
    assert contents.image is not None
    assert contents.timed_out is False


@pytest.mark.parametrize("platform,binary", [("linux", "xclip"), ("win32", "pwsh")])
def test_the_timeout_flag_is_reported_off_macos_too(
    platform: str, binary: str, monkeypatch, which_all
) -> None:
    """Every platform is a peer on this too: a timeout that only the macOS
    backend could report would leave Linux and Windows users with the same
    wrong diagnosis the flag exists to remove."""

    def wedged(argv, kwargs):
        time.sleep(CLIPBOARD_TIMEOUT_S + 0.4)
        return b""

    _install(monkeypatch, {binary: wedged})
    contents = read_clipboard(BIG, platform=platform, env={"DISPLAY": ":0"})
    assert contents.timed_out is True


def test_clipboard_text_is_bounded_by_the_paste_budget_not_the_image_ceiling(
    monkeypatch, which_all
) -> None:
    """Text has no downstream resize, so the 64 MB INGEST ceiling - which is
    generous precisely because images get bounded later - is not a bound on
    text at all.

    Measured before the cap: a 5 MB text clipboard blocked the UI for 52 s on a
    synchronous insert (code round 1, F3). The image ceiling stays where it is;
    only text takes the smaller budget, because only text reaches the document
    at its read size.
    """
    oversized = b"x" * (MAX_CLIPBOARD_TEXT_BYTES + 1)
    _install(monkeypatch, {"xclip": lambda argv, kwargs: oversized})
    contents = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert contents.text == "", "text past the paste budget must not reach the composer"

    at_limit = b"y" * MAX_CLIPBOARD_TEXT_BYTES
    _install(monkeypatch, {"xclip": lambda argv, kwargs: at_limit})
    contents = read_clipboard(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert len(contents.text) == MAX_CLIPBOARD_TEXT_BYTES, "the budget itself must fit"
