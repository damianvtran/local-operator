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
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from local_operator import clipboard as clipboard_module
from local_operator.clipboard import (
    CLIPBOARD_TIMEOUT_S,
    ClipboardImage,
    clipboard_reads_are_local,
    read_clipboard_file_paths,
    read_clipboard_image,
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


class FakeRun:
    """Stands in for ``subprocess.run`` and records the argv it was given.

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
        if answer is None:
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        return subprocess.CompletedProcess(argv, 0, answer, b"")


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
    monkeypatch.setattr(clipboard_module.subprocess, "run", fake)
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
    assert read_clipboard_image(BIG, platform=platform, env=env) is None
    assert read_clipboard_file_paths(platform=platform, env=env) == []
    assert fake.calls == [], "a remote session must not even spawn the reader"


# -- macOS --------------------------------------------------------------------
def test_macos_reads_the_png_flavor(monkeypatch, which_all, tmp_path) -> None:
    """The reported case: a native screenshot leaves PNG bytes and no text."""

    def osascript(argv, kwargs):
        # The script writes to the destination path passed as its argv, which
        # is what the backend then reads back.
        Path(argv[2]).write_bytes(PNG)
        return b"ok"

    _install(monkeypatch, {"osascript": osascript})
    image = read_clipboard_image(BIG, platform="darwin", env={})
    assert image == ClipboardImage(PNG, "image/png")


def test_macos_reads_the_script_from_stdin_not_the_argv(monkeypatch, which_all) -> None:
    """The AppleScript goes in over stdin, so it never appears in ``ps`` and
    the destination path is never spliced into source text."""

    def osascript(argv, kwargs):
        assert argv[1] == "-", "the script must come from stdin"
        assert b"NSPasteboard" in kwargs["input"]
        assert b"public.tiff" in kwargs["input"], "the TIFF fallback must be in the script"
        Path(argv[2]).write_bytes(PNG)
        return b"ok"

    _install(monkeypatch, {"osascript": osascript})
    assert read_clipboard_image(BIG, platform="darwin", env={}) is not None


def test_macos_with_an_empty_pasteboard_is_no_image(monkeypatch, which_all) -> None:
    """The script writes no file and reports "none"; the backend must not then
    hand back a zero-byte payload that sniffs as nothing."""
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b"none"})
    assert read_clipboard_image(BIG, platform="darwin", env={}) is None


def test_the_sniffed_type_is_what_comes_back_not_the_requested_one(monkeypatch, which_all) -> None:
    """The result names what the bytes ARE. A backend that reported its own
    request would have laundered the ``xclip`` text case above into a
    confident, wrong ``image/png``."""

    def wl_paste(argv, kwargs):
        # Offers PNG, delivers JPEG. Nothing stops a clipboard owner doing this.
        return b"image/png\n" if "--list-types" in argv else JPEG

    _install(monkeypatch, {"wl-paste": wl_paste})
    image = read_clipboard_image(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"})
    assert image is not None and image.mime_type == "image/jpeg"


def test_macos_file_urls_come_back_as_paths(monkeypatch, which_all) -> None:
    """Finder's Cmd+C puts only ``public.file-url`` on the pasteboard — no
    text, no image bytes — so this is the only route to what was copied."""
    _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/a.png\n/tmp/b c.png\n"})
    assert read_clipboard_file_paths(platform="darwin", env={}) == ["/tmp/a.png", "/tmp/b c.png"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_file_urls_are_a_macos_only_route(platform: str, monkeypatch, which_all) -> None:
    """Linux file managers put ``text/uri-list`` on the clipboard next to plain
    text of the same paths, so the terminal already bracket-pastes them and the
    composer's path branch already handles it. A second route there would be a
    way for one copy to be attached twice.
    """
    fake = _install(monkeypatch, {"osascript": lambda argv, kwargs: b"/tmp/a.png\n"})
    assert read_clipboard_file_paths(platform=platform, env={}) == []
    assert fake.calls == []


# -- X11 ----------------------------------------------------------------------
def test_x11_reads_the_png_target(monkeypatch, which_all) -> None:
    _install(monkeypatch, {"xclip": PNG})
    image = read_clipboard_image(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert image == ClipboardImage(PNG, "image/png")


def test_x11_asks_for_the_clipboard_selection_not_the_primary(monkeypatch, which_all) -> None:
    """``PRIMARY`` is the middle-click selection and holds whatever text was
    last highlighted; ``Cmd+V``/``Ctrl+V`` pastes ``CLIPBOARD``."""
    fake = _install(monkeypatch, {"xclip": PNG})
    read_clipboard_image(BIG, platform="linux", env={"DISPLAY": ":0"})
    assert fake.calls[0] == ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]


def test_x11_falls_through_to_the_next_type(monkeypatch, which_all) -> None:
    """A browser copy is often JPEG or WebP and never PNG."""
    _install(
        monkeypatch,
        {"xclip": lambda argv, kwargs: JPEG if "image/jpeg" in argv else None},
    )
    image = read_clipboard_image(BIG, platform="linux", env={"DISPLAY": ":0"})
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
    assert read_clipboard_image(BIG, platform="linux", env={"DISPLAY": ":0"}) is None


def test_an_unsendable_format_on_the_clipboard_is_refused(monkeypatch, which_all) -> None:
    """HEIC sniffs fine and no provider accepts it. "No image" is a better
    answer than an attachment that earns a 400 mid-turn."""
    heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
    _install(monkeypatch, {"xclip": heic})
    assert read_clipboard_image(BIG, platform="linux", env={"DISPLAY": ":0"}) is None


def test_x11_without_xclip_installed_is_silent(monkeypatch, which_none) -> None:
    """A missing ``xclip`` is not an error the user should see on every stray
    empty paste — it is simply a machine with no clipboard images."""
    fake = _install(monkeypatch, {})
    assert read_clipboard_image(BIG, platform="linux", env={"DISPLAY": ":0"}) is None
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
    image = read_clipboard_image(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"})
    assert image == ClipboardImage(PNG, "image/png")
    assert fake.calls[0][1] == "--list-types"


def test_wayland_with_no_image_type_on_offer_reads_nothing(monkeypatch, which_all) -> None:
    fake = _install(monkeypatch, {"wl-paste": lambda argv, kwargs: b"text/plain\ntext/html\n"})
    assert read_clipboard_image(BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"}) is None
    assert len(fake.calls) == 1, "a listing with no image type must not be followed by a read"


def test_wayland_is_preferred_when_xwayland_also_sets_display(monkeypatch, which_all) -> None:
    """A Wayland session usually runs XWayland too, so ``DISPLAY`` is set in
    both. Testing ``DISPLAY`` first would route a Wayland session to ``xclip``
    and read XWayland's separate, usually empty selection."""

    def wl_paste(argv, kwargs):
        return b"image/png\n" if "--list-types" in argv else PNG

    fake = _install(monkeypatch, {"wl-paste": wl_paste, "xclip": b"WRONG-SELECTION"})
    image = read_clipboard_image(
        BIG, platform="linux", env={"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    )
    assert image is not None and image.data == PNG
    assert all(call[0] != "xclip" for call in fake.calls)


def test_a_headless_linux_box_spawns_nothing(monkeypatch, which_all) -> None:
    """No display server means no clipboard, and discovering that by shelling
    out would cost a subprocess on every empty paste."""
    fake = _install(monkeypatch, {"xclip": PNG, "wl-paste": PNG})
    assert read_clipboard_image(BIG, platform="linux", env={}) is None
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
    assert read_clipboard_image(BIG, platform="win32", env={}) == ClipboardImage(PNG, "image/png")


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
    read_clipboard_image(BIG, platform="win32", env={})
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
    read_clipboard_image(BIG, platform="win32", env={})
    assert "-Encoding Byte" not in seen["script"]
    assert "-AsByteStream" not in seen["script"]


def test_windows_runs_in_a_single_threaded_apartment(monkeypatch, which_all) -> None:
    """The Windows clipboard API is STA-only and ``Clipboard::GetImage`` throws
    outright from the MTA that ``pwsh`` uses by default."""
    fake = _install(
        monkeypatch, {"pwsh": lambda argv, kwargs: (Path(argv[-1]).write_bytes(PNG), b"")[1]}
    )
    read_clipboard_image(BIG, platform="win32", env={})
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
    assert read_clipboard_image(BIG, platform="win32", env={}) is not None
    assert fake.calls[0][0] == "/c/powershell.exe"


def test_windows_without_any_powershell_is_silent(monkeypatch, which_none) -> None:
    fake = _install(monkeypatch, {})
    assert read_clipboard_image(BIG, platform="win32", env={}) is None
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
        assert kwargs["timeout"] == CLIPBOARD_TIMEOUT_S
        raise subprocess.TimeoutExpired(argv, CLIPBOARD_TIMEOUT_S)

    _install(monkeypatch, {binary: hang})
    assert read_clipboard_image(BIG, platform=platform, env=env) is None


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
    assert read_clipboard_image(1024, platform=platform, env=env) is None


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
    assert read_clipboard_image(BIG, platform=platform, env=env) is None


def test_an_unknown_platform_reads_nothing(monkeypatch, which_all) -> None:
    """No guessing. An unrecognised platform has no known clipboard tool, and
    spawning something speculative on a keystroke is worse than doing nothing.
    """
    fake = _install(monkeypatch, {"xclip": PNG, "osascript": PNG})
    assert read_clipboard_image(BIG, platform="sunos5", env={"DISPLAY": ":0"}) is None
    assert fake.calls == []
