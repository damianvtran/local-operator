"""Real-App release evidence for the new numeric Settings row.

These are ordinary CI tests, NOT a locally admitted lightweight capture rig.
The fake owns only the session boundary; Settings, config writes, validation,
production CSS and capture geometry remain real. Image review is still required.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

SETTING_KEY = "providers.openrouter.retire_below_tps"
SOURCE_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session", autouse=True)
def _prepay_session_import_warmup() -> None:
    """Shadow the parent SESSION fixture, before function monkeypatch setup.

    This module tests an injected facade, not owner construction. Paying the
    real owner's provider/MCP warmup defeats that boundary; its ordering remains
    covered by the dedicated startup tests. No production default is changed.
    """


def _metadata_text(path: Path, root: Path, limit: int = 16384) -> str:
    """Read a bounded Git metadata file, never a ref escaping its authority."""
    resolved = path.resolve()
    assert resolved.is_relative_to(root.resolve()), "Git metadata escaped its root"
    with resolved.open("rb") as stream:
        data = stream.read(limit + 1)
    assert len(data) <= limit, "Git metadata exceeded the bounded read"
    return data.decode("ascii").strip()


def checkout_revision(checkout: Path) -> str:
    """Resolve real detached/branch/worktree HEAD without spawning git.

    Actions checkout may leave main symbolic; worktrees use a gitfile and a
    separate common ref store. GITHUB_SHA can describe a different event ref,
    so it is not a substitute for the actual checkout's identity.
    """
    marker = checkout / ".git"
    if marker.is_file():
        pointer = _metadata_text(marker, checkout)
        assert pointer.startswith("gitdir: ") and "\n" not in pointer
        location = Path(pointer.removeprefix("gitdir: "))
        gitdir = (checkout / location).resolve()
    else:
        assert marker.is_dir(), "checkout has no Git metadata directory"
        gitdir = marker.resolve()
    common = gitdir
    if (gitdir / "commondir").exists():
        location = _metadata_text(gitdir / "commondir", gitdir)
        assert location and "\n" not in location
        common = (gitdir / location).resolve()
    assert gitdir.is_dir() and (common / "objects").is_dir()
    value = _metadata_text(gitdir / "HEAD", gitdir)
    seen = set()
    for _ in range(8):
        if re.fullmatch(r"[0-9a-f]{40}", value):
            return value
        assert value.startswith("ref: "), "invalid checkout revision"
        ref = value.removeprefix("ref: ")
        assert re.fullmatch(r"refs/[A-Za-z0-9._/-]+", ref)
        assert all(part not in ("", ".", "..") for part in ref.split("/"))
        assert ref not in seen, "cyclic symbolic ref"
        seen.add(ref)
        for authority in dict.fromkeys((gitdir, common)):
            path = authority / ref
            if path.exists():
                value = _metadata_text(path, authority)
                break
        else:
            matches = set()
            packed = common / "packed-refs"
            if packed.exists():
                for line in _metadata_text(packed, common, 1024 * 1024).splitlines():
                    if not line or line.startswith(("#", "^")):
                        continue
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == ref:
                        matches.add(fields[0])
            assert len(matches) == 1, "missing or ambiguous checkout ref"
            value = matches.pop()
    raise AssertionError("symbolic checkout ref exceeded bounded hops")


@pytest.mark.parametrize(
    "shape", ["detached", "loose-main", "packed-main", "worktree-loose", "worktree-packed"]
)
def test_checkout_revision_reads_actual_ref_shape(tmp_path: Path, shape: str) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    if shape.startswith("worktree"):
        common = tmp_path / "metadata"
        gitdir = common / "worktrees" / "proof"
        gitdir.mkdir(parents=True)
        (checkout / ".git").write_text("gitdir: ../metadata/worktrees/proof\n")
        (gitdir / "commondir").write_text("../..\n")
    else:
        common = gitdir = checkout / ".git"
        gitdir.mkdir()
    (common / "objects").mkdir()
    expected = "1" * 40
    if shape == "detached":
        (gitdir / "HEAD").write_text(expected + "\n")
    else:
        (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
        if "packed" in shape:
            (common / "packed-refs").write_text(
                "# pack-refs with: peeled\n" + expected + " refs/heads/main\n"
            )
        else:
            loose = common / "refs" / "heads" / "main"
            loose.parent.mkdir(parents=True)
            loose.write_text(expected + "\n")
    assert checkout_revision(checkout) == expected


@pytest.mark.parametrize(
    "fault", ["missing", "cycle", "traversal", "escaping-symlink", "oversized"]
)
def test_checkout_revision_rejects_bad_metadata(tmp_path: Path, fault: str) -> None:
    gitdir = tmp_path / ".git"
    (gitdir / "objects").mkdir(parents=True)
    head = gitdir / "HEAD"
    head.write_text("ref: refs/heads/main\n")
    if fault == "traversal":
        head.write_text("ref: refs/heads/../../outside\n")
    elif fault == "oversized":
        head.write_text("0" * 16385)
    elif fault in ("cycle", "escaping-symlink"):
        loose = gitdir / "refs" / "heads" / "main"
        loose.parent.mkdir(parents=True)
        if fault == "cycle":
            loose.write_text("ref: refs/heads/main\n")
        else:
            outside = tmp_path / "outside"
            outside.write_text("1" * 40)
            loose.symlink_to(outside)
    with pytest.raises(AssertionError):
        checkout_revision(tmp_path)


def refuse_external(violations: list[str], *_args, **_kwargs) -> NoReturn:
    violations.append("external-effect-or-provider")
    raise AssertionError("UI proof attempted an external effect or provider")


def test_refusal_counter_is_recorded_before_raising() -> None:
    """Pure primitive control, not a runtime/provider-hook sensitivity proof."""
    violations: list[str] = []
    with pytest.raises(AssertionError, match="external effect"):
        refuse_external(violations)
    assert violations == ["external-effect-or-provider"]


@pytest.fixture
def ui_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reject external effects and check counters outside swallowed App errors."""
    root = tmp_path / "synthetic-ui"
    root.mkdir()
    config = root / "config"
    config.mkdir()
    for key in tuple(os.environ):
        if key.startswith(("CMUX_", "LOP_", "LOCAL_OPERATOR_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_DESKTOP_LAUNCH", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    violations: list[str] = []

    def refuse(*_args, **_kwargs) -> NoReturn:
        refuse_external(violations, *_args, **_kwargs)

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)
    for name in ("system", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    import asyncio

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", refuse)

    # No App import occurs at module scope. These traps precede this module's
    # product imports, not pytest's already-collected parent conftest imports.
    # That remaining import closure is why this is remote-CI-only, not LIGHT.
    from local_operator.tui import app as app_module
    from local_operator.tui import quota_notice
    from scripts import visual_capture
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    assert Path(app_module.__file__).resolve() == (SOURCE_ROOT / "local_operator/tui/app.py")
    assert app_module.OperatorApp.CSS_PATH == "local_operator.tcss"
    assert (SOURCE_ROOT / "local_operator/tui/local_operator.tcss").is_file()

    async def no_quota_prewarm():
        return False

    monkeypatch.setattr(app_module, "start_prewarm", lambda: None)
    monkeypatch.setattr(app_module, "await_prewarm", no_quota_prewarm)
    monkeypatch.setattr(quota_notice, "_prewarm_started", True)

    async def no_attended_onboarding(_self):
        return None

    # A fresh synthetic HOME is deliberately not a first attended install.
    # Only this unrelated onboarding route is replaced, not boot/adoption,
    # on_mount, polling, Settings handlers or their real ConfigManager writes.
    monkeypatch.setattr(app_module.OperatorApp, "_route_first_run_boot", no_attended_onboarding)
    for name in (
        "_check_for_update",
        "_start_terminal_title",
        "_start_multiplexer_broadcast",
        "_start_herdr_reporter",
    ):
        monkeypatch.setattr(app_module.OperatorApp, name, lambda _self: None)

    # save_capture's optional fontconfig metadata normally spawns fc-match.
    # Record UNMEASURED rather than allow an incidental child or invent font
    # metrics. The real SVG exporter/CSS/geometry are untouched; the independent
    # rasterizer later owns actual font selection and viewed-pixel evidence.
    monkeypatch.setattr(
        visual_capture,
        "font_provenance",
        lambda _profile: {"status": "unmeasured: no font subprocess in UI proof"},
    )

    class BoundSession(FakeSession):
        async def prompt(self, *_args, **_kwargs):
            refuse()

        async def complete_once(self, system: str, prompt: str) -> str:
            refuse()

    session = BoundSession()
    factory_calls: list[str] = []

    async def factory():
        factory_calls.append("synthetic-facade")
        return await _factory(session)

    app = app_module.OperatorApp(factory, warm_session_imports=False)
    proof = SimpleNamespace(
        app=app,
        session=session,
        config=config,
        root=root,
        output=tmp_path / "proof",
        violations=violations,
        factory_calls=factory_calls,
    )
    try:
        yield proof
    finally:
        assert violations == [], "caught App exceptions must not hide tripwire calls"
        assert factory_calls == ["synthetic-facade"]
        assert not session.prompts and not session.completions
        assert getattr(app, "_notifier", None) is None


async def _await_transcript_scroll_settled(proof, pilot, *, tries: int = 400) -> None:
    """Await the transcript's OWN scroll completion before sampling a frame.

    WHY THIS REPLACES A PRE-LOOP SETTLE. A single
    ``pilot.wait_for_scheduled_animations()`` before the loop did NOT stop the
    between-sample motion (CI run 38842431374: the two ownership frames still
    differed by exactly two rows at both widths). The published sidecar says why:
    the transcript's ``virtual_size`` is IDENTICAL in both frames ([96, 47]), so
    nothing was appended between the samples -- the scroll OFFSET moved. Both
    moved cards shift by the same two rows, which is an offset change, not a
    relayout and not new content.

    The producer is an in-flight ANIMATED scroll. The fixture calls
    ``card.scroll_visible(immediate=True)``, and ``animate`` defaults to True;
    Textual's ``immediate=True`` only skips the ``call_after_refresh`` deferral --
    it still routes to ``_scroll_to(animate=True)``, which animates over the
    default duration. ``wait_for_scheduled_animations`` is a CPU-idle heuristic
    (20 ms granularity, 1 s cap) and so is not a completion condition for that
    animation.

    So this waits on the harness's own named state, the same predicate
    ``tests/unit/tui/test_rendered_history_paging.py`` uses: the animator is no
    longer driving this view's ``scroll_y`` AND the scroll has reached its
    target. That is a bounded await on a completion condition, NOT polling until
    the two geometries happen to agree (still forbidden) and NOT a longer idle
    guess. Running it before EACH sample is the point: motion that begins after
    sample 1 is covered by sample 2's own wait.

    ``tries`` is a wedge backstop, not the assertion: a transcript that never
    settles fails loudly here rather than silently shipping an unstable pair.
    """
    view = proof.app._transcript_view()
    for _ in range(tries):
        await pilot.pause()
        if (
            not proof.app.animator.is_being_animated(view, "scroll_y")
            and view.scroll_y == view.scroll_target_y
        ):
            # One extra beat for the anchor/gap settle the growth queued after
            # the scroll landed (the pattern the paging tests use).
            await pilot.pause()
            return
    raise AssertionError(
        "transcript scroll never settled before capture; "
        f"animating={proof.app.animator.is_being_animated(view, 'scroll_y')} "
        f"scroll_y={view.scroll_y} target={view.scroll_target_y}"
    )


async def capture_pair(proof, pilot, case: str, phase: str) -> set[str]:
    """Use the existing active-screen exporter; keep unviewed evidence labelled."""
    from scripts.visual_capture import save_capture

    proof.output.mkdir(exist_ok=True)
    generated: set[str] = set()
    for frame in ("post-pause-1", "post-pause-2"):
        # Settle BEFORE each sample: a pre-loop settle alone let an in-flight
        # animated card scroll continue into the second sample.
        await _await_transcript_scroll_settled(proof, pilot)
        await pilot.pause()
        stem = f"{case}-{phase}-{frame}"
        target = proof.output / f"{stem}.svg"
        save_capture(proof.app, target)
        generated.update((target.name, target.with_suffix(".geometry.json").name))
    publish_proof(proof, case, phase, generated)
    return generated


def publish_proof(
    proof, case: str, phase: str, names: set[str], *, _target: Path | None = None
) -> None:
    """Publish a fully validated pair BEFORE its unchanged strict verdict.

    Failed geometry is evidence, not a reason to discard the original frames.
    Privacy, provenance and the combined job quota still precede EVERY copy.
    The private target argument isolates pure publisher controls from the real
    CI artifact directory; it is not an App/exporter/renderer substitution.
    The strict verdict that follows compares two POST-SETTLE samples; see the
    narrowed claim at its assertion.
    """
    assert re.fullmatch(r"(?:host-speed|ownership)-(?:100|60)x30", case)
    assert phase in proof.expected_phases and re.fullmatch(r"[a-z-]+", phase)
    frames = ("post-pause-1", "post-pause-2")
    expected = {
        f"{case}-{phase}-{frame}{suffix}"
        for frame in frames
        for suffix in (".svg", ".geometry.json")
    }
    assert names == expected, "unexpected or incomplete proof pair"
    paths = {name: proof.output / name for name in names}
    assert all(path.resolve().parent == proof.output.resolve() for path in paths.values())
    assert (
        sum(path.stat().st_size for path in paths.values()) <= 12 * 1024 * 1024
    ), "proof pair byte quota exceeded before publication"
    incoming = {name: path.read_bytes() for name, path in paths.items()}
    measured = []
    for data in incoming.values():
        # Refuse the whole pair, rather than rewrite pixels or publish its safe
        # first member before discovering a leak in the second.
        for private_prefix in (proof.root, SOURCE_ROOT):
            assert str(private_prefix).encode() not in data, "private prefix in proof pair"
    for frame in frames:
        geometry = json.loads(incoming[f"{case}-{phase}-{frame}.geometry.json"])
        assert geometry["grid"] == list(proof.app.size)
        assert geometry["widgets"] and geometry["css_path"] == ["local_operator.tcss"]
        measured.append({key: geometry[key] for key in ("grid", "screen", "widgets")})
    stable = measured[0] == measured[1]
    if os.environ.get("GITHUB_ACTIONS") != "true":
        assert stable, "consecutive frames changed geometry"
        return
    # Unit CI is POSIX. A lock covers all xdist workers in THIS job; neither a
    # partial overwrite nor a per-worker quota may hide a per-job overflow.
    import fcntl

    target = _target if _target is not None else SOURCE_ROOT / "test-artifacts" / "tui-proof"
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        head = checkout_revision(SOURCE_ROOT)
        existing = {
            p.name: p.stat().st_size for p in target.iterdir() if p.suffix in (".svg", ".json")
        }
        assert sum(n.endswith(".svg") for n in existing.keys() | names) <= 48
        sources = (
            "local_operator/tui/local_operator.tcss",
            "local_operator/tui/app.py",
            "local_operator/tui/widgets/settings_view.py",
            "local_operator/tui/widgets/tool_card.py",
            "local_operator/tools/tool_docs.py",
            "scripts/visual_capture.py",
            "tests/unit/tui/conftest.py",
            "tests/unit/tui/test_app_pilot.py",
            "tests/unit/tui/test_host_speed_settings_rendered.py",
            "tests/unit/tui/test_ownership_inspection_rendered.py",
        )
        manifest.update(
            checkout_revision=head,
            run_id=os.environ.get("GITHUB_RUN_ID"),
            python=sys.version,
            node=None,
            node_note="Python/Textual capture; Node not used or measured",
            font_note="unmeasured here; independent rasterization/view still required",
            validation_note=(
                "Capture presence/counters are not final fixture/test PASS. "
                "Authenticate final JUnit, including teardown, independently."
            ),
            sampling_note=(
                "Post-pause geometry samples only; not initial-paint observation "
                "or pixel equality/reflow exclusion. Independent image view required."
            ),
            source_sha256={
                name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
                for name in sources
            },
        )
        manifest.setdefault("expected_phases", {})[case] = sorted(proof.expected_phases)
        manifest.setdefault("cases", {}).setdefault(case, {})[phase] = {
            "grid": list(proof.app.size),
            "pair_frames": list(frames),
            "geometry_equal_at_capture": stable,
            "pair_verdict_note": "Diagnostic capture only; strict assertion follows publication.",
            "files": {name: hashlib.sha256(incoming[name]).hexdigest() for name in sorted(names)},
            "external_attempts_at_capture": len(proof.violations),
            "coverage_note": proof.case_note,
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
        projected = (
            sum(
                size
                for name, size in existing.items()
                if name not in incoming and name != "manifest.json"
            )
            + sum(map(len, incoming.values()))
            + len(manifest_bytes)
        )
        assert projected <= 12 * 1024 * 1024, "proof quota exceeded before copying"
        for name, data in incoming.items():
            (target / name).write_bytes(data)
        manifest_path.write_bytes(manifest_bytes)
    # This is the SAME complete grid/screen/widgets equality. Publishing safe
    # failing evidence must never turn the call or final teardown into PASS.
    # NARROWED CLAIM: with both samples now post-settle, this measures "two
    # post-settle frames are identical". A reflow that lands BEFORE sample 1 is
    # therefore no longer observed here, while motion BETWEEN the two samples is
    # still detected. It is not a "no motion from first paint" guarantee.
    assert measured[0] == measured[1], "consecutive frames changed geometry"


@pytest.mark.asyncio
async def test_capture_pair_settles_before_sampling_both_frames(monkeypatch, tmp_path):
    """Ordered control for the sampling ORDER/contract only.

    The settle helper is stubbed, so this pins that a settle runs before EACH
    sample (once per frame) and fails if it is moved after a sample or dropped.
    SCOPE LIMIT: because the helper is stubbed, this proves placement, never
    completion -- it cannot detect whether the real predicate returns before the
    motion. Design round 3's viewed frames and the published sidecars are the
    separate evidence about the motion itself.
    """
    import scripts.visual_capture as visual_capture

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    order: list[str] = []

    async def fake_settle(_proof, _pilot, **_kwargs) -> None:
        order.append("settle")

    class _OrderPilot:
        async def pause(self) -> None:
            order.append("sample")

    def fake_save_capture(_app, target) -> str:
        order.append("export")
        Path(target).write_text("<svg/>")
        Path(target).with_suffix(".geometry.json").write_text(
            json.dumps(
                {
                    "grid": [100, 30],
                    "css_path": ["local_operator.tcss"],
                    "screen": {"class": "SyntheticOrderControl"},
                    "widgets": [{"region": [0, 0, 100, 30]}],
                }
            )
        )
        return str(target)

    monkeypatch.setattr(sys.modules[__name__], "_await_transcript_scroll_settled", fake_settle)
    monkeypatch.setattr(visual_capture, "save_capture", fake_save_capture)
    proof = SimpleNamespace(
        root=tmp_path / "synthetic-order",
        output=tmp_path / "order",
        app=SimpleNamespace(size=(100, 30)),
        expected_phases={"happy"},
        violations=[],
        case_note="Synthetic ordered control; not rendered evidence.",
    )
    generated = await capture_pair(proof, _OrderPilot(), "host-speed-100x30", "happy")
    assert order == [
        "settle",
        "sample",
        "export",
        "settle",
        "sample",
        "export",
    ]
    assert generated == {
        "host-speed-100x30-happy-post-pause-1.svg",
        "host-speed-100x30-happy-post-pause-1.geometry.json",
        "host-speed-100x30-happy-post-pause-2.svg",
        "host-speed-100x30-happy-post-pause-2.geometry.json",
    }


@pytest.mark.parametrize("fault", ["second-private", "combined-quota", "safe-unstable"])
def test_pair_publication_preserves_privacy_quota_and_strict_failure(tmp_path, monkeypatch, fault):
    """Pure publisher controls only: no App, UI-hook or pixel sensitivity proof."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    root = tmp_path / "synthetic-private"
    output = tmp_path / "pair"
    output.mkdir()
    target = tmp_path / "published"
    case, phase = "ownership-100x30", "happy"
    proof = SimpleNamespace(
        root=root,
        output=output,
        app=SimpleNamespace(size=(100, 30)),
        expected_phases={phase},
        violations=[],
        case_note="Synthetic pure publisher control; not rendered evidence.",
    )
    names = set()
    for index, frame in enumerate(("post-pause-1", "post-pause-2")):
        stem = f"{case}-{phase}-{frame}"
        svg = output / f"{stem}.svg"
        svg.write_text(
            "<svg/>" if fault != "combined-quota" else "<svg>" + " " * (6 * 1024 * 1024) + "</svg>"
        )
        geometry: dict[str, Any] = {
            "grid": [100, 30],
            "css_path": ["local_operator.tcss"],
            "screen": {"class": "SyntheticPublisherControl"},
            "widgets": [{"region": [0, index, 100, 30]}],
        }
        if fault == "second-private" and index == 1:
            geometry["screen"]["private"] = str(root)
        metadata = svg.with_suffix(".geometry.json")
        metadata.write_text(json.dumps(geometry))
        names.update((svg.name, metadata.name))
    if fault == "safe-unstable":
        with pytest.raises(AssertionError, match="consecutive frames changed geometry"):
            publish_proof(proof, case, phase, names, _target=target)
        manifest = json.loads((target / "manifest.json").read_text())
        record = manifest["cases"][case][phase]
        assert record["geometry_equal_at_capture"] is False
        assert record["pair_frames"] == ["post-pause-1", "post-pause-2"]
        assert set(record["files"]) == names
        for name in names:
            data = (target / name).read_bytes()
            assert data == (output / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == record["files"][name]
    else:
        expected_refusal = (
            "private prefix in proof pair"
            if fault == "second-private"
            else "proof pair byte quota exceeded before publication"
        )
        with pytest.raises(AssertionError, match=expected_refusal):
            publish_proof(proof, case, phase, names, _target=target)
        assert not target.exists(), "unsafe or over-quota pair was partially published"


async def replace_inline(pilot, view, value: str) -> None:
    # The row paints an inline buffer, not an Input widget. Exercise its actual
    # end/backspace/printable key handlers rather than writing _buffer directly.
    await pilot.press("end", *(["backspace"] * len(view._buffer)), *tuple(value))


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [100, 60])
async def test_retirement_float_row_real_app(ui_proof, width: int) -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.tui.widgets.editor import Editor
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.conftest import await_session_boot

    proof = ui_proof
    app = proof.app
    case = f"host-speed-{width}x30"
    phases: set[str] = set()
    proof.expected_phases = {"default", "editor", "saved", "reset", "invalid", "closed"}
    proof.case_note = (
        "Only not-a-number validation is painted; negative/max-range cases "
        "assert behavior and config bytes, not separate error frames."
    )
    setting = settings_io.resolve_key(SETTING_KEY)
    assert setting is not None
    assert setting.kind is settings_io.Kind.FLOAT
    assert (setting.default, setting.minimum, setting.maximum) == (0.0, 0.0, 100000.0)
    async with app.run_test(size=(width, 30)) as pilot:
        await await_session_boot(app)
        assert app._session is proof.session
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("/settings")
        await pilot.press("enter")
        view = app.query_one(SettingsView)
        indices = [
            i
            for i, row in enumerate(view._rows)
            if row.kind == "setting" and row.setting and row.setting.key == SETTING_KEY
        ]
        assert len(indices) == 1
        await pilot.press("home")
        # Section headers are not selectable, so a row index is not a number
        # of Down keys. Navigate through real focus/selection events instead.
        for _ in range(len(view._rows)):
            if view.selected_key == SETTING_KEY:
                break
            await pilot.press("down")
        assert view.selected_key == SETTING_KEY
        assert view.has_focus and setting.help
        manager = ConfigManager(proof.config)

        async def shot(phase):
            phases.add(phase)
            await capture_pair(proof, pilot, case, phase)

        assert settings_io.read_setting(manager, setting) == 0.0
        await shot("default")
        await pilot.press("enter")
        assert view.editing_key == SETTING_KEY
        await replace_inline(pilot, view, "25")
        await shot("editor")
        await pilot.press("enter")
        assert view.editing_key is None and not view.error_text
        assert settings_io.read_setting(ConfigManager(proof.config), setting) == 25.0
        await shot("saved")
        await pilot.press("r")
        assert settings_io.read_setting(ConfigManager(proof.config), setting) == 0.0
        await shot("reset")
        config_path = proof.config / "config.yml"
        before = config_path.read_bytes()
        for invalid in ("-1", "100001", "not-a-number"):
            if view.editing_key is None:
                await pilot.press("enter")
            await replace_inline(pilot, view, invalid)
            await pilot.press("enter")
            assert view.editing_key == SETTING_KEY and view.error_text
            assert config_path.read_bytes() == before
        await shot("invalid")
        await pilot.press("escape")
        assert view.editing_key is None and view.selected_key == SETTING_KEY
        await pilot.press("escape")
        assert app._settings_view is None and editor.has_focus
        assert config_path.read_bytes() == before
        await shot("closed")
        assert phases == proof.expected_phases
