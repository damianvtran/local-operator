"""Tests for the browser tool's file transfer: `download` and `upload`.

The two actions share one wire contract with both hosts, so what is asserted here
is the HARNESS half of that contract: the schema delta, the validation, the
approval tier, the capability refusal, and — the important one — that the answer
comes from Python's own inspection of the filesystem rather than from what the
host said. A fake host that reports a file which is not there must FAIL the call,
which is the "prove the test can still fail" requirement discharged for the
check the whole design rests on.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import local_operator.tools.builtin as builtin
from local_operator import browser_files as bf
from local_operator.browser_bridge.backend import HostCapabilities
from local_operator.harness.types import BrowserSurface, ToolContext


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every path this feature composes derives from the config root."""
    root = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return root


class FakeHost:
    """A host client whose `capabilities()` is a script and whose `call()` records.

    Modelled on the real client's surface only: `host`, `capabilities()` and
    ``await call(method, params)``. `on_call` is where a test emulates the HOST
    writing a file, which is what makes the landing/diff path real rather than
    mocked.
    """

    def __init__(
        self,
        *,
        host: str = "extension",
        methods: tuple[str, ...] = ("upload",),
        version: str = "0.1.18",
        capabilities_known: bool = True,
        disabled: tuple[str, ...] = (),
        switches_known: bool = False,
        result: dict[str, Any] | None = None,
        on_call: Any = None,
    ) -> None:
        self.host = host
        self._methods = methods
        self._version = version
        self._capabilities_known = capabilities_known
        self._disabled = disabled
        self._switches_known = switches_known
        self.result = result or {}
        self.on_call = on_call
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(
            methods=self._methods,
            version=self._version,
            capabilities_known=self._capabilities_known,
            disabled=self._disabled,
            switches_known=self._switches_known,
        )

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if self.on_call is not None:
            self.on_call(method, params)
        return dict(self.result)


def _ctx(session_id: str = "sess0001") -> ToolContext:
    ctx = ToolContext(cwd="/tmp", session_id=session_id)
    ctx.browser = BrowserSurface()
    ctx.browser.surface_id = "bridge:7:nonce"
    return ctx


def _surface() -> Any:
    return _ctx().browser


def _params(**kwargs: Any) -> builtin.BrowserParams:
    return builtin.BrowserParams(**kwargs)


def _flow(action: str, host: FakeHost, *, policy: Any = None, **kwargs: Any) -> builtin.ToolResult:
    """Run the REAL download/upload flow against a fake host client.

    The flow functions are called directly (rather than through
    `execute_browser`) because this module is about the policy and the
    post-hoc verification, which live in them; the dispatch that reaches them is
    covered in `test_browser_tool.py` and `tests/unit/browser_bridge/
    test_tool_selection.py`. ``policy`` injects a shrunken cap set the way the
    flow's own signature does — the caps are instance data on a `Policy` exactly
    so a test can shrink one without monkeypatching module globals.
    """
    flow = builtin._browser_download if action == "download" else builtin._browser_upload
    return asyncio.run(flow(**kwargs, client=host, policy=policy))


# --- the schema --------------------------------------------------------------


def test_download_and_upload_are_advertised_actions() -> None:
    assert "download" in builtin.BROWSER_ACTIONS
    assert "upload" in builtin.BROWSER_ACTIONS
    # cmux cannot serve either, and the action list and the degrade check are the
    # same set so they cannot drift.
    assert {"download", "upload"} <= set(builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS)


def test_paths_is_the_only_new_parameter() -> None:
    """The tool-surface ladder's rung 1: one field, two words in a description.

    `download` deliberately takes NO destination (the harness composes it), so a
    caller cannot name a directory a page will write into.
    """
    properties = builtin.BrowserParams.model_json_schema()["properties"]
    assert "paths" in properties
    assert properties["paths"]["type"] == "array"
    assert "download" in builtin.BROWSER_ACTIONS and "upload" in builtin.BROWSER_ACTIONS


def test_upload_needs_at_least_one_path() -> None:
    problem = builtin._validate_browser_args("upload", _params(action="upload", selector="#f"))
    assert "needs paths" in problem


def test_upload_refuses_an_empty_path_entry() -> None:
    problem = builtin._validate_browser_args(
        "upload", _params(action="upload", selector="#f", paths=["ok.pdf", "   "])
    )
    assert "non-empty" in problem


def test_paths_entries_must_be_strings() -> None:
    """`list[str]` refuses it at validation, which is the coarse-schema case."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        # Through the model's own validator, which is the path the loop takes, and
        # through a dict so the test states the hostile SHAPE rather than a type
        # error the type checker would (correctly) refuse to compile.
        builtin.BrowserParams.model_validate({"action": "upload", "paths": [1]})


def test_download_accepts_no_selector_and_refuses_a_flag_shaped_one() -> None:
    assert builtin._validate_browser_args("download", _params(action="download")) == ""
    assert (
        builtin._validate_browser_args("download", _params(action="download", selector="#go")) == ""
    )
    assert "flag-shaped" in builtin._validate_browser_args(
        "download", _params(action="download", selector="--all")
    )


def test_upload_escalates_to_exec_and_everything_else_stays_write(monkeypatch) -> None:
    """Upload transmits local bytes to a remote origin: the tier says so.

    It is NOT the protection — one callback serves both tiers today and `--yolo`
    installs none — which is why the policy in `browser_files` runs
    unconditionally. The tier is what a tier-sensitive host will honour.
    """
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "ui_browser_advertisable", lambda: False)
    tool = builtin.build_browser_tool(_ctx())
    assert tool is not None
    assert tool.approval_tier == "write"
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"action": "upload", "paths": ["a.pdf"]}) == "exec"
    assert tool.call_approval_tier({"action": "UPLOAD"}) == "exec"
    for action in ("download", "screenshot", "read", "", "open"):
        assert tool.call_approval_tier({"action": action}) == "write"


# --- the capability refusal --------------------------------------------------


def test_download_on_an_older_extension_sends_the_reader_to_the_update_and_the_switch() -> None:
    """A build that predates the capability: update it, then turn the switch on.

    This is state (a) of the three the record can now express. It used to be the
    only answer for `download` — the old copy said no build could serve it — which
    is why the assertion that the remedy is an UPDATE is paired here with the
    warning that the update alone is not enough: the capability is opt-in, and a
    reader sent to the update without the switch would meet the same refusal for a
    different reason and conclude the update failed.
    """
    host = FakeHost(methods=("upload",), version="0.1.18")
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert result.is_error
    assert host.calls == [], "the refusal must come from the record, not the wire"
    assert "does not provide 'download'" in result.text
    assert "first version that does is 0.1.19" in result.text
    assert "Allow downloads" in result.text
    assert (result.details or {}).get("error_code") == "capability_unsupported"


def test_download_with_the_operators_switch_off_names_the_switch_not_an_update() -> None:
    """State (b): the build CAN serve it and the operator has not enabled it.

    The distinction is the whole point of the `capability_switches` event. Sending
    this user to an update would be a remedy that cannot work (their build is
    current), and the copy must not offer it — asserted here as the ABSENCE of the
    update sentence, not only as the presence of the switch one.
    """
    host = FakeHost(methods=(), version="0.1.19", disabled=("download",), switches_known=True)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert result.is_error
    assert host.calls == [], "a switched-off capability costs no socket call"
    assert "'download' is switched off" in result.text
    assert "Allow downloads" in result.text
    assert "chrome://extensions" in result.text
    assert "No update is involved" in result.text
    assert "first version that does is" not in result.text


def test_upload_with_the_operators_switch_off_names_the_upload_switch() -> None:
    """The same state for the other capability, worded by ITS OWN label.

    A single shared sentence would tell a user looking for "Allow downloads" to
    flip the wrong control, so the labels come from `CAPABILITY_SWITCH_LABEL` and
    this test pins that the mapping is per-method rather than one string.
    """
    host = FakeHost(
        methods=("download",), version="0.1.19", disabled=("upload",), switches_known=True
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=["/tmp/x.pdf"]),
        context=_ctx(),
    )
    assert result.is_error
    assert host.calls == []
    assert "'upload' is switched off" in result.text
    assert "Allow uploads" in result.text
    assert "Allow downloads" not in result.text


def test_a_switched_on_extension_downloads_over_the_wire() -> None:
    """State (c): the switch is on, the method is advertised, the call is SENT.

    The inverse of the two refusals above, and the one that would silently rot:
    a gating change that never lets anything through still passes every
    "refused" assertion in this file.
    """

    def write_it(method: str, params: dict[str, Any]) -> None:
        # The app host's shape: bytes in the directory the harness composed. The
        # extension's shape (a file in the user's Downloads plus a reported path)
        # is exercised by the intake tests below.
        Path(params["dir"], "receipt.pdf").write_bytes(b"%PDF-1.4" + b"\x00" * 32)

    host = FakeHost(
        methods=("download", "upload"),
        version="0.1.19",
        disabled=(),
        switches_known=True,
        result={"armed": True, "url": "https://example.test/export"},
        on_call=write_it,
    )
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error
    assert [call[0] for call in host.calls] == ["download"]
    assert "receipt.pdf" in result.text


def test_upload_on_a_pre_feature_extension_names_the_first_version_that_has_it() -> None:
    host = FakeHost(methods=(), version="0.1.17")
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=["/tmp/x.pdf"]),
        context=_ctx(),
    )
    assert result.is_error
    assert host.calls == []
    assert "0.1.17" in result.text and "0.1.18" in result.text
    assert "update the browser extension" in result.text


def test_upload_on_the_app_host_sends_the_reader_to_the_app() -> None:
    host = FakeHost(host="ui", methods=(), version="0.29.2")
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=["/tmp/x.pdf"]),
        context=_ctx(),
    )
    assert result.is_error
    assert "desktop app" in result.text and "update the desktop app" in result.text


def test_a_current_extension_that_stopped_advertising_is_a_wedge_not_a_version() -> None:
    host = FakeHost(methods=(), version="0.1.18")
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=["/tmp/x.pdf"]),
        context=_ctx(),
    )
    assert "chrome://extensions" in result.text
    assert "0.1.18" in result.text


# --- download: the filesystem is the truth -----------------------------------


def test_download_reports_what_actually_landed(tmp_path: Path) -> None:
    payload = b"%PDF-1.4\n" + b"x" * 40

    def write_it(method: str, params: dict[str, Any]) -> None:
        Path(params["dir"], "receipt.pdf").write_bytes(payload)

    host = FakeHost(
        methods=("download",), result={"files": [], "armed": True, "reason": ""}, on_call=write_it
    )
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert "receipt.pdf" in result.text and "pdf" in result.text
    directory = bf.session_dir("sess0001").parent
    assert str(directory) in result.text
    # The destination is composed by the harness, and it is a private directory.
    assert host.calls[0][1]["dir"].startswith(str(directory))
    facts = (result.details or {}).get("files") or []
    assert facts and facts[0]["bytes"] == len(payload)
    assert len(facts[0]["sha256"]) == 64


def test_download_fails_when_the_host_reports_a_file_that_never_landed() -> None:
    """THE check this design rests on: the host's word is a hint, not evidence."""
    host = FakeHost(
        methods=("download",),
        result={
            "files": [{"name": "receipt.pdf", "path": "/nowhere/receipt.pdf", "bytes": 10}],
            "armed": True,
            "reason": "",
        },
    )
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert result.is_error
    # The intake refuses it, and the sentence has to say the right thing about the
    # right file: nothing was saved (the check this design rests on), the path the
    # host named was not there, and the entry was NOT deleted — a path we cannot
    # corroborate might be the user's own file, so the report must not claim we
    # cleaned it up (the R2/N7 over-reporting rule).
    assert "nothing was saved" in result.text
    assert "not there" in result.text
    assert "NOT deleted" in result.text


def test_download_deletes_executable_content_even_when_the_host_calls_it_a_pdf(
    tmp_path: Path,
) -> None:
    """Content wins over both the name and the host's report."""

    def write_it(method: str, params: dict[str, Any]) -> None:
        Path(params["dir"], "receipt.pdf").write_bytes(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 32)

    host = FakeHost(
        methods=("download",),
        result={
            "files": [{"name": "receipt.pdf", "mime": "application/pdf", "bytes": 40}],
            "armed": True,
            "reason": "",
        },
        on_call=write_it,
    )
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert result.is_error
    assert "executable" in result.text
    # DELETED, not merely reported: nothing executable survives.
    assert bf.snapshot(Path(host.calls[0][1]["dir"])) == {}


def test_download_renames_a_file_the_content_disagrees_with(tmp_path: Path) -> None:
    def write_it(method: str, params: dict[str, Any]) -> None:
        Path(params["dir"], "invoice.zip").write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")

    host = FakeHost(methods=("download",), result={"armed": True, "reason": ""}, on_call=write_it)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert "invoice.pdf" in result.text
    assert "the name said" in result.text
    landed = bf.snapshot(Path(host.calls[0][1]["dir"]))
    assert list(landed) == ["invoice.pdf"]


def test_the_rename_sentence_names_the_type_the_server_declared(tmp_path: Path) -> None:
    """R3: `declared_mime` is wired into the copy, not merely accepted.

    Design §7.4 quotes this sentence as "the server called it …", so the host's
    declared type has to reach the words the model reads. It is a HINT (content
    decides), which is exactly why it is quoted next to the disagreement instead
    of being allowed to change the verdict.
    """

    def write_it(method: str, params: dict[str, Any]) -> None:
        Path(params["dir"], "invoice.zip").write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")

    host = FakeHost(
        methods=("download",),
        result={"files": [{"name": "invoice.zip", "mime": "application/zip"}], "armed": True},
        on_call=write_it,
    )
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert "the server said 'application/zip'" in result.text
    assert "invoice.pdf" in result.text


def test_a_symlink_escaping_the_quarantine_root_loses_only_the_entry(tmp_path: Path) -> None:
    """R1: the refusal deletes the CANDIDATE ENTRY, never the resolved target.

    The target is by construction a path OUTSIDE the root, so unlinking it (what
    the first version did) destroyed the user's own file while leaving the
    escaping entry in the session directory — reported as a refusal, executed as
    silent data loss. Both halves are asserted: the outside file SURVIVES intact
    and the entry is GONE.
    """
    payload = b"%PDF-1.4\n1 0 obj\n%%EOF\n"
    outside = tmp_path / "important.pdf"
    outside.write_bytes(payload)

    def write_it(method: str, params: dict[str, Any]) -> None:
        os.symlink(outside, Path(params["dir"], "receipt.pdf"))

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert result.is_error
    assert "refused and deleted" in result.text
    assert "outside the download directory" in result.text
    # The escape is gone...
    assert list(Path(host.calls[0][1]["dir"]).iterdir()) == []
    # ...and the file outside the root is untouched, bytes and all.
    assert outside.read_bytes() == payload


def test_the_refusal_says_so_when_the_entry_could_not_be_deleted(tmp_path: Path) -> None:
    """The model-visible sentence claims only what happened (R1's second half)."""
    outside = tmp_path / "important.pdf"
    outside.write_bytes(b"%PDF-1.4\n")
    directory: dict[str, Path] = {}

    def write_it(method: str, params: dict[str, Any]) -> None:
        directory["path"] = Path(params["dir"])
        os.symlink(outside, directory["path"] / "receipt.pdf")
        # An unwritable directory is what makes `unlink` fail on a real disk.
        os.chmod(directory["path"], 0o500)

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    try:
        result = _flow(
            "download",
            host,
            tool_call_id="t1",
            state=_surface(),
            params=_params(action="download"),
            context=_ctx(),
        )
    finally:
        os.chmod(directory["path"], 0o700)
    assert result.is_error
    assert "refused, NOT deleted" in result.text
    assert (directory["path"] / "receipt.pdf").is_symlink()
    assert outside.exists()


def test_the_per_call_cap_is_applied_before_anything_is_audited(tmp_path: Path) -> None:
    """R2: a dropped file keeps no `verdict=allow` row naming a path that is gone."""

    def write_it(method: str, params: dict[str, Any]) -> None:
        for index in range(5):
            Path(params["dir"], f"f{index}.pdf").write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
        policy=bf.Policy(download_max_files=3),
    )
    assert not result.is_error, result.text
    assert len((result.details or {}).get("files") or []) == 3
    directory = Path(host.calls[0][1]["dir"])
    assert len(bf.snapshot(directory)) == 3
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    kept = [row for row in rows if row["verdict"] == "allow"]
    dropped = [row for row in rows if row["verdict"] == "deny"]
    assert len(kept) == 3 and len(dropped) == 2
    # Every allow row names a file that is really there, and every dropped file is
    # named by a row that says why.
    assert all(Path(str(row["path"])).exists() for row in kept)
    assert all("per call limit" in str(row["reason"]) for row in dropped)
    # N7/R8: the row answers "is it still on disk?", the same way for every rule.
    assert all("the entry was removed" in str(row["reason"]) for row in dropped)
    assert "files per call limit" in result.text


def test_the_cap_row_says_what_happened_when_the_delete_failed(tmp_path: Path) -> None:
    """R8: the over-cap row carries the delete outcome, not only the cap.

    The remediation reply claimed the over-cap path got the same treatment as the
    containment rule; the sentence did and the audit row did not. The row is what
    a later reader answers "what did this session keep?" from, so the claim is
    made true in the row — and it is exercised in the state where it matters: an
    unwritable session directory, where the unlink really fails and the dropped
    files really are still there.
    """
    directory: dict[str, Path] = {}

    def write_it(method: str, params: dict[str, Any]) -> None:
        directory["path"] = Path(params["dir"])
        for index in range(5):
            Path(params["dir"], f"f{index}.pdf").write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")
        # Readable, not writable: every unlink below fails on a real disk.
        os.chmod(directory["path"], 0o500)

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    try:
        result = _flow(
            "download",
            host,
            tool_call_id="t1",
            state=_surface(),
            params=_params(action="download"),
            context=_ctx(),
            policy=bf.Policy(download_max_files=3),
        )
    finally:
        os.chmod(directory["path"], 0o700)
    assert not result.is_error, result.text
    assert "refused, NOT deleted" in result.text
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    dropped = [row for row in rows if row["verdict"] == "deny"]
    assert len(dropped) == 2
    assert all("per call limit" in str(row["reason"]) for row in dropped)
    assert all("could NOT be removed" in str(row["reason"]) for row in dropped)
    # The rows are true: the two dropped files are still in the session directory.
    assert len(bf.snapshot(directory["path"])) == 5


def test_the_session_ceiling_refuses_before_anything_is_armed(tmp_path: Path) -> None:
    """R5: the 2 GB ceiling bounds the NEXT call, and refuses without arming."""
    session = bf.session_dir("sess0001")
    (session / "already.pdf").write_bytes(b"x" * 64)
    host = FakeHost(methods=("download",), result={"armed": True})
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
        policy=bf.Policy(download_max_session_bytes=16),
    )
    assert result.is_error
    assert "already downloaded more than 16 bytes" in result.text
    assert host.calls == []


def test_a_kept_artifact_is_tightened_to_0600(tmp_path: Path) -> None:
    """Q-2: §4.1's "files 0600" is enforced on the artifact the tool reports."""

    def write_it(method: str, params: dict[str, Any]) -> None:
        path = Path(params["dir"], "receipt.pdf")
        path.write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")
        # What an Electron/Chromium write lands by umask.
        path.chmod(0o644)

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    landed = Path(host.calls[0][1]["dir"]) / "receipt.pdf"
    assert stat.S_IMODE(landed.stat().st_mode) == 0o600


def test_the_mode_change_lands_on_the_entry_not_on_a_symlink_target(tmp_path: Path) -> None:
    """N8: `os.chmod` follows a link, so the entry it names must be lchmod'ed.

    An in-root symlink is a legitimate candidate (containment passes, because its
    target is inside the root), and the old `os.chmod` tightened the TARGET — a
    file this call did not land, does not report as an artifact, and did not
    choose: the page wrote the link. The target here sits in a subdirectory so it
    is not itself a candidate, which keeps the two modes independently observable.

    Platform-shaped on purpose (review round 4, item 3): a symlink's own mode is
    NOT settable portably — Linux has no `lchmod` — so what is asserted on every
    platform is the part that matters, that the TARGET is untouched, and the
    entry's 0600 is asserted only where the platform can do it. The Linux branch
    is executed rather than assumed by
    `test_a_symlink_entry_without_lchmod_reports_that_it_could_not_be_tightened`.
    """

    def write_it(method: str, params: dict[str, Any]) -> None:
        directory = Path(params["dir"])
        (directory / "sub").mkdir()
        target = directory / "sub" / "target.pdf"
        target.write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")
        target.chmod(0o644)
        os.symlink(target, directory / "receipt.pdf")

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    directory = Path(host.calls[0][1]["dir"])
    entry = directory / "receipt.pdf"
    assert stat.S_IMODE((directory / "sub" / "target.pdf").stat().st_mode) == 0o644
    if hasattr(os, "lchmod"):
        assert stat.S_IMODE(os.lstat(entry).st_mode) == 0o600
        assert "could not tighten the mode" not in result.text
    else:
        # Linux: the mode could not be set, and the result SAYS so rather than
        # implying a 0600 that is not there.
        assert "could not tighten the mode of receipt.pdf" in result.text


def test_a_symlink_entry_without_lchmod_reports_that_it_could_not_be_tightened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LINUX branch, executed on macOS by removing the attribute.

    macOS cannot reproduce the CI failure by itself, so the branch is driven
    rather than reasoned about: with `os.lchmod` gone (as on Linux),
    `chmod_private` must return False for a symlink entry, must NOT fall back to
    `chmod` (that is the N8 bug — it would tighten whatever the link points at),
    and the download result must carry the caveat instead of asserting a mode the
    harness never set.
    """
    monkeypatch.delattr(os, "lchmod", raising=False)

    def write_it(method: str, params: dict[str, Any]) -> None:
        directory = Path(params["dir"])
        (directory / "sub").mkdir()
        target = directory / "sub" / "target.pdf"
        target.write_bytes(b"%PDF-1.4\n1 0 obj\n%%EOF\n")
        target.chmod(0o644)
        os.symlink(target, directory / "receipt.pdf")

    host = FakeHost(methods=("download",), result={"armed": True}, on_call=write_it)
    result = _flow(
        "download",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    directory = Path(host.calls[0][1]["dir"])
    assert bf.chmod_private(directory / "receipt.pdf") is False
    assert "could not tighten the mode of receipt.pdf" in result.text
    # The target is untouched — the whole point of refusing rather than falling back.
    assert stat.S_IMODE((directory / "sub" / "target.pdf").stat().st_mode) == 0o644


# --- upload: the gate runs before anything reaches a browser -----------------


def _uploadable(tmp_path: Path, name: str = "deck.pptx") -> Path:
    path = tmp_path / name
    path.write_bytes(b"PK\x03\x04fake deck payload")
    return path


def test_upload_refuses_a_credential_file_without_calling_the_host(tmp_path: Path) -> None:
    key = tmp_path / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True)
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    host = FakeHost(methods=("upload",))
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(key)]),
        context=_ctx(),
    )
    assert result.is_error
    assert "credential deny-list" in result.text
    assert host.calls == [], "the gate is unconditional and runs before the wire"


def test_upload_refuses_a_file_inside_the_config_root(
    tmp_path: Path, isolated_config: Path
) -> None:
    isolated_config.mkdir(parents=True, exist_ok=True)
    secret = isolated_config / "config.yml"
    secret.write_text("token: hunter2\n")
    host = FakeHost(methods=("upload",))
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(secret)]),
        context=_ctx(),
    )
    assert result.is_error
    assert "config directory" in result.text
    assert host.calls == []


def test_upload_refuses_the_whole_call_when_one_path_is_refused(tmp_path: Path) -> None:
    good = _uploadable(tmp_path)
    bad = tmp_path / ".env"
    bad.write_text("SECRET=1\n")
    host = FakeHost(methods=("upload",))
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(good), str(bad)]),
        context=_ctx(),
    )
    assert result.is_error
    assert "nothing was attached" in result.text
    assert host.calls == []


def test_upload_refuses_more_files_than_the_cap(tmp_path: Path) -> None:
    paths = [
        str(_uploadable(tmp_path, f"file-{index}.pdf")) for index in range(bf.UPLOAD_MAX_FILES + 1)
    ]
    host = FakeHost(methods=("upload",))
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=paths),
        context=_ctx(),
    )
    assert result.is_error
    assert "files per call limit" in result.text
    assert host.calls == []


def test_upload_sends_resolved_paths_and_reports_the_read_back(tmp_path: Path) -> None:
    deck = _uploadable(tmp_path)
    size = deck.stat().st_size
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": size}],
            "refused": [],
            "accept": ".pdf",
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    method, params = host.calls[0]
    assert method == "upload"
    assert params["paths"] == [str(deck.resolve())]
    assert params["selector"] == "#f"
    assert "attached 1 file(s)" in result.text
    # `accept=` is REPORTED and never obeyed.
    assert "accept='.pdf'" in result.text
    facts = (result.details or {}).get("files") or []
    assert f"sha256 {facts[0]['sha256'][:12]}" in result.text


def test_upload_fails_when_the_dom_holds_something_else(tmp_path: Path) -> None:
    """A filled-looking input that ignored the attach must not read as success."""
    deck = _uploadable(tmp_path)
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": 1}],
            "refused": [],
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert result.is_error
    assert "did not take the attach" in result.text


def test_upload_surfaces_a_host_side_refusal(tmp_path: Path) -> None:
    deck = _uploadable(tmp_path)
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": [],
            "accepted": [],
            "refused": [{"path": "deck.pptx", "reason": "refused: not an absolute path"}],
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert result.is_error
    assert "not an absolute path" in result.text


def test_the_audit_file_records_both_directions(tmp_path: Path) -> None:
    deck = _uploadable(tmp_path)
    size = deck.stat().st_size
    downloader = FakeHost(
        methods=("download",),
        result={"armed": True, "reason": ""},
        on_call=lambda method, params: Path(params["dir"], "receipt.pdf").write_bytes(
            b"%PDF-1.4\n"
        ),
    )
    _flow(
        "download",
        downloader,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="download"),
        context=_ctx(),
    )
    uploader = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"path": str(deck.resolve()), "bytes": size}],
            "refused": [],
        },
    )
    _flow(
        "upload",
        uploader,
        tool_call_id="t2",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    assert [row["action"] for row in rows] == ["download", "upload"]
    assert rows[0]["host"] == "extension" and rows[0]["verdict"] == "allow"
    assert len(rows[0]["sha256"]) == 64
    assert rows[1]["name"] == "deck.pptx"


# --- the lost read-back, the stale bridge, and an absent client --------------


def test_an_attach_whose_read_back_was_lost_to_a_navigation_is_reported_unverified(
    tmp_path: Path,
) -> None:
    """Q-1: the bytes went; a bare CDP internal error here invites a double send.

    A form that submits itself from its `change` handler destroys the execution
    context the read-back runs in — while the attach has already happened and the
    files have already reached the server. The reported result must therefore be
    `accepted` facts plus an explicit unverified marker, with the audit row kept,
    and Python's own stat + digest (which the page cannot touch) is what the facts
    are built from.
    """
    deck = _uploadable(tmp_path)
    marker = (
        "unavailable — the page navigated out of the change event before the input "
        "could be read back"
    )
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [
                {
                    "name": "deck.pptx",
                    "path": str(deck.resolve()),
                    "bytes": -1,
                    "mime": "",
                    "sniffed": "",
                    "sha256": "",
                }
            ],
            "refused": [],
            "accept": "",
            "readback": marker,
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert "attached 1 file(s)" in result.text
    assert "could not be read back" in result.text
    assert "before re-sending" in result.text
    facts = (result.details or {}).get("files") or []
    assert len(facts) == 1
    assert len(str(facts[0]["sha256"])) == 64
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["action"] == "upload" and rows[0]["verdict"] == "allow"
    assert rows[0]["sha256"] == facts[0]["sha256"]
    assert "navigated" in str(rows[0]["reason"])


def test_the_unverified_marker_is_not_a_free_bypass(tmp_path: Path) -> None:
    """Without the marker, a read-back that reports nothing still fails the call."""
    deck = _uploadable(tmp_path)
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": -1}],
            "refused": [],
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert result.is_error
    assert "did not take the attach" in result.text


def test_the_marker_does_not_suppress_a_reported_mismatch(tmp_path: Path) -> None:
    """R6: the comparison is gated on the -1 sentinel, never on the marker.

    The marker means "I could not read it back"; it must never mean "do not
    check". A host that reports BOTH a real count and a marker used to skip the
    one comparison a page that ignored the attach is caught by — proven here with
    the same 8-byte file and the same host result as the failing case above,
    differing only in the marker.
    """
    deck = _uploadable(tmp_path)
    marker = (
        "unavailable — the page navigated out of the change event before the input "
        "could be read back"
    )
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": 999999}],
            "refused": [],
            "accept": "",
            "readback": marker,
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert result.is_error, result.text
    assert "did not take the attach" in result.text
    assert "999999" in result.text
    # No facts and no audit row for a call nothing was sent by — the same answer
    # the marker-less mismatch gives.
    assert not (result.details or {}).get("files")
    # A refused call writes no row at all: nothing was sent, and the trail says so
    # by being empty rather than by carrying an `allow` row for a call that failed.
    audit_path = bf.downloads_root() / bf.AUDIT_FILENAME
    assert not audit_path.exists() or audit_path.read_text().strip() == ""


def test_the_host_marker_is_sanitised_and_capped_before_the_transcript(
    tmp_path: Path,
) -> None:
    """R7: the marker is a string from OUTSIDE, so it gets the same door as the type.

    A marker carrying `\r\n` used to grow the tool result by a line the HOST
    chose, and a bidi override DISPLAYS one string while the bytes say another.
    Both are stripped, and what survives is capped, before it reaches the note or
    the audit row.
    """
    deck = _uploadable(tmp_path)
    injected = "\u202e" + "fake note" + "x" * 400
    marker = f"unavailable — the read-back was lost\r\n[injected line] {injected}"
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": -1}],
            "refused": [],
            "accept": "",
            "readback": marker,
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert "could not be read back" in result.text
    # The host's newline cannot start a line, and the override cannot travel.
    assert "\n[injected line]" not in result.text
    assert "\r" not in result.text
    assert "\u202e" not in result.text
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    reason = str(rows[0]["reason"])
    assert "\r" not in reason and "\n" not in reason
    assert "\u202e" not in reason
    assert len(reason.encode("utf-8")) <= bf.MAX_READBACK_BYTES
    # And the fact itself is marked unverified, so a consumer of `details` sees
    # the same caveat the prose carries (N6).
    facts = (result.details or {}).get("files") or []
    assert facts and facts[0]["verified"] is False


def test_a_marker_that_sanitises_away_still_marks_the_attach_unverified(
    tmp_path: Path,
) -> None:
    """R7's edge: the marker's PRESENCE is a fact, its TEXT is untrusted.

    A host that sends a marker made only of control characters is still saying
    its read failed. Sanitising cannot be allowed to turn that into a VERIFIED
    attach — the note is what a model re-checks from, and "no detail" is also the
    extension's own fallback for an empty error detail.
    """
    deck = _uploadable(tmp_path)
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": -1}],
            "refused": [],
            "accept": "",
            "readback": "\r\n\u202e",
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert "could not be read back (no detail)" in result.text
    facts = (result.details or {}).get("files") or []
    assert facts and facts[0]["verified"] is False
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    assert str(rows[0]["reason"]) == "no detail"


@pytest.mark.parametrize(
    "count",
    [None, "abc", {}, "12.5", 12.5, "--12", "++5", "+-3", "\u00b2", "9" * 4301],
    ids=[
        "null",
        "non-numeric-string",
        "dict",
        "float-shaped-string",
        "json-float",
        "double-sign",
        "double-plus",
        "plus-minus",
        "superscript",
        "4301-digits",
    ],
)
@pytest.mark.parametrize("marker", [False, True], ids=["no-marker", "marker"])
def test_a_malformed_host_byte_count_never_raises_out_of_the_tool(
    tmp_path: Path, count: Any, marker: bool
) -> None:
    """Q-1 / round 4: a count the host typed wrong is a typed answer, not an exception.

    `bytes` is a field the host chooses the type of, and `int()` on it raised
    straight out of `_browser_upload`, so the loop's generic handler turned a
    policy answer into `Tool raised: ...` plus a warning traceback. The shapes are
    driven through the REAL flow, in both marker shapes: with no marker the call
    is REFUSED naming what the host sent, and with a marker it stays the
    unverified attach it already was — a refusal there would say "the file input
    did not take the attach" over bytes the host reported setting, which is the
    double-send harm round 1's Q-1 exists to prevent.

    The last four ids are round 4's: a sign the coercion did not strip (`--12`,
    `++5`, `+-3` all reached `int()`), a Unicode digit `str.isdigit()` accepts and
    `int()` rejects (`\u00b2`), and a digit string past CPython's ~4300-digit
    `int()` limit. Every one of them escaped as `Tool raised:` before the guard.
    """
    deck = _uploadable(tmp_path)
    result = _flow(
        "upload",
        FakeHost(
            methods=("upload",),
            result={
                "inputs": ["#f"],
                "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": count}],
                "refused": [],
                "accept": "",
                "readback": "unavailable — the read-back was lost" if marker else "",
            },
        ),
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    if not marker:
        assert result.is_error, result.text
        assert "malformed byte count" in result.text
        # A PREFIX of the offending value: the label is sanitised and capped like
        # every other host-supplied string, so the 4301-digit shape arrives
        # clipped with an ellipsis rather than in full.
        assert repr(count)[:40] in result.text
        assert not (result.details or {}).get("files")
        return
    assert not result.is_error, result.text
    assert "malformed" in result.text
    facts = (result.details or {}).get("files") or []
    assert facts and facts[0]["verified"] is False
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    assert "malformed" in str(rows[0]["reason"])


def test_the_honest_composed_marker_survives_intact(tmp_path: Path) -> None:
    """Q-2: the marker the extension really composes is not clipped.

    `upload.ts` builds `"unavailable — the read-back failed (" + describeError(e) +
    ")"`, and `describeError` caps the ERROR TEXT at 120 characters — so the
    composed marker is ~159 bytes, above round 2's 120-byte ceiling, which dropped
    the closing paren and the tail of the diagnostic on exactly the branch that
    carries it. Measured here through the real flow, on the composed shape.
    """
    deck = _uploadable(tmp_path)
    marker = "unavailable — the read-back failed (" + "net::ERR_ABORTED " + "d" * 100 + ")"
    assert len(marker.encode("utf-8")) > 120
    assert len(marker.encode("utf-8")) <= bf.MAX_READBACK_BYTES
    result = _flow(
        "upload",
        FakeHost(
            methods=("upload",),
            result={
                "inputs": ["#f"],
                "accepted": [{"name": "deck.pptx", "path": str(deck.resolve()), "bytes": -1}],
                "refused": [],
                "accept": "",
                "readback": marker,
            },
        ),
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    assert f"could not be read back ({marker})" in result.text
    rows = [
        json.loads(line)
        for line in (bf.downloads_root() / bf.AUDIT_FILENAME).read_text().splitlines()
    ]
    assert rows[0]["reason"] == marker


def test_a_clipped_marker_says_that_it_was_clipped() -> None:
    """Q-2: the ceiling still bounds a hostile host, and a cut is visible.

    No fixed ceiling can bound an honest marker whose error text is multibyte
    (the extension caps 120 CHARACTERS), so what must hold is that a clipped value
    never reads as the whole one: the tail goes, the head stays, and an ellipsis
    says so — where `_truncate_bytes` would have kept a fragment of the tail.
    """
    label = bf.readback_label("unavailable — the read-back failed (" + "e" * 500 + ")")
    assert label.endswith(bf.CLIP_MARK)
    assert len(label.encode("utf-8")) <= bf.MAX_READBACK_BYTES
    assert "e" * 500 not in label
    # Under the ceiling nothing is added, and an absent marker stays absent.
    assert bf.readback_label("unavailable — the read-back was lost") == (
        "unavailable — the read-back was lost"
    )
    assert bf.readback_label("") == ""


def test_an_ordinary_attach_is_marked_verified_in_details(tmp_path: Path) -> None:
    """N6: the structured result distinguishes a verified attach from an unverified one."""
    deck = _uploadable(tmp_path)
    host = FakeHost(
        methods=("upload",),
        result={
            "inputs": ["#f"],
            "accepted": [
                {
                    "name": "deck.pptx",
                    "path": str(deck.resolve()),
                    "bytes": deck.stat().st_size,
                }
            ],
            "refused": [],
            "accept": "",
        },
    )
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=[str(deck)]),
        context=_ctx(),
    )
    assert not result.is_error, result.text
    facts = (result.details or {}).get("files") or []
    assert facts and facts[0]["verified"] is True
    assert "could not be read back" not in result.text


def test_a_bridge_that_predates_the_advertisement_sends_the_reader_to_a_restart() -> None:
    """R4: an empty capability list from an OLD BRIDGE is not an extension fault.

    Design §6.4's "new harness + old daemon" row promises the remedy names
    restarting the bridge. Toggling the extension cannot help — that daemon never
    asked it what it can do — and the record's own `capabilities_known` stamp is
    the only thing that separates the two causes.
    """
    host = FakeHost(methods=(), version="0.1.18", capabilities_known=False)
    result = _flow(
        "upload",
        host,
        tool_call_id="t1",
        state=_surface(),
        params=_params(action="upload", selector="#f", paths=["/tmp/whatever"]),
        context=_ctx(),
    )
    assert result.is_error
    assert "lop browser restart" in result.text
    assert "toggle" not in result.text
    # Decided from the record: no socket call is spent on a refusal.
    assert host.calls == []


def test_an_absent_client_is_a_typed_refusal_not_an_attribute_error() -> None:
    """N3: the one function that must answer before touching a socket never raises."""
    result = asyncio.run(
        builtin._browser_download(
            tool_call_id="t1",
            state=_surface(),
            params=_params(action="download"),
            context=_ctx(),
            client=None,
        )
    )
    assert result.is_error
    # "No browser is attached" is the right answer for an absent client even on a
    # method no host ever advertised: there is no peer, so no version and no switch
    # answer exists to attribute. The N3 point is the TYPE — a `capability_refusal`
    # built from an empty record, never an AttributeError out of the one function
    # that must answer before touching a socket.
    assert "no browser is attached" in result.text
    assert (result.details or {}).get("error_code") == "capability_unsupported"
