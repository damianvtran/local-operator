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
        result: dict[str, Any] | None = None,
        on_call: Any = None,
    ) -> None:
        self.host = host
        self._methods = methods
        self._version = version
        self._capabilities_known = capabilities_known
        self.result = result or {}
        self.on_call = on_call
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(self._methods, self._version, self._capabilities_known)

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


def test_download_on_the_extension_host_is_refused_without_a_socket_call() -> None:
    """No extension build can serve it, so the copy must not send the user to an
    update that cannot help (measured: Chrome refuses the browser-level download
    commands to an extension's debugger session)."""
    host = FakeHost(methods=("upload",))
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
    assert "browser extension cannot serve 'download'" in result.text
    assert "desktop app" in result.text
    assert "bash + curl" in result.text
    assert (result.details or {}).get("error_code") == "capability_unsupported"


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
    assert "no download started" in result.text
    assert "receipt.pdf" not in result.text


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
    assert "files per call limit" in result.text


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
    assert "cannot serve" in result.text
