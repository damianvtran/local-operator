"""C1 of remote files parity: the owner's referenced-digest scan is memoised and streamed.

docs/design/remote-files-parity.md §0 (spike) and §4 "RAM: paths that must stream".
``files._resolve_source`` authorises EVERY phase of a pull by the session's referenced
digest set; before C1 each phase re-read the whole transcript, so a 32 MiB pull (65
phases) read a 64 MiB journal 65 times on the owner. The real two-relay loopback pins
the count at ONE read per pull; the unit cells pin the window/overlap and invalidation.
"""

from __future__ import annotations

import builtins
import json
import os
import pathlib
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import sync
from local_operator.session.attachments import AttachmentStore
from tests.unit.network.test_mesh_file_pull import PEER, mesh  # noqa: F401 — fixture
from tests.unit.network.test_mobility import SESSION, Devices, devices, pair  # noqa: F401


def _count_transcript_reads(monkeypatch: pytest.MonkeyPatch, counter: list[int]) -> None:
    """Count every whole read (base: ``read_bytes``) and every open (C1: windows)."""
    original_read_bytes = pathlib.Path.read_bytes
    original_open = builtins.open

    def read_bytes(self: Path) -> bytes:
        if self.name == "transcript.jsonl":
            counter[0] += 1
        return original_read_bytes(self)

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(str(file)).name == "transcript.jsonl":
            counter[0] += 1
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    monkeypatch.setattr(sync, "open", counting_open, raising=False)


def test_a_32_mib_pull_scans_the_owner_transcript_once(
    mesh: Devices,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, _server_b, _host, _port = mesh
    payload = os.urandom(32 * 1024 * 1024)
    stored = AttachmentStore(Path(server_a.root) / "attachments").put_bytes(payload, "image/png")
    assert stored is not None
    transcript = Path(server_a.root) / "sessions" / SESSION / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "e9", "content": [{"attachment": stored.digest}]}) + "\n")
    counter = [0]
    _count_transcript_reads(monkeypatch, counter)
    sync._REFERENCE_MEMO.clear()

    capsys.readouterr()
    rc = net_cli.main(
        Namespace(
            network_command="files",
            files_command="pull",
            peer=PEER,
            session=SESSION,
            source=f"attachment:{stored.digest}",
            out=str(tmp_path / "big.png"),
            force=False,
            json=True,
        )
    )
    receipt = json.loads(capsys.readouterr().out)
    assert rc == 0, receipt
    assert (tmp_path / "big.png").stat().st_size == len(payload)
    # 1 stat phase + 64 chunks of 512 KiB on base = 65 scans; memoised = 1.
    assert counter[0] == 1, f"the owner scanned its transcript {counter[0]} times for one pull"


def test_a_reference_straddling_a_window_boundary_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "_SCAN_WINDOW", 64)
    sync._REFERENCE_MEMO.clear()
    digests = [f"{index:032x}" for index in range(40)]
    rows = "".join(
        json.dumps({"id": f"e{i}", "pad": "x" * (i * 7), "content": [{"attachment": d}]}) + "\n"
        for i, d in enumerate(digests)
    )
    (tmp_path / "transcript.jsonl").write_text(rows, encoding="utf-8")
    assert sync.referenced_attachments_in(tmp_path) == sorted(digests)


def test_the_memo_follows_an_append_and_is_bounded(tmp_path: Path) -> None:
    sync._REFERENCE_MEMO.clear()
    transcript = tmp_path / "transcript.jsonl"
    first, second = "a" * 32, "b" * 32
    transcript.write_text(json.dumps({"attachment": first}) + "\n", encoding="utf-8")
    assert sync.referenced_attachments_in(tmp_path) == [first]
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"attachment": second}) + "\n")
    assert sync.referenced_attachments_in(tmp_path) == [first, second]
    for index in range(sync._REFERENCE_MEMO_SIZE + 3):
        other = tmp_path / f"s{index}"
        other.mkdir()
        (other / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
        assert sync.referenced_attachments_in(other) == []
    assert len(sync._REFERENCE_MEMO) == sync._REFERENCE_MEMO_SIZE
