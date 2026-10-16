"""The pure scanner: which journal rows become files, and which must not.

Each test builds rows in the REAL shape (``_fixtures``) and asserts the records. The negative
controls are the point: a failed command, an upload receipt, a denied path and a pruned row
without ``details.artifacts`` must each produce exactly what the design says, and the parse
count test pins the cost discipline structurally (no wall-clock bound to flake).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.session_files import scan as scan_module
from local_operator.session_files.scan import FileScan, resolve_path

from ._fixtures import assistant, call, generate_image_rows, line, tool

DIGEST = "a" * 32
OTHER = "b" * 32
CWD, HOME = "/work/proj", "/home/u"


def _scan(rows: list[dict[str, Any]], **kwargs: Any) -> FileScan:
    scan = FileScan(cwd=kwargs.pop("cwd", CWD), home=kwargs.pop("home", HOME), **kwargs)
    scan.feed_all([line(row) for row in rows])
    return scan


def test_write_and_edit_record_the_path_with_origin() -> None:
    scan = _scan(
        [
            assistant("a", 1, [call("c1", "write", path="/work/proj/a.md", i="notes")]),
            tool(
                "t1", 2, "c1", "write", "ok", {"path": "/work/proj/a.md", "added": 3, "removed": 0}
            ),
            assistant("a2", 3, [call("c2", "edit", path="/work/proj/b.py")]),
            tool(
                "t2", 4, "c2", "edit", "ok", {"path": "/work/proj/b.py", "added": 1, "removed": 1}
            ),
        ]
    )
    by_name = {r.name: r for r in scan.records.values()}
    assert by_name["a.md"].origin == "written" and by_name["a.md"].intent == "notes"
    assert by_name["a.md"].kind == "markdown"
    assert by_name["b.py"].origin == "edited" and by_name["b.py"].kind == "code"
    assert by_name["a.md"].first_seq == 1 and by_name["a.md"].last_ts == 2


def test_bash_outputs_resolve_relative_to_cwd_and_pair_with_the_result() -> None:
    scan = _scan(
        [
            assistant("a", 1, [call("c1", "bash", command="convert a.png out/b.png > report.txt")]),
            tool("t1", 2, "c1", "bash", "exit code: 0\n--- stdout ---\n"),
        ]
    )
    paths = {r.path for r in scan.records.values()}
    assert paths == {
        "/work/proj/report.txt"
    }  # `>` is a recognised output door; `convert` args are not
    assert next(iter(scan.records.values())).origin == "command"


def test_failed_bash_wrote_nothing() -> None:
    """Both failure spellings: ``is_error`` and a non-zero ``exit code:`` line."""
    for failure in (
        tool("t1", 2, "c1", "bash", "boom", is_error=True),
        tool("t1", 2, "c1", "bash", "exit code: 2\n--- stdout ---\n"),
        tool("t1", 2, "c1", "bash", "TIMEOUT after 5s (process killed)\nexit code: 137\n"),
    ):
        scan = _scan(
            [assistant("a", 1, [call("c1", "bash", command="echo hi > out.txt")]), failure]
        )
        assert scan.records == {}, failure["payload"]["content"]
    ok = _scan(
        [
            assistant("a", 1, [call("c1", "bash", command="echo hi > out.txt")]),
            tool("t1", 2, "c1", "bash", "exit code: 0\n--- stdout ---\nhi"),
        ]
    )
    assert list(ok.records) == ["p:/work/proj/out.txt"]


def test_unpaired_bash_result_adds_nothing() -> None:
    scan = _scan([tool("t1", 2, "ghost", "bash", "exit code: 0\n")])
    assert scan.records == {}


def test_errored_write_is_not_a_file() -> None:
    scan = _scan(
        [
            assistant("a", 1, [call("c1", "write", path="/work/proj/a.md")]),
            tool("t1", 2, "c1", "write", "denied", {"path": "/work/proj/a.md"}, is_error=True),
        ]
    )
    assert scan.records == {}


def test_browser_screenshot_and_download_but_not_upload_receipt() -> None:
    scan = _scan(
        [
            tool(
                "t1", 1, "c1", "browser", "Screenshot saved", {"path": "/tmp/s.png", "bytes": 900}
            ),
            tool(
                "t2",
                2,
                "c2",
                "browser",
                "downloaded",
                {
                    "files": [{"path": "/Users/u/Downloads/r.pdf", "bytes": 12}],
                    "directory": "/Users/u/Downloads",
                },
            ),
            # An upload ATTACH receipt names files the model SENT - not ones it produced.
            tool(
                "t3",
                3,
                "c3",
                "browser",
                "attached",
                {"files": [{"path": "/Users/u/secret.txt", "bytes": 5}]},
            ),
        ]
    )
    got = {r.path: (r.origin, r.size) for r in scan.records.values()}
    assert got == {"/tmp/s.png": ("screenshot", 900), "/Users/u/Downloads/r.pdf": ("download", 12)}


def test_generate_image_old_row_uses_content_block() -> None:
    scan = _scan(generate_image_rows(1, 10.0, DIGEST))
    record = scan.records[f"a:{DIGEST}"]
    assert (record.origin, record.tool, record.kind) == ("generated", "generate_image", "image")
    assert (record.name, record.width, record.height, record.size) == ("flux-dev-01.png", 8, 6, 72)
    assert record.path is None and record.digest == DIGEST


def test_pruned_generate_image_row_survives_through_details_artifacts() -> None:
    """The latent hole the design found: pruning blanks ``content`` but keeps ``details``."""
    without = _scan(generate_image_rows(1, 10.0, DIGEST, pruned=True))
    assert (
        without.records == {}
    ), "a pruned OLD row has nothing left to derive from (documented limit)"
    with_details = _scan(generate_image_rows(1, 10.0, DIGEST, pruned=True, with_artifacts=True))
    assert list(with_details.records) == [f"a:{DIGEST}"]
    assert with_details.records[f"a:{DIGEST}"].name == "flux-dev-01.png"


def test_one_result_is_never_counted_twice() -> None:
    """details.artifacts AND the content block name the same digest: one record, one touch."""
    scan = _scan(generate_image_rows(1, 10.0, DIGEST, with_artifacts=True))
    assert len(scan.records) == 1 and scan.records[f"a:{DIGEST}"].touches == 1


def test_regenerating_the_same_bytes_is_one_file_touched_twice() -> None:
    scan = _scan(generate_image_rows(1, 10.0, DIGEST) + generate_image_rows(2, 20.0, DIGEST))
    record = scan.records[f"a:{DIGEST}"]
    assert record.touches == 2 and record.first_ts == 10.5 and record.last_ts == 20.5


def test_video_receipts_from_status_job_result_and_old_digest_only() -> None:
    new = {
        "job": "j1",
        "digest": DIGEST,
        "digests": [DIGEST],
        "artifacts": [
            {
                "digest": DIGEST,
                "kind": "video",
                "name": "wan-01.mp4",
                "content_type": "video/mp4",
                "size_bytes": 5000,
                "width": 640,
                "height": 360,
                "duration_s": 5.0,
            }
        ],
        "content_type": "video/mp4",
    }
    old = {
        "job_id": "j2",
        "digest": OTHER,
        "digests": [OTHER],
        "content_type": "video/mp4",
        "duration_s": 8.0,
    }
    job_result = {
        "id": "jr",
        "ts": 5.0,
        "type": "message",
        "payload": {
            "kind": "custom",
            "custom_type": "job_result",
            "details": {"job_id": "j1", **new},
        },
    }
    scan = _scan(
        [
            tool("t1", 1, "c1", "video_status", "Video ready", new),
            tool("t2", 2, "c2", "video_status", "Video ready", old),
            job_result,
        ]
    )
    video = scan.records[f"a:{DIGEST}"]
    assert (video.kind, video.name, video.duration_s, video.width) == (
        "video",
        "wan-01.mp4",
        5.0,
        640,
    )
    assert video.touches == 2  # status row + the background job_result row
    legacy = scan.records[f"a:{OTHER}"]
    assert (legacy.kind, legacy.duration_s, legacy.name.startswith("video-")) == (
        "video",
        8.0,
        True,
    )


def test_job_result_without_digest_is_not_even_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    parses: list[int] = []
    real = scan_module._row
    monkeypatch.setattr(scan_module, "_row", lambda raw: parses.append(1) or real(raw))
    chatter = {
        "id": "jr",
        "ts": 1.0,
        "type": "message",
        "payload": {
            "kind": "custom",
            "custom_type": "job_result",
            "details": {"job_id": "j", "text": "done"},
        },
    }
    _scan([chatter])
    assert parses == []


def test_denied_paths_never_become_records_and_are_cached_as_denied() -> None:
    calls: list[str] = []

    def deny(path: str) -> bool:
        calls.append(path)
        return path.endswith(".env")

    rows = []
    for index in range(3):
        rows += [
            assistant(f"a{index}", index, [call(f"c{index}", "write", path="/work/proj/.env")]),
            tool(f"t{index}", index + 0.5, f"c{index}", "write", "ok", {"path": "/work/proj/.env"}),
        ]
    scan = _scan(rows, deny=deny)
    assert scan.records == {}
    assert calls == [
        "/work/proj/.env"
    ], "the verdict is memoised: one denylist call per distinct path"


def test_origin_precedence_keeps_the_strongest_evidence() -> None:
    rows = [
        assistant("a", 1, [call("c1", "write", path="/work/proj/x.csv")]),
        tool("t1", 2, "c1", "write", "ok", {"path": "/work/proj/x.csv"}),
        assistant("a2", 3, [call("c2", "bash", command="echo > x.csv")]),
        tool("t2", 4, "c2", "bash", "exit code: 0"),
    ]
    record = _scan(rows).records["p:/work/proj/x.csv"]
    assert (record.origin, record.touches, record.last_ts) == ("written", 2, 4)
    flipped = _scan(rows[2:] + rows[:2]).records["p:/work/proj/x.csv"]
    assert flipped.origin == "written", "the stronger origin wins whatever order they arrive in"


def test_supplement_newest_version_wins_and_retracts() -> None:
    def supplement(version: int, files: list[str], state: str = "done") -> dict[str, Any]:
        return {
            "id": f"s{version}",
            "ts": float(version),
            "type": "custom",
            "payload": {
                "custom_type": "supplement_v1",
                "details": {
                    "anchor": "A",
                    "job": "j",
                    "version": version,
                    "state": state,
                    "at": float(version),
                    "files": [
                        {
                            "path": p,
                            "name": p.rsplit("/", 1)[-1],
                            "kind": "markdown",
                            "size_bytes": 10,
                            "mtime": 1.0,
                            "why": "written by write",
                        }
                        for p in files
                    ],
                    "components": [],
                },
            },
        }

    scan = _scan([supplement(1, ["docs/a.md"]), supplement(2, ["docs/b.md"])])
    assert [f["path"] for f in scan.anchors["A"].files] == ["docs/b.md"]
    out_of_order = _scan([supplement(2, ["docs/b.md"]), supplement(1, ["docs/a.md"])])
    assert [f["path"] for f in out_of_order.anchors["A"].files] == ["docs/b.md"]
    skipped = _scan([supplement(1, ["docs/a.md"]), supplement(2, ["docs/b.md"], state="skipped")])
    assert skipped.anchors["A"].files == [], "a skipped newest version retracts the older callout"
    malformed = _scan(
        [
            {
                "id": "m",
                "ts": 1.0,
                "type": "custom",
                "payload": {
                    "custom_type": "supplement_v1",
                    "details": {"anchor": "A", "version": "2"},
                },
            }
        ]
    )
    assert malformed.anchors == {}


def test_prose_rows_are_never_json_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cost invariant: only file-producing rows reach ``json.loads``.

    Structural, not timed - a wall-clock bound flakes under fleet load (AGENTS.md "Timing").
    """
    parses: list[int] = []
    real = scan_module._row
    monkeypatch.setattr(scan_module, "_row", lambda raw: parses.append(1) or real(raw))
    prose = [
        {
            "id": f"p{i}",
            "ts": float(i),
            "type": "message",
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": "x" * 50}]},
        }
        for i in range(500)
    ]
    reads = [
        tool(f"r{i}", float(i), f"rc{i}", "read", "content", {"path": "/work/proj/f.py"})
        for i in range(200)
    ]
    spend = [
        {
            "id": f"sp{i}",
            "ts": 1.0,
            "type": "custom",
            "payload": {"custom_type": "session_spend.v1", "details": {}},
        }
        for i in range(100)
    ]
    scan = _scan(prose + reads + spend)
    assert parses == [] and scan.records == {} and scan.rows == 800
    _scan(
        [
            assistant("a", 1, [call("c1", "write", path="/work/proj/a.md")]),
            tool("t", 2, "c1", "write", "ok", {"path": "/work/proj/a.md"}),
        ]
    )
    assert len(parses) == 2


def test_resolve_path_rules() -> None:
    assert resolve_path("~/x/y.md", cwd=CWD, home=HOME) == "/home/u/x/y.md"
    assert resolve_path("out/../a.md", cwd=CWD, home=HOME) == "/work/proj/a.md"
    assert resolve_path("a.md", cwd="", home=HOME) is None, "no cwd: refuse to guess"
    assert resolve_path("a\x00b", cwd=CWD, home=HOME) is None
    assert resolve_path("", cwd=CWD, home=HOME) is None


def test_payload_round_trip_resumes_exactly() -> None:
    rows = generate_image_rows(1, 1.0, DIGEST, with_artifacts=True) + [
        assistant("a", 5, [call("c9", "bash", command="echo > z.txt")]),
    ]
    first = _scan(rows)
    clone = FileScan.from_payload(json.loads(json.dumps(first.to_payload())), cwd=CWD, home=HOME)
    assert clone.rows == first.rows and clone.records.keys() == first.records.keys()
    clone.feed(clone.rows, line(tool("t9", 6, "c9", "bash", "exit code: 0")))
    assert "p:/work/proj/z.txt" in clone.records, "a call pending at save time pairs after reload"
    with pytest.raises(ValueError):
        FileScan.from_payload({"rows": 1, "records": [[1, 2]]})


def test_record_cap_is_counted_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan_module, "MAX_RECORDS", 3)
    rows = [
        tool(f"t{i}", float(i), f"c{i}", "browser", "s", {"path": f"/tmp/s{i}.png", "bytes": 1})
        for i in range(5)
    ]
    scan = _scan(rows)
    assert len(scan.records) == 3 and scan.truncated and scan.dropped == 2


@pytest.mark.parametrize("name,key", [("bash", "command"), ("eval", "code")])
def test_oversized_code_is_refused_before_shared_parser_work(name, key, monkeypatch):
    """Inventory reads FULL journal arguments, unlike the supplement's 8KiB snapshot.

    Structural guard: a 100k repeated-open command must not reach the lexer/regex
    or hold the inventory's between-row deadline hostage. A normal row still scans.
    """
    from local_operator.supplements import candidates

    def forbidden(*args):
        raise AssertionError("oversized journal argument reached parser")

    monkeypatch.setattr(candidates, "_lift_heredocs", forbidden)
    monkeypatch.setattr(candidates, "_variable_write", forbidden)
    scan = _scan(
        [assistant("a", 1, [call("large", name, **{key: 'python3 -c "' + "open(" * 20000 + '"'})])]
    )
    assert len(scan.pending) == 1
    assert not next(iter(scan.pending.values())).outputs
