"""Page coordinates must come from the same verified source as the payload.

Synthetic journals only. These pins cover the index/page seam; they are not
an HTTP negotiation test or a claim that the UI already consumes ordinals.
"""

import json
from pathlib import Path

from local_operator.session import transcript as transcript_module
from local_operator.session import transcript_index as ti
from local_operator.session.transcript import read_transcript_page

SID = "journal-order-page-fixture"


def _journal(root: Path) -> Path:
    return root / "sessions" / SID / "transcript.jsonl"


def _rows(root: Path, ids: list[str]) -> None:
    path = _journal(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately backwards clocks: source order is the complete-line ordinal.
    with path.open("a", encoding="utf-8") as handle:
        for at, id_ in enumerate(ids):
            handle.write(
                json.dumps(
                    {
                        "id": id_,
                        "ts": 100 - at,
                        "type": "message",
                        "payload": {
                            "kind": "message",
                            "role": "assistant",
                            "content": [{"text": id_}],
                        },
                    }
                )
                + "\n"
            )


def test_backward_pages_share_one_verified_domain(tmp_path: Path):
    _rows(tmp_path, ["page-a", "page-c"])
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    tail = read_transcript_page(_journal(tmp_path).parent, limit=1)
    older = read_transcript_page(_journal(tmp_path).parent, before_id="page-c", limit=1)
    assert ti.journal_coordinates_for_page(index, tail) == (index.journal_generation, {"page-c": 1})
    assert ti.journal_coordinates_for_page(index, older) == (
        index.journal_generation,
        {"page-a": 0},
    )


def test_page_from_old_file_cannot_receive_new_generation(tmp_path: Path):
    _rows(tmp_path, ["reused-id"])
    old_page = read_transcript_page(_journal(tmp_path).parent, limit=1)
    replacement = _journal(tmp_path).with_suffix(".replacement")
    replacement.write_text(
        json.dumps(
            {
                "id": "reused-id",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": "new payload"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replacement.replace(_journal(tmp_path))
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    assert ti.journal_coordinates_for_page(index, old_page) is None
    new_page = read_transcript_page(_journal(tmp_path).parent, limit=1)
    assert ti.journal_coordinates_for_page(index, new_page) == (
        index.journal_generation,
        {"reused-id": 0},
    )


def test_stale_or_unknown_index_has_no_page_authority(tmp_path: Path):
    _rows(tmp_path, ["stable-a"])
    stale = ti.refresh_index(tmp_path, SID)
    assert stale is not None
    _rows(tmp_path, ["appended-b"])
    page = read_transcript_page(_journal(tmp_path).parent, limit=2)
    assert ti.journal_coordinates_for_page(stale, page) is None
    ready = ti.refresh_index(tmp_path, SID)
    assert ready is not None
    ready.journal_generation = ""
    assert ti.journal_coordinates_for_page(ready, page) is None


def test_around_page_carries_verified_source_signature(tmp_path: Path):
    _rows(tmp_path, ["around-a", "around-b", "around-c"])
    page = read_transcript_page(_journal(tmp_path).parent, around_id="around-b", before=1, after=1)
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    assert ti.journal_coordinates_for_page(index, page) == (
        index.journal_generation,
        {"around-a": 0, "around-b": 1, "around-c": 2},
    )


def test_page_changed_during_read_has_no_source_authority(tmp_path: Path, monkeypatch):
    _rows(tmp_path, ["race-a", "race-b"])
    original = transcript_module._iter_complete_lines_backward

    def racing_reader(*args, **kwargs):
        for chunk in original(*args, **kwargs):
            _rows(tmp_path, ["race-appended"])
            yield chunk

    monkeypatch.setattr(transcript_module, "_iter_complete_lines_backward", racing_reader)
    page = read_transcript_page(_journal(tmp_path).parent, limit=1)
    assert page.source_signature is None
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None
    assert ti.journal_coordinates_for_page(index, page) is None
