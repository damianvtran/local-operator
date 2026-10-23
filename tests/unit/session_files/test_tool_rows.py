"""The tools' additive ``details.artifacts`` -> the real transcript writer -> the scanner.

End to end on the REAL pieces, because the claim is "an artifact survives pruning": the tool
produces the result, ``Message.tool_result`` + ``Transcript`` write the row exactly as a live
session does, ``_pruned_entry`` blanks it exactly as the prune pass does, and the scanner
reads the resulting file. A control row WITHOUT ``details.artifacts`` (what builds before this
change wrote) shows the gap the field closes.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.imagegen import ImageAttempt, ImageOutcome, ImageRoute, MediaAsset
from local_operator.session.transcript import Transcript, TranscriptEntry, _pruned_entry
from local_operator.session_files import ledger
from local_operator.tools import image_tool
from local_operator.video import jobs as video_jobs

from ._fixtures import SESSION, png

pytestmark = pytest.mark.asyncio


def _outcome(*seeds: int) -> ImageOutcome:
    return ImageOutcome(
        assets=tuple(
            MediaAsset(data=png(seed=s), content_type="image/png", source_url=f"https://x/{s}.png")
            for s in seeds
        ),
        route=ImageRoute.RADIENT,
        attempts=(ImageAttempt(route=ImageRoute.RADIENT, outcome="ok"),),
        model="flux/dev",
        prompt="a cat",
        seed=None,
        generation_id="g1",
        cost_usd=0.03,
        cost_source=None,
        billing_basis=None,
        cost_provenance=None,
        usage_record_id=None,
        strength_ignored=False,
    )


async def _journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeds: tuple[int, ...]
) -> tuple[Path, Any]:
    async def fake_cascade(**kwargs: Any) -> ImageOutcome:
        return _outcome(*seeds)

    monkeypatch.setattr(image_tool, "run_image_cascade", fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )
    assert result.is_error is False
    session_dir = tmp_path / "cfg" / "sessions" / SESSION
    transcript = Transcript(session_dir)
    await transcript.append_message(Message.tool_result(result))
    return session_dir, result


def _scan(session_dir: Path) -> ledger.Snapshot:
    ledger.forget_resident()
    snapshot = ledger.refresh(tmp_config(session_dir), SESSION, session_dir, cwd="/w", home="/h")
    assert snapshot is not None
    return snapshot


def tmp_config(session_dir: Path) -> Path:
    return session_dir.parent.parent


async def test_generate_image_writes_artifacts_without_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir, result = await _journal(tmp_path, monkeypatch, (1, 2))
    artifacts = (result.details or {})["artifacts"]
    assert [a["digest"] for a in artifacts] == [b.attachment for b in result.content[1:]]
    for entry in artifacts:
        assert set(entry) == {
            "digest",
            "kind",
            "name",
            "content_type",
            "size_bytes",
            "width",
            "height",
            "duration_s",
        }
        assert entry["kind"] == "image" and (entry["width"], entry["height"]) == (8, 6)
        assert "/" not in entry["name"], "a display handle, never a filesystem path"
    # The durable row: details carry the list on disk.
    row = json.loads((session_dir / "transcript.jsonl").read_text().splitlines()[-1])
    assert row["payload"]["provider_payload"]["details"]["artifacts"] == artifacts


async def test_the_artifact_survives_the_prune_pass_and_the_old_shape_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir, result = await _journal(tmp_path, monkeypatch, (1,))
    digest = result.details["artifacts"][0]["digest"]  # type: ignore[index]
    path = session_dir / "transcript.jsonl"
    entry = TranscriptEntry.from_json(path.read_text().splitlines()[-1])
    assert entry is not None

    def prune(row: TranscriptEntry) -> None:
        pruned = _pruned_entry(row, "[pruned]")
        path.write_text(pruned.to_json() + "\n")

    prune(entry)
    assert "digest" not in json.loads(path.read_text())["payload"]["content"][0]["text"]
    assert list(_scan(session_dir).scan.records) == [f"a:{digest}"], "details survived the prune"

    # Control: the pre-change shape (no details.artifacts) is invisible once pruned.
    old = copy.deepcopy(json.loads(entry.to_json()))
    old["payload"]["provider_payload"]["details"].pop("artifacts")
    old_entry = TranscriptEntry.from_json(json.dumps(old))
    assert old_entry is not None
    prune(old_entry)
    assert _scan(session_dir).scan.records == {}, "the known gap for old pruned sessions"


async def test_partial_registration_lists_only_what_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_cascade(**kwargs: Any) -> ImageOutcome:
        good = _outcome(1)
        bad = MediaAsset(data=b"", content_type="image/png", source_url="https://x/bad.png")
        return dataclasses.replace(good, assets=(*good.assets, bad))

    monkeypatch.setattr(image_tool, "run_image_cascade", fake_cascade)
    result = await image_tool.execute_generate_image(
        "call-1", {"prompt": "a cat"}, None, None, None
    )
    assert (
        len((result.details or {})["artifacts"]) == 1
    ), "an unregistered asset must not be advertised"


async def test_video_receipt_carries_the_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = MediaAsset(
        data=b"\x00\x00\x00\x18ftypmp42" + b"x" * 64,
        content_type="video/mp4",
        source_url="https://x/v.mp4",
        duration_s=4.0,
    )
    receipt, text = video_jobs.finish_video_result(
        record_job_id="job1",
        route="fal",
        model="wan/t2v",
        assets=[asset],
        cost_usd=0.5,
        cost_source="reported",
        billing_basis="billed",
        prompt="A kite over cliffs",
        filename="Kite Over Cliffs",
    )
    (artifact,) = receipt["artifacts"]
    assert artifact["digest"] == receipt["digest"] and artifact["kind"] == "video"
    assert artifact["name"] == "kite-over-cliffs.mp4", "the request names the asset"
    assert artifact["duration_s"] == 4.0 and "digest" in text
    # Omitted, the name derives from the prompt (the old-record fallback path).
    fallback, _ = video_jobs.finish_video_result(
        record_job_id="job3",
        route="fal",
        model="wan/t2v",
        assets=[asset],
        cost_usd=0.5,
        cost_source="reported",
        billing_basis="billed",
        prompt="A kite over cliffs",
    )
    assert fallback["artifacts"][0]["name"] == "kite-over-cliffs.mp4"
    # A failed registration advertises nothing.
    empty, _ = video_jobs.finish_video_result(
        record_job_id="job2",
        route="fal",
        model="m",
        assets=[MediaAsset(data=b"", content_type="video/mp4", source_url="https://x/e.mp4")],
        cost_usd=None,
        cost_source=None,
        billing_basis=None,
    )
    assert empty["artifacts"] == [] and empty["digest"] is None
