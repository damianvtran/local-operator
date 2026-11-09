"""Durable music jobs over the existing AsyncJobManager and audio attachment store.

The transcript record is written BEFORE paid submission. Recovery reuses its
idempotency key, never starts an unrelated replacement composition. Explicit
cancel requests keep an observer alive because a running provider can still
produce charged audio; session teardown merely stops local observation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from local_operator.clients.radient_cost import channel_record_from_cost, cost_from_payload
from local_operator.i18n.keys import wire_music as mk
from local_operator.i18n.messages import render_resolved
from local_operator.music.client import MusicClient
from local_operator.session.channel_spend import new_record_id, now_ms
from local_operator.video.rungs import VIDEO_POLL_INTERVAL_S

logger = logging.getLogger(__name__)
CUSTOM_TYPE = "music_job_v1"
TERMINAL = frozenset({"completed", "failed"})


def read_records(directory: Any) -> dict[str, dict[str, Any]]:
    from local_operator.session.transcript import read_custom_entries

    result: dict[str, dict[str, Any]] = {}
    if not directory:
        return result
    for entry in read_custom_entries(directory, CUSTOM_TYPE):
        payload = getattr(entry, "payload", None)
        details = payload.get("details") if isinstance(payload, dict) else None
        if isinstance(details, dict) and isinstance(details.get("job"), str):
            result[details["job"]] = dict(details)
    return result


async def journal(context: Any, record: dict[str, Any]) -> None:
    writer = context.journal_music_job
    if writer is None:
        # A paid submit with no durable request identity cannot be recovered.
        raise RuntimeError("music_durable_journal_unavailable")
    record["v"] = 1
    record["at"] = datetime.now(timezone.utc).isoformat()
    await writer(dict(record))


def _spend(context: Any, track: dict[str, Any]) -> None:
    callback = context.record_channel_spend
    if callback is None:
        return
    cost = cost_from_payload(track)
    if cost is None:
        return
    row = channel_record_from_cost(
        cost,
        channel="audio",
        record_id=new_record_id("audio", track["id"]),
        session_id=context.session_id,
        request_id=track["id"],
        ts_ms=now_ms(),
    )
    # Quote and terminal receipt share identity. A replay is an idempotent
    # revision, not a second ledger line; unknown/estimated never becomes billed.
    callback(replace(row, rev=2 if track.get("settled") else 1))


async def _register(
    context: Any, client: MusicClient, record: dict[str, Any], track: dict[str, Any]
) -> None:
    from local_operator.artifacts.naming import media_filename
    from local_operator.session.attachments import artifact_details, cache_media

    existing = record.setdefault("artifacts", [])
    for artifact in track.get("artifacts", []):
        identity = (track["id"], artifact["id"], artifact["role"])
        if any(
            (row.get("track_id"), row.get("artifact_id"), row.get("role")) == identity
            for row in existing
        ):
            continue
        asset = await client.download(artifact)
        title = (
            track.get("title")
            or record["params"].get("title")
            or record["params"].get("prompt")
            or track["id"]
        )
        block = cache_media(
            asset.data,
            asset.content_type or "audio/mpeg",
            kind="audio",
            name=media_filename(
                title, asset.content_type, kind="audio", index=track.get("variant_index", 0) + 1
            ),
            duration_s=artifact.get("duration_seconds"),
        )
        if block is None:
            raise RuntimeError("music_attachment_registration_failed")
        facts = artifact_details([block])[0]
        facts.update(
            {
                "material": "music",
                "track_id": track["id"],
                "artifact_id": artifact["id"],
                "role": artifact["role"],
                "variant_index": track.get("variant_index", 0),
                "title": track.get("title"),
                "operation": "compose",
                "attribution": record.get("attribution", ""),
            }
        )
        existing.append(facts)
        # Registration is journaled per asset, so recovery does not silently
        # re-register all siblings after the last download fails.
        await journal(context, record)


def runner(context: Any, initial: dict[str, Any], *, client_factory: Any = MusicClient) -> Any:
    async def run(watcher_id: str, signal: Any, report: Any) -> str:
        del signal  # cancellation is task-owned; turn interrupts do not reach it
        from local_operator.paths import config_dir

        record = dict(initial)
        record["watcher"] = watcher_id
        client = client_factory(config_dir=config_dir())
        try:
            await journal(context, record)
            if not record.get("tracks"):
                payload = await client.request("generate", body=record["params"], key=record["key"])
                tracks = payload.get("tracks")
                if not isinstance(tracks, list) or not tracks:
                    raise RuntimeError("music_response_has_no_tracks")
                # Only IDs/status/cost are durable here. Signed playback URLs
                # are bearer grants and are fetched afresh at result time.
                record["tracks"] = [
                    {
                        key: track[key]
                        for key in (
                            "id",
                            "title",
                            "variant_index",
                            "status",
                            "settled",
                            "cost",
                            "error_type",
                        )
                        if key in track
                    }
                    for track in tracks
                ]
                record["attribution"] = payload.get("attribution", "")
                if record.get("cancellation_requested"):
                    await cancel_record(context, record, client=client)
                else:
                    record["state"] = "running"
                await journal(context, record)
            while True:
                settled = 0
                for track in record["tracks"]:
                    if not track.get("settled"):
                        payload = await client.request("status", request_id=track["id"])
                        current = payload["tracks"][0]
                        track.update(
                            {
                                key: current[key]
                                for key in (
                                    "title",
                                    "variant_index",
                                    "status",
                                    "settled",
                                    "cost",
                                    "error_type",
                                )
                                if key in current
                            }
                        )
                    _spend(context, track)
                    if track.get("settled"):
                        if (
                            not track.get("error_type")
                            and track.get("status") == "COMPLETED"
                            and not track.get("artifacts_registered")
                        ):
                            try:
                                completed = await client.request("result", request_id=track["id"])
                                await _register(context, client, record, completed["tracks"][0])
                                track["artifacts_registered"] = True
                                track.pop("registration_error", None)
                            except Exception as exc:
                                # A CDN/store failure must not erase successful
                                # siblings or their already-authoritative costs.
                                track["registration_error"] = (
                                    getattr(exc, "code", None) or type(exc).__name__
                                )
                        settled += 1
                await journal(context, record)
                progress = render_resolved(
                    mk.progress(completed=settled, total=len(record["tracks"]))
                )
                report(
                    {
                        "progress": progress,
                        "music": {
                            "job": record["job"],
                            "stage": (
                                "cancel_requested"
                                if record.get("cancellation_requested")
                                else "polling"
                            ),
                            "tracks": record["tracks"],
                            "artifacts": record.get("artifacts", []),
                            "attribution": record.get("attribution", ""),
                        },
                    }
                )
                if settled == len(record["tracks"]):
                    break
                await asyncio.sleep(VIDEO_POLL_INTERVAL_S)
            record["state"] = (
                "interrupted"
                if any(track.get("registration_error") for track in record["tracks"])
                else "completed" if record.get("artifacts") else "failed"
            )
            await journal(context, record)
            receipt = {
                "job": record["job"],
                "material": "music",
                "model": "radient/music",
                "provider": "radient",
                "tracks": record["tracks"],
                "artifacts": record.get("artifacts", []),
                "attribution": record.get("attribution", ""),
            }
            report({"music_receipt": receipt, "music": {**receipt, "stage": record["state"]}})
            if not record.get("artifacts"):
                raise RuntimeError("music_no_playable_results")
            return render_resolved(mk.ready(job=record["job"]))
        except asyncio.CancelledError:
            disposed = context.session_disposed
            if disposed is None or not disposed():
                await cancel_record(context, record, client=client)
                # AsyncJobManager cancels the local task, not the upstream
                # synthesis. A new typed observer retains the paid outcome.
                record.pop("watcher", None)
                # The cancelling manager row still says running until this
                # handler returns. Replace only THAT watcher, never a live
                # successor whose record also carries cancellation intent.
                record["watcher"] = ""
                successor = register_observer(context, record)
                if successor:
                    record["watcher"] = successor
                    await journal(context, record)
            else:
                record["state"] = "interrupted"
                await journal(context, record)
            raise
        except Exception as exc:
            record["state"] = (
                "interrupted"
                if record.get("tracks") or not record.get("state") == "failed"
                else "failed"
            )
            record["error"] = getattr(exc, "code", None) or type(exc).__name__
            await journal(context, record)
            raise
        finally:
            await client.close()

    return run


async def cancel_record(
    context: Any, record: dict[str, Any], *, client: MusicClient | None = None
) -> None:
    from local_operator.paths import config_dir

    owned = client is None
    client = client or MusicClient(config_dir=config_dir())
    try:
        outcomes = []
        for track in record.get("tracks", []):
            if track.get("settled"):
                continue
            try:
                outcomes.append(await client.request("cancel", body={"request_id": track["id"]}))
            except Exception as exc:
                outcomes.append(
                    {
                        "request_id": track["id"],
                        "error_type": getattr(exc, "code", None) or type(exc).__name__,
                    }
                )
        record["state"] = "cancel_requested"
        record["cancellation_requested"] = True
        record["cancellation"] = [
            {
                key: value
                for key, value in outcome.items()
                if key in {"request_id", "requested", "vendor_stopped", "error_type"}
            }
            for outcome in outcomes
        ]
        await journal(context, record)
    finally:
        if owned:
            await client.close()


def register_observer(context: Any, record: dict[str, Any]) -> str | None:
    jobs = context.jobs
    if jobs is None:
        return None
    current = jobs.get(record.get("watcher", record["job"]))
    if current is not None and current.status == "running":
        return current.id
    return jobs.register(
        type="music", label=render_resolved(mk.label()), run=runner(context, record)
    )


async def recover(context: Any) -> None:
    try:
        records = read_records(context.session_dir)
        for record in records.values():
            if record.get("state") not in TERMINAL:
                watcher = register_observer(context, record)
                if watcher:
                    record["watcher"] = watcher
                    await journal(context, record)
    except Exception:  # noqa: BLE001 — recovery failure cannot prevent session open
        logger.warning("music recovery could not resume pending observations", exc_info=True)
