"""Durable identity and bounded-undelivered-command regressions."""

from __future__ import annotations

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.mobile.command_reservation import (
    MAX_PENDING_STEERS,
    CommandReservations,
)
from local_operator.session.transcript import Transcript


class _SessionAuthority:
    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript

    def has_admitted_command(self, command_id: str) -> bool:
        return self.transcript.has_admitted_command(command_id)

    def subscribe_admitted_commands(self, handler):  # noqa: ANN001, ANN202
        return self.transcript.subscribe_admitted_commands(handler)


@pytest.mark.asyncio
async def test_compacted_command_remains_admitted_after_reconstruction(tmp_path) -> None:
    directory = tmp_path / "session"
    transcript = Transcript(directory)
    command = Message.user("first", id="producer-1")
    await transcript.append_message(command, producer_command_id=command.id)
    kept = await transcript.append_message(Message.assistant("later"))
    await transcript.append_compaction("summary", kept.id, tokens_before=100)

    assert all(message.id != command.id for message in transcript.build_llm_history())
    assert transcript.has_admitted_command(command.id)
    assert Transcript(directory).has_admitted_command(command.id)


@pytest.mark.asyncio
async def test_only_explicit_producer_markers_enter_admission_index(tmp_path) -> None:
    directory = tmp_path / "session"
    transcript = Transcript(directory)
    collision = "shared-id"

    await transcript.append_message(Message.user("local", id=collision))
    await transcript.append_message(Message.assistant("assistant", id=collision))
    await transcript.append_message(
        CustomMessage(custom_type="imported", details={"text": "custom"}, id=collision)
    )
    await transcript.append_custom("ledger", {"command_id": collision})
    await transcript.append_compaction("summary", collision, tokens_before=1)
    with transcript.path.open("a", encoding="utf-8") as handle:
        handle.write('{"id":"shared-id","type":"message","payload":{"role":"user"\n')

    assert not transcript.has_admitted_command(collision)
    assert not Transcript(directory).has_admitted_command(collision)
    reservations = CommandReservations(_SessionAuthority(Transcript(directory)))
    assert reservations.reserve(collision, kind="prompt")

    await transcript.append_message(
        Message.user("mobile", id="different-message-id"),
        producer_command_id=collision,
    )
    reopened = Transcript(directory)
    assert reopened.has_admitted_command(collision)
    assert not CommandReservations(_SessionAuthority(reopened)).reserve(collision, kind="prompt")
    marker = reopened.entries()[-1].payload["producer_command_id"]
    replayed = reopened.build_llm_history()
    assert marker == collision
    assert all("producer_command_id" not in message.model_dump() for message in replayed)


@pytest.mark.asyncio
async def test_append_before_ack_hands_reservation_to_durable_authority(tmp_path) -> None:
    transcript = Transcript(tmp_path / "session")
    reservations = CommandReservations(_SessionAuthority(transcript))
    unsubscribe = reservations.subscribe_durable()
    assert reservations.reserve("crash-boundary", kind="prompt")

    await transcript.append_message(
        Message.user("persisted", id="crash-boundary"),
        producer_command_id="crash-boundary",
    )

    assert "crash-boundary" not in reservations._commands
    assert not reservations.reserve("crash-boundary", kind="prompt")
    unsubscribe()


@pytest.mark.asyncio
async def test_pending_steers_are_bounded_without_evicting_accepted_ids(tmp_path) -> None:
    transcript = Transcript(tmp_path / "session")
    reservations = CommandReservations(_SessionAuthority(transcript))
    reservations.subscribe_durable()

    for index in range(MAX_PENDING_STEERS):
        assert reservations.reserve(f"steer-{index}", kind="steer")
    assert reservations._pending_steers == MAX_PENDING_STEERS
    assert len(reservations._commands) == MAX_PENDING_STEERS
    assert not reservations.reserve("steer-0", kind="steer")
    with pytest.raises(RuntimeError, match=r"steering queue is full \(32\)"):
        reservations.reserve("overflow", kind="steer")

    await transcript.append_message(
        Message.user("delivered", id="steer-0"),
        producer_command_id="steer-0",
    )
    assert reservations._pending_steers == MAX_PENDING_STEERS - 1
    assert reservations.reserve("replacement", kind="steer")
    assert not reservations.reserve("steer-0", kind="steer")


def test_prompt_transfer_consumes_one_steer_slot_and_rejection_releases_it(tmp_path) -> None:
    reservations = CommandReservations(_SessionAuthority(Transcript(tmp_path / "session")))
    assert reservations.reserve("race", kind="prompt")
    reservations.reject("race", transfer_to_steer=True)
    assert reservations.reserve("race", kind="steer", prompt_transfer=True)
    assert reservations._pending_steers == 1

    reservations.reject("race")
    assert reservations._pending_steers == 0
    assert reservations.reserve("race", kind="steer")
    reservations.clear()
    assert reservations._pending_steers == 0
    assert not reservations._commands
