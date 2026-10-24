"""Deterministic storage/cancellation regressions; no running App required."""
import ast
import asyncio
import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from local_operator.session.runtime import inbox
from local_operator.wakes import spooled, supervisor


class JournalTests(unittest.TestCase):
    def test_short_append_and_failed_rollback_never_ack_poisoned_next_row(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            line = lambda text: inbox.InboxLine(text=text, sender={}, mode="mailbox", written_at=1)
            self.assertTrue(inbox.append_inbox(root, line("prior")))
            write = os.write
            with patch.object(os, "write", lambda fd, data: write(fd, data[:7])):
                self.assertFalse(inbox.append_inbox(root, line("short")))
            self.assertTrue(inbox.append_inbox(root, line("next")))
            self.assertEqual([r.text for r in inbox.peek_inbox(root)], ["prior", "next"])
            with patch.object(os, "write", lambda fd, data: write(fd, data[:7])), patch.object(
                os, "ftruncate", side_effect=OSError("controlled rollback failure")
            ):
                self.assertFalse(inbox.append_inbox(root, line("poison")))
            self.assertFalse(inbox.append_inbox(root, line("must not ack")))

    def test_short_recall_rolls_back_without_losing_the_owner_row(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            row = inbox.InboxLine(text="owner", sender={}, mode="mailbox", written_at=1,
                                  source=inbox.SOURCE_USER, command_id="synthetic-command")
            self.assertTrue(inbox.append_inbox(root, row))
            write = os.write
            with patch.object(os, "write", lambda fd, data: write(fd, data[:7])):
                self.assertFalse(inbox.withdraw_inbox(root, row.command_id))
            self.assertEqual(len(inbox.peek_inbox(root)), 1)
            self.assertTrue(inbox.withdraw_inbox(root, row.command_id))
            self.assertEqual(inbox.peek_inbox(root), [])


class OwnedWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_joins_io_and_preserves_both_failures(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        ready = asyncio.Event()
        failure = OSError("controlled storage failure")
        loop = asyncio.get_running_loop()
        loop_errors = []
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        def operation():
            entered.set()
            loop.call_soon_threadsafe(ready.set)
            try:
                release.wait(5)
                raise failure
            finally:
                finished.set()
        task = asyncio.create_task(spooled.await_spool_write(operation))
        try:
            await asyncio.wait_for(ready.wait(), 2)
            self.assertTrue(entered.is_set())
            task.cancel("first cancellation")
            await asyncio.sleep(0)
            task.cancel("second cancellation")
            await asyncio.sleep(0)
            # Two loop turns are a deterministic heartbeat while the I/O stays
            # parked, not a wall-clock timing assertion.
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(BaseExceptionGroup) as caught:
                await task
            self.assertTrue(finished.is_set())
            self.assertIsInstance(caught.exception.exceptions[0], asyncio.CancelledError)
            self.assertIs(caught.exception.exceptions[1], failure)
            await asyncio.sleep(0)
            self.assertEqual(loop_errors, [])
        finally:
            release.set()
            if not task.done():
                try:
                    await task
                except BaseException:
                    pass

    async def test_admission_marker_is_not_authority_until_append_commits(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            directory = root / "sessions" / "synthetic"
            directory.mkdir(parents=True)
            release = threading.Event()
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()
            append = inbox.append_inbox
            def held_append(*args):
                loop.call_soon_threadsafe(entered.set)
                release.wait(5)
                return append(*args)
            row = inbox.InboxLine(text="synthetic", sender={}, mode="mailbox", written_at=1,
                                  source=inbox.SOURCE_USER)
            with patch.object(inbox, "append_inbox", held_append), patch.object(spooled, "_raise_reader"):
                task = asyncio.create_task(spooled.await_spool_write(
                    inbox.append_for_successor, root, directory, row))
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    snapshot = spooled.read_spooled_turn(root, directory.name)
                    self.assertIsNotNone(snapshot)
                    self.assertIsNone(snapshot["generation"])
                    self.assertFalse(task.done())
                    release.set()
                    self.assertTrue(await task)
                    snapshot = spooled.read_spooled_turn(root, directory.name)
                    self.assertIsNotNone(snapshot["generation"])
                    self.assertFalse(spooled.clear_if_no_owed_turn(
                        root, directory.name, directory,
                        expected_generation=snapshot["generation"]))
                finally:
                    release.set()
                    if not task.done():
                        await task


class ReadBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_after_claim_keeps_owed_row_and_releases_shared_seat(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            row = inbox.InboxLine(text="owed", sender={}, mode="mailbox", written_at=1,
                                  source=inbox.SOURCE_USER)
            self.assertTrue(inbox.append_inbox(directory, row))
            batch = inbox.try_enter_read_batch(directory)
            self.assertIsNotNone(batch)
            ready = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()
            def claim():
                inbox.claim_inbox_batch(batch)
                loop.call_soon_threadsafe(ready.set)
                release.wait(5)
            async def owner():
                try:
                    await spooled.await_spool_write(claim)
                finally:
                    await inbox.finish_read_batch(batch)
            task = asyncio.create_task(owner())
            try:
                await asyncio.wait_for(ready.wait(), 2)
                self.assertIsNone(inbox.try_enter_read_batch(directory))
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertEqual([r.text for r in inbox.peek_inbox(directory)], ["owed"])
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(spooled.spool_owes_turn(directory))
                self.assertFalse(batch.acquired)
                next_batch = inbox.try_enter_read_batch(directory)
                self.assertIsNotNone(next_batch)
                inbox.leave_read_batch(next_batch)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_exact_ordinals_preserve_duplicate_idless_rows_and_concurrent_tail(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            line = lambda text: inbox.InboxLine(text=text, sender={}, mode="mailbox", written_at=1)
            for text in ("same", "same", "later"):
                self.assertTrue(inbox.append_inbox(directory, line(text)))
            batch = inbox.try_enter_read_batch(directory)
            try:
                await spooled.await_spool_write(inbox.claim_inbox_batch, batch)
                self.assertTrue(await spooled.await_spool_write(inbox.append_inbox, directory, line("tail")))
                rows = batch.delivery_rows()
                batch.handled.update(rows[0][0])
                await inbox.finish_read_batch(batch)
                self.assertTrue(batch.published and batch.durable)
                self.assertEqual([r.text for r in inbox.peek_inbox(directory)], ["same", "later", "tail"])
            finally:
                if batch.acquired:
                    await spooled.await_spool_write(inbox.release_inbox_batch, batch)
                inbox.leave_read_batch(batch)

    async def test_admission_accepted_just_before_cancel_is_committed_once(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            inbox.append_inbox(directory, inbox.InboxLine(text="one", sender={}, mode="mailbox", written_at=1))
            batch = inbox.try_enter_read_batch(directory)
            await spooled.await_spool_write(inbox.claim_inbox_batch, batch)
            entered = asyncio.Event()
            release = asyncio.Event()
            accepted = []
            async def admission():
                entered.set()
                await release.wait()
                accepted.append(1)
                return "accepted"
            task = asyncio.create_task(inbox.admit_batch_row(batch, batch.delivery_rows()[0][0], admission()))
            try:
                await entered.wait()
                task.cancel()
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await inbox.finish_read_batch(batch)
                self.assertEqual(accepted, [1])
                self.assertEqual(inbox.peek_inbox(directory), [])
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                if batch.acquired:
                    await spooled.await_spool_write(inbox.release_inbox_batch, batch)
                inbox.leave_read_batch(batch)

    async def test_prefix_conflict_never_overwrites_or_deletes_current_bytes(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            inbox.append_inbox(directory, inbox.InboxLine(text="one", sender={}, mode="mailbox", written_at=1))
            batch = inbox.try_enter_read_batch(directory)
            await spooled.await_spool_write(inbox.claim_inbox_batch, batch)
            batch.handled.add(0)
            path = inbox.inbox_path(directory)
            path.write_bytes(b"uncooperative mutation\n")
            with self.assertRaises(OSError):
                await inbox.finish_read_batch(batch)
            self.assertEqual(path.read_bytes(), b"uncooperative mutation\n")
            self.assertFalse(batch.acquired or batch.published)

    async def test_a_live_foreign_reader_defers_the_claim_without_raising(self):
        """The designed handover: the outgoing runtime still owns the lease.

        Deliveries hold the reader lease across the whole batch (owner prompts
        are measured in tens of seconds, past LOCK_WAIT_S), so a successor that
        raised here would fail its boot/close before its own admission work.
        """
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            inbox.append_inbox(directory, inbox.InboxLine(
                text="owed", sender={}, mode="mailbox", written_at=1, source=inbox.SOURCE_USER))
            batch = inbox.try_enter_read_batch(directory)

            class Busy:
                released = False
                def acquire(self):
                    raise inbox.WakeLockBusy("held by the outgoing runtime")
                def release(self):
                    self.released = True
            batch.reader = Busy()
            self.assertFalse(await inbox.claim_read_batch_or_defer(batch))
            await inbox.settle_read_batch(batch)  # must not raise, must not settle
            self.assertFalse(Busy.released, "release without acquisition")
            self.assertEqual([row.text for row in inbox.peek_inbox(directory)], ["owed"])
            self.assertTrue(spooled.spool_owes_turn(directory))
            self.assertIsNotNone(inbox.try_enter_read_batch(directory))

    async def test_a_claim_failing_after_the_lease_is_taken_releases_it(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            inbox.append_inbox(directory, inbox.InboxLine(
                text="owed", sender={}, mode="mailbox", written_at=1, source=inbox.SOURCE_USER))
            path = inbox.inbox_path(directory)
            target = directory / "elsewhere.jsonl"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
            batch = inbox.try_enter_read_batch(directory)
            try:
                self.assertFalse(await inbox.claim_read_batch_or_defer(batch))
            finally:
                await inbox.settle_read_batch(batch)
            self.assertFalse(batch.acquired)
            self.assertEqual(target.read_bytes(), path.read_bytes())
            lock = inbox.WakeWriteLock(directory, name=".inbox-reader.lock")
            lock.acquire()   # a leaked lease would block here
            lock.release()
            self.assertIsNotNone(inbox.try_enter_read_batch(directory))

    async def test_an_empty_inbox_latches_the_one_shot_flag(self):
        """An empty inbox must not re-acquire/settle on every pipeline pass."""
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            batch = inbox.try_enter_read_batch(directory)
            try:
                await spooled.await_spool_write(inbox.claim_inbox_batch, batch)
                self.assertIs(batch.pending_after_commit, False)
                await inbox.finish_read_batch(batch)
                self.assertIs(batch.pending_after_commit, False)
            finally:
                inbox.leave_read_batch(batch)
            inbox.append_inbox(directory, inbox.InboxLine(
                text="owed", sender={}, mode="mailbox", written_at=1))
            batch = inbox.try_enter_read_batch(directory)
            try:
                await spooled.await_spool_write(inbox.claim_inbox_batch, batch)
                self.assertIs(batch.pending_after_commit, True)
            finally:
                if batch.acquired:
                    await spooled.await_spool_write(inbox.release_inbox_batch, batch)
                inbox.leave_read_batch(batch)

    def test_the_e2e_stop_caller_awaits_the_async_action(self):
        """The committed caller must `await` the converted action.

        Source-level contract only: importing the Textual App here would be a
        different, separately allocated test.
        """
        tree = ast.parse(E2E_STOP_SOURCE, E2E_STOP_PATH)
        awaited, bare = 0, 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == "action_stop":
                    awaited += 1
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == "action_stop":
                    bare += 1
        self.assertEqual((awaited, bare), (1, 0))


class SupervisorLateFatalTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_late_failure_is_raised_after_every_engagement_is_joined(self):
        sweeper = supervisor._Sweeper()
        boom = RuntimeError("late bookkeeping failure")
        joined = []

        async def failing():
            raise boom

        async def forever():
            try:
                await asyncio.sleep(3600)
            finally:
                joined.append("forever")
            return False

        first = asyncio.create_task(failing())
        second = asyncio.create_task(forever())
        sweeper._in_flight[("s", 1)] = first
        sweeper._in_flight[("s", 2)] = second
        first.add_done_callback(sweeper._observe)
        await asyncio.sleep(0)
        with self.assertRaises(RuntimeError) as caught:
            await sweeper.drain()
        self.assertIs(caught.exception, boom)
        self.assertEqual(joined, ["forever"])
        self.assertEqual(sweeper._in_flight, {})

    async def test_the_same_failure_object_is_not_reported_as_a_group(self):
        sweeper = supervisor._Sweeper()
        boom = RuntimeError("one failure, two observers")
        sweeper._record_failure(boom)
        sweeper._record_failure(boom)
        self.assertIs(sweeper._failure, boom)
