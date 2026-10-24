"""Deterministic storage/cancellation regressions; no running App required."""
import asyncio
import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from local_operator.session.runtime import inbox
from local_operator.wakes import spooled


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
