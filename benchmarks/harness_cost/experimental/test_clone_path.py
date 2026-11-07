"""Offline clone-policy controls: tiny profile bytes, never runtime duplication.

Simulate an unsupported Linux reflink without invoking native benchmark tools.
Sparse files exercise limits without allocating large test data.
"""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runner_v2 as r


class ClonePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"])
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "basehome-public"
        self.source.mkdir(mode=0o750)
        self.skill = self.source / ".agents/skills/public/SKILL.md"
        self.skill.parent.mkdir(parents=True)
        self.skill.write_bytes(b"Synthetic public profile\x00\xff\n")
        self.skill.chmod(0o640)
        self.destination = self.root / "home"

    def unavailable(self, args, **kwargs):
        # GNU cp can create directories/empty files before returning ENOTSUP.
        self.assertIn("--reflink=always", args)
        destination = self.destination
        self.assertTrue(destination.is_dir())
        (destination / "partial-cow-output").write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, args, stderr=b"Operation not supported")

    def linux_copy(self):
        with (
            patch.object(r.sys, "platform", "linux"),
            patch.object(r.subprocess, "run", side_effect=self.unavailable),
        ):
            r.clone_path(self.source, self.destination)

    def test_unsupported_reflink_copies_identical_bounded_profile_bytes_and_modes(self):
        self.linux_copy()
        copied = self.destination / self.skill.relative_to(self.source)
        self.assertEqual(copied.read_bytes(), self.skill.read_bytes())
        self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o640)
        self.assertEqual(
            stat.S_IMODE(self.destination.stat().st_mode),
            stat.S_IMODE(self.source.stat().st_mode),
        )
        self.assertFalse((self.destination / "partial-cow-output").exists())

    def test_successful_cow_is_preferred_without_ordinary_copy(self):
        def clone(args, **kwargs):
            (self.destination / "cow-marker").write_bytes(b"clone")
            return subprocess.CompletedProcess(args, 0)

        with (
            patch.object(r.sys, "platform", "linux"),
            patch.object(r.subprocess, "run", side_effect=clone) as command,
            patch.object(r.os, "fwalk", side_effect=AssertionError("No fallback on CoW success")),
        ):
            r.clone_path(self.source, self.destination)
        self.assertEqual(command.call_count, 1)
        self.assertEqual((self.destination / "cow-marker").read_bytes(), b"clone")

    def test_dependency_and_cache_directories_require_cow(self):
        for name in (".venv", "venv", "node_modules", "site-packages", "installs", ".cache"):
            with self.subTest(name=name):
                directory = self.source / name
                directory.mkdir()
                try:
                    with self.assertRaisesRegex(ValueError, "CoW"):
                        self.linux_copy()
                    self.assertFalse(self.destination.exists())
                finally:
                    directory.rmdir()

    def test_sparse_oversized_profile_and_excess_entries_require_cow(self):
        sparse = self.source / "large-profile.json"
        with sparse.open("wb") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, "CoW"):
            self.linux_copy()
        self.assertFalse(self.destination.exists())
        sparse.unlink()
        # Repeated mock entries exercise count enforcement without 4097 inodes.
        real_walk = r.os.fwalk

        def too_many(*args, **kwargs):
            for rel, dirs, files, fd in real_walk(*args, **kwargs):
                if files:
                    yield rel, dirs, files * 4097, fd
                else:
                    yield rel, dirs, files, fd

        with patch.object(r.os, "fwalk", side_effect=too_many):
            with self.assertRaisesRegex(ValueError, "CoW"):
                self.linux_copy()
        self.assertFalse(self.destination.exists())

    def test_root_and_nested_symlinks_and_special_files_are_refused(self):
        link = self.root / "linked-profile"
        link.symlink_to(self.source, target_is_directory=True)
        with patch.object(r.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "symlink"):
                r.clone_path(link, self.destination)
            command.assert_not_called()
        for name, target in (("linked-skill", self.skill), ("linked-dir", self.skill.parent)):
            child = self.source / name
            child.symlink_to(target)
            with self.assertRaises(ValueError):
                self.linux_copy()
            self.assertFalse(self.destination.exists())
            child.unlink()
        fifo = self.source / "not-profile-data"
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            self.linux_copy()
        self.assertFalse(self.destination.exists())

    def test_existing_destination_and_dangling_link_are_untouched(self):
        self.destination.mkdir()
        sentinel = self.destination / "user-owned"
        sentinel.write_bytes(b"keep")
        with patch.object(r.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "already exist"):
                r.clone_path(self.source, self.destination)
            command.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"keep")
        sentinel.unlink()
        self.destination.rmdir()
        self.destination.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(ValueError, "already exist"):
            r.clone_path(self.source, self.destination)
        self.assertTrue(self.destination.is_symlink())

    def test_content_growth_after_preflight_is_bounded_and_partial_copy_removed(self):
        original = r.os.read
        count = 0
        grown = False

        def growing(fd, size):
            nonlocal count, grown
            if not grown:
                grown = True
                # Sparse growth after the size preflight, before the first read.
                with self.skill.open("r+b") as stream:
                    stream.truncate(8 * 1024 * 1024 + 1)
            block = original(fd, size)
            count += len(block)
            return block

        with patch.object(r.os, "read", side_effect=growing):
            with self.assertRaisesRegex(ValueError, "changed during copy"):
                self.linux_copy()
        self.assertLessEqual(count, 8 * 1024 * 1024)
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.skill.exists())

    def test_read_failure_cleans_only_created_destination(self):
        with patch.object(r.os, "read", side_effect=OSError("synthetic read failure")):
            with self.assertRaisesRegex(OSError, "synthetic"):
                self.linux_copy()
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.skill.exists())

    def test_readonly_partial_clone_is_removed_without_touching_link_targets(self):
        user_directory = self.root / "user-owned"
        user_directory.mkdir(mode=0o750)
        sentinel = user_directory / "keep"
        sentinel.write_bytes(b"untouched")

        def readonly_partial(args, **kwargs):
            destination = self.destination
            child = destination / "readonly"
            child.mkdir()
            (child / "link").symlink_to(user_directory, target_is_directory=True)
            child.chmod(0o500)
            destination.chmod(0o500)
            raise subprocess.CalledProcessError(1, args)

        with (
            patch.object(r.sys, "platform", "linux"),
            patch.object(r.subprocess, "run", side_effect=readonly_partial),
        ):
            r.clone_path(self.source, self.destination)
        self.assertFalse((self.destination / "readonly").exists())
        self.assertEqual(sentinel.read_bytes(), b"untouched")
        self.assertEqual(stat.S_IMODE(user_directory.stat().st_mode), 0o750)

    def test_replaced_partial_destination_is_not_cleaned_up(self):
        moved = self.root / "moved-partial"
        user_directory = self.root / "user-owned"
        user_directory.mkdir()
        sentinel = user_directory / "keep"
        sentinel.write_bytes(b"untouched")

        def replace_destination(args, **kwargs):
            self.destination.rename(moved)
            self.destination.symlink_to(user_directory, target_is_directory=True)
            raise subprocess.CalledProcessError(1, args)

        with (
            patch.object(r.sys, "platform", "linux"),
            patch.object(r.subprocess, "run", side_effect=replace_destination),
        ):
            with self.assertRaisesRegex(ValueError, "refusing cleanup"):
                r.clone_path(self.source, self.destination)
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_unreadable_source_directory_fails_closed_in_both_walks(self):
        unreadable = self.source / "unreadable"
        unreadable.mkdir()
        secret = unreadable / "must-not-be-omitted.txt"
        secret.write_bytes(b"Real public fixture bytes, not an empty subtree")
        original_walk = r.os.fwalk

        for pass_number in (1, 2):
            with self.subTest(pass_number=pass_number):
                calls = 0

                def unreadable_walk(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == pass_number:
                        unreadable.chmod(0)
                    yield from original_walk(*args, **kwargs)

                try:
                    with patch.object(r.os, "fwalk", side_effect=unreadable_walk):
                        with self.assertRaises(PermissionError):
                            self.linux_copy()
                    self.assertFalse(self.destination.exists())
                finally:
                    # Restore the real chmod000 fixture even when a regression
                    # unexpectedly reports success, so fixture cleanup is safe.
                    unreadable.chmod(0o700)
                self.assertEqual(
                    secret.read_bytes(), b"Real public fixture bytes, not an empty subtree"
                )

    def test_root_swap_after_source_open_never_writes_or_chmods_outside(self):
        # Same timing as R6-F2: real source open succeeds, then the owned root
        # is moved aside and its public pathname becomes an outside symlink.
        # Traversal, source read, destination writes/mode application are real.
        flat = self.source / "profile.txt"
        flat.write_bytes(b"public bytes")
        outside = self.root / "outside"
        outside.mkdir(mode=0o700)
        # Mirror existing source subdirectories so the old pathname-based copy
        # can demonstrably reach its write/chmod rather than fail on a missing
        # directory before exercising the reported race.
        (outside / ".agents/skills/public").mkdir(parents=True)
        moved = self.root / "moved-during-copy"
        original_open = r.os.open
        swapped = False

        def swap_after_source_open(path, flags, *args, **kwargs):
            nonlocal swapped
            fd = original_open(path, flags, *args, **kwargs)
            if path == "profile.txt" and flags & os.O_NONBLOCK and not swapped:
                swapped = True
                self.destination.rename(moved)
                self.destination.symlink_to(outside, target_is_directory=True)
            return fd

        with patch.object(r.os, "open", side_effect=swap_after_source_open):
            with self.assertRaisesRegex(ValueError, "destination changed"):
                self.linux_copy()
        self.assertTrue(swapped)
        self.assertTrue(self.destination.is_symlink())
        self.assertFalse((outside / "profile.txt").exists())
        self.assertFalse((outside / self.skill.relative_to(self.source)).exists())
        self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o700)
        self.assertFalse(moved.exists())  # only the inode we created was cleaned
        self.assertEqual(flat.read_bytes(), b"public bytes")

    def test_native_clone_carrier_uses_owned_cwd_and_checks_success_binding(self):
        # Exercise the real carrier/native cp too: replacing its destination
        # pathname must not redirect CoW writes before the final identity check.
        outside = self.root / "outside"
        outside.mkdir(mode=0o700)
        moved = self.root / "moved-before-native-clone"
        original_run = r.subprocess.run

        def replace_before_native_clone(args, **kwargs):
            self.destination.rename(moved)
            self.destination.symlink_to(outside, target_is_directory=True)
            return original_run(args, **kwargs)

        with patch.object(r.subprocess, "run", side_effect=replace_before_native_clone):
            with self.assertRaisesRegex(ValueError, "destination changed"):
                r.clone_path(self.source, self.destination)
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o700)
        self.assertFalse(moved.exists())

    def test_cross_filesystem_policy_still_refuses_before_copy(self):
        original = r.Path.stat

        def different_device(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == self.destination.parent:
                fields = list(info)
                fields[2] += 1  # st_dev; no second filesystem or large fixture needed.
                return os.stat_result(fields)
            return info

        with (
            patch.object(r.Path, "stat", autospec=True, side_effect=different_device),
            patch.object(r.subprocess, "run") as command,
        ):
            with self.assertRaisesRegex(ValueError, "share a filesystem"):
                r.clone_path(self.source, self.destination)
            command.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_runtime_named_source_root_cannot_use_fallback(self):
        runtime = self.root / ".venv"
        self.source.rename(runtime)
        self.source = runtime
        with self.assertRaisesRegex(ValueError, "CoW"):
            self.linux_copy()
        self.assertFalse(self.destination.exists())

    def test_non_linux_clone_failure_does_not_enable_copy(self):
        with (
            patch.object(r.sys, "platform", "darwin"),
            patch.object(r.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["cp"])),
            patch.object(r.os, "read", side_effect=AssertionError("No Darwin fallback")),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                r.clone_path(self.source, self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
