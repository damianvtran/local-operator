"""Synthetic filesystem cases for the post-run score-detail projection."""

import hashlib
import json
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Literal, TextIO
from unittest.mock import patch

from retain_score_detail import (
    MAX_DETAIL_BYTES,
    publish_score_detail,
    retain_score_detail,
)


class RetainScoreDetailTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.raw = root / "raw"
        self.seal = root / "partial"
        self.seal.mkdir()
        (self.raw / "artifacts").mkdir(parents=True)
        raw_bundle = self.raw / "evidence" / "episode"
        raw_bundle.mkdir(parents=True)
        self.raw_outcome = raw_bundle / "outcome.json"
        self.data = b'{"checks":[{"passed":false}]}\n'
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.source = self.raw / "artifacts" / self.digest
        self.target = self.seal / "artifacts" / self.digest
        self.details = {
            "sha256": self.digest,
            "byte_count": len(self.data),
            "media_type": "application/json",
        }
        self.outcome = {"score": {"status": "scored", "binary": 0, "details": self.details}}

    def retain(self) -> dict[str, str]:
        return retain_score_detail(self.outcome, self.raw, self.seal)

    def test_retains_exact_referenced_bytes_only(self) -> None:
        self.source.write_bytes(self.data)
        unrelated = self.source.with_name(hashlib.sha256(b"unrelated").hexdigest())
        unrelated.write_bytes(b"unrelated")
        before = json.dumps(self.outcome)
        self.raw_outcome.write_text(before, encoding="utf-8")
        sealed_outcome = self.seal / "outcome.json"
        sealed_outcome.write_text(before, encoding="utf-8")

        outcome = json.loads(sealed_outcome.read_text(encoding="utf-8"))
        self.assertEqual(retain_score_detail(outcome, self.raw, self.seal), {"status": "retained"})
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(self.source.read_bytes(), self.data)
        self.assertTrue(unrelated.is_file())
        self.assertEqual(json.dumps(outcome), before)
        self.assertEqual(self.raw_outcome.read_text(encoding="utf-8"), before)
        self.assertEqual(sealed_outcome.read_text(encoding="utf-8"), before)

    def test_publishes_optional_detail_once(self) -> None:
        self.source.write_bytes(self.data)
        publish_score_detail(self.outcome, self.raw, self.seal)
        sidecar = self.seal / "score-detail"
        receipt = sidecar / "score-detail-receipt.json"
        target = sidecar / "artifacts" / self.digest
        self.assertEqual(list(self.seal.iterdir()), [sidecar])
        self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")), {"status": "retained"})
        self.assertEqual(target.read_bytes(), self.data)

        # An existing sidecar must remain in place even if a repeat can't retain the source.
        original_inode = sidecar.stat().st_ino
        self.source.unlink()
        publish_score_detail(self.outcome, self.raw, self.seal)
        self.assertEqual(list(self.seal.iterdir()), [sidecar])
        self.assertEqual(sidecar.stat().st_ino, original_inode)
        self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")), {"status": "retained"})
        self.assertEqual(target.read_bytes(), self.data)

    def test_receipt_close_error_preserves_primary_seal_and_ledger(self) -> None:
        self.source.write_bytes(self.data)
        outcome_bytes = json.dumps(self.outcome).encode()
        events_bytes = b"synthetic events"
        (self.seal / "outcome.json").write_bytes(outcome_bytes)
        (self.seal / "events.jsonl.gz").write_bytes(events_bytes)
        sealed = self.seal.rename(self.seal.with_name("sealed"))
        ledger = sealed.with_name("results.jsonl")
        ledger_bytes = b'{"status":"scored","binary":0}\n'
        with ledger.open("ab") as stream:
            stream.write(ledger_bytes)

        open_path = Path.open
        receipt_paths: list[Path] = []

        @contextmanager
        def fail_receipt_close(
            path: Path, mode: Literal["x"], *, encoding: str
        ) -> Iterator[TextIO]:
            with open_path(path, mode, encoding=encoding) as stream:
                yield stream
            if path.name == "score-detail-receipt.json":
                receipt_paths.append(path)
                # Fail after actual close so suppressing the error but leaving
                # staged data behind cannot pass.
                raise OSError("synthetic receipt close failure")

        warning = StringIO()
        with patch.object(Path, "open", fail_receipt_close), redirect_stderr(warning):
            publish_score_detail(self.outcome, self.raw, sealed)

        self.assertEqual(len(receipt_paths), 1)
        self.assertFalse(receipt_paths[0].parent.exists())
        self.assertEqual(
            warning.getvalue().strip(), "sealed: WARNING: optional score detail sidecar I/O failed"
        )
        self.assertEqual(
            sorted(p.name for p in sealed.iterdir()), ["events.jsonl.gz", "outcome.json"]
        )
        self.assertEqual((sealed / "outcome.json").read_bytes(), outcome_bytes)
        self.assertEqual((sealed / "events.jsonl.gz").read_bytes(), events_bytes)
        self.assertEqual(ledger.read_bytes(), ledger_bytes)
        self.assertEqual(self.source.read_bytes(), self.data)

    def test_missing_referenced_file_is_not_missing_reference(self) -> None:
        self.assertEqual(self.retain(), {"status": "missing", "reason": "artifact_absent"})
        self.assertFalse(self.target.parent.exists())

    def test_no_outcome_score_or_detail_does_not_create_artifacts(self) -> None:
        for outcome in (None, {}, {"score": None}, {"score": {}}, {"score": {"details": None}}):
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    retain_score_detail(outcome, self.raw, self.seal),
                    {"status": "missing", "reason": "reference_absent"},
                )
        self.assertFalse(self.target.parent.exists())

    def test_refuses_wrong_digest_even_with_declared_length(self) -> None:
        self.source.write_bytes(b"x" * len(self.data))
        self.assertEqual(self.retain(), {"status": "refused", "reason": "digest_mismatch"})
        self.assertFalse(self.target.parent.exists())

    def test_refuses_wrong_length(self) -> None:
        self.source.write_bytes(self.data)
        for length in (len(self.data) - 1, len(self.data) + 1):
            with self.subTest(length=length):
                self.details["byte_count"] = length
                self.assertEqual(self.retain(), {"status": "refused", "reason": "size_mismatch"})
        self.assertFalse(self.target.parent.exists())

    def test_refuses_noncanonical_or_out_of_policy_reference(self) -> None:
        self.source.write_bytes(self.data)
        for field, value in (
            ("sha256", self.digest.upper()),
            ("sha256", self.digest + "\n"),
            ("byte_count", True),
            ("byte_count", MAX_DETAIL_BYTES + 1),
            ("media_type", "image/png"),
        ):
            with self.subTest(field=field, value=value):
                original = self.details[field]
                self.details[field] = value
                self.assertEqual(
                    self.retain(), {"status": "refused", "reason": "invalid_reference"}
                )
                self.details[field] = original
        self.assertFalse(self.target.parent.exists())

    def test_refuses_nonregular_artifact(self) -> None:
        self.source.mkdir()
        self.assertEqual(self.retain(), {"status": "refused", "reason": "unsafe_source"})
        self.assertFalse(self.target.parent.exists())

    def test_refuses_symlink_artifact(self) -> None:
        sibling = self.raw / "synthetic.json"
        sibling.write_bytes(self.data)
        self.source.symlink_to(sibling)
        self.assertEqual(self.retain(), {"status": "refused", "reason": "unsafe_source"})
        self.assertFalse(self.target.parent.exists())

    def test_refuses_symlink_artifact_directory(self) -> None:
        (self.raw / "artifacts").rmdir()
        sibling = self.raw / "synthetic"
        sibling.mkdir()
        (sibling / self.digest).write_bytes(self.data)
        (self.raw / "artifacts").symlink_to(sibling, target_is_directory=True)
        self.assertEqual(self.retain(), {"status": "refused", "reason": "unsafe_source"})
        self.assertFalse(self.target.parent.exists())

    def test_does_not_replace_an_existing_artifact(self) -> None:
        self.source.write_bytes(self.data)
        self.target.parent.mkdir()
        self.target.write_bytes(b"prior data")
        self.assertEqual(self.retain(), {"status": "refused", "reason": "already_exists"})
        self.assertEqual(self.target.read_bytes(), b"prior data")
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_refuses_symlink_destination_artifact_directory(self) -> None:
        self.source.write_bytes(self.data)
        (self.seal / "artifacts").symlink_to(self.source.parent, target_is_directory=True)
        self.assertEqual(self.retain(), {"status": "refused", "reason": "unsafe_destination"})
        self.assertEqual(list(self.source.parent.iterdir()), [self.source])
        self.assertEqual(self.source.read_bytes(), self.data)
