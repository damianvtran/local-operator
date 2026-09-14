"""Keep the bulk scanner byte/state-identical to its pre-optimization contract."""

import random
from pathlib import Path
from typing import SupportsIndex

import pytest

from local_operator.evaluation.evidence.store import (
    EvidenceBundleInvalid,
    EvidenceWriter,
    _RedactionScanner,
)
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import manifest


class _BytewiseScanner(_RedactionScanner):
    """Frozen decoder from e3f0837f9; deliberately not another bulk algorithm.

    Output equality alone misses delayed escapes at chunk boundaries. Keeping
    the old transition function lets the tests compare pending bytes after EACH
    operation, including invalid escapes and empty input, not just final output.
    """

    def _decode_percent(self, block: bytes) -> bytes:
        output = bytearray()
        data = bytes(self._percent_pending) + block
        self._percent_pending.clear()
        index = 0
        while index < len(data):
            if data[index] != 37:  # %
                output.append(data[index])
                index += 1
                continue
            if index + 2 >= len(data):
                self._percent_pending.extend(data[index:])
                break
            high = self._hex_value(data[index + 1])
            low = self._hex_value(data[index + 2])
            if high is None or low is None:
                output.append(data[index])
                index += 1
                continue
            output.append((high << 4) | low)
            index += 3
        return bytes(output)


def _compare_decoding(writer: EvidenceWriter, chunks: list[bytes]) -> None:
    current = _RedactionScanner(writer)
    previous = _BytewiseScanner(writer)
    for chunk in chunks:
        assert current._decode_percent(chunk) == previous._decode_percent(chunk)
        assert current._percent_pending == previous._percent_pending
        assert len(current._percent_pending) <= 2
    for _ in range(2):
        current.finish()
        previous.finish()
        assert vars(current) == vars(previous)
    for scanner in (current, previous):
        with pytest.raises(EvidenceBundleInvalid, match="finalized"):
            scanner.feed(b"")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"ordinary",
        b"%",
        b"%7",
        b"%g",
        b"%00",
        b"%aF",
        b"%FF",
        b"%G0",
        b"%0G",
        b"%%41",
        b"%%%",
        b"%4%41",
        b"%41%42%4",
        b"prefix%252541%25%32%35tail%",
        bytes(range(256)),
    ],
)
def test_decoder_matches_bytewise_at_every_split(tmp_path: Path, payload: bytes) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), RedactionSet.from_resolved_values(())
    ) as writer:
        for split in range(len(payload) + 1):
            _compare_decoding(writer, [b"", payload[:split], b"", payload[split:], b"", b""])
        _compare_decoding(writer, [payload[i : i + 1] for i in range(len(payload))])


def test_decoder_matches_bytewise_for_arbitrary_binary_streams(tmp_path: Path) -> None:
    rng = random.Random(20260914)
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), RedactionSet.from_resolved_values(())
    ) as writer:
        for _ in range(256):
            payload = rng.randbytes(rng.randrange(513))
            # Alternate arbitrary binary data with syntax-dense random data:
            # uniformly random bytes alone rarely exercise valid percent escapes.
            dense = bytes(rng.choice(b"%0123456789abcdefABCDEFgz") for _ in range(128))
            for data in (payload, dense):
                cuts = sorted([0, len(data)] + [rng.randrange(len(data) + 1) for _ in range(12)])
                _compare_decoding(writer, [data[a:b] for a, b in zip(cuts, cuts[1:])])


def test_ordinary_runs_do_not_append_each_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.evaluation.evidence import store

    appended: list[int] = []

    class CountingBytearray(bytearray):
        def append(self, value: SupportsIndex, /) -> None:
            appended.append(int(value))
            super().append(value)

    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), RedactionSet.from_resolved_values(())
    ) as writer:
        scanner = _RedactionScanner(writer)
        # Count scalar output work, not elapsed time on a contended CI runner.
        # Pending state is constructed before this spy: only the decoder's
        # output buffer is instrumented, and only the escape needs an append.
        monkeypatch.setattr(store, "bytearray", CountingBytearray, raising=False)
        assert scanner._decode_percent(b"x" * 4096 + b"%41" + b"y" * 4096) == (
            b"x" * 4096 + b"A" + b"y" * 4096
        )
        assert appended == [ord("A")]


def test_nested_percent_decoding_remains_one_pass_per_call(tmp_path: Path) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), RedactionSet.from_resolved_values(())
    ) as writer:
        for scanner_type in (_RedactionScanner, _BytewiseScanner):
            value = b"%252541%25"
            for expected in (b"%2541%", b"%41%", b"A%"):
                scanner = scanner_type(writer)
                # Feed every byte separately so each nested pass also crosses
                # the pending-byte boundary. A decoded '%' is not re-decoded
                # within the same pass, nor joined with that pass's next block.
                decoded = b"".join(scanner._decode_percent(bytes([byte])) for byte in value)
                decoded += bytes(scanner._percent_pending)
                scanner.finish()
                assert decoded == expected
                value = decoded


@pytest.mark.parametrize(
    ("payload", "secret", "rejected"),
    [
        (b"prefix-very-secret-value-tail", "very-secret-value", True),
        (b"prefix-%76%65%72%79%2dsecret-value-tail", "very-secret-value", True),
        (b"%76%65%72%79%2d%73%65%63%72%65%74%2d%76%61%6c%75%65", "very-secret-value", True),
        (b"%76%65%72%79%2d%73%65%63%72%65%74%2d%76%61%6c%75%65%", "very-secret-value%", True),
        (b"%76%65%72%79%2dsecret-value%7", "very-secret-value%7", True),
        (b"%2576%2565%2572%2579%252dsecret-value", "very-secret-value", False),
        (b"prefix-%76%65%72%79%2gsecret-value-tail", "very-secret-value", False),
        (b"prefix-VERY-SECRET-VALUE-tail", "very-secret-value", False),
    ],
)
def test_redaction_verdict_and_state_match_at_every_split(
    tmp_path: Path, payload: bytes, secret: str, rejected: bool
) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), RedactionSet.from_resolved_values((secret,))
    ) as writer:
        for split in range(len(payload) + 1):
            current = _RedactionScanner(writer)
            previous = _BytewiseScanner(writer)
            failures: list[str | None] = []
            for scanner in (current, previous):
                try:
                    for chunk in (payload[:split], b"", payload[split:], b""):
                        scanner.feed(chunk)
                    scanner.finish()
                    scanner.finish()
                except EvidenceBundleInvalid as error:
                    failures.append(str(error))
                else:
                    failures.append(None)
            assert failures[0] == failures[1]
            assert (failures[0] is not None) == rejected
            assert vars(current) == vars(previous)
