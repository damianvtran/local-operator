"""Local prefix proof for Bedrock's ordinary Sonnet 5.5 Messages route.

AWS rejects an edited prefix by default; do not opt into a beta or change
what gets published. The proof is over the ordered wire projection, not the
journal rows (merged tool results, repaired arguments, and media matter).
https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-thinking-block-binding.html
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

FIELD = "bedrock_messages_prefix_v2"
_V1_FIELD = "bedrock_messages_prefix_v1"
THINKING = frozenset({"thinking", "redacted_thinking"})
_DIGEST = re.compile(r"[0-9a-f]{64}")
_LINK = b"bedrock-messages-native-link-v2\0"
_EMPTY_TAIL = hashlib.sha256(_LINK).digest()


@dataclass(frozen=True)
class NativeTail:
    """Normalized chain tip and the number of native blocks actually sent."""

    digest: bytes
    count: int


@dataclass(frozen=True)
class Witness:
    exact: str
    portable: str
    native: NativeTail | None


def _unmarked(block: dict[str, Any]) -> dict[str, Any]:
    # Only remove the protocol field, never keys in a schema, tool input or
    # other nested data. These fields can have the same spelling by design.
    if "cache_control" not in block:
        return block
    return {k: v for k, v in block.items() if k != "cache_control"}


class Prefix:
    """Bound exact, portable and native-only views of one Messages attempt.

    Each native link hashes the previous tip, the block's portable position
    (including route/scope), and the block itself. Seed the first survivor
    from its own witness so an authorized leading cut has the same tail as
    before. Equal tips prove that the last N links match, but only if the
    minting attempt actually sent at least N blocks: the seed may remember
    older blocks that were not on that attempt's wire. No retained lineage
    grows with the transcript beyond one fixed-size witness per native block.

    The exact and portable views include roles and block/message boundaries;
    tool schema and argument key order is significant in both.
    """

    def __init__(
        self,
        route: str,
        scope: str | None,
        body: dict[str, Any],
        native_seed: bytes | None = None,
    ) -> None:
        self._exact = hashlib.sha256()
        for value in (
            _V1_FIELD,  # Keep the exact/portable hashing domain for v1 journals.
            route,
            scope,
            [_unmarked(block) for block in body.get("system", [])],
            [_unmarked(tool) for tool in body.get("tools", [])],
        ):
            self._add(self._exact, value)
        self._portable = self._exact.copy()
        self.native_seed = native_seed
        self._native_tip = native_seed if native_seed is not None else _EMPTY_TAIL
        self._native_count = 0

    @staticmethod
    def _encoded(value: Any) -> bytes:
        # These JSON records contain no raw newline delimiter. Process one
        # block at a time; don't materialize or retain a second conversation.
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"

    @classmethod
    def _add(cls, digest: Any, value: Any) -> None:
        digest.update(cls._encoded(value))

    def begin(self, role: str) -> None:
        self._add(self._exact, role)
        self._add(self._portable, role)

    def block(self, block: dict[str, Any], witness: Witness | None = None) -> None:
        encoded = self._encoded(_unmarked(block))
        self._exact.update(encoded)
        if block.get("type") not in THINKING:
            self._portable.update(encoded)
            return
        # A legacy first block can start a fresh chain but cannot recover an
        # older chain's tip. Exact replay still wins without that anchor.
        if not self._native_count and witness is not None and witness.native is not None:
            self.native_seed = self._native_tip = witness.native.digest
        link = hashlib.sha256(_LINK + self._native_tip + self._portable.digest())
        link.update(encoded)
        self._native_tip = link.digest()
        self._native_count += 1

    def end(self) -> None:
        self._add(self._exact, None)
        self._add(self._portable, None)

    def message(self, row: dict[str, Any]) -> None:
        self.begin(row["role"])
        for block in row["content"]:
            self.block(block)
        self.end()

    def _is_native_suffix(self, expected: NativeTail) -> bool:
        return 0 < self._native_count <= expected.count and self._native_tip == expected.digest

    def invalidates(self, expected: Witness, before_drops: Prefix | None) -> bool:
        """Exact replay wins; only a leading cut may change a witnessed prefix."""
        if self._exact.hexdigest() == expected.exact:
            return False
        if self._portable.hexdigest() != expected.portable:
            return True
        native = expected.native
        if native is not None and self._native_count:
            return not self._is_native_suffix(native)
        # All-thinking removal before this selector is valid. V1 cannot prove
        # any native-only mismatches beyond the exact pre-splice lineage; v2
        # can also catch a newly removed dependency after an earlier leading cut.
        return before_drops is not None and (
            before_drops._exact.hexdigest() == expected.exact
            or (native is not None and before_drops._is_native_suffix(native))
        )

    def fork(self) -> Prefix:
        # Remember the before-drops tip without storing the original history.
        original = object.__new__(Prefix)
        original._exact = self._exact.copy()
        original._portable = self._portable.copy()
        original.native_seed = self.native_seed
        original._native_tip = self._native_tip
        original._native_count = self._native_count
        return original

    def bind_assistant(self, payload: dict[str, Any], calls: list[dict[str, Any]]) -> None:
        args = {call["id"]: call["args"] for call in calls}
        if len(args) != len(calls):
            return  # Cannot reconstruct ambiguous tool inputs by ID.
        self.begin("assistant")
        witnesses = {}
        for index, block in enumerate(payload["items"]):
            if block["type"] in THINKING:
                witnesses[str(index)] = [
                    self._exact.hexdigest(),
                    self._portable.hexdigest(),
                    self._native_tip.hex(),
                    self._native_count,
                ]
            elif block["type"] == "tool_use":
                block = {**block, "input": args[block["id"]]}
            self.block(block)
        payload[FIELD] = witnesses


def read_witnesses(payload: dict[str, Any]) -> dict[str, Witness]:
    """Read v2 or its v1 predecessor; absent proofs do not govern legacy replay."""
    witnesses = {}
    for field, length, hashes in ((_V1_FIELD, 2, 2), (FIELD, 4, 3)):
        rows = payload.get(field)
        if not isinstance(rows, dict):
            continue
        for index, row in rows.items():
            if (
                not isinstance(row, list)
                or len(row) != length
                or not all(isinstance(s, str) and _DIGEST.fullmatch(s) for s in row[:hashes])
            ):
                continue
            native = None
            if field == FIELD:
                count = row[3]
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    continue
                native = NativeTail(bytes.fromhex(row[2]), count)
            witnesses[index] = Witness(row[0], row[1], native)
    return witnesses
