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
from typing import Any

FIELD = "bedrock_messages_prefix_v1"
THINKING = frozenset({"thinking", "redacted_thinking"})
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _unmarked(block: dict[str, Any]) -> dict[str, Any]:
    # Only remove the protocol field, never keys in a schema, tool input or
    # other nested data. These fields can have the same spelling by design.
    if "cache_control" not in block:
        return block
    return {k: v for k, v in block.items() if k != "cache_control"}


class Prefix:
    """Two incremental views: exact lineage and positive portable mismatch.

    AWS permits removing *leading* thinking. A difference only in prior
    thinking is therefore not on its own proof of a bad prefix. Keep that
    case unknown unless the producer matches the prefix before a proved drop.
    Both views include roles and block/message boundaries, and never sort
    the keys of tool schemas or arguments (their printed order can matter).
    """

    def __init__(self, route: str, scope: str | None, body: dict[str, Any]) -> None:
        self._exact = hashlib.sha256()
        for value in (
            FIELD,
            route,
            scope,
            [_unmarked(block) for block in body.get("system", [])],
            [_unmarked(tool) for tool in body.get("tools", [])],
        ):
            self._add(self._exact, value)
        self._portable = self._exact.copy()

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

    def block(self, block: dict[str, Any]) -> None:
        encoded = self._encoded(_unmarked(block))
        self._exact.update(encoded)
        if block.get("type") not in THINKING:
            self._portable.update(encoded)

    def end(self) -> None:
        self._add(self._exact, None)
        self._add(self._portable, None)

    def message(self, row: dict[str, Any]) -> None:
        self.begin(row["role"])
        for block in row["content"]:
            self.block(block)
        self.end()

    def witness(self) -> tuple[str, str]:
        return self._exact.hexdigest(), self._portable.hexdigest()

    def fork(self) -> Prefix:
        # When a splice first diverges, remember the original lineage as well.
        # This proves a later dependency without storing the original history.
        original = object.__new__(Prefix)
        original._exact = self._exact.copy()
        original._portable = self._portable.copy()
        return original

    def bind_assistant(self, payload: dict[str, Any], calls: list[dict[str, Any]]) -> None:
        args = {call["id"]: call["args"] for call in calls}
        if len(args) != len(calls):
            return  # Cannot reconstruct ambiguous tool inputs by ID.
        self.begin("assistant")
        witnesses = {}
        for index, block in enumerate(payload["items"]):
            if block["type"] in THINKING:
                witnesses[str(index)] = list(self.witness())
            elif block["type"] == "tool_use":
                block = {**block, "input": args[block["id"]]}
            self.block(block)
        payload[FIELD] = witnesses


def read_witnesses(payload: dict[str, Any]) -> dict[str, list[str]]:
    value = payload.get(FIELD)
    if not isinstance(value, dict):
        # Old sessions and unproven stream shapes keep the old behavior.
        return {}
    return {
        key: pair
        for key, pair in value.items()
        if isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(digest, str) and _DIGEST.fullmatch(digest) for digest in pair)
    }
