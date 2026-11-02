"""Opaque server routing for one logical turn, outside durable request data."""

from dataclasses import dataclass, field


@dataclass(repr=False)
class CodexTurnState:
    """First server value wins until an actual account or destination changes.

    A new instance belongs to each main turn, not the pooled client, cache
    lineage, or transcript. Generations prevent a late response from restoring
    an earlier route after failover. Nothing here is a cache-hit guarantee.
    """

    _attempt: tuple[str, str] | None = field(default=None, repr=False)
    _endpoint: str | None = field(default=None, repr=False)
    _token: str | None = field(default=None, repr=False)
    _generation: int = field(default=0, repr=False)

    def observe_attempt(self, provider: str, owner: str) -> None:
        if self._attempt != (provider, owner):
            self._attempt = (provider, owner)
            self._endpoint = None
            self._token = None
            self._generation += 1

    def bind_endpoint(self, endpoint: str) -> tuple[int, str | None]:
        if self._endpoint != endpoint:
            self._endpoint = endpoint
            self._token = None
            self._generation += 1
        return self._generation, self._token

    def receive(self, generation: int, value: str | None) -> None:
        # A header is opaque but cannot be unlimited or contain framing bytes.
        # Reject rather than truncate: a modified token routes a different turn.
        if (
            generation == self._generation
            and self._token is None
            and isinstance(value, str)
            and 0 < len(value) <= 4096
            and all(0x21 <= ord(char) <= 0x7E for char in value)
        ):
            self._token = value
