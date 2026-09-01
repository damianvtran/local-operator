"""Who answers when a model asks the user a question mid-episode.

Benchmarks that model a human collaborator let the policy emit an
``AskUserAction``. The environment cannot answer it, so the runner suspends the
step loop and asks a responder. Keeping that behind a Protocol means an
unattended benchmark run, a scripted user simulator, and an interactive
operator session all drive the same episode code.

``ask`` returns ``None`` to abandon: the question cannot be answered within the
deadline. The runner treats that as a harness-sourced cancellation rather than
an error, because an unanswered ask leaves an outstanding exchange open in
``HostVerifier``, and an episode may not proceed past an ask it never resolved.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class UserResponder(Protocol):
    """Answers one ask-user prompt raised during an episode.

    ``deadline_ms`` is the budget for this single answer, not for the episode.
    An implementation that cannot answer inside it must return ``None`` rather
    than block: the runner is holding an outstanding adapter exchange open for
    the duration of this call.
    """

    async def ask(self, prompt: str, deadline_ms: int) -> str | None: ...


class NullUserResponder:
    """Refuses every question immediately.

    This is the default because an unattended evaluation run has nobody to ask,
    and blocking would stall a batch indefinitely. Refusing immediately turns an
    ask into a deterministic, promptly-cancelled episode instead of a hang, and
    the resulting bundle records exactly why it ended.
    """

    async def ask(self, prompt: str, deadline_ms: int) -> str | None:
        del prompt, deadline_ms
        return None
