"""The runner's one-shot completion challenge, and the rule that scopes it.

WHY THIS EXISTS, and why it is not a claim-grounding check. The paid campaign
of 2026-09-25 scored 0 of 13 binary while the model's own finishes described
the screen it was looking at. Reading each finish against the frame artifact it
was bound to (``~/worktrees/osworld/INFRA.md`` §"CORRECTED (2026-09-25 PM)")
showed 3 of 3 claims were TRUE about the observation: the model looked, formed
a true claim, and then judged that screen to satisfy the task. Two mechanisms
that follow from the older, wrong diagnosis are unreachable or empty on this
corpus and are deliberately NOT built here:

* refusing a finish with no observation after the last mutation -- for 16 of 16
  ``status="done"`` finishes the finish's observation IS the newest one, bound
  structurally by ``PendingObservationToken``, ``ActionBatch.validate_for`` and
  ``parse_decision``'s validation pair, so the rule could never fire;
* comparing the model's notes against its claim -- the ``public_observations``
  agree with the ``reason`` in substance, so there is no inconsistency to catch.

What is left is the missing COMPARISON, so the gate asks for exactly that, once:
the claim is quoted back as a claim, the task is restated as it was stated, the
current observation's frames are re-attached, and the model is asked whether
THIS observation shows each thing the task requires -- answering either with the
same finish, unchanged, or with an action batch that closes the gap.

TWO PROPERTIES ARE THE WHOLE SAFETY ARGUMENT, and both are structural:

* the challenge count is BOUNDED and separate from ``max_decision_retries``
  (the precedent is ``MAX_EMPTY_TRUNCATION_RETRIES`` in ``episode.py``: spending
  one allowance must not eat another). At the default budget of 1 the SECOND
  ``done`` declaration is always accepted, so the gate can never drive an
  episode into ``_ModelFailure`` or an unscored seal. Worst case is one wasted
  cycle plus a warning-shaped ``error`` event.
* the gate lives in the RUNNER, not in the model client. A client that cannot
  re-present the end state cannot enforce the gate at all, which the driver
  reports loudly rather than degrading in silence; the client owns only the
  words, and returns them so the runner can publish exactly what was shown.

Nothing here names a benchmark, an application or a task family: the gate fires
on the protocol's own terminal action, and its content is the model's own words
plus the task text it was handed at reset. A task with nothing left to observe
-- terminal-only, or already correct -- is explicitly not blocked, because
re-declaring the same finish is a permitted reply.
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.evaluation.protocol import ActionBatch, FinishAction
from local_operator.evaluation.runner.model import CompletionChallenger, EpisodeTurn


def finish_claim(batch: ActionBatch) -> FinishAction:
    """The batch's finish action.

    A terminal action is the ONLY action in its batch (``ActionBatch``'s
    ``_bind_and_isolate_actions``), so there is exactly one to find and no
    question about which claim was declared. Raising rather than returning
    ``None`` keeps the caller free of a branch that could never be taken for a
    batch the caller has already classified as a finish.
    """

    for action in batch.actions:
        if isinstance(action, FinishAction):
            return action
    raise ValueError("batch carries no finish action")


class CompletionGate:
    """One episode's completion gate: whether to challenge, and how often.

    The counter is per EPISODE and lives here rather than on the model client,
    which is rebuilt per episode anyway: the bound is a property of the
    episode's protocol, not of the client that happens to answer.
    """

    def __init__(
        self,
        *,
        client: Any,
        enabled: bool,
        challenges: int,
    ) -> None:
        self._client = client
        self._enabled = enabled
        self._budget = challenges
        self._fired = 0

    @property
    def fired(self) -> int:
        """How many challenges this episode has made."""

        return self._fired

    def should_challenge(self, claim: FinishAction) -> bool:
        """Whether this terminal claim is the one to ask the model to re-check.

        Four conditions and each one is load-bearing:

        * the gate is enabled (``EpisodeConfig.completion_gate``), so the
          control arm of a campaign is a config flip rather than a code path;
        * the budget is not spent, which is what bounds the exchange;
        * the claim is ``done``. A ``failed`` or ``infeasible`` finish is the
          model reporting that it could NOT finish -- there is no completion to
          confirm, and challenging it would spend a cycle asking the model to
          re-examine an admission of failure;
        * the client can actually re-present the end state -- it satisfies
          :class:`~local_operator.evaluation.runner.model.CompletionChallenger`.
          A client that cannot must not be handed a gate whose only effect would
          be a recorded challenge the model never saw.
        """

        if not self._enabled or self._fired >= self._budget:
            return False
        if claim.status != "done":
            return False
        return isinstance(self._client, CompletionChallenger)

    async def challenge(self, batch: ActionBatch, turns: Sequence[EpisodeTurn]) -> str:
        """Append the challenge and return the text the model was shown.

        The task instruction is the RESET observation's text (``turns[0]``),
        which is the task as the adapter published it -- including the task
        files whose literal is interpolated at runtime, which is the whole point
        of reading the observation rather than a task id.

        The counter advances only after the client has accepted the turn: a
        client that raises did not show the model a challenge, and charging it
        here would spend the allowance on a turn that never happened.
        """

        observation = turns[-1].observation
        instruction = turns[0].observation.text or ""
        client = self._client
        if not isinstance(client, CompletionChallenger):
            # ``should_challenge`` is the only caller and it refuses exactly this
            # case, so reaching here is a caller bug -- and it is raised rather
            # than swallowed because the alternative is a challenge counted as
            # fired that the model never saw.
            raise TypeError("model client cannot re-present the end state")
        text = await client.challenge_completion(
            observation,
            tuple(turns),
            batch=batch,
            instruction=instruction,
        )
        self._fired += 1
        return text
