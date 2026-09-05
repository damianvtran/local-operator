"""One negotiated source for action admission, model instructions and evidence.

The neutral protocol still parses historic Unicode TypeAction records. Execution
restrictions belong here, not on that data model: an ASCII keyboard backend must
reject a lossy batch before its first click, while a Unicode backend may accept it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

from local_operator.evaluation.protocol import (
    NAMED_KEYS,
    ActionBatch,
    AskUserAction,
    ComputerAction,
    PasteTextAction,
    TypeAction,
)


class ActionAdmissionError(ValueError):
    """Known pre-dispatch invalid input; no environment mutation was attempted."""


@dataclass(frozen=True)
class ActionSurface:
    paste_text: bool = False
    type_text_mode: Literal["unicode", "ascii"] = "unicode"
    ask_user: bool = True

    @property
    def models(self) -> tuple[Any, ...]:
        return tuple(
            model
            for model in get_args(get_args(ComputerAction)[0])
            if (model is not PasteTextAction or self.paste_text)
            and (model is not AskUserAction or self.ask_user)
        )

    @property
    def named_keys(self) -> tuple[str, ...]:
        return tuple(sorted(NAMED_KEYS))

    def schema(self) -> dict[str, Any]:
        # Include the execution restriction in the identity: the same neutral
        # field schema on a lossy keyboard is not the same admitted action surface.
        return {
            "actions": [model.model_json_schema() for model in self.models],
            "type_text_mode": self.type_text_mode,
            "named_keys": list(self.named_keys),
        }

    def validate_batch(self, batch: ActionBatch) -> None:
        for action in batch.actions:
            if type(action) not in self.models:
                raise ActionAdmissionError("action kind is not supported by this adapter")
            if isinstance(action, TypeAction) and self.type_text_mode == "ascii":
                if not action.text.isascii():
                    raise ActionAdmissionError(
                        "type supports only ASCII on this adapter; "
                        "use paste_text with an explicit chord"
                        if self.paste_text
                        else "type supports only ASCII on this adapter"
                    )


LEGACY_ACTION_SURFACE = ActionSurface()
