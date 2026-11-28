"""Which provider a remote run declares, and why it is not this device's choice.

A REMOTE RUN ALWAYS DECLARES ``radient`` HOSTING. The far end runs the task under
a run token Radient issues to a host RADIENT OPERATES: a hosting value is the
token's destination as much as the model's, so naming any other provider would
address that token to a third party. Radient's own deployment enforces this at
admission (its remote-compute server admits only the ``radient`` hosting), and
the refusal there is a 400 ``invalid_request`` that reads like a client bug —
which is exactly what filling the field from this device's local model config
used to produce. So the field is decided here, once, for every surface.

THE DEVICE'S CONFIG IS NOT AN INPUT. ``hosting``/``model_name`` in
``config.yml`` describe what THIS machine runs locally and which provider keys it
holds; a remote run inherits neither. Reading them here is how a device set to
``anthropic``/``claude-opus-5-5`` sent a body Radient refused.

THE MODEL IS THE OPERATOR'S OR ``auto``. ``--model`` is a real choice about the
work and travels as given; everything else is left to the far end, which knows
which models it serves.

WHY A REFUSAL AND NOT A SILENT REWRITE. An explicit ``--hosting anthropic`` is
an instruction this surface cannot follow, and quietly running the job on a
different provider than the one asked for is the failure mode the CLI's
``LOCAL_ONLY_FLAGS`` list exists to avoid. It is refused before a request is
composed, with the remedy in the sentence the caller owns (i18n key
``wire.remote_compute.hosting_radient_only``).

IMPORTS ARE STDLIB ONLY, beside ``arguments.py``: this is imported by the exec
entry path's remote branch and by the desktop route.
"""

from __future__ import annotations

#: The one hosting a remote run may declare; see the module docstring.
REMOTE_HOSTING = "radient"

#: What the far end runs when the operator names no model.
REMOTE_DEFAULT_MODEL = "auto"


class RemoteHostingRefused(ValueError):
    """An explicit hosting this surface cannot send a remote run to.

    Carries the offered value so the caller can name it back to the operator: the
    sentence itself belongs to the surface (the CLI's keyed message), not here.

    A ``ValueError`` so existing ``except ValueError`` refusals on the remote path
    keep reporting this as a refused invocation rather than a crash.
    """

    def __init__(self, hosting: str) -> None:
        self.hosting = hosting
        super().__init__(
            f"a remote run uses {REMOTE_HOSTING} hosting, so {hosting!r} cannot be sent"
        )


def remote_hosting_model(
    *, hosting: str | None = None, model: str | None = None
) -> tuple[str, str]:
    """The ``(hosting, model)`` pair a remote run declares.

    ``hosting`` may be absent or ``radient`` and nothing else. Comparison is
    case-insensitive after stripping: ``--hosting Radient`` names the same
    provider, and refusing it would be a spelling lesson rather than a rule. An
    empty string is treated as absent — it names no provider — matching how the
    exec parser's other optional strings read "not passed".

    ``model`` is stripped and passed through, with
    :data:`REMOTE_DEFAULT_MODEL` when it is absent or empty.
    """
    offered = (hosting or "").strip()
    if offered and offered.lower() != REMOTE_HOSTING:
        raise RemoteHostingRefused(offered)
    return REMOTE_HOSTING, (model or "").strip() or REMOTE_DEFAULT_MODEL


__all__ = [
    "REMOTE_DEFAULT_MODEL",
    "REMOTE_HOSTING",
    "RemoteHostingRefused",
    "remote_hosting_model",
]
