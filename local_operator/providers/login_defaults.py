"""The ONE rule for what a successful login writes to config.

Why this module exists: this decision had two implementations — one in
``providers/auth_cli`` for ``local-operator login``, one in ``tui/app`` for
``/login`` — and they drifted on every axis that mattered. They wrote different
hosting ids for the same provider (raw ``provider_id`` vs
``credential_provider_id``), decided "is the current hosting broken?" by
different tests (a registry lookup vs a TUI-only state flag), and therefore
produced different config for identical input: logging into ``xai-oauth`` from a
corrupted config yielded ``xai`` + ``grok-3`` on one path and ``xai-oauth`` +
the dead model on the other.

That divergence is not incidental to the bug this module was extracted for. The
original defect — a config naming a provider the registry does not own leaving
the session unbootable and unrepairable — survived precisely because the
"recover from a bad provider" logic lived in more than one place and no single
place was responsible for being right. Fixing it in two copies would have
rebuilt the same trap. Both front ends now call :func:`plan_login_defaults`, so
a change to the policy is a change to one function.

Pure and side-effect-free by construction: it reads no config file, writes no
config file, and prints nothing. Callers own their own ``ConfigManager`` and
their own reporting surface (stdout for the CLI, a transcript notice for the
TUI), which is the only thing that legitimately differs between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_operator.model.defaults import default_model_for
from local_operator.providers.registry import (
    credential_provider_id,
    get_provider_definition,
)


@dataclass(frozen=True)
class LoginDefaults:
    """What a login should write, and how to describe it.

    ``model_name`` is ``None`` when the caller must leave the stored value
    alone, and ``""`` when it must CLEAR it — a distinction that carries real
    weight (see :func:`plan_login_defaults`), so it cannot be collapsed into a
    single falsy check.
    """

    #: The hosting id to write, or ``None`` to leave hosting untouched.
    hosting: str | None
    #: ``None`` = leave as-is, ``""`` = clear it, otherwise the id to write.
    model_name: str | None
    #: One-line receipt for the user, or ``None`` when nothing was written.
    receipt: str | None
    #: True when this replaces a hosting the registry does not own, as opposed
    #: to filling in an empty one. Callers use it only for wording.
    repairing: bool


def is_unusable_hosting(hosting: str | None) -> bool:
    """True when ``hosting`` is set but names no provider the engine accepts.

    Asked of the registry rather than a hardcoded id list, for the same reason
    the startup resolver does it that way: ``get_provider_definition`` resolves
    legacy aliases (``noop`` -> ``test``), so a membership test against ids
    would classify a WORKING alias config as corrupt and silently repoint a
    user's default. Empty hosting is not "unusable" — it is the separate
    nothing-configured case, which has its own first-run handling.
    """
    return bool(hosting) and get_provider_definition(str(hosting)) is None


def plan_login_defaults(
    provider_id: str, hosting: str | None, model_name: str | None
) -> LoginDefaults:
    """Decide what a just-completed login for ``provider_id`` should write.

    Three cases, in order:

    1. **Hosting already set and usable** — write nothing. A user logging into
       a second provider to switch models later has not asked to change their
       default, and silently repointing it would be a surprise.
    2. **Hosting empty** — adopt this provider and its default model. Without
       this, ``login`` stored a credential but left hosting empty, so the very
       command the "not configured" error recommends looped straight back to
       the same error.
    3. **Hosting set but unusable** — REPLACE it. That config cannot boot, and
       the error it produces recommends this command as the remedy, so
       "already set, leave it alone" would loop one level deeper.

    The model is resolved through ``credential_provider_id`` and, when
    repairing, is always overwritten — cleared if the provider has no known
    default. Both halves of that are load-bearing:

    - A login FLAVOUR (``xai-oauth``, ``openai-device``, ``zai-oauth``) is an
      authentication route, not a hosting id, and it has no default model of
      its own. Writing the raw flavour id as hosting left the stale model in
      place beside it, producing a config that BOOTS — ``configure_model``
      accepts the pair — and then fails at stream time on a model the provider
      never heard of. Trading a boot failure the app explains for a runtime
      failure it cannot is strictly worse than the bug being repaired.
    - Clearing (rather than keeping) a model with no known default is what
      makes the repair safe for ``alibaba-token-plan``, which resolves to no
      default at all. An empty ``model_name`` is a state the startup resolver
      already handles and reports precisely; a model belonging to a provider
      that never existed is not.
    """
    if hosting and not is_unusable_hosting(hosting):
        return LoginDefaults(hosting=None, model_name=None, receipt=None, repairing=False)

    repairing = is_unusable_hosting(hosting)
    # The credential's storage id is the real hosting: an OAuth flavour stores
    # under the provider it authenticates, and that is what the app must point
    # at. This is also what gives the flavour a default model to inherit.
    resolved = credential_provider_id(provider_id)
    default_model = default_model_for(resolved) or ""

    if repairing:
        # Always overwrite: the stored model belonged to the provider being
        # replaced. "" clears it rather than leaving a dead id behind.
        model_to_write: str | None = default_model
        receipt = f"replaced unusable hosting '{hosting}' with '{resolved}'"
        if default_model:
            receipt += f", model to '{default_model}'"
        else:
            # Named explicitly: a cleared model changes what the next launch
            # does, so it must not be a silent side effect of logging in.
            receipt += ", cleared the model it left behind (no default known)"
    else:
        # First-run: only fill an EMPTY model, so a user who deliberately chose
        # one keeps it.
        model_to_write = default_model if (default_model and not model_name) else None
        receipt = f"set default hosting to '{resolved}'"
        if model_to_write:
            receipt += f", model to '{model_to_write}'"

    return LoginDefaults(
        hosting=resolved,
        model_name=model_to_write,
        receipt=receipt,
        repairing=repairing,
    )
