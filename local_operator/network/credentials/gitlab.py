"""The ``gitlab`` forge credential — the GitHub adapter's sibling: a STUB.

WHY THIS FILE EXISTS AND WHY IT IS (ALMOST) EMPTY. ``mesh-consent-provisioning.md``
§3.6 fixed the shape for GitLab (sources, delivery, strong mode, lease semantics)
and deliberately left it OUT of the S2 slice: nothing exists today (no ``glab``
credential path anywhere in ``local_operator/**``), so building it inside S2
would have meant a second broker path, a second real-git loopback and a second
share surface riding one review — the note tracks it as its own slice instead of
letting it hide here. This module is that slice's landing point: it pins the
names the note DECIDED, so the follow-up cannot invent a second spelling of the
key or the secret name, and it contains no behaviour at all.

WHAT THE SLICE BUILDS, from §3.6 (the position, in substance):

- **Sources:** owner-side ``GITLAB_TOKEN`` (store secret) or the owner's
  ``glab``-stored login; a scoped, expiring PAT is the recommended durable
  source. Strong mode mints-or-uses a DEDICATED token rather than sharing the
  operator's primary login — GitLab's revocable-per-token PAT model is closer
  to the GitHub App than GitHub's OAuth token is.
- **Delivery:** ``GITLAB_TOKEN`` in the child env plus a **gitlab.com-scoped**
  helper — the same F1 close pattern (reset the helper list for gitlab.com, add
  one brokered helper), a sibling of ``github.py``'s delivery, never a fork of
  the broker.
- **Refresh/lease/revocation:** identical semantics to §3.3; ``glab`` refresh is
  owner-side like ``gh``'s.

Nothing imports this module yet; that is the stub's contract, not an oversight.
"""

from __future__ import annotations

#: The placement key AND the provider spelling (§3.6), the same one-string rule
#: ``github.py`` states: a second spelling is how a share and a borrow come to
#: disagree about which key they mean.
GITLAB_KEY = "gitlab"

#: The stored PAT-class secret's name. ``GITLAB_TOKEN`` on purpose: it is the
#: spelling a ``glab``/CI user already recognises, the sibling of
#: ``GITHUB_TOKEN``.
TOKEN_SECRET_NAME = "GITLAB_TOKEN"


def is_gitlab_key(key: str) -> bool:
    """Whether ``key`` names this adapter — the one classifier, no aliases."""
    return key == GITLAB_KEY
