"""The ``github`` credential: a forge bearer from the owner's source ladder.

WHAT THIS CREDENTIAL IS, AND WHY IT IS NOT A PROVIDER LOGIN. Every other key in
``placement`` names a login this device holds in its own ``auth.db`` and serves
through ``AuthStore``. ``github`` has no row anywhere: the OWNER serves a bearer
resolved at serve time from the source ladder below, hands it to a borrower over
the existing broker wire, and the borrower uses it for ``git`` and ``gh`` —
never for provider calls. So this module is the adapter the broker dispatches
to, plus the borrower-side delivery (env construction, the git credential
helper, and the self-revoke belt).

THE SOURCE LADDER (``docs/design/mesh-consent-provisioning.md`` §3.2), strongest
first, resolved by :func:`resolve_source` on the owner at EVERY serve:

1. ``GITHUB_APP`` — one JSON blob (``app_id``, ``installation_id``, PEM private
   key) in the owner's encrypted secret store; today's mint path runs unchanged.
   The optional STRONGER arm: server-side narrowed at mint and revocable per
   token (``DELETE /installation/token``). A configured-but-broken App REFUSES
   rather than falling through — falling through would silently lend a wider
   credential than the operator configured.
2. ``GITHUB_TOKEN`` — a PAT-class secret in the same store: the durable,
   revocable-at-the-forge arm the guide teaches (a fine-grained PAT scoped to
   the designated repositories with an expiry). Not minted here, so there is no
   server-side revoke handle and the revocation receipt says so (§3.3).
3. The owner's own ``gh`` CLI login — the zero-setup route, resolved BY ASKING
   ``gh`` ITSELF: ``gh auth token`` when the token lives in the OS keychain
   (the default on macOS — ``hosts.yml`` then carries the login with no
   ``oauth_token``), with gh's own hosts file read first when it holds one
   (``--insecure-storage`` installs). The owner is the trusted side; the
   borrower never reads it, and nothing new is stored — ``gh``'s tooling stays
   the source of truth.

Not-a-token material fails closed nowhere else: with no arm at all the adapter
refuses ``no_local_credential`` in the reader's own terms (push and PR-write are
unavailable until one of the three is configured; public clones and non-GitHub
work are unaffected).

THE NARROWING TRAVELS WITH THE GRANT (field finding, live remote-node E2E). The
allow-list is an owner-side designation, but its ENFORCEMENT is the borrower's
helper, so it must reach the borrower's config — a list that lives only on the
owner shipped a live node whose every push died at ``could not read Username``
because its own config had none. The owner now attaches its CURRENT designation
to every github grant (:func:`grant_narrowing` → ``Grant.narrowing``), and the
borrower writes it into its own config before the child runs
(:func:`remember_delivered_repositories` — idempotent, receipted, never fatal).
An owner with no designation sends nothing, and a device with no list still
refuses: the fail-closed backstop is unchanged, it is just no longer a manual
step on every node.

DEVICE-SCOPED BY CONSTRUCTION (design F3). The borrower side has no
rail-authenticated session identity — a same-uid process can claim any
``for_session`` — so a session-scoped grant would enforce nothing. A session id
therefore travels ONLY as attribution (audit rows, cache keying), and both
directions refuse a session-scoped row by name: the share verb at the document
(``placement.py``) and the owner at serve time (:meth:`MeshCredentialBroker`
dispatch in ``owner.py``). The disclosure that follows — any process or session
on the borrower node can borrow while the share stands — is stated on the share
receipt and in the network guide (the mandated T7(b) copy), and §3.4's honesty
clause rides it for the token arms: only the App is narrowed server-side, so
the end the operator wants must be named for the arm that served.

DEVICE-SCOPED BY CONSTRUCTION (design F3). The borrower side has no
rail-authenticated session identity — a same-uid process can claim any
``for_session`` — so a session-scoped grant would enforce nothing. A session id
therefore travels ONLY as attribution (audit rows, cache keying), and both
directions refuse a session-scoped row by name: the share verb at the document
(``placement.py``) and the owner at serve time (:meth:`MeshCredentialBroker`
dispatch in ``owner.py``). The disclosure that follows — any process or session
on the borrower node can borrow while the share stands — is stated on the share
receipt and in the network guide (the mandated T7(b) copy).

THE F1 CLOSE, WHICH IS WHY THE ENV IS SAFE. A node's global
``credential.helper=store`` would otherwise receive the brokered token on
``approve`` and write it to ``~/.git-credentials``. The delivery therefore resets
the helper list FOR GITHUB.COM ONLY before adding the brokered helper
(``GIT_CONFIG_COUNT=3`` + the empty first entry) — spike-proven: with the reset,
github.com resolves to the brokered helper alone while other hosts keep their
configured helpers; without it the store receives and persists the token. The
helper itself re-checks ``protocol``/``host`` and requires a well-formed ``path``
in the configured allow-list, so a lookalike host or a naive
``git credential fill`` extraction is refused. `git` rides the helper;
``gh`` rides ``GH_TOKEN``/``GITHUB_TOKEN`` (gh has no credential-helper
protocol, so the env is the only carrier for that leg).

MINT-REVOKE (design F4). The owner keeps a bounded in-memory registry of
minted-but-unrevoked tokens per ``(key, holder)`` and calls
``DELETE /installation/token`` at each token's window end on the broker's own
loop; ``credential revoke`` (and only that path today) revokes immediately via
the ``credential_revoke`` local op. Revoke is idempotent: a token that no longer
authenticates IS the goal state. A failed call is recorded and retried while
this process lives; if no revoke can ever be delivered, the token falls back to
its own 60-minute ceiling. The borrower holds a best-effort belt: it self-revokes
the tokens it fetched at their window end.

Stdlib only at import: the relay imports the credentials package at
construction, so ``jwt`` and ``httpx`` are imported inside the functions that
reach the wire.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger("local_operator.network.credentials.github")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The placement key AND the provider spelling: this credential is not a
#: provider login, so the two are the same string on purpose (a second spelling
#: is how a share and a borrow come to disagree about which key they mean).
GITHUB_KEY = "github"

#: The placement row's kind. ``CredentialRef.kind`` on the wire carries it too.
#: The kind names the ROW (one github placement), not the arm that served this
#: window: the ladder can change arm between windows (a gh login today, an App
#: tomorrow) while the share stands. Per-source wording therefore branches on
#: ``resolve_source``, never on this string.
GITHUB_KIND = "github-app"

#: The owner's secret-store entry. ONE JSON blob: ``app_id``, ``installation_id``,
#: ``private_key`` (PEM). One name, so the checklist is one command; see
#: ``local_operator/guides/network/GUIDE.md``.
APP_SECRET_NAME = "GITHUB_APP"

#: Where the designated repositories live. The SAME key is read by the owner (the
#: mint narrows to it — and refuses when it is empty rather than minting a token
#: that covers the whole installation) and by the borrower's helper as its
#: use-time backstop.
REPOSITORIES_PATH: tuple[str, ...] = ("network", "credentials", "github", "repositories")

#: The default: none. An empty list is a REFUSAL at mint time, never "no
#: narrowing" — ``repositories`` absent on the wire would grant the token access
#: to everything the installation covers.
DEFAULT_REPOSITORIES: tuple[str, ...] = ()

#: The API surface. ``2022-11-28`` is the version the shipped endpoint shapes
#: were verified against (mint: ``repositories`` + ``permissions``; revoke:
#: ``DELETE /installation/token`` -> 204, idempotent).
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

#: The permissions the mint asks for, fixed rather than configurable: the two a
#: git push and a pull request need. A narrower request would refuse at the
#: first ``git push``; a wider one would be a capability nobody asked for.
GITHUB_PERMISSIONS: dict[str, str] = {"contents": "write", "pull_requests": "write"}

#: The stored PAT-class secret (the ladder's second arm): ONE opaque short line,
#: the token itself — a fine-grained PAT scoped to the designated repositories
#: with an expiry is the shape the guide teaches. A ``GITHUB_TOKEN``-class NAME
#: on purpose: it is the spelling a ``gh``/CI user already recognises.
TOKEN_SECRET_NAME = "GITHUB_TOKEN"

#: The ladder's arms, strongest first (§3.2). ``""`` — no arm — is
#: ``no_local_credential`` at serve time. THE ONE ORDER: the lender resolves
#: with :func:`resolve_source` and the receipts name the arm with it, so
#: "which source serves" cannot drift between serving and the copy.
SOURCE_APP = "app"
SOURCE_TOKEN = "token"
SOURCE_GH = "gh"

#: The ladder's "we cannot tell" answer: the encrypted store EXISTS but cannot
#: be read, so an arm may be present that must not be skipped (review round 1,
#: M1). Deliberately distinct from ``""`` — an empty ladder falls through to
#: ``no_local_credential``; an unreadable store REFUSES by name, because serving
#: a wider arm than one that could not be read is exactly the silent downgrade
#: the ladder forbids.
SOURCE_UNREADABLE = "unreadable"

#: How much of ``gh``'s hosts file one read may take. gh writes a few hundred
#: bytes; the bound exists so a pathological file cannot make the lender read
#: it wholesale — and an oversized file is refused BY NAME, never parsed
#: partially (a half-read credential file must not serve half a token).
_GH_HOSTS_READ_LIMIT = 64 * 1024

#: The refusal codes the ladder's arms add to the broker's closed set
#: (registered in ``types.BROKER_ERROR_TTL_MS``; borrower sentences in
#: ``messages.py``). Both are OWNER-side states no retry from the borrower can
#: change — the operator repairs the owner — so they cache long like the
#: App's own codes.
CODE_TOKEN_UNUSABLE = "github_token_unusable"
CODE_GH_UNUSABLE = "github_gh_unusable"
#: The store itself could not be read, so no arm could be resolved. Its own
#: code rather than a token/App one: the refusal is about the door, not an arm.
CODE_STORE_UNREADABLE = "github_store_unreadable"

#: The refusal codes this adapter adds to the broker's closed set (registered in
#: ``types.BROKER_ERROR_TTL_MS``; sentences in ``messages.py``).
CODE_DEVICE_SCOPE = "device_scope_required"
CODE_APP_UNUSABLE = "github_app_unusable"
CODE_REPOSITORIES_UNSET = "github_repositories_unset"
CODE_REPO_REFUSED = "github_repo_refused"

#: How long a fetch may block on one HTTP round trip. Sequential and bounded: a
#: mint sits inside one slow-op worker, and the CLI's revoke op waits for its
#: DELETE calls.
HTTP_TIMEOUT_S = 10.0

#: How often a failed revoke is retried while this process lives.
REVOKE_RETRY_S = 60.0

#: Bound for one ``gh auth token`` call: gh answers locally (keychain read or
#: keyfile read), so this is generous for a healthy install and still bounded
#: for a wedged one. The call runs in the broker's executor, never on the relay
#: loop (a hung gh must not stall the relay — the mint's own rule).
GH_CLI_TIMEOUT_S = 10.0

#: The most tokens tracked per ``(key, holder)``. Each grant is one window
#: (<= ``grant_ttl_s``), so 16 is far above any honest overlap; past it the
#: OLDEST outstanding token is revoked immediately rather than forgotten — a
#: registry that drops records would silently lose revokability.
MAX_TRACKED_PER_HOLDER = 16

#: The most tokens one synchronous ``revoke_all`` will DELETE before scheduling
#: the rest. Bounds the control op's wait: each call is bounded by
#: ``HTTP_TIMEOUT_S``, and a holder's outstanding set is one window deep in any
#: honest workflow.
REVOKE_SYNC_MAX = 4


def is_github_key(key: str) -> bool:
    """Whether ``key`` names this adapter's credential. The one spelling rule."""
    return str(key or "") == GITHUB_KEY


# ---------------------------------------------------------------------------
# The configured allow-list
# ---------------------------------------------------------------------------


def normalise_repository(entry: Any) -> str:
    """``owner/repo`` from a config entry, or ``""`` when unusable.

    Tolerates a trailing ``.git`` (git writes ``path=owner/repo.git`` in the
    helper input, so an operator copying that spelling should not get a silent
    non-match). Case is preserved: GitHub treats owner/repo case-insensitively
    but the allow-list is compared exactly, and normalising case would make the
    receipt less copy-pasteable.
    """
    text = str(entry or "").strip()
    if text.endswith(".git"):
        text = text[: -len(".git")]
    parts = text.split("/")
    if len(parts) != 2:
        return ""
    owner, name = parts[0].strip(), parts[1].strip()
    if not owner or not name or owner in (".", "..") or name in (".", ".."):
        return ""
    return f"{owner}/{name}"


def repositories_for(root: Path | None = None) -> tuple[str, ...]:
    """The designated repositories, normalised, from this device's config.

    Read through ``network.store.read_config`` (the package's ONE reader), and
    re-read per call: the config can change while a relay runs, and the mint is
    the moment the list has to be true.
    """
    from local_operator.network import store

    raw = store.read_config(REPOSITORIES_PATH, DEFAULT_REPOSITORIES, root)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return DEFAULT_REPOSITORIES
    normalised = [normalise_repository(entry) for entry in raw]
    return tuple(entry for entry in normalised if entry)


def grant_narrowing(repositories: Sequence[str]) -> dict[str, Any]:
    """The narrowing metadata a github grant carries (design §3.4: the enforced
    bound is the helper's allow-list).

    NON-EMPTY LISTS ONLY: a cleared or never-set designation is delivered as
    "nothing", and the borrower leaves its own standing list alone — absence on
    the wire cannot be told apart from "this owner build predates the field",
    and wiping a working backstop from an ambiguous signal is the wrong
    direction for a fail-closed check. The borrower's half is
    :func:`remember_delivered_repositories`.
    """
    entries = [str(entry) for entry in repositories if str(entry)]
    return {"repositories": entries} if entries else {}


def mint_repository_names(repositories: Sequence[str]) -> list[str]:
    """The wire form of the allow-list: BARE repository names.

    ``POST /app/installations/{id}/access_tokens`` documents ``repositories`` as
    "list of repository names", and the official ``actions/create-github-app-token``
    action splits its ``owner/repo`` inputs down to the bare name before
    sending (``lib/main.js``: ``repositories.map(parseRepositoryInput)`` then
    ``repositoryNames``). The owner portion is fixed by the installation, so
    sending it is at best ignored and at worst a 422; the config keeps
    ``owner/repo`` because the helper's path check compares against git's own
    ``owner/repo`` spelling.
    """
    names: list[str] = []
    for entry in repositories:
        parts = str(entry).split("/", 1)
        if len(parts) == 2 and parts[1] and parts[1] not in names:
            names.append(parts[1])
    return names


# ---------------------------------------------------------------------------
# The App key (owner side)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GithubAppKey:
    """The three fields the checklist stores, parsed."""

    app_id: str
    installation_id: str
    private_key: str


class GithubAppKeyError(Exception):
    """``GITHUB_APP`` is absent or unusable. ``kind``: ``absent`` | ``unusable``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class GithubTokenError(Exception):
    """The ``GITHUB_TOKEN`` secret is absent or unusable.

    Same two kinds as :class:`GithubAppKeyError` on purpose: the ladder treats
    ``absent`` as "this arm is not configured" (fall through) and ``unusable``
    as a NAMED refusal (present, broken — fall-through would lend a wider thing
    than a configured arm the operator believes in).
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class GithubGhError(Exception):
    """The ``gh`` CLI login arm: absent (no stored login) or unusable (broken)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _secret_state(root: Path | None, name: str) -> bool | None:
    """Whether ``name`` is configured: ``True`` / ``False`` / ``None`` = UNKNOWN.

    The tri-state is what the LADDER needs and a single-arm probe did not:
    ``False`` must mean "definitely not configured" so resolution may fall
    through to the next arm, while an unreadable store is ``None`` — we cannot
    know whether a NARROWER arm is hiding there, and continuing would serve a
    wider credential than the one the operator may have set up (review round 1,
    M1 / QA Q2: a corrupted ``master.key`` used to read as "absent" and the
    ladder silently served the full ``gh`` login instead).

    ``describe`` only — metadata, never value material. A store that was never
    created is a definite ``False``: opening one with ``create=True`` would
    make a read a writer.
    """
    try:
        from local_operator.secrets import access
        from local_operator.secrets.errors import SecretNotFound
        from local_operator.secrets.keys import store_path

        if not store_path(root).exists():
            return False
        try:
            store = access.open_store(root)
        except Exception:  # noqa: BLE001 — present but unopenable is UNKNOWN
            return None
        try:
            store.describe(name)
            return True
        except SecretNotFound:
            return False
        except Exception:  # noqa: BLE001 — anything else is UNKNOWN, never False
            return None
    except Exception:  # noqa: BLE001 — even the store_path probe failing is UNKNOWN
        return None


def app_secret_present(root: Path | None = None) -> bool:
    """Whether a ``GITHUB_APP`` secret exists, WITHOUT reading its value.

    The closed-direction bool view of :func:`_secret_state` (``None`` folds to
    ``False``): for callers that only need "is a source configured here for a
    yes/no promise", an unknown store is not a promise. Callers that RESOLVE
    the ladder must use :func:`resolve_source`, which distinguishes the two.
    """
    return _secret_state(root, APP_SECRET_NAME) is True


def read_app_key(root: Path | None = None) -> GithubAppKey:
    """The App key from the owner's secret store. Raises :class:`GithubAppKeyError`.

    The VALUE is read through ``access.retrieve_secret`` — the one announced
    path (design §6) — rather than a bare ``open_store().get``, so the read is
    attributed exactly like every other value retrieval in this tree. The key
    itself never reaches a model-visible channel; only the tokens minted FROM it
    do (and those are exact-value registered on the borrower).
    """
    from local_operator.secrets import access
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.keys import store_path

    if not store_path(root).exists():
        raise GithubAppKeyError(
            "absent",
            f"no secret store exists here, so no {APP_SECRET_NAME} is configured",
        )
    try:
        raw = access.retrieve_secret(APP_SECRET_NAME, root)
    except SecretNotFound as exc:
        raise GithubAppKeyError("absent", str(exc)) from exc
    except SecretStoreError as exc:
        # Present but unreadable is NOT "absent": the ladder must refuse here
        # rather than fall through to a wider arm (review round 1, M1).
        raise GithubAppKeyError(
            "unusable", f"{APP_SECRET_NAME} exists but could not be read ({exc.__class__.__name__})"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — a locked/hardened store is a store that cannot serve
        raise GithubAppKeyError(
            "unusable", f"{APP_SECRET_NAME} could not be read ({exc.__class__.__name__})"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — malformed is its own answer
        raise GithubAppKeyError("unusable", f"{APP_SECRET_NAME} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise GithubAppKeyError("unusable", f"{APP_SECRET_NAME} is not a JSON object")
    app_id = str(payload.get("app_id") or "").strip()
    installation_id = str(payload.get("installation_id") or "").strip()
    private_key = str(payload.get("private_key") or "")
    missing = [
        name
        for name, value in (
            ("app_id", app_id),
            ("installation_id", installation_id),
            ("private_key", private_key),
        )
        if not value
    ]
    if missing:
        raise GithubAppKeyError("unusable", f"{APP_SECRET_NAME} is missing {', '.join(missing)}")
    return GithubAppKey(app_id=app_id, installation_id=installation_id, private_key=private_key)


# ---------------------------------------------------------------------------
# The source ladder: which arm serves here (owner side)
# ---------------------------------------------------------------------------


def token_secret_present(root: Path | None = None) -> bool:
    """Whether a ``GITHUB_TOKEN`` secret is configured, WITHOUT reading it.

    The closed-direction bool view of :func:`_secret_state`, like
    :func:`app_secret_present`; :func:`resolve_source` is the caller that must
    distinguish "absent" from "unreadable".
    """
    return _secret_state(root, TOKEN_SECRET_NAME) is True


def read_token_secret(root: Path | None = None) -> str:
    """The ``GITHUB_TOKEN``-class secret's VALUE. Raises :class:`GithubTokenError`.

    Read through ``access.retrieve_secret`` — the one announced path (design
    §6) — exactly like the App key: the read is attributed like every other
    value retrieval in this tree, and the value never reaches a model-visible
    channel (it is exact-value registered on the borrower when served).
    """
    from local_operator.secrets import access
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.keys import store_path

    if not store_path(root).exists():
        raise GithubTokenError(
            "absent", f"no secret store exists here, so no {TOKEN_SECRET_NAME} is configured"
        )
    try:
        raw = access.retrieve_secret(TOKEN_SECRET_NAME, root)
    except SecretNotFound as exc:
        raise GithubTokenError("absent", str(exc)) from exc
    except SecretStoreError as exc:
        # A store that exists but cannot produce the value is NOT "absent":
        # aborting the ladder here is the whole point of the tri-state (M1).
        raise GithubTokenError(
            "unusable",
            f"{TOKEN_SECRET_NAME} exists but could not be read ({exc.__class__.__name__})",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — a locked/hardened store cannot serve
        raise GithubTokenError(
            "unusable", f"{TOKEN_SECRET_NAME} could not be read ({exc.__class__.__name__})"
        ) from exc
    token = raw.decode("utf-8", "replace").strip()
    # A token is ONE short opaque line. A value that is empty, multi-line, or
    # absurdly long is not a token the forge will ever accept — refusing HERE,
    # by name, beats surfacing the shape error as a 401 at the first push.
    if not token or "\n" in token or len(token) > 512:
        raise GithubTokenError(
            "unusable",
            f"{TOKEN_SECRET_NAME} does not look like a token (one short line expected)",
        )
    return token


def gh_hosts_path(home: Path | None = None) -> Path:
    """``~/.config/gh/hosts.yml`` — the DEFAULT path only, deliberately.

    ``readiness._gh_auth_fact`` states the same position for its structural
    scan: a ``GH_CONFIG_DIR``/``XDG_CONFIG_HOME`` relocation is not chased in
    v1, so the report and the lender cannot come to different conclusions about
    which login exists.
    """
    root = Path.home() if home is None else home
    return root / ".config" / "gh" / "hosts.yml"


def _read_gh_hosts(home: Path | None = None) -> dict[str, Any]:
    """gh's hosts file parsed, or a :class:`GithubGhError` naming the state.

    Bounded read (``_GH_HOSTS_READ_LIMIT``): an oversized file is refused, not
    half-parsed. ``FileNotFoundError`` is ``absent``; every other failure is
    ``unusable`` — never a guess, and never a partial answer.
    """
    path = gh_hosts_path(home)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise GithubGhError("absent", f"no gh CLI login is stored here ({path})") from exc
    except OSError as exc:
        raise GithubGhError(
            "unusable", f"the gh CLI's hosts file could not be read ({exc.__class__.__name__})"
        ) from exc
    if len(raw) > _GH_HOSTS_READ_LIMIT:
        raise GithubGhError("unusable", "the gh CLI's hosts file is unexpectedly large")
    try:
        import yaml

        parsed = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — malformed is its own answer
        raise GithubGhError(
            "unusable", f"the gh CLI's hosts file is not valid YAML ({exc.__class__.__name__})"
        ) from exc
    if not isinstance(parsed, dict):
        raise GithubGhError("unusable", "the gh CLI's hosts file is not a YAML mapping")
    return parsed


def gh_login_present(home: Path | None = None) -> bool:
    """Whether gh stores a ``github.com`` login here, without reading the token.

    An unreadable-but-present hosts file resolves to the arm (``True``) so the
    serve-time refusal names it; only a definite absence — no file, or no
    ``github.com`` entry — is ``False``. The direction matters: a file left
    behind by ``gh auth logout`` (entry gone) must fall through to
    ``no_local_credential``, not refuse as though a login existed. The parse
    reads the file's mapping (the file IS gh's storage); the token string is
    only extracted where a serve needs it, never here.
    """
    try:
        hosts = _read_gh_hosts(home)
    except GithubGhError as exc:
        return exc.kind == "unusable"
    return isinstance(hosts.get("github.com"), dict)


#: Absolute fallback directories for gh discovery, after this process's PATH and
#: the user-local bin: the two standard install prefixes (Homebrew on arm64 and
#: on x86_64, plus locally-built installs). The relay's launchd job hands it
#: PATH=/usr/bin:/bin:/usr/sbin:/sbin (measured on the operator's machine), where
#: a Homebrew gh is invisible to a PATH probe — this fallback is what lets the
#: arm serve there (review round 2, B2), and ``relay.render_plist`` teaches the
#: job the same directories so the two cannot disagree.
GH_FALLBACK_BIN_DIRS: tuple[str, ...] = ("/opt/homebrew/bin", "/usr/local/bin")


def _resolve_program(name: str, *, path: str | None = None) -> str | None:
    """``shutil.which`` behind ONE name, so the probe has one seam.

    The same seam shape as ``readiness._resolve_program`` (that module's tooling
    row): tests pin discovery without touching the process PATH, and production
    keeps exactly one spelling of "where is gh".
    """
    import shutil

    return shutil.which(name, path=path)


def find_gh(home: Path | None = None) -> str | None:
    """The gh executable as this device would run it — off-PATH included.

    Probes, in order: this PROCESS's PATH; the user-local bin directory; then
    the standard install prefixes (:data:`GH_FALLBACK_BIN_DIRS`). The order is
    the one ``relay._launchd_path`` teaches the relay's own job, so the job's
    environment and this fallback agree about which gh wins. The relay runs
    under launchd with a minimal PATH on macOS and gh is commonly installed
    where that PATH does not reach; the borrower never runs this (owner-side
    only), so the probe answers for the same ambient user the broker serves as.
    """
    try:
        found = _resolve_program("gh")
    except Exception:  # noqa: BLE001 — a broken probe is "not found here"
        found = None
    if found:
        return found
    root = Path.home() if home is None else home
    fallbacks = (root / ".local" / "bin", *(Path(entry) for entry in GH_FALLBACK_BIN_DIRS))
    for directory in fallbacks:
        try:
            found = _resolve_program("gh", path=str(directory))
        except Exception:  # noqa: BLE001 — see above
            continue
        if found:
            return found
    return None


def read_gh_token(home: Path | None = None) -> str:
    """The token the owner's own gh CLI holds for ``github.com`` (owner side).

    ``by asking gh itself`` — the design's literal wording (§3.2), and what a
    default macOS install needs: gh keeps the token in the OS keychain there,
    and ``hosts.yml`` carries the login WITHOUT an ``oauth_token`` (verified
    live: gh 2.100.0, ``gh auth status`` shows ``(keyring)``). So resolution is
    two-step, cheapest first:

    1. gh's own hosts file, when it holds a token (``--insecure-storage`` /
       keyring-less installs) — a bounded local read, no process at all;
    2. ``gh auth token --hostname github.com`` — gh's own resolution, the
       default for keyring-stored logins. The binary is resolved off-PATH too
       (:func:`find_gh`, the launchd lesson); stdout is captured and never
       logged, and only a single-line answer is accepted.

    Resolved fresh at EVERY serve — no cache, so a re-login or rotation
    reaches the node's next command (§3.3). The value never reaches a
    model-visible channel (exact-value registered when served).
    """
    hosts = _read_gh_hosts(home)
    entry = hosts.get("github.com")
    if isinstance(entry, dict):
        token = str(entry.get("oauth_token") or "").strip()
        if token:
            return token
    exe = find_gh(home)
    if exe is None:
        if isinstance(entry, dict):
            raise GithubGhError(
                "unusable",
                "the gh CLI's github.com login stores no token in its file and no gh "
                "executable was found to ask (probed PATH, ~/.local/bin, "
                "/opt/homebrew/bin and /usr/local/bin): install gh if it is genuinely "
                "absent, and check that directory list if it is not — nothing was lent "
                "(the network guide has the ladder)",
            )
        raise GithubGhError("absent", "the gh CLI stores no github.com login here")
    try:
        proc = subprocess.run(
            [exe, "auth", "token", "--hostname", "github.com"],
            capture_output=True,
            text=True,
            timeout=GH_CLI_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — a timeout or spawn failure is the arm being unusable
        raise GithubGhError(
            "unusable", f"gh could not be asked for a token ({exc.__class__.__name__})"
        ) from exc
    token = proc.stdout.strip()
    if proc.returncode != 0 or not token or "\n" in token:
        # STRUCTURAL only: gh's stderr can carry account and host detail and is
        # never reproduced (the readiness probe's names-only discipline).
        raise GithubGhError(
            "unusable",
            f"gh could not produce a github.com token (exit {proc.returncode}): sign in "
            "again with the gh CLI on this device (the network guide has the ladder)",
        )
    return token


def no_source_arms() -> str:
    """The one enumeration of the ladder's arms, shared by every refusal voice.

    Kept CLAUSE-level so the device-side refusal (:func:`no_source_message`),
    the share verb's refusal and the borrower's rendered sentence
    (``messages.py``) cannot drift in WHICH arms they teach — the fact a person
    acts on. Order is remedy-first (cheapest to set up first); the network
    guide's ladder lists the same three strongest-first, and this clause points
    there.
    """
    return (
        "the gh CLI's own login, a GITHUB_TOKEN-class token, or the stronger "
        "GitHub App — the network guide has the ladder"
    )


def no_source_message() -> str:
    """The device-side sentence for "no arm of the ladder is configured here".

    This sentence (not a second literal) is what the broker refusal and the
    share verb emit; its arms clause is :func:`no_source_arms`, shared with the
    borrower's sentence in ``messages.py`` so the enumeration cannot drift.
    """
    return (
        f"no GitHub credential is configured here yet: {no_source_arms()}. Push and "
        "PR-write stay unavailable until one of them exists; public clones and "
        "non-GitHub work are unaffected"
    )


def resolve_source(root: Path | None = None, *, home: Path | None = None) -> str:
    """Which ladder arm serves github on this device right now.

    Answers ``SOURCE_APP`` / ``SOURCE_TOKEN`` / ``SOURCE_GH`` / ``""`` (no arm),
    or ``SOURCE_UNREADABLE`` when the encrypted store EXISTS but cannot be read:
    the ladder STOPS there rather than guessing, because an unreadable store
    may hold a narrower arm (App or token) and continuing would serve a wider
    credential than the one the operator set up (review round 1, M1 / QA Q2).
    Callers that only need "may this device serve at all" use
    :func:`source_present`, which folds the unknown to not-a-promise.

    PRESENCE, not usability, for each arm it does resolve: a configured but
    broken arm RESOLVES so the serve-time refusal can name it. THE ONE ORDER:
    the lender serves with this, and the receipts name the arm with it.
    """
    app_state = _secret_state(root, APP_SECRET_NAME)
    if app_state is None:
        return SOURCE_UNREADABLE
    if app_state:
        return SOURCE_APP
    token_state = _secret_state(root, TOKEN_SECRET_NAME)
    if token_state is None:
        return SOURCE_UNREADABLE
    if token_state:
        return SOURCE_TOKEN
    if gh_login_present(home):
        return SOURCE_GH
    return ""


def source_present(root: Path | None = None, *, home: Path | None = None) -> bool:
    """Whether any arm resolves to something servable (``""``/unknown are not).

    The closed direction for yes/no surfaces (the offer candidate gate, the
    ledger row): an unreadable store is not a promise, and neither is an empty
    ladder.
    """
    return resolve_source(root, home=home) in (SOURCE_APP, SOURCE_TOKEN, SOURCE_GH)


# ---------------------------------------------------------------------------
# The wire calls (mint + revoke)
# ---------------------------------------------------------------------------


class GithubApiError(Exception):
    """One failed API call. ``kind``: ``auth`` | ``coverage`` | ``offline`` | ``server``.

    The kinds are the owner's mapping table to refusal codes; the HTTP status is
    kept for the diagnostic (the borrower never sees it — messages are rendered
    from codes, and a status is not a sentence).
    """

    def __init__(self, kind: str, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclasses.dataclass(frozen=True)
class MintedToken:
    """One fresh installation token. The token is material; nothing logs it."""

    token: str
    expires_at_ms: int


@dataclasses.dataclass(frozen=True)
class _Reply:
    status: int
    body: dict[str, Any]


class _HttpxTransport:
    """The production transport: one ``httpx`` client per call.

    ``httpx`` is imported HERE (see the module docstring's stdlib-at-import
    rule). ``follow_redirects=False``: api.github.com does not redirect the two
    endpoints this module calls, and following one would re-send the
    Authorization header to wherever it pointed.
    """

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
    ) -> _Reply:
        import httpx

        with httpx.Client(timeout=HTTP_TIMEOUT_S, follow_redirects=False) as client:
            response = client.request(method, url, headers=headers, json=json_body)
        body: dict[str, Any] = {}
        try:
            decoded = response.json()
            if isinstance(decoded, dict):
                body = decoded
        except Exception:  # noqa: BLE001 — a non-JSON body is empty; the status is the answer
            body = {}
        return _Reply(status=response.status_code, body=body)


def _app_jwt(app: GithubAppKey, now: float | None = None) -> str:
    """The App JWT: ``iss`` = app id, ``iat`` 60 s back (clock skew), 10-min cap.

    GitHub's documented bounds: ``iat`` no more than 60 s in the past, ``exp`` no
    more than 10 minutes ahead. ``jwt`` (pyjwt[crypto]) is a direct dependency
    and is imported here, not at module scope.
    """
    import jwt

    moment = int(now if now is not None else time.time())
    return jwt.encode(
        {"iat": moment - 60, "exp": moment + 600, "iss": app.app_id},
        app.private_key,
        algorithm="RS256",
    )


def _expires_at_ms(expires_at: Any, now: float) -> int:
    """GitHub's ISO-8601 ``expires_at`` as epoch ms, or ``now + 1 h`` when unreadable.

    One hour is the documented lifetime of an installation token; a reply whose
    timestamp this build cannot parse still gets a correct-enough window rather
    than an epoch-zero expiry that would look already-dead.
    """
    try:
        parsed = datetime.datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000)
    except Exception:  # noqa: BLE001 — see above
        return int(now * 1000) + 3600_000


class GithubMinter:
    """The two API calls: mint one narrow token, revoke one token."""

    def __init__(self, *, base_url: str = GITHUB_API_BASE, transport: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport if transport is not None else _HttpxTransport()

    def mint(self, app: GithubAppKey, repositories: Sequence[str]) -> MintedToken:
        """Mint a token narrowed to ``repositories`` (bare names on the wire).

        THE MINT IS THE ALLOW-LIST CHECK (§D4): ``repositories`` names the
        covered set, ``permissions`` narrows the authority, and GitHub refuses
        (422) a repository the installation does not cover — "no silent
        widening, no separate convention". Callers let :class:`GithubApiError`
        carry the kind; nothing here retries (a retry loop lives in the caller
        only where the caller owns a schedule).
        """
        now = time.time()
        names = mint_repository_names(repositories)
        if not names:
            # THE BELT UNDER THE OWNER'S CHECK: minting without ``repositories``
            # would hand the token access to EVERYTHING the installation covers
            # (docs: absent means the installation's full set), so an empty
            # allow-list is refused here rather than widened silently.
            raise GithubApiError("coverage", "no repositories are designated")
        url = f"{self.base_url}/app/installations/{app.installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {_app_jwt(app, now)}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        try:
            reply = self._transport.request(
                "POST",
                url,
                headers=headers,
                json_body={"repositories": names, "permissions": dict(GITHUB_PERMISSIONS)},
            )
        except Exception as exc:  # noqa: BLE001 — any transport failure is "offline"
            raise GithubApiError("offline", f"{exc.__class__.__name__}") from exc
        if reply.status == 201:
            token = str(reply.body.get("token") or "")
            if not token:
                raise GithubApiError("server", "the mint reply carried no token", status=201)
            return MintedToken(
                token=token, expires_at_ms=_expires_at_ms(reply.body.get("expires_at"), now)
            )
        message = str(reply.body.get("message") or f"HTTP {reply.status}")
        if reply.status in (401, 403, 404):
            # The App's own authentication was refused, or the installation id
            # does not exist. The remedy is on the owner's setup, not a retry.
            raise GithubApiError("auth", message, status=reply.status)
        if reply.status == 422:
            # Under-covered repositories (or a malformed request). The remedy is
            # the installation/allow-list, and the sentence names the repos.
            raise GithubApiError("coverage", message, status=reply.status)
        raise GithubApiError("server", message, status=reply.status)

    def revoke(self, token: str) -> bool:
        """``DELETE /installation/token``. Idempotent: "already dead" is success.

        204 is the documented success. 401/403/404 are ALSO success: the only
        way this call can be refused authentication is that the token itself no
        longer authenticates — which is the goal state, because the call
        authenticates WITH the token it revokes. Anything else is a failure the
        caller records and retries while it lives.
        """
        url = f"{self.base_url}/installation/token"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        try:
            reply = self._transport.request("DELETE", url, headers=headers)
        except Exception:  # noqa: BLE001 — a failed call is retried by the lender
            return False
        return reply.status in (204, 401, 403, 404)


# ---------------------------------------------------------------------------
# The owner's mint-revoke registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TrackedToken:
    """One minted-but-unrevoked token, kept ONLY to be revocable."""

    key: str
    holder: str
    grant_id: str
    token: str
    grant_exp_ms: int
    token_exp_ms: int
    attempts: int = 0
    last_error: str = ""


class GithubLender:
    """Mints on demand and revokes at each token's window end (design F4).

    ONE instance per broker process, owned by :class:`MeshCredentialBroker`. The
    registry is IN MEMORY and intentionally so: a token's life is one window
    (<= ``grant_ttl_s``) and the hard fallback is the token's own one-hour
    ceiling, so persistence would buy a durability claim this design does not
    make. What it buys instead is the scheduled DELETE on the broker's own loop
    (:meth:`schedule`) and the immediate one on ``credential revoke``
    (:meth:`revoke_holder`).
    """

    def __init__(
        self,
        *,
        minter: GithubMinter | None = None,
        clock: Callable[[], float] = time.time,
        retry_s: float = REVOKE_RETRY_S,
    ) -> None:
        self._minter = minter if minter is not None else GithubMinter()
        self._clock = clock
        self._retry_s = retry_s
        self._lock = threading.Lock()
        #: ``(key, holder) -> {grant_id: _TrackedToken}``, insertion-ordered.
        self._tracked: dict[tuple[str, str], dict[str, _TrackedToken]] = {}
        self._loop: Any = None
        self._tasks: set[Any] = set()
        #: Set by the broker so scheduled fires can report without re-importing.
        self._audit_sink: Any = None

    # -- construction -------------------------------------------------------

    def attach_loop(self, loop: Any) -> None:
        """Bind the broker's one long-lived loop; call once, before any mint."""
        self._loop = loop

    def attach_audit(self, sink: "Callable[[str, str, str, int, int], None]") -> None:
        """``sink(cause, key, holder, revoked, deferred)`` — the broker's audit hook.

        A callable rather than the broker itself on purpose: this module must
        not import ``owner.py`` (the relay's construction path imports BOTH,
        and a cycle would be resolved by an import-time failure elsewhere).
        """
        self._audit_sink = sink

    # -- mint ---------------------------------------------------------------

    def mint(self, app: GithubAppKey, repositories: Sequence[str]) -> MintedToken:
        """Pass-through to the minter; kept on the lender so tests inject ONE seam."""
        return self._minter.mint(app, repositories)

    # -- registry -----------------------------------------------------------

    def register(
        self,
        *,
        key: str,
        holder: str,
        grant_id: str,
        token: str,
        grant_exp_ms: int,
        token_exp_ms: int,
    ) -> None:
        """Track a fresh token and schedule its window-end revoke."""
        record = _TrackedToken(
            key=key,
            holder=holder,
            grant_id=grant_id,
            token=token,
            grant_exp_ms=grant_exp_ms,
            token_exp_ms=token_exp_ms,
        )
        overflow: list[_TrackedToken] = []
        with self._lock:
            slot = self._tracked.setdefault((key, holder), {})
            slot[grant_id] = record
            while len(slot) > MAX_TRACKED_PER_HOLDER:
                _oldest_id, oldest = next(iter(slot.items()))
                del slot[_oldest_id]
                overflow.append(oldest)
        # Overflow is revoked EARLY, never forgotten: the registry's bound must
        # not be a way to lose revokability (see MAX_TRACKED_PER_HOLDER).
        for spilled in overflow:
            self._schedule(spilled, delay_s=0.0)
        self._schedule(record)

    def outstanding(self, *, key: str = GITHUB_KEY, holder: str = "") -> int:
        with self._lock:
            if holder:
                return len(self._tracked.get((key, holder), {}))
            return sum(
                len(slot) for (slot_key, _holder), slot in self._tracked.items() if slot_key == key
            )

    def revoke_holder(self, holder: str, *, key: str = GITHUB_KEY, cause: str = "revoked") -> int:
        """Revoke every outstanding token for ``(key, holder)`` NOW. Returns how many died.

        Blocking with a bounded shape (see REVOKE_SYNC_MAX): the caller is a
        control-op worker answering ``credential revoke``, and the operator is
        owed a count. Tokens past the sync budget are revoked via the loop (or
        synchronously when no loop is attached, which is the CLI-only case).
        """
        with self._lock:
            slot = self._tracked.get((key, holder), {})
            records = list(slot.values())
        revoked = 0
        deferred = 0
        for index, record in enumerate(records):
            if index < REVOKE_SYNC_MAX or self._loop is None:
                ok = self._revoke_now(record)
                if not ok and self._loop is not None:
                    # The retry contract covers the OPERATOR's immediate revoke
                    # too: a DELETE that failed must come back while the relay
                    # lives, not merely be counted as missing.
                    self._schedule(record, delay_s=self._retry_s)
            else:
                ok = False
                deferred += 1
                self._schedule(record, delay_s=0.0)
            if ok:
                revoked += 1
        self._record(cause=cause, key=key, holder=holder, revoked=revoked, deferred=deferred)
        return revoked

    def revoke_all_holders(self, *, key: str = GITHUB_KEY, cause: str = "revoked") -> int:
        """Every outstanding token for ``key``, whatever the holder. Returns the count."""
        with self._lock:
            holders = sorted({holder for (slot_key, holder) in self._tracked if slot_key == key})
        return sum(self.revoke_holder(holder, key=key, cause=cause) for holder in holders)

    # -- internals ----------------------------------------------------------

    def _revoke_now(self, record: _TrackedToken) -> bool:
        """One immediate DELETE. True when nothing is left to do for the token.

        True covers both success and "past the token's own ceiling": a token
        past its 1-hour expiry is dead by GitHub's clock, and a DELETE cannot
        authenticate with a dead token, so retrying it would be a loop with no
        reachable success. False means the caller should retry later.
        """
        ok = self._minter.revoke(record.token)
        with self._lock:
            slot = self._tracked.get((record.key, record.holder), {})
            if ok or self._clock() * 1000 >= record.token_exp_ms:
                slot.pop(record.grant_id, None)
                if not slot:
                    self._tracked.pop((record.key, record.holder), None)
                return True
            record.attempts += 1
            record.last_error = "revoke call failed"
            return False

    def _schedule(self, record: _TrackedToken, *, delay_s: float | None = None) -> None:
        """Queue one revoke on the broker's loop (thread-safe), or run it inline."""
        if delay_s is None:
            delay_s = max(0.05, record.grant_exp_ms / 1000.0 - self._clock())
        loop = self._loop
        if loop is None:
            # No loop (a unit-test lender, or a process that never bound one):
            # the token's own ceiling remains, stated rather than hidden.
            return
        loop.call_soon_threadsafe(self._fire_on_loop, record, max(0.05, delay_s))

    def _fire_on_loop(self, record: _TrackedToken, delay_s: float) -> None:
        """Runs on the loop thread: schedule the DELETE off the loop."""
        loop = self._loop
        if loop is None:  # pragma: no cover — close() raced the fire
            return
        loop.call_later(delay_s, self._start_reap, record)

    def _start_reap(self, record: _TrackedToken) -> None:
        """Create the reap task and RETAIN it: ``create_task`` holds a weak
        reference, so a task nobody keeps can be collected mid-run."""
        import asyncio

        task = asyncio.get_running_loop().create_task(self._reap(record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _reap(self, record: _TrackedToken) -> None:
        """Off-loop DELETE; on failure retry while this process lives."""
        import asyncio

        with self._lock:
            slot = self._tracked.get((record.key, record.holder), {})
            if record.grant_id not in slot:
                return  # already revoked through another path; nothing to report
        loop = asyncio.get_running_loop()
        retry = record.attempts > 0
        ok = await loop.run_in_executor(None, self._revoke_now, record)
        if not ok:
            self._schedule(record, delay_s=self._retry_s)
        # EVERY fire reports, so the window-end DELETE and its retries are visible
        # to an incident reader — not only the operator-issued ones. The cause
        # tells them apart (see ``_github_audit_sink``).
        self._record(
            cause="revoke_retry" if retry else "grant_expired",
            key=record.key,
            holder=record.holder,
            revoked=1 if ok else 0,
            deferred=0,
        )

    def _record(self, *, cause: str, key: str, holder: str, revoked: int, deferred: int) -> None:
        sink = self._audit_sink
        if sink is None:
            return
        try:
            sink(cause, key, holder, revoked, deferred)
        except Exception:  # noqa: BLE001 — a log that cannot be written is not a refusal
            pass

    def close(self) -> None:
        """Cancel any pending fires; the loop itself is the broker's to stop."""
        loop = self._loop
        if loop is None:
            return
        for task in list(self._tasks):
            try:
                loop.call_soon_threadsafe(task.cancel)
            except Exception:  # noqa: BLE001 — a closed loop is already stopped
                pass
        self._tasks.clear()


# ---------------------------------------------------------------------------
# The borrower's delivery
# ---------------------------------------------------------------------------


def helper_command() -> str:
    """The ``GIT_CONFIG`` helper value: ``!`` + THIS BUILD's CLI, shell-quoted.

    Spelled with ``python_argv`` rather than a bare ``lop``: the helper MUST be
    the build that constructed the env (its host/protocol semantics are what the
    injected reset pair is paired with), and a ``lop`` resolved from the child's
    PATH may be a different generation entirely — the same reasoning
    ``builtin._sessions_launch`` states for spawning this build's CLI. The
    subcommand still ships as ``lop credential git-helper`` for a person to run
    and for tests to invoke.
    """
    from local_operator.interpreter import python_argv

    return "!" + shlex.join(python_argv("-m", "local_operator.cli", "credential", "git-helper"))


def git_env_for_token(token: str) -> dict[str, str]:
    """The exact child env for one grant (design §D1, spike-verified form).

    ``GIT_CONFIG_COUNT=3`` with the EMPTY first entry: the github.com-scoped
    helper list is reset before the brokered helper is added, so a node's
    global ``credential.helper=store`` (or any other persisting helper, global
    or repo-local) never receives the value — while other hosts keep their own
    helpers untouched (an unscoped reset would strip those too). The order is
    load-bearing: git consults same-scope entries in order and the empty value
    RESETS the accumulated list, so the reset must precede the addition.
    ``useHttpPath=true`` is what puts ``path=owner/repo.git`` in the helper's
    input, which is what the helper's allow-list check reads.
    """
    return {
        "GH_TOKEN": token,
        "GITHUB_TOKEN": token,
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_KEY_0": "credential.https://github.com.helper",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": "credential.https://github.com.helper",
        "GIT_CONFIG_VALUE_1": helper_command(),
        "GIT_CONFIG_KEY_2": "credential.https://github.com.useHttpPath",
        "GIT_CONFIG_VALUE_2": "true",
    }


#: One client per process per root, built lazily. NOT keyed per session: the
#: GrantCache inside is already keyed ``(key, session_id)``, and the refusal
#: document is per network. A ``None`` answer is deliberately NOT cached — a
#: share that arrives while this process lives must be seen by the next command
#: (the same staleness class ``MeshCredentialBroker``'s docstring describes on
#: the other side).
_GIT_CLIENTS: dict[str, Any] = {}
_GIT_CLIENTS_LOCK = threading.Lock()

#: Tokens this process has already scheduled a self-revoke for, by token string.
_SELF_REVOKE_SCHEDULED: set[str] = set()
_SELF_REVOKE_LOCK = threading.Lock()


def _client_for(root: Path | None) -> Any:
    from local_operator.network.credentials.client import MeshCredentialClient

    cache_key = str(root or "")
    with _GIT_CLIENTS_LOCK:
        client = _GIT_CLIENTS.get(cache_key)
        if client is not None:
            return client
    client = MeshCredentialClient.for_this_device(root)
    if client is not None:
        with _GIT_CLIENTS_LOCK:
            _GIT_CLIENTS[cache_key] = client
    return client


def _schedule_self_revoke(token: str, grant_exp_ms: int) -> None:
    """The borrower's best-effort belt: DELETE the token at its window end.

    Fires only while THIS process lives; the owner-side revoker is the primary,
    and the token's own ceiling is the floor under both. Duplicate scheduling is
    suppressed per token: the env path re-reads the same cached grant for every
    command in a window, and one timer per command would be a thread storm.
    """
    if not token:
        return
    with _SELF_REVOKE_LOCK:
        if token in _SELF_REVOKE_SCHEDULED:
            return
        _SELF_REVOKE_SCHEDULED.add(token)
    delay = max(0.1, grant_exp_ms / 1000.0 - time.time())

    def _fire() -> None:
        try:
            revoke_installation_token(token)
        finally:
            with _SELF_REVOKE_LOCK:
                _SELF_REVOKE_SCHEDULED.discard(token)

    timer = threading.Timer(delay, _fire)
    timer.daemon = True
    timer.start()


def revoke_installation_token(token: str, *, base_url: str | None = None) -> bool:
    """One borrower-side DELETE. Best effort; never raises.

    ``base_url`` is read at CALL time (``None`` means the module's real host), so
    a test's loopback GitHub stands in without touching the constant — the same
    reason the minter is injectable on the owner side.
    """
    try:
        return GithubMinter(base_url=base_url or GITHUB_API_BASE).revoke(token)
    except Exception:  # noqa: BLE001 — a belt, not a mechanism of record
        return False


def remember_delivered_repositories(
    repositories: Any,
    *,
    root: Path | None = None,
    self_device: str = "",
    owner_device: str = "",
    network_id: str = "",
) -> dict[str, Any]:
    """Materialise a delivered narrowing into THIS device's config (borrower side).

    THE BOUND LANDS WITH THE BEARER (design §3.4): the owner attaches its current
    designation to every github grant it serves, and this writes it where the git
    credential helper — a separate process git starts later, reading
    :func:`repositories_for` at use time — will find it. Idempotent (an equal
    list writes nothing), receipted (a change appends
    ``credential.narrowing_applied`` to the audit trail), and NEVER fatal: every
    failure is returned in the result and recorded (``credential.narrowing_refused``
    where the trail is writable), because the borrow that carried the value must
    not fail over its own receipt.

    ``repositories`` is whatever the wire carried. Only a list/tuple with at
    least one USABLE ``owner/repo`` entry writes: a shape that normalises to
    nothing is a wire anomaly — recorded and ignored, never a silent clearing of
    a list an operator set here by hand.
    """
    result: dict[str, Any] = {"changed": False, "repositories": [], "error": ""}
    normalised: list[str] = []
    if isinstance(repositories, (list, tuple)):
        for entry in repositories:
            cleaned = normalise_repository(entry)
            if cleaned and cleaned not in normalised:
                normalised.append(cleaned)
    result["repositories"] = normalised
    if not normalised:
        reason = (
            "unreadable" if not isinstance(repositories, (list, tuple)) else "no_usable_entries"
        )
        result["error"] = reason
        _record_narrowing(root, self_device, owner_device, network_id, applied=False, reason=reason)
        return result
    try:
        current = list(repositories_for(root))
    except Exception as exc:  # noqa: BLE001 — an unreadable config reads as "unknown"
        result["error"] = exc.__class__.__name__
        _record_narrowing(
            root,
            self_device,
            owner_device,
            network_id,
            applied=False,
            reason=exc.__class__.__name__,
        )
        return result
    if normalised == current:
        return result
    try:
        _write_repositories(root, normalised)
    except Exception as exc:  # noqa: BLE001 — recorded, never fatal; see the docstring
        result["error"] = exc.__class__.__name__
        logger.warning("github narrowing: the delivered repositories were not written: %s", exc)
        _record_narrowing(
            root,
            self_device,
            owner_device,
            network_id,
            applied=False,
            reason=exc.__class__.__name__,
        )
        return result
    result["changed"] = True
    _record_narrowing(
        root, self_device, owner_device, network_id, applied=True, repositories=normalised
    )
    return result


def _write_repositories(root: Path | None, repositories: list[str]) -> None:
    """Write the allow-list through the settings registry — the ONE writer.

    The same row the ``/settings`` UI edits, so a delivered value and a
    hand-typed one cannot disagree about the path, the validation, or the merge
    rule; and the write funnels through ``write_setting``'s reload-before-write,
    so a config another process edited a moment ago is merged rather than
    reverted by a stale snapshot.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir

    directory = Path(root) if root is not None else config_dir()
    setting = next(
        candidate
        for candidate in settings_io.settings_for("network")
        # Matched on the PATH, not the dotted key: ``REPOSITORIES_PATH`` is the
        # one spelling both sides of the share already read, and a matching
        # criterion cannot drift from it.
        if candidate.path == REPOSITORIES_PATH
    )
    settings_io.write_setting(ConfigManager(directory), setting, list(repositories))


def _record_narrowing(
    root: Path | None,
    self_device: str,
    owner_device: str,
    network_id: str,
    *,
    applied: bool,
    repositories: Sequence[str] = (),
    reason: str = "",
) -> None:
    """One audit row for a narrowing that landed — or one that was refused.

    The borrow path's own discipline (``cli._audit_placement``'s shape): a
    one-shot writer, closed after its row, and a trail that cannot be written
    must never fail the delivery. ``act``/``sub`` are the devices the delegation
    ran between, so an incident reader can line the receipt up with the grant.
    """
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        detail: dict[str, Any] = {
            "credential_key": GITHUB_KEY,
            "act": self_device,
            "sub": owner_device,
        }
        if applied:
            detail["repositories"] = list(repositories)
            event = "credential.narrowing_applied"
        else:
            detail["reason"] = reason
            event = "credential.narrowing_refused"
        log = AuditLog(root)
        log.record(
            AuditEvent(
                event=event,
                network_id=network_id,
                actor=self_device,
                subject=owner_device,
                actor_kind="device",
                detail=detail,
            )
        )
        log.close()
    except Exception:  # noqa: BLE001 — a receipt is never worth a lost env
        pass


def borrowed_git_env(
    *,
    root: Path | None = None,
    session_id: str = "",
    client: Any = None,
) -> tuple[dict[str, str], str]:
    """Fetch-on-use: ``(env, token)`` for one child, or ``({}, "")``. Never raises.

    ``{}`` is every not-now answer on purpose — this runs on the bash tool's
    spawn path, and a command must never fail because a borrow could not: the
    child then behaves exactly as it did before brokering existed. A refusal is
    left in the client's own cache for the surfaces that render sentences.

    ``session_id`` travels as ATTRIBUTION (the audit row's ``for_session``, the
    grant cache's partition) and nothing else; see the module docstring.
    """
    try:
        if client is None:
            client = _client_for(root)
        if client is None:
            return {}, ""
        if not client.should_borrow(GITHUB_KEY):
            return {}, ""
        outcome = client.request_grant_sync(GITHUB_KEY, session_id=session_id, provider=GITHUB_KEY)
    except Exception:  # noqa: BLE001 — see the docstring
        return {}, ""
    token = str(getattr(outcome, "access_token", "") or "")
    if not token:
        return {}, ""
    # THE BOUND LANDS WITH THE BEARER (§3.4): a grant that carries the owner's
    # narrowing writes it to this device's config HERE, before the child that
    # will run the helper is spawned — the helper is a separate process that
    # reads ``repositories_for`` at use time, and this is the moment the value
    # and its use are guaranteed in order. Best effort by construction:
    # ``remember_delivered_repositories`` never raises, and a device that
    # received no narrowing keeps whatever its own config holds (no list still
    # refuses).
    try:
        narrowing = getattr(outcome, "narrowing", None)
        delivered = narrowing.get("repositories") if isinstance(narrowing, dict) else None
        if isinstance(delivered, list):
            owner_of = getattr(client, "owner_of", None)
            remember_delivered_repositories(
                delivered,
                root=root,
                self_device=str(getattr(client, "self_device", "") or ""),
                owner_device=str(owner_of(GITHUB_KEY) or "") if callable(owner_of) else "",
                network_id=str(getattr(client, "network_id", "") or ""),
            )
    except Exception:  # noqa: BLE001 — a delivery receipt is never worth a lost env
        pass
    grant_exp_ms = int(getattr(outcome, "grant_expires_at_ms", 0) or 0)
    _schedule_self_revoke(token, grant_exp_ms)
    return git_env_for_token(token), token


# ---------------------------------------------------------------------------
# The git credential helper
# ---------------------------------------------------------------------------

#: The hosts the helper serves, exactly, lowercased. ``github.com:443``
#: is the spelling git writes when the URL carried an explicit port; both mean
#: the same HTTPS endpoint, and anything else — ``github.com.evil.com``, a
#: different host, a different scheme — is refused.
_ALLOWED_HOSTS = frozenset({"github.com", "github.com:443"})


def _parse_credential_request(lines: Iterable[str]) -> dict[str, str]:
    """``key=value`` lines from git's credential protocol, last one wins.

    Malformed lines are skipped rather than fatal: the protocol is line-oriented
    and only the three fields this helper reads (`protocol`, `host`, `path`)
    matter. ``wwwauth[]``/``capability[]``/``url`` and friends are ignored.
    """
    fields: dict[str, str] = {}
    for line in lines:
        text = str(line).rstrip("\r\n")
        if not text or text == "\n":
            continue
        name, sep, value = text.partition("=")
        if not sep:
            continue
        fields[name.strip().lower()] = value
    return fields


def _path_allowed(path: str, repositories: Sequence[str]) -> bool:
    """Whether ``path`` is a repo this helper may serve.

    FAIL-CLOSED EDGES FIRST: absent, empty or malformed paths are refused (a
    bare ``git credential fill`` has no path, and serving it would hand the
    token out for any interaction against github.com), and NO LIST CONFIGURED IS
    ALSO A REFUSAL — not a wildcard. The mint refuses to produce a token under
    an empty list in the first place, so a token that reaches a helper without a
    list is outside the mechanism; the mechanism never widens for it.
    """
    normalised = normalise_repository(path)
    if not normalised:
        return False
    return normalised in set(repositories)


def git_helper_reply(
    operation: str,
    request_lines: Iterable[str],
    *,
    token: str,
    repositories: Sequence[str] = (),
) -> str:
    """The helper's pure core: git's request in, git's response out (or ``""``).

    Semantics are the spike's, verbatim: serve ONLY ``protocol=https`` to
    ``github.com``/``github.com:443`` with a present, allow-listed path; ``get``
    prints the two lines git expects and NOTHING else; ``store``/``erase`` are
    no-ops; every refusal is silence with exit 0, because a credential helper's
    way of saying "I have nothing" is to print nothing — an error message would
    surface it as git's own failure. The helper never reads or writes a file,
    and the token it serves is whatever ``$GH_TOKEN`` its environment carries
    (the session injected it; a helper started bare has none and refuses).
    """
    op = str(operation or "").strip().lower()
    if op in ("store", "erase"):
        return ""
    if op != "get":
        return ""
    fields = _parse_credential_request(request_lines)
    if fields.get("protocol", "").lower() != "https":
        return ""
    if fields.get("host", "").lower() not in _ALLOWED_HOSTS:
        return ""
    if not _path_allowed(fields.get("path", ""), repositories):
        return ""
    if not token:
        return ""
    return f"username=x-access-token\npassword={token}\n"


def run_git_helper_cli(operation: str) -> int:
    """The ``lop credential git-helper`` entry point. Always exit 0, mostly silence."""
    import sys

    try:
        lines = sys.stdin.read().splitlines()
    except Exception:  # noqa: BLE001 — no stdin is "no request"
        lines = []
    reply = git_helper_reply(
        operation,
        lines,
        token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "",
        repositories=repositories_for(None),
    )
    if reply:
        sys.stdout.write(reply)
        sys.stdout.flush()
    return 0
