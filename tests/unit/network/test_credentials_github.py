"""The ``github`` adapter: minted, narrowed, revoked — and what git does with it.

WHAT MAKES THIS EVIDENCE. Everything drivable is REAL: the broker on its own
event loop, the secret store on disk, the real ``git`` binary, OUR helper
(``lop credential git-helper``) invoked BY git through the exact env the session
injects, and a loopback GitHub whose mint verifies the App JWT signature and
whose ``DELETE /installation/token`` is the only thing that kills a token. The
one stub is the App MINT ENDPOINT — the brief allows it while the operator's App
does not exist yet — and it stands where GitHub stands: a real RSA keypair in
the secret store signs the real JWT, and a real HTTP endpoint validates it.

CELL MAP (the folded note's DoD; the note is the spec, this file is its proof):

* T1 — a borrow mints a NARROWED token (repositories + permissions on the wire)
  and a real ``git push`` through the borrowed env succeeds and leaves no trace
  (`test_a_borrow_is_minted_narrowed_and_audited`,
  `test_a_push_through_the_borrowed_env_succeeds_and_leaves_no_trace`).
* T2 — revoke races an in-flight push (held receive-pack): the push fails, the
  old token never authenticates again, the DELETE was issued for it, and the
  next borrow is refused by name (`test_revoke_races_an_in_flight_push`).
* T3 — window-end revoke by the owner (a), borrower self-revoke (b), the
  rescued value dying at revoke (c), the next command getting a NEW token (d),
  and a stale held-open child env refusing after the window (e).
* T4 — the helper-list reset pair (F1): under the injected env github.com's
  chain is OURS ALONE; the store file is untouched; other hosts keep their
  helpers; and the NEGATIVE CONTROL — the same flow with the reset entry
  dropped — really does write the token into the store, so the instrument can
  catch the bug it exists to catch.
* T5 — host/path bounds: lookalike host refused, ``github.com:443`` serves,
  ``insteadOf`` rewrites fail closed, direct helper invocation refuses
  non-https/non-github/absent/not-listed paths.
* T6 / T7 — RECORDED, ACCEPTED — NOT TESTED. See
  ``test_the_t6_t7_acceptance_records_are_present``: enforcement is external
  (GitHub-side mint narrowing + the helper path guard) and device-scope
  consequences are the accepted cost, disclosed in copy. No coverage is
  claimed for either.
* T8 — 0-peer devices get no injection and no dumps; the ``printenv`` invariant
  (masking on the model-visible result); audit rows and the refusal catalogue
  carry the new entries.

The git-level cells fake github.com LOCALLY (a CONNECT+TLS proxy in front of
real ``git http-backend``), so a real push exercises the real helper path
offline; the node's own git config carries a persisting ``store`` helper and a
marker, so the exclusion is measured against the thing it excludes.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.types import BrokerError, Grant, GrantScope
from local_operator.paths import CONFIG_DIR_ENV

OWNER_DEVICE = "d_00000000000000000000000000000031"
BORROWER_DEVICE = "d_00000000000000000000000000000032"
APP_ID = "123456"
SCRATCH = "damianvtran/scratch"

GIT = shutil.which("git") or "git"


# ---------------------------------------------------------------------------
# The loopback GitHub: a real API (JWT-verifying) on one port
# ---------------------------------------------------------------------------


class _FakeGithub:
    """``api.github.com`` over loopback: mint (JWT-verified) + revoke.

    The mint answers the RFC shapes the real endpoint documents; the revoke is
    the only authority on liveness — a token is dead exactly when this server
    says so, which is what lets every "it stops authenticating" assertion be
    about the mechanism and not about a stub's tolerance.
    """

    def __init__(self, public_pem: str) -> None:
        import jwt

        self._jwt = jwt
        self.public_pem = public_pem
        self.mint_requests: list[dict[str, Any]] = []
        self.revoke_requests: list[str] = []
        self.tokens: set[str] = set()
        self.fail_next_revokes = 0
        self._seq = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: D102 — silence
                pass

            def _json(self, status: int, payload: dict[str, Any] | None) -> None:
                data = b"" if payload is None else json.dumps(payload).encode("utf-8")
                self.send_response(status)
                if data:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                if not re.fullmatch(r"/app/installations/\d+/access_tokens", self.path):
                    self._json(404, {"message": "not found"})
                    return
                header = self.headers.get("Authorization", "")
                try:
                    server._jwt.decode(
                        header.removeprefix("Bearer "),
                        server.public_pem,
                        algorithms=["RS256"],
                        audience=None,
                        issuer=APP_ID,
                        options={"verify_aud": False},
                    )
                except Exception:  # noqa: BLE001 — a forged/absent JWT is a 401
                    self._json(401, {"message": "Bad credentials"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                server.mint_requests.append(body)
                server._seq += 1
                token = f"ghs_test_{server._seq}_{os.urandom(3).hex()}"
                server.tokens.add(token)
                expires = (
                    datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                self._json(201, {"token": token, "expires_at": expires})

            def do_DELETE(self) -> None:  # noqa: N802
                if self.path != "/installation/token":
                    self._json(404, {"message": "not found"})
                    return
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                if server.fail_next_revokes > 0:
                    server.fail_next_revokes -= 1
                    self._json(500, {"message": "transient"})
                    return
                server.revoke_requests.append(token)
                if token in server.tokens:
                    server.tokens.discard(token)
                    self._json(204, None)
                else:
                    self._json(401, {"message": "Bad credentials"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def live(self, token: str) -> bool:
        return token in self.tokens

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# The loopback github.com: git http-backend, behind a CONNECT+TLS proxy
# ---------------------------------------------------------------------------


class _GitBackend:
    """One authenticated git HTTP request → ``git http-backend``.

    Auth is the same shape GitHub serves: ``Basic`` with
    ``x-access-token:<installation token>``. The token must be LIVE per the fake
    GitHub above — that shared liveness is what makes "revoked" bite locally.
    A ``hold`` event (armed per request predicate) withholds the receive-pack
    POST before its auth decision, which is the deterministic race fixture T2
    needs: the client→server bytes are already stalled in flight while the
    revoke lands.
    """

    def __init__(self, project_root: Path, github: _FakeGithub) -> None:
        self.root = project_root
        self.github = github
        self.auth_attempts: list[dict[str, Any]] = []
        self.hold_event: threading.Event | None = None
        self.saw_receive_pack = threading.Event()

    def handle(self, method: str, target: str, headers: dict[str, str], body: bytes) -> tuple:
        path, _, query = target.partition("?")
        if method == "POST" and path.endswith("/git-receive-pack"):
            self.saw_receive_pack.set()
            if self.hold_event is not None:
                self.hold_event.wait(timeout=30.0)
        auth = headers.get("authorization", "")
        token = ""
        if auth.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                user, _, password = decoded.partition(":")
                if user == "x-access-token":
                    token = password
            except Exception:  # noqa: BLE001 — an unreadable header is no credential
                token = ""
        ok = bool(token) and self.github.live(token)
        self.auth_attempts.append(
            {"method": method, "path": path, "token": token, "authorised": ok}
        )
        if not ok:
            return ("401 Unauthorized", {"WWW-Authenticate": 'Basic realm="GitHub"'}, b"auth")
        env = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "QUERY_STRING": query,
            "CONTENT_TYPE": headers.get("content-type", ""),
            "CONTENT_LENGTH": str(len(body)),
            "GIT_PROJECT_ROOT": str(self.root),
            "GIT_HTTP_EXPORT_ALL": "1",
            "REMOTE_USER": "x-access-token",
            "REMOTE_ADDR": "127.0.0.1",
            "PATH": os.environ.get("PATH", ""),
        }
        proc = subprocess.run(
            ["git", "http-backend"], input=body, env=env, capture_output=True, timeout=30
        )
        out = proc.stdout
        sep = out.find(b"\r\n\r\n")
        if sep < 0:
            return ("500 Internal Server Error", {}, b"bad cgi output")
        status = "200 OK"
        resp_headers: dict[str, str] = {}
        for line in out[:sep].decode("latin-1").split("\r\n"):
            name, _, value = line.partition(":")
            if name.lower() == "status":
                status = value.strip()
            else:
                resp_headers[name.strip()] = value.strip()
        return (status, resp_headers, out[sep + 4 :])


class _HttpsGithubProxy:
    """github.com:443, terminated locally — CONNECT → TLS → git http-backend.

    WHY A TLS PROXY: git resolves credentials for the URL it is about to talk
    to, and our helper serves github.com ONLY. A plain local server can never be
    "github.com" to git, so a real push through the real helper needs the tunnel
    trick: ``https_proxy`` points at us, we accept the CONNECT, present a
    throwaway certificate for github.com, and speak real git http-backend on the
    other side. `GIT_SSL_NO_VERIFY` is the test's own concession for the
    throwaway cert and touches nothing in the product.
    """

    def __init__(self, backend: _GitBackend, certfile: Path, keyfile: Path) -> None:
        self.backend = backend
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ctx.load_cert_chain(certfile, keyfile)
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._closed = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn: socket.socket) -> None:
        try:
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            head, _, rest = buf.partition(b"\r\n\r\n")
            if not head.split(b"\r\n", 1)[0].startswith(b"CONNECT"):
                conn.close()
                return
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            tls = self.ctx.wrap_socket(conn, server_side=True)
            self._serve_http(tls, rest)
        except Exception:  # noqa: BLE001 — a broken connection is a failed clone
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _serve_http(self, tls: ssl.SSLSocket, buffered: bytes) -> None:
        buf = buffered
        while True:
            while b"\r\n\r\n" not in buf:
                chunk = tls.recv(65536)
                if not chunk:
                    return
                buf += chunk
            head, _, rest = buf.partition(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, target, _version = lines[0].split(" ")
            headers = {}
            for line in lines[1:]:
                name, _, value = line.partition(":")
                headers[name.strip().lower()] = value.strip()
            length = int(headers.get("content-length") or 0)
            body = rest
            while len(body) < length:
                chunk = tls.recv(65536)
                if not chunk:
                    return
                body += chunk
            buf = body[length:]
            body = body[:length]
            status, resp_headers, payload = self.backend.handle(method, target, headers, body)
            head_lines = [f"HTTP/1.1 {status}"]
            for name, value in resp_headers.items():
                head_lines.append(f"{name}: {value}")
            head_lines.append(f"Content-Length: {len(payload)}")
            head_lines.append("Connection: keep-alive")
            tls.sendall(("\r\n".join(head_lines) + "\r\n\r\n").encode("latin-1") + payload)

    def close(self) -> None:
        self._closed.set()
        try:
            self.sock.close()
        except OSError:
            pass


class _PlainGitServer:
    """The same backend on plain HTTP: the rewrite target for the ``insteadOf`` cell."""

    def __init__(self, backend: _GitBackend) -> None:
        self.backend = backend
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        backend = self.backend

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                headers = {k.lower(): v for k, v in self.headers.items()}
                status, resp_headers, payload = backend.handle(
                    self.command, self.path, headers, body
                )
                self.send_response(int(status.split(" ", 1)[0]))
                for name, value in resp_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                self._serve()

            def do_POST(self) -> None:  # noqa: N802
                self._serve()

        return Handler

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self.thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_keypair() -> tuple[str, str]:
    """A real RSA keypair: the private PEM goes in the store, the public verifies JWTs."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return pem, public


@pytest.fixture()
def github_api(app_keypair: tuple[str, str]) -> Any:
    fake = _FakeGithub(app_keypair[1])
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture(autouse=True)
def _no_secret_broker_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never spawn the secret broker daemon from a test.

    ``access.retrieve_secret`` reaches the broker first when one is reachable and
    falls back to the local decrypt when one cannot start — this pins the
    fallback so no cell launches a daemon process, and it exercises exactly the
    availability arms the code documents. The store tiers under test are
    keyfile-tier stores in isolated roots.
    """
    from local_operator.secrets import client as secrets_client

    monkeypatch.setattr(secrets_client, "ensure_broker", lambda *a, **k: False)


class _Env:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "config"
        self.root.mkdir()
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.tmp = tmp_path


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    holder = _Env(tmp_path)
    monkeypatch.setenv("HOME", str(holder.home))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(holder.root))
    return holder


def _seed_app_key(root: Path, pem: str, *, app_id: str = APP_ID) -> None:
    from local_operator.secrets.keys import load_master_key
    from local_operator.secrets.store import SecretStore

    # ``create=True`` writes the KEYFILE the production read path resolves
    # (``resolve_master_key``); an in-memory-only key would pass every same-object
    # read and fail exactly the one the broker makes.
    master_key = load_master_key(root, create=True)
    store = SecretStore(master_key, base=root)
    store.initialize()
    store.set(
        github_mod.APP_SECRET_NAME,
        json.dumps({"app_id": app_id, "installation_id": 4242, "private_key": pem}).encode(),
        description="test App key (the App itself does not exist yet)",
    )


def _write_config(root: Path, repositories: list[str], *, grant_ttl_s: float | None = None) -> None:
    credentials: dict[str, Any] = {"github": {"repositories": list(repositories)}}
    if grant_ttl_s is not None:
        credentials["grant_ttl_s"] = grant_ttl_s
    # ``values:`` is the level this store actually reads (the config.yml warning
    # says so out loud when a key sits at the top level).
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"network": {"credentials": credentials}}}), encoding="utf-8"
    )


class _Link:
    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.network_id = "n_gh"
        self.epoch = 1


def _grant_frame(device: str, **overrides: Any) -> dict[str, Any]:
    frame = {
        "op": "net_broker",
        "req": 1,
        "kind": "grant",
        "key": github_mod.GITHUB_KEY,
        "provider": github_mod.GITHUB_KEY,
        "from_device": device,
        "from_device_name": device[-4:],
        "for_session": "sess-1",
        "model_id": "test-model",
        "force_refresh": False,
    }
    frame.update(overrides)
    return frame


def _detail(reply: dict[str, Any]) -> dict[str, Any]:
    assert reply.get("op") == "ack", reply
    detail = reply.get("detail")
    assert isinstance(detail, dict), reply
    return detail


@pytest.fixture()
def owner(env: _Env, app_keypair: tuple[str, str], github_api: _FakeGithub) -> Any:
    """An owner device: App key in its secret store, a device-scoped share, the broker."""
    _seed_app_key(env.root, app_keypair[0])
    _write_config(env.root, [SCRATCH])
    document = placement_mod.PlacementDocument("n_gh", root=env.root, written_by=OWNER_DEVICE)
    document.declare(
        github_mod.GITHUB_KEY,
        owner_device=OWNER_DEVICE,
        owner_device_name="owner-laptop",
        provider=github_mod.GITHUB_KEY,
        identity_label="",
        by=OWNER_DEVICE,
    )
    document.grant(github_mod.GITHUB_KEY, BORROWER_DEVICE, scope="device", by=OWNER_DEVICE)
    document.save()
    from local_operator.network.audit import AuditLog

    audit = AuditLog(root=env.root)
    broker = owner_mod.MeshCredentialBroker(
        root=env.root,
        self_device=OWNER_DEVICE,
        self_device_name="owner-laptop",
        network_id="n_gh",
        audit=audit,
        github_minter=github_mod.GithubMinter(base_url=github_api.url),
    )
    try:
        yield type(
            "Owner",
            (),
            {
                "root": env.root,
                "broker": broker,
                "document": document,
                "audit": audit,
                "env": env,
            },
        )
    finally:
        broker.close()
        audit.close()


def _ask_grant(owner: Any, device: str, **kwargs: Any) -> dict[str, Any]:
    return _detail(owner.broker.on_broker(_Link(device), _grant_frame(device, **kwargs)))


def _audit_rows(owner: Any, event: str) -> list[dict[str, Any]]:
    return [row for row in owner.audit.tail(100, network_id="n_gh") if row.get("event") == event]


def _wait_for(predicate: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


# ---------------------------------------------------------------------------
# The node's git side: config with a persisting helper, markers, the backend
# ---------------------------------------------------------------------------


_MARKER_SCRIPT = """#!/bin/sh
{ echo "op=$1"; cat; echo "---"; } >> "$MARKER_LOG"
exit 0
"""


def _node_gitconfig(home: Path, marker: Path, marker_log: Path) -> None:
    (home / ".gitconfig").write_text(
        "[credential]\n"
        "\thelper = store\n"
        f"\thelper = {shlex.quote(str(marker))}\n"
        "[user]\n"
        "\tname = Node User\n"
        "\temail = node@example.test\n",
        encoding="utf-8",
    )


def _marker(path: Path) -> None:
    path.write_text(_MARKER_SCRIPT, encoding="utf-8")
    path.chmod(0o755)


def _node_env(env: _Env) -> dict[str, str]:
    """The node's own environment: no session injection, its config, its HOME.

    ``GIT_CONFIG_SYSTEM=/dev/null`` keeps the container/system config out of the
    chain — on this fleet the SYSTEM config carries ``credential.helper =
    osxkeychain``, which would ask the operator's keychain for a password (and
    the reset-pair semantics under test are about the GLOBAL and repo-local
    helpers, which this cell provides itself).
    """
    keep = ("PATH", "TERM", "TMPDIR", "LANG")
    base = {name: os.environ[name] for name in keep if name in os.environ}
    base.update(
        {
            "HOME": str(env.home),
            CONFIG_DIR_ENV: str(env.root),
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return base


def _injected_env(env: _Env, token: str) -> dict[str, str]:
    """The exact child env the session injects, merged over the node's own."""
    merged = _node_env(env)
    merged.update(github_mod.git_env_for_token(token))
    return merged


def _write_config_extra(env: _Env, extra: dict[str, str]) -> None:
    pass  # placeholder kept out of the way; config extras ride _with_config below


def _with_config(base: dict[str, str], entries: list[tuple[str, str]]) -> dict[str, str]:
    """Append git-config env entries (the node's own config-in-env, for tests)."""
    merged = dict(base)
    count = int(merged.get("GIT_CONFIG_COUNT") or 0)
    for offset, (key, value) in enumerate(entries):
        merged[f"GIT_CONFIG_KEY_{count + offset}"] = key
        merged[f"GIT_CONFIG_VALUE_{count + offset}"] = value
    merged["GIT_CONFIG_COUNT"] = str(count + len(entries))
    return merged


def _run(args: list[str], *, env: dict[str, str], cwd: Path | None = None, stdin: str = "") -> Any:
    return subprocess.run(
        args, env=env, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=60
    )


def _fill(env: dict[str, str], host: str, path: str = "", cwd: Path | None = None) -> Any:
    lines = f"protocol=https\nhost={host}\n"
    if path:
        lines += f"path={path}\n"
    lines += "\n"
    return _run([GIT, "credential", "fill"], env=env, cwd=cwd, stdin=lines)


def _github_git(env: _Env, tmp: Path, github_api: _FakeGithub) -> tuple[_GitBackend, Path]:
    """A bare scratch repo the proxy serves, plus the backend sharing liveness."""
    project_root = tmp / "gitroot"
    (project_root / "damianvtran").mkdir(parents=True)
    bare = project_root / "damianvtran" / "scratch.git"
    bare.mkdir()
    git_env = _node_env(env)
    for cmd in (["git", "init", "--bare", "-q"], ["git", "config", "http.receivepack", "true"]):
        result = _run(cmd, env=git_env, cwd=bare)
        assert result.returncode == 0, (cmd, result.stderr)
    return _GitBackend(project_root, github_api), bare


def _make_cert(cn: str, out: Path) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.UTC)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(cn)]), critical=False)
        .sign(key, hashes.SHA256())
    )
    certfile, keyfile = out / "cert.pem", out / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return certfile, keyfile


def _work_repo(env: _Env, tmp: Path) -> Path:
    work = tmp / "work"
    work.mkdir()
    git_env = _node_env(env)
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "node@example.test"],
        ["git", "config", "user.name", "Node User"],
    ):
        assert _run(cmd, env=git_env, cwd=work).returncode == 0
    (work / "README.md").write_text("hello\n", encoding="utf-8")
    assert _run(["git", "add", "."], env=git_env, cwd=work).returncode == 0
    assert _run(["git", "commit", "-q", "-m", "init"], env=git_env, cwd=work).returncode == 0
    return work


def _tree_bytes(root: Path) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out.append((str(path.relative_to(root)), path.read_bytes()))
    return out


# ---------------------------------------------------------------------------
# T1/T3/T8a — the owner side: mint, narrowing, refusals, audit
# ---------------------------------------------------------------------------


def test_a_borrow_is_minted_narrowed_and_audited(owner: Any, github_api: _FakeGithub) -> None:
    """T1 owner half: the token is minted against the App JWT, narrowed, audited.

    The JWT check lives IN the fake endpoint (it 401s a forged token), so a
    mint that succeeded is proof the real ``_app_jwt`` was accepted by a real
    verifier. ``repositories`` and ``permissions`` are read from the wire body —
    the narrowing GitHub enforces — not from our own claim about it.
    """
    detail = _ask_grant(owner, BORROWER_DEVICE)
    assert detail["kind"] == "grant", detail
    token = detail["access_token"]
    assert github_api.live(token)
    assert github_api.mint_requests == [
        {
            "repositories": ["scratch"],
            "permissions": {"contents": "write", "pull_requests": "write"},
        }
    ], github_api.mint_requests
    assert detail["credential_ref"]["kind"] == "github-app"
    assert detail["credential_ref"]["provider"] == github_mod.GITHUB_KEY
    assert detail["scope"] == {"kind": "device", "session_id": ""}
    assert detail["refreshed"] is True
    now_ms = time.time() * 1000
    assert detail["grant_expires_at_ms"] <= detail["token_expires_at_ms"]
    assert detail["grant_expires_at_ms"] <= now_ms + 900_000 + 5_000
    # The delegation record: act/sub spelled, the bearer absent (whitelist refuses
    # `access_token` by name).
    rows = _audit_rows(owner, "credential.grant")
    assert len(rows) == 1, rows
    record = rows[0]
    assert record["actor"] == OWNER_DEVICE and record["subject"] == BORROWER_DEVICE
    fields = record["detail"]
    assert fields["act"] == OWNER_DEVICE and fields["sub"] == BORROWER_DEVICE
    assert fields["credential_key"] == github_mod.GITHUB_KEY
    assert fields["credential_kind"] == "github-app"
    assert fields["scope"] == "device"
    assert token not in json.dumps(record)
    assert "access_token" not in fields


def test_a_session_scoped_share_is_refused_by_name_in_the_document(owner: Any) -> None:
    """F3, mechanism half: the document itself refuses ``--scope session`` for this key."""
    from local_operator.network.types import MeshRefusal

    document = owner.document
    with pytest.raises(MeshRefusal) as caught:
        document.grant(github_mod.GITHUB_KEY, "d_" + "9" * 32, scope="session", by=OWNER_DEVICE)
    assert caught.value.code == "device_scope_required"
    # And the owner refuses to SERVE a hand-written session row (belt), by name.
    owner.document.grant(github_mod.GITHUB_KEY, BORROWER_DEVICE, scope="device", by=OWNER_DEVICE)
    entry = owner.document.entry(github_mod.GITHUB_KEY)
    assert entry is not None
    for holder in entry.holders:
        if holder.device == BORROWER_DEVICE:
            holder.scope = "session"  # hand-written, below the verb
    owner.document.save()
    detail = _ask_grant(owner, BORROWER_DEVICE)
    assert detail["code"] == "device_scope_required", detail
    assert "nothing was lent" in detail["message"]


def test_the_missing_app_is_the_named_interim_state(env: _Env) -> None:
    """The state a user meets TODAY: no App → ``no_local_credential``, one small step."""
    from local_operator.network.credentials import messages

    root = env.root
    _write_config(root, [SCRATCH])
    document = placement_mod.PlacementDocument("n_gh", root=root, written_by=OWNER_DEVICE)
    document.declare(
        github_mod.GITHUB_KEY,
        owner_device=OWNER_DEVICE,
        owner_device_name="owner-laptop",
        provider=github_mod.GITHUB_KEY,
        identity_label="",
        by=OWNER_DEVICE,
    )
    document.grant(github_mod.GITHUB_KEY, BORROWER_DEVICE, scope="device", by=OWNER_DEVICE)
    document.save()
    broker = owner_mod.MeshCredentialBroker(
        root=root, self_device=OWNER_DEVICE, self_device_name="owner-laptop", network_id="n_gh"
    )
    try:
        detail = _detail(broker.on_broker(_Link(BORROWER_DEVICE), _grant_frame(BORROWER_DEVICE)))
    finally:
        broker.close()
    assert detail["code"] == "no_local_credential", detail
    sentence = messages.render_broker_error(
        BrokerError(
            code=detail["code"],
            key=detail["key"],
            owner_device_name=detail.get("owner_device_name") or "owner-laptop",
        ),
        key=github_mod.GITHUB_KEY,
        owner_name="owner-laptop",
    )
    assert "owner-laptop" in sentence
    assert "unavailable until a GitHub App is configured" in sentence
    assert "one small step" in sentence
    assert "Public clones" in sentence
    assert "lop login" not in sentence, "github has no local-login remedy to offer"


def test_an_unusable_app_key_refuses_by_name(env: _Env) -> None:
    """A secret that exists but is not a usable App key is its own named refusal."""
    from local_operator.secrets.keys import load_master_key
    from local_operator.secrets.store import SecretStore

    master_key = load_master_key(env.root, create=True)
    store = SecretStore(master_key, base=env.root)
    store.initialize()
    store.set(github_mod.APP_SECRET_NAME, b"not json", description="broken")
    _write_config(env.root, [SCRATCH])
    document = placement_mod.PlacementDocument("n_gh", root=env.root, written_by=OWNER_DEVICE)
    document.declare(
        github_mod.GITHUB_KEY,
        owner_device=OWNER_DEVICE,
        owner_device_name="owner-laptop",
        provider=github_mod.GITHUB_KEY,
        identity_label="",
        by=OWNER_DEVICE,
    )
    document.grant(github_mod.GITHUB_KEY, BORROWER_DEVICE, scope="device", by=OWNER_DEVICE)
    document.save()
    broker = owner_mod.MeshCredentialBroker(
        root=env.root, self_device=OWNER_DEVICE, self_device_name="owner-laptop", network_id="n_gh"
    )
    try:
        detail = _detail(broker.on_broker(_Link(BORROWER_DEVICE), _grant_frame(BORROWER_DEVICE)))
    finally:
        broker.close()
    assert detail["code"] == "github_app_unusable", detail


def test_no_designated_repositories_refuses_rather_than_widening(
    owner: Any, github_api: _FakeGithub, env: _Env
) -> None:
    """Empty allow-list at mint time ⇒ refusal; never "no narrowing"."""
    _write_config(owner.env.root, [])
    detail = _ask_grant(owner, BORROWER_DEVICE)
    assert detail["code"] == "github_repositories_unset", detail
    assert github_api.mint_requests == []
    assert not github_api.tokens, "a token was minted without the designated list"


# ---------------------------------------------------------------------------
# T2/T3 — mint-revoke: now, at the window end, and after failure
# ---------------------------------------------------------------------------


def test_revoke_deletes_the_outstanding_token_now(owner: Any, github_api: _FakeGithub) -> None:
    """The operator's immediate half: ``DELETE /installation/token``, audited, idempotent."""
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    count = owner.broker.revoke_outstanding(github_mod.GITHUB_KEY, BORROWER_DEVICE)
    assert count == 1
    assert github_api.revoke_requests == [token]
    assert not github_api.live(token)
    rows = _audit_rows(owner, "credential.revoke")
    assert len(rows) == 1, rows
    fields = rows[0]["detail"]
    assert rows[0]["actor"] == OWNER_DEVICE and rows[0]["subject"] == BORROWER_DEVICE
    assert fields["act"] == OWNER_DEVICE and fields["sub"] == BORROWER_DEVICE
    assert fields["cause"] == "revoked" and fields["revoked"] == 1
    # Idempotent: nothing left to revoke, no error, no second call.
    assert owner.broker.revoke_outstanding(github_mod.GITHUB_KEY, BORROWER_DEVICE) == 0
    assert github_api.revoke_requests == [token]


def test_window_end_revokes_on_the_broker_loop(
    owner: Any, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3(a): the broker's own loop issues the DELETE at ``grant_expires_at``.

    The window is shortened by patching the TTL reader the broker uses — the
    mechanism under test is the REVOKER (registration, scheduling on the loop,
    the DELETE), not the number of seconds a config says.
    """
    monkeypatch.setattr(owner_mod, "_grant_ttl_s", lambda root: 1.0)
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    assert github_api.live(token)
    assert _wait_for(
        lambda: github_api.revoke_requests == [token], timeout=10.0
    ), "the window-end revoker never fired"
    assert not github_api.live(token)
    rows = _audit_rows(owner, "credential.revoke")
    assert rows and rows[-1]["detail"]["cause"] == "grant_expired", rows


def test_a_failed_revoke_is_retried_and_recorded(owner: Any, github_api: _FakeGithub) -> None:
    """A failed DELETE is a recorded failure, then a success — never a silent no-op."""
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    github_api.fail_next_revokes = 1
    # The retry delay is a production constant (60 s); shortened here so the
    # retry CONTRACT is observed in bounded time rather than re-tested.
    owner.broker._lender()._retry_s = 0.25  # noqa: SLF001 — the cell's own dial
    assert owner.broker.revoke_outstanding(github_mod.GITHUB_KEY, BORROWER_DEVICE) == 0
    assert github_api.live(token), "the token died without a successful DELETE"
    assert _wait_for(
        lambda: github_api.revoke_requests == [token] and not github_api.live(token), timeout=10.0
    ), "the retry never landed"
    rows = _audit_rows(owner, "credential.revoke")
    assert [row["detail"]["revoked"] for row in rows] == [0, 1], rows


def test_borrower_self_revoke_fires_at_window_end(
    env: _Env, monkeypatch: pytest.MonkeyPatch, github_api: _FakeGithub
) -> None:
    """T3(b): the borrower's belt — a DELETE at its grant's window end, without a retry loop."""
    monkeypatch.setattr(github_mod, "GITHUB_API_BASE", github_api.url)
    token = f"ghs_self_{os.urandom(4).hex()}"
    github_api.tokens.add(token)
    grant = Grant(
        access_token=token,
        kind="bearer",
        token_expires_at_ms=int((time.time() + 3600) * 1000),
        grant_expires_at_ms=int((time.time() + 1.0) * 1000),
        credential_ref=github_mod_credential_ref(),
        served_by=OWNER_DEVICE,
        scope=GrantScope(kind="device", session_id=""),
    )

    class _Client:
        should = True

        def should_borrow(self, key: str) -> bool:
            return self.should

        def request_grant_sync(self, key: str, **kwargs: Any) -> Any:
            return grant

    env_map, served = github_mod.borrowed_git_env(client=_Client())
    assert served == token and env_map["GH_TOKEN"] == token
    assert _wait_for(
        lambda: not github_api.live(token), timeout=10.0
    ), "the borrower's self-revoke never fired"
    assert github_api.revoke_requests == [token]


def github_mod_credential_ref() -> Any:
    from local_operator.network.credentials.types import (
        CredentialRef,
        synthetic_credential_id,
    )

    return CredentialRef(
        owner_device=OWNER_DEVICE,
        owner_device_name="owner-laptop",
        provider=github_mod.GITHUB_KEY,
        kind=github_mod.GITHUB_KIND,
        credential_id=synthetic_credential_id(github_mod.GITHUB_KEY, OWNER_DEVICE),
    )


# ---------------------------------------------------------------------------
# T4 — the reset pair, the exclusion matrix, and the negative control
# ---------------------------------------------------------------------------


def test_the_reset_pair_is_exactly_the_spike_shape(env: _Env) -> None:
    """The env's config entries, pinned literally: count 3, reset FIRST, path on."""
    injected = github_mod.git_env_for_token("ghs_shape")
    assert injected["GIT_CONFIG_COUNT"] == "3"
    assert injected["GIT_CONFIG_KEY_0"] == "credential.https://github.com.helper"
    assert injected["GIT_CONFIG_VALUE_0"] == "", "the reset entry must be EMPTY"
    assert injected["GIT_CONFIG_KEY_1"] == "credential.https://github.com.helper"
    assert injected["GIT_CONFIG_VALUE_1"].startswith("!")
    assert "credential" in injected["GIT_CONFIG_VALUE_1"]
    assert "git-helper" in injected["GIT_CONFIG_VALUE_1"]
    assert injected["GIT_CONFIG_KEY_2"] == "credential.https://github.com.useHttpPath"
    assert injected["GIT_CONFIG_VALUE_2"] == "true"
    assert injected["GH_TOKEN"] == injected["GITHUB_TOKEN"] == "ghs_shape"


def test_helper_exclusion_matrix(env: _Env, tmp_path: Path) -> None:
    """T4 (F1): github.com runs OUR helper alone; the store is untouched; gitlab keeps its chain.

    The node is configured the hostile way on purpose: a persisting ``store``
    helper plus a marker helper are in its global config, and a repo-local
    marker sits in ``.git/config`` of the working repository. Under the
    injected env none of them may see the github.com interaction, while the
    SAME env must leave gitlab.com's chain fully populated (an unscoped reset
    would strip those too — the test can see the difference).
    """
    _write_config(env.root, [SCRATCH])
    marker = env.home / "marker.sh"
    marker_log = env.home / "marker.log"
    _marker(marker)
    _node_gitconfig(env.home, marker, marker_log)
    repo = tmp_path / "repo"
    repo.mkdir()
    local_marker = tmp_path / "local-marker.sh"
    local_marker_log = tmp_path / "local-marker.log"
    _marker(local_marker)
    git_env = _node_env(env)
    assert _run(["git", "init", "-q"], env=git_env, cwd=repo).returncode == 0
    assert (
        _run(
            ["git", "config", "credential.helper", str(local_marker)],
            env=git_env,
            cwd=repo,
        ).returncode
        == 0
    )
    token = f"ghs_excl_{os.urandom(4).hex()}"
    injected = _injected_env(env, token)
    injected["MARKER_LOG"] = str(marker_log)

    # (1) github.com fill: OURS answers; nothing else was consulted.
    filled = _fill(injected, "github.com", SCRATCH + ".git", cwd=repo)
    assert filled.returncode == 0, filled.stderr
    assert "username=x-access-token" in filled.stdout
    assert f"password={token}" in filled.stdout
    assert not marker_log.exists(), "a persisting helper was consulted for github.com"
    assert not local_marker_log.exists(), "the repo-local helper received github.com"
    assert not (env.home / ".git-credentials").exists()

    # (2) github.com approve: still nothing persists.
    approved = _run(
        [GIT, "credential", "approve"],
        env=injected,
        cwd=repo,
        stdin=(
            f"protocol=https\nhost=github.com\npath={SCRATCH}.git\n"
            f"username=x-access-token\npassword={token}\n\n"
        ),
    )
    assert approved.returncode == 0, approved.stderr
    assert not (env.home / ".git-credentials").exists(), "the store helper WROTE for github.com"
    assert not marker_log.exists(), "a persisting helper received the github.com approve"
    assert not local_marker_log.exists(), "the repo-local helper received the github.com approve"

    # (3) another host keeps its configured helpers: the marker IS consulted there.
    other = _fill(injected, "gitlab.com", "group/project.git")
    assert "password=" not in other.stdout and token not in other.stdout
    assert marker_log.exists(), "the reset stripped another host's helpers"
    log_text = marker_log.read_text(encoding="utf-8")
    assert "host=gitlab.com" in log_text
    assert "host=github.com" not in log_text


def test_the_negative_control_without_the_reset_writes_the_token_to_the_store(
    env: _Env, tmp_path: Path
) -> None:
    """THE NEGATIVE CONTROL (the blocker's whole proof): drop the reset ⇒ the bug shows.

    Same node, same flow, same helper chain — the ONLY difference is the empty
    first entry. With it, github.com's list is ours alone and ``store`` never
    writes; without it, the store receives the value and ``.git-credentials``
    appears. If this cell could not fail, the exclusion cells above would prove
    nothing.
    """
    marker = env.home / "marker.sh"
    marker_log = env.home / "marker.log"
    _marker(marker)
    _node_gitconfig(env.home, marker, marker_log)
    _write_config(env.root, [SCRATCH])
    token = "ghs_control"
    without_reset = _node_env(env)
    without_reset.update(
        {
            "GH_TOKEN": token,
            "GITHUB_TOKEN": token,
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "credential.https://github.com.helper",
            "GIT_CONFIG_VALUE_0": github_mod.helper_command(),
            "GIT_CONFIG_KEY_1": "credential.https://github.com.useHttpPath",
            "GIT_CONFIG_VALUE_1": "true",
            "MARKER_LOG": str(marker_log),
        }
    )
    approved = _run(
        [GIT, "credential", "approve"],
        env=without_reset,
        stdin=(
            f"protocol=https\nhost=github.com\npath={SCRATCH}.git\n"
            f"username=x-access-token\npassword={token}\n\n"
        ),
    )
    assert approved.returncode == 0, approved.stderr
    store_file = env.home / ".git-credentials"
    assert store_file.exists(), "the negative control did not reproduce the bug"
    assert token in store_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# T1 git half — a real push through the borrowed env
# ---------------------------------------------------------------------------


def test_a_push_through_the_borrowed_env_succeeds_and_leaves_no_trace(
    owner: Any, github_api: _FakeGithub, tmp_path: Path
) -> None:
    """T1: real ``git push`` to github.com, served by OUR helper, nothing on disk.

    The push goes through a CONNECT+TLS tunnel to a local git http-backend
    because a unit test cannot reach the real github.com — everything else is
    real: the helper is invoked by git, decides from ``$GH_TOKEN``, and the
    receive-pack is a real one. Afterward: no ``.git-credentials``, the marker
    helper never saw github.com, the token appears in NO file under the node's
    HOME or the config root, and the node's gitconfig is byte-identical.
    """
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    env = owner.env
    marker = env.home / "marker.sh"
    marker_log = env.home / "marker.log"
    _marker(marker)
    _node_gitconfig(env.home, marker, marker_log)
    before = _tree_bytes(env.home)

    backend, _bare = _github_git(env, tmp_path, github_api)
    certfile, keyfile = _make_cert("github.com", tmp_path)
    proxy = _HttpsGithubProxy(backend, certfile, keyfile)
    try:
        work = _work_repo(env, tmp_path)
        push_env = _injected_env(env, token)
        push_env.update(
            {
                "https_proxy": f"http://127.0.0.1:{proxy.port}",
                "GIT_SSL_NO_VERIFY": "true",
                "MARKER_LOG": str(marker_log),
            }
        )
        pushed = _run(
            [GIT, "push", "https://github.com/damianvtran/scratch.git", "main:main"],
            env=push_env,
            cwd=work,
        )
        assert pushed.returncode == 0, (pushed.stdout, pushed.stderr[-2000:])
        assert backend.saw_receive_pack.is_set(), "no receive-pack ever arrived"
        assert any(a["authorised"] and a["token"] == token for a in backend.auth_attempts)
    finally:
        proxy.close()

    # The trace checks, on the real filesystem.
    assert not (env.home / ".git-credentials").exists()
    marker_text = marker_log.read_text(encoding="utf-8") if marker_log.exists() else ""
    assert "host=github.com" not in marker_text
    after = _tree_bytes(env.home)
    assert _bytes_of("config.yml", after) == _bytes_of("config.yml", before)
    assert _bytes_of(".gitconfig", after) == _bytes_of(".gitconfig", before)
    for root in (env.home, env.root):
        for relative, content in _tree_bytes(root):
            assert token.encode() not in content, f"the token reached {root}/{relative}"


def _bytes_of(relative: str, tree: list[tuple[str, bytes]]) -> bytes | None:
    for name, content in tree:
        if name == relative:
            return content
    return None


def test_revoke_races_an_in_flight_push(
    owner: Any, github_api: _FakeGithub, tmp_path: Path
) -> None:
    """T2: the revoke lands while the receive-pack is in flight; the push dies.

    Deterministic by fixture: the receive-pack POST is HELD before its auth
    decision (``hold_event``), the revoke runs, the POST is released — the old
    token must not authenticate, the push must fail, and the attempt must be
    visible in the audit. Then the placement row is revoked and the next borrow
    is refused, sentence included.
    """
    from local_operator.network.credentials import messages

    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    env = owner.env
    backend, _bare = _github_git(env, tmp_path, github_api)
    certfile, keyfile = _make_cert("github.com", tmp_path)
    proxy = _HttpsGithubProxy(backend, certfile, keyfile)
    backend.hold_event = threading.Event()
    try:
        work = _work_repo(env, tmp_path)
        push_env = _injected_env(env, token)
        push_env.update(
            {
                "https_proxy": f"http://127.0.0.1:{proxy.port}",
                "GIT_SSL_NO_VERIFY": "true",
            }
        )
        push = subprocess.Popen(
            [GIT, "push", "https://github.com/damianvtran/scratch.git", "main:main"],
            env=push_env,
            cwd=work,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert backend.saw_receive_pack.wait(timeout=30.0), "the receive-pack never got in flight"
        # IN FLIGHT: revoke, then release the held request.
        assert owner.broker.revoke_outstanding(github_mod.GITHUB_KEY, BORROWER_DEVICE) == 1
        assert github_api.revoke_requests == [token]
        backend.hold_event.set()
        stdout, stderr = push.communicate(timeout=60.0)
    finally:
        proxy.close()
    assert push.returncode != 0, (stdout, stderr)
    assert not github_api.live(token)
    # Every request presenting the old token AFTER the revoke was refused.
    refused = [
        attempt
        for attempt in backend.auth_attempts
        if attempt["method"] == "POST" and attempt["token"] == token and not attempt["authorised"]
    ]
    assert refused, backend.auth_attempts
    assert not any(
        attempt["method"] == "POST" and attempt["authorised"] for attempt in backend.auth_attempts
    ), "a receive-pack authenticated after the revoke"
    rows = _audit_rows(owner, "credential.revoke")
    assert rows and rows[-1]["detail"]["cause"] == "revoked"

    # THE NEXT BORROW IS REFUSED, in the github voice (no local-login dead end).
    owner.document.revoke(github_mod.GITHUB_KEY, BORROWER_DEVICE, by=OWNER_DEVICE)
    owner.document.save()
    detail = _ask_grant(owner, BORROWER_DEVICE)
    assert detail["code"] == "not_a_holder", detail
    sentence = messages.render_broker_error(
        BrokerError(code=detail["code"], key=github_mod.GITHUB_KEY),
        key=github_mod.GITHUB_KEY,
        owner_name="owner-laptop",
    )
    assert "does not share its GitHub credential" in sentence
    assert "lop login" not in sentence


def test_a_stale_child_env_does_not_authenticate_after_the_window_ends(
    owner: Any, github_api: _FakeGithub, tmp_path: Path
) -> None:
    """T3(e): a child held open across the window keeps the old value in its env —
    and it must FAIL. The env cannot be un-exported; the token can be revoked."""
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    env = owner.env
    backend, _bare = _github_git(env, tmp_path, github_api)
    certfile, keyfile = _make_cert("github.com", tmp_path)
    proxy = _HttpsGithubProxy(backend, certfile, keyfile)
    try:
        work = _work_repo(env, tmp_path)
        stale_env = _injected_env(env, token)
        stale_env.update(
            {
                "https_proxy": f"http://127.0.0.1:{proxy.port}",
                "GIT_SSL_NO_VERIFY": "true",
            }
        )
        # While live, the same env works.
        first = _run(
            [GIT, "ls-remote", "https://github.com/damianvtran/scratch.git"],
            env=stale_env,
            cwd=work,
        )
        assert first.returncode == 0, first.stderr
        # Window ends (the owner's revoker runs).
        assert owner.broker.revoke_outstanding(github_mod.GITHUB_KEY, BORROWER_DEVICE) == 1
        second = _run(
            [GIT, "ls-remote", "https://github.com/damianvtran/scratch.git"],
            env=stale_env,
            cwd=work,
        )
        assert second.returncode != 0, "the stale child env still authenticated"
        assert not github_api.live(token)
    finally:
        proxy.close()


# ---------------------------------------------------------------------------
# T5 — host and path bounds
# ---------------------------------------------------------------------------


def test_helper_bounds_matrix(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """T5(a,b,d): the helper's own decisions, then the real git config scope.

    The unit matrix drives ``git_helper_reply`` directly — the same function the
    shipped subcommand wraps — and the subprocess cell runs the ACTUAL
    ``lop credential git-helper`` entry, so the wiring is not assumed.
    """
    token = "ghs_bounds"
    monkeypatch.setenv("GH_TOKEN", token)

    def reply(operation: str, lines: list[str], repositories: list[str]) -> str:
        return github_mod.git_helper_reply(operation, lines, token=token, repositories=repositories)

    listed = [SCRATCH]
    assert reply("get", ["protocol=https", "host=github.com", f"path={SCRATCH}.git"], listed)
    assert reply("get", ["protocol=https", "host=github.com:443", f"path={SCRATCH}.git"], listed)
    assert reply("get", ["protocol=https", "host=github.com", f"path={SCRATCH}.git"], listed)
    # Refusals: everything else is silence.
    assert reply("get", ["protocol=http", "host=github.com", f"path={SCRATCH}.git"], listed) == ""
    assert reply("get", ["protocol=https", "host=gitlab.com", f"path={SCRATCH}.git"], listed) == ""
    assert (
        reply("get", ["protocol=https", "host=github.com.evil.com", f"path={SCRATCH}.git"], listed)
        == ""
    )
    assert reply("get", ["protocol=https", "host=github.com"], listed) == "", "no path served"
    assert (
        reply("get", ["protocol=https", "host=github.com", "path=other/repo.git"], listed) == ""
    ), "a repo outside the list was served"
    assert (
        reply("get", ["protocol=https", "host=github.com", f"path={SCRATCH}.git"], []) == ""
    ), "an EMPTY allow-list must refuse, never widen"
    assert (
        reply("store", ["protocol=https", "host=github.com", f"path={SCRATCH}.git"], listed) == ""
    )
    assert (
        reply("erase", ["protocol=https", "host=github.com", f"path={SCRATCH}.git"], listed) == ""
    )

    # The real subcommand, as git invokes it.
    _write_config(env.root, [SCRATCH])
    completed = _run(
        [sys.executable, "-m", "local_operator.cli", "credential", "git-helper", "get"],
        env={**_node_env(env), "GH_TOKEN": token},
        stdin=f"protocol=https\nhost=github.com\npath={SCRATCH}.git\n\n",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"username=x-access-token\npassword={token}\n"
    refused = _run(
        [sys.executable, "-m", "local_operator.cli", "credential", "git-helper", "get"],
        env={**_node_env(env), "GH_TOKEN": token},
        stdin=f"protocol=http\nhost=github.com\npath={SCRATCH}.git\n\n",
    )
    assert refused.returncode == 0 and refused.stdout == ""


def test_an_instead_of_rewrite_fails_closed_and_never_offers_the_token(
    env: _Env, owner: Any, github_api: _FakeGithub, tmp_path: Path
) -> None:
    """T5(c): the spike's adversarial case, as a cell.

    ``url.<local>.insteadOf = https://github.com/`` rewrites the request to a
    non-github host; git resolves credentials for THAT host, so our scoped
    helper is not in its chain and the token is never offered. The local server
    proves it: it was reached, and none of the requests it recorded carried an
    Authorization it could have been given.
    """
    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    assert github_api.live(token)
    project_root = tmp_path / "rewrite-root"
    (project_root / "damianvtran").mkdir(parents=True)
    bare = project_root / "damianvtran" / "scratch.git"
    bare.mkdir()
    git_env = _node_env(env)
    for cmd in (["git", "init", "--bare", "-q"], ["git", "config", "http.receivepack", "true"]):
        assert _run(cmd, env=git_env, cwd=bare).returncode == 0, cmd
    backend = _GitBackend(project_root, github_api)
    server = _PlainGitServer(backend)
    try:
        injected = _injected_env(env, token)
        rewritten = _with_config(
            injected,
            [("url.http://127.0.0.1:%d/.insteadOf" % server.port, "https://github.com/")],
        )
        result = _run([GIT, "ls-remote", f"https://github.com/{SCRATCH}.git"], env=rewritten)
        assert result.returncode != 0, result.stdout
        # The rewritten host was REACHED and got NOTHING: every request it saw
        # was unauthenticated, so the token was never offered to it.
        assert backend.auth_attempts, "the local server was never reached"
        assert all(
            attempt["token"] == "" for attempt in backend.auth_attempts
        ), backend.auth_attempts
        assert not any(attempt["authorised"] for attempt in backend.auth_attempts)
        assert not (env.home / ".git-credentials").exists()
    finally:
        server.close()


# ---------------------------------------------------------------------------
# T8 — 0-peer devices, the printenv invariant, the catalogue and T6/T7 records
# ---------------------------------------------------------------------------


def test_a_zero_peer_device_gets_no_injection_and_no_config_writes(env: _Env) -> None:
    """T8a/b: a device that borrows nothing pays nothing — and writes nothing.

    ``borrowed_git_env`` answers ``({}, "")``, the auth-store constructor stays
    the plain ``AuthStore`` (mesh-awareness is only added for a device that
    actually borrows), no git config file exists in the node's home, and the env
    mechanism never touches disk — the whole git config arrives per command.
    """
    from local_operator.network.credentials import build_auth_store
    from local_operator.providers.auth_store import AuthStore

    assert github_mod.borrowed_git_env(root=env.root) == ({}, "")
    store = build_auth_store(env.root)
    try:
        assert type(store) is AuthStore, type(store)
    finally:
        store.close()
    assert not (env.home / ".gitconfig").exists()
    assert not (env.home / ".git-credentials").exists()


def test_the_printenv_invariant_masks_the_injected_token(
    env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T8e: the child CAN print the token; the model-visible result cannot carry it.

    The real ``execute_bash`` path runs with a monkeypatched fetch (the borrow
    itself has its own cells): the child sees ``GH_TOKEN`` and prints it, and
    the session store's exact-value masking — the control the design leans on —
    must scrub it from the result the model would read. The raw child output is
    captured too, so the assertion is about MASKING and not about the variable
    being absent for some other reason.
    """
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.tools import builtin
    from local_operator.variables import VariableStore

    token = f"ghs_printenv_{os.urandom(4).hex()}"
    injected = github_mod.git_env_for_token(token)
    monkeypatch.setattr(github_mod, "borrowed_git_env", lambda **kwargs: (injected, token))

    raw = _run(
        ["/bin/sh", "-c", "printenv GH_TOKEN"],
        env={**os.environ, **injected},
    )
    assert raw.stdout.strip() == token, "the child did not actually hold the token"

    variables = VariableStore(cwd=str(tmp_path))
    context = ToolContext(cwd=str(tmp_path), variables=variables, session_id="sess-printenv")
    result = asyncio.run(
        builtin.execute_bash("bash-printenv", {"command": "printenv GH_TOKEN"}, None, None, context)
    )
    assert token not in result.text, result.text
    assert "[redacted]" in result.text, "the result did not show the masking at all"


def test_the_github_refusal_codes_are_in_the_catalogue_and_carry_ttls() -> None:
    """T8d: the four new codes are wired into the shared refusal catalogue.

    The catalogue test already renders every code; this cell pins that these
    four ARE in it (a code that never landed there would be dropped from
    rendering while every other test stayed green) and that their retry
    classes are the deliberate ones.
    """
    from local_operator.network.credentials.types import (
        BROKER_ERROR_CODES,
        BROKER_ERROR_TTL_MS,
    )

    expected = {
        github_mod.CODE_DEVICE_SCOPE,
        github_mod.CODE_APP_UNUSABLE,
        github_mod.CODE_REPOSITORIES_UNSET,
        github_mod.CODE_REPO_REFUSED,
    }
    assert expected <= set(BROKER_ERROR_CODES)
    # The two states an operator repairs on the owner are cached LONG (no retry
    # from the borrower can change them); the repo-refused one is a per-mint
    # answer that must not be remembered at all.
    assert BROKER_ERROR_TTL_MS[github_mod.CODE_APP_UNUSABLE] > 60_000
    assert BROKER_ERROR_TTL_MS[github_mod.CODE_REPOSITORIES_UNSET] > 60_000
    assert BROKER_ERROR_TTL_MS[github_mod.CODE_REPO_REFUSED] == 0


def test_the_t6_t7_acceptance_records_are_present_and_claim_no_coverage() -> None:
    """T6/T7: RECORDED, ACCEPTED — NOT TESTED, and the disclosure copy exists.

    T6 (allow-list enforcement is external: GitHub-side mint narrowing + the
    helper path guard) and T7 (device-scope consequences) are accepted records,
    not cells — this test pins that the records and the mandated T7(b)
    disclosure copy are PRESENT, and deliberately asserts nothing about
    enforcement, because no coverage is claimed.
    """
    from local_operator.network import cli as network_cli

    disclosure = network_cli._github_share_disclosure(
        "peer-b"
    )  # noqa: SLF001 — the receipt's own copy
    assert "any process or session on peer-b" in disclosure
    assert "while the share stands" in disclosure
    guides = Path(__file__).resolve().parents[3] / "local_operator" / "guides" / "network"
    appendix = Path(__file__).resolve().parents[3] / "docs" / "design" / "mesh-credentials.md"
    guide = guides / "GUIDE.md"
    assert appendix.exists(), "the mesh-credentials appendix must exist"
    text = appendix.read_text(encoding="utf-8") + guide.read_text(encoding="utf-8")
    assert "any process or session" in text, "the T7(b) disclosure is missing from the docs"
    assert "ACCEPTED — NOT TESTED" in text, "the T6/T7 records must say what they are"


def test_the_revoke_op_deletes_now_and_the_receipt_carries_the_mint_revoke_copy(
    owner: Any, github_api: _FakeGithub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4 wiring: the control op DELETEs for real; the receipt copy says so.

    The op is driven against the broker's own lender (the DELETE it triggers is
    the same object the broker's scheduler would run). The receipt copy is
    asserted against the SAME helpers the receipt prints, so a drift between
    cell and copy cannot pass. The full CLI↔relay round trip belongs to the
    node's e2e lane; this stops at the seam rather than pretending to be it.
    """
    import inspect

    from local_operator.network import cli as network_cli
    from local_operator.network.credentials import (  # noqa: PLC2701 — the op object itself
        _LocalOps,
    )

    token = _ask_grant(owner, BORROWER_DEVICE)["access_token"]
    monkeypatch.setattr(owner_mod, "broker_for_relay", lambda server: owner.broker)
    ops = _LocalOps(server=object())
    acked = ops.revoke({"credential_key": github_mod.GITHUB_KEY, "holder": BORROWER_DEVICE})
    assert acked == {"kind": "ack", "key": github_mod.GITHUB_KEY, "revoked": 1}
    assert github_api.revoke_requests == [token]
    assert not github_api.live(token)

    # A non-github key answers zero WITHOUT a broker lookup — nothing to un-issue.
    monkeypatch.setattr(owner_mod, "broker_for_relay", lambda server: None)
    assert ops.revoke({"credential_key": "openai", "holder": BORROWER_DEVICE})["revoked"] == 0

    # The receipt copy, from the helpers the receipt itself uses.
    lines = network_cli._github_revoke_lines("peer-b", 1)  # noqa: SLF001 — the receipt's own copy
    assert "refused now" in lines[0] and "DELETE /installation/token" in lines[0]
    assert "1 outstanding token(s) revoked at GitHub just now" in lines[1]
    payload = network_cli._github_revocation_payload(1, 900)  # noqa: SLF001
    assert payload["minted_tokens_revoked"] == 1
    assert "60-minute ceiling" in payload["copied_bearer"]
    fallback = network_cli._github_revoke_lines("peer-b", None)  # noqa: SLF001
    assert "relay is not running" in fallback[1]

    # The CLI routing call: the op name and the (key, holder) payload exist at
    # the one place that can ask this device's relay.
    source = inspect.getsource(network_cli._cmd_credential)
    assert '"credential_revoke"' in source
    assert "credential_key=key" in source and "holder=device" in source
